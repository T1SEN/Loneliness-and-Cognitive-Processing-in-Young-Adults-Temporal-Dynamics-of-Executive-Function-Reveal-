"""
Power & Reliability Analysis Suite
==================================

Unified power analysis and reliability assessment for UCLA × Executive Function research.

Consolidates 4 analyses:
- power_sensitivity_analysis.py
- power_reliability_adjusted.py
- reliability_corrected_effects.py
- reliability_enhancement_composites.py

Usage:
    python -m analysis.validation.power_suite                    # Run all
    python -m analysis.validation.power_suite --analysis power_sensitivity
    python -m analysis.validation.power_suite --list

    from analysis.validation import power_suite
    power_suite.run('power_sensitivity')
    power_suite.run()  # All analyses

CRITICAL: All models control for DASS-21 subscales (depression, anxiety, stress) + age.

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX, PRP_RT_MAX
)
from analysis.preprocessing import (
    safe_zscore,
    prepare_gender_variable
)

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "power_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable
    source_script: str


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, source_script: str):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            source_script=source_script
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_power_data() -> pd.DataFrame:
    """Load and prepare master dataset for power analyses."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    # Ensure ucla_total exists
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors using NaN-safe z-score (ddof=1)
    required_cols = ['age', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']

    for col in required_cols:
        if col not in master.columns:
            raise ValueError(f"Missing required column: {col}")

    master['z_age'] = safe_zscore(master['age'])
    master['z_ucla'] = safe_zscore(master['ucla_total'])
    master['z_dass_dep'] = safe_zscore(master['dass_depression'])
    master['z_dass_anx'] = safe_zscore(master['dass_anxiety'])
    master['z_dass_str'] = safe_zscore(master['dass_stress'])

    return master


def get_outcomes(df: pd.DataFrame) -> list:
    """Get available outcome variables."""
    outcomes = []

    if 'pe_rate' in df.columns:
        outcomes.append(('pe_rate', 'WCST Perseverative Error Rate'))
    if 'prp_bottleneck' in df.columns:
        outcomes.append(('prp_bottleneck', 'PRP Bottleneck Effect'))
    if 'stroop_interference' in df.columns:
        outcomes.append(('stroop_interference', 'Stroop Interference'))

    return outcomes


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def power_correlation(n: int, r: float, alpha: float = 0.05) -> float:
    """Calculate power for correlation test using Fisher Z transformation."""
    if abs(r) >= 1 or n < 4:
        return np.nan

    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(z_crit - abs(z_r) / se) + stats.norm.cdf(-z_crit - abs(z_r) / se)

    return power


def power_ttest_ind(n1: int, n2: int, d: float, alpha: float = 0.05) -> float:
    """Calculate power for independent t-test (Cohen's d)."""
    if n1 < 2 or n2 < 2:
        return np.nan

    se = np.sqrt(1 / n1 + 1 / n2)
    ncp = d / se
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    return power


def power_regression_coefficient(n: int, r2_full: float, r2_reduced: float,
                                 n_predictors: int, alpha: float = 0.05) -> float:
    """Calculate power for testing a regression coefficient (F-test for R² change)."""
    if n <= n_predictors + 1:
        return np.nan

    delta_r2 = r2_full - r2_reduced
    f2 = delta_r2 / (1 - r2_full) if r2_full < 1 else np.nan

    if pd.isna(f2) or f2 < 0:
        return np.nan

    df1 = 1
    df2 = n - n_predictors - 1
    ncp = f2 * (df1 + df2 + 1)
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

    return power


def minimum_detectable_effect_correlation(n: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """Calculate minimum r detectable with given N and power."""
    low, high = 0.001, 0.999

    for _ in range(50):
        mid = (low + high) / 2
        current_power = power_correlation(n, mid, alpha)

        if current_power < power:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def sample_size_for_correlation(r: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Calculate required N for correlation with given power."""
    if abs(r) >= 1:
        return np.nan

    z_r = 0.5 * np.log((1 + abs(r)) / (1 - abs(r)))
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = ((z_alpha + z_beta) / z_r) ** 2 + 3

    return int(np.ceil(n))


def cronbach_alpha(df: pd.DataFrame) -> float:
    """Calculate Cronbach's alpha for reliability."""
    df_clean = df.dropna()
    if len(df_clean) < 2:
        return np.nan

    n_items = df_clean.shape[1]
    if n_items < 2:
        return np.nan

    item_vars = df_clean.var(axis=0, ddof=1)
    total_var = df_clean.sum(axis=1).var(ddof=1)

    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


def correct_for_attenuation(r_observed: float, reliability_x: float, reliability_y: float) -> float:
    """Correct correlation for measurement error."""
    if reliability_x <= 0 or reliability_y <= 0:
        return np.nan

    r_true = r_observed / np.sqrt(reliability_x * reliability_y)
    return min(abs(r_true), 1.0) * np.sign(r_observed)


# =============================================================================
# ANALYSIS 1: POWER SENSITIVITY
# =============================================================================

@register_analysis(
    name="power_sensitivity",
    description="Post-hoc power analysis and sensitivity curves",
    source_script="power_sensitivity_analysis.py"
)
def analyze_power_sensitivity(verbose: bool = True) -> pd.DataFrame:
    """
    Comprehensive power and sensitivity analysis.

    Features:
    - Post-hoc power for observed effect sizes
    - Minimum detectable effects
    - Required sample sizes
    - Power curves
    """
    if verbose:
        print("\n" + "=" * 70)
        print("POWER SENSITIVITY ANALYSIS")
        print("=" * 70)

    df = load_power_data()

    N_total = len(df)
    N_male = len(df[df['gender'] == 'male'])
    N_female = len(df[df['gender'] == 'female'])

    if verbose:
        print(f"\nSample sizes: Total={N_total}, Male={N_male}, Female={N_female}")

    power_results = []
    outcomes = get_outcomes(df)

    # Post-hoc power for correlations
    for col, label in outcomes:
        if col not in df.columns or 'ucla_total' not in df.columns:
            continue

        # Full sample
        valid = df.dropna(subset=[col, 'ucla_total'])
        n = len(valid)
        if n < 5:
            continue

        r, p = stats.pearsonr(valid['ucla_total'], valid[col])
        pwr = power_correlation(n, r)

        power_results.append({
            'analysis': 'Correlation',
            'outcome': label,
            'subgroup': 'Full Sample',
            'n': n,
            'effect_size': r,
            'effect_type': 'r',
            'power': pwr,
            'observed_p': p,
            'significant': p < 0.05
        })

        # By gender
        for gender in ['male', 'female']:
            gender_df = df[df['gender'] == gender]
            valid = gender_df.dropna(subset=[col, 'ucla_total'])
            n = len(valid)
            if n < 5:
                continue

            r, p = stats.pearsonr(valid['ucla_total'], valid[col])
            pwr = power_correlation(n, r)

            power_results.append({
                'analysis': 'Correlation',
                'outcome': label,
                'subgroup': gender.capitalize(),
                'n': n,
                'effect_size': r,
                'effect_type': 'r',
                'power': pwr,
                'observed_p': p,
                'significant': p < 0.05
            })

    # Group comparison power
    ucla_median = df['ucla_total'].median()
    df['ucla_group'] = df['ucla_total'].apply(lambda x: 'High' if x > ucla_median else 'Low')

    for col, label in outcomes:
        if col not in df.columns:
            continue

        valid = df.dropna(subset=[col, 'ucla_group'])
        high = valid[valid['ucla_group'] == 'High'][col]
        low = valid[valid['ucla_group'] == 'Low'][col]

        n_high, n_low = len(high), len(low)
        if n_high < 5 or n_low < 5:
            continue

        pooled_std = np.sqrt(((n_high - 1) * high.var() + (n_low - 1) * low.var()) / (n_high + n_low - 2))
        d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0

        t_stat, p = stats.ttest_ind(high, low)
        pwr = power_ttest_ind(n_high, n_low, d)

        power_results.append({
            'analysis': 'Group Comparison',
            'outcome': label,
            'subgroup': 'Full Sample',
            'n': n_high + n_low,
            'effect_size': d,
            'effect_type': "Cohen's d",
            'power': pwr,
            'observed_p': p,
            'significant': p < 0.05
        })

    power_df = pd.DataFrame(power_results)

    # Minimum detectable effects
    mde_results = []
    for subgroup, n in [('Full Sample', N_total), ('Males', N_male), ('Females', N_female)]:
        mde_r = minimum_detectable_effect_correlation(n)
        mde_results.append({
            'subgroup': subgroup,
            'n': n,
            'mde_80': mde_r,
            'mde_90': minimum_detectable_effect_correlation(n, power=0.90),
            'effect_type': 'r'
        })

    mde_df = pd.DataFrame(mde_results)

    # Required sample sizes
    target_effects = {
        'Small (r=0.10)': 0.10,
        'Small-Medium (r=0.20)': 0.20,
        'Medium (r=0.30)': 0.30,
        'Medium-Large (r=0.40)': 0.40,
        'Large (r=0.50)': 0.50
    }

    required_n_results = []
    for label, r in target_effects.items():
        n_80 = sample_size_for_correlation(r, power=0.80)
        n_90 = sample_size_for_correlation(r, power=0.90)

        required_n_results.append({
            'effect_size_label': label,
            'r': r,
            'n_for_80_power': n_80,
            'n_for_90_power': n_90
        })

    required_n_df = pd.DataFrame(required_n_results)

    # Save outputs
    power_df.to_csv(OUTPUT_DIR / "posthoc_power_analysis.csv", index=False, encoding='utf-8-sig')
    mde_df.to_csv(OUTPUT_DIR / "minimum_detectable_effects.csv", index=False, encoding='utf-8-sig')
    required_n_df.to_csv(OUTPUT_DIR / "required_sample_sizes.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\nPost-hoc Power Summary:")
        print(f"  Total analyses: {len(power_df)}")
        if len(power_df) > 0:
            print(f"  Adequately powered (≥80%): {(power_df['power'] >= 0.80).sum()}")
            print(f"  Mean power: {power_df['power'].mean():.3f}")

        print(f"\nMinimum Detectable Effects (80% power):")
        for _, row in mde_df.iterrows():
            print(f"  {row['subgroup']} (N={row['n']}): r ≥ {row['mde_80']:.3f}")

        print(f"\nRequired Sample Sizes:")
        for _, row in required_n_df.iterrows():
            print(f"  {row['effect_size_label']}: N = {row['n_for_80_power']}")

    return power_df


# =============================================================================
# ANALYSIS 2: RELIABILITY-ADJUSTED POWER
# =============================================================================

@register_analysis(
    name="power_reliability",
    description="Reliability-adjusted power projections",
    source_script="power_reliability_adjusted.py"
)
def analyze_power_reliability(verbose: bool = True) -> pd.DataFrame:
    """
    Reliability-adjusted power analysis.

    Adjusts observed partial R² for attenuation due to measurement error
    and projects power at various sample sizes.
    """
    import math

    if verbose:
        print("\n" + "=" * 70)
        print("RELIABILITY-ADJUSTED POWER ANALYSIS")
        print("=" * 70)

    df = load_power_data()
    outcomes = get_outcomes(df)

    # Default reliability estimates
    reliability = {
        'ucla': 0.92,  # Literature-based UCLA alpha
        'stroop': 0.60,
        'prp': 0.50,
        'wcst': 0.50
    }

    # Try to load existing reliability if available
    rel_path = OUTPUT_DIR / "reliability_estimates.csv"
    if rel_path.exists():
        rel_df = pd.read_csv(rel_path)
        if not rel_df.empty:
            row = rel_df.iloc[0]
            reliability['stroop'] = row.get('r_spearman_brown', 0.60) if 'stroop' in str(row.get('task', '')) else 0.60

    results = []

    for col, label in outcomes:
        if col not in df.columns:
            continue

        valid = df.dropna(subset=[col, 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])
        n = len(valid)
        if n < 20:
            continue

        # Fit OLS model
        formula = f"{col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        try:
            model = smf.ols(formula, data=valid).fit()
        except Exception:
            continue

        # Get UCLA coefficient
        if 'z_ucla' not in model.params:
            continue

        t_stat = model.tvalues['z_ucla']
        k_params = len(model.params)
        df2 = n - k_params

        # Observed partial R²
        pr2_obs = (t_stat ** 2) / (t_stat ** 2 + df2) if df2 > 0 else 0

        # Reliability-adjusted (true) partial R²
        r_x = reliability['ucla']
        r_y = reliability.get(col.split('_')[0], 0.6)
        atten = max(r_x * r_y, 1e-6)
        pr2_true = min(pr2_obs / atten, 0.99)

        # Power projection at N=150
        def p_from_pr2(df2: int, pr2: float) -> float:
            if df2 <= 0 or pr2 <= 0:
                return 1.0
            t2 = df2 * pr2 / (1 - pr2)
            t = math.sqrt(max(t2, 0.0))
            return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))

        n_target = 150
        df2_target = n_target - k_params
        p_obs_150 = p_from_pr2(df2_target, pr2_obs)
        p_true_150 = p_from_pr2(df2_target, pr2_true)

        # Required N for 80% power
        t_needed = 1.96 + 0.84  # z_alpha + z_beta

        def n_required(pr2: float) -> float:
            if pr2 <= 0:
                return float("inf")
            df2_req = (t_needed ** 2) * (1 - pr2) / pr2
            return df2_req + k_params

        n80_obs = n_required(pr2_obs)
        n80_true = n_required(pr2_true)

        results.append({
            'outcome': label,
            'n_current': n,
            'partialR2_observed': pr2_obs,
            'partialR2_true_est': pr2_true,
            'reliability_ucla': r_x,
            'reliability_outcome': r_y,
            'pred_p_at_150_obs': p_obs_150,
            'pred_p_at_150_true': p_true_150,
            'N_needed_80power_obs': n80_obs,
            'N_needed_80power_true': n80_true
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "power_reliability_adjusted.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print("\nReliability-Adjusted Power Projections:")
        print("-" * 60)
        for _, row in results_df.iterrows():
            print(f"\n{row['outcome']}:")
            print(f"  Observed partial R²: {row['partialR2_observed']:.4f}")
            print(f"  True partial R² (est.): {row['partialR2_true_est']:.4f}")
            print(f"  N needed for 80% power (obs): {row['N_needed_80power_obs']:.0f}")
            print(f"  N needed for 80% power (true): {row['N_needed_80power_true']:.0f}")

    return results_df


# =============================================================================
# ANALYSIS 3: RELIABILITY-CORRECTED EFFECTS
# =============================================================================

@register_analysis(
    name="reliability_corrected",
    description="Attenuation-corrected UCLA-EF correlations",
    source_script="reliability_corrected_effects.py"
)
def analyze_reliability_corrected(verbose: bool = True) -> pd.DataFrame:
    """
    Compute reliability-corrected (disattenuated) correlations.

    Uses split-half reliability estimates with Spearman-Brown correction
    to estimate true correlations free of measurement error.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RELIABILITY-CORRECTED EFFECTS ANALYSIS")
        print("=" * 70)

    df = load_power_data()

    # Compute UCLA reliability (Cronbach's alpha)
    ucla_items = [f'ucla_{i}' for i in range(1, 21)]
    available_items = [c for c in ucla_items if c in df.columns]

    if len(available_items) >= 10:
        item_data = df[available_items].dropna()
        if len(item_data) >= 30:
            n_items = len(available_items)
            item_vars = item_data.var()
            total_var = item_data.sum(axis=1).var()
            ucla_rel = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
        else:
            ucla_rel = 0.92
    else:
        ucla_rel = 0.92  # Literature default

    if verbose:
        print(f"\nUCLA Reliability: alpha = {ucla_rel:.3f}")

    # Compute EF split-half reliabilities
    ef_reliabilities = {}

    # Stroop split-half
    stroop_path = RESULTS_DIR / "4c_stroop_trials.csv"
    if stroop_path.exists():
        stroop_trials = pd.read_csv(stroop_path, encoding='utf-8')
        if 'participantId' in stroop_trials.columns:
            stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})

        rt_col = 'rt_ms' if 'rt_ms' in stroop_trials.columns else 'rt'
        cond_col = 'type' if 'type' in stroop_trials.columns else 'cond'

        stroop_trials = stroop_trials[stroop_trials['timeout'] == False].copy()
        stroop_trials = stroop_trials[stroop_trials['correct'] == True].copy()
        stroop_trials = stroop_trials[(stroop_trials[rt_col] > DEFAULT_RT_MIN) &
                                      (stroop_trials[rt_col] < STROOP_RT_MAX)].copy()

        results = []
        for pid in stroop_trials['participant_id'].unique():
            pdata = stroop_trials[stroop_trials['participant_id'] == pid].copy()
            pdata['trial_num'] = range(len(pdata))
            pdata['split'] = pdata['trial_num'] % 2

            for split in [0, 1]:
                split_data = pdata[pdata['split'] == split]
                cong = split_data[split_data[cond_col].str.lower() == 'congruent'][rt_col].mean()
                incong = split_data[split_data[cond_col].str.lower() == 'incongruent'][rt_col].mean()

                if not np.isnan(cong) and not np.isnan(incong):
                    results.append({
                        'participant_id': pid,
                        'split': split,
                        'interference': incong - cong
                    })

        results_df = pd.DataFrame(results)
        even = results_df[results_df['split'] == 0].set_index('participant_id')['interference']
        odd = results_df[results_df['split'] == 1].set_index('participant_id')['interference']

        common_pids = even.index.intersection(odd.index)
        if len(common_pids) >= 30:
            r_split, _ = stats.pearsonr(even.loc[common_pids], odd.loc[common_pids])
            r_sb = 2 * r_split / (1 + r_split)
            ef_reliabilities['stroop'] = {
                'task': 'stroop',
                'measure': 'stroop_interference',
                'r_split_half': r_split,
                'r_spearman_brown': r_sb,
                'n': len(common_pids)
            }
            if verbose:
                print(f"Stroop Split-Half: r={r_split:.3f}, SB={r_sb:.3f}")

    # WCST and PRP split-half (simplified)
    wcst_path = RESULTS_DIR / "4b_wcst_trials.csv"
    if wcst_path.exists():
        ef_reliabilities['wcst'] = {
            'task': 'wcst',
            'measure': 'pe_rate',
            'r_spearman_brown': 0.50,  # Default from literature
            'n': len(df)
        }

    prp_path = RESULTS_DIR / "4a_prp_trials.csv"
    if prp_path.exists():
        ef_reliabilities['prp'] = {
            'task': 'prp',
            'measure': 'prp_bottleneck',
            'r_spearman_brown': 0.50,  # Default
            'n': len(df)
        }

    # Compute corrected correlations
    corrected_results = []

    task_col_map = {
        'stroop': 'stroop_interference',
        'wcst': 'pe_rate',
        'prp': 'prp_bottleneck'
    }

    for task, rel_info in ef_reliabilities.items():
        measure = task_col_map.get(task)
        if measure not in df.columns:
            continue

        ef_rel = rel_info.get('r_spearman_brown', 0.5)

        valid = df[['ucla_total', measure]].dropna()
        if len(valid) < 30:
            continue

        r_observed, p_observed = stats.pearsonr(valid['ucla_total'], valid[measure])
        r_corrected = correct_for_attenuation(r_observed, ucla_rel, ef_rel)

        # CI scaling
        n = len(valid)
        se_r = np.sqrt((1 - r_observed ** 2) / (n - 2))
        correction_factor = 1 / np.sqrt(ucla_rel * ef_rel)

        corrected_results.append({
            'task': task,
            'ef_measure': measure,
            'n': n,
            'ucla_reliability': ucla_rel,
            'ef_reliability': ef_rel,
            'r_observed': r_observed,
            'p_observed': p_observed,
            'r_corrected': r_corrected,
            'ci_low_obs': r_observed - 1.96 * se_r,
            'ci_high_obs': r_observed + 1.96 * se_r,
            'correction_factor': correction_factor
        })

    corrected_df = pd.DataFrame(corrected_results)

    # Save reliability estimates
    rel_list = [r for r in ef_reliabilities.values() if r is not None]
    if rel_list:
        rel_df = pd.DataFrame(rel_list)
        rel_df['ucla_alpha'] = ucla_rel
        rel_df.to_csv(OUTPUT_DIR / "reliability_estimates.csv", index=False, encoding='utf-8-sig')

    corrected_df.to_csv(OUTPUT_DIR / "corrected_correlations.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print("\nAttenuation-Corrected Correlations:")
        print("-" * 60)
        for _, row in corrected_df.iterrows():
            change = ((row['r_corrected'] / row['r_observed']) - 1) * 100 if row['r_observed'] != 0 else 0
            print(f"  {row['task'].upper()}: r_obs={row['r_observed']:.3f} → r_corr={row['r_corrected']:.3f} ({change:+.0f}%)")

    return corrected_df


# =============================================================================
# ANALYSIS 4: RELIABILITY ENHANCED COMPOSITES
# =============================================================================

@register_analysis(
    name="reliability_enhanced",
    description="Composite EF scores with enhanced reliability",
    source_script="reliability_enhancement_composites.py"
)
def analyze_reliability_enhanced(verbose: bool = True) -> pd.DataFrame:
    """
    Generate composite EF scores with enhanced reliability.

    Creates multi-indicator composites for each task and a meta-control
    factor combining all three tasks.
    """
    import ast

    if verbose:
        print("\n" + "=" * 70)
        print("RELIABILITY ENHANCED COMPOSITES")
        print("=" * 70)

    df = load_power_data()
    results = []

    # Load trial data
    wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
    stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8')

    # Normalize participant ID
    for tdf in [wcst_trials, stroop_trials, prp_trials]:
        if 'participantId' in tdf.columns:
            tdf.rename(columns={'participantId': 'participant_id'}, inplace=True)

    # Parse WCST extra field
    def parse_wcst_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except:
            return {}

    if 'extra' in wcst_trials.columns:
        wcst_trials['extra_dict'] = wcst_trials['extra'].apply(parse_wcst_extra)
        wcst_trials['isPE'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))
    elif 'isPE' not in wcst_trials.columns:
        wcst_trials['isPE'] = False

    # WCST metrics
    rt_col = 'reactionTimeMs' if 'reactionTimeMs' in wcst_trials.columns else 'rt_ms'
    wcst_metrics = []

    for pid in wcst_trials['participant_id'].unique():
        trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()
        trials = trials[(trials['timeout'] == False) & (trials[rt_col] > 0)].copy()

        if len(trials) < 10:
            continue

        pe_rate = (trials['isPE'].sum() / len(trials)) * 100
        error_rate = ((trials['correct'] == False).sum() / len(trials)) * 100
        rt_cv = trials[rt_col].std() / trials[rt_col].mean() * 100 if trials[rt_col].mean() > 0 else np.nan

        wcst_metrics.append({
            'participant_id': pid,
            'pe_rate': pe_rate,
            'wcst_error_rate': error_rate,
            'wcst_rt_cv': rt_cv
        })

    wcst_df = pd.DataFrame(wcst_metrics)

    if len(wcst_df) > 0 and verbose:
        wcst_items = wcst_df[['pe_rate', 'wcst_error_rate', 'wcst_rt_cv']].dropna()
        if len(wcst_items) > 10:
            scaler = StandardScaler()
            wcst_z = pd.DataFrame(scaler.fit_transform(wcst_items), columns=wcst_items.columns)
            wcst_alpha = cronbach_alpha(wcst_z)
            print(f"\nWCST Composite: alpha = {wcst_alpha:.3f}, N = {len(wcst_items)}")
            results.append({'task': 'WCST', 'alpha': wcst_alpha, 'n_items': 3, 'n': len(wcst_items)})

    # Stroop metrics
    rt_col = 'rt_ms' if 'rt_ms' in stroop_trials.columns else 'rt'
    type_col = 'type' if 'type' in stroop_trials.columns else 'trialType'
    stroop_metrics = []

    for pid in stroop_trials['participant_id'].unique():
        trials = stroop_trials[stroop_trials['participant_id'] == pid].copy()

        if 'timeout' in trials.columns:
            trials = trials[(trials['timeout'].fillna(False) == False) & (trials[rt_col] > 0)].copy()
        else:
            trials = trials[trials[rt_col] > 0].copy()

        if len(trials) < 10:
            continue

        incongruent_rt = trials[trials[type_col] == 'incongruent'][rt_col].mean()
        congruent_rt = trials[trials[type_col] == 'congruent'][rt_col].mean()
        interference = incongruent_rt - congruent_rt

        if np.isnan(interference):
            continue

        incongruent_trials = trials[trials[type_col] == 'incongruent']
        error_rate = ((incongruent_trials['correct'] == False).sum() / len(incongruent_trials)) * 100 if len(incongruent_trials) > 0 else np.nan
        rt_cv = incongruent_trials[rt_col].std() / incongruent_trials[rt_col].mean() * 100 if len(incongruent_trials) > 0 else np.nan

        stroop_metrics.append({
            'participant_id': pid,
            'stroop_interference': interference,
            'stroop_error_rate': error_rate,
            'stroop_rt_cv': rt_cv
        })

    stroop_df = pd.DataFrame(stroop_metrics)

    if len(stroop_df) > 0 and verbose:
        stroop_items = stroop_df[['stroop_interference', 'stroop_error_rate', 'stroop_rt_cv']].dropna()
        if len(stroop_items) > 10:
            scaler = StandardScaler()
            stroop_z = pd.DataFrame(scaler.fit_transform(stroop_items), columns=stroop_items.columns)
            stroop_alpha = cronbach_alpha(stroop_z)
            print(f"Stroop Composite: alpha = {stroop_alpha:.3f}, N = {len(stroop_items)}")
            results.append({'task': 'Stroop', 'alpha': stroop_alpha, 'n_items': 3, 'n': len(stroop_items)})

    # PRP metrics
    prp_metrics = []

    for pid in prp_trials['participant_id'].unique():
        trials = prp_trials[prp_trials['participant_id'] == pid].copy()
        trials = trials[(trials['t2_timeout'] == False) & (trials['t2_rt_ms'] > 0)].copy()

        if len(trials) < 10:
            continue

        soa_col = 'soa_measured_ms' if 'soa_measured_ms' in trials.columns else 'soa_nominal_ms'
        short_rt = trials[trials[soa_col] <= 150]['t2_rt_ms'].mean()
        long_rt = trials[trials[soa_col] >= 1200]['t2_rt_ms'].mean()
        bottleneck = short_rt - long_rt

        t2_error_rate = ((trials['t2_correct'] == False).sum() / len(trials)) * 100
        t2_rt_cv = trials['t2_rt_ms'].std() / trials['t2_rt_ms'].mean() * 100

        prp_metrics.append({
            'participant_id': pid,
            'prp_bottleneck': bottleneck,
            'prp_error_rate': t2_error_rate,
            'prp_rt_cv': t2_rt_cv
        })

    prp_df = pd.DataFrame(prp_metrics)

    if len(prp_df) > 0 and verbose:
        prp_items = prp_df[['prp_bottleneck', 'prp_error_rate', 'prp_rt_cv']].dropna()
        if len(prp_items) > 10:
            scaler = StandardScaler()
            prp_z = pd.DataFrame(scaler.fit_transform(prp_items), columns=prp_items.columns)
            prp_alpha = cronbach_alpha(prp_z)
            print(f"PRP Composite: alpha = {prp_alpha:.3f}, N = {len(prp_items)}")
            results.append({'task': 'PRP', 'alpha': prp_alpha, 'n_items': 3, 'n': len(prp_items)})

    # Create composites
    scaler = StandardScaler()

    if len(wcst_df) > 0:
        wcst_items = wcst_df[['pe_rate', 'wcst_error_rate', 'wcst_rt_cv']].fillna(wcst_df[['pe_rate', 'wcst_error_rate', 'wcst_rt_cv']].mean())
        wcst_z = pd.DataFrame(scaler.fit_transform(wcst_items), columns=wcst_items.columns, index=wcst_df.index)
        wcst_df['wcst_composite'] = wcst_z.mean(axis=1)

    if len(stroop_df) > 0:
        stroop_items = stroop_df[['stroop_interference', 'stroop_error_rate', 'stroop_rt_cv']].fillna(stroop_df[['stroop_interference', 'stroop_error_rate', 'stroop_rt_cv']].mean())
        stroop_z = pd.DataFrame(scaler.fit_transform(stroop_items), columns=stroop_items.columns, index=stroop_df.index)
        stroop_df['stroop_composite'] = stroop_z.mean(axis=1)

    if len(prp_df) > 0:
        prp_items = prp_df[['prp_bottleneck', 'prp_error_rate', 'prp_rt_cv']].fillna(prp_df[['prp_bottleneck', 'prp_error_rate', 'prp_rt_cv']].mean())
        prp_z = pd.DataFrame(scaler.fit_transform(prp_items), columns=prp_items.columns, index=prp_df.index)
        prp_df['prp_composite'] = prp_z.mean(axis=1)

    # Merge composites with master data
    composite_df = df[['participant_id', 'gender', 'ucla_total']].copy()

    if len(wcst_df) > 0:
        composite_df = composite_df.merge(wcst_df[['participant_id', 'wcst_composite', 'pe_rate']],
                                          on='participant_id', how='left')
    if len(stroop_df) > 0:
        composite_df = composite_df.merge(stroop_df[['participant_id', 'stroop_composite', 'stroop_interference']],
                                          on='participant_id', how='left')
    if len(prp_df) > 0:
        composite_df = composite_df.merge(prp_df[['participant_id', 'prp_composite', 'prp_bottleneck']],
                                          on='participant_id', how='left')

    # Meta-control composite
    composite_cols = []
    for col in ['wcst_composite', 'stroop_composite', 'prp_composite']:
        if col in composite_df.columns:
            composite_cols.append(col)

    if len(composite_cols) > 0:
        composite_df['meta_control'] = composite_df[composite_cols].mean(axis=1)

        # Meta-control alpha
        cross_task = composite_df[composite_cols].dropna()
        if len(cross_task) > 10:
            meta_alpha = cronbach_alpha(cross_task)
            print(f"Meta-Control: alpha = {meta_alpha:.3f}, N = {len(cross_task)}")
            results.append({'task': 'Meta-Control', 'alpha': meta_alpha, 'n_items': len(composite_cols), 'n': len(cross_task)})

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "composite_reliability.csv", index=False, encoding='utf-8-sig')
    composite_df.to_csv(OUTPUT_DIR / "ef_composite_scores.csv", index=False, encoding='utf-8-sig')

    # Compare single vs composite correlations
    if verbose and 'ucla_total' in composite_df.columns:
        print("\nUCLA-EF Correlations (Single vs Composite):")
        print("-" * 50)

        comparisons = [
            ('pe_rate', 'wcst_composite', 'WCST'),
            ('stroop_interference', 'stroop_composite', 'Stroop'),
            ('prp_bottleneck', 'prp_composite', 'PRP')
        ]

        for single, comp, name in comparisons:
            if single in composite_df.columns and comp in composite_df.columns:
                valid_single = composite_df.dropna(subset=[single, 'ucla_total'])
                valid_comp = composite_df.dropna(subset=[comp, 'ucla_total'])

                if len(valid_single) > 10 and len(valid_comp) > 10:
                    r_single, p_single = stats.pearsonr(valid_single['ucla_total'], valid_single[single])
                    r_comp, p_comp = stats.pearsonr(valid_comp['ucla_total'], valid_comp[comp])
                    print(f"  {name}: Single r={r_single:.3f}, Composite r={r_comp:.3f}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def list_analyses() -> None:
    """List all available analyses."""
    print("\nAvailable Power & Reliability Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")
        print(f"      Source: {spec.source_script}")
    print()


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run power and reliability analyses.

    Args:
        analysis: Specific analysis to run, or None for all
        verbose: Print output

    Returns:
        Dictionary of results DataFrames
    """
    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Use list_analyses() to see available.")

        spec = ANALYSES[analysis]
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Running: {name}")
                print(f"{'=' * 70}")
            results[name] = spec.function(verbose=verbose)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Power & Reliability Analysis Suite")
    parser.add_argument('--analysis', type=str, help='Specific analysis to run')
    parser.add_argument('--list', action='store_true', help='List available analyses')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if args.list:
        list_analyses()
        return

    run(analysis=args.analysis, verbose=not args.quiet)


if __name__ == "__main__":
    main()
