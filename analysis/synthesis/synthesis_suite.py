"""
Synthesis Analysis Suite
========================

Unified integration and summary analyses for UCLA × Executive Function research.

Consolidates all 8 analyses:
- synthesis_1_group_comparisons.py
- synthesis_2_forest_plot.py
- synthesis_3_reliability_analysis.py
- synthesis_4_gender_stratified_regressions.py
- synthesis_5_ucla_dass_correlation_by_gender.py
- synthesis_6_variance_tests.py
- synthesis_7_power_analysis.py
- synthesis_8_robust_sensitivity.py

Usage:
    python -m analysis.synthesis.synthesis_suite                    # Run all
    python -m analysis.synthesis.synthesis_suite --analysis forest_plot
    python -m analysis.synthesis.synthesis_suite --list

    from analysis.synthesis import synthesis_suite
    synthesis_suite.run('forest_plot')
    synthesis_suite.run()  # All analyses

CRITICAL: All models control for DASS-21 subscales + age.

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
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "synthesis_suite"
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
# UTILITY FUNCTIONS
# =============================================================================

def load_synthesis_data() -> pd.DataFrame:
    """Load and prepare master dataset for synthesis analyses."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender
    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)

    # Standardize predictors
    scaler = StandardScaler()
    required_cols = ['age', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']

    for col in required_cols:
        if col not in master.columns:
            raise ValueError(f"Missing required column: {col}")

    master['z_age'] = scaler.fit_transform(master[['age']])
    master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
    master['z_dass_dep'] = scaler.fit_transform(master[['dass_depression']])
    master['z_dass_anx'] = scaler.fit_transform(master[['dass_anxiety']])
    master['z_dass_str'] = scaler.fit_transform(master[['dass_stress']])

    return master


def get_outcomes(df: pd.DataFrame) -> list:
    """Get available outcome variables."""
    outcomes = []
    if 'pe_rate' in df.columns:
        outcomes.append(('pe_rate', 'WCST PE Rate'))
    if 'wcst_accuracy' in df.columns:
        outcomes.append(('wcst_accuracy', 'WCST Accuracy'))
    if 'prp_bottleneck' in df.columns:
        outcomes.append(('prp_bottleneck', 'PRP Bottleneck'))
    if 'stroop_interference' in df.columns:
        outcomes.append(('stroop_interference', 'Stroop Interference'))
    return outcomes


def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0


# =============================================================================
# ANALYSIS 1: GROUP COMPARISONS
# =============================================================================

@register_analysis(
    name="group_comparisons",
    description="2x2x2 group comparisons: UCLA (High/Low) x DASS (High/Low) x Gender",
    source_script="synthesis_1_group_comparisons.py"
)
def analyze_group_comparisons(verbose: bool = True) -> pd.DataFrame:
    """
    Group comparison analysis with median/quartile splits.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GROUP COMPARISONS ANALYSIS")
        print("=" * 70)

    master = load_synthesis_data()
    outcomes = get_outcomes(master)

    if verbose:
        print(f"  N = {len(master)}")

    # Create group variables
    ucla_median = master['ucla_total'].median()
    dass_total = master['dass_depression'] + master['dass_anxiety'] + master['dass_stress']
    dass_median = dass_total.median()

    master['ucla_group'] = np.where(master['ucla_total'] >= ucla_median, 'High', 'Low')
    master['dass_group'] = np.where(dass_total >= dass_median, 'High', 'Low')
    master['gender_label'] = np.where(master['gender_male'] == 1, 'Male', 'Female')

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if outcome_col not in master.columns:
            continue

        df = master.dropna(subset=[outcome_col, 'ucla_total', 'dass_depression']).copy()

        if len(df) < 30:
            continue

        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        # Group statistics
        for gender in ['Male', 'Female']:
            for ucla_grp in ['High', 'Low']:
                for dass_grp in ['High', 'Low']:
                    mask = (
                        (df['gender_label'] == gender) &
                        (df['ucla_group'] == ucla_grp) &
                        (df['dass_group'] == dass_grp)
                    )
                    subset = df.loc[mask, outcome_col]

                    if len(subset) >= 3:
                        result = {
                            'outcome': outcome_col,
                            'outcome_label': outcome_label,
                            'gender': gender,
                            'ucla_group': ucla_grp,
                            'dass_group': dass_grp,
                            'n': len(subset),
                            'mean': subset.mean(),
                            'sd': subset.std(),
                            'median': subset.median()
                        }
                        all_results.append(result)

        # Overall group comparison (High UCLA vs Low UCLA)
        high_ucla = df[df['ucla_group'] == 'High'][outcome_col]
        low_ucla = df[df['ucla_group'] == 'Low'][outcome_col]

        if len(high_ucla) >= 5 and len(low_ucla) >= 5:
            t_stat, p_val = stats.ttest_ind(high_ucla, low_ucla)
            d = cohen_d(high_ucla, low_ucla)

            if verbose:
                print(f"    High UCLA (n={len(high_ucla)}): M={high_ucla.mean():.2f}, SD={high_ucla.std():.2f}")
                print(f"    Low UCLA (n={len(low_ucla)}): M={low_ucla.mean():.2f}, SD={low_ucla.std():.2f}")
                print(f"    t={t_stat:.2f}, p={p_val:.4f}, d={d:.2f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "group_comparison_statistics.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'group_comparison_statistics.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 2: FOREST PLOT
# =============================================================================

@register_analysis(
    name="forest_plot",
    description="Forest plot of DASS-controlled effect sizes with 95% CIs",
    source_script="synthesis_2_forest_plot.py"
)
def analyze_forest_plot(verbose: bool = True) -> pd.DataFrame:
    """
    Extract standardized effect sizes from hierarchical regressions.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("FOREST PLOT ANALYSIS")
        print("=" * 70)

    master = load_synthesis_data()
    outcomes = get_outcomes(master)

    all_effects = []

    for outcome_col, outcome_label in outcomes:
        if outcome_col not in master.columns:
            continue

        df = master.dropna(subset=[outcome_col]).copy()

        if len(df) < 30:
            continue

        if verbose:
            print(f"\n  {outcome_label} (N={len(df)})")

        # Full model with interaction
        formula = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

        try:
            model = smf.ols(formula, data=df).fit()

            # Extract UCLA main effect
            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                se = model.bse['z_ucla']
                p = model.pvalues['z_ucla']
                ci_low, ci_high = model.conf_int().loc['z_ucla']

                all_effects.append({
                    'outcome': outcome_col,
                    'outcome_label': outcome_label,
                    'effect': 'UCLA main effect',
                    'beta': beta,
                    'se': se,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    'p': p,
                    'n': len(df)
                })

                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"    UCLA: beta={beta:.3f}, 95%CI=[{ci_low:.3f}, {ci_high:.3f}], p={p:.4f}{sig}")

            # Extract interaction term
            int_term = 'z_ucla:C(gender_male)[T.1]'
            if int_term in model.params:
                beta = model.params[int_term]
                se = model.bse[int_term]
                p = model.pvalues[int_term]
                ci_low, ci_high = model.conf_int().loc[int_term]

                all_effects.append({
                    'outcome': outcome_col,
                    'outcome_label': outcome_label,
                    'effect': 'UCLA x Gender',
                    'beta': beta,
                    'se': se,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    'p': p,
                    'n': len(df)
                })

                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"    UCLA x Gender: beta={beta:.3f}, 95%CI=[{ci_low:.3f}, {ci_high:.3f}], p={p:.4f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")

    results_df = pd.DataFrame(all_effects)
    results_df.to_csv(OUTPUT_DIR / "forest_plot_effect_sizes.csv", index=False, encoding='utf-8-sig')

    # Create forest plot figure
    if len(results_df) > 0:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            y_positions = range(len(results_df))
            colors = ['#1f77b4' if 'main' in eff else '#ff7f0e' for eff in results_df['effect']]

            ax.errorbar(
                results_df['beta'],
                y_positions,
                xerr=[results_df['beta'] - results_df['ci_lower'],
                      results_df['ci_upper'] - results_df['beta']],
                fmt='o',
                capsize=3,
                color='black'
            )

            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"{row['outcome_label']}\n({row['effect']})" for _, row in results_df.iterrows()])
            ax.set_xlabel('Standardized Beta (95% CI)')
            ax.set_title('Forest Plot: UCLA Effects on Executive Function\n(DASS-controlled)')

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'forest_plot.png', dpi=150, bbox_inches='tight')
            plt.close()

            if verbose:
                print(f"\n  Figure: {OUTPUT_DIR / 'forest_plot.png'}")

        except Exception as e:
            if verbose:
                print(f"  Could not create figure: {e}")

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'forest_plot_effect_sizes.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: GENDER-STRATIFIED REGRESSIONS
# =============================================================================

@register_analysis(
    name="gender_stratified",
    description="Separate DASS-controlled regressions for males and females",
    source_script="synthesis_4_gender_stratified_regressions.py"
)
def analyze_gender_stratified(verbose: bool = True) -> pd.DataFrame:
    """
    Run separate regression models for males and females.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENDER-STRATIFIED REGRESSIONS")
        print("=" * 70)

    master = load_synthesis_data()
    outcomes = get_outcomes(master)

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if outcome_col not in master.columns:
            continue

        df = master.dropna(subset=[outcome_col]).copy()

        if len(df) < 30:
            continue

        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        formula = f"{outcome_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"

        for gender, gender_label in [(1, 'Male'), (0, 'Female')]:
            df_gender = df[df['gender_male'] == gender]

            if len(df_gender) < 15:
                continue

            try:
                model = smf.ols(formula, data=df_gender).fit()

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    se = model.bse['z_ucla']
                    p = model.pvalues['z_ucla']
                    ci_low, ci_high = model.conf_int().loc['z_ucla']

                    all_results.append({
                        'outcome': outcome_col,
                        'outcome_label': outcome_label,
                        'gender': gender_label,
                        'n': len(df_gender),
                        'beta_ucla': beta,
                        'se': se,
                        'ci_lower': ci_low,
                        'ci_upper': ci_high,
                        'p': p,
                        'r_squared': model.rsquared
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    {gender_label} (n={len(df_gender)}): beta={beta:.3f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {gender_label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "gender_stratified_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gender_stratified_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 4: UCLA-DASS CORRELATIONS BY GENDER
# =============================================================================

@register_analysis(
    name="ucla_dass_correlations",
    description="UCLA-DASS correlations stratified by gender",
    source_script="synthesis_5_ucla_dass_correlation_by_gender.py"
)
def analyze_ucla_dass_correlations(verbose: bool = True) -> pd.DataFrame:
    """
    Examine UCLA-DASS correlation patterns by gender.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA-DASS CORRELATIONS BY GENDER")
        print("=" * 70)

    master = load_synthesis_data()

    dass_vars = ['dass_depression', 'dass_anxiety', 'dass_stress']
    all_results = []

    for gender, gender_label in [(1, 'Male'), (0, 'Female')]:
        df_gender = master[master['gender_male'] == gender].dropna(
            subset=['ucla_total'] + dass_vars
        )

        if len(df_gender) < 10:
            continue

        if verbose:
            print(f"\n  {gender_label} (n={len(df_gender)})")
            print("  " + "-" * 50)

        for dass_var in dass_vars:
            r, p = stats.pearsonr(df_gender['ucla_total'], df_gender[dass_var])

            # Fisher's z for CI
            z = np.arctanh(r)
            se_z = 1 / np.sqrt(len(df_gender) - 3)
            ci_z_low = z - 1.96 * se_z
            ci_z_high = z + 1.96 * se_z
            ci_r_low = np.tanh(ci_z_low)
            ci_r_high = np.tanh(ci_z_high)

            dass_label = dass_var.replace('dass_', '').capitalize()

            all_results.append({
                'gender': gender_label,
                'n': len(df_gender),
                'dass_subscale': dass_label,
                'r': r,
                'ci_lower': ci_r_low,
                'ci_upper': ci_r_high,
                'p': p
            })

            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    UCLA-{dass_label}: r={r:.3f}, 95%CI=[{ci_r_low:.3f}, {ci_r_high:.3f}], p={p:.4f}{sig}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "ucla_dass_correlations.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ucla_dass_correlations.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 5: VARIANCE TESTS
# =============================================================================

@register_analysis(
    name="variance_tests",
    description="Levene's test for variance equality across groups",
    source_script="synthesis_6_variance_tests.py"
)
def analyze_variance_tests(verbose: bool = True) -> pd.DataFrame:
    """
    Test if outcome variances differ between UCLA High/Low and Male/Female groups.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("VARIANCE EQUALITY TESTS (LEVENE'S)")
        print("=" * 70)

    master = load_synthesis_data()
    outcomes = get_outcomes(master)

    # Create UCLA groups
    ucla_median = master['ucla_total'].median()
    master['ucla_group'] = np.where(master['ucla_total'] >= ucla_median, 'High', 'Low')

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if outcome_col not in master.columns:
            continue

        df = master.dropna(subset=[outcome_col]).copy()

        if len(df) < 30:
            continue

        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        # UCLA High vs Low
        high_ucla = df[df['ucla_group'] == 'High'][outcome_col]
        low_ucla = df[df['ucla_group'] == 'Low'][outcome_col]

        if len(high_ucla) >= 5 and len(low_ucla) >= 5:
            stat, p = stats.levene(high_ucla, low_ucla)
            var_ratio = high_ucla.var() / low_ucla.var()

            all_results.append({
                'outcome': outcome_col,
                'outcome_label': outcome_label,
                'comparison': 'UCLA High vs Low',
                'var_group1': high_ucla.var(),
                'var_group2': low_ucla.var(),
                'var_ratio': var_ratio,
                'levene_stat': stat,
                'p': p
            })

            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    UCLA: Var ratio={var_ratio:.2f}, Levene p={p:.4f}{sig}")

        # Male vs Female
        male = df[df['gender_male'] == 1][outcome_col]
        female = df[df['gender_male'] == 0][outcome_col]

        if len(male) >= 5 and len(female) >= 5:
            stat, p = stats.levene(male, female)
            var_ratio = male.var() / female.var()

            all_results.append({
                'outcome': outcome_col,
                'outcome_label': outcome_label,
                'comparison': 'Male vs Female',
                'var_group1': male.var(),
                'var_group2': female.var(),
                'var_ratio': var_ratio,
                'levene_stat': stat,
                'p': p
            })

            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    Gender: Var ratio={var_ratio:.2f}, Levene p={p:.4f}{sig}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "variance_test_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'variance_test_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 6: RELIABILITY ANALYSIS
# =============================================================================

@register_analysis(
    name="reliability",
    description="Split-half reliability of EF metrics with bootstrap CI",
    source_script="synthesis_3_reliability_analysis.py"
)
def analyze_reliability(verbose: bool = True) -> pd.DataFrame:
    """
    Compute split-half reliability for EF metrics using trial-level data.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RELIABILITY ANALYSIS")
        print("=" * 70)

    all_results = []

    # Load trial data for each task
    trial_files = {
        'wcst': RESULTS_DIR / '4b_wcst_trials.csv',
        'stroop': RESULTS_DIR / '4c_stroop_trials.csv',
        'prp': RESULTS_DIR / '4a_prp_trials.csv'
    }

    for task, filepath in trial_files.items():
        if not filepath.exists():
            if verbose:
                print(f"  {task.upper()}: No trial file")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        trials = pd.read_csv(filepath, encoding='utf-8')
        trials.columns = trials.columns.str.lower()

        if 'participantid' in trials.columns:
            trials = trials.rename(columns={'participantid': 'participant_id'})

        # Determine metric column
        if task == 'wcst':
            acc_col = 'correct' if 'correct' in trials.columns else 'is_correct'
            if acc_col not in trials.columns:
                continue
            metric_col = acc_col
            metric_name = 'error_rate'
        elif task == 'stroop':
            rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
            if rt_col not in trials.columns:
                continue
            metric_col = rt_col
            metric_name = 'rt'
        else:  # prp
            rt_col = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 'rt_ms'
            if rt_col not in trials.columns:
                continue
            metric_col = rt_col
            metric_name = 't2_rt'

        # Compute split-half reliability per participant
        odd_half = []
        even_half = []
        pids = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 20:
                continue

            pdata = pdata.reset_index(drop=True)

            odd_trials = pdata.iloc[::2]
            even_trials = pdata.iloc[1::2]

            if task == 'wcst':
                odd_val = (odd_trials[metric_col] == False).mean() * 100
                even_val = (even_trials[metric_col] == False).mean() * 100
            else:
                odd_val = odd_trials[metric_col].mean()
                even_val = even_trials[metric_col].mean()

            odd_half.append(odd_val)
            even_half.append(even_val)
            pids.append(pid)

        if len(odd_half) < 20:
            if verbose:
                print(f"    Insufficient data (N={len(odd_half)})")
            continue

        # Correlation
        r, p = stats.pearsonr(odd_half, even_half)

        # Spearman-Brown correction
        r_sb = (2 * r) / (1 + r) if r > 0 else r

        all_results.append({
            'task': task,
            'metric': metric_name,
            'n_participants': len(odd_half),
            'split_half_r': r,
            'spearman_brown_r': r_sb,
            'p': p,
            'interpretation': 'Good' if r_sb >= 0.70 else ('Moderate' if r_sb >= 0.50 else 'Poor')
        })

        if verbose:
            print(f"    N = {len(odd_half)}")
            print(f"    Split-half r = {r:.3f}")
            print(f"    Spearman-Brown r = {r_sb:.3f}")
            print(f"    Reliability: {all_results[-1]['interpretation']}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "reliability_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'reliability_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 7: POWER ANALYSIS
# =============================================================================

@register_analysis(
    name="power_analysis",
    description="Post-hoc and prospective power calculations",
    source_script="synthesis_7_power_analysis.py"
)
def analyze_power(verbose: bool = True) -> pd.DataFrame:
    """
    Calculate power for observed effects and replication planning.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("POWER ANALYSIS")
        print("=" * 70)

    master = load_synthesis_data()
    outcomes = get_outcomes(master)

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if outcome_col not in master.columns:
            continue

        analysis_cols = ['z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome_col]
        df = master[analysis_cols].dropna()

        if len(df) < 30:
            continue

        n = len(df)

        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        # Fit model to get observed effect
        formula = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

        try:
            model = smf.ols(formula, data=df).fit()

            interaction_term = 'z_ucla:C(gender_male)[T.1]'
            if interaction_term not in model.params.index:
                continue

            beta = model.params[interaction_term]
            se = model.bse[interaction_term]
            t_stat = abs(beta / se)

            # Post-hoc power calculation
            df_resid = model.df_resid
            ncp = t_stat  # Non-centrality parameter
            crit_t = stats.t.ppf(0.975, df_resid)

            # Power = P(T > crit | NCP)
            power_observed = 1 - stats.nct.cdf(crit_t, df_resid, ncp) + stats.nct.cdf(-crit_t, df_resid, ncp)

            # Prospective power for N=200
            n_rep = 200
            scale_factor = np.sqrt(n_rep / n)
            ncp_rep = t_stat * scale_factor
            df_rep = n_rep - 7  # Approximate df for full model
            crit_t_rep = stats.t.ppf(0.975, df_rep)
            power_replication = 1 - stats.nct.cdf(crit_t_rep, df_rep, ncp_rep) + stats.nct.cdf(-crit_t_rep, df_rep, ncp_rep)

            all_results.append({
                'outcome': outcome_col,
                'outcome_label': outcome_label,
                'n_observed': n,
                'beta': beta,
                'se': se,
                't_stat': t_stat,
                'power_observed': power_observed,
                'n_replication': n_rep,
                'power_replication': power_replication
            })

            if verbose:
                print(f"    N = {n}, β = {beta:.3f}, t = {t_stat:.2f}")
                print(f"    Power (N={n}): {power_observed:.1%}")
                print(f"    Power (N={n_rep}): {power_replication:.1%}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "power_analysis_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'power_analysis_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 8: ROBUST SENSITIVITY
# =============================================================================

@register_analysis(
    name="robust_sensitivity",
    description="Bootstrap sensitivity analysis for effect robustness",
    source_script="synthesis_8_robust_sensitivity.py"
)
def analyze_robust_sensitivity(verbose: bool = True, n_bootstrap: int = 500) -> pd.DataFrame:
    """
    Bootstrap analysis of effect stability.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ROBUST SENSITIVITY ANALYSIS")
        print("=" * 70)
        print(f"  Bootstrap iterations: {n_bootstrap}")

    master = load_synthesis_data()
    outcomes = get_outcomes(master)

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if outcome_col not in master.columns:
            continue

        analysis_cols = ['z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome_col]
        df = master[analysis_cols].dropna()

        if len(df) < 30:
            continue

        n = len(df)

        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        formula = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

        # Bootstrap
        boot_betas = []
        boot_pvals = []
        interaction_term = 'z_ucla:C(gender_male)[T.1]'

        for i in range(n_bootstrap):
            boot_idx = np.random.choice(n, size=n, replace=True)
            df_boot = df.iloc[boot_idx]

            try:
                model_boot = smf.ols(formula, data=df_boot).fit()
                if interaction_term in model_boot.params.index:
                    boot_betas.append(model_boot.params[interaction_term])
                    boot_pvals.append(model_boot.pvalues[interaction_term])
            except:
                continue

        if len(boot_betas) < 100:
            if verbose:
                print(f"    Insufficient bootstrap samples ({len(boot_betas)})")
            continue

        boot_betas = np.array(boot_betas)
        boot_pvals = np.array(boot_pvals)

        # Statistics
        mean_beta = np.mean(boot_betas)
        ci_lower = np.percentile(boot_betas, 2.5)
        ci_upper = np.percentile(boot_betas, 97.5)
        pct_significant = (np.array(boot_pvals) < 0.05).mean() * 100
        pct_positive = (boot_betas > 0).mean() * 100

        all_results.append({
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': n,
            'n_bootstrap': len(boot_betas),
            'mean_beta': mean_beta,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pct_significant': pct_significant,
            'pct_positive': pct_positive
        })

        if verbose:
            print(f"    N = {n}, Bootstrap samples: {len(boot_betas)}")
            print(f"    Mean β = {mean_beta:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"    % Significant: {pct_significant:.1f}%")
            print(f"    % Same sign: {pct_positive:.1f}%")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "robust_sensitivity_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'robust_sensitivity_results.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run synthesis analyses.

    Parameters
    ----------
    analysis : str, optional
        Specific analysis to run. If None, runs all.
    verbose : bool
        Print progress messages.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Results from each analysis.
    """
    if verbose:
        print("=" * 70)
        print("SYNTHESIS ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
            print(f"  Description: {spec.description}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Running: {spec.name}")
            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("SYNTHESIS SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Synthesis Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")
        print(f"    Source: {spec.source_script}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesis Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                        help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
