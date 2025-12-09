"""
Stroop Neutral Condition Decomposition Suite
=============================================

Decomposes Stroop interference into facilitation and interference components
using the neutral condition as baseline.

Decomposition:
- Facilitation = RT_neutral - RT_congruent (benefit from congruent info)
- Interference = RT_incongruent - RT_neutral (cost of conflicting info)
- Total Stroop = RT_incongruent - RT_congruent = Facilitation + Interference

Key Research Questions:
1. Does UCLA affect facilitation, interference, or both?
2. Is the DDM drift rate effect (p=0.013) driven by specific components?
3. Do males/females differ in which component UCLA affects?

Analyses:
- condition_metrics: Extract RT/accuracy per condition (congruent/incongruent/neutral)
- decomposition: Calculate facilitation and interference indices
- ddm_by_condition: Fit EZ-DDM for each condition (including neutral)
- ucla_facilitation_interference: Test UCLA effects on each component (DASS-controlled)

Usage:
    python -m analysis.advanced.stroop_decomposition_suite
    python -m analysis.advanced.stroop_decomposition_suite --analysis decomposition
    python -m analysis.advanced.stroop_decomposition_suite --list

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
from typing import Dict, Optional, Callable, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX, find_interaction_term,
    apply_fdr_correction
)
from analysis.preprocessing.loaders import ensure_participant_id
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "stroop_decomposition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DDM scaling parameter
S = 0.1


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


def register_analysis(name: str, description: str, source_script: str = "stroop_decomposition_suite.py"):
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

def load_stroop_data() -> pd.DataFrame:
    """Load master dataset with standardized predictors."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    return master


def load_stroop_trials_all_conditions() -> pd.DataFrame:
    """
    Load Stroop trials including ALL conditions (congruent, incongruent, neutral).

    Unlike the standard trial loader, this retains neutral trials for decomposition.
    """
    path = RESULTS_DIR / '4c_stroop_trials.csv'
    if not path.exists():
        raise FileNotFoundError(f"Stroop trials not found: {path}")

    df = pd.read_csv(path, encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    # RT column handling
    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    if rt_col not in df.columns:
        raise KeyError("Stroop trials missing rt column")
    if rt_col != 'rt':
        if 'rt' in df.columns:
            df = df.drop(columns=['rt'])
        df = df.rename(columns={rt_col: 'rt'})

    # Condition column
    cond_col = None
    for cand in ['type', 'condition', 'cond']:
        if cand in df.columns:
            cond_col = cand
            break
    if cond_col is None:
        raise KeyError("Stroop trials missing condition column")
    if cond_col != 'condition':
        df['condition'] = df[cond_col]

    # Handle correct column - ensure boolean
    if 'correct' in df.columns:
        correct_vals = df['correct']
        if correct_vals.dtype == object:
            df['correct'] = correct_vals.astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df['correct'] = correct_vals.astype(bool)

    # Handle timeout column - ensure boolean
    if 'timeout' in df.columns:
        timeout_vals = df['timeout']
        if timeout_vals.dtype == object:
            df['timeout'] = timeout_vals.astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df['timeout'] = timeout_vals.fillna(False).astype(bool)

    # Filter valid trials
    before = len(df)
    if 'timeout' in df.columns:
        df = df[df['timeout'] == False]
    df = df[df['rt'].between(DEFAULT_RT_MIN, STROOP_RT_MAX)]

    # Ensure all three conditions present
    conditions = df['condition'].unique()
    print(f"  Found conditions: {list(conditions)}")
    print(f"  Trials after filtering: {len(df)} (from {before})")

    return df


# =============================================================================
# EZ-DDM IMPLEMENTATION (copied from ddm_suite for consistency)
# =============================================================================

def ez_diffusion(pc: float, vrt: float, mrt: float, s: float = S) -> Tuple[float, float, float]:
    """Compute EZ-diffusion model parameters."""
    if pc <= 0.5 or pc >= 1.0:
        return np.nan, np.nan, np.nan
    if vrt <= 0 or mrt <= 0:
        return np.nan, np.nan, np.nan

    try:
        L = np.log(pc / (1 - pc))
        x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt

        if x <= 0:
            return np.nan, np.nan, np.nan

        v = np.sign(pc - 0.5) * s * (x ** 0.25)
        a = s**2 * L / v

        if a <= 0:
            return np.nan, np.nan, np.nan

        y = -v * a / s**2
        mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
        t = mrt - mdt

        if not np.isfinite(v) or not np.isfinite(a) or not np.isfinite(t):
            return np.nan, np.nan, np.nan

        if t < 0:
            t = 0.0

        return v, a, t

    except Exception:
        return np.nan, np.nan, np.nan


def fit_ez_ddm_condition(trials: pd.DataFrame) -> Dict:
    """Fit EZ-DDM to condition-specific data."""
    if len(trials) < 20:
        return None

    # Convert RT to seconds
    trials = trials.copy()
    trials['rt_sec'] = trials['rt'] / 1000.0

    # Handle correct column - could be bool, string, or int
    if 'correct' in trials.columns:
        # Convert to boolean properly
        correct_vals = trials['correct']
        if correct_vals.dtype == object:
            # String values
            trials['correct_bool'] = correct_vals.astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            trials['correct_bool'] = correct_vals.astype(bool)
    else:
        return None

    correct_trials = trials[trials['correct_bool'] == True]
    if len(correct_trials) < 10:
        return None

    pc = trials['correct_bool'].mean()
    mrt = correct_trials['rt_sec'].mean()
    vrt = correct_trials['rt_sec'].var()

    v, a, t = ez_diffusion(pc, vrt, mrt)

    if np.isnan(v):
        return None

    return {
        'v': v,
        'a': a,
        't': t,
        'pc': pc,
        'mrt': mrt * 1000,
        'vrt': vrt * 1e6,
        'n_trials': len(trials),
        'n_correct': len(correct_trials)
    }


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="condition_metrics",
    description="Extract RT and accuracy for each condition (congruent/incongruent/neutral)"
)
def analyze_condition_metrics(verbose: bool = True) -> pd.DataFrame:
    """Extract participant-level metrics for each Stroop condition."""
    if verbose:
        print("\n" + "=" * 70)
        print("CONDITION-LEVEL METRICS (CONGRUENT/INCONGRUENT/NEUTRAL)")
        print("=" * 70)

    trials = load_stroop_trials_all_conditions()

    results = []

    for pid, pdata in trials.groupby('participant_id'):
        row = {'participant_id': pid}

        for cond in ['congruent', 'incongruent', 'neutral']:
            cond_data = pdata[pdata['condition'] == cond]

            if len(cond_data) < 5:
                continue

            correct_trials = cond_data[cond_data['correct'] == True]

            row[f'{cond}_n_trials'] = len(cond_data)
            row[f'{cond}_accuracy'] = cond_data['correct'].mean()
            row[f'{cond}_rt_mean'] = correct_trials['rt'].mean() if len(correct_trials) > 0 else np.nan
            row[f'{cond}_rt_sd'] = correct_trials['rt'].std() if len(correct_trials) > 0 else np.nan
            row[f'{cond}_rt_median'] = correct_trials['rt'].median() if len(correct_trials) > 0 else np.nan

        # Only include if all three conditions have data
        if all(f'{c}_rt_mean' in row and pd.notna(row.get(f'{c}_rt_mean')) for c in ['congruent', 'incongruent', 'neutral']):
            results.append(row)

    if len(results) == 0:
        if verbose:
            print("  No participants with all three conditions")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  Participants with all conditions: {len(results_df)}")
        print(f"\n  Mean RT by condition:")
        for cond in ['congruent', 'neutral', 'incongruent']:
            rt_col = f'{cond}_rt_mean'
            if rt_col in results_df.columns:
                print(f"    {cond}: {results_df[rt_col].mean():.1f} ms (SD={results_df[rt_col].std():.1f})")

        print(f"\n  Mean Accuracy by condition:")
        for cond in ['congruent', 'neutral', 'incongruent']:
            acc_col = f'{cond}_accuracy'
            if acc_col in results_df.columns:
                print(f"    {cond}: {results_df[acc_col].mean()*100:.1f}%")

    results_df.to_csv(OUTPUT_DIR / "condition_metrics.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'condition_metrics.csv'}")

    return results_df


@register_analysis(
    name="decomposition",
    description="Calculate facilitation and interference indices from neutral baseline"
)
def analyze_decomposition(verbose: bool = True) -> pd.DataFrame:
    """
    Decompose Stroop effect into facilitation and interference.

    Facilitation = RT_neutral - RT_congruent (benefit from congruent info)
    Interference = RT_incongruent - RT_neutral (cost of conflicting info)
    Total = Facilitation + Interference = RT_incongruent - RT_congruent
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STROOP EFFECT DECOMPOSITION")
        print("Facilitation = RT_neutral - RT_congruent")
        print("Interference = RT_incongruent - RT_neutral")
        print("=" * 70)

    # Load or compute condition metrics
    metrics_file = OUTPUT_DIR / "condition_metrics.csv"
    if not metrics_file.exists():
        metrics = analyze_condition_metrics(verbose=False)
    else:
        metrics = pd.read_csv(metrics_file)

    if len(metrics) == 0:
        if verbose:
            print("  No condition metrics available")
        return pd.DataFrame()

    # Compute decomposition
    metrics['facilitation'] = metrics['neutral_rt_mean'] - metrics['congruent_rt_mean']
    metrics['interference'] = metrics['incongruent_rt_mean'] - metrics['neutral_rt_mean']
    metrics['total_stroop'] = metrics['incongruent_rt_mean'] - metrics['congruent_rt_mean']

    # Verify: total = facilitation + interference
    metrics['check'] = metrics['facilitation'] + metrics['interference']
    discrepancy = (metrics['total_stroop'] - metrics['check']).abs().max()

    if verbose:
        print(f"\n  N = {len(metrics)}")
        print(f"\n  DECOMPOSITION RESULTS:")
        print(f"    Facilitation: {metrics['facilitation'].mean():.1f} ms (SD={metrics['facilitation'].std():.1f})")
        print(f"    Interference: {metrics['interference'].mean():.1f} ms (SD={metrics['interference'].std():.1f})")
        print(f"    Total Stroop: {metrics['total_stroop'].mean():.1f} ms (SD={metrics['total_stroop'].std():.1f})")
        print(f"\n    Sum check (F + I = Total): max discrepancy = {discrepancy:.4f}")

        # Proportion of total effect from each component
        mean_fac = metrics['facilitation'].mean()
        mean_int = metrics['interference'].mean()
        mean_total = metrics['total_stroop'].mean()

        if mean_total > 0:
            print(f"\n    Proportion of total effect:")
            print(f"      Facilitation: {mean_fac/mean_total*100:.1f}%")
            print(f"      Interference: {mean_int/mean_total*100:.1f}%")

        # Test if each component significantly > 0
        print(f"\n  One-sample t-tests (vs 0):")
        for comp, name in [('facilitation', 'Facilitation'), ('interference', 'Interference')]:
            t_stat, p_val = stats.ttest_1samp(metrics[comp].dropna(), 0)
            sig = "*" if p_val < 0.05 else ""
            print(f"    {name}: t={t_stat:.2f}, p={p_val:.4f}{sig}")

    # Save
    decomp_cols = ['participant_id', 'congruent_rt_mean', 'neutral_rt_mean', 'incongruent_rt_mean',
                   'facilitation', 'interference', 'total_stroop']
    decomp_df = metrics[decomp_cols].copy()
    decomp_df.to_csv(OUTPUT_DIR / "decomposition_indices.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'decomposition_indices.csv'}")

    return decomp_df


@register_analysis(
    name="ddm_by_condition",
    description="Fit EZ-DDM separately for congruent, incongruent, and neutral"
)
def analyze_ddm_by_condition(verbose: bool = True) -> pd.DataFrame:
    """
    Fit DDM parameters for each condition including neutral.

    This allows testing whether UCLA affects drift rate specifically
    in neutral, congruent, or incongruent trials.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DDM BY CONDITION (INCLUDING NEUTRAL)")
        print("=" * 70)

    trials = load_stroop_trials_all_conditions()

    results = []

    for pid, pdata in trials.groupby('participant_id'):
        row = {'participant_id': pid}

        for cond in ['congruent', 'incongruent', 'neutral']:
            cond_data = pdata[pdata['condition'] == cond]
            fit = fit_ez_ddm_condition(cond_data)

            if fit is not None:
                for key, val in fit.items():
                    row[f'{cond}_{key}'] = val

        # Compute differences
        if 'neutral_v' in row and 'congruent_v' in row:
            row['delta_v_facilitation'] = row['neutral_v'] - row['congruent_v']  # Should be negative if congruent has faster drift
        if 'neutral_v' in row and 'incongruent_v' in row:
            row['delta_v_interference'] = row['incongruent_v'] - row['neutral_v']  # Should be negative if incongruent has slower drift
        if 'congruent_v' in row and 'incongruent_v' in row:
            row['delta_v_total'] = row['incongruent_v'] - row['congruent_v']

        # Same for boundary
        if 'neutral_a' in row and 'congruent_a' in row:
            row['delta_a_facilitation'] = row['neutral_a'] - row['congruent_a']
        if 'neutral_a' in row and 'incongruent_a' in row:
            row['delta_a_interference'] = row['incongruent_a'] - row['neutral_a']

        results.append(row)

    results_df = pd.DataFrame(results)

    # Filter to participants with all conditions
    has_all = results_df[['congruent_v', 'incongruent_v', 'neutral_v']].notna().all(axis=1)
    complete_df = results_df[has_all].copy()

    if verbose:
        print(f"\n  Total participants: {len(results_df)}")
        print(f"  With all three conditions: {len(complete_df)}")

        if len(complete_df) > 0:
            print(f"\n  Mean drift rate (v) by condition:")
            for cond in ['congruent', 'neutral', 'incongruent']:
                v_col = f'{cond}_v'
                if v_col in complete_df.columns:
                    print(f"    {cond}: {complete_df[v_col].mean():.4f} (SD={complete_df[v_col].std():.4f})")

            print(f"\n  Drift rate differences:")
            if 'delta_v_facilitation' in complete_df.columns:
                mean_fac = complete_df['delta_v_facilitation'].mean()
                t_fac, p_fac = stats.ttest_1samp(complete_df['delta_v_facilitation'].dropna(), 0)
                sig = "*" if p_fac < 0.05 else ""
                print(f"    Facilitation (neutral - cong): {mean_fac:.4f}, t={t_fac:.2f}, p={p_fac:.4f}{sig}")

            if 'delta_v_interference' in complete_df.columns:
                mean_int = complete_df['delta_v_interference'].mean()
                t_int, p_int = stats.ttest_1samp(complete_df['delta_v_interference'].dropna(), 0)
                sig = "*" if p_int < 0.05 else ""
                print(f"    Interference (incong - neutral): {mean_int:.4f}, t={t_int:.2f}, p={p_int:.4f}{sig}")

    complete_df.to_csv(OUTPUT_DIR / "ddm_by_condition.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ddm_by_condition.csv'}")

    return complete_df


@register_analysis(
    name="ucla_facilitation_interference",
    description="Test UCLA effects on facilitation vs interference (DASS-controlled)"
)
def analyze_ucla_effects(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether UCLA affects facilitation, interference, or both.

    Key Research Questions:
    1. Does UCLA affect interference (conflict resolution) more than facilitation?
    2. Does the UCLA × Gender interaction differ between components?
    3. Can this explain the DDM drift rate effect?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECTS ON STROOP COMPONENTS (DASS-CONTROLLED)")
        print("=" * 70)

    # Load master data
    master = load_stroop_data()

    # Load decomposition indices
    decomp_file = OUTPUT_DIR / "decomposition_indices.csv"
    if not decomp_file.exists():
        analyze_decomposition(verbose=False)
    decomp = pd.read_csv(decomp_file)

    # Load DDM by condition
    ddm_file = OUTPUT_DIR / "ddm_by_condition.csv"
    if not ddm_file.exists():
        analyze_ddm_by_condition(verbose=False)
    ddm = pd.read_csv(ddm_file)

    # Merge
    merged = master.merge(decomp, on='participant_id', how='inner')
    merged = merged.merge(ddm[['participant_id', 'congruent_v', 'incongruent_v', 'neutral_v',
                               'delta_v_facilitation', 'delta_v_interference', 'delta_v_total']],
                          on='participant_id', how='left')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  N = {len(merged)}")

    # Outcomes to test
    outcomes = [
        # RT-based decomposition
        ('facilitation', 'Facilitation (RT)'),
        ('interference', 'Interference (RT)'),
        ('total_stroop', 'Total Stroop (RT)'),
        # DDM-based
        ('neutral_v', 'Neutral Drift Rate'),
        ('congruent_v', 'Congruent Drift Rate'),
        ('incongruent_v', 'Incongruent Drift Rate'),
        ('delta_v_interference', 'Drift Rate Interference Delta'),
    ]

    all_results = []

    for outcome_col, outcome_name in outcomes:
        if outcome_col not in merged.columns:
            continue

        if merged[outcome_col].isna().all():
            continue

        try:
            formula = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            # Extract results
            result_row = {
                'outcome': outcome_name,
                'outcome_col': outcome_col,
                'n': model.nobs,
                'r_squared': model.rsquared,
            }

            # UCLA main effect
            if 'z_ucla' in model.params:
                result_row['beta_ucla'] = model.params['z_ucla']
                result_row['se_ucla'] = model.bse['z_ucla']
                result_row['p_ucla'] = model.pvalues['z_ucla']

            # UCLA × Gender interaction
            int_term = find_interaction_term(model.params.index)
            if int_term:
                result_row['beta_interaction'] = model.params[int_term]
                result_row['se_interaction'] = model.bse[int_term]
                result_row['p_interaction'] = model.pvalues[int_term]

            all_results.append(result_row)

        except Exception as e:
            if verbose:
                print(f"  ERROR with {outcome_name}: {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Apply FDR correction
    if 'p_ucla' in results_df.columns:
        results_df = apply_fdr_correction(results_df, p_col='p_ucla')
        if 'p_fdr' in results_df.columns:
            results_df = results_df.rename(columns={'p_fdr': 'p_ucla_fdr', 'significant_fdr': 'sig_ucla_fdr'})
    if 'p_interaction' in results_df.columns:
        results_df = apply_fdr_correction(results_df, p_col='p_interaction')
        if 'p_fdr' in results_df.columns:
            results_df = results_df.rename(columns={'p_fdr': 'p_interaction_fdr', 'significant_fdr': 'sig_interaction_fdr'})

    if verbose:
        print("\n  RESULTS (UCLA main effects):")
        print("  " + "-" * 60)
        for _, row in results_df.iterrows():
            if 'beta_ucla' in row and pd.notna(row['beta_ucla']):
                sig = ""
                if row.get('p_ucla', 1) < 0.05:
                    sig = "*"
                if row.get('p_ucla_fdr', 1) < 0.05:
                    sig = "** (FDR)"
                print(f"    {row['outcome']}: beta={row['beta_ucla']:.4f}, p={row.get('p_ucla', np.nan):.4f}{sig}")

        # Highlight any significant interactions
        sig_int = results_df[results_df.get('p_interaction', pd.Series([1]*len(results_df))) < 0.10]
        if len(sig_int) > 0:
            print("\n  UCLA × Gender Interactions (p < 0.10):")
            print("  " + "-" * 60)
            for _, row in sig_int.iterrows():
                sig = "*" if row.get('p_interaction', 1) < 0.05 else "†"
                print(f"    {row['outcome']}: beta={row['beta_interaction']:.4f}, p={row.get('p_interaction', np.nan):.4f}{sig}")

    results_df.to_csv(OUTPUT_DIR / "ucla_facilitation_interference.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ucla_facilitation_interference.csv'}")

    return results_df


@register_analysis(
    name="gender_stratified",
    description="Gender-stratified analysis of facilitation vs interference"
)
def analyze_gender_stratified(verbose: bool = True) -> pd.DataFrame:
    """
    Examine facilitation/interference separately for males and females.

    This helps understand the UCLA × Gender interaction mechanism.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENDER-STRATIFIED STROOP DECOMPOSITION")
        print("=" * 70)

    # Load master data
    master = load_stroop_data()

    # Load decomposition
    decomp_file = OUTPUT_DIR / "decomposition_indices.csv"
    if not decomp_file.exists():
        analyze_decomposition(verbose=False)
    decomp = pd.read_csv(decomp_file)

    # Merge
    merged = master.merge(decomp, on='participant_id', how='inner')

    all_results = []

    for gender_val, gender_name in [(0, 'Female'), (1, 'Male')]:
        subset = merged[merged['gender_male'] == gender_val]

        if len(subset) < 15:
            if verbose:
                print(f"\n  {gender_name}: Insufficient data (N={len(subset)})")
            continue

        if verbose:
            print(f"\n  {gender_name.upper()} (N={len(subset)})")
            print("  " + "-" * 50)

        for outcome in ['facilitation', 'interference', 'total_stroop']:
            if outcome not in subset.columns:
                continue

            try:
                # DASS-controlled model (no gender term since stratified)
                formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    se = model.bse['z_ucla']
                    p = model.pvalues['z_ucla']

                    all_results.append({
                        'gender': gender_name,
                        'outcome': outcome,
                        'beta_ucla': beta,
                        'se_ucla': se,
                        'p_ucla': p,
                        'n': len(subset)
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    UCLA -> {outcome}: beta={beta:.3f}, SE={se:.3f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {outcome}: Error - {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "gender_stratified_decomposition.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gender_stratified_decomposition.csv'}")

    return results_df


@register_analysis(
    name="summary",
    description="Summary report and key findings"
)
def analyze_summary(verbose: bool = True) -> Dict:
    """Generate summary report of all findings."""
    if verbose:
        print("\n" + "=" * 70)
        print("STROOP DECOMPOSITION SUMMARY")
        print("=" * 70)

    summary = {}

    # Load all results
    decomp_file = OUTPUT_DIR / "decomposition_indices.csv"
    ucla_file = OUTPUT_DIR / "ucla_facilitation_interference.csv"
    gender_file = OUTPUT_DIR / "gender_stratified_decomposition.csv"

    if decomp_file.exists():
        decomp = pd.read_csv(decomp_file)
        summary['decomposition'] = {
            'n': len(decomp),
            'mean_facilitation': decomp['facilitation'].mean(),
            'mean_interference': decomp['interference'].mean(),
            'mean_total': decomp['total_stroop'].mean(),
            'facilitation_pct': decomp['facilitation'].mean() / decomp['total_stroop'].mean() * 100,
            'interference_pct': decomp['interference'].mean() / decomp['total_stroop'].mean() * 100,
        }

    if ucla_file.exists():
        ucla = pd.read_csv(ucla_file)
        sig_main = ucla[ucla['p_ucla'] < 0.05] if 'p_ucla' in ucla.columns else pd.DataFrame()
        sig_int = ucla[ucla['p_interaction'] < 0.05] if 'p_interaction' in ucla.columns else pd.DataFrame()

        summary['ucla_effects'] = {
            'n_tested': len(ucla),
            'n_significant_main': len(sig_main),
            'n_significant_interaction': len(sig_int),
            'significant_main_effects': sig_main['outcome'].tolist() if len(sig_main) > 0 else [],
            'significant_interactions': sig_int['outcome'].tolist() if len(sig_int) > 0 else [],
        }

    if gender_file.exists():
        gender = pd.read_csv(gender_file)
        summary['gender_stratified'] = {}
        for g in ['Male', 'Female']:
            g_data = gender[gender['gender'] == g]
            sig = g_data[g_data['p_ucla'] < 0.05] if 'p_ucla' in g_data.columns else pd.DataFrame()
            summary['gender_stratified'][g] = {
                'n_significant': len(sig),
                'significant_outcomes': sig['outcome'].tolist() if len(sig) > 0 else []
            }

    if verbose:
        if 'decomposition' in summary:
            d = summary['decomposition']
            print(f"\n  DECOMPOSITION (N={d['n']}):")
            print(f"    Total Stroop Effect: {d['mean_total']:.1f} ms")
            print(f"    - Facilitation: {d['mean_facilitation']:.1f} ms ({d['facilitation_pct']:.1f}%)")
            print(f"    - Interference: {d['mean_interference']:.1f} ms ({d['interference_pct']:.1f}%)")

        if 'ucla_effects' in summary:
            u = summary['ucla_effects']
            print(f"\n  UCLA EFFECTS:")
            print(f"    Significant main effects: {u['n_significant_main']}/{u['n_tested']}")
            if u['significant_main_effects']:
                for eff in u['significant_main_effects']:
                    print(f"      - {eff}")
            print(f"    Significant interactions: {u['n_significant_interaction']}/{u['n_tested']}")
            if u['significant_interactions']:
                for eff in u['significant_interactions']:
                    print(f"      - {eff}")

        if 'gender_stratified' in summary:
            print(f"\n  GENDER-STRATIFIED:")
            for g, g_data in summary['gender_stratified'].items():
                print(f"    {g}: {g_data['n_significant']} significant effects")
                for eff in g_data['significant_outcomes']:
                    print(f"      - {eff}")

    # Save summary
    import json
    with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'summary.json'}")

    return summary


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run Stroop decomposition analyses."""
    if verbose:
        print("=" * 70)
        print("STROOP NEUTRAL CONDITION DECOMPOSITION SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        # Run all analyses in order
        analysis_order = [
            'condition_metrics',
            'decomposition',
            'ddm_by_condition',
            'ucla_facilitation_interference',
            'gender_stratified',
            'summary',
        ]

        for name in analysis_order:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("STROOP DECOMPOSITION SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Stroop Decomposition Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stroop Decomposition Suite")
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
