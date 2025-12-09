"""
Drift-Diffusion Model (DDM) Analysis Suite
===========================================

Decomposes RT distributions into cognitive process parameters using DDM.

Parameters:
- v (drift rate): Rate of evidence accumulation (information processing efficiency)
- a (boundary separation): Response caution / speed-accuracy tradeoff
- t (non-decision time): Encoding and motor response time
- z (starting point): Response bias (often fixed at a/2)

Analyses:
- ez_ddm: Quick closed-form DDM parameter estimation (Wagenmakers et al., 2007)
- full_ddm: MLE-based full DDM fitting
- condition_ddm: Condition-specific DDM (congruent vs incongruent)
- ucla_relationship: UCLA effects on DDM parameters (DASS-controlled)

Theoretical Background:
    DDM models decision-making as noisy evidence accumulation toward
    response boundaries. Loneliness may affect:
    - Drift rate: Lower efficiency in information processing
    - Boundary: More cautious responding (higher a) or impulsive (lower a)
    - Non-decision time: Slower encoding/motor processes

Usage:
    python -m analysis.advanced.ddm_suite              # Run all
    python -m analysis.advanced.ddm_suite --analysis ez_ddm
    python -m analysis.advanced.ddm_suite --list

Reference:
    Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007).
    An EZ-diffusion model for response time and accuracy.
    Psychonomic Bulletin & Review, 14(1), 3-22.

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
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX, find_interaction_term,
    apply_fdr_correction
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "ddm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DDM scaling parameter (convention)
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


def register_analysis(name: str, description: str, source_script: str = "ddm_suite.py"):
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

def load_ddm_data() -> pd.DataFrame:
    """Load and prepare master dataset."""
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


def load_stroop_trials() -> pd.DataFrame:
    """Load Stroop trial-level data."""
    path = RESULTS_DIR / '4c_stroop_trials.csv'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8')
    df.columns = df.columns.str.lower()

    # Handle participant_id
    if 'participantid' in df.columns and 'participant_id' in df.columns:
        df = df.drop(columns=['participantid'])
    elif 'participantid' in df.columns:
        df = df.rename(columns={'participantid': 'participant_id'})

    return df


# =============================================================================
# EZ-DDM IMPLEMENTATION
# =============================================================================

def ez_diffusion(pc: float, vrt: float, mrt: float, s: float = S) -> Tuple[float, float, float]:
    """
    Compute EZ-diffusion model parameters.

    Based on Wagenmakers et al. (2007).

    Parameters:
        pc: Proportion correct (accuracy)
        vrt: Variance of RT for correct responses (in seconds^2)
        mrt: Mean RT for correct responses (in seconds)
        s: Scaling parameter (default 0.1)

    Returns:
        v: Drift rate
        a: Boundary separation
        t: Non-decision time

    Note: Returns (nan, nan, nan) if computation fails.
    """
    # Edge case handling
    if pc <= 0.5 or pc >= 1.0:
        return np.nan, np.nan, np.nan

    if vrt <= 0 or mrt <= 0:
        return np.nan, np.nan, np.nan

    try:
        # Logit of accuracy
        L = np.log(pc / (1 - pc))  # = qlogis(pc)

        # Compute x
        x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt

        if x <= 0:
            return np.nan, np.nan, np.nan

        # Drift rate
        v = np.sign(pc - 0.5) * s * (x ** 0.25)

        # Boundary separation
        a = s**2 * L / v

        if a <= 0:
            return np.nan, np.nan, np.nan

        # Non-decision time
        # MDT = a/(2v) * (1 - exp(-va/s^2)) / (1 + exp(-va/s^2))
        y = -v * a / s**2
        mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
        t = mrt - mdt

        # Sanity checks
        if not np.isfinite(v) or not np.isfinite(a) or not np.isfinite(t):
            return np.nan, np.nan, np.nan

        if t < 0:
            t = 0.0  # Non-decision time can't be negative

        return v, a, t

    except Exception:
        return np.nan, np.nan, np.nan


def fit_ez_ddm_participant(trials: pd.DataFrame, rt_col: str = 'rt_ms',
                           correct_col: str = 'correct') -> Dict:
    """
    Fit EZ-DDM to a single participant's data.

    Parameters:
        trials: Trial-level data for one participant
        rt_col: Name of RT column (in ms)
        correct_col: Name of correct/incorrect column

    Returns:
        Dictionary with DDM parameters and fit statistics
    """
    # Filter valid trials
    valid = trials[
        (trials[rt_col] > DEFAULT_RT_MIN) &
        (trials[rt_col] < STROOP_RT_MAX)
    ].copy()

    if len(valid) < 20:
        return None

    # Convert RT to seconds
    valid['rt_sec'] = valid[rt_col] / 1000.0

    # Compute summary statistics
    correct_trials = valid[valid[correct_col] == True]

    if len(correct_trials) < 10:
        return None

    pc = valid[correct_col].mean()
    mrt = correct_trials['rt_sec'].mean()
    vrt = correct_trials['rt_sec'].var()

    # Fit EZ-DDM
    v, a, t = ez_diffusion(pc, vrt, mrt)

    if np.isnan(v):
        return None

    return {
        'v': v,
        'a': a,
        't': t,
        'pc': pc,
        'mrt': mrt * 1000,  # Convert back to ms for reporting
        'vrt': vrt * 1e6,   # Variance in ms^2
        'n_trials': len(valid),
        'n_correct': len(correct_trials)
    }


def fit_ez_ddm_by_condition(trials: pd.DataFrame, rt_col: str = 'rt_ms',
                             correct_col: str = 'correct',
                             cond_col: str = 'type') -> Dict:
    """
    Fit EZ-DDM separately for each condition (congruent/incongruent).
    """
    results = {}

    for cond in ['congruent', 'incongruent']:
        cond_trials = trials[trials[cond_col] == cond]

        if len(cond_trials) < 20:
            continue

        fit = fit_ez_ddm_participant(cond_trials, rt_col, correct_col)

        if fit is not None:
            for key, val in fit.items():
                results[f'{cond}_{key}'] = val

    # Compute differences (incongruent - congruent)
    if f'congruent_v' in results and f'incongruent_v' in results:
        results['delta_v'] = results['incongruent_v'] - results['congruent_v']
        results['delta_a'] = results['incongruent_a'] - results['congruent_a']
        results['delta_t'] = results['incongruent_t'] - results['congruent_t']

    return results


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="ez_ddm",
    description="EZ-diffusion model parameter estimation (Wagenmakers et al., 2007)"
)
def analyze_ez_ddm(verbose: bool = True) -> pd.DataFrame:
    """
    Fit EZ-DDM to Stroop data for each participant.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EZ-DIFFUSION MODEL FITTING")
        print("=" * 70)

    trials = load_stroop_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient Stroop trial data")
        return pd.DataFrame()

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    if rt_col not in trials.columns:
        if verbose:
            print("  Missing RT column")
        return pd.DataFrame()

    if verbose:
        print(f"  Total trials: {len(trials)}")
        print(f"  Participants: {trials['participant_id'].nunique()}")
        print("\n  Fitting EZ-DDM to each participant...")

    results = []

    for pid, pdata in trials.groupby('participant_id'):
        fit = fit_ez_ddm_participant(pdata, rt_col=rt_col)

        if fit is not None:
            fit['participant_id'] = pid
            results.append(fit)

    if len(results) < 20:
        if verbose:
            print(f"  Only {len(results)} participants fitted successfully")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  Fitted participants: {len(results_df)}")
        print(f"  Mean drift rate (v): {results_df['v'].mean():.4f} (SD={results_df['v'].std():.4f})")
        print(f"  Mean boundary (a): {results_df['a'].mean():.4f} (SD={results_df['a'].std():.4f})")
        print(f"  Mean non-decision time (t): {results_df['t'].mean()*1000:.1f} ms (SD={results_df['t'].std()*1000:.1f})")
        print(f"  Mean accuracy: {results_df['pc'].mean()*100:.1f}%")

    results_df.to_csv(OUTPUT_DIR / "ez_ddm_parameters.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ez_ddm_parameters.csv'}")

    return results_df


@register_analysis(
    name="condition_ddm",
    description="Condition-specific DDM (congruent vs incongruent)"
)
def analyze_condition_ddm(verbose: bool = True) -> pd.DataFrame:
    """
    Fit EZ-DDM separately for congruent and incongruent conditions.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CONDITION-SPECIFIC DDM")
        print("=" * 70)

    trials = load_stroop_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient Stroop trial data")
        return pd.DataFrame()

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    cond_col = 'type' if 'type' in trials.columns else 'condition'

    if rt_col not in trials.columns or cond_col not in trials.columns:
        if verbose:
            print("  Missing required columns")
        return pd.DataFrame()

    if verbose:
        print(f"  Fitting DDM by condition (congruent/incongruent)...")

    results = []

    for pid, pdata in trials.groupby('participant_id'):
        fit = fit_ez_ddm_by_condition(pdata, rt_col=rt_col, cond_col=cond_col)

        if fit and 'delta_v' in fit:
            fit['participant_id'] = pid
            results.append(fit)

    if len(results) < 20:
        if verbose:
            print(f"  Only {len(results)} participants fitted successfully")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  Fitted participants: {len(results_df)}")

        # Congruent parameters
        print(f"\n  CONGRUENT:")
        print(f"    Drift rate: {results_df['congruent_v'].mean():.4f}")
        print(f"    Boundary: {results_df['congruent_a'].mean():.4f}")

        # Incongruent parameters
        print(f"\n  INCONGRUENT:")
        print(f"    Drift rate: {results_df['incongruent_v'].mean():.4f}")
        print(f"    Boundary: {results_df['incongruent_a'].mean():.4f}")

        # Differences
        print(f"\n  CONFLICT EFFECT (incongruent - congruent):")
        print(f"    Delta drift rate: {results_df['delta_v'].mean():.4f}")

        # Test if delta_v differs from zero
        t_stat, p_val = stats.ttest_1samp(results_df['delta_v'].dropna(), 0)
        sig = "*" if p_val < 0.05 else ""
        print(f"    Delta v vs 0: t={t_stat:.2f}, p={p_val:.4f}{sig}")

    results_df.to_csv(OUTPUT_DIR / "condition_ddm_parameters.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'condition_ddm_parameters.csv'}")

    return results_df


@register_analysis(
    name="ucla_relationship",
    description="UCLA effects on DDM parameters (DASS-controlled)"
)
def analyze_ucla_relationship(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether loneliness relates to DDM parameters.

    Hypotheses:
    - H1: Higher UCLA -> lower drift rate (slower evidence accumulation)
    - H2: Higher UCLA -> higher boundary (more cautious responding)
    - H3: Higher UCLA -> longer non-decision time (slower encoding)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECTS ON DDM PARAMETERS (DASS-CONTROLLED)")
        print("=" * 70)

    # Load master data
    master = load_ddm_data()

    # Load or fit DDM parameters
    param_files = {
        'overall': OUTPUT_DIR / "ez_ddm_parameters.csv",
        'condition': OUTPUT_DIR / "condition_ddm_parameters.csv"
    }

    all_results = []

    for param_type, param_file in param_files.items():
        if not param_file.exists():
            if verbose:
                print(f"\n  {param_type} parameters not found - fitting now...")

            if param_type == 'overall':
                analyze_ez_ddm(verbose=False)
            elif param_type == 'condition':
                analyze_condition_ddm(verbose=False)

        if not param_file.exists():
            continue

        params = pd.read_csv(param_file)
        merged = master.merge(params, on='participant_id', how='inner')

        if len(merged) < 30:
            if verbose:
                print(f"  {param_type}: Insufficient merged data (N={len(merged)})")
            continue

        if verbose:
            print(f"\n  {param_type.upper()} DDM (N={len(merged)})")
            print("  " + "-" * 50)

        # Determine which parameters to test
        if param_type == 'overall':
            param_cols = ['v', 'a', 't']
        else:
            param_cols = ['congruent_v', 'incongruent_v', 'delta_v',
                         'congruent_a', 'incongruent_a', 'delta_a',
                         'congruent_t', 'incongruent_t', 'delta_t']

        for param in param_cols:
            if param not in merged.columns:
                continue

            # DASS-controlled regression
            try:
                formula = f"{param} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta_ucla = model.params['z_ucla']
                    se_ucla = model.bse['z_ucla']
                    p_ucla = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p_ucla < 0.05 else ""
                        print(f"    UCLA -> {param}: beta={beta_ucla:.4f}, SE={se_ucla:.4f}, p={p_ucla:.4f}{sig}")

                    all_results.append({
                        'param_type': param_type,
                        'parameter': param,
                        'beta_ucla': beta_ucla,
                        'se_ucla': se_ucla,
                        'p_ucla': p_ucla,
                        'r_squared': model.rsquared,
                        'n': len(merged)
                    })

                # Check interaction
                int_term = find_interaction_term(model.params.index)
                if int_term:
                    beta_int = model.params[int_term]
                    p_int = model.pvalues[int_term]

                    if p_int < 0.10 and verbose:
                        sig = "*" if p_int < 0.05 else "†"
                        print(f"    UCLA x Gender -> {param}: beta={beta_int:.4f}, p={p_int:.4f}{sig}")

                    all_results.append({
                        'param_type': param_type,
                        'parameter': f'{param}_interaction',
                        'beta_ucla': beta_int,
                        'se_ucla': model.bse.get(int_term, np.nan),
                        'p_ucla': p_int,
                        'r_squared': model.rsquared,
                        'n': len(merged)
                    })

            except Exception as e:
                if verbose:
                    print(f"    {param}: Regression error - {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "ucla_ddm_relationship.csv", index=False, encoding='utf-8-sig')

    # Summary of significant effects
    sig_effects = results_df[results_df['p_ucla'] < 0.05]

    if len(sig_effects) > 0 and verbose:
        print("\n  SIGNIFICANT EFFECTS (p < 0.05):")
        print("  " + "-" * 50)
        for _, row in sig_effects.iterrows():
            print(f"    {row['param_type']} - {row['parameter']}: "
                  f"beta={row['beta_ucla']:.4f}, p={row['p_ucla']:.4f}")

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ucla_ddm_relationship.csv'}")

    return results_df


@register_analysis(
    name="ddm_hmm_correlation",
    description="Correlations between DDM parameters and HMM attentional states"
)
def analyze_ddm_hmm_correlation(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether DDM parameters correlate with HMM-derived attentional states.

    Theoretical link:
    - Low drift rate (v) may correlate with high lapse occupancy
    - High boundary (a) may correlate with lower lapse transitions
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DDM-HMM CORRELATION ANALYSIS")
        print("=" * 70)

    # Load DDM parameters
    param_file = OUTPUT_DIR / "ez_ddm_parameters.csv"
    if not param_file.exists():
        analyze_ez_ddm(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  DDM parameters not available")
        return pd.DataFrame()

    ddm_params = pd.read_csv(param_file)

    # Load HMM states
    hmm_file = ANALYSIS_OUTPUT_DIR / "hmm_mechanism" / "hmm_states_by_participant.csv"
    if not hmm_file.exists():
        hmm_file = ANALYSIS_OUTPUT_DIR / "hmm_attentional_states" / "hmm_merged_with_predictors.csv"

    if not hmm_file.exists():
        if verbose:
            print("  HMM states not available")
        return pd.DataFrame()

    hmm_states = pd.read_csv(hmm_file)

    # Merge
    merged = ddm_params.merge(hmm_states, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient merged data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(merged)}")
        print("\n  DDM-HMM Correlations:")
        print("  " + "-" * 60)

    ddm_cols = ['v', 'a', 't']
    hmm_cols = ['lapse_occupancy', 'trans_to_lapse', 'trans_to_focus', 'rt_diff']

    correlations = []

    for ddm_col in ddm_cols:
        if ddm_col not in merged.columns:
            continue

        for hmm_col in hmm_cols:
            if hmm_col not in merged.columns:
                continue

            valid = merged[[ddm_col, hmm_col]].dropna()

            if len(valid) < 20:
                continue

            r, p = stats.pearsonr(valid[ddm_col], valid[hmm_col])

            correlations.append({
                'ddm_param': ddm_col,
                'hmm_state': hmm_col,
                'r': r,
                'p': p,
                'n': len(valid)
            })

            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    {ddm_col} × {hmm_col}: r={r:.3f}, p={p:.4f}{sig}")

    if len(correlations) == 0:
        return pd.DataFrame()

    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(OUTPUT_DIR / "ddm_hmm_correlations.csv", index=False, encoding='utf-8-sig')

    # Key finding: drift rate and lapse
    if 'v' in corr_df['ddm_param'].values and 'lapse_occupancy' in corr_df['hmm_state'].values:
        row = corr_df[(corr_df['ddm_param'] == 'v') & (corr_df['hmm_state'] == 'lapse_occupancy')]
        if len(row) > 0 and verbose:
            r_val = row['r'].values[0]
            p_val = row['p'].values[0]
            direction = "negative" if r_val < 0 else "positive"
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"\n  KEY: Drift rate × Lapse occupancy: r={r_val:.3f}{sig} ({direction})")
            print(f"        Higher drift rate associated with {'lower' if r_val < 0 else 'higher'} lapse time")

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ddm_hmm_correlations.csv'}")

    return corr_df


@register_analysis(
    name="drift_boundary_tradeoff",
    description="Analyze drift rate vs boundary separation tradeoff"
)
def analyze_drift_boundary_tradeoff(verbose: bool = True) -> Dict:
    """
    Analyze the speed-accuracy tradeoff through drift-boundary relationship.

    Higher boundary = more cautious, lower drift = slower accumulation.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DRIFT-BOUNDARY TRADEOFF ANALYSIS")
        print("=" * 70)

    # Load DDM parameters
    param_file = OUTPUT_DIR / "ez_ddm_parameters.csv"
    if not param_file.exists():
        analyze_ez_ddm(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  DDM parameters not available")
        return {}

    params = pd.read_csv(param_file)
    master = load_ddm_data()
    merged = master.merge(params, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return {}

    results = {'n': len(merged)}

    # 1. Basic v-a correlation
    r_va, p_va = stats.pearsonr(merged['v'], merged['a'])
    results['r_v_a'] = r_va
    results['p_v_a'] = p_va

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"\n  1. Drift-Boundary Correlation:")
        sig = "*" if p_va < 0.05 else ""
        print(f"     r(v, a) = {r_va:.3f}, p = {p_va:.4f}{sig}")

    # 2. Efficiency ratio: v/a (higher = more efficient)
    merged['efficiency'] = merged['v'] / merged['a']
    results['mean_efficiency'] = merged['efficiency'].mean()
    results['sd_efficiency'] = merged['efficiency'].std()

    if verbose:
        print(f"\n  2. Processing Efficiency (v/a ratio):")
        print(f"     Mean = {results['mean_efficiency']:.3f} (SD = {results['sd_efficiency']:.3f})")

    # 3. UCLA effect on efficiency
    try:
        formula = "efficiency ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=merged).fit(cov_type='HC3')

        if 'z_ucla' in model.params:
            results['beta_ucla_efficiency'] = model.params['z_ucla']
            results['p_ucla_efficiency'] = model.pvalues['z_ucla']

            if verbose:
                sig = "*" if results['p_ucla_efficiency'] < 0.05 else ""
                print(f"\n  3. UCLA → Efficiency (DASS-controlled):")
                print(f"     beta = {results['beta_ucla_efficiency']:.4f}, p = {results['p_ucla_efficiency']:.4f}{sig}")

        # Check interaction
        int_term = find_interaction_term(model.params.index)
        if int_term:
            results['beta_interaction_efficiency'] = model.params[int_term]
            results['p_interaction_efficiency'] = model.pvalues[int_term]

            if verbose and results['p_interaction_efficiency'] < 0.10:
                sig = "*" if results['p_interaction_efficiency'] < 0.05 else "†"
                print(f"     UCLA × Gender: beta = {results['beta_interaction_efficiency']:.4f}, p = {results['p_interaction_efficiency']:.4f}{sig}")

    except Exception as e:
        if verbose:
            print(f"  Efficiency regression error: {e}")

    # 4. Gender differences in tradeoff
    if verbose:
        print(f"\n  4. Gender Differences:")

    for gender, label in [(0, 'Female'), (1, 'Male')]:
        subset = merged[merged['gender_male'] == gender]
        if len(subset) < 20:
            continue

        r_g, p_g = stats.pearsonr(subset['v'], subset['a'])
        results[f'r_v_a_{label.lower()}'] = r_g
        results[f'p_v_a_{label.lower()}'] = p_g

        if verbose:
            sig = "*" if p_g < 0.05 else ""
            print(f"     {label}: r(v, a) = {r_g:.3f}, p = {p_g:.4f}{sig}")

    # Save results
    import json
    with open(OUTPUT_DIR / "drift_boundary_tradeoff.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'drift_boundary_tradeoff.json'}")

    return results


@register_analysis(
    name="gender_stratified_ddm",
    description="Gender-stratified DDM analysis for UCLA effects"
)
def analyze_gender_stratified_ddm(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze UCLA effects on DDM parameters separately by gender.

    Based on the UCLA × Gender interaction found in WCST PE, we test whether
    DDM parameters also show gender-specific UCLA effects.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENDER-STRATIFIED DDM ANALYSIS")
        print("=" * 70)

    # Load DDM parameters
    param_file = OUTPUT_DIR / "ez_ddm_parameters.csv"
    if not param_file.exists():
        analyze_ez_ddm(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  DDM parameters not available")
        return pd.DataFrame()

    params = pd.read_csv(param_file)
    master = load_ddm_data()
    merged = master.merge(params, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    all_results = []

    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = merged[merged['gender_male'] == gender]

        if len(subset) < 15:
            if verbose:
                print(f"  {label.upper()}: Insufficient data (N={len(subset)})")
            continue

        if verbose:
            print(f"\n  {label.upper()} (N={len(subset)})")
            print("  " + "-" * 50)

        for param in ['v', 'a', 't']:
            if param not in subset.columns:
                continue

            try:
                # Simple UCLA model without gender interaction (already stratified)
                formula = f"{param} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    se = model.bse['z_ucla']
                    p = model.pvalues['z_ucla']

                    all_results.append({
                        'gender': label,
                        'parameter': param,
                        'beta_ucla': beta,
                        'se_ucla': se,
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(subset)
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    UCLA → {param}: beta={beta:.4f}, SE={se:.4f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {param}: Error - {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Compare male vs female effects
    if verbose:
        print("\n  GENDER COMPARISON:")
        print("  " + "-" * 50)

        for param in ['v', 'a', 't']:
            male_row = results_df[(results_df['gender'] == 'male') & (results_df['parameter'] == param)]
            female_row = results_df[(results_df['gender'] == 'female') & (results_df['parameter'] == param)]

            if len(male_row) > 0 and len(female_row) > 0:
                m_beta = male_row['beta_ucla'].values[0]
                f_beta = female_row['beta_ucla'].values[0]
                m_p = male_row['p_ucla'].values[0]
                f_p = female_row['p_ucla'].values[0]

                m_sig = "*" if m_p < 0.05 else ""
                f_sig = "*" if f_p < 0.05 else ""

                print(f"    {param}: Male β={m_beta:.4f}{m_sig} vs Female β={f_beta:.4f}{f_sig}")

    results_df.to_csv(OUTPUT_DIR / "gender_stratified_ddm.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gender_stratified_ddm.csv'}")

    return results_df


@register_analysis(
    name="mediation_drift_stroop",
    description="Mediation analysis: UCLA → Drift Rate → Stroop Interference"
)
def analyze_mediation_drift_stroop(verbose: bool = True) -> Dict:
    """
    Test whether drift rate mediates the UCLA → Stroop interference relationship.

    Path model:
    - c: UCLA → Stroop interference (total effect)
    - a: UCLA → Drift rate
    - b: Drift rate → Stroop interference (controlling for UCLA)
    - c': UCLA → Stroop interference (controlling for Drift rate = direct effect)
    - Indirect effect = a × b
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MEDIATION: UCLA → DRIFT RATE → STROOP INTERFERENCE")
        print("=" * 70)

    # Load DDM parameters
    param_file = OUTPUT_DIR / "ez_ddm_parameters.csv"
    if not param_file.exists():
        analyze_ez_ddm(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  DDM parameters not available")
        return {}

    params = pd.read_csv(param_file)
    master = load_ddm_data()
    merged = master.merge(params, on='participant_id', how='inner')

    # Need Stroop interference
    if 'stroop_interference' not in merged.columns and 'stroop_effect' not in merged.columns:
        if verbose:
            print("  Stroop interference measure not available")
        return {}

    outcome = 'stroop_interference' if 'stroop_interference' in merged.columns else 'stroop_effect'

    # Standardize for interpretability
    merged['z_v'] = (merged['v'] - merged['v'].mean()) / merged['v'].std()
    merged[f'z_{outcome}'] = (merged[outcome] - merged[outcome].mean()) / merged[outcome].std()

    valid = merged.dropna(subset=['z_ucla', 'z_v', f'z_{outcome}', 'z_dass_dep', 'z_dass_anx', 'z_dass_str'])

    if len(valid) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(valid)})")
        return {}

    results = {'n': len(valid)}

    if verbose:
        print(f"  N = {len(valid)}")
        print(f"  Mediator: Drift rate (v)")
        print(f"  Outcome: {outcome}")

    # Path c: Total effect (UCLA → Stroop)
    try:
        model_c = smf.ols(f"z_{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                          data=valid).fit(cov_type='HC3')
        c = model_c.params.get('z_ucla', np.nan)
        c_p = model_c.pvalues.get('z_ucla', np.nan)
        results['c_total'] = c
        results['c_p'] = c_p

        if verbose:
            sig = "*" if c_p < 0.05 else ""
            print(f"\n  Path c (total effect): β={c:.4f}, p={c_p:.4f}{sig}")
    except:
        return {}

    # Path a: UCLA → Drift
    try:
        model_a = smf.ols("z_v ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                          data=valid).fit(cov_type='HC3')
        a = model_a.params.get('z_ucla', np.nan)
        a_p = model_a.pvalues.get('z_ucla', np.nan)
        results['a_path'] = a
        results['a_p'] = a_p

        if verbose:
            sig = "*" if a_p < 0.05 else ""
            print(f"  Path a (UCLA → Drift): β={a:.4f}, p={a_p:.4f}{sig}")
    except:
        return results

    # Path b and c': Drift → Stroop (controlling UCLA)
    try:
        model_bc = smf.ols(f"z_{outcome} ~ z_ucla + z_v + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                           data=valid).fit(cov_type='HC3')
        b = model_bc.params.get('z_v', np.nan)
        b_p = model_bc.pvalues.get('z_v', np.nan)
        c_prime = model_bc.params.get('z_ucla', np.nan)
        c_prime_p = model_bc.pvalues.get('z_ucla', np.nan)

        results['b_path'] = b
        results['b_p'] = b_p
        results['c_prime_direct'] = c_prime
        results['c_prime_p'] = c_prime_p

        if verbose:
            sig = "*" if b_p < 0.05 else ""
            print(f"  Path b (Drift → Stroop): β={b:.4f}, p={b_p:.4f}{sig}")
            sig = "*" if c_prime_p < 0.05 else ""
            print(f"  Path c' (direct effect): β={c_prime:.4f}, p={c_prime_p:.4f}{sig}")
    except:
        return results

    # Indirect effect = a × b
    indirect = a * b
    results['indirect_ab'] = indirect

    # Sobel test for indirect effect
    try:
        se_a = model_a.bse.get('z_ucla', np.nan)
        se_b = model_bc.bse.get('z_v', np.nan)
        se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
        z_sobel = indirect / se_indirect
        p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))

        results['se_indirect'] = se_indirect
        results['z_sobel'] = z_sobel
        results['p_sobel'] = p_sobel

        if verbose:
            sig = "*" if p_sobel < 0.05 else ""
            print(f"\n  Indirect effect (a×b): β={indirect:.4f}, SE={se_indirect:.4f}")
            print(f"  Sobel test: z={z_sobel:.3f}, p={p_sobel:.4f}{sig}")
    except:
        pass

    # Proportion mediated
    if abs(c) > 0.001:
        prop_mediated = indirect / c
        results['proportion_mediated'] = prop_mediated

        if verbose:
            print(f"\n  Proportion mediated: {prop_mediated*100:.1f}%")

    # Bootstrap confidence interval for indirect effect
    if verbose:
        print("\n  Bootstrap 95% CI (1000 iterations)...")

    n_boot = 1000
    boot_indirect = []

    for _ in range(n_boot):
        idx = np.random.choice(len(valid), size=len(valid), replace=True)
        boot_data = valid.iloc[idx]

        try:
            m_a = smf.ols("z_v ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                          data=boot_data).fit()
            m_bc = smf.ols(f"z_{outcome} ~ z_ucla + z_v + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                           data=boot_data).fit()

            boot_a = m_a.params.get('z_ucla', np.nan)
            boot_b = m_bc.params.get('z_v', np.nan)
            boot_indirect.append(boot_a * boot_b)
        except:
            continue

    if len(boot_indirect) > 100:
        ci_low = np.percentile(boot_indirect, 2.5)
        ci_high = np.percentile(boot_indirect, 97.5)
        results['ci_low'] = ci_low
        results['ci_high'] = ci_high

        # Significant if CI doesn't include 0
        results['boot_significant'] = (ci_low > 0) or (ci_high < 0)

        if verbose:
            sig = "*" if results['boot_significant'] else ""
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]{sig}")

    # Save results
    import json
    with open(OUTPUT_DIR / "mediation_drift_stroop.json", 'w', encoding='utf-8') as f:
        # Convert numpy types for JSON
        json_results = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, np.integer)):
                json_results[k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                json_results[k] = bool(v)
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'mediation_drift_stroop.json'}")

    return results


@register_analysis(
    name="condition_by_gender",
    description="Condition-specific DDM analysis by gender (congruent vs incongruent)"
)
def analyze_condition_by_gender(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze whether UCLA affects condition-specific DDM parameters differently by gender.

    Tests:
    - Does female vulnerability in drift rate appear specifically in high-conflict (incongruent) trials?
    - Is there a 3-way interaction: UCLA × condition × gender?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CONDITION-SPECIFIC DDM BY GENDER")
        print("=" * 70)

    # Load condition DDM parameters
    param_file = OUTPUT_DIR / "condition_ddm_parameters.csv"
    if not param_file.exists():
        analyze_condition_ddm(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  Condition DDM parameters not available")
        return pd.DataFrame()

    params = pd.read_csv(param_file)
    master = load_ddm_data()
    merged = master.merge(params, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(merged)}")

    all_results = []

    # Condition-specific drift rates
    drift_cols = ['congruent_v', 'incongruent_v', 'delta_v']

    # Gender-stratified analysis
    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = merged[merged['gender_male'] == gender]

        if len(subset) < 15:
            continue

        if verbose:
            print(f"\n  {label.upper()} (N={len(subset)})")
            print("  " + "-" * 50)

        for drift_col in drift_cols:
            if drift_col not in subset.columns:
                continue

            try:
                formula = f"{drift_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    all_results.append({
                        'gender': label,
                        'parameter': drift_col,
                        'beta_ucla': beta,
                        'se_ucla': model.bse['z_ucla'],
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(subset)
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else "†" if p < 0.10 else ""
                        cond_label = {'congruent_v': 'Congruent drift',
                                     'incongruent_v': 'Incongruent drift',
                                     'delta_v': 'Conflict cost (Δv)'}.get(drift_col, drift_col)
                        print(f"    UCLA → {cond_label}: β={beta:.4f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {drift_col}: Error - {e}")

    # Test 3-way interactions
    if verbose:
        print("\n  UCLA × Condition Interactions (pooled):")
        print("  " + "-" * 50)

    for drift_col in drift_cols:
        if drift_col not in merged.columns:
            continue

        try:
            formula = f"{drift_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            int_term = find_interaction_term(model.params.index)
            if int_term:
                beta_int = model.params[int_term]
                p_int = model.pvalues[int_term]

                all_results.append({
                    'gender': 'interaction',
                    'parameter': drift_col,
                    'beta_ucla': beta_int,
                    'se_ucla': model.bse[int_term],
                    'p_ucla': p_int,
                    'r_squared': model.rsquared,
                    'n': len(merged)
                })

                if verbose:
                    sig = "*" if p_int < 0.05 else "†" if p_int < 0.10 else ""
                    print(f"    UCLA × Gender → {drift_col}: β={beta_int:.4f}, p={p_int:.4f}{sig}")

        except Exception:
            continue

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "condition_by_gender.csv", index=False, encoding='utf-8-sig')

    # Summary
    if verbose:
        print("\n  FEMALE-SPECIFIC DRIFT RATE EFFECTS:")
        print("  " + "-" * 50)
        female_sig = results_df[(results_df['gender'] == 'female') & (results_df['p_ucla'] < 0.05)]
        if len(female_sig) > 0:
            for _, row in female_sig.iterrows():
                direction = "lower" if row['beta_ucla'] < 0 else "higher"
                print(f"    Higher UCLA → {direction} {row['parameter']}")
        else:
            print("    No significant effects at p < 0.05")

        print(f"\n  Output: {OUTPUT_DIR / 'condition_by_gender.csv'}")

    return results_df


@register_analysis(
    name="efficiency_by_gender",
    description="Processing efficiency (v/a ratio) analysis by gender"
)
def analyze_efficiency_by_gender(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze processing efficiency (drift/boundary ratio) by gender.

    Efficiency = v/a
    Higher efficiency = faster accumulation relative to caution level
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PROCESSING EFFICIENCY BY GENDER")
        print("=" * 70)

    # Load DDM parameters
    param_file = OUTPUT_DIR / "ez_ddm_parameters.csv"
    if not param_file.exists():
        analyze_ez_ddm(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  DDM parameters not available")
        return pd.DataFrame()

    params = pd.read_csv(param_file)
    master = load_ddm_data()
    merged = master.merge(params, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    # Compute efficiency (ratio) and log-transformed efficiency
    # Log transformation for ratio variables helps with non-linearity
    merged['efficiency'] = merged['v'] / merged['a']
    merged['log_efficiency'] = np.log(merged['efficiency'].clip(lower=0.001))

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"  Mean efficiency: {merged['efficiency'].mean():.3f} (SD={merged['efficiency'].std():.3f})")
        print(f"  Using LOG-TRANSFORMED efficiency for regression (corrects non-linearity)")

    all_results = []

    # Gender-stratified
    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = merged[merged['gender_male'] == gender]

        if len(subset) < 15:
            continue

        if verbose:
            print(f"\n  {label.upper()} (N={len(subset)})")
            print(f"    Mean efficiency: {subset['efficiency'].mean():.3f}")
            print("  " + "-" * 50)

        # UCLA → efficiency (using log-transformed efficiency for linearity)
        try:
            formula = "log_efficiency ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=subset).fit(cov_type='HC3')

            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']

                all_results.append({
                    'gender': label,
                    'parameter': 'efficiency',
                    'beta_ucla': beta,
                    'se_ucla': model.bse['z_ucla'],
                    'p_ucla': p,
                    'r_squared': model.rsquared,
                    'n': len(subset)
                })

                if verbose:
                    sig = "*" if p < 0.05 else "†" if p < 0.10 else ""
                    direction = "lower" if beta < 0 else "higher"
                    print(f"    UCLA → Efficiency: β={beta:.4f}, p={p:.4f}{sig}")
                    if p < 0.10:
                        print(f"      Higher UCLA → {direction} processing efficiency")

        except Exception as e:
            if verbose:
                print(f"    Efficiency regression error: {e}")

        # Also test v and a separately
        for param, param_name in [('v', 'drift'), ('a', 'boundary')]:
            if param not in subset.columns:
                continue

            try:
                formula = f"{param} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    all_results.append({
                        'gender': label,
                        'parameter': param,
                        'beta_ucla': beta,
                        'se_ucla': model.bse['z_ucla'],
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(subset)
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else "†" if p < 0.10 else ""
                        print(f"    UCLA → {param_name}: β={beta:.4f}, p={p:.4f}{sig}")

            except Exception:
                continue

    # Interaction test
    if verbose:
        print("\n  UCLA × Gender Interaction:")
        print("  " + "-" * 50)

    try:
        formula = "log_efficiency ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=merged).fit(cov_type='HC3')

        int_term = find_interaction_term(model.params.index)
        if int_term:
            beta_int = model.params[int_term]
            p_int = model.pvalues[int_term]

            all_results.append({
                'gender': 'interaction',
                'parameter': 'efficiency',
                'beta_ucla': beta_int,
                'se_ucla': model.bse[int_term],
                'p_ucla': p_int,
                'r_squared': model.rsquared,
                'n': len(merged)
            })

            if verbose:
                sig = "*" if p_int < 0.05 else "†" if p_int < 0.10 else ""
                print(f"    UCLA × Gender → Efficiency: β={beta_int:.4f}, p={p_int:.4f}{sig}")

    except Exception as e:
        if verbose:
            print(f"    Interaction error: {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Apply FDR correction (Benjamini-Hochberg) across all tests
    if 'p_ucla' in results_df.columns and len(results_df) > 1:
        results_df = apply_fdr_correction(results_df, p_col='p_ucla')

        if verbose:
            print("\n  FDR Correction Applied (Benjamini-Hochberg):")
            print("  " + "-" * 50)
            sig_raw = (results_df['p_ucla'] < 0.05).sum()
            sig_fdr = (results_df['p_fdr'] < 0.05).sum() if 'p_fdr' in results_df.columns else 0
            print(f"    Significant (raw p < 0.05): {sig_raw}/{len(results_df)}")
            print(f"    Significant (FDR q < 0.05): {sig_fdr}/{len(results_df)}")

    results_df.to_csv(OUTPUT_DIR / "efficiency_by_gender.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'efficiency_by_gender.csv'}")

    return results_df


@register_analysis(
    name="parameter_correlations",
    description="Correlations between DDM parameters and other EF measures"
)
def analyze_parameter_correlations(verbose: bool = True) -> pd.DataFrame:
    """
    Examine correlations between DDM parameters and traditional EF measures.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DDM PARAMETER CORRELATIONS")
        print("=" * 70)

    master = load_ddm_data()

    # Load DDM parameters
    param_file = OUTPUT_DIR / "ez_ddm_parameters.csv"
    if not param_file.exists():
        analyze_ez_ddm(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  DDM parameters not available")
        return pd.DataFrame()

    params = pd.read_csv(param_file)
    merged = master.merge(params, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print("  Insufficient data")
        return pd.DataFrame()

    # EF measures to correlate with
    ef_measures = ['stroop_interference', 'pe_rate', 'prp_bottleneck',
                   'wcst_pe', 'stroop_effect']

    ddm_params = ['v', 'a', 't']

    correlations = []

    if verbose:
        print(f"  N = {len(merged)}")
        print("\n  Correlations:")
        print("  " + "-" * 60)

    for ddm_param in ddm_params:
        if ddm_param not in merged.columns:
            continue

        for ef_measure in ef_measures:
            if ef_measure not in merged.columns:
                continue

            valid = merged[[ddm_param, ef_measure]].dropna()

            if len(valid) < 20:
                continue

            r, p = stats.pearsonr(valid[ddm_param], valid[ef_measure])

            correlations.append({
                'ddm_param': ddm_param,
                'ef_measure': ef_measure,
                'r': r,
                'p': p,
                'n': len(valid)
            })

            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    {ddm_param} x {ef_measure}: r={r:.3f}, p={p:.4f}{sig}")

    if len(correlations) == 0:
        if verbose:
            print("    No valid correlations computed")
        return pd.DataFrame()

    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(OUTPUT_DIR / "ddm_ef_correlations.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ddm_ef_correlations.csv'}")

    return corr_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run DDM analyses.
    """
    if verbose:
        print("=" * 70)
        print("DRIFT-DIFFUSION MODEL ANALYSIS SUITE")
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
            'ez_ddm',
            'condition_ddm',
            'ucla_relationship',
            'ddm_hmm_correlation',
            'drift_boundary_tradeoff',
            'gender_stratified_ddm',
            'condition_by_gender',
            'efficiency_by_gender',
            'mediation_drift_stroop',
            'parameter_correlations'
        ]

        for name in analysis_order:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("DDM SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable DDM Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDM Analysis Suite")
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
