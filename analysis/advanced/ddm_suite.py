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
    DEFAULT_RT_MIN, STROOP_RT_MAX
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
                interaction_term = 'z_ucla:C(gender_male)[T.1]'
                if interaction_term in model.params:
                    beta_int = model.params[interaction_term]
                    p_int = model.pvalues[interaction_term]

                    if p_int < 0.10 and verbose:
                        sig = "*" if p_int < 0.05 else "†"
                        print(f"    UCLA x Gender -> {param}: beta={beta_int:.4f}, p={p_int:.4f}{sig}")

                    all_results.append({
                        'param_type': param_type,
                        'parameter': f'{param}_interaction',
                        'beta_ucla': beta_int,
                        'se_ucla': model.bse.get(interaction_term, np.nan),
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
