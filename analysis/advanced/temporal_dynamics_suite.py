"""
Temporal Dynamics Suite
======================

Trial-level time series analysis of RT patterns and attentional dynamics.

Research Question:
------------------
Do lonely individuals show different temporal patterns in their performance?
- Higher autocorrelation (more serial dependency, less responsive to feedback)
- Different long-range correlations (DFA scaling exponent)
- Fewer adaptive change points (less flexible performance adjustment)

Analyses:
---------
1. autocorrelation: RT autocorrelation function (ACF) lag 1-10
2. dfa: Detrended Fluctuation Analysis - long-range correlations
3. change_point: Change point detection - adaptive transitions
4. variability_decomposition: Decompose IIV into slow drift and fast fluctuations

Usage:
    python -m analysis.advanced.temporal_dynamics_suite
    python -m analysis.advanced.temporal_dynamics_suite --analysis dfa
    python -m analysis.advanced.temporal_dynamics_suite --list

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
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import detrend
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, DEFAULT_RT_MAX, PRP_RT_MAX, STROOP_RT_MAX, find_interaction_term
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "temporal_dynamics"
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


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_temporal_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load master dataset and trial-level data for all tasks."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    # Load trial data for each task
    trial_data = {}

    trial_files = {
        'stroop': (RESULTS_DIR / '4c_stroop_trials.csv', STROOP_RT_MAX),
        'wcst': (RESULTS_DIR / '4b_wcst_trials.csv', DEFAULT_RT_MAX),
        'prp': (RESULTS_DIR / '4a_prp_trials.csv', PRP_RT_MAX)
    }

    for task, (filepath, rt_max) in trial_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath, encoding='utf-8')
            df.columns = df.columns.str.lower()

            # Normalize participant_id
            if 'participantid' in df.columns and 'participant_id' not in df.columns:
                df = df.rename(columns={'participantid': 'participant_id'})
            elif 'participantid' in df.columns:
                df = df.drop(columns=['participantid'])

            # Get RT column
            if task == 'prp':
                rt_col = 't2_rt_ms' if 't2_rt_ms' in df.columns else 't2_rt'
            else:
                rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'

            if rt_col in df.columns:
                df['rt'] = df[rt_col]
                df = df[(df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < rt_max)].copy()

            # Sort trials
            sort_cols = ['participant_id']
            for cand in ['trialindex', 'trial_index', 'timestamp']:
                if cand in df.columns:
                    sort_cols.append(cand)
                    break
            df = df.sort_values(sort_cols).reset_index(drop=True)

            # Add trial number per participant
            df['trial_num'] = df.groupby('participant_id').cumcount() + 1

            trial_data[task] = df

    return master, trial_data


def get_dass_controlled_formula(outcome: str) -> str:
    """Get DASS-controlled regression formula per CLAUDE.md."""
    return f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"


# =============================================================================
# ANALYSIS 1: AUTOCORRELATION FUNCTION
# =============================================================================

def compute_acf(series: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Compute autocorrelation function for a time series."""
    n = len(series)
    if n < max_lag + 10:
        return np.full(max_lag, np.nan)

    series = series - np.mean(series)
    acf = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        if n - lag < 5:
            acf[lag - 1] = np.nan
        else:
            acf[lag - 1] = np.corrcoef(series[:-lag], series[lag:])[0, 1]

    return acf


@register_analysis(
    name="autocorrelation",
    description="RT autocorrelation function (ACF) lag 1-10"
)
def analyze_autocorrelation(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze RT autocorrelation patterns.

    Hypothesis: Lonely individuals show higher autocorrelation
    (more serial dependency, less responsive to feedback).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: RT AUTOCORRELATION FUNCTION")
        print("=" * 70)

    master, trial_data = load_temporal_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        if task not in trial_data:
            continue

        trials = trial_data[task]

        if 'rt' not in trials.columns:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        acf_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            rts = pdata['rt'].values
            acf = compute_acf(rts, max_lag=10)

            result = {
                'participant_id': pid,
                'acf_lag1': acf[0],
                'acf_lag2': acf[1],
                'acf_lag3': acf[2],
                'acf_mean': np.nanmean(acf[:5]),  # Mean of lag 1-5
                'n_trials': len(pdata)
            }
            acf_results.append(result)

        if len(acf_results) < 20:
            continue

        acf_df = pd.DataFrame(acf_results)
        merged = master.merge(acf_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean lag-1 autocorrelation: {merged['acf_lag1'].mean():.3f}")

        # Test UCLA effects
        for metric in ['acf_lag1', 'acf_mean']:
            merged_clean = merged.dropna(subset=[metric])
            if len(merged_clean) < 20:
                continue

            try:
                formula = get_dass_controlled_formula(metric)
                model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

                int_term = find_interaction_term(model.params.index)
                terms_to_check = ['z_ucla']
                if int_term:
                    terms_to_check.append(int_term)
                for term in terms_to_check:
                    if term in model.params:
                        beta = model.params[term]
                        p = model.pvalues[term]

                        label = term.replace('z_ucla', 'UCLA').replace(':C(gender_male)[T.1]', ' x Male').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                        if verbose:
                            sig = "*" if p < 0.05 else ""
                            print(f"    {metric} ~ {label}: beta={beta:.4f}, p={p:.4f}{sig}")

                        all_results.append({
                            'task': task,
                            'outcome': metric,
                            'term': label,
                            'beta': beta,
                            'p': p,
                            'n': len(merged_clean)
                        })

            except Exception as e:
                if verbose:
                    print(f"    Error for {metric}: {e}")

        # Save per-participant data
        acf_df.to_csv(OUTPUT_DIR / f'acf_by_participant_{task}.csv', index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "autocorrelation_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'autocorrelation_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 2: DETRENDED FLUCTUATION ANALYSIS (DFA)
# =============================================================================

def compute_dfa(series: np.ndarray, min_scale: int = 4, max_scale: int = None) -> float:
    """
    Compute DFA scaling exponent (alpha).

    Alpha > 0.5: Long-range positive correlations (more persistent)
    Alpha = 0.5: White noise (uncorrelated)
    Alpha < 0.5: Anti-correlated

    Parameters
    ----------
    series : array
        Time series data
    min_scale : int
        Minimum box size
    max_scale : int
        Maximum box size (default: n/4)

    Returns
    -------
    float
        DFA scaling exponent (alpha)
    """
    n = len(series)
    if n < 50:
        return np.nan

    if max_scale is None:
        max_scale = n // 4

    # Step 1: Integrate the series (cumulative sum of deviations from mean)
    y = np.cumsum(series - np.mean(series))

    # Step 2: Divide into boxes and compute fluctuation for each scale
    scales = []
    fluctuations = []

    for scale in range(min_scale, min(max_scale, n // 4) + 1):
        # Number of complete boxes
        n_boxes = n // scale

        if n_boxes < 2:
            continue

        rms = []
        for i in range(n_boxes):
            start = i * scale
            end = start + scale
            segment = y[start:end]

            # Linear detrend
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residuals = segment - trend

            rms.append(np.sqrt(np.mean(residuals ** 2)))

        if len(rms) > 0:
            scales.append(scale)
            fluctuations.append(np.mean(rms))

    if len(scales) < 3:
        return np.nan

    # Step 3: Fit log-log plot
    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)

    # Linear regression
    slope, intercept, r, p, se = stats.linregress(log_scales, log_fluct)

    return slope  # This is the DFA alpha


@register_analysis(
    name="dfa",
    description="Detrended Fluctuation Analysis - long-range correlations"
)
def analyze_dfa(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze long-range correlations using DFA.

    Hypothesis: Lonely individuals show different DFA alpha
    (possibly higher = more persistent/less flexible).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: DETRENDED FLUCTUATION ANALYSIS (DFA)")
        print("=" * 70)

    master, trial_data = load_temporal_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        if task not in trial_data:
            continue

        trials = trial_data[task]

        if 'rt' not in trials.columns:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        dfa_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 50:  # Need at least 50 trials for reliable DFA
                continue

            rts = pdata['rt'].values
            alpha = compute_dfa(rts)

            if not np.isnan(alpha):
                dfa_results.append({
                    'participant_id': pid,
                    'dfa_alpha': alpha,
                    'n_trials': len(pdata)
                })

        if len(dfa_results) < 20:
            if verbose:
                print(f"    Insufficient data ({len(dfa_results)} participants)")
            continue

        dfa_df = pd.DataFrame(dfa_results)
        merged = master.merge(dfa_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean DFA alpha: {merged['dfa_alpha'].mean():.3f}")
            print(f"    (alpha=0.5 = white noise, alpha>0.5 = persistent)")

        # Test UCLA effects
        try:
            formula = get_dass_controlled_formula('dfa_alpha')
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            int_term = find_interaction_term(model.params.index)
            terms_to_check = ['z_ucla']
            if int_term:
                terms_to_check.append(int_term)
            for term in terms_to_check:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = term.replace('z_ucla', 'UCLA').replace(':C(gender_male)[T.1]', ' x Male').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    dfa_alpha ~ {label}: beta={beta:.4f}, p={p:.4f}{sig}")

                    all_results.append({
                        'task': task,
                        'outcome': 'dfa_alpha',
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged)
                    })

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

        # Save per-participant data
        dfa_df.to_csv(OUTPUT_DIR / f'dfa_by_participant_{task}.csv', index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "dfa_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'dfa_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: CHANGE POINT DETECTION
# =============================================================================

def detect_change_points(series: np.ndarray, threshold: float = 2.0) -> List[int]:
    """
    Simple change point detection based on cumulative sum (CUSUM).

    Parameters
    ----------
    series : array
        Time series data
    threshold : float
        Z-score threshold for detecting change points

    Returns
    -------
    list
        Indices of detected change points
    """
    n = len(series)
    if n < 20:
        return []

    # Standardize
    series = (series - np.mean(series)) / np.std(series)

    # Cumulative sum
    cusum = np.cumsum(series)

    # Detect significant shifts
    change_points = []
    window = max(5, n // 10)

    for i in range(window, n - window):
        # Compare mean before and after
        before = np.mean(series[max(0, i-window):i])
        after = np.mean(series[i:min(n, i+window)])
        diff = abs(after - before)

        if diff > threshold:
            # Check if this is a local maximum
            if len(change_points) == 0 or i - change_points[-1] > window:
                change_points.append(i)

    return change_points


@register_analysis(
    name="change_point",
    description="Change point detection - adaptive transitions"
)
def analyze_change_points(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze change point frequency in performance.

    Hypothesis: Lonely individuals show fewer change points
    (less adaptive performance adjustment).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: CHANGE POINT DETECTION")
        print("=" * 70)

    master, trial_data = load_temporal_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        if task not in trial_data:
            continue

        trials = trial_data[task]

        if 'rt' not in trials.columns:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        cp_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            rts = pdata['rt'].values
            change_points = detect_change_points(rts)

            # Normalize by number of trials
            n_trials = len(pdata)
            cp_rate = len(change_points) / n_trials * 100  # per 100 trials

            cp_results.append({
                'participant_id': pid,
                'n_change_points': len(change_points),
                'cp_rate': cp_rate,
                'n_trials': n_trials
            })

        if len(cp_results) < 20:
            continue

        cp_df = pd.DataFrame(cp_results)
        merged = master.merge(cp_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean change points: {merged['n_change_points'].mean():.1f}")
            print(f"    Mean CP rate: {merged['cp_rate'].mean():.2f} per 100 trials")

        # Test UCLA effects
        for metric in ['n_change_points', 'cp_rate']:
            try:
                formula = get_dass_controlled_formula(metric)
                model = smf.ols(formula, data=merged).fit(cov_type='HC3')

                int_term = find_interaction_term(model.params.index)
                terms_to_check = ['z_ucla']
                if int_term:
                    terms_to_check.append(int_term)
                for term in terms_to_check:
                    if term in model.params:
                        beta = model.params[term]
                        p = model.pvalues[term]

                        label = term.replace('z_ucla', 'UCLA').replace(':C(gender_male)[T.1]', ' x Male').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                        if verbose:
                            sig = "*" if p < 0.05 else ""
                            print(f"    {metric} ~ {label}: beta={beta:.3f}, p={p:.4f}{sig}")

                        all_results.append({
                            'task': task,
                            'outcome': metric,
                            'term': label,
                            'beta': beta,
                            'p': p,
                            'n': len(merged)
                        })

            except Exception as e:
                if verbose:
                    print(f"    Error for {metric}: {e}")

        # Save per-participant data
        cp_df.to_csv(OUTPUT_DIR / f'change_points_by_participant_{task}.csv', index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "change_point_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'change_point_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 4: VARIABILITY DECOMPOSITION
# =============================================================================

@register_analysis(
    name="variability_decomposition",
    description="Decompose IIV into slow drift and fast fluctuations"
)
def analyze_variability_decomposition(verbose: bool = True) -> pd.DataFrame:
    """
    Decompose RT variability into slow (trend) and fast (trial-to-trial) components.

    Hypothesis: UCLA might specifically predict fast fluctuations
    (attentional instability) rather than slow drift (fatigue).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: VARIABILITY DECOMPOSITION")
        print("=" * 70)

    master, trial_data = load_temporal_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        if task not in trial_data:
            continue

        trials = trial_data[task]

        if 'rt' not in trials.columns:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        var_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            rts = pdata['rt'].values
            n = len(rts)

            # Total variability
            total_var = np.var(rts)

            # Slow drift: variance of moving average (window = 10 trials)
            window = min(10, n // 4)
            if window >= 3:
                moving_avg = np.convolve(rts, np.ones(window)/window, mode='valid')
                slow_var = np.var(moving_avg)
            else:
                slow_var = np.nan

            # Fast fluctuations: variance of residuals after removing trend
            trend = np.convolve(rts, np.ones(window)/window, mode='same') if window >= 3 else np.mean(rts)
            residuals = rts - trend
            fast_var = np.var(residuals)

            # Trial-to-trial changes
            diff = np.diff(rts)
            diff_var = np.var(diff)

            var_results.append({
                'participant_id': pid,
                'total_var': total_var,
                'slow_var': slow_var,
                'fast_var': fast_var,
                'diff_var': diff_var,
                'fast_ratio': fast_var / total_var if total_var > 0 else np.nan,
                'n_trials': n
            })

        if len(var_results) < 20:
            continue

        var_df = pd.DataFrame(var_results)
        merged = master.merge(var_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean fast/total ratio: {merged['fast_ratio'].mean():.3f}")

        # Test UCLA effects on each component
        for metric in ['total_var', 'slow_var', 'fast_var', 'fast_ratio']:
            merged_clean = merged.dropna(subset=[metric])
            if len(merged_clean) < 20:
                continue

            try:
                formula = get_dass_controlled_formula(metric)
                model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

                int_term = find_interaction_term(model.params.index)
                terms_to_check = ['z_ucla']
                if int_term:
                    terms_to_check.append(int_term)
                for term in terms_to_check:
                    if term in model.params:
                        beta = model.params[term]
                        p = model.pvalues[term]

                        label = term.replace('z_ucla', 'UCLA').replace(':C(gender_male)[T.1]', ' x Male').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                        if verbose:
                            sig = "*" if p < 0.05 else ""
                            print(f"    {metric} ~ {label}: beta={beta:.3f}, p={p:.4f}{sig}")

                        all_results.append({
                            'task': task,
                            'outcome': metric,
                            'term': label,
                            'beta': beta,
                            'p': p,
                            'n': len(merged_clean)
                        })

            except Exception as e:
                if verbose:
                    print(f"    Error for {metric}: {e}")

        # Save per-participant data
        var_df.to_csv(OUTPUT_DIR / f'variability_by_participant_{task}.csv', index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "variability_decomposition_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'variability_decomposition_results.csv'}")

    return results_df


# =============================================================================
# SUMMARY VISUALIZATION
# =============================================================================

def create_summary_visualization(verbose: bool = True) -> None:
    """Create summary figure for temporal dynamics analysis."""
    if verbose:
        print("\n" + "=" * 70)
        print("CREATING SUMMARY VISUALIZATION")
        print("=" * 70)

    # Collect all results
    result_files = {
        'ACF': OUTPUT_DIR / 'autocorrelation_results.csv',
        'DFA': OUTPUT_DIR / 'dfa_results.csv',
        'Change Points': OUTPUT_DIR / 'change_point_results.csv',
        'Variability': OUTPUT_DIR / 'variability_decomposition_results.csv'
    }

    all_effects = []

    for analysis_name, filepath in result_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Filter for interaction term
            interaction = df[df['term'].str.contains('x Male', case=False, na=False)]
            if len(interaction) > 0:
                for _, row in interaction.iterrows():
                    all_effects.append({
                        'Analysis': analysis_name,
                        'Task': row.get('task', ''),
                        'Outcome': row.get('outcome', ''),
                        'Beta': row['beta'],
                        'P-value': row['p']
                    })

    if len(all_effects) == 0:
        if verbose:
            print("  No results to visualize")
        return

    effects_df = pd.DataFrame(all_effects)

    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(effects_df) * 0.4)))

    y_pos = range(len(effects_df))
    colors = ['#E74C3C' if p < 0.05 else '#3498DB' for p in effects_df['P-value']]

    ax.barh(y_pos, effects_df['Beta'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    labels = [f"{row['Analysis']} ({row['Task']}): {row['Outcome']}" for _, row in effects_df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('UCLA x Gender Interaction (Beta)')
    ax.set_title('Temporal Dynamics: UCLA x Gender Effects\n(Red = p < 0.05)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_dynamics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {OUTPUT_DIR / 'temporal_dynamics_summary.png'}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run temporal dynamics analyses."""
    if verbose:
        print("=" * 70)
        print("TEMPORAL DYNAMICS SUITE")
        print("=" * 70)
        print("\nExploring time series properties of trial-level RT data")

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        # Run all analyses
        for name, spec in ANALYSES.items():
            try:
                if verbose:
                    print(f"\n--- Running: {spec.description} ---")
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

        # Create summary visualization
        create_summary_visualization(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TEMPORAL DYNAMICS SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Temporal Dynamics Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Dynamics Analysis Suite")
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
