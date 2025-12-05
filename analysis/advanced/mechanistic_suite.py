"""
Mechanistic Analysis Suite
==========================

Advanced mechanistic decomposition analyses for UCLA × Executive Function.

Consolidates Tier 1 & Tier 2 analyses:
- tier1_fatigue_analysis.py
- tier1_speed_accuracy_tradeoff.py
- tier1_autocorrelation.py
- tier1_lapse_decomposition.py
- tier1_exgaussian_cross_task.py
- tier2_meta_analysis.py
- tier2_pre_error_trajectories.py
- tier2_prp_coupling.py
- tier2_hmm_attentional_states.py

Usage:
    python -m analysis.advanced.mechanistic_suite                    # Run all
    python -m analysis.advanced.mechanistic_suite --analysis fatigue
    python -m analysis.advanced.mechanistic_suite --list

    from analysis.advanced import mechanistic_suite
    mechanistic_suite.run('fatigue')
    mechanistic_suite.run()  # All analyses

NOTE: These are ADVANCED analyses requiring careful interpretation.

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
from scipy.stats import median_abs_deviation
from scipy.optimize import minimize
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR, find_interaction_term
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "mechanistic_suite"
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
    tier: int


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, source_script: str, tier: int = 1):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            source_script=source_script,
            tier=tier
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mechanistic_data() -> pd.DataFrame:
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


def load_trial_data(task: str) -> pd.DataFrame:
    """Load trial-level data for specific task."""
    trial_files = {
        'prp': RESULTS_DIR / '4a_prp_trials.csv',
        'stroop': RESULTS_DIR / '4c_stroop_trials.csv',
        'wcst': RESULTS_DIR / '4b_wcst_trials.csv'
    }

    if task not in trial_files:
        return pd.DataFrame()

    path = trial_files[task]
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8')
    df.columns = df.columns.str.lower()

    # Handle duplicate participant_id columns (some CSVs have both participantid and participant_id)
    if 'participantid' in df.columns and 'participant_id' in df.columns:
        df = df.drop(columns=['participantid'])
    elif 'participantid' in df.columns:
        df = df.rename(columns={'participantid': 'participant_id'})

    return df


# =============================================================================
# TIER 1 ANALYSES
# =============================================================================

@register_analysis(
    name="fatigue",
    description="Time-on-task fatigue effects and UCLA moderation",
    source_script="tier1_fatigue_analysis.py",
    tier=1
)
def analyze_fatigue(verbose: bool = True) -> pd.DataFrame:
    """
    Does loneliness amplify fatigue-related performance decline?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 1: FATIGUE ANALYSIS")
        print("=" * 70)

    master = load_mechanistic_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient trial data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        # Add trial number per participant
        trials = trials.sort_values(['participant_id', 'trial_index' if 'trial_index' in trials.columns else trials.columns[0]])
        trials['trial_num'] = trials.groupby('participant_id').cumcount() + 1

        # Get RT column
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        if rt_col not in trials.columns:
            if verbose:
                print(f"    No RT column found")
            continue

        # Filter valid RTs
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Calculate fatigue slope per participant (RT trend over trials)
        fatigue_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 20:
                continue

            # Linear regression: RT ~ trial_num
            slope, intercept, r, p, se = stats.linregress(pdata['trial_num'], pdata[rt_col])

            fatigue_results.append({
                'participant_id': pid,
                'fatigue_slope': slope,  # Positive = slowing over time
                'fatigue_intercept': intercept,
                'fatigue_r': r,
                'n_trials': len(pdata)
            })

        if len(fatigue_results) < 20:
            continue

        fatigue_df = pd.DataFrame(fatigue_results)

        # Merge with master
        merged = master.merge(fatigue_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean fatigue slope: {merged['fatigue_slope'].mean():.3f} ms/trial")

        # Test UCLA effect on fatigue (DASS-controlled)
        try:
            formula = "fatigue_slope ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit()

            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']

                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"    UCLA -> Fatigue: beta={beta:.3f}, p={p:.4f}{sig}")

                all_results.append({
                    'task': task,
                    'analysis': 'fatigue_slope',
                    'beta_ucla': beta,
                    'p_ucla': p,
                    'n': len(merged)
                })

        except Exception as e:
            if verbose:
                print(f"    Regression error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "fatigue_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'fatigue_results.csv'}")

    return results_df


@register_analysis(
    name="speed_accuracy",
    description="Speed-accuracy tradeoff patterns by UCLA",
    source_script="tier1_speed_accuracy_tradeoff.py",
    tier=1
)
def analyze_speed_accuracy(verbose: bool = True) -> pd.DataFrame:
    """
    Do lonely individuals show different speed-accuracy tradeoff patterns?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 1: SPEED-ACCURACY TRADEOFF")
        print("=" * 70)

    master = load_mechanistic_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        acc_col = 'correct' if 'correct' in trials.columns else 'is_correct'

        if rt_col not in trials.columns or acc_col not in trials.columns:
            continue

        # Filter valid
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Calculate per-participant SAT metrics
        sat_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 20:
                continue

            mean_rt = pdata[rt_col].mean()
            accuracy = pdata[acc_col].mean()

            # Inverse efficiency score (IES) = RT / Accuracy
            ies = mean_rt / accuracy if accuracy > 0 else np.nan

            sat_results.append({
                'participant_id': pid,
                'sat_mean_rt': mean_rt,
                'sat_accuracy': accuracy,
                'sat_ies': ies,
                'sat_n_trials': len(pdata)
            })

        if len(sat_results) < 20:
            continue

        sat_df = pd.DataFrame(sat_results)
        merged = master.merge(sat_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")

        # RT-Accuracy correlation by UCLA group
        merged['ucla_high'] = merged['ucla_total'] > merged['ucla_total'].median()

        for ucla_grp in [True, False]:
            grp_data = merged[merged['ucla_high'] == ucla_grp]
            if len(grp_data) >= 10:
                r, p = stats.pearsonr(grp_data['sat_mean_rt'], grp_data['sat_accuracy'])
                grp_label = "High UCLA" if ucla_grp else "Low UCLA"

                if verbose:
                    print(f"    {grp_label}: RT-Acc r={r:.3f}, p={p:.4f}")

                all_results.append({
                    'task': task,
                    'ucla_group': grp_label,
                    'r_rt_accuracy': r,
                    'p': p,
                    'n': len(grp_data)
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "speed_accuracy_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'speed_accuracy_results.csv'}")

    return results_df


@register_analysis(
    name="autocorrelation",
    description="Trial-to-trial RT autocorrelation patterns",
    source_script="tier1_autocorrelation.py",
    tier=1
)
def analyze_autocorrelation(verbose: bool = True) -> pd.DataFrame:
    """
    Do lonely individuals show higher RT autocorrelation (more serial dependency)?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 1: RT AUTOCORRELATION")
        print("=" * 70)

    master = load_mechanistic_data()
    all_results = []

    for task in ['stroop', 'prp']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else ('t2_rt_ms' if 't2_rt_ms' in trials.columns else 'rt')
        if rt_col not in trials.columns:
            continue

        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Calculate lag-1 autocorrelation per participant
        ac_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.sort_values(pdata.columns[0])  # Sort by first column
            rts = pdata[rt_col].values

            # Lag-1 autocorrelation
            if len(rts) > 2:
                ac1 = np.corrcoef(rts[:-1], rts[1:])[0, 1]
            else:
                ac1 = np.nan

            ac_results.append({
                'participant_id': pid,
                'autocorr_lag1': ac1,
                'n_trials': len(pdata)
            })

        if len(ac_results) < 20:
            continue

        ac_df = pd.DataFrame(ac_results)
        merged = master.merge(ac_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean autocorr: {merged['autocorr_lag1'].mean():.3f}")

        # Test UCLA effect
        merged = merged.dropna(subset=['autocorr_lag1'])
        r, p = stats.pearsonr(merged['z_ucla'], merged['autocorr_lag1'])

        if verbose:
            sig = "*" if p < 0.05 else ""
            print(f"    UCLA vs Autocorr: r={r:.3f}, p={p:.4f}{sig}")

        all_results.append({
            'task': task,
            'analysis': 'autocorrelation',
            'r_ucla_autocorr': r,
            'p': p,
            'n': len(merged)
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "autocorrelation_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'autocorrelation_results.csv'}")

    return results_df


# =============================================================================
# TIER 2 ANALYSES
# =============================================================================

@register_analysis(
    name="pre_error_trajectories",
    description="RT/accuracy trajectories leading up to errors",
    source_script="tier2_pre_error_trajectories.py",
    tier=2
)
def analyze_pre_error_trajectories(verbose: bool = True) -> pd.DataFrame:
    """
    Do pre-error RT patterns differ between lonely and non-lonely individuals?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 2: PRE-ERROR TRAJECTORIES")
        print("=" * 70)

    master = load_mechanistic_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        acc_col = 'correct' if 'correct' in trials.columns else 'is_correct'

        if rt_col not in trials.columns or acc_col not in trials.columns:
            continue

        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Calculate pre-error RT slope per participant
        pre_error_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.reset_index(drop=True)

            # Find error indices
            error_indices = pdata[pdata[acc_col] == False].index.tolist()

            if len(error_indices) < 3:
                continue

            # Get RT in trials before errors (n-3 to n-1)
            pre_error_rts = []
            for err_idx in error_indices:
                if err_idx >= 3:
                    pre_rts = pdata.loc[err_idx-3:err_idx-1, rt_col].values
                    pre_error_rts.extend(pre_rts)

            if len(pre_error_rts) < 5:
                continue

            # Average pre-error RT
            mean_pre_error_rt = np.mean(pre_error_rts)

            # Compare to overall mean
            overall_mean_rt = pdata[rt_col].mean()
            pre_error_elevation = mean_pre_error_rt - overall_mean_rt

            pre_error_results.append({
                'participant_id': pid,
                'mean_pre_error_rt': mean_pre_error_rt,
                'overall_mean_rt': overall_mean_rt,
                'pre_error_elevation': pre_error_elevation,
                'n_errors': len(error_indices)
            })

        if len(pre_error_results) < 20:
            continue

        pre_err_df = pd.DataFrame(pre_error_results)
        merged = master.merge(pre_err_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean pre-error elevation: {merged['pre_error_elevation'].mean():.1f} ms")

        # Test UCLA effect
        r, p = stats.pearsonr(merged['z_ucla'], merged['pre_error_elevation'])

        if verbose:
            sig = "*" if p < 0.05 else ""
            print(f"    UCLA vs Pre-error elevation: r={r:.3f}, p={p:.4f}{sig}")

        all_results.append({
            'task': task,
            'analysis': 'pre_error_elevation',
            'r_ucla': r,
            'p': p,
            'n': len(merged)
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "pre_error_trajectories.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'pre_error_trajectories.csv'}")

    return results_df


@register_analysis(
    name="cross_task_coupling",
    description="Cross-task performance coupling patterns",
    source_script="tier2_prp_coupling.py",
    tier=2
)
def analyze_cross_task_coupling(verbose: bool = True) -> pd.DataFrame:
    """
    Do performance patterns couple across tasks? Does UCLA affect coupling?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 2: CROSS-TASK COUPLING")
        print("=" * 70)

    master = load_mechanistic_data()

    # Get EF metrics
    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 2:
        if verbose:
            print("  Need at least 2 EF metrics")
        return pd.DataFrame()

    df = master.dropna(subset=available + ['ucla_total']).copy()

    if len(df) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")

    # Compute coupling matrix by UCLA group
    df['ucla_high'] = df['ucla_total'] > df['ucla_total'].median()

    results = []

    for ucla_grp in [True, False]:
        grp_data = df[df['ucla_high'] == ucla_grp]
        grp_label = "High UCLA" if ucla_grp else "Low UCLA"

        if len(grp_data) < 15:
            continue

        if verbose:
            print(f"\n  {grp_label} (n={len(grp_data)})")

        # Correlation matrix
        for i, col1 in enumerate(available):
            for j, col2 in enumerate(available):
                if i < j:
                    r, p = stats.pearsonr(grp_data[col1], grp_data[col2])

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    {col1} x {col2}: r={r:.3f}{sig}")

                    results.append({
                        'ucla_group': grp_label,
                        'task1': col1,
                        'task2': col2,
                        'r': r,
                        'p': p,
                        'n': len(grp_data)
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "cross_task_coupling.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'cross_task_coupling.csv'}")

    return results_df


@register_analysis(
    name="lapse_decomposition",
    description="Lapse frequency vs magnitude decomposition",
    source_script="tier1_lapse_decomposition.py",
    tier=1
)
def analyze_lapse_decomposition(verbose: bool = True) -> pd.DataFrame:
    """
    Decompose RT variability into lapse frequency and magnitude.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 1: LAPSE DECOMPOSITION")
        print("=" * 70)

    master = load_mechanistic_data()
    all_results = []

    for task in ['stroop', 'wcst']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient trial data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        if rt_col not in trials.columns:
            continue

        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Classify lapses per participant
        lapse_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 20:
                continue

            rt_series = pdata[rt_col]
            median_rt = rt_series.median()
            mad_rt = median_abs_deviation(rt_series, nan_policy='omit')

            if mad_rt == 0:
                mad_rt = rt_series.std() * 0.6745

            threshold = median_rt + 2.5 * mad_rt
            is_lapse = rt_series >= threshold

            lapse_freq = is_lapse.mean() * 100
            normal_rt = rt_series[~is_lapse].mean()
            lapse_rt = rt_series[is_lapse].mean() if is_lapse.sum() > 0 else np.nan
            lapse_magnitude = lapse_rt - normal_rt if not np.isnan(lapse_rt) else np.nan

            lapse_results.append({
                'participant_id': pid,
                'lapse_freq': lapse_freq,
                'lapse_magnitude': lapse_magnitude,
                'n_trials': len(pdata)
            })

        if len(lapse_results) < 20:
            continue

        lapse_df = pd.DataFrame(lapse_results)
        merged = master.merge(lapse_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean lapse frequency: {merged['lapse_freq'].mean():.1f}%")

        # Test UCLA effects
        for metric in ['lapse_freq', 'lapse_magnitude']:
            merged_clean = merged.dropna(subset=[metric])
            if len(merged_clean) < 20:
                continue

            try:
                formula = f"{metric} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged_clean).fit()

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    UCLA -> {metric}: β={beta:.3f}, p={p:.4f}{sig}")

                    all_results.append({
                        'task': task,
                        'metric': metric,
                        'beta_ucla': beta,
                        'p_ucla': p,
                        'n': len(merged_clean)
                    })

            except Exception as e:
                if verbose:
                    print(f"    Regression error for {metric}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "lapse_decomposition.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'lapse_decomposition.csv'}")

    return results_df


@register_analysis(
    name="exgaussian",
    description="Ex-Gaussian RT decomposition (mu, sigma, tau) with FDR correction",
    source_script="tier1_exgaussian_cross_task.py (enhanced)",
    tier=1
)
def analyze_exgaussian(verbose: bool = True) -> pd.DataFrame:
    """
    Ex-Gaussian Decomposition of RT Distributions with FDR Correction.

    Enhanced from legacy scripts with:
    1. Full parameter extraction: mu (decision), sigma (motor), tau (lapses)
    2. FDR correction for 9 tests (3 tasks × 3 parameters)
    3. Bootstrap SE for parameter uncertainty weighting
    4. DASS-controlled regression with robust SE

    Parameters:
    - mu: Mean of Gaussian component (decision/motor execution)
    - sigma: SD of Gaussian component (motor variability)
    - tau: Exponential tail (attentional lapses)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 1: EX-GAUSSIAN DECOMPOSITION (FDR-corrected)")
        print("=" * 70)
        print("  Parameters: mu (decision), sigma (motor), tau (lapses)")

    def fit_exgaussian_moments(rts, n_bootstrap=100):
        """
        Fit ex-Gaussian using method of moments with bootstrap SE.

        Returns:
            dict with mu, sigma, tau and their bootstrap SEs
        """
        rts = np.array(rts)
        rts = rts[(rts > 0) & np.isfinite(rts)]

        if len(rts) < 30:
            return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan,
                    'mu_se': np.nan, 'sigma_se': np.nan, 'tau_se': np.nan}

        def _fit_moments(data):
            """
            Method of moments estimation for ex-Gaussian parameters.

            Based on Heathcote (1996) "RTSYS: A DOS application for the analysis
            of reaction time data" and the standard ex-Gaussian moments:
              E[X] = mu + tau
              Var[X] = sigma^2 + tau^2
              M3 (3rd central moment) = 2 * tau^3

            Therefore: tau = (M3 / 2)^(1/3)
            """
            m1 = np.mean(data)
            m2 = np.var(data)
            m3 = stats.moment(data, moment=3)  # 3rd central moment

            # Negative skewness: ex-Gaussian not appropriate, return Gaussian
            if m3 <= 0:
                return m1, np.sqrt(m2), 0.0

            # tau from 3rd central moment: M3 = 2*tau^3 => tau = (M3/2)^(1/3)
            tau = (m3 / 2) ** (1/3)
            mu = m1 - tau
            sigma_sq = m2 - tau**2

            if sigma_sq <= 0:
                # Fallback: allocate variance proportionally
                sigma = np.sqrt(m2) * 0.5
            else:
                sigma = np.sqrt(sigma_sq)

            return mu, sigma, tau

        # Main fit
        mu, sigma, tau = _fit_moments(rts)

        # Bootstrap for SE
        boot_mus, boot_sigmas, boot_taus = [], [], []
        np.random.seed(42)

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(rts, size=len(rts), replace=True)
            try:
                b_mu, b_sigma, b_tau = _fit_moments(boot_sample)
                if not np.isnan(b_mu) and not np.isnan(b_tau):
                    boot_mus.append(b_mu)
                    boot_sigmas.append(b_sigma)
                    boot_taus.append(b_tau)
            except Exception:
                continue

        return {
            'mu': mu,
            'sigma': sigma,
            'tau': tau,
            'mu_se': np.std(boot_mus) if len(boot_mus) > 10 else np.nan,
            'sigma_se': np.std(boot_sigmas) if len(boot_sigmas) > 10 else np.nan,
            'tau_se': np.std(boot_taus) if len(boot_taus) > 10 else np.nan
        }

    master = load_mechanistic_data()
    all_results = []

    for task in ['stroop', 'wcst', 'prp']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient trial data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else ('t2_rt_ms' if 't2_rt_ms' in trials.columns else 'rt')
        if rt_col not in trials.columns:
            continue

        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Fit ex-Gaussian per participant with bootstrap SE
        exg_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            fit_result = fit_exgaussian_moments(pdata[rt_col].values, n_bootstrap=100)

            if np.isnan(fit_result['mu']):
                continue

            exg_results.append({
                'participant_id': pid,
                'exg_mu': fit_result['mu'],
                'exg_sigma': fit_result['sigma'],
                'exg_tau': fit_result['tau'],
                'exg_mu_se': fit_result['mu_se'],
                'exg_sigma_se': fit_result['sigma_se'],
                'exg_tau_se': fit_result['tau_se'],
                'n_trials': len(pdata)
            })

        if len(exg_results) < 20:
            continue

        exg_df = pd.DataFrame(exg_results)
        merged = master.merge(exg_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean: mu={merged['exg_mu'].mean():.0f}ms, "
                  f"sigma={merged['exg_sigma'].mean():.0f}ms, "
                  f"tau={merged['exg_tau'].mean():.0f}ms")

        # Save task-specific ex-gaussian parameters
        exg_df.to_csv(OUTPUT_DIR / f"exgaussian_{task}.csv", index=False, encoding='utf-8-sig')

        # Test UCLA effects on ALL three parameters
        parameters = [
            ('exg_mu', 'mu (decision)'),
            ('exg_sigma', 'sigma (motor)'),
            ('exg_tau', 'tau (lapse)')
        ]

        for param_col, param_label in parameters:
            try:
                formula = f"{param_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'parameter': param_col.replace('exg_', ''),
                    'parameter_label': param_label,
                    'n': len(merged),
                    'mean_value': merged[param_col].mean(),
                    'sd_value': merged[param_col].std()
                }

                if 'z_ucla' in model.params:
                    result['beta_ucla'] = model.params['z_ucla']
                    result['se_ucla'] = model.bse['z_ucla']
                    result['p_ucla'] = model.pvalues['z_ucla']

                int_term = find_interaction_term(model.params.index)
                if int_term:
                    result['beta_interaction'] = model.params[int_term]
                    result['p_interaction'] = model.pvalues[int_term]

                result['r_squared'] = model.rsquared

                all_results.append(result)

                if verbose and result.get('p_ucla', 1) < 0.10:
                    sig = "***" if result['p_ucla'] < 0.001 else "**" if result['p_ucla'] < 0.01 else "*" if result['p_ucla'] < 0.05 else "+"
                    print(f"    UCLA → {param_label}: β={result['beta_ucla']:.3f}, p={result['p_ucla']:.4f} {sig}")

            except Exception as e:
                if verbose:
                    print(f"    Regression error for {param_label}: {e}")

    # Apply FDR correction
    results_df = pd.DataFrame(all_results)

    if len(results_df) > 1 and 'p_ucla' in results_df.columns:
        if verbose:
            print("\n  Applying FDR correction (Benjamini-Hochberg)...")

        p_vals = results_df['p_ucla'].dropna().values
        n_tests = len(p_vals)

        if n_tests > 1:
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_vals)
            ranks = np.empty(n_tests)
            ranks[sorted_idx] = np.arange(1, n_tests + 1)
            q_vals = p_vals * n_tests / ranks

            # Ensure monotonicity (use .copy() to avoid aliasing)
            q_sorted = q_vals[sorted_idx].copy()
            for i in range(n_tests - 2, -1, -1):
                q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
            q_vals[sorted_idx] = q_sorted
            q_vals = np.clip(q_vals, 0, 1)

            results_df.loc[results_df['p_ucla'].notna(), 'p_ucla_fdr'] = q_vals
            results_df['sig_fdr'] = results_df['p_ucla_fdr'] < 0.05

            if verbose:
                print(f"    {n_tests} tests corrected")
                sig_fdr = results_df[results_df['sig_fdr'] == True]
                if len(sig_fdr) > 0:
                    print("\n    *** FDR-Significant Results ***")
                    for _, row in sig_fdr.iterrows():
                        print(f"      {row['task']} {row['parameter']}: "
                              f"β={row['beta_ucla']:.3f}, p_fdr={row['p_ucla_fdr']:.4f}")
                else:
                    print("    No FDR-significant UCLA main effects")

    # Also apply FDR to interaction terms if present
    if 'p_interaction' in results_df.columns:
        p_int_vals = results_df['p_interaction'].dropna().values
        if len(p_int_vals) > 1:
            n_int = len(p_int_vals)
            sorted_idx = np.argsort(p_int_vals)
            ranks = np.empty(n_int)
            ranks[sorted_idx] = np.arange(1, n_int + 1)
            q_int_vals = p_int_vals * n_int / ranks

            # Enforce monotonicity: q[i] <= q[i+1] in sorted order
            q_sorted = q_int_vals[sorted_idx].copy()
            for i in range(n_int - 2, -1, -1):
                q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
            q_int_vals[sorted_idx] = q_sorted
            q_int_vals = np.clip(q_int_vals, 0, 1)

            results_df.loc[results_df['p_interaction'].notna(), 'p_interaction_fdr'] = q_int_vals

            sig_int = results_df[(results_df['p_interaction_fdr'] < 0.05) & (results_df['p_interaction_fdr'].notna())]
            if verbose and len(sig_int) > 0:
                print("\n    *** FDR-Significant Interactions ***")
                for _, row in sig_int.iterrows():
                    print(f"      {row['task']} {row['parameter']}: "
                          f"β={row['beta_interaction']:.3f}, p_fdr={row['p_interaction_fdr']:.4f}")

    results_df.to_csv(OUTPUT_DIR / "exgaussian_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'exgaussian_results.csv'}")

    return results_df


@register_analysis(
    name="meta_analysis",
    description="Random-effects meta-analysis across tasks",
    source_script="tier2_meta_analysis.py",
    tier=2
)
def analyze_meta_analysis(verbose: bool = True) -> pd.DataFrame:
    """
    Meta-analysis of UCLA effects across tasks.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 2: CROSS-TASK META-ANALYSIS")
        print("=" * 70)

    master = load_mechanistic_data()
    effect_sizes = []

    # Compute UCLA -> RT variability effect for each task
    for task in ['stroop', 'wcst', 'prp']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            continue

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else ('t2_rt_ms' if 't2_rt_ms' in trials.columns else 'rt')
        if rt_col not in trials.columns:
            continue

        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Compute RT variability per participant
        var_results = []
        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 20:
                continue

            rt_sd = pdata[rt_col].std()
            rt_cv = pdata[rt_col].std() / pdata[rt_col].mean() if pdata[rt_col].mean() > 0 else np.nan

            var_results.append({
                'participant_id': pid,
                'rt_sd': rt_sd,
                'rt_cv': rt_cv
            })

        if len(var_results) < 20:
            continue

        var_df = pd.DataFrame(var_results)
        merged = master.merge(var_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        # Get UCLA effect size
        try:
            formula = "rt_sd ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)"
            model = smf.ols(formula, data=merged).fit()

            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                se = model.bse['z_ucla']

                effect_sizes.append({
                    'task': task,
                    'beta': beta,
                    'se': se,
                    'n': len(merged)
                })

        except Exception:
            pass

    if len(effect_sizes) < 2:
        if verbose:
            print("  Insufficient tasks for meta-analysis")
        return pd.DataFrame()

    effect_df = pd.DataFrame(effect_sizes)

    if verbose:
        print(f"\n  Tasks: {list(effect_df['task'])}")

    # Random-effects meta-analysis (DerSimonian-Laird)
    betas = effect_df['beta'].values
    ses = effect_df['se'].values
    weights = 1 / (ses ** 2)

    # Fixed effect estimate
    beta_fe = np.sum(weights * betas) / np.sum(weights)

    # Q statistic for heterogeneity
    Q = np.sum(weights * (betas - beta_fe) ** 2)
    df = len(betas) - 1
    p_q = 1 - stats.chi2.cdf(Q, df) if df > 0 else np.nan

    # I-squared
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    # Tau-squared (between-study variance)
    C = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau_sq = max(0, (Q - df) / C) if C > 0 else 0

    # Random effects estimate
    weights_re = 1 / (ses ** 2 + tau_sq)
    beta_re = np.sum(weights_re * betas) / np.sum(weights_re)
    se_re = np.sqrt(1 / np.sum(weights_re))

    results = [{
        'pooled_beta': beta_re,
        'pooled_se': se_re,
        'ci_lower': beta_re - 1.96 * se_re,
        'ci_upper': beta_re + 1.96 * se_re,
        'Q': Q,
        'p_Q': p_q,
        'I2': I2,
        'tau_sq': tau_sq,
        'n_tasks': len(betas)
    }]

    if verbose:
        print(f"\n  Pooled effect (RE): β = {beta_re:.3f} [{beta_re - 1.96*se_re:.3f}, {beta_re + 1.96*se_re:.3f}]")
        print(f"  Heterogeneity: I² = {I2:.1f}%, Q = {Q:.2f}, p = {p_q:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "meta_analysis_results.csv", index=False, encoding='utf-8-sig')

    # Also save per-task effects
    effect_df.to_csv(OUTPUT_DIR / "meta_analysis_effects.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'meta_analysis_results.csv'}")

    return results_df


@register_analysis(
    name="hmm_states",
    description="HMM-based attentional state identification",
    source_script="tier2_hmm_attentional_states.py",
    tier=2
)
def analyze_hmm_states(verbose: bool = True) -> pd.DataFrame:
    """
    HMM-based identification of attentional states.
    Note: Requires hmmlearn package.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TIER 2: HMM ATTENTIONAL STATES")
        print("=" * 70)

    try:
        from hmmlearn import hmm as hmm_module
    except ImportError:
        if verbose:
            print("  hmmlearn package not available - skipping")
        return pd.DataFrame()

    master = load_mechanistic_data()
    all_results = []

    # Use WCST (most trials per participant)
    trials = load_trial_data('wcst')

    if len(trials) < 100:
        if verbose:
            print("  Insufficient WCST trial data")
        return pd.DataFrame()

    # Sort trials by explicit order; fall back to participant-only if missing
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'timestamp', 'trialnum']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols)

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    if rt_col not in trials.columns:
        if verbose:
            print("  No RT column found")
        return pd.DataFrame()

    trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

    if verbose:
        print(f"  Fitting 2-state HMM per participant...")

    hmm_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 50:
            continue

        pdata = pdata.sort_values(pdata.columns[0])
        rts = pdata[rt_col].values.reshape(-1, 1)

        try:
            model = hmm_module.GaussianHMM(
                n_components=2,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            model.fit(rts)

            states = model.predict(rts)

            # Identify "lapse" state (higher mean RT)
            means = model.means_.flatten()
            lapse_state = np.argmax(means)

            # State occupancy
            lapse_occupancy = (states == lapse_state).mean() * 100

            # Transition probabilities
            trans_matrix = model.transmat_
            stay_focused = trans_matrix[1-lapse_state, 1-lapse_state] if lapse_state == 1 else trans_matrix[1, 1]
            trans_to_lapse = trans_matrix[1-lapse_state, lapse_state] if lapse_state == 1 else trans_matrix[0, 1]

            hmm_results.append({
                'participant_id': pid,
                'lapse_occupancy': lapse_occupancy,
                'stay_focused': stay_focused,
                'trans_to_lapse': trans_to_lapse,
                'n_trials': len(pdata)
            })

        except Exception:
            continue

    if len(hmm_results) < 20:
        if verbose:
            print(f"  Only {len(hmm_results)} participants fitted successfully")
        return pd.DataFrame()

    hmm_df = pd.DataFrame(hmm_results)
    merged = master.merge(hmm_df, on='participant_id', how='inner')

    if len(merged) < 20:
        if verbose:
            print(f"  Insufficient merged data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"  Mean lapse occupancy: {merged['lapse_occupancy'].mean():.1f}%")

    # Prepare logit-transformed probabilities (avoid boundary issues)
    epsilon = 1e-4
    for prob_col in ['lapse_occupancy', 'trans_to_lapse']:
        prob = np.clip(merged[prob_col] / 100.0, epsilon, 1 - epsilon)
        merged[f'{prob_col}_logit'] = np.log(prob / (1 - prob))

    # Test UCLA effects
    for metric in ['lapse_occupancy', 'trans_to_lapse']:
        try:
            formula = f"{metric}_logit ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.wls(formula, data=merged, weights=merged.get('n_trials', 1)).fit(cov_type='HC3')

            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']

                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"  UCLA -> {metric} (logit): β={beta:.3f}, p={p:.4f}{sig}")

                all_results.append({
                    'metric': metric,
                    'beta_ucla_logit': beta,
                    'p_ucla': p,
                    'n': len(merged),
                    'cov_type': 'HC3',
                    'weight_var': 'n_trials'
                })

        except Exception as e:
            if verbose:
                print(f"  Regression error for {metric}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "hmm_states_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'hmm_states_results.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run mechanistic analyses.
    """
    if verbose:
        print("=" * 70)
        print("MECHANISTIC ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name} (Tier {spec.tier})")
        results[analysis] = spec.function(verbose=verbose)
    else:
        # Run by tier
        for tier in [1, 2]:
            tier_analyses = {k: v for k, v in ANALYSES.items() if v.tier == tier}
            if tier_analyses:
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"TIER {tier} ANALYSES")
                    print("=" * 70)

                for name, spec in tier_analyses.items():
                    try:
                        results[name] = spec.function(verbose=verbose)
                    except Exception as e:
                        print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("MECHANISTIC SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Mechanistic Analyses:")
    print("-" * 60)
    for tier in [1, 2]:
        tier_analyses = {k: v for k, v in ANALYSES.items() if v.tier == tier}
        if tier_analyses:
            print(f"\n  TIER {tier}:")
            for name, spec in tier_analyses.items():
                print(f"    {name}")
                print(f"      {spec.description}")
                print(f"      Source: {spec.source_script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mechanistic Analysis Suite")
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
