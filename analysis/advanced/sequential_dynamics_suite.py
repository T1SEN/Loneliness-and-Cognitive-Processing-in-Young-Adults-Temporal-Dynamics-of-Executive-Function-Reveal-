"""
Sequential Dynamics Analysis Suite
===================================

Trial-level sequential dynamics analysis for UCLA × Executive Function.

Analyses:
1. error_cascade: 연속 오류 패턴 탐지 및 UCLA 관계
2. recovery_trajectory: 오류 후 1-5 시행 회복 궤적
3. momentum: 연속 정답 후 RT 가속 효과
4. volatility: 시행간 변동성 (stable vs volatile performer)

Usage:
    python -m analysis.advanced.sequential_dynamics_suite              # Run all
    python -m analysis.advanced.sequential_dynamics_suite --analysis error_cascade
    python -m analysis.advanced.sequential_dynamics_suite --list

    from analysis.advanced import sequential_dynamics_suite
    sequential_dynamics_suite.run('error_cascade')
    sequential_dynamics_suite.run()  # All analyses

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
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Bayesian imports (optional)
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR, find_interaction_term
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "sequential_deep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable
    priority: int


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, priority: int = 1):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            priority=priority
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_master_data() -> pd.DataFrame:
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

    # Handle duplicate participant_id columns
    if 'participantid' in df.columns and 'participant_id' in df.columns:
        df = df.drop(columns=['participantid'])
    elif 'participantid' in df.columns:
        df = df.rename(columns={'participantid': 'participant_id'})

    return df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_error_runs(errors: np.ndarray, threshold: int = 2) -> List[dict]:
    """
    Detect consecutive error runs (cascades).

    Args:
        errors: Binary array (1=error, 0=correct)
        threshold: Minimum consecutive errors to count as cascade

    Returns:
        List of cascade info dicts
    """
    cascades = []
    current_run = 0
    run_start = 0

    for i, e in enumerate(errors):
        if e == 1:
            if current_run == 0:
                run_start = i
            current_run += 1
        else:
            if current_run >= threshold:
                cascades.append({
                    'start': run_start,
                    'length': current_run,
                    'end': i - 1
                })
            current_run = 0

    # Handle run at end
    if current_run >= threshold:
        cascades.append({
            'start': run_start,
            'length': current_run,
            'end': len(errors) - 1
        })

    return cascades


def compute_recovery_metrics(
    df: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    max_lag: int = 5
) -> pd.DataFrame:
    """
    Compute post-error recovery trajectory metrics.

    Returns DataFrame with lag-wise RT and accuracy after errors.
    """
    results = []

    for pid, pdata in df.groupby('participant_id'):
        pdata = pdata.sort_values('trial_index' if 'trial_index' in pdata.columns else pdata.columns[0]).reset_index(drop=True)

        errors = (pdata[correct_col] == 0).values if correct_col else None
        rts = pdata[rt_col].values

        if errors is None:
            continue

        error_indices = np.where(errors)[0]

        if len(error_indices) < 3:
            continue

        lag_data = {f'rt_lag_{lag}': [] for lag in range(1, max_lag + 1)}
        lag_data.update({f'acc_lag_{lag}': [] for lag in range(1, max_lag + 1)})

        for err_idx in error_indices:
            for lag in range(1, max_lag + 1):
                next_idx = err_idx + lag
                if next_idx < len(rts):
                    lag_data[f'rt_lag_{lag}'].append(rts[next_idx])
                    lag_data[f'acc_lag_{lag}'].append(1 - errors[next_idx])

        result = {
            'participant_id': pid,
            'n_errors': len(error_indices),
            'n_trials': len(pdata)
        }

        for lag in range(1, max_lag + 1):
            if len(lag_data[f'rt_lag_{lag}']) > 0:
                result[f'mean_rt_lag_{lag}'] = np.mean(lag_data[f'rt_lag_{lag}'])
                result[f'mean_acc_lag_{lag}'] = np.mean(lag_data[f'acc_lag_{lag}'])

        results.append(result)

    return pd.DataFrame(results)


def compute_momentum_metrics(
    df: pd.DataFrame,
    rt_col: str,
    correct_col: str
) -> pd.DataFrame:
    """
    Compute momentum effect: RT change with consecutive correct responses.
    """
    results = []

    for pid, pdata in df.groupby('participant_id'):
        pdata = pdata.sort_values('trial_index' if 'trial_index' in pdata.columns else pdata.columns[0]).reset_index(drop=True)

        correct = pdata[correct_col].values
        rts = pdata[rt_col].values

        # Compute correct streak
        streaks = []
        current_streak = 0
        for c in correct:
            if c == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)

        streaks = np.array(streaks)

        # Mean RT by streak length
        streak_rt = {}
        for streak in [0, 1, 2, 3, 4, 5]:
            mask = streaks == streak
            if mask.sum() > 5:
                streak_rt[f'rt_streak_{streak}'] = np.nanmean(rts[mask])

        if len(streak_rt) < 3:
            continue

        result = {
            'participant_id': pid,
            'n_trials': len(pdata),
            'mean_streak': np.mean(streaks),
            'max_streak': np.max(streaks),
            **streak_rt
        }

        # Momentum slope (RT change per streak increment)
        x = np.array([0, 1, 2, 3, 4, 5])
        y = [streak_rt.get(f'rt_streak_{s}', np.nan) for s in x]
        valid = ~np.isnan(y)
        if valid.sum() >= 3:
            slope, _, _, _, _ = stats.linregress(x[valid], np.array(y)[valid])
            result['momentum_slope'] = slope

        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# EXPONENTIAL RECOVERY FITTING UTILITIES
# =============================================================================

def _exponential_decay(t: np.ndarray, delta: float, tau: float) -> np.ndarray:
    """Exponential decay function: delta * exp(-t/tau)"""
    return delta * np.exp(-t / tau)


def fit_exponential_recovery(
    lag_rts: List[float],
    n_bootstrap: int = 200,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Fit exponential decay to post-event RT recovery with bootstrap SE.

    RT(t) = baseline + delta * exp(-t/tau)
    - delta: Initial slowing magnitude (ms)
    - tau: Recovery time constant (higher = slower recovery)

    Parameters
    ----------
    lag_rts : list
        RT deviations from baseline at lags 1, 2, 3, ...
    n_bootstrap : int
        Number of bootstrap iterations for SE estimation
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        - tau: Recovery time constant
        - tau_se: Bootstrap SE of tau
        - delta: Initial slowing magnitude
        - delta_se: Bootstrap SE of delta
        - r_squared: Fit quality
        - converged: Whether fitting converged
    """
    np.random.seed(random_state)

    # Filter NaN values
    valid_rts = [(i+1, rt) for i, rt in enumerate(lag_rts) if pd.notna(rt)]

    if len(valid_rts) < 3:
        return {
            'tau': np.nan, 'tau_se': np.nan,
            'delta': np.nan, 'delta_se': np.nan,
            'r_squared': np.nan, 'converged': False
        }

    lags = np.array([v[0] for v in valid_rts])
    rts = np.array([v[1] for v in valid_rts])

    try:
        # Initial guess
        delta_init = max(rts[0], 10) if rts[0] > 0 else 50
        tau_init = 2.0

        popt, pcov = curve_fit(
            _exponential_decay, lags, rts,
            p0=[delta_init, tau_init],
            bounds=([0, 0.5], [500, 10]),
            maxfev=1000
        )

        # R-squared
        predicted = _exponential_decay(lags, *popt)
        ss_res = np.sum((rts - predicted) ** 2)
        ss_tot = np.sum((rts - np.mean(rts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Bootstrap for SE estimation
        bootstrap_taus = []
        bootstrap_deltas = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(len(lags), size=len(lags), replace=True)
            boot_lags = lags[idx]
            boot_rts = rts[idx]

            try:
                boot_popt, _ = curve_fit(
                    _exponential_decay, boot_lags, boot_rts,
                    p0=popt,
                    bounds=([0, 0.5], [500, 10]),
                    maxfev=500
                )
                bootstrap_deltas.append(boot_popt[0])
                bootstrap_taus.append(boot_popt[1])
            except Exception:
                continue

        tau_se = np.std(bootstrap_taus) if len(bootstrap_taus) > 10 else np.nan
        delta_se = np.std(bootstrap_deltas) if len(bootstrap_deltas) > 10 else np.nan

        return {
            'tau': popt[1],
            'tau_se': tau_se,
            'delta': popt[0],
            'delta_se': delta_se,
            'r_squared': r_squared,
            'converged': True
        }

    except Exception:
        return {
            'tau': np.nan, 'tau_se': np.nan,
            'delta': np.nan, 'delta_se': np.nan,
            'r_squared': np.nan, 'converged': False
        }


def compute_proactive_reactive_indices(
    trial_df: pd.DataFrame,
    cond_col: str,
    rt_col: str
) -> pd.DataFrame:
    """
    Compute proactive vs reactive control indices from Stroop task.

    Proactive control: Sustained slowing (overall mean RT)
    Reactive control: Transient slowing after incongruent (CSE effect)

    Parameters
    ----------
    trial_df : DataFrame
        Trial-level Stroop data with condition and RT columns
    cond_col : str
        Column name for trial condition (congruent/incongruent)
    rt_col : str
        Column name for reaction time

    Returns
    -------
    DataFrame with participant-level indices
    """
    results = []

    for pid, grp in trial_df.groupby('participant_id'):
        # Sort by trial order
        sort_col = None
        for cand in ['trialindex', 'trial_index', 'timestamp']:
            if cand in grp.columns:
                sort_col = cand
                break

        if sort_col:
            grp = grp.sort_values(sort_col).reset_index(drop=True)
        else:
            grp = grp.reset_index(drop=True)

        if len(grp) < 30:
            continue

        # Normalize condition names
        grp['cond_norm'] = grp[cond_col].astype(str).str.lower().str.strip()
        grp['prev_cond'] = grp['cond_norm'].shift(1)

        # CSE: (iI - cI) vs (iC - cC)
        # iI: incongruent following incongruent
        # cI: incongruent following congruent
        iI = grp[(grp['cond_norm'] == 'incongruent') &
                 (grp['prev_cond'] == 'incongruent')][rt_col]
        cI = grp[(grp['cond_norm'] == 'incongruent') &
                 (grp['prev_cond'] == 'congruent')][rt_col]

        if len(iI) >= 5 and len(cI) >= 5:
            cse = iI.mean() - cI.mean()  # Negative = conflict adaptation
        else:
            cse = np.nan

        # Overall slowing (proactive index)
        mean_rt = grp[rt_col].mean()

        # Reactive index: RT after incongruent vs after congruent
        post_inc = grp[grp['prev_cond'] == 'incongruent'][rt_col].mean()
        post_con = grp[grp['prev_cond'] == 'congruent'][rt_col].mean()
        reactive_index = post_inc - post_con if pd.notna(post_inc) and pd.notna(post_con) else np.nan

        # Proactive index: overall mean RT relative to task difficulty
        congruent_rt = grp[grp['cond_norm'] == 'congruent'][rt_col].mean()
        proactive_index = mean_rt - congruent_rt if pd.notna(congruent_rt) else np.nan

        results.append({
            'participant_id': pid,
            'stroop_cse': cse,
            'stroop_mean_rt': mean_rt,
            'stroop_reactive_index': reactive_index,
            'stroop_proactive_index': proactive_index,
            'stroop_n_trials': len(grp)
        })

    return pd.DataFrame(results)


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="adaptive_recovery",
    description="Exponential recovery dynamics with tau fitting and proactive/reactive indices",
    priority=1
)
def analyze_adaptive_recovery(verbose: bool = True) -> pd.DataFrame:
    """
    Adaptive Recovery Dynamics Analysis.

    Enhanced from legacy adaptive_recovery_dynamics.py with:
    1. Exponential fitting: RT(t) = baseline + delta * exp(-t/tau)
    2. Bootstrap SE for tau (for inverse-variance weighting)
    3. Proactive/reactive control indices from Stroop CSE
    4. FDR correction for multiple outcomes
    5. Optional Bayesian analysis with 4 chains × 2000 draws

    Research Question:
    Do lonely individuals show impaired dynamic adjustment after setbacks?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ADAPTIVE RECOVERY DYNAMICS ANALYSIS")
        print("=" * 70)
        print("  Exponential fitting: RT(t) = baseline + delta * exp(-t/tau)")
        print("  tau = recovery time constant (higher = slower recovery)")

    master = load_master_data()
    all_results = []

    # ===========================
    # Part 1: Post-Error Recovery (WCST, Stroop)
    # ===========================
    if verbose:
        print("\n  [1] Computing post-error recovery trajectories...")

    for task in ['wcst', 'stroop']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"    {task.upper()}: Insufficient data")
            continue

        # Get columns
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        correct_col = None
        for c in ['correct', 'iscorrect', 'is_correct', 'accuracy']:
            if c in trials.columns:
                correct_col = c
                break

        if correct_col is None:
            continue

        # Sort trials
        sort_cols = ['participant_id']
        for cand in ['trialindex', 'trial_index', 'timestamp']:
            if cand in trials.columns:
                sort_cols.append(cand)
                break
        trials = trials.sort_values(sort_cols)

        # Filter valid trials
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        recovery_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.reset_index(drop=True)
            errors = (pdata[correct_col] == 0).values
            rts = pdata[rt_col].values

            # Find error positions
            error_positions = np.where(errors)[0]
            if len(error_positions) < 3:
                continue

            # Compute baseline RT (correct trials only)
            baseline_rt = np.median(rts[~errors])

            # Collect RT deviations at each lag after error
            max_lag = 5
            lag_rts_list = {lag: [] for lag in range(1, max_lag + 1)}

            for err_idx in error_positions:
                for lag in range(1, max_lag + 1):
                    next_idx = err_idx + lag
                    if next_idx < len(rts) and not np.isnan(rts[next_idx]):
                        # RT deviation from baseline
                        lag_rts_list[lag].append(rts[next_idx] - baseline_rt)

            # Mean RT deviation at each lag
            lag_rt_devs = []
            for lag in range(1, max_lag + 1):
                if len(lag_rts_list[lag]) >= 3:
                    lag_rt_devs.append(np.mean(lag_rts_list[lag]))
                else:
                    lag_rt_devs.append(np.nan)

            # Fit exponential recovery
            fit_result = fit_exponential_recovery(lag_rt_devs, n_bootstrap=200)

            recovery_results.append({
                'participant_id': pid,
                'task': task,
                'n_trials': len(pdata),
                'n_errors': len(error_positions),
                'baseline_rt': baseline_rt,
                'initial_slowing': lag_rt_devs[0] if pd.notna(lag_rt_devs[0]) else np.nan,
                'recovery_tau': fit_result['tau'],
                'recovery_tau_se': fit_result['tau_se'],
                'recovery_delta': fit_result['delta'],
                'recovery_delta_se': fit_result['delta_se'],
                'recovery_r2': fit_result['r_squared'],
                'recovery_converged': fit_result['converged'],
                **{f'rt_lag{i+1}': lag_rt_devs[i] for i in range(len(lag_rt_devs))}
            })

        if len(recovery_results) < 20:
            continue

        recovery_df = pd.DataFrame(recovery_results)

        if verbose:
            converged = recovery_df['recovery_converged'].sum()
            print(f"    {task.upper()}: N={len(recovery_df)}, Converged={converged}")
            print(f"      Mean tau={recovery_df['recovery_tau'].mean():.2f}, "
                  f"Mean delta={recovery_df['recovery_delta'].mean():.1f}ms")

        # Merge with master
        merged = master.merge(recovery_df, on='participant_id', how='inner')

        # Regression on recovery outcomes
        outcomes = [
            ('recovery_tau', f'{task.upper()} Recovery Time Constant'),
            ('recovery_delta', f'{task.upper()} Initial Slowing Magnitude'),
            ('initial_slowing', f'{task.upper()} Post-Error Slowing (lag 1)')
        ]

        for outcome, label in outcomes:
            if outcome not in merged.columns or merged[outcome].isna().all():
                continue

            clean_data = merged.dropna(subset=[
                'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome
            ])

            if len(clean_data) < 25:
                continue

            try:
                formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'outcome': outcome,
                    'outcome_label': label,
                    'n': len(clean_data),
                    'beta_ucla': model.params.get('z_ucla', np.nan),
                    'se_ucla': model.bse.get('z_ucla', np.nan),
                    'p_ucla': model.pvalues.get('z_ucla', np.nan),
                    'r_squared': model.rsquared
                }
                # Dynamic interaction term detection
                int_term = find_interaction_term(model.params.index)
                result['beta_ucla_x_gender'] = model.params.get(int_term, np.nan) if int_term else np.nan
                result['p_ucla_x_gender'] = model.pvalues.get(int_term, np.nan) if int_term else np.nan

                all_results.append(result)

                if verbose and result['p_ucla'] < 0.10:
                    sig = "***" if result['p_ucla'] < 0.001 else "**" if result['p_ucla'] < 0.01 else "*" if result['p_ucla'] < 0.05 else "+"
                    print(f"      UCLA → {label}: β={result['beta_ucla']:.3f}, p={result['p_ucla']:.4f} {sig}")

            except Exception as e:
                if verbose:
                    print(f"      {label}: ERROR - {e}")

        # Save task-specific recovery data
        recovery_df.to_csv(OUTPUT_DIR / f"adaptive_recovery_{task}.csv",
                           index=False, encoding='utf-8-sig')

    # ===========================
    # Part 2: Proactive/Reactive Control (Stroop)
    # ===========================
    if verbose:
        print("\n  [2] Computing proactive/reactive control indices...")

    stroop_trials = load_trial_data('stroop')

    if len(stroop_trials) > 100:
        rt_col = 'rt_ms' if 'rt_ms' in stroop_trials.columns else 'rt'
        cond_col = None
        for c in ['type', 'condition', 'congruency']:
            if c in stroop_trials.columns:
                cond_col = c
                break

        if cond_col:
            stroop_trials = stroop_trials[(stroop_trials[rt_col] > 100) &
                                          (stroop_trials[rt_col] < 3000)].copy()

            control_df = compute_proactive_reactive_indices(stroop_trials, cond_col, rt_col)

            if len(control_df) >= 20:
                if verbose:
                    print(f"    Stroop control indices: N={len(control_df)}")
                    print(f"      Mean CSE={control_df['stroop_cse'].mean():.1f}ms")

                merged = master.merge(control_df, on='participant_id', how='inner')

                outcomes = [
                    ('stroop_cse', 'Stroop Conflict Adaptation (CSE)'),
                    ('stroop_reactive_index', 'Stroop Reactive Control Index'),
                    ('stroop_proactive_index', 'Stroop Proactive Control Index')
                ]

                for outcome, label in outcomes:
                    if outcome not in merged.columns or merged[outcome].isna().all():
                        continue

                    clean_data = merged.dropna(subset=[
                        'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome
                    ])

                    if len(clean_data) < 25:
                        continue

                    try:
                        formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                        model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                        result = {
                            'task': 'stroop',
                            'outcome': outcome,
                            'outcome_label': label,
                            'n': len(clean_data),
                            'beta_ucla': model.params.get('z_ucla', np.nan),
                            'se_ucla': model.bse.get('z_ucla', np.nan),
                            'p_ucla': model.pvalues.get('z_ucla', np.nan),
                            'r_squared': model.rsquared
                        }
                        # Dynamic interaction term detection
                        int_term = find_interaction_term(model.params.index)
                        result['beta_ucla_x_gender'] = model.params.get(int_term, np.nan) if int_term else np.nan
                        result['p_ucla_x_gender'] = model.pvalues.get(int_term, np.nan) if int_term else np.nan

                        all_results.append(result)

                        if verbose and result['p_ucla'] < 0.10:
                            sig = "***" if result['p_ucla'] < 0.001 else "**" if result['p_ucla'] < 0.01 else "*" if result['p_ucla'] < 0.05 else "+"
                            print(f"      UCLA → {label}: β={result['beta_ucla']:.3f}, p={result['p_ucla']:.4f} {sig}")

                    except Exception as e:
                        if verbose:
                            print(f"      {label}: ERROR - {e}")

                # Save control indices
                control_df.to_csv(OUTPUT_DIR / "adaptive_recovery_control_indices.csv",
                                  index=False, encoding='utf-8-sig')

    # ===========================
    # Part 3: FDR Correction
    # ===========================
    results_df = pd.DataFrame(all_results)

    if len(results_df) > 1:
        if verbose:
            print("\n  [3] Applying FDR correction...")

        # FDR for UCLA main effects using Benjamini-Hochberg
        p_vals = results_df['p_ucla'].dropna().values
        if len(p_vals) > 1:
            # Manual BH correction (more reliable than scipy version)
            n = len(p_vals)
            sorted_idx = np.argsort(p_vals)
            ranks = np.empty(n)
            ranks[sorted_idx] = np.arange(1, n + 1)
            q_vals = p_vals * n / ranks

            # Enforce monotonicity: q[i] <= q[i+1] in sorted order
            q_sorted = q_vals[sorted_idx].copy()
            for i in range(n - 2, -1, -1):
                q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
            q_vals[sorted_idx] = q_sorted
            q_vals = np.clip(q_vals, 0, 1)

            results_df.loc[results_df['p_ucla'].notna(), 'p_ucla_fdr'] = q_vals

    # ===========================
    # Part 4: Bayesian Analysis (Optional)
    # ===========================
    if HAS_PYMC and len(results_df) > 0:
        if verbose:
            print("\n  [4] Running Bayesian analysis (4 chains × 2000 draws)...")

        # Select key outcomes for Bayesian analysis
        key_outcomes = ['recovery_tau', 'stroop_cse']

        for task in ['wcst', 'stroop']:
            trials = load_trial_data(task)
            if len(trials) < 100:
                continue

            # Load recovery data for this task
            recovery_path = OUTPUT_DIR / f"adaptive_recovery_{task}.csv"
            if not recovery_path.exists():
                continue

            recovery_df = pd.read_csv(recovery_path)
            merged = master.merge(recovery_df, on='participant_id', how='inner')

            for outcome in ['recovery_tau']:
                if outcome not in merged.columns:
                    continue

                clean_data = merged.dropna(subset=[
                    'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx',
                    'z_dass_str', 'z_age', outcome
                ])

                if len(clean_data) < 30:
                    continue

                try:
                    y = clean_data[outcome].values
                    y_std = (y - np.mean(y)) / np.std(y) if np.std(y) > 0 else y

                    with pm.Model() as model:
                        # Priors
                        intercept = pm.Normal('intercept', mu=0, sigma=1)
                        b_ucla = pm.Normal('b_ucla', mu=0, sigma=0.5)
                        b_gender = pm.Normal('b_gender', mu=0, sigma=0.5)
                        b_interaction = pm.Normal('b_ucla_x_gender', mu=0, sigma=0.5)
                        b_dass_d = pm.Normal('b_dass_d', mu=0, sigma=0.5)
                        b_dass_a = pm.Normal('b_dass_a', mu=0, sigma=0.5)
                        b_dass_s = pm.Normal('b_dass_s', mu=0, sigma=0.5)
                        b_age = pm.Normal('b_age', mu=0, sigma=0.5)
                        sigma = pm.HalfNormal('sigma', sigma=1)

                        mu = (intercept +
                              b_ucla * clean_data['z_ucla'].values +
                              b_gender * clean_data['gender_male'].values +
                              b_interaction * clean_data['z_ucla'].values * clean_data['gender_male'].values +
                              b_dass_d * clean_data['z_dass_dep'].values +
                              b_dass_a * clean_data['z_dass_anx'].values +
                              b_dass_s * clean_data['z_dass_str'].values +
                              b_age * clean_data['z_age'].values)

                        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_std)

                        # Sample with improved settings
                        trace = pm.sample(
                            draws=2000,
                            tune=1000,
                            chains=4,
                            cores=1,
                            random_seed=42,
                            progressbar=False,
                            return_inferencedata=True
                        )

                    # Extract results
                    posterior_ucla = trace.posterior['b_ucla'].values.flatten()
                    rope_interval = (-0.1, 0.1)
                    in_rope = np.mean((posterior_ucla >= rope_interval[0]) &
                                      (posterior_ucla <= rope_interval[1]))

                    if verbose:
                        print(f"    {task.upper()} {outcome}:")
                        print(f"      Posterior mean = {np.mean(posterior_ucla):.3f}")
                        print(f"      94% HDI = [{np.percentile(posterior_ucla, 3):.3f}, "
                              f"{np.percentile(posterior_ucla, 97):.3f}]")
                        print(f"      ROPE in = {in_rope*100:.1f}%")

                except Exception as e:
                    if verbose:
                        print(f"    Bayesian error for {task} {outcome}: {e}")

    # Save results
    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "adaptive_recovery_regression.csv",
                          index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'adaptive_recovery_regression.csv'}")

        # Summary
        sig_main = results_df[results_df['p_ucla'] < 0.05] if 'p_ucla' in results_df.columns else pd.DataFrame()
        sig_int = results_df[results_df['p_ucla_x_gender'] < 0.05] if 'p_ucla_x_gender' in results_df.columns else pd.DataFrame()

        if len(sig_main) > 0 or len(sig_int) > 0:
            print("\n  *** SIGNIFICANT RESULTS ***")
            for _, row in sig_main.iterrows():
                print(f"    {row['outcome_label']}: UCLA β={row['beta_ucla']:.3f}, p={row['p_ucla']:.4f}")
            for _, row in sig_int.iterrows():
                print(f"    {row['outcome_label']}: UCLA×Gender β={row['beta_ucla_x_gender']:.3f}, p={row['p_ucla_x_gender']:.4f}")

    return results_df


@register_analysis(
    name="error_cascade",
    description="Error cascade (consecutive error) detection and UCLA relationship",
    priority=1
)
def analyze_error_cascade(verbose: bool = True) -> pd.DataFrame:
    """
    Detect error cascades and test UCLA relationship.

    Cascade = 2+ consecutive errors
    Tests: Do lonely people have more/longer cascades?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ERROR CASCADE ANALYSIS")
        print("=" * 70)

    master = load_master_data()
    all_results = []

    for task in ['wcst', 'stroop']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        # Get columns
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        correct_col = None
        for c in ['correct', 'iscorrect', 'is_correct', 'accuracy']:
            if c in trials.columns:
                correct_col = c
                break

        if correct_col is None:
            if verbose:
                print(f"    No correct column found")
            continue

        # Sort by trial order
        sort_cols = ['participant_id']
        for cand in ['trialindex', 'trial_index', 'timestamp']:
            if cand in trials.columns:
                sort_cols.append(cand)
                break
        trials = trials.sort_values(sort_cols)

        # Filter valid trials
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Create error column
        trials['error'] = (trials[correct_col] == 0).astype(int)

        cascade_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 20:
                continue

            errors = pdata['error'].values
            cascades = detect_error_runs(errors, threshold=2)

            if len(cascades) == 0:
                lengths = [0]
            else:
                lengths = [c['length'] for c in cascades]

            cascade_results.append({
                'participant_id': pid,
                'task': task,
                'n_trials': len(pdata),
                'n_cascades': len(cascades),
                'cascade_rate': len(cascades) / len(pdata) * 100,
                'mean_cascade_length': np.mean(lengths) if lengths else 0,
                'max_cascade_length': np.max(lengths) if lengths else 0,
                'total_cascade_trials': sum(lengths),
                'cascade_proportion': sum(lengths) / len(pdata) * 100,
                'total_errors': errors.sum(),
                'error_rate': errors.mean() * 100
            })

        cascade_df = pd.DataFrame(cascade_results)

        if len(cascade_df) < 20:
            continue

        # Merge with master
        merged = master.merge(cascade_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean cascades/participant: {merged['n_cascades'].mean():.2f}")
            print(f"    Mean cascade length: {merged['mean_cascade_length'].mean():.2f}")

        # Regression
        outcomes = [
            ('n_cascades', 'Number of Cascades'),
            ('mean_cascade_length', 'Mean Cascade Length'),
            ('cascade_proportion', 'Cascade Proportion (%)')
        ]

        for outcome, label in outcomes:
            try:
                formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'outcome': outcome,
                    'outcome_label': label,
                    'n': len(merged)
                }

                if 'z_ucla' in model.params:
                    result['beta_ucla'] = model.params['z_ucla']
                    result['p_ucla'] = model.pvalues['z_ucla']

                # Dynamic interaction term detection
                int_term = find_interaction_term(model.params.index)
                if int_term:
                    result['beta_ucla_x_gender'] = model.params[int_term]
                    result['p_ucla_x_gender'] = model.pvalues[int_term]

                result['r_squared'] = model.rsquared
                all_results.append(result)

                if verbose and 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    UCLA → {label}: β={beta:.4f}, p={p:.4f} {sig}")

            except Exception as e:
                if verbose:
                    print(f"    {label}: ERROR - {e}")

        # Save task-specific results
        cascade_df.to_csv(OUTPUT_DIR / f"cascade_{task}.csv", index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "cascade_regression.csv", index=False, encoding='utf-8-sig')

    # Visualization
    if len(all_results) > 0:
        _plot_cascade_results(all_results)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'cascade_regression.csv'}")

    return results_df


def _plot_cascade_results(results: List[dict]):
    """Create cascade analysis visualization."""
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Effect sizes by task and outcome
    tasks = df['task'].unique()
    outcomes = df['outcome'].unique()

    x_pos = np.arange(len(outcomes))
    width = 0.35

    for i, task in enumerate(tasks):
        task_df = df[df['task'] == task]
        betas = [task_df[task_df['outcome'] == o]['beta_ucla'].values[0]
                 if len(task_df[task_df['outcome'] == o]) > 0 else 0 for o in outcomes]
        ax.bar(x_pos + i * width, betas, width, label=task.upper())

    ax.set_ylabel('UCLA β (DASS-controlled)')
    ax.set_title('Error Cascade: UCLA Effect by Task and Outcome')
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes], rotation=15)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cascade_effects.png", dpi=300, bbox_inches='tight')
    plt.close()


@register_analysis(
    name="recovery_trajectory",
    description="Post-error recovery trajectory (lag 1-5 analysis)",
    priority=1
)
def analyze_recovery_trajectory(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze post-error recovery trajectory.

    Questions:
    - How do RT and accuracy change in trials after an error?
    - Does UCLA moderate recovery speed?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("POST-ERROR RECOVERY TRAJECTORY ANALYSIS")
        print("=" * 70)

    master = load_master_data()
    all_results = []

    for task in ['wcst', 'stroop']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        # Get columns
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        correct_col = None
        for c in ['correct', 'iscorrect', 'is_correct']:
            if c in trials.columns:
                correct_col = c
                break

        if correct_col is None:
            continue

        # Add trial index if missing
        if 'trial_index' not in trials.columns:
            for cand in ['trialindex', 'timestamp']:
                if cand in trials.columns:
                    trials['trial_index'] = trials[cand]
                    break

        # Filter
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Compute recovery metrics
        recovery_df = compute_recovery_metrics(trials, rt_col, correct_col, max_lag=5)

        if len(recovery_df) < 20:
            continue

        # Merge
        merged = master.merge(recovery_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")

        # Regression for each lag
        for lag in range(1, 6):
            rt_col_lag = f'mean_rt_lag_{lag}'
            acc_col_lag = f'mean_acc_lag_{lag}'

            if rt_col_lag not in merged.columns:
                continue

            try:
                formula = f"{rt_col_lag} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'lag': lag,
                    'metric': 'RT',
                    'n': len(merged)
                }

                if 'z_ucla' in model.params:
                    result['beta_ucla'] = model.params['z_ucla']
                    result['p_ucla'] = model.pvalues['z_ucla']
                    result['r_squared'] = model.rsquared

                all_results.append(result)

                if verbose and 'z_ucla' in model.params and lag == 1:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    UCLA → RT at Lag 1: β={beta:.4f}, p={p:.4f} {sig}")

            except Exception:
                pass

        # Save task-specific
        recovery_df.to_csv(OUTPUT_DIR / f"recovery_{task}.csv", index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "recovery_regression.csv", index=False, encoding='utf-8-sig')

        # Visualization
        _plot_recovery_trajectory(results_df)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'recovery_regression.csv'}")

    return results_df


def _plot_recovery_trajectory(df: pd.DataFrame):
    """Plot recovery trajectory effects."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        lags = task_df['lag'].values
        betas = task_df['beta_ucla'].values
        pvals = task_df['p_ucla'].values

        ax.plot(lags, betas, 'o-', label=task.upper(), linewidth=2, markersize=8)

        # Mark significant points
        for lag, beta, p in zip(lags, betas, pvals):
            if p < 0.05:
                ax.annotate('*', (lag, beta), textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=14, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (trials after error)')
    ax.set_ylabel('UCLA β on RT (DASS-controlled)')
    ax.set_title('Post-Error Recovery: UCLA Effect Across Lags')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recovery_trajectory.png", dpi=300, bbox_inches='tight')
    plt.close()


@register_analysis(
    name="momentum",
    description="Correct-streak momentum effect and UCLA moderation",
    priority=1
)
def analyze_momentum(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze momentum effect: RT acceleration with consecutive correct responses.

    Question: Do lonely people show different momentum patterns?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MOMENTUM EFFECT ANALYSIS")
        print("=" * 70)

    master = load_master_data()
    all_results = []

    for task in ['wcst', 'stroop']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        # Get columns
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        correct_col = None
        for c in ['correct', 'iscorrect', 'is_correct']:
            if c in trials.columns:
                correct_col = c
                break

        if correct_col is None:
            continue

        # Add trial index if missing
        if 'trial_index' not in trials.columns:
            for cand in ['trialindex', 'timestamp']:
                if cand in trials.columns:
                    trials['trial_index'] = trials[cand]  # Fixed: was assigning string literal
                    break

        # Filter
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Compute momentum metrics
        momentum_df = compute_momentum_metrics(trials, rt_col, correct_col)

        if len(momentum_df) < 20:
            continue

        # Merge
        merged = master.merge(momentum_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            if 'momentum_slope' in merged.columns:
                print(f"    Mean momentum slope: {merged['momentum_slope'].mean():.2f} ms/streak")

        # Regression on momentum slope
        if 'momentum_slope' in merged.columns:
            try:
                formula = "momentum_slope ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'outcome': 'momentum_slope',
                    'n': len(merged)
                }

                if 'z_ucla' in model.params:
                    result['beta_ucla'] = model.params['z_ucla']
                    result['p_ucla'] = model.pvalues['z_ucla']

                # Dynamic interaction term detection
                int_term = find_interaction_term(model.params.index)
                if int_term:
                    result['beta_ucla_x_gender'] = model.params[int_term]
                    result['p_ucla_x_gender'] = model.pvalues[int_term]

                result['r_squared'] = model.rsquared
                all_results.append(result)

                if verbose and 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    UCLA → Momentum Slope: β={beta:.4f}, p={p:.4f} {sig}")

                # Save model
                with open(OUTPUT_DIR / f"momentum_{task}_regression.txt", 'w', encoding='utf-8') as f:
                    f.write(str(model.summary()))

            except Exception as e:
                if verbose:
                    print(f"    Momentum regression error: {e}")

        # Save task-specific
        momentum_df.to_csv(OUTPUT_DIR / f"momentum_{task}.csv", index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "momentum_regression.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'momentum_regression.csv'}")

    return results_df


@register_analysis(
    name="resilience",
    description="Error burst resilience: recovery speed and inter-burst correct streaks",
    priority=1
)
def analyze_resilience(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    오류 버스트 회복탄력성 분석

    연구 질문: 오류 버스트 후 기준선 성능으로 복귀 속도가 UCLA에 의해 조절되는가?

    회복탄력성 지수 = 버스트 간 평균 정답 연속 길이
    버스트 후 5시행 RT 궤적 분석

    Returns:
        (resilience_df, significant_findings)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ERROR BURST RESILIENCE ANALYSIS")
        print("=" * 70)

    master = load_master_data()
    all_results = []
    significant_findings = []

    for task in ['wcst', 'stroop']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        # Get columns
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        correct_col = None
        for c in ['correct', 'iscorrect', 'is_correct']:
            if c in trials.columns:
                correct_col = c
                break

        if correct_col is None:
            continue

        # Sort trials
        sort_cols = ['participant_id']
        for cand in ['trialindex', 'trial_index', 'timestamp']:
            if cand in trials.columns:
                sort_cols.append(cand)
                break
        trials = trials.sort_values(sort_cols)

        # Filter valid trials
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()
        trials['error'] = (trials[correct_col] == 0).astype(int)

        resilience_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            errors = pdata['error'].values
            rts = pdata[rt_col].values
            correct = (1 - errors)

            # Detect error bursts (2+ consecutive errors)
            bursts = detect_error_runs(errors, threshold=2)

            if len(bursts) < 2:
                # 버스트가 2개 미만이면 버스트 간 회복 분석 불가
                continue

            # 1. 버스트 간 정답 연속 길이 계산 (Inter-burst correct streaks)
            inter_burst_streaks = []
            for i in range(len(bursts) - 1):
                burst_end = bursts[i]['end']
                next_burst_start = bursts[i + 1]['start']

                # 버스트 사이의 정답 수
                between_trials = correct[burst_end + 1:next_burst_start]
                streak_length = len(between_trials)

                if streak_length > 0:
                    inter_burst_streaks.append(streak_length)

            # 2. 버스트 후 5시행 RT 궤적
            post_burst_rts = {lag: [] for lag in range(1, 6)}

            for burst in bursts:
                burst_end = burst['end']
                for lag in range(1, 6):
                    idx = burst_end + lag
                    if idx < len(rts) and not np.isnan(rts[idx]):
                        post_burst_rts[lag].append(rts[idx])

            # 3. 버스트 후 회복 기울기 계산
            recovery_slope = np.nan
            if all(len(post_burst_rts[lag]) >= 3 for lag in range(1, 4)):
                mean_rts = [np.mean(post_burst_rts[lag]) for lag in range(1, 4)]
                lags = [1, 2, 3]
                slope, _, _, p_slope, _ = stats.linregress(lags, mean_rts)
                recovery_slope = slope  # 음수면 RT 감소 = 빠른 회복

            # 4. 기준선 RT와 버스트 후 RT 비교
            baseline_rt = np.mean(rts[correct == 1]) if np.sum(correct) > 10 else np.nan
            post_burst_rt_lag1 = np.mean(post_burst_rts[1]) if len(post_burst_rts[1]) > 0 else np.nan

            resilience_results.append({
                'participant_id': pid,
                'task': task,
                'n_trials': len(pdata),
                'n_bursts': len(bursts),
                'burst_rate': len(bursts) / len(pdata) * 100,
                'mean_burst_length': np.mean([b['length'] for b in bursts]),
                'resilience_index': np.mean(inter_burst_streaks) if inter_burst_streaks else 0,
                'max_inter_burst_streak': np.max(inter_burst_streaks) if inter_burst_streaks else 0,
                'post_burst_rt_lag1': post_burst_rt_lag1,
                'post_burst_rt_lag2': np.mean(post_burst_rts[2]) if len(post_burst_rts[2]) > 0 else np.nan,
                'post_burst_rt_lag3': np.mean(post_burst_rts[3]) if len(post_burst_rts[3]) > 0 else np.nan,
                'post_burst_rt_lag4': np.mean(post_burst_rts[4]) if len(post_burst_rts[4]) > 0 else np.nan,
                'post_burst_rt_lag5': np.mean(post_burst_rts[5]) if len(post_burst_rts[5]) > 0 else np.nan,
                'recovery_slope': recovery_slope,
                'baseline_rt': baseline_rt,
                'rt_elevation_lag1': (post_burst_rt_lag1 - baseline_rt) if not np.isnan(post_burst_rt_lag1) and not np.isnan(baseline_rt) else np.nan
            })

        res_df = pd.DataFrame(resilience_results)

        if len(res_df) < 20:
            continue

        # Merge with master
        merged = master.merge(res_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean resilience index: {merged['resilience_index'].mean():.2f}")
            print(f"    Mean recovery slope: {merged['recovery_slope'].mean():.2f} ms/lag")

        # Regression on resilience outcomes
        outcomes = [
            ('resilience_index', 'Resilience Index (inter-burst streak)'),
            ('recovery_slope', 'Recovery Slope (RT change/lag)'),
            ('rt_elevation_lag1', 'RT Elevation at Lag 1'),
            ('burst_rate', 'Burst Rate (%)')
        ]

        for outcome, label in outcomes:
            if outcome not in merged.columns or merged[outcome].isna().all():
                continue

            clean_data = merged.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])
            if len(clean_data) < 25:
                continue

            try:
                formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'outcome': outcome,
                    'outcome_label': label,
                    'n': len(clean_data),
                    'beta_ucla': model.params.get('z_ucla', np.nan),
                    'p_ucla': model.pvalues.get('z_ucla', np.nan),
                    'r_squared': model.rsquared
                }
                # Dynamic interaction term detection
                int_term = find_interaction_term(model.params.index)
                result['beta_ucla_x_gender'] = model.params.get(int_term, np.nan) if int_term else np.nan
                result['p_ucla_x_gender'] = model.pvalues.get(int_term, np.nan) if int_term else np.nan

                all_results.append(result)

                # Check significance
                if result['p_ucla'] < 0.05:
                    significant_findings.append({
                        'analysis': 'Resilience',
                        'outcome': f"{task.upper()} {outcome}",
                        'effect': 'UCLA main',
                        'beta': result['beta_ucla'],
                        'p': result['p_ucla'],
                        'n': result['n']
                    })
                    if verbose:
                        print(f"    * {label}: UCLA β={result['beta_ucla']:.4f}, p={result['p_ucla']:.4f}")

                if result['p_ucla_x_gender'] < 0.05:
                    significant_findings.append({
                        'analysis': 'Resilience',
                        'outcome': f"{task.upper()} {outcome}",
                        'effect': 'UCLA x Gender',
                        'beta': result['beta_ucla_x_gender'],
                        'p': result['p_ucla_x_gender'],
                        'n': result['n']
                    })
                    if verbose:
                        print(f"    * {label}: UCLA×Gender β={result['beta_ucla_x_gender']:.4f}, p={result['p_ucla_x_gender']:.4f}")

            except Exception as e:
                if verbose:
                    print(f"    {label}: ERROR - {e}")

        # Save task-specific
        res_df.to_csv(OUTPUT_DIR / f"resilience_{task}.csv", index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "resilience_regression.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'resilience_regression.csv'}")
        if significant_findings:
            print(f"\n  SIGNIFICANT FINDINGS: {len(significant_findings)}")

    return results_df, significant_findings


@register_analysis(
    name="volatility",
    description="Trial-by-trial RT volatility (stable vs volatile performers)",
    priority=2
)
def analyze_volatility(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze trial-by-trial volatility patterns.

    Volatility = trial-to-trial RT fluctuation
    Question: Are lonely people more volatile (less stable)?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RT VOLATILITY ANALYSIS")
        print("=" * 70)

    master = load_master_data()
    all_results = []

    for task in ['wcst', 'stroop', 'prp']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'

        # Sort trials
        sort_cols = ['participant_id']
        for cand in ['trialindex', 'trial_index', 'timestamp']:
            if cand in trials.columns:
                sort_cols.append(cand)
                break
        trials = trials.sort_values(sort_cols)

        # Filter
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        volatility_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            rts = pdata[rt_col].values
            rts = rts[~np.isnan(rts)]

            if len(rts) < 30:
                continue

            # Volatility metrics
            # 1. RMSSD (Root Mean Square of Successive Differences)
            diffs = np.diff(rts)
            rmssd = np.sqrt(np.mean(diffs ** 2))

            # 2. CV (Coefficient of Variation)
            cv = np.std(rts) / np.mean(rts) * 100

            # 3. IIV (Intra-individual Variability) as SD
            iiv = np.std(rts)

            # 4. MSSD (Mean Squared Successive Difference)
            mssd = np.mean(diffs ** 2)

            # 5. Trend-adjusted volatility (residual SD after detrending)
            trial_nums = np.arange(len(rts))
            if len(trial_nums) > 10:
                slope, intercept, _, _, _ = stats.linregress(trial_nums, rts)
                predicted = slope * trial_nums + intercept
                residuals = rts - predicted
                adj_volatility = np.std(residuals)
            else:
                adj_volatility = np.nan

            volatility_results.append({
                'participant_id': pid,
                'task': task,
                'n_trials': len(rts),
                'mean_rt': np.mean(rts),
                'rmssd': rmssd,
                'cv': cv,
                'iiv': iiv,
                'mssd': mssd,
                'adj_volatility': adj_volatility
            })

        vol_df = pd.DataFrame(volatility_results)

        if len(vol_df) < 20:
            continue

        # Merge
        merged = master.merge(vol_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean RMSSD: {merged['rmssd'].mean():.1f} ms")
            print(f"    Mean CV: {merged['cv'].mean():.1f}%")

        # Regression
        outcomes = [
            ('rmssd', 'RMSSD (successive difference)'),
            ('cv', 'Coefficient of Variation'),
            ('iiv', 'IIV (SD)'),
            ('adj_volatility', 'Trend-adjusted volatility')
        ]

        for outcome, label in outcomes:
            if outcome not in merged.columns or merged[outcome].isna().all():
                continue

            try:
                formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged.dropna(subset=[outcome])).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'outcome': outcome,
                    'outcome_label': label,
                    'n': len(merged.dropna(subset=[outcome]))
                }

                if 'z_ucla' in model.params:
                    result['beta_ucla'] = model.params['z_ucla']
                    result['p_ucla'] = model.pvalues['z_ucla']
                    result['r_squared'] = model.rsquared

                all_results.append(result)

                if verbose and 'z_ucla' in model.params and outcome == 'rmssd':
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    UCLA → RMSSD: β={beta:.4f}, p={p:.4f} {sig}")

            except Exception as e:
                if verbose:
                    print(f"    {label}: ERROR - {e}")

        # Save task-specific
        vol_df.to_csv(OUTPUT_DIR / f"volatility_{task}.csv", index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "volatility_regression.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'volatility_regression.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run sequential dynamics analyses.
    """
    if verbose:
        print("=" * 70)
        print("SEQUENTIAL DYNAMICS ANALYSIS SUITE")
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
        for priority in [1, 2, 3]:
            priority_analyses = {k: v for k, v in ANALYSES.items() if v.priority == priority}
            for name, spec in priority_analyses.items():
                try:
                    results[name] = spec.function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("SEQUENTIAL DYNAMICS SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Sequential Dynamics Analyses:")
    print("-" * 60)
    for priority in [1, 2, 3]:
        priority_analyses = {k: v for k, v in ANALYSES.items() if v.priority == priority}
        if priority_analyses:
            label = {1: 'High', 2: 'Medium', 3: 'Low'}[priority]
            print(f"\n  Priority {priority} ({label}):")
            for name, spec in priority_analyses.items():
                print(f"    {name}")
                print(f"      {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Dynamics Analysis Suite")
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
