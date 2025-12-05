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
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from analysis.utils.data_loader_utils import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR
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
# ANALYSES
# =============================================================================

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

                if 'z_ucla:C(gender_male)[T.1]' in model.params:
                    result['beta_ucla_x_gender'] = model.params['z_ucla:C(gender_male)[T.1]']
                    result['p_ucla_x_gender'] = model.pvalues['z_ucla:C(gender_male)[T.1]']

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
                    trials['trial_index'] = cand
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

                if 'z_ucla:C(gender_male)[T.1]' in model.params:
                    result['beta_ucla_x_gender'] = model.params['z_ucla:C(gender_male)[T.1]']
                    result['p_ucla_x_gender'] = model.pvalues['z_ucla:C(gender_male)[T.1]']

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
                    'beta_ucla_x_gender': model.params.get('z_ucla:C(gender_male)[T.1]', np.nan),
                    'p_ucla_x_gender': model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan),
                    'r_squared': model.rsquared
                }
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
