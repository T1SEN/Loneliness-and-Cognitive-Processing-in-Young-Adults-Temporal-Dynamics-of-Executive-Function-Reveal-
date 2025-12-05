"""
Attention Depletion Analysis Suite
==================================

Examines whether loneliness accelerates attention resource depletion
during prolonged cognitive tasks.

Analyses:
- fatigue_trajectory: Performance decline across task quartiles
- tau_accumulation: Ex-Gaussian tau increase rate over time
- depletion_recovery: Rest period recovery patterns
- sustained_attention_index: Composite sustained attention measure
- ucla_moderation: UCLA moderation of fatigue effects (DASS-controlled)

Theoretical Background:
    Loneliness may tax cognitive resources through:
    1. Increased vigilance/threat monitoring
    2. Self-regulatory depletion
    3. Sleep disruption effects
    4. Stress-induced attention narrowing

Usage:
    python -m analysis.advanced.attention_depletion_suite              # Run all
    python -m analysis.advanced.attention_depletion_suite --analysis fatigue_trajectory
    python -m analysis.advanced.attention_depletion_suite --list

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
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX, PRP_RT_MAX
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "attention_depletion"
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


def register_analysis(name: str, description: str, source_script: str = "attention_depletion_suite.py"):
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

def load_depletion_data() -> pd.DataFrame:
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

    # Handle participant_id
    if 'participantid' in df.columns and 'participant_id' in df.columns:
        df = df.drop(columns=['participantid'])
    elif 'participantid' in df.columns:
        df = df.rename(columns={'participantid': 'participant_id'})

    return df


def fit_exgaussian_moments(rts: np.ndarray) -> tuple:
    """Fit ex-Gaussian using method of moments."""
    rts = np.array(rts)
    rts = rts[(rts > 0) & np.isfinite(rts)]

    if len(rts) < 20:
        return np.nan, np.nan, np.nan

    m1 = np.mean(rts)
    m2 = np.var(rts)
    m3 = stats.moment(rts, moment=3)

    if m3 <= 0:
        return m1, np.sqrt(m2), 0.0

    tau = (m3 / 2) ** (1/3)
    mu = m1 - tau
    sigma_sq = m2 - tau**2

    if sigma_sq <= 0:
        sigma = np.sqrt(m2) * 0.5
    else:
        sigma = np.sqrt(sigma_sq)

    return mu, sigma, tau


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="fatigue_trajectory",
    description="Analyze performance decline across task quartiles"
)
def analyze_fatigue_trajectory(verbose: bool = True) -> pd.DataFrame:
    """
    Examine how RT and accuracy change across task quartiles.

    Tests whether UCLA moderates the rate of performance decline.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("FATIGUE TRAJECTORY ANALYSIS")
        print("=" * 70)

    master = load_depletion_data()
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

        # Get RT column
        if task == 'prp':
            rt_col = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 't2_rt'
            rt_max = PRP_RT_MAX
        else:
            rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
            rt_max = STROOP_RT_MAX

        if rt_col not in trials.columns:
            continue

        # Sort by trial order
        sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
        if sort_col not in trials.columns:
            sort_col = trials.columns[0]

        trials = trials.sort_values(['participant_id', sort_col])

        # Filter valid RT
        valid_trials = trials[
            (trials[rt_col] > DEFAULT_RT_MIN) &
            (trials[rt_col] < rt_max)
        ].copy()

        # Assign quartiles per participant
        quartile_results = []

        for pid, pdata in valid_trials.groupby('participant_id'):
            n_trials = len(pdata)
            if n_trials < 20:
                continue

            pdata = pdata.reset_index(drop=True)
            pdata['quartile'] = pd.qcut(range(n_trials), q=4, labels=[1, 2, 3, 4])

            # Compute metrics per quartile
            for q in [1, 2, 3, 4]:
                q_data = pdata[pdata['quartile'] == q]
                if len(q_data) < 5:
                    continue

                quartile_results.append({
                    'participant_id': pid,
                    'quartile': q,
                    'mean_rt': q_data[rt_col].mean(),
                    'sd_rt': q_data[rt_col].std(),
                    'cv_rt': q_data[rt_col].std() / q_data[rt_col].mean() if q_data[rt_col].mean() > 0 else np.nan,
                    'accuracy': q_data['correct'].mean() if 'correct' in q_data.columns else np.nan,
                    'n_trials': len(q_data)
                })

        if len(quartile_results) < 40:  # At least 10 participants x 4 quartiles
            continue

        q_df = pd.DataFrame(quartile_results)

        # Compute per-participant fatigue slopes (Q4 - Q1)
        fatigue_metrics = []

        for pid in q_df['participant_id'].unique():
            p_q = q_df[q_df['participant_id'] == pid]
            if len(p_q) < 4:
                continue

            q1 = p_q[p_q['quartile'] == 1]
            q4 = p_q[p_q['quartile'] == 4]

            if len(q1) == 0 or len(q4) == 0:
                continue

            rt_slope = q4['mean_rt'].values[0] - q1['mean_rt'].values[0]
            cv_slope = q4['cv_rt'].values[0] - q1['cv_rt'].values[0] if not pd.isna(q4['cv_rt'].values[0]) else np.nan
            acc_slope = q4['accuracy'].values[0] - q1['accuracy'].values[0] if not pd.isna(q4['accuracy'].values[0]) else np.nan

            fatigue_metrics.append({
                'participant_id': pid,
                'rt_fatigue_slope': rt_slope,
                'cv_fatigue_slope': cv_slope,
                'accuracy_fatigue_slope': acc_slope
            })

        if len(fatigue_metrics) < 20:
            continue

        fatigue_df = pd.DataFrame(fatigue_metrics)
        merged = master.merge(fatigue_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean RT fatigue slope: {merged['rt_fatigue_slope'].mean():.1f} ms")

        # Test UCLA moderation of fatigue
        for metric in ['rt_fatigue_slope', 'cv_fatigue_slope', 'accuracy_fatigue_slope']:
            merged_clean = merged.dropna(subset=[metric])
            if len(merged_clean) < 20:
                continue

            try:
                formula = f"{metric} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    UCLA -> {metric}: beta={beta:.4f}, p={p:.4f}{sig}")

                    all_results.append({
                        'task': task,
                        'metric': metric,
                        'beta_ucla': beta,
                        'se_ucla': model.bse['z_ucla'],
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(merged_clean)
                    })

            except Exception as e:
                if verbose:
                    print(f"    {metric}: Regression error - {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "fatigue_trajectory.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'fatigue_trajectory.csv'}")

    return results_df


@register_analysis(
    name="tau_accumulation",
    description="Ex-Gaussian tau accumulation rate over task progression"
)
def analyze_tau_accumulation(verbose: bool = True) -> pd.DataFrame:
    """
    Examine whether tau (attentional lapses) increases faster for lonely individuals.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TAU ACCUMULATION ANALYSIS")
        print("=" * 70)

    master = load_depletion_data()
    all_results = []
    all_tau_results = []  # Collect tau data from all tasks

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
        rt_max = STROOP_RT_MAX

        if rt_col not in trials.columns:
            continue

        sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
        if sort_col not in trials.columns:
            sort_col = trials.columns[0]

        trials = trials.sort_values(['participant_id', sort_col])

        valid_trials = trials[
            (trials[rt_col] > DEFAULT_RT_MIN) &
            (trials[rt_col] < rt_max) &
            (trials['correct'] == True)
        ].copy()

        # Fit ex-Gaussian per quartile per participant
        tau_results = []

        for pid, pdata in valid_trials.groupby('participant_id'):
            n_trials = len(pdata)
            if n_trials < 40:  # Need enough trials for 4 quartiles
                continue

            pdata = pdata.reset_index(drop=True)
            pdata['quartile'] = pd.qcut(range(n_trials), q=4, labels=[1, 2, 3, 4])

            tau_by_quartile = {}
            for q in [1, 2, 3, 4]:
                q_rts = pdata[pdata['quartile'] == q][rt_col].values
                if len(q_rts) >= 10:
                    _, _, tau = fit_exgaussian_moments(q_rts)
                    tau_by_quartile[q] = tau

            if len(tau_by_quartile) < 4:
                continue

            # Compute tau slope (Q4 - Q1)
            tau_slope = tau_by_quartile.get(4, np.nan) - tau_by_quartile.get(1, np.nan)

            if not np.isnan(tau_slope):
                tau_results.append({
                    'task': task,
                    'participant_id': pid,
                    'tau_q1': tau_by_quartile.get(1),
                    'tau_q2': tau_by_quartile.get(2),
                    'tau_q3': tau_by_quartile.get(3),
                    'tau_q4': tau_by_quartile.get(4),
                    'tau_slope': tau_slope,
                    'n_trials': n_trials
                })

        if len(tau_results) < 20:
            if verbose:
                print(f"    Insufficient tau results ({len(tau_results)} participants)")
            continue

        tau_df = pd.DataFrame(tau_results)
        merged = master.merge(tau_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean tau Q1: {merged['tau_q1'].mean():.1f} ms")
            print(f"    Mean tau Q4: {merged['tau_q4'].mean():.1f} ms")
            print(f"    Mean tau slope: {merged['tau_slope'].mean():.1f} ms")

        # Test UCLA effect on tau slope
        try:
            formula = "tau_slope ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']

                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"\n    UCLA -> tau_slope: beta={beta:.4f}, p={p:.4f}{sig}")

                all_results.append({
                    'task': task,
                    'metric': 'tau_slope',
                    'beta_ucla': beta,
                    'se_ucla': model.bse['z_ucla'],
                    'p_ucla': p,
                    'r_squared': model.rsquared,
                    'n': len(merged)
                })

        except Exception as e:
            if verbose:
                print(f"    Regression error: {e}")

        # Collect tau results from this task
        all_tau_results.extend(tau_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "tau_accumulation.csv", index=False, encoding='utf-8-sig')

    # Save detailed tau by quartile from all tasks
    if len(all_tau_results) > 0:
        all_tau_df = pd.DataFrame(all_tau_results)
        all_tau_df.to_csv(OUTPUT_DIR / "tau_by_quartile.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'tau_accumulation.csv'}")

    return results_df


@register_analysis(
    name="sustained_attention_index",
    description="Compute composite sustained attention measure"
)
def analyze_sustained_attention_index(verbose: bool = True) -> pd.DataFrame:
    """
    Create a composite index of sustained attention from multiple indicators:
    - RT variability trend
    - Tau slope
    - Accuracy decline
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SUSTAINED ATTENTION INDEX")
        print("=" * 70)

    master = load_depletion_data()
    all_components = []

    for task in ['stroop', 'wcst']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        if rt_col not in trials.columns:
            continue

        sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
        if sort_col not in trials.columns:
            sort_col = trials.columns[0]

        trials = trials.sort_values(['participant_id', sort_col])

        valid_trials = trials[
            (trials[rt_col] > DEFAULT_RT_MIN) &
            (trials[rt_col] < STROOP_RT_MAX)
        ].copy()

        # Compute multiple attention metrics per participant
        for pid, pdata in valid_trials.groupby('participant_id'):
            if len(pdata) < 40:
                continue

            pdata = pdata.reset_index(drop=True)
            pdata['quartile'] = pd.qcut(range(len(pdata)), q=4, labels=[1, 2, 3, 4])

            q1 = pdata[pdata['quartile'] == 1]
            q4 = pdata[pdata['quartile'] == 4]

            if len(q1) < 5 or len(q4) < 5:
                continue

            # 1. RT variability slope (CV change)
            cv_q1 = q1[rt_col].std() / q1[rt_col].mean() if q1[rt_col].mean() > 0 else np.nan
            cv_q4 = q4[rt_col].std() / q4[rt_col].mean() if q4[rt_col].mean() > 0 else np.nan
            cv_slope = cv_q4 - cv_q1 if not (np.isnan(cv_q1) or np.isnan(cv_q4)) else np.nan

            # 2. Tau slope
            correct_q1 = q1[q1['correct'] == True] if 'correct' in q1.columns else q1
            correct_q4 = q4[q4['correct'] == True] if 'correct' in q4.columns else q4

            if len(correct_q1) >= 10 and len(correct_q4) >= 10:
                _, _, tau_q1 = fit_exgaussian_moments(correct_q1[rt_col].values)
                _, _, tau_q4 = fit_exgaussian_moments(correct_q4[rt_col].values)
                tau_slope = tau_q4 - tau_q1 if not (np.isnan(tau_q1) or np.isnan(tau_q4)) else np.nan
            else:
                tau_slope = np.nan

            # 3. Accuracy decline
            if 'correct' in pdata.columns:
                acc_q1 = q1['correct'].mean()
                acc_q4 = q4['correct'].mean()
                acc_slope = acc_q4 - acc_q1  # Negative = decline
            else:
                acc_slope = np.nan

            # 4. Mean RT slope (slowing)
            rt_slope = q4[rt_col].mean() - q1[rt_col].mean()

            all_components.append({
                'participant_id': pid,
                'task': task,
                'cv_slope': cv_slope,
                'tau_slope': tau_slope,
                'acc_slope': acc_slope,
                'rt_slope': rt_slope
            })

    if len(all_components) < 40:
        if verbose:
            print("  Insufficient data for composite index")
        return pd.DataFrame()

    comp_df = pd.DataFrame(all_components)

    # Average across tasks for participants with multiple tasks
    avg_df = comp_df.groupby('participant_id').agg({
        'cv_slope': 'mean',
        'tau_slope': 'mean',
        'acc_slope': 'mean',
        'rt_slope': 'mean'
    }).reset_index()

    # Z-score components and create composite
    for col in ['cv_slope', 'tau_slope', 'acc_slope', 'rt_slope']:
        std = avg_df[col].std()
        avg_df[f'z_{col}'] = (avg_df[col] - avg_df[col].mean()) / std if std > 0 else 0

    # Composite: higher = worse sustained attention
    # (higher CV slope, higher tau slope, lower accuracy, higher RT slope)
    avg_df['sustained_attention_index'] = (
        avg_df['z_cv_slope'] +
        avg_df['z_tau_slope'] +
        (-avg_df['z_acc_slope']) +  # Reverse accuracy (decline is bad)
        avg_df['z_rt_slope']
    ) / 4

    # Merge with master and test UCLA effect
    merged = master.merge(avg_df, on='participant_id', how='inner')

    if len(merged) < 20:
        if verbose:
            print("  Insufficient merged data")
        return pd.DataFrame()

    if verbose:
        print(f"\n  Composite Index (N = {len(merged)})")
        print("  " + "-" * 50)
        print(f"    Mean index: {merged['sustained_attention_index'].mean():.3f}")
        print(f"    SD index: {merged['sustained_attention_index'].std():.3f}")

    # Test UCLA effect
    results = []
    try:
        formula = "sustained_attention_index ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=merged).fit(cov_type='HC3')

        if 'z_ucla' in model.params:
            beta = model.params['z_ucla']
            p = model.pvalues['z_ucla']

            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"\n    UCLA -> Sustained Attention Index: beta={beta:.4f}, p={p:.4f}{sig}")

            results.append({
                'metric': 'sustained_attention_index',
                'beta_ucla': beta,
                'se_ucla': model.bse['z_ucla'],
                'p_ucla': p,
                'r_squared': model.rsquared,
                'n': len(merged)
            })

    except Exception as e:
        if verbose:
            print(f"    Regression error: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "sustained_attention_index.csv", index=False, encoding='utf-8-sig')
    avg_df.to_csv(OUTPUT_DIR / "attention_components.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'sustained_attention_index.csv'}")

    return results_df


@register_analysis(
    name="ucla_moderation",
    description="UCLA moderation of time-on-task effects (DASS-controlled)"
)
def analyze_ucla_moderation(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether UCLA moderates the effect of time-on-task on performance.

    Uses trial-level mixed-effects model with UCLA x trial interaction.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA MODERATION OF TIME-ON-TASK EFFECTS")
        print("=" * 70)

    master = load_depletion_data()
    all_results = []

    for task in ['stroop']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        if rt_col not in trials.columns:
            continue

        sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
        if sort_col not in trials.columns:
            sort_col = trials.columns[0]

        trials = trials.sort_values(['participant_id', sort_col])

        valid_trials = trials[
            (trials[rt_col] > DEFAULT_RT_MIN) &
            (trials[rt_col] < STROOP_RT_MAX) &
            (trials['correct'] == True)
        ].copy()

        # Add trial number per participant
        valid_trials['trial_num'] = valid_trials.groupby('participant_id').cumcount()

        # Merge with master data
        merged = valid_trials.merge(
            master[['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male']],
            on='participant_id',
            how='inner'
        )

        if len(merged) < 1000:
            if verbose:
                print(f"    Insufficient merged trials ({len(merged)})")
            continue

        # Scale trial number to 0-1
        merged['trial_scaled'] = merged['trial_num'] / merged.groupby('participant_id')['trial_num'].transform('max')

        if verbose:
            print(f"    Trials: {len(merged)}")
            print(f"    Participants: {merged['participant_id'].nunique()}")

        # Test UCLA x trial interaction
        # Using aggregated approach for computational efficiency
        # Group by participant and trial quartile
        merged['trial_quartile'] = pd.cut(merged['trial_scaled'], bins=4, labels=[1, 2, 3, 4])

        agg = merged.groupby(['participant_id', 'trial_quartile']).agg({
            rt_col: 'mean',
            'z_ucla': 'first',
            'z_dass_dep': 'first',
            'z_dass_anx': 'first',
            'z_dass_str': 'first',
            'z_age': 'first',
            'gender_male': 'first'
        }).reset_index()

        agg['trial_quartile'] = agg['trial_quartile'].astype(int)

        try:
            formula = f"{rt_col} ~ trial_quartile * z_ucla + trial_quartile * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=agg).fit(cov_type='HC3')

            # Check for trial x UCLA interaction
            interaction_term = 'trial_quartile:z_ucla'
            if interaction_term in model.params:
                beta = model.params[interaction_term]
                p = model.pvalues[interaction_term]

                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"\n    Trial x UCLA interaction: beta={beta:.4f}, p={p:.4f}{sig}")

                all_results.append({
                    'task': task,
                    'effect': 'trial_x_ucla',
                    'beta': beta,
                    'se': model.bse[interaction_term],
                    'p': p,
                    'r_squared': model.rsquared,
                    'n_trials': len(agg),
                    'n_participants': agg['participant_id'].nunique()
                })

        except Exception as e:
            if verbose:
                print(f"    Model error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "ucla_moderation.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ucla_moderation.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run attention depletion analyses.
    """
    if verbose:
        print("=" * 70)
        print("ATTENTION DEPLETION ANALYSIS SUITE")
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
        for name, spec in ANALYSES.items():
            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("ATTENTION DEPLETION SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Attention Depletion Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Depletion Analysis Suite")
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
