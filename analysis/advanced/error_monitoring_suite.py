"""
Error Monitoring Analysis Suite
===============================

Examines error detection, correction, and adaptation patterns in relation to loneliness.

Analyses:
- pes_decomposition: Adaptive vs maladaptive post-error slowing
- error_awareness: Error awareness proxy measures
- error_cascade: Consecutive error patterns and recovery
- conflict_detection: Conflict monitoring efficiency (Stroop)
- ucla_relationship: UCLA effects on error monitoring (DASS-controlled)

Theoretical Background:
    Error monitoring involves:
    1. Error detection (ACC/dACC activity)
    2. Error awareness
    3. Adaptive behavioral adjustment
    4. Performance recovery

    Loneliness may impair error monitoring through:
    - Attentional resource depletion
    - Self-focused rumination interfering with task monitoring
    - Reduced motivation for performance optimization

Usage:
    python -m analysis.advanced.error_monitoring_suite              # Run all
    python -m analysis.advanced.error_monitoring_suite --analysis pes_decomposition
    python -m analysis.advanced.error_monitoring_suite --list

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

# Project imports
from analysis.utils.data_loader_utils import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "error_monitoring"
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


def register_analysis(name: str, description: str, source_script: str = "error_monitoring_suite.py"):
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

def load_monitoring_data() -> pd.DataFrame:
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


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="pes_decomposition",
    description="Decompose PES into adaptive vs maladaptive components"
)
def analyze_pes_decomposition(verbose: bool = True) -> pd.DataFrame:
    """
    Decompose Post-Error Slowing (PES) into adaptive and maladaptive components.

    Adaptive PES: Slowing accompanied by accuracy improvement
    Maladaptive PES: Slowing without accuracy benefit (or with accuracy cost)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PES DECOMPOSITION ANALYSIS")
        print("=" * 70)

    master = load_monitoring_data()
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

        # Sort by participant and trial
        sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
        if sort_col not in trials.columns:
            sort_col = trials.columns[0]

        trials = trials.sort_values(['participant_id', sort_col]).reset_index(drop=True)

        # Compute PES metrics per participant
        pes_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.reset_index(drop=True)

            # Identify error and correct trials
            pdata['prev_correct'] = pdata['correct'].shift(1)
            pdata['valid_rt'] = (pdata[rt_col] > DEFAULT_RT_MIN) & (pdata[rt_col] < STROOP_RT_MAX)

            # Post-error trials (trial after error)
            post_error = pdata[(pdata['prev_correct'] == False) & pdata['valid_rt']]
            # Post-correct trials
            post_correct = pdata[(pdata['prev_correct'] == True) & pdata['valid_rt']]

            if len(post_error) < 3 or len(post_correct) < 10:
                continue

            # Basic PES
            pes = post_error[rt_col].mean() - post_correct[rt_col].mean()

            # Post-error accuracy
            post_error_acc = post_error['correct'].mean()
            post_correct_acc = post_correct['correct'].mean()
            post_error_acc_diff = post_error_acc - post_correct_acc

            # Classify PES type
            if pes > 0 and post_error_acc_diff >= 0:
                pes_type = 'adaptive'
            elif pes > 0 and post_error_acc_diff < 0:
                pes_type = 'maladaptive'
            elif pes <= 0:
                pes_type = 'no_slowing'
            else:
                pes_type = 'unknown'

            # RT variability after errors
            post_error_cv = post_error[rt_col].std() / post_error[rt_col].mean() if post_error[rt_col].mean() > 0 else np.nan
            post_correct_cv = post_correct[rt_col].std() / post_correct[rt_col].mean() if post_correct[rt_col].mean() > 0 else np.nan
            cv_reduction = post_correct_cv - post_error_cv  # Positive = less variable after error

            pes_results.append({
                'participant_id': pid,
                'pes': pes,
                'post_error_acc': post_error_acc,
                'post_correct_acc': post_correct_acc,
                'acc_diff': post_error_acc_diff,
                'pes_type': pes_type,
                'cv_reduction': cv_reduction,
                'n_errors': len(post_error)
            })

        if len(pes_results) < 20:
            if verbose:
                print(f"    Insufficient PES data ({len(pes_results)} participants)")
            continue

        pes_df = pd.DataFrame(pes_results)
        merged = master.merge(pes_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean PES: {merged['pes'].mean():.1f} ms")
            type_counts = merged['pes_type'].value_counts()
            for ptype, count in type_counts.items():
                print(f"    {ptype}: {count} ({count/len(merged)*100:.1f}%)")

        # Test UCLA effects on PES components
        for metric in ['pes', 'acc_diff', 'cv_reduction']:
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
    results_df.to_csv(OUTPUT_DIR / "pes_decomposition.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'pes_decomposition.csv'}")

    return results_df


@register_analysis(
    name="error_cascade",
    description="Analyze consecutive error patterns and recovery"
)
def analyze_error_cascade(verbose: bool = True) -> pd.DataFrame:
    """
    Examine patterns of consecutive errors (error bursts) and recovery time.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ERROR CASCADE ANALYSIS")
        print("=" * 70)

    master = load_monitoring_data()
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

        sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
        if sort_col not in trials.columns:
            sort_col = trials.columns[0]

        trials = trials.sort_values(['participant_id', sort_col]).reset_index(drop=True)

        # Compute error burst metrics per participant
        cascade_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.reset_index(drop=True)
            correct_seq = pdata['correct'].values

            # Find error bursts (consecutive errors)
            bursts = []
            burst_length = 0

            for i, c in enumerate(correct_seq):
                if not c:  # Error
                    burst_length += 1
                else:  # Correct
                    if burst_length >= 2:  # At least 2 consecutive errors
                        bursts.append(burst_length)
                    burst_length = 0

            # Handle final burst
            if burst_length >= 2:
                bursts.append(burst_length)

            n_errors = (~pdata['correct']).sum()
            n_bursts = len(bursts)
            mean_burst_length = np.mean(bursts) if bursts else 0
            max_burst_length = max(bursts) if bursts else 0

            # Burst frequency (proportion of errors in bursts)
            errors_in_bursts = sum(bursts) if bursts else 0
            burst_proportion = errors_in_bursts / n_errors if n_errors > 0 else 0

            # Recovery time (trials from burst end to next correct)
            # Simplified: just use burst frequency and length
            cascade_results.append({
                'participant_id': pid,
                'n_errors': n_errors,
                'n_bursts': n_bursts,
                'mean_burst_length': mean_burst_length,
                'max_burst_length': max_burst_length,
                'burst_proportion': burst_proportion,
                'error_rate': n_errors / len(pdata)
            })

        if len(cascade_results) < 20:
            continue

        cascade_df = pd.DataFrame(cascade_results)
        merged = master.merge(cascade_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean bursts: {merged['n_bursts'].mean():.1f}")
            print(f"    Mean burst length: {merged['mean_burst_length'].mean():.2f}")

        # Test UCLA effects
        for metric in ['n_bursts', 'mean_burst_length', 'burst_proportion']:
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
    results_df.to_csv(OUTPUT_DIR / "error_cascade.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'error_cascade.csv'}")

    return results_df


@register_analysis(
    name="conflict_detection",
    description="Conflict monitoring efficiency in Stroop task"
)
def analyze_conflict_detection(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze conflict monitoring using Stroop congruency effects.

    Based on Botvinick's conflict monitoring theory:
    - Conflict signal triggers increased control
    - Measured via sequential congruency effects (CSE)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CONFLICT DETECTION ANALYSIS (STROOP)")
        print("=" * 70)

    master = load_monitoring_data()
    trials = load_trial_data('stroop')

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

    sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
    if sort_col not in trials.columns:
        sort_col = trials.columns[0]

    trials = trials.sort_values(['participant_id', sort_col]).reset_index(drop=True)

    # Filter valid trials
    valid = trials[
        (trials[rt_col] > DEFAULT_RT_MIN) &
        (trials[rt_col] < STROOP_RT_MAX) &
        (trials['correct'] == True)
    ].copy()

    # Add previous trial condition
    valid['prev_cond'] = valid.groupby('participant_id')[cond_col].shift(1)

    # Compute CSE per participant
    cse_results = []

    for pid, pdata in valid.groupby('participant_id'):
        if len(pdata) < 30:
            continue

        pdata = pdata.dropna(subset=['prev_cond'])

        # Get RT by current x previous condition
        conditions = {}
        for prev in ['congruent', 'incongruent']:
            for curr in ['congruent', 'incongruent']:
                subset = pdata[(pdata['prev_cond'] == prev) & (pdata[cond_col] == curr)]
                if len(subset) >= 3:
                    conditions[f'{prev[:1]}{curr[:1]}'] = subset[rt_col].mean()

        if len(conditions) < 4:
            continue

        # CSE = (cI - cC) - (iI - iC)
        # = Conflict effect after congruent - conflict effect after incongruent
        # Positive CSE = conflict adaptation
        try:
            cse = (conditions['ci'] - conditions['cc']) - (conditions['ii'] - conditions['ic'])

            # Also compute basic interference effect
            incongruent_rt = pdata[pdata[cond_col] == 'incongruent'][rt_col].mean()
            congruent_rt = pdata[pdata[cond_col] == 'congruent'][rt_col].mean()
            interference = incongruent_rt - congruent_rt

            cse_results.append({
                'participant_id': pid,
                'cse': cse,
                'interference': interference,
                'rt_cc': conditions.get('cc'),
                'rt_ci': conditions.get('ci'),
                'rt_ic': conditions.get('ic'),
                'rt_ii': conditions.get('ii')
            })

        except Exception:
            continue

    if len(cse_results) < 20:
        if verbose:
            print(f"  Insufficient CSE data ({len(cse_results)} participants)")
        return pd.DataFrame()

    cse_df = pd.DataFrame(cse_results)
    merged = master.merge(cse_df, on='participant_id', how='inner')

    if len(merged) < 20:
        if verbose:
            print("  Insufficient merged data")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"  Mean CSE: {merged['cse'].mean():.1f} ms")
        print(f"  Mean interference: {merged['interference'].mean():.1f} ms")

        # Test if CSE is significantly different from 0
        t_stat, p_val = stats.ttest_1samp(merged['cse'].dropna(), 0)
        sig = "*" if p_val < 0.05 else ""
        print(f"  CSE vs 0: t={t_stat:.2f}, p={p_val:.4f}{sig}")

    all_results = []

    # Test UCLA effects
    for metric in ['cse', 'interference']:
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
                    print(f"  UCLA -> {metric}: beta={beta:.4f}, p={p:.4f}{sig}")

                all_results.append({
                    'task': 'stroop',
                    'metric': metric,
                    'beta_ucla': beta,
                    'se_ucla': model.bse['z_ucla'],
                    'p_ucla': p,
                    'r_squared': model.rsquared,
                    'n': len(merged_clean)
                })

        except Exception as e:
            if verbose:
                print(f"  {metric}: Regression error - {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "conflict_detection.csv", index=False, encoding='utf-8-sig')
    cse_df.to_csv(OUTPUT_DIR / "cse_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'conflict_detection.csv'}")

    return results_df


@register_analysis(
    name="error_awareness",
    description="Error awareness proxy measures"
)
def analyze_error_awareness(verbose: bool = True) -> pd.DataFrame:
    """
    Estimate error awareness through behavioral proxies:
    - Post-error RT variability reduction (focused attention after error)
    - Post-error accuracy improvement
    - Error correction speed
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ERROR AWARENESS ANALYSIS")
        print("=" * 70)

    master = load_monitoring_data()
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

        sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
        if sort_col not in trials.columns:
            sort_col = trials.columns[0]

        trials = trials.sort_values(['participant_id', sort_col]).reset_index(drop=True)

        # Compute awareness metrics per participant
        awareness_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.reset_index(drop=True)
            pdata['prev_correct'] = pdata['correct'].shift(1)
            pdata['valid_rt'] = (pdata[rt_col] > DEFAULT_RT_MIN) & (pdata[rt_col] < STROOP_RT_MAX)

            post_error = pdata[(pdata['prev_correct'] == False) & pdata['valid_rt']]
            post_correct = pdata[(pdata['prev_correct'] == True) & pdata['valid_rt']]

            if len(post_error) < 3 or len(post_correct) < 10:
                continue

            # 1. RT variability reduction after error
            post_error_cv = post_error[rt_col].std() / post_error[rt_col].mean() if post_error[rt_col].mean() > 0 else np.nan
            post_correct_cv = post_correct[rt_col].std() / post_correct[rt_col].mean() if post_correct[rt_col].mean() > 0 else np.nan
            cv_reduction = post_correct_cv - post_error_cv  # Positive = less variable after error

            # 2. Post-error accuracy
            post_error_acc = post_error['correct'].mean()

            # 3. Error recovery index: proportion of errors followed by correct
            pdata['next_correct'] = pdata['correct'].shift(-1)
            error_trials = pdata[pdata['correct'] == False]
            recovery_rate = error_trials['next_correct'].mean() if len(error_trials) > 3 else np.nan

            # Composite awareness index (z-scored components, averaged)
            awareness_results.append({
                'participant_id': pid,
                'cv_reduction': cv_reduction,
                'post_error_acc': post_error_acc,
                'recovery_rate': recovery_rate,
                'n_errors': len(post_error)
            })

        if len(awareness_results) < 20:
            continue

        aware_df = pd.DataFrame(awareness_results)

        # Z-score and create composite
        for col in ['cv_reduction', 'post_error_acc', 'recovery_rate']:
            std = aware_df[col].std()
            aware_df[f'z_{col}'] = (aware_df[col] - aware_df[col].mean()) / std if std > 0 else 0

        aware_df['awareness_index'] = (
            aware_df['z_cv_reduction'] +
            aware_df['z_post_error_acc'] +
            aware_df['z_recovery_rate']
        ) / 3

        merged = master.merge(aware_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean recovery rate: {merged['recovery_rate'].mean():.2%}")

        # Test UCLA effects
        for metric in ['cv_reduction', 'post_error_acc', 'recovery_rate', 'awareness_index']:
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
    results_df.to_csv(OUTPUT_DIR / "error_awareness.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'error_awareness.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run error monitoring analyses.
    """
    if verbose:
        print("=" * 70)
        print("ERROR MONITORING ANALYSIS SUITE")
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
        print("ERROR MONITORING SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Error Monitoring Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error Monitoring Analysis Suite")
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
