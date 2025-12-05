"""
Control Strategy Analysis Suite
===============================

Examines proactive vs reactive cognitive control strategies in relation to loneliness.

Based on Braver's (2012) Dual Mechanisms of Control (DMC) framework:
- Proactive control: Sustained maintenance of goal-relevant information
- Reactive control: Transient retrieval of goal-relevant information when needed

Analyses:
- congruency_proportion: Stroop congruency proportion effect (proactive adaptation)
- rule_anticipation: WCST pre-switch slowing (proactive rule monitoring)
- post_conflict_adjustment: Reactive control after conflict trials
- dmc_composite: Combined proactive/reactive index
- ucla_relationship: UCLA effects on control strategy (DASS-controlled)

Usage:
    python -m analysis.advanced.control_strategy_suite              # Run all
    python -m analysis.advanced.control_strategy_suite --analysis congruency_proportion
    python -m analysis.advanced.control_strategy_suite --list

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
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "control_strategy"
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


def register_analysis(name: str, description: str, source_script: str = "control_strategy_suite.py"):
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

def load_strategy_data() -> pd.DataFrame:
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
    name="congruency_proportion",
    description="Stroop congruency proportion effect (proactive adaptation)"
)
def analyze_congruency_proportion(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze how participants adapt to the overall proportion of congruent trials.

    Proactive control is indicated by:
    - Smaller interference effect in high-congruent contexts
    - Larger interference effect in low-congruent contexts

    Note: Since we cannot manipulate proportion, we examine adaptation across blocks.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CONGRUENCY PROPORTION EFFECT ANALYSIS")
        print("=" * 70)

    master = load_strategy_data()
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

    # Compute per-participant metrics
    prop_results = []

    for pid, pdata in valid.groupby('participant_id'):
        if len(pdata) < 40:
            continue

        # Split into halves
        mid = len(pdata) // 2
        first_half = pdata.iloc[:mid]
        second_half = pdata.iloc[mid:]

        # Compute congruency proportion and interference for each half
        results_by_half = []

        for half_name, half_data in [('first', first_half), ('second', second_half)]:
            cong_prop = (half_data[cond_col] == 'congruent').mean()

            cong_rt = half_data[half_data[cond_col] == 'congruent'][rt_col].mean()
            incong_rt = half_data[half_data[cond_col] == 'incongruent'][rt_col].mean()

            if np.isnan(cong_rt) or np.isnan(incong_rt):
                continue

            interference = incong_rt - cong_rt

            results_by_half.append({
                'half': half_name,
                'cong_prop': cong_prop,
                'interference': interference
            })

        if len(results_by_half) < 2:
            continue

        # Compute adaptation: correlation between proportion and interference
        # (negative correlation = proactive control)
        first = results_by_half[0]
        second = results_by_half[1]

        # Change in interference from first to second half
        interference_change = second['interference'] - first['interference']

        # Overall proactive index: how much does interference adapt?
        overall_interference = (first['interference'] + second['interference']) / 2
        overall_cong_prop = (first['cong_prop'] + second['cong_prop']) / 2

        prop_results.append({
            'participant_id': pid,
            'first_interference': first['interference'],
            'second_interference': second['interference'],
            'interference_change': interference_change,
            'overall_interference': overall_interference,
            'overall_cong_prop': overall_cong_prop
        })

    if len(prop_results) < 20:
        if verbose:
            print(f"  Insufficient data ({len(prop_results)} participants)")
        return pd.DataFrame()

    prop_df = pd.DataFrame(prop_results)
    merged = master.merge(prop_df, on='participant_id', how='inner')

    if len(merged) < 20:
        if verbose:
            print("  Insufficient merged data")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"  Mean overall interference: {merged['overall_interference'].mean():.1f} ms")
        print(f"  Mean interference change: {merged['interference_change'].mean():.1f} ms")

    all_results = []

    # Test UCLA effects
    for metric in ['overall_interference', 'interference_change']:
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
    results_df.to_csv(OUTPUT_DIR / "congruency_proportion.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'congruency_proportion.csv'}")

    return results_df


@register_analysis(
    name="rule_anticipation",
    description="WCST pre-switch slowing (proactive rule monitoring)"
)
def analyze_rule_anticipation(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze pre-switch slowing in WCST as indicator of proactive control.

    Proactive control is indicated by:
    - Slowing before rule switches (anticipation of change)
    - Better performance after anticipated switches
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RULE ANTICIPATION ANALYSIS (WCST)")
        print("=" * 70)

    master = load_strategy_data()
    trials = load_trial_data('wcst')

    if len(trials) < 100:
        if verbose:
            print("  Insufficient WCST trial data")
        return pd.DataFrame()

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'reactiontimems'
    if rt_col not in trials.columns:
        if verbose:
            print("  Missing RT column")
        return pd.DataFrame()

    sort_col = 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
    if sort_col not in trials.columns:
        sort_col = trials.columns[0]

    trials = trials.sort_values(['participant_id', sort_col]).reset_index(drop=True)

    # Identify rule switches
    if 'ruleatthattime' in trials.columns:
        rule_col = 'ruleatthattime'
    elif 'rule' in trials.columns:
        rule_col = 'rule'
    else:
        if verbose:
            print("  Missing rule column")
        return pd.DataFrame()

    # Compute pre-switch metrics per participant
    anticipation_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 30:
            continue

        pdata = pdata.reset_index(drop=True)

        # Identify switch points
        pdata['prev_rule'] = pdata[rule_col].shift(1)
        pdata['is_switch'] = pdata[rule_col] != pdata['prev_rule']

        switch_indices = pdata[pdata['is_switch'] == True].index.tolist()

        if len(switch_indices) < 3:
            continue

        # Filter valid RT
        valid = pdata[(pdata[rt_col] > DEFAULT_RT_MIN) & (pdata[rt_col] < 5000)].copy()

        # Compute pre-switch RT (5 trials before switch)
        pre_switch_rts = []
        post_switch_rts = []
        baseline_rts = []

        for sw_idx in switch_indices:
            # Pre-switch (5 trials before)
            pre_start = max(0, sw_idx - 5)
            pre_trials = pdata.iloc[pre_start:sw_idx]
            pre_valid = pre_trials[pre_trials[rt_col] > DEFAULT_RT_MIN][rt_col]
            if len(pre_valid) >= 2:
                pre_switch_rts.extend(pre_valid.tolist())

            # Post-switch (5 trials after)
            post_end = min(len(pdata), sw_idx + 5)
            post_trials = pdata.iloc[sw_idx:post_end]
            post_valid = post_trials[post_trials[rt_col] > DEFAULT_RT_MIN][rt_col]
            if len(post_valid) >= 2:
                post_switch_rts.extend(post_valid.tolist())

        # Baseline: trials not near switches (at least 5 away)
        switch_window = set()
        for sw_idx in switch_indices:
            for i in range(max(0, sw_idx - 5), min(len(pdata), sw_idx + 5)):
                switch_window.add(i)

        baseline_trials = pdata[~pdata.index.isin(switch_window)]
        baseline_valid = baseline_trials[baseline_trials[rt_col] > DEFAULT_RT_MIN][rt_col]
        baseline_rts = baseline_valid.tolist()

        if len(pre_switch_rts) < 5 or len(baseline_rts) < 10:
            continue

        mean_pre_switch = np.mean(pre_switch_rts)
        mean_post_switch = np.mean(post_switch_rts) if post_switch_rts else np.nan
        mean_baseline = np.mean(baseline_rts)

        # Pre-switch slowing = pre-switch RT - baseline RT (positive = slowing)
        pre_switch_slowing = mean_pre_switch - mean_baseline

        # Switch cost = post-switch - baseline
        switch_cost = mean_post_switch - mean_baseline if not np.isnan(mean_post_switch) else np.nan

        anticipation_results.append({
            'participant_id': pid,
            'pre_switch_slowing': pre_switch_slowing,
            'switch_cost': switch_cost,
            'mean_pre_switch_rt': mean_pre_switch,
            'mean_post_switch_rt': mean_post_switch,
            'mean_baseline_rt': mean_baseline,
            'n_switches': len(switch_indices)
        })

    if len(anticipation_results) < 20:
        if verbose:
            print(f"  Insufficient data ({len(anticipation_results)} participants)")
        return pd.DataFrame()

    antic_df = pd.DataFrame(anticipation_results)
    merged = master.merge(antic_df, on='participant_id', how='inner')

    if len(merged) < 20:
        if verbose:
            print("  Insufficient merged data")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"  Mean pre-switch slowing: {merged['pre_switch_slowing'].mean():.1f} ms")
        print(f"  Mean switch cost: {merged['switch_cost'].mean():.1f} ms")

        # Test if pre-switch slowing is significant
        t_stat, p_val = stats.ttest_1samp(merged['pre_switch_slowing'].dropna(), 0)
        sig = "*" if p_val < 0.05 else ""
        print(f"  Pre-switch slowing vs 0: t={t_stat:.2f}, p={p_val:.4f}{sig}")

    all_results = []

    # Test UCLA effects
    for metric in ['pre_switch_slowing', 'switch_cost']:
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
                    'task': 'wcst',
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
    results_df.to_csv(OUTPUT_DIR / "rule_anticipation.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'rule_anticipation.csv'}")

    return results_df


@register_analysis(
    name="post_conflict_adjustment",
    description="Reactive control after conflict trials (Stroop)"
)
def analyze_post_conflict_adjustment(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze reactive control through post-conflict adjustments.

    Reactive control is indicated by:
    - RT adjustments following incongruent trials
    - Accuracy improvements after detecting conflict
    """
    if verbose:
        print("\n" + "=" * 70)
        print("POST-CONFLICT ADJUSTMENT ANALYSIS")
        print("=" * 70)

    master = load_strategy_data()
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

    # Compute post-conflict metrics per participant
    adjust_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 40:
            continue

        pdata = pdata.reset_index(drop=True)

        # Add previous trial info
        pdata['prev_cond'] = pdata[cond_col].shift(1)
        pdata['valid_rt'] = (pdata[rt_col] > DEFAULT_RT_MIN) & (pdata[rt_col] < STROOP_RT_MAX)
        pdata = pdata[pdata['valid_rt'] & pdata['correct']].copy()

        # Post-incongruent (reactive after conflict)
        post_incong = pdata[pdata['prev_cond'] == 'incongruent']
        post_cong = pdata[pdata['prev_cond'] == 'congruent']

        if len(post_incong) < 5 or len(post_cong) < 5:
            continue

        # RT adjustment after conflict
        rt_post_incong = post_incong[rt_col].mean()
        rt_post_cong = post_cong[rt_col].mean()
        reactive_adjustment = rt_post_incong - rt_post_cong  # Typically negative (faster after conflict)

        # Accuracy after conflict
        post_incong_all = pdata[pdata['prev_cond'] == 'incongruent']
        post_cong_all = pdata[pdata['prev_cond'] == 'congruent']

        # Note: Since we filtered correct only, use original data for accuracy
        orig = trials[trials['participant_id'] == pid].copy()
        orig['prev_cond'] = orig[cond_col].shift(1)

        post_incong_acc = orig[orig['prev_cond'] == 'incongruent']['correct'].mean()
        post_cong_acc = orig[orig['prev_cond'] == 'congruent']['correct'].mean()
        reactive_acc = post_incong_acc - post_cong_acc

        adjust_results.append({
            'participant_id': pid,
            'reactive_rt_adjustment': reactive_adjustment,
            'reactive_acc_adjustment': reactive_acc,
            'rt_post_incongruent': rt_post_incong,
            'rt_post_congruent': rt_post_cong
        })

    if len(adjust_results) < 20:
        if verbose:
            print(f"  Insufficient data ({len(adjust_results)} participants)")
        return pd.DataFrame()

    adjust_df = pd.DataFrame(adjust_results)
    merged = master.merge(adjust_df, on='participant_id', how='inner')

    if len(merged) < 20:
        if verbose:
            print("  Insufficient merged data")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"  Mean reactive RT adjustment: {merged['reactive_rt_adjustment'].mean():.1f} ms")

    all_results = []

    # Test UCLA effects
    for metric in ['reactive_rt_adjustment', 'reactive_acc_adjustment']:
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
    results_df.to_csv(OUTPUT_DIR / "post_conflict_adjustment.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'post_conflict_adjustment.csv'}")

    return results_df


@register_analysis(
    name="dmc_composite",
    description="Dual Mechanisms of Control composite index"
)
def analyze_dmc_composite(verbose: bool = True) -> pd.DataFrame:
    """
    Create composite indices of proactive and reactive control based on DMC framework.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DMC COMPOSITE INDEX")
        print("=" * 70)

    master = load_strategy_data()

    # Load individual metric files if they exist
    proactive_metrics = []
    reactive_metrics = []

    # Try to load pre-computed metrics
    prop_file = OUTPUT_DIR / "congruency_proportion.csv"
    antic_file = OUTPUT_DIR / "rule_anticipation.csv"
    adjust_file = OUTPUT_DIR / "post_conflict_adjustment.csv"

    # If files don't exist, run the analyses first
    if not prop_file.exists():
        analyze_congruency_proportion(verbose=False)
    if not antic_file.exists():
        analyze_rule_anticipation(verbose=False)
    if not adjust_file.exists():
        analyze_post_conflict_adjustment(verbose=False)

    # Collect raw participant-level data
    stroop_trials = load_trial_data('stroop')
    wcst_trials = load_trial_data('wcst')

    dmc_results = []

    for pid in master['participant_id'].unique():
        result = {'participant_id': pid}

        # Stroop proactive: interference change
        stroop_p = stroop_trials[stroop_trials['participant_id'] == pid] if len(stroop_trials) > 0 else pd.DataFrame()
        if len(stroop_p) >= 40:
            rt_col = 'rt_ms' if 'rt_ms' in stroop_p.columns else 'rt'
            cond_col = 'type' if 'type' in stroop_p.columns else 'condition'

            if rt_col in stroop_p.columns and cond_col in stroop_p.columns:
                valid = stroop_p[
                    (stroop_p[rt_col] > DEFAULT_RT_MIN) &
                    (stroop_p[rt_col] < STROOP_RT_MAX) &
                    (stroop_p['correct'] == True)
                ]

                if len(valid) >= 20:
                    cong_rt = valid[valid[cond_col] == 'congruent'][rt_col].mean()
                    incong_rt = valid[valid[cond_col] == 'incongruent'][rt_col].mean()
                    result['stroop_interference'] = incong_rt - cong_rt if not (np.isnan(cong_rt) or np.isnan(incong_rt)) else np.nan

        # WCST proactive: pre-switch slowing (simplified - switch cost)
        wcst_p = wcst_trials[wcst_trials['participant_id'] == pid] if len(wcst_trials) > 0 else pd.DataFrame()
        if len(wcst_p) >= 30:
            rt_col = 'rt_ms' if 'rt_ms' in wcst_p.columns else 'reactiontimems'
            if rt_col in wcst_p.columns:
                result['wcst_mean_rt'] = wcst_p[wcst_p[rt_col] > DEFAULT_RT_MIN][rt_col].mean()

        dmc_results.append(result)

    if len(dmc_results) < 20:
        if verbose:
            print("  Insufficient data")
        return pd.DataFrame()

    dmc_df = pd.DataFrame(dmc_results)
    merged = master.merge(dmc_df, on='participant_id', how='inner')

    if verbose:
        print(f"  N = {len(merged)}")

    # Create DMC indices by z-scoring available metrics
    if 'stroop_interference' in merged.columns:
        std = merged['stroop_interference'].std()
        merged['z_stroop_interference'] = (merged['stroop_interference'] - merged['stroop_interference'].mean()) / std if std > 0 else 0

    # Proactive index (lower interference = more proactive)
    if 'z_stroop_interference' in merged.columns:
        merged['proactive_index'] = -merged['z_stroop_interference']  # Reverse so higher = more proactive

        if verbose:
            print(f"  Mean proactive index: {merged['proactive_index'].mean():.3f}")

    all_results = []

    # Test UCLA effect on proactive index
    if 'proactive_index' in merged.columns:
        merged_clean = merged.dropna(subset=['proactive_index'])
        if len(merged_clean) >= 20:
            try:
                formula = "proactive_index ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"  UCLA -> proactive_index: beta={beta:.4f}, p={p:.4f}{sig}")

                    all_results.append({
                        'metric': 'proactive_index',
                        'beta_ucla': beta,
                        'se_ucla': model.bse['z_ucla'],
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(merged_clean)
                    })

            except Exception as e:
                if verbose:
                    print(f"  Regression error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "dmc_composite.csv", index=False, encoding='utf-8-sig')
    merged[['participant_id', 'stroop_interference', 'proactive_index']].dropna().to_csv(
        OUTPUT_DIR / "dmc_by_participant.csv", index=False, encoding='utf-8-sig'
    )

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'dmc_composite.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run control strategy analyses.
    """
    if verbose:
        print("=" * 70)
        print("CONTROL STRATEGY ANALYSIS SUITE")
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
        print("CONTROL STRATEGY SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Control Strategy Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control Strategy Analysis Suite")
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
