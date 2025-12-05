"""
WCST PE Gender-Specific Mechanism Suite
========================================

Deep mechanistic analysis to understand WHY lonely males show elevated
perseverative errors (PE) in WCST.

Research Question:
------------------
The Gold Standard analysis found UCLA x Gender interaction for WCST PE
(p = 0.025). What cognitive mechanism underlies this effect?

Analyses:
---------
1. error_burst: Consecutive PE sequences (burst length) by UCLA x Gender
2. rule_learning: Learning trajectory for first vs. subsequent rules
3. strategy_patterns: Win-Stay/Lose-Shift behavioral patterns
4. post_error_adjustment: Post-error slowing (PES) and accuracy recovery
5. rule_dimension: PE rates by rule dimension (color/shape/number)

Usage:
    python -m analysis.advanced.wcst_mechanism_deep_suite
    python -m analysis.advanced.wcst_mechanism_deep_suite --analysis error_burst
    python -m analysis.advanced.wcst_mechanism_deep_suite --list

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
import ast
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from analysis.utils.data_loader_utils import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, DEFAULT_RT_MAX
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "wcst_mechanism_deep"
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

def load_wcst_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare WCST trial-level data with master dataset."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    # Load WCST trials
    wcst_path = RESULTS_DIR / '4b_wcst_trials.csv'
    if not wcst_path.exists():
        raise FileNotFoundError(f"WCST trials file not found: {wcst_path}")

    trials = pd.read_csv(wcst_path, encoding='utf-8')
    trials.columns = trials.columns.str.lower()

    # Normalize participant_id
    if 'participantid' in trials.columns and 'participant_id' not in trials.columns:
        trials = trials.rename(columns={'participantid': 'participant_id'})
    elif 'participantid' in trials.columns:
        trials = trials.drop(columns=['participantid'])

    # Parse extra field for PE
    def _parse_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except (ValueError, SyntaxError):
            return {}

    trials['extra_dict'] = trials['extra'].apply(_parse_extra)
    trials['is_pe'] = trials['extra_dict'].apply(lambda x: x.get('isPE', False))
    trials['is_npe'] = trials['extra_dict'].apply(lambda x: x.get('isNPE', False))

    # RT column
    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'reactiontimems'
    if rt_col in trials.columns:
        trials['rt'] = trials[rt_col]

    # Filter valid RT
    if 'rt' in trials.columns:
        trials = trials[(trials['rt'] > DEFAULT_RT_MIN) & (trials['rt'] < DEFAULT_RT_MAX)].copy()

    # Sort trials
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'timestamp']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols).reset_index(drop=True)

    # Add trial number per participant
    trials['trial_num'] = trials.groupby('participant_id').cumcount() + 1

    return master, trials


def get_dass_controlled_formula(outcome: str) -> str:
    """Get DASS-controlled regression formula per CLAUDE.md."""
    return f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"


# =============================================================================
# ANALYSIS 1: ERROR BURST ANALYSIS
# =============================================================================

@register_analysis(
    name="error_burst",
    description="Consecutive PE sequences (burst length) by UCLA x Gender"
)
def analyze_error_burst(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze consecutive PE bursts.

    Hypothesis: Lonely males show longer PE bursts (more perseverative).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: ERROR BURST PATTERNS")
        print("=" * 70)

    master, trials = load_wcst_data()

    # Calculate burst statistics per participant
    burst_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 20:
            continue

        pdata = pdata.reset_index(drop=True)
        pe_sequence = pdata['is_pe'].astype(int).values

        # Find PE bursts (consecutive PEs)
        burst_lengths = []
        current_burst = 0

        for is_pe in pe_sequence:
            if is_pe:
                current_burst += 1
            else:
                if current_burst > 0:
                    burst_lengths.append(current_burst)
                current_burst = 0

        # Don't forget last burst
        if current_burst > 0:
            burst_lengths.append(current_burst)

        if len(burst_lengths) == 0:
            burst_lengths = [0]

        burst_results.append({
            'participant_id': pid,
            'n_bursts': len(burst_lengths),
            'mean_burst_length': np.mean(burst_lengths),
            'max_burst_length': np.max(burst_lengths),
            'total_pe': sum(burst_lengths),
            'n_trials': len(pdata)
        })

    if len(burst_results) < 20:
        if verbose:
            print("  Insufficient data for burst analysis")
        return pd.DataFrame()

    burst_df = pd.DataFrame(burst_results)
    merged = master.merge(burst_df, on='participant_id', how='inner')

    if verbose:
        print(f"\n  N = {len(merged)}")
        print(f"  Mean burst count: {merged['n_bursts'].mean():.1f}")
        print(f"  Mean burst length: {merged['mean_burst_length'].mean():.2f}")

    all_results = []

    # Test UCLA effects on burst metrics
    for metric in ['mean_burst_length', 'max_burst_length', 'n_bursts']:
        try:
            formula = get_dass_controlled_formula(metric)
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            # Extract coefficients
            for term in ['z_ucla', 'C(gender_male)[T.1]', 'z_ucla:C(gender_male)[T.1]']:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = term.replace('z_ucla', 'UCLA').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"  {metric} ~ {label}: beta={beta:.3f}, p={p:.4f}{sig}")

                    all_results.append({
                        'outcome': metric,
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged)
                    })

        except Exception as e:
            if verbose:
                print(f"  Error for {metric}: {e}")

    # Gender-stratified correlations
    if verbose:
        print("\n  Gender-Stratified Correlations:")

    for gender, gender_label in [(0, 'Female'), (1, 'Male')]:
        gdata = merged[merged['gender_male'] == gender]
        if len(gdata) >= 10:
            r, p = stats.pearsonr(gdata['z_ucla'], gdata['mean_burst_length'])
            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    {gender_label}: UCLA-BurstLength r={r:.3f}, p={p:.4f}{sig}")

            all_results.append({
                'outcome': 'mean_burst_length',
                'term': f'UCLA_corr_{gender_label}',
                'beta': r,
                'p': p,
                'n': len(gdata)
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "error_burst_results.csv", index=False, encoding='utf-8-sig')

    # Save participant-level data
    burst_df.to_csv(OUTPUT_DIR / "error_burst_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'error_burst_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 2: RULE LEARNING TRAJECTORY
# =============================================================================

@register_analysis(
    name="rule_learning",
    description="Learning trajectory for first vs. subsequent rules"
)
def analyze_rule_learning(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze rule learning trajectories.

    Hypothesis: Lonely males show slower learning for later rules (flexibility deficit).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: RULE LEARNING TRAJECTORY")
        print("=" * 70)

    master, trials = load_wcst_data()

    # Identify rule transitions
    # A rule switch typically happens when the previous correct dimension changes

    if 'currentrule' not in trials.columns and 'current_rule' not in trials.columns:
        if verbose:
            print("  No rule column found - attempting to infer from consecutive correct responses")

        # Alternative: Analyze accuracy over trial blocks
        learning_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 40:
                continue

            pdata = pdata.reset_index(drop=True)
            n_trials = len(pdata)

            # Split into quarters
            quarter_size = n_trials // 4
            quarters = []

            for q in range(4):
                start = q * quarter_size
                end = (q + 1) * quarter_size if q < 3 else n_trials
                q_data = pdata.iloc[start:end]

                quarters.append({
                    'accuracy': q_data['correct'].mean(),
                    'pe_rate': q_data['is_pe'].mean() * 100,
                    'mean_rt': q_data['rt'].mean() if 'rt' in q_data.columns else np.nan
                })

            # Learning slope (accuracy improvement)
            acc_slope = (quarters[3]['accuracy'] - quarters[0]['accuracy']) / 3
            pe_slope = (quarters[3]['pe_rate'] - quarters[0]['pe_rate']) / 3

            learning_results.append({
                'participant_id': pid,
                'acc_q1': quarters[0]['accuracy'],
                'acc_q4': quarters[3]['accuracy'],
                'acc_slope': acc_slope,
                'pe_q1': quarters[0]['pe_rate'],
                'pe_q4': quarters[3]['pe_rate'],
                'pe_slope': pe_slope,
                'n_trials': n_trials
            })

        if len(learning_results) < 20:
            if verbose:
                print("  Insufficient data")
            return pd.DataFrame()

        learning_df = pd.DataFrame(learning_results)
        merged = master.merge(learning_df, on='participant_id', how='inner')

        if verbose:
            print(f"\n  N = {len(merged)}")
            print(f"  Mean accuracy slope: {merged['acc_slope'].mean():.4f}")
            print(f"  Mean PE slope: {merged['pe_slope'].mean():.4f}")

    else:
        # Use actual rule column
        rule_col = 'currentrule' if 'currentrule' in trials.columns else 'current_rule'

        learning_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 40:
                continue

            pdata = pdata.reset_index(drop=True)

            # Find rule switches
            pdata['rule_changed'] = pdata[rule_col] != pdata[rule_col].shift(1)
            pdata['rule_num'] = pdata['rule_changed'].cumsum()

            # Trials to criterion per rule
            rule_stats = []
            for rule_num, rule_data in pdata.groupby('rule_num'):
                rule_stats.append({
                    'rule_num': rule_num,
                    'trials_to_complete': len(rule_data),
                    'pe_rate': rule_data['is_pe'].mean() * 100,
                    'accuracy': rule_data['correct'].mean()
                })

            if len(rule_stats) < 2:
                continue

            first_rule = rule_stats[0]
            later_rules = rule_stats[1:]

            learning_results.append({
                'participant_id': pid,
                'first_rule_trials': first_rule['trials_to_complete'],
                'first_rule_pe': first_rule['pe_rate'],
                'later_rules_trials': np.mean([r['trials_to_complete'] for r in later_rules]),
                'later_rules_pe': np.mean([r['pe_rate'] for r in later_rules]),
                'n_rules_completed': len(rule_stats)
            })

        if len(learning_results) < 20:
            if verbose:
                print("  Insufficient data")
            return pd.DataFrame()

        learning_df = pd.DataFrame(learning_results)
        merged = master.merge(learning_df, on='participant_id', how='inner')

    all_results = []

    # Test UCLA effects on learning metrics
    for metric in ['acc_slope', 'pe_slope']:
        if metric not in merged.columns:
            continue

        try:
            formula = get_dass_controlled_formula(metric)
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            for term in ['z_ucla', 'z_ucla:C(gender_male)[T.1]']:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = term.replace('z_ucla', 'UCLA').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"  {metric} ~ {label}: beta={beta:.4f}, p={p:.4f}{sig}")

                    all_results.append({
                        'outcome': metric,
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged)
                    })

        except Exception as e:
            if verbose:
                print(f"  Error for {metric}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "rule_learning_results.csv", index=False, encoding='utf-8-sig')
    learning_df.to_csv(OUTPUT_DIR / "rule_learning_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'rule_learning_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: WIN-STAY/LOSE-SHIFT STRATEGY
# =============================================================================

@register_analysis(
    name="strategy_patterns",
    description="Win-Stay/Lose-Shift behavioral patterns"
)
def analyze_strategy_patterns(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze Win-Stay/Lose-Shift patterns.

    Hypothesis: Lonely males show excessive "Lose-Stay" (perseveration after negative feedback).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: WIN-STAY/LOSE-SHIFT STRATEGY")
        print("=" * 70)

    master, trials = load_wcst_data()

    strategy_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 20:
            continue

        pdata = pdata.reset_index(drop=True)

        # Add previous trial info
        pdata['prev_correct'] = pdata['correct'].shift(1)
        pdata['prev_response'] = pdata.get('response', pdata.get('selectedcard', None))

        if pdata['prev_response'] is None:
            # If no response column, use PE pattern
            pdata['same_response'] = pdata['is_pe']  # Proxy: PE means "same" incorrect response
        else:
            pdata['same_response'] = pdata['prev_response'] == pdata.get('response', pdata.get('selectedcard'))

        # Filter out first trial
        pdata = pdata.iloc[1:].copy()

        # Win-Stay: After correct, stayed (correct again)
        win_trials = pdata[pdata['prev_correct'] == True]
        if len(win_trials) > 0:
            win_stay_rate = (win_trials['correct'] == True).mean()
        else:
            win_stay_rate = np.nan

        # Lose-Shift: After incorrect, shifted (changed response)
        lose_trials = pdata[pdata['prev_correct'] == False]
        if len(lose_trials) > 0:
            # Lose-Shift = correct after error (shifted to correct)
            lose_shift_rate = (lose_trials['correct'] == True).mean()
            # Lose-Stay (perseveration) = PE after error
            lose_stay_rate = lose_trials['is_pe'].mean()
        else:
            lose_shift_rate = np.nan
            lose_stay_rate = np.nan

        strategy_results.append({
            'participant_id': pid,
            'win_stay_rate': win_stay_rate,
            'lose_shift_rate': lose_shift_rate,
            'lose_stay_rate': lose_stay_rate,  # Key metric for perseveration
            'n_win_trials': len(win_trials),
            'n_lose_trials': len(lose_trials)
        })

    if len(strategy_results) < 20:
        if verbose:
            print("  Insufficient data")
        return pd.DataFrame()

    strategy_df = pd.DataFrame(strategy_results)
    merged = master.merge(strategy_df, on='participant_id', how='inner')

    if verbose:
        print(f"\n  N = {len(merged)}")
        print(f"  Mean Win-Stay rate: {merged['win_stay_rate'].mean():.3f}")
        print(f"  Mean Lose-Shift rate: {merged['lose_shift_rate'].mean():.3f}")
        print(f"  Mean Lose-Stay (perseveration) rate: {merged['lose_stay_rate'].mean():.3f}")

    all_results = []

    # Key metric: Lose-Stay (perseveration after negative feedback)
    for metric in ['lose_stay_rate', 'lose_shift_rate', 'win_stay_rate']:
        merged_clean = merged.dropna(subset=[metric])
        if len(merged_clean) < 20:
            continue

        try:
            formula = get_dass_controlled_formula(metric)
            model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

            for term in ['z_ucla', 'z_ucla:C(gender_male)[T.1]']:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = term.replace('z_ucla', 'UCLA').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"  {metric} ~ {label}: beta={beta:.4f}, p={p:.4f}{sig}")

                    all_results.append({
                        'outcome': metric,
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged_clean)
                    })

        except Exception as e:
            if verbose:
                print(f"  Error for {metric}: {e}")

    # Gender-stratified correlations for Lose-Stay
    if verbose:
        print("\n  Gender-Stratified Correlations (Lose-Stay):")

    for gender, gender_label in [(0, 'Female'), (1, 'Male')]:
        gdata = merged.dropna(subset=['lose_stay_rate'])
        gdata = gdata[gdata['gender_male'] == gender]
        if len(gdata) >= 10:
            r, p = stats.pearsonr(gdata['z_ucla'], gdata['lose_stay_rate'])
            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    {gender_label}: UCLA-LoseStay r={r:.3f}, p={p:.4f}{sig}")

            all_results.append({
                'outcome': 'lose_stay_rate',
                'term': f'UCLA_corr_{gender_label}',
                'beta': r,
                'p': p,
                'n': len(gdata)
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "strategy_patterns_results.csv", index=False, encoding='utf-8-sig')
    strategy_df.to_csv(OUTPUT_DIR / "strategy_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'strategy_patterns_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 4: POST-ERROR ADJUSTMENT
# =============================================================================

@register_analysis(
    name="post_error_adjustment",
    description="Post-error slowing (PES) and accuracy recovery"
)
def analyze_post_error_adjustment(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze post-error behavioral adjustments.

    Hypothesis: Lonely males show reduced PES (less error monitoring) or
    reduced post-error accuracy recovery.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: POST-ERROR BEHAVIORAL ADJUSTMENTS")
        print("=" * 70)

    master, trials = load_wcst_data()

    if 'rt' not in trials.columns:
        if verbose:
            print("  No RT column - skipping PES analysis")
        return pd.DataFrame()

    adjustment_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 30:
            continue

        pdata = pdata.reset_index(drop=True)

        # Add previous trial info
        pdata['prev_correct'] = pdata['correct'].shift(1)
        pdata['prev_rt'] = pdata['rt'].shift(1)

        # Filter out first trial
        pdata = pdata.iloc[1:].copy()

        # Post-Error Slowing (PES)
        post_error = pdata[pdata['prev_correct'] == False]
        post_correct = pdata[pdata['prev_correct'] == True]

        if len(post_error) >= 5 and len(post_correct) >= 5:
            pes = post_error['rt'].mean() - post_correct['rt'].mean()
        else:
            pes = np.nan

        # Post-Error Accuracy (PEA)
        if len(post_error) >= 5:
            post_error_accuracy = post_error['correct'].mean()
        else:
            post_error_accuracy = np.nan

        # Post-Correct Accuracy
        if len(post_correct) >= 5:
            post_correct_accuracy = post_correct['correct'].mean()
        else:
            post_correct_accuracy = np.nan

        # Recovery difference
        if not np.isnan(post_error_accuracy) and not np.isnan(post_correct_accuracy):
            accuracy_recovery = post_correct_accuracy - post_error_accuracy
        else:
            accuracy_recovery = np.nan

        adjustment_results.append({
            'participant_id': pid,
            'pes': pes,
            'post_error_accuracy': post_error_accuracy,
            'post_correct_accuracy': post_correct_accuracy,
            'accuracy_recovery': accuracy_recovery,
            'n_post_error': len(post_error),
            'n_post_correct': len(post_correct)
        })

    if len(adjustment_results) < 20:
        if verbose:
            print("  Insufficient data")
        return pd.DataFrame()

    adj_df = pd.DataFrame(adjustment_results)
    merged = master.merge(adj_df, on='participant_id', how='inner')

    if verbose:
        print(f"\n  N = {len(merged)}")
        print(f"  Mean PES: {merged['pes'].mean():.1f} ms")
        print(f"  Mean post-error accuracy: {merged['post_error_accuracy'].mean():.3f}")

    all_results = []

    # Test UCLA effects
    for metric in ['pes', 'post_error_accuracy', 'accuracy_recovery']:
        merged_clean = merged.dropna(subset=[metric])
        if len(merged_clean) < 20:
            continue

        try:
            formula = get_dass_controlled_formula(metric)
            model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

            for term in ['z_ucla', 'z_ucla:C(gender_male)[T.1]']:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = term.replace('z_ucla', 'UCLA').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"  {metric} ~ {label}: beta={beta:.3f}, p={p:.4f}{sig}")

                    all_results.append({
                        'outcome': metric,
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged_clean)
                    })

        except Exception as e:
            if verbose:
                print(f"  Error for {metric}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "post_error_adjustment_results.csv", index=False, encoding='utf-8-sig')
    adj_df.to_csv(OUTPUT_DIR / "post_error_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'post_error_adjustment_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 5: RULE DIMENSION ANALYSIS
# =============================================================================

@register_analysis(
    name="rule_dimension",
    description="PE rates by rule dimension (color/shape/number)"
)
def analyze_rule_dimension(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze PE rates by rule dimension.

    Hypothesis: UCLA effect might be dimension-specific
    (e.g., stronger for shape/number than color).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: PE BY RULE DIMENSION")
        print("=" * 70)

    master, trials = load_wcst_data()

    # Check for rule dimension column
    rule_col = None
    for cand in ['currentrule', 'current_rule', 'rule', 'dimension']:
        if cand in trials.columns:
            rule_col = cand
            break

    if rule_col is None:
        if verbose:
            print("  No rule dimension column found")
        return pd.DataFrame()

    dimension_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 20:
            continue

        result = {'participant_id': pid}

        for dim in pdata[rule_col].unique():
            dim_data = pdata[pdata[rule_col] == dim]
            if len(dim_data) >= 5:
                result[f'pe_rate_{dim}'] = dim_data['is_pe'].mean() * 100
                result[f'n_trials_{dim}'] = len(dim_data)

        dimension_results.append(result)

    if len(dimension_results) < 20:
        if verbose:
            print("  Insufficient data")
        return pd.DataFrame()

    dim_df = pd.DataFrame(dimension_results)
    merged = master.merge(dim_df, on='participant_id', how='inner')

    all_results = []

    # Get available dimension columns
    pe_cols = [c for c in merged.columns if c.startswith('pe_rate_')]

    if verbose:
        print(f"\n  N = {len(merged)}")
        print(f"  Dimensions found: {[c.replace('pe_rate_', '') for c in pe_cols]}")

    for pe_col in pe_cols:
        dim = pe_col.replace('pe_rate_', '')
        merged_clean = merged.dropna(subset=[pe_col])

        if len(merged_clean) < 20:
            continue

        if verbose:
            print(f"\n  {dim.upper()} dimension:")

        try:
            formula = get_dass_controlled_formula(pe_col)
            model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

            for term in ['z_ucla', 'z_ucla:C(gender_male)[T.1]']:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = term.replace('z_ucla', 'UCLA').replace('C(gender_male)[T.1]', 'Male').replace(':', ' x ')

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    {label}: beta={beta:.3f}, p={p:.4f}{sig}")

                    all_results.append({
                        'dimension': dim,
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged_clean)
                    })

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "rule_dimension_results.csv", index=False, encoding='utf-8-sig')
    dim_df.to_csv(OUTPUT_DIR / "rule_dimension_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'rule_dimension_results.csv'}")

    return results_df


# =============================================================================
# SUMMARY VISUALIZATION
# =============================================================================

def create_summary_visualization(verbose: bool = True) -> None:
    """Create summary figure for WCST mechanism analysis."""
    if verbose:
        print("\n" + "=" * 70)
        print("CREATING SUMMARY VISUALIZATION")
        print("=" * 70)

    # Load results
    result_files = {
        'Error Burst': OUTPUT_DIR / 'error_burst_results.csv',
        'Strategy': OUTPUT_DIR / 'strategy_patterns_results.csv',
        'Post-Error': OUTPUT_DIR / 'post_error_adjustment_results.csv',
        'Learning': OUTPUT_DIR / 'rule_learning_results.csv'
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
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(effects_df))
    colors = ['#E74C3C' if p < 0.05 else '#3498DB' for p in effects_df['P-value']]

    ax.barh(y_pos, effects_df['Beta'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['Analysis']}: {row['Outcome']}" for _, row in effects_df.iterrows()])
    ax.set_xlabel('UCLA x Gender Interaction (Beta)')
    ax.set_title('WCST Mechanism Analysis: UCLA x Gender Effects\n(Red = p < 0.05)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mechanism_summary_forest.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {OUTPUT_DIR / 'mechanism_summary_forest.png'}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run WCST mechanism analyses."""
    if verbose:
        print("=" * 70)
        print("WCST PE GENDER-SPECIFIC MECHANISM SUITE")
        print("=" * 70)
        print("\nResearch Question: Why do lonely males show elevated")
        print("perseverative errors in WCST?")

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
        print("WCST MECHANISM SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable WCST Mechanism Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WCST PE Mechanism Analysis Suite")
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
