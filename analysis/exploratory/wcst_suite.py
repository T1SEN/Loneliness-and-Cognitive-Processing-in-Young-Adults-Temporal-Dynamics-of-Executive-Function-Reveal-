"""
WCST Exploratory Analysis Suite
================================

Unified exploratory analyses for the WCST (Wisconsin Card Sorting Test).

Consolidates:
- wcst_learning_trajectory.py
- wcst_error_type_decomposition.py
- wcst_switching_dynamics_quick.py
- wcst_post_error_adaptation_quick.py
- wcst_mechanism_comprehensive.py

Usage:
    python -m analysis.exploratory.wcst_suite
    python -m analysis.exploratory.wcst_suite --analysis learning_trajectory

NOTE: These are EXPLORATORY analyses.
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
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from analysis.preprocessing import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors
from analysis.preprocessing import (
    prepare_gender_variable,
    find_interaction_term
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "wcst_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_wcst_extra(extra_str):
    """Parse WCST extra column."""
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}


def load_wcst_trials() -> pd.DataFrame:
    """Load WCST trial data."""
    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    # Parse extra column for isPE
    if 'extra' in df.columns:
        extra_parsed = df['extra'].apply(parse_wcst_extra)
        df['is_pe'] = extra_parsed.apply(lambda x: x.get('isPE', False))
    elif 'ispe' in df.columns:
        df['is_pe'] = df['ispe']
    else:
        df['is_pe'] = False

    # Standardize correct column
    if 'iscorrect' in df.columns:
        df['correct'] = df['iscorrect']
    elif 'is_correct' in df.columns:
        df['correct'] = df['is_correct']

    return df


def load_master_with_wcst() -> pd.DataFrame:
    """Load master dataset with WCST metrics."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master = standardize_predictors(master)
    return master


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_learning_trajectory(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze block-wise learning trajectory.

    Examines how accuracy/PE rate changes across trial blocks.
    """
    output_dir = OUTPUT_DIR / "learning_trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[LEARNING TRAJECTORY] Analyzing block-wise learning...")

    trials = load_wcst_trials()
    master = load_master_with_wcst()

    # Add trial block (10 trials per block)
    trials['trial_num'] = trials.groupby('participant_id').cumcount()
    trials['block'] = trials['trial_num'] // 10

    # Compute metrics by block per participant
    block_metrics = trials.groupby(['participant_id', 'block']).agg({
        'correct': 'mean',
        'is_pe': 'mean'
    }).reset_index()
    block_metrics.columns = ['participant_id', 'block', 'accuracy', 'pe_rate']

    # Compute learning slope per participant
    learning_results = []
    for pid in block_metrics['participant_id'].unique():
        pdata = block_metrics[block_metrics['participant_id'] == pid]

        if len(pdata) < 3:
            continue

        # Accuracy slope
        slope_acc, _, r_acc, p_acc, _ = stats.linregress(pdata['block'], pdata['accuracy'])

        # PE rate slope
        slope_pe, _, r_pe, p_pe, _ = stats.linregress(pdata['block'], pdata['pe_rate'])

        learning_results.append({
            'participant_id': pid,
            'accuracy_slope': slope_acc,
            'accuracy_r': r_acc,
            'pe_slope': slope_pe,
            'pe_r': r_pe,
            'n_blocks': len(pdata)
        })

    learning_df = pd.DataFrame(learning_results)
    analysis_df = learning_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean accuracy slope: {analysis_df['accuracy_slope'].mean():.4f}")

    # DASS-controlled regression on learning slope
    if len(analysis_df) >= 30:
        formula = "accuracy_slope ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'accuracy_slope'])).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → Learning: β={model.params.get('z_ucla', np.nan):.4f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    learning_df.to_csv(output_dir / "learning_trajectory_metrics.csv", index=False, encoding='utf-8-sig')
    return learning_df


def analyze_error_decomposition(verbose: bool = True) -> pd.DataFrame:
    """
    Decompose errors into perseverative vs non-perseverative.

    Tests whether UCLA differentially predicts error types.
    """
    output_dir = OUTPUT_DIR / "error_decomposition"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[ERROR DECOMPOSITION] Analyzing PE vs NPE...")

    trials = load_wcst_trials()
    master = load_master_with_wcst()

    # Compute error metrics per participant
    error_results = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid]
        n_trials = len(pdata)

        if n_trials < 20:
            continue

        n_errors = (pdata['correct'] == 0).sum()
        n_pe = pdata['is_pe'].sum()
        n_npe = n_errors - n_pe

        error_results.append({
            'participant_id': pid,
            'n_trials': n_trials,
            'n_errors': n_errors,
            'n_pe': n_pe,
            'n_npe': n_npe,
            'pe_rate': n_pe / n_trials * 100,
            'npe_rate': n_npe / n_trials * 100,
            'error_rate': n_errors / n_trials * 100,
            'pe_proportion': n_pe / n_errors * 100 if n_errors > 0 else 0
        })

    error_df = pd.DataFrame(error_results)

    # Drop conflicting columns from master before merge
    master_cols = [c for c in master.columns if c not in ['pe_rate', 'npe_rate', 'error_rate']]
    analysis_df = error_df.merge(master[master_cols], on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean PE rate: {analysis_df['pe_rate'].mean():.1f}%")
        print(f"  Mean NPE rate: {analysis_df['npe_rate'].mean():.1f}%")

    # Compare UCLA effects on PE vs NPE
    reg_results = []
    if len(analysis_df) >= 30:
        for dv in ['pe_rate', 'npe_rate']:
            formula = f"{dv} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', dv])).fit(cov_type='HC3')

            int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
            reg_results.append({
                'outcome': dv,
                'ucla_beta': model.params.get('z_ucla', np.nan),
                'ucla_p': model.pvalues.get('z_ucla', np.nan),
                'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
                'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
                'n': int(model.nobs),
                'r_squared': model.rsquared
            })

            if verbose:
                print(f"  UCLA → {dv}: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    error_df.to_csv(output_dir / "error_decomposition_metrics.csv", index=False, encoding='utf-8-sig')

    if reg_results:
        reg_df = pd.DataFrame(reg_results)
        reg_df.to_csv(output_dir / "error_decomposition_regression.csv", index=False, encoding='utf-8-sig')

    return error_df


def analyze_post_error_adaptation(verbose: bool = True) -> pd.DataFrame:
    """Analyze post-error adaptation in WCST."""
    output_dir = OUTPUT_DIR / "post_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[POST-ERROR] Analyzing post-error adaptation...")

    trials = load_wcst_trials()
    master = load_master_with_wcst()

    # Sort and add previous trial info
    trials = trials.sort_values(['participant_id', 'idx' if 'idx' in trials.columns else 'trial_index'])
    trials['prev_correct'] = trials.groupby('participant_id')['correct'].shift(1)
    trials['prev_pe'] = trials.groupby('participant_id')['is_pe'].shift(1)

    # Compute post-error accuracy
    pe_adaptation = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid].dropna(subset=['prev_correct'])

        if len(pdata) < 20:
            continue

        acc_after_error = pdata[pdata['prev_correct'] == 0]['correct'].mean()
        acc_after_correct = pdata[pdata['prev_correct'] == 1]['correct'].mean()
        acc_after_pe = pdata[pdata['prev_pe'] == 1]['correct'].mean()

        pe_adaptation.append({
            'participant_id': pid,
            'acc_after_error': acc_after_error,
            'acc_after_correct': acc_after_correct,
            'acc_after_pe': acc_after_pe,
            'post_error_improvement': acc_after_error - acc_after_correct
        })

    adapt_df = pd.DataFrame(pe_adaptation)
    analysis_df = adapt_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean post-error improvement: {analysis_df['post_error_improvement'].mean():.3f}")

    # DASS/나이/성별 통제 회귀로 UCLA 효과 검정
    reg_results = pd.DataFrame()
    clean_df = analysis_df.dropna(subset=[
        'post_error_improvement', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'
    ])
    if len(clean_df) >= 20:
        formula = "post_error_improvement ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=clean_df).fit(cov_type='HC3')
        int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
        reg_results = pd.DataFrame([{
            'ucla_beta': model.params.get('z_ucla', np.nan),
            'ucla_p': model.pvalues.get('z_ucla', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
            'n': len(clean_df),
            'formula': formula,
            'cov_type': 'HC3'
        }])
        if verbose:
            print(f"  UCLA effect (DASS/age/gender controlled): β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    adapt_df.to_csv(output_dir / "wcst_post_error_metrics.csv", index=False, encoding='utf-8-sig')
    if not reg_results.empty:
        reg_results.to_csv(output_dir / "wcst_post_error_regression.csv", index=False, encoding='utf-8-sig')

    return adapt_df


def analyze_switching_dynamics(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze rule-switching dynamics.

    Examines how participants adapt after rule changes:
    - Post-switch error rate (first 5 trials after rule change)
    - Trials-to-criterion (5 consecutive correct)
    - UCLA × Gender effects on switching efficiency

    Source: wcst_switching_dynamics_quick.py
    """
    output_dir = OUTPUT_DIR / "switching_dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[SWITCHING DYNAMICS] Analyzing rule-switching behavior...")

    trials = load_wcst_trials()
    master = load_master_with_wcst()

    # Find rule column
    rule_col = None
    for cand in ['ruleatthattime', 'rule_at_that_time', 'rule_at_time', 'rule']:
        if cand in trials.columns:
            rule_col = cand
            break

    if not rule_col:
        if verbose:
            print("  No rule column found. Skipping switching analysis.")
        return pd.DataFrame()

    # Sort by trial order
    trial_col = 'idx' if 'idx' in trials.columns else 'trial_index' if 'trial_index' in trials.columns else 'trialindex'
    if trial_col not in trials.columns:
        trial_col = trials.columns[0]

    trials = trials.sort_values(['participant_id', trial_col])

    # Detect rule changes and compute metrics
    switching_metrics = []

    for pid, grp in trials.groupby('participant_id'):
        rules = grp[rule_col].values
        is_correct = grp['correct'].values

        if len(rules) < 20:
            continue

        # Find rule change points
        rule_changes = [0]
        for i in range(1, len(rules)):
            if rules[i] != rules[i - 1]:
                rule_changes.append(i)

        if len(rule_changes) < 2:
            continue

        post_switch_errors = []
        trials_to_criterion = []

        for change_idx in rule_changes[1:]:
            window_end = min(change_idx + 10, len(grp))
            post_switch_trials = is_correct[change_idx:window_end]

            if len(post_switch_trials) >= 5:
                # Error rate in first 5 trials
                post_switch_errors.append(1 - post_switch_trials[:5].mean())

                # Trials to 5 consecutive correct
                criterion_met = False
                for j in range(change_idx, len(is_correct) - 4):
                    if is_correct[j:j+5].sum() == 5:
                        trials_to_criterion.append(j - change_idx)
                        criterion_met = True
                        break
                if not criterion_met:
                    trials_to_criterion.append(np.nan)

        switching_metrics.append({
            'participant_id': pid,
            'n_rule_changes': len(rule_changes) - 1,
            'post_switch_error_rate': np.nanmean(post_switch_errors) if post_switch_errors else np.nan,
            'avg_trials_to_criterion': np.nanmean(trials_to_criterion) if trials_to_criterion else np.nan
        })

    switching_df = pd.DataFrame(switching_metrics)
    analysis_df = switching_df.merge(master, on='participant_id', how='inner')
    analysis_df = analysis_df.dropna(subset=['post_switch_error_rate'])

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        if len(analysis_df) > 0:
            print(f"  Mean post-switch error rate: {analysis_df['post_switch_error_rate'].mean():.3f}")

    # Gender-stratified correlations
    for gender_val, label in [(1, 'Male'), (0, 'Female')]:
        subset = analysis_df[analysis_df['gender_male'] == gender_val]
        if len(subset) >= 10:
            valid = subset.dropna(subset=['ucla_total', 'post_switch_error_rate'])
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid['ucla_total'], valid['post_switch_error_rate'])
                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"  {label}: UCLA × post_switch_error: r={r:.3f}, p={p:.4f}{sig}")

    # DASS-controlled regression
    clean_df = analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'post_switch_error_rate'])
    if len(clean_df) >= 30:
        formula = "post_switch_error_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=clean_df).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → Switching: β={model.params.get('z_ucla', np.nan):.4f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        reg_df = pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'se': model.bse.values,
            'p': model.pvalues.values
        })
        reg_df.to_csv(output_dir / "switching_regression.csv", index=False, encoding='utf-8-sig')

    switching_df.to_csv(output_dir / "switching_metrics.csv", index=False, encoding='utf-8-sig')
    return switching_df


def analyze_mechanism(verbose: bool = True) -> pd.DataFrame:
    """
    Comprehensive WCST mechanism analysis.

    Examines:
    - Feedback sensitivity (RT/accuracy changes after positive/negative feedback)
    - Adaptive index (accuracy improvement relative to RT cost)
    - UCLA × Gender effects on feedback processing

    Source: wcst_mechanism_comprehensive.py
    """
    output_dir = OUTPUT_DIR / "mechanism"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[MECHANISM] Analyzing feedback sensitivity and adaptation...")

    trials = load_wcst_trials()
    master = load_master_with_wcst()

    # Find RT column
    rt_col = None
    for cand in ['reactiontimems', 'rt_ms', 'rt']:
        if cand in trials.columns:
            rt_col = cand
            break

    if not rt_col:
        if verbose:
            print("  No RT column found. Skipping mechanism analysis.")
        return pd.DataFrame()

    # Sort and compute previous trial feedback
    trial_col = 'idx' if 'idx' in trials.columns else 'trialindex' if 'trialindex' in trials.columns else 'trial_index'
    if trial_col not in trials.columns:
        trial_col = trials.columns[0]

    trials = trials.sort_values(['participant_id', trial_col])
    trials['prev_correct'] = trials.groupby('participant_id')['correct'].shift(1)

    mechanism_results = []

    for pid in trials['participant_id'].unique():
        pdata = trials[(trials['participant_id'] == pid) & (trials[rt_col] > 0)].copy()

        if len(pdata) < 20:
            continue

        # After negative feedback (previous error)
        post_neg = pdata[pdata['prev_correct'] == 0]
        # After positive feedback (previous correct)
        post_pos = pdata[pdata['prev_correct'] == 1]

        if len(post_neg) < 3 or len(post_pos) < 3:
            continue

        # RT changes
        rt_post_neg = post_neg[rt_col].mean()
        rt_post_pos = post_pos[rt_col].mean()
        rt_change = rt_post_neg - rt_post_pos  # positive = slowing after error

        # Accuracy changes
        acc_post_neg = post_neg['correct'].mean() * 100
        acc_post_pos = post_pos['correct'].mean() * 100
        acc_change = acc_post_neg - acc_post_pos  # positive = improvement after error

        # Adaptive index: accuracy improvement relative to RT cost
        adaptive_index = acc_change - (rt_change / 100)

        mechanism_results.append({
            'participant_id': pid,
            'rt_post_negative': rt_post_neg,
            'rt_post_positive': rt_post_pos,
            'rt_feedback_sensitivity': rt_change,
            'acc_post_negative': acc_post_neg,
            'acc_post_positive': acc_post_pos,
            'acc_feedback_sensitivity': acc_change,
            'adaptive_index': adaptive_index,
            'n_trials': len(pdata)
        })

    mechanism_df = pd.DataFrame(mechanism_results)
    analysis_df = mechanism_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        if len(analysis_df) > 0:
            print(f"  Mean RT feedback sensitivity: {analysis_df['rt_feedback_sensitivity'].mean():.1f} ms")
            print(f"  Mean adaptive index: {analysis_df['adaptive_index'].mean():.2f}")

    # Gender-stratified correlations on adaptive index
    for gender_val, label in [(1, 'Male'), (0, 'Female')]:
        subset = analysis_df[analysis_df['gender_male'] == gender_val]
        if len(subset) >= 10:
            valid = subset.dropna(subset=['ucla_total', 'adaptive_index'])
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid['ucla_total'], valid['adaptive_index'])
                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"  {label}: UCLA × adaptive_index: r={r:.3f}, p={p:.4f}{sig}")

    # DASS-controlled regression on adaptive index
    clean_df = analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'adaptive_index'])
    if len(clean_df) >= 30:
        formula = "adaptive_index ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=clean_df).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → Adaptive Index: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        reg_df = pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'se': model.bse.values,
            'p': model.pvalues.values
        })
        reg_df.to_csv(output_dir / "mechanism_regression.csv", index=False, encoding='utf-8-sig')

    mechanism_df.to_csv(output_dir / "mechanism_metrics.csv", index=False, encoding='utf-8-sig')
    return mechanism_df


ANALYSES = {
    'learning_trajectory': ('Block-wise learning trajectory', analyze_learning_trajectory),
    'error_decomposition': ('PE vs NPE decomposition', analyze_error_decomposition),
    'post_error': ('Post-error adaptation', analyze_post_error_adaptation),
    'switching_dynamics': ('Rule-switching dynamics', analyze_switching_dynamics),
    'mechanism': ('Feedback sensitivity and adaptation', analyze_mechanism),
}


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run WCST exploratory analyses."""
    if verbose:
        print("=" * 70)
        print("WCST EXPLORATORY ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        desc, func = ANALYSES[analysis]
        results[analysis] = func(verbose=verbose)
    else:
        for name, (desc, func) in ANALYSES.items():
            if verbose:
                print(f"\n[{name.upper()}] {desc}")
            try:
                results[name] = func(verbose=verbose)
            except Exception as e:
                print(f"  ERROR: {e}")

    if verbose:
        print(f"\nOutput: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WCST Exploratory Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()
    run(analysis=args.analysis, verbose=not args.quiet)
