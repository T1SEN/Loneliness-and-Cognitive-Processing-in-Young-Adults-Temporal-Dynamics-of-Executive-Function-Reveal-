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

from analysis.utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

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

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
    master['gender_male'] = (master['gender'] == 'male').astype(int)

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
    analysis_df = error_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean PE rate: {analysis_df['pe_rate'].mean():.1f}%")
        print(f"  Mean NPE rate: {analysis_df['npe_rate'].mean():.1f}%")

    # Compare UCLA effects on PE vs NPE
    if len(analysis_df) >= 30:
        for dv in ['pe_rate', 'npe_rate']:
            formula = f"{dv} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', dv])).fit(cov_type='HC3')

            if verbose:
                print(f"  UCLA → {dv}: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    error_df.to_csv(output_dir / "error_decomposition_metrics.csv", index=False, encoding='utf-8-sig')
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

    adapt_df.to_csv(output_dir / "wcst_post_error_metrics.csv", index=False, encoding='utf-8-sig')
    return adapt_df


ANALYSES = {
    'learning_trajectory': ('Block-wise learning trajectory', analyze_learning_trajectory),
    'error_decomposition': ('PE vs NPE decomposition', analyze_error_decomposition),
    'post_error': ('Post-error adaptation', analyze_post_error_adaptation),
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
