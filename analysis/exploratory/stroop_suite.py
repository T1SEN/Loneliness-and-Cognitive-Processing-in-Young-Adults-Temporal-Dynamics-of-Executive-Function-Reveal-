"""
Stroop Exploratory Analysis Suite
=================================

Unified exploratory analyses for the Stroop task.

Consolidates:
- stroop_conflict_adaptation.py
- stroop_neutral_baseline.py
- stroop_exgaussian_decomposition.py
- stroop_post_error_adjustments.py
- stroop_cse_conflict_adaptation.py

Usage:
    python -m analysis.exploratory.stroop_suite
    python -m analysis.exploratory.stroop_suite --analysis conflict_adaptation

NOTE: These are EXPLORATORY analyses.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from analysis.utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX
)
from analysis.utils.modeling import standardize_predictors

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "stroop_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_stroop_trials() -> pd.DataFrame:
    """Load Stroop trial data."""
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    # Standardize RT column
    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    df['rt'] = df[rt_col]

    # Standardize congruency - actual data uses 'type' column with values: congruent, incongruent, neutral
    if 'type' in df.columns:
        df['congruent'] = df['type'].str.lower() == 'congruent'
        df['incongruent'] = df['type'].str.lower() == 'incongruent'
        df['is_neutral'] = df['type'].str.lower() == 'neutral'
    elif 'congruency' in df.columns:
        df['congruent'] = df['congruency'].str.lower() == 'congruent'
        df['incongruent'] = df['congruency'].str.lower() == 'incongruent'
        df['is_neutral'] = df['congruency'].str.lower() == 'neutral'
    elif 'iscongruent' in df.columns:
        # Boolean column: True/1 = congruent, False/0 = incongruent
        df['congruent'] = df['iscongruent'].astype(bool)
        df['incongruent'] = ~df['congruent']
        df['is_neutral'] = False

    # Filter valid trials
    df = df[
        (df['rt'] > DEFAULT_RT_MIN) &
        (df['rt'] < STROOP_RT_MAX) &
        (df['correct'] == 1)
    ].copy()

    return df


def load_master_with_stroop() -> pd.DataFrame:
    """Load master dataset with Stroop metrics."""
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

ANALYSES = {}


def analyze_conflict_adaptation(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze Congruency Sequence Effects (Gratton effect).

    Tests whether current trial congruency effect depends on previous trial.
    """
    output_dir = OUTPUT_DIR / "conflict_adaptation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[CONFLICT ADAPTATION] Analyzing congruency sequence effects...")

    trials = load_stroop_trials()
    master = load_master_with_stroop()

    # Sort and add previous trial info
    trials = trials.sort_values(['participant_id', 'idx' if 'idx' in trials.columns else 'trial'])
    trials['prev_congruent'] = trials.groupby('participant_id')['congruent'].shift(1)

    # Compute CSE per participant
    cse_results = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid].dropna(subset=['prev_congruent'])

        if len(pdata) < 20:
            continue

        # Current congruency effect after congruent vs incongruent
        after_cong = pdata[pdata['prev_congruent'] == True]
        after_incong = pdata[pdata['prev_congruent'] == False]

        # Congruency effect = Incongruent RT - Congruent RT
        ce_after_cong = after_cong[after_cong['incongruent'] == True]['rt'].mean() - after_cong[after_cong['congruent'] == True]['rt'].mean()
        ce_after_incong = after_incong[after_incong['incongruent'] == True]['rt'].mean() - after_incong[after_incong['congruent'] == True]['rt'].mean()

        cse = ce_after_cong - ce_after_incong  # Gratton effect

        cse_results.append({
            'participant_id': pid,
            'ce_after_congruent': ce_after_cong,
            'ce_after_incongruent': ce_after_incong,
            'cse': cse,
            'n_trials': len(pdata)
        })

    cse_df = pd.DataFrame(cse_results)
    analysis_df = cse_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean CSE (Gratton): {analysis_df['cse'].mean():.1f} ms")

    # DASS-controlled regression
    if len(analysis_df) >= 30:
        formula = "cse ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'cse'])).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → CSE: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'p': model.pvalues.values
        }).to_csv(output_dir / "cse_regression.csv", index=False, encoding='utf-8-sig')

    cse_df.to_csv(output_dir / "cse_metrics.csv", index=False, encoding='utf-8-sig')
    return cse_df


def analyze_neutral_baseline(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze facilitation vs interference using neutral baseline.

    Decomposes Stroop effect into:
    - Facilitation: Neutral - Congruent
    - Interference: Incongruent - Neutral
    """
    output_dir = OUTPUT_DIR / "neutral_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[NEUTRAL BASELINE] Analyzing facilitation vs interference...")

    trials = load_stroop_trials()
    master = load_master_with_stroop()

    # Check for neutral trials (is_neutral already set in load_stroop_trials)
    if 'is_neutral' not in trials.columns or not trials['is_neutral'].any():
        if verbose:
            print("  No neutral trials found. Skipping.")
        return pd.DataFrame()

    # Compute metrics per participant
    results = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid]

        rt_cong = pdata[pdata['congruent'] == True]['rt'].mean()
        rt_incong = pdata[pdata['incongruent'] == True]['rt'].mean()
        rt_neutral = pdata[pdata['is_neutral'] == True]['rt'].mean()

        if pd.isna(rt_neutral):
            continue

        results.append({
            'participant_id': pid,
            'rt_congruent': rt_cong,
            'rt_incongruent': rt_incong,
            'rt_neutral': rt_neutral,
            'facilitation': rt_neutral - rt_cong,
            'interference': rt_incong - rt_neutral,
            'stroop_effect': rt_incong - rt_cong
        })

    if not results:
        if verbose:
            print("  Insufficient neutral trial data.")
        return pd.DataFrame()

    baseline_df = pd.DataFrame(results)
    analysis_df = baseline_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean facilitation: {analysis_df['facilitation'].mean():.1f} ms")
        print(f"  Mean interference: {analysis_df['interference'].mean():.1f} ms")

    baseline_df.to_csv(output_dir / "neutral_baseline_metrics.csv", index=False, encoding='utf-8-sig')
    return baseline_df


def analyze_post_error(verbose: bool = True) -> pd.DataFrame:
    """Analyze post-error slowing in Stroop task."""
    output_dir = OUTPUT_DIR / "post_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[POST-ERROR] Analyzing post-error adjustments...")

    # Load all trials (including errors)
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    df['rt'] = df[rt_col]
    df = df[(df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < STROOP_RT_MAX)].copy()

    master = load_master_with_stroop()

    # Sort and add previous trial info
    df = df.sort_values(['participant_id', 'idx' if 'idx' in df.columns else 'trial'])
    df['prev_correct'] = df.groupby('participant_id')['correct'].shift(1)

    # Compute PES
    pes_results = []
    for pid in df['participant_id'].unique():
        pdata = df[(df['participant_id'] == pid) & (df['correct'] == 1)].dropna(subset=['prev_correct'])

        if len(pdata) < 20:
            continue

        rt_after_error = pdata[pdata['prev_correct'] == 0]['rt'].mean()
        rt_after_correct = pdata[pdata['prev_correct'] == 1]['rt'].mean()
        pes = rt_after_error - rt_after_correct if not pd.isna(rt_after_error) else np.nan

        pes_results.append({
            'participant_id': pid,
            'pes': pes,
            'n_post_error': (pdata['prev_correct'] == 0).sum(),
            'n_post_correct': (pdata['prev_correct'] == 1).sum()
        })

    pes_df = pd.DataFrame(pes_results)
    analysis_df = pes_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean PES: {analysis_df['pes'].mean():.1f} ms")

    # DASS-controlled regression
    if len(analysis_df) >= 30:
        formula = "pes ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'pes'])).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → PES: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    pes_df.to_csv(output_dir / "stroop_pes_metrics.csv", index=False, encoding='utf-8-sig')
    return pes_df


ANALYSES = {
    'conflict_adaptation': ('Congruency Sequence Effects (Gratton)', analyze_conflict_adaptation),
    'neutral_baseline': ('Facilitation vs Interference', analyze_neutral_baseline),
    'post_error': ('Post-Error Slowing', analyze_post_error),
}


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run Stroop exploratory analyses."""
    if verbose:
        print("=" * 70)
        print("STROOP EXPLORATORY ANALYSIS SUITE")
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
    parser = argparse.ArgumentParser(description="Stroop Exploratory Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()
    run(analysis=args.analysis, verbose=not args.quiet)
