"""
WCST Error Type Decomposition Suite
====================================

Decomposes WCST errors into perseverative (PE) and non-perseverative (NPE) types
to understand which error mechanism UCLA affects.

Error Types:
- Perseverative Error (PE): Continuing to respond based on a previously correct
  rule after the rule has changed. Indicates cognitive rigidity.
- Non-Perseverative Error (NPE): Errors that don't follow perseverative pattern.
  May indicate inattention, confusion, or random responding.
- Perseverative Response (PR): Any response matching previous rule, whether
  correct or not.

Key Research Questions:
1. Does UCLA affect PE more than NPE? (cognitive rigidity vs inattention)
2. Is the UCLA × Gender interaction specific to one error type?
3. What is the temporal pattern of PE vs NPE across the task?

Analyses:
- error_metrics: Compute PE/NPE/PR rates per participant
- ucla_error_types: Test UCLA effects on each error type (DASS-controlled)
- gender_stratified: Gender-specific error type patterns
- error_sequences: Sequential patterns of error types
- summary: Summary report

Usage:
    python -m analysis.advanced.wcst_error_decomposition_suite
    python -m analysis.advanced.wcst_error_decomposition_suite --list

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
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    find_interaction_term, apply_fdr_correction
)
from analysis.preprocessing.loaders import ensure_participant_id
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "wcst_error_decomposition"
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


def register_analysis(name: str, description: str, source_script: str = "wcst_error_decomposition_suite.py"):
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

def parse_wcst_extra(extra_str):
    """Parse the extra column which may contain error type flags."""
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except Exception:
        return {}


def load_wcst_trials() -> pd.DataFrame:
    """
    Load WCST trials and extract/compute error type flags.

    Returns DataFrame with isPE, isNPE, isPR columns.
    """
    path = RESULTS_DIR / '4b_wcst_trials.csv'
    if not path.exists():
        raise FileNotFoundError(f"WCST trials not found: {path}")

    df = pd.read_csv(path, encoding='utf-8')
    df = ensure_participant_id(df)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Handle correct column
    if 'correct' in df.columns:
        correct_vals = df['correct']
        if correct_vals.dtype == object:
            df['correct'] = correct_vals.astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df['correct'] = correct_vals.fillna(False).astype(bool)

    # Check if error flags exist directly
    has_pe = 'ispe' in df.columns and df['ispe'].notna().any()
    has_npe = 'isnpe' in df.columns and df['isnpe'].notna().any()
    has_pr = 'ispr' in df.columns and df['ispr'].notna().any()

    # If not available, try to parse from extra column
    if not has_pe or not has_npe:
        if 'extra' in df.columns:
            print("  Parsing error flags from extra column...")
            extra_dicts = df['extra'].apply(parse_wcst_extra)

            # Extract flags
            df['ispe'] = extra_dicts.apply(lambda x: x.get('isPE', False) if isinstance(x, dict) else False)
            df['isnpe'] = extra_dicts.apply(lambda x: x.get('isNPE', False) if isinstance(x, dict) else False)
            df['ispr'] = extra_dicts.apply(lambda x: x.get('isPR', False) if isinstance(x, dict) else False)
        else:
            # Compute error types based on rule matching logic
            print("  Computing error types from rule matching...")
            df = compute_error_types(df)

    # Convert to boolean
    for col in ['ispe', 'isnpe', 'ispr']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes'])
            else:
                df[col] = df[col].fillna(False).astype(bool)

    # Create error column (any error)
    df['is_error'] = ~df['correct']

    print(f"  WCST trials loaded: {len(df)} trials, {df['participant_id'].nunique()} participants")

    # Count error types
    n_pe = df['ispe'].sum() if 'ispe' in df.columns else 0
    n_npe = df['isnpe'].sum() if 'isnpe' in df.columns else 0
    n_errors = df['is_error'].sum()

    print(f"  Total errors: {n_errors}, PE: {n_pe}, NPE: {n_npe}")

    return df


def compute_error_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PE/NPE/PR based on rule matching.

    Perseverative response: Response matches the previous rule
    Perseverative error: Incorrect response that matches previous rule
    Non-perseverative error: Incorrect response that doesn't match previous rule
    """
    df = df.copy()

    # Sort by participant and trial
    df = df.sort_values(['participant_id', 'trialindex'])

    # Initialize columns
    df['ispr'] = False
    df['ispe'] = False
    df['isnpe'] = False

    # Track previous rule per participant
    df['prev_rule'] = df.groupby('participant_id')['ruleatthattime'].shift(1)

    # Identify rule changes
    df['rule_changed'] = df['ruleatthattime'] != df['prev_rule']

    # For each trial after a rule change, check if response matches old rule
    # This is a simplified heuristic - full computation would require matching logic

    # For now, mark errors as either PE or NPE based on whether it's immediately after rule change
    # Errors within 5 trials after rule change that follow old pattern = PE
    # Other errors = NPE

    # Track trials since rule change
    df['trials_since_change'] = 0
    current_pid = None
    trials_since = 0

    for idx in df.index:
        if df.loc[idx, 'participant_id'] != current_pid:
            current_pid = df.loc[idx, 'participant_id']
            trials_since = 0

        if df.loc[idx, 'rule_changed']:
            trials_since = 0
        else:
            trials_since += 1

        df.loc[idx, 'trials_since_change'] = trials_since

    # Classify errors
    # If error occurs within first 5 trials after rule change, consider as PE
    # Otherwise NPE
    error_mask = df['is_error']
    early_after_change = df['trials_since_change'] <= 5

    df.loc[error_mask & early_after_change, 'ispe'] = True
    df.loc[error_mask & ~early_after_change, 'isnpe'] = True

    return df


def load_master_with_standardization() -> pd.DataFrame:
    """Load master dataset with standardized predictors."""
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


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="error_metrics",
    description="Compute PE/NPE rates and ratios per participant"
)
def analyze_error_metrics(verbose: bool = True) -> pd.DataFrame:
    """Compute error type metrics for each participant."""
    if verbose:
        print("\n" + "=" * 70)
        print("WCST ERROR TYPE METRICS")
        print("=" * 70)

    trials = load_wcst_trials()

    results = []

    for pid, pdata in trials.groupby('participant_id'):
        n_trials = len(pdata)
        n_errors = pdata['is_error'].sum()
        n_pe = pdata['ispe'].sum() if 'ispe' in pdata.columns else 0
        n_npe = pdata['isnpe'].sum() if 'isnpe' in pdata.columns else 0
        n_correct = n_trials - n_errors

        # Rates (as proportion of total trials)
        error_rate = n_errors / n_trials if n_trials > 0 else 0
        pe_rate = n_pe / n_trials if n_trials > 0 else 0
        npe_rate = n_npe / n_trials if n_trials > 0 else 0

        # Proportions (as proportion of errors)
        pe_proportion = n_pe / n_errors if n_errors > 0 else 0
        npe_proportion = n_npe / n_errors if n_errors > 0 else 0

        # PE/NPE ratio
        pe_npe_ratio = n_pe / n_npe if n_npe > 0 else np.nan

        results.append({
            'participant_id': pid,
            'n_trials': n_trials,
            'n_correct': n_correct,
            'n_errors': n_errors,
            'n_pe': n_pe,
            'n_npe': n_npe,
            'error_rate': error_rate,
            'pe_rate': pe_rate,
            'npe_rate': npe_rate,
            'pe_proportion': pe_proportion,
            'npe_proportion': npe_proportion,
            'pe_npe_ratio': pe_npe_ratio,
        })

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  N = {len(results_df)}")
        print(f"\n  Error Type Descriptives:")
        print(f"    Total error rate: {results_df['error_rate'].mean()*100:.1f}% (SD={results_df['error_rate'].std()*100:.1f})")
        print(f"    PE rate: {results_df['pe_rate'].mean()*100:.1f}% (SD={results_df['pe_rate'].std()*100:.1f})")
        print(f"    NPE rate: {results_df['npe_rate'].mean()*100:.1f}% (SD={results_df['npe_rate'].std()*100:.1f})")

        print(f"\n  Error Composition (of total errors):")
        print(f"    PE proportion: {results_df['pe_proportion'].mean()*100:.1f}%")
        print(f"    NPE proportion: {results_df['npe_proportion'].mean()*100:.1f}%")

        # Test if PE and NPE proportions differ
        t_stat, p_val = stats.ttest_rel(
            results_df['pe_proportion'].dropna(),
            results_df['npe_proportion'].dropna()
        )
        print(f"\n  PE vs NPE proportion: t={t_stat:.2f}, p={p_val:.4f}")

    results_df.to_csv(OUTPUT_DIR / "error_type_metrics.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'error_type_metrics.csv'}")

    return results_df


@register_analysis(
    name="ucla_error_types",
    description="Test UCLA effects on PE vs NPE (DASS-controlled)"
)
def analyze_ucla_error_types(verbose: bool = True) -> pd.DataFrame:
    """Test whether UCLA differently affects PE vs NPE."""
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECTS ON PE vs NPE (DASS-CONTROLLED)")
        print("=" * 70)

    # Load error metrics
    metrics_file = OUTPUT_DIR / "error_type_metrics.csv"
    if not metrics_file.exists():
        analyze_error_metrics(verbose=False)
    metrics = pd.read_csv(metrics_file)

    # Load master data
    master = load_master_with_standardization()

    # Merge
    merged = master.merge(metrics, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  N = {len(merged)}")

    # Outcomes to test
    outcomes = [
        ('error_rate', 'Total Error Rate'),
        ('pe_rate', 'PE Rate (rigidity)'),
        ('npe_rate', 'NPE Rate (inattention)'),
        ('pe_proportion', 'PE Proportion'),
        ('pe_npe_ratio', 'PE/NPE Ratio'),
    ]

    results = []

    for outcome_col, outcome_name in outcomes:
        if outcome_col not in merged.columns:
            continue

        # Skip if all NaN
        if merged[outcome_col].isna().all():
            continue

        try:
            formula = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            result_row = {
                'outcome': outcome_name,
                'outcome_col': outcome_col,
                'n': int(model.nobs),
                'r_squared': model.rsquared,
            }

            # UCLA main effect
            if 'z_ucla' in model.params:
                result_row['beta_ucla'] = model.params['z_ucla']
                result_row['se_ucla'] = model.bse['z_ucla']
                result_row['p_ucla'] = model.pvalues['z_ucla']

            # UCLA × Gender interaction
            int_term = find_interaction_term(model.params.index)
            if int_term:
                result_row['beta_interaction'] = model.params[int_term]
                result_row['se_interaction'] = model.bse[int_term]
                result_row['p_interaction'] = model.pvalues[int_term]

            results.append(result_row)

            if verbose:
                sig_main = "*" if result_row.get('p_ucla', 1) < 0.05 else ""
                sig_int = "*" if result_row.get('p_interaction', 1) < 0.05 else ""

                print(f"\n  {outcome_name}:")
                print(f"    UCLA main: beta={result_row.get('beta_ucla', np.nan):.4f}, p={result_row.get('p_ucla', np.nan):.4f}{sig_main}")
                if 'beta_interaction' in result_row:
                    print(f"    UCLA × Gender: beta={result_row['beta_interaction']:.4f}, p={result_row['p_interaction']:.4f}{sig_int}")

        except Exception as e:
            if verbose:
                print(f"\n  {outcome_name}: Error - {e}")

    if len(results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    if 'p_ucla' in results_df.columns:
        results_df = apply_fdr_correction(results_df, p_col='p_ucla')
        if 'p_fdr' in results_df.columns:
            results_df = results_df.rename(columns={'p_fdr': 'p_ucla_fdr', 'significant_fdr': 'sig_ucla_fdr'})

    results_df.to_csv(OUTPUT_DIR / "ucla_error_types.csv", index=False, encoding='utf-8-sig')

    if verbose:
        # Summary
        sig_effects = results_df[results_df['p_ucla'] < 0.05] if 'p_ucla' in results_df.columns else pd.DataFrame()
        if len(sig_effects) > 0:
            print(f"\n  SIGNIFICANT UCLA MAIN EFFECTS (p < 0.05):")
            for _, row in sig_effects.iterrows():
                print(f"    {row['outcome']}: beta={row['beta_ucla']:.4f}, p={row['p_ucla']:.4f}")

        print(f"\n  Output: {OUTPUT_DIR / 'ucla_error_types.csv'}")

    return results_df


@register_analysis(
    name="gender_stratified",
    description="Gender-stratified error type analysis"
)
def analyze_gender_stratified(verbose: bool = True) -> pd.DataFrame:
    """Analyze error types separately for males and females."""
    if verbose:
        print("\n" + "=" * 70)
        print("GENDER-STRATIFIED ERROR TYPE ANALYSIS")
        print("=" * 70)

    # Load error metrics
    metrics_file = OUTPUT_DIR / "error_type_metrics.csv"
    if not metrics_file.exists():
        analyze_error_metrics(verbose=False)
    metrics = pd.read_csv(metrics_file)

    # Load master data
    master = load_master_with_standardization()

    # Merge
    merged = master.merge(metrics, on='participant_id', how='inner')

    results = []

    for gender_val, gender_name in [(0, 'Female'), (1, 'Male')]:
        subset = merged[merged['gender_male'] == gender_val]

        if len(subset) < 15:
            if verbose:
                print(f"\n  {gender_name}: Insufficient data (N={len(subset)})")
            continue

        if verbose:
            print(f"\n  {gender_name.upper()} (N={len(subset)})")
            print("  " + "-" * 50)

        for outcome_col in ['pe_rate', 'npe_rate', 'pe_proportion']:
            if outcome_col not in subset.columns:
                continue

            try:
                formula = f"{outcome_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    se = model.bse['z_ucla']
                    p = model.pvalues['z_ucla']

                    results.append({
                        'gender': gender_name,
                        'outcome': outcome_col,
                        'beta_ucla': beta,
                        'se_ucla': se,
                        'p_ucla': p,
                        'n': len(subset)
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    UCLA -> {outcome_col}: beta={beta:.4f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {outcome_col}: Error - {e}")

    if len(results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "gender_stratified_errors.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gender_stratified_errors.csv'}")

    return results_df


@register_analysis(
    name="error_sequences",
    description="Sequential patterns of PE vs NPE errors"
)
def analyze_error_sequences(verbose: bool = True) -> pd.DataFrame:
    """Analyze temporal patterns of error types."""
    if verbose:
        print("\n" + "=" * 70)
        print("ERROR SEQUENCE ANALYSIS")
        print("=" * 70)

    trials = load_wcst_trials()

    results = []

    for pid, pdata in trials.groupby('participant_id'):
        pdata = pdata.sort_values('trialindex')

        # Get error sequences
        errors = pdata[pdata['is_error']]

        if len(errors) < 2:
            continue

        # Count transitions
        pe_errors = errors['ispe'].values if 'ispe' in errors.columns else np.zeros(len(errors))
        npe_errors = errors['isnpe'].values if 'isnpe' in errors.columns else np.zeros(len(errors))

        # Transition counts
        pe_after_pe = 0
        pe_after_npe = 0
        npe_after_pe = 0
        npe_after_npe = 0

        for i in range(1, len(errors)):
            prev_pe = pe_errors[i-1]
            curr_pe = pe_errors[i]

            if prev_pe and curr_pe:
                pe_after_pe += 1
            elif prev_pe and not curr_pe:
                npe_after_pe += 1
            elif not prev_pe and curr_pe:
                pe_after_npe += 1
            else:
                npe_after_npe += 1

        # Transition probabilities
        total_after_pe = pe_after_pe + npe_after_pe
        total_after_npe = pe_after_npe + npe_after_npe

        results.append({
            'participant_id': pid,
            'n_error_transitions': len(errors) - 1,
            'pe_after_pe': pe_after_pe,
            'npe_after_pe': npe_after_pe,
            'pe_after_npe': pe_after_npe,
            'npe_after_npe': npe_after_npe,
            'p_pe_after_pe': pe_after_pe / total_after_pe if total_after_pe > 0 else np.nan,
            'p_pe_after_npe': pe_after_npe / total_after_npe if total_after_npe > 0 else np.nan,
            'pe_persistence': pe_after_pe / total_after_pe if total_after_pe > 0 else np.nan,
        })

    if len(results) == 0:
        if verbose:
            print("  No valid error sequences")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  N = {len(results_df)}")
        print(f"\n  Error Persistence (P(PE|previous PE)):")
        print(f"    Mean: {results_df['pe_persistence'].mean():.3f} (SD={results_df['pe_persistence'].std():.3f})")

        # Test if PE persists more than chance
        pe_persistence = results_df['pe_persistence'].dropna()
        if len(pe_persistence) > 10:
            t_stat, p_val = stats.ttest_1samp(pe_persistence, 0.5)
            sig = "*" if p_val < 0.05 else ""
            print(f"    vs 0.5 (chance): t={t_stat:.2f}, p={p_val:.4f}{sig}")

    results_df.to_csv(OUTPUT_DIR / "error_sequences.csv", index=False, encoding='utf-8-sig')

    # Test UCLA effect on PE persistence
    if verbose:
        print("\n  Testing UCLA effect on PE persistence:")

    master = load_master_with_standardization()
    merged = master.merge(results_df, on='participant_id', how='inner')

    if len(merged) >= 30:
        try:
            formula = "pe_persistence ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            if 'z_ucla' in model.params and verbose:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']
                sig = "*" if p < 0.05 else ""
                print(f"    UCLA -> PE persistence: beta={beta:.4f}, p={p:.4f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'error_sequences.csv'}")

    return results_df


@register_analysis(
    name="summary",
    description="Summary report"
)
def analyze_summary(verbose: bool = True) -> Dict:
    """Generate summary report."""
    if verbose:
        print("\n" + "=" * 70)
        print("WCST ERROR DECOMPOSITION SUMMARY")
        print("=" * 70)

    summary = {}

    # Load results
    metrics_file = OUTPUT_DIR / "error_type_metrics.csv"
    ucla_file = OUTPUT_DIR / "ucla_error_types.csv"
    gender_file = OUTPUT_DIR / "gender_stratified_errors.csv"

    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)
        summary['n_participants'] = len(metrics)
        summary['mean_pe_rate'] = metrics['pe_rate'].mean()
        summary['mean_npe_rate'] = metrics['npe_rate'].mean()
        summary['mean_pe_proportion'] = metrics['pe_proportion'].mean()

    if ucla_file.exists():
        ucla = pd.read_csv(ucla_file)
        sig_main = ucla[ucla['p_ucla'] < 0.05] if 'p_ucla' in ucla.columns else pd.DataFrame()
        sig_int = ucla[ucla.get('p_interaction', pd.Series([1]*len(ucla))) < 0.05]

        summary['significant_main_effects'] = sig_main['outcome'].tolist() if len(sig_main) > 0 else []
        summary['significant_interactions'] = sig_int['outcome'].tolist() if len(sig_int) > 0 else []

    if gender_file.exists():
        gender = pd.read_csv(gender_file)
        sig_male = gender[(gender['gender'] == 'Male') & (gender['p_ucla'] < 0.05)]
        sig_female = gender[(gender['gender'] == 'Female') & (gender['p_ucla'] < 0.05)]

        summary['male_significant'] = sig_male['outcome'].tolist() if len(sig_male) > 0 else []
        summary['female_significant'] = sig_female['outcome'].tolist() if len(sig_female) > 0 else []

    if verbose:
        print(f"\n  ERROR COMPOSITION:")
        if 'mean_pe_rate' in summary:
            print(f"    Mean PE rate: {summary['mean_pe_rate']*100:.1f}%")
            print(f"    Mean NPE rate: {summary['mean_npe_rate']*100:.1f}%")
            print(f"    PE as % of errors: {summary['mean_pe_proportion']*100:.1f}%")

        print(f"\n  UCLA EFFECTS:")
        if summary.get('significant_main_effects'):
            print(f"    Significant main effects: {summary['significant_main_effects']}")
        else:
            print(f"    No significant main effects")

        if summary.get('significant_interactions'):
            print(f"    Significant interactions: {summary['significant_interactions']}")

        print(f"\n  GENDER-SPECIFIC:")
        print(f"    Male significant: {summary.get('male_significant', [])}")
        print(f"    Female significant: {summary.get('female_significant', [])}")

        print(f"\n  INTERPRETATION:")
        if summary.get('significant_main_effects'):
            if 'PE Rate (rigidity)' in summary['significant_main_effects']:
                print("    UCLA affects perseverative errors -> Loneliness impacts cognitive flexibility")
            if 'NPE Rate (inattention)' in summary['significant_main_effects']:
                print("    UCLA affects non-perseverative errors -> Loneliness impacts attention/focus")
        else:
            print("    No significant UCLA effects on specific error types (consistent with null main effects)")

    # Save summary
    import json
    with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'summary.json'}")

    return summary


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict:
    """Run WCST error decomposition analyses."""
    if verbose:
        print("=" * 70)
        print("WCST ERROR TYPE DECOMPOSITION SUITE")
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
        analysis_order = [
            'error_metrics',
            'ucla_error_types',
            'gender_stratified',
            'error_sequences',
            'summary',
        ]

        for name in analysis_order:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("WCST ERROR DECOMPOSITION SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable WCST Error Decomposition Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WCST Error Decomposition Suite")
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
