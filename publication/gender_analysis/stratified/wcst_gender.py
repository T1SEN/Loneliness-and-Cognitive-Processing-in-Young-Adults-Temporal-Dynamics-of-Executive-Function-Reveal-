"""
WCST Gender-Stratified Analysis
===============================

Analyze UCLA effects on WCST error types separately by gender.

WCST Error Types:
- PE (Perseverative Errors): Rigid, failure to shift
- NPE (Non-Perseverative Errors): Random, attentional lapses

Key Finding: Males show UCLA x Gender interaction for PE rate (p=0.025).

Source: analysis/advanced/wcst_error_decomposition_suite.py
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from publication.preprocessing import (
    load_master_dataset,
    standardize_predictors,
    prepare_gender_variable,
    ANALYSIS_OUTPUT_DIR,
)

from .._utils import load_gender_data
from .._constants import MIN_SAMPLE_STRATIFIED

from . import OUTPUT_DIR


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_wcst_error_metrics() -> pd.DataFrame:
    """
    Load WCST error type metrics from pre-computed file.

    Returns
    -------
    pd.DataFrame
        Error metrics (pe_rate, npe_rate, etc.) per participant
    """
    possible_paths = [
        ANALYSIS_OUTPUT_DIR / "wcst_error_decomposition" / "error_type_metrics.csv",
        ANALYSIS_OUTPUT_DIR / "wcst_error_decomposition_suite" / "error_type_metrics.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return pd.DataFrame()


# =============================================================================
# ANALYSIS: GENDER-STRATIFIED WCST
# =============================================================================

def analyze_gender_stratified_wcst(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze error types (PE, NPE) separately for males and females.

    This is the key analysis - males show UCLA effect on PE rate.

    Returns
    -------
    pd.DataFrame
        Results with beta, se, p for each gender x error type
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENDER-STRATIFIED WCST ERROR TYPE ANALYSIS")
        print("=" * 70)

    # Load master data
    master = load_gender_data(verbose=verbose)

    # Load error metrics (if pre-computed)
    metrics = load_wcst_error_metrics()

    if not metrics.empty:
        merged = master.merge(metrics, on='participant_id', how='inner')
    else:
        # Use master data directly if metrics not available
        merged = master

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    all_results = []
    wcst_outcomes = ['pe_rate', 'npe_rate', 'total_error_rate', 'pe_proportion',
                     'categories_completed', 'conceptual_level_responses']

    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = merged[merged['gender_male'] == gender]

        if len(subset) < MIN_SAMPLE_STRATIFIED:
            if verbose:
                print(f"\n  {label.upper()}: Insufficient data (N={len(subset)})")
            continue

        if verbose:
            print(f"\n  {label.upper()} (N={len(subset)})")
            print("  " + "-" * 50)

        for outcome in wcst_outcomes:
            if outcome not in subset.columns:
                continue

            # Drop missing values for all covariates to ensure accurate N
            valid = subset.dropna(subset=['z_ucla', outcome, 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])

            if len(valid) < MIN_SAMPLE_STRATIFIED:
                if verbose:
                    print(f"    {outcome}: Insufficient valid data (N={len(valid)})")
                continue

            try:
                formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=valid).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    se = model.bse['z_ucla']
                    p = model.pvalues['z_ucla']

                    all_results.append({
                        'gender': label,
                        'outcome': outcome,
                        'beta_ucla': beta,
                        'se_ucla': se,
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(valid)  # Use valid N, not subset N
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    UCLA -> {outcome}: beta={beta:.4f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {outcome}: Error - {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Highlight key finding
    if verbose:
        print("\n  KEY FINDING:")
        print("  " + "-" * 50)

        pe_male = results_df[(results_df['gender'] == 'male') & (results_df['outcome'] == 'pe_rate')]
        pe_female = results_df[(results_df['gender'] == 'female') & (results_df['outcome'] == 'pe_rate')]

        if len(pe_male) > 0:
            beta = pe_male['beta_ucla'].values[0]
            p = pe_male['p_ucla'].values[0]
            sig = "*" if p < 0.05 else ""
            print(f"    Male PE rate: beta={beta:.4f}, p={p:.4f}{sig}")

        if len(pe_female) > 0:
            beta = pe_female['beta_ucla'].values[0]
            p = pe_female['p_ucla'].values[0]
            sig = "*" if p < 0.05 else ""
            print(f"    Female PE rate: beta={beta:.4f}, p={p:.4f}{sig}")

    output_file = OUTPUT_DIR / "wcst_gender_stratified.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return results_df


# =============================================================================
# RUNNER
# =============================================================================

def run(verbose: bool = True) -> Dict:
    """
    Run WCST gender-stratified analysis.

    Returns
    -------
    dict
        Analysis results
    """
    results = {}

    try:
        results['gender_stratified'] = analyze_gender_stratified_wcst(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    run(verbose=True)
