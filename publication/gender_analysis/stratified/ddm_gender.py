"""
DDM Gender-Stratified Analysis
==============================

Analyze UCLA effects on Drift-Diffusion Model parameters separately by gender.

DDM Parameters:
- v: Drift rate (evidence accumulation speed)
- a: Boundary separation (response caution)
- t: Non-decision time (encoding + motor)

Key Finding: Drift rate shows UCLA effect in FEMALES (p=0.021), not males.

Source: analysis/advanced/ddm_suite.py
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
from .._constants import MIN_SAMPLE_STRATIFIED, DDM_PARAMS

from . import OUTPUT_DIR


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_ddm_parameters() -> pd.DataFrame:
    """
    Load DDM parameters from pre-computed file.

    Returns
    -------
    pd.DataFrame
        DDM parameters (v, a, t) per participant
    """
    # Check multiple possible locations
    possible_paths = [
        ANALYSIS_OUTPUT_DIR / "ddm" / "ez_ddm_parameters.csv",
        ANALYSIS_OUTPUT_DIR / "ddm_suite" / "ez_ddm_parameters.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return pd.DataFrame()


# =============================================================================
# ANALYSIS: GENDER-STRATIFIED DDM
# =============================================================================

def analyze_gender_stratified_ddm(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze UCLA effects on DDM parameters separately by gender.

    Tests v, a, t parameters for UCLA association within each gender,
    controlling for DASS-21 subscales.

    Returns
    -------
    pd.DataFrame
        Results with beta, se, p for each gender x parameter
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENDER-STRATIFIED DDM ANALYSIS")
        print("=" * 70)

    # Load DDM parameters
    params = load_ddm_parameters()

    if params.empty:
        if verbose:
            print("  DDM parameters not available")
            print("  Run 'python -m analysis.advanced.ddm_suite' first")
        return pd.DataFrame()

    # Load master data
    master = load_gender_data(verbose=verbose)

    # Merge
    merged = master.merge(params, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    all_results = []

    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = merged[merged['gender_male'] == gender]

        if len(subset) < MIN_SAMPLE_STRATIFIED:
            if verbose:
                print(f"  {label.upper()}: Insufficient data (N={len(subset)})")
            continue

        if verbose:
            print(f"\n  {label.upper()} (N={len(subset)})")
            print("  " + "-" * 50)

        for param in ['v', 'a', 't']:
            if param not in subset.columns:
                continue

            try:
                formula = f"{param} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    se = model.bse['z_ucla']
                    p = model.pvalues['z_ucla']

                    all_results.append({
                        'gender': label,
                        'parameter': param,
                        'beta_ucla': beta,
                        'se_ucla': se,
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(subset)
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        param_name = {'v': 'Drift rate', 'a': 'Boundary', 't': 'Non-decision'}[param]
                        print(f"    UCLA -> {param_name} ({param}): beta={beta:.4f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {param}: Error - {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Gender comparison
    if verbose:
        print("\n  GENDER COMPARISON:")
        print("  " + "-" * 50)

        for param in ['v', 'a', 't']:
            male_row = results_df[(results_df['gender'] == 'male') & (results_df['parameter'] == param)]
            female_row = results_df[(results_df['gender'] == 'female') & (results_df['parameter'] == param)]

            if len(male_row) > 0 and len(female_row) > 0:
                m_beta = male_row['beta_ucla'].values[0]
                f_beta = female_row['beta_ucla'].values[0]
                m_p = male_row['p_ucla'].values[0]
                f_p = female_row['p_ucla'].values[0]

                m_sig = "*" if m_p < 0.05 else ""
                f_sig = "*" if f_p < 0.05 else ""

                param_name = {'v': 'Drift', 'a': 'Boundary', 't': 'Non-dec'}[param]
                print(f"    {param_name}: Male beta={m_beta:.4f}{m_sig} vs Female beta={f_beta:.4f}{f_sig}")

    output_file = OUTPUT_DIR / "ddm_gender_stratified.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return results_df


# =============================================================================
# RUNNER
# =============================================================================

def run(verbose: bool = True) -> Dict:
    """
    Run DDM gender-stratified analysis.

    Returns
    -------
    dict
        Analysis results
    """
    results = {}

    try:
        results['gender_stratified'] = analyze_gender_stratified_ddm(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    run(verbose=True)
