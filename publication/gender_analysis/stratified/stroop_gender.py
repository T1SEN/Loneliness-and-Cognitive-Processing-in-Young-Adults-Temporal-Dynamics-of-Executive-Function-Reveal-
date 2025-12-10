"""
Stroop Gender-Stratified Analysis
=================================

Analyze UCLA effects on Stroop decomposition indices separately by gender.

Stroop Components:
- facilitation: Congruent advantage (faster than neutral)
- interference: Incongruent cost (slower than neutral)
- total_stroop: Overall Stroop effect (incongruent - congruent)

Source: analysis/advanced/stroop_decomposition_suite.py
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

def load_stroop_decomposition() -> pd.DataFrame:
    """
    Load Stroop decomposition indices from pre-computed file.

    Returns
    -------
    pd.DataFrame
        Stroop indices (facilitation, interference, total) per participant
    """
    possible_paths = [
        ANALYSIS_OUTPUT_DIR / "stroop_decomposition" / "decomposition_indices.csv",
        ANALYSIS_OUTPUT_DIR / "stroop_decomposition_suite" / "decomposition_indices.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return pd.DataFrame()


# =============================================================================
# ANALYSIS: GENDER-STRATIFIED STROOP
# =============================================================================

def analyze_gender_stratified_stroop(verbose: bool = True) -> pd.DataFrame:
    """
    Examine facilitation/interference separately for males and females.

    This helps understand the UCLA x Gender interaction mechanism.

    Returns
    -------
    pd.DataFrame
        Results with beta, se, p for each gender x outcome
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENDER-STRATIFIED STROOP DECOMPOSITION")
        print("=" * 70)

    # Load master data
    master = load_gender_data(verbose=verbose)

    # Load decomposition
    decomp = load_stroop_decomposition()

    if decomp.empty:
        if verbose:
            print("  Stroop decomposition indices not available")
            print("  Run 'python -m analysis.advanced.stroop_decomposition_suite' first")
        return pd.DataFrame()

    # Merge
    merged = master.merge(decomp, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    all_results = []
    stroop_outcomes = ['facilitation', 'interference', 'total_stroop', 'stroop_interference']

    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = merged[merged['gender_male'] == gender]

        if len(subset) < MIN_SAMPLE_STRATIFIED:
            if verbose:
                print(f"\n  {label.upper()}: Insufficient data (N={len(subset)})")
            continue

        if verbose:
            print(f"\n  {label.upper()} (N={len(subset)})")
            print("  " + "-" * 50)

        for outcome in stroop_outcomes:
            if outcome not in subset.columns:
                continue

            try:
                formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

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
                        'n': len(subset)
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

    output_file = OUTPUT_DIR / "stroop_gender_stratified.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return results_df


# =============================================================================
# RUNNER
# =============================================================================

def run(verbose: bool = True) -> Dict:
    """
    Run Stroop gender-stratified analysis.

    Returns
    -------
    dict
        Analysis results
    """
    results = {}

    try:
        results['gender_stratified'] = analyze_gender_stratified_stroop(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    run(verbose=True)
