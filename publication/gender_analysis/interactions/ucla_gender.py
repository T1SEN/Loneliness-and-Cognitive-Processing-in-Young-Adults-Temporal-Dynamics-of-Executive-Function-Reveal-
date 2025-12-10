"""
UCLA x Gender Interaction Analysis
==================================

Systematic testing of UCLA x Gender interactions across all EF outcomes.

Uses FDR correction for multiple comparisons.

Key Finding: WCST PE shows significant UCLA x Gender interaction (p=0.025).

Source: analysis/gold_standard/pipeline.py, analysis/advanced/male_vulnerability_suite.py
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from publication.preprocessing import (
    apply_fdr_correction,
    find_interaction_term,
)

from .._utils import (
    load_gender_data,
    run_all_gender_interactions,
    test_gender_interaction,
)

from .._constants import (
    EF_OUTCOMES,
    PRIMARY_OUTCOMES,
    MIN_SAMPLE_INTERACTION,
    INTERACTION_FORMULA,
)

from . import OUTPUT_DIR


# =============================================================================
# ANALYSIS: ALL INTERACTIONS
# =============================================================================

def analyze_all_interactions(verbose: bool = True) -> pd.DataFrame:
    """
    Test UCLA x Gender interactions across all EF outcomes.

    Uses DASS-controlled formula with FDR correction.

    Returns
    -------
    pd.DataFrame
        Interaction test results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA x GENDER INTERACTION ANALYSIS")
        print("=" * 70)

    df = load_gender_data(verbose=verbose)

    if verbose:
        print(f"  Testing {len(EF_OUTCOMES)} outcomes...")
        print("  " + "-" * 60)

    results_df = run_all_gender_interactions(df, EF_OUTCOMES, apply_fdr=True, verbose=verbose)

    if results_df.empty:
        return pd.DataFrame()

    # Summarize
    n_sig_raw = (results_df['p_interaction'] < 0.05).sum()
    n_sig_fdr = (results_df['p_interaction_fdr'] < 0.05).sum() if 'p_interaction_fdr' in results_df.columns else 0

    if verbose:
        print(f"\n  SUMMARY:")
        print("  " + "-" * 60)
        print(f"  Outcomes tested: {len(results_df)}")
        print(f"  Significant (p < 0.05): {n_sig_raw}")
        print(f"  Significant (FDR < 0.05): {n_sig_fdr}")

        # Show significant interactions
        sig = results_df[results_df['p_interaction'] < 0.05].sort_values('p_interaction')
        if len(sig) > 0:
            print(f"\n  SIGNIFICANT INTERACTIONS:")
            for _, row in sig.iterrows():
                direction = "+" if row['beta_interaction'] > 0 else "-"
                print(f"    {row['outcome']}: beta={row['beta_interaction']:.4f} ({direction}), p={row['p_interaction']:.4f}")

    output_file = OUTPUT_DIR / "ucla_gender_interactions.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return results_df


# =============================================================================
# ANALYSIS: PRIMARY OUTCOMES
# =============================================================================

def analyze_primary_outcomes(verbose: bool = True) -> pd.DataFrame:
    """
    Focus on primary outcomes (PE, Stroop interference, PRP bottleneck).

    Provides detailed results for the three main EF metrics.

    Returns
    -------
    pd.DataFrame
        Detailed results for primary outcomes
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PRIMARY OUTCOME INTERACTIONS")
        print("=" * 70)

    df = load_gender_data(verbose=verbose)

    all_results = []

    for outcome in PRIMARY_OUTCOMES:
        if outcome not in df.columns:
            continue

        if verbose:
            print(f"\n  {outcome.upper()}")
            print("  " + "-" * 50)

        result = test_gender_interaction(df, outcome, verbose=False)

        if result:
            # Add gender-stratified betas
            for gender, label in [(0, 'female'), (1, 'male')]:
                subset = df[df['gender_male'] == gender].dropna(
                    subset=['z_ucla', outcome, 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
                )

                if len(subset) >= 15:
                    try:
                        formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                        model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                        result[f'beta_{label}'] = model.params.get('z_ucla', np.nan)
                        result[f'p_{label}'] = model.pvalues.get('z_ucla', np.nan)
                        result[f'n_{label}'] = len(subset)
                    except:
                        pass

            all_results.append(result)

            if verbose:
                print(f"    Interaction: beta={result['beta_interaction']:.4f}, p={result['p_interaction']:.4f}")

                if 'beta_male' in result:
                    m_sig = "*" if result.get('p_male', 1) < 0.05 else ""
                    f_sig = "*" if result.get('p_female', 1) < 0.05 else ""
                    print(f"    Male: beta={result['beta_male']:.4f}, p={result.get('p_male', np.nan):.4f}{m_sig}")
                    print(f"    Female: beta={result['beta_female']:.4f}, p={result.get('p_female', np.nan):.4f}{f_sig}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    output_file = OUTPUT_DIR / "primary_outcome_interactions.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return results_df


# =============================================================================
# RUNNER
# =============================================================================

def run(verbose: bool = True) -> Dict:
    """
    Run all UCLA x Gender interaction analyses.

    Returns
    -------
    dict
        Analysis results
    """
    results = {}

    try:
        results['all_interactions'] = analyze_all_interactions(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ERROR in all_interactions: {e}")

    try:
        results['primary_outcomes'] = analyze_primary_outcomes(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ERROR in primary_outcomes: {e}")

    return results


if __name__ == "__main__":
    run(verbose=True)
