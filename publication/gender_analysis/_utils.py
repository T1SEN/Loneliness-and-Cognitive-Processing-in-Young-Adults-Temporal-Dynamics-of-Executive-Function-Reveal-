"""
Gender Analysis Utilities
=========================

Shared functions for gender-specific analyses.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from publication.preprocessing import (
    load_master_dataset,
    standardize_predictors,
    prepare_gender_variable,
    apply_fdr_correction,
    find_interaction_term,
)

from ._constants import (
    MIN_SAMPLE_STRATIFIED,
    MIN_SAMPLE_INTERACTION,
    DASS_COVARIATES,
    STRATIFIED_FORMULA,
    INTERACTION_FORMULA,
)

# =============================================================================
# OUTPUT PATHS (constants.py에서 중앙 관리)
# =============================================================================

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "gender_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_gender_data(verbose: bool = True) -> pd.DataFrame:
    """
    Load master dataset with gender variable prepared and predictors standardized.

    Returns
    -------
    pd.DataFrame
        Master dataset with:
        - 'gender': lowercase 'male' or 'female'
        - 'gender_male': binary 0/1
        - Standardized predictors: z_ucla, z_dass_dep, z_dass_anx, z_dass_str, z_age
    """
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Prepare gender variable
    master = prepare_gender_variable(master)

    # Standardize predictors
    master = standardize_predictors(master)

    if verbose:
        n_male = (master['gender_male'] == 1).sum()
        n_female = (master['gender_male'] == 0).sum()
        print(f"  Data loaded: N={len(master)} (Male={n_male}, Female={n_female})")

    return master


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def run_gender_stratified_regression(
    df: pd.DataFrame,
    outcome: str,
    formula_template: str = STRATIFIED_FORMULA,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run UCLA regression separately for males and females.

    Parameters
    ----------
    df : pd.DataFrame
        Data with gender_male and predictor columns
    outcome : str
        Dependent variable column name
    formula_template : str
        Formula template with {outcome} placeholder
    verbose : bool
        Print results

    Returns
    -------
    list of dict
        Results for each gender with beta, se, p, r_squared, n
    """
    results = []

    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = df[df['gender_male'] == gender].copy()

        if len(subset) < MIN_SAMPLE_STRATIFIED:
            if verbose:
                print(f"    {label}: Insufficient data (N={len(subset)})")
            continue

        # Drop rows with missing values
        required_cols = ['z_ucla', outcome] + DASS_COVARIATES
        valid = subset.dropna(subset=[c for c in required_cols if c in subset.columns])

        if len(valid) < MIN_SAMPLE_STRATIFIED:
            continue

        try:
            formula = formula_template.format(outcome=outcome)
            model = smf.ols(formula, data=valid).fit(cov_type='HC3')

            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                se = model.bse['z_ucla']
                p = model.pvalues['z_ucla']

                results.append({
                    'gender': label,
                    'outcome': outcome,
                    'beta_ucla': beta,
                    'se_ucla': se,
                    'p_ucla': p,
                    'r_squared': model.rsquared,
                    'n': len(valid)
                })

                if verbose and p < 0.10:
                    sig = "*" if p < 0.05 else "+"
                    print(f"    {label}: {outcome} beta={beta:.4f}, p={p:.4f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    {label}: Error - {e}")
            continue

    return results


def test_gender_interaction(
    df: pd.DataFrame,
    outcome: str,
    formula_template: str = INTERACTION_FORMULA,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Test UCLA x Gender interaction for a single outcome.

    Parameters
    ----------
    df : pd.DataFrame
        Data with gender_male and predictor columns
    outcome : str
        Dependent variable column name
    formula_template : str
        Formula template with {outcome} placeholder
    verbose : bool
        Print results

    Returns
    -------
    dict or None
        Interaction test results with beta, se, p values
    """
    if outcome not in df.columns:
        return None

    required_cols = ['z_ucla', outcome, 'gender_male'] + DASS_COVARIATES
    valid = df.dropna(subset=[c for c in required_cols if c in df.columns])

    if len(valid) < MIN_SAMPLE_INTERACTION:
        if verbose:
            print(f"    {outcome}: Insufficient data (N={len(valid)})")
        return None

    try:
        formula = formula_template.format(outcome=outcome)
        model = smf.ols(formula, data=valid).fit(cov_type='HC3')

        # Main effect
        beta_main = model.params.get('z_ucla', np.nan)
        p_main = model.pvalues.get('z_ucla', np.nan)

        # Interaction term
        int_term = find_interaction_term(model.params.index)
        beta_int = model.params.get(int_term, np.nan) if int_term else np.nan
        se_int = model.bse.get(int_term, np.nan) if int_term else np.nan
        p_int = model.pvalues.get(int_term, np.nan) if int_term else np.nan

        result = {
            'outcome': outcome,
            'beta_main': beta_main,
            'p_main': p_main,
            'beta_interaction': beta_int,
            'se_interaction': se_int,
            'p_interaction': p_int,
            'r_squared': model.rsquared,
            'n': len(valid)
        }

        if verbose and p_int < 0.10:
            sig = "*" if p_int < 0.05 else "+"
            print(f"    {outcome}: interaction beta={beta_int:.4f}, p={p_int:.4f}{sig}")

        return result

    except Exception as e:
        if verbose:
            print(f"    {outcome}: Error - {e}")
        return None


def run_all_gender_interactions(
    df: pd.DataFrame,
    outcomes: List[str],
    apply_fdr: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test UCLA x Gender interactions across multiple outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Data with required columns
    outcomes : list of str
        Outcome variables to test
    apply_fdr : bool
        Apply FDR correction
    verbose : bool
        Print results

    Returns
    -------
    pd.DataFrame
        Results with all interaction tests
    """
    results = []

    for outcome in outcomes:
        result = test_gender_interaction(df, outcome, verbose=verbose)
        if result:
            results.append(result)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if apply_fdr and len(results_df) > 1:
        results_df = apply_fdr_correction(results_df, p_col='p_interaction')
        results_df = results_df.rename(columns={'p_fdr': 'p_interaction_fdr'})

    return results_df


# =============================================================================
# EFFECT SIZE FUNCTIONS
# =============================================================================

def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Compute Cohen's d effect size.

    Parameters
    ----------
    group1, group2 : pd.Series
        Two groups to compare

    Returns
    -------
    float
        Cohen's d (positive if group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (group1.mean() - group2.mean()) / pooled_std


def partial_eta_squared(model) -> float:
    """
    Compute partial eta-squared from regression model.

    Parameters
    ----------
    model : statsmodels RegressionResults
        Fitted OLS model

    Returns
    -------
    float
        Partial eta-squared
    """
    ss_effect = model.ess
    ss_error = model.ssr
    return ss_effect / (ss_effect + ss_error)


def compute_gender_effect_sizes(
    df: pd.DataFrame,
    outcomes: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute effect sizes for gender differences across outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Data with gender_male column
    outcomes : list of str
        Variables to compare
    verbose : bool
        Print results

    Returns
    -------
    pd.DataFrame
        Effect size results
    """
    results = []

    for outcome in outcomes:
        if outcome not in df.columns:
            continue

        male = df[df['gender_male'] == 1][outcome].dropna()
        female = df[df['gender_male'] == 0][outcome].dropna()

        if len(male) < 10 or len(female) < 10:
            continue

        d = cohens_d(male, female)

        # Independent samples t-test (Welch's, unequal variance)
        t, p = stats.ttest_ind(male, female, equal_var=False)

        results.append({
            'outcome': outcome,
            'mean_male': male.mean(),
            'mean_female': female.mean(),
            'sd_male': male.std(),
            'sd_female': female.std(),
            'cohens_d': d,
            't_statistic': t,
            'p_value': p,
            'n_male': len(male),
            'n_female': len(female)
        })

        if verbose and p < 0.10:
            sig = "*" if p < 0.05 else "+"
            direction = "M>F" if d > 0 else "F>M"
            print(f"    {outcome}: d={d:.3f} ({direction}), p={p:.4f}{sig}")

    return pd.DataFrame(results)


# =============================================================================
# CORRELATION FUNCTIONS
# =============================================================================

def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> Tuple[float, float]:
    """
    Fisher z-test for comparing two correlations.

    Parameters
    ----------
    r1 : float
        Correlation 1
    n1 : int
        Sample size 1
    r2 : float
        Correlation 2
    n2 : int
        Sample size 2

    Returns
    -------
    tuple
        (z-statistic, p-value)
    """
    # Fisher z transformation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Standard error of difference
    se_diff = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))

    # Z-statistic
    z = (z1 - z2) / se_diff

    # Two-tailed p-value
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p


def compare_correlations_by_gender(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare correlation between two variables across genders.

    Parameters
    ----------
    df : pd.DataFrame
        Data with gender_male column
    var1, var2 : str
        Variables to correlate
    verbose : bool
        Print results

    Returns
    -------
    dict
        Correlation comparison results
    """
    results = {}

    for gender, label in [(0, 'female'), (1, 'male')]:
        subset = df[df['gender_male'] == gender][[var1, var2]].dropna()

        if len(subset) < MIN_SAMPLE_STRATIFIED:
            continue

        r, p = stats.pearsonr(subset[var1], subset[var2])
        results[f'r_{label}'] = r
        results[f'p_{label}'] = p
        results[f'n_{label}'] = len(subset)

    # Fisher z-test for difference
    if 'r_male' in results and 'r_female' in results:
        z, p_diff = fisher_z_test(
            results['r_male'], results['n_male'],
            results['r_female'], results['n_female']
        )
        results['z_diff'] = z
        results['p_diff'] = p_diff

        if verbose:
            sig = "*" if p_diff < 0.05 else ""
            print(f"    {var1} x {var2}:")
            print(f"      Male: r={results['r_male']:.3f}, Female: r={results['r_female']:.3f}")
            print(f"      Difference: z={z:.3f}, p={p_diff:.4f}{sig}")

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Paths
    'RESULTS_DIR',
    'OUTPUT_DIR',
    # Data loading
    'load_gender_data',
    # Regression
    'run_gender_stratified_regression',
    'test_gender_interaction',
    'run_all_gender_interactions',
    # Effect sizes
    'cohens_d',
    'partial_eta_squared',
    'compute_gender_effect_sizes',
    # Correlations
    'fisher_z_test',
    'compare_correlations_by_gender',
]
