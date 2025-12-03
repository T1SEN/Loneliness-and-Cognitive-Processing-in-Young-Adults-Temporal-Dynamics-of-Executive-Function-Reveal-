"""
Modeling Utilities
==================

Shared functions for regression analysis with DASS-21 control.

This module provides:
- Standard formula templates
- Model fitting with robust SE
- Result extraction and formatting
- DASS control verification

Usage:
    from analysis.utils.modeling import (
        DASS_CONTROL_FORMULA,
        fit_dass_controlled_model,
        extract_coefficients,
        verify_dass_control
    )
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.preprocessing import StandardScaler


# =============================================================================
# FORMULA TEMPLATES
# =============================================================================

# Standard DASS-controlled formula (MANDATORY for confirmatory analyses)
DASS_CONTROL_FORMULA = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

# Base covariates only
COVARIATES_ONLY_FORMULA = "{outcome} ~ z_age + C(gender_male)"

# DASS without UCLA
DASS_WITHOUT_UCLA_FORMULA = "{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str"

# UCLA without interaction
UCLA_MAIN_ONLY_FORMULA = "{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla"

# Required predictors for Gold Standard
REQUIRED_PREDICTORS = ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male']


# =============================================================================
# DATA PREPARATION
# =============================================================================

def standardize_predictors(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Standardize predictor columns (z-score).

    Args:
        df: DataFrame with raw columns
        columns: Columns to standardize (default: ucla_total, dass_*, age)

    Returns:
        DataFrame with z_ prefixed standardized columns added
    """
    result = df.copy()

    if columns is None:
        columns = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']

    scaler = StandardScaler()

    col_mapping = {
        'ucla_total': 'z_ucla',
        'dass_depression': 'z_dass_dep',
        'dass_anxiety': 'z_dass_anx',
        'dass_stress': 'z_dass_str',
        'age': 'z_age'
    }

    for col in columns:
        if col in result.columns:
            z_col = col_mapping.get(col, f'z_{col}')
            result[z_col] = scaler.fit_transform(result[[col]])

    return result


def prepare_gender_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize gender coding.

    Args:
        df: DataFrame with 'gender' or 'gender_normalized' column

    Returns:
        DataFrame with 'gender_male' binary column (1=male, 0=female)
    """
    result = df.copy()

    if 'gender_normalized' in result.columns:
        gender_col = 'gender_normalized'
    elif 'gender' in result.columns:
        gender_col = 'gender'
    else:
        raise ValueError("No gender column found")

    result['gender'] = result[gender_col].fillna('').astype(str).str.strip().str.lower()
    result['gender_male'] = (result['gender'] == 'male').astype(int)

    return result


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_dass_controlled_model(
    data: pd.DataFrame,
    outcome: str,
    robust: str = "HC3",
    min_n: int = 30
) -> Optional[RegressionResultsWrapper]:
    """
    Fit OLS model with full DASS control.

    Args:
        data: DataFrame with standardized predictors
        outcome: Column name of dependent variable
        robust: Robust SE type (HC0, HC1, HC2, HC3)
        min_n: Minimum sample size

    Returns:
        Fitted model or None if insufficient data
    """
    df = data.dropna(subset=[outcome] + REQUIRED_PREDICTORS).copy()

    if len(df) < min_n:
        return None

    formula = DASS_CONTROL_FORMULA.format(outcome=outcome)
    model = smf.ols(formula, data=df).fit(cov_type=robust)

    return model


def fit_hierarchical_models(
    data: pd.DataFrame,
    outcome: str,
    robust: str = "HC3"
) -> Dict[str, RegressionResultsWrapper]:
    """
    Fit hierarchical regression models.

    Model 0: Covariates only (age, gender)
    Model 1: + DASS subscales
    Model 2: + UCLA main effect
    Model 3: + UCLA × Gender interaction

    Args:
        data: DataFrame with standardized predictors
        outcome: Column name of dependent variable
        robust: Robust SE type

    Returns:
        Dict mapping model names to fitted models
    """
    formulas = {
        'model0': COVARIATES_ONLY_FORMULA,
        'model1': DASS_WITHOUT_UCLA_FORMULA,
        'model2': UCLA_MAIN_ONLY_FORMULA,
        'model3': DASS_CONTROL_FORMULA
    }

    models = {}
    df = data.dropna(subset=[outcome] + REQUIRED_PREDICTORS).copy()

    for name, formula_template in formulas.items():
        formula = formula_template.format(outcome=outcome)
        models[name] = smf.ols(formula, data=df).fit(cov_type=robust)

    return models


# =============================================================================
# RESULT EXTRACTION
# =============================================================================

def extract_coefficients(
    model: RegressionResultsWrapper,
    predictors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract coefficients, SE, t, and p-values from fitted model.

    Args:
        model: Fitted statsmodels model
        predictors: Specific predictors to extract (default: all)

    Returns:
        DataFrame with beta, se, t, p columns
    """
    if predictors is None:
        predictors = model.params.index.tolist()

    results = []
    for pred in predictors:
        if pred in model.params:
            results.append({
                'predictor': pred,
                'beta': model.params[pred],
                'se': model.bse[pred],
                't': model.tvalues[pred],
                'p': model.pvalues[pred]
            })

    return pd.DataFrame(results)


def extract_ucla_effects(model: RegressionResultsWrapper) -> Dict[str, Any]:
    """
    Extract UCLA-specific effects from model.

    Returns:
        Dict with ucla_beta, ucla_p, interaction_beta, interaction_p
    """
    result = {
        'ucla_beta': model.params.get('z_ucla', np.nan),
        'ucla_se': model.bse.get('z_ucla', np.nan),
        'ucla_p': model.pvalues.get('z_ucla', np.nan),
    }

    # Interaction term
    int_term = 'z_ucla:C(gender_male)[T.1]'
    if int_term in model.params:
        result['interaction_beta'] = model.params[int_term]
        result['interaction_se'] = model.bse[int_term]
        result['interaction_p'] = model.pvalues[int_term]
    else:
        result['interaction_beta'] = np.nan
        result['interaction_se'] = np.nan
        result['interaction_p'] = np.nan

    return result


def compute_delta_r2(models: Dict[str, RegressionResultsWrapper]) -> Dict[str, float]:
    """
    Compute ΔR² between hierarchical models.

    Args:
        models: Dict with model0, model1, model2, model3

    Returns:
        Dict with delta_r2_dass, delta_r2_ucla, delta_r2_interaction
    """
    return {
        'delta_r2_dass': models['model1'].rsquared - models['model0'].rsquared,
        'delta_r2_ucla': models['model2'].rsquared - models['model1'].rsquared,
        'delta_r2_interaction': models['model3'].rsquared - models['model2'].rsquared
    }


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_dass_control(formula: str) -> Tuple[bool, List[str]]:
    """
    Verify that a formula includes all required DASS controls.

    Args:
        formula: Model formula string

    Returns:
        (is_valid, missing_terms) tuple
    """
    required = ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
    missing = [term for term in required if term not in formula]

    # Check for gender interaction
    if 'z_ucla' in formula and 'gender_male' in formula:
        if '*' not in formula and ':' not in formula:
            missing.append('UCLA×Gender interaction')

    return len(missing) == 0, missing


def check_formula_compliance(formula: str) -> Dict[str, bool]:
    """
    Check formula compliance with Gold Standard requirements.

    Returns:
        Dict with boolean flags for each requirement
    """
    return {
        'has_dass_dep': 'z_dass_dep' in formula,
        'has_dass_anx': 'z_dass_anx' in formula,
        'has_dass_str': 'z_dass_str' in formula,
        'has_age': 'z_age' in formula,
        'has_gender': 'gender_male' in formula,
        'has_ucla': 'z_ucla' in formula,
        'has_interaction': ('*' in formula or 'z_ucla:' in formula) and 'gender_male' in formula
    }


# =============================================================================
# FORMATTING
# =============================================================================

def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for display."""
    if p < threshold:
        return f"< {threshold}"
    return f"{p:.3f}"


def format_coefficient(beta: float, se: float, p: float) -> str:
    """Format coefficient with SE and significance stars."""
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    return f"{beta:.3f} ({se:.3f}){stars}"


def results_to_apa_table(
    results: List[Dict[str, Any]],
    outcome_col: str = 'outcome'
) -> pd.DataFrame:
    """
    Format results as APA-style regression table.

    Args:
        results: List of result dicts from hierarchical_regression
        outcome_col: Column name for outcome labels

    Returns:
        Formatted DataFrame for publication
    """
    rows = []
    for r in results:
        rows.append({
            'Outcome': r.get(outcome_col, ''),
            'N': r.get('n', ''),
            'UCLA β': format_coefficient(r.get('ucla_beta', np.nan),
                                         r.get('ucla_se', np.nan),
                                         r.get('ucla_p', 1.0)),
            'UCLA×Gender β': format_coefficient(r.get('interaction_beta', np.nan),
                                                r.get('interaction_se', np.nan),
                                                r.get('interaction_p', 1.0)),
            'ΔR² (DASS)': f"{r.get('delta_r2_dass', 0):.3f}",
            'ΔR² (UCLA)': f"{r.get('delta_r2_ucla', 0):.3f}",
            'ΔR² (Int.)': f"{r.get('delta_r2_interaction', 0):.3f}",
        })

    return pd.DataFrame(rows)
