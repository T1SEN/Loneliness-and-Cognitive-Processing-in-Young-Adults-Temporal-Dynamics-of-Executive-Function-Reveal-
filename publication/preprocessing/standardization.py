"""
Standardization Utilities
=========================

Provides consistent z-score standardization across all analysis modules.

Key features:
- NaN-safe: Uses pandas operations that skip NaN values by default
- Consistent ddof: Uses ddof=1 (sample standard deviation) throughout
- Column mapping: Standardized naming convention (z_ucla, z_dass_dep, etc.)

Usage:
    from publication.preprocessing import (
        safe_zscore,
        standardize_predictors,
        PREDICTOR_COLUMN_MAPPING
    )

    # Single column
    df['z_ucla'] = safe_zscore(df['ucla_total'])

    # Multiple columns with standard mapping
    df = standardize_predictors(df)

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import warnings


# =============================================================================
# COLUMN MAPPING
# =============================================================================

# Standard mapping from raw column names to z-score column names
PREDICTOR_COLUMN_MAPPING: Dict[str, str] = {
    'ucla_total': 'z_ucla',
    'ucla_score': 'z_ucla',
    'dass_depression': 'z_dass_dep',
    'dass_anxiety': 'z_dass_anx',
    'dass_stress': 'z_dass_str',
    'age': 'z_age',
}

# Default columns to standardize
DEFAULT_PREDICTOR_COLUMNS: List[str] = [
    'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age'
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def safe_zscore(series: pd.Series, ddof: int = 1, fill_constant: float = 0.0) -> pd.Series:
    """
    Calculate z-score with NaN-safe operations.

    Uses pandas operations that skip NaN values by default.
    Uses sample standard deviation (ddof=1) for consistency with
    statistical conventions.

    Parameters
    ----------
    series : pd.Series
        Input data series.
    ddof : int, default 1
        Delta degrees of freedom. Use 1 for sample std, 0 for population std.
    fill_constant : float, default 0.0
        Value to use when the column is constant or std is undefined.
        This mimics StandardScaler behavior (returns zeros).

    Returns
    -------
    pd.Series
        Z-scored series with same index as input.

    Notes
    -----
    - NaN values in input remain NaN in output
    - Returns all NaN if std is 0 or all values are NaN
    - Uses pandas mean/std which skip NaN by default (skipna=True)

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, np.nan, 5])
    >>> safe_zscore(s)
    0   -1.161895
    1   -0.387298
    2    0.387298
    3         NaN
    4    1.161895
    dtype: float64
    """
    mean_val = series.mean()  # skipna=True by default
    std_val = series.std(ddof=ddof)  # skipna=True by default

    if pd.isna(std_val) or std_val == 0:
        warnings.warn(f"Constant or undefined std ({std_val}) detected. Filling with {fill_constant}.")
        # Preserve NaN positions from original series
        result = pd.Series(fill_constant, index=series.index)
        result[series.isna()] = np.nan
        return result

    return (series - mean_val) / std_val


def standardize_predictors(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    ddof: int = 1,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Standardize multiple predictor columns to z-scores.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to standardize. If None, uses DEFAULT_PREDICTOR_COLUMNS.
    column_mapping : dict, optional
        Mapping from input column names to output z-score column names.
        If None, uses PREDICTOR_COLUMN_MAPPING.
    ddof : int, default 1
        Delta degrees of freedom for std calculation.
    inplace : bool, default False
        If True, modify df in place. Otherwise, return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with added z-score columns.

    Examples
    --------
    >>> df = pd.DataFrame({'ucla_total': [30, 40, 50], 'age': [20, 25, 30]})
    >>> df = standardize_predictors(df, columns=['ucla_total', 'age'])
    >>> 'z_ucla' in df.columns and 'z_age' in df.columns
    True
    """
    if columns is None:
        columns = DEFAULT_PREDICTOR_COLUMNS
    if column_mapping is None:
        column_mapping = PREDICTOR_COLUMN_MAPPING

    result = df if inplace else df.copy()

    standardized_cols = []
    for col in columns:
        if col in result.columns:
            z_col = column_mapping.get(col, f'z_{col}')
            result[z_col] = safe_zscore(result[col], ddof=ddof)
            standardized_cols.append(f"{col} -> {z_col}")

    return result


def prepare_gender_variable(
    df: pd.DataFrame,
    gender_col: Optional[str] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Standardize gender coding to binary 'gender_male' variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    gender_col : str, optional
        Column containing gender data. If None, searches for
        'gender_normalized' or 'gender'.
    inplace : bool, default False
        If True, modify df in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'gender' (lowercase) and 'gender_male' (0/1) columns.

    Raises
    ------
    ValueError
        If no gender column is found.
    """
    result = df if inplace else df.copy()

    # Find gender column
    if gender_col is None:
        if 'gender_normalized' in result.columns:
            gender_col = 'gender_normalized'
        elif 'gender' in result.columns:
            gender_col = 'gender'
        else:
            raise ValueError("No gender column found. Provide gender_col parameter.")

    # Normalize to canonical tokens using shared preprocessing logic
    from .core import normalize_gender_series

    normalized = normalize_gender_series(result[gender_col])
    fallback = (
        result[gender_col]
        .fillna('')
        .astype(str)
        .str.strip()
        .str.lower()
        .replace('', np.nan)
    )

    result['gender_normalized'] = normalized
    result['gender'] = normalized.where(normalized.notna(), fallback)

    # Create binary variable: 1 = male, 0 = female, NaN = other/missing
    # Unrecognized values remain NaN and will be excluded from analyses
    result['gender_male'] = normalized.map({'male': 1, 'female': 0})

    return result


# =============================================================================
# FDR CORRECTION UTILITIES
# =============================================================================

def apply_fdr_correction(
    results_df: pd.DataFrame,
    p_col: str = 'p',
    alpha: float = 0.05,
    method: str = 'fdr_bh'
) -> pd.DataFrame:
    """
    Apply FDR (Benjamini-Hochberg) correction to p-values in results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing p-values.
    p_col : str, default 'p'
        Column name containing p-values.
    alpha : float, default 0.05
        Significance threshold.
    method : str, default 'fdr_bh'
        Correction method. Options: 'fdr_bh', 'bonferroni', 'holm', etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'p_fdr' and 'significant_fdr' columns.
    """
    from statsmodels.stats.multitest import multipletests

    result = results_df.copy()

    if p_col not in result.columns:
        warnings.warn(f"Column '{p_col}' not found. Returning unchanged DataFrame.")
        return result

    pvals = result[p_col].values
    valid_mask = ~np.isnan(pvals)

    if valid_mask.sum() < 2:
        warnings.warn("Fewer than 2 valid p-values. Cannot apply FDR correction.")
        result['p_fdr'] = result[p_col]
        result['significant_fdr'] = result[p_col] < alpha
        return result

    # Apply correction only to valid p-values
    _, p_adj, _, _ = multipletests(pvals[valid_mask], method=method, alpha=alpha)

    # Initialize with NaN
    result['p_fdr'] = np.nan
    result.loc[valid_mask, 'p_fdr'] = p_adj
    result['significant_fdr'] = result['p_fdr'] < alpha

    return result


# =============================================================================
# INTERACTION TERM UTILITIES
# =============================================================================

def find_interaction_term(
    model_params_index: pd.Index,
    term1: str = 'ucla',
    term2: str = 'gender'
) -> Optional[str]:
    """
    Dynamically find interaction term in model parameters.

    statsmodels names interaction terms differently depending on how
    variables are specified (categorical vs numeric). This function
    searches for the interaction term regardless of exact naming.

    Parameters
    ----------
    model_params_index : pd.Index
        Index of model.params (parameter names).
    term1 : str, default 'ucla'
        First term to search for in interaction.
    term2 : str, default 'gender'
        Second term to search for in interaction.

    Returns
    -------
    str or None
        Interaction term name if found, None otherwise.

    Examples
    --------
    >>> # Works with both categorical and numeric gender
    >>> find_interaction_term(model.params.index)
    'z_ucla:C(gender_male)[T.1]'  # or 'z_ucla:gender_male'
    """
    candidates = [
        k for k in model_params_index
        if term1.lower() in k.lower() and term2.lower() in k.lower()
        and ':' in k  # Must be an interaction (contains :)
    ]

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        warnings.warn(f"Multiple interaction terms found: {candidates}. Using first.")
        return candidates[0]
    else:
        return None
