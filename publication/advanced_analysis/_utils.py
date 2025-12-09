"""
Advanced Analysis Utilities
===========================

Shared functions for path analysis and mediation suites.

This module provides:
- Bootstrap mediation analysis
- SEM model fitting (semopy with OLS fallback)
- EF composite creation
- Path coefficient extraction
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# =============================================================================
# SEMOPY IMPORT WITH FALLBACK
# =============================================================================

try:
    from semopy import Model as SemopyModel
    from semopy import calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    SemopyModel = None
    calc_stats = None

# =============================================================================
# OUTPUT PATHS (publication/basic_analysis/utils.py 패턴 따름)
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BASE_OUTPUT = RESULTS_DIR / "publication" / "advanced_analysis"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MEDIATION FUNCTIONS
# =============================================================================

def bootstrap_mediation(
    df: pd.DataFrame,
    x_col: str,
    m_col: str,
    y_col: str,
    covariates: Optional[List[str]] = None,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Bootstrap mediation with optional covariate control.

    Tests pathway: X → M → Y controlling for covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Data for analysis
    x_col : str
        Independent variable column name
    m_col : str
        Mediator column name
    y_col : str
        Dependent variable column name
    covariates : list of str, optional
        Covariate column names
    n_bootstrap : int
        Number of bootstrap iterations (default: 5000)
    ci : float
        Confidence interval level (default: 0.95)
    seed : int
        Random seed (default: 42)

    Returns
    -------
    dict
        Mediation results including a, b, c, c', indirect effect, and CIs
    """
    covariates = covariates or []
    cov_str = (" + " + " + ".join(covariates)) if covariates else ""

    formula_a = f"{m_col} ~ {x_col}{cov_str}"
    formula_b = f"{y_col} ~ {x_col} + {m_col}{cov_str}"
    formula_c = f"{y_col} ~ {x_col}{cov_str}"

    a_model = smf.ols(formula_a, data=df).fit()
    bc_model = smf.ols(formula_b, data=df).fit()
    c_model = smf.ols(formula_c, data=df).fit()

    a = a_model.params.get(x_col, np.nan)
    b = bc_model.params.get(m_col, np.nan)
    c_prime = bc_model.params.get(x_col, np.nan)
    c = c_model.params.get(x_col, np.nan)

    indirect = a * b
    indirect_effects = []
    alpha = (1 - ci) / 2
    n = len(df)
    rng = np.random.default_rng(seed)
    fail_count = 0

    for i in range(n_bootstrap):
        sample = df.sample(n=n, replace=True, random_state=int(rng.integers(0, 1e9)))
        try:
            a_boot = smf.ols(formula_a, data=sample).fit().params.get(x_col, np.nan)
            b_boot = smf.ols(formula_b, data=sample).fit().params.get(m_col, np.nan)
            if np.isfinite(a_boot) and np.isfinite(b_boot):
                indirect_effects.append(a_boot * b_boot)
            else:
                fail_count += 1
        except Exception:
            fail_count += 1
            continue

    # Warn if high failure rate
    if fail_count > n_bootstrap * 0.1:
        warnings.warn(f"High bootstrap failure rate: {fail_count}/{n_bootstrap} ({100*fail_count/n_bootstrap:.1f}%)")

    indirect_effects = np.array(indirect_effects)
    ci_low = np.percentile(indirect_effects, alpha * 100) if len(indirect_effects) else np.nan
    ci_high = np.percentile(indirect_effects, (1 - alpha) * 100) if len(indirect_effects) else np.nan

    significant = False
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        significant = (ci_low > 0) or (ci_high < 0)

    return {
        'a': a,
        'b': b,
        'c': c,
        'c_prime': c_prime,
        'indirect': indirect,
        'indirect_ci_low': ci_low,
        'indirect_ci_high': ci_high,
        'indirect_significant': significant,
        'proportion_mediated': indirect / c if c not in (0, np.nan) else np.nan,
        'n_bootstrap': len(indirect_effects),
        'covariates_used': covariates
    }


def sobel_test(a: float, b: float, se_a: float, se_b: float) -> Tuple[float, float]:
    """
    Sobel test for indirect effect significance.

    Parameters
    ----------
    a : float
        a-path coefficient (X → M)
    b : float
        b-path coefficient (M → Y)
    se_a : float
        Standard error of a
    se_b : float
        Standard error of b

    Returns
    -------
    tuple
        (z-statistic, p-value)
    """
    se_indirect = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    z = (a * b) / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# =============================================================================
# EF COMPOSITE FUNCTIONS
# =============================================================================

def create_ef_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create EF composite from three task metrics.

    EF composite = mean(z_pe_rate, z_stroop_interference, z_prp_bottleneck)
    Higher values = worse performance

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pe_rate, stroop_interference, prp_bottleneck columns

    Returns
    -------
    pd.DataFrame
        DataFrame with ef_composite column added
    """
    from publication.preprocessing import safe_zscore

    ef_metrics = ['pe_rate', 'stroop_interference', 'prp_bottleneck']

    # Check available metrics
    available = [m for m in ef_metrics if m in df.columns]

    if not available:
        raise ValueError(f"No EF metrics found. Expected: {ef_metrics}")

    if len(available) < len(ef_metrics):
        print(f"[WARNING] Only {len(available)}/{len(ef_metrics)} EF metrics available: {available}")

    # Z-score each metric
    z_scores = pd.DataFrame(index=df.index)
    for metric in available:
        z_scores[f'z_{metric}'] = safe_zscore(df[metric])

    # Mean composite (higher = worse)
    df = df.copy()
    df['ef_composite'] = z_scores.mean(axis=1)

    return df


# =============================================================================
# PATH MODEL FITTING
# =============================================================================

def fit_path_model_semopy(model_spec: str, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """
    Fit path model using semopy and extract fit indices.

    Falls back to OLS approximation if semopy is not available.

    Parameters
    ----------
    model_spec : str
        Semopy model specification
    df : pd.DataFrame
        Data for fitting
    model_name : str
        Name for identification

    Returns
    -------
    dict
        Model fit results including AIC, BIC, CFI, RMSEA, etc.
    """
    if not SEMOPY_AVAILABLE:
        return fit_path_model_ols(model_spec, df, model_name)

    try:
        model = SemopyModel(model_spec)
        model.fit(df)

        # Extract fit statistics
        stats_df = calc_stats(model)

        # Extract parameters
        params = model.inspect()

        result = {
            'model_name': model_name,
            'n_obs': len(df),
            'converged': True,
            'method': 'semopy',
        }

        # Add fit indices - stats_df has columns as index names and 'Value' row
        fit_indices = ['AIC', 'BIC', 'CFI', 'GFI', 'AGFI', 'NFI', 'TLI', 'RMSEA', 'chi2', 'DoF']
        for idx_name in fit_indices:
            if idx_name in stats_df.columns:
                val = stats_df.loc['Value', idx_name]
                result[idx_name.lower().replace(' ', '_').replace('-', '_')] = val

        # Store parameters
        result['params'] = params
        result['model_object'] = model

        return result

    except Exception as e:
        print(f"[WARNING] Semopy fitting failed for {model_name}: {e}")
        return fit_path_model_ols(model_spec, df, model_name)


def fit_path_model_ols(model_spec: str, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """
    Fallback: Fit path model using OLS regression (approximation).

    Note: This does not provide true SEM fit indices.

    Parameters
    ----------
    model_spec : str
        Semopy model specification
    df : pd.DataFrame
        Data for fitting
    model_name : str
        Name for identification

    Returns
    -------
    dict
        Model fit results with pseudo fit indices
    """
    print(f"[INFO] Using OLS approximation for {model_name}")

    # Parse the model specification
    lines = [l.strip() for l in model_spec.strip().split('\n') if l.strip() and '~' in l]

    models = {}
    for line in lines:
        parts = line.split('~')
        if len(parts) != 2:
            continue

        dv = parts[0].strip()
        rhs = parts[1].strip()

        # Remove parameter labels (e.g., "c1*ef_composite" -> "ef_composite")
        rhs_clean = re.sub(r'\w+\*', '', rhs)

        formula = f"{dv} ~ {rhs_clean}"

        try:
            model = smf.ols(formula, data=df).fit()
            models[dv] = model
        except Exception as e:
            print(f"[ERROR] OLS fit failed for {dv}: {e}")

    # Compute pseudo fit indices
    total_aic = sum(m.aic for m in models.values())
    total_bic = sum(m.bic for m in models.values())

    result = {
        'model_name': model_name,
        'n_obs': len(df),
        'converged': True,
        'aic': total_aic,
        'bic': total_bic,
        'cfi': np.nan,  # Not available for OLS
        'rmsea': np.nan,
        'chi2': np.nan,
        'dof': np.nan,
        'ols_models': models,
        'method': 'OLS_approximation',
    }

    # Extract path coefficients
    params_list = []
    for dv, model in models.items():
        for param, coef in model.params.items():
            if param != 'Intercept':
                params_list.append({
                    'lval': dv,
                    'op': '~',
                    'rval': param,
                    'Estimate': coef,
                    'Std. Err': model.bse.get(param, np.nan),
                    'p-value': model.pvalues.get(param, np.nan),
                })

    result['params'] = pd.DataFrame(params_list)

    return result


def extract_path_coefficients(result: Dict[str, Any], model_info: Dict[str, str]) -> Dict[str, float]:
    """
    Extract a-path and b-path coefficients from model result.

    For semopy output:
    - params DataFrame: lval (DV), op (~), rval (predictor), Estimate, Std. Err, p-value
    - a-path: exogenous → mediator
    - b-path: mediator → endogenous

    Parameters
    ----------
    result : dict
        Model fitting result from fit_path_model_semopy or fit_path_model_ols
    model_info : dict
        Model specification with keys: exogenous, mediator, endogenous

    Returns
    -------
    dict
        a_coef, b_coef, indirect_effect, a_se, b_se, a_p, b_p
    """
    params = result.get('params')

    if params is None or (isinstance(params, pd.DataFrame) and params.empty):
        return {'a_coef': np.nan, 'b_coef': np.nan, 'indirect_effect': np.nan,
                'a_se': np.nan, 'b_se': np.nan, 'a_p': np.nan, 'b_p': np.nan}

    exog = model_info.get('exogenous', '')
    mediator = model_info.get('mediator', '')
    endog = model_info.get('endogenous', '')

    a_coef = b_coef = a_se = b_se = a_p = b_p = np.nan

    if isinstance(params, pd.DataFrame) and len(params) > 0:
        # Filter regression paths only (op == '~')
        reg_params = params[params['op'] == '~'].copy()

        # a-path: mediator ~ exogenous (exogenous predicting mediator)
        # lval = mediator, rval = exogenous
        a_row = reg_params[(reg_params['lval'] == mediator) & (reg_params['rval'] == exog)]

        # b-path: endogenous ~ mediator (mediator predicting endogenous)
        # lval = endogenous, rval = mediator
        b_row = reg_params[(reg_params['lval'] == endog) & (reg_params['rval'] == mediator)]

        if len(a_row) > 0:
            a_coef = a_row['Estimate'].values[0]
            a_se = a_row['Std. Err'].values[0] if 'Std. Err' in a_row.columns else np.nan
            a_p = a_row['p-value'].values[0] if 'p-value' in a_row.columns else np.nan

        if len(b_row) > 0:
            b_coef = b_row['Estimate'].values[0]
            b_se = b_row['Std. Err'].values[0] if 'Std. Err' in b_row.columns else np.nan
            b_p = b_row['p-value'].values[0] if 'p-value' in b_row.columns else np.nan

    indirect = a_coef * b_coef if not (np.isnan(a_coef) or np.isnan(b_coef)) else np.nan

    return {
        'a_coef': a_coef,
        'b_coef': b_coef,
        'indirect_effect': indirect,
        'a_se': a_se,
        'b_se': b_se,
        'a_p': a_p,
        'b_p': b_p,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'RESULTS_DIR',
    'BASE_OUTPUT',
    'SEMOPY_AVAILABLE',
    # Mediation
    'bootstrap_mediation',
    'sobel_test',
    # EF composite
    'create_ef_composite',
    # Path models
    'fit_path_model_semopy',
    'fit_path_model_ols',
    'extract_path_coefficients',
]
