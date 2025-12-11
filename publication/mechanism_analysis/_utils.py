"""
Mechanism Analysis - Shared Utilities
=====================================

Common constants and helper functions for mechanism analysis suite.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Any, Dict, List, Optional

import pandas as pd

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR
from publication.advanced_analysis._utils import (
    fit_path_model_semopy,
    fit_path_model_ols,
    extract_path_coefficients,
)

# =============================================================================
# OUTPUT PATHS
# =============================================================================

BASE_OUTPUT = ANALYSIS_OUTPUT_DIR / "mechanism_analysis"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

# Default covariates used in mechanism-level path analyses
DEFAULT_PATH_COVARIATES = ['z_age', 'gender_male']


def _format_covariates(df: pd.DataFrame, covariates: Optional[List[str]] = None) -> str:
    """Build RHS covariate string for SEM equations."""
    covariates = covariates or []
    available = [cov for cov in covariates if cov in df.columns]
    return "".join(f" + {cov}" for cov in available)


def run_path_models(
    df: pd.DataFrame,
    param_col: str,
    param_label: str,
    module_tag: str,
    dass_var: str = 'z_dass_dep',
    covariates: Optional[List[str]] = None,
    min_n: int = 30,
) -> List[Dict[str, Any]]:
    """Fit core loneliness↔DASS path models for a mechanism parameter."""
    if param_col not in df.columns or dass_var not in df.columns or 'z_ucla' not in df.columns:
        return []

    covariates = covariates or DEFAULT_PATH_COVARIATES
    available_covs = [cov for cov in covariates if cov in df.columns]

    required_cols = set(['z_ucla', param_col, dass_var] + available_covs)
    clean_df = df.dropna(subset=list(required_cols))

    if len(clean_df) < min_n:
        return []

    cov_str = _format_covariates(clean_df, available_covs)

    model_specs = [
        {
            'name': 'loneliness_to_mechanism_to_dass',
            'description': 'Loneliness → mechanism → DASS',
            'spec': (
                f"{param_col} ~ a*z_ucla{cov_str}\n"
                f"{dass_var} ~ b*{param_col} + z_ucla{cov_str}\n"
            ),
            'exogenous': 'z_ucla',
            'mediator': param_col,
            'endogenous': dass_var,
        },
        {
            'name': 'loneliness_to_dass_to_mechanism',
            'description': 'Loneliness → DASS → mechanism',
            'spec': (
                f"{dass_var} ~ a*z_ucla{cov_str}\n"
                f"{param_col} ~ b*{dass_var} + z_ucla{cov_str}\n"
            ),
            'exogenous': 'z_ucla',
            'mediator': dass_var,
            'endogenous': param_col,
        },
        {
            'name': 'dass_to_loneliness_to_mechanism',
            'description': 'DASS → loneliness → mechanism',
            'spec': (
                f"z_ucla ~ a*{dass_var}{cov_str}\n"
                f"{param_col} ~ b*z_ucla + {dass_var}{cov_str}\n"
            ),
            'exogenous': dass_var,
            'mediator': 'z_ucla',
            'endogenous': param_col,
        },
    ]

    records: List[Dict[str, Any]] = []

    for spec in model_specs:
        try:
            fit_result = fit_path_model_semopy(
                spec['spec'],
                clean_df,
                f"{module_tag}_{param_col}_{dass_var}_{spec['name']}"
            )
        except ImportError:
            fit_result = fit_path_model_ols(
                spec['spec'],
                clean_df,
                f"{module_tag}_{param_col}_{dass_var}_{spec['name']}"
            )

        path_stats = extract_path_coefficients(fit_result, {
            'exogenous': spec['exogenous'],
            'mediator': spec['mediator'],
            'endogenous': spec['endogenous'],
        })

        records.append({
            'module': module_tag,
            'parameter': param_label,
            'param_col': param_col,
            'dass_var': dass_var,
            'model_name': spec['name'],
            'model_description': spec['description'],
            'n': len(clean_df),
            'a_coef': path_stats['a_coef'],
            'b_coef': path_stats['b_coef'],
            'indirect_effect': path_stats['indirect_effect'],
            'a_p': path_stats['a_p'],
            'b_p': path_stats['b_p'],
            'method': fit_result.get('method'),
            'aic': fit_result.get('aic'),
            'bic': fit_result.get('bic'),
            'cfi': fit_result.get('cfi'),
            'rmsea': fit_result.get('rmsea'),
        })

    return records

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BASE_OUTPUT',
    'DEFAULT_PATH_COVARIATES',
    'run_path_models',
]
