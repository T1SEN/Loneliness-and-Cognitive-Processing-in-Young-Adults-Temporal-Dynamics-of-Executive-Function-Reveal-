"""
Modeling Utilities
==================

DASS-controlled regression templates and model fitting utilities.

For data loading and preprocessing, use: analysis.preprocessing
For statistics (Ex-Gaussian, post-error), use: analysis.statistics
For visualization, use: analysis.visualization
"""

from analysis.utils.modeling import (
    DASS_CONTROL_FORMULA,
    fit_dass_controlled_model,
    fit_hierarchical_models,
    extract_coefficients,
    extract_ucla_effects,
    verify_dass_control,
    results_to_apa_table,
)

# Convenience re-exports from preprocessing (commonly used with modeling)
from analysis.preprocessing import (
    standardize_predictors,
    prepare_gender_variable,
)

__all__ = [
    # Core modeling
    "DASS_CONTROL_FORMULA",
    "fit_dass_controlled_model",
    "fit_hierarchical_models",
    "extract_coefficients",
    "extract_ucla_effects",
    "verify_dass_control",
    "results_to_apa_table",
    # Convenience re-exports
    "standardize_predictors",
    "prepare_gender_variable",
]
