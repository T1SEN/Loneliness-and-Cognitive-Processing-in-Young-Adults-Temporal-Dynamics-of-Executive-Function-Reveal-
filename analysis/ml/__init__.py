"""
Machine Learning Analysis Suite
===============================

Machine learning approaches for predicting loneliness from EF measures.

Modules:
- nested_cv: Nested cross-validation with hyperparameter tuning
- classification: Binary loneliness classification (high/low UCLA)
- feature_selection: Recursive feature elimination
- residual_prediction: Predict EF from DASS residuals
- dass_prediction: Predict DASS from EF (reverse causality test)

Usage:
    # Run nested CV classification
    python -m analysis.ml.nested_cv --task classification --features demo_dass

    # Run nested CV regression
    python -m analysis.ml.nested_cv --task regression --features ef_demo_dass
"""

# Module exports
__all__ = [
    "nested_cv",
    "classification",
    "feature_selection",
    "residual_prediction",
    "dass_prediction",
]
