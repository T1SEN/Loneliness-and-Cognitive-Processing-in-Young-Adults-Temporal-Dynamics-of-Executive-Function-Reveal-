"""
Publication Advanced Analysis Suite
====================================

경로분석, 매개분석, 베이지안 SEM 스위트

Scripts:
    mediation_suite.py         - UCLA → DASS → EF 매개분석
    path_depression_suite.py   - 경로모형 비교 (Depression)
    path_anxiety_suite.py      - 경로모형 비교 (Anxiety)
    path_stress_suite.py       - 경로모형 비교 (Stress)
    bayesian_suite.py          - 베이지안 SEM

Usage:
    python -m publication.advanced_analysis.mediation_suite
    python -m publication.advanced_analysis.path_depression_suite
    python -m publication.advanced_analysis.path_anxiety_suite
    python -m publication.advanced_analysis.path_stress_suite
    python -m publication.advanced_analysis.bayesian_suite
"""

import warnings

# Shared utilities
from ._utils import (
    BASE_OUTPUT,
    RESULTS_DIR,
    SEMOPY_AVAILABLE,
    bootstrap_mediation,
    sobel_test,
    create_ef_composite,
    fit_path_model_semopy,
    fit_path_model_ols,
    extract_path_coefficients,
)

# Suite run functions (lazy import to avoid circular dependencies)
def run_mediation(*args, **kwargs):
    """Run mediation suite."""
    from .mediation_suite import run
    return run(*args, **kwargs)


def run_path_depression(*args, **kwargs):
    """Run path depression suite (formerly path_comparison)."""
    from .path_depression_suite import run
    return run(*args, **kwargs)


def run_path_comparison(*args, **kwargs):
    """Backward-compatible alias for the depression path suite."""
    warnings.warn(
        "publication.advanced_analysis.path_comparison_suite has been renamed "
        "to path_depression_suite. Please update imports/CLI usage accordingly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_path_depression(*args, **kwargs)


def run_path_anxiety(*args, **kwargs):
    """Run path anxiety suite."""
    from .path_anxiety_suite import run
    return run(*args, **kwargs)


def run_path_stress(*args, **kwargs):
    """Run path stress suite."""
    from .path_stress_suite import run
    return run(*args, **kwargs)


def run_bayesian(*args, **kwargs):
    """Run Bayesian suite."""
    from .bayesian_suite import run
    return run(*args, **kwargs)


__all__ = [
    # Constants
    'BASE_OUTPUT',
    'RESULTS_DIR',
    'SEMOPY_AVAILABLE',
    # Shared functions
    'bootstrap_mediation',
    'sobel_test',
    'create_ef_composite',
    'fit_path_model_semopy',
    'fit_path_model_ols',
    'extract_path_coefficients',
    # Suite runners
    'run_mediation',
    'run_path_depression',
    'run_path_comparison',
    'run_path_anxiety',
    'run_path_stress',
    'run_bayesian',
]
