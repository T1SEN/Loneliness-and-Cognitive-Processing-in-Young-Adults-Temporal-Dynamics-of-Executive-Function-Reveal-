"""
Statistics Module
=================

Statistical analysis utilities - Ex-Gaussian fitting, post-error metrics.

Usage:
    from analysis.statistics import (
        fit_exgaussian,
        compute_pes,
    )
"""

# Ex-Gaussian distribution fitting
from analysis.statistics.exgaussian import (
    exgaussian_pdf,
    exgaussian_cdf,
    exgaussian_mean,
    exgaussian_var,
    fit_exgaussian,
    fit_exgaussian_dict,
    fit_exgaussian_by_participant,
    fit_exgaussian_by_condition,
    compute_exgaussian_metrics,
)

# Post-error analysis
from analysis.statistics.post_error import (
    compute_pes,
    compute_post_error_accuracy,
    compute_pes_robust,
    compute_prp_pes,
    compute_stroop_pes,
    compute_wcst_pes,
    compute_all_task_pes,
)

__all__ = [
    # Ex-Gaussian
    'exgaussian_pdf',
    'exgaussian_cdf',
    'exgaussian_mean',
    'exgaussian_var',
    'fit_exgaussian',
    'fit_exgaussian_dict',
    'fit_exgaussian_by_participant',
    'fit_exgaussian_by_condition',
    'compute_exgaussian_metrics',
    # Post-error
    'compute_pes',
    'compute_post_error_accuracy',
    'compute_pes_robust',
    'compute_prp_pes',
    'compute_stroop_pes',
    'compute_wcst_pes',
    'compute_all_task_pes',
]
