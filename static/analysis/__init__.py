"""
Publication Basic Analysis Suite
================================

Core descriptive, correlational, and hierarchical-regression scripts for the
basic IRB analysis section.
"""

from .utils import (
    get_analysis_data,
    filter_vars,
    get_output_dir,
    DESCRIPTIVE_VARS,
    CORRELATION_VARS,
    PRIMARY_OUTCOMES,
    OUTCOMES_BY_TASK,
    get_primary_outcomes,
    STANDARDIZED_PREDICTORS,
    prepare_regression_data,
    format_pvalue,
    format_coefficient,
    print_section_header,
)

__all__ = [
    "get_analysis_data",
    "filter_vars",
    "get_output_dir",
    "DESCRIPTIVE_VARS",
    "CORRELATION_VARS",
    "PRIMARY_OUTCOMES",
    "OUTCOMES_BY_TASK",
    "get_primary_outcomes",
    "STANDARDIZED_PREDICTORS",
    "prepare_regression_data",
    "format_pvalue",
    "format_coefficient",
    "print_section_header",
]
