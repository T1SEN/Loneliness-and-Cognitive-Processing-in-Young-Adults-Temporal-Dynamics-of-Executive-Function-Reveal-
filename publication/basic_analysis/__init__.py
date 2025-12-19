"""
Publication Basic Analysis Suite
================================

Core descriptive, correlational, and hierarchical-regression scripts for the
basic IRB analysis section.

Scripts:
    descriptive_statistics.py   - Descriptive summary (N, Mean, SD, Min, Max)
    correlation_analysis.py     - Pearson correlation matrix
    hierarchical_regression.py  - Hierarchical multiple regression (DASS-controlled)

Usage:
    python -m publication.basic_analysis.descriptive_statistics --task overall
    python -m publication.basic_analysis.correlation_analysis --task overall
    python -m publication.basic_analysis.hierarchical_regression --task overall
"""

from .utils import (
    get_analysis_data,
    filter_vars,
    get_output_dir,
    DESCRIPTIVE_VARS,
    CORRELATION_VARS,
    TIER1_OUTCOMES,
    TIER1_OUTCOMES_BY_TASK,
    get_tier1_outcomes,
    STANDARDIZED_PREDICTORS,
    prepare_regression_data,
    format_pvalue,
    format_coefficient,
    print_section_header,
)

__all__ = [
    'get_analysis_data',
    'filter_vars',
    'get_output_dir',
    'DESCRIPTIVE_VARS',
    'CORRELATION_VARS',
    'TIER1_OUTCOMES',
    'TIER1_OUTCOMES_BY_TASK',
    'get_tier1_outcomes',
    'STANDARDIZED_PREDICTORS',
    'prepare_regression_data',
    'format_pvalue',
    'format_coefficient',
    'print_section_header',
]
