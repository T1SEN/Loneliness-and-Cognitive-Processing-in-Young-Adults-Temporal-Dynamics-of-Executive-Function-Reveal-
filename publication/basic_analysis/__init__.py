"""
Publication Basic Analysis Suite
================================

IRB 연구계획서에 명시된 기본 통계분석 스크립트 모음

Scripts:
    01_descriptive_statistics.py  - 기술통계분석 (N, Mean, SD, Min, Max)
    02_correlation_analysis.py    - 상관분석 (전체 상관행렬)
    03_hierarchical_regression.py - 위계적 다중회귀분석 (DASS 통제)

Usage:
    python -m publication.basic_analysis.01_descriptive_statistics
    python -m publication.basic_analysis.02_correlation_analysis
    python -m publication.basic_analysis.03_hierarchical_regression
"""

from .utils import (
    get_analysis_data,
    OUTPUT_DIR,
    DESCRIPTIVE_VARS,
    CORRELATION_VARS,
    TIER1_OUTCOMES,
    STANDARDIZED_PREDICTORS,
    prepare_regression_data,
    format_pvalue,
    format_coefficient,
    print_section_header,
)

__all__ = [
    'get_analysis_data',
    'OUTPUT_DIR',
    'DESCRIPTIVE_VARS',
    'CORRELATION_VARS',
    'TIER1_OUTCOMES',
    'STANDARDIZED_PREDICTORS',
    'prepare_regression_data',
    'format_pvalue',
    'format_coefficient',
    'print_section_header',
]
