"""
Visualization Module
====================

Plotting utilities and publication helpers.

Usage:
    from analysis.visualization import (
        set_publication_style,
        create_forest_plot,
        bootstrap_ci,
        cohens_d,
    )
"""

# Plotting utilities
from analysis.visualization.plotting import (
    COLORS,
    SIG_MARKERS,
    get_significance_marker,
    set_publication_style,
    create_model_comparison_plot,
    create_forest_plot,
    create_scatter_with_regression,
    create_distribution_comparison,
    create_correlation_heatmap,
    save_figure,
)

# Publication helpers
from analysis.visualization.publication import (
    bootstrap_ci,
    bootstrap_regression_coef,
    cohens_d,
    eta_squared,
    partial_eta_squared,
    r_to_d,
    d_to_r,
    format_pvalue,
    format_ci,
    format_statistic,
    create_apa_table,
    save_publication_figure,
    nested_cv_with_ci,
    describe_sample,
)

__all__ = [
    # Plotting
    'COLORS',
    'SIG_MARKERS',
    'get_significance_marker',
    'set_publication_style',
    'create_model_comparison_plot',
    'create_forest_plot',
    'create_scatter_with_regression',
    'create_distribution_comparison',
    'create_correlation_heatmap',
    'save_figure',
    # Publication
    'bootstrap_ci',
    'bootstrap_regression_coef',
    'cohens_d',
    'eta_squared',
    'partial_eta_squared',
    'r_to_d',
    'd_to_r',
    'format_pvalue',
    'format_ci',
    'format_statistic',
    'create_apa_table',
    'save_publication_figure',
    'nested_cv_with_ci',
    'describe_sample',
]
