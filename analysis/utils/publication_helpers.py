"""
Legacy Publication Helper Compatibility Layer
=============================================

Framework and archive scripts historically imported utilities from
``analysis.utils.publication_helpers``.  The modern implementation lives
in ``analysis.visualization.publication``; this shim re-exports the
relevant helpers (and the frequently co-imported ``load_master_dataset``)
so legacy scripts keep functioning.
"""

from analysis.visualization.publication import (  # noqa: F401
    bootstrap_ci,
    cohens_d,
    format_ci,
    format_pvalue,
    save_publication_figure,
    set_publication_style,
    standardize_variables,
)
from analysis.preprocessing import load_master_dataset  # noqa: F401

__all__ = [
    "bootstrap_ci",
    "cohens_d",
    "format_ci",
    "format_pvalue",
    "save_publication_figure",
    "set_publication_style",
    "standardize_variables",
    "load_master_dataset",
]
