"""
Publication Network Analysis Package
====================================

High-level API for running domain-level Gaussian graphical models (GGMs)
on UCLA/DASS/EF variables, including gender-stratified comparisons.

Usage:
    python -m publication.network_analysis --analysis domain_gender

Programmatic:
    from publication.network_analysis import run
    run(analysis="domain_overall")
"""

from ._config import BASE_OUTPUT, NetworkVariableSet, VARIABLE_SETS
from .network_suite import AVAILABLE_ANALYSES, list_analyses, run

__all__ = [
    "BASE_OUTPUT",
    "NetworkVariableSet",
    "VARIABLE_SETS",
    "AVAILABLE_ANALYSES",
    "run",
    "list_analyses",
]
