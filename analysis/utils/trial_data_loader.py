"""
Legacy Trial Loader Compatibility Layer
=======================================

The refactor that introduced ``analysis.preprocessing.trial_loaders``
removed the original ``analysis.utils.trial_data_loader`` module that
many archived scripts still import.  This shim simply re-exports the
current trial-level loader functions so those scripts can run without
additional changes.
"""

from analysis.preprocessing.trial_loaders import (
    load_prp_trials,
    load_stroop_trials,
    load_wcst_trials,
)

__all__ = ["load_prp_trials", "load_stroop_trials", "load_wcst_trials"]
