"""Stroop preprocessing helpers."""

from .qc import (
    STROOP_REQUIRED_TRIALS,
    STROOP_MIN_ACCURACY,
    clean_stroop_trials,
    prepare_stroop_trials,
    compute_stroop_qc_ids,
)
from .features import (
    compute_stroop_interference,
    compute_stroop_interference_slope,
    build_stroop_features,
)

__all__ = [
    "STROOP_REQUIRED_TRIALS",
    "STROOP_MIN_ACCURACY",
    "clean_stroop_trials",
    "prepare_stroop_trials",
    "compute_stroop_qc_ids",
    "compute_stroop_interference",
    "compute_stroop_interference_slope",
    "build_stroop_features",
]
