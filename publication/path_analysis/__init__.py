"""
EF-focused path analysis modules.

Each script (path_anxiety, path_depression, path_stress) estimates the
gender-moderated pathways linking UCLA loneliness, the corresponding DASS
subscale, and each EF outcome separately (WCST PE rate, Stroop interference,
PRP bottleneck).
"""

from . import path_anxiety, path_depression, path_stress  # noqa: F401

__all__ = [
    'path_anxiety',
    'path_depression',
    'path_stress',
]
