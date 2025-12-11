"""
EF-focused path analysis modules.

Each script (path_anxiety, path_depression, path_stress) estimates the
gender-moderated pathways linking UCLA loneliness, the corresponding DASS
subscale, and the EF composite outcome.
"""

from . import path_anxiety, path_depression, path_stress  # noqa: F401

__all__ = [
    'path_anxiety',
    'path_depression',
    'path_stress',
]
