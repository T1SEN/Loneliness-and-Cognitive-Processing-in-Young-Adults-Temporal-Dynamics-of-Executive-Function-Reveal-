"""
Compatibility shim for the renamed depression path suite.

The original module was ``path_comparison_suite``; it has been renamed to
``path_depression_suite`` to align with the anxiety/stress naming scheme.
This file re-exports everything from the new module while issuing a
deprecation warning when run as a script.
"""

from __future__ import annotations

import warnings

from .path_depression_suite import *  # noqa: F401,F403

warnings.warn(
    "publication.advanced_analysis.path_comparison_suite is deprecated. "
    "Use publication.advanced_analysis.path_depression_suite instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == '__main__':
    import runpy

    runpy.run_module(
        'publication.advanced_analysis.path_depression_suite',
        run_name='__main__',
    )
