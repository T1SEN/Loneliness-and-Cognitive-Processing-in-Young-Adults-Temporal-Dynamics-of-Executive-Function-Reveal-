"""
Path Analysis (Depression)
==========================

Tests gender-moderated pathways linking UCLA loneliness, DASS depression, and
each EF outcome (WCST PE rate, Stroop interference, PRP bottleneck).
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from ._utils import (
    BASE_OUTPUT,
    load_common_path_data,
    fit_all_path_models,
)

OUTPUT_DIR = BASE_OUTPUT / "depression"
DASS_COL = 'z_dass_dep'


def run(verbose: bool = True):
    """
    Run gender-moderated path analysis for DASS Depression.
    """
    df = load_common_path_data(DASS_COL)
    return fit_all_path_models(
        df,
        dass_col=DASS_COL,
        dass_label="DASS Depression",
        output_dir=OUTPUT_DIR,
        verbose=verbose,
    )


if __name__ == "__main__":
    run(verbose=True)
