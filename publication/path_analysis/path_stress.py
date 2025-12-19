"""
Path Analysis (Stress)
======================

Tests gender-moderated pathways linking UCLA loneliness, DASS stress, and
each EF outcome (WCST PE rate, Stroop interference, PRP bottleneck).
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse

from publication.preprocessing.constants import VALID_TASKS

from ._utils import (
    get_output_dir,
    load_common_path_data,
    fit_all_path_models,
)

DASS_COL = 'z_dass_str'

def parse_args():
    parser = argparse.ArgumentParser(description="Run path analysis for DASS Stress.")
    parser.add_argument("--task", required=True, choices=sorted(VALID_TASKS))
    return parser.parse_args()

def run(task: str, verbose: bool = True):
    """
    Run gender-moderated path analysis for DASS Stress.
    """
    df = load_common_path_data(DASS_COL, task=task)
    output_dir = get_output_dir(task, "stress")
    return fit_all_path_models(
        df,
        dass_col=DASS_COL,
        dass_label="DASS Stress",
        output_dir=output_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    args = parse_args()
    run(task=args.task, verbose=True)
