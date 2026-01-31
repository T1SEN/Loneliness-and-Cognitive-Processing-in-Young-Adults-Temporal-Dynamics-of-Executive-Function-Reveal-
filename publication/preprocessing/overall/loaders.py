"""Overall loaders (merged summaries)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..stroop.loaders import load_stroop_summary
from ..wcst.loaders import load_wcst_summary


def load_overall_summary(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge task summaries for overall datasets.

    Returns a per-participant summary that contains Stroop and WCST metrics.
    """
    stroop_summary = load_stroop_summary(data_dir)
    wcst_summary = load_wcst_summary(data_dir)

    merged = stroop_summary.merge(wcst_summary, on="participant_id", how="inner")
    return merged
