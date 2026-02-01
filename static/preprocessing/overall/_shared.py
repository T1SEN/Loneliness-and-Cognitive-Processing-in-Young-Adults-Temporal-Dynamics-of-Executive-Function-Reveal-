"""Shared helpers for overall feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd

from ..constants import get_results_dir
from .loaders import load_overall_summary


def get_overall_participant_ids(data_dir: Path | None = None) -> Set[str]:
    if data_dir is None:
        data_dir = get_results_dir("overall")
    summary = load_overall_summary(data_dir)
    if summary.empty or "participant_id" not in summary.columns:
        return set()
    return set(summary["participant_id"].dropna().astype(str))


def filter_to_overall_ids(df: pd.DataFrame, overall_ids: Set[str]) -> pd.DataFrame:
    if df.empty or not overall_ids or "participant_id" not in df.columns:
        return df
    mask = df["participant_id"].astype(str).isin(overall_ids)
    return df[mask].copy()
