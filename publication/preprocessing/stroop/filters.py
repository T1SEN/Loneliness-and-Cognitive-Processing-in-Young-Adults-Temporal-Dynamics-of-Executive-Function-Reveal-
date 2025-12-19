"""Stroop QC filters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ..constants import RAW_DIR


@dataclass
class StroopQCCriteria:
    min_accuracy: Optional[float] = None
    max_mean_rt: Optional[float] = None
    require_metrics: bool = True


def get_stroop_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[StroopQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = StroopQCCriteria()

    summary_path = data_dir / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] cognitive summary file not found: {summary_path}")
        return set()

    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary_df["testName"] = summary_df["testName"].str.lower()

    stroop_df = summary_df[summary_df["testName"] == "stroop"].copy()
    if stroop_df.empty:
        return set()

    if criteria.require_metrics:
        mask = (
            stroop_df["mrt_cong"].notna()
            & stroop_df["mrt_incong"].notna()
            & stroop_df["stroop_effect"].notna()
            & stroop_df["accuracy"].notna()
        )
        stroop_df = stroop_df[mask]

    if criteria.min_accuracy is not None:
        stroop_df = stroop_df[stroop_df["accuracy"] >= criteria.min_accuracy]

    if criteria.max_mean_rt is not None:
        stroop_df = stroop_df[
            ((stroop_df["mrt_cong"] + stroop_df["mrt_incong"]) / 2) <= criteria.max_mean_rt
        ]

    valid_ids = set(stroop_df["participantId"].unique())

    if verbose:
        all_stroop = set(summary_df[summary_df["testName"] == "stroop"]["participantId"].unique())
        excluded = all_stroop - valid_ids
        if excluded:
            print(f"  [INFO] Stroop QC failed: {len(excluded)}")

    return valid_ids
