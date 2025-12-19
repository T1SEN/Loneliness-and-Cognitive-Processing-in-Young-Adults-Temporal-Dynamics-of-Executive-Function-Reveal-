"""WCST QC filters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ..constants import RAW_DIR


@dataclass
class WCSTQCCriteria:
    max_total_errors: int = 60
    max_perseverative_responses: int = 60
    min_completed_categories: int = 1
    require_metrics: bool = True


def get_wcst_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[WCSTQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = WCSTQCCriteria()

    summary_path = data_dir / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] cognitive summary file not found: {summary_path}")
        return set()

    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary_df["testName"] = summary_df["testName"].str.lower()

    wcst_df = summary_df[summary_df["testName"] == "wcst"].copy()
    if wcst_df.empty:
        return set()

    if criteria.require_metrics:
        mask = (
            (wcst_df["totalTrialCount"] > 0)
            & (wcst_df["completedCategories"] >= criteria.min_completed_categories)
            & (wcst_df["perseverativeErrorCount"].notna())
        )
        wcst_df = wcst_df[mask]

    if criteria.max_total_errors > 0 and "totalErrorCount" in wcst_df.columns:
        wcst_df = wcst_df[wcst_df["totalErrorCount"] < criteria.max_total_errors]

    if criteria.max_perseverative_responses > 0 and "perseverativeResponses" in wcst_df.columns:
        wcst_df = wcst_df[wcst_df["perseverativeResponses"] < criteria.max_perseverative_responses]

    valid_ids = set(wcst_df["participantId"].unique())

    if verbose:
        all_wcst = set(summary_df[summary_df["testName"] == "wcst"]["participantId"].unique())
        excluded = all_wcst - valid_ids
        if excluded:
            print(f"  [INFO] WCST QC failed: {len(excluded)}")

    return valid_ids
