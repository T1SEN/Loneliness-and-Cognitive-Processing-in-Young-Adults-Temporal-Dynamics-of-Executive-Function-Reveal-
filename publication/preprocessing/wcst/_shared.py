"""Shared helpers for WCST feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..constants import get_results_dir
from ..core import ensure_participant_id


def _load_wcst_summary_metrics(data_dir: Path | None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("wcst")
    summary_path = Path(data_dir) / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()

    summary = pd.read_csv(summary_path, encoding="utf-8")
    summary = ensure_participant_id(summary)
    if "testName" not in summary.columns:
        return pd.DataFrame()
    summary["testName"] = summary["testName"].astype(str).str.strip().str.lower()
    summary = summary[summary["testName"] == "wcst"].copy()
    if summary.empty:
        return pd.DataFrame()

    if "completedCategories" not in summary.columns:
        return summary[["participant_id"]]

    summary = summary[["participant_id", "completedCategories"]].rename(
        columns={"completedCategories": "wcst_completed_categories"}
    )
    summary["wcst_completed_categories"] = pd.to_numeric(
        summary["wcst_completed_categories"], errors="coerce"
    )
    return summary


def prepare_wcst_trials(
    data_dir: None | str | Path = None,
    filter_rt: bool = False,
) -> Dict[str, object]:
    from .loaders import load_wcst_trials

    if filter_rt:
        wcst, _ = load_wcst_trials(data_dir=data_dir, clean=True, filter_rt=True, apply_trial_filters=False)
    else:
        wcst, _ = load_wcst_trials(data_dir=data_dir, apply_trial_filters=True)

    rt_col = None
    for cand in ("reactionTimeMs", "rt_ms", "reaction_time_ms", "rt"):
        if cand in wcst.columns:
            rt_col = cand
            break

    trial_col = None
    for cand in ("trialIndex", "trial_index", "trial"):
        if cand in wcst.columns:
            trial_col = cand
            break
    if trial_col:
        wcst = wcst.sort_values(["participant_id", trial_col])

    rule_col = None
    for cand in ("ruleAtThatTime", "rule_at_that_time", "rule_at_time", "rule"):
        if cand in wcst.columns:
            rule_col = cand
            break

    return {
        "wcst": wcst,
        "rt_col": rt_col,
        "trial_col": trial_col,
        "rule_col": rule_col,
    }
