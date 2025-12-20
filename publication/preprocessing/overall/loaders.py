"""Overall loaders (merged summaries)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core import ensure_participant_id
from ..prp.loaders import load_prp_summary
from ..stroop.loaders import load_stroop_summary
from ..wcst.loaders import load_wcst_summary


def _load_cognitive_summary_metrics(data_dir: Path) -> pd.DataFrame:
    summary_path = data_dir / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(columns=["participant_id"])

    cognitive = pd.read_csv(summary_path, encoding="utf-8")
    cognitive = ensure_participant_id(cognitive)
    if "testName" not in cognitive.columns:
        return pd.DataFrame(columns=["participant_id"])

    cognitive["testName"] = cognitive["testName"].astype(str).str.strip().str.lower()
    metrics_by_test = {
        "stroop": ["stroop_effect"],
        "wcst": [
            "perseverativeResponses",
            "perseverativeErrorCount",
            "perseverativeResponsesPercent",
        ],
    }

    merged = None
    for test_name, columns in metrics_by_test.items():
        available = [col for col in columns if col in cognitive.columns]
        if not available:
            continue
        subset = cognitive[cognitive["testName"] == test_name][["participant_id"] + available].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("participant_id").drop_duplicates(subset=["participant_id"])
        for col in available:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
        merged = subset if merged is None else merged.merge(subset, on="participant_id", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["participant_id"])
    return merged


def load_overall_summary(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge task summaries for overall datasets.

    Returns a per-participant summary that contains PRP, Stroop, and WCST metrics.
    """
    prp_summary = load_prp_summary(data_dir)
    stroop_summary = load_stroop_summary(data_dir)
    wcst_summary = load_wcst_summary(data_dir)

    merged = prp_summary.merge(stroop_summary, on="participant_id", how="inner")
    merged = merged.merge(wcst_summary, on="participant_id", how="inner")
    cognitive_extra = _load_cognitive_summary_metrics(data_dir)
    if not cognitive_extra.empty:
        merged = merged.merge(cognitive_extra, on="participant_id", how="left")
    return merged
