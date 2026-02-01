"""Overall loaders (merged summaries)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core import ensure_participant_id


def _prepare_task_summary(df: pd.DataFrame, task: str) -> pd.DataFrame:
    subset = df[df["testName"] == task].copy()
    if subset.empty:
        return pd.DataFrame()

    subset = ensure_participant_id(subset)
    if "timestamp" in subset.columns:
        subset = subset.sort_values("timestamp")
    subset = subset.drop_duplicates(subset=["participant_id"], keep="last")

    subset = subset.drop(columns=["testName"], errors="ignore")
    rename_map = {
        col: f"{task}_{col}" for col in subset.columns if col != "participant_id"
    }
    return subset.rename(columns=rename_map)


def _add_stroop_interference(df: pd.DataFrame) -> pd.DataFrame:
    if "stroop_interference" in df.columns:
        return df
    if "stroop_mrt_incong" in df.columns and "stroop_mrt_cong" in df.columns:
        incong = pd.to_numeric(df["stroop_mrt_incong"], errors="coerce")
        cong = pd.to_numeric(df["stroop_mrt_cong"], errors="coerce")
        df["stroop_interference"] = incong - cong
        return df
    if "stroop_stroop_effect" in df.columns:
        df["stroop_interference"] = pd.to_numeric(df["stroop_stroop_effect"], errors="coerce")
        return df
    return df


def _add_wcst_perseverative_error_rate(df: pd.DataFrame) -> pd.DataFrame:
    if "wcst_perseverative_error_rate" in df.columns:
        return df
    if "wcst_perseverativeErrorCount" in df.columns and "wcst_totalTrialCount" in df.columns:
        pe = pd.to_numeric(df["wcst_perseverativeErrorCount"], errors="coerce")
        total = pd.to_numeric(df["wcst_totalTrialCount"], errors="coerce")
        rate = (pe / total) * 100
        df["wcst_perseverative_error_rate"] = rate.where(total > 0)
    return df


def load_overall_summary(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge task summaries for overall datasets.

    Returns a per-participant summary that contains Stroop and WCST metrics.
    """
    summary_path = data_dir / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(columns=["participant_id"])

    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame(columns=["participant_id"])

    if "testName" not in df.columns:
        return ensure_participant_id(df)

    df["testName"] = df["testName"].astype(str).str.lower()
    stroop_summary = _prepare_task_summary(df, "stroop")
    wcst_summary = _prepare_task_summary(df, "wcst")

    if stroop_summary.empty or wcst_summary.empty:
        return pd.DataFrame(columns=["participant_id"])

    merged = stroop_summary.merge(wcst_summary, on="participant_id", how="inner")
    merged = _add_stroop_interference(merged)
    merged = _add_wcst_perseverative_error_rate(merged)
    return merged
