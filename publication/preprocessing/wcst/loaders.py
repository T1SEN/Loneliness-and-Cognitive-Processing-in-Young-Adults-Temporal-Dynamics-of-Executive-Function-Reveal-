"""WCST loaders."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from ..constants import WCST_RT_MIN, get_results_dir
from .filters import clean_wcst_trials, filter_wcst_rt_trials
from ..core import ensure_participant_id

def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    mapped = text.map(mapping)
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_bool = numeric.map(lambda v: bool(int(v)) if pd.notna(v) else None)
        mapped = mapped.fillna(numeric_bool)
    return mapped.fillna(False).astype(bool)


def load_wcst_trials(
    data_dir: Path | None = None,
    clean: bool = True,
    filter_rt: bool = False,
    apply_trial_filters: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if data_dir is None:
        data_dir = get_results_dir("wcst")

    df = pd.read_csv(data_dir / "4b_wcst_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)
    summary = {
        "rows_before": len(df),
        "rows_after": len(df),
        "n_participants": df["participant_id"].nunique(),
    }

    if apply_trial_filters:
        df, clean_stats = clean_wcst_trials(df)
        summary.update(clean_stats)
        df["rt_ms"] = pd.to_numeric(df["rt_ms"], errors="coerce")
        df = df[df["rt_ms"].notna()]
        df = df[df["rt_ms"] >= WCST_RT_MIN]
        summary["rows_after"] = len(df)
    else:
        if clean:
            df, clean_stats = clean_wcst_trials(df)
            summary.update(clean_stats)
            summary["rows_after"] = len(df)

        if filter_rt:
            before_rt = len(df)
            df = filter_wcst_rt_trials(df)
            summary["rt_filtered"] = before_rt - len(df)
            summary["rows_after"] = len(df)

    if "correct" in df.columns:
        df["correct"] = _coerce_bool_series(df["correct"])

    if "isPE" not in df.columns:
        def parse_extra(extra_str):
            if not isinstance(extra_str, str):
                return {}
            try:
                return ast.literal_eval(extra_str)
            except Exception:
                return {}
        df["extra_dict"] = df["extra"].apply(parse_extra) if "extra" in df.columns else {}
        df["isPE"] = df.get("extra_dict", {}).apply(lambda x: x.get("isPE", False) if isinstance(x, dict) else False)

    return df, summary


def load_wcst_summary(data_dir: Path) -> pd.DataFrame:
    wcst_trials = pd.read_csv(data_dir / "4b_wcst_trials.csv", encoding="utf-8")
    wcst_trials = ensure_participant_id(wcst_trials)
    wcst_trials, _ = clean_wcst_trials(wcst_trials)

    def _parse_wcst_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except (ValueError, SyntaxError):
            return {}

    wcst_trials["extra_dict"] = wcst_trials["extra"].apply(_parse_wcst_extra)
    wcst_trials["is_pe"] = wcst_trials["extra_dict"].apply(lambda x: x.get("isPE", False))

    wcst_trials["rt_ms"] = pd.to_numeric(wcst_trials["rt_ms"], errors="coerce")
    wcst_trials = wcst_trials[wcst_trials["rt_ms"].notna()]
    wcst_trials = wcst_trials[wcst_trials["rt_ms"] >= WCST_RT_MIN]

    wcst_trials["correct"] = _coerce_bool_series(wcst_trials["correct"])
    wcst_summary = wcst_trials.groupby("participant_id").agg(
        pe_count=("is_pe", "sum"),
        total_trials=("is_pe", "count"),
        wcst_accuracy=("correct", lambda x: (x.sum() / len(x)) * 100),
    ).reset_index()
    rt_summary = wcst_trials.groupby("participant_id").agg(
        wcst_mean_rt=("rt_ms", "mean"),
        wcst_sd_rt=("rt_ms", "std"),
    ).reset_index()
    wcst_summary = wcst_summary.merge(rt_summary, on="participant_id", how="left")
    wcst_summary["pe_rate"] = (wcst_summary["pe_count"] / wcst_summary["total_trials"]) * 100

    return wcst_summary
