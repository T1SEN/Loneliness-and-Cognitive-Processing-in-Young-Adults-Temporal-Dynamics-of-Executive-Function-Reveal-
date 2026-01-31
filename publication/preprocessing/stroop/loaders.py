"""Stroop loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np

from ..constants import (
    STROOP_RT_MIN,
    STROOP_RT_MAX,
    get_results_dir,
)
from ..core import ensure_participant_id


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    })
    return mapped.fillna(False).astype(bool)


def load_stroop_trials(
    data_dir: Path | None = None,
    rt_min: int = STROOP_RT_MIN,
    rt_max: int | None = STROOP_RT_MAX,
    require_correct_for_rt: bool = False,
    drop_timeouts: bool = True,
    apply_trial_filters: bool = True,
    exclude_timeout: bool | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if data_dir is None:
        data_dir = get_results_dir("stroop")

    df = pd.read_csv(data_dir / "4c_stroop_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)

    rt_col = "rt_ms" if "rt_ms" in df.columns else "rt" if "rt" in df.columns else None
    if not rt_col:
        raise KeyError("Stroop trials missing rt/rt_ms column")
    if rt_col != "rt":
        if "rt" in df.columns:
            df[rt_col] = df[rt_col].fillna(df["rt"])
            df = df.drop(columns=["rt"])
        df = df.rename(columns={rt_col: "rt"})
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    cond_col = None
    for cand in ["type", "condition", "cond"]:
        if cand in df.columns:
            cond_col = cand
            break
    if cond_col is None:
        raise KeyError("Stroop trials missing condition column")

    if "timeout" in df.columns:
        df["timeout"] = _coerce_bool(df["timeout"])
    else:
        df["timeout"] = False

    if "correct" in df.columns:
        df["correct"] = _coerce_bool(df["correct"])
    else:
        df["correct"] = False

    df["correct"] = df["correct"] & (~df["timeout"])
    df["is_timeout"] = df["timeout"]

    before = len(df)
    if exclude_timeout is None:
        exclude_timeout = drop_timeouts
    if apply_trial_filters:
        df["is_rt_valid"] = df["rt"].between(rt_min, rt_max) if rt_max is not None else (df["rt"] >= rt_min)
        if exclude_timeout:
            df = df[df["is_timeout"] == False]
            df = df[df["is_rt_valid"]]
        else:
            df = df[df["is_rt_valid"] | (df["is_timeout"] == True)]
        if require_correct_for_rt:
            df = df[df["correct"] == True]
    else:
        df["is_rt_valid"] = df["rt"].between(rt_min, rt_max) if rt_max is not None else (df["rt"] >= rt_min)

    summary = {
        "rows_before": before,
        "rows_after": len(df),
        "n_participants": df["participant_id"].nunique(),
        "rt_min": rt_min,
        "rt_max": rt_max,
        "filters_applied": apply_trial_filters,
    }
    return df, summary


def load_stroop_summary(data_dir: Path) -> pd.DataFrame:
    stroop_trials = pd.read_csv(data_dir / "4c_stroop_trials.csv", encoding="utf-8")
    stroop_trials = ensure_participant_id(stroop_trials)

    rt_col = "rt_ms" if "rt_ms" in stroop_trials.columns else ("rt" if "rt" in stroop_trials.columns else None)
    if rt_col is None:
        raise KeyError("Stroop trials missing RT column ('rt' or 'rt_ms').")
    if rt_col != "rt":
        if "rt" in stroop_trials.columns:
            stroop_trials[rt_col] = stroop_trials[rt_col].fillna(stroop_trials["rt"])
            stroop_trials = stroop_trials.drop(columns=["rt"])
        stroop_trials = stroop_trials.rename(columns={rt_col: "rt"})
        rt_col = "rt"

    cond_col = "type" if "type" in stroop_trials.columns else ("condition" if "condition" in stroop_trials.columns else ("cond" if "cond" in stroop_trials.columns else None))
    if cond_col is None:
        raise KeyError("Stroop trials missing condition column ('type', 'condition', or 'cond').")

    stroop_trials["correct"] = _coerce_bool(stroop_trials["correct"])
    if "timeout" in stroop_trials.columns:
        stroop_trials["timeout"] = _coerce_bool(stroop_trials["timeout"])
    else:
        stroop_trials["timeout"] = False
    stroop_trials["correct"] = stroop_trials["correct"] & (~stroop_trials["timeout"])
    stroop_trials[rt_col] = pd.to_numeric(stroop_trials[rt_col], errors="coerce")
    acc_summary = stroop_trials.groupby(["participant_id", cond_col]).agg(
        accuracy=("correct", "mean")
    ).reset_index()

    rt_trials = stroop_trials.copy()
    if "timeout" in rt_trials.columns:
        rt_trials = rt_trials[rt_trials["timeout"] == False]
    rt_trials = rt_trials[rt_trials[rt_col].notna()]
    rt_trials = rt_trials[rt_trials[rt_col].between(STROOP_RT_MIN, STROOP_RT_MAX)]

    rt_summary = rt_trials.groupby(["participant_id", cond_col]).agg(
        rt_mean=(rt_col, "mean")
    ).reset_index()

    rt_trials_correct = rt_trials[rt_trials["correct"] == True]
    rt_summary_correct = rt_trials_correct.groupby(["participant_id", cond_col]).agg(
        rt_mean_correct=(rt_col, "mean")
    ).reset_index()

    stroop_summary = acc_summary.merge(rt_summary, on=["participant_id", cond_col], how="left")
    stroop_summary = stroop_summary.merge(rt_summary_correct, on=["participant_id", cond_col], how="left")

    stroop_wide = stroop_summary.pivot(
        index="participant_id", columns=cond_col, values=["rt_mean", "rt_mean_correct", "accuracy"]
    )
    stroop_wide.columns = ["_".join(col).rstrip("_") for col in stroop_wide.columns.values]
    stroop_wide = stroop_wide.reset_index()

    interference = np.nan
    if (
        "rt_mean_correct_incongruent" in stroop_wide.columns
        and "rt_mean_correct_congruent" in stroop_wide.columns
    ):
        interference = (
            stroop_wide["rt_mean_correct_incongruent"] - stroop_wide["rt_mean_correct_congruent"]
        )
    elif "rt_mean_incongruent" in stroop_wide.columns and "rt_mean_congruent" in stroop_wide.columns:
        interference = stroop_wide["rt_mean_incongruent"] - stroop_wide["rt_mean_congruent"]

    out = pd.DataFrame({
        "participant_id": stroop_wide["participant_id"].astype(str),
        "stroop_interference": interference,
    })
    return out
