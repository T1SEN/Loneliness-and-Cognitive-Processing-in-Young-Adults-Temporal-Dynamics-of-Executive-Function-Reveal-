"""Stroop loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from ..constants import (
    STROOP_RT_MIN,
    STROOP_RT_MAX,
    get_results_dir,
)
from ..core import ensure_participant_id


def load_stroop_trials(
    data_dir: Path | None = None,
    rt_min: int = STROOP_RT_MIN,
    rt_max: int = STROOP_RT_MAX,
    require_correct_for_rt: bool = True,
    drop_timeouts: bool = True,
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

    cond_col = None
    for cand in ["type", "condition", "cond"]:
        if cand in df.columns:
            cond_col = cand
            break
    if cond_col is None:
        raise KeyError("Stroop trials missing condition column")

    for bool_col in ["correct", "timeout"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(False).astype(bool)

    before = len(df)
    if drop_timeouts and "timeout" in df.columns:
        df = df[df["timeout"] == False]
    if require_correct_for_rt and "correct" in df.columns:
        df = df[df["correct"] == True]
    df = df[df["rt"].between(rt_min, rt_max)]

    summary = {
        "rows_before": before,
        "rows_after": len(df),
        "n_participants": df["participant_id"].nunique(),
        "rt_min": rt_min,
        "rt_max": rt_max,
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

    stroop_trials["correct"] = stroop_trials["correct"].fillna(False).astype(bool)
    if "timeout" in stroop_trials.columns:
        stroop_trials["timeout"] = stroop_trials["timeout"].fillna(False).astype(bool)
    acc_summary = stroop_trials.groupby(["participant_id", cond_col]).agg(
        accuracy=("correct", "mean")
    ).reset_index()

    rt_trials = stroop_trials[
        ((stroop_trials["timeout"] == False) if "timeout" in stroop_trials.columns else True)
        & (stroop_trials["correct"] == True)
        & (stroop_trials[rt_col] >= STROOP_RT_MIN)
        & (stroop_trials[rt_col] <= STROOP_RT_MAX)
    ].copy()

    rt_summary = rt_trials.groupby(["participant_id", cond_col]).agg(
        rt_mean=(rt_col, "mean")
    ).reset_index()

    stroop_summary = acc_summary.merge(rt_summary, on=["participant_id", cond_col], how="left")

    stroop_wide = stroop_summary.pivot(index="participant_id", columns=cond_col, values=["rt_mean", "accuracy"])
    stroop_wide.columns = ["_".join(col).rstrip("_") for col in stroop_wide.columns.values]
    stroop_wide = stroop_wide.reset_index()

    if "rt_mean_incongruent" in stroop_wide.columns and "rt_mean_congruent" in stroop_wide.columns:
        stroop_wide["stroop_interference"] = stroop_wide["rt_mean_incongruent"] - stroop_wide["rt_mean_congruent"]

    return stroop_wide
