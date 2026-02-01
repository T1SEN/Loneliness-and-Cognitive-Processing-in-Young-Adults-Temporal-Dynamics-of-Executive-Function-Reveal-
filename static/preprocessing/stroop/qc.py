"""
Stroop trial cleaning + QC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd

from ..constants import STROOP_RT_MIN, STROOP_RT_MAX
from ..core import ensure_participant_id

STROOP_REQUIRED_TRIALS = 108
STROOP_MIN_ACCURACY = 0.70


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )
    return mapped.fillna(False)


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def clean_stroop_trials(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_participant_id(df.copy())

    cond_col = _pick_column(df, ["cond", "type"])
    trial_col = _pick_column(df, ["trial", "trial_index", "trialIndex"])
    rt_col = _pick_column(df, ["rt_ms", "rt", "resp_time_ms"])
    timeout_col = _pick_column(df, ["timeout", "is_timeout"])
    correct_col = _pick_column(df, ["correct"])

    if cond_col is not None:
        df["cond"] = df[cond_col].astype(str).str.lower()
        df = df[df["cond"].isin({"congruent", "incongruent", "neutral"})]
    else:
        df["cond"] = pd.NA

    if trial_col is not None:
        df["trial_order"] = pd.to_numeric(df[trial_col], errors="coerce")
    else:
        df["trial_order"] = np.nan

    if rt_col is not None:
        df["rt_ms"] = pd.to_numeric(df[rt_col], errors="coerce")
    else:
        df["rt_ms"] = np.nan

    if timeout_col is not None:
        df["timeout"] = _coerce_bool_series(df[timeout_col]).astype(bool)
    else:
        df["timeout"] = False

    if correct_col is not None:
        df["correct"] = _coerce_bool_series(df[correct_col]).astype(bool)
    else:
        df["correct"] = False

    # Timeouts are treated as incorrect
    df["correct"] = df["correct"] & (~df["timeout"])
    df["is_rt_valid"] = df["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
    return df


def prepare_stroop_trials(data_dir: Path) -> pd.DataFrame:
    trials_path = data_dir / "4a_stroop_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame(columns=["participant_id"])
    df = pd.read_csv(trials_path, encoding="utf-8-sig")
    return clean_stroop_trials(df)


def compute_stroop_qc_ids(
    stroop_df: pd.DataFrame,
    min_trials: int = STROOP_REQUIRED_TRIALS,
    min_accuracy: float = STROOP_MIN_ACCURACY,
) -> Set[str]:
    if stroop_df.empty:
        return set()
    counts = stroop_df.groupby("participant_id").size()
    acc = stroop_df.groupby("participant_id")["correct"].mean()
    valid = (counts >= min_trials) & (acc >= min_accuracy)
    return set(valid[valid].index.astype(str))
