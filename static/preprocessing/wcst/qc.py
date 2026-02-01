"""
WCST trial cleaning + QC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd

from ..constants import (
    WCST_RT_MIN,
    WCST_RT_MAX,
    WCST_VALID_CONDS,
    WCST_VALID_CARDS,
    WCST_MIN_TRIALS,
    WCST_MAX_SINGLE_CHOICE,
)
from ..core import ensure_participant_id


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


def clean_wcst_trials(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_participant_id(df.copy())

    rule_col = _pick_column(df, ["ruleAtThatTime", "rule_at_that_time", "rule"])
    trial_col = _pick_column(df, ["trialIndex", "trial_index", "trial"])
    chosen_col = _pick_column(df, ["chosenCard", "chosen_card"])
    rt_col = _pick_column(df, ["rt_ms", "reactionTimeMs", "resp_time_ms"])
    timeout_col = _pick_column(df, ["timeout"])
    correct_col = _pick_column(df, ["correct"])
    ispe_col = _pick_column(df, ["isPE"])
    isnpe_col = _pick_column(df, ["isNPE"])

    if rule_col is not None:
        df["rule"] = df[rule_col].astype(str).str.lower().replace({"color": "colour"})
    else:
        df["rule"] = pd.NA

    if chosen_col is not None:
        df["chosen_card"] = df[chosen_col].astype(str)
    else:
        df["chosen_card"] = pd.NA

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

    if ispe_col is not None:
        df["isPE"] = _coerce_bool_series(df[ispe_col]).astype(bool)
    else:
        df["isPE"] = False

    if isnpe_col is not None:
        df["isNPE"] = _coerce_bool_series(df[isnpe_col]).astype(bool)
    else:
        df["isNPE"] = False

    # Timeouts are treated as incorrect
    df["correct"] = df["correct"] & (~df["timeout"])

    valid_rule = df["rule"].isin(WCST_VALID_CONDS)
    valid_card = df["chosen_card"].isin(WCST_VALID_CARDS)
    df = df[valid_rule & valid_card].copy()

    # Remove anticipatory RTs (< 200 ms) but keep missing RTs for non-RT metrics
    df = df[df["rt_ms"].isna() | (df["rt_ms"] >= WCST_RT_MIN)]
    df["is_rt_valid"] = df["rt_ms"].between(WCST_RT_MIN, WCST_RT_MAX)
    return df


def prepare_wcst_trials(data_dir: Path) -> pd.DataFrame:
    trials_path = data_dir / "4b_wcst_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame(columns=["participant_id"])
    df = pd.read_csv(trials_path, encoding="utf-8-sig")
    return clean_wcst_trials(df)


def compute_wcst_qc_ids(
    wcst_df: pd.DataFrame,
    min_trials: int = WCST_MIN_TRIALS,
    max_single_choice: float = WCST_MAX_SINGLE_CHOICE,
) -> Set[str]:
    if wcst_df.empty:
        return set()
    counts = wcst_df.groupby("participant_id").size()
    choice_ratio = (
        wcst_df.groupby("participant_id")["chosen_card"]
        .value_counts(normalize=True)
        .groupby(level=0)
        .max()
    )
    choice_ratio = choice_ratio.reindex(counts.index).fillna(1.0)
    valid = (counts >= min_trials) & (choice_ratio <= max_single_choice)
    return set(valid[valid].index.astype(str))
