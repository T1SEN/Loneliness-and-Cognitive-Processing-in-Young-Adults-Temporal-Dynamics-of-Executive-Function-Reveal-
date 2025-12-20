"""Stroop QC filters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ..constants import RAW_DIR, STROOP_RT_MIN, STROOP_RT_MAX
from ..core import ensure_participant_id


@dataclass
class StroopQCCriteria:
    n_trials: int = 108
    max_timeout_rate: float = 0.15
    min_accuracy: float = 0.80
    min_valid_per_condition: int = 20
    rt_min: float = STROOP_RT_MIN
    rt_max: float = STROOP_RT_MAX


def _normalize_condition(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    mapping = {
        "cong": "congruent",
        "congruent": "congruent",
        "inc": "incongruent",
        "incong": "incongruent",
        "incongruent": "incongruent",
        "neutral": "neutral",
        "neut": "neutral",
        "neu": "neutral",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if "incong" in cleaned:
        return "incongruent"
    if "cong" in cleaned:
        return "congruent"
    if "neut" in cleaned:
        return "neutral"
    return None


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


def compute_stroop_qc_stats(
    trials_df: pd.DataFrame,
    rt_min: float = STROOP_RT_MIN,
    rt_max: float = STROOP_RT_MAX,
) -> pd.DataFrame:
    trials_df = ensure_participant_id(trials_df)

    cond_col = None
    for cand in ("type", "condition", "cond", "congruency"):
        if cand in trials_df.columns:
            cond_col = cand
            break
    if cond_col is None:
        raise KeyError("Stroop trials missing condition column.")

    trials_df["condition_norm"] = trials_df[cond_col].apply(_normalize_condition)

    timeout = trials_df.get("timeout")
    if isinstance(timeout, pd.Series):
        timeout = _coerce_bool(timeout)
    else:
        timeout = pd.Series(False, index=trials_df.index)
    trials_df["timeout"] = timeout

    correct = trials_df.get("correct")
    if isinstance(correct, pd.Series):
        correct = _coerce_bool(correct)
    else:
        correct = pd.Series(False, index=trials_df.index)
    trials_df["correct"] = correct & (~trials_df["timeout"])

    rt_col = None
    for cand in ("rt_ms", "rt", "reactionTimeMs"):
        if cand in trials_df.columns:
            rt_col = cand
            break
    if rt_col is None:
        raise KeyError("Stroop trials missing RT column.")

    trials_df["rt_ms"] = pd.to_numeric(trials_df[rt_col], errors="coerce")

    valid_rt = trials_df["rt_ms"].between(rt_min, rt_max)
    valid_trial = (~trials_df["timeout"]) & (trials_df["correct"]) & valid_rt

    base_counts = trials_df.groupby("participant_id").size().rename("n_trials")
    timeout_rate = trials_df.groupby("participant_id")["timeout"].mean().rename("timeout_rate")
    accuracy = trials_df.groupby("participant_id")["correct"].mean().rename("accuracy_total")

    def _count_valid(condition: str) -> pd.Series:
        mask = trials_df["condition_norm"] == condition
        return valid_trial[mask].groupby(trials_df.loc[mask, "participant_id"]).sum()

    valid_cong = _count_valid("congruent").rename("valid_cong_n")
    valid_incong = _count_valid("incongruent").rename("valid_incong_n")
    valid_neutral = _count_valid("neutral").rename("valid_neutral_n")

    qc = (
        pd.concat([base_counts, timeout_rate, accuracy, valid_cong, valid_incong, valid_neutral], axis=1)
        .fillna(0)
        .reset_index()
    )
    return qc


def get_stroop_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[StroopQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = StroopQCCriteria()

    trials_path = data_dir / "4c_stroop_trials.csv"
    if not trials_path.exists():
        if verbose:
            print(f"[WARN] Stroop trials file not found: {trials_path}")
        return set()

    trials = pd.read_csv(trials_path, encoding="utf-8-sig")
    qc = compute_stroop_qc_stats(trials, rt_min=criteria.rt_min, rt_max=criteria.rt_max)

    mask = (
        (qc["n_trials"] == criteria.n_trials)
        & (qc["timeout_rate"] <= criteria.max_timeout_rate)
        & (qc["accuracy_total"] >= criteria.min_accuracy)
        & (qc["valid_cong_n"] >= criteria.min_valid_per_condition)
        & (qc["valid_incong_n"] >= criteria.min_valid_per_condition)
        & (qc["valid_neutral_n"] >= criteria.min_valid_per_condition)
    )
    valid_ids = set(qc.loc[mask, "participant_id"].unique())

    if verbose:
        excluded = set(qc["participant_id"].unique()) - valid_ids
        if excluded:
            print(f"  [INFO] Stroop QC failed: {len(excluded)}")

    return valid_ids
