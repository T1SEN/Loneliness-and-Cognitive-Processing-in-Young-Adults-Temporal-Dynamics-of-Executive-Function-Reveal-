"""PRP QC filters (trial-level + participant-level)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set
import re

import pandas as pd

from ..constants import (
    RAW_DIR,
    PRP_LONG_SOA_MIN,
)
from ..core import ensure_participant_id


@dataclass
class PRPQCCriteria:
    min_both_correct_rate: float = 0.50
    n_trials_required: int = 120
    long_soa_min: float = PRP_LONG_SOA_MIN


def _normalize_response_order(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    token = re.sub(r"[^a-z0-9]", "", cleaned)
    if token.startswith("t1t2"):
        return "t1_t2"
    if token.startswith("t2t1"):
        return "t2_t1"
    if token in {"t1only", "t2only", "none"}:
        return token
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


def compute_prp_qc_stats(
    trials_df: pd.DataFrame,
    criteria: Optional[PRPQCCriteria] = None,
) -> pd.DataFrame:
    if criteria is None:
        criteria = PRPQCCriteria()

    trials_df = ensure_participant_id(trials_df)

    if "task" in trials_df.columns:
        trials_df = trials_df[trials_df["task"].astype(str).str.lower() == "prp"].copy()

    soa_col = "soa_nominal_ms" if "soa_nominal_ms" in trials_df.columns else (
        "soa" if "soa" in trials_df.columns else None
    )
    if soa_col is None:
        raise KeyError("PRP trials missing soa or soa_nominal_ms column.")

    for bool_col in ["t1_timeout", "t2_timeout", "t1_correct", "t2_correct"]:
        if bool_col in trials_df.columns:
            trials_df[bool_col] = _coerce_bool(trials_df[bool_col])
        else:
            trials_df[bool_col] = False

    trials_df["soa_ms"] = pd.to_numeric(trials_df[soa_col], errors="coerce")

    if "response_order" in trials_df.columns:
        trials_df["order_norm"] = trials_df["response_order"].apply(_normalize_response_order)
        trials_df["is_order_valid"] = trials_df["order_norm"] == "t1_t2"
    else:
        trials_df["order_norm"] = None
        trials_df["is_order_valid"] = True

    if "t2_pressed_while_t1_pending" in trials_df.columns:
        trials_df["t2_pressed_while_t1_pending"] = _coerce_bool(
            trials_df["t2_pressed_while_t1_pending"]
        )
        trials_df["is_pending_valid"] = ~trials_df["t2_pressed_while_t1_pending"]
    else:
        trials_df["t2_pressed_while_t1_pending"] = False
        trials_df["is_pending_valid"] = True

    trials_df["is_structurally_valid"] = (
        trials_df["is_order_valid"] & trials_df["is_pending_valid"]
    )

    n_trials = trials_df.groupby("participant_id").size().rename("n_trials")

    t1_denom = trials_df[~trials_df["t1_timeout"]].groupby("participant_id").size()
    t1_num = trials_df[~trials_df["t1_timeout"]].groupby("participant_id")["t1_correct"].sum()
    acc_t1 = (t1_num / t1_denom).rename("acc_t1")

    long_mask = (~trials_df["t2_timeout"]) & (trials_df["soa_ms"] >= criteria.long_soa_min)
    t2_long_denom = trials_df[long_mask].groupby("participant_id").size()
    t2_long_num = trials_df[long_mask].groupby("participant_id")["t2_correct"].sum()
    acc_t2_long = (t2_long_num / t2_long_denom).rename("acc_t2_long")

    trials_df["both_correct"] = (
        trials_df["t1_correct"]
        & trials_df["t2_correct"]
        & (~trials_df["t1_timeout"])
        & (~trials_df["t2_timeout"])
        & trials_df["is_structurally_valid"]
    )
    both_correct_rate = (
        trials_df.groupby("participant_id")["both_correct"]
        .mean()
        .rename("both_correct_rate")
    )

    qc = (
        pd.concat([n_trials, acc_t1, acc_t2_long, both_correct_rate], axis=1)
        .fillna(0)
        .reset_index()
    )
    return qc


def get_prp_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[PRPQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = PRPQCCriteria()

    trials_path = data_dir / "4a_prp_trials.csv"
    if not trials_path.exists():
        if verbose:
            print(f"[WARN] PRP trials file not found: {trials_path}")
        return set()

    trials = pd.read_csv(trials_path, encoding="utf-8-sig")
    qc = compute_prp_qc_stats(trials, criteria)

    mask = qc["both_correct_rate"] >= criteria.min_both_correct_rate
    if criteria.n_trials_required > 0:
        mask = mask & (qc["n_trials"] == criteria.n_trials_required)

    valid_ids = set(qc.loc[mask, "participant_id"].astype(str))

    if verbose:
        excluded = set(qc["participant_id"].astype(str)) - valid_ids
        if excluded:
            print(f"  [INFO] PRP QC failed: {len(excluded)}")

    return valid_ids
