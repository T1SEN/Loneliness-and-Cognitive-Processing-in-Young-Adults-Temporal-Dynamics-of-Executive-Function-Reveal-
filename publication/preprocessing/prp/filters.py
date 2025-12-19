"""PRP QC filters (trial-level + participant-level)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set
import re

import pandas as pd

from ..constants import (
    RAW_DIR,
    PRP_RT_MIN,
    PRP_RT_MAX,
    PRP_IRI_MIN,
    PRP_ACC_THRESHOLD,
    PRP_LONG_SOA_MIN,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
)
from ..core import ensure_participant_id


@dataclass
class PRPQCCriteria:
    n_trials_required: int = 120
    rt_min: float = PRP_RT_MIN
    rt_max: float = PRP_RT_MAX
    iri_min: float = PRP_IRI_MIN
    min_acc_t1: float = PRP_ACC_THRESHOLD
    min_acc_t2_long: float = PRP_ACC_THRESHOLD
    long_soa_min: float = PRP_LONG_SOA_MIN
    require_valid_order: bool = True


def _normalize_response_order(value: object) -> Optional[str]:
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


def compute_prp_qc_stats(
    trials_df: pd.DataFrame,
    criteria: Optional[PRPQCCriteria] = None,
) -> pd.DataFrame:
    if criteria is None:
        criteria = PRPQCCriteria()

    trials_df = ensure_participant_id(trials_df)

    if "task" in trials_df.columns:
        trials_df = trials_df[trials_df["task"].astype(str).str.lower() == "prp"].copy()

    rt_col = "t2_rt_ms" if "t2_rt_ms" in trials_df.columns else (
        "t2_rt" if "t2_rt" in trials_df.columns else None
    )
    if rt_col is None:
        raise KeyError("PRP trials missing t2_rt or t2_rt_ms column.")

    soa_col = "soa_nominal_ms" if "soa_nominal_ms" in trials_df.columns else (
        "soa" if "soa" in trials_df.columns else None
    )
    if soa_col is None:
        raise KeyError("PRP trials missing soa or soa_nominal_ms column.")

    for bool_col in ["t1_timeout", "t2_timeout", "t1_correct", "t2_correct"]:
        if bool_col in trials_df.columns:
            trials_df[bool_col] = trials_df[bool_col].fillna(False).astype(bool)
        else:
            trials_df[bool_col] = False

    trials_df["rt_ms"] = pd.to_numeric(trials_df[rt_col], errors="coerce")
    trials_df["soa_ms"] = pd.to_numeric(trials_df[soa_col], errors="coerce")

    if "response_order" in trials_df.columns:
        trials_df["order_norm"] = trials_df["response_order"].apply(_normalize_response_order)
    else:
        trials_df["order_norm"] = None

    if "t2_pressed_while_t1_pending" in trials_df.columns:
        trials_df["t2_pressed_while_t1_pending"] = (
            trials_df["t2_pressed_while_t1_pending"].fillna(False).astype(bool)
        )
    else:
        trials_df["t2_pressed_while_t1_pending"] = False

    if "t1_resp_ms" in trials_df.columns and "t2_resp_ms" in trials_df.columns:
        trials_df["t1_resp_ms"] = pd.to_numeric(trials_df["t1_resp_ms"], errors="coerce")
        trials_df["t2_resp_ms"] = pd.to_numeric(trials_df["t2_resp_ms"], errors="coerce")
        trials_df["iri_ms"] = trials_df["t2_resp_ms"] - trials_df["t1_resp_ms"]
    else:
        trials_df["t1_resp_ms"] = pd.NA
        trials_df["t2_resp_ms"] = pd.NA
        trials_df["iri_ms"] = pd.NA

    non_timeout = (~trials_df["t1_timeout"]) & (~trials_df["t2_timeout"])
    order_valid = trials_df["order_norm"] == "t1_t2"
    if "t2_pressed_while_t1_pending" in trials_df.columns:
        order_valid = order_valid & (~trials_df["t2_pressed_while_t1_pending"].fillna(False).astype(bool))

    rt_valid = trials_df["rt_ms"].between(criteria.rt_min, criteria.rt_max)
    correct_both = trials_df["t1_correct"] & trials_df["t2_correct"]

    iri_applicable = non_timeout & trials_df["t1_resp_ms"].notna() & trials_df["t2_resp_ms"].notna()
    iri_valid = iri_applicable & (trials_df["iri_ms"] >= criteria.iri_min)

    valid_trial = non_timeout & correct_both & rt_valid
    if criteria.require_valid_order and "response_order" in trials_df.columns:
        valid_trial = valid_trial & order_valid
    if criteria.iri_min > 0:
        valid_trial = valid_trial & iri_valid

    n_trials = trials_df.groupby("participant_id").size().rename("n_trials")

    t1_denom = trials_df[~trials_df["t1_timeout"]].groupby("participant_id").size()
    t1_num = trials_df[~trials_df["t1_timeout"]].groupby("participant_id")["t1_correct"].sum()
    acc_t1 = (t1_num / t1_denom).rename("acc_t1")

    long_mask = (~trials_df["t2_timeout"]) & (trials_df["soa_ms"] >= criteria.long_soa_min)
    t2_long_denom = trials_df[long_mask].groupby("participant_id").size()
    t2_long_num = trials_df[long_mask].groupby("participant_id")["t2_correct"].sum()
    acc_t2_long = (t2_long_num / t2_long_denom).rename("acc_t2_long")

    valid_trial_n = valid_trial.groupby(trials_df["participant_id"]).sum().rename("valid_trial_n")

    def _count_valid(soa_min: Optional[float], soa_max: Optional[float], label: str) -> pd.Series:
        mask = valid_trial.copy()
        if soa_min is not None:
            mask = mask & (trials_df["soa_ms"] >= soa_min)
        if soa_max is not None:
            mask = mask & (trials_df["soa_ms"] <= soa_max)
        return mask.groupby(trials_df["participant_id"]).sum().rename(label)

    valid_short_n = _count_valid(None, DEFAULT_SOA_SHORT, "valid_short_n")
    valid_long_n = _count_valid(DEFAULT_SOA_LONG, None, "valid_long_n")

    order_violation_n = None
    if "response_order" in trials_df.columns:
        order_violation_n = (~order_valid).groupby(trials_df["participant_id"]).sum().rename("order_violation_n")

    iri_applicable_n = iri_applicable.groupby(trials_df["participant_id"]).sum().rename("iri_applicable_n")
    iri_violation_n = (iri_applicable & (trials_df["iri_ms"] < criteria.iri_min)).groupby(
        trials_df["participant_id"]
    ).sum().rename("iri_violation_n")

    parts = [n_trials, acc_t1, acc_t2_long, valid_trial_n, valid_short_n, valid_long_n]
    if order_violation_n is not None:
        parts.append(order_violation_n)
    parts.extend([iri_applicable_n, iri_violation_n])

    qc = pd.concat(parts, axis=1).fillna(0).reset_index()
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

    mask = (
        (qc["acc_t1"] >= criteria.min_acc_t1)
        & (qc["acc_t2_long"] >= criteria.min_acc_t2_long)
    )
    if criteria.n_trials_required > 0:
        mask = mask & (qc["n_trials"] == criteria.n_trials_required)

    valid_ids = set(qc.loc[mask, "participant_id"].astype(str))

    if verbose:
        excluded = set(qc["participant_id"].astype(str)) - valid_ids
        if excluded:
            print(f"  [INFO] PRP QC failed: {len(excluded)}")

    return valid_ids
