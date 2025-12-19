"""PRP loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import re

import pandas as pd

from ..constants import (
    PRP_RT_MIN,
    PRP_RT_MAX,
    PRP_IRI_MIN,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
    get_results_dir,
)
from ..core import ensure_participant_id

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


def load_prp_trials(
    data_dir: Path | None = None,
    rt_min: int = PRP_RT_MIN,
    rt_max: int = PRP_RT_MAX,
    require_t1_correct: bool = True,
    enforce_short_long_only: bool = True,
    require_t2_correct_for_rt: bool = True,
    drop_timeouts: bool = True,
    require_valid_order: bool = True,
    iri_min: int = PRP_IRI_MIN,
    exclude_t2_pressed_while_pending: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if data_dir is None:
        data_dir = get_results_dir("prp")

    df = pd.read_csv(data_dir / "4a_prp_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)

    rt_col = "t2_rt_ms" if "t2_rt_ms" in df.columns else "t2_rt" if "t2_rt" in df.columns else None
    if rt_col and rt_col != "t2_rt":
        if "t2_rt" in df.columns:
            df[rt_col] = df[rt_col].fillna(df["t2_rt"])
            df = df.drop(columns=["t2_rt"])
        df = df.rename(columns={rt_col: "t2_rt"})

    soa_col = None
    for cand in ["soa_nominal_ms", "soa_ms", "soa"]:
        if cand in df.columns:
            soa_col = cand
            break
    if soa_col and soa_col != "soa":
        if "soa" in df.columns:
            df[soa_col] = df[soa_col].fillna(df["soa"])
            df = df.drop(columns=["soa"])
        df = df.rename(columns={soa_col: "soa"})
    if "t1_correct" not in df.columns:
        raise KeyError("PRP trials missing t1_correct column")
    if "t2_rt" not in df.columns or "soa" not in df.columns:
        raise KeyError("PRP trials missing t2_rt or soa column")

    for bool_col in ["t1_correct", "t2_correct", "t1_timeout", "t2_timeout"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(False).astype(bool)
    if "t2_pressed_while_t1_pending" in df.columns:
        df["t2_pressed_while_t1_pending"] = (
            df["t2_pressed_while_t1_pending"].fillna(False).astype(bool)
        )

    before = len(df)
    df = df[df["t2_rt"].between(rt_min, rt_max)]
    if drop_timeouts:
        if "t1_timeout" in df.columns:
            df = df[df["t1_timeout"] == False]
        if "t2_timeout" in df.columns:
            df = df[df["t2_timeout"] == False]
    if require_t1_correct:
        df = df[df["t1_correct"] == True]
    if require_t2_correct_for_rt and "t2_correct" in df.columns:
        df = df[df["t2_correct"] == True]

    if require_valid_order and "response_order" in df.columns:
        df["order_norm"] = df["response_order"].apply(_normalize_response_order)
        df = df[df["order_norm"] == "t1_t2"]
        if exclude_t2_pressed_while_pending and "t2_pressed_while_t1_pending" in df.columns:
            df = df[df["t2_pressed_while_t1_pending"] == False]

    if iri_min > 0 and "t1_resp_ms" in df.columns and "t2_resp_ms" in df.columns:
        df["t1_resp_ms"] = pd.to_numeric(df["t1_resp_ms"], errors="coerce")
        df["t2_resp_ms"] = pd.to_numeric(df["t2_resp_ms"], errors="coerce")
        df["iri_ms"] = df["t2_resp_ms"] - df["t1_resp_ms"]
        iri_valid = df["iri_ms"] >= iri_min
        df = df[iri_valid]

    def bin_soa(soa_val):
        if soa_val <= DEFAULT_SOA_SHORT:
            return "short"
        if soa_val >= DEFAULT_SOA_LONG:
            return "long"
        return "other"

    df["soa_bin"] = df["soa"].apply(bin_soa)
    if enforce_short_long_only:
        df = df[df["soa_bin"].isin(["short", "long"])]

    summary = {
        "rows_before": before,
        "rows_after": len(df),
        "n_participants": df["participant_id"].nunique(),
        "rt_min": rt_min,
        "rt_max": rt_max,
    }
    return df, summary


def load_prp_summary(data_dir: Path) -> pd.DataFrame:
    prp_trials = pd.read_csv(data_dir / "4a_prp_trials.csv", encoding="utf-8")
    prp_trials = ensure_participant_id(prp_trials)

    rt_col = "t2_rt_ms" if "t2_rt_ms" in prp_trials.columns else ("t2_rt" if "t2_rt" in prp_trials.columns else None)
    if rt_col is None:
        raise KeyError("PRP trials missing T2 RT column ('t2_rt' or 't2_rt_ms').")
    if rt_col != "t2_rt":
        if "t2_rt" in prp_trials.columns:
            prp_trials[rt_col] = prp_trials[rt_col].fillna(prp_trials["t2_rt"])
            prp_trials = prp_trials.drop(columns=["t2_rt"])
        prp_trials = prp_trials.rename(columns={rt_col: "t2_rt"})

    soa_col = None
    for cand in ["soa_nominal_ms", "soa_ms", "soa"]:
        if cand in prp_trials.columns:
            soa_col = cand
            break
    if soa_col is None:
        raise KeyError("PRP trials missing SOA column ('soa', 'soa_ms', or 'soa_nominal_ms').")
    if soa_col != "soa":
        if "soa" in prp_trials.columns:
            prp_trials[soa_col] = prp_trials[soa_col].fillna(prp_trials["soa"])
            prp_trials = prp_trials.drop(columns=["soa"])
        prp_trials = prp_trials.rename(columns={soa_col: "soa"})

    prp_trials["t1_correct"] = prp_trials["t1_correct"].fillna(False).astype(bool)
    if "t2_correct" in prp_trials.columns:
        prp_trials["t2_correct"] = prp_trials["t2_correct"].fillna(False).astype(bool)
    else:
        prp_trials["t2_correct"] = True
    prp_trials["t1_timeout"] = prp_trials.get("t1_timeout", False)
    if isinstance(prp_trials["t1_timeout"], pd.Series):
        prp_trials["t1_timeout"] = prp_trials["t1_timeout"].fillna(False).astype(bool)
    prp_trials["t2_timeout"] = prp_trials.get("t2_timeout", False)
    if isinstance(prp_trials["t2_timeout"], pd.Series):
        prp_trials["t2_timeout"] = prp_trials["t2_timeout"].fillna(False).astype(bool)
    if "t2_pressed_while_t1_pending" in prp_trials.columns:
        prp_trials["t2_pressed_while_t1_pending"] = (
            prp_trials["t2_pressed_while_t1_pending"].fillna(False).astype(bool)
        )

    if "response_order" in prp_trials.columns:
        prp_trials["order_norm"] = prp_trials["response_order"].apply(_normalize_response_order)
        prp_trials = prp_trials[prp_trials["order_norm"] == "t1_t2"]
        if "t2_pressed_while_t1_pending" in prp_trials.columns:
            prp_trials = prp_trials[prp_trials["t2_pressed_while_t1_pending"] == False]

    if "t1_resp_ms" in prp_trials.columns and "t2_resp_ms" in prp_trials.columns:
        prp_trials["t1_resp_ms"] = pd.to_numeric(prp_trials["t1_resp_ms"], errors="coerce")
        prp_trials["t2_resp_ms"] = pd.to_numeric(prp_trials["t2_resp_ms"], errors="coerce")
        prp_trials["iri_ms"] = prp_trials["t2_resp_ms"] - prp_trials["t1_resp_ms"]

    prp_rt = prp_trials[
        (prp_trials["t1_correct"] == True)
        & (prp_trials["t2_correct"] == True)
        & (prp_trials["t1_timeout"] == False)
        & (prp_trials["t2_timeout"] == False)
        & (prp_trials["t2_rt"] >= PRP_RT_MIN)
        & (prp_trials["t2_rt"] <= PRP_RT_MAX)
    ].copy()

    if "iri_ms" in prp_rt.columns:
        prp_rt = prp_rt[prp_rt["iri_ms"] >= PRP_IRI_MIN]

    def bin_soa(soa):
        if soa <= DEFAULT_SOA_SHORT:
            return "short"
        if soa >= DEFAULT_SOA_LONG:
            return "long"
        return "other"

    prp_rt["soa_bin"] = prp_rt["soa"].apply(bin_soa)
    prp_rt = prp_rt[prp_rt["soa_bin"].isin(["short", "long"])].copy()

    prp_summary = prp_rt.groupby(["participant_id", "soa_bin"]).agg(
        t2_rt_mean=("t2_rt", "mean"),
        t2_rt_sd=("t2_rt", "std"),
        n_trials=("t2_rt", "count"),
    ).reset_index()

    prp_wide = prp_summary.pivot(index="participant_id", columns="soa_bin", values=["t2_rt_mean", "t2_rt_sd"])
    prp_wide.columns = ["_".join(col).rstrip("_") for col in prp_wide.columns.values]
    prp_wide = prp_wide.reset_index()

    if "t2_rt_mean_short" in prp_wide.columns and "t2_rt_mean_long" in prp_wide.columns:
        prp_wide["prp_bottleneck"] = prp_wide["t2_rt_mean_short"] - prp_wide["t2_rt_mean_long"]

    return prp_wide
