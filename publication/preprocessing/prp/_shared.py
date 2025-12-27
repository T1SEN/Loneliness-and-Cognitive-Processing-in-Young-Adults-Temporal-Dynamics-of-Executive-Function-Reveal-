"""Shared helpers for PRP feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..constants import PRP_RT_MIN
from .trial_level_loaders import load_prp_trials


def _run_lengths(mask: pd.Series) -> List[int]:
    lengths: List[int] = []
    count = 0
    for val in mask:
        if val:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def _apply_timeout_as_incorrect(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "t1_timeout" in df.columns and "t1_correct" in df.columns:
        df.loc[df["t1_timeout"] == True, "t1_correct"] = False
    if "t2_timeout" in df.columns and "t2_correct" in df.columns:
        df.loc[df["t2_timeout"] == True, "t2_correct"] = False
    return df


def prepare_prp_trials(
    rt_min: float = PRP_RT_MIN,
    rt_max: float | None = None,
    data_dir: Path | None = None,
) -> Dict[str, object]:
    prp_rt, _ = load_prp_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=True,
        require_t1_correct=True,
        require_t2_correct_for_rt=True,
    )
    prp_raw, _ = load_prp_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=False,
    )

    prp = prp_rt

    prp["t2_rt_ms"] = pd.to_numeric(prp.get("t2_rt", prp.get("t2_rt_ms")), errors="coerce")
    prp["soa_ms"] = pd.to_numeric(prp.get("soa", prp.get("soa_nominal_ms")), errors="coerce")
    prp["t1_rt_ms"] = pd.to_numeric(prp.get("t1_rt_ms", prp.get("t1_rt")), errors="coerce")

    prp_raw["t2_rt_ms"] = pd.to_numeric(prp_raw.get("t2_rt", prp_raw.get("t2_rt_ms")), errors="coerce")
    prp_raw["soa_ms"] = pd.to_numeric(prp_raw.get("soa", prp_raw.get("soa_nominal_ms")), errors="coerce")
    prp_raw["t1_rt_ms"] = pd.to_numeric(prp_raw.get("t1_rt_ms", prp_raw.get("t1_rt")), errors="coerce")

    for col in ["t1_correct", "t2_correct"]:
        if col in prp.columns and prp[col].dtype == "object":
            prp[col] = prp[col].map({
                True: True, "True": True, 1: True,
                False: False, "False": False, 0: False,
            })

    if "t1_correct" in prp_raw.columns and prp_raw["t1_correct"].dtype == "object":
        prp_raw["t1_correct"] = prp_raw["t1_correct"].map({
            True: True, "True": True, 1: True,
            False: False, "False": False, 0: False,
        })
    if "t2_correct" in prp_raw.columns and prp_raw["t2_correct"].dtype == "object":
        prp_raw["t2_correct"] = prp_raw["t2_correct"].map({
            True: True, "True": True, 1: True,
            False: False, "False": False, 0: False,
        })

    prp_raw = _apply_timeout_as_incorrect(prp_raw)

    prp_acc = prp_raw.copy()
    if "order_norm" in prp_acc.columns:
        prp_acc = prp_acc[prp_acc["order_norm"] == "t1_t2"]
    if "t2_pressed_while_t1_pending" in prp_acc.columns:
        prp_acc = prp_acc[prp_acc["t2_pressed_while_t1_pending"] == False]
    if not prp.empty and "participant_id" in prp_acc.columns:
        prp_acc = prp_acc[prp_acc["participant_id"].isin(prp["participant_id"].unique())]

    soa_levels = sorted(prp["soa_ms"].dropna().unique())

    return {
        "prp": prp,
        "prp_acc": prp_acc,
        "prp_raw": prp_raw,
        "soa_levels": soa_levels,
    }
