"""Shared helpers for PRP feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
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


def compute_bottleneck_slope_block_change(
    grp: pd.DataFrame,
    rt_col: str,
    soa_col: str,
    trial_col: str | None,
    block_size: int = 30,
    min_trials_per_soa: int = 2,
    min_soa_levels: int = 3,
    min_blocks: int = 3,
) -> float:
    if grp.empty or rt_col not in grp.columns or soa_col not in grp.columns:
        return np.nan

    if trial_col and trial_col in grp.columns:
        trial_order = pd.to_numeric(grp[trial_col], errors="coerce")
        if trial_order.notna().any():
            base_order = trial_order - trial_order.min()
        else:
            base_order = pd.Series(np.arange(len(grp)), index=grp.index, dtype=float)
    else:
        base_order = pd.Series(np.arange(len(grp)), index=grp.index, dtype=float)

    df = grp.copy()
    df["_trial_order"] = base_order
    df = df[df["_trial_order"].notna()]
    if df.empty:
        return np.nan

    df["_block"] = (df["_trial_order"] // block_size).astype(int)
    block_slopes = []
    block_ids = []

    for block_id, block_df in df.groupby("_block"):
        agg = (
            block_df.groupby(soa_col)[rt_col]
            .agg(rt_mean="mean", n="count")
            .reset_index()
        )
        agg = agg[agg["n"] >= int(min_trials_per_soa)]
        if len(agg) < min_soa_levels:
            continue
        if agg[soa_col].nunique() < 2:
            continue
        slope = np.polyfit(
            agg[soa_col].to_numpy(dtype=float),
            agg["rt_mean"].to_numpy(dtype=float),
            1,
        )[0]
        if np.isfinite(slope):
            block_ids.append(block_id)
            block_slopes.append(slope)

    if len(block_slopes) < min_blocks:
        return np.nan

    order = np.argsort(block_ids)
    x = np.array([block_ids[i] for i in order], dtype=float) + 1.0
    y = np.array([block_slopes[i] for i in order], dtype=float)
    if len(np.unique(x)) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


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
