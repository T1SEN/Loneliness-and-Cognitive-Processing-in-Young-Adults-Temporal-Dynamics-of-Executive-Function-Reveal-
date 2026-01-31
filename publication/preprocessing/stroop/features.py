"""Stroop feature derivation (manuscript core only)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..constants import STROOP_RT_MIN, STROOP_RT_MAX
from .loaders import load_stroop_trials


def _normalize_condition(value: object) -> str | None:
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


def _trial_order(grp: pd.DataFrame) -> pd.Series:
    for cand in ("trial", "trialIndex", "trial_index", "idx", "trialOrder"):
        if cand in grp.columns:
            return pd.to_numeric(grp[cand], errors="coerce")
    return pd.Series(np.arange(len(grp), dtype=float), index=grp.index)


def _interference_slope(grp: pd.DataFrame) -> float:
    if "trial_order" not in grp.columns or "condition_norm" not in grp.columns:
        return np.nan
    subset = grp[grp["condition_norm"].isin(["congruent", "incongruent"])].copy()
    subset = subset.dropna(subset=["trial_order", "rt_ms"])
    subset = subset.sort_values("trial_order").reset_index(drop=True)
    if len(subset) < 20:
        return np.nan

    order_rank = subset["trial_order"].rank(method="first")
    subset["quartile"] = pd.qcut(order_rank, q=4, labels=[1, 2, 3, 4])

    def _quartile_interference(q: int) -> float:
        q_df = subset[subset["quartile"] == q]
        inc = q_df[q_df["condition_norm"] == "incongruent"]["rt_ms"]
        con = q_df[q_df["condition_norm"] == "congruent"]["rt_ms"]
        if len(inc) >= 3 and len(con) >= 3:
            return float(inc.mean() - con.mean())
        return np.nan

    int_vals = [_quartile_interference(q) for q in [1, 2, 3, 4]]
    valid_idx = [i for i, v in enumerate(int_vals) if pd.notna(v)]
    if len(valid_idx) < 3:
        return np.nan

    x_fit = np.array(valid_idx, dtype=float)
    y_fit = np.array([int_vals[i] for i in valid_idx], dtype=float)
    return float(np.polyfit(x_fit, y_fit, 1)[0])


def derive_stroop_features(
    rt_min: float = STROOP_RT_MIN,
    rt_max: float | None = None,
    data_dir: None | str | Path = None,
) -> pd.DataFrame:
    rt_max_val = STROOP_RT_MAX if rt_max is None else rt_max
    stroop, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max_val,
        apply_trial_filters=True,
        require_correct_for_rt=True,
    )

    if not isinstance(stroop, pd.DataFrame) or stroop.empty:
        return pd.DataFrame(columns=["participant_id", "stroop_interference_slope"])

    if "rt_ms" not in stroop.columns:
        stroop["rt_ms"] = pd.to_numeric(stroop.get("rt"), errors="coerce")

    cond_col = None
    for cand in ("type", "condition", "cond", "congruency"):
        if cand in stroop.columns:
            cond_col = cand
            break
    if cond_col:
        stroop["condition_norm"] = stroop[cond_col].apply(_normalize_condition)
    else:
        stroop["condition_norm"] = None

    records: List[dict[str, float | str]] = []
    for pid, grp in stroop.groupby("participant_id"):
        grp = grp.copy()
        grp["trial_order"] = _trial_order(grp)
        slope = _interference_slope(grp)
        records.append({
            "participant_id": pid,
            "stroop_interference_slope": slope,
        })

    return pd.DataFrame(records)
