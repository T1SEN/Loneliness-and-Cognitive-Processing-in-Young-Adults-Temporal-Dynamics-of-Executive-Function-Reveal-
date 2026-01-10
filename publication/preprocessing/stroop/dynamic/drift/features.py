"""Stroop drift feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ....constants import STROOP_RT_MIN
from ....core import dfa_alpha, lag1_autocorrelation
from ..._shared import prepare_stroop_trials


def derive_stroop_drift_features(
    rt_min: float = STROOP_RT_MIN,
    rt_max: float | None = None,
    data_dir: None | str | Path = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_stroop_trials(rt_min=rt_min, rt_max=rt_max, data_dir=data_dir)

    stroop = prepared["stroop"]
    cond_col = prepared["cond_col"]

    if not isinstance(stroop, pd.DataFrame) or stroop.empty:
        return pd.DataFrame()

    records: List[Dict] = []
    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()
        slopes = {
            "congruent": np.nan,
            "incongruent": np.nan,
            "neutral": np.nan,
        }

        if "trial_order" in grp.columns:
            rt_ordered = grp.sort_values("trial_order")["rt_ms"]
        else:
            rt_ordered = grp["rt_ms"]
        rt_lag1 = lag1_autocorrelation(rt_ordered)
        rt_dfa = dfa_alpha(rt_ordered)

        if cond_col and "trial_order" in grp.columns:
            for cond in slopes:
                subset = grp[grp["condition_norm"] == cond]
                if "correct" in subset.columns:
                    subset = subset[subset["correct"] == True]
                subset = subset.dropna(subset=["trial_order", "rt_ms"])
                if len(subset) >= 5 and subset["trial_order"].nunique() > 1:
                    x = subset["trial_order"].values
                    y = subset["rt_ms"].values
                    slopes[cond] = float(np.polyfit(x, y, 1)[0])

        # Interference slope: quartile-based linear regression
        interference_slope = np.nan
        if cond_col and "trial_order" in grp.columns:
            grp_sorted = grp.copy()
            if "correct" in grp_sorted.columns:
                grp_sorted = grp_sorted[grp_sorted["correct"] == True]
            grp_sorted = grp_sorted[
                grp_sorted["condition_norm"].isin(["congruent", "incongruent"])
            ]
            grp_sorted = grp_sorted.dropna(subset=["trial_order"])
            grp_sorted = grp_sorted.sort_values("trial_order").reset_index(drop=True)
            n = len(grp_sorted)
            if n >= 20:
                order_rank = grp_sorted["trial_order"].rank(method="first")
                grp_sorted["quartile"] = pd.qcut(order_rank, q=4, labels=[1, 2, 3, 4])

                def _quartile_interference(q: int) -> float:
                    q_df = grp_sorted[grp_sorted["quartile"] == q]
                    inc = q_df[q_df["condition_norm"] == "incongruent"]["rt_ms"]
                    con = q_df[q_df["condition_norm"] == "congruent"]["rt_ms"]
                    if len(inc) >= 3 and len(con) >= 3:
                        return float(inc.mean() - con.mean())
                    return np.nan

                int_vals = [_quartile_interference(q) for q in [1, 2, 3, 4]]
                valid_idx = [i for i, v in enumerate(int_vals) if pd.notna(v)]
                if len(valid_idx) >= 3:
                    x_fit = np.array(valid_idx, dtype=float)
                    y_fit = np.array([int_vals[i] for i in valid_idx], dtype=float)
                    interference_slope = float(np.polyfit(x_fit, y_fit, 1)[0])

        records.append({
            "participant_id": pid,
            "stroop_cong_slope": slopes["congruent"],
            "stroop_incong_slope": slopes["incongruent"],
            "stroop_neutral_slope": slopes["neutral"],
            "stroop_interference_slope": interference_slope,
            "stroop_rt_lag1": rt_lag1,
            "stroop_rt_dfa_alpha": rt_dfa,
        })

    return pd.DataFrame(records)
