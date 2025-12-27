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

        records.append({
            "participant_id": pid,
            "stroop_cong_slope": slopes["congruent"],
            "stroop_incong_slope": slopes["incongruent"],
            "stroop_neutral_slope": slopes["neutral"],
            "stroop_rt_lag1": rt_lag1,
            "stroop_rt_dfa_alpha": rt_dfa,
        })

    return pd.DataFrame(records)
