"""Stroop feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..core import coefficient_of_variation
from .loaders import load_stroop_trials


def derive_stroop_features(
    rt_min: float = 200,
    rt_max: float = 3000,
    data_dir: None | str | Path = None,
) -> pd.DataFrame:
    stroop, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        drop_timeouts=True,
        require_correct_for_rt=False,
    )

    rt_col = "rt" if "rt" in stroop.columns else "rt_ms"
    stroop["rt_ms"] = pd.to_numeric(stroop[rt_col], errors="coerce")
    stroop = stroop[stroop["rt_ms"].notna()]

    trial_col = None
    for cand in ("trial", "trialIndex", "idx", "trial_index"):
        if cand in stroop.columns:
            trial_col = cand
            break
    if trial_col:
        stroop["trial_order"] = pd.to_numeric(stroop[trial_col], errors="coerce")
        stroop = stroop.sort_values(["participant_id", "trial_order"])

    cond_col = None
    for cand in ("type", "condition", "cond", "congruency"):
        if cand in stroop.columns:
            cond_col = cand
            break

    records: List[Dict] = []
    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()

        if "correct" in grp.columns:
            grp["prev_correct"] = grp["correct"].shift(1)
            post_error = grp[grp["prev_correct"] == False]
            post_correct = grp[grp["prev_correct"] == True]
            post_error_mean = post_error["rt_ms"].mean() if len(post_error) > 0 else np.nan
            post_correct_mean = post_correct["rt_ms"].mean() if len(post_correct) > 0 else np.nan
            pes = (
                post_error_mean - post_correct_mean
                if pd.notna(post_error_mean) and pd.notna(post_correct_mean)
                else np.nan
            )
        else:
            pes = np.nan
            post_error_mean = np.nan
            post_correct_mean = np.nan

        slope = np.nan
        cv_incong = np.nan
        cv_cong = np.nan
        if cond_col and "trial_order" in grp.columns:
            cond_lower = grp[cond_col].astype(str).str.lower()
            incong = grp[cond_lower == "incongruent"]
            cong = grp[cond_lower == "congruent"]

            if len(incong) >= 5 and incong["trial_order"].nunique() > 1:
                x = incong["trial_order"].values
                y = incong["rt_ms"].values
                coef = np.polyfit(x, y, 1)
                slope = coef[0]

            cv_incong = coefficient_of_variation(incong["rt_ms"])
            cv_cong = coefficient_of_variation(cong["rt_ms"])

        records.append({
            "participant_id": pid,
            "stroop_post_error_slowing": pes,
            "stroop_post_error_rt": post_error_mean,
            "stroop_post_correct_rt": post_correct_mean,
            "stroop_incong_slope": slope,
            "stroop_cv_all": coefficient_of_variation(grp["rt_ms"]),
            "stroop_cv_incong": cv_incong,
            "stroop_cv_cong": cv_cong,
            "stroop_trials": len(grp),
        })

    return pd.DataFrame(records)
