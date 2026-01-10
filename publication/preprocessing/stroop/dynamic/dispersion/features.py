"""Stroop dispersion feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ....constants import STROOP_RT_MIN
from ....core import (
    coefficient_of_variation,
    compute_fatigue_slopes,
    compute_temporal_variability_slopes,
    interquartile_range,
    mean_squared_successive_differences,
    skewness,
)
from ...mechanism import load_or_compute_stroop_mechanism_features
from ..._shared import prepare_stroop_trials


def derive_stroop_dispersion_features(
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
        cv_incong = np.nan
        cv_cong = np.nan
        rt_sd_all = np.nan
        rt_iqr_all = np.nan
        rt_skew_all = np.nan
        rt_mssd_all = np.nan

        if cond_col:
            incong = grp[grp["condition_norm"] == "incongruent"]
            cong = grp[grp["condition_norm"] == "congruent"]
            cv_incong = coefficient_of_variation(incong["rt_ms"].dropna())
            cv_cong = coefficient_of_variation(cong["rt_ms"].dropna())

        rt_all = grp["rt_ms"].dropna()
        if not rt_all.empty:
            rt_sd_all = float(rt_all.std(ddof=1)) if len(rt_all) > 1 else np.nan
            rt_iqr_all = interquartile_range(rt_all)
            rt_skew_all = skewness(rt_all)

        if "trial_order" in grp.columns:
            rt_ordered = grp.sort_values("trial_order")["rt_ms"]
        else:
            rt_ordered = grp["rt_ms"]
        rt_mssd_all = mean_squared_successive_differences(rt_ordered)

        if "trial_order" in grp.columns:
            seq = grp.sort_values("trial_order")
            seq_rt = seq["rt_ms"]
            seq_correct = seq["correct"] if "correct" in seq.columns else None
        else:
            seq_rt = grp["rt_ms"]
            seq_correct = grp["correct"] if "correct" in grp.columns else None

        if seq_correct is not None:
            seq_rt_correct = seq_rt[seq_correct == True]
        else:
            seq_rt_correct = seq_rt
        fatigue_metrics = compute_fatigue_slopes(seq_rt_correct)
        variability_slopes = compute_temporal_variability_slopes(seq_rt_correct)

        records.append({
            "participant_id": pid,
            "stroop_cv_all": coefficient_of_variation(grp["rt_ms"].dropna()),
            "stroop_cv_incong": cv_incong,
            "stroop_cv_cong": cv_cong,
            "stroop_rt_sd_all": rt_sd_all,
            "stroop_rt_iqr_all": rt_iqr_all,
            "stroop_rt_skew_all": rt_skew_all,
            "stroop_rt_mssd_all": rt_mssd_all,
            "stroop_cv_fatigue_slope": fatigue_metrics["cv_fatigue_slope"],
            "stroop_cv_fatigue_slope_rolling": fatigue_metrics["cv_fatigue_slope_rolling"],
            "stroop_rt_sd_block_slope": variability_slopes["sd_slope"],
            "stroop_rt_p90_block_slope": variability_slopes["p90_slope"],
            "stroop_residual_sd_block_slope": variability_slopes["residual_sd_slope"],
        })

    features_df = pd.DataFrame(records)

    mech = load_or_compute_stroop_mechanism_features(data_dir=data_dir)
    sigma_cols = [
        "stroop_exg_congruent_sigma",
        "stroop_exg_incongruent_sigma",
        "stroop_exg_neutral_sigma",
        "stroop_exg_sigma_interference",
    ]
    if not mech.empty:
        available = [c for c in sigma_cols if c in mech.columns]
        if available:
            mech = mech[["participant_id"] + available]
            features_df = features_df.merge(mech, on="participant_id", how="left")

    return features_df
