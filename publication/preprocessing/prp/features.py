"""PRP feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..constants import DEFAULT_RT_MIN
from ..core import coefficient_of_variation
from .loaders import load_prp_trials
from .exgaussian_mechanism import load_or_compute_prp_mechanism_features
from .bottleneck_mechanism import load_or_compute_prp_bottleneck_mechanism_features


def derive_prp_features(
    rt_min: float = DEFAULT_RT_MIN,
    rt_max: float | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    prp, _ = load_prp_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=True,
    )

    prp["t2_rt_ms"] = pd.to_numeric(prp.get("t2_rt", prp.get("t2_rt_ms")), errors="coerce")
    prp["soa_ms"] = pd.to_numeric(prp.get("soa", prp.get("soa_nominal_ms")), errors="coerce")

    for col in ["t1_correct", "t2_correct"]:
        if col in prp.columns and prp[col].dtype == "object":
            prp[col] = prp[col].map({
                True: True, "True": True, 1: True,
                False: False, "False": False, 0: False,
            })

    records: List[Dict] = []
    for pid, group in prp.groupby("participant_id"):
        overall_cv = coefficient_of_variation(group["t2_rt_ms"].dropna())
        short_soa = group[group["soa_ms"] <= 150]
        long_soa = group[group["soa_ms"] >= 1200]

        if "t1_correct" in group.columns and "t2_correct" in group.columns:
            valid = group[(group["t1_correct"].notna()) & (group["t2_correct"].notna())]
            t1_errors = valid[~valid["t1_correct"]]
            n_t1_errors = len(t1_errors)

            if n_t1_errors == 0:
                cascade_rate = np.nan
                cascade_inflation = np.nan
            else:
                t1_and_t2_errors = t1_errors[~t1_errors["t2_correct"]]
                n_cascade = len(t1_and_t2_errors)
                cascade_rate = n_cascade / n_t1_errors
                t2_error_rate = 1 - valid["t2_correct"].mean() if len(valid) > 0 else np.nan
                cascade_inflation = (
                    cascade_rate / t2_error_rate
                    if pd.notna(t2_error_rate) and t2_error_rate > 0
                    else np.nan
                )
        else:
            cascade_rate = np.nan
            cascade_inflation = np.nan

        grp_sorted = group.sort_values("trial_index" if "trial_index" in group.columns else "trial")
        if "t2_correct" in grp_sorted.columns:
            grp_sorted["prev_t2_error"] = (~grp_sorted["t2_correct"]).shift(1).fillna(False)
            post_error = grp_sorted[grp_sorted["prev_t2_error"] == True]["t2_rt_ms"]
            post_correct = grp_sorted[grp_sorted["prev_t2_error"] == False]["t2_rt_ms"]
            pes = (
                post_error.mean() - post_correct.mean()
                if len(post_error) > 0 and len(post_correct) > 0
                else np.nan
            )
        else:
            pes = np.nan

        records.append({
            "participant_id": pid,
            "prp_t2_cv_all": overall_cv,
            "prp_t2_cv_short": coefficient_of_variation(short_soa["t2_rt_ms"].dropna()),
            "prp_t2_cv_long": coefficient_of_variation(long_soa["t2_rt_ms"].dropna()),
            "prp_cascade_rate": cascade_rate,
            "prp_cascade_inflation": cascade_inflation,
            "prp_pes": pes,
            "prp_t2_trials": len(group),
        })

    features_df = pd.DataFrame(records)

    mechanism_df = load_or_compute_prp_mechanism_features(data_dir=data_dir)
    if not mechanism_df.empty:
        if features_df.empty:
            features_df = mechanism_df
        else:
            overlap = [c for c in mechanism_df.columns if c != "participant_id" and c in features_df.columns]
            if overlap:
                features_df = features_df.drop(columns=overlap)
            features_df = features_df.merge(mechanism_df, on="participant_id", how="left")

    bottleneck_df = load_or_compute_prp_bottleneck_mechanism_features(data_dir=data_dir)
    if bottleneck_df.empty:
        return features_df
    if features_df.empty:
        return bottleneck_df

    overlap = [c for c in bottleneck_df.columns if c != "participant_id" and c in features_df.columns]
    if overlap:
        features_df = features_df.drop(columns=overlap)

    return features_df.merge(bottleneck_df, on="participant_id", how="left")
