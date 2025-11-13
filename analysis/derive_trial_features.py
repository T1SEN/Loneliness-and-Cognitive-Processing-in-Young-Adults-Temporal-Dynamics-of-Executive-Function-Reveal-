#!/usr/bin/env python3
"""
Derive trial-level features from PRP, Stroop, and WCST trial logs.

Outputs participant-level aggregates (e.g., PRP T2 CV, Stroop post-error
slowing, Stroop RT slope) to results/analysis_outputs/trial_level_features.csv
for use in higher-order hypotheses (Hγ8~Hγ10 등).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "analysis_outputs" / "trial_level_features.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_prp_features() -> pd.DataFrame:
    path = RESULTS_DIR / "4a_prp_trials.csv"
    prp = pd.read_csv(path)

    prp["participant_id"] = prp["participant_id"].fillna(prp["participantId"])
    prp["t2_rt_ms"] = pd.to_numeric(prp["t2_rt_ms"], errors="coerce")
    prp["soa_nominal_ms"] = pd.to_numeric(prp["soa_nominal_ms"], errors="coerce")
    prp = prp[
        (prp["t2_timeout"] == False)
        & (prp["t2_rt_ms"].notna())
        & (prp["t2_rt_ms"] > 0)
    ]

    def cv(series: pd.Series) -> float:
        if len(series) < 3 or series.mean() == 0:
            return np.nan
        return series.std(ddof=1) / series.mean()

    records: List[Dict] = []
    for pid, group in prp.groupby("participant_id"):
        overall_cv = cv(group["t2_rt_ms"])
        short = group[group["soa_nominal_ms"] <= 150]
        long = group[group["soa_nominal_ms"] >= 600]
        records.append(
            {
                "participant_id": pid,
                "prp_t2_cv_all": overall_cv,
                "prp_t2_cv_short": cv(short["t2_rt_ms"]),
                "prp_t2_cv_long": cv(long["t2_rt_ms"]),
                "prp_t2_trials": len(group),
            }
        )
    return pd.DataFrame(records)


def load_stroop_features() -> pd.DataFrame:
    path = RESULTS_DIR / "4c_stroop_trials.csv"
    stroop = pd.read_csv(path)
    stroop["participant_id"] = stroop["participant_id"].fillna(stroop["participantId"])
    stroop["rt_ms"] = pd.to_numeric(stroop["rt_ms"], errors="coerce")
    stroop["trial"] = pd.to_numeric(stroop["trial"], errors="coerce")
    stroop = stroop[(stroop["timeout"] == False) & stroop["rt_ms"].notna()]
    stroop = stroop.sort_values(["participant_id", "trial"])

    records: List[Dict] = []
    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()
        grp["prev_correct"] = grp["correct"].shift(1)
        grp["prev_rt"] = grp["rt_ms"].shift(1)
        post_error = grp[(grp["prev_correct"] == False)]
        post_correct = grp[(grp["prev_correct"] == True)]
        post_error_mean = post_error["rt_ms"].mean()
        post_correct_mean = post_correct["rt_ms"].mean()
        post_error_diff = (
            post_error_mean - post_correct_mean
            if pd.notna(post_error_mean) and pd.notna(post_correct_mean)
            else np.nan
        )

        incong = grp[grp["type"].str.lower() == "incongruent"]
        slope = np.nan
        if len(incong) >= 5 and incong["trial"].nunique() > 1:
            x = incong["trial"].values
            y = incong["rt_ms"].values
            coef = np.polyfit(x, y, 1)
            slope = coef[0]

        records.append(
            {
                "participant_id": pid,
                "stroop_post_error_slowing": post_error_diff,
                "stroop_post_error_rt": post_error_mean,
                "stroop_post_correct_rt": post_correct_mean,
                "stroop_incong_slope": slope,
                "stroop_trials": len(grp),
            }
        )
    return pd.DataFrame(records)


def main():
    prp = load_prp_features()
    stroop = load_stroop_features()

    features = pd.merge(prp, stroop, on="participant_id", how="outer")
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved trial-level features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
