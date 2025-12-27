"""PRP dispersion feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ....constants import DEFAULT_SOA_LONG, DEFAULT_SOA_SHORT, PRP_RT_MIN
from ....core import (
    coefficient_of_variation,
    interquartile_range,
    mean_squared_successive_differences,
    skewness,
)
from ...mechanism import load_or_compute_prp_mechanism_features
from ..._shared import prepare_prp_trials


def derive_prp_dispersion_features(
    rt_min: float = PRP_RT_MIN,
    rt_max: float | None = None,
    data_dir: Path | None = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_prp_trials(rt_min=rt_min, rt_max=rt_max, data_dir=data_dir)

    prp = prepared["prp"]
    soa_levels = prepared["soa_levels"]

    if not isinstance(prp, pd.DataFrame) or prp.empty:
        return pd.DataFrame()

    records: List[Dict] = []

    for pid, group in prp.groupby("participant_id"):
        short_soa = group[group["soa_ms"] <= DEFAULT_SOA_SHORT]
        long_soa = group[group["soa_ms"] >= DEFAULT_SOA_LONG]

        rt_all = group["t2_rt_ms"].dropna()
        order_col = None
        for cand in ("trial_order", "trial_index", "trialIndex", "idx", "trial", "trialIndexInBlock"):
            if cand in group.columns:
                order_col = cand
                break
        if order_col:
            rt_ordered = group.sort_values(order_col)["t2_rt_ms"]
        else:
            rt_ordered = group["t2_rt_ms"]

        records.append({
            "participant_id": pid,
            "prp_t2_cv_all": coefficient_of_variation(group["t2_rt_ms"].dropna()),
            "prp_t2_cv_short": coefficient_of_variation(short_soa["t2_rt_ms"].dropna()),
            "prp_t2_cv_long": coefficient_of_variation(long_soa["t2_rt_ms"].dropna()),
            "prp_t2_rt_sd_all": float(rt_all.std(ddof=1)) if len(rt_all) > 1 else np.nan,
            "prp_t2_rt_iqr_all": interquartile_range(rt_all),
            "prp_t2_rt_skew_all": skewness(rt_all),
            "prp_t2_rt_mssd_all": mean_squared_successive_differences(rt_ordered),
        })

        if soa_levels:
            record = records[-1]
            for soa_val in soa_levels:
                label = int(round(float(soa_val)))
                subset = group[group["soa_ms"] == soa_val]
                record[f"prp_t2_rt_sd_soa_{label}"] = subset["t2_rt_ms"].std()
                if "t1_rt_ms" in subset.columns:
                    record[f"prp_t1_rt_sd_soa_{label}"] = subset["t1_rt_ms"].std()

    features_df = pd.DataFrame(records)

    mech = load_or_compute_prp_mechanism_features(data_dir=data_dir)
    sigma_cols = [
        "prp_exg_short_sigma",
        "prp_exg_long_sigma",
        "prp_exg_overall_sigma",
        "prp_exg_sigma_bottleneck",
    ]
    if not mech.empty:
        available = [c for c in sigma_cols if c in mech.columns]
        if available:
            mech = mech[["participant_id"] + available]
            features_df = features_df.merge(mech, on="participant_id", how="left")

    return features_df
