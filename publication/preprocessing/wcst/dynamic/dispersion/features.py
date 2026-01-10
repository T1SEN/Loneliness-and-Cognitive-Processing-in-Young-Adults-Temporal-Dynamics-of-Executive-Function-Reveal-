"""WCST dispersion feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ....core import (
    coefficient_of_variation,
    compute_temporal_variability_slopes,
    interquartile_range,
    mean_squared_successive_differences,
    skewness,
)
from ....constants import WCST_RT_MIN, WCST_RT_MAX
from ..._shared import prepare_wcst_trials


def derive_wcst_dispersion_features(
    data_dir: None | str | Path = None,
    filter_rt: bool = False,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_wcst_trials(data_dir=data_dir, filter_rt=filter_rt)

    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]

    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None:
        return pd.DataFrame()

    records: List[Dict] = []
    trial_col = prepared.get("trial_col")
    for pid, grp in wcst.groupby("participant_id"):
        rt_all = pd.to_numeric(grp[rt_col], errors="coerce")
        if "is_rt_valid" in grp.columns:
            rt_all = rt_all.where(grp["is_rt_valid"].astype(bool))
        else:
            rt_all = rt_all.where(rt_all.between(WCST_RT_MIN, WCST_RT_MAX))

        if trial_col and trial_col in grp.columns:
            grp_sorted = grp.sort_values(trial_col)
            rt_ordered = pd.to_numeric(grp_sorted[rt_col], errors="coerce")
            if "is_rt_valid" in grp_sorted.columns:
                rt_ordered = rt_ordered.where(grp_sorted["is_rt_valid"].astype(bool))
            else:
                rt_ordered = rt_ordered.where(rt_ordered.between(WCST_RT_MIN, WCST_RT_MAX))
            correct_ordered = grp_sorted["correct"] if "correct" in grp_sorted.columns else None
        else:
            rt_ordered = rt_all
            correct_ordered = grp["correct"] if "correct" in grp.columns else None

        rt_ordered_all = rt_ordered

        if correct_ordered is not None:
            rt_ordered = rt_ordered[correct_ordered == True]
        variability_slopes = compute_temporal_variability_slopes(rt_ordered)
        records.append({
            "participant_id": pid,
            "wcst_cv_rt": coefficient_of_variation(rt_all.dropna()),
            "wcst_rt_sd_all": float(rt_all.std(ddof=1)) if rt_all.notna().sum() > 1 else np.nan,
            "wcst_rt_iqr_all": interquartile_range(rt_all),
            "wcst_rt_skew_all": skewness(rt_all),
            "wcst_rt_mssd_all": mean_squared_successive_differences(rt_ordered_all),
            "wcst_rt_sd_block_slope": variability_slopes["sd_slope"],
            "wcst_rt_p90_block_slope": variability_slopes["p90_slope"],
            "wcst_residual_sd_block_slope": variability_slopes["residual_sd_slope"],
        })

    return pd.DataFrame(records)
