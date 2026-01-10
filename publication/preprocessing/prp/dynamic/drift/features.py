"""PRP drift feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from ....constants import PRP_RT_MIN, DEFAULT_SOA_SHORT, DEFAULT_SOA_LONG
from ....core import dfa_alpha, lag1_autocorrelation
from ..._shared import (
    prepare_prp_trials,
    compute_bottleneck_effect_slope_block_change,
    compute_bottleneck_slope_block_change,
)


def derive_prp_drift_features(
    rt_min: float = PRP_RT_MIN,
    rt_max: float | None = None,
    data_dir: Path | None = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_prp_trials(rt_min=rt_min, rt_max=rt_max, data_dir=data_dir)

    prp = prepared["prp"]
    if not isinstance(prp, pd.DataFrame) or prp.empty:
        return pd.DataFrame()

    rt_col = "t2_rt_ms" if "t2_rt_ms" in prp.columns else "t2_rt"
    trial_col = None
    for cand in ("trial_order", "trial_index", "trialIndex", "idx", "trial", "trialIndexInBlock"):
        if cand in prp.columns:
            trial_col = cand
            break

    prp_block_size = 30
    records: List[Dict] = []
    for pid, group in prp.groupby("participant_id"):
        grp = group.copy()

        if trial_col:
            grp["trial_order"] = pd.to_numeric(grp[trial_col], errors="coerce")
            grp = grp.sort_values("trial_order")
        else:
            grp = grp.reset_index(drop=True)
            grp["trial_order"] = grp.index.astype(float)

        rt_vals = pd.to_numeric(grp.get(rt_col), errors="coerce")
        rt_lag1 = lag1_autocorrelation(rt_vals)
        rt_dfa = dfa_alpha(rt_vals)
        valid = grp["trial_order"].notna() & rt_vals.notna()
        slope = np.nan
        slope_short = np.nan
        slope_long = np.nan
        if valid.sum() >= 5 and grp.loc[valid, "trial_order"].nunique() > 1:
            x = grp.loc[valid, "trial_order"].values
            y = rt_vals.loc[valid].values
            slope = float(np.polyfit(x, y, 1)[0])

        soa_col = "soa_ms" if "soa_ms" in grp.columns else "soa"
        if soa_col in grp.columns:
            short_mask = valid & (grp[soa_col] <= DEFAULT_SOA_SHORT)
            long_mask = valid & (grp[soa_col] >= DEFAULT_SOA_LONG)
            if short_mask.sum() >= 5 and grp.loc[short_mask, "trial_order"].nunique() > 1:
                x = grp.loc[short_mask, "trial_order"].values
                y = rt_vals.loc[short_mask].values
                slope_short = float(np.polyfit(x, y, 1)[0])
            if long_mask.sum() >= 5 and grp.loc[long_mask, "trial_order"].nunique() > 1:
                x = grp.loc[long_mask, "trial_order"].values
                y = rt_vals.loc[long_mask].values
                slope_long = float(np.polyfit(x, y, 1)[0])

        bottleneck_slope_block_change = compute_bottleneck_slope_block_change(
            grp,
            rt_col=rt_col,
            soa_col=soa_col,
            trial_col=trial_col,
            block_size=prp_block_size,
        )
        bottleneck_effect_slope_block_change = compute_bottleneck_effect_slope_block_change(
            grp,
            rt_col=rt_col,
            soa_col=soa_col,
            trial_col=trial_col,
            short_soa_max=DEFAULT_SOA_SHORT,
            long_soa_min=DEFAULT_SOA_LONG,
            block_size=prp_block_size,
        )

        records.append({
            "participant_id": pid,
            "prp_t2_rt_slope": slope,
            "prp_t2_rt_slope_short": slope_short,
            "prp_t2_rt_slope_long": slope_long,
            "prp_bottleneck_slope_block_change": bottleneck_slope_block_change,
            "prp_bottleneck_effect_slope_block_change": bottleneck_effect_slope_block_change,
            "prp_t2_rt_lag1": rt_lag1,
            "prp_t2_rt_dfa_alpha": rt_dfa,
        })

    return pd.DataFrame(records)
