"""PRP drift feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from ....constants import PRP_RT_MIN
from ....core import dfa_alpha, lag1_autocorrelation
from ..._shared import prepare_prp_trials


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
        if valid.sum() >= 5 and grp.loc[valid, "trial_order"].nunique() > 1:
            x = grp.loc[valid, "trial_order"].values
            y = rt_vals.loc[valid].values
            slope = float(np.polyfit(x, y, 1)[0])

        records.append({
            "participant_id": pid,
            "prp_t2_rt_slope": slope,
            "prp_t2_rt_lag1": rt_lag1,
            "prp_t2_rt_dfa_alpha": rt_dfa,
        })

    return pd.DataFrame(records)
