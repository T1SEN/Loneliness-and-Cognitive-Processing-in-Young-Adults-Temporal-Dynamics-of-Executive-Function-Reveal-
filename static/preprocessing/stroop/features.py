"""
Stroop feature derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import STROOP_RT_MIN, STROOP_RT_MAX


def _assign_stroop_segments(stroop: pd.DataFrame) -> pd.DataFrame:
    df = stroop.sort_values(["participant_id", "trial_order"]).copy()

    def _assign_segment(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["segment"] = np.nan
        valid = group["trial_order"].notna()
        if not valid.any():
            return group
        sub = group[valid].sort_values("trial_order").copy()
        n_trials = len(sub)
        if n_trials == 0:
            return group
        positions = np.arange(1, n_trials + 1)
        edges = np.linspace(0, n_trials, 5)
        sub["segment"] = pd.cut(
            positions,
            bins=edges,
            labels=[1, 2, 3, 4],
            include_lowest=True,
        ).astype(float)
        group.loc[sub.index, "segment"] = sub["segment"]
        return group

    return df.groupby("participant_id", group_keys=False).apply(_assign_segment)


def compute_stroop_interference(stroop: pd.DataFrame) -> pd.Series:
    valid = (
        (stroop["cond"].isin({"congruent", "incongruent"}))
        & (stroop["correct"])
        & (~stroop["timeout"])
        & (stroop["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX))
    )
    subset = stroop[valid].copy()
    means = subset.groupby(["participant_id", "cond"])["rt_ms"].mean().unstack()
    if "incongruent" not in means.columns or "congruent" not in means.columns:
        return pd.Series(dtype=float)
    return means["incongruent"] - means["congruent"]


def compute_stroop_interference_slope(stroop: pd.DataFrame) -> pd.Series:
    if stroop.empty:
        return pd.Series(dtype=float)

    stroop = _assign_stroop_segments(stroop)
    valid = (
        (stroop["cond"].isin({"congruent", "incongruent"}))
        & (stroop["correct"])
        & (~stroop["timeout"])
        & (stroop["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX))
    )
    subset = stroop[valid].copy()
    seg_means = (
        subset.groupby(["participant_id", "segment", "cond"])["rt_ms"].mean().unstack()
    )
    if "incongruent" not in seg_means.columns or "congruent" not in seg_means.columns:
        return pd.Series(dtype=float)

    seg_means["interference"] = seg_means["incongruent"] - seg_means["congruent"]
    seg_means = seg_means.reset_index()[["participant_id", "segment", "interference"]]

    def _slope(group: pd.DataFrame) -> float:
        group = group.dropna(subset=["interference"])
        if len(group) < 2:
            return np.nan
        x = group["segment"].astype(float).to_numpy()
        y = group["interference"].to_numpy()
        return float(np.polyfit(x, y, 1)[0])

    return seg_means.groupby("participant_id").apply(_slope)


def build_stroop_features(stroop: pd.DataFrame) -> pd.DataFrame:
    if stroop.empty:
        return pd.DataFrame(columns=["participant_id", "stroop_interference", "stroop_interference_slope"])

    participant_ids = sorted(stroop["participant_id"].dropna().astype(str).unique())
    features = pd.DataFrame({"participant_id": participant_ids})

    interference = compute_stroop_interference(stroop)
    slope = compute_stroop_interference_slope(stroop)

    features = features.merge(
        interference.rename("stroop_interference"),
        on="participant_id",
        how="left",
    )
    features = features.merge(
        slope.rename("stroop_interference_slope"),
        on="participant_id",
        how="left",
    )

    return features
