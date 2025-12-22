"""Overall feature derivation (merged trial features)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..prp.features import derive_prp_features
from ..stroop.features import derive_stroop_features
from ..wcst.features import derive_wcst_features
from ..standardization import safe_zscore


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Merge trial-derived features from PRP, Stroop, and WCST.
    """
    prp_features = derive_prp_features(data_dir=data_dir)
    stroop_features = derive_stroop_features(data_dir=data_dir)
    wcst_features = derive_wcst_features(data_dir=data_dir)

    merged = prp_features.merge(stroop_features, on="participant_id", how="inner")
    merged = merged.merge(wcst_features, on="participant_id", how="inner")

    if not merged.empty:
        cv_cols = ["stroop_cv_fatigue_slope", "wcst_cv_fatigue_slope"]
        tau_cols = ["stroop_tau_slope", "wcst_tau_slope"]
        acc_cols = ["stroop_acc_fatigue_slope", "wcst_acc_fatigue_slope"]
        rt_cols = ["stroop_rt_fatigue_slope", "wcst_rt_fatigue_slope"]

        merged["sustained_attention_cv_slope"] = merged[cv_cols].mean(axis=1, skipna=True)
        merged["sustained_attention_tau_slope"] = merged[tau_cols].mean(axis=1, skipna=True)
        merged["sustained_attention_acc_slope"] = merged[acc_cols].mean(axis=1, skipna=True)
        merged["sustained_attention_rt_slope"] = merged[rt_cols].mean(axis=1, skipna=True)

        z_cv = safe_zscore(merged["sustained_attention_cv_slope"])
        z_tau = safe_zscore(merged["sustained_attention_tau_slope"])
        z_acc = safe_zscore(merged["sustained_attention_acc_slope"])
        z_rt = safe_zscore(merged["sustained_attention_rt_slope"])

        merged["sustained_attention_index"] = (z_cv + z_tau + (-z_acc) + z_rt) / 4

    return merged
