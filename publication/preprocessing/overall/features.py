"""Overall feature derivation (merged trial features)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import get_results_dir
from ..stroop.features import derive_stroop_features
from ..wcst.features import derive_wcst_features
from ..standardization import safe_zscore
from ._shared import get_overall_participant_ids, filter_to_overall_ids
from .loaders import load_overall_summary


def _compute_cross_task_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    required = ["pe_rate", "stroop_interference"]
    if summary.empty or not all(col in summary.columns for col in required):
        return pd.DataFrame()

    records = []
    for _, row in summary.iterrows():
        values = [row[col] for col in required if pd.notna(row[col])]
        n_tasks = len(values)
        if n_tasks < 2:
            records.append({
                "participant_id": row["participant_id"],
                "cross_task_cv": np.nan,
                "cross_task_range": np.nan,
                "cross_task_mean": np.nan,
                "cross_task_sd": np.nan,
                "cross_task_n_tasks": float(n_tasks),
            })
            continue

        mean_val = float(np.mean(values))
        sd_val = float(np.std(values))
        cv_val = float(sd_val / abs(mean_val)) if mean_val != 0 else np.nan
        range_val = float(max(values) - min(values))

        records.append({
            "participant_id": row["participant_id"],
            "cross_task_cv": cv_val,
            "cross_task_range": range_val,
            "cross_task_mean": mean_val,
            "cross_task_sd": sd_val,
            "cross_task_n_tasks": float(n_tasks),
        })

    return pd.DataFrame(records)


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Merge trial-derived features from Stroop and WCST.
    """
    if data_dir is None:
        data_dir = get_results_dir("overall")
    overall_ids = get_overall_participant_ids(data_dir)

    stroop_features = derive_stroop_features(data_dir=get_results_dir("stroop"))
    wcst_features = derive_wcst_features(data_dir=get_results_dir("wcst"))

    if overall_ids:
        stroop_features = filter_to_overall_ids(stroop_features, overall_ids)
        wcst_features = filter_to_overall_ids(wcst_features, overall_ids)

    merged = stroop_features.merge(wcst_features, on="participant_id", how="inner")
    summary = load_overall_summary(data_dir)
    cross_task = _compute_cross_task_consistency(summary)
    if not cross_task.empty:
        merged = merged.merge(cross_task, on="participant_id", how="left")

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
