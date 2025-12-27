"""WCST drift feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..._shared import _load_wcst_summary_metrics, prepare_wcst_trials
from ....core import dfa_alpha, lag1_autocorrelation


def derive_wcst_drift_features(
    data_dir: None | str | Path = None,
    filter_rt: bool = False,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_wcst_trials(data_dir=data_dir, filter_rt=filter_rt)

    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    rule_col = prepared["rule_col"]
    trial_col = prepared["trial_col"]

    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None:
        return pd.DataFrame()

    records: List[Dict] = []
    for pid, grp in wcst.groupby("participant_id"):
        grp = grp.reset_index(drop=True)
        total_trials = int(len(grp))
        correct = grp["correct"].astype(bool).values if "correct" in grp.columns else np.zeros(total_trials, dtype=bool)

        trials_per_category = []
        rt_slope_overall = np.nan
        rt_slope_within = np.nan
        acc_slope_within = np.nan
        delta_trials = np.nan
        learning_slope_trials = np.nan

        rt_lag1 = np.nan
        rt_dfa = np.nan
        if rt_col:
            if trial_col and trial_col in grp.columns:
                grp = grp.sort_values(trial_col)
                trial_order = pd.to_numeric(grp[trial_col], errors="coerce")
            else:
                trial_order = pd.Series(np.arange(len(grp)), index=grp.index, dtype=float)

            rt_vals = pd.to_numeric(grp[rt_col], errors="coerce")
            rt_lag1 = lag1_autocorrelation(rt_vals)
            rt_dfa = dfa_alpha(rt_vals)
            valid = trial_order.notna() & rt_vals.notna()
            if valid.sum() >= 5 and trial_order[valid].nunique() > 1:
                rt_slope_overall = float(np.polyfit(trial_order[valid].values, rt_vals[valid].values, 1)[0])

        if rule_col:
            rules = grp[rule_col].astype(str).str.lower().values
            change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
            segment_starts = [0] + change_indices
            segment_ends = change_indices + [len(rules)]
            trials_per_category = [end - start for start, end in zip(segment_starts, segment_ends)]

            rt_slope_vals = []
            acc_slope_vals = []
            rt_vals = pd.to_numeric(grp[rt_col], errors="coerce").values.astype(float) if rt_col else np.array([])

            for start, end in zip(segment_starts, segment_ends):
                seg_len = end - start
                if seg_len >= 5:
                    x = np.arange(seg_len)
                    seg_rt = rt_vals[start:end]
                    seg_acc = correct[start:end].astype(float)
                    rt_mask = np.isfinite(seg_rt)
                    if rt_mask.sum() >= 3:
                        rt_slope_vals.append(float(np.polyfit(x[rt_mask], seg_rt[rt_mask], 1)[0]))
                    acc_slope_vals.append(float(np.polyfit(x, seg_acc, 1)[0]))

            if rt_slope_vals:
                rt_slope_within = float(np.mean(rt_slope_vals))
            if acc_slope_vals:
                acc_slope_within = float(np.mean(acc_slope_vals))

            if len(trials_per_category) >= 6:
                first = np.mean(trials_per_category[:3])
                last = np.mean(trials_per_category[-3:])
                delta_trials = float(first - last)
            if len(trials_per_category) >= 2:
                x = np.arange(1, len(trials_per_category) + 1)
                learning_slope_trials = float(np.polyfit(x, trials_per_category, 1)[0])

        records.append({
            "participant_id": pid,
            "wcst_rt_slope_overall": rt_slope_overall,
            "wcst_rt_slope_within_category": rt_slope_within,
            "wcst_acc_slope_within_category": acc_slope_within,
            "wcst_delta_trials_first3_last3": delta_trials,
            "wcst_learning_slope_trials": learning_slope_trials,
            "wcst_rt_lag1": rt_lag1,
            "wcst_rt_dfa_alpha": rt_dfa,
        })

    features_df = pd.DataFrame(records)

    summary_df = _load_wcst_summary_metrics(
        data_dir if isinstance(data_dir, Path) else Path(data_dir) if data_dir else None
    )
    if not summary_df.empty:
        drift_cols = ["wcst_delta_clr_percent_first3_last3", "wcst_learning_slope_clr_percent"]
        summary_df = summary_df[["participant_id"] + [c for c in drift_cols if c in summary_df.columns]]
        summary_df = summary_df.set_index("participant_id")
        features_df = features_df.set_index("participant_id")
        for col in summary_df.columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].combine_first(summary_df[col])
            else:
                features_df[col] = summary_df[col]
        features_df = features_df.reset_index()

    return features_df
