"""WCST traditional feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ...constants import get_results_dir
from .._shared import _load_wcst_summary_metrics, prepare_wcst_trials


def derive_wcst_traditional_features(
    data_dir: None | str | Path = None,
    filter_rt: bool = False,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_wcst_trials(data_dir=data_dir, filter_rt=filter_rt)

    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    rule_col = prepared["rule_col"]

    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None:
        return pd.DataFrame()

    records: List[Dict] = []
    for pid, grp in wcst.groupby("participant_id"):
        grp = grp.reset_index(drop=True)

        total_trials = int(len(grp))
        correct = grp["correct"].astype(bool).values if "correct" in grp.columns else np.zeros(total_trials, dtype=bool)
        errors = ~correct
        total_errors = int(errors.sum()) if total_trials else 0
        error_rate = (total_errors / total_trials) if total_trials else np.nan

        is_pe = grp["isPE"].astype(bool).values if "isPE" in grp.columns else np.zeros(total_trials, dtype=bool)
        is_pr = grp["isPR"].astype(bool).values if "isPR" in grp.columns else np.zeros(total_trials, dtype=bool)
        is_npe = grp["isNPE"].astype(bool).values if "isNPE" in grp.columns else np.zeros(total_trials, dtype=bool)
        has_is_npe = "isNPE" in grp.columns

        pe_count = int(is_pe.sum())
        pr_count = int(is_pr.sum())
        npe_count = int(is_npe.sum()) if has_is_npe else max(total_errors - pe_count, 0)

        pe_rate = (pe_count / total_trials) * 100 if total_trials else np.nan
        pr_percent = (pr_count / total_trials) * 100 if total_trials else np.nan
        error_pr_ratio = (int((is_pr & errors).sum()) / total_errors) if total_errors else np.nan
        error_npe_ratio = (npe_count / total_errors) if total_errors else np.nan

        trials_per_category = []
        trials_to_first_category = np.nan
        categories_completed = np.nan

        if rule_col:
            rules = grp[rule_col].astype(str).str.lower().values
            change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
            segment_starts = [0] + change_indices
            segment_ends = change_indices + [len(rules)]
            trials_per_category = [end - start for start, end in zip(segment_starts, segment_ends)]
            categories_completed = float(len(trials_per_category)) if trials_per_category else np.nan
            if trials_per_category:
                trials_to_first_category = float(trials_per_category[0])

        record = {
            "participant_id": pid,
            "wcst_trials": total_trials,
            "wcst_total_errors": total_errors,
            "wcst_error_rate": error_rate,
            "wcst_perseverative_errors": pe_count,
            "wcst_perseverative_error_rate": pe_rate,
            "wcst_nonperseverative_errors": npe_count,
            "wcst_perseverative_responses": pr_count,
            "wcst_perseverative_response_percent": pr_percent,
            "wcst_error_pr_ratio": error_pr_ratio,
            "wcst_error_npe_ratio": error_npe_ratio,
            "wcst_trials_to_first_category": trials_to_first_category,
            "wcst_categories_completed": categories_completed,
            "wcst_trials_per_category_mean": float(np.mean(trials_per_category)) if trials_per_category else np.nan,
            "wcst_trials_per_category_sd": float(np.std(trials_per_category)) if trials_per_category else np.nan,
        }

        if trials_per_category:
            for idx, val in enumerate(trials_per_category, start=1):
                record[f"wcst_trials_per_category_{idx}"] = float(val)

        records.append(record)

    features_df = pd.DataFrame(records)

    summary_df = _load_wcst_summary_metrics(
        data_dir if isinstance(data_dir, Path) else Path(data_dir) if data_dir else None
    )
    if not summary_df.empty:
        drift_cols = {
            "wcst_delta_clr_percent_first3_last3",
            "wcst_learning_slope_clr_percent",
        }
        summary_df = summary_df.set_index("participant_id")
        features_df = features_df.set_index("participant_id")
        for col in summary_df.columns:
            if col in drift_cols:
                continue
            if col in features_df.columns:
                features_df[col] = features_df[col].combine_first(summary_df[col])
            else:
                features_df[col] = summary_df[col]
        features_df = features_df.reset_index()

    return features_df
