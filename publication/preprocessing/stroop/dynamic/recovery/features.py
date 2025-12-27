"""Stroop recovery feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ....constants import RAW_DIR, STROOP_RT_MIN, STROOP_RT_MAX
from ..._shared import _run_lengths, prepare_stroop_trials


def derive_stroop_recovery_features(
    rt_min: float = STROOP_RT_MIN,
    rt_max: float | None = None,
    data_dir: None | str | Path = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    rt_max_val = STROOP_RT_MAX if rt_max is None else rt_max

    def _rt_valid(series: pd.Series) -> pd.Series:
        valid = series.notna() & (series >= rt_min)
        if rt_max_val is not None:
            valid &= series <= rt_max_val
        return valid

    if prepared is None:
        prepared = prepare_stroop_trials(rt_min=rt_min, rt_max=rt_max, data_dir=data_dir)

    stroop = prepared["stroop"]
    stroop_acc = prepared["stroop_acc"]
    cond_col = prepared["cond_col"]

    if not isinstance(stroop, pd.DataFrame) or stroop.empty:
        return pd.DataFrame()

    acc_groups = {}
    if isinstance(stroop_acc, pd.DataFrame) and not stroop_acc.empty:
        acc_groups = {pid: grp for pid, grp in stroop_acc.groupby("participant_id")}

    raw_post_error = {}
    if isinstance(stroop, pd.DataFrame) and not stroop.empty:
        pids = set(stroop["participant_id"].unique())
        raw_prepared = prepare_stroop_trials(rt_min=rt_min, rt_max=rt_max, data_dir=RAW_DIR)
        raw_df = raw_prepared.get("stroop_raw")
        raw_cond_col = raw_prepared.get("cond_col")
        if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
            raw_df = raw_df[raw_df["participant_id"].isin(pids)].copy()
            if "timeout" in raw_df.columns:
                raw_df = raw_df[raw_df["timeout"] == False]
            elif "is_timeout" in raw_df.columns:
                raw_df = raw_df[raw_df["is_timeout"] == False]
            if "trial_order" not in raw_df.columns:
                for cand in ("trial", "trialIndex", "idx", "trial_index"):
                    if cand in raw_df.columns:
                        raw_df["trial_order"] = pd.to_numeric(raw_df[cand], errors="coerce")
                        break
            for pid, grp in raw_df.groupby("participant_id"):
                if "trial_order" in grp.columns:
                    grp = grp.sort_values("trial_order")
                else:
                    grp = grp.sort_index()
                grp = grp.reset_index(drop=True)

                pes = np.nan
                post_error_mean = np.nan
                post_correct_mean = np.nan
                post_error_accuracy = np.nan
                post_error_interference_rt = np.nan
                post_error_interference_acc = np.nan
                post_error_recovery_slope = np.nan

                if "correct" in grp.columns:
                    grp["prev_correct"] = grp["correct"].shift(1)
                    rt_valid = _rt_valid(grp["rt_ms"])

                    post_error = grp[rt_valid & (grp["prev_correct"] == False)]
                    post_correct = grp[rt_valid & (grp["prev_correct"] == True)]
                    post_error_mean = post_error["rt_ms"].mean() if len(post_error) > 0 else np.nan
                    post_correct_mean = post_correct["rt_ms"].mean() if len(post_correct) > 0 else np.nan
                    if pd.notna(post_error_mean) and pd.notna(post_correct_mean):
                        pes = post_error_mean - post_correct_mean

                    post_error_accuracy = grp[grp["prev_correct"] == False]["correct"].mean()

                if raw_cond_col and "condition_norm" in grp.columns and "prev_correct" in grp.columns:
                    seq = grp.copy()
                    seq["prev_cond"] = seq["condition_norm"].shift(1)

                    seq_rt_all = seq[_rt_valid(seq["rt_ms"])].copy()
                    if "correct" in seq_rt_all.columns:
                        seq_rt_all = seq_rt_all[seq_rt_all["correct"] == True]
                    seq_rt_all = seq_rt_all[
                        seq_rt_all["condition_norm"].isin(["congruent", "incongruent"])
                        & seq_rt_all["prev_cond"].isin(["congruent", "incongruent"])
                    ]

                    def _post_error_rt_diff(prev_flag: bool) -> float:
                        subset = seq_rt_all[seq_rt_all["prev_correct"] == prev_flag]
                        rt_incong = subset[subset["condition_norm"] == "incongruent"]["rt_ms"].mean()
                        rt_cong = subset[subset["condition_norm"] == "congruent"]["rt_ms"].mean()
                        if pd.notna(rt_incong) and pd.notna(rt_cong):
                            return rt_incong - rt_cong
                        return np.nan

                    diff_error_rt = _post_error_rt_diff(False)
                    diff_correct_rt = _post_error_rt_diff(True)
                    if pd.notna(diff_error_rt) and pd.notna(diff_correct_rt):
                        post_error_interference_rt = diff_error_rt - diff_correct_rt

                    seq_acc_all = seq.copy()
                    seq_acc_all = seq_acc_all[
                        seq_acc_all["condition_norm"].isin(["congruent", "incongruent"])
                        & seq_acc_all["prev_cond"].isin(["congruent", "incongruent"])
                    ]

                    def _post_error_acc_diff(prev_flag: bool) -> float:
                        subset = seq_acc_all[seq_acc_all["prev_correct"] == prev_flag]
                        acc_incong = subset[subset["condition_norm"] == "incongruent"]["correct"].mean()
                        acc_cong = subset[subset["condition_norm"] == "congruent"]["correct"].mean()
                        if pd.notna(acc_incong) and pd.notna(acc_cong):
                            return acc_incong - acc_cong
                        return np.nan

                    diff_error_acc = _post_error_acc_diff(False)
                    diff_correct_acc = _post_error_acc_diff(True)
                    if pd.notna(diff_error_acc) and pd.notna(diff_correct_acc):
                        post_error_interference_acc = diff_error_acc - diff_correct_acc

                if "correct" in grp.columns and "rt_ms" in grp.columns:
                    if grp["rt_ms"].notna().any():
                        post_slopes = []
                        error_indices = grp.index[grp["correct"] == False].tolist()
                        for idx in error_indices:
                            window = []
                            j = idx + 1
                            while j < len(grp) and len(window) < 5:
                                rt_val = grp.loc[j, "rt_ms"]
                                if pd.notna(rt_val) and rt_val >= rt_min and (
                                    rt_max_val is None or rt_val <= rt_max_val
                                ):
                                    window.append(float(rt_val))
                                j += 1
                            if len(window) == 5:
                                x = np.arange(1, len(window) + 1)
                                post_slopes.append(float(np.polyfit(x, window, 1)[0]))
                        if post_slopes:
                            post_error_recovery_slope = float(np.mean(post_slopes))

                raw_post_error[pid] = {
                    "stroop_post_error_slowing": pes,
                    "stroop_post_error_rt": post_error_mean,
                    "stroop_post_correct_rt": post_correct_mean,
                    "stroop_post_error_accuracy": post_error_accuracy,
                    "stroop_post_error_interference_rt": post_error_interference_rt,
                    "stroop_post_error_interference_acc": post_error_interference_acc,
                    "stroop_post_error_recovery_slope": post_error_recovery_slope,
                }

    records: List[Dict] = []

    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()
        acc_grp = acc_groups.get(pid, pd.DataFrame())

        if "correct" in grp.columns:
            grp["prev_correct"] = grp["correct"].shift(1)
            rt_valid = _rt_valid(grp["rt_ms"])
            post_error = grp[rt_valid & (grp["prev_correct"] == False)]
            post_correct = grp[rt_valid & (grp["prev_correct"] == True)]
            post_error_mean = post_error["rt_ms"].mean() if len(post_error) > 0 else np.nan
            post_correct_mean = post_correct["rt_ms"].mean() if len(post_correct) > 0 else np.nan
            pes = (
                post_error_mean - post_correct_mean
                if pd.notna(post_error_mean) and pd.notna(post_correct_mean)
                else np.nan
            )
        else:
            pes = np.nan
            post_error_mean = np.nan
            post_correct_mean = np.nan

        cse_rt = np.nan
        cse_acc = np.nan
        post_conflict_slowing = np.nan
        post_conflict_accuracy = np.nan
        post_error_accuracy = np.nan
        post_error_interference_rt = np.nan
        post_error_interference_acc = np.nan
        error_run_mean = np.nan
        error_run_max = np.nan
        error_recovery_rt = np.nan
        post_error_recovery_slope = np.nan

        if cond_col and "trial_order" in grp.columns:
            seq = grp.copy()
            seq["prev_cond"] = seq["condition_norm"].shift(1)
            seq["prev_correct"] = seq["correct"].shift(1) if "correct" in seq.columns else np.nan

            seq_rt_all = seq[_rt_valid(seq["rt_ms"])].copy()
            if "correct" in seq_rt_all.columns:
                seq_rt_all = seq_rt_all[seq_rt_all["correct"] == True]
            seq_rt_all = seq_rt_all[
                seq_rt_all["condition_norm"].isin(["congruent", "incongruent"])
                & seq_rt_all["prev_cond"].isin(["congruent", "incongruent"])
            ]

            seq_rt = seq_rt_all.copy()
            if "prev_correct" in seq_rt.columns:
                seq_rt = seq_rt[seq_rt["prev_correct"] == True]

            seq_rt = seq_rt[
                seq_rt["condition_norm"].isin(["congruent", "incongruent"])
                & seq_rt["prev_cond"].isin(["congruent", "incongruent"])
            ]

            def _seq_mean_rt(prev_cond: str, cond: str) -> float:
                subset = seq_rt[(seq_rt["prev_cond"] == prev_cond) & (seq_rt["condition_norm"] == cond)]
                return subset["rt_ms"].mean()

            rt_cc = _seq_mean_rt("congruent", "congruent")
            rt_ci = _seq_mean_rt("congruent", "incongruent")
            rt_ic = _seq_mean_rt("incongruent", "congruent")
            rt_ii = _seq_mean_rt("incongruent", "incongruent")
            if all(pd.notna(v) for v in (rt_cc, rt_ci, rt_ic, rt_ii)):
                cse_rt = (rt_ci - rt_cc) - (rt_ii - rt_ic)

            if not acc_grp.empty:
                seq_acc_all = acc_grp.copy()
                seq_acc_all["prev_cond"] = seq_acc_all["condition_norm"].shift(1)
                seq_acc_all["prev_correct"] = (
                    seq_acc_all["correct"].shift(1) if "correct" in seq_acc_all.columns else np.nan
                )
                seq_acc_all = seq_acc_all[
                    seq_acc_all["condition_norm"].isin(["congruent", "incongruent"])
                    & seq_acc_all["prev_cond"].isin(["congruent", "incongruent"])
                ]

                seq_acc = seq_acc_all.copy()
                if "prev_correct" in seq_acc.columns:
                    seq_acc = seq_acc[seq_acc["prev_correct"] == True]

                seq_acc = seq_acc[
                    seq_acc["condition_norm"].isin(["congruent", "incongruent"])
                    & seq_acc["prev_cond"].isin(["congruent", "incongruent"])
                ]

                def _seq_acc(prev_cond: str, cond: str) -> float:
                    subset = seq_acc[(seq_acc["prev_cond"] == prev_cond) & (seq_acc["condition_norm"] == cond)]
                    return subset["correct"].mean()

                acc_cc = _seq_acc("congruent", "congruent")
                acc_ci = _seq_acc("congruent", "incongruent")
                acc_ic = _seq_acc("incongruent", "congruent")
                acc_ii = _seq_acc("incongruent", "incongruent")
                if all(pd.notna(v) for v in (acc_cc, acc_ci, acc_ic, acc_ii)):
                    cse_acc = (acc_ci - acc_cc) - (acc_ii - acc_ic)

                prev_incong = seq_acc[seq_acc["prev_cond"] == "incongruent"]["correct"].mean()
                prev_cong = seq_acc[seq_acc["prev_cond"] == "congruent"]["correct"].mean()
                if pd.notna(prev_incong) and pd.notna(prev_cong):
                    post_conflict_accuracy = prev_incong - prev_cong

                post_error_trials = seq_acc_all[seq_acc_all["prev_correct"] == False]
                post_error_accuracy = post_error_trials["correct"].mean() if len(post_error_trials) else np.nan

                def _post_error_acc_diff(prev_flag: bool) -> float:
                    subset = seq_acc_all[seq_acc_all["prev_correct"] == prev_flag]
                    acc_incong = subset[subset["condition_norm"] == "incongruent"]["correct"].mean()
                    acc_cong = subset[subset["condition_norm"] == "congruent"]["correct"].mean()
                    if pd.notna(acc_incong) and pd.notna(acc_cong):
                        return acc_incong - acc_cong
                    return np.nan

                diff_error = _post_error_acc_diff(False)
                diff_correct = _post_error_acc_diff(True)
                if pd.notna(diff_error) and pd.notna(diff_correct):
                    post_error_interference_acc = diff_error - diff_correct

            prev_incong_rt = seq_rt[seq_rt["prev_cond"] == "incongruent"]["rt_ms"].mean()
            prev_cong_rt = seq_rt[seq_rt["prev_cond"] == "congruent"]["rt_ms"].mean()
            if pd.notna(prev_incong_rt) and pd.notna(prev_cong_rt):
                post_conflict_slowing = prev_incong_rt - prev_cong_rt

            def _post_error_rt_diff(prev_flag: bool) -> float:
                subset = seq_rt_all[seq_rt_all["prev_correct"] == prev_flag]
                rt_incong = subset[subset["condition_norm"] == "incongruent"]["rt_ms"].mean()
                rt_cong = subset[subset["condition_norm"] == "congruent"]["rt_ms"].mean()
                if pd.notna(rt_incong) and pd.notna(rt_cong):
                    return rt_incong - rt_cong
                return np.nan

            diff_error_rt = _post_error_rt_diff(False)
            diff_correct_rt = _post_error_rt_diff(True)
            if pd.notna(diff_error_rt) and pd.notna(diff_correct_rt):
                post_error_interference_rt = diff_error_rt - diff_correct_rt

        if not acc_grp.empty and "correct" in acc_grp.columns:
            error_runs = _run_lengths(~acc_grp["correct"])
            if error_runs:
                error_run_mean = float(np.mean(error_runs))
                error_run_max = float(np.max(error_runs))
            else:
                error_run_mean = 0.0
                error_run_max = 0.0

        if "correct" in grp.columns and "trial_order" in grp.columns:
            seq_full = grp.sort_values("trial_order").reset_index(drop=True)
            if seq_full["rt_ms"].notna().any():
                max_lag = 3
                lag_means = []
                for lag in range(1, max_lag + 1):
                    lag_rts = []
                    for i in range(len(seq_full) - lag):
                        if seq_full.loc[i, "correct"] == False:
                            rt_val = seq_full.loc[i + lag, "rt_ms"]
                            if (
                                pd.notna(rt_val)
                                and seq_full.loc[i + lag, "correct"] == True
                                and rt_val >= rt_min
                                and (rt_max_val is None or rt_val <= rt_max_val)
                            ):
                                lag_rts.append(rt_val)
                    lag_means.append(np.mean(lag_rts) if lag_rts else np.nan)
                if all(pd.notna(v) for v in lag_means):
                    error_recovery_rt = float(np.polyfit(range(1, max_lag + 1), lag_means, 1)[0])

            if seq_full["rt_ms"].notna().any():
                post_slopes = []
                error_indices = seq_full.index[seq_full["correct"] == False].tolist()
                for idx in error_indices:
                    window = []
                    j = idx + 1
                    while j < len(seq_full) and len(window) < 5:
                        rt_val = seq_full.loc[j, "rt_ms"]
                        if pd.notna(rt_val) and rt_val >= rt_min and (
                            rt_max_val is None or rt_val <= rt_max_val
                        ):
                            window.append(float(rt_val))
                        j += 1
                    if len(window) == 5:
                        x = np.arange(1, len(window) + 1)
                        post_slopes.append(float(np.polyfit(x, window, 1)[0]))
                if post_slopes:
                    post_error_recovery_slope = float(np.mean(post_slopes))

        raw_vals = raw_post_error.get(pid)
        if raw_vals:
            pes = raw_vals.get("stroop_post_error_slowing", pes)
            post_error_mean = raw_vals.get("stroop_post_error_rt", post_error_mean)
            post_correct_mean = raw_vals.get("stroop_post_correct_rt", post_correct_mean)
            post_error_accuracy = raw_vals.get("stroop_post_error_accuracy", post_error_accuracy)
            post_error_interference_rt = raw_vals.get(
                "stroop_post_error_interference_rt", post_error_interference_rt
            )
            post_error_interference_acc = raw_vals.get(
                "stroop_post_error_interference_acc", post_error_interference_acc
            )
            post_error_recovery_slope = raw_vals.get(
                "stroop_post_error_recovery_slope", post_error_recovery_slope
            )

        records.append({
            "participant_id": pid,
            "stroop_post_error_slowing": pes,
            "stroop_post_error_rt": post_error_mean,
            "stroop_post_correct_rt": post_correct_mean,
            "stroop_cse_rt": cse_rt,
            "stroop_cse_acc": cse_acc,
            "stroop_post_conflict_slowing": post_conflict_slowing,
            "stroop_post_conflict_accuracy": post_conflict_accuracy,
            "stroop_post_error_accuracy": post_error_accuracy,
            "stroop_post_error_interference_rt": post_error_interference_rt,
            "stroop_post_error_interference_acc": post_error_interference_acc,
            "stroop_error_run_length_mean": error_run_mean,
            "stroop_error_run_length_max": error_run_max,
            "stroop_error_recovery_rt": error_recovery_rt,
            "stroop_post_error_recovery_slope": post_error_recovery_slope,
        })

    return pd.DataFrame(records)
