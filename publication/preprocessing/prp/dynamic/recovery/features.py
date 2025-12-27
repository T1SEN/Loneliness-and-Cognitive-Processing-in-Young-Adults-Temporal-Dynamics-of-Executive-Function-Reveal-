"""PRP recovery feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ....constants import PRP_RT_MIN, PRP_RT_MAX, RAW_DIR
from ..._shared import _run_lengths, prepare_prp_trials


def derive_prp_recovery_features(
    rt_min: float = PRP_RT_MIN,
    rt_max: float | None = None,
    data_dir: Path | None = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    rt_max_val = PRP_RT_MAX if rt_max is None else rt_max

    def _rt_valid(series: pd.Series) -> pd.Series:
        valid = series.notna() & (series >= rt_min)
        if rt_max_val is not None:
            valid &= series <= rt_max_val
        return valid

    if prepared is None:
        prepared = prepare_prp_trials(rt_min=rt_min, rt_max=rt_max, data_dir=data_dir)

    prp = prepared["prp"]
    prp_acc = prepared.get("prp_acc", prp)
    raw_post_error = {}
    if isinstance(prp_acc, pd.DataFrame) and not prp_acc.empty:
        pids = set(prp_acc["participant_id"].unique())
        raw_prepared = prepare_prp_trials(rt_min=rt_min, rt_max=rt_max, data_dir=RAW_DIR)
        raw_df = raw_prepared.get("prp_raw")
        if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
            raw_df = raw_df[raw_df["participant_id"].isin(pids)].copy()
            for timeout_col in ("t1_timeout", "t2_timeout"):
                if timeout_col in raw_df.columns:
                    raw_df[timeout_col] = (
                        raw_df[timeout_col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map({"true": True, "1": True, "false": False, "0": False})
                        .fillna(False)
                        .astype(bool)
                    )
                    raw_df = raw_df[raw_df[timeout_col] == False]
            if "t2_correct" in raw_df.columns and raw_df["t2_correct"].dtype == "object":
                raw_df["t2_correct"] = raw_df["t2_correct"].map({
                    True: True, "True": True, 1: True,
                    False: False, "False": False, 0: False,
                })
            if "t2_rt_ms" not in raw_df.columns:
                raw_df["t2_rt_ms"] = pd.to_numeric(
                    raw_df.get("t2_rt", raw_df.get("t2_rt_ms")),
                    errors="coerce",
                )

            order_col = None
            for cand in ("trial_order", "trial_index", "trialIndex", "idx", "trial", "trialIndexInBlock"):
                if cand in raw_df.columns:
                    order_col = cand
                    break
            if order_col:
                raw_df["trial_order"] = pd.to_numeric(raw_df[order_col], errors="coerce")

            for pid, grp in raw_df.groupby("participant_id"):
                if "trial_order" in grp.columns:
                    grp = grp.sort_values("trial_order")
                else:
                    grp = grp.sort_index()
                grp = grp.reset_index(drop=True)

                pes = np.nan
                post_error_accuracy = np.nan
                post_error_accuracy_drop = np.nan

                if "t2_correct" in grp.columns:
                    grp["prev_t2_error"] = (~grp["t2_correct"]).shift(1)
                    rt_valid = _rt_valid(grp["t2_rt_ms"])

                    post_error_rt = grp.loc[rt_valid & (grp["prev_t2_error"] == True), "t2_rt_ms"].mean()
                    post_correct_rt = grp.loc[rt_valid & (grp["prev_t2_error"] == False), "t2_rt_ms"].mean()
                    if pd.notna(post_error_rt) and pd.notna(post_correct_rt):
                        pes = post_error_rt - post_correct_rt

                    post_error_accuracy = grp.loc[grp["prev_t2_error"] == True, "t2_correct"].mean()
                    post_correct_accuracy = grp.loc[grp["prev_t2_error"] == False, "t2_correct"].mean()
                    if pd.notna(post_error_accuracy) and pd.notna(post_correct_accuracy):
                        post_error_accuracy_drop = float(post_error_accuracy - post_correct_accuracy)

                raw_post_error[pid] = {
                    "prp_pes": pes,
                    "prp_post_error_accuracy": post_error_accuracy,
                    "prp_post_error_accuracy_drop": post_error_accuracy_drop,
                }
    if not isinstance(prp_acc, pd.DataFrame) or prp_acc.empty:
        return pd.DataFrame()

    records: List[Dict] = []

    for pid, group in prp_acc.groupby("participant_id"):
        if "t1_correct" in group.columns and "t2_correct" in group.columns:
            valid = group[(group["t1_correct"].notna()) & (group["t2_correct"].notna())]
            t1_errors = valid[~valid["t1_correct"]]
            n_t1_errors = len(t1_errors)

            if n_t1_errors == 0:
                cascade_rate = np.nan
                cascade_inflation = np.nan
            else:
                t1_and_t2_errors = t1_errors[~t1_errors["t2_correct"]]
                n_cascade = len(t1_and_t2_errors)
                cascade_rate = n_cascade / n_t1_errors
                t2_error_rate = 1 - valid["t2_correct"].mean() if len(valid) > 0 else np.nan
                cascade_inflation = (
                    cascade_rate / t2_error_rate
                    if pd.notna(t2_error_rate) and t2_error_rate > 0
                    else np.nan
                )
        else:
            cascade_rate = np.nan
            cascade_inflation = np.nan

        grp_sorted = group.sort_values("trial_index" if "trial_index" in group.columns else "trial")
        if "t2_correct" in grp_sorted.columns:
            grp_sorted["prev_t2_error"] = (~grp_sorted["t2_correct"]).shift(1).fillna(False)
            rt_valid = _rt_valid(grp_sorted["t2_rt_ms"])
            post_error = grp_sorted[rt_valid & (grp_sorted["prev_t2_error"] == True)]["t2_rt_ms"]
            post_correct = grp_sorted[rt_valid & (grp_sorted["prev_t2_error"] == False)]["t2_rt_ms"]
            pes = (
                post_error.mean() - post_correct.mean()
                if len(post_error) > 0 and len(post_correct) > 0
                else np.nan
            )
            post_error_acc = grp_sorted[grp_sorted["prev_t2_error"] == True]["t2_correct"].mean()
            post_correct_acc = grp_sorted[grp_sorted["prev_t2_error"] == False]["t2_correct"].mean()
            if pd.notna(post_error_acc) and pd.notna(post_correct_acc):
                post_error_accuracy_drop = float(post_error_acc - post_correct_acc)
            else:
                post_error_accuracy_drop = np.nan
            post_error_accuracy = float(post_error_acc) if pd.notna(post_error_acc) else np.nan
        else:
            pes = np.nan
            post_error_accuracy = np.nan
            post_error_accuracy_drop = np.nan

        t2_rt_t1_error = np.nan
        t2_rt_t1_correct = np.nan
        t2_interference_t1_error = np.nan
        if "t1_correct" in group.columns:
            t2_rt_t1_error = group[~group["t1_correct"]]["t2_rt_ms"].mean()
            t2_rt_t1_correct = group[group["t1_correct"]]["t2_rt_ms"].mean()
            if pd.notna(t2_rt_t1_error) and pd.notna(t2_rt_t1_correct):
                t2_interference_t1_error = t2_rt_t1_error - t2_rt_t1_correct

        t1_error_runs = _run_lengths(~group["t1_correct"]) if "t1_correct" in group.columns else []
        t2_error_runs = _run_lengths(~group["t2_correct"]) if "t2_correct" in group.columns else []
        cascade_runs = _run_lengths((~group["t1_correct"]) & (~group["t2_correct"])) if (
            "t1_correct" in group.columns and "t2_correct" in group.columns
        ) else []

        raw_vals = raw_post_error.get(pid)
        if raw_vals:
            pes = raw_vals.get("prp_pes", pes)
            post_error_accuracy = raw_vals.get("prp_post_error_accuracy", post_error_accuracy)
            post_error_accuracy_drop = raw_vals.get("prp_post_error_accuracy_drop", post_error_accuracy_drop)

        records.append({
            "participant_id": pid,
            "prp_cascade_rate": cascade_rate,
            "prp_cascade_inflation": cascade_inflation,
            "prp_pes": pes,
            "prp_post_error_accuracy": post_error_accuracy,
            "prp_post_error_accuracy_drop": post_error_accuracy_drop,
            "prp_t2_rt_t1_error": t2_rt_t1_error,
            "prp_t2_rt_t1_correct": t2_rt_t1_correct,
            "prp_t2_interference_t1_error": t2_interference_t1_error,
            "prp_t1_error_run_mean": float(np.mean(t1_error_runs)) if t1_error_runs else 0.0,
            "prp_t1_error_run_max": float(np.max(t1_error_runs)) if t1_error_runs else 0.0,
            "prp_t2_error_run_mean": float(np.mean(t2_error_runs)) if t2_error_runs else 0.0,
            "prp_t2_error_run_max": float(np.max(t2_error_runs)) if t2_error_runs else 0.0,
            "prp_cascade_run_mean": float(np.mean(cascade_runs)) if cascade_runs else 0.0,
            "prp_cascade_run_max": float(np.max(cascade_runs)) if cascade_runs else 0.0,
        })

    return pd.DataFrame(records)
