"""PRP feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..constants import DEFAULT_RT_MIN, DEFAULT_SOA_LONG, DEFAULT_SOA_SHORT
from ..core import coefficient_of_variation
from .loaders import load_prp_trials
from .exgaussian_mechanism import load_or_compute_prp_mechanism_features
from .bottleneck_mechanism import load_or_compute_prp_bottleneck_mechanism_features


def _run_lengths(mask: pd.Series) -> List[int]:
    lengths: List[int] = []
    count = 0
    for val in mask:
        if val:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def derive_prp_features(
    rt_min: float = DEFAULT_RT_MIN,
    rt_max: float | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    prp, _ = load_prp_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=True,
    )
    prp_raw, _ = load_prp_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=False,
    )

    prp["t2_rt_ms"] = pd.to_numeric(prp.get("t2_rt", prp.get("t2_rt_ms")), errors="coerce")
    prp["soa_ms"] = pd.to_numeric(prp.get("soa", prp.get("soa_nominal_ms")), errors="coerce")
    prp["t1_rt_ms"] = pd.to_numeric(prp.get("t1_rt_ms", prp.get("t1_rt")), errors="coerce")

    prp_raw["t2_rt_ms"] = pd.to_numeric(prp_raw.get("t2_rt", prp_raw.get("t2_rt_ms")), errors="coerce")
    prp_raw["soa_ms"] = pd.to_numeric(prp_raw.get("soa", prp_raw.get("soa_nominal_ms")), errors="coerce")
    prp_raw["t1_rt_ms"] = pd.to_numeric(prp_raw.get("t1_rt_ms", prp_raw.get("t1_rt")), errors="coerce")

    for col in ["t1_correct", "t2_correct"]:
        if col in prp.columns and prp[col].dtype == "object":
            prp[col] = prp[col].map({
                True: True, "True": True, 1: True,
                False: False, "False": False, 0: False,
            })

    if "t1_correct" in prp_raw.columns and prp_raw["t1_correct"].dtype == "object":
        prp_raw["t1_correct"] = prp_raw["t1_correct"].map({
            True: True, "True": True, 1: True,
            False: False, "False": False, 0: False,
        })
    if "t2_correct" in prp_raw.columns and prp_raw["t2_correct"].dtype == "object":
        prp_raw["t2_correct"] = prp_raw["t2_correct"].map({
            True: True, "True": True, 1: True,
            False: False, "False": False, 0: False,
        })

    soa_levels = sorted(prp["soa_ms"].dropna().unique())

    records: List[Dict] = []
    raw_groups = {pid: grp for pid, grp in prp_raw.groupby("participant_id")}

    for pid, group in prp.groupby("participant_id"):
        raw_grp = raw_groups.get(pid, pd.DataFrame())
        overall_cv = coefficient_of_variation(group["t2_rt_ms"].dropna())
        short_soa = group[group["soa_ms"] <= DEFAULT_SOA_SHORT]
        long_soa = group[group["soa_ms"] >= DEFAULT_SOA_LONG]

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
            post_error = grp_sorted[grp_sorted["prev_t2_error"] == True]["t2_rt_ms"]
            post_correct = grp_sorted[grp_sorted["prev_t2_error"] == False]["t2_rt_ms"]
            pes = (
                post_error.mean() - post_correct.mean()
                if len(post_error) > 0 and len(post_correct) > 0
                else np.nan
            )
        else:
            pes = np.nan

        order_violation_rate = np.nan
        t1_only_rate = np.nan
        t2_only_rate = np.nan
        t2_pending_rate = np.nan
        iri_mean = np.nan
        iri_median = np.nan
        iri_p10 = np.nan
        iri_p25 = np.nan

        if not raw_grp.empty:
            if "order_norm" in raw_grp.columns:
                order_counts = raw_grp["order_norm"].value_counts(dropna=True)
                total_orders = int(order_counts.sum())
                if total_orders > 0:
                    order_violation_rate = order_counts.get("t2_t1", 0) / total_orders
                    t1_only_rate = order_counts.get("t1only", 0) / total_orders
                    t2_only_rate = order_counts.get("t2only", 0) / total_orders

            if "t2_pressed_while_t1_pending" in raw_grp.columns:
                t2_pending_rate = float(pd.to_numeric(raw_grp["t2_pressed_while_t1_pending"], errors="coerce").mean())

            iri_df = raw_grp.copy()
            for timeout_col in ("t1_timeout", "t2_timeout"):
                if timeout_col in iri_df.columns:
                    iri_df = iri_df[iri_df[timeout_col] == False]
            if "iri_ms" in iri_df.columns:
                iri_vals = pd.to_numeric(iri_df["iri_ms"], errors="coerce").dropna()
                if len(iri_vals) > 0:
                    iri_mean = float(iri_vals.mean())
                    iri_median = float(iri_vals.median())
                    iri_p10 = float(iri_vals.quantile(0.1))
                    iri_p25 = float(iri_vals.quantile(0.25))

        t1_cost = np.nan
        if "t1_rt_ms" in group.columns:
            t1_short = group[group["soa_ms"] <= DEFAULT_SOA_SHORT]["t1_rt_ms"].mean()
            t1_long = group[group["soa_ms"] >= DEFAULT_SOA_LONG]["t1_rt_ms"].mean()
            if pd.notna(t1_short) and pd.notna(t1_long):
                t1_cost = t1_short - t1_long

        rt1_rt2_coupling = np.nan
        if "t1_rt_ms" in group.columns:
            pair = group[["t1_rt_ms", "t2_rt_ms"]].dropna()
            if len(pair) >= 3:
                rt1_rt2_coupling = float(np.corrcoef(pair["t1_rt_ms"], pair["t2_rt_ms"])[0, 1])

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

        records.append({
            "participant_id": pid,
            "prp_t2_cv_all": overall_cv,
            "prp_t2_cv_short": coefficient_of_variation(short_soa["t2_rt_ms"].dropna()),
            "prp_t2_cv_long": coefficient_of_variation(long_soa["t2_rt_ms"].dropna()),
            "prp_cascade_rate": cascade_rate,
            "prp_cascade_inflation": cascade_inflation,
            "prp_pes": pes,
            "prp_t1_cost": t1_cost,
            "prp_rt1_rt2_coupling": rt1_rt2_coupling,
            "prp_t2_rt_t1_error": t2_rt_t1_error,
            "prp_t2_rt_t1_correct": t2_rt_t1_correct,
            "prp_t2_interference_t1_error": t2_interference_t1_error,
            "prp_order_violation_rate": order_violation_rate,
            "prp_t1_only_rate": t1_only_rate,
            "prp_t2_only_rate": t2_only_rate,
            "prp_t2_while_t1_pending_rate": t2_pending_rate,
            "prp_iri_mean": iri_mean,
            "prp_iri_median": iri_median,
            "prp_iri_p10": iri_p10,
            "prp_iri_p25": iri_p25,
            "prp_t1_error_run_mean": float(np.mean(t1_error_runs)) if t1_error_runs else 0.0,
            "prp_t1_error_run_max": float(np.max(t1_error_runs)) if t1_error_runs else 0.0,
            "prp_t2_error_run_mean": float(np.mean(t2_error_runs)) if t2_error_runs else 0.0,
            "prp_t2_error_run_max": float(np.max(t2_error_runs)) if t2_error_runs else 0.0,
            "prp_cascade_run_mean": float(np.mean(cascade_runs)) if cascade_runs else 0.0,
            "prp_cascade_run_max": float(np.max(cascade_runs)) if cascade_runs else 0.0,
            "prp_t2_trials": len(group),
        })

        if soa_levels:
            record = records[-1]
            for soa_val in soa_levels:
                label = int(round(float(soa_val)))
                subset = group[group["soa_ms"] == soa_val]
                record[f"prp_t2_rt_mean_soa_{label}"] = subset["t2_rt_ms"].mean()
                record[f"prp_t2_rt_median_soa_{label}"] = subset["t2_rt_ms"].median()
                record[f"prp_t2_rt_sd_soa_{label}"] = subset["t2_rt_ms"].std()
                if "t1_rt_ms" in subset.columns:
                    record[f"prp_t1_rt_mean_soa_{label}"] = subset["t1_rt_ms"].mean()
                    record[f"prp_t1_rt_median_soa_{label}"] = subset["t1_rt_ms"].median()
                    record[f"prp_t1_rt_sd_soa_{label}"] = subset["t1_rt_ms"].std()
                if "t1_correct" in subset.columns:
                    record[f"prp_t1_acc_soa_{label}"] = subset["t1_correct"].mean()
                if "t1_rt_ms" in subset.columns:
                    pair = subset[["t1_rt_ms", "t2_rt_ms"]].dropna()
                    if len(pair) >= 3:
                        record[f"prp_rt1_rt2_coupling_soa_{label}"] = float(
                            np.corrcoef(pair["t1_rt_ms"], pair["t2_rt_ms"])[0, 1]
                        )
                    else:
                        record[f"prp_rt1_rt2_coupling_soa_{label}"] = np.nan

    features_df = pd.DataFrame(records)

    mechanism_df = load_or_compute_prp_mechanism_features(data_dir=data_dir)
    if not mechanism_df.empty:
        if features_df.empty:
            features_df = mechanism_df
        else:
            overlap = [c for c in mechanism_df.columns if c != "participant_id" and c in features_df.columns]
            if overlap:
                features_df = features_df.drop(columns=overlap)
            features_df = features_df.merge(mechanism_df, on="participant_id", how="left")

    bottleneck_df = load_or_compute_prp_bottleneck_mechanism_features(data_dir=data_dir)
    if bottleneck_df.empty:
        return features_df
    if features_df.empty:
        return bottleneck_df

    overlap = [c for c in bottleneck_df.columns if c != "participant_id" and c in features_df.columns]
    if overlap:
        features_df = features_df.drop(columns=overlap)

    return features_df.merge(bottleneck_df, on="participant_id", how="left")
