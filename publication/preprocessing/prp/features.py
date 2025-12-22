"""PRP feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..constants import PRP_RT_MIN, DEFAULT_SOA_LONG, DEFAULT_SOA_SHORT
from ..core import (
    coefficient_of_variation,
    median_absolute_deviation,
    interquartile_range,
    dfa_alpha,
    lag1_autocorrelation,
    run_length_stats,
    compute_error_cascade_metrics,
    compute_post_error_recovery_metrics,
    compute_momentum_metrics,
    compute_volatility_metrics,
    compute_iiv_parameters,
    compute_fatigue_slopes,
    compute_error_awareness_metrics,
)
from ..standardization import safe_zscore
from .loaders import load_prp_trials
from .exgaussian_mechanism import load_or_compute_prp_mechanism_features
from .hmm_event_features import load_or_compute_prp_hmm_event_features
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
    rt_min: float = PRP_RT_MIN,
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
    order_col = None
    for cand in ("trial_index", "trialIndex", "trial"):
        if cand in prp.columns:
            order_col = cand
            break

    records: List[Dict] = []
    raw_groups = {pid: grp for pid, grp in prp_raw.groupby("participant_id")}

    for pid, group in prp.groupby("participant_id"):
        raw_grp = raw_groups.get(pid, pd.DataFrame())
        grp_sorted = group.sort_values(order_col) if order_col and order_col in group.columns else group.sort_index()
        overall_cv = coefficient_of_variation(group["t2_rt_ms"].dropna())
        short_soa = group[group["soa_ms"] <= DEFAULT_SOA_SHORT]
        long_soa = group[group["soa_ms"] >= DEFAULT_SOA_LONG]

        if "t2_correct" in group.columns:
            group_correct = group[group["t2_correct"] == True]
        else:
            group_correct = group
        short_soa_correct = group_correct[group_correct["soa_ms"] <= DEFAULT_SOA_SHORT]
        long_soa_correct = group_correct[group_correct["soa_ms"] >= DEFAULT_SOA_LONG]

        if "t2_correct" in grp_sorted.columns:
            grp_sorted_correct = grp_sorted[grp_sorted["t2_correct"] == True]
        else:
            grp_sorted_correct = grp_sorted
        short_seq = grp_sorted_correct[grp_sorted_correct["soa_ms"] <= DEFAULT_SOA_SHORT]
        long_seq = grp_sorted_correct[grp_sorted_correct["soa_ms"] >= DEFAULT_SOA_LONG]
        rt_short_seq = short_seq["t2_rt_ms"].dropna()
        rt_long_seq = long_seq["t2_rt_ms"].dropna()

        dfa_short = dfa_alpha(rt_short_seq)
        dfa_long = dfa_alpha(rt_long_seq)
        lag1_short = lag1_autocorrelation(rt_short_seq)
        lag1_long = lag1_autocorrelation(rt_long_seq)
        run_short = run_length_stats(rt_short_seq)
        run_long = run_length_stats(rt_long_seq)

        mad_all = median_absolute_deviation(group["t2_rt_ms"].dropna())
        mad_short = median_absolute_deviation(short_soa["t2_rt_ms"].dropna())
        mad_long = median_absolute_deviation(long_soa["t2_rt_ms"].dropna())
        iqr_all = interquartile_range(group["t2_rt_ms"].dropna())
        iqr_short = interquartile_range(short_soa["t2_rt_ms"].dropna())
        iqr_long = interquartile_range(long_soa["t2_rt_ms"].dropna())

        mad_all_correct = median_absolute_deviation(group_correct["t2_rt_ms"].dropna())
        mad_short_correct = median_absolute_deviation(short_soa_correct["t2_rt_ms"].dropna())
        mad_long_correct = median_absolute_deviation(long_soa_correct["t2_rt_ms"].dropna())
        iqr_all_correct = interquartile_range(group_correct["t2_rt_ms"].dropna())
        iqr_short_correct = interquartile_range(short_soa_correct["t2_rt_ms"].dropna())
        iqr_long_correct = interquartile_range(long_soa_correct["t2_rt_ms"].dropna())

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

        vincentile_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        vincentile_labels = [10, 30, 50, 70, 90]
        vincentile_diffs = {f"prp_vincentile_bottleneck_p{label}_correct": np.nan for label in vincentile_labels}
        delta_plot_slope = np.nan

        short_vals = short_soa_correct["t2_rt_ms"].dropna()
        long_vals = long_soa_correct["t2_rt_ms"].dropna()
        if len(short_vals) >= 20 and len(long_vals) >= 20:
            short_q = np.quantile(short_vals, vincentile_quantiles)
            long_q = np.quantile(long_vals, vincentile_quantiles)
            diffs = short_q - long_q
            for label, diff in zip(vincentile_labels, diffs):
                vincentile_diffs[f"prp_vincentile_bottleneck_p{label}_correct"] = float(diff)
            delta_plot_slope = float(diffs[-1] - diffs[0])

        seq_rt = grp_sorted["t2_rt_ms"] if "t2_rt_ms" in grp_sorted.columns else pd.Series(dtype=float)
        seq_correct = grp_sorted["t2_correct"] if "t2_correct" in grp_sorted.columns else pd.Series(dtype=object)
        fatigue_metrics = compute_fatigue_slopes(seq_rt, seq_correct)
        cascade_metrics = compute_error_cascade_metrics(seq_correct)
        recovery_metrics = compute_post_error_recovery_metrics(seq_rt, seq_correct, max_lag=5)
        momentum_metrics = compute_momentum_metrics(seq_rt, seq_correct)
        volatility_metrics = compute_volatility_metrics(seq_rt)
        iiv_metrics = compute_iiv_parameters(seq_rt)
        awareness_metrics = compute_error_awareness_metrics(seq_rt, seq_correct)

        record = {
            "participant_id": pid,
            "prp_t2_cv_all": overall_cv,
            "prp_t2_cv_short": coefficient_of_variation(short_soa["t2_rt_ms"].dropna()),
            "prp_t2_cv_long": coefficient_of_variation(long_soa["t2_rt_ms"].dropna()),
            "prp_t2_mad_all": mad_all,
            "prp_t2_mad_short": mad_short,
            "prp_t2_mad_long": mad_long,
            "prp_t2_iqr_all": iqr_all,
            "prp_t2_iqr_short": iqr_short,
            "prp_t2_iqr_long": iqr_long,
            "prp_t2_mad_all_correct": mad_all_correct,
            "prp_t2_mad_short_correct": mad_short_correct,
            "prp_t2_mad_long_correct": mad_long_correct,
            "prp_t2_iqr_all_correct": iqr_all_correct,
            "prp_t2_iqr_short_correct": iqr_short_correct,
            "prp_t2_iqr_long_correct": iqr_long_correct,
            "prp_dfa_alpha_short_correct": dfa_short,
            "prp_dfa_alpha_long_correct": dfa_long,
            "prp_lag1_short_correct": lag1_short,
            "prp_lag1_long_correct": lag1_long,
            "prp_slow_run_mean_short_correct": run_short["slow_run_mean"],
            "prp_slow_run_max_short_correct": run_short["slow_run_max"],
            "prp_fast_run_mean_short_correct": run_short["fast_run_mean"],
            "prp_fast_run_max_short_correct": run_short["fast_run_max"],
            "prp_slow_run_mean_long_correct": run_long["slow_run_mean"],
            "prp_slow_run_max_long_correct": run_long["slow_run_max"],
            "prp_fast_run_mean_long_correct": run_long["fast_run_mean"],
            "prp_fast_run_max_long_correct": run_long["fast_run_max"],
            "prp_cascade_rate": cascade_rate,
            "prp_cascade_inflation": cascade_inflation,
            "prp_pes": pes,
            "prp_rt_fatigue_slope": fatigue_metrics["rt_fatigue_slope"],
            "prp_cv_fatigue_slope": fatigue_metrics["cv_fatigue_slope"],
            "prp_acc_fatigue_slope": fatigue_metrics["acc_fatigue_slope"],
            "prp_error_cascade_count": cascade_metrics["error_cascade_count"],
            "prp_error_cascade_rate": cascade_metrics["error_cascade_rate"],
            "prp_error_cascade_mean_len": cascade_metrics["error_cascade_mean_len"],
            "prp_error_cascade_max_len": cascade_metrics["error_cascade_max_len"],
            "prp_error_cascade_trials": cascade_metrics["error_cascade_trials"],
            "prp_error_cascade_prop": cascade_metrics["error_cascade_prop"],
            "prp_recovery_rt_lag1": recovery_metrics["recovery_rt_lag1"],
            "prp_recovery_rt_lag2": recovery_metrics["recovery_rt_lag2"],
            "prp_recovery_rt_lag3": recovery_metrics["recovery_rt_lag3"],
            "prp_recovery_rt_lag4": recovery_metrics["recovery_rt_lag4"],
            "prp_recovery_rt_lag5": recovery_metrics["recovery_rt_lag5"],
            "prp_recovery_acc_lag1": recovery_metrics["recovery_acc_lag1"],
            "prp_recovery_acc_lag2": recovery_metrics["recovery_acc_lag2"],
            "prp_recovery_acc_lag3": recovery_metrics["recovery_acc_lag3"],
            "prp_recovery_acc_lag4": recovery_metrics["recovery_acc_lag4"],
            "prp_recovery_acc_lag5": recovery_metrics["recovery_acc_lag5"],
            "prp_recovery_rt_slope": recovery_metrics["recovery_rt_slope"],
            "prp_recovery_acc_slope": recovery_metrics["recovery_acc_slope"],
            "prp_momentum_slope": momentum_metrics["momentum_slope"],
            "prp_momentum_mean_streak": momentum_metrics["momentum_mean_streak"],
            "prp_momentum_max_streak": momentum_metrics["momentum_max_streak"],
            "prp_momentum_rt_streak_0": momentum_metrics["momentum_rt_streak_0"],
            "prp_momentum_rt_streak_1": momentum_metrics["momentum_rt_streak_1"],
            "prp_momentum_rt_streak_2": momentum_metrics["momentum_rt_streak_2"],
            "prp_momentum_rt_streak_3": momentum_metrics["momentum_rt_streak_3"],
            "prp_momentum_rt_streak_4": momentum_metrics["momentum_rt_streak_4"],
            "prp_momentum_rt_streak_5": momentum_metrics["momentum_rt_streak_5"],
            "prp_volatility_rmssd": volatility_metrics["volatility_rmssd"],
            "prp_volatility_adj": volatility_metrics["volatility_adj"],
            "prp_intercept": iiv_metrics["iiv_intercept"],
            "prp_slope": iiv_metrics["iiv_slope"],
            "prp_slope_p": iiv_metrics["iiv_slope_p"],
            "prp_residual_sd": iiv_metrics["iiv_residual_sd"],
            "prp_raw_cv": iiv_metrics["iiv_raw_cv"],
            "prp_iiv_trials": iiv_metrics["iiv_n_trials"],
            "prp_iiv_r_squared": iiv_metrics["iiv_r_squared"],
            "prp_post_error_cv": awareness_metrics["post_error_cv"],
            "prp_post_correct_cv": awareness_metrics["post_correct_cv"],
            "prp_post_error_cv_reduction": awareness_metrics["post_error_cv_reduction"],
            "prp_post_error_accuracy": awareness_metrics["post_error_acc"],
            "prp_post_correct_accuracy": awareness_metrics["post_correct_acc"],
            "prp_post_error_acc_diff": awareness_metrics["post_error_acc_diff"],
            "prp_post_error_recovery_rate": awareness_metrics["post_error_recovery_rate"],
            "prp_pes_adaptive": awareness_metrics["pes_adaptive"],
            "prp_pes_maladaptive": awareness_metrics["pes_maladaptive"],
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
            "prp_delta_plot_slope_bottleneck_correct": delta_plot_slope,
        }
        record.update(vincentile_diffs)
        records.append(record)

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
    if not features_df.empty:
        required_cols = [
            "prp_post_error_cv_reduction",
            "prp_post_error_accuracy",
            "prp_post_error_recovery_rate",
        ]
        if all(col in features_df.columns for col in required_cols):
            z_cv = safe_zscore(features_df["prp_post_error_cv_reduction"])
            z_acc = safe_zscore(features_df["prp_post_error_accuracy"])
            z_rec = safe_zscore(features_df["prp_post_error_recovery_rate"])
            features_df["prp_error_awareness_index"] = (z_cv + z_acc + z_rec) / 3

    mechanism_df = load_or_compute_prp_mechanism_features(data_dir=data_dir)
    if not mechanism_df.empty:
        if features_df.empty:
            features_df = mechanism_df
        else:
            overlap = [c for c in mechanism_df.columns if c != "participant_id" and c in features_df.columns]
            if overlap:
                features_df = features_df.drop(columns=overlap)
            features_df = features_df.merge(mechanism_df, on="participant_id", how="left")

    hmm_df = load_or_compute_prp_hmm_event_features(data_dir=data_dir)
    if not hmm_df.empty:
        if features_df.empty:
            features_df = hmm_df
        else:
            overlap = [c for c in hmm_df.columns if c != "participant_id" and c in features_df.columns]
            if overlap:
                features_df = features_df.drop(columns=overlap)
            features_df = features_df.merge(hmm_df, on="participant_id", how="left")

    bottleneck_df = load_or_compute_prp_bottleneck_mechanism_features(data_dir=data_dir)
    if bottleneck_df.empty:
        return features_df
    if features_df.empty:
        return bottleneck_df

    overlap = [c for c in bottleneck_df.columns if c != "participant_id" and c in features_df.columns]
    if overlap:
        features_df = features_df.drop(columns=overlap)

    return features_df.merge(bottleneck_df, on="participant_id", how="left")
