"""Stroop feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..constants import STROOP_RT_MIN, STROOP_RT_MAX
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
    compute_tau_quartile_metrics,
    compute_error_awareness_metrics,
    compute_pre_error_slope_metrics,
    compute_speed_accuracy_metrics,
)
from ..standardization import safe_zscore
from .loaders import load_stroop_trials
from .exgaussian_mechanism import load_or_compute_stroop_mechanism_features
from .hmm_event_features import load_or_compute_stroop_hmm_event_features


def _normalize_condition(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    mapping = {
        "cong": "congruent",
        "congruent": "congruent",
        "inc": "incongruent",
        "incong": "incongruent",
        "incongruent": "incongruent",
        "neutral": "neutral",
        "neut": "neutral",
        "neu": "neutral",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if "incong" in cleaned:
        return "incongruent"
    if "cong" in cleaned:
        return "congruent"
    if "neut" in cleaned:
        return "neutral"
    return None


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


def _ez_ddm_params(rt_correct: pd.Series, acc: float, n_trials: int, s: float = 0.1) -> Dict[str, float]:
    if n_trials < 10:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}
    if rt_correct.empty:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}
    mrt = rt_correct.mean()
    vrt = rt_correct.var(ddof=1)
    if pd.isna(mrt) or pd.isna(vrt) or vrt <= 0:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}

    eps = 1.0 / (2.0 * n_trials)
    p = float(np.clip(acc, eps, 1.0 - eps))
    s2 = s ** 2
    logit = np.log(p / (1 - p))
    x = (logit * (p ** 2 * logit - p * logit + p - 0.5)) / vrt
    if x <= 0:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}
    v = np.sign(p - 0.5) * s * (x ** 0.25)
    a = (s2 * logit) / v
    y = -v * a / s2
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    t0 = mrt - mdt
    return {"v": float(v), "a": float(a), "t0": float(t0)}


def derive_stroop_features(
    rt_min: float = STROOP_RT_MIN,
    rt_max: float | None = None,
    data_dir: None | str | Path = None,
) -> pd.DataFrame:
    rt_max_val = STROOP_RT_MAX if rt_max is None else rt_max
    stroop, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max_val,
        apply_trial_filters=True,
    )
    stroop_raw, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max_val,
        apply_trial_filters=False,
    )

    for df in (stroop, stroop_raw):
        if "rt" in df.columns:
            df["rt_ms"] = pd.to_numeric(df["rt"], errors="coerce")

    trial_col = None
    for cand in ("trial", "trialIndex", "idx", "trial_index"):
        if cand in stroop.columns or cand in stroop_raw.columns:
            trial_col = cand
            break
    if trial_col:
        stroop["trial_order"] = pd.to_numeric(stroop[trial_col], errors="coerce")
        stroop_raw["trial_order"] = pd.to_numeric(stroop_raw[trial_col], errors="coerce")
        stroop = stroop.sort_values(["participant_id", "trial_order"])
        stroop_raw = stroop_raw.sort_values(["participant_id", "trial_order"])

    cond_col = None
    for cand in ("type", "condition", "cond", "congruency"):
        if cand in stroop.columns or cand in stroop_raw.columns:
            cond_col = cand
            break
    if cond_col:
        stroop["condition_norm"] = stroop[cond_col].apply(_normalize_condition)
        stroop_raw["condition_norm"] = stroop_raw[cond_col].apply(_normalize_condition)

    stroop_acc = stroop_raw.copy()
    if "timeout" in stroop_acc.columns and "correct" in stroop_acc.columns:
        stroop_acc["correct"] = stroop_acc["correct"] & (~stroop_acc["timeout"])

    def _rt_valid(series: pd.Series) -> pd.Series:
        valid = series.notna() & (series >= rt_min)
        if rt_max_val is not None:
            valid &= series <= rt_max_val
        return valid

    records: List[Dict] = []
    acc_groups = {pid: grp for pid, grp in stroop_acc.groupby("participant_id")}

    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()
        acc_grp = acc_groups.get(pid, pd.DataFrame())
        grp_sorted = grp.sort_values("trial_order") if "trial_order" in grp.columns else grp.sort_index()

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

        rt_all = grp["rt_ms"].dropna()
        mad_all = median_absolute_deviation(rt_all)
        iqr_all = interquartile_range(rt_all)

        slope = np.nan
        cv_incong = np.nan
        cv_cong = np.nan
        rt_sd_incong = np.nan
        mad_incong = np.nan
        iqr_incong = np.nan
        mad_cong = np.nan
        iqr_cong = np.nan
        if cond_col:
            incong = grp[grp["condition_norm"] == "incongruent"]
            cong = grp[grp["condition_norm"] == "congruent"]

            if "trial_order" in grp.columns:
                incong_valid = incong.dropna(subset=["trial_order", "rt_ms"])
                if len(incong_valid) >= 5 and incong_valid["trial_order"].nunique() > 1:
                    x = incong_valid["trial_order"].values
                    y = incong_valid["rt_ms"].values
                    coef = np.polyfit(x, y, 1)
                    slope = coef[0]

            cv_incong = coefficient_of_variation(incong["rt_ms"].dropna())
            cv_cong = coefficient_of_variation(cong["rt_ms"].dropna())
            rt_sd_vals = incong["rt_ms"].dropna()
            rt_sd_incong = float(rt_sd_vals.std(ddof=1)) if len(rt_sd_vals) > 1 else np.nan
            mad_incong = median_absolute_deviation(incong["rt_ms"].dropna())
            iqr_incong = interquartile_range(incong["rt_ms"].dropna())
            mad_cong = median_absolute_deviation(cong["rt_ms"].dropna())
            iqr_cong = interquartile_range(cong["rt_ms"].dropna())

        rt_correct = grp_sorted[grp_sorted["correct"] == True] if "correct" in grp_sorted.columns else grp_sorted
        rt_correct_all = rt_correct["rt_ms"].dropna()
        temporal_all = run_length_stats(rt_correct_all)
        dfa_all = dfa_alpha(rt_correct_all)
        lag1_all = lag1_autocorrelation(rt_correct_all)

        dfa_incong = np.nan
        lag1_incong = np.nan
        temporal_incong = {
            "slow_run_mean": np.nan,
            "slow_run_max": np.nan,
            "fast_run_mean": np.nan,
            "fast_run_max": np.nan,
        }
        if cond_col:
            incong_correct = rt_correct[rt_correct["condition_norm"] == "incongruent"]
            rt_incong = incong_correct["rt_ms"].dropna()
            dfa_incong = dfa_alpha(rt_incong)
            lag1_incong = lag1_autocorrelation(rt_incong)
            temporal_incong = run_length_stats(rt_incong)
        rt_correct_all = rt_correct["rt_ms"].dropna()
        mad_all_correct = median_absolute_deviation(rt_correct_all)
        iqr_all_correct = interquartile_range(rt_correct_all)
        mad_incong_correct = np.nan
        iqr_incong_correct = np.nan
        mad_cong_correct = np.nan
        iqr_cong_correct = np.nan
        if cond_col:
            incong_correct = rt_correct[rt_correct["condition_norm"] == "incongruent"]
            cong_correct = rt_correct[rt_correct["condition_norm"] == "congruent"]
            mad_incong_correct = median_absolute_deviation(incong_correct["rt_ms"].dropna())
            iqr_incong_correct = interquartile_range(incong_correct["rt_ms"].dropna())
            mad_cong_correct = median_absolute_deviation(cong_correct["rt_ms"].dropna())
            iqr_cong_correct = interquartile_range(cong_correct["rt_ms"].dropna())

        rt_interference = np.nan
        rt_facilitation = np.nan
        rt_incong_neutral = np.nan
        rt_neutral_cong = np.nan
        rt_interference_correct = np.nan
        rt_facilitation_correct = np.nan
        rt_incong_neutral_correct = np.nan
        rt_neutral_cong_correct = np.nan
        acc_interference = np.nan
        acc_facilitation = np.nan

        if cond_col:
            for df, suffix in ((grp, ""), (rt_correct, "_correct")):
                rt_cong = df.loc[df["condition_norm"] == "congruent", "rt_ms"].mean()
                rt_incong = df.loc[df["condition_norm"] == "incongruent", "rt_ms"].mean()
                rt_neutral = df.loc[df["condition_norm"] == "neutral", "rt_ms"].mean()

                if pd.notna(rt_incong) and pd.notna(rt_cong):
                    if suffix:
                        rt_interference_correct = rt_incong - rt_cong
                    else:
                        rt_interference = rt_incong - rt_cong
                if pd.notna(rt_neutral) and pd.notna(rt_cong):
                    if suffix:
                        rt_facilitation_correct = rt_neutral - rt_cong
                        rt_neutral_cong_correct = rt_neutral - rt_cong
                    else:
                        rt_facilitation = rt_neutral - rt_cong
                        rt_neutral_cong = rt_neutral - rt_cong
                if pd.notna(rt_incong) and pd.notna(rt_neutral):
                    if suffix:
                        rt_incong_neutral_correct = rt_incong - rt_neutral
                    else:
                        rt_incong_neutral = rt_incong - rt_neutral

            if not acc_grp.empty and "correct" in acc_grp.columns:
                acc_cong = acc_grp.loc[acc_grp["condition_norm"] == "congruent", "correct"].mean()
                acc_incong = acc_grp.loc[acc_grp["condition_norm"] == "incongruent", "correct"].mean()
                acc_neutral = acc_grp.loc[acc_grp["condition_norm"] == "neutral", "correct"].mean()
                if pd.notna(acc_incong) and pd.notna(acc_cong):
                    acc_interference = acc_incong - acc_cong
                if pd.notna(acc_neutral) and pd.notna(acc_cong):
                    acc_facilitation = acc_neutral - acc_cong

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
        ez_params = {}

        if cond_col and "trial_order" in grp.columns:
            seq = grp.copy()
            seq["prev_cond"] = seq["condition_norm"].shift(1)
            seq["prev_correct"] = seq["correct"].shift(1) if "correct" in seq.columns else np.nan

            seq_rt_all = seq[seq["rt_ms"].notna()].copy()
            if "correct" in seq_rt_all.columns:
                seq_rt_all = seq_rt_all[seq_rt_all["correct"] == True]
            seq_rt_all = seq_rt_all[
                seq_rt_all["condition_norm"].isin(["congruent", "incongruent"])
                & seq_rt_all["prev_cond"].isin(["congruent", "incongruent"])
            ]
            seq_rt_all_pe = seq_rt_all[_rt_valid(seq_rt_all["rt_ms"])]

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
                subset = seq_rt_all_pe[seq_rt_all_pe["prev_correct"] == prev_flag]
                rt_incong = subset[subset["condition_norm"] == "incongruent"]["rt_ms"].mean()
                rt_cong = subset[subset["condition_norm"] == "congruent"]["rt_ms"].mean()
                if pd.notna(rt_incong) and pd.notna(rt_cong):
                    return rt_incong - rt_cong
                return np.nan

            diff_error_rt = _post_error_rt_diff(False)
            diff_correct_rt = _post_error_rt_diff(True)
            if pd.notna(diff_error_rt) and pd.notna(diff_correct_rt):
                post_error_interference_rt = diff_error_rt - diff_correct_rt

        if cond_col:
            for cond in ("congruent", "incongruent", "neutral"):
                cond_mask = grp["condition_norm"] == cond
                n_trials = int(cond_mask.sum())
                if "correct" in grp.columns:
                    acc = grp.loc[cond_mask, "correct"].mean()
                    rt_correct = grp.loc[cond_mask & (grp["correct"] == True), "rt_ms"]
                else:
                    acc = np.nan
                    rt_correct = pd.Series(dtype=float)
                params = _ez_ddm_params(rt_correct, acc, n_trials)
                ez_params[cond] = params

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

        seq_rt = grp_sorted["rt_ms"] if "rt_ms" in grp_sorted.columns else pd.Series(dtype=float)
        seq_correct = grp_sorted["correct"] if "correct" in grp_sorted.columns else pd.Series(dtype=object)

        fatigue_metrics = compute_fatigue_slopes(seq_rt, seq_correct)
        tau_metrics = compute_tau_quartile_metrics(seq_rt, seq_correct if "correct" in grp_sorted.columns else None)
        cascade_metrics = compute_error_cascade_metrics(seq_correct)
        recovery_metrics = compute_post_error_recovery_metrics(seq_rt, seq_correct, max_lag=5)
        momentum_metrics = compute_momentum_metrics(seq_rt, seq_correct)
        volatility_metrics = compute_volatility_metrics(seq_rt)
        iiv_metrics = compute_iiv_parameters(seq_rt)
        awareness_metrics = compute_error_awareness_metrics(seq_rt, seq_correct)
        speed_metrics = compute_speed_accuracy_metrics(seq_rt, seq_correct)
        pre_error_metrics = compute_pre_error_slope_metrics(seq_rt, seq_correct)

        accuracy_all = np.nan
        error_rate_all = np.nan
        timeout_rate = np.nan
        error_rate_non_timeout = np.nan
        if not acc_grp.empty and "correct" in acc_grp.columns:
            accuracy_all = float(acc_grp["correct"].mean())
            timeout_col = "is_timeout" if "is_timeout" in acc_grp.columns else ("timeout" if "timeout" in acc_grp.columns else None)
            if timeout_col is not None:
                timeout_rate = float(acc_grp[timeout_col].mean())
                error_rate_non_timeout = float(((~acc_grp[timeout_col]) & (~acc_grp["correct"])).mean())
            error_rate_all = float(1.0 - accuracy_all)

        ies = np.nan
        if pd.notna(speed_metrics["mean_rt"]) and pd.notna(accuracy_all) and accuracy_all > 0:
            ies = float(speed_metrics["mean_rt"] / accuracy_all)

        records.append({
            "participant_id": pid,
            "stroop_post_error_slowing": pes,
            "stroop_post_error_rt": post_error_mean,
            "stroop_post_correct_rt": post_correct_mean,
            "stroop_mean_rt_all": speed_metrics["mean_rt"],
            "stroop_accuracy_all": accuracy_all,
            "stroop_error_rate_all": error_rate_all,
            "stroop_timeout_rate": timeout_rate,
            "stroop_error_rate_non_timeout": error_rate_non_timeout,
            "stroop_ies": ies,
            "stroop_pre_error_slope_mean": pre_error_metrics["pre_error_slope_mean"],
            "stroop_pre_error_slope_std": pre_error_metrics["pre_error_slope_std"],
            "stroop_pre_error_n": pre_error_metrics["pre_error_n"],
            "stroop_incong_slope": slope,
            "stroop_cv_all": coefficient_of_variation(grp["rt_ms"].dropna()),
            "stroop_cv_incong": cv_incong,
            "stroop_cv_cong": cv_cong,
            "stroop_rt_sd_incong": rt_sd_incong,
            "stroop_mad_all": mad_all,
            "stroop_iqr_all": iqr_all,
            "stroop_mad_incong": mad_incong,
            "stroop_iqr_incong": iqr_incong,
            "stroop_mad_cong": mad_cong,
            "stroop_iqr_cong": iqr_cong,
            "stroop_mad_all_correct": mad_all_correct,
            "stroop_iqr_all_correct": iqr_all_correct,
            "stroop_mad_incong_correct": mad_incong_correct,
            "stroop_iqr_incong_correct": iqr_incong_correct,
            "stroop_mad_cong_correct": mad_cong_correct,
            "stroop_iqr_cong_correct": iqr_cong_correct,
            "stroop_dfa_alpha_correct": dfa_all,
            "stroop_lag1_correct": lag1_all,
            "stroop_slow_run_mean_correct": temporal_all["slow_run_mean"],
            "stroop_slow_run_max_correct": temporal_all["slow_run_max"],
            "stroop_fast_run_mean_correct": temporal_all["fast_run_mean"],
            "stroop_fast_run_max_correct": temporal_all["fast_run_max"],
            "stroop_dfa_alpha_incong_correct": dfa_incong,
            "stroop_lag1_incong_correct": lag1_incong,
            "stroop_slow_run_mean_incong_correct": temporal_incong["slow_run_mean"],
            "stroop_slow_run_max_incong_correct": temporal_incong["slow_run_max"],
            "stroop_fast_run_mean_incong_correct": temporal_incong["fast_run_mean"],
            "stroop_fast_run_max_incong_correct": temporal_incong["fast_run_max"],
            "stroop_rt_interference": rt_interference,
            "stroop_rt_facilitation": rt_facilitation,
            "stroop_rt_incong_minus_neutral": rt_incong_neutral,
            "stroop_rt_neutral_minus_cong": rt_neutral_cong,
            "stroop_rt_interference_correct": rt_interference_correct,
            "stroop_rt_facilitation_correct": rt_facilitation_correct,
            "stroop_rt_incong_minus_neutral_correct": rt_incong_neutral_correct,
            "stroop_rt_neutral_minus_cong_correct": rt_neutral_cong_correct,
            "stroop_acc_interference": acc_interference,
            "stroop_acc_facilitation": acc_facilitation,
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
            "stroop_rt_fatigue_slope": fatigue_metrics["rt_fatigue_slope"],
            "stroop_cv_fatigue_slope": fatigue_metrics["cv_fatigue_slope"],
            "stroop_acc_fatigue_slope": fatigue_metrics["acc_fatigue_slope"],
            "stroop_tau_q1": tau_metrics["tau_q1"],
            "stroop_tau_q2": tau_metrics["tau_q2"],
            "stroop_tau_q3": tau_metrics["tau_q3"],
            "stroop_tau_q4": tau_metrics["tau_q4"],
            "stroop_tau_slope": tau_metrics["tau_slope"],
            "stroop_error_cascade_count": cascade_metrics["error_cascade_count"],
            "stroop_error_cascade_rate": cascade_metrics["error_cascade_rate"],
            "stroop_error_cascade_mean_len": cascade_metrics["error_cascade_mean_len"],
            "stroop_error_cascade_max_len": cascade_metrics["error_cascade_max_len"],
            "stroop_error_cascade_trials": cascade_metrics["error_cascade_trials"],
            "stroop_error_cascade_prop": cascade_metrics["error_cascade_prop"],
            "stroop_recovery_rt_lag1": recovery_metrics["recovery_rt_lag1"],
            "stroop_recovery_rt_lag2": recovery_metrics["recovery_rt_lag2"],
            "stroop_recovery_rt_lag3": recovery_metrics["recovery_rt_lag3"],
            "stroop_recovery_rt_lag4": recovery_metrics["recovery_rt_lag4"],
            "stroop_recovery_rt_lag5": recovery_metrics["recovery_rt_lag5"],
            "stroop_recovery_acc_lag1": recovery_metrics["recovery_acc_lag1"],
            "stroop_recovery_acc_lag2": recovery_metrics["recovery_acc_lag2"],
            "stroop_recovery_acc_lag3": recovery_metrics["recovery_acc_lag3"],
            "stroop_recovery_acc_lag4": recovery_metrics["recovery_acc_lag4"],
            "stroop_recovery_acc_lag5": recovery_metrics["recovery_acc_lag5"],
            "stroop_recovery_rt_slope": recovery_metrics["recovery_rt_slope"],
            "stroop_recovery_acc_slope": recovery_metrics["recovery_acc_slope"],
            "stroop_momentum_slope": momentum_metrics["momentum_slope"],
            "stroop_momentum_mean_streak": momentum_metrics["momentum_mean_streak"],
            "stroop_momentum_max_streak": momentum_metrics["momentum_max_streak"],
            "stroop_momentum_rt_streak_0": momentum_metrics["momentum_rt_streak_0"],
            "stroop_momentum_rt_streak_1": momentum_metrics["momentum_rt_streak_1"],
            "stroop_momentum_rt_streak_2": momentum_metrics["momentum_rt_streak_2"],
            "stroop_momentum_rt_streak_3": momentum_metrics["momentum_rt_streak_3"],
            "stroop_momentum_rt_streak_4": momentum_metrics["momentum_rt_streak_4"],
            "stroop_momentum_rt_streak_5": momentum_metrics["momentum_rt_streak_5"],
            "stroop_volatility_rmssd": volatility_metrics["volatility_rmssd"],
            "stroop_volatility_adj": volatility_metrics["volatility_adj"],
            "stroop_intercept": iiv_metrics["iiv_intercept"],
            "stroop_slope": iiv_metrics["iiv_slope"],
            "stroop_slope_p": iiv_metrics["iiv_slope_p"],
            "stroop_residual_sd": iiv_metrics["iiv_residual_sd"],
            "stroop_raw_cv": iiv_metrics["iiv_raw_cv"],
            "stroop_iiv_trials": iiv_metrics["iiv_n_trials"],
            "stroop_iiv_r_squared": iiv_metrics["iiv_r_squared"],
            "stroop_post_error_cv": awareness_metrics["post_error_cv"],
            "stroop_post_correct_cv": awareness_metrics["post_correct_cv"],
            "stroop_post_error_cv_reduction": awareness_metrics["post_error_cv_reduction"],
            "stroop_post_error_acc_diff": awareness_metrics["post_error_acc_diff"],
            "stroop_post_error_recovery_rate": awareness_metrics["post_error_recovery_rate"],
            "stroop_pes_adaptive": awareness_metrics["pes_adaptive"],
            "stroop_pes_maladaptive": awareness_metrics["pes_maladaptive"],
            "stroop_trials": len(grp),
        })

        if ez_params:
            record = records[-1]
            for cond in ("congruent", "incongruent", "neutral"):
                params = ez_params.get(cond, {})
                record[f"stroop_ez_{cond}_v"] = params.get("v", np.nan)
                record[f"stroop_ez_{cond}_a"] = params.get("a", np.nan)
                record[f"stroop_ez_{cond}_t0"] = params.get("t0", np.nan)
                record[f"stroop_ez_{cond}_sv"] = 0.0
                record[f"stroop_ez_{cond}_sz"] = 0.0
                record[f"stroop_ez_{cond}_st0"] = 0.0

            if (
                pd.notna(record.get("stroop_ez_incongruent_v"))
                and pd.notna(record.get("stroop_ez_congruent_v"))
            ):
                record["stroop_ez_v_interference"] = (
                    record["stroop_ez_incongruent_v"] - record["stroop_ez_congruent_v"]
                )
            else:
                record["stroop_ez_v_interference"] = np.nan

            if (
                pd.notna(record.get("stroop_ez_incongruent_a"))
                and pd.notna(record.get("stroop_ez_congruent_a"))
            ):
                record["stroop_ez_a_interference"] = (
                    record["stroop_ez_incongruent_a"] - record["stroop_ez_congruent_a"]
                )
            else:
                record["stroop_ez_a_interference"] = np.nan

            if (
                pd.notna(record.get("stroop_ez_incongruent_t0"))
                and pd.notna(record.get("stroop_ez_congruent_t0"))
            ):
                record["stroop_ez_t0_interference"] = (
                    record["stroop_ez_incongruent_t0"] - record["stroop_ez_congruent_t0"]
                )
            else:
                record["stroop_ez_t0_interference"] = np.nan

    features_df = pd.DataFrame(records)
    if not features_df.empty:
        required_cols = [
            "stroop_post_error_cv_reduction",
            "stroop_post_error_accuracy",
            "stroop_post_error_recovery_rate",
        ]
        if all(col in features_df.columns for col in required_cols):
            z_cv = safe_zscore(features_df["stroop_post_error_cv_reduction"])
            z_acc = safe_zscore(features_df["stroop_post_error_accuracy"])
            z_rec = safe_zscore(features_df["stroop_post_error_recovery_rate"])
            features_df["stroop_error_awareness_index"] = (z_cv + z_acc + z_rec) / 3

    mechanism_df = load_or_compute_stroop_mechanism_features(data_dir=data_dir)
    if not mechanism_df.empty:
        if features_df.empty:
            features_df = mechanism_df
        else:
            overlap = [c for c in mechanism_df.columns if c != "participant_id" and c in features_df.columns]
            if overlap:
                features_df = features_df.drop(columns=overlap)
            features_df = features_df.merge(mechanism_df, on="participant_id", how="left")

    hmm_df = load_or_compute_stroop_hmm_event_features(data_dir=data_dir)
    if not hmm_df.empty:
        if features_df.empty:
            features_df = hmm_df
        else:
            overlap = [c for c in hmm_df.columns if c != "participant_id" and c in features_df.columns]
            if overlap:
                features_df = features_df.drop(columns=overlap)
            features_df = features_df.merge(hmm_df, on="participant_id", how="left")

    return features_df
