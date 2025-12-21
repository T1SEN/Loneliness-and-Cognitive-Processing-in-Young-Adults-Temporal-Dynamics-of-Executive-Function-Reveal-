"""Stroop feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..constants import STROOP_RT_MIN
from ..core import coefficient_of_variation
from .loaders import load_stroop_trials
from .exgaussian_mechanism import load_or_compute_stroop_mechanism_features
from .lba_mechanism import load_or_compute_stroop_lba_mechanism_features


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
    stroop, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=True,
    )
    stroop_raw, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
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
    if "timeout" in stroop_acc.columns:
        stroop_acc = stroop_acc[stroop_acc["timeout"] == False]

    records: List[Dict] = []
    acc_groups = {pid: grp for pid, grp in stroop_acc.groupby("participant_id")}

    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()
        acc_grp = acc_groups.get(pid, pd.DataFrame())

        if "correct" in grp.columns:
            grp["prev_correct"] = grp["correct"].shift(1)
            post_error = grp[grp["prev_correct"] == False]
            post_correct = grp[grp["prev_correct"] == True]
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

        slope = np.nan
        cv_incong = np.nan
        cv_cong = np.nan
        if cond_col and "trial_order" in grp.columns:
            incong = grp[grp["condition_norm"] == "incongruent"]
            cong = grp[grp["condition_norm"] == "congruent"]

            incong_valid = incong.dropna(subset=["trial_order", "rt_ms"])
            if len(incong_valid) >= 5 and incong_valid["trial_order"].nunique() > 1:
                x = incong_valid["trial_order"].values
                y = incong_valid["rt_ms"].values
                coef = np.polyfit(x, y, 1)
                slope = coef[0]

            cv_incong = coefficient_of_variation(incong["rt_ms"].dropna())
            cv_cong = coefficient_of_variation(cong["rt_ms"].dropna())

        rt_correct = grp[grp["correct"] == True] if "correct" in grp.columns else grp

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
                            if pd.notna(rt_val) and seq_full.loc[i + lag, "correct"] == True:
                                lag_rts.append(rt_val)
                    lag_means.append(np.mean(lag_rts) if lag_rts else np.nan)
                if all(pd.notna(v) for v in lag_means):
                    error_recovery_rt = float(np.polyfit(range(1, max_lag + 1), lag_means, 1)[0])

        records.append({
            "participant_id": pid,
            "stroop_post_error_slowing": pes,
            "stroop_post_error_rt": post_error_mean,
            "stroop_post_correct_rt": post_correct_mean,
            "stroop_incong_slope": slope,
            "stroop_cv_all": coefficient_of_variation(grp["rt_ms"].dropna()),
            "stroop_cv_incong": cv_incong,
            "stroop_cv_cong": cv_cong,
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

    mechanism_df = load_or_compute_stroop_mechanism_features(data_dir=data_dir)
    if not mechanism_df.empty:
        if features_df.empty:
            features_df = mechanism_df
        else:
            overlap = [c for c in mechanism_df.columns if c != "participant_id" and c in features_df.columns]
            if overlap:
                features_df = features_df.drop(columns=overlap)
            features_df = features_df.merge(mechanism_df, on="participant_id", how="left")

    lba_df = load_or_compute_stroop_lba_mechanism_features(data_dir=data_dir)
    if lba_df.empty:
        return features_df
    if features_df.empty:
        return lba_df

    overlap = [c for c in lba_df.columns if c != "participant_id" and c in features_df.columns]
    if overlap:
        features_df = features_df.drop(columns=overlap)

    return features_df.merge(lba_df, on="participant_id", how="left")
