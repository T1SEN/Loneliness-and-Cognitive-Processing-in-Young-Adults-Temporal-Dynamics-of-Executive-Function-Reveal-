"""WCST recovery feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..._shared import _run_lengths, prepare_wcst_trials
from ....constants import RAW_DIR, WCST_RT_MIN, WCST_RT_MAX
from ....core import ensure_participant_id


def derive_wcst_recovery_features(
    data_dir: None | str | Path = None,
    filter_rt: bool = False,
    rt_max: float | None = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    rt_max_val = WCST_RT_MAX if rt_max is None else rt_max

    def _rt_valid(series: pd.Series) -> pd.Series:
        valid = series.notna() & (series >= WCST_RT_MIN)
        if rt_max_val is not None:
            valid &= series <= rt_max_val
        return valid

    if prepared is None:
        prepared = prepare_wcst_trials(data_dir=data_dir, filter_rt=filter_rt)

    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    rule_col = prepared["rule_col"]
    trial_col = prepared["trial_col"]

    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None:
        return pd.DataFrame()

    raw_post_error: dict[str, dict[str, float]] = {}
    pids = set(wcst["participant_id"].unique())
    raw_path = RAW_DIR / "4b_wcst_trials.csv"
    if raw_path.exists() and pids:
        raw_df = pd.read_csv(raw_path, encoding="utf-8-sig")
        raw_df = ensure_participant_id(raw_df)
        raw_df = raw_df[raw_df["participant_id"].isin(pids)].copy()

        if "trial_index" not in raw_df.columns and "trialIndex" in raw_df.columns:
            raw_df = raw_df.rename(columns={"trialIndex": "trial_index"})

        if "rt_ms" not in raw_df.columns:
            for cand in ("reactionTimeMs", "resp_time_ms"):
                if cand in raw_df.columns:
                    raw_df["rt_ms"] = raw_df[cand]
                    break
        raw_df["rt_ms"] = pd.to_numeric(raw_df.get("rt_ms"), errors="coerce")

        if "timeout" in raw_df.columns:
            raw_df["timeout"] = (
                raw_df["timeout"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "1": True, "false": False, "0": False})
                .fillna(False)
                .astype(bool)
            )
            raw_df = raw_df[raw_df["timeout"] == False]

        if "correct" in raw_df.columns:
            raw_df["correct"] = (
                raw_df["correct"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "1": True, "false": False, "0": False})
                .fillna(False)
                .astype(bool)
            )

        if "trial_index" in raw_df.columns:
            raw_df["trial_order"] = pd.to_numeric(raw_df["trial_index"], errors="coerce")
        else:
            raw_df["trial_order"] = np.arange(len(raw_df))

        for pid, grp in raw_df.groupby("participant_id"):
            if "trial_order" in grp.columns:
                grp = grp.sort_values("trial_order")
            else:
                grp = grp.sort_index()
            grp = grp.reset_index(drop=True)

            pes = np.nan
            post_error_accuracy_drop = np.nan

            if "correct" in grp.columns:
                grp["prev_error"] = (~grp["correct"]).shift(1)
                rt_valid = _rt_valid(grp["rt_ms"])

                post_error_rt = grp.loc[rt_valid & (grp["prev_error"] == True), "rt_ms"].mean()
                post_correct_rt = grp.loc[rt_valid & (grp["prev_error"] == False), "rt_ms"].mean()
                if pd.notna(post_error_rt) and pd.notna(post_correct_rt):
                    pes = post_error_rt - post_correct_rt

                post_error_acc = grp.loc[grp["prev_error"] == True, "correct"].mean()
                post_correct_acc = grp.loc[grp["prev_error"] == False, "correct"].mean()
                if pd.notna(post_error_acc) and pd.notna(post_correct_acc):
                    post_error_accuracy_drop = float(post_error_acc - post_correct_acc)

            raw_post_error[pid] = {
                "wcst_pes": pes,
                "wcst_post_error_accuracy_drop": post_error_accuracy_drop,
            }

    records: List[Dict] = []
    for pid, grp in wcst.groupby("participant_id"):
        if trial_col and trial_col in grp.columns:
            grp = grp.sort_values(trial_col)
        grp = grp.reset_index(drop=True)

        total_trials = int(len(grp))
        has_correct = "correct" in grp.columns
        correct = (
            grp["correct"].astype(bool).values
            if has_correct
            else np.zeros(total_trials, dtype=bool)
        )
        rt_vals = pd.to_numeric(grp[rt_col], errors="coerce")
        if "is_rt_valid" in grp.columns:
            rt_vals = rt_vals.where(grp["is_rt_valid"].astype(bool))
        else:
            rt_vals = rt_vals.where(_rt_valid(rt_vals))
        rt_vals = rt_vals.values.astype(float)

        pes = np.nan
        post_error_accuracy_drop = np.nan
        if has_correct:
            post_pe_rts = []
            post_correct_rts = []
            post_error_acc = []
            post_correct_acc = []
            for i in range(len(grp) - 1):
                if correct[i] == False:
                    post_error_acc.append(int(correct[i + 1]))
                    rt_next = rt_vals[i + 1]
                    if np.isfinite(rt_next) and rt_next >= WCST_RT_MIN and (
                        rt_max_val is None or rt_next <= rt_max_val
                    ):
                        post_pe_rts.append(rt_next)
                elif correct[i] == True:
                    post_correct_acc.append(int(correct[i + 1]))
                    rt_next = rt_vals[i + 1]
                    if np.isfinite(rt_next) and rt_next >= WCST_RT_MIN and (
                        rt_max_val is None or rt_next <= rt_max_val
                    ):
                        post_correct_rts.append(rt_next)
            if post_pe_rts and post_correct_rts:
                pes = np.mean(post_pe_rts) - np.mean(post_correct_rts)
            if post_error_acc and post_correct_acc:
                post_error_accuracy_drop = np.mean(post_error_acc) - np.mean(post_correct_acc)

        post_switch_error_rate = np.nan
        post_switch_error_delta = np.nan
        post_switch_pe_rate = np.nan
        post_switch_rt_cost_correct = np.nan
        trials_to_first_correct_after_shift = np.nan

        errors = ~correct
        total_errors = int(errors.sum()) if total_trials else 0

        is_pe = grp["isPE"].astype(bool).values if "isPE" in grp.columns else np.zeros(total_trials, dtype=bool)
        is_pr = grp["isPR"].astype(bool).values if "isPR" in grp.columns else np.zeros(total_trials, dtype=bool)
        is_npe = grp["isNPE"].astype(bool).values if "isNPE" in grp.columns else np.zeros(total_trials, dtype=bool)
        has_is_npe = "isNPE" in grp.columns

        pe_count = int(is_pe.sum())
        pr_count = int(is_pr.sum())
        npe_count = int(is_npe.sum()) if has_is_npe else max(total_errors - pe_count, 0)

        pe_runs = _run_lengths(is_pe)
        pr_runs = _run_lengths(is_pr)
        pe_run_mean = float(np.mean(pe_runs)) if pe_runs else 0.0
        pe_run_max = float(np.max(pe_runs)) if pe_runs else 0.0
        pr_run_mean = float(np.mean(pr_runs)) if pr_runs else 0.0
        pr_run_max = float(np.max(pr_runs)) if pr_runs else 0.0
        pe_cluster_rate = (pe_run_max / pe_count) if pe_count else np.nan

        rt_jump_at_switch = np.nan
        switch_cost_rt = {k: [] for k in range(1, 6)}
        switch_cost_err = {k: [] for k in range(1, 6)}
        trials_to_reacq = np.nan
        post_switch_recovery_slope = np.nan

        if rule_col:
            rules = grp[rule_col].astype(str).str.lower().values
            change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
            post_switch_errors = []
            post_switch_error_deltas = []
            post_switch_pe_rates = []
            post_switch_rt_costs_correct = []
            trials_to_first_correct = []

            for idx in change_indices:
                post_end = min(idx + 5, len(correct))
                post_slice = slice(idx, post_end)
                post_vals = correct[post_slice]
                if len(post_vals) >= 3:
                    post_error_rate = 1 - post_vals.mean()
                    post_switch_errors.append(post_error_rate)
                    post_switch_pe_rates.append(np.mean(is_pe[post_slice]))
                else:
                    post_error_rate = np.nan

                pre_start = max(0, idx - 3)
                pre_slice = slice(pre_start, idx)
                pre_vals = correct[pre_slice]
                if len(pre_vals) >= 3:
                    pre_error_rate = 1 - pre_vals.mean()
                else:
                    pre_error_rate = np.nan

                if np.isfinite(post_error_rate) and np.isfinite(pre_error_rate):
                    post_switch_error_deltas.append(post_error_rate - pre_error_rate)

                pre_rt = rt_vals[pre_slice]
                post_rt = rt_vals[post_slice]
                pre_mask = pre_vals & np.isfinite(pre_rt)
                post_mask = post_vals & np.isfinite(post_rt)
                pre_rt_corr = pre_rt[pre_mask]
                post_rt_corr = post_rt[post_mask]
                if len(pre_rt_corr) >= 2 and len(post_rt_corr) >= 2:
                    post_switch_rt_costs_correct.append(
                        float(np.mean(post_rt_corr) - np.mean(pre_rt_corr))
                    )

                for j in range(idx, len(correct)):
                    if correct[j]:
                        trials_to_first_correct.append(float((j - idx) + 1))
                        break

            if post_switch_errors:
                post_switch_error_rate = float(np.nanmean(post_switch_errors))
            if post_switch_error_deltas:
                post_switch_error_delta = float(np.nanmean(post_switch_error_deltas))
            if post_switch_pe_rates:
                post_switch_pe_rate = float(np.nanmean(post_switch_pe_rates))
            if post_switch_rt_costs_correct:
                post_switch_rt_cost_correct = float(np.nanmean(post_switch_rt_costs_correct))
            if trials_to_first_correct:
                trials_to_first_correct_after_shift = float(np.nanmean(trials_to_first_correct))

            rt_jump_vals = []
            for idx in change_indices:
                pre_start = max(0, idx - 3)
                pre_rt = rt_vals[pre_start:idx]
                post_rt = rt_vals[idx: min(idx + 3, len(rt_vals))]
                if len(pre_rt) and len(post_rt):
                    pre_mean = np.nanmean(pre_rt)
                    post_mean = np.nanmean(post_rt)
                    if np.isfinite(pre_mean) and np.isfinite(post_mean):
                        rt_jump_vals.append(float(post_mean - pre_mean))

                for k in range(1, 6):
                    if idx + k < len(rt_vals):
                        err_val = 1 - int(correct[idx + k])
                        switch_cost_err[k].append(err_val)
                        if len(pre_rt) and np.isfinite(rt_vals[idx + k]):
                            pre_mean = np.nanmean(pre_rt)
                            if np.isfinite(pre_mean):
                                switch_cost_rt[k].append(float(rt_vals[idx + k] - pre_mean))

            for k in range(1, 6):
                switch_cost_err[k] = float(np.mean(switch_cost_err[k])) if switch_cost_err[k] else np.nan
                switch_cost_rt[k] = float(np.mean(switch_cost_rt[k])) if switch_cost_rt[k] else np.nan

            if rt_jump_vals:
                rt_jump_at_switch = float(np.mean(rt_jump_vals))

            post_slopes = []
            for idx in change_indices:
                window = []
                j = idx
                while j < len(rt_vals) and len(window) < 5:
                    rt_val = rt_vals[j]
                    if np.isfinite(rt_val):
                        window.append(float(rt_val))
                    j += 1
                if len(window) == 5:
                    x = np.arange(1, len(window) + 1)
                    post_slopes.append(float(np.polyfit(x, window, 1)[0]))
            if post_slopes:
                post_switch_recovery_slope = float(np.mean(post_slopes))

            reacq_vals = []
            for idx in change_indices:
                found = False
                for j in range(idx, len(correct) - 2):
                    if correct[j] and correct[j + 1] and correct[j + 2]:
                        reacq_vals.append(float((j + 3) - idx))
                        found = True
                        break
                if not found:
                    continue
            if reacq_vals:
                trials_to_reacq = float(np.mean(reacq_vals))

        failure_to_maintain_set = np.nan
        if total_trials and has_is_npe:
            ftm = 0
            consecutive_correct = 0
            in_fms_eligible = False
            fms_episode_active = False
            rules = grp[rule_col].astype(str).str.lower().values if rule_col else None
            prev_rule = None

            for i in range(total_trials):
                if rules is not None:
                    rule = rules[i]
                    if prev_rule is None:
                        prev_rule = rule
                    elif rule != prev_rule:
                        consecutive_correct = 0
                        in_fms_eligible = False
                        fms_episode_active = False
                        prev_rule = rule

                is_correct = bool(correct[i])
                is_npe_flag = bool(is_npe[i])
                if (not is_correct) and is_npe_flag and in_fms_eligible and (not fms_episode_active):
                    ftm += 1
                    fms_episode_active = True

                if is_correct:
                    consecutive_correct += 1
                    if consecutive_correct == 5:
                        in_fms_eligible = True
                        fms_episode_active = False
                    if fms_episode_active:
                        fms_episode_active = False
                else:
                    consecutive_correct = 0
                    in_fms_eligible = False

            failure_to_maintain_set = float(ftm)

        raw_vals = raw_post_error.get(pid)
        if raw_vals:
            pes = raw_vals.get("wcst_pes", pes)
            post_error_accuracy_drop = raw_vals.get(
                "wcst_post_error_accuracy_drop", post_error_accuracy_drop
            )

        record = {
            "participant_id": pid,
            "wcst_pes": pes,
            "wcst_post_error_accuracy_drop": post_error_accuracy_drop,
            "wcst_post_switch_error_rate": post_switch_error_rate,
            "wcst_post_switch_error_delta": post_switch_error_delta,
            "wcst_post_switch_pe_rate": post_switch_pe_rate,
            "wcst_post_switch_rt_cost_correct": post_switch_rt_cost_correct,
            "wcst_trials_to_first_correct_after_shift": trials_to_first_correct_after_shift,
            "wcst_pe_run_length_mean": pe_run_mean,
            "wcst_pe_run_length_max": pe_run_max,
            "wcst_pr_run_length_mean": pr_run_mean,
            "wcst_pr_run_length_max": pr_run_max,
            "wcst_pe_cluster_rate": pe_cluster_rate,
            "wcst_rt_jump_at_switch": rt_jump_at_switch,
            "wcst_failure_to_maintain_set": failure_to_maintain_set,
            "wcst_trials_to_rule_reacquisition": trials_to_reacq,
            "wcst_post_switch_recovery_slope": post_switch_recovery_slope,
        }

        for k in range(1, 6):
            record[f"wcst_switch_cost_error_k{k}"] = switch_cost_err[k]
            record[f"wcst_switch_cost_rt_k{k}"] = switch_cost_rt[k]

        records.append(record)

    return pd.DataFrame(records)
