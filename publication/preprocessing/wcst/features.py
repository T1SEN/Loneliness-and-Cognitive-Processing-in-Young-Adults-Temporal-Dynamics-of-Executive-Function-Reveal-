"""WCST feature derivation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..constants import get_results_dir
from ..core import coefficient_of_variation, ensure_participant_id
from .loaders import load_wcst_trials
from .hmm_mechanism import load_or_compute_wcst_hmm_mechanism_features
from .rl_mechanism import load_or_compute_wcst_rl_mechanism_features
from .wsls_mechanism import load_or_compute_wcst_wsls_mechanism_features
from .bayesianrl_mechanism import load_or_compute_wcst_bayesianrl_mechanism_features


def _run_lengths(mask: np.ndarray) -> List[int]:
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


def _load_wcst_summary_metrics(data_dir: Path | None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("wcst")
    summary_path = Path(data_dir) / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()

    summary = pd.read_csv(summary_path, encoding="utf-8")
    summary = ensure_participant_id(summary)
    if "testName" not in summary.columns:
        return pd.DataFrame()
    summary["testName"] = summary["testName"].astype(str).str.strip().str.lower()
    summary = summary[summary["testName"] == "wcst"].copy()
    if summary.empty:
        return pd.DataFrame()

    rename = {
        "completedCategories": "wcst_completed_categories",
        "totalErrorCount": "wcst_total_errors",
        "totalTrialCount": "wcst_total_trials_summary",
        "totalCorrectCount": "wcst_total_correct_summary",
        "perseverativeResponses": "wcst_perseverative_responses",
        "perseverativeResponsesPercent": "wcst_perseverative_response_percent",
        "perseverativeErrorCount": "wcst_perseverative_errors",
        "nonPerseverativeErrorCount": "wcst_nonperseverative_errors",
        "trialsToCompleteFirstCategory": "wcst_trials_to_first_category",
        "failureToMaintainSet": "wcst_failure_to_maintain_set",
        "conceptualLevelResponses": "wcst_clr_count",
        "conceptualLevelResponsesPercent": "wcst_clr_percent",
        "learningToLearn": "wcst_learning_to_learn",
        "learningToLearnHeatonClrDelta": "wcst_learning_to_learn_heaton_clr_delta",
        "learningEfficiencyDeltaTrials": "wcst_learning_efficiency_delta_trials",
        "trialsToFirstConceptualResp": "wcst_trials_to_first_conceptual_resp",
        "trialsToFirstConceptualResp0": "wcst_trials_to_first_conceptual_resp0",
        "hasFirstCLR": "wcst_has_first_clr",
        "categoryClrPercents": "wcst_category_clr_percents",
    }
    cols = [c for c in rename if c in summary.columns]
    summary = summary[["participant_id"] + cols].rename(columns=rename)
    for col in summary.columns:
        if col == "participant_id" or col == "wcst_category_clr_percents":
            continue
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    def _parse_list(val: object) -> List[float]:
        if isinstance(val, list):
            return [float(x) for x in val]
        if not isinstance(val, str):
            return []
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        return []

    clr_lists = summary["wcst_category_clr_percents"].apply(_parse_list)
    summary["wcst_category_clr_percents"] = clr_lists.apply(lambda x: x if x else np.nan)

    delta_clr = []
    slope_clr = []
    for vals in clr_lists:
        if not vals or len(vals) < 3:
            delta_clr.append(np.nan)
            slope_clr.append(np.nan)
            continue
        first = np.mean(vals[:3])
        last = np.mean(vals[-3:])
        delta_clr.append(float(first - last))
        if len(vals) >= 2:
            x = np.arange(1, len(vals) + 1)
            slope = np.polyfit(x, vals, 1)[0]
            slope_clr.append(float(slope))
        else:
            slope_clr.append(np.nan)

    summary["wcst_delta_clr_percent_first3_last3"] = delta_clr
    summary["wcst_learning_slope_clr_percent"] = slope_clr

    return summary


def derive_wcst_features(
    data_dir: None | str | Path = None,
    filter_rt: bool = False,
) -> pd.DataFrame:
    if filter_rt:
        wcst, _ = load_wcst_trials(data_dir=data_dir, clean=True, filter_rt=True, apply_trial_filters=False)
    else:
        wcst, _ = load_wcst_trials(data_dir=data_dir, apply_trial_filters=True)

    rt_col = None
    for cand in ("reactionTimeMs", "rt_ms", "reaction_time_ms", "rt"):
        if cand in wcst.columns:
            rt_col = cand
            break
    if not rt_col:
        return pd.DataFrame(columns=["participant_id", "wcst_pes", "wcst_post_switch_error_rate", "wcst_cv_rt", "wcst_trials"])

    trial_col = None
    for cand in ("trialIndex", "trial_index", "trial"):
        if cand in wcst.columns:
            trial_col = cand
            break

    if trial_col:
        wcst = wcst.sort_values(["participant_id", trial_col])

    rule_col = None
    for cand in ("ruleAtThatTime", "rule_at_that_time", "rule_at_time", "rule"):
        if cand in wcst.columns:
            rule_col = cand
            break

    records: List[Dict] = []
    for pid, grp in wcst.groupby("participant_id"):
        grp = grp.reset_index(drop=True)

        pes = np.nan
        if "correct" in grp.columns:
            rt_vals = grp[rt_col].values
            correct = grp["correct"].values
            post_pe_rts = []
            post_correct_rts = []
            for i in range(len(grp) - 1):
                if correct[i] == False:
                    post_pe_rts.append(rt_vals[i + 1])
                elif correct[i] == True:
                    post_correct_rts.append(rt_vals[i + 1])
            if post_pe_rts and post_correct_rts:
                pes = np.mean(post_pe_rts) - np.mean(post_correct_rts)

        post_switch_error_rate = np.nan
        if rule_col and "correct" in grp.columns:
            rules = grp[rule_col].values
            is_correct = grp["correct"].values
            rule_changes = []
            for i in range(1, len(rules)):
                if rules[i] != rules[i - 1]:
                    rule_changes.append(i)

            post_switch_errors = []
            for change_idx in rule_changes:
                window_end = min(change_idx + 5, len(grp))
                post_switch_trials = is_correct[change_idx:window_end]
                if len(post_switch_trials) >= 3:
                    post_switch_errors.append(1 - post_switch_trials.mean())

            if post_switch_errors:
                post_switch_error_rate = np.nanmean(post_switch_errors)

        total_trials = int(len(grp))
        correct = grp["correct"].astype(bool).values if "correct" in grp.columns else np.zeros(total_trials, dtype=bool)
        errors = ~correct
        total_errors = int(errors.sum()) if total_trials else 0
        error_rate = (total_errors / total_trials) if total_trials else np.nan

        is_pe = grp["isPE"].astype(bool).values if "isPE" in grp.columns else np.zeros(total_trials, dtype=bool)
        is_pr = grp["isPR"].astype(bool).values if "isPR" in grp.columns else np.zeros(total_trials, dtype=bool)
        is_npe = grp["isNPE"].astype(bool).values if "isNPE" in grp.columns else np.zeros(total_trials, dtype=bool)

        pe_count = int(is_pe.sum())
        pr_count = int(is_pr.sum())
        npe_count = int(is_npe.sum()) if "isNPE" in grp.columns else max(total_errors - pe_count, 0)

        pe_rate = (pe_count / total_trials) * 100 if total_trials else np.nan
        pr_percent = (pr_count / total_trials) * 100 if total_trials else np.nan
        error_pr_ratio = (int((is_pr & errors).sum()) / total_errors) if total_errors else np.nan
        error_npe_ratio = (npe_count / total_errors) if total_errors else np.nan

        pe_runs = _run_lengths(is_pe)
        pr_runs = _run_lengths(is_pr)
        pe_run_mean = float(np.mean(pe_runs)) if pe_runs else 0.0
        pe_run_max = float(np.max(pe_runs)) if pe_runs else 0.0
        pr_run_mean = float(np.mean(pr_runs)) if pr_runs else 0.0
        pr_run_max = float(np.max(pr_runs)) if pr_runs else 0.0
        pe_cluster_rate = (pe_run_max / pe_count) if pe_count else np.nan

        trials_per_category = []
        trials_to_first_category = np.nan
        categories_completed = np.nan
        rt_slope_within = np.nan
        acc_slope_within = np.nan
        rt_jump_at_switch = np.nan
        switch_cost_rt = {k: [] for k in range(1, 6)}
        switch_cost_err = {k: [] for k in range(1, 6)}
        trials_to_reacq = np.nan

        if rule_col:
            rules = grp[rule_col].astype(str).str.lower().values
            change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
            segment_starts = [0] + change_indices
            segment_ends = change_indices + [len(rules)]
            trials_per_category = [end - start for start, end in zip(segment_starts, segment_ends)]
            categories_completed = float(len(trials_per_category)) if trials_per_category else np.nan
            if trials_per_category:
                trials_to_first_category = float(trials_per_category[0])

            rt_jump_vals = []
            rt_slope_vals = []
            acc_slope_vals = []
            rt_vals = pd.to_numeric(grp[rt_col], errors="coerce").values.astype(float)

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

            for idx in change_indices:
                pre_start = max(0, idx - 3)
                pre_rt = rt_vals[pre_start:idx]
                post_rt = rt_vals[idx: min(idx + 3, len(rt_vals))]
                if len(pre_rt) and len(post_rt):
                    rt_jump_vals.append(float(np.nanmean(post_rt) - np.nanmean(pre_rt)))

                for k in range(1, 6):
                    if idx + k < len(rt_vals):
                        err_val = 1 - int(correct[idx + k])
                        switch_cost_err[k].append(err_val)
                        if len(pre_rt):
                            switch_cost_rt[k].append(float(rt_vals[idx + k] - np.nanmean(pre_rt)))

            for k in range(1, 6):
                switch_cost_err[k] = float(np.mean(switch_cost_err[k])) if switch_cost_err[k] else np.nan
                switch_cost_rt[k] = float(np.mean(switch_cost_rt[k])) if switch_cost_rt[k] else np.nan

            if rt_jump_vals:
                rt_jump_at_switch = float(np.mean(rt_jump_vals))

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
        if rule_col and total_trials:
            ftm = 0
            streak = 0
            rules = grp[rule_col].astype(str).str.lower().values
            for i in range(total_trials):
                if i > 0 and rules[i] != rules[i - 1]:
                    streak = 0
                if correct[i]:
                    streak += 1
                else:
                    if streak >= 5:
                        ftm += 1
                    streak = 0
            failure_to_maintain_set = float(ftm)

        records.append({
            "participant_id": pid,
            "wcst_pes": pes,
            "wcst_post_switch_error_rate": post_switch_error_rate,
            "wcst_cv_rt": coefficient_of_variation(grp[rt_col].dropna()),
            "wcst_trials": len(grp),
            "wcst_total_errors": total_errors,
            "wcst_error_rate": error_rate,
            "wcst_perseverative_errors": pe_count,
            "wcst_perseverative_error_rate": pe_rate,
            "wcst_nonperseverative_errors": npe_count,
            "wcst_perseverative_responses": pr_count,
            "wcst_perseverative_response_percent": pr_percent,
            "wcst_error_pr_ratio": error_pr_ratio,
            "wcst_error_npe_ratio": error_npe_ratio,
            "wcst_pe_run_length_mean": pe_run_mean,
            "wcst_pe_run_length_max": pe_run_max,
            "wcst_pr_run_length_mean": pr_run_mean,
            "wcst_pr_run_length_max": pr_run_max,
            "wcst_pe_cluster_rate": pe_cluster_rate,
            "wcst_trials_to_first_category": trials_to_first_category,
            "wcst_categories_completed": categories_completed,
            "wcst_trials_per_category_mean": float(np.mean(trials_per_category)) if trials_per_category else np.nan,
            "wcst_trials_per_category_sd": float(np.std(trials_per_category)) if trials_per_category else np.nan,
            "wcst_rt_slope_within_category": rt_slope_within,
            "wcst_acc_slope_within_category": acc_slope_within,
            "wcst_rt_jump_at_switch": rt_jump_at_switch,
            "wcst_failure_to_maintain_set": failure_to_maintain_set,
            "wcst_trials_to_rule_reacquisition": trials_to_reacq,
        })

        if trials_per_category:
            record = records[-1]
            for idx, val in enumerate(trials_per_category, start=1):
                record[f"wcst_trials_per_category_{idx}"] = float(val)
            if len(trials_per_category) >= 6:
                first = np.mean(trials_per_category[:3])
                last = np.mean(trials_per_category[-3:])
                record["wcst_delta_trials_first3_last3"] = float(first - last)
            else:
                record["wcst_delta_trials_first3_last3"] = np.nan
            if len(trials_per_category) >= 2:
                x = np.arange(1, len(trials_per_category) + 1)
                record["wcst_learning_slope_trials"] = float(np.polyfit(x, trials_per_category, 1)[0])
            else:
                record["wcst_learning_slope_trials"] = np.nan

            for k in range(1, 6):
                record[f"wcst_switch_cost_error_k{k}"] = switch_cost_err[k]
                record[f"wcst_switch_cost_rt_k{k}"] = switch_cost_rt[k]

    features_df = pd.DataFrame(records)

    hmm_df = load_or_compute_wcst_hmm_mechanism_features(data_dir=data_dir)
    rl_df = load_or_compute_wcst_rl_mechanism_features(data_dir=data_dir)
    wsls_df = load_or_compute_wcst_wsls_mechanism_features(data_dir=data_dir)
    brl_df = load_or_compute_wcst_bayesianrl_mechanism_features(data_dir=data_dir)
    mech_frames = [df for df in (hmm_df, rl_df, wsls_df, brl_df) if not df.empty]
    if not mech_frames:
        return features_df

    if features_df.empty:
        combined = mech_frames[0]
        for extra_df in mech_frames[1:]:
            overlap = [c for c in extra_df.columns if c != "participant_id" and c in combined.columns]
            if overlap:
                extra_df = extra_df.drop(columns=overlap)
            combined = combined.merge(extra_df, on="participant_id", how="outer")
        return combined

    for mech_df in mech_frames:
        overlap = [c for c in mech_df.columns if c != "participant_id" and c in features_df.columns]
        if overlap:
            features_df = features_df.drop(columns=overlap)
        features_df = features_df.merge(mech_df, on="participant_id", how="left")

    summary_df = _load_wcst_summary_metrics(data_dir if isinstance(data_dir, Path) else Path(data_dir) if data_dir else None)
    if not summary_df.empty:
        summary_df = summary_df.set_index("participant_id")
        features_df = features_df.set_index("participant_id")
        for col in summary_df.columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].combine_first(summary_df[col])
            else:
                features_df[col] = summary_df[col]
        features_df = features_df.reset_index()

    return features_df
