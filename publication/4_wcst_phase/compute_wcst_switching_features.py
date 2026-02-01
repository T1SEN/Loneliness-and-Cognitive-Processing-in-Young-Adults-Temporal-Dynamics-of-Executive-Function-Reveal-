"""Compute WCST rule-switching features (excluding slow-state profiles)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import WCST_RT_MIN, WCST_RT_MAX, get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.wcst_phase_utils import prepare_wcst_trials


RULE_TYPES = ("colour", "shape", "number")


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _sd_or_nan(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def _slope_or_nan(x_vals: list[float], y_vals: list[float], min_n: int = 3) -> float:
    if len(y_vals) < min_n:
        return float("nan")
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return float("nan")


def _split_early_late(n_switches: int) -> tuple[list[int], list[int]]:
    if n_switches >= 4:
        return [0, 1], [n_switches - 2, n_switches - 1]
    if n_switches == 3:
        return [0], [2]
    if n_switches == 2:
        return [0], [1]
    return [], []


def _summarize_switch_costs(df: pd.DataFrame) -> pd.DataFrame:
    dyn_path = get_results_dir("overall") / "5_wcst_dynamic_recovery_features.csv"
    if not dyn_path.exists():
        return pd.DataFrame()
    dyn = pd.read_csv(dyn_path, encoding="utf-8-sig")
    dyn = ensure_participant_id(dyn)

    records: list[dict[str, float]] = []
    for _, row in dyn.iterrows():
        pid = row["participant_id"]
        rt_vals = []
        err_vals = []
        x_vals = []
        for k in range(1, 6):
            rt = row.get(f"wcst_switch_cost_rt_k{k}")
            err = row.get(f"wcst_switch_cost_error_k{k}")
            if pd.notna(rt):
                rt_vals.append(float(rt))
                x_vals.append(float(k))
            if pd.notna(err):
                err_vals.append(float(err))

        record = {
            "participant_id": pid,
            "wcst_switch_cost_rt_mean_k1_k5": _mean_or_nan(rt_vals),
            "wcst_switch_cost_error_mean_k1_k5": _mean_or_nan(err_vals),
            "wcst_switch_cost_rt_slope_k1_k5": _slope_or_nan(x_vals, rt_vals, min_n=3),
            "wcst_switch_cost_error_slope_k1_k5": _slope_or_nan(x_vals, err_vals, min_n=3),
        }
        records.append(record)

    return pd.DataFrame(records)


def compute_switching_features() -> pd.DataFrame:
    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]

    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or rule_col is None:
        return pd.DataFrame()

    records: list[dict[str, float]] = []

    for pid, grp in wcst.groupby("participant_id"):
        if trial_col and trial_col in grp.columns:
            grp = grp.sort_values(trial_col)
        grp = grp.reset_index(drop=True)

        rules = grp[rule_col].astype(str).str.lower().replace({"color": "colour"}).values
        rt_vals = pd.to_numeric(grp[rt_col], errors="coerce").astype(float).values
        correct = grp["correct"].astype(bool).values if "correct" in grp.columns else None
        is_pe = grp["isPE"].astype(bool).values if "isPE" in grp.columns else None

        if "is_rt_valid" in grp.columns:
            valid = grp["is_rt_valid"].astype(bool).values
        else:
            valid = np.isfinite(rt_vals)
            valid &= (rt_vals >= WCST_RT_MIN) & (rt_vals <= WCST_RT_MAX)

        rt_vals = rt_vals.copy()
        rt_vals[~valid] = np.nan

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        n_switches = len(change_indices)

        switch_cost_rt = [float("nan")] * n_switches
        switch_cost_err = [float("nan")] * n_switches
        switch_rules = [""] * n_switches

        exploration_means = []
        exploration_sds = []
        exploitation_means = []
        exploitation_sds = []
        switch_first_rt = []
        switch_first_correct = []
        switch_first_error_latency = []
        switch_first_error_rt = []
        switch_signal_rt = []
        switch_signal_delta = []
        lucky_streak_lens = []
        exploration_slopes = []
        exploration_error_rates = []
        confirmation_error_rates = []
        confirmation_rt_means = []
        confirmation_slopes = []
        post_reacq_rt_cvs = []
        shift_pe_rt_means = []
        shift_pe_counts = []
        shift_pe_rates = []
        shift_pe_post_rt_means = []
        shift_pe_slowings = []
        shift_pe_rt_means_reacq = []
        shift_pe_counts_reacq = []
        shift_pe_rates_reacq = []
        shift_pe_post_rt_means_reacq = []
        shift_pe_slowings_reacq = []
        stable_pe_rt_means = []
        stable_pe_counts = []
        stable_pe_rates = []
        shift_error_rts = []
        shift_post_error_rts = []
        shift_pes_vals = []
        stable_error_rts = []
        stable_post_error_rts = []
        stable_pes_vals = []
        shift_error_count_total = 0
        shift_trial_count_total = 0
        exploitation_slopes = []
        exploitation_error_rates = []

        total_trials = int(len(correct)) if correct is not None else 0
        total_errors = int(np.sum(~correct)) if correct is not None else 0
        shift_window_mask = np.zeros(len(correct), dtype=bool) if correct is not None else None

        for s_idx, idx in enumerate(change_indices):
            next_idx = change_indices[s_idx + 1] if s_idx + 1 < n_switches else len(rt_vals)
            switch_rules[s_idx] = rules[idx]

            pre_start = max(0, idx - 3)
            pre_rt = rt_vals[pre_start:idx]
            pre_rt = pre_rt[np.isfinite(pre_rt)]
            pre_mean = float(np.mean(pre_rt)) if len(pre_rt) >= 2 else float("nan")
            if len(pre_rt) >= 2 and idx + 1 < len(rt_vals):
                rt_k1 = rt_vals[idx + 1]
                if np.isfinite(rt_k1):
                    switch_cost_rt[s_idx] = float(rt_k1 - np.mean(pre_rt))
            if correct is not None and idx + 1 < len(correct):
                switch_cost_err[s_idx] = float(1 - int(correct[idx + 1]))

            if correct is not None:
                reacq_idx_local = None
                reacq_start_local = None
                for j in range(idx, next_idx - 2):
                    if correct[j] and correct[j + 1] and correct[j + 2]:
                        reacq_start_local = j
                        reacq_idx_local = j + 2
                        break

                if idx < len(rt_vals):
                    rt_k0 = rt_vals[idx]
                    if np.isfinite(rt_k0):
                        switch_first_rt.append(float(rt_k0))
                if idx < len(correct):
                    switch_first_correct.append(float(int(correct[idx])))

                first_error_idx = None
                for j in range(idx, len(correct)):
                    if not correct[j]:
                        first_error_idx = j
                        break
                if first_error_idx is not None:
                    switch_first_error_latency.append(float((first_error_idx - idx) + 1))
                    rt_err = rt_vals[first_error_idx]
                    if np.isfinite(rt_err):
                        switch_first_error_rt.append(float(rt_err))
                    if first_error_idx + 1 < len(rt_vals):
                        rt_signal = rt_vals[first_error_idx + 1]
                        if np.isfinite(rt_signal):
                            switch_signal_rt.append(float(rt_signal))
                            if np.isfinite(pre_mean):
                                switch_signal_delta.append(float(rt_signal - pre_mean))
                    lucky = int(np.sum(correct[idx:first_error_idx]))
                    lucky_streak_lens.append(float(lucky))

                explore_end = reacq_start_local if reacq_start_local is not None else next_idx
                if explore_end > idx:
                    segment_rt = rt_vals[idx:explore_end]
                    segment_valid = segment_rt[np.isfinite(segment_rt)]
                    if len(segment_valid) > 0:
                        exploration_means.append(float(np.mean(segment_valid)))
                        exploration_sds.append(float(np.std(segment_valid, ddof=1)) if len(segment_valid) > 1 else 0.0)
                    exp_correct = correct[idx:explore_end]
                    if len(exp_correct):
                        exploration_error_rates.append(float(1.0 - np.mean(exp_correct)))
                    if len(segment_rt) >= 3 and np.isfinite(segment_rt).sum() >= 3:
                        x_vals = np.arange(1, len(segment_rt) + 1)
                        mask = np.isfinite(segment_rt)
                        exploration_slopes.append(float(np.polyfit(x_vals[mask], segment_rt[mask], 1)[0]))

                shift_end_err = reacq_idx_local if reacq_idx_local is not None else next_idx - 1
                if shift_end_err >= idx:
                    if shift_window_mask is not None:
                        shift_window_mask[idx:shift_end_err + 1] = True
                    shift_trial_count_total += int(shift_end_err - idx + 1)
                    shift_error_count_total += int(np.sum(~correct[idx:shift_end_err + 1]))
                    for j in range(idx, shift_end_err + 1):
                        if correct[j]:
                            continue
                        rt_err = rt_vals[j]
                        if np.isfinite(rt_err):
                            shift_error_rts.append(float(rt_err))
                        if j + 1 < next_idx:
                            rt_next = rt_vals[j + 1]
                            if np.isfinite(rt_next):
                                shift_post_error_rts.append(float(rt_next))
                            if np.isfinite(rt_err) and np.isfinite(rt_next):
                                shift_pes_vals.append(float(rt_next - rt_err))

                if reacq_idx_local is not None:
                    stable_start_err = reacq_idx_local + 1
                    stable_end_err = next_idx - 1
                    if stable_start_err <= stable_end_err:
                        for j in range(stable_start_err, stable_end_err + 1):
                            if correct[j]:
                                continue
                            rt_err = rt_vals[j]
                            if np.isfinite(rt_err):
                                stable_error_rts.append(float(rt_err))
                            if j + 1 < next_idx:
                                rt_next = rt_vals[j + 1]
                                if np.isfinite(rt_next):
                                    stable_post_error_rts.append(float(rt_next))
                                if np.isfinite(rt_err) and np.isfinite(rt_next):
                                    stable_pes_vals.append(float(rt_next - rt_err))

                if is_pe is not None:
                    first_correct_idx_local = None
                    for j in range(idx, next_idx):
                        if correct[j]:
                            first_correct_idx_local = j
                            break

                    shift_end = first_correct_idx_local if first_correct_idx_local is not None else next_idx - 1
                    if shift_end >= idx:
                        shift_pe_idx = [j for j in range(idx, shift_end + 1) if is_pe[j]]
                        window_len = shift_end - idx + 1
                        shift_pe_counts.append(float(len(shift_pe_idx)))
                        shift_pe_rates.append(float(len(shift_pe_idx) / window_len) if window_len > 0 else float("nan"))
                        shift_pe_rts = [rt_vals[j] for j in shift_pe_idx if np.isfinite(rt_vals[j])]
                        if shift_pe_rts:
                            shift_pe_rt_means.append(float(np.mean(shift_pe_rts)))
                        post_rts = []
                        for j in shift_pe_idx:
                            if j + 1 < next_idx:
                                rt_next = rt_vals[j + 1]
                                if np.isfinite(rt_next):
                                    post_rts.append(float(rt_next))
                        if post_rts:
                            shift_pe_post_rt_means.append(float(np.mean(post_rts)))
                        if shift_pe_rts and post_rts:
                            shift_pe_slowings.append(float(np.mean(post_rts) - np.mean(shift_pe_rts)))

                    shift_end_reacq = reacq_idx_local if reacq_idx_local is not None else next_idx - 1
                    if shift_end_reacq >= idx:
                        shift_pe_idx = [j for j in range(idx, shift_end_reacq + 1) if is_pe[j]]
                        window_len = shift_end_reacq - idx + 1
                        shift_pe_counts_reacq.append(float(len(shift_pe_idx)))
                        shift_pe_rates_reacq.append(
                            float(len(shift_pe_idx) / window_len) if window_len > 0 else float("nan")
                        )
                        shift_pe_rts = [rt_vals[j] for j in shift_pe_idx if np.isfinite(rt_vals[j])]
                        if shift_pe_rts:
                            shift_pe_rt_means_reacq.append(float(np.mean(shift_pe_rts)))
                        post_rts = []
                        for j in shift_pe_idx:
                            if j + 1 < next_idx:
                                rt_next = rt_vals[j + 1]
                                if np.isfinite(rt_next):
                                    post_rts.append(float(rt_next))
                        if post_rts:
                            shift_pe_post_rt_means_reacq.append(float(np.mean(post_rts)))
                        if shift_pe_rts and post_rts:
                            shift_pe_slowings_reacq.append(float(np.mean(post_rts) - np.mean(shift_pe_rts)))

                    if reacq_idx_local is not None:
                        stable_start = reacq_idx_local + 1
                        stable_end = next_idx - 1
                        if stable_start <= stable_end:
                            stable_pe_idx = [j for j in range(stable_start, stable_end + 1) if is_pe[j]]
                            stable_len = stable_end - stable_start + 1
                            stable_pe_counts.append(float(len(stable_pe_idx)))
                            stable_pe_rates.append(
                                float(len(stable_pe_idx) / stable_len) if stable_len > 0 else float("nan")
                            )
                            stable_pe_rts = [rt_vals[j] for j in stable_pe_idx if np.isfinite(rt_vals[j])]
                            if stable_pe_rts:
                                stable_pe_rt_means.append(float(np.mean(stable_pe_rts)))

                reacq_idx = reacq_idx_local
                reacq_start = reacq_start_local
                if reacq_idx is not None:
                    confirm_rt = rt_vals[reacq_start:reacq_idx + 1]
                    confirm_rt = confirm_rt[np.isfinite(confirm_rt)]
                    if len(confirm_rt) > 0:
                        confirmation_rt_means.append(float(np.mean(confirm_rt)))
                    confirm_correct = correct[reacq_start:reacq_idx + 1]
                    if len(confirm_correct):
                        confirmation_error_rates.append(float(1.0 - np.mean(confirm_correct)))
                    if len(confirm_rt) >= 3:
                        x_vals = np.arange(1, len(confirm_rt) + 1)
                        confirmation_slopes.append(float(np.polyfit(x_vals, confirm_rt, 1)[0]))

                    next_idx = change_indices[s_idx + 1] if s_idx + 1 < n_switches else len(rt_vals)
                    start = reacq_idx + 1
                    if start < next_idx:
                        exp_segment = rt_vals[start:next_idx]
                        exp_valid = exp_segment[np.isfinite(exp_segment)]
                        if len(exp_valid) > 0:
                            exploitation_means.append(float(np.mean(exp_valid)))
                            exploitation_sds.append(float(np.std(exp_valid, ddof=1)) if len(exp_valid) > 1 else 0.0)
                            if len(exp_segment) >= 3 and np.isfinite(exp_segment).sum() >= 3:
                                x_vals = np.arange(1, len(exp_segment) + 1)
                                mask = np.isfinite(exp_segment)
                                exploitation_slopes.append(float(np.polyfit(x_vals[mask], exp_segment[mask], 1)[0]))
                            if len(exp_valid) > 1:
                                mean_rt = float(np.mean(exp_valid))
                                if mean_rt > 0:
                                    post_reacq_rt_cvs.append(float(np.std(exp_valid, ddof=1) / mean_rt))
                        exp_correct = correct[start:next_idx]
                        if len(exp_correct):
                            exploitation_error_rates.append(float(1.0 - np.mean(exp_correct)))

        early_idx, late_idx = _split_early_late(n_switches)
        early_rt = [switch_cost_rt[i] for i in early_idx if np.isfinite(switch_cost_rt[i])]
        late_rt = [switch_cost_rt[i] for i in late_idx if np.isfinite(switch_cost_rt[i])]
        early_err = [switch_cost_err[i] for i in early_idx if np.isfinite(switch_cost_err[i])]
        late_err = [switch_cost_err[i] for i in late_idx if np.isfinite(switch_cost_err[i])]

        rule_rt: dict[str, list[float]] = {r: [] for r in RULE_TYPES}
        rule_err: dict[str, list[float]] = {r: [] for r in RULE_TYPES}
        for s_idx, rule in enumerate(switch_rules):
            if rule not in RULE_TYPES:
                continue
            rt_val = switch_cost_rt[s_idx]
            err_val = switch_cost_err[s_idx]
            if np.isfinite(rt_val):
                rule_rt[rule].append(rt_val)
            if np.isfinite(err_val):
                rule_err[rule].append(err_val)

        nonshift_error_rts = []
        if correct is not None and shift_window_mask is not None:
            error_mask = ~correct
            valid_mask = np.isfinite(rt_vals)
            nonshift_error_rts = rt_vals[error_mask & ~shift_window_mask & valid_mask].tolist()

        record = {
            "participant_id": pid,
            "wcst_switch_cost_rt_k1_sd": _sd_or_nan([v for v in switch_cost_rt if np.isfinite(v)]),
            "wcst_switch_cost_error_k1_sd": _sd_or_nan([v for v in switch_cost_err if np.isfinite(v)]),
            "wcst_switch_cost_rt_early": _mean_or_nan(early_rt),
            "wcst_switch_cost_rt_late": _mean_or_nan(late_rt),
            "wcst_switch_cost_rt_early_late_delta": (
                _mean_or_nan(early_rt) - _mean_or_nan(late_rt)
                if early_rt and late_rt else float("nan")
            ),
            "wcst_switch_cost_error_early": _mean_or_nan(early_err),
            "wcst_switch_cost_error_late": _mean_or_nan(late_err),
            "wcst_switch_cost_error_early_late_delta": (
                _mean_or_nan(early_err) - _mean_or_nan(late_err)
                if early_err and late_err else float("nan")
            ),
            "wcst_exploration_rt_mean": _mean_or_nan(exploration_means),
            "wcst_exploration_rt_sd": _mean_or_nan(exploration_sds),
            "wcst_exploitation_rt_mean": _mean_or_nan(exploitation_means),
            "wcst_exploitation_rt_sd": _mean_or_nan(exploitation_sds),
            "wcst_switch_first_trial_rt": _mean_or_nan(switch_first_rt),
            "wcst_switch_first_trial_correct": _mean_or_nan(switch_first_correct),
            "wcst_switch_first_error_latency": _mean_or_nan(switch_first_error_latency),
            "wcst_switch_first_error_rt": _mean_or_nan(switch_first_error_rt),
            "wcst_switch_signal_rt": _mean_or_nan(switch_signal_rt),
            "wcst_switch_signal_delta": _mean_or_nan(switch_signal_delta),
            "wcst_lucky_streak_len": _mean_or_nan(lucky_streak_lens),
            "wcst_exploration_rt_slope": _mean_or_nan(exploration_slopes),
            "wcst_exploration_error_rate": _mean_or_nan(exploration_error_rates),
            "wcst_confirmation_rt_mean": _mean_or_nan(confirmation_rt_means),
            "wcst_confirmation_rt_slope": _mean_or_nan(confirmation_slopes),
            "wcst_confirmation_error_rate": _mean_or_nan(confirmation_error_rates),
            "wcst_post_reacq_rt_cv": _mean_or_nan(post_reacq_rt_cvs),
            "wcst_exploitation_rt_slope": _mean_or_nan(exploitation_slopes),
            "wcst_exploitation_error_rate": _mean_or_nan(exploitation_error_rates),
            "wcst_shift_pe_rt_mean": _mean_or_nan(shift_pe_rt_means),
            "wcst_shift_pe_count_mean": _mean_or_nan(shift_pe_counts),
            "wcst_shift_pe_rate_mean": _mean_or_nan(shift_pe_rates),
            "wcst_post_shift_pe_rt_mean": _mean_or_nan(shift_pe_post_rt_means),
            "wcst_post_shift_pe_slowing": _mean_or_nan(shift_pe_slowings),
            "wcst_shift_pe_rt_mean_reacq": _mean_or_nan(shift_pe_rt_means_reacq),
            "wcst_shift_pe_count_mean_reacq": _mean_or_nan(shift_pe_counts_reacq),
            "wcst_shift_pe_rate_mean_reacq": _mean_or_nan(shift_pe_rates_reacq),
            "wcst_post_shift_pe_rt_mean_reacq": _mean_or_nan(shift_pe_post_rt_means_reacq),
            "wcst_post_shift_pe_slowing_reacq": _mean_or_nan(shift_pe_slowings_reacq),
            "wcst_stable_pe_rt_mean": _mean_or_nan(stable_pe_rt_means),
            "wcst_stable_pe_count_mean": _mean_or_nan(stable_pe_counts),
            "wcst_stable_pe_rate_mean": _mean_or_nan(stable_pe_rates),
            "wcst_shift_error_rt_mean": _mean_or_nan(shift_error_rts),
            "wcst_post_shift_error_rt_mean": _mean_or_nan(shift_post_error_rts),
            "wcst_shift_post_error_slowing": _mean_or_nan(shift_pes_vals),
            "wcst_stable_error_rt_mean": _mean_or_nan(stable_error_rts),
            "wcst_post_stable_error_rt_mean": _mean_or_nan(stable_post_error_rts),
            "wcst_stable_post_error_slowing": _mean_or_nan(stable_pes_vals),
            "wcst_nonshift_error_rt_mean": _mean_or_nan(nonshift_error_rts),
            "wcst_shift_error_count_total": float(shift_error_count_total)
            if correct is not None else float("nan"),
            "wcst_shift_error_rate": (
                float(shift_error_count_total) / float(shift_trial_count_total)
                if shift_trial_count_total > 0 else float("nan")
            ),
            "wcst_nonshift_error_count_total": (
                float(total_errors - shift_error_count_total)
                if correct is not None else float("nan")
            ),
            "wcst_nonshift_error_rate": (
                float(total_errors - shift_error_count_total) / float(total_trials - shift_trial_count_total)
                if total_trials > shift_trial_count_total else float("nan")
            ),
        }
        if shift_pe_rt_means and stable_pe_rt_means:
            record["wcst_pe_context_rt_delta"] = float(
                np.mean(stable_pe_rt_means) - np.mean(shift_pe_rt_means)
            )
        else:
            record["wcst_pe_context_rt_delta"] = float("nan")
        if shift_pe_rates and stable_pe_rates:
            record["wcst_pe_context_rate_delta"] = float(
                np.mean(stable_pe_rates) - np.mean(shift_pe_rates)
            )
        else:
            record["wcst_pe_context_rate_delta"] = float("nan")
        if shift_pe_rt_means_reacq and stable_pe_rt_means:
            record["wcst_pe_context_rt_delta_reacq"] = float(
                np.mean(stable_pe_rt_means) - np.mean(shift_pe_rt_means_reacq)
            )
        else:
            record["wcst_pe_context_rt_delta_reacq"] = float("nan")
        if shift_pe_rates_reacq and stable_pe_rates:
            record["wcst_pe_context_rate_delta_reacq"] = float(
                np.mean(stable_pe_rates) - np.mean(shift_pe_rates_reacq)
            )
        else:
            record["wcst_pe_context_rate_delta_reacq"] = float("nan")
        if shift_post_error_rts and stable_post_error_rts:
            record["wcst_error_context_post_error_rt_delta"] = float(
                np.mean(stable_post_error_rts) - np.mean(shift_post_error_rts)
            )
        else:
            record["wcst_error_context_post_error_rt_delta"] = float("nan")
        if shift_pes_vals and stable_pes_vals:
            record["wcst_error_context_pes_delta"] = float(
                np.mean(stable_pes_vals) - np.mean(shift_pes_vals)
            )
        else:
            record["wcst_error_context_pes_delta"] = float("nan")
        if exploration_means and exploitation_means:
            record["wcst_exploration_penalty"] = (
                float(np.mean(exploration_means)) - float(np.mean(exploitation_means))
            )
        else:
            record["wcst_exploration_penalty"] = float("nan")

        for rule in RULE_TYPES:
            record[f"wcst_switch_cost_rt_k1_to_{rule}"] = _mean_or_nan(rule_rt[rule])
            record[f"wcst_switch_cost_error_k1_to_{rule}"] = _mean_or_nan(rule_err[rule])

        records.append(record)

    features = pd.DataFrame(records)
    summary = _summarize_switch_costs(features)
    if summary.empty:
        return features
    return features.merge(summary, on="participant_id", how="left")


def main() -> None:
    df = compute_switching_features()
    out_path = get_results_dir("overall") / "5_wcst_switching_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not df.empty:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()



