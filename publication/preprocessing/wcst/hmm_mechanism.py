"""WCST HMM mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import get_results_dir
from .loaders import load_wcst_trials

MECHANISM_FILENAME = "5_wcst_hmm_mechanism_features.csv"
SHIFT_K_MAX = 5
STABLE_RUN_MIN = 5
EVENT_COLUMNS = [
    "wcst_slow_prob_baseline",
    "wcst_slow_prob_post_error",
    "wcst_slow_prob_post_error_delta",
    "wcst_slow_prob_stable",
    "wcst_slow_prob_stable_delta",
] + [
    f"wcst_slow_prob_shift_k{k}" for k in range(SHIFT_K_MAX + 1)
] + [
    f"wcst_slow_prob_shift_k{k}_delta" for k in range(SHIFT_K_MAX + 1)
]

try:
    from hmmlearn import hmm as hmm_module
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def _run_lengths(states: np.ndarray, target: int) -> list[int]:
    lengths = []
    count = 0
    for val in states:
        if val == target:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def _compute_event_metrics(
    p_slow: np.ndarray,
    rules: np.ndarray | None,
    correct: np.ndarray | None,
    stable_len: int = STABLE_RUN_MIN,
    max_k: int = SHIFT_K_MAX,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    baseline = float(np.nanmean(p_slow)) if len(p_slow) else np.nan
    metrics["wcst_slow_prob_baseline"] = baseline

    # Shift-aligned slow-state probability (k=0..max_k).
    if rules is None:
        for k in range(max_k + 1):
            metrics[f"wcst_slow_prob_shift_k{k}"] = np.nan
            metrics[f"wcst_slow_prob_shift_k{k}_delta"] = np.nan
    else:
        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        for k in range(max_k + 1):
            vals = []
            for idx in change_indices:
                j = idx + k
                if j < len(p_slow):
                    vals.append(p_slow[j])
            if vals:
                mean_val = float(np.mean(vals))
                metrics[f"wcst_slow_prob_shift_k{k}"] = mean_val
                metrics[f"wcst_slow_prob_shift_k{k}_delta"] = (
                    mean_val - baseline if np.isfinite(baseline) else np.nan
                )
            else:
                metrics[f"wcst_slow_prob_shift_k{k}"] = np.nan
                metrics[f"wcst_slow_prob_shift_k{k}_delta"] = np.nan

    # Post-error and stable-run metrics.
    if correct is None or len(correct) == 0:
        metrics["wcst_slow_prob_post_error"] = np.nan
        metrics["wcst_slow_prob_post_error_delta"] = np.nan
        metrics["wcst_slow_prob_stable"] = np.nan
        metrics["wcst_slow_prob_stable_delta"] = np.nan
        return metrics

    post_error_vals = [
        p_slow[i + 1] for i in range(len(correct) - 1) if not correct[i]
    ]
    if post_error_vals:
        post_error_mean = float(np.mean(post_error_vals))
        metrics["wcst_slow_prob_post_error"] = post_error_mean
        metrics["wcst_slow_prob_post_error_delta"] = (
            post_error_mean - baseline if np.isfinite(baseline) else np.nan
        )
    else:
        metrics["wcst_slow_prob_post_error"] = np.nan
        metrics["wcst_slow_prob_post_error_delta"] = np.nan

    stable_mask = np.zeros(len(correct), dtype=bool)
    run_len = 0
    prev_rule = rules[0] if rules is not None and len(rules) else None
    for i in range(len(correct)):
        if rules is not None and rules[i] != prev_rule:
            run_len = 0
            prev_rule = rules[i]
        if correct[i]:
            run_len += 1
        else:
            run_len = 0
        if run_len >= stable_len:
            stable_mask[i] = True
    if stable_mask.any():
        stable_mean = float(np.mean(p_slow[stable_mask]))
        metrics["wcst_slow_prob_stable"] = stable_mean
        metrics["wcst_slow_prob_stable_delta"] = (
            stable_mean - baseline if np.isfinite(baseline) else np.nan
        )
    else:
        metrics["wcst_slow_prob_stable"] = np.nan
        metrics["wcst_slow_prob_stable_delta"] = np.nan

    return metrics


def compute_wcst_hmm_features(
    data_dir: Path | None = None,
    n_states: int = 2,
    min_trials: int = 50,
) -> pd.DataFrame:
    if not HMM_AVAILABLE:
        return pd.DataFrame()

    trials, _ = load_wcst_trials(data_dir=data_dir, apply_trial_filters=True)

    rt_col = None
    for cand in ("rt_ms", "reactionTimeMs", "reaction_time_ms", "reactiontimems", "rt"):
        if cand in trials.columns:
            rt_col = cand
            break
    if rt_col is None:
        return pd.DataFrame()
    trials["rt_ms"] = pd.to_numeric(trials[rt_col], errors="coerce")
    trials = trials[trials["rt_ms"].notna()]

    trial_col = None
    for cand in ("trialIndex", "trial_index", "trial"):
        if cand in trials.columns:
            trial_col = cand
            break
    if trial_col:
        trials = trials.sort_values(["participant_id", trial_col])

    rule_col = None
    for cand in ("ruleAtThatTime", "rule_at_that_time", "rule", "cond"):
        if cand in trials.columns:
            rule_col = cand
            break

    results = []
    for pid, pdata in trials.groupby("participant_id"):
        if len(pdata) < min_trials:
            continue

        rts = pdata["rt_ms"].values.reshape(-1, 1)
        try:
            model = hmm_module.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(rts)
            states = model.predict(rts)
            means = model.means_.flatten()

            lapse_state = int(np.argmax(means))
            focus_state = 1 - lapse_state

            trans_matrix = model.transmat_
            try:
                post = model.predict_proba(rts)
                p_slow = post[:, lapse_state]
            except Exception:
                p_slow = (states == lapse_state).astype(float)

            lapse_runs = _run_lengths(states, lapse_state)
            focus_runs = _run_lengths(states, focus_state)
            lapse_dwell = float(np.mean(lapse_runs)) if lapse_runs else np.nan
            focus_dwell = float(np.mean(focus_runs)) if focus_runs else np.nan
            long_lapse_rate = (
                float(np.mean([length >= 5 for length in lapse_runs])) if lapse_runs else np.nan
            )
            lapse_rt_sd = float(np.nanstd(rts[states == lapse_state])) if np.any(states == lapse_state) else np.nan
            focus_rt_sd = float(np.nanstd(rts[states == focus_state])) if np.any(states == focus_state) else np.nan
            occ = np.array([(states == focus_state).mean(), (states == lapse_state).mean()])
            state_entropy = float(-np.sum(occ * np.log(np.clip(occ, 1e-12, 1.0))))

            record = {
                "participant_id": pid,
                "wcst_hmm_lapse_occupancy": (states == lapse_state).mean() * 100,
                "wcst_hmm_trans_to_lapse": trans_matrix[focus_state, lapse_state],
                "wcst_hmm_trans_to_focus": trans_matrix[lapse_state, focus_state],
                "wcst_hmm_stay_lapse": trans_matrix[lapse_state, lapse_state],
                "wcst_hmm_stay_focus": trans_matrix[focus_state, focus_state],
                "wcst_hmm_lapse_rt_mean": means[lapse_state],
                "wcst_hmm_focus_rt_mean": means[focus_state],
                "wcst_hmm_rt_diff": means[lapse_state] - means[focus_state],
                "wcst_hmm_state_changes": int(np.sum(np.diff(states) != 0)),
                "wcst_hmm_n_trials": int(len(pdata)),
                "wcst_hmm_lapse_dwell_mean": lapse_dwell,
                "wcst_hmm_focus_dwell_mean": focus_dwell,
                "wcst_hmm_long_lapse_episode_rate": long_lapse_rate,
                "wcst_hmm_lapse_rt_sd": lapse_rt_sd,
                "wcst_hmm_focus_rt_sd": focus_rt_sd,
                "wcst_hmm_state_entropy": state_entropy,
            }
            rules = (
                pdata[rule_col].astype(str).str.lower().values
                if rule_col and rule_col in pdata.columns
                else None
            )
            correct = (
                pdata["correct"].astype(bool).values
                if "correct" in pdata.columns
                else None
            )
            record.update(_compute_event_metrics(p_slow, rules, correct))

            if "correct" in pdata.columns:
                acc = pdata["correct"].astype(float).values.reshape(-1, 1)
                features = np.hstack([rts, acc])
                mean = np.nanmean(features, axis=0)
                std = np.nanstd(features, axis=0)
                std[std == 0] = 1.0
                features = (features - mean) / std
                try:
                    model2d = hmm_module.GaussianHMM(
                        n_components=n_states,
                        covariance_type="full",
                        n_iter=100,
                        random_state=42,
                    )
                    model2d.fit(features)
                    states2d = model2d.predict(features)
                    lapse_state2d = int(np.argmax([np.mean(rts[states2d == s]) for s in range(n_states)]))
                    focus_state2d = 1 - lapse_state2d
                    trans2d = model2d.transmat_
                    lapse_runs2d = _run_lengths(states2d, lapse_state2d)
                    focus_runs2d = _run_lengths(states2d, focus_state2d)
                    occ2d = np.array([
                        (states2d == focus_state2d).mean(),
                        (states2d == lapse_state2d).mean(),
                    ])
                    record.update({
                        "wcst_hmm2d_lapse_occupancy": (states2d == lapse_state2d).mean() * 100,
                        "wcst_hmm2d_trans_to_lapse": trans2d[focus_state2d, lapse_state2d],
                        "wcst_hmm2d_trans_to_focus": trans2d[lapse_state2d, focus_state2d],
                        "wcst_hmm2d_stay_lapse": trans2d[lapse_state2d, lapse_state2d],
                        "wcst_hmm2d_stay_focus": trans2d[focus_state2d, focus_state2d],
                        "wcst_hmm2d_lapse_dwell_mean": float(np.mean(lapse_runs2d)) if lapse_runs2d else np.nan,
                        "wcst_hmm2d_focus_dwell_mean": float(np.mean(focus_runs2d)) if focus_runs2d else np.nan,
                        "wcst_hmm2d_state_entropy": float(-np.sum(occ2d * np.log(np.clip(occ2d, 1e-12, 1.0)))),
                    })
                except Exception:
                    pass
            results.append(record)
        except Exception:
            continue

    return pd.DataFrame(results)


def load_or_compute_wcst_hmm_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("wcst")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        existing = pd.read_csv(output_path, encoding="utf-8-sig")
        if all(col in existing.columns for col in EVENT_COLUMNS):
            return existing

    features = compute_wcst_hmm_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] WCST HMM mechanism features saved: {output_path}")
    return features
