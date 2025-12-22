"""PRP HMM event feature derivation (slow-state probabilities)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import get_results_dir
from .loaders import load_prp_trials

MECHANISM_FILENAME = "5_prp_hmm_event_features.csv"
STABLE_RUN_MIN = 5
EVENT_COLUMNS = [
    "prp_slow_prob_baseline",
    "prp_slow_prob_post_error",
    "prp_slow_prob_post_error_delta",
    "prp_slow_prob_stable",
    "prp_slow_prob_stable_delta",
]

try:
    from hmmlearn import hmm as hmm_module
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def _compute_event_metrics(
    p_slow: np.ndarray,
    error: np.ndarray,
    correct: np.ndarray,
    stable_len: int = STABLE_RUN_MIN,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    baseline = float(np.nanmean(p_slow)) if len(p_slow) else np.nan
    metrics["prp_slow_prob_baseline"] = baseline

    post_error_vals = [
        p_slow[i + 1] for i in range(len(error) - 1) if error[i]
    ]
    if post_error_vals:
        post_error_mean = float(np.mean(post_error_vals))
        metrics["prp_slow_prob_post_error"] = post_error_mean
        metrics["prp_slow_prob_post_error_delta"] = (
            post_error_mean - baseline if np.isfinite(baseline) else np.nan
        )
    else:
        metrics["prp_slow_prob_post_error"] = np.nan
        metrics["prp_slow_prob_post_error_delta"] = np.nan

    stable_mask = np.zeros(len(correct), dtype=bool)
    run_len = 0
    for i in range(len(correct)):
        if correct[i]:
            run_len += 1
        else:
            run_len = 0
        if run_len >= stable_len:
            stable_mask[i] = True
    if stable_mask.any():
        stable_mean = float(np.mean(p_slow[stable_mask]))
        metrics["prp_slow_prob_stable"] = stable_mean
        metrics["prp_slow_prob_stable_delta"] = (
            stable_mean - baseline if np.isfinite(baseline) else np.nan
        )
    else:
        metrics["prp_slow_prob_stable"] = np.nan
        metrics["prp_slow_prob_stable_delta"] = np.nan

    return metrics


def compute_prp_hmm_event_features(
    data_dir: Path | None = None,
    n_states: int = 2,
    min_trials: int = 50,
) -> pd.DataFrame:
    if not HMM_AVAILABLE:
        return pd.DataFrame()

    trials, _ = load_prp_trials(
        data_dir=data_dir,
        apply_trial_filters=True,
    )

    if "t2_rt" not in trials.columns:
        return pd.DataFrame()

    trials["t2_rt"] = pd.to_numeric(trials["t2_rt"], errors="coerce")
    trials = trials[trials["t2_rt"].notna()]

    trial_col = None
    for cand in ("trial_index", "idx", "trial"):
        if cand in trials.columns:
            trial_col = cand
            break
    if trial_col:
        trials = trials.sort_values(["participant_id", trial_col])

    results = []
    for pid, pdata in trials.groupby("participant_id"):
        pdata = pdata.reset_index(drop=True)
        if len(pdata) < min_trials:
            continue

        rts = pdata["t2_rt"].values.reshape(-1, 1)
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
            slow_state = int(np.argmax(means))
            try:
                post = model.predict_proba(rts)
                p_slow = post[:, slow_state]
            except Exception:
                p_slow = (states == slow_state).astype(float)

            error = np.zeros(len(pdata), dtype=bool)
            if "t1_correct" in pdata.columns:
                error |= ~pdata["t1_correct"].astype(bool).values
            if "t2_correct" in pdata.columns:
                error |= ~pdata["t2_correct"].astype(bool).values
            if "t1_timeout" in pdata.columns:
                error |= pdata["t1_timeout"].astype(bool).values
            if "t2_timeout" in pdata.columns:
                error |= pdata["t2_timeout"].astype(bool).values

            if "t1_correct" in pdata.columns and "t2_correct" in pdata.columns:
                correct = (
                    pdata["t1_correct"].astype(bool).values
                    & pdata["t2_correct"].astype(bool).values
                )
            else:
                correct = ~error

            if "t1_timeout" in pdata.columns:
                correct &= ~pdata["t1_timeout"].astype(bool).values
            if "t2_timeout" in pdata.columns:
                correct &= ~pdata["t2_timeout"].astype(bool).values

            record = {"participant_id": pid}
            record.update(_compute_event_metrics(p_slow, error, correct))
            results.append(record)
        except Exception:
            continue

    return pd.DataFrame(results)


def load_or_compute_prp_hmm_event_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("prp")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        existing = pd.read_csv(output_path, encoding="utf-8-sig")
        if all(col in existing.columns for col in EVENT_COLUMNS):
            return existing

    features = compute_prp_hmm_event_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] PRP HMM event features saved: {output_path}")
    return features
