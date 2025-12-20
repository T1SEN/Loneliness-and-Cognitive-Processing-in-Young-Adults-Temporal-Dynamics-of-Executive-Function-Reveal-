"""WCST HMM mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import get_results_dir
from .loaders import load_wcst_trials

MECHANISM_FILENAME = "5_wcst_hmm_mechanism_features.csv"

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
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_wcst_hmm_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] WCST HMM mechanism features saved: {output_path}")
    return features
