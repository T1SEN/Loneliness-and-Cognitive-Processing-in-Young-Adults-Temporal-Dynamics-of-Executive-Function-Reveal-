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
            }
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
