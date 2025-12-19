"""WCST mechanism feature derivation (HMM + RL)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..constants import WCST_VALID_CARDS, get_results_dir
from .loaders import load_wcst_trials

MECHANISM_FILENAME = "5_wcst_mechanism_features.csv"

try:
    from hmmlearn import hmm as hmm_module
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


WCST_CARD_ORDER = [
    "one_yellow_circle",
    "two_black_rectangle",
    "three_blue_star",
    "four_red_triangle",
]


def _prepare_actions(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()
    if "chosenCard" not in trials.columns:
        return trials

    trials["chosenCard"] = trials["chosenCard"].astype(str).str.strip().str.lower()
    card_map = {card: idx for idx, card in enumerate(WCST_CARD_ORDER)}
    trials["action"] = trials["chosenCard"].map(card_map)

    missing_map = trials["action"].isna().any()
    if missing_map:
        valid_cards = sorted(
            [card for card in trials["chosenCard"].dropna().unique() if card in WCST_VALID_CARDS]
        )
        if valid_cards:
            fallback_map = {card: idx for idx, card in enumerate(valid_cards)}
            trials["action"] = trials["chosenCard"].map(fallback_map)

    return trials


def compute_wcst_hmm_features(
    data_dir: Path | None = None,
    n_states: int = 2,
    min_trials: int = 50,
) -> pd.DataFrame:
    if not HMM_AVAILABLE:
        return pd.DataFrame()

    trials, _ = load_wcst_trials(data_dir=data_dir, filter_rt=True)

    rt_col = "rt_ms" if "rt_ms" in trials.columns else "reactiontimems" if "reactiontimems" in trials.columns else None
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


class RescorlaWagnerModel:
    def __init__(self, alpha: float = 0.5, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.n_actions = 4
        self.Q = np.zeros(self.n_actions)

    def reset(self) -> None:
        self.Q = np.zeros(self.n_actions)

    def get_action_prob(self, action: int) -> float:
        exp_q = np.exp(self.beta * (self.Q - np.max(self.Q)))
        probs = exp_q / np.sum(exp_q)
        return probs[action]

    def update(self, action: int, reward: float) -> None:
        prediction_error = reward - self.Q[action]
        self.Q[action] += self.alpha * prediction_error

    def negative_log_likelihood(self, trials: pd.DataFrame) -> float:
        self.reset()
        nll = 0.0
        for _, trial in trials.iterrows():
            action = trial.get("action", np.nan)
            if pd.isna(action):
                continue
            action = int(action) % self.n_actions
            reward = 1.0 if bool(trial.get("correct", False)) else 0.0
            prob = self.get_action_prob(action)
            nll -= np.log(max(prob, 1e-10))
            self.update(action, reward)
        return nll


class AsymmetricRWModel(RescorlaWagnerModel):
    def __init__(self, alpha_pos: float = 0.5, alpha_neg: float = 0.5, beta: float = 1.0):
        super().__init__(alpha_pos, beta)
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def update(self, action: int, reward: float) -> None:
        prediction_error = reward - self.Q[action]
        if prediction_error >= 0:
            self.Q[action] += self.alpha_pos * prediction_error
        else:
            self.Q[action] += self.alpha_neg * prediction_error


def _fit_model(
    model_class,
    trials: pd.DataFrame,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    n_restarts: int = 3,
) -> Dict[str, float] | None:
    if len(trials) < 20:
        return None

    best_nll = np.inf
    best_params = None

    def objective(params: List[float]) -> float:
        try:
            model = model_class(*params)
            return model.negative_log_likelihood(trials)
        except Exception:
            return np.inf

    for _ in range(n_restarts):
        x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
        try:
            result = minimize(
                objective,
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 500},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return None

    result = {}
    for name, value in zip(param_names, best_params):
        result[name] = float(value)
    result["n_trials"] = int(len(trials))
    return result


def compute_wcst_rl_features(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    trials, _ = load_wcst_trials(data_dir=data_dir, filter_rt=False)
    trials = _prepare_actions(trials)

    if "action" not in trials.columns:
        return pd.DataFrame()
    trials = trials[trials["action"].notna()].copy()

    results = []
    for pid, pdata in trials.groupby("participant_id"):
        record = {"participant_id": pid}

        basic = _fit_model(
            RescorlaWagnerModel,
            pdata,
            bounds=[(0.01, 0.99), (0.1, 10.0)],
            param_names=["alpha", "beta"],
        )
        if basic is not None:
            record["wcst_rl_alpha"] = basic["alpha"]
            record["wcst_rl_beta"] = basic["beta"]
            record["wcst_rl_n_trials"] = basic["n_trials"]
        else:
            record["wcst_rl_alpha"] = np.nan
            record["wcst_rl_beta"] = np.nan
            record["wcst_rl_n_trials"] = np.nan

        asym = _fit_model(
            AsymmetricRWModel,
            pdata,
            bounds=[(0.01, 0.99), (0.01, 0.99), (0.1, 10.0)],
            param_names=["alpha_pos", "alpha_neg", "beta"],
        )
        if asym is not None:
            record["wcst_rl_alpha_pos"] = asym["alpha_pos"]
            record["wcst_rl_alpha_neg"] = asym["alpha_neg"]
            record["wcst_rl_alpha_asymmetry"] = asym["alpha_pos"] - asym["alpha_neg"]
            record["wcst_rl_beta_asym"] = asym["beta"]
            record["wcst_rl_asym_n_trials"] = asym["n_trials"]
        else:
            record["wcst_rl_alpha_pos"] = np.nan
            record["wcst_rl_alpha_neg"] = np.nan
            record["wcst_rl_alpha_asymmetry"] = np.nan
            record["wcst_rl_beta_asym"] = np.nan
            record["wcst_rl_asym_n_trials"] = np.nan

        results.append(record)

    return pd.DataFrame(results)


def compute_wcst_mechanism_features(data_dir: Path | None = None) -> pd.DataFrame:
    hmm_df = compute_wcst_hmm_features(data_dir=data_dir)
    rl_df = compute_wcst_rl_features(data_dir=data_dir)

    if hmm_df.empty and rl_df.empty:
        return pd.DataFrame()
    if hmm_df.empty:
        return rl_df
    if rl_df.empty:
        return hmm_df

    overlap = [c for c in rl_df.columns if c != "participant_id" and c in hmm_df.columns]
    if overlap:
        rl_df = rl_df.drop(columns=overlap)

    return hmm_df.merge(rl_df, on="participant_id", how="outer")


def load_or_compute_wcst_mechanism_features(
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

    features = compute_wcst_mechanism_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] WCST mechanism features saved: {output_path}")
    return features
