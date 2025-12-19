"""WCST RL mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..constants import WCST_VALID_CARDS, get_results_dir
from .loaders import load_wcst_trials

MECHANISM_FILENAME = "5_wcst_rl_mechanism_features.csv"


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
            reward = 1.0 if trial.get("correct", False) else 0.0
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


def load_or_compute_wcst_rl_mechanism_features(
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

    features = compute_wcst_rl_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] WCST RL mechanism features saved: {output_path}")
    return features
