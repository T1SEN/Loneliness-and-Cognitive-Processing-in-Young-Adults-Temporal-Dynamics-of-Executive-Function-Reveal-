"""WCST Bayesian rule learner mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..constants import get_results_dir
from .loaders import load_wcst_trials

MECHANISM_FILENAME = "5_wcst_bayesianrl_mechanism_features.csv"

REFERENCE_CARDS = [
    {"name": "one_yellow_circle", "count": 1, "color": "yellow", "shape": "circle"},
    {"name": "two_black_rectangle", "count": 2, "color": "black", "shape": "rectangle"},
    {"name": "three_blue_star", "count": 3, "color": "blue", "shape": "star"},
    {"name": "four_red_triangle", "count": 4, "color": "red", "shape": "triangle"},
]

CARD_ATTRS = {card["name"]: card for card in REFERENCE_CARDS}
CARD_ORDER = [card["name"] for card in REFERENCE_CARDS]
CARD_INDEX = {name: idx for idx, name in enumerate(CARD_ORDER)}
COLOR_TO_CARD = {card["color"]: card["name"] for card in REFERENCE_CARDS}
SHAPE_TO_CARD = {card["shape"]: card["name"] for card in REFERENCE_CARDS}
NUMBER_TO_CARD = {card["count"]: card["name"] for card in REFERENCE_CARDS}

RULES = ["colour", "shape", "number"]


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    total = np.sum(exp_x)
    if total <= 0:
        return np.ones_like(x) / len(x)
    return exp_x / total


def _prepare_brl_trials(df: pd.DataFrame) -> pd.DataFrame:
    chosen_col = _pick_column(df, ["chosenCard", "chosen_card"])
    color_col = _pick_column(df, ["cardColor", "card_color", "color"])
    shape_col = _pick_column(df, ["cardShape", "card_shape", "shape"])
    number_col = _pick_column(df, ["cardNumber", "card_number", "count"])
    trial_col = _pick_column(df, ["trialIndex", "trial_index", "trial"])

    required = [chosen_col, color_col, shape_col, number_col, "correct", "participant_id"]
    if any(col is None for col in required):
        return pd.DataFrame()

    out = df.copy()
    out["chosen_card"] = out[chosen_col].astype(str).str.strip().str.lower()
    out["stim_color_raw"] = out[color_col]
    out["stim_shape_raw"] = out[shape_col]
    out["stim_number_raw"] = out[number_col]

    out["stim_color"] = out["stim_color_raw"].astype(str).str.strip().str.lower()
    out["stim_shape"] = out["stim_shape_raw"].astype(str).str.strip().str.lower()
    out["stim_number"] = pd.to_numeric(out["stim_number_raw"], errors="coerce")

    out["chosen_idx"] = out["chosen_card"].map(CARD_INDEX)
    out["rule_idx_color"] = out["stim_color"].map(COLOR_TO_CARD).map(CARD_INDEX)
    out["rule_idx_shape"] = out["stim_shape"].map(SHAPE_TO_CARD).map(CARD_INDEX)
    out["rule_idx_number"] = out["stim_number"].map(NUMBER_TO_CARD).map(CARD_INDEX)

    valid_mask = (
        out["chosen_idx"].notna()
        & out["rule_idx_color"].notna()
        & out["rule_idx_shape"].notna()
        & out["rule_idx_number"].notna()
        & out["correct"].notna()
    )

    out = out[valid_mask].copy()
    if out.empty:
        return out

    if trial_col:
        out["trial_order"] = pd.to_numeric(out[trial_col], errors="coerce")
    else:
        out["trial_order"] = np.arange(len(out))

    return out


class BayesianRuleLearner:
    def __init__(self, hazard: float, noise: float, beta: float):
        self.hazard = hazard
        self.noise = noise
        self.beta = beta
        self.n_rules = len(RULES)
        self.belief = np.ones(self.n_rules) / self.n_rules

    def reset(self) -> None:
        self.belief = np.ones(self.n_rules) / self.n_rules

    def _apply_hazard(self) -> np.ndarray:
        switch_prob = self.hazard
        return (
            (1.0 - switch_prob) * self.belief
            + switch_prob * (1.0 - self.belief) / (self.n_rules - 1)
        )

    def _choice_prob(self, prior: np.ndarray, rule_indices: Tuple[int, int, int]) -> np.ndarray:
        weights = np.zeros(len(CARD_ORDER))
        for idx, card_idx in enumerate(rule_indices):
            weights[int(card_idx)] += prior[idx]
        logits = self.beta * weights
        return _softmax(logits)

    def negative_log_likelihood(self, trials: pd.DataFrame) -> Tuple[float, int]:
        self.reset()
        nll = 0.0
        n_used = 0

        for row in trials.itertuples(index=False):
            rule_indices = (
                int(row.rule_idx_color),
                int(row.rule_idx_shape),
                int(row.rule_idx_number),
            )
            chosen_idx = int(row.chosen_idx)
            correct = bool(row.correct)

            prior = self._apply_hazard()
            probs = self._choice_prob(prior, rule_indices)
            prob = probs[chosen_idx]
            nll -= np.log(max(prob, 1e-12))

            likelihood = np.zeros(self.n_rules)
            for r_idx, card_idx in enumerate(rule_indices):
                predicts_correct = card_idx == chosen_idx
                if correct:
                    likelihood[r_idx] = (1.0 - self.noise) if predicts_correct else self.noise
                else:
                    likelihood[r_idx] = (1.0 - self.noise) if not predicts_correct else self.noise

            posterior = prior * likelihood
            total = posterior.sum()
            if total <= 0:
                posterior = np.ones(self.n_rules) / self.n_rules
            else:
                posterior = posterior / total
            self.belief = posterior
            n_used += 1

        return nll, n_used


def _fit_bayesian_rule_model(
    trials: pd.DataFrame,
    bounds: List[Tuple[float, float]],
    n_restarts: int = 5,
) -> Optional[Dict[str, float]]:
    if len(trials) < 20:
        return None

    best_nll = np.inf
    best_params = None
    best_n = 0

    def objective(params: List[float]) -> float:
        hazard, noise, beta = params
        model = BayesianRuleLearner(hazard, noise, beta)
        nll, _ = model.negative_log_likelihood(trials)
        return nll

    for _ in range(n_restarts):
        x0 = [np.random.uniform(low, high) for low, high in bounds]
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

    model = BayesianRuleLearner(*best_params)
    nll, n_used = model.negative_log_likelihood(trials)
    return {
        "hazard": float(best_params[0]),
        "noise": float(best_params[1]),
        "beta": float(best_params[2]),
        "neg_loglik": float(nll),
        "n_trials": int(n_used),
    }


def compute_wcst_bayesianrl_features(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    trials, _ = load_wcst_trials(data_dir=data_dir, apply_trial_filters=True)
    prepared = _prepare_brl_trials(trials)
    if prepared.empty:
        return pd.DataFrame()

    results = []
    bounds = [(0.001, 0.3), (0.001, 0.3), (0.1, 10.0)]

    for pid, pdata in prepared.groupby("participant_id"):
        pdata = pdata.sort_values("trial_order")
        fit = _fit_bayesian_rule_model(pdata, bounds=bounds)

        record = {"participant_id": pid}
        if fit is None:
            record.update({
                "wcst_brl_hazard": np.nan,
                "wcst_brl_noise": np.nan,
                "wcst_brl_beta": np.nan,
                "wcst_brl_negloglik": np.nan,
                "wcst_brl_n_trials": np.nan,
            })
        else:
            record.update({
                "wcst_brl_hazard": fit["hazard"],
                "wcst_brl_noise": fit["noise"],
                "wcst_brl_beta": fit["beta"],
                "wcst_brl_negloglik": fit["neg_loglik"],
                "wcst_brl_n_trials": fit["n_trials"],
            })
        results.append(record)

    return pd.DataFrame(results)


def load_or_compute_wcst_bayesianrl_mechanism_features(
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

    features = compute_wcst_bayesianrl_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] WCST Bayesian RL mechanism features saved: {output_path}")
    return features
