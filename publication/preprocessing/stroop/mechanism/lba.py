"""Stroop mechanism feature derivation (LBA)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ...constants import get_results_dir
from ..loaders import load_stroop_trials

MECHANISM_FILENAME = "5_stroop_lba_mechanism_features.csv"


def _lba_cdf(t: np.ndarray, v: float, sv: float, A: float, b: float) -> np.ndarray:
    t = np.maximum(t, 1e-9)
    x = (b - A - v * t) / (sv * t)
    y = (b - v * t) / (sv * t)
    cdf = (
        1.0
        - ((b - A - v * t) / A) * norm.cdf(x)
        - ((b - v * t) / A) * norm.cdf(y)
        + ((sv * t) / A) * (norm.pdf(x) - norm.pdf(y))
    )
    return np.clip(cdf, 0.0, 1.0)


def _lba_pdf(t: np.ndarray, v: float, sv: float, A: float, b: float) -> np.ndarray:
    t = np.maximum(t, 1e-9)
    x = (b - A - v * t) / (sv * t)
    y = (b - v * t) / (sv * t)
    pdf = (1.0 / A) * (v * (norm.cdf(x) - norm.cdf(y)) + sv * (norm.pdf(x) - norm.pdf(y)))
    return np.maximum(pdf, 1e-12)


def _lba_nll(
    params: np.ndarray,
    rts_sec: np.ndarray,
    correct: np.ndarray,
    n_accumulators: int,
    sv: float,
) -> float:
    v_correct, v_incorrect, A, k, t0 = params
    if v_correct <= v_incorrect:
        return 1e12
    if A <= 0 or k <= 0:
        return 1e12
    b = A + k
    min_rt = float(np.min(rts_sec))
    if t0 <= 0 or t0 >= min_rt:
        return 1e12

    t = rts_sec - t0
    f_c = _lba_pdf(t, v_correct, sv, A, b)
    F_c = _lba_cdf(t, v_correct, sv, A, b)
    f_i = _lba_pdf(t, v_incorrect, sv, A, b)
    F_i = _lba_cdf(t, v_incorrect, sv, A, b)

    n_other = n_accumulators - 1
    prob_correct = f_c * (1.0 - F_i) ** n_other
    prob_incorrect = n_other * f_i * (1.0 - F_i) ** (n_other - 1) * (1.0 - F_c)

    prob = np.where(correct, prob_correct, prob_incorrect)
    return -float(np.sum(np.log(np.clip(prob, 1e-12, None))))


def _fit_lba_model(
    rts_ms: np.ndarray,
    correct: np.ndarray,
    n_accumulators: int = 4,
    min_trials: int = 50,
    n_restarts: int = 3,
) -> Dict[str, float]:
    rts_ms = np.asarray(rts_ms, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    if len(rts_ms) < min_trials or len(np.unique(correct)) < 2:
        return {}

    rts_sec = rts_ms / 1000.0
    sv = 1.0
    bounds = [
        (0.5, 5.0),   # v_correct
        (0.1, 3.0),   # v_incorrect
        (0.1, 1.0),   # A
        (0.1, 1.5),   # k (b = A + k)
        (0.1, 0.5),   # t0
    ]

    rng = np.random.default_rng(42)
    best = None
    best_nll = np.inf

    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)
        result = minimize(
            _lba_nll,
            x0=x0,
            args=(rts_sec, correct, n_accumulators, sv),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.success and result.fun < best_nll:
            best_nll = float(result.fun)
            best = result.x

    if best is None:
        return {}

    v_correct, v_incorrect, A, k, t0 = best
    b = A + k
    return {
        "v_correct": float(v_correct),
        "v_incorrect": float(v_incorrect),
        "A": float(A),
        "b": float(b),
        "t0": float(t0),
        "n_trials": int(len(rts_ms)),
        "nll": float(best_nll),
        "converged": True,
    }


def compute_stroop_lba_features(
    data_dir: Path | None = None,
    min_trials: int = 50,
    min_trials_per_condition: int = 15,
) -> pd.DataFrame:
    trials, _ = load_stroop_trials(
        data_dir=data_dir,
        apply_trial_filters=True,
    )

    rt_col = "rt" if "rt" in trials.columns else "rt_ms"
    trials["rt_ms"] = pd.to_numeric(trials[rt_col], errors="coerce")
    trials = trials[trials["rt_ms"].notna()]

    cond_col = None
    for cand in ("type", "condition", "cond"):
        if cand in trials.columns:
            cond_col = cand
            break
    if cond_col is None:
        return pd.DataFrame()

    trials["condition"] = trials[cond_col].astype(str).str.lower().str.strip()
    valid_conditions = ("congruent", "incongruent", "neutral")
    trials = trials[trials["condition"].isin(valid_conditions)]

    results = []
    for pid, group in trials.groupby("participant_id"):
        record = {"participant_id": pid}
        total_trials = int(len(group))

        for cond in valid_conditions:
            subset = group[group["condition"] == cond]
            n_cond = int(len(subset))
            fit = {}
            if total_trials >= min_trials and n_cond >= min_trials_per_condition:
                fit = _fit_lba_model(
                    subset["rt_ms"].values,
                    subset["correct"].values,
                    min_trials=min_trials_per_condition,
                )

            record[f"stroop_lba_{cond}_v_correct"] = fit.get("v_correct", np.nan)
            record[f"stroop_lba_{cond}_v_incorrect"] = fit.get("v_incorrect", np.nan)
            record[f"stroop_lba_{cond}_A"] = fit.get("A", np.nan)
            record[f"stroop_lba_{cond}_b"] = fit.get("b", np.nan)
            record[f"stroop_lba_{cond}_t0"] = fit.get("t0", np.nan)
            record[f"stroop_lba_{cond}_n_trials"] = n_cond
            record[f"stroop_lba_{cond}_negloglik"] = fit.get("nll", np.nan)
            record[f"stroop_lba_{cond}_converged"] = fit.get("converged", False)

            if "nll" in fit and pd.notna(fit["nll"]):
                k_params = 5
                n_obs = max(n_cond, 1)
                aic = 2 * k_params + 2 * fit["nll"]
                bic = k_params * np.log(n_obs) + 2 * fit["nll"]
                record[f"stroop_lba_{cond}_aic"] = float(aic)
                record[f"stroop_lba_{cond}_bic"] = float(bic)
            else:
                record[f"stroop_lba_{cond}_aic"] = np.nan
                record[f"stroop_lba_{cond}_bic"] = np.nan

        if (
            pd.notna(record["stroop_lba_congruent_v_correct"])
            and pd.notna(record["stroop_lba_incongruent_v_correct"])
        ):
            record["stroop_lba_v_correct_interference"] = (
                record["stroop_lba_congruent_v_correct"]
                - record["stroop_lba_incongruent_v_correct"]
            )
        else:
            record["stroop_lba_v_correct_interference"] = np.nan

        if (
            pd.notna(record["stroop_lba_congruent_b"])
            and pd.notna(record["stroop_lba_incongruent_b"])
        ):
            record["stroop_lba_b_interference"] = (
                record["stroop_lba_incongruent_b"] - record["stroop_lba_congruent_b"]
            )
        else:
            record["stroop_lba_b_interference"] = np.nan

        if (
            pd.notna(record["stroop_lba_congruent_t0"])
            and pd.notna(record["stroop_lba_incongruent_t0"])
        ):
            record["stroop_lba_t0_interference"] = (
                record["stroop_lba_incongruent_t0"] - record["stroop_lba_congruent_t0"]
            )
        else:
            record["stroop_lba_t0_interference"] = np.nan

        results.append(record)

    return pd.DataFrame(results)


def load_or_compute_stroop_lba_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("stroop")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_stroop_lba_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] Stroop LBA features saved: {output_path}")
    return features
