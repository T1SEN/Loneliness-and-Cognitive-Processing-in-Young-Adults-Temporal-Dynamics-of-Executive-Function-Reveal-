"""Stroop mechanism feature derivation (Ex-Gaussian)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import exponnorm

from ..constants import STROOP_RT_MIN, STROOP_RT_MAX, get_results_dir
from .loaders import load_stroop_trials

MECHANISM_FILENAME = "5_stroop_mechanism_features.csv"


def _fit_exgaussian(rts: np.ndarray, min_trials: int = 20) -> Dict[str, float]:
    rts = np.asarray(rts)
    rts = rts[~np.isnan(rts)]
    if len(rts) < min_trials:
        return {"mu": np.nan, "sigma": np.nan, "tau": np.nan}

    m = np.mean(rts)
    s = np.std(rts)
    if s <= 0:
        return {"mu": np.nan, "sigma": np.nan, "tau": np.nan}

    skew = np.mean(((rts - m) / s) ** 3)
    tau_init = max(10, (abs(skew) / 2) ** (1 / 3) * s)
    mu_init = max(100, m - tau_init)
    sigma_init = max(10, np.sqrt(max(0, s ** 2 - tau_init ** 2)))

    def neg_loglik(params: np.ndarray) -> float:
        mu, sigma, tau = params
        if sigma <= 0 or tau <= 0:
            return 1e10
        k = tau / sigma
        try:
            return -np.sum(exponnorm.logpdf(rts, k, loc=mu, scale=sigma))
        except Exception:
            return 1e10

    result = minimize(
        neg_loglik,
        x0=[mu_init, sigma_init, tau_init],
        bounds=[(100, STROOP_RT_MAX), (10, 1000), (10, 2000)],
        method="L-BFGS-B",
    )
    if not result.success:
        return {"mu": np.nan, "sigma": np.nan, "tau": np.nan}

    mu, sigma, tau = result.x
    return {"mu": float(mu), "sigma": float(sigma), "tau": float(tau)}


def compute_stroop_exgaussian_features(
    data_dir: Path | None = None,
    min_trials: int = 20,
) -> pd.DataFrame:
    trials, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=STROOP_RT_MIN,
        rt_max=STROOP_RT_MAX,
        require_correct_for_rt=True,
        drop_timeouts=True,
    )

    rt_col = "rt" if "rt" in trials.columns else "rt_ms"
    trials["rt_ms"] = pd.to_numeric(trials[rt_col], errors="coerce")
    trials = trials[trials["rt_ms"].notna()]

    cond_col = None
    for cand in ("type", "condition", "cond", "congruency"):
        if cand in trials.columns:
            cond_col = cand
            break
    if cond_col is None:
        return pd.DataFrame()

    trials["condition"] = trials[cond_col].astype(str).str.lower().str.strip()
    valid_conditions = {"congruent", "incongruent", "neutral"}
    trials = trials[trials["condition"].isin(valid_conditions)]

    results = []
    for pid, group in trials.groupby("participant_id"):
        record = {"participant_id": pid}

        for cond in ("congruent", "incongruent", "neutral"):
            subset = group[group["condition"] == cond]
            params = _fit_exgaussian(subset["rt_ms"].values, min_trials=min_trials)
            record[f"stroop_exg_{cond}_mu"] = params["mu"]
            record[f"stroop_exg_{cond}_sigma"] = params["sigma"]
            record[f"stroop_exg_{cond}_tau"] = params["tau"]

        record["stroop_exg_mu_interference"] = (
            record["stroop_exg_incongruent_mu"] - record["stroop_exg_congruent_mu"]
            if pd.notna(record["stroop_exg_incongruent_mu"]) and pd.notna(record["stroop_exg_congruent_mu"])
            else np.nan
        )
        record["stroop_exg_sigma_interference"] = (
            record["stroop_exg_incongruent_sigma"] - record["stroop_exg_congruent_sigma"]
            if pd.notna(record["stroop_exg_incongruent_sigma"]) and pd.notna(record["stroop_exg_congruent_sigma"])
            else np.nan
        )
        record["stroop_exg_tau_interference"] = (
            record["stroop_exg_incongruent_tau"] - record["stroop_exg_congruent_tau"]
            if pd.notna(record["stroop_exg_incongruent_tau"]) and pd.notna(record["stroop_exg_congruent_tau"])
            else np.nan
        )

        results.append(record)

    return pd.DataFrame(results)


def load_or_compute_stroop_mechanism_features(
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

    features = compute_stroop_exgaussian_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] Stroop mechanism features saved: {output_path}")
    return features
