"""PRP mechanism feature derivation (Ex-Gaussian)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import exponnorm

from ..constants import (
    PRP_IRI_MIN,
    PRP_RT_MIN,
    PRP_RT_MAX,
    get_results_dir,
)
from .loaders import load_prp_trials

MECHANISM_FILENAME = "5_prp_mechanism_features.csv"


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
        bounds=[(50, 1500), (10, 1000), (10, 2000)],
        method="L-BFGS-B",
    )
    if not result.success:
        return {"mu": np.nan, "sigma": np.nan, "tau": np.nan}

    mu, sigma, tau = result.x
    return {"mu": float(mu), "sigma": float(sigma), "tau": float(tau)}


def compute_prp_exgaussian_features(
    data_dir: Path | None = None,
    min_trials: int = 20,
) -> pd.DataFrame:
    trials, _ = load_prp_trials(
        data_dir=data_dir,
        rt_min=PRP_RT_MIN,
        rt_max=PRP_RT_MAX,
        require_t1_correct=True,
        require_t2_correct_for_rt=True,
        enforce_short_long_only=True,
        drop_timeouts=True,
        require_valid_order=True,
        iri_min=PRP_IRI_MIN,
        exclude_t2_pressed_while_pending=True,
    )

    trials["t2_rt"] = pd.to_numeric(trials["t2_rt"], errors="coerce")
    trials = trials[trials["t2_rt"].notna()]

    results = []
    for pid, group in trials.groupby("participant_id"):
        record = {"participant_id": pid}
        for soa in ("short", "long"):
            subset = group[group["soa_bin"] == soa]
            params = _fit_exgaussian(subset["t2_rt"].values, min_trials=min_trials)
            record[f"prp_exg_{soa}_mu"] = params["mu"]
            record[f"prp_exg_{soa}_sigma"] = params["sigma"]
            record[f"prp_exg_{soa}_tau"] = params["tau"]

        overall_params = _fit_exgaussian(group["t2_rt"].values, min_trials=min_trials)
        record["prp_exg_overall_mu"] = overall_params["mu"]
        record["prp_exg_overall_sigma"] = overall_params["sigma"]
        record["prp_exg_overall_tau"] = overall_params["tau"]

        record["prp_exg_mu_bottleneck"] = (
            record["prp_exg_short_mu"] - record["prp_exg_long_mu"]
            if pd.notna(record["prp_exg_short_mu"]) and pd.notna(record["prp_exg_long_mu"])
            else np.nan
        )
        record["prp_exg_sigma_bottleneck"] = (
            record["prp_exg_short_sigma"] - record["prp_exg_long_sigma"]
            if pd.notna(record["prp_exg_short_sigma"]) and pd.notna(record["prp_exg_long_sigma"])
            else np.nan
        )
        record["prp_exg_tau_bottleneck"] = (
            record["prp_exg_short_tau"] - record["prp_exg_long_tau"]
            if pd.notna(record["prp_exg_short_tau"]) and pd.notna(record["prp_exg_long_tau"])
            else np.nan
        )

        results.append(record)

    return pd.DataFrame(results)


def load_or_compute_prp_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("prp")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_prp_exgaussian_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] PRP mechanism features saved: {output_path}")
    return features
