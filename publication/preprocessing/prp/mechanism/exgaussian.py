"""PRP mechanism feature derivation (Ex-Gaussian)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import exponnorm

from ...constants import PRP_RT_MAX, get_results_dir
from ..trial_level_loaders import load_prp_trials

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
        bounds=[(100, PRP_RT_MAX), (10, 1000), (10, 2000)],
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
        apply_trial_filters=True,
    )

    trials["t2_rt"] = pd.to_numeric(trials["t2_rt"], errors="coerce")
    trials = trials[trials["t2_rt"].notna()]

    soa_levels = sorted(trials["soa"].dropna().unique()) if "soa" in trials.columns else []

    results = []
    for pid, group in trials.groupby("participant_id"):
        record = {"participant_id": pid}
        if "t2_correct" in group.columns:
            group_correct = group[group["t2_correct"] == True].copy()
        else:
            group_correct = group.copy()

        for soa in ("short", "long"):
            subset = group[group["soa_bin"] == soa]
            params = _fit_exgaussian(subset["t2_rt"].values, min_trials=min_trials)
            record[f"prp_exg_{soa}_mu"] = params["mu"]
            record[f"prp_exg_{soa}_sigma"] = params["sigma"]
            record[f"prp_exg_{soa}_tau"] = params["tau"]

            subset_correct = group_correct[group_correct["soa_bin"] == soa]
            params_correct = _fit_exgaussian(subset_correct["t2_rt"].values, min_trials=min_trials)
            record[f"prp_exg_correct_{soa}_mu"] = params_correct["mu"]
            record[f"prp_exg_correct_{soa}_sigma"] = params_correct["sigma"]
            record[f"prp_exg_correct_{soa}_tau"] = params_correct["tau"]

        overall_params = _fit_exgaussian(group["t2_rt"].values, min_trials=min_trials)
        record["prp_exg_overall_mu"] = overall_params["mu"]
        record["prp_exg_overall_sigma"] = overall_params["sigma"]
        record["prp_exg_overall_tau"] = overall_params["tau"]

        overall_correct = _fit_exgaussian(group_correct["t2_rt"].values, min_trials=min_trials)
        record["prp_exg_correct_overall_mu"] = overall_correct["mu"]
        record["prp_exg_correct_overall_sigma"] = overall_correct["sigma"]
        record["prp_exg_correct_overall_tau"] = overall_correct["tau"]

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

        record["prp_exg_correct_mu_bottleneck"] = (
            record["prp_exg_correct_short_mu"] - record["prp_exg_correct_long_mu"]
            if pd.notna(record["prp_exg_correct_short_mu"]) and pd.notna(record["prp_exg_correct_long_mu"])
            else np.nan
        )
        record["prp_exg_correct_sigma_bottleneck"] = (
            record["prp_exg_correct_short_sigma"] - record["prp_exg_correct_long_sigma"]
            if pd.notna(record["prp_exg_correct_short_sigma"]) and pd.notna(record["prp_exg_correct_long_sigma"])
            else np.nan
        )
        record["prp_exg_correct_tau_bottleneck"] = (
            record["prp_exg_correct_short_tau"] - record["prp_exg_correct_long_tau"]
            if pd.notna(record["prp_exg_correct_short_tau"]) and pd.notna(record["prp_exg_correct_long_tau"])
            else np.nan
        )

        if pd.notna(record["prp_exg_short_tau"]) and pd.notna(record["prp_exg_long_tau"]):
            if record["prp_exg_long_tau"] > 0:
                record["prp_exg_tau_ratio_short_long"] = record["prp_exg_short_tau"] / record["prp_exg_long_tau"]
            else:
                record["prp_exg_tau_ratio_short_long"] = np.nan
        else:
            record["prp_exg_tau_ratio_short_long"] = np.nan

        if pd.notna(record["prp_exg_correct_short_tau"]) and pd.notna(record["prp_exg_correct_long_tau"]):
            if record["prp_exg_correct_long_tau"] > 0:
                record["prp_exg_correct_tau_ratio_short_long"] = (
                    record["prp_exg_correct_short_tau"] / record["prp_exg_correct_long_tau"]
                )
            else:
                record["prp_exg_correct_tau_ratio_short_long"] = np.nan
        else:
            record["prp_exg_correct_tau_ratio_short_long"] = np.nan

        for soa_val in soa_levels:
            label = int(round(float(soa_val)))
            subset = group[group["soa"] == soa_val]
            params = _fit_exgaussian(subset["t2_rt"].values, min_trials=min_trials)
            record[f"prp_exg_soa_{label}_mu"] = params["mu"]
            record[f"prp_exg_soa_{label}_sigma"] = params["sigma"]
            record[f"prp_exg_soa_{label}_tau"] = params["tau"]

            subset_correct = group_correct[group_correct["soa"] == soa_val]
            params_correct = _fit_exgaussian(subset_correct["t2_rt"].values, min_trials=min_trials)
            record[f"prp_exg_correct_soa_{label}_mu"] = params_correct["mu"]
            record[f"prp_exg_correct_soa_{label}_sigma"] = params_correct["sigma"]
            record[f"prp_exg_correct_soa_{label}_tau"] = params_correct["tau"]

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
