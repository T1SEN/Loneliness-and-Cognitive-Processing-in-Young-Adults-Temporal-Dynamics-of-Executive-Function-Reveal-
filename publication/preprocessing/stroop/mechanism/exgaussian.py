"""Stroop mechanism feature derivation (Ex-Gaussian)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import exponnorm, lognorm, norm

from ...constants import STROOP_RT_MAX, get_results_dir
from ..loaders import load_stroop_trials

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


def _fit_shifted_lognormal(rts: np.ndarray, min_trials: int = 20) -> Dict[str, float]:
    rts = np.asarray(rts, dtype=float)
    rts = rts[~np.isnan(rts)]
    if len(rts) < min_trials:
        return {"mu": np.nan, "sigma": np.nan, "shift": np.nan}

    min_rt = float(np.min(rts))
    shift = max(0.0, min_rt - 1.0)
    shifted = rts - shift
    if np.any(shifted <= 0):
        shift = max(0.0, min_rt * 0.9)
        shifted = rts - shift
    if np.any(shifted <= 0):
        return {"mu": np.nan, "sigma": np.nan, "shift": np.nan}

    try:
        shape, _, scale = lognorm.fit(shifted, floc=0)
    except Exception:
        return {"mu": np.nan, "sigma": np.nan, "shift": np.nan}

    return {"mu": float(np.log(scale)), "sigma": float(shape), "shift": float(shift)}


def _fit_log_mixture(
    rts: np.ndarray,
    min_trials: int = 30,
    n_iter: int = 100,
) -> Dict[str, float]:
    rts = np.asarray(rts, dtype=float)
    rts = rts[~np.isnan(rts)]
    if len(rts) < min_trials:
        return {"slow_prop": np.nan}

    log_rts = np.log(rts)
    mu1, mu2 = np.quantile(log_rts, [0.3, 0.7])
    var = float(np.var(log_rts)) if np.var(log_rts) > 0 else 1.0
    var1 = var
    var2 = var
    w1 = 0.5
    w2 = 0.5

    for _ in range(n_iter):
        sd1 = np.sqrt(var1)
        sd2 = np.sqrt(var2)
        p1 = w1 * norm.pdf(log_rts, mu1, sd1)
        p2 = w2 * norm.pdf(log_rts, mu2, sd2)
        total = p1 + p2
        if np.any(total <= 0):
            break
        r1 = p1 / total
        r2 = 1.0 - r1
        w1 = float(np.mean(r1))
        w2 = 1.0 - w1
        mu1 = float(np.sum(r1 * log_rts) / max(r1.sum(), 1e-9))
        mu2 = float(np.sum(r2 * log_rts) / max(r2.sum(), 1e-9))
        var1 = float(np.sum(r1 * (log_rts - mu1) ** 2) / max(r1.sum(), 1e-9))
        var2 = float(np.sum(r2 * (log_rts - mu2) ** 2) / max(r2.sum(), 1e-9))
        var1 = max(var1, 1e-6)
        var2 = max(var2, 1e-6)

    if mu1 >= mu2:
        slow_prop = w1
    else:
        slow_prop = w2
    return {"slow_prop": float(slow_prop)}


def compute_stroop_exgaussian_features(
    data_dir: Path | None = None,
    min_trials: int = 20,
) -> pd.DataFrame:
    trials, _ = load_stroop_trials(
        data_dir=data_dir,
        apply_trial_filters=True,
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
        group_all = group.copy()
        if "correct" in group_all.columns:
            group_correct = group_all[group_all["correct"] == True].copy()
        else:
            group_correct = group_all.copy()

        for cond in ("congruent", "incongruent", "neutral"):
            subset = group_all[group_all["condition"] == cond]
            params = _fit_exgaussian(subset["rt_ms"].values, min_trials=min_trials)
            record[f"stroop_exg_{cond}_mu"] = params["mu"]
            record[f"stroop_exg_{cond}_sigma"] = params["sigma"]
            record[f"stroop_exg_{cond}_tau"] = params["tau"]

            subset_correct = group_correct[group_correct["condition"] == cond]
            params_correct = _fit_exgaussian(subset_correct["rt_ms"].values, min_trials=min_trials)
            record[f"stroop_exg_correct_{cond}_mu"] = params_correct["mu"]
            record[f"stroop_exg_correct_{cond}_sigma"] = params_correct["sigma"]
            record[f"stroop_exg_correct_{cond}_tau"] = params_correct["tau"]

            lognorm_params = _fit_shifted_lognormal(subset_correct["rt_ms"].values, min_trials=min_trials)
            record[f"stroop_lognorm_correct_{cond}_mu"] = lognorm_params["mu"]
            record[f"stroop_lognorm_correct_{cond}_sigma"] = lognorm_params["sigma"]
            record[f"stroop_lognorm_correct_{cond}_shift"] = lognorm_params["shift"]

            mix_params = _fit_log_mixture(subset_correct["rt_ms"].values, min_trials=min_trials)
            record[f"stroop_mix_slow_prop_correct_{cond}"] = mix_params["slow_prop"]

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

        record["stroop_exg_correct_mu_interference"] = (
            record["stroop_exg_correct_incongruent_mu"] - record["stroop_exg_correct_congruent_mu"]
            if pd.notna(record["stroop_exg_correct_incongruent_mu"]) and pd.notna(record["stroop_exg_correct_congruent_mu"])
            else np.nan
        )
        record["stroop_exg_correct_sigma_interference"] = (
            record["stroop_exg_correct_incongruent_sigma"] - record["stroop_exg_correct_congruent_sigma"]
            if pd.notna(record["stroop_exg_correct_incongruent_sigma"]) and pd.notna(record["stroop_exg_correct_congruent_sigma"])
            else np.nan
        )
        record["stroop_exg_correct_tau_interference"] = (
            record["stroop_exg_correct_incongruent_tau"] - record["stroop_exg_correct_congruent_tau"]
            if pd.notna(record["stroop_exg_correct_incongruent_tau"]) and pd.notna(record["stroop_exg_correct_congruent_tau"])
            else np.nan
        )

        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        cong_rts = group_correct[group_correct["condition"] == "congruent"]["rt_ms"].dropna().values
        incong_rts = group_correct[group_correct["condition"] == "incongruent"]["rt_ms"].dropna().values
        if len(cong_rts) >= min_trials and len(incong_rts) >= min_trials:
            cong_q = np.quantile(cong_rts, quantiles)
            incong_q = np.quantile(incong_rts, quantiles)
            diffs = incong_q - cong_q
            for q, diff in zip(quantiles, diffs):
                label = int(q * 100)
                record[f"stroop_vincentile_interference_p{label}_correct"] = float(diff)
            record["stroop_delta_plot_slope_correct"] = float(diffs[-1] - diffs[0])
        else:
            for q in quantiles:
                label = int(q * 100)
                record[f"stroop_vincentile_interference_p{label}_correct"] = np.nan
            record["stroop_delta_plot_slope_correct"] = np.nan

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
