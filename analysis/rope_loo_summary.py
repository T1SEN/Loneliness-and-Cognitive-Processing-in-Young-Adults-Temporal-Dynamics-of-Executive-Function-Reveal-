"""
ROPE/SESOI and LOO comparison for trial-level GLMMs
---------------------------------------------------
This script:
1) Loads trial-level GLMM traces (with UCLA) and computes ROPE probabilities
   for key coefficients across several ROPE widths.
2) Refits reduced models (without UCLA and its interaction) and compares
   models via LOO (elpd) and stacking weights.

Outputs (results/analysis_outputs/):
- rope_summary_glmm.csv
- loo_compare_glmm_stroop.csv
- loo_compare_glmm_prp.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_stroop_trials, load_prp_trials


BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().std() in (0, None) or s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / s.std()


def load_covariates() -> pd.DataFrame:
    """
    Pull covariates from the unified master dataset (UCLA + DASS with z-scores).
    Falls back to on-the-fly z-scoring if precomputed cols are missing.
    """
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
    cov = master.copy()

    # Harmonize column names used in this script
    if "ucla_total" not in cov.columns and "ucla_score" in cov.columns:
        cov["ucla_total"] = cov["ucla_score"]

    if "z_ucla" not in cov.columns:
        base = cov["ucla_total"] if "ucla_total" in cov.columns else cov.get("ucla_score")
        cov["z_ucla"] = _z(base) if base is not None else np.nan

    cov["z_dass_dep"] = cov["z_dass_depression"] if "z_dass_depression" in cov.columns else _z(cov.get("dass_depression"))
    cov["z_dass_anx"] = cov["z_dass_anxiety"] if "z_dass_anxiety" in cov.columns else _z(cov.get("dass_anxiety"))
    cov["z_dass_stress"] = cov["z_dass_stress"] if "z_dass_stress" in cov.columns else _z(cov.get("dass_stress"))

    needed = ["participant_id", "ucla_total", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    cov = cov[[c for c in needed if c in cov.columns]].dropna(subset=["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"])
    cov["participant_id"] = cov["participant_id"].astype(str)
    return cov


def prepare_stroop(cov: pd.DataFrame) -> pd.DataFrame:
    trials, _ = load_stroop_trials(use_cache=True, rt_min=200, rt_max=3000, drop_timeouts=True, require_correct_for_rt=True)
    cond_col = None
    for cand in ("type", "condition", "cond"):
        if cand in trials.columns:
            cond_col = cand
            break
    if cond_col is None:
        raise RuntimeError("Stroop trials missing condition column")

    cond = trials[cond_col].astype(str).str.lower()
    code_map = {"congruent": 0.0, "neutral": 0.5, "incongruent": 1.0}
    trials["condition_code"] = cond.map(code_map)
    trials = trials[trials["condition_code"].notna()]

    trials["participant_id"] = trials["participant_id"].astype(str)
    df = trials.merge(cov, on="participant_id", how="inner").dropna(
        subset=["participant_id", "rt", "condition_code", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    )
    df["rt_sec"] = df["rt"] / 1000.0
    df["log_rt"] = np.log(df["rt_sec"].clip(lower=1e-3))
    pid_uni = {pid: i for i, pid in enumerate(sorted(df["participant_id"].unique()))}
    df["pid_idx"] = df["participant_id"].map(pid_uni)
    return df


def prepare_prp(cov: pd.DataFrame) -> pd.DataFrame:
    trials, _ = load_prp_trials(
        use_cache=True,
        rt_min=200,
        rt_max=4000,
        require_t1_correct=False,
        require_t2_correct_for_rt=False,
        enforce_short_long_only=False,
    )
    trials["participant_id"] = trials["participant_id"].astype(str)

    soa_col = "soa"
    trials["soa_scaled"] = pd.to_numeric(trials[soa_col], errors="coerce") / 1000.0

    df = trials.merge(cov, on="participant_id", how="inner").dropna(
        subset=["participant_id", "t2_rt", "soa_scaled", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    )
    df["t2_rt_sec"] = df["t2_rt"] / 1000.0
    df["log_t2_rt"] = np.log(df["t2_rt_sec"].clip(lower=1e-3))
    pid_uni = {pid: i for i, pid in enumerate(sorted(df["participant_id"].unique()))}
    df["pid_idx"] = df["participant_id"].map(pid_uni)
    return df


def rope_summary_from_trace(idata: az.InferenceData, model: str, term_map: Dict[str, str], ropes: List[float]) -> pd.DataFrame:
    rows = []
    post = idata.posterior

    def _extract_values(key: str) -> np.ndarray:
        base = key.strip()
        idx = None
        if "[" in key and key.endswith("]"):
            base, idx_str = key[:-1].split("[", 1)
            idx = int(idx_str.strip())
        data = post[base]
        if idx is not None:
            extra_dims = [d for d in data.dims if d not in ("chain", "draw")]
            if not extra_dims:
                raise ValueError(f"No extra dimension to index for {base}")
            data = data.isel({extra_dims[-1]: idx})
        return data.values.ravel()

    for var, label in term_map.items():
        s = _extract_values(var)
        for r in ropes:
            prob_in = (np.abs(s) < r).mean()
            pdirection = max((s > 0).mean(), (s < 0).mean())
            rows.append(
                {
                    "model": model,
                    "term": label,
                    "rope_width": r,
                    "post_mean": float(s.mean()),
                    "hdi_3%": float(np.percentile(s, 3)),
                    "hdi_97%": float(np.percentile(s, 97)),
                    "P(|beta|<rope)": float(prob_in),
                    "PD": float(pdirection),
                }
            )
    return pd.DataFrame(rows)


def fit_stroop_no_ucla(df: pd.DataFrame, draws=1000, tune=1000) -> az.InferenceData:
    X = df[["condition_code", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].to_numpy()
    pid = df["pid_idx"].to_numpy().astype(int)
    n_pid = df["pid_idx"].nunique()
    y = df["log_rt"].to_numpy()
    with pm.Model() as m:
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a = pm.Normal("a", 0.0, sigma_a, shape=n_pid)
        beta = pm.Normal("beta", 0.0, 1.0, shape=X.shape[1])
        alpha = pm.Normal("alpha", 0.0, 1.0)
        mu = alpha + a[pid] + pm.math.dot(X, beta)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("obs", mu, sigma, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )
    return idata


def fit_prp_no_ucla(df: pd.DataFrame, draws=1000, tune=1000) -> az.InferenceData:
    X = df[["soa_scaled", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].to_numpy()
    pid = df["pid_idx"].to_numpy().astype(int)
    n_pid = df["pid_idx"].nunique()
    y = df["log_t2_rt"].to_numpy()
    with pm.Model() as m:
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a = pm.Normal("a", 0.0, sigma_a, shape=n_pid)
        beta = pm.Normal("beta", 0.0, 1.0, shape=X.shape[1])
        alpha = pm.Normal("alpha", 0.0, 1.0)
        mu = alpha + a[pid] + pm.math.dot(X, beta)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("obs", mu, sigma, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )
    return idata


def fit_stroop_with_ucla(df: pd.DataFrame, draws=1000, tune=1000) -> az.InferenceData:
    X = df[["condition_code", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].to_numpy()
    x_int = (df["z_ucla"] * df["condition_code"]).to_numpy()
    pid = df["pid_idx"].to_numpy().astype(int)
    n_pid = df["pid_idx"].nunique()
    y = df["log_rt"].to_numpy()
    with pm.Model() as m:
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a = pm.Normal("a", 0.0, sigma_a, shape=n_pid)
        beta = pm.Normal("beta", 0.0, 1.0, shape=X.shape[1])
        beta_int = pm.Normal("beta_int", 0.0, 1.0)
        alpha = pm.Normal("alpha", 0.0, 1.0)
        mu = alpha + a[pid] + pm.math.dot(X, beta) + beta_int * x_int
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("obs", mu, sigma, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )
    return idata


def fit_prp_with_ucla(df: pd.DataFrame, draws=1000, tune=1000) -> az.InferenceData:
    X = df[["soa_scaled", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].to_numpy()
    x_int = (df["z_ucla"] * df["soa_scaled"]).to_numpy()
    pid = df["pid_idx"].to_numpy().astype(int)
    n_pid = df["pid_idx"].nunique()
    y = df["log_t2_rt"].to_numpy()
    with pm.Model() as m:
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a = pm.Normal("a", 0.0, sigma_a, shape=n_pid)
        beta = pm.Normal("beta", 0.0, 1.0, shape=X.shape[1])
        beta_int = pm.Normal("beta_int", 0.0, 1.0)
        alpha = pm.Normal("alpha", 0.0, 1.0)
        mu = alpha + a[pid] + pm.math.dot(X, beta) + beta_int * x_int
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("obs", mu, sigma, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )
    return idata

def main():
    # 1) ROPE from existing traces (with UCLA)
    ropes = [0.01, 0.02, 0.05]
    rope_rows = []
    try:
        id_s = az.from_netcdf(OUT / "trial_bayes_stroop_rt_trace.nc")
        rope_rows.append(
            rope_summary_from_trace(
                id_s,
                model="stroop_logrt",
                term_map={"beta[1]": "ucla", "beta_int": "ucla:cond"},
                ropes=ropes,
            )
        )
    except Exception:
        pass
    try:
        id_p = az.from_netcdf(OUT / "trial_bayes_prp_rt_trace.nc")
        rope_rows.append(
            rope_summary_from_trace(
                id_p,
                model="prp_logt2rt",
                term_map={"beta[1]": "ucla", "beta_int": "ucla:soa"},
                ropes=ropes,
            )
        )
    except Exception:
        pass
    if rope_rows:
        rope_df = pd.concat(rope_rows, ignore_index=True)
        rope_df.to_csv(OUT / "rope_summary_glmm.csv", index=False)

    # 2) LOO compare (refit models without UCLA)
    cov = load_covariates()
    stroop = prepare_stroop(cov)
    prp = prepare_prp(cov)

    # Fit WITH/WO UCLA models (ensure log_likelihood present)
    id_s_with = fit_stroop_with_ucla(stroop)
    id_s_wo = fit_stroop_no_ucla(stroop)
    id_p_with = fit_prp_with_ucla(prp)
    id_p_wo = fit_prp_no_ucla(prp)

    # LOO and stacking weights
    comp_s = az.compare({"with_ucla": id_s_with, "without_ucla": id_s_wo}, method="BB-pseudo-BMA", ic="loo")
    comp_p = az.compare({"with_ucla": id_p_with, "without_ucla": id_p_wo}, method="BB-pseudo-BMA", ic="loo")
    comp_s.to_csv(OUT / "loo_compare_glmm_stroop.csv")
    comp_p.to_csv(OUT / "loo_compare_glmm_prp.csv")
    print("Saved ROPE and LOO summaries to analysis_outputs/")


if __name__ == "__main__":
    main()
