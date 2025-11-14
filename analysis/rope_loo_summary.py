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
    surv = pd.read_csv(RES / "2_surveys_results.csv")
    surv.columns = surv.columns.str.lower()
    ucla = (
        surv[surv["surveyname"].str.lower() == "ucla"]
        [["participantid", "score"]]
        .rename(columns={"participantid": "participant_id", "score": "ucla_total"})
    )
    dass = (
        surv[surv["surveyname"].str.lower() == "dass"]
        [["participantid", "score_d", "score_a", "score_s"]]
        .rename(
            columns={
                "participantid": "participant_id",
                "score_d": "dass_dep",
                "score_a": "dass_anx",
                "score_s": "dass_stress",
            }
        )
    )
    cov = ucla.merge(dass, on="participant_id", how="inner")
    cov["z_ucla"] = _z(cov["ucla_total"]) 
    cov["z_dass_dep"] = _z(cov["dass_dep"]) 
    cov["z_dass_anx"] = _z(cov["dass_anx"]) 
    cov["z_dass_stress"] = _z(cov["dass_stress"]) 
    return cov


def prepare_stroop(cov: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RES / "4c_stroop_trials.csv")
    pid = df.get("participant_id", df.get("participantId"))
    df["participant_id"] = pid
    if "timeout" in df.columns:
        df = df[df["timeout"] == False]
    df = df[df["rt_ms"].notna() & (df["rt_ms"] >= 200) & (df["rt_ms"] <= 3000)].copy()
    if "type" in df.columns:
        cond = df["type"].astype(str).str.lower()
    elif "cond" in df.columns:
        cond = df["cond"].astype(str).str.lower()
    else:
        raise RuntimeError("Stroop trials missing condition column")
    code_map = {"congruent": 0.0, "neutral": 0.5, "incongruent": 1.0}
    df["condition_code"] = cond.map(code_map)
    df = df[df["condition_code"].notna()]
    df = df.merge(cov, on="participant_id", how="inner").dropna(
        subset=["participant_id", "rt_ms", "condition_code", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    )
    df["rt_sec"] = df["rt_ms"] / 1000.0
    df["log_rt"] = np.log(df["rt_sec"].clip(lower=1e-3))
    pid_uni = {pid: i for i, pid in enumerate(sorted(df["participant_id"].unique()))}
    df["pid_idx"] = df["participant_id"].map(pid_uni)
    return df


def prepare_prp(cov: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RES / "4a_prp_trials.csv")
    df = df[(df["t2_timeout"] == False) & df["t2_rt_ms"].notna()].copy()
    df = df[(df["t2_rt_ms"] >= 200) & (df["t2_rt_ms"] <= 4000)].copy()
    soa_col = "soa_nominal_ms" if "soa_nominal_ms" in df.columns else "soa"
    df["soa_scaled"] = pd.to_numeric(df[soa_col], errors="coerce") / 1000.0
    df = df.merge(cov, on="participant_id", how="inner").dropna(
        subset=["participant_id", "t2_rt_ms", "soa_scaled", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    )
    df["t2_rt_sec"] = df["t2_rt_ms"] / 1000.0
    df["log_t2_rt"] = np.log(df["t2_rt_sec"].clip(lower=1e-3))
    pid_uni = {pid: i for i, pid in enumerate(sorted(df["participant_id"].unique()))}
    df["pid_idx"] = df["participant_id"].map(pid_uni)
    return df


def rope_summary_from_trace(idata: az.InferenceData, model: str, term_map: Dict[str, str], ropes: List[float]) -> pd.DataFrame:
    rows = []
    post = idata.posterior
    for var, label in term_map.items():
        s = post[var].values.ravel()
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
