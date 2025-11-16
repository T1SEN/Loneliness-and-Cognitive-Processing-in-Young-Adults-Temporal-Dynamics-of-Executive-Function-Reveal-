"""
Trial-level Bayesian modeling of EF tasks
----------------------------------------
Builds hierarchical Bayesian models on trial-level data to probe how
loneliness (UCLA), controlling for DASS (dep/anx/stress), relates to
RT dynamics in Stroop and PRP tasks.

Models (PyMC, NUTS):
- Stroop RT (log-seconds) ~ condition_code + z_ucla + DASS + z_ucla:condition
  with participant random intercepts.
- PRP T2 RT (log-seconds) ~ soa_scaled + z_ucla + DASS + z_ucla:soa
  with participant random intercepts.

Outputs are saved to results/analysis_outputs/:
- trial_bayes_stroop_rt_summary.csv
- trial_bayes_prp_rt_summary.csv
- trial_bayes_stroop_rt_trace.nc
- trial_bayes_prp_rt_trace.nc
- trial_bayes_ppc_*.png

Notes:
- Keep models intentionally simple for stability/speed. Random slopes can be
  added later if sampling speed allows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = RESULTS_DIR / "analysis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().std() in (0, None) or s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / s.std()


def load_covariates() -> pd.DataFrame:
    surv = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
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
    df = ucla.merge(dass, on="participant_id", how="inner").copy()
    df["z_ucla"] = _z(df["ucla_total"])
    df["z_dass_dep"] = _z(df["dass_dep"])
    df["z_dass_anx"] = _z(df["dass_anx"])
    df["z_dass_stress"] = _z(df["dass_stress"])
    return df


def prepare_stroop_trials(cov: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
    pid = df.get("participant_id", df.get("participantId"))
    df["participant_id"] = pid
    if "rt_ms" not in df.columns:
        raise RuntimeError("Stroop trials missing rt_ms column")
    if "timeout" in df.columns:
        df = df[df["timeout"] == False]
    df = df[df["rt_ms"].notna()].copy()
    df = df[(df["rt_ms"] >= 200) & (df["rt_ms"] <= 3000)]
    # condition coding
    if "type" in df.columns:
        cond = df["type"].astype(str).str.lower()
    elif "cond" in df.columns:
        cond = df["cond"].astype(str).str.lower()
    else:
        raise RuntimeError("Stroop trials missing condition column ('type' or 'cond')")
    code_map = {"congruent": 0.0, "neutral": 0.5, "incongruent": 1.0}
    df["condition"] = cond
    df["condition_code"] = cond.map(code_map)
    df = df[df["condition_code"].notna()]
    # response correctness if present
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(int)
    df = df.merge(cov, on="participant_id", how="inner").dropna(
        subset=["participant_id", "rt_ms", "condition_code", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    )
    df["rt_sec"] = df["rt_ms"] / 1000.0
    df["log_rt"] = np.log(df["rt_sec"].clip(lower=1e-3))
    # participant index
    pid_uni = {pid: i for i, pid in enumerate(sorted(df["participant_id"].unique()))}
    df["pid_idx"] = df["participant_id"].map(pid_uni)
    return df


def prepare_prp_trials(cov: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")
    if "t2_rt_ms" not in df.columns:
        raise RuntimeError("PRP trials missing t2_rt_ms column")
    df = df[(df["t2_timeout"] == False) & df["t2_rt_ms"].notna()].copy()
    df = df[(df["t2_rt_ms"] >= 200) & (df["t2_rt_ms"] <= 4000)]
    # SOA scaling to seconds
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


# ---------------------------------------------------------------------------
# Bayesian models
# ---------------------------------------------------------------------------
@dataclass
class FitConfig:
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    random_seed: int = 42


def fit_stroop_rt_model(df: pd.DataFrame, cfg: FitConfig) -> Tuple[az.InferenceData, pd.DataFrame]:
    X = df[["condition_code", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].to_numpy()
    # interaction term: z_ucla * condition_code
    x_int = (df["z_ucla"] * df["condition_code"]).to_numpy()
    pid_idx = df["pid_idx"].to_numpy().astype(int)
    n_pid = df["pid_idx"].nunique()
    y = df["log_rt"].to_numpy()

    with pm.Model() as model:
        # Random intercepts by participant
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a = pm.Normal("a", mu=0.0, sigma=sigma_a, shape=n_pid)

        # Fixed effects
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
        beta_int = pm.Normal("beta_int", mu=0.0, sigma=1.0)
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        mu = alpha + a[pid_idx] + pm.math.dot(X, beta) + beta_int * x_int
        sigma = pm.HalfNormal("sigma", 1.0)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    summary = az.summary(
        idata, var_names=["alpha", "beta", "beta_int", "sigma", "sigma_a"], round_to=3
    )
    # Map coefficients to labels
    coef_labels = ["cond", "ucla", "dass_dep", "dass_anx", "dass_stress"]
    rows = []
    for i, lab in enumerate(coef_labels):
        s = summary.loc[f"beta[{i}]"].to_dict()
        s.update({"term": lab, "model": "stroop_logrt"})
        rows.append(s)
    s_int = summary.loc["beta_int"].to_dict()
    s_int.update({"term": "ucla:cond", "model": "stroop_logrt"})
    rows.append(s_int)
    out_df = pd.DataFrame(rows)
    return idata, out_df


def fit_prp_t2_rt_model(df: pd.DataFrame, cfg: FitConfig) -> Tuple[az.InferenceData, pd.DataFrame]:
    X = df[["soa_scaled", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].to_numpy()
    x_int = (df["z_ucla"] * df["soa_scaled"]).to_numpy()
    pid_idx = df["pid_idx"].to_numpy().astype(int)
    n_pid = df["pid_idx"].nunique()
    y = df["log_t2_rt"].to_numpy()

    with pm.Model() as model:
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a = pm.Normal("a", mu=0.0, sigma=sigma_a, shape=n_pid)

        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
        beta_int = pm.Normal("beta_int", mu=0.0, sigma=1.0)
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        mu = alpha + a[pid_idx] + pm.math.dot(X, beta) + beta_int * x_int
        sigma = pm.HalfNormal("sigma", 1.0)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    summary = az.summary(
        idata, var_names=["alpha", "beta", "beta_int", "sigma", "sigma_a"], round_to=3
    )
    coef_labels = ["soa", "ucla", "dass_dep", "dass_anx", "dass_stress"]
    rows = []
    for i, lab in enumerate(coef_labels):
        s = summary.loc[f"beta[{i}]"].to_dict()
        s.update({"term": lab, "model": "prp_logt2rt"})
        rows.append(s)
    s_int = summary.loc["beta_int"].to_dict()
    s_int.update({"term": "ucla:soa", "model": "prp_logt2rt"})
    rows.append(s_int)
    out_df = pd.DataFrame(rows)
    return idata, out_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    print("Trial-level Bayesian modeling: loading covariates…")
    cov = load_covariates()

    print("Preparing Stroop trials…")
    stroop = prepare_stroop_trials(cov)
    print(f"  Stroop: {len(stroop)} trials | {stroop['participant_id'].nunique()} participants")

    print("Preparing PRP trials…")
    prp = prepare_prp_trials(cov)
    print(f"  PRP: {len(prp)} trials | {prp['participant_id'].nunique()} participants")

    cfg = FitConfig(draws=1000, tune=1000, chains=4, target_accept=0.9)

    # Stroop RT model
    print("\nFitting Stroop log-RT model (random intercepts)…")
    idata_s, sum_s = fit_stroop_rt_model(stroop, cfg)
    sum_s.to_csv(OUT_DIR / "trial_bayes_stroop_rt_summary.csv", index=False)
    idata_s.to_netcdf(OUT_DIR / "trial_bayes_stroop_rt_trace.nc")
    try:
        az.plot_ppc(az.from_pymc3(idata=idata_s))
    except Exception:
        pass

    # PRP T2 RT model
    print("\nFitting PRP log-T2-RT model (random intercepts)…")
    idata_p, sum_p = fit_prp_t2_rt_model(prp, cfg)
    sum_p.to_csv(OUT_DIR / "trial_bayes_prp_rt_summary.csv", index=False)
    idata_p.to_netcdf(OUT_DIR / "trial_bayes_prp_rt_trace.nc")

    # Merge key terms for quick reading
    key = pd.concat([sum_s, sum_p], ignore_index=True)
    key = key[[c for c in key.columns if c in ("term", "mean", "hdi_3%", "hdi_97%", "sd", "model") or c.startswith("hdi_")]]
    key.to_csv(OUT_DIR / "trial_bayes_key_terms.csv", index=False)
    print("Saved posterior summaries to analysis_outputs/")

    return {
        "stroop_trials": len(stroop),
        "stroop_participants": stroop["participant_id"].nunique(),
        "prp_trials": len(prp),
        "prp_participants": prp["participant_id"].nunique(),
    }


if __name__ == "__main__":
    main()

