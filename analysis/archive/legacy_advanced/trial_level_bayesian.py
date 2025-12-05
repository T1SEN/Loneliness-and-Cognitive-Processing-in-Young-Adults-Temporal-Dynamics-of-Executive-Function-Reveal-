"""
Trial-level Bayesian modeling of EF tasks
-----------------------------------------
Hierarchical Bayesian models (PyMC) on Stroop and PRP trials with UCLA/DASS covariates.

Models:
- Stroop RT (log-seconds) ~ condition + z_ucla + DASS + z_ucla:condition + (1|participant)
- PRP T2 RT (log-seconds) ~ soa_scaled + z_ucla + DASS + z_ucla:soa + (1|participant)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials, load_stroop_trials

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
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
    if "ucla_total" not in master.columns and "ucla_score" in master.columns:
        master["ucla_total"] = master["ucla_score"]

    cov = master[
        [
            "participant_id",
            "ucla_total",
            "dass_depression",
            "dass_anxiety",
            "dass_stress",
            "gender_normalized",
        ]
    ].copy()
    cov = cov.rename(columns={"gender_normalized": "gender"})
    cov["gender"] = cov["gender"].fillna("").astype(str).str.strip().str.lower()
    cov["z_ucla"] = _z(cov["ucla_total"])
    cov["z_dass_dep"] = _z(cov["dass_depression"])
    cov["z_dass_anx"] = _z(cov["dass_anxiety"])
    cov["z_dass_stress"] = _z(cov["dass_stress"])
    return cov


def prepare_stroop_trials(cov: pd.DataFrame) -> pd.DataFrame:
    trials, summary = load_stroop_trials(use_cache=True)

    rt_col = "rt" if "rt" in trials.columns else "rt_ms" if "rt_ms" in trials.columns else None
    if not rt_col:
        raise KeyError("Stroop trials missing rt/rt_ms")
    if rt_col != "rt":
        trials["rt"] = trials[rt_col]

    if "type" not in trials.columns:
        for cand in ["condition", "cond"]:
            if cand in trials.columns:
                trials = trials.rename(columns={cand: "type"})
                break
    if "type" not in trials.columns:
        raise KeyError("Stroop trials missing condition/type")

    if "timeout" not in trials.columns:
        trials["timeout"] = False

    df = trials[
        (trials["timeout"] == False)
        & (trials["rt"].between(200, 5000))
        & (trials["type"].isin(["congruent", "incongruent"]))
    ].copy()
    df["condition_code"] = df["type"].map({"congruent": 0, "incongruent": 1})
    df["log_rt"] = np.log(df["rt"] / 1000.0)  # seconds

    df = df.merge(
        cov[
            ["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "gender"]
        ],
        on="participant_id",
        how="inner",
    )
    df = df.dropna(subset=["log_rt", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"])
    return df


def prepare_prp_trials(cov: pd.DataFrame) -> pd.DataFrame:
    trials, summary = load_prp_trials(
        use_cache=True,
        rt_min=200,
        rt_max=5000,
        require_t1_correct=False,      # keep T1 errors
        enforce_short_long_only=False, # allow all SOA bins
    )

    if "soa" not in trials.columns:
        raise KeyError("PRP trials missing soa")

    df = trials[trials["t2_rt"].between(200, 5000)].copy()
    df["log_t2_rt"] = np.log(df["t2_rt"] / 1000.0)  # seconds
    df["soa_scaled"] = _z(df["soa"])

    df = df.merge(
        cov[
            ["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "gender"]
        ],
        on="participant_id",
        how="inner",
    )
    df = df.dropna(subset=["log_t2_rt", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"])
    return df


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class FitConfig:
    draws: int = 3000
    tune: int = 3000
    chains: int = 4
    target_accept: float = 0.95
    random_seed: int = 42


def fit_stroop_rt_model(df: pd.DataFrame, cfg: FitConfig) -> Tuple[az.InferenceData, pd.DataFrame]:
    with pm.Model() as model:
        a_part = pm.Normal("alpha_part", 0, 1, shape=df["participant_id"].nunique())
        sigma_a = pm.HalfNormal("sigma_a", 1.0)

        condition = df["condition_code"].values
        z_ucla = df["z_ucla"].values
        dass_dep = df["z_dass_dep"].values
        dass_anx = df["z_dass_anx"].values
        dass_str = df["z_dass_stress"].values
        pid_idx, uniques = pd.factorize(df["participant_id"])

        beta = pm.Normal("beta", 0, 1, shape=4)  # condition, ucla, dep, anx; stress added separately
        beta_int = pm.Normal("beta_int", 0, 1)

        mu = (
            beta[0] * condition
            + beta[1] * z_ucla
            + beta[2] * dass_dep
            + beta[3] * dass_anx
            + beta_int * condition * z_ucla
            + a_part[pid_idx] * sigma_a
            + dass_str  # include stress additively
        )

        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("log_rt_obs", mu=mu, sigma=sigma, observed=df["log_rt"].values)

        idata = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            init="jitter+adapt_diag_grad",
            random_seed=cfg.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    summary = az.summary(idata, var_names=["beta", "beta_int", "sigma", "sigma_a"], round_to=3)
    coef_labels = ["condition", "ucla", "dass_dep", "dass_anx"]
    rows = []
    for i, lab in enumerate(coef_labels):
        s = summary.loc[f"beta[{i}]"].to_dict()
        s.update({"term": lab, "model": "stroop_logrt"})
        rows.append(s)
    s_int = summary.loc["beta_int"].to_dict()
    s_int.update({"term": "ucla:condition", "model": "stroop_logrt"})
    rows.append(s_int)
    out_df = pd.DataFrame(rows)
    return idata, out_df


def fit_prp_t2_rt_model(df: pd.DataFrame, cfg: FitConfig) -> Tuple[az.InferenceData, pd.DataFrame]:
    with pm.Model() as model:
        a_part = pm.Normal("alpha_part", 0, 1, shape=df["participant_id"].nunique())
        sigma_a = pm.HalfNormal("sigma_a", 1.0)

        soa = df["soa_scaled"].values
        z_ucla = df["z_ucla"].values
        dass_dep = df["z_dass_dep"].values
        dass_anx = df["z_dass_anx"].values
        dass_str = df["z_dass_stress"].values
        pid_idx, uniques = pd.factorize(df["participant_id"])

        beta = pm.Normal("beta", 0, 1, shape=5)  # soa, ucla, dep, anx, stress
        beta_int = pm.Normal("beta_int", 0, 1)

        mu = (
            beta[0] * soa
            + beta[1] * z_ucla
            + beta[2] * dass_dep
            + beta[3] * dass_anx
            + beta[4] * dass_str
            + beta_int * soa * z_ucla
            + a_part[pid_idx] * sigma_a
        )

        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("log_t2_rt_obs", mu=mu, sigma=sigma, observed=df["log_t2_rt"].values)

        idata = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    summary = az.summary(idata, var_names=["beta", "beta_int", "sigma", "sigma_a"], round_to=3)
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

    cfg = FitConfig(draws=3000, tune=3000, chains=4, target_accept=0.95)

    print("\nFitting Stroop log-RT model (random intercepts)…")
    idata_s, sum_s = fit_stroop_rt_model(stroop, cfg)
    sum_s.to_csv(OUT_DIR / "trial_bayes_stroop_rt_summary.csv", index=False)
    idata_s.to_netcdf(OUT_DIR / "trial_bayes_stroop_rt_trace.nc")

    print("\nFitting PRP log-T2-RT model (random intercepts)…")
    idata_p, sum_p = fit_prp_t2_rt_model(prp, cfg)
    sum_p.to_csv(OUT_DIR / "trial_bayes_prp_rt_summary.csv", index=False)
    idata_p.to_netcdf(OUT_DIR / "trial_bayes_prp_rt_trace.nc")

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
