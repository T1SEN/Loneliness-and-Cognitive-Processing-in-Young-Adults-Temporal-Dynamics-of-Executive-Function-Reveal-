"""
Joint Bayesian Latent-Factor Model (EF hierarchy)
=================================================
Implements a mid-level PRP construct that depends on two lower-level
executive functions:

  - Inhibition  ← measured mainly by Stroop interference
  - Shifting    ← measured mainly by WCST perseveration
  - PRP bottleneck depends on both Inhibition and Shifting

Identification:
  - Latent factors have unit variance (standard normal priors around a
    regression mean), and the primary loadings are fixed to 1.0:
      stroop_z ~ Normal(alpha_s + 1*Inhib, sigma_s)
      wcst_z   ~ Normal(alpha_w + 1*Shift, sigma_w)
      prp_z    ~ Normal(alpha_p + a*Inhib + b*Shift, sigma_p)

  - Inhib and Shift means are regressed on z-scored predictors
    (UCLA loneliness + DASS subscales). Factors are independent in this
    first implementation for stability; the correlation structure can be
    added via LKJ in a later revision if desired.

Outputs (results/analysis_outputs/):
  - joint_model_trace.nc               (PyMC InferenceData)
  - joint_model_summary.csv            (parameter summary)
  - joint_factor_scores.csv            (participant-level latent means)
  - joint_model_readme.txt             (model spec + notes)

Usage:
  python analysis/joint_ef_model.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu, sd = x.mean(skipna=True), x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0 or x.dropna().empty:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def load_master() -> pd.DataFrame:
    """Load master dataset built by run_analysis.py or reconstruct on the fly."""
    master_path = OUT / "master_dataset.csv"
    if master_path.exists():
        df = pd.read_csv(master_path)
        # Backward-compatibility: align column names expected downstream
        rename_map = {
            "pe_rate": "perseverative_error_rate",
            "stroop_effect": "stroop_interference",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        return df

    # Fallback: reconstruct minimal master from raw results
    participants = pd.read_csv(RES / "1_participants_info.csv")
    surveys = pd.read_csv(RES / "2_surveys_results.csv")
    cognitive = pd.read_csv(RES / "3_cognitive_tests_summary.csv")

    # Normalize column names
    surveys_l = surveys.copy(); surveys_l.columns = surveys_l.columns.str.lower()
    parts_l = participants.copy(); parts_l.columns = parts_l.columns.str.lower()
    parts_l = parts_l.rename(columns={"participantid": "participant_id"})

    ucla = (
        surveys_l[surveys_l["surveyname"].str.lower() == "ucla"]
        [["participantid", "score"]]
        .rename(columns={"participantid": "participant_id", "score": "ucla_total"})
    )
    dass = (
        surveys_l[surveys_l["surveyname"].str.lower() == "dass"]
        [["participantid", "score_d", "score_a", "score_s"]]
        .rename(columns={
            "participantid": "participant_id",
            "score_d": "dass_depression",
            "score_a": "dass_anxiety",
            "score_s": "dass_stress",
        })
    )

    # Summary metrics from cognitive summary
    cog = cognitive.copy(); cog.columns = cog.columns.str.lower()
    stroop = cog[cog["test_name"] == "stroop"][
        ["participant_id", "stroop_effect"]
    ]
    wcst = cog[cog["test_name"] == "wcst"][
        ["participant_id", "perseverative_error_count", "total_trial_count"]
    ]
    wcst = wcst.assign(
        perseverative_error_rate=lambda d: (d["perseverative_error_count"] / d["total_trial_count"]) * 100.0
    )[["participant_id", "perseverative_error_rate"]]
    prp = cog[cog["test_name"] == "prp"][
        ["participant_id", "rt2_soa_50", "rt2_soa_1200"]
    ].assign(prp_bottleneck=lambda d: d["rt2_soa_50"] - d["rt2_soa_1200"]).dropna()
    prp = prp[["participant_id", "prp_bottleneck"]]

    master = (
        parts_l[["participant_id", "age", "gender"]]
        .merge(ucla, on="participant_id", how="left")
        .merge(dass, on="participant_id", how="left")
        .merge(stroop, on="participant_id", how="left")
        .merge(wcst, on="participant_id", how="left")
        .merge(prp, on="participant_id", how="left")
    )
    return master


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep complete cases, z-score outcomes and predictors."""
    cols_needed = [
        "participant_id",
        "ucla_total",
        "dass_depression",
        "dass_anxiety",
        "dass_stress",
        "stroop_interference",
        "perseverative_error_rate",
        "prp_bottleneck",
    ]
    sub = df[cols_needed].dropna().copy()
    if len(sub) < 30:
        raise RuntimeError("Not enough complete rows for joint model (need ≥30).")

    # Z-scores of observed indicators and predictors
    sub["stroop_z"] = _z(sub["stroop_interference"])  # larger = worse interference
    sub["wcst_z"] = _z(sub["perseverative_error_rate"])  # larger = more perseveration
    sub["prp_z"] = _z(sub["prp_bottleneck"])  # larger = bigger bottleneck
    sub["z_ucla"] = _z(sub["ucla_total"])  # loneliness
    sub["z_dep"] = _z(sub["dass_depression"])  # DASS controls
    sub["z_anx"] = _z(sub["dass_anxiety"]) 
    sub["z_str"] = _z(sub["dass_stress"]) 

    complete = sub.dropna(subset=["stroop_z", "wcst_z", "prp_z", "z_ucla", "z_dep", "z_anx", "z_str"]).copy()
    # Keep design matrix for predictors
    X = complete[["z_ucla", "z_dep", "z_anx", "z_str"]].copy()
    return complete, X


def build_and_fit(complete: pd.DataFrame, X: pd.DataFrame, draws=1500, tune=1500, chains=4):
    """Build the joint EF model and run MCMC."""
    y_s = complete["stroop_z"].values
    y_w = complete["wcst_z"].values
    y_p = complete["prp_z"].values
    X_mat = X.values  # N x 4
    N = len(complete)
    K = X_mat.shape[1]

    with pm.Model() as mdl:
        # Regressions on latent means (Inhib, Shift); unit variance for identification
        beta0_inhib = pm.Normal("beta0_inhib", 0.0, 1.0)
        beta_inhib = pm.Normal("beta_inhib", 0.0, 1.0, shape=K)  # [UCLA, DEP, ANX, STR]

        beta0_shift = pm.Normal("beta0_shift", 0.0, 1.0)
        beta_shift = pm.Normal("beta_shift", 0.0, 1.0, shape=K)

        mu_inhib = beta0_inhib + pm.math.dot(X_mat, beta_inhib)
        mu_shift = beta0_shift + pm.math.dot(X_mat, beta_shift)

        # Latent factors (independent standard normals around means)
        Inhib = pm.Normal("Inhib", mu=mu_inhib, sigma=1.0, shape=N)
        Shift = pm.Normal("Shift", mu=mu_shift, sigma=1.0, shape=N)

        # Measurement model
        alpha_s = pm.Normal("alpha_stroop", 0.0, 0.5)
        alpha_w = pm.Normal("alpha_wcst", 0.0, 0.5)
        alpha_p = pm.Normal("alpha_prp", 0.0, 0.5)

        # Fixed primary loadings to 1.0 for identification
        sigma_s = pm.HalfNormal("sigma_stroop", 1.0)
        sigma_w = pm.HalfNormal("sigma_wcst", 1.0)
        sigma_p = pm.HalfNormal("sigma_prp", 1.0)

        # Cross-loadings for PRP on both factors
        a = pm.Normal("loading_prp_inhib", 0.0, 1.0)
        b = pm.Normal("loading_prp_shift", 0.0, 1.0)

        # Likelihoods
        pm.Normal("stroop_obs", mu=alpha_s + Inhib, sigma=sigma_s, observed=y_s)
        pm.Normal("wcst_obs",   mu=alpha_w + Shift,  sigma=sigma_w, observed=y_w)
        pm.Normal("prp_obs",    mu=alpha_p + a*Inhib + b*Shift, sigma=sigma_p, observed=y_p)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.99,
            return_inferencedata=True,
            random_seed=42,
        )

    return idata


def save_outputs(idata: az.InferenceData, complete: pd.DataFrame) -> None:
    # Trace
    trace_path = OUT / "joint_model_trace.nc"
    idata.to_netcdf(trace_path)

    # Summary for key parameters
    vars_of_interest = [
        "beta0_inhib", "beta_inhib",
        "beta0_shift", "beta_shift",
        "alpha_stroop", "alpha_wcst", "alpha_prp",
        "sigma_stroop", "sigma_wcst", "sigma_prp",
        "loading_prp_inhib", "loading_prp_shift",
    ]
    summ = az.summary(idata, var_names=vars_of_interest, round_to=3)
    summ.to_csv(OUT / "joint_model_summary.csv")

    # Participant-level latent means (posterior means)
    inhib = idata.posterior["Inhib"].values.mean(axis=(0, 1))
    shift = idata.posterior["Shift"].values.mean(axis=(0, 1))
    scores = pd.DataFrame({
        "participant_id": complete["participant_id"].values,
        "inhibition_score": inhib,
        "shifting_score": shift,
    })
    scores.to_csv(OUT / "joint_factor_scores.csv", index=False)

    # Readme/spec
    (OUT / "joint_model_readme.txt").write_text(
        "Joint EF model: stroop→Inhib(1.0), wcst→Shift(1.0), prp→a*Inhib + b*Shift\n"
        "Latent means ~ UCLA + DASS; unit latent SD; factors independent in v1.\n"
        f"N(participants)={len(scores)}\n",
        encoding="utf-8",
    )


def main():
    print("\n" + "="*70)
    print("JOINT EF MODEL (Latent Inhibition + Shifting → PRP)")
    print("="*70)

    df = load_master()
    complete, X = prepare_data(df)
    print(f"Participants with complete data: N={len(complete)}")

    idata = build_and_fit(complete, X)
    save_outputs(idata, complete)

    # Console brief
    a = idata.posterior["loading_prp_inhib"].values.flatten()
    b = idata.posterior["loading_prp_shift"].values.flatten()
    print("\nKey loadings (posterior means, 95% HDI):")
    print(f"  PRP ← Inhib:  mean={a.mean():.3f}, 95% HDI=[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}], P(>0)={(a>0).mean():.2%}")
    print(f"  PRP ← Shift:  mean={b.mean():.3f}, 95% HDI=[{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}], P(>0)={(b>0).mean():.2%}")

    # Report UCLA effects on latent means
    beta_inhib = idata.posterior["beta_inhib"].values[..., 0].ravel()  # UCLA is first col
    beta_shift = idata.posterior["beta_shift"].values[..., 0].ravel()
    print("\nUCLA effects on latent factors (coefficient on z_ucla):")
    print(f"  Inhibition: mean={beta_inhib.mean():.3f}, 95% HDI=[{np.percentile(beta_inhib,2.5):.3f}, {np.percentile(beta_inhib,97.5):.3f}], P(>0)={(beta_inhib>0).mean():.2%}")
    print(f"  Shifting:   mean={beta_shift.mean():.3f}, 95% HDI=[{np.percentile(beta_shift,2.5):.3f}, {np.percentile(beta_shift,97.5):.3f}], P(>0)={(beta_shift>0).mean():.2%}")

    print("\nSaved outputs to results/analysis_outputs")


if __name__ == "__main__":
    main()
