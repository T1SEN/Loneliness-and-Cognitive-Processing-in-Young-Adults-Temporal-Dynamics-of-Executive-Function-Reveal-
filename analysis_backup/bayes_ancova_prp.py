"""
Bayesian ANCOVA for PRP bottleneck (extreme groups)
---------------------------------------------------
Matches the frequentist extreme-group ANCOVA by comparing bottom vs top
UCLA quartiles while controlling for DASS subscales, age, and gender.

Model (PyMC):
  prp_bottleneck_ms ~ Normal(mu, sigma)
  mu = alpha + b_group*I(high) + b_dep*z_dass_dep + b_anx*z_dass_anx +
        b_str*z_dass_stress + b_age*age + b_male*I(gender=='male')

Outputs:
- results/analysis_outputs/bayes_ancova_prp_summary.csv
- results/analysis_outputs/bayes_ancova_prp_bf.csv (Savage–Dickey BF across priors)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def make_quartile_groups(df: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = df["ucla_total"].quantile([0.25, 0.75])
    g = np.where(df["ucla_total"] <= q1, "low", np.where(df["ucla_total"] >= q3, "high", None))
    out = df.copy()
    out["group"] = g
    return out.dropna(subset=["group"]).copy()


def sd_bayes_factor_normal0(samples: np.ndarray, prior_sd: float) -> tuple[float, float, float]:
    s = np.asarray(samples).ravel()
    s = s[np.isfinite(s)]
    if len(s) < 10 or prior_sd <= 0:
        return np.nan, np.nan, np.nan
    m = s.mean()
    sd = s.std(ddof=1)
    try:
        kde = gaussian_kde(s)
        post_pdf0 = float(kde.evaluate([0.0])[0])
    except Exception:
        return np.nan, np.nan, np.nan
    prior_pdf0 = 1.0 / np.sqrt(2 * np.pi) / prior_sd
    if post_pdf0 <= 0:
        return m, sd, np.nan
    bf01 = post_pdf0 / prior_pdf0
    return m, sd, bf01


def main():
    df = build_analysis_dataframe()
    df = df.dropna(subset=["ucla_total", "prp_bottleneck"])  # ensure PRP present
    df_q = make_quartile_groups(df)
    # Design matrix
    data = df_q[[
        "participant_id", "prp_bottleneck", "group", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"
    ]].dropna().copy()
    if data["group"].nunique() < 2 or len(data) < 30:
        raise SystemExit("Not enough data for Bayesian ANCOVA (need ≥30 and two groups).")
    data["is_high"] = (data["group"] == "high").astype(int)
    data["is_male"] = (data["gender"] == "male").astype(int)
    y = data["prp_bottleneck"].to_numpy()
    X = data[["is_high", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "is_male"]].to_numpy()

    with pm.Model() as anc:
        alpha = pm.Normal("alpha", 0.0, 200.0)
        beta = pm.Normal("beta", 0.0, 100.0, shape=X.shape[1])  # weakly-informative on ms scale
        mu = alpha + pm.math.dot(X, beta)
        sigma = pm.HalfNormal("sigma", 100.0)
        pm.Normal("obs", mu, sigma, observed=y)
        idata = pm.sample(draws=2000, tune=2000, chains=4, target_accept=0.95, return_inferencedata=True, random_seed=42)

    summ = az.summary(idata, var_names=["alpha", "beta", "sigma"], round_to=3)
    # Label coefficients
    coef_names = ["group_high", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "male"]
    rows = []
    for i, name in enumerate(coef_names):
        s = idata.posterior["beta"].values[..., i].ravel()
        rows.append({
            "term": name,
            "post_mean": float(s.mean()),
            "hdi_3%": float(np.percentile(s, 3)),
            "hdi_97%": float(np.percentile(s, 97)),
            "PD": float(max((s > 0).mean(), (s < 0).mean())),
        })
    pd.DataFrame(rows).to_csv(OUT / "bayes_ancova_prp_summary.csv", index=False)

    # Savage–Dickey BF across plausible priors for group difference (ms)
    g = idata.posterior["beta"].values[..., 0]  # group_high
    priors = [25.0, 50.0, 100.0]
    bf_rows = []
    for psd in priors:
        m, sd, bf01 = sd_bayes_factor_normal0(g, psd)
        bf_rows.append({"prior_sd_ms": psd, "post_mean": m, "post_sd": sd, "BF01": bf01, "BF10": (1.0/bf01) if bf01>0 else np.nan})
    pd.DataFrame(bf_rows).to_csv(OUT / "bayes_ancova_prp_bf.csv", index=False)
    print("Saved Bayesian ANCOVA summary and BF to analysis_outputs/")


if __name__ == "__main__":
    main()
