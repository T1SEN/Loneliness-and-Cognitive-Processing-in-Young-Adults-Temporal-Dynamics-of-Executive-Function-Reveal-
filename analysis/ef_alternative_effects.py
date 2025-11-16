"""
Supplementary EF-focused analyses to highlight gender × loneliness effects.

Analyses included:
1. Gender × loneliness tertile ANOVA/OLS on core EF metrics.
2. WCST learning-curve visualization (early vs late PE rates).
3. Reverse OLS models predicting EF metrics from loneliness (plus controls).
4. Bayesian regression for WCST perseverative errors with gender interaction.

Outputs are written to results/analysis_outputs/ef_alternative/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

try:  # pragma: no cover - optional acceleration
    from pymc.sampling_jax import sample_numpyro_nuts
except Exception:
    sample_numpyro_nuts = None
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis_outputs" / "ef_alternative"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_EXPANDED = RESULTS_DIR / "analysis_outputs" / "master_expanded_metrics.csv"
LEARNING_SLOPES = RESULTS_DIR / "analysis_outputs" / "learning_curves" / "individual_learning_slopes.csv"

EF_METRICS = [
    "wcst_pe_rate",
    "wcst_accuracy",
    "wcst_rt_cv",
    "stroop_interference",
    "prp_bottleneck",
    "prp_t1_slowing",
]


def zscore(series: pd.Series) -> pd.Series:
    if series.std(skipna=True) == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean(skipna=True)) / series.std(skipna=True)


def normalize_gender(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s:
        return None
    if any(tok in s for tok in ("female", "여", "f")):
        return "female"
    if any(tok in s for tok in ("male", "남", "m")):
        return "male"
    return None


def load_master() -> pd.DataFrame:
    base = pd.read_csv(MASTER_EXPANDED).rename(columns={"pe_rate": "wcst_pe_rate"})
    if LEARNING_SLOPES.exists():
        slopes = pd.read_csv(LEARNING_SLOPES)[
            ["participant_id", "gender", "age", "pe_slope", "early_pe_rate", "late_pe_rate"]
        ].rename(columns={"pe_slope": "wcst_pe_slope", "gender": "gender_slopes", "age": "age_slopes"})
        merged = base.merge(slopes, on="participant_id", how="left")
        if "gender" in merged.columns:
            merged["gender"] = merged["gender"].combine_first(merged.get("gender_slopes"))
        else:
            merged["gender"] = merged.get("gender_slopes")
        if "age" in merged.columns:
            merged["age"] = merged["age"].combine_first(merged.get("age_slopes"))
        else:
            merged["age"] = merged.get("age_slopes")
    else:
        merged = base.copy()
    merged["gender_clean"] = merged["gender"].apply(normalize_gender)
    merged = merged.dropna(subset=["gender_clean", "ucla_total"])
    merged["gender_male"] = merged["gender_clean"].map({"male": 1, "female": 0})
    merged["z_ucla"] = zscore(merged["ucla_total"])
    merged["z_age"] = zscore(merged["age"])
    merged["z_dass_dep"] = zscore(merged["dass_depression"])
    merged["z_dass_anx"] = zscore(merged["dass_anxiety"])
    merged["z_dass_stress"] = zscore(merged["dass_stress"])
    merged["loneliness_group"] = pd.qcut(
        merged["ucla_total"],
        q=[0, 1 / 3, 2 / 3, 1],
        labels=["low", "mid", "high"],
    )
    return merged


def run_subgroup_anova(df: pd.DataFrame) -> None:
    anova_records: List[pd.DataFrame] = []
    coef_records: List[pd.DataFrame] = []
    means_rows = []
    for metric in EF_METRICS:
        data = df[["gender_male", "loneliness_group", metric]].dropna()
        if data.empty:
            continue
        model = smf.ols(f"{metric} ~ C(gender_male) * C(loneliness_group)", data=data).fit()
        coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        coef.insert(0, "metric", metric)
        coef_records.append(coef)
        try:
            anova = anova_lm(model, typ=2)
            anova = anova.reset_index().rename(columns={"index": "term"})
            anova["metric"] = metric
            anova_records.append(anova)
        except ValueError:
            pass

        summary = (
            data.assign(gender=data["gender_male"].map({0: "female", 1: "male"}))
            .groupby(["loneliness_group", "gender"])
            [metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["metric"] = metric
        means_rows.append(summary)

    if anova_records:
        anova_df = pd.concat(anova_records, ignore_index=True)
        anova_df.to_csv(OUTPUT_DIR / "ef_subgroup_anova.csv", index=False, encoding="utf-8-sig")
    if coef_records:
        coef_df = pd.concat(coef_records, ignore_index=True)
        coef_df.to_csv(OUTPUT_DIR / "ef_subgroup_coefficients.csv", index=False, encoding="utf-8-sig")
    if means_rows:
        means_df = pd.concat(means_rows, ignore_index=True)
        means_df.to_csv(OUTPUT_DIR / "ef_subgroup_means.csv", index=False, encoding="utf-8-sig")


def plot_wcst_learning_curves(df: pd.DataFrame) -> None:
    slopes = df[["gender_clean", "loneliness_group", "early_pe_rate", "late_pe_rate"]].dropna()
    if slopes.empty:
        return
    focus = slopes[slopes["loneliness_group"].isin(["low", "high"])].copy()
    if focus.empty:
        return
    groups = sorted(focus["gender_clean"].unique())
    stages = ["early_pe_rate", "late_pe_rate"]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"female": "#1f77b4", "male": "#d62728"}
    stage_labels = ["Early (first third)", "Late (last third)"]
    for gender in groups:
        for group_label, linestyle in [("low", "-"), ("high", "--")]:
            subset = focus[(focus["gender_clean"] == gender) & (focus["loneliness_group"] == group_label)]
            if subset.empty:
                continue
            means = [subset[stage].mean() * 100 for stage in stages]
            ax.plot(stage_labels, means, label=f"{gender.title()} / {group_label.title()}", color=colors.get(gender, "gray"), linestyle=linestyle, marker="o")
    ax.set_ylabel("Perseverative Error Rate (%)")
    ax.set_title("WCST Early vs Late PE Rates by Gender × Loneliness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wcst_learning_curve_by_group.png", dpi=300)
    plt.close(fig)


def run_reverse_models(df: pd.DataFrame) -> None:
    rows = []
    for metric in EF_METRICS:
        columns = [
            metric,
            "z_ucla",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]
        if metric not in df.columns:
            continue
        data = df[columns].dropna()
        if len(data) < 30:
            continue
        formula = f"{metric} ~ z_ucla * gender_male + z_dass_dep + z_dass_anx + z_dass_stress + z_age"
        model = smf.ols(formula, data=data).fit()
        coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        coef.insert(0, "metric", metric)
        coef["n"] = len(data)
        coef["r_squared"] = model.rsquared
        rows.append(coef)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(OUTPUT_DIR / "ef_reverse_regression_coefficients.csv", index=False, encoding="utf-8-sig")


def run_bayesian_wcst(df: pd.DataFrame) -> None:
    data = df[["wcst_pe_rate", "z_ucla", "gender_male"]].dropna()
    if len(data) < 30:
        return
    y = data["wcst_pe_rate"].values
    z = data["z_ucla"].values
    gender = data["gender_male"].values
    interaction = z * gender

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_ucla = pm.Normal("beta_ucla", mu=0, sigma=5)
        beta_gender = pm.Normal("beta_gender", mu=0, sigma=5)
        beta_inter = pm.Normal("beta_inter", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = alpha + beta_ucla * z + beta_gender * gender + beta_inter * interaction
        pm.Normal("wcst_obs", mu=mu, sigma=sigma, observed=y)
        if sample_numpyro_nuts is not None:
            idata = sample_numpyro_nuts(
                draws=1000,
                tune=1000,
                target_accept=0.9,
                random_seed=42,
                chains=2,
                progress_bar=False,
            )
        else:
            idata = pm.sample(
                draws=100,
                tune=100,
                target_accept=0.9,
                random_seed=42,
                chains=1,
                cores=1,
                progressbar=False,
            )

    summary = az.summary(idata, var_names=["alpha", "beta_ucla", "beta_gender", "beta_inter", "sigma"], kind="stats")
    summary.to_csv(OUTPUT_DIR / "bayesian_wcst_summary.csv", encoding="utf-8-sig")

    probs = {
        "P(beta_ucla > 0)": float((idata.posterior["beta_ucla"] > 0).mean()),
        "P(beta_inter > 0)": float((idata.posterior["beta_inter"] > 0).mean()),
    }
    with (OUTPUT_DIR / "bayesian_wcst_probs.json").open("w", encoding="utf-8") as fh:
        json.dump(probs, fh, indent=2)


def main() -> None:
    df = load_master()
    run_subgroup_anova(df)
    plot_wcst_learning_curves(df)
    run_reverse_models(df)
    run_bayesian_wcst(df)
    print("EF alternative analyses complete. Outputs saved to ef_alternative/.")


if __name__ == "__main__":
    main()
