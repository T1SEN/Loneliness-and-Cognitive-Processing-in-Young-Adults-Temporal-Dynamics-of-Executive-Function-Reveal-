#!/usr/bin/env python3
"""
고급 분석 파이프라인 (Pilot N≈70)
==================================
1) PyMC trace를 이용한 Bayes Factor / ROPE 요약
2) UCLA × DASS 프로파일 클러스터링
3) DDM 파라미터에 외로움 공변량을 투입한 간단한 PyMC 회귀
4) 외로움 → 정서(DASS) → EF 매개 효과 (Sobel 테스트)
5) Trial index × 외로움 피로 효과 (혼합효과)
6) spline 기반 비선형/threshold 회귀

결과 텍스트: results/analysis_outputs/advanced_analysis_report.txt
중간 산출물: 같은 폴더 내 CSV 파일들
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from patsy import dmatrix
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
RESULTS_DIR = ROOT_DIR / "results" / "analysis_outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = RESULTS_DIR / "advanced_analysis_report.txt"


def _load_analysis_df() -> pd.DataFrame:
    from loneliness_exec_models import build_analysis_dataframe, add_meta_control

    df = build_analysis_dataframe()
    df = add_meta_control(df)
    return df


# ---------------------------------------------------------------------------
# 1. Bayes Factor / ROPE
# ---------------------------------------------------------------------------
def savage_dickey(trace: az.InferenceData, var_name: str, prior_sd: float = 1.0) -> dict:
    posterior = trace.posterior[var_name].values.flatten()
    posterior_kde = stats.gaussian_kde(posterior)
    posterior_density_at_zero = posterior_kde(0)[0]
    prior_density_at_zero = stats.norm.pdf(0, loc=0, scale=prior_sd)
    bf01 = posterior_density_at_zero / prior_density_at_zero
    return {
        "posterior_mean": float(posterior.mean()),
        "posterior_sd": float(posterior.std()),
        "bf01": float(bf01),
    }


def rope_probability(trace: az.InferenceData, var_name: str, rope=(-0.05, 0.05)) -> float:
    posterior = trace.posterior[var_name].values.flatten()
    return float(((posterior >= rope[0]) & (posterior <= rope[1])).mean())


def bayes_factor_section() -> str:
    lines = ["# Bayes Factor / ROPE"]
    targets = {
        "loneliness_stroop_trace.nc": "Stroop β(ucla)",
        "loneliness_prp_trace.nc": "PRP β(ucla)",
        "wcst_switching_trace.nc": "WCST β(ucla)",
    }
    for fname, label in targets.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            lines.append(f"- {label}: trace file not found ({path.name})")
            continue
        trace = az.from_netcdf(path)
        if "beta_ucla" not in trace.posterior:
            lines.append(f"- {label}: beta_ucla not in trace")
            continue
        metrics = savage_dickey(trace, "beta_ucla")
        rope_p = rope_probability(trace, "beta_ucla")
        lines.append(
            f"- {label}: mean={metrics['posterior_mean']:.3f}, sd={metrics['posterior_sd']:.3f}, "
            f"BF01={metrics['bf01']:.3f}, P(ROPE ±0.05)={rope_p:.2f}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 2. K-means 프로파일
# ---------------------------------------------------------------------------
def profile_section(df: pd.DataFrame) -> str:
    use_cols = ["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    data = df.dropna(subset=use_cols + ["prp_bottleneck", "stroop_effect", "wcst_total_errors"]).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[use_cols])

    best_k, best_sil = None, -np.inf
    for k in range(2, 6):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_sil:
            best_sil = score
            best_k = k

    cluster_model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    data["cluster"] = cluster_model.fit_predict(X)
    summary = (
        data.groupby("cluster")[["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress",
                                 "stroop_effect", "prp_bottleneck", "wcst_total_errors"]]
        .mean()
        .reset_index()
    )
    summary_path = RESULTS_DIR / "cluster_profile_summary.csv"
    summary.to_csv(summary_path, index=False)

    # 간단한 ANOVA
    aov_results = []
    for outcome in ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]:
        model = ols(f"{outcome} ~ C(cluster)", data=data).fit()
        table = stats.f_oneway(
            *[group[outcome].values for _, group in data.groupby("cluster")]
        )
        aov_results.append((outcome, table.statistic, table.pvalue))

    lines = [
        "# UCLA × DASS Profile (K-Means)",
        f"- Best cluster k={best_k} (silhouette={best_sil:.3f})",
        f"- Profile summary saved to {summary_path.name}",
    ]
    for outcome, fstat, pval in aov_results:
        lines.append(f"  * {outcome}: F={fstat:.2f}, p={pval:.3f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 3. PyMC 회귀 (DDM 파라미터)
# ---------------------------------------------------------------------------
def ddm_covariate_section(df: pd.DataFrame) -> str:
    params_path = RESULTS_DIR / "stroop_ddm_parameters.csv"
    if not params_path.exists():
        return "# DDM Covariate: stroop_ddm_parameters.csv not found\n"
    params = pd.read_csv(params_path)
    merged = params.merge(df[["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]],
                          on="participant_id", how="inner")
    merged["drift_diff"] = merged["drift_congruent"] - merged["drift_incongruent"]
    data = merged.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["drift_diff", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"])
    y = data["drift_diff"].values
    X = data[["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].values

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.Deterministic("mu", (X * beta).sum(axis=1))
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(1000, tune=1000, chains=4, target_accept=0.9, progressbar=False)

    az.to_netcdf(trace, RESULTS_DIR / "ddm_covariate_trace.nc")
    beta_summary = az.summary(trace, var_names=["beta"])
    beta_summary.to_csv(RESULTS_DIR / "ddm_covariate_summary.csv")
    lines = ["# DDM Covariate (drift difference)",
             beta_summary.to_string()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 4. 매개/조절 (Sobel test)
# ---------------------------------------------------------------------------
def mediation_section(df: pd.DataFrame) -> str:
    data = df.dropna(subset=["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "prp_bottleneck"]).copy()
    mediator = data["z_dass_dep"]
    predictor = data["z_ucla"]
    outcome = data["prp_bottleneck"]

    med_df = data.rename(columns={"z_ucla": "predictor", "z_dass_dep": "mediator"})
    med_model = ols("mediator ~ predictor", data=med_df).fit()
    out_model = ols("prp_bottleneck ~ z_ucla + z_dass_dep", data=data).fit()
    a = med_model.params["predictor"]
    b = out_model.params["z_dass_dep"]
    sa = med_model.bse["predictor"]
    sb = out_model.bse["z_dass_dep"]
    sobel_z = (a * b) / np.sqrt(b ** 2 * sa ** 2 + a ** 2 * sb ** 2)
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    lines = [
        "# Mediation (UCLA -> Depression -> PRP)",
        f"- a (UCLA→Dep) = {a:.3f}",
        f"- b (Dep→PRP|UCLA) = {b:.3f}",
        f"- Sobel z = {sobel_z:.3f}, p = {sobel_p:.3f}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 5. Trial index × Loneliness 혼합모형
# ---------------------------------------------------------------------------
def fatigue_section(df: pd.DataFrame) -> str:
    stroop_trials = pd.read_csv(ROOT_DIR / "results" / "4c_stroop_trials.csv")
    stroop_trials = stroop_trials[(stroop_trials["timeout"] == False) & stroop_trials["rt_ms"].notna()]
    stroop_trials = stroop_trials.merge(
        df[["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]],
        on="participant_id",
        how="inner",
    )
    stroop_trials["trial_scaled"] = stroop_trials["trial"] / stroop_trials["trial"].max()
    stroop_data = stroop_trials.dropna(subset=["trial_scaled", "participant_id"]).copy()

    def slope_func(group):
        if group["trial_scaled"].nunique() < 3:
            return np.nan
        coef = np.polyfit(group["trial_scaled"], group["rt_ms"], deg=1)
        return coef[0]  # slope

    slopes = (
        stroop_data.groupby("participant_id")
        .apply(slope_func)
        .rename("rt_slope")
        .reset_index()
        .dropna()
    )
    slopes = slopes.merge(
        df[["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]],
        on="participant_id",
        how="left",
    ).dropna()
    slope_model = ols(
        "rt_slope ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress",
        data=slopes,
    ).fit()

    lines = [
        "# Fatigue / Time trends (per-participant slope)",
        f"- Mean slope = {slopes['rt_slope'].mean():.2f}",
        f"- UCLA effect on slope: coef = {slope_model.params['z_ucla']:.3f}, "
        f"p = {slope_model.pvalues['z_ucla']:.3f}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 6. 비선형 (spline)
# ---------------------------------------------------------------------------
def threshold_section(df: pd.DataFrame) -> str:
    data = df.dropna(subset=["z_ucla", "prp_bottleneck", "z_dass_dep", "z_dass_anx", "z_dass_stress"]).copy()
    spline = dmatrix("bs(z_ucla, df=4, degree=3, include_intercept=False)", data=data, return_type="dataframe")
    spline_cols = [f"spline_{i}" for i in range(spline.shape[1])]
    spline.columns = spline_cols
    design = pd.concat([data.reset_index(drop=True), spline], axis=1)
    formula = "prp_bottleneck ~ " + " + ".join(spline_cols) + " + z_dass_dep + z_dass_anx + z_dass_stress"
    model = ols(formula, data=design).fit()
    lines = [
        "# Non-linear spline regression (PRP)",
        f"- R^2 = {model.rsquared:.3f}",
        f"- Max |t| for spline terms = {model.tvalues[spline_cols].abs().max():.2f}",
    ]
    return "\n".join(lines) + "\n"


def main():
    df = _load_analysis_df()
    sections = [
        bayes_factor_section(),
        profile_section(df),
        ddm_covariate_section(df),
        mediation_section(df),
        fatigue_section(df),
        threshold_section(df),
    ]
    report = "\n".join(sections)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n정밀 분석 보고서 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
