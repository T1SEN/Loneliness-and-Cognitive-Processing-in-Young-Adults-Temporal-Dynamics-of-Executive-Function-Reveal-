"""Bayesian hierarchical model: EF outcomes ~ D/A/S (shared effects)

목적
- DASS(우울/불안/스트레스)가 여러 EF 지표(Stroop, PRP, WCST, meta-control)에
  걸쳐 일관된 작은 신호를 가지는지, 과제 간 부분풀링(hierarchical pooling)을
  통해 보다 민감하게 탐지.

모형 개요
- 관측단위: (참가자, 과제) 쌍. 과제별 EF 점수 y_{ij}를 z-점수로 표준화.
- 예측변수: z_dass_dep, z_dass_anx, z_dass_stress, z_age, gender(0/1).
- 과제별 회귀계수 β_j,k 를 공유 hyper-mean μ_k와 hyper-sd τ_k에 묶음:
    β_dep[j] ~ Normal(μ_dep, τ_dep) 등.
- 관심 파라미터: μ_dep, μ_anx, μ_str (D/A/S 전반에 걸친 평균 효과)와
  각 β_dep[j], β_anx[j], β_str[j]의 후분포.

출력 (results/analysis_outputs/)
- dass_ef_hier_summary.csv : hyper/과제별 계수 후분포 요약
- dass_ef_hier_task_level.csv : 과제별(Outcome×Predictor) 정리 테이블
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from dass_exec_models import build_df, add_meta_control


BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().std() in (0, None) or s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / s.std()


def build_long_df() -> pd.DataFrame:
    df = build_df()
    df = add_meta_control(df)

    outcomes = [
        ("stroop_effect", "Stroop interference"),
        ("prp_bottleneck", "PRP bottleneck"),
        ("wcst_total_errors", "WCST total errors"),
        ("meta_control", "Meta-control"),
    ]

    rows: List[Dict] = []
    for idx, (col, label) in enumerate(outcomes):
        needed = [
            "participant_id",
            col,
            "dass_dep",
            "dass_anx",
            "dass_stress",
            "age",
            "gender",
        ]
        sub = df[needed].dropna().copy()
        if sub.empty:
            continue
        # encode gender: female=0, male=1 (rough heuristic)
        g = sub["gender"].astype(str)
        g_enc = np.where(g.str.contains("남"), 1.0, 0.0)

        rows.append(
            pd.DataFrame(
                {
                    "participant_id": sub["participant_id"].values,
                    "task_idx": idx,
                    "task_label": label,
                    "y_raw": pd.to_numeric(sub[col], errors="coerce").values,
                    "d_dep": pd.to_numeric(sub["dass_dep"], errors="coerce").values,
                    "d_anx": pd.to_numeric(sub["dass_anx"], errors="coerce").values,
                    "d_str": pd.to_numeric(sub["dass_stress"], errors="coerce").values,
                    "age": pd.to_numeric(sub["age"], errors="coerce").values,
                    "gender01": g_enc,
                }
            )
        )

    long = pd.concat(rows, ignore_index=True)
    # z-score y and predictors
    long["y"] = long.groupby("task_idx")["y_raw"].transform(z)
    long["z_dep"] = z(long["d_dep"])
    long["z_anx"] = z(long["d_anx"])
    long["z_str"] = z(long["d_str"])
    long["z_age"] = z(long["age"])
    # gender01는 이미 0/1, 필요시 중심화만
    long["g_center"] = long["gender01"] - long["gender01"].mean()

    # 완전한 케이스 필터링
    long = long.dropna(subset=["y", "z_dep", "z_anx", "z_str", "z_age", "g_center"])
    return long


def fit_model(long: pd.DataFrame) -> az.InferenceData:
    task_idx = long["task_idx"].to_numpy(dtype="int32")
    J = int(long["task_idx"].max() + 1)

    y = long["y"].to_numpy(dtype=float)
    z_dep = long["z_dep"].to_numpy(dtype=float)
    z_anx = long["z_anx"].to_numpy(dtype=float)
    z_str = long["z_str"].to_numpy(dtype=float)
    z_age = long["z_age"].to_numpy(dtype=float)
    g = long["g_center"].to_numpy(dtype=float)

    with pm.Model() as m:
        # Task-specific intercepts
        alpha = pm.Normal("alpha", 0.0, 1.0, shape=J)

        # Hierarchical priors for D/A/S slopes
        mu_dep = pm.Normal("mu_dep", 0.0, 0.5)
        tau_dep = pm.HalfNormal("tau_dep", 0.5)
        beta_dep = pm.Normal("beta_dep", mu_dep, tau_dep, shape=J)

        mu_anx = pm.Normal("mu_anx", 0.0, 0.5)
        tau_anx = pm.HalfNormal("tau_anx", 0.5)
        beta_anx = pm.Normal("beta_anx", mu_anx, tau_anx, shape=J)

        mu_str = pm.Normal("mu_str", 0.0, 0.5)
        tau_str = pm.HalfNormal("tau_str", 0.5)
        beta_str = pm.Normal("beta_str", mu_str, tau_str, shape=J)

        # Non-hierarchical slopes for age/gender (per task)
        beta_age = pm.Normal("beta_age", 0.0, 0.5, shape=J)
        beta_gen = pm.Normal("beta_gen", 0.0, 0.5, shape=J)

        sigma = pm.HalfNormal("sigma", 1.0, shape=J)

        mu = (
            alpha[task_idx]
            + beta_dep[task_idx] * z_dep
            + beta_anx[task_idx] * z_anx
            + beta_str[task_idx] * z_str
            + beta_age[task_idx] * z_age
            + beta_gen[task_idx] * g
        )

        pm.Normal("y_obs", mu=mu, sigma=sigma[task_idx], observed=y)

        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=42,
            return_inferencedata=True,
        )

    return idata


def summarize(idata: az.InferenceData, long: pd.DataFrame) -> None:
    # Hyper-level summary
    vars_of_interest = [
        "mu_dep",
        "mu_anx",
        "mu_str",
        "tau_dep",
        "tau_anx",
        "tau_str",
    ]
    summary = az.summary(idata, var_names=vars_of_interest, hdi_prob=0.95)
    summary.to_csv(OUT / "dass_ef_hier_summary.csv")

    # Task-specific D/A/S slopes
    beta_dep = idata.posterior["beta_dep"].stack(draws=("chain", "draw"))
    beta_anx = idata.posterior["beta_anx"].stack(draws=("chain", "draw"))
    beta_str = idata.posterior["beta_str"].stack(draws=("chain", "draw"))

    task_labels = (
        long[["task_idx", "task_label"]]
        .drop_duplicates()
        .set_index("task_idx")
        .sort_index()["task_label"]
        .tolist()
    )

    rows: List[Dict] = []
    rope = 0.1  # in SD units
    for j, name in enumerate(task_labels):
        for var_name, arr in [("dep", beta_dep), ("anx", beta_anx), ("str", beta_str)]:
            if var_name == "dep":
                v = arr.sel(beta_dep_dim_0=j)
            elif var_name == "anx":
                v = arr.sel(beta_anx_dim_0=j)
            else:
                v = arr.sel(beta_str_dim_0=j)
            vals = v.values
            mean = float(vals.mean())
            hdi = az.hdi(vals, hdi_prob=0.95)
            prob_dir = float(max((vals > 0).mean(), (vals < 0).mean()))
            prob_in_rope = float((np.abs(vals) < rope).mean())
            rows.append(
                {
                    "outcome": name,
                    "predictor": var_name,
                    "post_mean": mean,
                    "hdi_low": float(hdi[0]),
                    "hdi_high": float(hdi[1]),
                    "PD": prob_dir,
                    "P(|beta|<ROPE)": prob_in_rope,
                }
            )

    task_df = pd.DataFrame(rows)
    task_df.to_csv(OUT / "dass_ef_hier_task_level.csv", index=False)


def main() -> None:
    long = build_long_df()
    if long.empty:
        print("No complete data for hierarchical model.")
        return
    print(f"Fitting hierarchical model on N={len(long)} obs, J={long['task_idx'].nunique()} tasks...")
    idata = fit_model(long)
    summarize(idata, long)
    print("Saved hierarchical summaries to dass_ef_hier_summary.csv and dass_ef_hier_task_level.csv")


if __name__ == "__main__":
    main()

