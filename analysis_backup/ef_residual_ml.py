"""
EF -> Loneliness residual prediction (DASS-controlled) with ML

Pipeline:
 1) Load participants, surveys (UCLA, DASS), EF summaries, and DDM-derived EF params.
 2) Fit baseline OLS: UCLA ~ DASS + age + gender; compute residuals (ucla_resid).
 3) Build EF feature sets (base EF + extended including DDM features + PCA meta-control).
 4) Cross-validated ML to predict ucla_resid using EF features; report R2/MAE/RMSE.
 5) Also compute incremental CV-R2 for predicting UCLA directly: (DASS+demo) vs (DASS+demo+EF).
 6) Power analysis: required N for detecting r in [0.10, 0.15] at alpha=0.05, power=0.80.
 7) Save summary CSVs and simple figures.

Outputs written to results/analysis_outputs/:
  - ef_residual_ml_metrics.csv
  - ef_residual_incremental_r2.csv
  - ef_residual_feature_importance_*.csv
  - ef_residual_r2_bar.png
  - ef_residual_power_curve.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT = RESULTS_DIR / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_base() -> pd.DataFrame:
    participants = _read_csv(RESULTS_DIR / "1_participants_info.csv").rename(
        columns={"participantId": "participant_id"}
    )
    surveys = _read_csv(RESULTS_DIR / "2_surveys_results.csv").rename(
        columns={"participantId": "participant_id", "surveyName": "survey"}
    )
    cog = _read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv").rename(
        columns={"participantId": "participant_id", "testName": "test"}
    )

    # Participants
    demo = participants[["participant_id", "age", "gender"]].copy()
    demo["age"] = _to_num(demo["age"])  # gender left as raw (Korean)

    # Surveys
    ucla = surveys[surveys["survey"].str.lower() == "ucla"][
        ["participant_id", "score"]
    ].rename(columns={"score": "ucla_total"})
    ucla["ucla_total"] = _to_num(ucla["ucla_total"]) 

    dass = surveys[surveys["survey"].str.lower() == "dass"][
        ["participant_id", "score_D", "score_A", "score_S"]
    ].rename(columns={"score_D": "dass_dep", "score_A": "dass_anx", "score_S": "dass_stress"})
    for c in ["dass_dep", "dass_anx", "dass_stress"]:
        dass[c] = _to_num(dass[c])

    # EF from summary
    prp = cog[cog["test"].str.lower() == "prp"].copy()
    for c in ["rt2_soa_50", "rt2_soa_1200", "mrt_t1", "mrt_t2", "acc_t1", "acc_t2"]:
        prp[c] = _to_num(prp[c])
    prp = prp.assign(
        prp_rt_short=prp["rt2_soa_50"],
        prp_rt_long=prp["rt2_soa_1200"],
        prp_bottleneck=lambda d: d["rt2_soa_50"] - d["rt2_soa_1200"],
        prp_rt_slope=lambda d: (d["rt2_soa_50"] - d["rt2_soa_1200"]) / 1150.0,
    )[[
        "participant_id",
        "prp_bottleneck",
        "prp_rt_slope",
        "mrt_t1",
        "mrt_t2",
        "acc_t1",
        "acc_t2",
    ]].rename(columns={
        "mrt_t1": "prp_mrt_t1",
        "mrt_t2": "prp_mrt_t2",
        "acc_t1": "prp_acc_t1",
        "acc_t2": "prp_acc_t2",
    })

    stroop = cog[cog["test"].str.lower() == "stroop"][
        ["participant_id", "stroop_effect", "accuracy", "mrt_incong", "mrt_cong", "mrt_total"]
    ].rename(columns={"accuracy": "stroop_accuracy"})
    for c in ["stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong", "mrt_total"]:
        stroop[c] = _to_num(stroop[c])
    stroop["stroop_ratio_incong_cong"] = stroop["mrt_incong"] / stroop["mrt_cong"]

    wcst = cog[cog["test"].str.lower() == "wcst"][
        [
            "participant_id",
            "totalErrorCount",
            "perseverativeErrorCount",
            "nonPerseverativeErrorCount",
            "conceptualLevelResponsesPercent",
            "perseverativeResponsesPercent",
            "failureToMaintainSet",
        ]
    ].rename(
        columns={
            "totalErrorCount": "wcst_total_errors",
            "perseverativeErrorCount": "wcst_persev_errors",
            "nonPerseverativeErrorCount": "wcst_nonpersev_errors",
            "conceptualLevelResponsesPercent": "wcst_conceptual_pct",
            "perseverativeResponsesPercent": "wcst_persev_resp_pct",
            "failureToMaintainSet": "wcst_failure_to_maintain_set",
        }
    )
    for c in wcst.columns:
        if c != "participant_id":
            wcst[c] = _to_num(wcst[c])

    # DDM / advanced EF params if available
    ddm_stroop_path = OUT / "stroop_ddm_parameters.csv"
    ddm_prp_path = OUT / "prp_bottleneck_parameters.csv"
    wcst_switch_path = OUT / "wcst_switching_parameters.csv"
    ddm_stroop = pd.read_csv(ddm_stroop_path) if ddm_stroop_path.exists() else pd.DataFrame()
    ddm_prp = pd.read_csv(ddm_prp_path) if ddm_prp_path.exists() else pd.DataFrame()
    wcst_adv = pd.read_csv(wcst_switch_path) if wcst_switch_path.exists() else pd.DataFrame()

    df = demo.merge(ucla, on="participant_id", how="left")
    df = df.merge(dass, on="participant_id", how="left")
    df = df.merge(stroop, on="participant_id", how="left")
    df = df.merge(prp, on="participant_id", how="left")
    df = df.merge(wcst, on="participant_id", how="left")
    if not ddm_stroop.empty:
        df = df.merge(ddm_stroop, on="participant_id", how="left", suffixes=(None, "_ddm"))
    if not ddm_prp.empty:
        df = df.merge(ddm_prp, on="participant_id", how="left")
    if not wcst_adv.empty:
        df = df.merge(wcst_adv, on="participant_id", how="left")
    return df


def gender_ohe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cat = df[["gender"]].fillna("미상").astype(str)
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    arr = ohe.fit_transform(cat)
    cols = [f"gender={c}" for c in ohe.categories_[0]]
    out = pd.DataFrame(arr, columns=cols, index=df.index)
    return out, cols


def fit_baseline_residuals(df: pd.DataFrame) -> pd.DataFrame:
    base = df[["ucla_total", "dass_dep", "dass_anx", "dass_stress", "age", "gender"]].copy()
    base = base.dropna(subset=["ucla_total"]).copy()
    # One-hot for gender within statsmodels formula
    base = base.rename(columns={"ucla_total": "y"})
    model = smf.ols("y ~ dass_dep + dass_anx + dass_stress + age + C(gender)", data=base).fit()
    resid = model.resid
    out = df.copy()
    out.loc[base.index, "ucla_resid"] = resid
    # Save baseline summary metrics
    summ = {
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    pd.DataFrame([summ]).to_csv(OUT / "ef_residual_baseline_ucla_on_dass_demo.csv", index=False)
    return out


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    ef_basic = [
        "stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong", "mrt_total", "stroop_ratio_incong_cong",
        "prp_bottleneck", "prp_rt_slope", "prp_mrt_t1", "prp_mrt_t2", "prp_acc_t1", "prp_acc_t2",
        "wcst_total_errors", "wcst_persev_errors", "wcst_nonpersev_errors", "wcst_conceptual_pct", "wcst_persev_resp_pct", "wcst_failure_to_maintain_set",
    ]
    ef_ddm = [
        # Stroop DDM
        "drift_congruent", "drift_incongruent", "drift_neutral", "stroop_interference", "pop_interference_mean", "pop_interference_sd",
        # PRP bottleneck param
        "prp_bottleneck_effect", "pop_prp_mean", "pop_prp_sd",
        # WCST advanced
        "wcst_accuracy", "wcst_persev_rate", "wcst_nonpersev_rate", "wcst_total_error_rate", "wcst_switch_cost_ms",
    ]
    ef_ddm = [c for c in ef_ddm if c in df.columns]
    ef_all = [c for c in ef_basic if c in df.columns] + ef_ddm
    return {
        "ef_basic": [c for c in ef_basic if c in df.columns],
        "ef_extended": ef_all,
    }


def add_meta_control(df: pd.DataFrame, ef_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in ef_cols if c in df.columns]
    complete = df[cols].dropna()
    if len(complete) < 20 or len(cols) < 3:
        df["meta_control"] = np.nan
        return df, ["meta_control"]
    scaler = StandardScaler()
    X = scaler.fit_transform(complete[cols])
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(X).ravel()
    tmp = complete.assign(meta_control=scores)
    df = df.merge(tmp[["meta_control"]], left_index=True, right_index=True, how="left")
    return df, ["meta_control"]


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def run_residual_ml(df: pd.DataFrame) -> None:
    # Build feature sets
    feats = build_feature_sets(df)
    # Add latent factor to extended EF
    df_mc, mc_cols = add_meta_control(df.copy(), feats["ef_extended"]) 
    feats["ef_plus_meta"] = feats["ef_extended"] + mc_cols

    y = df_mc["ucla_resid"].dropna()
    idx = y.index
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics_rows = []
    importances_rows = []
    r2_bars = []

    for feats_key, cols in feats.items():
        X = df_mc.loc[idx, cols]
        # Build models for this feature set
        models = {
            "ridge": Pipeline([("prep", ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols)
            ], remainder="drop")), ("mdl", Ridge(alpha=1.0, random_state=42))]),
            "lasso": Pipeline([("prep", ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols)
            ], remainder="drop")), ("mdl", Lasso(alpha=0.01, max_iter=10000, random_state=42))]),
            "rf": Pipeline([("prep", ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), cols)
            ], remainder="drop")), ("mdl", RandomForestRegressor(n_estimators=500, random_state=42))]),
            "gbrt": Pipeline([("prep", ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), cols)
            ], remainder="drop")), ("mdl", GradientBoostingRegressor(random_state=42))]),
        }

        # Train/eval
        for name, pipe in models.items():
            
            y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
            m = eval_regression(y, y_pred)
            m.update({"feature_set": feats_key, "model": name, "n": int(len(y))})
            metrics_rows.append(m)
            r2_bars.append((feats_key, name, m["R2"]))

            # Fit once to all to get permutation importance
            try:
                fitted = pipe.fit(X, y)
                perm = permutation_importance(fitted, X, y, n_repeats=30, random_state=42, scoring="r2")
                imp_df = pd.DataFrame({
                    "feature": cols[: len(perm.importances_mean)],
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                    "feature_set": feats_key,
                    "model": name,
                })
                importances_rows.append(imp_df)
            except Exception as e:
                pass

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT / "ef_residual_ml_metrics.csv", index=False)
    if importances_rows:
        pd.concat(importances_rows, ignore_index=True).to_csv(OUT / "ef_residual_feature_importance.csv", index=False)

    # Simple bar plot for R2 by model within each feature set (best model highlighted)
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot = metrics_df.pivot_table(index="feature_set", columns="model", values="R2")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("CV R2 (UCLA residual)")
        ax.set_title("EF-only ML prediction of UCLA residual (DASS+demo controlled)")
        plt.tight_layout()
        fig.savefig(OUT / "ef_residual_r2_bar.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass


def incremental_r2(df: pd.DataFrame) -> None:
    # Predict UCLA directly with CV: (demo+DASS) vs (demo+DASS+EF)
    work = df.dropna(subset=["ucla_total"]).copy()
    y = work["ucla_total"].astype(float).values
    demo_dass = [c for c in ["age", "dass_dep", "dass_anx", "dass_stress", "gender"] if c in work.columns]
    ef_sets = build_feature_sets(work)
    # Preprocessors
    prep_demo = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), [c for c in demo_dass if c != "gender"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), ["gender"]) if "gender" in demo_dass else ("drop", "drop", [])
    ], remainder="drop")
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline (demo + DASS)
    Xb = work[[c for c in demo_dass if c != "gender"] + (["gender"] if "gender" in demo_dass else [])]
    pipe_b = Pipeline([("prep", prep_demo), ("mdl", rf)])
    y_pred_b = cross_val_predict(pipe_b, Xb, y, cv=cv)
    r2_b = r2_score(y, y_pred_b)

    rows = [{"feature_set": "demo_dass", "R2": float(r2_b)}]
    for name, cols in ef_sets.items():
        Xe = work[[c for c in demo_dass if c != "gender"] + (["gender"] if "gender" in demo_dass else []) + cols]
        prep_e = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), [c for c in Xe.columns if c != "gender"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), ["gender"]) if "gender" in Xe.columns else ("drop", "drop", [])
        ], remainder="drop")
        pipe_e = Pipeline([("prep", prep_e), ("mdl", rf)])
        y_pred_e = cross_val_predict(pipe_e, Xe, y, cv=cv)
        r2_e = r2_score(y, y_pred_e)
        rows.append({"feature_set": f"demo_dass+{name}", "R2": float(r2_e), "DeltaR2": float(r2_e - r2_b)})
    pd.DataFrame(rows).to_csv(OUT / "ef_residual_incremental_r2.csv", index=False)


def required_n_for_r(r: float, alpha: float = 0.05, power: float = 0.80) -> int:
    # Fisher z method
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    zr = 0.5 * np.log((1 + r) / (1 - r))
    n = ((z_alpha + z_beta) / zr) ** 2 + 3
    return int(np.ceil(n))


def power_curve() -> None:
    rs = np.linspace(0.05, 0.30, 11)
    ns = [required_n_for_r(float(r)) for r in rs]
    df = pd.DataFrame({"r": rs, "required_n": ns})
    df.to_csv(OUT / "ef_residual_power_curve.csv", index=False)
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["r"], df["required_n"], marker="o")
        ax.set_xlabel("Target correlation (r)")
        ax.set_ylabel("Required N (alpha=0.05, power=0.80)")
        ax.set_title("Power curve for detecting EF–loneliness effect")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT / "ef_residual_power_curve.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass


def main() -> None:
    df = load_base()
    df = fit_baseline_residuals(df)
    run_residual_ml(df)
    incremental_r2(df)
    power_curve()
    # Print a quick power note for r=0.10, 0.15
    n10 = required_n_for_r(0.10)
    n15 = required_n_for_r(0.15)
    print(f"Required N for r=0.10: ~{n10}")
    print(f"Required N for r=0.15: ~{n15}")


if __name__ == "__main__":
    main()
