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
from data_loader_utils import load_master_dataset
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_base() -> pd.DataFrame:
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    def _pick(candidates):
        for c in candidates:
            if c in master.columns:
                return master[c]
        return pd.Series(np.nan, index=master.index)

    df = pd.DataFrame({
        "participant_id": master["participant_id"],
        "age": _to_num(master.get("age")),
        "gender": master.get("gender_normalized", master.get("gender")),
    })
    df["gender"] = df["gender"].fillna("").astype(str).str.strip().str.lower()

    # Surveys
    if "ucla_total" in master.columns:
        df["ucla_total"] = _to_num(master["ucla_total"])
    elif "ucla_score" in master.columns:
        df["ucla_total"] = _to_num(master["ucla_score"])

    df["dass_dep"] = _to_num(master.get("dass_depression"))
    df["dass_anx"] = _to_num(master.get("dass_anxiety"))
    df["dass_stress"] = _to_num(master.get("dass_stress"))

    # EF metrics (with fallbacks)
    df["prp_bottleneck"] = _to_num(master.get("prp_bottleneck"))
    if df["prp_bottleneck"].isna().all():
        short_col = _to_num(_pick(["rt2_soa_50", "rt2_soa_150"]))
        long_col = _to_num(_pick(["rt2_soa_1200", "rt2_soa_1200_ms"]))
        if not short_col.isna().all() and not long_col.isna().all():
            df["prp_bottleneck"] = short_col - long_col
    df["prp_rt_slope"] = np.where(
        df["prp_bottleneck"].notna(),
        df["prp_bottleneck"] / 1150.0,
        np.nan,
    )
    df["prp_mrt_t1"] = _to_num(_pick(["mrt_t1"]))
    df["prp_mrt_t2"] = _to_num(_pick(["mrt_t2"]))
    df["prp_acc_t1"] = _to_num(_pick(["acc_t1"]))
    df["prp_acc_t2"] = _to_num(_pick(["acc_t2"]))

    df["stroop_effect"] = _to_num(_pick(["stroop_interference", "stroop_effect"]))
    if df["stroop_effect"].isna().all():
        incong = _to_num(_pick(["mrt_incong", "rt_mean_incongruent"]))
        cong = _to_num(_pick(["mrt_cong", "rt_mean_congruent"]))
        if not incong.isna().all() and not cong.isna().all():
            df["stroop_effect"] = incong - cong
    df["stroop_accuracy"] = _to_num(_pick(["accuracy", "accuracy_stroop"]))
    df["mrt_incong"] = _to_num(_pick(["mrt_incong"]))
    df["mrt_cong"] = _to_num(_pick(["mrt_cong"]))
    df["mrt_total"] = _to_num(_pick(["mrt_total"]))
    if "mrt_cong" in df.columns and "mrt_incong" in df.columns:
        df["stroop_ratio_incong_cong"] = df["mrt_incong"] / df["mrt_cong"]

    df["wcst_total_errors"] = _to_num(_pick(["wcst_total_errors", "totalErrorCount", "total_error_count"]))
    df["wcst_persev_errors"] = _to_num(_pick(["wcst_persev_errors", "perseverativeErrorCount", "perseverative_error_count"]))
    df["wcst_nonpersev_errors"] = _to_num(_pick(["wcst_nonpersev_errors", "nonPerseverativeErrorCount", "non_perseverative_error_count"]))
    df["wcst_conceptual_pct"] = _to_num(_pick(["wcst_conceptual_pct", "conceptualLevelResponsesPercent", "conceptual_level_responses_percent"]))
    df["wcst_persev_resp_pct"] = _to_num(_pick(["wcst_persev_resp_pct", "perseverativeResponsesPercent", "perseverative_responses_percent"]))
    df["wcst_failure_to_maintain_set"] = _to_num(_pick(["wcst_failure_to_maintain_set", "failureToMaintainSet", "failure_to_maintain_set"]))

    # DDM / advanced EF params if available (keep optional merges)
    ddm_stroop_path = OUT / "stroop_ddm_parameters.csv"
    ddm_prp_path = OUT / "prp_bottleneck_parameters.csv"
    wcst_switch_path = OUT / "wcst_switching_parameters.csv"
    ddm_stroop = pd.read_csv(ddm_stroop_path) if ddm_stroop_path.exists() else pd.DataFrame()
    ddm_prp = pd.read_csv(ddm_prp_path) if ddm_prp_path.exists() else pd.DataFrame()
    wcst_adv = pd.read_csv(wcst_switch_path) if wcst_switch_path.exists() else pd.DataFrame()

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
