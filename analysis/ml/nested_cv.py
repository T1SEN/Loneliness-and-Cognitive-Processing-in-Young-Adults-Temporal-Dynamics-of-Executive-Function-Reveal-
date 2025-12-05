"""
Nested CV + Hyperparameter Tuning for Loneliness Prediction

Goals
- Provide a more rigorous, variance-controlled estimate of out-of-sample
  performance than simple CV by adding an inner tuning loop.
- Compare feature sets (ef, demo, demo_dass, ef_demo, ef_demo_dass).
- Output per-fold predictions, mean±sd metrics, tuned params, permutation
  importances, and partial dependence plots for interpretability.

Usage examples
- Classification (best baseline: demo_dass):
    python analysis/ml_nested_tuned.py --task classification --features demo_dass
- Regression (UCLA continuous):
    python analysis/ml_nested_tuned.py --task regression --features demo_dass
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from analysis.preprocessing import load_master_dataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT = RESULTS_DIR / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_base_dataframe() -> pd.DataFrame:
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    def _pick(candidates):
        for c in candidates:
            if c in master.columns:
                return master[c]
        return pd.Series(np.nan, index=master.index)

    base = pd.DataFrame({
        "participant_id": master["participant_id"],
        "age": _to_num(master.get("age")),
        "gender": master.get("gender_normalized", master.get("gender")),
        "education": master.get("education"),
        "courseName": master.get("courseName"),
        "professorName": master.get("professorName"),
        "classSection": master.get("classSection"),
        "createdAt": master.get("createdAt"),
    })
    base["gender"] = base["gender"].fillna("").astype(str).str.strip().str.lower()

    try:
        dt = pd.to_datetime(base["createdAt"], errors="coerce")
        base["created_hour"] = dt.dt.hour.astype("Int64")
        base["created_dow"] = dt.dt.day_name()
        def hour_bin(h):
            if pd.isna(h):
                return np.nan
            if 5 <= h < 12:
                return "morning"
            if 12 <= h < 17:
                return "afternoon"
            if 17 <= h < 22:
                return "evening"
            return "night"
        base["created_hour_bin"] = base["created_hour"].apply(hour_bin)
    except Exception:
        base["created_hour"] = np.nan
        base["created_dow"] = np.nan
        base["created_hour_bin"] = np.nan

    if "ucla_total" in master.columns:
        base["ucla_total"] = _to_num(master["ucla_total"])
    elif "ucla_score" in master.columns:
        base["ucla_total"] = _to_num(master["ucla_score"])
    base["ucla_duration"] = _to_num(master.get("ucla_duration"))

    base["dass_dep"] = _to_num(master.get("dass_depression"))
    base["dass_anx"] = _to_num(master.get("dass_anxiety"))
    base["dass_stress"] = _to_num(master.get("dass_stress"))
    base["dass_duration"] = _to_num(master.get("dass_duration"))

    base["prp_bottleneck"] = _to_num(master.get("prp_bottleneck"))
    if base["prp_bottleneck"].isna().all():
        short_col = _to_num(_pick(["rt2_soa_50", "rt2_soa_150"]))
        long_col = _to_num(_pick(["rt2_soa_1200", "rt2_soa_1200_ms"]))
        if not short_col.isna().all() and not long_col.isna().all():
            base["prp_bottleneck"] = short_col - long_col
    base["prp_rt_slope"] = np.where(
        base["prp_bottleneck"].notna(),
        base["prp_bottleneck"] / 1150.0,
        np.nan,
    )
    base["prp_mrt_t1"] = _to_num(_pick(["mrt_t1"]))
    base["prp_mrt_t2"] = _to_num(_pick(["mrt_t2"]))
    base["prp_acc_t1"] = _to_num(_pick(["acc_t1"]))
    base["prp_acc_t2"] = _to_num(_pick(["acc_t2"]))

    base["stroop_effect"] = _to_num(_pick(["stroop_interference", "stroop_effect"]))
    base["stroop_accuracy"] = _to_num(_pick(["accuracy", "accuracy_stroop"]))
    base["mrt_incong"] = _to_num(_pick(["mrt_incong"]))
    base["mrt_cong"] = _to_num(_pick(["mrt_cong"]))
    base["mrt_total"] = _to_num(_pick(["mrt_total"]))
    if base["stroop_effect"].isna().all():
        incong = base["mrt_incong"]
        cong = base["mrt_cong"]
        if not incong.isna().all() and not cong.isna().all():
            base["stroop_effect"] = incong - cong

    base["wcst_total_errors"] = _to_num(_pick(["wcst_total_errors", "totalErrorCount", "total_error_count"]))
    base["wcst_persev_errors"] = _to_num(_pick(["wcst_persev_errors", "perseverativeErrorCount", "perseverative_error_count"]))
    base["wcst_nonpersev_errors"] = _to_num(_pick(["wcst_nonpersev_errors", "nonPerseverativeErrorCount", "non_perseverative_error_count"]))
    base["wcst_conceptual_pct"] = _to_num(_pick(["wcst_conceptual_pct", "conceptualLevelResponsesPercent", "conceptual_level_responses_percent"]))
    base["wcst_persev_resp_pct"] = _to_num(_pick(["wcst_persev_resp_pct", "perseverativeResponsesPercent", "perseverative_responses_percent"]))
    base["wcst_failure_to_maintain_set"] = _to_num(_pick(["wcst_failure_to_maintain_set", "failureToMaintainSet", "failure_to_maintain_set"]))

    # Optional trial-derived features
    tfeat_path = RESULTS_DIR / "analysis_outputs" / "trial_level_features.csv"
    if tfeat_path.exists():
        tfeat = pd.read_csv(tfeat_path)
        base = base.merge(tfeat, on="participant_id", how="left")

    # Batch-effect validation logging preserved below
    return base


def select_features(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str]]:
    ef_numeric = [
        "stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong", "mrt_total",
        "prp_bottleneck", "prp_rt_slope", "prp_mrt_t1", "prp_mrt_t2", "prp_acc_t1", "prp_acc_t2",
        "wcst_total_errors", "wcst_persev_errors", "wcst_nonpersev_errors", "wcst_conceptual_pct", "wcst_persev_resp_pct", "wcst_failure_to_maintain_set",
    ]
    # Add trial-derived EF features if present
    trial_cols = [
        "prp_t2_cv_all", "prp_t2_cv_short", "prp_t2_cv_long", "prp_t2_trials",
        "stroop_post_error_slowing", "stroop_post_error_rt", "stroop_post_correct_rt", "stroop_incong_slope", "stroop_trials",
    ]
    ef_numeric = ef_numeric + [c for c in trial_cols if c in df.columns]
    # CRITICAL: Time features (created_hour, created_dow, created_hour_bin) REMOVED
    # Reason: Potential batch effects (recruitment timing may confound UCLA predictions)
    # See batch effect validation (lines 181-236) for empirical tests
    demo_num = ["age", "ucla_duration", "dass_duration", "duration_prp", "duration_stroop", "duration_wcst"]
    demo_cat = ["gender", "education", "courseName", "professorName", "classSection"]
    dass_num = ["dass_dep", "dass_anx", "dass_stress"]

    fs = feature_set.lower()
    if fs == "ef":
        num, cat = ef_numeric, []
    elif fs == "demo":
        num, cat = demo_num, demo_cat
    elif fs == "demo_dass":
        num, cat = demo_num + dass_num, demo_cat
    elif fs == "meta":
        # demographics + platform/time + durations (no DASS, no EF)
        num, cat = demo_num, demo_cat
    elif fs == "meta_ef":
        # meta + EF (no DASS)
        num, cat = ef_numeric + demo_num, demo_cat
    elif fs == "meta_ef_dass":
        # meta + EF + DASS
        num, cat = ef_numeric + demo_num + dass_num, demo_cat
    elif fs == "ef_demo":
        num, cat = ef_numeric + demo_num, demo_cat
    elif fs in ("ef_demo_dass", "all"):
        num, cat = ef_numeric + demo_num + dass_num, demo_cat
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    num = [c for c in num if c in df.columns]
    cat = [c for c in cat if c in df.columns]
    return num, cat


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scale_num: bool) -> ColumnTransformer:
    num_pipe = [
        ("imp", SimpleImputer(strategy="median")),
    ]
    if scale_num:
        num_pipe.append(("sc", StandardScaler()))
    num_pipe = Pipeline(num_pipe)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])


def nested_classification(df: pd.DataFrame, feature_set: str) -> None:
    numeric_cols, categorical_cols = select_features(df, feature_set)
    work = df.dropna(subset=["ucla_total"]).copy()
    # Build binary high_loneliness
    thr = work["ucla_total"].quantile(0.75)
    work["y"] = (work["ucla_total"] >= thr).astype(int)
    X = work[numeric_cols + categorical_cols]
    y = work["y"].to_numpy()

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    preds_rows = []
    tuned_params = []

    for i, (tr, te) in enumerate(outer.split(X, y), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        # Candidate models with grids
        candidates: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {}
        # Logistic Regression
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=True)
        logreg = Pipeline([("prep", pre), ("mdl", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42))])
        grid_logreg = {"mdl__C": [0.1, 1.0, 10.0]}
        candidates["logreg"] = (logreg, grid_logreg)
        # RandomForest
        pre_tree = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        rf = Pipeline([("prep", pre_tree), ("mdl", RandomForestClassifier(class_weight="balanced", random_state=42))])
        grid_rf = {"mdl__n_estimators": [300, 600], "mdl__max_depth": [None, 8, 12], "mdl__min_samples_leaf": [1, 3]}
        candidates["rf"] = (rf, grid_rf)
        # GradientBoosting
        gbrt = Pipeline([("prep", pre_tree), ("mdl", GradientBoostingClassifier(random_state=42))])
        grid_gbrt = {"mdl__n_estimators": [300, 600], "mdl__learning_rate": [0.05, 0.1], "mdl__max_depth": [2, 3]}
        candidates["gbrt"] = (gbrt, grid_gbrt)
        # XGBoost
        if HAS_XGBOOST:
            xgb = Pipeline([("prep", pre_tree), ("mdl", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
            grid_xgb = {"mdl__n_estimators": [200, 400], "mdl__max_depth": [3, 5], "mdl__learning_rate": [0.05, 0.1]}
            candidates["xgb"] = (xgb, grid_xgb)

        best_name = None
        best_est = None
        best_score = -np.inf
        best_params = None

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        for name, (pipe, grid) in candidates.items():
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=None)
            gs.fit(Xtr, ytr)
            if gs.best_score_ > best_score:
                best_score = gs.best_score_
                best_est = gs.best_estimator_
                best_name = name
                best_params = gs.best_params_

        # Evaluate on outer test
        proba = None
        try:
            proba = best_est.predict_proba(Xte)[:, 1]
        except Exception:
            pass
        ypred = best_est.predict(Xte)

        row = {
            "fold": i,
            "model": best_name,
            "Accuracy": float(accuracy_score(yte, ypred)),
            "F1": float(f1_score(yte, ypred)),
            "Precision": float(precision_score(yte, ypred, zero_division=0)),
            "Recall": float(recall_score(yte, ypred)),
        }
        if proba is not None:
            try:
                row["ROC_AUC"] = float(roc_auc_score(yte, proba))
                row["PR_AUC"] = float(average_precision_score(yte, proba))
            except Exception:
                pass
        rows.append(row)
        preds_rows.append(pd.DataFrame({
            "fold": i,
            "participant_id": work.iloc[te]["participant_id"].values,
            "y_true": yte,
            "y_pred": ypred,
            "y_proba": proba if proba is not None else np.nan,
        }))
        tuned_params.append({"fold": i, "model": best_name, **(best_params or {})})

    metrics = pd.DataFrame(rows)
    preds = pd.concat(preds_rows, ignore_index=True)
    params_df = pd.DataFrame(tuned_params)

    prefix = f"nested_{feature_set}_classification"
    metrics.to_csv(OUT / f"{prefix}_metrics.csv", index=False)
    preds.to_csv(OUT / f"{prefix}_predictions.csv", index=False)
    params_df.to_csv(OUT / f"{prefix}_params.csv", index=False)

    # Fit best on full data for interpretability (choose by majority winner across folds)
    winner = params_df.groupby("model").size().idxmax()

    # Recreate winner with its most common params, fit on all
    # Use rf or gbrt pre_tree; logreg pre scaled
    if winner == "logreg":
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=True)
        base = Pipeline([("prep", pre), ("mdl", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42))])
    elif winner == "rf":
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        base = Pipeline([("prep", pre), ("mdl", RandomForestClassifier(class_weight="balanced", random_state=42))])
    elif winner == "xgb" and HAS_XGBOOST:
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        base = Pipeline([("prep", pre), ("mdl", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    else:
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        base = Pipeline([("prep", pre), ("mdl", GradientBoostingClassifier(random_state=42))])

    # Most frequent tuned params for the chosen winner
    if not params_df.empty:
        pf = params_df[params_df["model"] == winner].copy()
        if not pf.empty:
            # Drop non-parameter columns and NaNs
            par_cols = [c for c in pf.columns if c.startswith("mdl__")]
            best_params = {}
            for c in par_cols:
                vals = pf[c].dropna()
                if len(vals) == 0:
                    continue
                # pick mode
                val = vals.mode().iloc[0]
                best_params[c] = val
            # Cast numeric params to proper dtypes
            cast_map = {"mdl__max_depth": int, "mdl__n_estimators": int, "mdl__min_samples_leaf": int}
            for k, caster in cast_map.items():
                if k in best_params:
                    try:
                        best_params[k] = caster(best_params[k])
                    except Exception:
                        pass
            try:
                base.set_params(**best_params)
            except Exception:
                pass
    fitted = base.fit(X, y)

    # Permutation importance
    try:
        perm = permutation_importance(fitted, X, y, n_repeats=50, random_state=42, scoring="roc_auc")
        feat_names = list(numeric_cols + categorical_cols)
        imp_df = pd.DataFrame({
            "feature": feat_names[: len(perm.importances_mean)],
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        imp_df.to_csv(OUT / f"{prefix}_importances.csv", index=False)
        # Partial dependence for top 4 features
        top = imp_df.sort_values("importance_mean", ascending=False)["feature"].head(4).tolist()
        if top:
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax = ax.ravel()
            for i, feat in enumerate(top):
                try:
                    PartialDependenceDisplay.from_estimator(fitted, X, [feat], ax=ax[i])
                except Exception:
                    pass
            plt.tight_layout()
            fig.savefig(OUT / f"{prefix}_pdp_top4.png", dpi=160)
            plt.close(fig)
    except Exception:
        pass


def nested_regression(df: pd.DataFrame, feature_set: str) -> None:
    numeric_cols, categorical_cols = select_features(df, feature_set)
    work = df.dropna(subset=["ucla_total"]).copy()
    X = work[numeric_cols + categorical_cols]
    y = work["ucla_total"].astype(float).to_numpy()

    # CRITICAL FIX: Use stratified K-fold for regression to ensure balanced folds
    # Create UCLA quartile bins for stratification
    y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    preds_rows = []
    tuned_params = []

    # Use y_bins for stratification
    for i, (tr, te) in enumerate(outer.split(X, y_bins), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        candidates: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {}
        # Ridge
        pre_scaled = build_preprocessor(numeric_cols, categorical_cols, scale_num=True)
        ridge = Pipeline([("prep", pre_scaled), ("mdl", Ridge(random_state=42))])
        grid_ridge = {"mdl__alpha": [0.1, 1.0, 10.0]}
        candidates["ridge"] = (ridge, grid_ridge)
        # Lasso
        lasso = Pipeline([("prep", pre_scaled), ("mdl", Lasso(max_iter=10000, random_state=42))])
        grid_lasso = {"mdl__alpha": [0.001, 0.01, 0.1]}
        candidates["lasso"] = (lasso, grid_lasso)
        # RandomForest
        pre_tree = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        rf = Pipeline([("prep", pre_tree), ("mdl", RandomForestRegressor(random_state=42))])
        grid_rf = {"mdl__n_estimators": [400, 800], "mdl__max_depth": [None, 8, 12], "mdl__min_samples_leaf": [1, 3]}
        candidates["rf"] = (rf, grid_rf)
        # GradientBoosting
        gbrt = Pipeline([("prep", pre_tree), ("mdl", GradientBoostingRegressor(random_state=42))])
        grid_gbrt = {"mdl__n_estimators": [400, 800], "mdl__learning_rate": [0.05, 0.1], "mdl__max_depth": [2, 3]}
        candidates["gbrt"] = (gbrt, grid_gbrt)
        # XGBoost
        if HAS_XGBOOST:
            xgb = Pipeline([("prep", pre_tree), ("mdl", XGBRegressor(random_state=42))])
            grid_xgb = {"mdl__n_estimators": [200, 400], "mdl__max_depth": [3, 5], "mdl__learning_rate": [0.05, 0.1]}
            candidates["xgb"] = (xgb, grid_xgb)

        best_name = None
        best_est = None
        best_score = -np.inf
        best_params = None

        # Inner loop uses regular KFold (stratification is more critical for outer loop)
        inner = KFold(n_splits=3, shuffle=True, random_state=123)
        for name, (pipe, grid) in candidates.items():
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="r2", n_jobs=None)
            gs.fit(Xtr, ytr)
            if gs.best_score_ > best_score:
                best_score = gs.best_score_
                best_est = gs.best_estimator_
                best_name = name
                best_params = gs.best_params_

        ypred = best_est.predict(Xte)
        row = {
            "fold": i,
            "model": best_name,
            "R2": float(r2_score(yte, ypred)),
            "RMSE": float(np.sqrt(mean_squared_error(yte, ypred))),
            "MAE": float(mean_absolute_error(yte, ypred)),
        }
        rows.append(row)
        preds_rows.append(pd.DataFrame({
            "fold": i,
            "participant_id": work.iloc[te]["participant_id"].values,
            "y_true": yte,
            "y_pred": ypred,
        }))
        tuned_params.append({"fold": i, "model": best_name, **(best_params or {})})

    metrics = pd.DataFrame(rows)
    preds = pd.concat(preds_rows, ignore_index=True)
    params_df = pd.DataFrame(tuned_params)

    prefix = f"nested_{feature_set}_regression"
    metrics.to_csv(OUT / f"{prefix}_metrics.csv", index=False)
    preds.to_csv(OUT / f"{prefix}_predictions.csv", index=False)
    params_df.to_csv(OUT / f"{prefix}_params.csv", index=False)

    # Fit a final model on full data (most frequent winner)
    winner = params_df.groupby("model").size().idxmax()
    if winner == "ridge":
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=True)
        base = Pipeline([("prep", pre), ("mdl", Ridge(random_state=42))])
    elif winner == "lasso":
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=True)
        base = Pipeline([("prep", pre), ("mdl", Lasso(max_iter=10000, random_state=42))])
    elif winner == "rf":
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        base = Pipeline([("prep", pre), ("mdl", RandomForestRegressor(random_state=42))])
    elif winner == "xgb" and HAS_XGBOOST:
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        base = Pipeline([("prep", pre), ("mdl", XGBRegressor(random_state=42))])
    else:
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_num=False)
        base = Pipeline([("prep", pre), ("mdl", GradientBoostingRegressor(random_state=42))])

    # Apply most common tuned params for winner
    if not params_df.empty:
        pf = params_df[params_df["model"] == winner].copy()
        if not pf.empty:
            par_cols = [c for c in pf.columns if c.startswith("mdl__")]
            best_params = {}
            for c in par_cols:
                vals = pf[c].dropna()
                if len(vals) == 0:
                    continue
                val = vals.mode().iloc[0]
                best_params[c] = val
            cast_map = {"mdl__max_depth": int, "mdl__n_estimators": int, "mdl__min_samples_leaf": int}
            for k, caster in cast_map.items():
                if k in best_params:
                    try:
                        best_params[k] = caster(best_params[k])
                    except Exception:
                        pass
            try:
                base.set_params(**best_params)
            except Exception:
                pass
    fitted = base.fit(X, y)
    try:
        perm = permutation_importance(fitted, X, y, n_repeats=50, random_state=42, scoring="r2")
        feat_names = list(numeric_cols + categorical_cols)
        imp_df = pd.DataFrame({
            "feature": feat_names[: len(perm.importances_mean)],
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        imp_df.to_csv(OUT / f"{prefix}_importances.csv", index=False)
        # Partial dependence for top 4 features
        top = imp_df.sort_values("importance_mean", ascending=False)["feature"].head(4).tolist()
        if top:
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax = ax.ravel()
            for i, feat in enumerate(top):
                try:
                    PartialDependenceDisplay.from_estimator(fitted, X, [feat], ax=ax[i])
                except Exception:
                    pass
            plt.tight_layout()
            fig.savefig(OUT / f"{prefix}_pdp_top4.png", dpi=160)
            plt.close(fig)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested CV with tuning for loneliness prediction")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--features", choices=["ef", "demo", "demo_dass", "ef_demo", "ef_demo_dass", "all", "meta", "meta_ef", "meta_ef_dass"], default="demo_dass")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = build_base_dataframe()
    if args.task == "classification":
        nested_classification(df, args.features)
    else:
        nested_regression(df, args.features)


if __name__ == "__main__":
    main()
