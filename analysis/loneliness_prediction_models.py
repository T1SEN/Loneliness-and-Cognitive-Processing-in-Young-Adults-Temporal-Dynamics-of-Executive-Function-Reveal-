"""
Dual-track loneliness prediction pipeline (regression + classification + RF).

This script operationalizes the "interpretability-first" plan:
1. Assemble a compact, theory-driven predictor set from exported CSVs.
2. Fit nested-cross-validated Elastic Net regression models for UCLA totals.
3. Fit a logistic LASSO that distinguishes high vs low loneliness tertiles.
4. Train a shallow random-forest regressor only for permutation importance.

Outputs are written to results/analysis_outputs/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader_utils import normalize_gender_value


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


MASTER_EXPANDED = OUTPUT_DIR / "master_expanded_metrics.csv"
LEARNING_SLOPES = OUTPUT_DIR / "learning_curves" / "individual_learning_slopes.csv"

EF_FEATURES = [
    "wcst_pe_rate",
    "late_minus_early_pe",
    "wcst_rt_cv",
    "wcst_accuracy",
    "stroop_interference",
    "prp_bottleneck",
    "prp_t1_slowing",
]

DASS_FEATURES = ["dass_depression", "dass_stress", "dass_anxiety"]
DEMO_FEATURES = ["gender_male", "age"]
MAIN_FEATURES = EF_FEATURES + DASS_FEATURES + DEMO_FEATURES
EF_ONLY_FEATURES = EF_FEATURES + DEMO_FEATURES

INTERACTION_SPECS = {
    "gender_x_wcst_pe": ("gender_male", "wcst_pe_rate"),
    "gender_x_late_minus_early": ("gender_male", "late_minus_early_pe"),
    "gender_x_wcst_rt_cv": ("gender_male", "wcst_rt_cv"),
}


@dataclass
class FoldMetrics:
    fold: int
    r2: float
    rmse: float
    mae: float
    best_alpha: float
    best_l1_ratio: float

def load_analysis_dataframe() -> pd.DataFrame:
    base = pd.read_csv(MASTER_EXPANDED).rename(columns={"pe_rate": "wcst_pe_rate"})
    if not LEARNING_SLOPES.exists():
        raise FileNotFoundError("WCST learning slopes CSV was not found.")
    slopes = pd.read_csv(LEARNING_SLOPES)[
        ["participant_id", "gender", "age", "pe_change"]
    ].rename(columns={"pe_change": "late_minus_early_pe"})
    df = base.merge(slopes, on="participant_id", how="left")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender_clean"] = df["gender"].apply(normalize_gender_value)
    df["gender_male"] = df["gender_clean"].map({"male": 1, "female": 0})
    required = ["ucla_total", "age", "gender_male", "late_minus_early_pe"]
    df = df.dropna(subset=required)
    for new_col, (left, right) in INTERACTION_SPECS.items():
        df[new_col] = df[left] * df[right]
    return df


def build_design_matrix(df: pd.DataFrame, include_dass: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    base = EF_ONLY_FEATURES.copy()
    if include_dass:
        base += [col for col in DASS_FEATURES if col in df.columns]
    feature_cols = base + list(INTERACTION_SPECS.keys())
    design = df[feature_cols].copy()
    return design, feature_cols


def _recover_coefs(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    coef_scaled = model.coef_.ravel()
    scale = getattr(scaler, "scale_", np.ones_like(coef_scaled))
    mean = getattr(scaler, "mean_", np.zeros_like(coef_scaled))
    coef = coef_scaled / scale
    intercept_val = model.intercept_ - np.sum(coef_scaled * mean / scale)
    intercept = float(np.squeeze(intercept_val))
    rows = [{"feature": name, "coefficient": val} for name, val in zip(feature_names, coef)]
    rows.append({"feature": "intercept", "coefficient": intercept})
    return pd.DataFrame(rows)


def run_nested_elastic_net_regression(
    X: pd.DataFrame,
    y: pd.Series,
    residualizer_covariates: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    outer = KFold(n_splits=5, shuffle=True, random_state=42)
    inner = KFold(n_splits=5, shuffle=True, random_state=131)
    param_grid = {
        "model__alpha": np.logspace(-3, 0.5, 10),
        "model__l1_ratio": [1.0, 0.75, 0.5],
    }
    fold_metrics: List[FoldMetrics] = []
    coef_rows: List[Dict[str, object]] = []
    preds = []
    truths = []
    fold_ids = []

    if residualizer_covariates is not None:
        residualizer_covariates = residualizer_covariates.loc[X.index]

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if residualizer_covariates is not None:
            cov_train = residualizer_covariates.iloc[train_idx]
            cov_test = residualizer_covariates.iloc[test_idx]
            resid_model = LinearRegression()
            resid_model.fit(cov_train, y_train)
            y_train = y_train - resid_model.predict(cov_train)
            y_test = y_test - resid_model.predict(cov_test)

        y_train_arr = np.asarray(y_train, dtype=float)
        y_test_arr = np.asarray(y_test, dtype=float)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=10000, random_state=42)),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        search.fit(X_train, y_train_arr)
        best_model: Pipeline = search.best_estimator_
        y_pred = best_model.predict(X_test)
        preds.append(y_pred)
        truths.append(y_test_arr)
        fold_ids.extend([fold_idx] * len(y_test_arr))
        fold_metrics.append(
            FoldMetrics(
                fold=fold_idx,
                r2=r2_score(y_test_arr, y_pred),
                rmse=np.sqrt(mean_squared_error(y_test_arr, y_pred)),
                mae=mean_absolute_error(y_test_arr, y_pred),
                best_alpha=search.best_params_["model__alpha"],
                best_l1_ratio=search.best_params_["model__l1_ratio"],
            )
        )
        coef = best_model.named_steps["model"].coef_
        nonzero = np.where(np.abs(coef) > 1e-6)[0]
        for idx in nonzero:
            coef_rows.append(
                {
                    "fold": fold_idx,
                    "feature": X.columns[idx],
                    "coefficient": coef[idx],
                }
            )

    oof_pred = np.concatenate(preds)
    oof_truth = np.concatenate(truths)
    overall_metrics = {
        "r2": r2_score(oof_truth, oof_pred),
        "rmse": np.sqrt(mean_squared_error(oof_truth, oof_pred)),
        "mae": mean_absolute_error(oof_truth, oof_pred),
    }

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=10000, random_state=42)),
        ]
    )
    final_search = GridSearchCV(
        final_pipeline,
        param_grid,
        cv=inner,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    if residualizer_covariates is not None:
        final_residualizer = LinearRegression()
        final_residualizer.fit(residualizer_covariates, y)
        y_final = (y - final_residualizer.predict(residualizer_covariates)).values
        residualizer_info = {
            "features": list(residualizer_covariates.columns),
            "coef": final_residualizer.coef_.tolist(),
            "intercept": float(final_residualizer.intercept_),
        }
    else:
        final_residualizer = None
        y_final = y.values
        residualizer_info = None

    final_search.fit(X, y_final)
    final_model: Pipeline = final_search.best_estimator_
    coef_df = _recover_coefs(final_model, list(X.columns))

    return {
        "fold_metrics": fold_metrics,
        "coef_rows": coef_rows,
        "overall_metrics": overall_metrics,
        "final_model": final_model,
        "coef_df": coef_df,
        "final_params": final_search.best_params_,
        "residualizer_info": residualizer_info,
    }


def create_high_low_labels(df: pd.DataFrame) -> pd.Series:
    lower = df["ucla_total"].quantile(1 / 3)
    upper = df["ucla_total"].quantile(2 / 3)
    labels = np.where(
        df["ucla_total"] <= lower,
        0,
        np.where(df["ucla_total"] >= upper, 1, np.nan),
    )
    return pd.Series(labels, index=df.index)


def run_logistic_lasso(X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=52)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=91)
    param_grid = {"model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5]}
    fold_metrics = []
    coef_rows = []
    preds = []
    true_vals = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner,
            scoring="roc_auc",
            n_jobs=1,
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        best_model: Pipeline = search.best_estimator_
        proba = best_model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        preds.append(proba)
        true_vals.append(y.iloc[test_idx].values)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "auc": roc_auc_score(y.iloc[test_idx], proba),
                "accuracy": accuracy_score(y.iloc[test_idx], pred),
                "f1": f1_score(y.iloc[test_idx], pred),
                "best_C": search.best_params_["model__C"],
            }
        )
        coef = best_model.named_steps["model"].coef_.ravel()
        nonzero = np.where(np.abs(coef) > 1e-6)[0]
        for idx in nonzero:
            coef_rows.append(
                {
                    "fold": fold_idx,
                    "feature": X.columns[idx],
                    "coefficient": coef[idx],
                }
            )

    oof_proba = np.concatenate(preds)
    oof_truth = np.concatenate(true_vals)
    oof_pred = (oof_proba >= 0.5).astype(int)
    summary = {
        "auc": roc_auc_score(oof_truth, oof_proba),
        "accuracy": accuracy_score(oof_truth, oof_pred),
        "f1": f1_score(oof_truth, oof_pred),
        "log_loss": log_loss(oof_truth, oof_proba),
    }

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    final_search = GridSearchCV(
        final_pipeline,
        param_grid,
        cv=inner,
        scoring="roc_auc",
        n_jobs=1,
    )
    final_search.fit(X, y)
    final_model: Pipeline = final_search.best_estimator_
    coef_df = _recover_coefs(final_model, list(X.columns))

    return {
        "fold_metrics": fold_metrics,
        "coef_rows": coef_rows,
        "summary": summary,
        "coef_df": coef_df,
        "final_params": final_search.best_params_,
    }


def run_random_forest_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    residualizer_covariates: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    def _build_rf() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            random_state=42,
            min_samples_leaf=4,
        )

    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    fold_rows = []
    perm_rows: List[pd.DataFrame] = []
    if residualizer_covariates is not None:
        residualizer_covariates = residualizer_covariates.loc[X.index]

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if residualizer_covariates is not None:
            cov_train = residualizer_covariates.iloc[train_idx]
            cov_test = residualizer_covariates.iloc[test_idx]
            resid_model = LinearRegression()
            resid_model.fit(cov_train, y_train)
            y_train = y_train - resid_model.predict(cov_train)
            y_test = y_test - resid_model.predict(cov_test)
        rf = _build_rf()
        rf.fit(X.iloc[train_idx], y_train)
        pred = rf.predict(X.iloc[test_idx])
        fold_rows.append(
            {
                "fold": fold_idx,
                "r2": r2_score(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "mae": mean_absolute_error(y_test, pred),
            }
        )
        perm = permutation_importance(
            rf,
            X.iloc[test_idx],
            y_test,
            n_repeats=100,
            random_state=11,
            n_jobs=1,
        )
        perm_rows.append(
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                    "fold": fold_idx,
                }
            )
        )

    perm_df = (
        pd.concat(perm_rows, ignore_index=True)
        .groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_std", "mean"),
            importance_fold_std=("importance_mean", "std"),
        )
        .fillna({"importance_fold_std": 0.0})
        .sort_values("importance_mean", ascending=False)
    )
    return {"fold_metrics": fold_rows, "perm_df": perm_df}


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {filename} ({len(df)} rows)")


def save_json(obj: Dict[str, object], filename: str) -> None:
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    print(f"Saved {filename}")


def main() -> None:
    print("Loading combined dataset...")
    df = load_analysis_dataframe()

    # ---------------------- Full model with DASS predictors ----------------------
    design_full, _ = build_design_matrix(df, include_dass=True)
    mask_full = design_full.notna().all(axis=1)
    if not mask_full.all():
        dropped = len(design_full) - int(mask_full.sum())
        print(f"  Dropping {dropped} rows with missing predictors (full model).")
    design_full = design_full.loc[mask_full]
    df_full = df.loc[mask_full].copy()
    y_reg = df_full["ucla_total"]
    print(f"Complete cases for regression (full): {len(design_full)}")

    print("\nRunning nested Elastic Net regression (with DASS)...")
    reg_results = run_nested_elastic_net_regression(design_full, y_reg)
    fold_df = pd.DataFrame([vars(f) for f in reg_results["fold_metrics"]])
    save_dataframe(fold_df, "loneliness_prediction_regression_folds.csv")
    save_dataframe(
        pd.DataFrame(reg_results["coef_rows"]),
        "loneliness_prediction_regression_fold_coefs.csv",
    )
    save_dataframe(
        reg_results["coef_df"], "loneliness_prediction_regression_final_coefs.csv"
    )
    save_json(
        {
            "overall": reg_results["overall_metrics"],
            "final_params": reg_results["final_params"],
            "residualizer": reg_results.get("residualizer_info"),
        },
        "loneliness_prediction_regression_summary.json",
    )

    print("\nRunning logistic LASSO (high vs low loneliness)...")
    labels = create_high_low_labels(df_full)
    mask = labels.notna()
    X_clf = design_full.loc[mask]
    y_clf = labels.loc[mask].astype(int)
    print(
        f"Classification sample: {len(X_clf)} (per class {y_clf.value_counts().to_dict()})"
    )
    clf_results = run_logistic_lasso(X_clf, y_clf)
    save_dataframe(
        pd.DataFrame(clf_results["fold_metrics"]),
        "loneliness_prediction_logistic_folds.csv",
    )
    save_dataframe(
        pd.DataFrame(clf_results["coef_rows"]),
        "loneliness_prediction_logistic_fold_coefs.csv",
    )
    save_dataframe(
        clf_results["coef_df"], "loneliness_prediction_logistic_final_coefs.csv"
    )
    save_json(
        {
            "summary": clf_results["summary"],
            "final_params": clf_results["final_params"],
        },
        "loneliness_prediction_logistic_summary.json",
    )

    print("\nRunning random-forest permutation importance (with DASS)...")
    rf_results = run_random_forest_analysis(design_full, y_reg)
    save_dataframe(
        pd.DataFrame(rf_results["fold_metrics"]),
        "loneliness_prediction_random_forest_folds.csv",
    )
    save_dataframe(
        rf_results["perm_df"],
        "loneliness_prediction_random_forest_importance.csv",
    )

    # ---------------- EF-only model on DASS-residualized loneliness -------------
    print("\nRunning EF-only models on DASS-residualized loneliness...")
    design_ef, _ = build_design_matrix(df, include_dass=False)
    mask_ef = design_ef.notna().all(axis=1)
    if not mask_ef.all():
        dropped = len(design_ef) - int(mask_ef.sum())
        print(f"  Dropping {dropped} rows with missing predictors (EF-only).")
    design_ef = design_ef.loc[mask_ef]
    df_ef = df.loc[mask_ef].copy()
    dass_cov = df_ef[[col for col in DASS_FEATURES if col in df_ef.columns]].copy()
    required_cols = [col for col in DASS_FEATURES if col in dass_cov.columns]
    if len(required_cols) < 2:
        raise ValueError("EF-only models require at least two DASS columns for residualization.")
    dass_cov = dass_cov[required_cols]
    resid_mask = dass_cov.notna().all(axis=1) & df_ef["ucla_total"].notna()
    design_ef = design_ef.loc[resid_mask]
    dass_cov = dass_cov.loc[resid_mask]
    y_ef = df_ef.loc[resid_mask, "ucla_total"]
    print(f"Complete cases for EF-only regression: {len(design_ef)}")

    ef_reg_results = run_nested_elastic_net_regression(
        design_ef,
        y_ef,
        residualizer_covariates=dass_cov,
    )
    save_dataframe(
        pd.DataFrame([vars(f) for f in ef_reg_results["fold_metrics"]]),
        "loneliness_prediction_resid_regression_folds.csv",
    )
    save_dataframe(
        pd.DataFrame(ef_reg_results["coef_rows"]),
        "loneliness_prediction_resid_regression_fold_coefs.csv",
    )
    save_dataframe(
        ef_reg_results["coef_df"],
        "loneliness_prediction_resid_regression_final_coefs.csv",
    )
    save_json(
        {
            "overall": ef_reg_results["overall_metrics"],
            "final_params": ef_reg_results["final_params"],
            "residualizer": ef_reg_results.get("residualizer_info"),
        },
        "loneliness_prediction_resid_regression_summary.json",
    )

    ef_rf_results = run_random_forest_analysis(
        design_ef,
        y_ef,
        residualizer_covariates=dass_cov,
    )
    save_dataframe(
        pd.DataFrame(ef_rf_results["fold_metrics"]),
        "loneliness_prediction_resid_random_forest_folds.csv",
    )
    save_dataframe(
        ef_rf_results["perm_df"],
        "loneliness_prediction_resid_random_forest_importance.csv",
    )

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
