"""
Tree/Ensemble models for EF outcomes
------------------------------------
Tries baseline linear and tree-ensemble regressors to predict EF outcomes
from UCLA, DASS, age, gender. Uses repeated K-fold CV to estimate
generalization (R^2, MAE) and permutation importance on held-out folds.

⚠️ WARNING: PREDICTION MODEL, NOT CAUSAL INFERENCE ⚠️
================================================================================
This is a MACHINE LEARNING prediction model, not a causal analysis.

Feature importance does NOT establish that UCLA has independent causal effects.
High UCLA importance may be due to:
  1. Collinearity with DASS (r ~ 0.5-0.7)
  2. Joint prediction without isolating unique variance
  3. Non-causal associations

For causal inference with proper DASS control, use:
  - master_dass_controlled_analysis.py (hierarchical regression)

This script is for EXPLORATORY PREDICTION ONLY.
================================================================================

Outputs (results/analysis_outputs/):
- tree_ensemble_cv_metrics.csv
- tree_ensemble_perm_importance_<outcome>.csv
- tree_ensemble_gb_importance_<outcome>.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe  # noqa: E402


BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


OUTCOMES = {
    "stroop_effect": "Stroop interference (ms)",
    "prp_bottleneck": "PRP bottleneck (ms)",
    "wcst_total_errors": "WCST total errors",
}

FEATURES = ["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]


def build_design() -> pd.DataFrame:
    df = build_analysis_dataframe()
    # Keep only needed columns and drop missing
    cols = ["participant_id"] + FEATURES + list(OUTCOMES.keys())
    df = df[cols].copy()
    # One-hot encode gender
    df["gender"] = df["gender"].fillna("female").astype(str)
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)
    return df


def cv_metrics(X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> List[Dict]:
    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    models = {
        "DummyMean": DummyRegressor(),
        "Ridge": Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=1.0, random_state=42))]),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }
    rows = []
    for name, model in models.items():
        r2 = cross_val_score(model, X, y, cv=rkf, scoring="r2", n_jobs=None)
        mae = -cross_val_score(model, X, y, cv=rkf, scoring="neg_mean_absolute_error", n_jobs=None)
        rows.append({
            "model": name,
            "r2_mean": float(np.mean(r2)),
            "r2_std": float(np.std(r2, ddof=1)),
            "mae_mean": float(np.mean(mae)),
            "mae_std": float(np.std(mae, ddof=1)),
        })
    return rows


def cv_permutation_importance(X: pd.DataFrame, y: pd.Series, feat_names: List[str]) -> pd.DataFrame:
    # Use GradientBoosting as a representative ensemble; compute test-set permutation importances across folds.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    importances = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_tr, y_tr)
        pi = permutation_importance(model, X_te, y_te, n_repeats=30, random_state=42)
        importances.append(pi.importances.T)  # shape: repeats x n_features
    # Concatenate and summarize
    all_imps = np.vstack(importances)  # (repeats*folds) x n_features
    imp_mean = np.mean(all_imps, axis=0)
    imp_std = np.std(all_imps, axis=0, ddof=1)
    df_imp = pd.DataFrame({
        "feature": feat_names,
        "perm_importance_mean": imp_mean,
        "perm_importance_sd": imp_std,
    }).sort_values("perm_importance_mean", ascending=False)
    return df_imp


def gb_feature_importance(X: pd.DataFrame, y: pd.Series, feat_names: List[str]) -> pd.DataFrame:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    imp = model.feature_importances_
    return pd.DataFrame({"feature": feat_names, "gb_feature_importance": imp}).sort_values("gb_feature_importance", ascending=False)


def main():
    df = build_design()
    metrics_rows = []
    for outcome_key, label in OUTCOMES.items():
        sub = df.dropna(subset=[outcome_key] + [c for c in df.columns if c.startswith("z_") or c.startswith("age") or c.startswith("gender_")])
        if len(sub) < 40:
            continue
        y = sub[outcome_key]
        # Build feature matrix (z_*, age, gender_*)
        feat_cols = [c for c in sub.columns if c.startswith("z_") or c == "age" or c.startswith("gender_")]
        X = sub[feat_cols]

        # Cross-validated metrics
        rows = cv_metrics(X, y)
        for r in rows:
            r.update({"outcome": label, "n": int(len(sub))})
        metrics_rows.extend(rows)

        # Permutation importance and GB feature importance
        pi_df = cv_permutation_importance(X, y, feat_cols)
        pi_df.to_csv(OUT / f"tree_ensemble_perm_importance_{outcome_key}.csv", index=False)
        gb_df = gb_feature_importance(X, y, feat_cols)
        gb_df.to_csv(OUT / f"tree_ensemble_gb_importance_{outcome_key}.csv", index=False)

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(OUT / "tree_ensemble_cv_metrics.csv", index=False)
        print("Saved tree ensemble CV metrics and importances to analysis_outputs/")
    else:
        print("Not enough data to run tree ensemble models.")


if __name__ == "__main__":
    main()

