"""
SHAP Analysis for UCLA Loneliness Prediction

Provides interpretable explanations for ML model predictions using SHAP values.
Supports both classification (high/low loneliness) and regression (UCLA score).

Features:
- SHAP summary plots (global feature importance)
- SHAP dependence plots (feature-target relationships)
- SHAP waterfall plots (individual predictions)

Usage:
    python -m analysis.ml.shap_analysis --task classification --features ef
    python -m analysis.ml.shap_analysis --task regression --features ef_demo_dass
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from analysis.preprocessing import load_master_dataset

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT = RESULTS_DIR / "analysis_outputs" / "shap"
OUT.mkdir(parents=True, exist_ok=True)


def build_base_dataframe() -> pd.DataFrame:
    """Load and prepare data for ML analysis."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def _pick(candidates):
        for c in candidates:
            if c in master.columns:
                return master[c]
        return pd.Series(np.nan, index=master.index)

    base = pd.DataFrame({
        "participant_id": master["participant_id"],
        "age": _to_num(master.get("age")),
        "gender": master.get("gender_normalized", master.get("gender")),
    })
    base["gender"] = base["gender"].fillna("").astype(str).str.strip().str.lower()
    base["gender_male"] = (base["gender"] == "male").astype(int)

    # UCLA target
    if "ucla_total" in master.columns:
        base["ucla_total"] = _to_num(master["ucla_total"])
    elif "ucla_score" in master.columns:
        base["ucla_total"] = _to_num(master["ucla_score"])

    # DASS
    base["dass_dep"] = _to_num(master.get("dass_depression"))
    base["dass_anx"] = _to_num(master.get("dass_anxiety"))
    base["dass_stress"] = _to_num(master.get("dass_stress"))

    # EF features - PRP
    base["prp_bottleneck"] = _to_num(master.get("prp_bottleneck"))
    if base["prp_bottleneck"].isna().all():
        short_col = _to_num(_pick(["rt2_soa_50", "rt2_soa_150"]))
        long_col = _to_num(_pick(["rt2_soa_1200"]))
        if not short_col.isna().all() and not long_col.isna().all():
            base["prp_bottleneck"] = short_col - long_col
    base["prp_mrt_t1"] = _to_num(_pick(["mrt_t1"]))
    base["prp_mrt_t2"] = _to_num(_pick(["mrt_t2"]))
    base["prp_acc_t1"] = _to_num(_pick(["acc_t1"]))
    base["prp_acc_t2"] = _to_num(_pick(["acc_t2"]))

    # EF features - Stroop
    base["stroop_effect"] = _to_num(_pick(["stroop_interference", "stroop_effect"]))
    base["stroop_accuracy"] = _to_num(_pick(["accuracy", "accuracy_stroop"]))
    base["mrt_incong"] = _to_num(_pick(["mrt_incong"]))
    base["mrt_cong"] = _to_num(_pick(["mrt_cong"]))
    if base["stroop_effect"].isna().all():
        incong = base["mrt_incong"]
        cong = base["mrt_cong"]
        if not incong.isna().all() and not cong.isna().all():
            base["stroop_effect"] = incong - cong

    # EF features - WCST
    base["wcst_total_errors"] = _to_num(_pick(["wcst_total_errors", "totalErrorCount"]))
    base["wcst_persev_errors"] = _to_num(_pick(["wcst_persev_errors", "perseverativeErrorCount"]))
    base["wcst_conceptual_pct"] = _to_num(_pick(["wcst_conceptual_pct", "conceptualLevelResponsesPercent"]))

    return base


def select_features(df: pd.DataFrame, feature_set: str) -> List[str]:
    """Select feature columns based on feature set name."""
    ef_cols = [
        "stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong",
        "prp_bottleneck", "prp_mrt_t1", "prp_mrt_t2", "prp_acc_t1", "prp_acc_t2",
        "wcst_total_errors", "wcst_persev_errors", "wcst_conceptual_pct",
    ]
    demo_cols = ["age", "gender_male"]
    dass_cols = ["dass_dep", "dass_anx", "dass_stress"]

    fs = feature_set.lower()
    if fs == "ef":
        cols = ef_cols
    elif fs == "demo":
        cols = demo_cols
    elif fs == "demo_dass":
        cols = demo_cols + dass_cols
    elif fs == "ef_demo":
        cols = ef_cols + demo_cols
    elif fs in ("ef_demo_dass", "all"):
        cols = ef_cols + demo_cols + dass_cols
    else:
        cols = ef_cols  # default

    return [c for c in cols if c in df.columns]


def run_shap_classification(df: pd.DataFrame, feature_set: str) -> None:
    """Run SHAP analysis for classification task."""
    if not HAS_SHAP:
        print("SHAP not available")
        return

    feature_cols = select_features(df, feature_set)
    work = df.dropna(subset=["ucla_total"]).copy()

    # Create binary target
    thr = work["ucla_total"].quantile(0.75)
    work["y"] = (work["ucla_total"] >= thr).astype(int)

    X = work[feature_cols].copy()
    y = work["y"].values

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

    # Train model
    print(f"\nTraining classifier for SHAP analysis...")
    print(f"  Features: {feature_cols}")
    print(f"  N samples: {len(X_imp)}, N features: {len(feature_cols)}")

    # Use XGBoost if available, otherwise RandomForest
    if HAS_XGBOOST:
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_imp, y)
        explainer = shap.TreeExplainer(model)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X_imp, y)
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_imp)

    # For binary classification, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary plot
    print("Creating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_imp, show=False, max_display=15)
    plt.title(f"SHAP Summary - Classification ({feature_set})")
    plt.tight_layout()
    plt.savefig(OUT / f"shap_summary_classification_{feature_set}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_imp, plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP Feature Importance - Classification ({feature_set})")
    plt.tight_layout()
    plt.savefig(OUT / f"shap_bar_classification_{feature_set}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["participant_id"] = work["participant_id"].values
    shap_df.to_csv(OUT / f"shap_values_classification_{feature_set}.csv", index=False)

    # Feature importance summary
    importance = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": importance
    }).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(OUT / f"shap_importance_classification_{feature_set}.csv", index=False)

    print(f"\nTop 5 features by SHAP importance:")
    for _, row in imp_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Dependence plots for top 3 features
    print("Creating SHAP dependence plots...")
    top_features = imp_df.head(3)["feature"].tolist()
    for feat in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_imp, show=False)
        plt.title(f"SHAP Dependence - {feat}")
        plt.tight_layout()
        plt.savefig(OUT / f"shap_dep_{feat}_classification_{feature_set}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✓ SHAP classification analysis complete. Results saved to {OUT}")


def run_shap_regression(df: pd.DataFrame, feature_set: str) -> None:
    """Run SHAP analysis for regression task."""
    if not HAS_SHAP:
        print("SHAP not available")
        return

    feature_cols = select_features(df, feature_set)
    work = df.dropna(subset=["ucla_total"]).copy()

    X = work[feature_cols].copy()
    y = work["ucla_total"].values

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

    # Train model
    print(f"\nTraining regressor for SHAP analysis...")
    print(f"  Features: {feature_cols}")
    print(f"  N samples: {len(X_imp)}, N features: {len(feature_cols)}")

    if HAS_XGBOOST:
        model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_imp, y)
        explainer = shap.TreeExplainer(model)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X_imp, y)
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_imp)

    # Summary plot
    print("Creating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_imp, show=False, max_display=15)
    plt.title(f"SHAP Summary - Regression ({feature_set})")
    plt.tight_layout()
    plt.savefig(OUT / f"shap_summary_regression_{feature_set}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_imp, plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP Feature Importance - Regression ({feature_set})")
    plt.tight_layout()
    plt.savefig(OUT / f"shap_bar_regression_{feature_set}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["participant_id"] = work["participant_id"].values
    shap_df.to_csv(OUT / f"shap_values_regression_{feature_set}.csv", index=False)

    # Feature importance summary
    importance = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": importance
    }).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(OUT / f"shap_importance_regression_{feature_set}.csv", index=False)

    print(f"\nTop 5 features by SHAP importance:")
    for _, row in imp_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Dependence plots for top 3 features
    print("Creating SHAP dependence plots...")
    top_features = imp_df.head(3)["feature"].tolist()
    for feat in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_imp, show=False)
        plt.title(f"SHAP Dependence - {feat}")
        plt.tight_layout()
        plt.savefig(OUT / f"shap_dep_{feat}_regression_{feature_set}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✓ SHAP regression analysis complete. Results saved to {OUT}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP analysis for UCLA loneliness prediction")
    p.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    p.add_argument("--features", choices=["ef", "demo", "demo_dass", "ef_demo", "ef_demo_dass", "all"], default="ef")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("SHAP ANALYSIS FOR UCLA LONELINESS PREDICTION")
    print("=" * 70)

    df = build_base_dataframe()
    print(f"Loaded {len(df)} participants")

    if args.task in ("classification", "both"):
        run_shap_classification(df, args.features)

    if args.task in ("regression", "both"):
        run_shap_regression(df, args.features)

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
