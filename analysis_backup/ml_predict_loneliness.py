"""
Predicting Loneliness with ML (Young Adults)

This script builds ML models to predict loneliness among young adults using
executive function (EF) summaries (Stroop, PRP, WCST) with optional
demographics and DASS controls. It supports both:

- regression: predict continuous UCLA total (or z-scored)
- classification: predict high loneliness (top 25%) vs others

Features are assembled directly from CSV exports under `results/` to avoid
dependencies on other analysis scripts. Models are evaluated with cross-
validation and key metrics are exported alongside permutation importances.

Usage examples:
  - Regression (EF only, age 18–35):
      python analysis/ml_predict_loneliness.py --task regression --age-min 18 --age-max 35 --features ef

  - Classification (EF + demo + DASS):
      python analysis/ml_predict_loneliness.py --task classification --features ef_demo_dass

Outputs (results/analysis_outputs/):
  - ml_loneliness_metrics_<task>.csv
  - ml_loneliness_predictions_<task>.csv
  - ml_loneliness_feature_importance_<model>_<task>.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Data loading and featurization
# -------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_base_dataframe() -> pd.DataFrame:
    """Merge participants, surveys (UCLA/DASS), and EF summaries into one DataFrame.

    Columns produced (subset):
      - participant_id, age, gender
      - ucla_total, dass_dep, dass_anx, dass_stress
      - stroop_effect, prp_bottleneck, wcst_total_errors, wcst_persev_resp_pct
    """
    # Participants
    participants = _read_csv(RESULTS_DIR / "1_participants_info.csv")
    participants = participants.rename(columns={
        "participantId": "participant_id",
        "age": "age",
        "gender": "gender",
    })[["participant_id", "age", "gender"]]
    participants["age"] = _to_num(participants["age"])  # keep gender as string (Korean ok)

    # Surveys
    surveys = _read_csv(RESULTS_DIR / "2_surveys_results.csv")
    # UCLA
    ucla = (
        surveys.loc[surveys["surveyName"].str.lower() == "ucla", ["participantId", "score"]]
        .rename(columns={"participantId": "participant_id", "score": "ucla_total"})
    )
    ucla["ucla_total"] = _to_num(ucla["ucla_total"])  # numeric

    # DASS
    dass = (
        surveys.loc[surveys["surveyName"].str.lower() == "dass", ["participantId", "score_D", "score_A", "score_S"]]
        .rename(columns={
            "participantId": "participant_id",
            "score_D": "dass_dep",
            "score_A": "dass_anx",
            "score_S": "dass_stress",
        })
    )
    for c in ["dass_dep", "dass_anx", "dass_stress"]:
        dass[c] = _to_num(dass[c])

    # Cognitive tests summary
    cog = _read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv")
    # PRP
    prp = cog.loc[cog["testName"].str.lower() == "prp"].copy()
    prp = prp.rename(columns={
        "participantId": "participant_id",
        "rt2_soa_50": "prp_rt_short",
        "rt2_soa_1200": "prp_rt_long",
        "mrt_t1": "prp_mrt_t1",
        "mrt_t2": "prp_mrt_t2",
        "acc_t1": "prp_acc_t1",
        "acc_t2": "prp_acc_t2",
        "n_trials": "prp_n_trials",
    })
    for c in ["prp_rt_short", "prp_rt_long", "prp_mrt_t1", "prp_mrt_t2", "prp_acc_t1", "prp_acc_t2", "prp_n_trials"]:
        prp[c] = _to_num(prp[c])
    prp["prp_bottleneck"] = prp["prp_rt_short"] - prp["prp_rt_long"]
    prp_keep = [
        "participant_id",
        "prp_bottleneck",
        "prp_rt_short",
        "prp_rt_long",
        "prp_mrt_t1",
        "prp_mrt_t2",
        "prp_acc_t1",
        "prp_acc_t2",
        "prp_n_trials",
    ]
    prp = prp[prp_keep]

    # Stroop
    stroop = cog.loc[cog["testName"].str.lower() == "stroop", [
        "participantId", "stroop_effect", "accuracy", "mrt_incong", "mrt_cong", "mrt_total", "total"
    ]].rename(columns={
        "participantId": "participant_id",
        "accuracy": "stroop_accuracy",
        "total": "stroop_total_trials",
    })
    for c in ["stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong", "mrt_total", "stroop_total_trials"]:
        stroop[c] = _to_num(stroop[c])

    # WCST
    wcst = cog.loc[cog["testName"].str.lower() == "wcst", [
        "participantId",
        "totalErrorCount",
        "perseverativeErrorCount",
        "nonPerseverativeErrorCount",
        "completedCategories",
        "conceptualLevelResponsesPercent",
        "perseverativeResponsesPercent",
        "failureToMaintainSet",
    ]].rename(columns={
        "participantId": "participant_id",
        "totalErrorCount": "wcst_total_errors",
        "perseverativeErrorCount": "wcst_persev_errors",
        "nonPerseverativeErrorCount": "wcst_nonpersev_errors",
        "completedCategories": "wcst_completed_categories",
        "conceptualLevelResponsesPercent": "wcst_conceptual_pct",
        "perseverativeResponsesPercent": "wcst_persev_resp_pct",
        "failureToMaintainSet": "wcst_failure_to_maintain_set",
    })
    for c in wcst.columns:
        if c != "participant_id":
            wcst[c] = _to_num(wcst[c])

    # Merge all
    df = participants.merge(ucla, on="participant_id", how="left")
    df = df.merge(dass, on="participant_id", how="left")
    df = df.merge(prp, on="participant_id", how="left")
    df = df.merge(stroop, on="participant_id", how="left")
    df = df.merge(wcst, on="participant_id", how="left")

    # Targets
    def _z(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if x.dropna().std() in (0, None) or x.dropna().empty:
            return pd.Series(np.nan, index=x.index)
        return (x - x.mean()) / x.std()

    df["z_ucla"] = _z(df["ucla_total"])
    # High loneliness: top quartile
    if df["ucla_total"].notna().sum() > 0:
        thr = df["ucla_total"].quantile(0.75)
    else:
        thr = np.nan
    df["high_loneliness"] = (df["ucla_total"] >= thr).astype(float)
    df.attrs["loneliness_threshold"] = float(thr) if np.isfinite(thr) else thr

    return df


def select_feature_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) for the requested feature set."""
    ef_numeric = [
        "stroop_effect",
        "stroop_accuracy",
        "prp_bottleneck",
        "prp_mrt_t1",
        "prp_mrt_t2",
        "prp_acc_t1",
        "prp_acc_t2",
        "wcst_total_errors",
        "wcst_persev_errors",
        "wcst_nonpersev_errors",
        "wcst_conceptual_pct",
        "wcst_persev_resp_pct",
        "wcst_failure_to_maintain_set",
    ]
    demo_num = ["age"]
    demo_cat = ["gender"]
    dass_num = ["dass_dep", "dass_anx", "dass_stress"]

    feature_set = feature_set.lower()
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    if feature_set == "ef":
        numeric_cols = ef_numeric
    elif feature_set == "ef_demo":
        numeric_cols = ef_numeric + demo_num
        categorical_cols = demo_cat
    elif feature_set in ("ef_demo_dass", "all"):
        numeric_cols = ef_numeric + demo_num + dass_num
        categorical_cols = demo_cat
    elif feature_set == "demo_dass":
        numeric_cols = demo_num + dass_num
        categorical_cols = demo_cat
    elif feature_set == "demo":
        numeric_cols = demo_num
        categorical_cols = demo_cat
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    # Keep only columns that actually exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    return numeric_cols, categorical_cols


# -----------------
# Modeling helpers
# -----------------
@dataclass
class CVRun:
    name: str
    estimator: Pipeline


def build_models(task: str, numeric_cols: List[str], categorical_cols: List[str]) -> List[CVRun]:
    # Preprocessors
    numeric_impute_scale = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    numeric_impute_only = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_proc = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Column transformers
    preproc_scaled = ColumnTransformer([
        ("num", numeric_impute_scale, numeric_cols),
        ("cat", categorical_proc, categorical_cols),
    ])
    preproc_tree = ColumnTransformer([
        ("num", numeric_impute_only, numeric_cols),
        ("cat", categorical_proc, categorical_cols),
    ])

    models: List[CVRun] = []
    if task == "regression":
        models.append(CVRun("ridge", Pipeline([("prep", preproc_scaled), ("mdl", Ridge(alpha=1.0, random_state=42))])))
        models.append(CVRun("lasso", Pipeline([("prep", preproc_scaled), ("mdl", Lasso(alpha=0.01, random_state=42, max_iter=10000))])))
        models.append(CVRun("rf", Pipeline([("prep", preproc_tree), ("mdl", RandomForestRegressor(n_estimators=500, random_state=42))])))
        models.append(CVRun("gbrt", Pipeline([("prep", preproc_tree), ("mdl", GradientBoostingRegressor(random_state=42))])))
    else:
        # classification
        models.append(CVRun("logreg", Pipeline([("prep", preproc_scaled), ("mdl", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42))])))
        models.append(CVRun("rf", Pipeline([("prep", preproc_tree), ("mdl", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42))])))
        models.append(CVRun("gbrt", Pipeline([("prep", preproc_tree), ("mdl", GradientBoostingClassifier(random_state=42))])))
    return models


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred))
    metrics = {"Accuracy": acc, "F1": f1, "Precision": pre, "Recall": rec}
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
            metrics["ROC_AUC"] = auc
        except Exception:
            pass
        try:
            ap = float(average_precision_score(y_true, y_proba))
            metrics["PR_AUC"] = ap
        except Exception:
            pass
    return metrics


def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Find a probability threshold that maximizes F1 on given predictions."""
    if y_proba is None or len(np.unique(y_true)) < 2:
        return {}
    # Use PR curve thresholds
    pre, rec, thr = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns thresholds for all but first point
    thr = np.asarray(thr)
    if thr.size == 0:
        return {}
    # Compute F1 for each threshold index aligned to rec[1:], pre[1:]
    f1s = 2 * (pre[1:] * rec[1:]) / (pre[1:] + rec[1:] + 1e-12)
    i = int(np.nanargmax(f1s))
    best_thr = float(thr[i])
    y_pred_opt = (y_proba >= best_thr).astype(int)
    m = evaluate_classification(y_true, y_pred_opt, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt).ravel()
    m.update({
        "BestThreshold": best_thr,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    })
    return m


def run_cv_models(df: pd.DataFrame, task: str, feature_set: str, age_min: int | None, age_max: int | None) -> None:
    # Filter age range if requested
    work = df.copy()
    if age_min is not None:
        work = work[work["age"].fillna(-1) >= age_min]
    if age_max is not None:
        work = work[work["age"].fillna(10**9) <= age_max]

    # Assemble features and target
    numeric_cols, categorical_cols = select_feature_columns(work, feature_set)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No features available after selection.")

    if task == "regression":
        target_col = "ucla_total"  # could switch to z_ucla
        work = work.dropna(subset=[target_col])
        y = work[target_col].to_numpy(dtype=float)
        is_classification = False
    else:
        target_col = "high_loneliness"
        work = work.dropna(subset=[target_col])
        y = work[target_col].to_numpy(dtype=int)
        is_classification = True

    # Build X
    X = work[numeric_cols + categorical_cols]

    # Choose CV strategy
    if is_classification:
        # Ensure both classes present
        if len(np.unique(y)) < 2:
            raise ValueError("High-loneliness classification has only one class in data. Adjust threshold or filters.")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = build_models(task, numeric_cols, categorical_cols)
    metrics_rows = []
    metrics_rows_opt = []
    preds_rows = []

    for run in models:
        est = run.estimator
        # Get cross-validated predictions
        if is_classification:
            # Predicted labels and probabilities (if available)
            y_pred = cross_val_predict(est, X, y, cv=cv, n_jobs=None, method="predict")
            y_proba = None
            try:
                y_proba = cross_val_predict(est, X, y, cv=cv, n_jobs=None, method="predict_proba")[:, 1]
            except Exception:
                # Some models may not implement predict_proba
                pass
            m = evaluate_classification(y, y_pred, y_proba)
            metrics_rows.append({"model": run.name, **m, "n": int(len(y))})
            # Threshold optimization (best F1)
            if y_proba is not None:
                mopt = optimize_threshold(y, y_proba)
                if mopt:
                    metrics_rows_opt.append({"model": run.name, **mopt, "n": int(len(y))})
            out = pd.DataFrame({
                "participant_id": work["participant_id"].values,
                "y_true": y,
                "y_pred": y_pred,
            })
            if y_proba is not None:
                out["y_proba"] = y_proba
        else:
            y_pred = cross_val_predict(est, X, y, cv=cv, n_jobs=None, method="predict")
            m = evaluate_regression(y, y_pred)
            metrics_rows.append({"model": run.name, **m, "n": int(len(y))})
            out = pd.DataFrame({
                "participant_id": work["participant_id"].values,
                "y_true": y,
                "y_pred": y_pred,
            })

        out.insert(0, "model", run.name)
        preds_rows.append(out)

        # Fit on all data and compute permutation importance for interpretability
        try:
            fitted = est.fit(X, y)
            scoring = "roc_auc" if is_classification else "r2"
            perm = permutation_importance(fitted, X, y, n_repeats=50, random_state=42, scoring=scoring)
            # Get expanded feature names after preprocessing
            try:
                prep = fitted.named_steps.get("prep")
                feat_names: List[str] = []
                if prep is not None:
                    # numeric
                    feat_names.extend(numeric_cols)
                    # categorical one-hot
                    if any(tr[0] == "cat" for tr in prep.transformers_):
                        cats = [tr for tr in prep.transformers_ if tr[0] == "cat"][0][1]
                        encoder = cats.named_steps["onehot"]
                        cat_feature_names = []
                        for c in categorical_cols:
                            try:
                                cats_for_c = [f"{c}=" + str(v) for v in encoder.categories_[categorical_cols.index(c)]]
                            except Exception:
                                cats_for_c = [c]
                            cat_feature_names.extend(cats_for_c)
                        feat_names.extend(cat_feature_names)
                else:
                    feat_names = list(X.columns)
            except Exception:
                feat_names = list(X.columns)
            import_df = pd.DataFrame({
                "feature": feat_names[: len(perm.importances_mean)],
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            })
            import_df.insert(0, "model", run.name)
            import_df.insert(1, "task", task)
            import_path = OUTPUT_DIR / f"ml_loneliness_feature_importance_{run.name}_{task}.csv"
            import_df.to_csv(import_path, index=False)
        except Exception as e:
            # Importance is best-effort; continue if it fails
            print(f"[warn] Importance failed for {run.name}: {e}")

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(preds_rows, ignore_index=True)

    metrics_path = OUTPUT_DIR / f"ml_loneliness_metrics_{task}.csv"
    preds_path = OUTPUT_DIR / f"ml_loneliness_predictions_{task}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved metrics to {metrics_path.name}")
    print(f"Saved predictions to {preds_path.name}")
    if metrics_rows_opt:
        metrics_opt_df = pd.DataFrame(metrics_rows_opt)
        metrics_opt_path = OUTPUT_DIR / f"ml_loneliness_metrics_{task}_best_threshold.csv"
        metrics_opt_df.to_csv(metrics_opt_path, index=False)
        print(f"Saved best-threshold metrics to {metrics_opt_path.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML models to predict loneliness (young adults)")
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--features", choices=["ef", "ef_demo", "ef_demo_dass", "demo_dass", "demo", "all"], default="ef")
    p.add_argument("--age-min", type=int, default=18, help="Minimum age filter (inclusive). Use -1 to disable.")
    p.add_argument("--age-max", type=int, default=35, help="Maximum age filter (inclusive). Use -1 to disable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = build_base_dataframe()
    age_min = None if args.age_min is not None and args.age_min < 0 else args.age_min
    age_max = None if args.age_max is not None and args.age_max < 0 else args.age_max

    # Report sample and threshold
    thr = df.attrs.get("loneliness_threshold", np.nan)
    if isinstance(thr, (int, float)) and np.isfinite(thr):
        print(f"High-loneliness threshold (75th percentile UCLA): {thr:.2f}")
    else:
        print("High-loneliness threshold could not be computed (missing UCLA data).")

    run_cv_models(df, args.task, args.features, age_min, age_max)


if __name__ == "__main__":
    main()
