"""
Recursive Feature Elimination (RFE) with cross-validation
--------------------------------------------------------
Ranks predictors (UCLA, DASS, age, gender) for EF outcomes using
Ridge and LinearSVR estimators within repeated K-fold CV.

Outputs (results/analysis_outputs/):
- rfe_selection_frequencies.csv   (per outcome/estimator, selection rate)
- rfe_cv_scores.csv               (CV R^2/MAE per estimator/outcome)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

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
    cols = ["participant_id"] + FEATURES + list(OUTCOMES.keys())
    df = df[cols].copy()
    # One-hot encode gender (male vs baseline)
    df["gender"] = df["gender"].fillna("female").astype(str)
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)
    return df


def run_rfe_for_outcome(df: pd.DataFrame, outcome: str, label: str) -> Dict[str, pd.DataFrame]:
    sub = df.dropna(
        subset=[outcome] + [c for c in df.columns if c.startswith("z_") or c == "age" or c.startswith("gender_")]
    ).copy()
    if len(sub) < 40:
        return {}

    y = sub[outcome].to_numpy()
    feat_cols = [c for c in sub.columns if c.startswith("z_") or c == "age" or c.startswith("gender_")]
    X = sub[feat_cols].to_numpy()

    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    estimators = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "LinearSVR": LinearSVR(random_state=42, loss="epsilon_insensitive", dual=True)
    }

    sel_rows = []
    score_rows = []
    for est_name, base_est in estimators.items():
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("est", base_est),
        ])
        rfe = RFECV(
            estimator=pipe.named_steps["est"],
            step=1,
            min_features_to_select=1,
            cv=rkf,
            scoring=make_scorer(r2_score),
            n_jobs=None,
        )
        # Fit RFE on standardized features so underlying estimator has coef_
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)
        rfe.fit(Xs, y)
        # RFECV with a pipeline sets support_ on transformed model; features correspond to feat_cols
        support = rfe.support_
        ranking = rfe.ranking_
        # Selection frequency approx: since RFECV used CV internally, support_ reflects best subset;
        # expose relative importance via ranking; we also compute external CV scores quickly
        for name, rank, keep in zip(feat_cols, ranking, support):
            sel_rows.append({
                "outcome": label,
                "estimator": est_name,
                "feature": name,
                "selected": bool(keep),
                "rank": int(rank),
                "n": int(len(sub)),
            })

        # External CV scores using same splits
        r2_scores: List[float] = []
        mae_scores: List[float] = []
        # Refit on each split with selected features only
        keep_idx = np.where(support)[0]
        for tr, te in rkf.split(X):
            Xt, Xv = Xs[tr][:, keep_idx], Xs[te][:, keep_idx]
            yt, yv = y[tr], y[te]
            base_est.fit(Xt, yt)
            yhat = base_est.predict(Xv)
            r2_scores.append(r2_score(yv, yhat))
            mae_scores.append(mean_absolute_error(yv, yhat))
        score_rows.append({
            "outcome": label,
            "estimator": est_name,
            "k_selected": int(len(keep_idx)),
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores, ddof=1)),
            "mae_mean": float(np.mean(mae_scores)),
            "mae_std": float(np.std(mae_scores, ddof=1)),
        })

    return {
        "sel": pd.DataFrame(sel_rows),
        "scores": pd.DataFrame(score_rows),
    }


def main():
    df = build_design()
    all_sel = []
    all_scores = []
    for key, label in OUTCOMES.items():
        res = run_rfe_for_outcome(df, key, label)
        if not res:
            continue
        all_sel.append(res["sel"]) 
        all_scores.append(res["scores"]) 
    if all_sel:
        sel_df = pd.concat(all_sel, ignore_index=True)
        # Summarize selection by feature (selected True vs False) per outcome/estimator
        freq = (
            sel_df.groupby(["outcome", "estimator", "feature"]) ["selected"].mean().reset_index()
            .rename(columns={"selected": "selection_rate"})
            .sort_values(["outcome", "estimator", "selection_rate"], ascending=[True, True, False])
        )
        freq.to_csv(OUT / "rfe_selection_frequencies.csv", index=False)
        sel_df.to_csv(OUT / "rfe_selection_raw.csv", index=False)
    if all_scores:
        score_df = pd.concat(all_scores, ignore_index=True)
        score_df.to_csv(OUT / "rfe_cv_scores.csv", index=False)
    print("Saved RFE selection frequencies and CV scores to analysis_outputs/")


if __name__ == "__main__":
    main()
