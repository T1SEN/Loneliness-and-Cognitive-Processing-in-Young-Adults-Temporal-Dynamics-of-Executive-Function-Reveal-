"""
EF -> DASS subscales prediction (trend check)

Build EF features from Stroop/PRP/WCST (plus DDM params if present) and test
whether they predict DASS-21 subscales: Depression (D), Anxiety (A), Stress (S).
We run light ML (Ridge/Lasso/RF/GBRT) with 5-fold CV and report R2/RMSE/MAE.

Outputs:
  - results/analysis_outputs/ef_predict_dass_metrics.csv
  - results/analysis_outputs/ef_predict_dass_importance_<target>.csv
  - results/analysis_outputs/ef_predict_dass_ols_<target>.csv (simple OLS with 3 EF)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT = RESULTS_DIR / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_dataset() -> pd.DataFrame:
    surveys = _read_csv(RESULTS_DIR / "2_surveys_results.csv")
    summary = _read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv")
    surveys.columns = surveys.columns.str.lower()
    summary.columns = summary.columns.str.lower()

    dass = (
        surveys[surveys["surveyname"] == "dass"][
            ["participantid", "score_d", "score_a", "score_s"]
        ]
        .rename(columns={"participantid": "participant_id", "score_d": "dass_dep", "score_a": "dass_anx", "score_s": "dass_stress"})
        .assign(dass_dep=lambda d: _to_num(d["dass_dep"]), dass_anx=lambda d: _to_num(d["dass_anx"]), dass_stress=lambda d: _to_num(d["dass_stress"]))
    )

    prp = summary[summary["testname"] == "prp"].copy()
    for c in ["rt2_soa_50", "rt2_soa_1200", "mrt_t1", "mrt_t2", "acc_t1", "acc_t2"]:
        prp[c] = _to_num(prp[c])
    prp = prp.assign(
        prp_bottleneck=lambda d: d["rt2_soa_50"] - d["rt2_soa_1200"],
        prp_rt_slope=lambda d: (d["rt2_soa_50"] - d["rt2_soa_1200"]) / 1150.0,
    )[["participantid", "prp_bottleneck", "prp_rt_slope", "mrt_t1", "mrt_t2", "acc_t1", "acc_t2"]].rename(
        columns={
            "participantid": "participant_id",
            "mrt_t1": "prp_mrt_t1",
            "mrt_t2": "prp_mrt_t2",
            "acc_t1": "prp_acc_t1",
            "acc_t2": "prp_acc_t2",
        }
    )

    stroop = summary[summary["testname"] == "stroop"][
        ["participantid", "stroop_effect", "accuracy", "mrt_incong", "mrt_cong", "mrt_total"]
    ].rename(columns={"participantid": "participant_id", "accuracy": "stroop_accuracy"})
    for c in ["stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong", "mrt_total"]:
        stroop[c] = _to_num(stroop[c])
    stroop["stroop_ratio_incong_cong"] = stroop["mrt_incong"] / stroop["mrt_cong"]

    wcst = summary[summary["testname"] == "wcst"][
        [
            "participantid",
            "totalerrorcount",
            "perseverativeerrorcount",
            "nonperseverativeerrorcount",
            "conceptuallevelresponsespercent",
            "perseverativeresponsespercent",
            "failuretomaintainset",
        ]
    ].rename(
        columns={
            "participantid": "participant_id",
            "totalerrorcount": "wcst_total_errors",
            "perseverativeerrorcount": "wcst_persev_errors",
            "nonperseverativeerrorcount": "wcst_nonpersev_errors",
            "conceptuallevelresponsespercent": "wcst_conceptual_pct",
            "perseverativeresponsespercent": "wcst_persev_resp_pct",
            "failuretomaintainset": "wcst_failure_to_maintain_set",
        }
    )
    for c in wcst.columns:
        if c != "participant_id":
            wcst[c] = _to_num(wcst[c])

    # Advanced EF (optional)
    ddm_stroop_path = OUT / "stroop_ddm_parameters.csv"
    ddm_prp_path = OUT / "prp_bottleneck_parameters.csv"
    wcst_switch_path = OUT / "wcst_switching_parameters.csv"
    ddm_stroop = pd.read_csv(ddm_stroop_path) if ddm_stroop_path.exists() else pd.DataFrame()
    ddm_prp = pd.read_csv(ddm_prp_path) if ddm_prp_path.exists() else pd.DataFrame()
    wcst_adv = pd.read_csv(wcst_switch_path) if wcst_switch_path.exists() else pd.DataFrame()

    df = dass.merge(stroop, on="participant_id", how="left")
    df = df.merge(prp, on="participant_id", how="left")
    df = df.merge(wcst, on="participant_id", how="left")
    if not ddm_stroop.empty:
        df = df.merge(ddm_stroop, on="participant_id", how="left", suffixes=(None, "_ddm"))
    if not ddm_prp.empty:
        df = df.merge(ddm_prp, on="participant_id", how="left")
    if not wcst_adv.empty:
        df = df.merge(wcst_adv, on="participant_id", how="left")
    return df


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    ef_basic = [
        "stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong", "mrt_total", "stroop_ratio_incong_cong",
        "prp_bottleneck", "prp_rt_slope", "prp_mrt_t1", "prp_mrt_t2", "prp_acc_t1", "prp_acc_t2",
        "wcst_total_errors", "wcst_persev_errors", "wcst_nonpersev_errors", "wcst_conceptual_pct", "wcst_persev_resp_pct", "wcst_failure_to_maintain_set",
    ]
    ef_ddm = [
        "drift_congruent", "drift_incongruent", "drift_neutral", "stroop_interference", "pop_interference_mean", "pop_interference_sd",
        "prp_bottleneck_effect", "pop_prp_mean", "pop_prp_sd",
        "wcst_accuracy", "wcst_persev_rate", "wcst_nonpersev_rate", "wcst_total_error_rate", "wcst_switch_cost_ms",
    ]
    ef_ddm = [c for c in ef_ddm if c in df.columns]
    ef_all = [c for c in ef_basic if c in df.columns] + ef_ddm
    return {"ef_basic": [c for c in ef_basic if c in df.columns], "ef_extended": ef_all}


def eval_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def run_ml(df: pd.DataFrame) -> None:
    feats = feature_sets(df)
    targets = ["dass_dep", "dass_anx", "dass_stress"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics_rows = []

    for target in targets:
        dfy = df.dropna(subset=[target]).copy()
        y = dfy[target].astype(float).values
        for fset_name, cols in feats.items():
            if not cols:
                continue
            X = dfy[cols]
            models = {
                "ridge": Pipeline([("prep", ColumnTransformer([
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols)
                ])), ("mdl", Ridge(alpha=1.0, random_state=42))]),
                "lasso": Pipeline([("prep", ColumnTransformer([
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols)
                ])), ("mdl", Lasso(alpha=0.01, max_iter=10000, random_state=42))]),
                "rf": Pipeline([("prep", ColumnTransformer([
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), cols)
                ])), ("mdl", RandomForestRegressor(n_estimators=500, random_state=42))]),
                "gbrt": Pipeline([("prep", ColumnTransformer([
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), cols)
                ])), ("mdl", GradientBoostingRegressor(random_state=42))]),
            }

            for mname, pipe in models.items():
                y_pred = cross_val_predict(pipe, X, y, cv=cv)
                m = eval_reg(y, y_pred)
                m.update({"target": target, "feature_set": fset_name, "model": mname, "n": int(len(y))})
                metrics_rows.append(m)

            # Fit once to all and dump permutation importances for RF (as proxy)
            try:
                rf_pipe = models["rf"].fit(X, y)
                perm = permutation_importance(rf_pipe, X, y, n_repeats=30, random_state=42, scoring="r2")
                imp = pd.DataFrame({
                    "feature": cols[: len(perm.importances_mean)],
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                })
                imp.to_csv(OUT / f"ef_predict_dass_importance_{target}_{fset_name}.csv", index=False)
            except Exception:
                pass

        # Simple OLS with 3 core EF predictors (readability)
        core = ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]
        core = [c for c in core if c in dfy.columns]
        if len(core) >= 2:
            ols_df = dfy[[target] + core].dropna()
            if len(ols_df) >= 30:
                formula = f"{target} ~ " + " + ".join(core)
                mod = smf.ols(formula, data=ols_df).fit(cov_type="HC3")
                tab = mod.summary2().tables[1]
                # Harmonize columns
                out = tab.rename(columns={"Coef.": "estimate", "Std.Err.": "std_error", "[0.025": "conf_low", "0.975]": "conf_high", "P>|t|": "p_value"})
                out.to_csv(OUT / f"ef_predict_dass_ols_{target}.csv", index=True)

    pd.DataFrame(metrics_rows).to_csv(OUT / "ef_predict_dass_metrics.csv", index=False)


def main() -> None:
    df = build_dataset()
    run_ml(df)
    print("Saved metrics and importances for EF->DASS trend check.")


if __name__ == "__main__":
    main()

