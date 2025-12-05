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
from analysis.preprocessing import load_master_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT = RESULTS_DIR / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_dataset() -> pd.DataFrame:
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    def _pick(cols):
        for c in cols:
            if c in master.columns:
                return master[c]
        return pd.Series(np.nan, index=master.index)

    df = pd.DataFrame({
        "participant_id": master["participant_id"],
        "dass_dep": _to_num(master.get("dass_depression")),
        "dass_anx": _to_num(master.get("dass_anxiety")),
        "dass_stress": _to_num(master.get("dass_stress")),
    })

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

    df["wcst_total_errors"] = _to_num(_pick(["wcst_total_errors", "totalerrorcount", "total_error_count"]))
    df["wcst_persev_errors"] = _to_num(_pick(["wcst_persev_errors", "perseverativeerrorcount", "perseverative_error_count"]))
    df["wcst_nonpersev_errors"] = _to_num(_pick(["wcst_nonpersev_errors", "nonperseverativeerrorcount", "non_perseverative_error_count"]))
    df["wcst_conceptual_pct"] = _to_num(_pick(["wcst_conceptual_pct", "conceptuallevelresponsespercent", "conceptual_level_responses_percent"]))
    df["wcst_persev_resp_pct"] = _to_num(_pick(["wcst_persev_resp_pct", "perseverativeresponsespercent", "perseverative_responses_percent"]))
    df["wcst_failure_to_maintain_set"] = _to_num(_pick(["wcst_failure_to_maintain_set", "failuretomaintainset", "failure_to_maintain_set"]))

    # Advanced EF (optional)
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
