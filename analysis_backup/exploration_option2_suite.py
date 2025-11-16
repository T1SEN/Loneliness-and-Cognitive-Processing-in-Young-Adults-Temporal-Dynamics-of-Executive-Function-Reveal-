#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe, add_meta_control, zscore  # noqa: E402


BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def _trim_z(s: pd.Series, thr: float = 3.0) -> pd.Series:
    z = (s - s.mean(skipna=True)) / s.std(skipna=True)
    return s.where(z.abs() <= thr)


def multiverse(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = []
    rows = []
    out_defs = [
        ("stroop_effect", "z_ucla"),
        ("prp_bottleneck", "z_ucla"),
        ("wcst_total_errors", "z_ucla"),
    ]
    covar_sets = {
        "full": ["z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"],
        "nodemog": ["z_dass_dep", "z_dass_anx", "z_dass_stress"],
        "dass_total": ["dass_total", "age", "gender"],
    }
    outlier_ops = {"none": lambda s: s, "winsor1": lambda s: _winsorize(s, 0.01), "ztrim3": lambda s: _trim_z(s, 3.0)}
    ytx = {"raw": lambda s: s, "z": lambda s: zscore(pd.to_numeric(s, errors="coerce"))}

    spec_id = 0
    for y, x in out_defs:
        for cov_name, covs in covar_sets.items():
            for ol_name, ol_fn in outlier_ops.items():
                for ytf_name, ytf in ytx.items():
                    spec_id += 1
                    cols = [y, x] + covs
                    data = df[cols].copy()
                    data[y] = ytf(ol_fn(pd.to_numeric(data[y], errors="coerce")))
                    if cov_name == "dass_total" and "dass_total" not in data.columns:
                        data["dass_total"] = df[["dass_dep", "dass_anx", "dass_stress"]].sum(axis=1)
                        data["dass_total"] = zscore(data["dass_total"])  # z
                    data = data.dropna()
                    if len(data) < 40:
                        continue
                    formula = f"y ~ {x} + "
                    if cov_name == "dass_total":
                        formula += "dass_total"
                    else:
                        formula += "z_dass_dep + z_dass_anx + z_dass_stress"
                    if "age" in covs:
                        formula += " + age"
                    if "gender" in covs:
                        formula += " + C(gender)"
                    model = smf.ols(formula, data=data.rename(columns={y: "y"})).fit(cov_type="HC3")
                    est = model.params.get(x, np.nan)
                    se = model.bse.get(x, np.nan)
                    p = model.pvalues.get(x, np.nan)
                    sign = np.sign(est) if pd.notna(est) else 0
                    rows.append({
                        "spec_id": spec_id,
                        "outcome": y,
                        "predictor": x,
                        "covars": cov_name,
                        "outlier": ol_name,
                        "y_transform": ytf_name,
                        "n": int(model.nobs),
                        "estimate": est,
                        "std_error": se,
                        "p_value": p,
                        "sign": sign,
                    })
                    specs.append({
                        "spec_id": spec_id,
                        "formula": formula,
                    })
    res = pd.DataFrame(rows)
    sp = pd.DataFrame(specs)
    if not res.empty:
        from statsmodels.stats.multitest import multipletests
        res["q_value"] = multipletests(res["p_value"].astype(float).values, method="fdr_bh")[1]
        summ = res.groupby(["outcome"]).agg(
            n_specs=("spec_id", "count"),
            median_est=("estimate", "median"),
            pos_share=("sign", lambda s: float((s > 0).mean())),
            q_min=("q_value", "min"),
        ).reset_index()
        summ.to_csv(OUT / "multiverse_summary.csv", index=False)
    res.to_csv(OUT / "multiverse_specs.csv", index=False)
    return res, sp


def influence_and_robust(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]:
        cols = [y, "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]
        data = df[cols].dropna().rename(columns={y: "y"})
        if len(data) < 40:
            continue
        model = smf.ols("y ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)", data=data).fit(cov_type="HC3")
        infl = OLSInfluence(model)
        cook = infl.cooks_distance[0]
        dfb = infl.dfbetas
        # VIF for numeric covariates (encode gender as dummy)
        X = pd.get_dummies(data[["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]], drop_first=True)
        X = sm.add_constant(X, has_constant="add").astype(float)
        vif_vals = []
        for i in range(1, X.shape[1]):
            vif_vals.append(variance_inflation_factor(X.values, i))
        vif_mean = float(np.nanmean(vif_vals)) if len(vif_vals) else np.nan
        # Robust regression (HuberT)
        rX = sm.add_constant(pd.get_dummies(data[["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]], drop_first=True), has_constant="add").astype(float)
        rlm = RLM(data["y"].astype(float), rX, M=sm.robust.norms.HuberT())
        rfit = rlm.fit()
        est_ols = model.params.get("z_ucla", np.nan)
        est_rlm = rfit.params.filter(like="z_ucla").mean() if not pd.isna(rfit.params).all() else np.nan
        rows.append({
            "outcome": y,
            "nobs": int(model.nobs),
            "est_ols": est_ols,
            "est_rlm": est_rlm,
            "vif_mean": vif_mean,
            "max_cook": float(np.nanmax(cook)) if len(cook) else np.nan,
            "max_dfbetas_abs_ucla": float(np.nanmax(np.abs(dfb[:, list(model.params.index).index("z_ucla")]))) if "z_ucla" in model.params.index else np.nan,
        })
        pd.DataFrame({"cook": cook}).to_csv(OUT / f"influence_cooks_{y}.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "influence_robust_summary.csv", index=False)
    return out


def imputation_sensitivity(df: pd.DataFrame, m: int = 5, seed: int = 42) -> pd.DataFrame:
    # Prepare matrix for imputation
    use_cols = [
        "stroop_effect", "prp_bottleneck", "wcst_total_errors",
        "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age",
        "gender",
    ]
    base = df[use_cols].copy()
    # Encode gender
    base_d = pd.get_dummies(base, columns=["gender"], drop_first=True)
    imp_rows = []
    for r in range(m):
        imp = IterativeImputer(random_state=seed + r, sample_posterior=True, max_iter=10)
        X_imp = pd.DataFrame(imp.fit_transform(base_d), columns=base_d.columns)
        # Round gender dummy back to 0/1 if exists
        for c in X_imp.columns:
            if c.startswith("gender_"):
                X_imp[c] = (X_imp[c] > 0.5).astype(int)
        for y in ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]:
            cols = [
                y, "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age",
            ] + [c for c in X_imp.columns if c.startswith("gender_")]
            data = X_imp[cols].dropna().rename(columns={y: "y"})
            if len(data) < 40:
                continue
            f = "y ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age"
            if any(c.startswith("gender_") for c in cols):
                f += " + " + " + ".join([c for c in cols if c.startswith("gender_")])
            model = smf.ols(f, data=data).fit(cov_type="HC3")
            imp_rows.append({
                "imp": r + 1,
                "outcome": y,
                "nobs": int(model.nobs),
                "estimate": model.params.get("z_ucla", np.nan),
                "p_value": model.pvalues.get("z_ucla", np.nan),
            })
    out = pd.DataFrame(imp_rows)
    if not out.empty:
        summ = out.groupby("outcome").agg(
            n_imps=("imp", "nunique"),
            mean_est=("estimate", "mean"),
            sd_est=("estimate", "std"),
            mean_p=("p_value", "mean"),
        ).reset_index()
        summ.to_csv(OUT / "imputation_sensitivity_summary.csv", index=False)
    out.to_csv(OUT / "imputation_sensitivity_samples.csv", index=False)
    return out


def interactions_exploratory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]:
        cols = [y, "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]
        data = df[cols].dropna().rename(columns={y: "y"})
        if len(data) < 40:
            continue
        f = "y ~ z_ucla * C(gender) + z_dass_dep + z_dass_anx + z_dass_stress + age"
        model = smf.ols(f, data=data).fit(cov_type="HC3")
        for term in ["z_ucla", "z_ucla:C(gender)[T.male]"]:
            if term in model.params.index:
                rows.append({
                    "outcome": y,
                    "term": term,
                    "estimate": model.params[term],
                    "p_value": model.pvalues[term],
                    "nobs": int(model.nobs),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        from statsmodels.stats.multitest import multipletests
        out["q_value"] = multipletests(out["p_value"].astype(float).values, method="fdr_bh")[1]
    out.to_csv(OUT / "interaction_exploratory.csv", index=False)
    return out


def time_and_order_checks(df: pd.DataFrame) -> str:
    # Time-of-day: link residuals ~ hour median from WCST trials
    p = RES / "4b_wcst_trials.csv"
    lines = []
    if p.exists():
        tr = pd.read_csv(p)
        tr["participant_id"] = tr["participant_id"].fillna(tr.get("participantId"))
        tr = tr.dropna(subset=["participant_id", "timestamp"]).copy()
        tr["ts"] = pd.to_datetime(tr["timestamp"], errors="coerce")
        tr = tr.dropna(subset=["ts"]) \
             .assign(hour=lambda d: d["ts"].dt.hour)
        med = tr.groupby("participant_id")["hour"].median().reset_index().rename(columns={"hour": "median_hour"})
        ef = df[["participant_id", "stroop_effect", "prp_bottleneck", "wcst_total_errors", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]].dropna()
        use = ef.merge(med, on="participant_id", how="left").dropna(subset=["median_hour"]).copy()
        if len(use) >= 40:
            for col in ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]:
                X = use[["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age"]].astype(float)
                X = sm.add_constant(X)
                y = use[col].astype(float)
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X.dot(beta)
                r = pd.Series(resid).corr(use["median_hour"].astype(float))
                lines.append(f"{col}: resid vs hour r={r:.3f} (n={len(use)})")
    # Trial-index slope (Stroop)
    p2 = RES / "4c_stroop_trials.csv"
    if p2.exists():
        st = pd.read_csv(p2)
        st = st[(st.get("timeout", False) == False) & st["rt_ms"].notna()]
        st = st.merge(df[["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]], on="participant_id", how="inner")
        st = st.dropna(subset=["trial", "participant_id"]).copy()
        st["trial_scaled"] = st.groupby("participant_id")["trial"].transform(lambda s: s / s.max())
        slopes = st.groupby("participant_id").apply(lambda g: np.polyfit(g["trial_scaled"], g["rt_ms"], deg=1)[0] if g["trial_scaled"].nunique() >= 3 else np.nan).rename("rt_slope").dropna().reset_index()
        slopes = slopes.merge(df[["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]], on="participant_id", how="left").dropna()
        if len(slopes) >= 30:
            m = smf.ols("rt_slope ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress", data=slopes).fit(cov_type="HC3")
            lines.append(f"Stroop slope ~ UCLA: est={m.params.get('z_ucla', np.nan):.3f}, p={m.pvalues.get('z_ucla', np.nan):.3f} (n={int(m.nobs)})")
    out = "\n".join(lines)
    (OUT / "time_order_summary.txt").write_text(out, encoding="utf-8")
    return out


def mixed_random_slopes(df: pd.DataFrame) -> str:
    lines = []
    # Stroop random slope of incongruency
    ps = RES / "4c_stroop_trials.csv"
    if ps.exists():
        st = pd.read_csv(ps)
        st = st[(st.get("timeout", False) == False) & st["rt_ms"].notna()]
        st["participant_id"] = st["participant_id"].fillna(st.get("participantId"))
        st = st.dropna(subset=["participant_id"]).copy()
        cond = st["cond"] if "cond" in st.columns else st.get("type")
        st["incong"] = (cond.astype(str).str.lower() == "incongruent").astype(float)
        use = st.merge(df[["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]], on="participant_id", how="inner").dropna()
        if len(use) >= 2000 and use["participant_id"].nunique() >= 20:
            md = sm.MixedLM.from_formula("rt_ms ~ incong + z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + z_ucla:incong", groups="participant_id", re_formula="~incong", data=use)
            mfit = md.fit(method="lbfgs", maxiter=200, disp=False)
            est = mfit.params.get("z_ucla:incong", np.nan)
            lines.append(f"Stroop MixedLM z_ucla×incong: est={est:.3f}")
    # PRP random slope of SOA
    pp = RES / "4a_prp_trials.csv"
    if pp.exists():
        pr = pd.read_csv(pp)
        pr = pr[(pr.get("t2_timeout", False) == False) & pr["t2_rt_ms"].notna()]
        pr["participant_id"] = pr["participant_id"].fillna(pr.get("participantId"))
        pr = pr.dropna(subset=["participant_id"]).copy()
        pr["soa_scaled"] = pd.to_numeric(pr.get("soa_nominal_ms"), errors="coerce") / 1000.0
        use = pr.merge(df[["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress"]], on="participant_id", how="inner").dropna()
        if len(use) >= 2000 and use["participant_id"].nunique() >= 20:
            md = sm.MixedLM.from_formula("t2_rt_ms ~ soa_scaled + z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + z_ucla:soa_scaled", groups="participant_id", re_formula="~soa_scaled", data=use)
            mfit = md.fit(method="lbfgs", maxiter=200, disp=False)
            est = mfit.params.get("z_ucla:soa_scaled", np.nan)
            lines.append(f"PRP MixedLM z_ucla×soa: est={est:.3f}")
    out = "\n".join(lines)
    (OUT / "mixed_random_slopes_summary.txt").write_text(out, encoding="utf-8")
    return out


def main():
    df = build_analysis_dataframe()
    df = add_meta_control(df)
    mv, _ = multiverse(df)
    infl = influence_and_robust(df)
    imp = imputation_sensitivity(df)
    inter = interactions_exploratory(df)
    tos = time_and_order_checks(df)
    mrs = mixed_random_slopes(df)
    print("[OK] Multiverse specs:", len(mv))
    print("[OK] Influence summary rows:", len(infl))
    print("[OK] Imputation rows:", len(imp))
    print("[OK] Interaction rows:", len(inter))
    print("[OK] Time/order summary:\n", tos)
    print("[OK] Mixed random slopes summary:\n", mrs)


if __name__ == "__main__":
    main()
