#!/usr/bin/env python3
"""
Equivalence (TOST) + Model Comparison (LOO/WAIC) + WCST windowed switch cost + Invariance
Outputs under results/analysis_outputs/:
- tost_summary.csv
- model_compare_waic_loo.csv
- wcst_switch_cost_window.csv
- invariance_summary.txt
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
from scipy import stats

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_wcst_trials

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------
# Helpers
# ------------------------

def tost_means(x, y, low, high, alpha=0.05):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 5 or ny < 5:
        return {"p_lower": np.nan, "p_upper": np.nan, "equivalent": False, "nx": nx, "ny": ny}
    diff = np.mean(x) - np.mean(y)
    se = np.sqrt(np.var(x, ddof=1)/nx + np.var(y, ddof=1)/ny)
    if se == 0 or not np.isfinite(se):
        return {"p_lower": np.nan, "p_upper": np.nan, "equivalent": False, "nx": nx, "ny": ny}
    t_low = (diff - low)/se
    t_up  = (diff - high)/se
    df = nx + ny - 2
    p_low = 1 - stats.t.cdf(t_low, df)
    p_up  = stats.t.cdf(t_up, df)
    equivalent = (p_low < alpha) and (p_up < alpha)
    return {"p_lower": float(p_low), "p_upper": float(p_up), "equivalent": bool(equivalent), "nx": nx, "ny": ny,
            "diff": float(diff)}


def build_analysis_df():
    from loneliness_exec_models import build_analysis_dataframe, add_meta_control
    df = build_analysis_dataframe()
    df = add_meta_control(df)
    return df

# ------------------------
# 1) TOST on EF outcomes (high vs low loneliness)
# ------------------------

def run_tost(df: pd.DataFrame, ms_bounds=(10.0, 10.0)) -> pd.DataFrame:
    data = df.copy()
    q3 = data["ucla_total"].quantile(0.75)
    q1 = data["ucla_total"].quantile(0.25)
    # Define extreme groups
    group = pd.Series(index=data.index, dtype='object')
    group[data['ucla_total'] >= q3] = 'high'
    group[data['ucla_total'] <= q1] = 'low'
    data = data.assign(group=group)
    data = data.dropna(subset=['group'])

    # Equivalence bounds (units: ms for RT differences; rate for error proportion)
    specs = [
        ("stroop_effect", "ms", (-ms_bounds[0], ms_bounds[1])),
        ("prp_bottleneck", "ms", (-ms_bounds[0], ms_bounds[1])),
        ("wcst_total_errors", "rate", (-0.5, 0.5)),
    ]

    rows = []
    for col, unit, (lo, hi) in specs:
        x = data.loc[data.group == "high", col]
        y = data.loc[data.group == "low", col]
        res = tost_means(x, y, low=lo, high=hi)
        rows.append({"outcome": col, "unit": unit, **res, "low_bound": lo, "high_bound": hi})
    return pd.DataFrame(rows)

# ------------------------
# 2) LOO/WAIC model comparison (PyMC)
# ------------------------

def fit_model_py(y, X, draws=1000, tune=1000, chains=2):
    """Simple Bayesian linear model returning ArviZ InferenceData."""
    with pm.Model() as m:
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.Deterministic("mu", (X * beta).sum(axis=1))
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            progressbar=False,
            return_inferencedata=True,
        )
    return idata


def run_model_compare(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    outcomes = ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]
    for col in outcomes:
        cols = [col, "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender", "z_ucla"]
        data = df[cols].dropna()
        if len(data) < 50:
            out_rows.append({"outcome": col, "n": len(data), "waic_diff": np.nan, "loo_diff": np.nan})
            continue
        # Encode gender
        g = (data["gender"].astype(str) == "male").astype(int).values.reshape(-1,1)
        age = pd.to_numeric(data["age"], errors="coerce").fillna(data["age"].median()).values.reshape(-1,1)
        cov = data[["z_dass_dep","z_dass_anx","z_dass_stress"]].values
        X0 = np.hstack([cov, age, g])
        X1 = np.hstack([cov, age, g, data[["z_ucla"]].values])
        y = data[col].values
        id0 = fit_model_py(y, X0)
        id1 = fit_model_py(y, X1)
        waic0, waic1 = az.waic(id0), az.waic(id1)
        loo0, loo1 = az.loo(id0), az.loo(id1)
        # Use ELPD metrics where higher is better. Positive diff => UCLA improves fit.
        waic0_elpd = float(getattr(waic0, 'elpd_waic', getattr(waic0, 'waic', np.nan)))
        waic1_elpd = float(getattr(waic1, 'elpd_waic', getattr(waic1, 'waic', np.nan)))
        loo0_elpd = float(getattr(loo0, 'elpd_loo', getattr(loo0, 'loo', np.nan)))
        loo1_elpd = float(getattr(loo1, 'elpd_loo', getattr(loo1, 'loo', np.nan)))

        out_rows.append({
            "outcome": col,
            "n": len(data),
            "waic0": waic0_elpd,
            "waic1": waic1_elpd,
            "waic_diff": waic1_elpd - waic0_elpd,
            "loo0": loo0_elpd,
            "loo1": loo1_elpd,
            "loo_diff": loo1_elpd - loo0_elpd,
        })
    return pd.DataFrame(out_rows)

# ------------------------
# 3) WCST windowed switch cost around category changes
# ------------------------

def wcst_window_switch() -> pd.DataFrame:
    df, _ = load_wcst_trials(use_cache=True)
    if "reactionTimeMs" not in df.columns and "rt_ms" in df.columns:
        df = df.rename(columns={"rt_ms": "reactionTimeMs"})
    if "reactionTimeMs" not in df.columns or "ruleAtThatTime" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["participant_id"]).copy()
    # basic RT filters
    df = df[(pd.to_numeric(df["reactionTimeMs"], errors="coerce") >= 200) & (pd.to_numeric(df["reactionTimeMs"], errors="coerce") <= 5000)]
    df = df.sort_values(["participant_id", df.get("trialIndex","trialIndexInBlock")])
    rows = []
    for pid, grp in df.groupby("participant_id"):
        grp = grp.copy()
        grp["prev_rule"] = grp["ruleAtThatTime"].shift(1)
        switches = grp.index[ (grp["ruleAtThatTime"] != grp["prev_rule"]) & grp["prev_rule"].notna() ]
        for idx in switches:
            i = grp.index.get_loc(idx)
            # skip early learning trials
            if i < 5: 
                continue
            pre = grp.iloc[max(0, i-3):i]["reactionTimeMs"].mean()
            post = grp.iloc[i+1:i+1+3]["reactionTimeMs"].mean()
            if np.isfinite(pre) and np.isfinite(post):
                rows.append({"participant_id": pid, "switch_idx": int(i), "pre3_mean": float(pre), "post3_mean": float(post), "diff": float(post-pre)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(OUT / "wcst_switch_cost_window.csv", index=False)
    return out

# ------------------------
# 4) Invariance by time-of-day (as proxy for environment)
# ------------------------

def invariance_time_of_day(df: pd.DataFrame) -> str:
    tr, _ = load_wcst_trials(use_cache=True)
    if "timestamp" not in tr.columns:
        return "[Invariance] timestamp column not available; skipping.\n"
    use = tr.dropna(subset=["participant_id","timestamp"]).copy()
    use["ts"] = pd.to_datetime(use["timestamp"], errors="coerce")
    use = use.dropna(subset=["ts"]) 
    use["hour"] = use["ts"].dt.hour
    # Merge EF outcomes per participant
    ef = df[["participant_id","stroop_effect","prp_bottleneck","wcst_total_errors","z_ucla","z_dass_dep","z_dass_anx","z_dass_stress","age","gender"]].dropna()
    sd = use.groupby("participant_id")["hour"].median().reset_index().rename(columns={"hour":"median_hour"})
    ef2 = ef.merge(sd, on="participant_id", how="left").dropna(subset=["median_hour"]) 
    if len(ef2) < 40:
        return "[Invariance] not enough participants with timestamp to test.\n"
    # Simple OLS via scipy stats on residuals
    lines = ["# Invariance (time-of-day as covariate)"]
    for col in ["stroop_effect","prp_bottleneck","wcst_total_errors"]:
        # regress outcome on covariates w/o hour, get residuals
        X = ef2[["z_ucla","z_dass_dep","z_dass_anx","z_dass_stress","age"]].astype(float).values
        y = ef2[col].astype(float).values
        X = np.hstack([np.ones((len(X),1)), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X.dot(beta)
        r = stats.pearsonr(resid, ef2["median_hour"].astype(float).values)
        lines.append(f"- {col}: resid vs hour r={r.statistic:.03f}, p={r.pvalue:.3f}")
    return "\n".join(lines) + "\n"


def main():
    df = build_analysis_df()
    # 1) TOST
    tost = run_tost(df, ms_bounds=(10.0,10.0))
    tost.to_csv(OUT / "tost_summary.csv", index=False)
    # 2) LOO/WAIC
    mc = run_model_compare(df)
    mc.to_csv(OUT / "model_compare_waic_loo.csv", index=False)
    # 3) WCST windowed switch
    wcst_win = wcst_window_switch()
    # 4) Invariance
    inv_text = invariance_time_of_day(df)
    (OUT / "invariance_summary.txt").write_text(inv_text, encoding="utf-8")
    # Console brief
    print("TOST summary:\n", tost.to_string(index=False))
    print("\nModel comparison (ΔWAIC, ΔLOO > 0 means UCLA improves fit):\n", mc.to_string(index=False))
    if not wcst_win.empty:
        print("\nWCST windowed switch cost (first rows):\n", wcst_win.head().to_string(index=False))
    print("\nInvariance:\n", inv_text)

if __name__ == "__main__":
    main()



