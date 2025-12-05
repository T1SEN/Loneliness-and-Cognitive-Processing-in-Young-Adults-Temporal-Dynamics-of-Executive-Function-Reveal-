#!/usr/bin/env python3
"""
Reliability-adjusted power and sensitivity analysis for loneliness models.

Reads the coefficient tables from the core OLS models and adjusts the
observed partial R^2 for attenuation due to measurement error using
available reliability estimates. Produces projections at N=150 and the
sample size required for ~80% power at alpha=.05.

Outputs: results/analysis_outputs/power_reliability_adjusted.csv
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "analysis_outputs"


def load_effects() -> pd.DataFrame:
    coef = pd.read_csv(OUTDIR / "loneliness_models_coefficients_py.csv")
    fit = pd.read_csv(OUTDIR / "loneliness_models_fit_py.csv")
    lon = coef.query("term == 'z_ucla'").copy()
    k = coef.groupby("outcome").size().rename("k")
    lon = lon.merge(k, on="outcome", how="left")
    lon = lon.merge(fit[["outcome", "nobs"]], on="outcome", how="left")
    # compute observed partial R^2 from t-stat
    pr2_rows = []
    for _, r in lon.iterrows():
        nobs = int(r["nobs"])
        kparams = int(r["k"])  # total parameters in model summary
        df2 = nobs - kparams
        t = r["estimate"] / r["std_error"] if r["std_error"] else 0.0
        pr2 = (t**2) / (t**2 + df2) if df2 > 0 else 0.0
        pr2_rows.append({"outcome": r["outcome"], "partialR2_obs": pr2, "df2": df2, "k": kparams})
    return pd.DataFrame(pr2_rows)


def load_reliability() -> dict:
    rel = {"ucla": float(os.environ.get("UCLA_RELIABILITY", 0.90))}
    path = OUTDIR / "rv_reliability.csv"
    if not path.exists():
        return {**rel, "stroop": 0.6, "prp": 0.5, "wcst": 0.5}
    df = pd.read_csv(path)
    if df.empty:
        return {**rel, "stroop": 0.6, "prp": 0.5, "wcst": 0.5}
    row = df.iloc[0]
    # Stroop: prefer Spearman-Brown if present
    stroop_sb = pd.to_numeric(row.get("sb"), errors="coerce")
    stroop_r = pd.to_numeric(row.get("stroop_split_half_r"), errors="coerce")
    stroop = float(stroop_sb if pd.notna(stroop_sb) else (2*stroop_r/(1+stroop_r) if pd.notna(stroop_r) else 0.6))
    # PRP: compute Spearman-Brown from split-half r if needed
    prp_r = pd.to_numeric(row.get("prp_split_half_r"), errors="coerce")
    prp = float(2*prp_r/(1+prp_r)) if pd.notna(prp_r) else 0.5
    # WCST: use deck r if available
    wcst_r = pd.to_numeric(row.get("wcst_deck_r"), errors="coerce")
    wcst = float(wcst_r) if pd.notna(wcst_r) else 0.5
    return {**rel, "stroop": stroop, "prp": prp, "wcst": wcst}


def map_reliability_for_outcome(outcome: str, rel: dict) -> float:
    outcome = str(outcome).lower()
    if "stroop" in outcome:
        return rel.get("stroop", 0.6)
    if "prp" in outcome:
        return rel.get("prp", 0.5)
    if "wcst" in outcome:
        return rel.get("wcst", 0.5)
    if "meta-control" in outcome or "meta control" in outcome:
        vals = [rel.get("stroop", 0.6), rel.get("prp", 0.5), rel.get("wcst", 0.5)]
        return float(np.nanmean(vals))
    return 0.6


def main():
    eff = load_effects()
    rel = load_reliability()
    rows = []
    for _, r in eff.iterrows():
        outcome = r["outcome"]
        pr2_obs = float(r["partialR2_obs"]) if pd.notna(r["partialR2_obs"]) else 0.0
        df2_now = int(r["df2"]) if pd.notna(r["df2"]) else 0
        kparams = int(r["k"]) if pd.notna(r["k"]) else 7

        r_x = rel.get("ucla", 0.9)
        r_y = map_reliability_for_outcome(outcome, rel)
        atten = max(r_x * r_y, 1e-6)
        pr2_true = min(pr2_obs / atten, 0.99) if atten > 0 else pr2_obs

        # Predicted p-value at N=150 using approximate t from partial R^2
        n_target = 150
        df2_target = n_target - kparams
        def p_from_pr2(df2: int, pr2: float) -> float:
            if df2 <= 0 or pr2 <= 0:
                return 1.0
            t2 = df2 * pr2 / (1 - pr2)
            t = math.sqrt(max(t2, 0.0))
            # normal approx
            return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))

        p_obs_150 = p_from_pr2(df2_target, pr2_obs)
        p_true_150 = p_from_pr2(df2_target, pr2_true)

        # Required N for 80% power (approx) given partial R^2
        t_needed = 1.96 + 0.84
        def n_required(pr2: float) -> float:
            if pr2 <= 0:
                return float("inf")
            df2_req = (t_needed**2) * (1 - pr2) / pr2
            return df2_req + kparams

        n80_obs = n_required(pr2_obs)
        n80_true = n_required(pr2_true)

        rows.append({
            "outcome": outcome,
            "partialR2_obs": pr2_obs,
            "partialR2_true_est": pr2_true,
            "reliability_ucla": r_x,
            "reliability_outcome": r_y,
            "pred_p_at_150_obs": p_obs_150,
            "pred_p_at_150_true": p_true_150,
            "N_needed_80power_obs": n80_obs,
            "N_needed_80power_true": n80_true,
        })

    out = pd.DataFrame(rows)
    out_path = OUTDIR / "power_reliability_adjusted.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved reliability-adjusted power table to {out_path}")


if __name__ == "__main__":
    main()

