#!/usr/bin/env python3
"""
Re-estimate key nonlinear/interaction hypotheses with robust SEs (HC3) and FDR.

Outputs: results/analysis_outputs/gamma_hypothesis_summary_fixed.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe, add_meta_control  # noqa: E402


OUT = Path(__file__).resolve().parent.parent / "results" / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def add_terms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["z_ucla_sq"] = df["z_ucla"] ** 2
    df["log_ucla"] = np.log1p(df["ucla_total"])
    df["dass_total_z"] = (df["dass_total"] - df["dass_total"].mean()) / df["dass_total"].std()
    return df


def fit(formula: str, data: pd.DataFrame, term: str, label: str, desc: str) -> Dict:
    if len(data) < 30:
        return {"hypothesis": label, "description": desc, "status": f"insufficient(n={len(data)})"}
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    tbl = model.summary2().tables[1]
    if term not in tbl.index:
        return {"hypothesis": label, "description": desc, "status": f"term_missing:{term}"}
    row = tbl.loc[term]
    p_col = "P>|t|" if "P>|t|" in tbl.columns else ("P>|z|" if "P>|z|" in tbl.columns else None)
    if p_col is None:
        return {"hypothesis": label, "description": desc, "status": "table_format_error"}
    return {
        "hypothesis": label,
        "description": desc,
        "status": "estimated",
        "term": term,
        "nobs": int(model.nobs),
        "estimate": float(row["Coef."]),
        "p_value": float(row[p_col]),
        "conf_low": float(row["[0.025"]),
        "conf_high": float(row["0.975]"]),
    }


def main():
    df = build_analysis_dataframe()
    df = add_meta_control(df)
    df = add_terms(df)

    covars = ["z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]
    results: List[Dict] = []

    # Quadratic PRP ~ z_ucla + z_ucla^2
    f1 = "prp_bottleneck ~ z_ucla + z_ucla_sq + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    d1 = df[["prp_bottleneck", "z_ucla", "z_ucla_sq"] + covars].dropna()
    results.append(fit(f1, d1, "z_ucla_sq", "Hγ1", "PRP 병목 U형",))

    # Log Stroop ~ log_ucla
    f2 = "stroop_effect ~ log_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    d2 = df[["stroop_effect", "log_ucla"] + covars].dropna()
    results.append(fit(f2, d2, "log_ucla", "Hγ2", "Stroop 로그 관계"))

    # Gender moderation on WCST perseverative errors
    f3 = "wcst_persev_errors ~ z_ucla * C(gender) + z_dass_dep + z_dass_anx + z_dass_stress + age"
    d3 = df[["wcst_persev_errors", "z_ucla", "gender", "age", "z_dass_dep", "z_dass_anx", "z_dass_stress"]].dropna()
    results.append(fit(f3, d3, "z_ucla:C(gender)[T.male]", "Hγ3", "Loneliness×Gender on WCST perseverative errors"))

    # Age moderation on PRP
    f4 = "prp_bottleneck ~ z_ucla * age + z_dass_dep + z_dass_anx + z_dass_stress + C(gender)"
    d4 = df[["prp_bottleneck", "z_ucla", "age", "z_dass_dep", "z_dass_anx", "z_dass_stress", "gender"]].dropna()
    results.append(fit(f4, d4, "z_ucla:age", "Hγ4", "Loneliness×Age on PRP"))

    # Interaction with DASS depression on Stroop
    f5 = "stroop_effect ~ z_ucla * z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    d5 = df[["stroop_effect", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]].dropna()
    results.append(fit(f5, d5, "z_ucla:z_dass_dep", "Hγ5", "Loneliness×Depression on Stroop"))

    # Meta-control ~ loneliness
    f6 = "meta_control ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    d6 = df[["meta_control", "z_ucla"] + covars].dropna()
    results.append(fit(f6, d6, "z_ucla", "Hγ6", "Meta-control ~ Loneliness"))

    # PRP ~ meta_control * DASS total
    f7 = "prp_bottleneck ~ meta_control * dass_total_z + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    d7 = df[["prp_bottleneck", "meta_control", "dass_total_z"] + covars].dropna()
    results.append(fit(f7, d7, "meta_control:dass_total_z", "Hγ7", "Meta-control×Stress on PRP"))

    summary = pd.DataFrame(results)
    # FDR q-values across estimated rows
    try:
        from statsmodels.stats.multitest import multipletests
        mask = summary["status"].eq("estimated") & summary["p_value"].notna()
        if mask.any():
            summary.loc[mask, "q_value"] = multipletests(summary.loc[mask, "p_value"].astype(float).values, method="fdr_bh")[1]
    except Exception:
        pass

    out_path = OUT / "gamma_hypothesis_summary_fixed.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
