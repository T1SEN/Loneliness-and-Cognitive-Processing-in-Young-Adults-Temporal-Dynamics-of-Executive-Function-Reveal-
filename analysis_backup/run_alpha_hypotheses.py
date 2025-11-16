"""
Test the Hα1~Hα10 hypothesis batch.

This reuses the preprocessed dataset from loneliness_exec_models and attempts
to operationalize each hypothesis. Items that require missing metrics are
flagged as "데이터 부족".
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe, add_meta_control  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_model(df, formula, cols, label, desc, term):
    data = df[cols].dropna()
    if len(data) < 20:
        return {"hypothesis": label, "description": desc, "status": f"표본 부족 (n={len(data)})"}
    model = smf.ols(formula=formula, data=data).fit()
    table = model.summary2().tables[1]
    if term not in table.index:
        return {"hypothesis": label, "description": desc, "status": f"{term} 계수 없음"}
    row = table.loc[term]
    return {
        "hypothesis": label,
        "description": desc,
        "status": "estimated",
        "term": term,
        "nobs": int(model.nobs),
        "estimate": float(row["Coef."]),
        "p_value": float(row["P>|t|"]),
        "conf_low": float(row["[0.025"]),
        "conf_high": float(row["0.975]"]),
    }


def main():
    df = build_analysis_dataframe()
    df = add_meta_control(df)
    df["high_loneliness"] = (df["ucla_total"] >= df["ucla_total"].quantile(0.75)).astype(int)

    results: List[Dict] = []
    covars = ["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]

    # Hα1: PRP bottleneck vs loneliness vs DASS anxiety (two rows)
    formula_a1 = "prp_bottleneck ~ z_ucla + z_dass_anx + z_dass_dep + z_dass_stress + age + C(gender)"
    cols_a1 = ["prp_bottleneck"] + covars
    results.append(run_model(df, formula_a1, cols_a1, "Hα1-1", "PRP 병목 ~ Loneliness", "z_ucla"))
    results.append(run_model(df, formula_a1, cols_a1, "Hα1-2", "PRP 병목 ~ DASS-Anxiety", "z_dass_anx"))

    # Hα2: Stroop incong-neutral diff (not available)
    results.append({"hypothesis": "Hα2", "description": "Stroop 비일치-중립 차이 지표 부재", "status": "데이터 부족"})

    # Hα3: perseverative resp % vs conceptual %
    formula_a3a = "wcst_persev_resp_pct ~ high_loneliness + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols_a3 = ["wcst_persev_resp_pct"] + covars + ["high_loneliness"]
    results.append(run_model(df, formula_a3a, cols_a3, "Hα3-1", "WCST PR% ~ 고외로움", "high_loneliness"))

    formula_a3b = "wcst_conceptual_pct ~ high_loneliness + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols_a3b = ["wcst_conceptual_pct"] + covars + ["high_loneliness"]
    results.append(run_model(df, formula_a3b, cols_a3b, "Hα3-2", "WCST CL% ~ 고외로움", "high_loneliness"))

    # Hα4: PRP slope vs loneliness controlling T1
    formula_a4 = ("prp_rt_slope ~ z_ucla + prp_mrt_t1 + z_dass_dep + "
                  "z_dass_anx + z_dass_stress + age + C(gender)")
    cols_a4 = ["prp_rt_slope", "prp_mrt_t1"] + covars
    results.append(run_model(df, formula_a4, cols_a4, "Hα4", "PRP slope ~ Loneliness", "z_ucla"))

    # Hα5: gender moderation
    formula_a5 = "prp_bottleneck ~ z_ucla * C(gender) + z_dass_dep + z_dass_anx + z_dass_stress + age"
    cols_a5 = ["prp_bottleneck"] + covars
    results.append(run_model(df, formula_a5, cols_a5, "Hα5", "PRP 병목 ~ Loneliness×성별", "z_ucla:C(gender)[T.male]"))

    # Hα6: correlation difference (persev vs non-persev)
    formula_a6a = "wcst_persev_errors ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols_a6 = ["wcst_persev_errors"] + covars
    results.append(run_model(df, formula_a6a, cols_a6, "Hα6-1", "WCST perseverative errors ~ Loneliness", "z_ucla"))

    formula_a6b = "wcst_nonpersev_errors ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols_a6b = ["wcst_nonpersev_errors"] + covars
    results.append(run_model(df, formula_a6b, cols_a6b, "Hα6-2", "WCST non-perseverative errors ~ Loneliness", "z_ucla"))

    # Hα7: interaction with DASS stress
    formula_a7 = "wcst_total_errors ~ z_ucla * z_dass_stress + z_dass_dep + z_dass_anx + age + C(gender)"
    cols_a7 = ["wcst_total_errors"] + covars
    results.append(run_model(df, formula_a7, cols_a7, "Hα7", "WCST total errors ~ Loneliness×Stress", "z_ucla:z_dass_stress"))

    # Hα8: PRP 연습 정확도 (missing)
    results.append({"hypothesis": "Hα8", "description": "PRP 연습 정확도 자료 부재", "status": "데이터 부족"})

    # Hα9: Stroop RT variance (missing)
    results.append({"hypothesis": "Hα9", "description": "Stroop RT 변동성 자료 부재", "status": "데이터 부족"})

    # Hα10: meta-control moderation
    formula_a10 = ("meta_control ~ prp_bottleneck * wcst_nonpersev_errors * z_ucla + "
                   "z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)")
    cols_a10 = ["meta_control", "prp_bottleneck", "wcst_nonpersev_errors"] + covars
    results.append(run_model(df, formula_a10, cols_a10, "Hα10",
                             "Meta-control ~ PRP*WCST*Loneliness",
                             "prp_bottleneck:wcst_nonpersev_errors:z_ucla"))

    summary = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "alpha_hypothesis_summary.csv"
    summary.to_csv(out_path, index=False)

    for row in results:
        if row.get("status") == "estimated":
            print(f"{row['hypothesis']} {row['description']}: {row['term']} estimate={row['estimate']:.3f}, "
                  f"p={row['p_value']:.3f} (n={row['nobs']})")
        else:
            print(f"{row['hypothesis']} {row['description']}: {row['status']}")
    print(f"\n저장 위치: {out_path}")


if __name__ == "__main__":
    main()
