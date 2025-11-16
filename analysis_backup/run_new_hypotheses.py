"""
Evaluate the second batch of 10 hypotheses (G1~G10).

Available hypotheses are tested with statsmodels OLS; ones that require
trial-level or missing summary metrics are flagged as "데이터 부족".
Outputs summary CSV at results/analysis_outputs/new_hypothesis_summary.csv.
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


def run_formula(df: pd.DataFrame, formula: str, required: List[str], label: str, desc: str,
                target_term: str) -> Dict:
    data = df[required].dropna()
    if len(data) < 20:
        return {"hypothesis": label, "description": desc, "status": f"표본 부족 (n={len(data)})"}
    model = smf.ols(formula=formula, data=data).fit()
    table = model.summary2().tables[1]
    if target_term not in table.index:
        return {"hypothesis": label, "description": desc, "status": f"{target_term} 계수 없음"}
    row = table.loc[target_term]
    return {
        "hypothesis": label,
        "description": desc,
        "status": "estimated",
        "term": target_term,
        "nobs": int(model.nobs),
        "estimate": float(row["Coef."]),
        "p_value": float(row["P>|t|"]),
        "conf_low": float(row["[0.025"]),
        "conf_high": float(row["0.975]"]),
    }


def main():
    df = build_analysis_dataframe()
    df = add_meta_control(df)
    cutoff = df["ucla_total"].quantile(0.75)
    df["high_loneliness"] = np.where(df["ucla_total"] >= cutoff, 1, 0)

    # Hypothesis mappings
    results: List[Dict] = []

    covars = ["z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]

    # G1 - practice accuracy (not available)
    results.append({
        "hypothesis": "G1",
        "description": "PRP 연습 정확도 자료 부재",
        "status": "데이터 부족"
    })

    # G2 - prp_bottleneck vs loneliness & comparison to DASS anxiety (report both)
    formula_g2 = "prp_bottleneck ~ z_ucla + z_dass_anx + z_dass_dep + z_dass_stress + age + C(gender)"
    req_g2 = ["prp_bottleneck"] + covars
    res_ucla = run_formula(df, formula_g2, req_g2, "G2a",
                           "PRP 병목 ~ Loneliness", "z_ucla")
    res_anx = run_formula(df, formula_g2, req_g2, "G2b",
                          "PRP 병목 ~ DASS-Anxiety", "z_dass_anx")
    results.extend([res_ucla, res_anx])

    # G3 - Stroop neutral difference (not available)
    results.append({
        "hypothesis": "G3",
        "description": "Stroop 중립 조건/차이 자료 부재",
        "status": "데이터 부족"
    })

    # G4 - wcst perseverative response %
    formula_g4 = "wcst_persev_resp_pct ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    req_g4 = ["wcst_persev_resp_pct"] + covars
    results.append(run_formula(df, formula_g4, req_g4, "G4",
                               "WCST perseverative response % ~ Loneliness", "z_ucla"))

    # G5 - moderation between PRP bottleneck and WCST non-perseverative errors
    formula_g5 = ("wcst_nonpersev_errors ~ prp_bottleneck * z_ucla + "
                  "z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)")
    req_g5 = ["wcst_nonpersev_errors", "prp_bottleneck"] + covars
    results.append(run_formula(df, formula_g5, req_g5, "G5",
                               "WCST non-persev errors ~ PRP*Loneliness",
                               "prp_bottleneck:z_ucla"))

    # G6 - gender moderation on PRP bottleneck
    formula_g6 = ("prp_bottleneck ~ z_ucla * C(gender) + "
                  "z_dass_dep + z_dass_anx + z_dass_stress + age")
    req_g6 = ["prp_bottleneck"] + covars
    results.append(run_formula(df, formula_g6, req_g6, "G6",
                               "PRP 병목 ~ Loneliness×Gender",
                               "z_ucla:C(gender)[T.male]"))

    # G7 - WCST conceptual %
    formula_g7 = "wcst_conceptual_pct ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    req_g7 = ["wcst_conceptual_pct"] + covars
    results.append(run_formula(df, formula_g7, req_g7, "G7",
                               "WCST conceptual % ~ Loneliness", "z_ucla"))

    # G8 - Stroop RT variability missing
    results.append({
        "hypothesis": "G8",
        "description": "Stroop RT 변동성 지표 부재",
        "status": "데이터 부족"
    })

    # G9a - PRP T1 RT ~ Loneliness
    formula_g9a = "prp_mrt_t1 ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    req_g9a = ["prp_mrt_t1"] + covars
    results.append(run_formula(df, formula_g9a, req_g9a, "G9a",
                               "PRP T1 RT ~ Loneliness", "z_ucla"))

    # G9b - PRP RT slope ~ Loneliness controlling T1 RT
    formula_g9b = ("prp_rt_slope ~ z_ucla + prp_mrt_t1 + "
                   "z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)")
    req_g9b = ["prp_rt_slope", "prp_mrt_t1"] + covars
    results.append(run_formula(df, formula_g9b, req_g9b, "G9b",
                               "PRP SOA slope ~ Loneliness (T1 RT 통제)", "z_ucla"))

    # G10 - interaction with DASS stress
    formula_g10 = ("prp_bottleneck ~ z_ucla * z_dass_stress + "
                   "z_dass_dep + z_dass_anx + age + C(gender)")
    req_g10 = ["prp_bottleneck"] + covars
    results.append(run_formula(df, formula_g10, req_g10, "G10",
                               "PRP 병목 ~ Loneliness×DASS-Stress",
                               "z_ucla:z_dass_stress"))

    summary = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "new_hypothesis_summary.csv"
    summary.to_csv(out_path, index=False)

    for row in results:
        if row.get("status") == "estimated":
            print(f"{row['hypothesis']} {row['description']}: "
                  f"{row['term']} estimate={row['estimate']:.3f}, p={row['p_value']:.3f} (n={row['nobs']})")
        else:
            print(f"{row['hypothesis']} {row['description']}: {row['status']}")
    print(f"\n저장 위치: {out_path}")


if __name__ == "__main__":
    main()
