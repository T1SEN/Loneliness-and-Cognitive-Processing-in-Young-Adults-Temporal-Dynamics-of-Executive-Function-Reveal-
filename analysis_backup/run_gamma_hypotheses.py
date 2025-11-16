"""
Test the Hγ1~Hγ10 exploratory hypotheses using available summary data.

Feasible components (quadratic terms, interactions, latent meta-control) are
estimated with statsmodels. Hypotheses requiring trial-level metrics (Hγ8~Hγ10)
are flagged as "데이터 부족".
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


def add_extra_terms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["z_ucla_sq"] = df["z_ucla"] ** 2
    df["log_ucla"] = np.log1p(df["ucla_total"])
    df["dass_total_z"] = (df["dass_total"] - df["dass_total"].mean()) / df["dass_total"].std()
    return df


def fit_model(df: pd.DataFrame, formula: str, cols: List[str], label: str, desc: str, term: str) -> Dict:
    data = df[cols].dropna()
    if len(data) < 30:
        return {"hypothesis": label, "description": desc, "status": f"표본 부족 (n={len(data)})"}
    model = smf.ols(formula, data=data).fit()
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
    df = add_extra_terms(df)
    feature_path = Path(__file__).resolve().parent.parent / "results" / "analysis_outputs" / "trial_level_features.csv"
    trial_feature_cols = [
        "prp_t2_cv_all",
        "prp_t2_cv_short",
        "prp_t2_cv_long",
        "prp_t2_trials",
        "stroop_post_error_slowing",
        "stroop_post_error_rt",
        "stroop_post_correct_rt",
        "stroop_incong_slope",
        "stroop_trials",
    ]

    if feature_path.exists():
        extra = pd.read_csv(feature_path)
        df = df.merge(extra, on="participant_id", how="left")
    else:
        print(f"[경고] Trial-level feature 파일을 찾을 수 없습니다: {feature_path}")
        for col in trial_feature_cols:
            if col not in df.columns:
                df[col] = np.nan

    results: List[Dict] = []
    covars = ["z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]

    # Hγ1 Quadratic (PRP bottleneck ~ z_ucla + z_ucla^2)
    formula1 = "prp_bottleneck ~ z_ucla + z_ucla_sq + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols1 = ["prp_bottleneck", "z_ucla", "z_ucla_sq"] + covars
    results.append(fit_model(df, formula1, cols1, "Hγ1", "PRP 병목 U자형", "z_ucla_sq"))

    # Hγ2 Log diminishing returns (Stroop effect ~ log_ucla)
    formula2 = "stroop_effect ~ log_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols2 = ["stroop_effect", "log_ucla"] + covars
    results.append(fit_model(df, formula2, cols2, "Hγ2", "Stroop 로그 관계", "log_ucla"))

    # Hγ3 Interaction with gender on WCST perseverative errors
    formula3 = "wcst_persev_errors ~ z_ucla * C(gender) + z_dass_dep + z_dass_anx + z_dass_stress + age"
    cols3 = ["wcst_persev_errors", "z_ucla", "gender", "age", "z_dass_dep", "z_dass_anx", "z_dass_stress"]
    results.append(fit_model(df, formula3, cols3, "Hγ3", "외로움×성별 → WCST 보속 오류", "z_ucla:C(gender)[T.male]"))

    # Hγ4 Interaction with age on PRP bottleneck
    formula4 = "prp_bottleneck ~ z_ucla * age + z_dass_dep + z_dass_anx + z_dass_stress + C(gender)"
    cols4 = ["prp_bottleneck", "z_ucla", "age", "z_dass_dep", "z_dass_anx", "z_dass_stress", "gender"]
    results.append(fit_model(df, formula4, cols4, "Hγ4", "외로움×나이 → PRP 병목", "z_ucla:age"))

    # Hγ5 Interaction with DASS depression on Stroop
    formula5 = "stroop_effect ~ z_ucla * z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols5 = ["stroop_effect", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]
    results.append(fit_model(df, formula5, cols5, "Hγ5", "외로움×DASS-우울 → Stroop", "z_ucla:z_dass_dep"))

    # Hγ6 Meta-control factor predicted by loneliness (latent)
    formula6 = "meta_control ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols6 = ["meta_control", "z_ucla"] + covars
    results.append(fit_model(df, formula6, cols6, "Hγ6", "Meta-control 요인 ~ 외로움", "z_ucla"))

    # Hγ7 Interaction: meta_control * DASS total predicting PRP bottleneck
    formula7 = "prp_bottleneck ~ meta_control * dass_total_z + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols7 = ["prp_bottleneck", "meta_control", "dass_total_z"] + covars
    results.append(fit_model(df, formula7, cols7, "Hγ7", "Meta-control×정서 → PRP", "meta_control:dass_total_z"))

    # Hγ8: PRP T2 CV ~ loneliness
    formula8 = "prp_t2_cv_short ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols8 = ["prp_t2_cv_short", "z_ucla"] + covars
    results.append(fit_model(df, formula8, cols8, "Hγ8", "PRP T2 CV(짧은 SOA) ~ 외로움", "z_ucla"))

    # Hγ9: Post-error slowing ~ loneliness
    formula9 = "stroop_post_error_slowing ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols9 = ["stroop_post_error_slowing", "z_ucla"] + covars
    results.append(fit_model(df, formula9, cols9, "Hγ9", "Stroop post-error slowing ~ 외로움", "z_ucla"))

    # Hγ10: Stroop incongruent slope ~ loneliness
    formula10 = "stroop_incong_slope ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    cols10 = ["stroop_incong_slope", "z_ucla"] + covars
    results.append(fit_model(df, formula10, cols10, "Hγ10", "Stroop incongruent slope ~ 외로움", "z_ucla"))

    # Hγ11: Loneliness-noise coupling (PRP CV * loneliness predicting WCST non-persev errors)
    formula11 = ("wcst_nonpersev_errors ~ prp_t2_cv_short * z_ucla + "
                 "z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)")
    cols11 = ["wcst_nonpersev_errors", "prp_t2_cv_short", "z_ucla"] + covars
    results.append(fit_model(df, formula11, cols11, "Hγ11",
                             "PRP CV×Loneliness → WCST non-persev errors",
                             "prp_t2_cv_short:z_ucla"))

    summary = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "gamma_hypothesis_summary.csv"
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
