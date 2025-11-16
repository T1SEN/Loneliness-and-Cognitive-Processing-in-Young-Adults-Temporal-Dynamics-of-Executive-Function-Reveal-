#!/usr/bin/env python3
"""
Trial-level mixed-effects 분석
------------------------------
Stroop / PRP / WCST trial 로그 전체를 활용해,
UCLA 외로움(z_ucla)과 DASS 보정 변수가 반응시간에 미치는 효과를
선형 혼합효과 모델로 추정한다.

결과는 results/analysis_outputs/trial_level_mixedlm_summary.txt 에 저장된다.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "analysis_outputs" / "trial_level_mixedlm_summary.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_predictor_table() -> pd.DataFrame:
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
    surveys.columns = surveys.columns.str.lower()

    ucla = (
        surveys[surveys["surveyname"].str.lower() == "ucla"]
        [["participantid", "score"]]
        .rename(columns={"participantid": "participant_id", "score": "ucla_total"})
    )
    dass = (
        surveys[surveys["surveyname"].str.lower() == "dass"]
        [["participantid", "score_d", "score_a", "score_s"]]
        .rename(
            columns={
                "participantid": "participant_id",
                "score_d": "dass_depression",
                "score_a": "dass_anxiety",
                "score_s": "dass_stress",
            }
        )
    )
    predictors = ucla.merge(dass, on="participant_id", how="inner")

    def z(series):
        if series.std(skipna=True) == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean(skipna=True)) / series.std(skipna=True)

    predictors["z_ucla"] = z(predictors["ucla_total"])
    predictors["z_dass_dep"] = z(predictors["dass_depression"])
    predictors["z_dass_anx"] = z(predictors["dass_anxiety"])
    predictors["z_dass_stress"] = z(predictors["dass_stress"])
    return predictors


def fit_stroop_mixedlm(predictors: pd.DataFrame) -> str:
    stroop = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
    stroop["participant_id"] = (
        stroop["participant_id"].fillna(stroop.get("participantId"))
    )
    stroop = stroop[(stroop["timeout"] == False) & stroop["rt_ms"].notna()]
    stroop = stroop.dropna(subset=["participant_id"])
    stroop["condition"] = (
        stroop["cond"]
        if "cond" in stroop.columns
        else stroop.get("type", "unknown").astype(str)
    )

    data = stroop.merge(predictors, on="participant_id", how="inner")
    data = data.dropna(
        subset=[
            "rt_ms",
            "z_ucla",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
            "condition",
        ]
    )
    if len(data) < 2000 or data["participant_id"].nunique() < 20:
        return "Stroop: 데이터가 부족하여 혼합모형을 적합할 수 없습니다.\n"
    data = data.reset_index(drop=True).copy()
    formula = (
        "rt_ms ~ C(condition) + z_ucla + z_dass_dep + z_dass_anx + "
        "z_dass_stress + z_ucla:C(condition)"
    )
    formula = (
        "rt_ms ~ C(condition) + z_ucla + z_dass_dep + z_dass_anx + "
        "z_dass_stress + z_ucla:C(condition)"
    )
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["participant_id"],
        re_formula="~1",
    )
    result = model.fit(method="lbfgs", maxiter=200, disp=False)
    return "\n=== Stroop MixedLM (trial-level RT) ===\n" + str(result.summary()) + "\n"


def fit_prp_mixedlm(predictors: pd.DataFrame) -> str:
    prp = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")
    prp = prp[prp["t2_rt_ms"].notna()]
    prp = prp[prp["t2_timeout"] == False]
    prp["soa_scaled"] = prp["soa_nominal_ms"] / 1000.0
    data = prp.merge(predictors, on="participant_id", how="inner")
    data = data.dropna(
        subset=[
            "t2_rt_ms",
            "soa_scaled",
            "z_ucla",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
        ]
    )
    if len(data) < 2000 or data["participant_id"].nunique() < 20:
        return "PRP: 데이터가 부족하여 혼합모형을 적합할 수 없습니다.\n"
    data = data.reset_index(drop=True).copy()

    formula = (
        "t2_rt_ms ~ soa_scaled + z_ucla + z_dass_dep + "
        "z_dass_anx + z_dass_stress + z_ucla:soa_scaled"
    )
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["participant_id"],
        re_formula="~1",
    )
    result = model.fit(method="lbfgs", maxiter=200, disp=False)
    return "\n=== PRP MixedLM (trial-level T2 RT) ===\n" + str(result.summary()) + "\n"


def fit_wcst_mixedlm(predictors: pd.DataFrame) -> str:
    wcst = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
    wcst["participant_id"] = wcst["participant_id"].fillna(
        wcst.get("participantId")
    )
    if "reactionTimeMs" not in wcst.columns:
        return "WCST: reactionTimeMs 컬럼이 없어 분석을 건너뜁니다.\n"
    wcst = wcst[wcst["reactionTimeMs"].notna()]
    wcst = wcst.dropna(subset=["participant_id"])
    data = wcst.merge(predictors, on="participant_id", how="inner")
    data = data.dropna(
        subset=[
            "reactionTimeMs",
            "z_ucla",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
            "ruleAtThatTime",
        ]
    )
    if len(data) < 1500 or data["participant_id"].nunique() < 20:
        return "WCST: 데이터가 부족하여 혼합모형을 적합할 수 없습니다.\n"
    data = data.reset_index(drop=True).copy()

    formula = (
        "reactionTimeMs ~ C(ruleAtThatTime) + z_ucla + "
        "z_dass_dep + z_dass_anx + z_dass_stress + z_ucla:C(ruleAtThatTime)"
    )
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["participant_id"],
        re_formula="~1",
    )
    result = model.fit(method="lbfgs", maxiter=200, disp=False)
    return "\n=== WCST MixedLM (trial-level RT) ===\n" + str(result.summary()) + "\n"


def main():
    predictors = build_predictor_table()
    sections = [
        "TRIAL-LEVEL MIXED EFFECTS ANALYSIS\n===============================\n",
        fit_stroop_mixedlm(predictors),
        fit_prp_mixedlm(predictors),
        fit_wcst_mixedlm(predictors),
    ]
    report = "\n".join(sections)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n결과 요약 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
