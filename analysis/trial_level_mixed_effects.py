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
from analysis.utils.trial_data_loader import (
    load_prp_trials,
    load_stroop_trials,
    load_wcst_trials,
)
from analysis.utils.data_loader_utils import load_master_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/trial_level_mixed_effects.py')
BASE_DIR = _this_file.resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "analysis_outputs" / "trial_level_mixedlm_summary.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_predictor_table() -> pd.DataFrame:
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
    # Use gender_normalized if available
    if "gender_normalized" in master.columns:
        master["gender"] = master["gender_normalized"].fillna("").astype(str).str.strip().str.lower()
    else:
        master["gender"] = master["gender"].fillna("").astype(str).str.strip().str.lower()

    if "ucla_total" not in master.columns and "ucla_score" in master.columns:
        master["ucla_total"] = master["ucla_score"]

    predictors = master[["participant_id", "ucla_total", "dass_depression", "dass_anxiety", "dass_stress"]].dropna()

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
    stroop, _ = load_stroop_trials(use_cache=True)
    rt_col = "rt" if "rt" in stroop.columns else "rt_ms" if "rt_ms" in stroop.columns else None
    if rt_col is None:
        return "Stroop: RT column missing.\n"
    if "timeout" in stroop.columns:
        stroop = stroop[stroop["timeout"] == False]
    stroop = stroop.dropna(subset=[rt_col, "participant_id"])
    stroop["condition"] = (
        stroop["cond"]
        if "cond" in stroop.columns
        else stroop.get("type", "unknown").astype(str)
    )

    data = stroop.merge(predictors, on="participant_id", how="inner")
    data = data.dropna(
        subset=[
            rt_col,
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
        f"{rt_col} ~ C(condition) + z_ucla + z_dass_dep + z_dass_anx + "
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
    prp, _ = load_prp_trials(use_cache=True)

    rt_col = "t2_rt" if "t2_rt" in prp.columns else "t2_rt_ms" if "t2_rt_ms" in prp.columns else None
    if rt_col is None:
        return "PRP: RT column missing.\n"
    if rt_col != "t2_rt":
        prp = prp.rename(columns={rt_col: "t2_rt"})
        rt_col = "t2_rt"

    if "t2_timeout" in prp.columns:
        prp = prp[prp["t2_timeout"] == False]

    prp = prp.dropna(subset=["participant_id", rt_col, "soa"])
    prp["soa_scaled"] = prp["soa"] / 1000.0

    data = prp.merge(predictors, on="participant_id", how="inner")
    data = data.dropna(
        subset=[
            rt_col,
            "soa_scaled",
            "z_ucla",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
        ]
    )
    if len(data) < 2000 or data["participant_id"].nunique() < 20:
        return "PRP: insufficient data for mixed model.\n"
    data = data.reset_index(drop=True).copy()

    formula = (
        f"{rt_col} ~ soa_scaled + z_ucla + z_dass_dep + "
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
    wcst, _ = load_wcst_trials(use_cache=True)

    rt_col = "reactionTimeMs" if "reactionTimeMs" in wcst.columns else "rt_ms" if "rt_ms" in wcst.columns else None
    if rt_col is None:
        return "WCST: reaction time column missing.\n"

    wcst = wcst.dropna(subset=["participant_id", rt_col, "ruleAtThatTime"])

    data = wcst.merge(predictors, on="participant_id", how="inner")
    data = data.dropna(
        subset=[
            rt_col,
            "z_ucla",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
            "ruleAtThatTime",
        ]
    )
    if len(data) < 1500 or data["participant_id"].nunique() < 20:
        return "WCST: insufficient data for mixed model.\n"
    data = data.reset_index(drop=True).copy()

    formula = (
        f"{rt_col} ~ C(ruleAtThatTime) + z_ucla + "
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
