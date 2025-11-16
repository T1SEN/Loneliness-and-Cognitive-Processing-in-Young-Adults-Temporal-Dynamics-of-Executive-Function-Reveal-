#!/usr/bin/env python3
"""
Explores the directional influence of DASS-21 subscales (Depression, Anxiety,
Stress) on each executive-function indicator using multiple regression.

Outputs coefficient tables for Stroop interference, PRP bottleneck, and WCST
error metrics, allowing us to see which affective factors show the clearest
trends.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
import statsmodels.formula.api as smf


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv")
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
    participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")

    summary.columns = summary.columns.str.lower()
    surveys.columns = surveys.columns.str.lower()
    participants.columns = participants.columns.str.lower()

    dass = (
        surveys[surveys["surveyname"] == "dass"]
        .rename(columns={"score_a": "dass_anx", "score_s": "dass_stress", "score_d": "dass_dep"})
        [["participantid", "dass_anx", "dass_stress", "dass_dep"]]
    )

    stroop = summary[summary["testname"] == "stroop"][["participantid", "stroop_effect"]]
    prp = summary[summary["testname"] == "prp"][
        ["participantid", "rt2_soa_50", "rt2_soa_1200"]
    ].copy()
    prp["prp_bottleneck"] = prp["rt2_soa_50"] - prp["rt2_soa_1200"]

    wcst = summary[summary["testname"] == "wcst"][
        ["participantid", "totalerrorcount", "perseverativeerrorcount", "nonperseverativeerrorcount"]
    ].rename(
        columns={
            "totalerrorcount": "wcst_total_errors",
            "perseverativeerrorcount": "wcst_persev_errors",
            "nonperseverativeerrorcount": "wcst_nonpersev_errors",
        }
    )

    df = participants[["participantid", "age", "gender"]]
    df = df.merge(dass, on="participantid", how="left")
    df = df.merge(stroop, on="participantid", how="left")
    df = df.merge(prp[["participantid", "prp_bottleneck"]], on="participantid", how="left")
    df = df.merge(wcst, on="participantid", how="left")

    df = df.rename(columns={"participantid": "participant_id"})
    return df


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        ("stroop_effect", "Stroop 간섭"),
        ("prp_bottleneck", "PRP 병목"),
        ("wcst_total_errors", "WCST 총 오류"),
        ("wcst_persev_errors", "WCST 보속 오류"),
        ("wcst_nonpersev_errors", "WCST 비보속 오류"),
    ]

    rows: List[Dict] = []
    for col, label in outcomes:
        data = df[["dass_dep", "dass_anx", "dass_stress", col]].dropna()
        if len(data) < 30:
            continue
        model = smf.ols(f"{col} ~ dass_dep + dass_anx + dass_stress", data=data).fit()
        table = model.summary2().tables[1].loc[["dass_dep", "dass_anx", "dass_stress"]]
        for term, row in table.iterrows():
            rows.append(
                {
                    "measure": label,
                    "predictor": term,
                    "estimate": row["Coef."],
                    "std_error": row["Std.Err."],
                    "p_value": row["P>|t|"],
                    "conf_low": row["[0.025"],
                    "conf_high": row["0.975]"],
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                }
            )
    return pd.DataFrame(rows)


def main():
    df = load_data()
    result = analyze(df)
    out_path = OUTPUT_DIR / "dass_effects_summary.csv"
    result.to_csv(out_path, index=False)
    print(f"DASS effect summary saved to {out_path}")
    if not result.empty:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
