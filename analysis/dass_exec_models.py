"""
Frequentist OLS: EF outcomes ~ DASS subscales (D/A/S)

Fits OLS models with HC3 robust SEs to test associations between each
executive-function indicator and DASS-21 subscales (Depression/Anxiety/Stress),
controlling for age and gender. Runs both univariate (one subscale at a time)
and multivariate (all three together) versions, and saves tidy coefficient
tables and model fit stats.

Outputs (results/analysis_outputs/):
  - dass_exec_models_coefficients.csv
  - dass_exec_models_fit.csv
  - dass_key_pvalues.csv (quick filter of D/A/S terms)

Usage:
  python analysis/dass_exec_models.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv_lower(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "_", regex=False)
    )
    return df


def clean_gender(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str)
    out = pd.Series(np.nan, index=s.index, dtype="object")
    out[x.str.contains("남")] = "male"
    out[x.str.contains("여")] = "female"
    return out


def z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().std() in (0, None) or s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / s.std()


def build_df() -> pd.DataFrame:
    participants = read_csv_lower(RES / "1_participants_info.csv")
    surveys = read_csv_lower(RES / "2_surveys_results.csv")
    cog = read_csv_lower(RES / "3_cognitive_tests_summary.csv")

    demo = participants.loc[:, ["participantid", "age", "gender"]].rename(
        columns={"participantid": "participant_id"}
    )
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
    demo["gender"] = clean_gender(demo["gender"])

    dass = (
        surveys.query("surveyname == 'dass'")
        .loc[:, ["participantid", "score_d", "score_a", "score_s"]]
        .rename(columns={"participantid": "participant_id", "score_d": "dass_dep", "score_a": "dass_anx", "score_s": "dass_stress"})
    )

    prp = cog.query("testname == 'prp'").copy()
    prp = prp.assign(
        prp_bottleneck=pd.to_numeric(prp["rt2_soa_50"], errors="coerce") - pd.to_numeric(prp["rt2_soa_1200"], errors="coerce"),
    ).rename(columns={"participantid": "participant_id"})
    prp_keep = prp.loc[:, ["participant_id", "prp_bottleneck", "mrt_t1", "mrt_t2", "acc_t1", "acc_t2"]]
    for c in ["mrt_t1", "mrt_t2", "acc_t1", "acc_t2"]:
        prp_keep[c] = pd.to_numeric(prp_keep[c], errors="coerce")

    stroop = (
        cog.query("testname == 'stroop'")
        .loc[:, ["participantid", "stroop_effect", "accuracy", "mrt_incong", "mrt_cong", "mrt_total"]]
        .rename(columns={"participantid": "participant_id", "accuracy": "stroop_accuracy"})
    )
    for c in ["stroop_effect", "stroop_accuracy", "mrt_incong", "mrt_cong", "mrt_total"]:
        stroop[c] = pd.to_numeric(stroop[c], errors="coerce")

    wcst = (
        cog.query("testname == 'wcst'")
        .loc[:, [
            "participantid",
            "totalerrorcount",
            "perseverativeerrorcount",
            "nonperseverativeerrorcount",
            "conceptuallevelresponsespercent",
            "perseverativeresponsespercent",
            "failuretomaintainset",
        ]]
        .rename(
            columns={
                "participantid": "participant_id",
                "totalerrorcount": "wcst_total_errors",
                "perseverativeerrorcount": "wcst_persev_errors",
                "nonperseverativeerrorcount": "wcst_nonpersev_errors",
                "conceptuallevelresponsespercent": "wcst_conceptual_pct",
                "perseverativeresponsespercent": "wcst_persev_resp_pct",
                "failuretomaintainset": "wcst_failure_to_maintain_set",
            }
        )
    )
    for c in wcst.columns:
        if c != "participant_id":
            wcst[c] = pd.to_numeric(wcst[c], errors="coerce")

    df = demo.merge(dass, on="participant_id", how="left")
    df = df.merge(prp_keep, on="participant_id", how="left")
    df = df.merge(stroop, on="participant_id", how="left")
    df = df.merge(wcst, on="participant_id", how="left")

    # z-scores for D/A/S
    df["z_dass_dep"] = z(df["dass_dep"])
    df["z_dass_anx"] = z(df["dass_anx"])
    df["z_dass_stress"] = z(df["dass_stress"])
    return df


def add_meta_control(df: pd.DataFrame) -> pd.DataFrame:
    ef_cols = ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]
    complete = df.dropna(subset=ef_cols).copy()
    if len(complete) < 15:
        df["meta_control"] = np.nan
        return df
    scaler = StandardScaler()
    X = scaler.fit_transform(complete[ef_cols])
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(X).ravel()
    tmp = complete[["participant_id"]].assign(meta_control=scores)
    return df.merge(tmp, on="participant_id", how="left")


@dataclass
class FitRow:
    outcome: str
    spec: str
    nobs: int
    r2: float
    aic: float
    bic: float


def run_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("stroop_effect", "Stroop interference (ms)"),
        ("prp_bottleneck", "PRP bottleneck (ms)"),
        ("wcst_total_errors", "WCST total errors"),
        ("meta_control", "Latent meta-control"),
    ]

    rows_coef: List[pd.DataFrame] = []
    rows_fit: List[dict] = []

    # Model specs
    specs = {
        "dep_only": "y ~ z_dass_dep + age + C(gender)",
        "anx_only": "y ~ z_dass_anx + age + C(gender)",
        "str_only": "y ~ z_dass_stress + age + C(gender)",
        "all_three": "y ~ z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)",
    }

    for col, nice in outcomes:
        need = [col, "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]
        data = df[need].dropna()
        if len(data) < 25:
            continue
        data = data.rename(columns={col: "y"})
        for spec_name, formula in specs.items():
            try:
                model = smf.ols(formula=formula, data=data).fit(cov_type="HC3")
            except Exception:
                continue
            tab = model.summary2().tables[1].copy()
            tab = tab.rename(columns={
                "Coef.": "estimate",
                "Std.Err.": "std_error",
                "[0.025": "conf_low",
                "0.975]": "conf_high",
                "P>|t|": "p_value",
                "t": "stat",
            })
            coef_df = tab.reset_index().rename(columns={"index": "term"})
            coef_df.insert(0, "outcome", nice)
            coef_df.insert(1, "spec", spec_name)
            rows_coef.append(coef_df)

            rows_fit.append(
                {
                    "outcome": nice,
                    "spec": spec_name,
                    "nobs": int(model.nobs),
                    "r2": float(model.rsquared),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                }
            )

    coef_all = pd.concat(rows_coef, ignore_index=True) if rows_coef else pd.DataFrame()
    fit_all = pd.DataFrame(rows_fit)
    return coef_all, fit_all


def main() -> None:
    df = build_df()
    df = add_meta_control(df)
    coef, fit = run_models(df)
    coef_path = OUT / "dass_exec_models_coefficients.csv"
    fit_path = OUT / "dass_exec_models_fit.csv"
    coef.to_csv(coef_path, index=False)
    fit.to_csv(fit_path, index=False)
    print(f"Saved coefficients to {coef_path.name}")
    print(f"Saved fit stats to {fit_path.name}")
    # Quick key rows
    if not coef.empty:
        key = coef[coef["term"].isin(["z_dass_dep", "z_dass_anx", "z_dass_stress"])]
        key.to_csv(OUT / "dass_key_pvalues.csv", index=False)
        print("\n=== Key D/A/S coefficients (HC3) ===")
        cols = [c for c in ["outcome", "spec", "term", "estimate", "p_value", "conf_low", "conf_high"] if c in key.columns]
        print(key[cols].to_string(index=False))


if __name__ == "__main__":
    main()
