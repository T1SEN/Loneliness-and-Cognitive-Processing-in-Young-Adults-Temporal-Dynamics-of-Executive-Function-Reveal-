"""
Python implementation of the Loneliness × Executive Function analysis.

Replicates the R workflow in a pure Python stack so it can be executed
directly here. Steps:
1. Load participants, survey, and cognitive-summary CSV exports.
2. Derive task-level EF metrics (Stroop, PRP, WCST) and survey predictors.
3. Merge into a single analysis DataFrame with z-scored loneliness/DASS.
4. Optional: run 1-factor PCA across EF indicators to extract a latent
   "meta-control" score, saving loadings if ≥15 complete cases.
5. Fit OLS models for each EF outcome with loneliness + DASS controls,
   plus age and gender covariates (requiring ≥20 complete rows per model).
6. Save coefficient and model-fit tables to results/analysis_outputs/.

Usage:
    python analysis/loneliness_exec_models.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


COLUMN_RENAMES = {
    "participantid": "participant_id",
    "surveyname": "survey_name",
    "testname": "test_name",
    "conceptuallevelresponses": "conceptual_level_responses",
    "nonperseverativeerrorcount": "non_perseverative_error_count",
    "perseverativeresponses": "perseverative_responses",
    "trialstofirstconceptualresp": "trials_to_first_conceptual_resp",
    "totaltrialcount": "total_trial_count",
    "totalcorrectcount": "total_correct_count",
    "conceptuallevelresponsespercent": "conceptual_level_responses_percent",
    "completedcategories": "completed_categories",
    "totalerrorcount": "total_error_count",
    "perseverativeerrorcount": "perseverative_error_count",
    "trialstocompletefirstcategory": "trials_to_complete_first_category",
    "learningtolearn": "learning_to_learn",
    "perseverativeresponsespercent": "perseverative_responses_percent",
    "failuretomaintainset": "failure_to_maintain_set",
    "learningtolearnheatonclrdelta": "learning_to_learn_heaton_clr_delta",
    "learningefficiencydeltatrials": "learning_efficiency_delta_trials",
    "hasfirstclr": "has_first_clr",
    "trialstofirstconceptualresp0": "trials_to_first_conceptual_resp0",
    "categoryclrpercents": "category_clr_percents",
}


def read_csv_lower(path: Path) -> pd.DataFrame:
    """Read CSV with pandas and standardize column names to snake_case-ish."""
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "_", regex=False)
    )
    df = df.rename(columns=COLUMN_RENAMES)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def clean_gender(series: pd.Series) -> pd.Series:
    """Map Korean gender labels to English tokens."""
    s = series.fillna("").astype(str)
    male_mask = s.str.contains("남")
    female_mask = s.str.contains("여")
    out = pd.Series(np.nan, index=series.index, dtype="object")
    out[male_mask] = "male"
    out[female_mask] = "female"
    return out


def zscore(series: pd.Series) -> pd.Series:
    if series.std(skipna=True) == 0 or series.dropna().empty:
        return pd.Series([np.nan] * len(series), index=series.index)
    return (series - series.mean(skipna=True)) / series.std(skipna=True)


def robust_clean_gender(series: pd.Series) -> pd.Series:
    """Robust mapping of gender labels to 'male'/'female'.

    Supports Korean labels (남/남성, 여/여성) and English (M/F, male/female),
    case-insensitive, tolerating minor encoding artifacts.
    """
    s = series.fillna("").astype(str).str.strip().str.lower()
    s_norm = s.str.replace(r"[^a-z\u3131-\u318e\uac00-\ud7a3]", "", regex=True)
    male_mask = (
        s_norm.str.contains("남") | s_norm.str.contains("남성") | s_norm.str.contains("male") | s_norm.eq("m")
    )
    female_mask = (
        s_norm.str.contains("여") | s_norm.str.contains("여성") | s_norm.str.contains("female") | s_norm.eq("f")
    )
    out = pd.Series(np.nan, index=series.index, dtype="object")
    out[male_mask] = "male"
    out[female_mask] = "female"
    return out


def build_analysis_dataframe() -> pd.DataFrame:
    participants = read_csv_lower(RESULTS_DIR / "1_participants_info.csv")
    surveys = read_csv_lower(RESULTS_DIR / "2_surveys_results.csv")
    cog = read_csv_lower(RESULTS_DIR / "3_cognitive_tests_summary.csv")

    # Survey splits
    ucla = (
        surveys.query("survey_name == 'ucla'")
        .loc[:, ["participant_id", "score"]]
        .rename(columns={"score": "ucla_total"})
    )

    dass = (
        surveys.query("survey_name == 'dass'")
        .loc[:, ["participant_id", "score_a", "score_s", "score_d"]]
        .rename(columns={"score_a": "dass_anx", "score_s": "dass_stress", "score_d": "dass_dep"})
    )
    dass["dass_total"] = dass[["dass_anx", "dass_stress", "dass_dep"]].sum(axis=1)

    # Cognitive metrics
    prp = cog.query("test_name == 'prp'").copy()
    prp = prp.assign(
        prp_bottleneck=prp["rt2_soa_50"] - prp["rt2_soa_1200"],
        prp_rt_slope=(prp["rt2_soa_50"] - prp["rt2_soa_1200"]) / 1150.0,
    )[
        [
            "participant_id",
            "n_trials",
            "acc_t1",
            "acc_t2",
            "mrt_t1",
            "mrt_t2",
            "rt2_soa_50",
            "rt2_soa_1200",
            "prp_bottleneck",
            "prp_rt_slope",
        ]
    ].rename(
        columns={
            "n_trials": "prp_n_trials",
            "acc_t1": "prp_acc_t1",
            "acc_t2": "prp_acc_t2",
            "mrt_t1": "prp_mrt_t1",
            "mrt_t2": "prp_mrt_t2",
            "rt2_soa_50": "prp_rt_short",
            "rt2_soa_1200": "prp_rt_long",
        }
    )

    stroop = (
        cog.query("test_name == 'stroop'")
        .loc[:, ["participant_id", "stroop_effect", "accuracy", "total", "mrt_incong", "mrt_cong", "mrt_total"]]
        .rename(
            columns={
                "accuracy": "stroop_accuracy",
                "total": "stroop_total_trials",
            }
        )
    )

    wcst = (
        cog.query("test_name == 'wcst'")
        .loc[
            :,
            [
                "participant_id",
                "total_error_count",
                "perseverative_error_count",
                "non_perseverative_error_count",
                "completed_categories",
                "conceptual_level_responses_percent",
                "perseverative_responses_percent",
                "failure_to_maintain_set",
            ],
        ]
        .rename(
            columns={
                "total_error_count": "wcst_total_errors",
                "perseverative_error_count": "wcst_persev_errors",
                "non_perseverative_error_count": "wcst_nonpersev_errors",
                "completed_categories": "wcst_completed_categories",
                "conceptual_level_responses_percent": "wcst_conceptual_pct",
                "perseverative_responses_percent": "wcst_persev_resp_pct",
                "failure_to_maintain_set": "wcst_failure_to_maintain_set",
            }
        )
    )

    analysis = (
        participants.loc[:, ["participant_id", "age", "gender"]]
        .assign(
            age=lambda df: pd.to_numeric(df["age"], errors="coerce"),
            gender=lambda df: robust_clean_gender(df["gender"]),
        )
        .merge(ucla, on="participant_id", how="left")
        .merge(dass, on="participant_id", how="left")
        .merge(prp, on="participant_id", how="left")
        .merge(stroop, on="participant_id", how="left")
        .merge(wcst, on="participant_id", how="left")
    )

    # Add z-scores
    for col, new_col in [
        ("ucla_total", "z_ucla"),
        ("dass_dep", "z_dass_dep"),
        ("dass_anx", "z_dass_anx"),
        ("dass_stress", "z_dass_stress"),
    ]:
        analysis[new_col] = zscore(analysis[col])

    return analysis


# ---------------------------------------------------------------------------
# PCA meta-control factor
# ---------------------------------------------------------------------------
def add_meta_control(df: pd.DataFrame) -> pd.DataFrame:
    ef_cols = ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]
    complete = df.dropna(subset=ef_cols).copy()
    if len(complete) < 15:
        print("PCA skipped (need at least 15 complete EF rows).")
        df["meta_control"] = np.nan
        return df

    scaler = StandardScaler()
    scaled = scaler.fit_transform(complete[ef_cols])
    pca = PCA(n_components=1)
    scores = pca.fit_transform(scaled).ravel()
    complete = complete.assign(meta_control=scores)
    df = df.merge(complete[["participant_id", "meta_control"]], on="participant_id", how="left")

    loadings = pd.DataFrame(
        {
            "indicator": ef_cols,
            "loading": pca.components_[0],
        }
    )
    loadings.to_csv(OUTPUT_DIR / "meta_control_loadings.csv", index=False)
    print("Saved PCA loadings to meta_control_loadings.csv")
    return df


# ---------------------------------------------------------------------------
# Regression modeling
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    outcome: str
    nobs: int
    r_squared: float
    adj_r_squared: float
    aic: float
    bic: float
    p_value: float
    coefficients: pd.DataFrame


def run_models(df: pd.DataFrame) -> List[ModelResult]:
    specs = [
        ("stroop_effect", "Stroop interference (ms)"),
        ("prp_bottleneck", "PRP bottleneck (short-long RT)"),
        ("wcst_total_errors", "WCST total errors"),
        ("meta_control", "Latent meta-control factor"),
    ]
    results: List[ModelResult] = []
    formula_template = "y ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"

    for outcome, nice in specs:
        cols_needed = [
            outcome,
            "z_ucla",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
            "age",
            "gender",
        ]
        data = df[cols_needed].dropna()
        if len(data) < 20:
            print(f"Skipping {outcome}: only {len(data)} complete rows.")
            continue

        data = data.rename(columns={outcome: "y"})
        model = smf.ols(formula=formula_template, data=data).fit(cov_type="HC3")
        coef_raw = model.summary2().tables[1].copy()
        # Harmonize column names across robust/non-robust variants
        rename_map = {"Coef.": "estimate", "Std.Err.": "std_error", "[0.025": "conf_low", "0.975]": "conf_high"}
        coef_raw = coef_raw.rename(columns=rename_map)
        if "P>|t|" in coef_raw.columns:
            coef_raw = coef_raw.rename(columns={"P>|t|": "p_value", "t": "stat"})
        if "P>|z|" in coef_raw.columns:
            coef_raw = coef_raw.rename(columns={"P>|z|": "p_value", "z": "stat"})
        coef_df = coef_raw.reset_index().rename(columns={"index": "term"})

        results.append(
            ModelResult(
                outcome=nice,
                nobs=int(model.nobs),
                r_squared=float(model.rsquared),
                adj_r_squared=float(model.rsquared_adj),
                aic=float(model.aic),
                bic=float(model.bic),
                p_value=float(model.f_pvalue),
                coefficients=coef_df,
            )
        )
        print(f"Fitted model for {nice} (n={int(model.nobs)})")

    return results


def save_outputs(results: List[ModelResult]) -> None:
    if not results:
        print("No models were fit; nothing to save.")
        return

    coef_rows = []
    fit_rows = []
    for res in results:
        coef = res.coefficients.copy()
        coef.insert(0, "outcome", res.outcome)
        coef_rows.append(coef)

        fit_rows.append(
            {
                "outcome": res.outcome,
                "r_squared": res.r_squared,
                "adj_r_squared": res.adj_r_squared,
                "AIC": res.aic,
                "BIC": res.bic,
                "p_value": res.p_value,
                "nobs": res.nobs,
            }
        )

    coef_df = pd.concat(coef_rows, ignore_index=True)
    # Add FDR (BH) q-values across all coefficients
    try:
        from statsmodels.stats.multitest import multipletests
        if "p_value" in coef_df.columns:
            coef_df["q_value"] = multipletests(coef_df["p_value"].astype(float).values, method="fdr_bh")[1]
    except Exception:
        pass
    fit_df = pd.DataFrame(fit_rows)

    coef_path = OUTPUT_DIR / "loneliness_models_coefficients_py.csv"
    fit_path = OUTPUT_DIR / "loneliness_models_fit_py.csv"
    coef_df.to_csv(coef_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    print(f"Saved coefficients to {coef_path.name}")
    print(f"Saved fit stats to {fit_path.name}")

    lon_rows = coef_df.query("term == 'z_ucla'")
    if not lon_rows.empty:
        print("\n=== Key p-values for UCLA Loneliness (Python) ===")
        print(lon_rows[["outcome", "estimate", "std_error", "conf_low", "conf_high", "p_value"]])


def main():
    analysis_df = build_analysis_dataframe()
    analysis_df = add_meta_control(analysis_df)
    results = run_models(analysis_df)
    save_outputs(results)


if __name__ == "__main__":
    main()
