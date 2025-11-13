"""
Quartile-based loneliness analysis.

Creates a binary indicator for participants in the top 25% of UCLA loneliness
scores and tests whether that “high loneliness” group shows different
executive-function outcomes after controlling for DASS-21 subscales, age, and
gender. Outputs tidy coefficient/f-stat tables similar to the continuous
models.

Usage:
    python analysis/loneliness_quartile_analysis.py
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from loneliness_exec_models import (
    build_analysis_dataframe,
    add_meta_control,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_high_loneliness_indicator(df: pd.DataFrame) -> pd.DataFrame:
    threshold = df["ucla_total"].quantile(0.75)
    df = df.copy()
    df["high_loneliness"] = np.where(df["ucla_total"] >= threshold, 1, 0)
    df.attrs["loneliness_threshold"] = threshold
    return df


def run_quartile_models(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("stroop_effect", "Stroop interference (ms)"),
        ("prp_bottleneck", "PRP bottleneck (short-long RT)"),
        ("wcst_total_errors", "WCST total errors"),
        ("meta_control", "Latent meta-control factor"),
    ]
    formula = "y ~ high_loneliness + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    rows = []
    for outcome, nice in specs:
        cols = [
            outcome,
            "high_loneliness",
            "z_dass_dep",
            "z_dass_anx",
            "z_dass_stress",
            "age",
            "gender",
        ]
        data = df[cols].dropna()
        if len(data) < 20:
            print(f"Skipping {nice}: insufficient complete cases (n={len(data)}).")
            continue
        model = smf.ols(formula=formula, data=data.rename(columns={outcome: "y"})).fit()
        coef = (
            model.summary2().tables[1]
            .rename(
                columns={
                    "Coef.": "estimate",
                    "Std.Err.": "std_error",
                    "P>|t|": "p_value",
                    "[0.025": "conf_low",
                    "0.975]": "conf_high",
                }
            )
            .reset_index()
            .rename(columns={"index": "term"})
        )
        coef.insert(0, "outcome", nice)
        rows.append(
            {
                "outcome": nice,
                "nobs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "AIC": float(model.aic),
                "BIC": float(model.bic),
                "p_value": float(model.f_pvalue),
                "coef_table": coef,
            }
        )
        print(f"Fitted quartile model for {nice} (n={int(model.nobs)})")
    return pd.DataFrame(rows)


def main():
    analysis_df = build_analysis_dataframe()
    analysis_df = add_meta_control(analysis_df)
    analysis_df = make_high_loneliness_indicator(analysis_df)
    threshold = analysis_df.attrs.get("loneliness_threshold")
    print(f"High-loneliness threshold (75th percentile UCLA): {threshold:.2f}")
    results = run_quartile_models(analysis_df)
    if results.empty:
        print("No quartile models were estimated.")
        return

    coef_df = pd.concat(results["coef_table"].tolist(), ignore_index=True)
    fit_df = results.drop(columns=["coef_table"])

    coef_path = OUTPUT_DIR / "loneliness_quartile_coefficients_py.csv"
    fit_path = OUTPUT_DIR / "loneliness_quartile_fit_py.csv"
    coef_df.to_csv(coef_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    print(f"Saved quartile coefficients to {coef_path.name}")
    print(f"Saved quartile fit stats to {fit_path.name}")

    key = coef_df.query("term == 'high_loneliness'")
    if not key.empty:
        print("\n=== High vs. other loneliness effect (binary) ===")
        print(key[["outcome", "estimate", "conf_low", "conf_high", "p_value"]])


if __name__ == "__main__":
    main()
