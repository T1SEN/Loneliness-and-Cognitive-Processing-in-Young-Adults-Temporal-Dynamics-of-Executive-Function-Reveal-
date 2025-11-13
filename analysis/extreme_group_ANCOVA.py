#!/usr/bin/env python3
"""
Extreme-group ANCOVA with DASS covariates.

1) Creates UCLA 상/하위 집단 (사분위 & 중앙값)
2) 각 집단 간 집행기능 지표를 ANCOVA로 비교하면서
   DASS-21 하위척도, 나이, 성별을 공변량으로 통제
3) 결과 표와 박스플롯/바이올린플롯을 results/analysis_outputs/에 저장
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEPENDENTS = [
    ("stroop_effect", "Stroop 간섭 (ms)"),
    ("prp_bottleneck", "PRP 병목 (ms)"),
    ("wcst_total_errors", "WCST 총 오류"),
]

COVARS = ["z_dass_dep", "z_dass_anx", "z_dass_stress", "age"]


def make_group_labels(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    quartile = df.copy()
    q1, q3 = quartile["ucla_total"].quantile([0.25, 0.75])
    quartile["group"] = np.where(
        quartile["ucla_total"] <= q1,
        "low",
        np.where(quartile["ucla_total"] >= q3, "high", None),
    )
    quartile = quartile.dropna(subset=["group"])

    median = df.copy()
    med = median["ucla_total"].median()
    median["group"] = np.where(median["ucla_total"] < med, "low", "high")

    return {"quartile": quartile, "median": median}


def adjusted_means(model, group_col: str, covar_means: Dict[str, float], gender_mode: str):
    records = []
    for level in ["low", "high"]:
        row = {group_col: level, "gender": gender_mode}
        row.update(covar_means)
        records.append(row)
    df_pred = pd.DataFrame(records)
    preds = model.get_prediction(df_pred).summary_frame()
    df_pred["adj_mean"] = preds["mean"]
    return df_pred.set_index(group_col)["adj_mean"].to_dict()


def ancova_table(df: pd.DataFrame, group_type: str) -> pd.DataFrame:
    rows = []
    gender_mode = df["gender"].mode(dropna=True)
    gender_mode = gender_mode.iloc[0] if not gender_mode.empty else "female"
    covar_means = {c: df[c].mean() for c in COVARS}

    for dep, label in DEPENDENTS:
        cols_needed = [dep, "group", "gender"] + COVARS
        data = df[cols_needed].dropna()
        if data["group"].nunique() < 2 or len(data) < 20:
            rows.append(
                {
                    "group_type": group_type,
                    "measure": label,
                    "n_low": int((data["group"] == "low").sum()),
                    "n_high": int((data["group"] == "high").sum()),
                    "F": np.nan,
                    "p_value": np.nan,
                    "partial_eta2": np.nan,
                    "adj_mean_low": np.nan,
                    "adj_mean_high": np.nan,
                }
            )
            continue

        covar_formula = " + ".join(COVARS)
        formula = f"{dep} ~ C(group) + {covar_formula} + C(gender)"
        model = smf.ols(formula, data=data).fit()
        anova = anova_lm(model, typ=2)
        effect = anova.loc["C(group)"]
        residual = anova.loc["Residual"]
        partial_eta = effect["sum_sq"] / (effect["sum_sq"] + residual["sum_sq"])
        means = adjusted_means(model, "group", covar_means, gender_mode)

        rows.append(
            {
                "group_type": group_type,
                "measure": label,
                "n_low": int((data["group"] == "low").sum()),
                "n_high": int((data["group"] == "high").sum()),
                "F": effect["F"],
                "p_value": effect["PR(>F)"],
                "partial_eta2": partial_eta,
                "adj_mean_low": means.get("low", np.nan),
                "adj_mean_high": means.get("high", np.nan),
            }
        )
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, prefix: str):
    long_df = df[["group"] + [dep for dep, _ in DEPENDENTS]].melt(
        id_vars="group", var_name="measure", value_name="value"
    )
    measure_labels = {dep: label for dep, label in DEPENDENTS}
    long_df["measure"] = long_df["measure"].map(measure_labels)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=long_df, x="measure", y="value", hue="group")
    plt.title(f"극단 집단 비교 (박스플롯) - {prefix}")
    plt.xlabel("측정치")
    plt.ylabel("값")
    plt.legend(title="UCLA 그룹")
    box_path = OUTPUT_DIR / f"extreme_group_boxplots_{prefix}.png"
    plt.tight_layout()
    plt.savefig(box_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=long_df, x="measure", y="value", hue="group", split=True)
    plt.title(f"극단 집단 비교 (바이올린플롯) - {prefix}")
    plt.xlabel("측정치")
    plt.ylabel("값")
    plt.legend(title="UCLA 그룹")
    violin_path = OUTPUT_DIR / f"extreme_group_violinplots_{prefix}.png"
    plt.tight_layout()
    plt.savefig(violin_path, dpi=300)
    plt.close()


def main():
    df = build_analysis_dataframe()
    if df.empty:
        raise SystemExit("분석 가능한 데이터가 없습니다.")
    df["ucla_total"] = pd.to_numeric(df["ucla_total"], errors="coerce")

    grouped = make_group_labels(df)

    quartile_table = ancova_table(grouped["quartile"], "quartile")
    median_table = ancova_table(grouped["median"], "median")

    quartile_table.to_csv(OUTPUT_DIR / "extreme_group_quartile_results.csv", index=False)
    median_table.to_csv(OUTPUT_DIR / "extreme_group_median_results.csv", index=False)

    save_plots(grouped["quartile"], "quartile")
    save_plots(grouped["median"], "median")

    print("사분위 집단 결과:")
    print(quartile_table.to_string(index=False))
    print("\n중앙값 집단 결과:")
    print(median_table.to_string(index=False))
    print(f"\n결과 파일 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
