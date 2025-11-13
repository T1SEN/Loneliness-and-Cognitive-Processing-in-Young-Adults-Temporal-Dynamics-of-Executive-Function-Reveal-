"""
Batch execution of the 10 exploratory 가설.

Only the hypotheses that can be operationalized with the current summary data
are estimated. Others are reported as "데이터 부족" because the necessary
trial-level metrics (예: post-error slowing, T2-first 비율) are not present in
`3_cognitive_tests_summary.csv`.

Usage:
    python analysis/run_hypothesis_tests.py
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe, add_meta_control  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_high_loneliness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cutoff = df["ucla_total"].quantile(0.75)
    df["high_loneliness"] = np.where(df["ucla_total"] >= cutoff, 1, 0)
    df.attrs["cutoff"] = cutoff
    return df


def add_conflict_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Latent factor from Stroop effect + PRP bottleneck."""
    cols = ["stroop_effect", "prp_bottleneck"]
    complete = df.dropna(subset=cols)
    if len(complete) < 15:
        df["conflict_factor"] = np.nan
        return df
    scaled = StandardScaler().fit_transform(complete[cols])
    pca = PCA(n_components=1)
    scores = pca.fit_transform(scaled).ravel()
    complete = complete.assign(conflict_factor=scores)
    df = df.merge(complete[["participant_id", "conflict_factor"]], on="participant_id", how="left")
    loadings = pd.DataFrame({"indicator": cols, "loading": pca.components_[0]})
    loadings.to_csv(OUTPUT_DIR / "conflict_factor_loadings.csv", index=False)
    return df


def run_model(df: pd.DataFrame, outcome: str, predictor: str, label: str) -> Dict:
    cols = [outcome, predictor, "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]
    data = df[cols].dropna()
    if len(data) < 20:
        return {"hypothesis": label, "status": f"표본 부족 (n={len(data)})"}
    formula = f"y ~ {predictor} + z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)"
    model = smf.ols(formula=formula, data=data.rename(columns={outcome: "y"})).fit()
    main = model.summary2().tables[1].loc[predictor]
    return {
        "hypothesis": label,
        "status": "estimated",
        "nobs": int(model.nobs),
        "estimate": float(main["Coef."]),
        "p_value": float(main["P>|t|"]),
        "conf_low": float(main["[0.025"]),
        "conf_high": float(main["0.975]"]),
    }


def main():
    df = build_analysis_dataframe()
    df = add_meta_control(df)
    df = add_high_loneliness(df)
    df = add_conflict_factor(df)

    tests = [
        ("H1", "Stroop 효과 ~ Loneliness", "stroop_effect", "z_ucla"),
        ("H2", "PRP 병목 ~ Loneliness", "prp_bottleneck", "z_ucla"),
        ("H3", "WCST 비보속 오류 ~ Loneliness", "wcst_nonpersev_errors", "z_ucla"),
        ("H4", "Meta-control ~ 고외로움(상위25%)", "meta_control", "high_loneliness"),
        ("H5", "갈등 요인 ~ Loneliness", "conflict_factor", "z_ucla"),
    ]

    not_available = [
        ("H6", "Post-error slowing (자료 없음)"),
        ("H7", "WCST 첫 범주 완료 시행 수 (요약에 없음)"),
        ("H8", "PRP T2 선반응 비율 (trial-level 필요)"),
        ("H9", "갈등 민감도 vs UCLA 상관 (H5와 중복, 추가 지표 없음)"),
        ("H10", "RT 변동성 (trial-level 필요)"),
    ]

    rows: List[Dict] = []
    for code, desc, outcome, predictor in tests:
        result = run_model(df, outcome, predictor, code)
        result["description"] = desc
        rows.append(result)

    for code, reason in not_available:
        rows.append({"hypothesis": code, "description": reason, "status": "데이터 부족"})

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "hypothesis_test_summary.csv"
    out_df.to_csv(out_path, index=False)

    for row in rows:
        if row.get("status") == "estimated":
            print(
                f"{row['hypothesis']} {row['description']}: "
                f"estimate={row['estimate']:.3f}, p={row['p_value']:.3f} (n={row['nobs']})"
            )
        else:
            print(f"{row['hypothesis']} {row['description']}: {row['status']}")
    print(f"\n저장 위치: {out_path}")


if __name__ == "__main__":
    main()
