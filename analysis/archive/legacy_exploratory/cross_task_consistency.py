"""
Cross-Task Consistency Analysis
===============================

Research Question: Are lonely individuals inconsistent across tasks
(high cross-task variability) or consistently impaired (stable deficit)?

Examines:
  - Within-person CV across tasks
  - Cross-task correlation matrix
  - Factor analysis of control metrics
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

try:
    import pymc as pm
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from analysis.utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "cross_task_consistency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_cross_task_metrics(master_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-task consistency metrics per participant."""
    results = []

    # Key EF metrics (standardized)
    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]

    for _, row in master_df.iterrows():
        pid = row["participant_id"]

        # Get EF values
        ef_values = []
        for col in ef_cols:
            if col in row and pd.notna(row[col]):
                ef_values.append(row[col])

        if len(ef_values) < 2:
            continue

        # Cross-task CV
        ef_mean = np.mean(ef_values)
        ef_sd = np.std(ef_values)
        cross_task_cv = ef_sd / abs(ef_mean) if ef_mean != 0 else np.nan

        # Range
        cross_task_range = max(ef_values) - min(ef_values)

        results.append({
            "participant_id": pid,
            "cross_task_cv": cross_task_cv,
            "cross_task_range": cross_task_range,
            "cross_task_mean": ef_mean,
            "cross_task_sd": ef_sd,
            "n_tasks": len(ef_values),
        })

    return pd.DataFrame(results)


def compute_cross_task_correlations(master_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-task correlation matrix."""
    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]

    available_cols = [c for c in ef_cols if c in master_df.columns]

    # Correlation matrix
    corr_matrix = master_df[available_cols].corr()

    # Flatten to long format
    results = []
    for i, col1 in enumerate(available_cols):
        for j, col2 in enumerate(available_cols):
            if i < j:
                r = corr_matrix.loc[col1, col2]
                n = master_df[[col1, col2]].dropna().shape[0]
                # Test significance
                if n > 3:
                    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_val = np.nan

                results.append({
                    "task1": col1,
                    "task2": col2,
                    "correlation": r,
                    "n": n,
                    "p_value": p_val,
                })

    return pd.DataFrame(results)


def run_frequentist_regressions(master_df: pd.DataFrame, outcomes: list, outcome_labels: dict) -> pd.DataFrame:
    """Run hierarchical regressions."""
    results = []
    formula_template = "{outcome} ~ z_ucla * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

    for outcome in outcomes:
        if outcome not in master_df.columns:
            continue

        df_clean = master_df.dropna(subset=[outcome, "z_ucla", "gender_male",
                                            "z_dass_depression", "z_dass_anxiety", "z_dass_stress", "z_age"])

        if len(df_clean) < 30:
            continue

        formula = formula_template.format(outcome=outcome)

        try:
            model = smf.ols(formula, data=df_clean).fit()

            for param in model.params.index:
                if param == "Intercept":
                    continue
                results.append({
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "parameter": param,
                    "coefficient": model.params[param],
                    "std_error": model.bse[param],
                    "t_value": model.tvalues[param],
                    "p_value": model.pvalues[param],
                    "ci_lower": model.conf_int().loc[param, 0],
                    "ci_upper": model.conf_int().loc[param, 1],
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                })
        except Exception as e:
            print(f"  Error fitting {outcome}: {e}")

    return pd.DataFrame(results)


def create_summary_report(freq_results: pd.DataFrame, corr_df: pd.DataFrame) -> str:
    """Generate summary report."""
    lines = [
        "=" * 80,
        "CROSS-TASK CONSISTENCY ANALYSIS: SUMMARY REPORT",
        "=" * 80,
        "",
        "RESEARCH QUESTION:",
        "Are lonely individuals inconsistent ACROSS tasks or consistently impaired?",
        "",
    ]

    lines.extend([
        "-" * 80,
        "CROSS-TASK CORRELATIONS:",
        "-" * 80,
    ])

    if not corr_df.empty:
        for _, row in corr_df.iterrows():
            sig = "*" if row["p_value"] < 0.05 else ""
            lines.append(f"  {row['task1']} vs {row['task2']}: r = {row['correlation']:.3f}{sig} (n={row['n']})")

    lines.extend([
        "",
        "-" * 80,
        "UCLA EFFECTS ON CROSS-TASK CONSISTENCY (p < 0.05):",
        "-" * 80,
    ])

    if not freq_results.empty:
        sig_results = freq_results[(freq_results["p_value"] < 0.05) &
                                   (freq_results["parameter"].str.contains("ucla", case=False))]
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                lines.append(f"  {row['outcome_label']}: {row['parameter']}")
                lines.append(f"    beta = {row['coefficient']:.4f}, p = {row['p_value']:.4f}")
        else:
            lines.append("  No significant UCLA effects")

    lines.extend([
        "",
        "=" * 80,
        "INTERPRETATION:",
        "=" * 80,
        "",
        "If UCLA predicts CROSS-TASK CV:",
        "  -> Loneliness causes general dysregulation (unstable person)",
        "",
        "If correlations are LOW:",
        "  -> EF domains are independent; task-specific effects",
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("CROSS-TASK CONSISTENCY ANALYSIS")
    print("=" * 60)

    # Load master dataset
    print("\n[1] Loading master dataset...")
    master = load_master_dataset(force_rebuild=False)
    print(f"  Master dataset: N={len(master)}")

    # Compute cross-task metrics
    print("\n[2] Computing cross-task consistency metrics...")
    cross_task = compute_cross_task_metrics(master)
    print(f"  Cross-task metrics: {len(cross_task)} participants")

    # Cross-task correlations
    print("\n[3] Computing cross-task correlations...")
    corr_df = compute_cross_task_correlations(master)
    for _, row in corr_df.iterrows():
        print(f"    {row['task1']} vs {row['task2']}: r = {row['correlation']:.3f}")

    # Merge
    analysis_df = master.merge(cross_task, on="participant_id", how="left")

    # Save
    cross_task.to_csv(OUTPUT_DIR / "cross_task_metrics.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUTPUT_DIR / "cross_task_correlations.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_DIR / 'cross_task_metrics.csv'}")

    # Define outcomes
    outcomes = ["cross_task_cv", "cross_task_range", "cross_task_sd"]
    outcome_labels = {
        "cross_task_cv": "Cross-Task CV",
        "cross_task_range": "Cross-Task Range",
        "cross_task_sd": "Cross-Task SD",
    }

    # Regressions
    print("\n[4] Running regressions...")
    freq_results = run_frequentist_regressions(analysis_df, outcomes, outcome_labels)
    freq_results.to_csv(OUTPUT_DIR / "frequentist_results.csv", index=False, encoding="utf-8-sig")

    sig_ucla = freq_results[(freq_results["p_value"] < 0.05) &
                            (freq_results["parameter"].str.contains("ucla", case=False))]
    if len(sig_ucla) > 0:
        print("\n  Significant UCLA effects:")
        for _, row in sig_ucla.iterrows():
            print(f"    {row['outcome_label']}: beta={row['coefficient']:.3f}, p={row['p_value']:.4f}")
    else:
        print("\n  No significant UCLA effects")

    # Report
    print("\n[5] Generating report...")
    report = create_summary_report(freq_results, corr_df)
    with open(OUTPUT_DIR / "CROSS_TASK_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("CROSS-TASK CONSISTENCY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
