"""Generate Table 3-6 for static (core outcomes only)."""
import sys
from pathlib import Path

import pandas as pd

from static.preprocessing.surveys import (
    SurveyQCCriteria,
    get_survey_valid_participants,
    load_dass_scores,
    load_participants,
    load_ucla_scores,
)
from static.preprocessing.constants import OUTPUT_STATS_DIR, OUTPUT_TABLES_DIR, RAW_DIR, get_results_dir
from static.preprocessing.overall.loaders import load_overall_summary

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Paths
repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data"
output_dir = OUTPUT_TABLES_DIR
output_dir.mkdir(parents=True, exist_ok=True)


def calc_stats(df, col, var_name):
    """Calculate descriptive statistics."""
    valid = df[col].dropna()
    return {
        "Variable": var_name,
        "N": len(valid),
        "Mean": valid.mean(),
        "SD": valid.std(),
    }


def _format_ci(beta: float, se: float) -> str:
    if pd.isna(beta) or pd.isna(se):
        return "NA"
    lower = beta - 1.96 * se
    upper = beta + 1.96 * se
    return f"[{lower:.3f}, {upper:.3f}]"


def _infer_cov_type(hr_by_task: dict[str, pd.DataFrame]) -> str:
    labels = []
    for hr in hr_by_task.values():
        if hr is None or hr.empty:
            continue
        if "cov_type" in hr.columns:
            labels.extend([str(x) for x in hr["cov_type"].dropna().unique()])
    labels = [l for l in labels if l and l != "nan"]
    if not labels:
        return "OLS"
    unique = sorted(set(labels))
    if len(unique) == 1:
        return unique[0]
    return "/".join(unique)


def _format_cov_note(cov_label: str) -> str:
    if cov_label.strip().upper() == "OLS":
        return "OLS"
    return f"{cov_label} robust"


def _load_hierarchical(task: str) -> pd.DataFrame:
    hr_path = OUTPUT_STATS_DIR / "analysis" / task / "hierarchical_results.csv"
    if not hr_path.exists():
        print(f"  Warning: {hr_path} not found")
        return pd.DataFrame()
    hr = pd.read_csv(hr_path, encoding="utf-8-sig")
    hr["task"] = task.upper()
    return hr


def _build_selected_table(specs, hr_by_task, include_axis: bool) -> pd.DataFrame:
    rows = []
    for spec in specs:
        task = spec["task"]
        col = spec["outcome_column"]
        hr = hr_by_task.get(task.lower()) or hr_by_task.get("overall")
        if hr is None or hr.empty:
            continue
        match = hr[hr["outcome_column"] == col]
        if match.empty:
            print(f"  Warning: {task} outcome not found: {col}")
            continue
        row = match.iloc[0]
        p_val = row.get("ucla_p", row.get("p_ucla_wald", float("nan")))
        out = {
            "Task": task,
            "DV": spec["dv"],
            "b": row["ucla_beta"],
            "SE": row["ucla_se"],
            "95% CI": _format_ci(row["ucla_beta"], row["ucla_se"]),
            "p": p_val,
            "delta_R2": row["delta_r2_ucla"],
        }
        if include_axis:
            out = {
                "Task": task,
                "Axis": spec["axis"],
                "DV": spec["dv"],
                "Type": spec["type"],
                "b": row["ucla_beta"],
                "SE": row["ucla_se"],
                "95% CI": _format_ci(row["ucla_beta"], row["ucla_se"]),
                "p": p_val,
                "delta_R2": row["delta_r2_ucla"],
            }
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(rows)
    out_df["b"] = out_df["b"].round(3)
    out_df["SE"] = out_df["SE"].round(3)
    out_df["p"] = out_df["p"].round(3)
    out_df["delta_R2"] = out_df["delta_R2"].round(4)
    return out_df


def generate_table3():
    """Generate Table 3: Descriptive Statistics."""
    print("=" * 60)
    print("Generating Table 3: Descriptive Statistics")
    print("=" * 60)

    survey_ids = set(get_survey_valid_participants(RAW_DIR, SurveyQCCriteria()))
    print(f"Survey complete N: {len(survey_ids)}")

    participants = load_participants(RAW_DIR)
    participants = participants[participants["participant_id"].isin(survey_ids)].copy()

    ucla = load_ucla_scores(RAW_DIR)
    ucla = ucla[ucla["participant_id"].isin(survey_ids)].copy()

    dass = load_dass_scores(RAW_DIR)
    dass = dass[dass["participant_id"].isin(survey_ids)].copy()

    rows = []

    # Demographics and surveys (survey sample)
    rows.append(calc_stats(participants, "age", "Age (years)"))
    rows.append(calc_stats(ucla, "ucla_total", "UCLA Loneliness"))
    rows.append(calc_stats(dass, "dass_depression", "DASS-21 Depression"))
    rows.append(calc_stats(dass, "dass_anxiety", "DASS-21 Anxiety"))
    rows.append(calc_stats(dass, "dass_stress", "DASS-21 Stress"))

    overall_summary = load_overall_summary(get_results_dir("overall"))

    print(f"Overall task-complete N: {len(overall_summary)}")

    rows.append(
        calc_stats(
            overall_summary,
            "stroop_interference",
            "Stroop Interference RT (ms)",
        )
    )
    rows.append(
        calc_stats(
            overall_summary,
            "wcst_perseverative_error_rate",
            "WCST Perseverative Error Rate (%)",
        )
    )

    table3 = pd.DataFrame(rows)
    table3["Mean"] = table3["Mean"].round(2)
    table3["SD"] = table3["SD"].round(2)

    table3.to_csv(output_dir / "Table3_descriptives.csv", index=False, encoding="utf-8-sig")

    md = table3.to_markdown(index=False)
    with open(output_dir / "Table3_descriptives.md", "w", encoding="utf-8") as f:
        f.write("# Table 3. Descriptive Statistics\n\n")
        f.write("*Note.* Demographics and self-report measures based on survey-complete sample. ")
        f.write("Cognitive task metrics based on overall task-complete sample.\n\n")
        f.write(md)

    print(f"\nTable 3: {len(table3)} rows")
    print(table3.to_string(index=False))

    return table3


def generate_table4():
    """Generate Table 4: Correlation Matrix (overall sample)."""
    print("\n" + "=" * 60)
    print("Generating Table 4: Correlation Matrix")
    print("=" * 60)

    r = pd.read_csv(
        OUTPUT_STATS_DIR / "analysis" / "overall" / "correlation_matrix.csv",
        index_col=0,
        encoding="utf-8-sig",
    )
    p = pd.read_csv(
        OUTPUT_STATS_DIR / "analysis" / "overall" / "correlation_pvalues.csv",
        index_col=0,
        encoding="utf-8-sig",
    )

    out = r.copy().astype(str)
    for i in range(len(r)):
        for j in range(len(r)):
            if i <= j:
                out.iat[i, j] = ""
                continue
            r_val = r.iat[i, j]
            p_val = p.iat[i, j]
            if pd.isna(r_val) or pd.isna(p_val):
                out.iat[i, j] = ""
                continue
            star = ""
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            out.iat[i, j] = f"{r_val:.2f}{star}"

    out.to_csv(output_dir / "Table4_correlation_matrix.csv", encoding="utf-8-sig")

    with open(output_dir / "Table4_correlation_matrix.md", "w", encoding="utf-8") as f:
        f.write("# Table 4. Correlation Matrix\n\n")
        f.write("*Note.* Overall task-complete sample. Lower triangle shows Pearson correlations.\n")
        f.write("*p < .05, **p < .01, ***p < .001\n\n")
        f.write(out.to_markdown())

    print("\nTable 4 (lower triangle with stars):")
    print(out.to_string())

    return out


def generate_table5_selected():
    """Generate Table 5: Conventional indices (selected set)."""
    print("\n" + "=" * 60)
    print("Generating Table 5 (Selected): Conventional Indices")
    print("=" * 60)

    hr_by_task = {"overall": _load_hierarchical("overall")}
    cov_label = _infer_cov_type(hr_by_task)
    cov_note = _format_cov_note(cov_label)

    specs = [
        {"task": "STROOP", "outcome_column": "stroop_interference", "dv": "RT interference"},
        {
            "task": "WCST",
            "outcome_column": "wcst_perseverative_error_rate",
            "dv": "Perseverative error rate",
        },
    ]

    table = _build_selected_table(specs, hr_by_task, include_axis=False)
    if table.empty:
        print("No data for selected Table 5")
        return None

    table.to_csv(output_dir / "Table5_conventional_selected.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "Table5_conventional_selected.md", "w", encoding="utf-8") as f:
        f.write("# Table 5. Conventional Indices\n\n")
        f.write("*Note.* b = coefficient for z-scored UCLA predictor (Step 3, controlling for DASS-21).\n")
        f.write(f"95% CI computed as b +/- 1.96 * SE ({cov_note}).\n\n")
        f.write(table.to_markdown(index=False))

    print(f"\nTable 5 Selected: {len(table)} rows")
    print(table.to_string(index=False))

    return table


def generate_table6_selected():
    """Generate Table 6: Temporal dynamics indices (selected set)."""
    print("\n" + "=" * 60)
    print("Generating Table 6 (Selected): Temporal Dynamics Indices")
    print("=" * 60)

    hr_by_task = {"overall": _load_hierarchical("overall")}
    cov_label = _infer_cov_type(hr_by_task)
    cov_note = _format_cov_note(cov_label)

    specs = [
        {
            "task": "STROOP",
            "axis": "Time-on-task drift",
            "dv": "Interference slope",
            "type": "RT",
            "outcome_column": "stroop_interference_slope",
        },
        {
            "task": "WCST",
            "axis": "Phase-specific RT",
            "dv": "Confirmation RT",
            "type": "RT",
            "outcome_column": "wcst_confirmation_rt",
        },
        {
            "task": "WCST",
            "axis": "Phase-specific RT",
            "dv": "Exploitation RT",
            "type": "RT",
            "outcome_column": "wcst_exploitation_rt",
        },
        {
            "task": "WCST",
            "axis": "Phase contrast",
            "dv": "Confirmation - Exploitation",
            "type": "RT",
            "outcome_column": "wcst_confirmation_minus_exploitation_rt",
        },
    ]

    table = _build_selected_table(specs, hr_by_task, include_axis=True)
    if table.empty:
        print("No data for selected Table 6")
        return None

    table.to_csv(output_dir / "Table6_temporal_selected.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "Table6_temporal_selected.md", "w", encoding="utf-8") as f:
        f.write("# Table 6. Temporal Dynamics Indices\n\n")
        f.write("*Note.* b = coefficient for z-scored UCLA predictor (Step 3, controlling for DASS-21).\n")
        f.write(f"95% CI computed as b +/- 1.96 * SE ({cov_note}).\n\n")
        f.write(table.to_markdown(index=False))

    print(f"\nTable 6 Selected: {len(table)} rows")
    print(table.to_string(index=False))

    return table


if __name__ == "__main__":
    print("Generating Tables 3-6 for Publication")
    print("=" * 60)

    table3 = generate_table3()
    table4 = generate_table4()
    table5_selected = generate_table5_selected()
    table6_selected = generate_table6_selected()

    print("\n" + "=" * 60)
    print("All tables generated successfully!")
    print(f"Output directory: {output_dir}")
