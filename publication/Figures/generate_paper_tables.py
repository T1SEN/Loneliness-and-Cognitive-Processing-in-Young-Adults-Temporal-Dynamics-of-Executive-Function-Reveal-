"""Generate Table 3-6 for publication."""
import sys
from pathlib import Path

import pandas as pd

from publication.preprocessing.surveys import (
    SurveyQCCriteria,
    get_survey_valid_participants,
    load_dass_scores,
    load_participants,
    load_ucla_scores,
)
from publication.preprocessing.constants import RAW_DIR, get_results_dir
from publication.preprocessing.stroop.loaders import load_stroop_summary
from publication.preprocessing.wcst.loaders import load_wcst_summary

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Paths
repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "publication" / "data"
output_dir = data_dir / "outputs" / "paper_tables"
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


def format_regression_table(df):
    cols = [
        "task",
        "outcome",
        "n",
        "ucla_beta",
        "ucla_se",
        "ucla_t",
        "p_ucla_wald",
        "delta_r2_ucla",
    ]
    out = df[cols].copy()
    out.columns = ["Task", "Outcome", "N", "b", "SE", "t", "p", "delta_R2"]
    out["b"] = out["b"].round(3)
    out["SE"] = out["SE"].round(3)
    out["t"] = out["t"].round(2)
    out["p"] = out["p"].round(3)
    out["delta_R2"] = out["delta_R2"].round(4)
    return out


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
    hr_path = data_dir / "outputs" / "analysis" / task / "hierarchical_results.csv"
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
        hr = hr_by_task.get(task.lower())
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

    stroop_summary = load_stroop_summary(get_results_dir("stroop"))
    wcst_summary = load_wcst_summary(get_results_dir("wcst"))

    print(
        f"Stroop N: {len(stroop_summary)}, "
        f"WCST N: {len(wcst_summary)}"
    )

    rows.append(
        calc_stats(
            stroop_summary,
            "stroop_interference",
            "Stroop Interference Effect (ms)",
        )
    )
    rows.append(calc_stats(wcst_summary, "pe_rate", "WCST Perseverative Error Rate (%)"))

    table3 = pd.DataFrame(rows)
    table3["Mean"] = table3["Mean"].round(2)
    table3["SD"] = table3["SD"].round(2)

    table3.to_csv(output_dir / "Table3_descriptives.csv", index=False, encoding="utf-8-sig")

    md = table3.to_markdown(index=False)
    with open(output_dir / "Table3_descriptives.md", "w", encoding="utf-8") as f:
        f.write("# Table 3. Descriptive Statistics\n\n")
        f.write("*Note.* Demographics and self-report measures based on survey-complete sample. ")
        f.write("Cognitive task metrics based on task-specific samples.\n\n")
        f.write(md)

    # Gender distribution
    gender = participants["gender"].astype(str).str.strip().str.lower()
    male_n = (gender == "male").sum()
    female_n = (gender == "female").sum()
    total_n = len(participants)

    gender_df = pd.DataFrame(
        {
            "Variable": ["Gender"],
            "Male_N": [male_n],
            "Female_N": [female_n],
            "Male_Percent": [round(100 * male_n / total_n, 1)],
            "Female_Percent": [round(100 * female_n / total_n, 1)],
        }
    )
    gender_df.to_csv(output_dir / "Table3_gender.csv", index=False, encoding="utf-8-sig")

    print("\nTable 3 generated:")
    print(table3.to_string(index=False))
    print(
        f"\nGender: Male {male_n} ({100*male_n/total_n:.1f}%), "
        f"Female {female_n} ({100*female_n/total_n:.1f}%)"
    )

    return table3


def generate_table4():
    """Generate Table 4: Correlation Matrix (overall sample)."""
    print("\n" + "=" * 60)
    print("Generating Table 4: Correlation Matrix")
    print("=" * 60)

    r = pd.read_csv(
        data_dir / "outputs" / "analysis" / "overall" / "correlation_matrix.csv",
        index_col=0,
        encoding="utf-8-sig",
    )
    p = pd.read_csv(
        data_dir / "outputs" / "analysis" / "overall" / "correlation_pvalues.csv",
        index_col=0,
        encoding="utf-8-sig",
    )

    print(f"Correlation matrix shape: {r.shape}")
    print(f"Variables: {list(r.columns)}")

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
        f.write("*Note.* N = 197 (all tasks complete). Lower triangle shows Pearson correlations.\n")
        f.write("*p < .05, **p < .01, ***p < .001\n\n")
        f.write(out.to_markdown())

    print("\nTable 4 (lower triangle with stars):")
    print(out.to_string())

    return out


def get_feature_columns(task, feature_type):
    """Get column names from feature files."""
    suffix_map = {
        "traditional": "traditional",
        "dynamic_drift": "dynamic_drift",
        "dynamic_dispersion": "dynamic_dispersion",
        "dynamic_recovery": "dynamic_recovery",
    }
    suffix = suffix_map.get(feature_type, feature_type)
    path = data_dir / f"complete_{task}" / f"5_{task}_{suffix}_features.csv"

    if not path.exists():
        print(f"  Warning: {path} not found")
        return []

    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in ["participant_id", "participantId"]]
    return cols


def generate_table5():
    """Generate Table 5: Conventional indices regression results."""
    print("\n" + "=" * 60)
    print("Generating Table 5: Conventional Indices")
    print("=" * 60)

    conventional_extras = {
        "wcst": ["pe_rate"],
        "stroop": [],
    }

    all_rows = []

    for task in ["stroop", "wcst"]:
        hr_path = data_dir / "outputs" / "analysis" / task / "hierarchical_results.csv"
        if not hr_path.exists():
            print(f"  Warning: {hr_path} not found")
            continue

        hr = pd.read_csv(hr_path, encoding="utf-8-sig")

        trad_cols = get_feature_columns(task, "traditional")
        extra_cols = conventional_extras.get(task, [])
        conv_cols = set(trad_cols + extra_cols)

        print(f"  {task.upper()}: {len(conv_cols)} conventional columns")

        filtered = hr[hr["outcome_column"].isin(conv_cols)].copy()
        if len(filtered) == 0:
            print("    No matching rows found in hierarchical_results")
            continue

        filtered["task"] = task.upper()
        all_rows.append(filtered)

    if not all_rows:
        print("No data for Table 5")
        return None

    table5 = pd.concat(all_rows, ignore_index=True)

    key_outcomes = ["stroop_rt_interference", "pe_rate"]
    summary_mask = (table5["p_ucla_wald"] < 0.05) | (
        table5["outcome_column"].isin(key_outcomes)
    )
    summary = table5[summary_mask].copy()
    summary = summary.sort_values(["task", "p_ucla_wald"])

    table5_out = format_regression_table(table5)
    summary_out = format_regression_table(summary)

    table5_out.to_csv(output_dir / "Table5_conventional_full.csv", index=False, encoding="utf-8-sig")
    summary_out.to_csv(
        output_dir / "Table5_conventional_summary.csv", index=False, encoding="utf-8-sig"
    )

    with open(output_dir / "Table5_conventional_summary.md", "w", encoding="utf-8") as f:
        f.write("# Table 5. Hierarchical Regression Results: Conventional Indices\n\n")
        f.write("*Note.* b = coefficient for z-scored UCLA predictor (Step 3, controlling for DASS-21).\n")
        f.write("delta_R2 = incremental R^2 for UCLA step. Key outcomes and any significant results shown.\n\n")
        f.write(summary_out.to_markdown(index=False))

    print(f"\nTable 5 Full: {len(table5_out)} rows")
    print(f"Table 5 Summary: {len(summary_out)} rows")
    print("\nSummary:")
    print(summary_out.to_string(index=False))

    return table5_out


def generate_table6():
    """Generate Table 6: Temporal dynamics indices regression results."""
    print("\n" + "=" * 60)
    print("Generating Table 6: Temporal Dynamics Indices")
    print("=" * 60)

    all_rows = []

    for task in ["stroop", "wcst"]:
        hr_path = data_dir / "outputs" / "analysis" / task / "hierarchical_results.csv"
        if not hr_path.exists():
            continue

        hr = pd.read_csv(hr_path, encoding="utf-8-sig")

        temporal_cols = set()
        for ftype in ["dynamic_drift", "dynamic_dispersion", "dynamic_recovery"]:
            temporal_cols.update(get_feature_columns(task, ftype))

        print(f"  {task.upper()}: {len(temporal_cols)} temporal columns")

        filtered = hr[hr["outcome_column"].isin(temporal_cols)].copy()
        if len(filtered) == 0:
            continue

        filtered["task"] = task.upper()
        all_rows.append(filtered)

    if not all_rows:
        print("No data for Table 6")
        return None

    table6 = pd.concat(all_rows, ignore_index=True)
    table6_out = format_regression_table(table6)

    table6_out.to_csv(output_dir / "Table6_temporal_full.csv", index=False, encoding="utf-8-sig")

    summary = table6[table6["p_ucla_wald"] < 0.05].sort_values(["task", "p_ucla_wald"])
    summary_out = format_regression_table(summary)
    summary_out.to_csv(
        output_dir / "Table6_temporal_summary.csv", index=False, encoding="utf-8-sig"
    )

    with open(output_dir / "Table6_temporal_summary.md", "w", encoding="utf-8") as f:
        f.write("# Table 6. Hierarchical Regression Results: Temporal Dynamics Indices\n\n")
        f.write("*Note.* b = coefficient for z-scored UCLA predictor (Step 3, controlling for DASS-21).\n")
        f.write("delta_R2 = incremental R^2 for UCLA step. Only significant results (p < .05) shown.\n\n")
        f.write(summary_out.to_markdown(index=False))

    print(f"\nTable 6 Full: {len(table6_out)} rows")
    print(f"Table 6 Summary: {len(summary_out)} rows")
    print("\nSummary:")
    print(summary_out.to_string(index=False))

    return table6_out


def generate_table5_selected():
    """Generate Table 5: Conventional indices (selected set)."""
    print("\n" + "=" * 60)
    print("Generating Table 5 (Selected): Conventional Indices")
    print("=" * 60)

    hr_by_task = {t: _load_hierarchical(t) for t in ["stroop", "wcst"]}
    cov_label = _infer_cov_type(hr_by_task)
    cov_note = _format_cov_note(cov_label)

    specs = [
        {"task": "STROOP", "outcome_column": "stroop_rt_interference", "dv": "RT interference"},
        {"task": "STROOP", "outcome_column": "stroop_acc_interference", "dv": "Accuracy interference"},
        {"task": "WCST", "outcome_column": "wcst_categories_completed", "dv": "Categories completed"},
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
        f.write(f"95% CI computed as b ± 1.96 × SE ({cov_note}).\n\n")
        f.write(table.to_markdown(index=False))

    print(f"\nTable 5 Selected: {len(table)} rows")
    print(table.to_string(index=False))

    return table


def generate_table6_selected():
    """Generate Table 6: Temporal dynamics indices (selected set)."""
    print("\n" + "=" * 60)
    print("Generating Table 6 (Selected): Temporal Dynamics Indices")
    print("=" * 60)

    hr_by_task = {t: _load_hierarchical(t) for t in ["stroop", "wcst"]}
    cov_label = _infer_cov_type(hr_by_task)
    cov_note = _format_cov_note(cov_label)

    specs = [
        {
            "task": "STROOP",
            "axis": "Time-on-task drift",
            "dv": "RT slope (incongruent)",
            "type": "RT",
            "outcome_column": "stroop_incong_slope",
        },
        {
            "task": "STROOP",
            "axis": "Within-task dispersion",
            "dv": "RT SD (incongruent)",
            "type": "RT",
            "outcome_column": "stroop_rt_sd_incong",
        },
        {
            "task": "WCST",
            "axis": "Time-on-task drift",
            "dv": "RT slope (within-category)",
            "type": "RT",
            "outcome_column": "wcst_rt_slope_within_category",
        },
        {
            "task": "STROOP",
            "axis": "Post-perturbation recovery",
            "dv": "Post-error slowing",
            "type": "RT",
            "outcome_column": "stroop_post_error_slowing",
        },
        {
            "task": "STROOP",
            "axis": "Post-perturbation recovery",
            "dv": "Post-error accuracy drop",
            "type": "Acc",
            "outcome_column": "stroop_post_error_acc_diff",
        },
        {
            "task": "WCST",
            "axis": "Post-perturbation recovery",
            "dv": "Post-switch RT cost (k1)",
            "type": "RT",
            "outcome_column": "wcst_switch_cost_rt_k1",
        },
        {
            "task": "WCST",
            "axis": "Post-perturbation recovery",
            "dv": "Post-error accuracy",
            "type": "Acc",
            "outcome_column": "wcst_post_error_accuracy",
        },
        {
            "task": "WCST",
            "axis": "Post-perturbation recovery",
            "dv": "Post-switch error rate",
            "type": "Acc",
            "outcome_column": "wcst_post_switch_error_rate",
        },
        {
            "task": "WCST",
            "axis": "Post-perturbation recovery",
            "dv": "Trials to reacquisition",
            "type": "Acc",
            "outcome_column": "wcst_trials_to_rule_reacquisition",
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
        f.write(f"95% CI computed as b ± 1.96 × SE ({cov_note}).\n\n")
        f.write(table.to_markdown(index=False))

    print(f"\nTable 6 Selected: {len(table)} rows")
    print(table.to_string(index=False))

    return table


if __name__ == "__main__":
    print("Generating Tables 3-6 for Publication")
    print("=" * 60)

    table3 = generate_table3()
    table4 = generate_table4()
    table5 = generate_table5()
    table6 = generate_table6()
    table5_selected = generate_table5_selected()
    table6_selected = generate_table6_selected()

    print("\n" + "=" * 60)
    print("All tables generated successfully!")
    print(f"Output directory: {output_dir}")
