"""Generate Figures 1-3 for static.

Figure 1: Participant Flow Diagram (CONSORT style)
Figure 2: UCLA-DASS Correlation Heatmap (4x4)
Figure 3: Conventional vs Temporal Forest Plot (Standardized beta)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from static.analysis.utils import STANDARDIZED_PREDICTORS, get_analysis_data, get_figures_dir
from static.preprocessing.constants import OUTPUT_FIGURES_DIR, OUTPUT_STATS_DIR, RAW_DIR

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Paths
repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data"
figures_dir = get_figures_dir()
output_dir = OUTPUT_FIGURES_DIR
stats_dir = OUTPUT_STATS_DIR / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 10

# =============================================================================
# Primary Indicators for Figure 3
# =============================================================================

# Conventional indicators (core manuscript outcomes)
CONV_PRIMARY = [
    ("STROOP", "stroop_interference", "Stroop Interference RT"),
    ("WCST", "wcst_perseverative_error_rate", "WCST PE Rate"),
]

# Temporal dynamics indicators (core manuscript outcomes)
TEMP_PRIMARY = [
    ("STROOP", "stroop_interference_slope", "Stroop Interference Slope"),
    ("WCST", "wcst_confirmation_rt", "WCST Confirmation RT"),
    ("WCST", "wcst_exploitation_rt", "WCST Exploitation RT"),
    ("WCST", "wcst_confirmation_minus_exploitation_rt", "WCST Confirm-Exploit RT"),
]


def _count_participants(path):
    """Count unique participants in a CSV file."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "participantId" in df.columns:
        return df["participantId"].nunique()
    if "participant_id" in df.columns:
        return df["participant_id"].nunique()
    return len(df)

def _load_participant_ids(path):
    """Load unique participant ids from a CSV file."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "participantId" in df.columns:
        return set(df["participantId"].dropna().astype(str))
    if "participant_id" in df.columns:
        return set(df["participant_id"].dropna().astype(str))
    return set()


def generate_figure1():
    """Generate Figure 1: Participant Flow Diagram (CONSORT style)."""
    print("=" * 60)
    print("Generating Figure 1: Participant Flow Diagram")
    print("=" * 60)

    from static.preprocessing.surveys import SurveyQCCriteria, get_survey_valid_participants

    n_raw = _count_participants(data_dir / "raw" / "1_participants_info.csv")
    survey_ids = {str(pid) for pid in get_survey_valid_participants(RAW_DIR, SurveyQCCriteria())}
    n_survey = len(survey_ids)

    stroop_attempted = _load_participant_ids(data_dir / "raw" / "4a_stroop_trials.csv") & survey_ids
    wcst_attempted = _load_participant_ids(data_dir / "raw" / "4b_wcst_trials.csv") & survey_ids

    n_stroop_attempted = len(stroop_attempted)
    n_wcst_attempted = len(wcst_attempted)

    summary_path = data_dir / "raw" / "3_cognitive_tests_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    else:
        summary = pd.DataFrame()

    if not summary.empty and "participantId" in summary.columns and "testName" in summary.columns:
        summary = summary[summary["participantId"].astype(str).isin(survey_ids)].copy()
        summary["testName"] = summary["testName"].astype(str).str.lower()
        stroop_complete_ids = set(summary[summary["testName"] == "stroop"]["participantId"].dropna().astype(str))
        wcst_complete_ids = set(summary[summary["testName"] == "wcst"]["participantId"].dropna().astype(str))
    else:
        stroop_complete_ids = set()
        wcst_complete_ids = set()

    n_stroop = len(stroop_complete_ids)
    n_wcst = len(wcst_complete_ids)
    n_all = len(_load_participant_ids(data_dir / "complete_overall" / "filtered_participant_ids.csv"))

    print(f"Raw: {n_raw}")
    print(f"Survey Complete: {n_survey}")
    print(f"Stroop attempted: {n_stroop_attempted}, complete: {n_stroop}")
    print(f"WCST attempted: {n_wcst_attempted}, complete: {n_wcst}")
    print(f"All Tasks Complete (overall sample): {n_all}")

    excluded_survey = n_raw - n_survey
    stroop_not_attempted = max(0, n_survey - n_stroop_attempted)
    wcst_not_attempted = max(0, n_survey - n_wcst_attempted)
    excluded_stroop = max(0, n_stroop_attempted - n_stroop)
    excluded_wcst = max(0, n_wcst_attempted - n_wcst)

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    box_props = dict(
        boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black", linewidth=2
    )
    excl_props = dict(
        boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", linewidth=1
    )

    # Enrolled
    ax.text(
        5, 13, f"Enrolled Participants\nN = {n_raw}",
        ha="center", va="center", fontsize=12, fontweight="bold", bbox=box_props,
    )

    # Survey complete
    ax.text(
        5, 10.5, f"Survey Complete\n(UCLA + DASS-21)\nN = {n_survey}",
        ha="center", va="center", fontsize=11, fontweight="bold", bbox=box_props,
    )

    # Exclusion: Survey
    ax.text(
        8.5, 11.75,
        f"Excluded (n = {excluded_survey}):\n- Incomplete surveys\n- Missing gender\n- Date filter",
        ha="left", va="center", fontsize=9, bbox=excl_props,
    )

    # Task boxes
    ax.text(
        3, 7.5, f"Stroop Complete\n(survey-eligible)\nN = {n_stroop}",
        ha="center", va="center", fontsize=10, fontweight="bold", bbox=box_props,
    )
    ax.text(
        7, 7.5, f"WCST Complete\n(survey-eligible)\nN = {n_wcst}",
        ha="center", va="center", fontsize=10, fontweight="bold", bbox=box_props,
    )

    # Task exclusions with QC reasons
    ax.text(
        3, 5.5,
        f"Not attempted: {stroop_not_attempted}\nNot complete: {excluded_stroop}",
        ha="center", va="center", fontsize=8, color="gray"
    )
    ax.text(
        7, 5.5,
        f"Not attempted: {wcst_not_attempted}\nNot complete: {excluded_wcst}",
        ha="center", va="center", fontsize=8, color="gray"
    )

    # Final sample
    ax.text(
        5, 3.5, f"All Tasks Complete\n(overall sample)\nN = {n_all}",
        ha="center", va="center", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", edgecolor="black", linewidth=2),
    )

    # Arrows
    arrow_props = dict(arrowstyle="->", color="black", lw=1.5)
    ax.annotate("", xy=(5, 11.5), xytext=(5, 12.2), arrowprops=arrow_props)
    ax.annotate("", xy=(3, 8.5), xytext=(4, 9.5), arrowprops=arrow_props)
    ax.annotate("", xy=(7, 8.5), xytext=(6, 9.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 4.3), xytext=(3, 6.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6, 4.3), xytext=(7, 6.5), arrowprops=arrow_props)

    # Caption
    ax.text(
        5, 1.5, "Figure 1. Participant Flow Diagram",
        ha="center", va="center", fontsize=12, fontstyle="italic",
    )

    plt.tight_layout()
    plt.savefig(
        figures_dir / "Figure1_participant_flow.png",
        dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none",
    )
    plt.close()
    print(f"Figure 1 saved to {figures_dir / 'Figure1_participant_flow.png'}")


def generate_figure2():
    """Generate Figure 2: UCLA-DASS Correlation Heatmap (4x4)."""
    print("\n" + "=" * 60)
    print("Generating Figure 2: UCLA-DASS Correlation Heatmap")
    print("=" * 60)

    corr_path = stats_dir / "overall" / "correlation_matrix.csv"
    pval_path = stats_dir / "overall" / "correlation_pvalues.csv"

    if not corr_path.exists():
        print(f"WARNING: {corr_path} not found. Skipping Figure 2.")
        return

    corr = pd.read_csv(corr_path, index_col=0)
    pval = pd.read_csv(pval_path, index_col=0) if pval_path.exists() else None

    # Extract UCLA-DASS only (4x4)
    vars_keep = ["UCLA", "DASS-Dep", "DASS-Anx", "DASS-Str"]
    vars_available = [v for v in vars_keep if v in corr.index]

    if len(vars_available) < 4:
        print(f"WARNING: Missing variables. Available: {corr.index.tolist()}")
        return

    sub_corr = corr.loc[vars_keep, vars_keep]

    # Create annotation matrix with significance markers
    annot_matrix = sub_corr.copy().astype(str)
    for i in range(len(vars_keep)):
        for j in range(len(vars_keep)):
            r = sub_corr.iloc[i, j]
            if i == j:
                annot_matrix.iloc[i, j] = ""
            elif pval is not None:
                p = pval.loc[vars_keep[i], vars_keep[j]]
                if p < 0.001:
                    annot_matrix.iloc[i, j] = f"{r:.2f}***"
                elif p < 0.01:
                    annot_matrix.iloc[i, j] = f"{r:.2f}**"
                elif p < 0.05:
                    annot_matrix.iloc[i, j] = f"{r:.2f}*"
                else:
                    annot_matrix.iloc[i, j] = f"{r:.2f}"
            else:
                annot_matrix.iloc[i, j] = f"{r:.2f}"

    fig, ax = plt.subplots(figsize=(5, 4))

    # Lower triangle mask
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))

    sns.heatmap(
        sub_corr, annot=annot_matrix, fmt="", cmap="RdBu_r",
        vmin=-1, vmax=1, mask=mask, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "r"},
        ax=ax
    )

    ax.set_title("Zero-Order Correlations: UCLA and DASS-21", fontsize=11)

    # Note
    fig.text(
        0.5, 0.02, "Note: ***p < .001, **p < .01, *p < .05",
        ha="center", fontsize=8, style="italic"
    )

    plt.tight_layout()
    plt.savefig(
        figures_dir / "Figure2_ucla_dass_heatmap.png",
        dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none",
    )
    plt.close()
    print(f"Figure 2 saved to {figures_dir / 'Figure2_ucla_dass_heatmap.png'}")


def generate_figure3():
    """Generate Figure 3: Conventional vs Temporal Forest Plot (Standardized beta)."""
    print("\n" + "=" * 60)
    print("Generating Figure 3: Forest Plot (Standardized beta)")
    print("=" * 60)

    hr_path = stats_dir / "overall" / "hierarchical_results.csv"
    if not hr_path.exists():
        print(f"WARNING: {hr_path} not found. Skipping Figure 3.")
        return

    hr = pd.read_csv(hr_path, encoding="utf-8-sig")
    hr["task"] = "overall"
    print(f"Loaded {len(hr)} outcomes from hierarchical results")

    # Filter to Primary indicators
    rows = []

    for task, outcome_col, label in CONV_PRIMARY:
        match = hr[hr["outcome_column"] == outcome_col]
        if len(match) > 0:
            row = match.iloc[0].to_dict()
            row["task"] = "overall"
            row["label"] = label
            row["group"] = "Conventional"
            rows.append(row)
        else:
            print(f"  Missing: {task} - {outcome_col}")

    for task, outcome_col, label in TEMP_PRIMARY:
        match = hr[hr["outcome_column"] == outcome_col]
        if len(match) > 0:
            row = match.iloc[0].to_dict()
            row["task"] = "overall"
            row["label"] = label
            row["group"] = "Temporal"
            rows.append(row)
        else:
            print(f"  Missing: {task} - {outcome_col}")

    if not rows:
        print("ERROR: No primary indicators matched. Skipping Figure 3.")
        return

    df = pd.DataFrame(rows)
    print(f"Matched {len(df)} primary indicators")

    # Compute standardized effects (DV scaled by SD within task sample)
    task_cache = {}

    def _get_task_data(task_name):
        if task_name not in task_cache:
            task_cache[task_name] = get_analysis_data(task_name)
        return task_cache[task_name]

    def _get_outcome_sd(task_name, outcome_col):
        data = _get_task_data(task_name)
        required = [outcome_col] + STANDARDIZED_PREDICTORS + ["gender_male"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            return np.nan
        subset = data.dropna(subset=required)
        if subset.empty:
            return np.nan
        return subset[outcome_col].std(ddof=1)

    df["outcome_sd"] = df.apply(
        lambda row: _get_outcome_sd(row["task"].lower(), row["outcome_column"]), axis=1
    )
    df["beta"] = df["ucla_beta"] / df["outcome_sd"]
    df["se"] = df["ucla_se"] / df["outcome_sd"]
    df.loc[df["outcome_sd"] <= 0, ["beta", "se"]] = np.nan
    df["ci_low"] = df["beta"] - 1.96 * df["se"]
    df["ci_high"] = df["beta"] + 1.96 * df["se"]
    df["p"] = df["p_ucla_wald"]
    df["delta_r2"] = df["delta_r2_ucla"]

    # Sort: Conventional first, then Temporal; within each group, by delta_r2 descending
    df["group_order"] = df["group"].map({"Conventional": 0, "Temporal": 1})
    df = df.sort_values(["group_order", "delta_r2"], ascending=[True, False]).reset_index(drop=True)

    # Calculate positions
    n_total = len(df)
    n_conv = len(df[df["group"] == "Conventional"])

    fig, ax = plt.subplots(figsize=(9, 10))

    # Y positions (top to bottom)
    y_positions = list(range(n_total))[::-1]

    # Draw separator line between Conventional and Temporal
    if n_conv > 0 and n_conv < n_total:
        separator_y = y_positions[n_conv - 1] - 0.5
        ax.axhline(separator_y, color="gray", linestyle="-", lw=0.5, alpha=0.7)

    # Draw forest plot
    label_transform = ax.get_yaxis_transform()
    for idx, (_, row) in enumerate(df.iterrows()):
        y = y_positions[idx]

        # CI line
        ax.plot(
            [row["ci_low"], row["ci_high"]], [y, y],
            color="gray", lw=1.5, solid_capstyle="round"
        )

        # Point (size proportional to delta_r2)
        size = 60 + 2500 * max(0, row["delta_r2"])
        color = "steelblue" if row["group"] == "Conventional" else "coral"
        ax.scatter(
            row["beta"], y, s=size, c=color, zorder=3,
            edgecolor="white", linewidth=0.5
        )

        # Significance marker
        if row["p"] < 0.01:
            sig_text = "**"
        elif row["p"] < 0.05:
            sig_text = "*"
        else:
            sig_text = ""

        # Label on left
        label_text = f"{row['label']}"
        if sig_text:
            label_text += f" {sig_text}"
        ax.text(-0.02, y, label_text, transform=label_transform, ha="right", va="center", fontsize=9)

        # Beta and CI on right
        ci_text = f"{row['beta']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        ax.text(1.02, y, ci_text, transform=label_transform, ha="left", va="center", fontsize=8, color="gray")

    # Zero reference line
    ax.axvline(0, color="black", linestyle="--", lw=1, alpha=0.7)

    # Group labels on far left
    if n_conv > 0:
        conv_center_y = np.mean(y_positions[:n_conv])
        ax.text(
            -0.12, conv_center_y, "CONVENTIONAL",
            transform=label_transform, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold"
        )

    if n_conv < n_total:
        temp_center_y = np.mean(y_positions[n_conv:])
        ax.text(
            -0.12, temp_center_y, "TEMPORAL",
            transform=label_transform, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold"
        )

    # Axis settings
    ax.set_xlabel("Standardized Beta (DV z-scored; UCLA effect controlling DASS)", fontsize=11)
    finite_ci = df[["ci_low", "ci_high"]].to_numpy().astype(float)
    finite_ci = finite_ci[np.isfinite(finite_ci)]
    if finite_ci.size:
        min_ci = float(np.min(finite_ci))
        max_ci = float(np.max(finite_ci))
        span = max_ci - min_ci
        pad = 0.08 * span if span > 0 else 0.1
        ax.set_xlim(min_ci - pad, max_ci + pad)
    else:
        ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, n_total - 0.5)
    ax.set_yticks([])

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=10, label='Conventional'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='coral',
               markersize=10, label='Temporal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='Point size = Delta R^2'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Title
    ax.set_title("Figure 3. UCLA Effects on Executive Function Indices", fontsize=12)

    fig.subplots_adjust(left=0.35, right=0.82)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "Figure3_forest_plot.png",
        dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none",
    )
    plt.close()

    # Save data table
    df_export = df[
        [
            "task",
            "group",
            "label",
            "outcome",
            "outcome_column",
            "n",
            "beta",
            "se",
            "ci_low",
            "ci_high",
            "p",
            "delta_r2",
            "outcome_sd",
            "ucla_beta",
            "ucla_se",
        ]
    ].copy()
    df_export.to_csv(output_dir / "Figure3_forest_data.csv", index=False, encoding="utf-8-sig")

    print(f"Figure 3 saved to {figures_dir / 'Figure3_forest_plot.png'}")
    print(f"Data table saved to {output_dir / 'Figure3_forest_data.csv'}")

    # Summary statistics
    n_sig_conv = len(df[(df["group"] == "Conventional") & (df["p"] < 0.05)])
    n_sig_temp = len(df[(df["group"] == "Temporal") & (df["p"] < 0.05)])
    print(f"\nSummary:")
    print(f"  Conventional: {n_sig_conv}/{n_conv} significant (p < .05)")
    print(f"  Temporal: {n_sig_temp}/{n_total - n_conv} significant (p < .05)")


if __name__ == "__main__":
    print("Generating Figures 1-3 for Publication")
    print("=" * 60)

    generate_figure1()
    generate_figure2()
    generate_figure3()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Figures directory: {figures_dir}")
    print(f"Data directory: {output_dir}")

