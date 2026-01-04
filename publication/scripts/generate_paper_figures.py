"""Generate Figure 1-2 for publication."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from publication.preprocessing.constants import RAW_DIR
from publication.preprocessing.surveys import SurveyQCCriteria, get_survey_valid_participants

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Paths
repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "publication" / "data"
output_dir = data_dir / "outputs" / "paper_figures"
table_dir = data_dir / "outputs" / "paper_tables"
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 10


def _count_participants(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "participantId" in df.columns:
        return df["participantId"].nunique()
    if "participant_id" in df.columns:
        return df["participant_id"].nunique()
    return len(df)


def generate_figure1():
    """Generate Figure 1: Participant Flow Diagram."""
    print("=" * 60)
    print("Generating Figure 1: Participant Flow Diagram")
    print("=" * 60)

    n_raw = _count_participants(data_dir / "raw" / "1_participants_info.csv")

    survey_ids = get_survey_valid_participants(RAW_DIR, SurveyQCCriteria())
    n_survey = len(survey_ids)

    n_stroop = _count_participants(data_dir / "complete_stroop" / "1_participants_info.csv")
    n_prp = _count_participants(data_dir / "complete_prp" / "1_participants_info.csv")
    n_wcst = _count_participants(data_dir / "complete_wcst" / "1_participants_info.csv")
    n_all = _count_participants(data_dir / "complete_overall" / "1_participants_info.csv")

    print(f"Raw: {n_raw}")
    print(f"Survey Complete: {n_survey}")
    print(f"Stroop: {n_stroop}, PRP: {n_prp}, WCST: {n_wcst}")
    print(f"All Tasks: {n_all}")

    excluded_survey = n_raw - n_survey
    excluded_stroop = n_survey - n_stroop
    excluded_prp = n_survey - n_prp
    excluded_wcst = n_survey - n_wcst

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

    ax.text(
        5,
        13,
        f"Enrolled Participants\nN = {n_raw}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=box_props,
    )

    ax.text(
        5,
        10.5,
        f"Survey Complete\n(UCLA + DASS-21)\nN = {n_survey}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=box_props,
    )

    ax.text(
        8.5,
        11.75,
        f"Excluded:\n- Incomplete surveys\n  (n = {excluded_survey})",
        ha="left",
        va="center",
        fontsize=9,
        bbox=excl_props,
    )

    ax.text(
        2,
        7.5,
        f"Stroop Complete\nN = {n_stroop}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=box_props,
    )
    ax.text(
        5,
        7.5,
        f"PRP Complete\nN = {n_prp}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=box_props,
    )
    ax.text(
        8,
        7.5,
        f"WCST Complete\nN = {n_wcst}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=box_props,
    )

    ax.text(2, 5.8, f"Excluded: {excluded_stroop}", ha="center", va="center", fontsize=8, color="gray")
    ax.text(5, 5.8, f"Excluded: {excluded_prp}", ha="center", va="center", fontsize=8, color="gray")
    ax.text(8, 5.8, f"Excluded: {excluded_wcst}", ha="center", va="center", fontsize=8, color="gray")

    ax.text(
        5,
        4,
        f"All Tasks Complete\n(Analysis Sample)\nN = {n_all}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightgreen", edgecolor="black", linewidth=2
        ),
    )

    arrow_props = dict(arrowstyle="->", color="black", lw=1.5)

    ax.annotate("", xy=(5, 11.5), xytext=(5, 12.2), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 8.5), xytext=(4, 9.5), arrowprops=arrow_props)
    ax.annotate("", xy=(5, 8.5), xytext=(5, 9.5), arrowprops=arrow_props)
    ax.annotate("", xy=(8, 8.5), xytext=(6, 9.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 4.8), xytext=(2, 6.5), arrowprops=arrow_props)
    ax.annotate("", xy=(5, 4.8), xytext=(5, 6.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6, 4.8), xytext=(8, 6.5), arrowprops=arrow_props)

    ax.text(
        5,
        1.5,
        "Figure 1. Participant Flow Diagram",
        ha="center",
        va="center",
        fontsize=12,
        fontstyle="italic",
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure1_participant_flow.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    print(f"Figure 1 saved to {output_dir / 'Figure1_participant_flow.png'}")


def generate_figure2():
    """Generate Figure 2: delta R^2 comparison (bar + forest plot)."""
    print("\n" + "=" * 60)
    print("Generating Figure 2: delta R^2 Comparison")
    print("=" * 60)

    table5 = pd.read_csv(table_dir / "Table5_conventional_full.csv", encoding="utf-8-sig")
    table6 = pd.read_csv(table_dir / "Table6_temporal_full.csv", encoding="utf-8-sig")

    table5["Approach"] = "Conventional"
    table6["Approach"] = "Temporal"

    df = pd.concat([table5, table6], ignore_index=True)

    print(f"Total outcomes: {len(df)} (Conventional: {len(table5)}, Temporal: {len(table6)})")

    fig, ax = plt.subplots(figsize=(8, 5))

    summary = df.groupby(["Task", "Approach"])["delta_R2"].mean().reset_index()
    summary_pivot = summary.pivot(index="Task", columns="Approach", values="delta_R2")

    x = np.arange(len(summary_pivot.index))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        summary_pivot["Conventional"],
        width,
        label="Conventional",
        color="steelblue",
    )
    bars2 = ax.bar(
        x + width / 2,
        summary_pivot["Temporal"],
        width,
        label="Temporal",
        color="coral",
    )

    ax.set_ylabel("Mean delta R^2 (UCLA step)", fontsize=11)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_pivot.index)
    ax.legend(title="Approach")
    ax.set_title("Figure 2A. Mean Incremental R^2 by Task and Approach", fontsize=12)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure2_delta_r2_bar.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print("Figure 2A (bar chart) saved")

    df_plot = df.copy()
    df_plot["significant"] = df_plot["p"] < 0.05
    df_plot = df_plot.sort_values(["Approach", "Task", "delta_R2"], ascending=[True, True, False])

    df_notable = df_plot[(df_plot["significant"]) | (df_plot["delta_R2"] > 0.02)]
    if len(df_notable) < 5:
        df_notable = df_plot.nlargest(20, "delta_R2")

    fig, ax = plt.subplots(figsize=(10, max(8, len(df_notable) * 0.3)))

    colors = {"Conventional": "steelblue", "Temporal": "coral"}
    markers = {True: "s", False: "o"}

    for i, (_, row) in enumerate(df_notable.iterrows()):
        color = colors[row["Approach"]]
        marker = markers[row["significant"]]
        size = 80 if row["significant"] else 40

        ax.scatter(row["delta_R2"], i, c=color, marker=marker, s=size, zorder=3)

        label = f"{row['Task']}: {row['Outcome'][:30]}"
        if row["significant"]:
            label += " *"
        ax.text(-0.005, i, label, ha="right", va="center", fontsize=8)

    ax.axvline(0, color="gray", linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("delta R^2 (UCLA step)", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Figure 2B. Forest Plot of Incremental R^2 by Outcome", fontsize=12)

    conv_patch = mpatches.Patch(color="steelblue", label="Conventional")
    temp_patch = mpatches.Patch(color="coral", label="Temporal")
    sig_marker = plt.Line2D(
        [0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="p < .05"
    )
    nonsig_marker = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="p >= .05"
    )
    ax.legend(handles=[conv_patch, temp_patch, sig_marker, nonsig_marker], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure2_delta_r2_forest.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print("Figure 2B (forest plot) saved")
    print(f"Notable outcomes shown: {len(df_notable)}")


if __name__ == "__main__":
    print("Generating Figures 1-2 for Publication")
    print("=" * 60)

    generate_figure1()
    generate_figure2()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
