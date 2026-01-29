from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_output_dir, get_figures_dir

LOW_LABEL = "Low loneliness (bottom 25%)"
HIGH_LABEL = "High loneliness (top 25%)"


def _trend_line(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_vals)
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x_vals[mask], y_vals[mask], 1)
    return float(slope), float(intercept)


def _load_summary(output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "stroop_interference_quartile_loneliness_extremes25_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path, encoding="utf-8-sig")

    table_path = output_dir / "stroop_rt_timeseg4_loneliness_extremes25_table.csv"
    if not table_path.exists():
        raise FileNotFoundError("Missing stroop interference summary/table files.")

    table = pd.read_csv(table_path, encoding="utf-8-sig")
    summary = table.rename(
        columns={
            "segment": "quartile",
            "mean_interference_rt": "mean_interference",
            "sem_interference_rt": "sem_interference",
        }
    )
    summary = summary[["loneliness_group", "quartile", "mean_interference", "sem_interference", "n"]]
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output_dir = get_output_dir("stroop")
    figures_dir = get_figures_dir()

    summary = _load_summary(output_dir)
    summary = summary.dropna(subset=["quartile"]).copy()
    summary["quartile"] = pd.to_numeric(summary["quartile"], errors="coerce")

    segments = [1, 2, 3, 4]
    x = np.array(segments, dtype=float)

    group_data: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for group in (HIGH_LABEL, LOW_LABEL):
        group_df = summary[summary["loneliness_group"] == group]
        means = []
        sems = []
        n_val = int(group_df["n"].max()) if not group_df.empty else 0
        for seg in segments:
            row = group_df[group_df["quartile"] == seg]
            means.append(float(row["mean_interference"].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row["sem_interference"].iloc[0]) if not row.empty else np.nan)
        group_data[group] = (np.array(means, dtype=float), np.array(sems, dtype=float), n_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {LOW_LABEL: "#1f77b4", HIGH_LABEL: "#d62728"}

    slopes_rows = []
    for group in (HIGH_LABEL, LOW_LABEL):
        means, sems, n_val = group_data[group]
        overall_mean = float(np.nanmean(means)) if np.isfinite(means).any() else np.nan
        ax.errorbar(
            x,
            means,
            yerr=sems,
            color=colors[group],
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"{group} (n={n_val})",
        )
        if np.isfinite(overall_mean):
            ax.hlines(
                overall_mean,
                x.min(),
                x.max(),
                colors=colors[group],
                linestyles=":",
                linewidth=2,
                alpha=0.8,
                label=f"{group} mean ({overall_mean:.1f} ms)",
            )
        slope, _ = _trend_line(x, means)
        slopes_rows.append({
            "loneliness_group": group,
            "slope_ms_per_segment": slope,
            "mean_interference": overall_mean,
        })

    ax.set_title("Stroop Interference RT by Quartile (Top/Bottom 25%)")
    ax.set_xlabel("Quartile (time on task)")
    ax.set_ylabel("Stroop interference RT (incongruent - congruent, ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in segments])
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = figures_dir / "stroop_interference_quartile_loneliness_extremes25_trend.png"
    fig.savefig(fig_path, dpi=160)
    fig_path_pdf = figures_dir / "stroop_interference_quartile_loneliness_extremes25_trend.pdf"
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    slopes_path = output_dir / "stroop_interference_quartile_loneliness_extremes25_trend_slopes.csv"
    pd.DataFrame(slopes_rows).to_csv(slopes_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {fig_path}")
    print(f"Saved: {fig_path_pdf}")
    print(f"Saved: {slopes_path}")


if __name__ == "__main__":
    main()
