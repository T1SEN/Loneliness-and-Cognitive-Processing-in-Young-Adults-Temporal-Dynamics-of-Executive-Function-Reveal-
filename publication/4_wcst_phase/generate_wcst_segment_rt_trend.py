from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_analysis_data, get_output_dir, get_figures_dir


LOW_LABEL = "Low loneliness (bottom 25%)"
HIGH_LABEL = "High loneliness (top 25%)"

SEGMENTS = [
    ("shift_trial", "wcst_switch_first_trial_rt"),
    ("post_shift_error", "wcst_post_shift_error_rt_mean"),
    ("exploration", "wcst_exploration_rt_mean"),
    ("confirmation", "wcst_confirmation_rt_mean"),
    ("exploitation", "wcst_exploitation_rt_mean"),
]


def _assign_loneliness_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ucla_score"]).copy()
    q25 = df["ucla_score"].quantile(0.25)
    q75 = df["ucla_score"].quantile(0.75)
    df["loneliness_group"] = pd.NA
    df.loc[df["ucla_score"] <= q25, "loneliness_group"] = LOW_LABEL
    df.loc[df["ucla_score"] >= q75, "loneliness_group"] = HIGH_LABEL
    return df[df["loneliness_group"].notna()].copy()


def _summary_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for segment, col in SEGMENTS:
        if col not in df.columns:
            continue
        for group in (LOW_LABEL, HIGH_LABEL):
            vals = df.loc[df["loneliness_group"] == group, col].dropna()
            n = int(len(vals))
            mean = float(vals.mean()) if n else np.nan
            sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            records.append({
                "segment": segment,
                "loneliness_group": group,
                "mean_rt": mean,
                "sem_rt": sem,
                "n": n,
            })
    return pd.DataFrame(records)


def _trend_line(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_vals)
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x_vals[mask], y_vals[mask], 1)
    return float(slope), float(intercept)


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = get_analysis_data("overall")
    df = _assign_loneliness_groups(df)

    summary = _summary_by_segment(df)
    output_dir = get_output_dir("overall")
    figures_dir = get_figures_dir()
    summary_path = output_dir / "wcst_segment_rt_line_extremes25_mean_trend_nodiscovery_nopreswitch_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    segments = [seg for seg, _ in SEGMENTS]
    x = np.arange(1, len(segments) + 1, dtype=float)
    group_data = {}
    for group in (LOW_LABEL, HIGH_LABEL):
        group_df = summary[summary["loneliness_group"] == group]
        means = []
        sems = []
        for seg in segments:
            row = group_df[group_df["segment"] == seg]
            means.append(float(row["mean_rt"].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row["sem_rt"].iloc[0]) if not row.empty else np.nan)
        group_data[group] = (np.array(means, dtype=float), np.array(sems, dtype=float))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {LOW_LABEL: "#1f77b4", HIGH_LABEL: "#d62728"}

    for group in (LOW_LABEL, HIGH_LABEL):
        means, sems = group_data[group]
        ax.errorbar(
            x,
            means,
            yerr=sems,
            color=colors[group],
            marker="o",
            linewidth=2,
            capsize=4,
            label=group,
        )
        slope, intercept = _trend_line(x, means)
        if np.isfinite(slope) and np.isfinite(intercept):
            ax.plot(
                x,
                slope * x + intercept,
                linestyle="--",
                color=colors[group],
                alpha=0.6,
                label=f"{group} trend ({slope:.1f} ms/seg)",
            )

    ax.set_title("WCST RT Across Segments (Mean + Trend, no discovery)")
    ax.set_xlabel("Segment (main only)")
    ax.set_ylabel("RT (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = figures_dir / "wcst_segment_rt_line_extremes25_mean_trend_nodiscovery_nopreswitch.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    slopes = []
    for group in (LOW_LABEL, HIGH_LABEL):
        means, _ = group_data[group]
        slope, _ = _trend_line(x, means)
        slopes.append({
            "loneliness_group": group,
            "slope_ms_per_segment": slope,
        })
    slopes_path = output_dir / "wcst_segment_rt_line_extremes25_mean_trend_nodiscovery_nopreswitch_slopes.csv"
    pd.DataFrame(slopes).to_csv(slopes_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {summary_path}")
    print(f"Saved: {slopes_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()



