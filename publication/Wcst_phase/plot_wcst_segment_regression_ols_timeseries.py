from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
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

PREDICTORS = [
    "z_ucla_score",
    "z_dass_depression",
    "z_dass_anxiety",
    "z_dass_stress",
    "z_age",
    "gender_male",
]


def _fit_model(df: pd.DataFrame, outcome: str) -> tuple[object, pd.DataFrame] | tuple[None, pd.DataFrame]:
    cols = [outcome] + PREDICTORS
    sub = df[cols].dropna()
    if sub.empty:
        return None, sub
    formula = (
        f"{outcome} ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    )
    model = smf.ols(formula, data=sub).fit()
    return model, sub


def _adjusted_stats(model: object, data: pd.DataFrame, z_ucla_value: float) -> tuple[float, float, int]:
    temp = data.copy()
    temp["z_ucla_score"] = z_ucla_value
    preds = model.predict(temp)
    n = int(len(preds))
    mean = float(preds.mean()) if n else np.nan
    sem = float(preds.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return mean, sem, n


def _trend_line(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_vals)
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x_vals[mask], y_vals[mask], 1)
    return float(slope), float(intercept)


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = get_analysis_data("wcst")
    q25 = float(df["z_ucla_score"].quantile(0.25))
    q75 = float(df["z_ucla_score"].quantile(0.75))

    records: list[dict[str, float | str | int]] = []
    for segment, col in SEGMENTS:
        model, sub = _fit_model(df, col)
        if model is None or sub.empty:
            continue
        mean_low, sem_low, n_low = _adjusted_stats(model, sub, q25)
        mean_high, sem_high, n_high = _adjusted_stats(model, sub, q75)
        records.extend([
            {
                "segment": segment,
                "loneliness_group": LOW_LABEL,
                "mean_rt": mean_low,
                "sem_rt": sem_low,
                "n": n_low,
            },
            {
                "segment": segment,
                "loneliness_group": HIGH_LABEL,
                "mean_rt": mean_high,
                "sem_rt": sem_high,
                "n": n_high,
            },
        ])

    summary = pd.DataFrame(records)
    output_dir = get_output_dir("wcst")
    figures_dir = get_figures_dir()
    summary_path = output_dir / "wcst_segment_rt_regression_ols_timeseries_summary.csv"
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

    fig, ax = plt.subplots(figsize=(11, 5.5))
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

    ax.set_title("WCST RT Across Segments (OLS-adjusted, mean + trend)")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Adjusted RT (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = figures_dir / "wcst_segment_rt_regression_ols_timeseries.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {summary_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()


