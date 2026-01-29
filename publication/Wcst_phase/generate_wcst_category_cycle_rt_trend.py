from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_analysis_data, get_output_dir, get_figures_dir
from publication.preprocessing.wcst._shared import prepare_wcst_trials


LOW_LABEL = "Low loneliness (bottom 25%)"
HIGH_LABEL = "High loneliness (top 25%)"
N_CATEGORIES = 6


def _assign_loneliness_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ucla_score"]).copy()
    q25 = df["ucla_score"].quantile(0.25)
    q75 = df["ucla_score"].quantile(0.75)
    df["loneliness_group"] = pd.NA
    df.loc[df["ucla_score"] <= q25, "loneliness_group"] = LOW_LABEL
    df.loc[df["ucla_score"] >= q75, "loneliness_group"] = HIGH_LABEL
    return df[df["loneliness_group"].notna()].copy()


def _compute_category_means(
    trials: pd.DataFrame,
    rt_col: str,
    trial_col: str | None,
    rule_col: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pid, grp in trials.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col) if trial_col else grp
        rt_series = pd.to_numeric(grp_sorted[rt_col], errors="coerce")
        if "is_rt_valid" in grp_sorted.columns:
            rt_valid = grp_sorted["is_rt_valid"].astype(bool) & rt_series.notna()
        else:
            rt_valid = rt_series.notna()
        rt_vals = rt_series.where(rt_valid).to_numpy(dtype=float)
        if "correct" not in grp_sorted.columns:
            continue
        correct = grp_sorted["correct"].astype(bool).to_numpy()
        rules = grp_sorted[rule_col].astype(str).str.lower().to_numpy()
        change_idx = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_idx
        segment_ends = change_idx + [len(rules)]
        for idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if idx > N_CATEGORIES:
                break
            seg_rt = rt_vals[start:end]
            seg_correct = correct[start:end]
            seg_rt_correct = seg_rt[seg_correct]
            mean_rt = float(np.nanmean(seg_rt_correct)) if np.isfinite(seg_rt_correct).sum() else np.nan
            records.append({
                "participant_id": pid,
                "category_num": idx,
                "mean_rt": mean_rt,
            })
    return pd.DataFrame(records)


def _summary_by_category(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str | int]] = []
    for category_num in range(1, N_CATEGORIES + 1):
        for group in (LOW_LABEL, HIGH_LABEL):
            vals = df.loc[
                (df["loneliness_group"] == group) & (df["category_num"] == category_num),
                "mean_rt",
            ].dropna()
            n = int(len(vals))
            mean = float(vals.mean()) if n else np.nan
            sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            records.append({
                "category_num": category_num,
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

    master = get_analysis_data("wcst")
    master = master[master["wcst_categories_completed"].fillna(0) >= N_CATEGORIES].copy()
    master = master[["participant_id", "ucla_score"]].dropna()

    trials_pack = prepare_wcst_trials()
    wcst = trials_pack["wcst"]
    rt_col = trials_pack["rt_col"]
    trial_col = trials_pack["trial_col"]
    rule_col = trials_pack["rule_col"]
    if wcst is None or rt_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    wcst = wcst[wcst["participant_id"].isin(master["participant_id"])].copy()
    category_means = _compute_category_means(wcst, rt_col, trial_col, rule_col)
    category_means = category_means.merge(master, on="participant_id", how="inner")
    category_means = _assign_loneliness_groups(category_means)

    summary = _summary_by_category(category_means)
    output_dir = get_output_dir("wcst")
    figures_dir = get_figures_dir()
    summary_path = output_dir / "wcst_category_cycle_rt_line_extremes25_6cat_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    x = np.arange(1, N_CATEGORIES + 1, dtype=float)
    group_data = {}
    n_by_group = {}
    for group in (LOW_LABEL, HIGH_LABEL):
        group_df = summary[summary["loneliness_group"] == group]
        means = []
        sems = []
        for cat in range(1, N_CATEGORIES + 1):
            row = group_df[group_df["category_num"] == cat]
            means.append(float(row["mean_rt"].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row["sem_rt"].iloc[0]) if not row.empty else np.nan)
        group_data[group] = (np.array(means, dtype=float), np.array(sems, dtype=float))
        n_by_group[group] = int(category_means[category_means["loneliness_group"] == group]["participant_id"].nunique())

    fig, ax = plt.subplots(figsize=(10.5, 6))
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
            label=f"{group} (n={n_by_group[group]})",
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

    ax.set_title("WCST RT Across Category Cycles (1-6, 6 categories completed)")
    ax.set_xlabel("Category cycle (1-6)")
    ax.set_ylabel("Mean RT (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = figures_dir / "wcst_category_cycle_rt_line_extremes25_6cat.png"
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
    slopes_path = output_dir / "wcst_category_cycle_rt_line_extremes25_6cat_slopes.csv"
    pd.DataFrame(slopes).to_csv(slopes_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {summary_path}")
    print(f"Saved: {slopes_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()



