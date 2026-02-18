from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from static.analysis.utils import get_output_dir, get_figures_dir
from static.preprocessing.constants import STROOP_RT_MIN, STROOP_RT_MAX, get_results_dir
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.surveys import load_ucla_scores

LOW_LABEL = "Low loneliness (bottom 25%)"
HIGH_LABEL = "High loneliness (top 25%)"
LOW_COLOR = "#5B8DB8"   # Steel blue
HIGH_COLOR = "#E8886F"  # Soft coral
PLOT_NOTES = [
    "Points are means; error bars indicate 95% CIs.",
]


def _trend_line(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_vals)
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x_vals[mask], y_vals[mask], 1)
    return float(slope), float(intercept)


def _ci95_halfwidth(sd: float, n: int) -> float:
    if not np.isfinite(sd) or n <= 1:
        return np.nan
    sem = sd / np.sqrt(n)
    t_crit = float(student_t.ppf(0.975, df=n - 1))
    return float(t_crit * sem)


def _sem_to_ci95(sem: float, n: int) -> float:
    if not np.isfinite(sem) or n <= 1:
        return np.nan
    t_crit = float(student_t.ppf(0.975, df=n - 1))
    return float(t_crit * sem)


def _ensure_ci95_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if "ci95_interference_rt" in out.columns and "ci95_interference" not in out.columns:
        out = out.rename(columns={"ci95_interference_rt": "ci95_interference"})
    if "ci95_interference" not in out.columns:
        if "sem_interference" in out.columns:
            out["ci95_interference"] = out.apply(
                lambda row: _sem_to_ci95(float(row["sem_interference"]), int(row["n"]))
                if pd.notna(row.get("n"))
                else np.nan,
                axis=1,
            )
        elif "sem_interference_rt" in out.columns:
            out["ci95_interference"] = out.apply(
                lambda row: _sem_to_ci95(float(row["sem_interference_rt"]), int(row["n"]))
                if pd.notna(row.get("n"))
                else np.nan,
                axis=1,
            )
    return out


def _legend_with_notes(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    note_handles = [Line2D([], [], linestyle="", marker=None) for _ in PLOT_NOTES]
    ax.legend(handles + note_handles, labels + PLOT_NOTES, loc="upper right")


def _group_label(group: str, n_val: int) -> str:
    if group == HIGH_LABEL:
        return f"{group} (n={n_val}, 4 ties)"
    return f"{group} (n={n_val})"


def _load_qc_ids() -> set[str]:
    task_dir = get_results_dir("overall")
    qc_ids_path = task_dir / "filtered_participant_ids.csv"
    if not qc_ids_path.exists():
        return set()
    qc_ids = pd.read_csv(qc_ids_path, encoding="utf-8-sig")
    qc_ids = ensure_participant_id(qc_ids)
    if "participant_id" not in qc_ids.columns:
        return set()
    return set(qc_ids["participant_id"].dropna().astype(str))


def _build_table(output_dir: Path) -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    trials_path = data_dir / "4a_stroop_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError("Missing Stroop trials file.")

    trials = pd.read_csv(trials_path, encoding="utf-8-sig")
    trials = ensure_participant_id(trials)

    qc_ids = _load_qc_ids()
    if qc_ids:
        trials = trials[trials["participant_id"].isin(qc_ids)]

    if "cond" in trials.columns:
        trials["cond"] = trials["cond"].astype(str).str.lower()
    elif "type" in trials.columns:
        trials["cond"] = trials["type"].astype(str).str.lower()
    else:
        raise KeyError("No condition column found in Stroop trials.")

    if "trial_index" in trials.columns:
        trials["trial_order"] = pd.to_numeric(trials["trial_index"], errors="coerce")
    elif "trial_order" in trials.columns:
        trials["trial_order"] = pd.to_numeric(trials["trial_order"], errors="coerce")
    elif "trial" in trials.columns:
        trials["trial_order"] = pd.to_numeric(trials["trial"], errors="coerce")
    else:
        raise KeyError("No trial order column found in Stroop trials.")

    trials["is_rt_valid"] = trials["is_rt_valid"].astype(str).str.lower().isin({"true", "1", "t", "yes"})
    trials["timeout"] = trials["timeout"].astype(str).str.lower().isin({"true", "1", "t", "yes"})
    trials["correct"] = trials["correct"].astype(str).str.lower().isin({"true", "1", "t", "yes"})
    trials["rt_ms"] = pd.to_numeric(trials["rt_ms"], errors="coerce")

    valid = (
        trials["cond"].isin({"congruent", "incongruent"})
        & trials["correct"]
        & (~trials["timeout"])
        & trials["is_rt_valid"]
        & trials["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
    )
    trials = trials[valid].dropna(subset=["participant_id", "trial_order", "rt_ms"])
    if trials.empty:
        raise RuntimeError("No valid Stroop trials after filtering.")

    trials = trials.sort_values(["participant_id", "trial_order"]).reset_index(drop=True)
    trials["segment"] = np.nan
    for pid, grp in trials.groupby("participant_id"):
        order_rank = grp["trial_order"].rank(method="first")
        try:
            seg = pd.qcut(order_rank, q=4, labels=[1, 2, 3, 4], duplicates="drop")
        except ValueError:
            continue
        trials.loc[grp.index, "segment"] = seg.astype(float)
    trials = trials.dropna(subset=["segment"])

    means = (
        trials.groupby(["participant_id", "segment", "cond"])["rt_ms"]
        .mean()
        .unstack()
    )
    if "incongruent" not in means.columns or "congruent" not in means.columns:
        raise RuntimeError("Missing condition means for interference computation.")
    means["interference_rt"] = means["incongruent"] - means["congruent"]
    means = means.reset_index()

    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    ucla = ensure_participant_id(ucla)
    ucla = ucla[["participant_id", "ucla_score"]].dropna()
    q25 = ucla["ucla_score"].quantile(0.25)
    q75 = ucla["ucla_score"].quantile(0.75)
    ucla["loneliness_group"] = pd.NA
    ucla.loc[ucla["ucla_score"] <= q25, "loneliness_group"] = LOW_LABEL
    ucla.loc[ucla["ucla_score"] >= q75, "loneliness_group"] = HIGH_LABEL
    ucla = ucla[ucla["loneliness_group"].notna()]

    merged = means.merge(ucla, on="participant_id", how="inner")
    if merged.empty:
        raise RuntimeError("No participants in loneliness extremes after merge.")

    summary = (
        merged.groupby(["loneliness_group", "segment"])["interference_rt"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_interference_rt", "std": "sd_interference_rt", "count": "n"})
    )
    summary["ci95_interference_rt"] = summary.apply(
        lambda row: _ci95_halfwidth(float(row["sd_interference_rt"]), int(row["n"])),
        axis=1,
    )
    summary = summary.drop(columns=["sd_interference_rt"])

    table_path = output_dir / "stroop_rt_timeseg4_loneliness_extremes25_table.csv"
    summary.to_csv(table_path, index=False, encoding="utf-8-sig")
    return summary


def _load_summary(output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "stroop_interference_quartile_loneliness_extremes25_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, encoding="utf-8-sig")
        summary = _ensure_ci95_summary(summary)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary

    table_path = output_dir / "stroop_rt_timeseg4_loneliness_extremes25_table.csv"
    if not table_path.exists():
        table = _build_table(output_dir)
    else:
        table = pd.read_csv(table_path, encoding="utf-8-sig")
    summary = table.rename(
        columns={
            "segment": "quartile",
            "mean_interference_rt": "mean_interference",
            "ci95_interference_rt": "ci95_interference",
            "sem_interference_rt": "sem_interference",
        }
    )
    summary = _ensure_ci95_summary(summary)
    summary = summary[["loneliness_group", "quartile", "mean_interference", "ci95_interference", "n"]]
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output_dir = get_output_dir("overall", bucket="supplementary")
    figures_dir = get_figures_dir(bucket="supplementary")

    summary = _load_summary(output_dir)
    summary = summary.dropna(subset=["quartile"]).copy()
    summary["quartile"] = pd.to_numeric(summary["quartile"], errors="coerce")

    segments = [1, 2, 3, 4]
    x = np.array(segments, dtype=float)

    group_data: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for group in (HIGH_LABEL, LOW_LABEL):
        group_df = summary[summary["loneliness_group"] == group]
        means = []
        ci95 = []
        n_val = int(group_df["n"].max()) if not group_df.empty else 0
        for seg in segments:
            row = group_df[group_df["quartile"] == seg]
            means.append(float(row["mean_interference"].iloc[0]) if not row.empty else np.nan)
            ci95.append(float(row["ci95_interference"].iloc[0]) if not row.empty else np.nan)
        group_data[group] = (np.array(means, dtype=float), np.array(ci95, dtype=float), n_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {LOW_LABEL: LOW_COLOR, HIGH_LABEL: HIGH_COLOR}

    slopes_rows = []
    for group in (HIGH_LABEL, LOW_LABEL):
        means, ci95, n_val = group_data[group]
        overall_mean = float(np.nanmean(means)) if np.isfinite(means).any() else np.nan
        ax.errorbar(
            x,
            means,
            yerr=ci95,
            color=colors[group],
            marker="o",
            linewidth=2,
            capsize=4,
            label=_group_label(group, n_val),
        )
        slope, intercept = _trend_line(x, means)
        if np.isfinite(slope) and np.isfinite(intercept):
            ax.plot(
                x,
                slope * x + intercept,
                linestyle="--",
                color=colors[group],
                alpha=0.6,
                label=f"{group} trend ({slope:.1f} ms/segment)",
            )
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
    _legend_with_notes(ax)
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
