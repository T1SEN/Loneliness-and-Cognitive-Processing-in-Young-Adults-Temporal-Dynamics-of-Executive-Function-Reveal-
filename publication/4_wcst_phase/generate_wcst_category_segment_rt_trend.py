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
from publication.wcst_phase_utils import prepare_wcst_trials


LOW_LABEL = "Low loneliness (bottom 25%)"
HIGH_LABEL = "High loneliness (top 25%)"
N_CATEGORIES = 6

SEGMENTS = [
    ("shift_trial", "Shift"),
    ("post_shift_error", "PostErr"),
    ("exploration", "Explore"),
    ("confirmation", "Confirm"),
    ("exploitation", "Exploit"),
]
FIRST_SEGMENTS = [
    ("exploration", "Explore"),
    ("confirmation", "Confirm"),
    ("exploitation", "Exploit"),
]


def _segment_plan() -> tuple[list[str], dict[tuple[int, str], int], dict[int, list[tuple[str, str]]]]:
    labels: list[str] = []
    order_map: dict[tuple[int, str], int] = {}
    segments_by_category: dict[int, list[tuple[str, str]]] = {}
    order = 1
    for cat in range(1, N_CATEGORIES + 1):
        segs = FIRST_SEGMENTS if cat == 1 else SEGMENTS
        segments_by_category[cat] = segs
        for seg_key, seg_short in segs:
            order_map[(cat, seg_key)] = order
            labels.append(f"C{cat}-{seg_short}")
            order += 1
    return labels, order_map, segments_by_category


def _assign_loneliness_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ucla_score"]).copy()
    q25 = df["ucla_score"].quantile(0.25)
    q75 = df["ucla_score"].quantile(0.75)
    df["loneliness_group"] = pd.NA
    df.loc[df["ucla_score"] <= q25, "loneliness_group"] = LOW_LABEL
    df.loc[df["ucla_score"] >= q75, "loneliness_group"] = HIGH_LABEL
    return df[df["loneliness_group"].notna()].copy()


def _segment_means_for_cycle(
    rt_vals: np.ndarray,
    correct: np.ndarray,
    idx: int,
    next_idx: int,
) -> dict[str, float]:
    seg_means: dict[str, float] = {}

    shift_rt = rt_vals[idx] if idx < len(rt_vals) else np.nan
    seg_means["shift_trial"] = float(shift_rt) if np.isfinite(shift_rt) else np.nan

    reacq_start = None
    reacq_idx = None
    for j in range(idx, next_idx - 2):
        if correct[j] and correct[j + 1] and correct[j + 2]:
            reacq_start = j
            reacq_idx = j + 2
            break

    explore_end = reacq_start if reacq_start is not None else next_idx
    if explore_end > idx:
        segment_rt = rt_vals[idx:explore_end]
        segment_rt = segment_rt[np.isfinite(segment_rt)]
        seg_means["exploration"] = float(np.mean(segment_rt)) if len(segment_rt) else np.nan
    else:
        seg_means["exploration"] = np.nan

    if reacq_start is not None and reacq_idx is not None:
        confirm_rt = rt_vals[reacq_start:reacq_idx + 1]
        confirm_rt = confirm_rt[np.isfinite(confirm_rt)]
        seg_means["confirmation"] = float(np.mean(confirm_rt)) if len(confirm_rt) else np.nan
    else:
        seg_means["confirmation"] = np.nan

    if reacq_idx is not None:
        exp_start = reacq_idx + 1
        if exp_start < next_idx:
            exp_rt = rt_vals[exp_start:next_idx]
            exp_rt = exp_rt[np.isfinite(exp_rt)]
            seg_means["exploitation"] = float(np.mean(exp_rt)) if len(exp_rt) else np.nan
        else:
            seg_means["exploitation"] = np.nan
    else:
        seg_means["exploitation"] = np.nan

    shift_end_err = reacq_idx if reacq_idx is not None else next_idx - 1
    post_error_rts = []
    if shift_end_err >= idx:
        for j in range(idx, shift_end_err + 1):
            if correct[j]:
                continue
            if j + 1 < next_idx:
                rt_next = rt_vals[j + 1]
                if np.isfinite(rt_next):
                    post_error_rts.append(float(rt_next))
    seg_means["post_shift_error"] = float(np.mean(post_error_rts)) if post_error_rts else np.nan

    return seg_means


def _build_long_df(
    trials: pd.DataFrame,
    rt_col: str,
    trial_col: str | None,
    rule_col: str,
    segments_by_category: dict[int, list[tuple[str, str]]],
    order_map: dict[tuple[int, str], int],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pid, grp in trials.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col) if trial_col else grp
        rt_series = pd.to_numeric(grp_sorted[rt_col], errors="coerce")
        rt_vals = rt_series.to_numpy(dtype=float)
        if "correct" not in grp_sorted.columns:
            continue
        correct = grp_sorted["correct"].astype(bool).to_numpy()
        rules = grp_sorted[rule_col].astype(str).str.lower().to_numpy()
        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]
        for cat_idx, (idx, next_idx) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > N_CATEGORIES:
                break
            seg_means = _segment_means_for_cycle(rt_vals, correct, idx, next_idx)
            for seg_key, seg_short in segments_by_category[cat_idx]:
                order = order_map[(cat_idx, seg_key)]
                records.append({
                    "participant_id": pid,
                    "category_num": cat_idx,
                    "segment": seg_key,
                    "segment_label": f"C{cat_idx}-{seg_short}",
                    "segment_order": order,
                    "mean_rt": seg_means.get(seg_key, np.nan),
                })
    return pd.DataFrame(records)


def _summary_by_block(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for order, label in enumerate(labels, start=1):
        block_df = df[df["segment_order"] == order]
        for group in (LOW_LABEL, HIGH_LABEL):
            vals = block_df.loc[block_df["loneliness_group"] == group, "mean_rt"].dropna()
            n = int(len(vals))
            mean = float(vals.mean()) if n else np.nan
            sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            records.append({
                "segment_order": order,
                "segment_label": label,
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

    master = get_analysis_data("overall")
    master = master[["participant_id", "ucla_score"]].dropna()

    trials_pack = prepare_wcst_trials()
    wcst = trials_pack["wcst"]
    rt_col = trials_pack["rt_col"]
    trial_col = trials_pack["trial_col"]
    rule_col = trials_pack["rule_col"]
    if wcst is None or rt_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    wcst = wcst[wcst["participant_id"].isin(master["participant_id"])].copy()
    labels, order_map, segments_by_category = _segment_plan()
    long_df = _build_long_df(wcst, rt_col, trial_col, rule_col, segments_by_category, order_map)
    long_df = long_df.merge(master, on="participant_id", how="inner")
    long_df = _assign_loneliness_groups(long_df)

    summary = _summary_by_block(long_df, labels)
    output_dir = get_output_dir("overall")
    figures_dir = get_figures_dir()
    summary_path = output_dir / "wcst_category_segment_rt_line_extremes25_6cat_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    x = np.arange(1, len(labels) + 1, dtype=float)

    group_data = {}
    n_by_group = {}
    for group in (LOW_LABEL, HIGH_LABEL):
        group_df = summary[summary["loneliness_group"] == group]
        means = []
        sems = []
        for order in range(1, len(labels) + 1):
            row = group_df[group_df["segment_order"] == order]
            means.append(float(row["mean_rt"].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row["sem_rt"].iloc[0]) if not row.empty else np.nan)
        group_data[group] = (np.array(means, dtype=float), np.array(sems, dtype=float))
        n_by_group[group] = int(long_df[long_df["loneliness_group"] == group]["participant_id"].nunique())

    fig, ax = plt.subplots(figsize=(14, 6.5))
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
            capsize=3,
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

    ax.set_title("WCST RT Across Category Segments (C1 starts at Explore, Top/Bottom 25%)")
    ax.set_xlabel(f"Category cycle x segment ({len(labels)} blocks)")
    ax.set_ylabel("Mean RT (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = figures_dir / "wcst_category_segment_rt_line_extremes25_6cat.png"
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
    slopes_path = output_dir / "wcst_category_segment_rt_line_extremes25_6cat_slopes.csv"
    pd.DataFrame(slopes).to_csv(slopes_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {summary_path}")
    print(f"Saved: {slopes_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()



