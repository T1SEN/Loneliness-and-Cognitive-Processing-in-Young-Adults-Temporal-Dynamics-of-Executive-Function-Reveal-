from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_output_dir, get_figures_dir
from publication.preprocessing.constants import get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from publication.preprocessing.wcst._shared import prepare_wcst_trials

from run_wcst_post_shift_error_log_ols import label_wcst_phases


LOW_LABEL = "Low loneliness (bottom 25%)"
HIGH_LABEL = "High loneliness (top 25%)"
PHASES = ["exploration", "confirmation", "exploitation"]
PHASE_LABELS = ["Exploration", "Confirmation", "Exploitation"]


def _assign_loneliness_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ucla_score"]).copy()
    q25 = df["ucla_score"].quantile(0.25)
    q75 = df["ucla_score"].quantile(0.75)
    df["loneliness_group"] = pd.NA
    df.loc[df["ucla_score"] <= q25, "loneliness_group"] = LOW_LABEL
    df.loc[df["ucla_score"] >= q75, "loneliness_group"] = HIGH_LABEL
    return df[df["loneliness_group"].notna()].copy()


def _trend_line(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_vals)
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x_vals[mask], y_vals[mask], 1)
    return float(slope), float(intercept)


def _load_base_ucla() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    participants = load_participants(data_dir)
    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    dass = load_dass_scores(data_dir)
    base = participants.merge(ucla, on="participant_id", how="inner")
    base = base.merge(dass, on="participant_id", how="left")
    base = ensure_participant_id(base)
    return base[["participant_id", "ucla_score"]]


def _load_qc_ids(task: str) -> set[str]:
    task_dir = get_results_dir(task)
    qc_ids_path = task_dir / "filtered_participant_ids.csv"
    if not qc_ids_path.exists():
        return set()
    qc_ids = pd.read_csv(qc_ids_path, encoding="utf-8-sig")
    qc_ids = ensure_participant_id(qc_ids)
    if "participant_id" not in qc_ids.columns:
        return set()
    return set(qc_ids["participant_id"].dropna().astype(str))


def _phase_means(confirm_len: int, include_errors: bool) -> pd.DataFrame:
    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    qc_ids = _load_qc_ids("wcst")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col=trial_col, confirm_len=confirm_len)
    wcst = wcst.dropna(subset=["phase"])

    if not include_errors:
        wcst = wcst[wcst["correct"].astype(bool)]

    phase_means = (
        wcst.groupby(["participant_id", "phase"], observed=False)["rt_ms"]
        .mean()
        .unstack()
        .reset_index()
    )
    return phase_means


def main(confirm_len: int, include_errors: bool) -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output_dir = get_output_dir("wcst")
    figures_dir = get_figures_dir()

    base = _load_base_ucla()
    phase_means = _phase_means(confirm_len=confirm_len, include_errors=include_errors)
    df = phase_means.merge(base, on="participant_id", how="inner")
    df = _assign_loneliness_groups(df)

    summary_rows = []
    for phase in PHASES:
        for group in (LOW_LABEL, HIGH_LABEL):
            vals = df.loc[df["loneliness_group"] == group, phase].dropna()
            n = int(len(vals))
            mean = float(vals.mean()) if n else np.nan
            sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            summary_rows.append({
                "phase": phase,
                "loneliness_group": group,
                "mean_rt": mean,
                "sem_rt": sem,
                "n": n,
            })
    summary = pd.DataFrame(summary_rows)

    err_tag = "alltrials" if include_errors else "correct"
    summary_path = output_dir / f"wcst_phase_rt_loneliness_extremes25_{err_tag}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    x = np.arange(1, len(PHASES) + 1, dtype=float)
    group_data = {}
    for group in (LOW_LABEL, HIGH_LABEL):
        group_df = summary[summary["loneliness_group"] == group]
        means = []
        sems = []
        n_val = int(group_df["n"].max()) if not group_df.empty else 0
        for phase in PHASES:
            row = group_df[group_df["phase"] == phase]
            means.append(float(row["mean_rt"].iloc[0]) if not row.empty else np.nan)
            sems.append(float(row["sem_rt"].iloc[0]) if not row.empty else np.nan)
        group_data[group] = (np.array(means, dtype=float), np.array(sems, dtype=float), n_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {LOW_LABEL: "#1f77b4", HIGH_LABEL: "#d62728"}

    slopes_rows = []
    for group in (HIGH_LABEL, LOW_LABEL):
        means, sems, n_val = group_data[group]
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
        slope, intercept = _trend_line(x, means)
        if np.isfinite(slope) and np.isfinite(intercept):
            ax.plot(
                x,
                slope * x + intercept,
                linestyle="--",
                color=colors[group],
                alpha=0.6,
                label=f"{group} trend ({slope:.1f} ms/phase)",
            )
        slopes_rows.append({
            "loneliness_group": group,
            "slope_ms_per_phase": slope,
        })

    ax.set_title("WCST Phase RT by Loneliness Group (Top/Bottom 25%)")
    ax.set_xlabel("Phase")
    ax.set_ylabel("RT (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_LABELS)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = figures_dir / f"wcst_phase_rt_loneliness_extremes25_{err_tag}.png"
    fig.savefig(fig_path, dpi=160)
    fig_path_pdf = figures_dir / f"wcst_phase_rt_loneliness_extremes25_{err_tag}.pdf"
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    slopes_path = output_dir / f"wcst_phase_rt_loneliness_extremes25_{err_tag}_slopes.csv"
    pd.DataFrame(slopes_rows).to_csv(slopes_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {summary_path}")
    print(f"Saved: {slopes_path}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {fig_path_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-len", type=int, default=3)
    parser.add_argument("--correct-only", action="store_true")
    args = parser.parse_args()
    main(args.confirm_len, include_errors=not args.correct_only)
