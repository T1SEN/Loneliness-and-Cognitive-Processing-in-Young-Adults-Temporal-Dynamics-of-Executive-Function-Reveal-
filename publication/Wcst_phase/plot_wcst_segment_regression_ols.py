from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_output_dir, get_figures_dir


MEAN_ORDER = [
    "shift_trial",
    "post_shift_error",
    "exploration",
    "confirmation",
    "exploitation",
]

SLOPE_ORDER = [
    "exploration",
    "confirmation",
    "exploitation",
    "cycle",
]


def _prepare_panel(df: pd.DataFrame, outcome_type: str, order: list[str]) -> pd.DataFrame:
    panel = df[df["outcome_type"] == outcome_type].copy()
    panel["order"] = panel["segment"].apply(lambda x: order.index(x) if x in order else len(order))
    panel = panel.sort_values("order")
    if outcome_type == "segment_mean":
        panel["label"] = panel["segment"]
    elif outcome_type == "segment_slope":
        panel["label"] = panel["segment"] + " slope"
    else:
        panel["label"] = panel["segment"] + " slope"
    return panel


def _plot_panel(ax: plt.Axes, panel: pd.DataFrame, title: str, xlabel: str) -> None:
    y = np.arange(len(panel))
    betas = panel["ucla_beta"].to_numpy(dtype=float)
    ses = panel["ucla_se"].to_numpy(dtype=float)
    pvals = panel["ucla_p"].to_numpy(dtype=float)
    ci = 1.96 * ses

    ax.errorbar(
        betas,
        y,
        xerr=ci,
        fmt="o",
        color="#444444",
        ecolor="#9e9e9e",
        capsize=4,
        zorder=2,
    )
    sig_colors = np.where(pvals < 0.05, "#d62728", "#1f77b4")
    ax.scatter(betas, y, c=sig_colors, s=60, zorder=3)

    ax.axvline(0, color="#808080", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(panel["label"].tolist())
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output_dir = get_output_dir("wcst")
    figures_dir = get_figures_dir()
    data_path = output_dir / "wcst_segment_rt_regression_nodiscovery_nopreswitch_ols.csv"
    df = pd.read_csv(data_path)

    mean_panel = _prepare_panel(df, "segment_mean", MEAN_ORDER)
    slope_panel = _prepare_panel(df, "segment_slope", SLOPE_ORDER)
    cycle_row = df[df["outcome_type"] == "cycle_slope"].copy()
    if not cycle_row.empty:
        cycle_row["segment"] = "cycle"
        cycle_row["label"] = "cycle slope"
        slope_panel = pd.concat([slope_panel, cycle_row], ignore_index=True)
        slope_panel["order"] = slope_panel["segment"].apply(
            lambda x: SLOPE_ORDER.index(x) if x in SLOPE_ORDER else len(SLOPE_ORDER)
        )
        slope_panel = slope_panel.sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    _plot_panel(
        axes[0],
        mean_panel,
        "Segment Mean RT (OLS)",
        "UCLA beta (ms)",
    )
    _plot_panel(
        axes[1],
        slope_panel,
        "Segment/Cycle Slopes (OLS)",
        "UCLA beta (ms per step)",
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="p<0.05"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="n.s."),
    ]
    axes[1].legend(handles=handles, loc="lower right")

    fig.tight_layout()
    fig_path = figures_dir / "wcst_segment_rt_regression_nodiscovery_nopreswitch_ols.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()


