from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_analysis_data, get_output_dir, run_ucla_regression


COV_TYPE = "nonrobust"

SEGMENT_MEAN_COLS = [
    ("shift_trial", "wcst_switch_first_trial_rt"),
    ("post_shift_error", "wcst_post_shift_error_rt_mean"),
    ("exploration", "wcst_exploration_rt_mean"),
    ("confirmation", "wcst_confirmation_rt_mean"),
    ("exploitation", "wcst_exploitation_rt_mean"),
]

SEGMENT_SLOPE_COLS = [
    ("exploration", "wcst_exploration_rt_slope"),
    ("confirmation", "wcst_confirmation_rt_slope"),
    ("exploitation", "wcst_exploitation_rt_slope"),
]


def _cycle_slope(row: pd.Series) -> float:
    x_vals = []
    y_vals = []
    for idx, (_, col) in enumerate(SEGMENT_MEAN_COLS, start=1):
        val = row.get(col, np.nan)
        if pd.notna(val):
            x_vals.append(float(idx))
            y_vals.append(float(val))
    if len(y_vals) < 3:
        return np.nan
    return float(np.polyfit(np.array(x_vals), np.array(y_vals), 1)[0])


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = get_analysis_data("wcst")
    df["wcst_cycle_rt_slope"] = df.apply(_cycle_slope, axis=1)

    results: list[dict[str, object]] = []

    for label, col in SEGMENT_MEAN_COLS:
        res = run_ucla_regression(df, col, cov_type=COV_TYPE)
        if res:
            res["outcome"] = f"WCST {label} RT Mean"
            res["outcome_type"] = "segment_mean"
            res["segment"] = label
            results.append(res)

    for label, col in SEGMENT_SLOPE_COLS:
        if col not in df.columns:
            continue
        res = run_ucla_regression(df, col, cov_type=COV_TYPE)
        if res:
            res["outcome"] = f"WCST {label} RT Slope"
            res["outcome_type"] = "segment_slope"
            res["segment"] = label
            results.append(res)

    res = run_ucla_regression(df, "wcst_cycle_rt_slope", cov_type=COV_TYPE)
    if res:
        res["outcome"] = "WCST Cycle RT Slope (segment trend)"
        res["outcome_type"] = "cycle_slope"
        res["segment"] = "cycle"
        results.append(res)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir("wcst")
    out_path = output_dir / "wcst_segment_rt_regression_nodiscovery_nopreswitch_ols.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df[["outcome", "n", "ucla_beta", "ucla_p", "cov_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
