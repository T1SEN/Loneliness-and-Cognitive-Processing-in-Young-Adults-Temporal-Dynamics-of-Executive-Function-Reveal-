from __future__ import annotations

import sys
from pathlib import Path

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


def _delta_col_name(start_label: str, end_label: str) -> str:
    return f"wcst_rt_delta_{start_label}_to_{end_label}"


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = get_analysis_data("wcst")

    results: list[dict[str, object]] = []
    for idx in range(len(SEGMENT_MEAN_COLS) - 1):
        start_label, start_col = SEGMENT_MEAN_COLS[idx]
        end_label, end_col = SEGMENT_MEAN_COLS[idx + 1]
        delta_col = _delta_col_name(start_label, end_label)
        if start_col not in df.columns or end_col not in df.columns:
            continue
        df[delta_col] = df[end_col] - df[start_col]
        res = run_ucla_regression(df, delta_col, cov_type=COV_TYPE)
        if res:
            res["outcome"] = f"WCST RT Delta {start_label} -> {end_label}"
            res["outcome_type"] = "segment_delta"
            res["segment"] = f"{start_label}_to_{end_label}"
            results.append(res)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir("wcst")
    out_path = output_dir / "wcst_segment_rt_delta_regression_nodiscovery_nopreswitch_ols.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df[["outcome", "n", "ucla_beta", "ucla_p", "cov_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
