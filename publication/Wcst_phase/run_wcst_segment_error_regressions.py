from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_analysis_data, get_output_dir, run_ucla_regression


COV_TYPE = "nonrobust"

SEGMENT_ERROR_COLS = [
    ("exploration", "wcst_exploration_error_rate"),
    ("confirmation", "wcst_confirmation_error_rate"),
    ("exploitation", "wcst_exploitation_error_rate"),
]


def _has_variance(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    return values.nunique() > 1


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = get_analysis_data("wcst")
    results: list[dict[str, object]] = []

    for label, col in SEGMENT_ERROR_COLS:
        if col not in df.columns:
            continue
        if not _has_variance(df[col]):
            continue
        res = run_ucla_regression(df, col, cov_type=COV_TYPE)
        if res:
            res["outcome"] = f"WCST {label} Error Rate"
            res["outcome_type"] = "segment_error_rate"
            res["segment"] = label
            results.append(res)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir("wcst")
    out_path = output_dir / "wcst_segment_error_rate_regression_nodiscovery_nopreswitch_ols.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df[["outcome", "n", "ucla_beta", "ucla_p", "cov_type"]].to_string(index=False))


if __name__ == "__main__":
    main()


