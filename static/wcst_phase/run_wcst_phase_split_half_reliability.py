"""Compute WCST phase split-half reliability using odd/even categories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.analysis.utils import get_output_dir
from static.preprocessing.constants import get_results_dir
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.wcst.phase import label_wcst_phases
from static.preprocessing.wcst.utils import prepare_wcst_trials


PHASES = ["exploration", "confirmation", "exploitation"]


def load_qc_ids(task: str) -> set[str]:
    task_dir = get_results_dir(task)
    qc_ids_path = task_dir / "filtered_participant_ids.csv"
    if not qc_ids_path.exists():
        return set()
    qc_ids = pd.read_csv(qc_ids_path, encoding="utf-8-sig")
    qc_ids = ensure_participant_id(qc_ids)
    if "participant_id" not in qc_ids.columns:
        return set()
    return set(qc_ids["participant_id"].dropna().astype(str))


def _split_half_stats(df: pd.DataFrame, phase_col: str, phase_value: str) -> dict[str, float]:
    subset = df[df[phase_col] == phase_value]
    means = (
        subset.groupby(["participant_id", "half"], observed=False)["rt_ms"]
        .mean()
        .unstack()
    )
    if means.empty:
        return {
            "phase": phase_value,
            "n": 0,
            "r": np.nan,
            "spearman_brown": np.nan,
        }
    if "odd" not in means.columns or "even" not in means.columns:
        return {
            "phase": phase_value,
            "n": 0,
            "r": np.nan,
            "spearman_brown": np.nan,
        }
    both = means.dropna(subset=["odd", "even"])
    n = int(len(both))
    if n < 2:
        return {
            "phase": phase_value,
            "n": n,
            "r": np.nan,
            "spearman_brown": np.nan,
        }
    r = float(both["odd"].corr(both["even"]))
    sb = float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else np.nan
    return {
        "phase": phase_value,
        "n": n,
        "r": r,
        "spearman_brown": sb,
    }


def main(confirm_len: int, correct_only: bool) -> None:
    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    qc_ids = load_qc_ids("overall")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col=trial_col, confirm_len=confirm_len)
    wcst = wcst.dropna(subset=["phase", "category_num"])

    if correct_only:
        wcst = wcst[wcst["correct"].astype(bool)]

    wcst["category_num"] = pd.to_numeric(wcst["category_num"], errors="coerce")
    wcst = wcst.dropna(subset=["category_num"])
    wcst["half"] = np.where(wcst["category_num"].astype(int) % 2 == 1, "odd", "even")

    results = []
    for phase in PHASES:
        results.append(_split_half_stats(wcst, "phase", phase))

    wcst["phase_merge"] = wcst["phase"].map(
        {
            "exploration": "pre_exploitation",
            "confirmation": "pre_exploitation",
            "exploitation": "exploitation",
        }
    )
    results.append(_split_half_stats(wcst, "phase_merge", "pre_exploitation"))

    results_df = pd.DataFrame(results)

    output_dir = get_output_dir("overall", bucket="supplementary")
    suffix = "_correct" if correct_only else ""
    if confirm_len == 3:
        filename = f"wcst_phase_split_half_reliability{suffix}.csv"
    else:
        filename = f"wcst_phase_split_half_reliability_m{confirm_len}{suffix}.csv"
    out_path = output_dir / filename
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-len", type=int, default=3)
    parser.add_argument("--correct-only", action="store_true")
    args = parser.parse_args()
    main(args.confirm_len, args.correct_only)
