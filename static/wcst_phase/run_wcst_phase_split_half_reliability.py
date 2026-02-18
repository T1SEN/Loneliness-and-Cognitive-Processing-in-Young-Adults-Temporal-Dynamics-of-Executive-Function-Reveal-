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
from static.preprocessing.public_validate import get_common_public_ids
from static.preprocessing.wcst.phase import label_wcst_phases
from static.preprocessing.wcst.utils import prepare_wcst_trials


PHASES = ["exploration", "confirmation", "exploitation"]


def load_qc_ids(task: str) -> set[str]:
    _ = task
    return get_common_public_ids(validate=True)


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


def _get_complete6_ids(wcst: pd.DataFrame) -> set[str]:
    if wcst.empty or "rule" not in wcst.columns or "trial_order" not in wcst.columns:
        return set()
    wcst = wcst.sort_values(["participant_id", "trial_order"]).copy()
    cat_counts = (
        wcst.groupby("participant_id")["rule"]
        .apply(lambda s: s.ne(s.shift()).sum())
    )
    return set(cat_counts[cat_counts >= 6].index.astype(str))


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
    if "timeout" in wcst.columns:
        wcst = wcst[~wcst["timeout"]]

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


def run_complete6_outputs(confirm_len: int = 3, correct_only: bool = False) -> pd.DataFrame:
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

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst["trial_order"] = pd.to_numeric(wcst[trial_col], errors="coerce")
    complete6_ids = _get_complete6_ids(wcst)
    if complete6_ids:
        wcst = wcst[wcst["participant_id"].isin(complete6_ids)].copy()
    else:
        return pd.DataFrame()

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]
    if "timeout" in wcst.columns:
        wcst = wcst[~wcst["timeout"]]

    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col="trial_order", confirm_len=confirm_len)
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
    out_path = output_dir / "wcst_phase_3_2phase_6categories_split_half_reliability.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-len", type=int, default=3)
    parser.add_argument("--correct-only", action="store_true")
    args = parser.parse_args()
    main(args.confirm_len, args.correct_only)
