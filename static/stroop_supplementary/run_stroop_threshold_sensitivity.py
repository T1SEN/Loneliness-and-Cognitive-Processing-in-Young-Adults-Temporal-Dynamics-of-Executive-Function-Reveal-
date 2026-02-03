"""
Stroop threshold sensitivity analysis (segment-count variants).

Computes Stroop interference slope using alternative segment counts
(e.g., 2/3/4/6) and runs OLS regressions controlling for DASS + covariates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "static").exists():
    ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.analysis.utils import get_output_dir
from static.preprocessing.constants import STROOP_RT_MIN, STROOP_RT_MAX, get_results_dir
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.datasets import load_master_dataset
from static.preprocessing.stroop.qc import clean_stroop_trials


DEFAULT_SEGMENTS = [2, 3, 4, 6]


def _load_qc_ids() -> set[str]:
    ids_path = get_results_dir("overall") / "filtered_participant_ids.csv"
    if not ids_path.exists():
        return set()
    ids_df = pd.read_csv(ids_path, encoding="utf-8-sig")
    ids_df = ensure_participant_id(ids_df)
    if "participant_id" not in ids_df.columns:
        return set()
    return set(ids_df["participant_id"].dropna().astype(str))


def _load_stroop_trials() -> pd.DataFrame:
    trials_path = get_results_dir("overall") / "4a_stroop_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(trials_path, encoding="utf-8-sig")
    if df.empty:
        return df
    df = clean_stroop_trials(df)
    qc_ids = _load_qc_ids()
    if qc_ids:
        df = df[df["participant_id"].isin(qc_ids)].copy()
    return df


def _compute_slopes_for_k(stroop: pd.DataFrame, k: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    if stroop.empty:
        return pd.DataFrame(columns=["participant_id", f"stroop_slope_k{k}"])

    stroop = stroop.dropna(subset=["trial_order"]).copy()
    stroop["trial_order"] = pd.to_numeric(stroop["trial_order"], errors="coerce")
    stroop = stroop.dropna(subset=["trial_order"]).copy()

    for pid, grp in stroop.groupby("participant_id"):
        grp = grp.sort_values("trial_order").copy()
        n_trials = len(grp)
        if n_trials == 0:
            continue
        positions = np.arange(1, n_trials + 1)
        edges = np.linspace(0, n_trials, k + 1)
        seg = pd.cut(
            positions,
            bins=edges,
            labels=list(range(1, k + 1)),
            include_lowest=True,
        )
        grp["segment_k"] = seg.astype(int)

        valid = (
            grp["cond"].isin({"congruent", "incongruent"})
            & grp["correct"]
            & (~grp["timeout"])
            & grp["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
        )
        gvalid = grp[valid].copy()
        if gvalid.empty:
            slope = np.nan
        else:
            seg_means = (
                gvalid.groupby(["segment_k", "cond"])["rt_ms"].mean().unstack()
            )
            if "incongruent" not in seg_means.columns or "congruent" not in seg_means.columns:
                slope = np.nan
            else:
                seg_means["interference"] = (
                    seg_means["incongruent"] - seg_means["congruent"]
                )
                seg_means = seg_means.reset_index()[["segment_k", "interference"]].dropna()
                if len(seg_means) < 2:
                    slope = np.nan
                else:
                    x = seg_means["segment_k"].astype(float).to_numpy()
                    y = seg_means["interference"].to_numpy()
                    slope = float(np.polyfit(x, y, 1)[0])

        rows.append({"participant_id": pid, f"stroop_slope_k{k}": slope})

    return pd.DataFrame(rows)


def _run_ols(
    df: pd.DataFrame,
    outcome: str,
) -> dict[str, float] | None:
    required = [
        outcome,
        "z_ucla_score",
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
    ]
    sub = df.dropna(subset=required)
    if len(sub) < 30:
        return None

    full = smf.ols(
        f"{outcome} ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + C(gender_male)",
        data=sub,
    ).fit()
    reduced = smf.ols(
        f"{outcome} ~ z_dass_depression + z_dass_anxiety + z_dass_stress + "
        "z_age + C(gender_male)",
        data=sub,
    ).fit()

    delta_r2 = float(full.rsquared - reduced.rsquared)

    return {
        "n": int(len(sub)),
        "ucla_beta": float(full.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(full.bse.get("z_ucla_score", np.nan)),
        "ucla_p": float(full.pvalues.get("z_ucla_score", np.nan)),
        "delta_r2": delta_r2,
    }


def main(segment_counts: list[int] | None = None) -> None:
    segments = segment_counts or DEFAULT_SEGMENTS
    segments = [int(s) for s in segments]

    stroop = _load_stroop_trials()
    if stroop.empty:
        raise RuntimeError("No Stroop trials available after filtering.")

    # Compute slopes for each segment count.
    slopes = None
    for k in segments:
        dfk = _compute_slopes_for_k(stroop, k)
        if slopes is None:
            slopes = dfk
        else:
            slopes = slopes.merge(dfk, on="participant_id", how="outer")

    if slopes is None or slopes.empty:
        raise RuntimeError("No slope data computed.")

    # Merge with master dataset for standardized predictors.
    master = load_master_dataset(task="overall")
    qc_ids = _load_qc_ids()
    if qc_ids:
        master = master[master["participant_id"].isin(qc_ids)].copy()

    merged = master.merge(slopes, on="participant_id", how="inner")

    results: list[dict[str, float]] = []
    for k in segments:
        outcome = f"stroop_slope_k{k}"
        res = _run_ols(merged, outcome)
        if res is None:
            continue
        results.append({"segments": k, **res})

    results_df = pd.DataFrame(results)

    out_dir = get_output_dir("overall", bucket="supplementary")
    results_path = out_dir / "stroop_threshold_sensitivity.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    slopes_path = out_dir / "stroop_threshold_sensitivity_slopes.csv"
    slopes.to_csv(slopes_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {results_path}")
    if not results_df.empty:
        print(results_df.to_string(index=False))
    print(f"Saved: {slopes_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stroop segment-count sensitivity analysis.")
    parser.add_argument(
        "--segments",
        nargs="+",
        type=int,
        default=DEFAULT_SEGMENTS,
        help="Segment counts to test (default: 2 3 4 6).",
    )
    args = parser.parse_args()
    main(segment_counts=args.segments)
