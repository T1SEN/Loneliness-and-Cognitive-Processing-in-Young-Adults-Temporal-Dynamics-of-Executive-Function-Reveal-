"""
Bootstrap stability for Stroop interference slope (4 segments).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "static").exists():
    ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.analysis.utils import get_output_dir
from static.preprocessing.constants import STROOP_RT_MIN, STROOP_RT_MAX, get_results_dir
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.stroop.qc import clean_stroop_trials


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


def _prepare_boot_data(df: pd.DataFrame) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    df = df.dropna(subset=["trial_order"]).copy()
    df["trial_order"] = pd.to_numeric(df["trial_order"], errors="coerce")
    df = df.dropna(subset=["trial_order"]).copy()

    boot_data: dict[str, dict[int, dict[str, np.ndarray]]] = {}

    for pid, grp in df.groupby("participant_id"):
        grp = grp.sort_values("trial_order").copy()
        n_trials = len(grp)
        if n_trials == 0:
            continue
        positions = np.arange(1, n_trials + 1)
        edges = np.linspace(0, n_trials, 5)
        seg = pd.cut(positions, bins=edges, labels=[1, 2, 3, 4], include_lowest=True)
        grp["segment"] = seg.astype(int)

        valid = (
            grp["cond"].isin({"congruent", "incongruent"})
            & grp["correct"]
            & (~grp["timeout"])
            & grp["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
        )
        gvalid = grp[valid].copy()
        if gvalid.empty:
            continue

        per_seg = {s: {"congruent": None, "incongruent": None} for s in [1, 2, 3, 4]}
        for s, segdf in gvalid.groupby("segment"):
            for cond in ["congruent", "incongruent"]:
                arr = segdf.loc[segdf["cond"] == cond, "rt_ms"].to_numpy()
                per_seg[s][cond] = arr

        boot_data[str(pid)] = per_seg

    return boot_data


def main(n_boot: int = 500, seed: int = 2026, save_per_participant: bool = False) -> None:
    df = _load_stroop_trials()
    if df.empty:
        raise RuntimeError("No Stroop trials available after filtering.")

    rng = np.random.default_rng(seed)
    boot_data = _prepare_boot_data(df)
    if not boot_data:
        raise RuntimeError("No bootstrap data available.")

    rows = []
    for pid, per_seg in boot_data.items():
        slopes = []
        for _ in range(n_boot):
            inters = []
            valid_seg = True
            for s in [1, 2, 3, 4]:
                inc = per_seg[s]["incongruent"]
                con = per_seg[s]["congruent"]
                if inc is None or con is None or len(inc) == 0 or len(con) == 0:
                    valid_seg = False
                    break
                inc_s = rng.choice(inc, size=len(inc), replace=True)
                con_s = rng.choice(con, size=len(con), replace=True)
                inters.append(inc_s.mean() - con_s.mean())
            if not valid_seg:
                slopes.append(np.nan)
                continue
            x = np.array([1, 2, 3, 4], dtype=float)
            slopes.append(np.polyfit(x, np.array(inters, dtype=float), 1)[0])

        slopes = np.array(slopes, dtype=float)
        slopes = slopes[np.isfinite(slopes)]
        if len(slopes) < 10:
            continue
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        rows.append(
            {
                "participant_id": pid,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "ci_width": float(hi - lo),
                "ci_includes_zero": bool(lo <= 0 <= hi),
            }
        )

    boot_df = pd.DataFrame(rows)
    if boot_df.empty:
        raise RuntimeError("Bootstrap results empty.")

    summary = pd.DataFrame(
        [
            {
                "n_participants": int(len(boot_df)),
                "n_boot": int(n_boot),
                "mean_ci_width": float(boot_df["ci_width"].mean()),
                "median_ci_width": float(boot_df["ci_width"].median()),
                "pct_ci_includes_zero": float(boot_df["ci_includes_zero"].mean()),
            }
        ]
    )

    out_dir = get_output_dir("overall", bucket="supplementary")
    summary_path = out_dir / "stroop_slope_bootstrap_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")
    print(summary.to_string(index=False))

    if save_per_participant:
        full_path = out_dir / "stroop_slope_bootstrap_per_participant.csv"
        boot_df.to_csv(full_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {full_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap stability for Stroop slope.")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-per-participant", action="store_true")
    args = parser.parse_args()
    main(n_boot=args.n_boot, seed=args.seed, save_per_participant=args.save_per_participant)
