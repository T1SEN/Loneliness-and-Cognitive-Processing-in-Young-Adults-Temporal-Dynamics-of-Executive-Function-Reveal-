"""
Stroop mean interference RT split-half reliability.

Computes odd/even split-half reliability and permutation split-half
for the mean interference RT (incongruent - congruent).
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
from static.preprocessing.constants import get_results_dir, get_stroop_trials_path
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
    trials_path = get_stroop_trials_path("overall")
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


def _odd_even_interference(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    odd_vals = []
    even_vals = []
    for pid, grp in df.groupby("participant_id"):
        sub_con = grp[grp["cond"] == "congruent"].sort_values("trial_order")
        sub_inc = grp[grp["cond"] == "incongruent"].sort_values("trial_order")
        if sub_con.empty or sub_inc.empty:
            continue

        con = sub_con["rt_ms"].to_numpy()
        inc = sub_inc["rt_ms"].to_numpy()

        con_odd = con[::2].mean() if len(con[::2]) > 0 else np.nan
        con_even = con[1::2].mean() if len(con[1::2]) > 0 else np.nan
        inc_odd = inc[::2].mean() if len(inc[::2]) > 0 else np.nan
        inc_even = inc[1::2].mean() if len(inc[1::2]) > 0 else np.nan

        if np.isfinite(con_odd) and np.isfinite(inc_odd) and np.isfinite(con_even) and np.isfinite(inc_even):
            odd_vals.append(inc_odd - con_odd)
            even_vals.append(inc_even - con_even)

    return np.array(odd_vals, dtype=float), np.array(even_vals, dtype=float)


def _perm_split_interference(df: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    rs = []
    for _ in range(n_perm):
        a_vals = []
        b_vals = []
        for _, grp in df.groupby("participant_id"):
            sub_con = grp[grp["cond"] == "congruent"]
            sub_inc = grp[grp["cond"] == "incongruent"]
            if sub_con.empty or sub_inc.empty:
                continue

            con = sub_con["rt_ms"].to_numpy()
            inc = sub_inc["rt_ms"].to_numpy()

            idx_con = np.arange(len(con))
            idx_inc = np.arange(len(inc))
            rng.shuffle(idx_con)
            rng.shuffle(idx_inc)

            half_con = len(con) // 2
            half_inc = len(inc) // 2

            con_a = con[idx_con[:half_con]]
            con_b = con[idx_con[half_con:]]
            inc_a = inc[idx_inc[:half_inc]]
            inc_b = inc[idx_inc[half_inc:]]

            if len(con_a) == 0 or len(con_b) == 0 or len(inc_a) == 0 or len(inc_b) == 0:
                continue

            a_vals.append(inc_a.mean() - con_a.mean())
            b_vals.append(inc_b.mean() - con_b.mean())

        a_vals = np.array(a_vals, dtype=float)
        b_vals = np.array(b_vals, dtype=float)
        if len(a_vals) > 2:
            rs.append(np.corrcoef(a_vals, b_vals)[0, 1])

    rs = np.array(rs, dtype=float)
    return rs[np.isfinite(rs)]


def main(n_perm: int = 500, seed: int = 2026) -> None:
    df = _load_stroop_trials()
    if df.empty:
        raise RuntimeError("No Stroop trials available after filtering.")

    df = df[df["cond"].isin({"congruent", "incongruent"})].copy()
    df = df[df["correct"] & (~df["timeout"]) & (df["is_rt_valid"])].copy()
    df = df.dropna(subset=["trial_order", "rt_ms"]).copy()

    odd_vals, even_vals = _odd_even_interference(df)
    if len(odd_vals) < 3:
        raise RuntimeError("Not enough participants for split-half reliability.")

    r_odd_even = float(np.corrcoef(odd_vals, even_vals)[0, 1])
    r_sb = float(2 * r_odd_even / (1 + r_odd_even)) if r_odd_even > -1 else np.nan

    rng = np.random.default_rng(seed)
    rs_perm = _perm_split_interference(df, n_perm=n_perm, rng=rng)
    mean_r = float(np.mean(rs_perm)) if len(rs_perm) else np.nan
    ci_low = float(np.percentile(rs_perm, 2.5)) if len(rs_perm) else np.nan
    ci_high = float(np.percentile(rs_perm, 97.5)) if len(rs_perm) else np.nan
    mean_r_sb = float(2 * mean_r / (1 + mean_r)) if np.isfinite(mean_r) and (1 + mean_r) != 0 else np.nan

    results = pd.DataFrame(
        [
            {
                "method": "odd_even",
                "n": int(len(odd_vals)),
                "r": r_odd_even,
                "r_sb": r_sb,
            },
            {
                "method": "perm_mean",
                "n": int(len(odd_vals)),
                "n_perm": int(len(rs_perm)),
                "r": mean_r,
                "r_ci_low": ci_low,
                "r_ci_high": ci_high,
                "r_sb": mean_r_sb,
            },
        ]
    )

    out_dir = get_output_dir("overall", bucket="supplementary")
    out_path = out_dir / "stroop_interference_reliability.csv"
    results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stroop interference RT reliability.")
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    main(n_perm=args.n_perm, seed=args.seed)
