"""Compute WCST pre-switch RT metrics for windows 1-5."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import WCST_RT_MIN, WCST_RT_MAX, get_results_dir
from publication.preprocessing.wcst._shared import prepare_wcst_trials


def _compute_pre_switch_features(
    windows: tuple[int, ...] = (1, 2, 3, 4, 5),
    correct_only: bool = False,
    suffix: str = "",
) -> pd.DataFrame:
    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]

    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or rule_col is None:
        return pd.DataFrame()

    records: list[dict[str, float]] = []

    for pid, grp in wcst.groupby("participant_id"):
        if trial_col and trial_col in grp.columns:
            grp = grp.sort_values(trial_col)
        grp = grp.reset_index(drop=True)

        rules = grp[rule_col].astype(str).str.lower().values
        rt_vals = pd.to_numeric(grp[rt_col], errors="coerce").astype(float).values

        correct = None
        if correct_only:
            if "correct" not in grp.columns:
                continue
            correct = grp["correct"].astype(bool).values

        if "is_rt_valid" in grp.columns:
            valid = grp["is_rt_valid"].astype(bool).values
        else:
            valid = np.isfinite(rt_vals)
            valid &= (rt_vals >= WCST_RT_MIN) & (rt_vals <= WCST_RT_MAX)
        if correct is not None:
            valid &= correct

        rt_vals = rt_vals.copy()
        rt_vals[~valid] = np.nan

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]

        per_window_means = {w: [] for w in windows}
        per_window_deltas = {w: [] for w in windows}
        per_window_counts = {w: 0 for w in windows}
        per_window_delta_counts = {w: 0 for w in windows}

        prev_idx = 0
        for idx in change_indices:
            seg_start = prev_idx
            for w in windows:
                if idx < w:
                    continue
                pre_slice = slice(idx - w, idx)
                pre_rts = rt_vals[pre_slice]
                if np.any(~np.isfinite(pre_rts)):
                    continue
                pre_mean = float(np.mean(pre_rts))
                per_window_means[w].append(pre_mean)
                per_window_counts[w] += 1

                base_end = idx - w
                if base_end > seg_start:
                    base_rts = rt_vals[seg_start:base_end]
                    base_rts = base_rts[np.isfinite(base_rts)]
                    if len(base_rts) > 0:
                        delta = pre_mean - float(np.mean(base_rts))
                        per_window_deltas[w].append(delta)
                        per_window_delta_counts[w] += 1

            prev_idx = idx

        record: dict[str, float] = {
            "participant_id": pid,
            f"wcst_pre_switch_n_switches{suffix}": float(len(change_indices)),
        }
        for w in windows:
            record[f"wcst_pre_switch_rt_mean_w{w}{suffix}"] = (
                float(np.mean(per_window_means[w])) if per_window_means[w] else np.nan
            )
            record[f"wcst_pre_switch_rt_delta_w{w}{suffix}"] = (
                float(np.mean(per_window_deltas[w])) if per_window_deltas[w] else np.nan
            )
            record[f"wcst_pre_switch_n_valid_w{w}{suffix}"] = float(per_window_counts[w])
            record[f"wcst_pre_switch_n_valid_delta_w{w}{suffix}"] = float(per_window_delta_counts[w])

        records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    df = _compute_pre_switch_features()
    out_path = get_results_dir("wcst") / "5_wcst_pre_switch_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not df.empty:
        print(df.head(10).to_string(index=False))

    df_correct = _compute_pre_switch_features(correct_only=True, suffix="_correct")
    out_path_correct = get_results_dir("wcst") / "5_wcst_pre_switch_features_correct.csv"
    df_correct.to_csv(out_path_correct, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path_correct}")
    if not df_correct.empty:
        print(df_correct.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

