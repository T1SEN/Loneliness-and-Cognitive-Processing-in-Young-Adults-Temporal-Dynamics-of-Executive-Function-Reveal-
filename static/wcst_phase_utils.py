"""WCST phase helpers for overall-only pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from static.preprocessing.constants import get_results_dir
from static.preprocessing.core import ensure_participant_id


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )
    return mapped.fillna(False)


def _read_trials_csv(data_dir: Path) -> pd.DataFrame:
    trials_path = data_dir / "4b_wcst_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(trials_path, encoding="utf-8-sig")
    if df.empty:
        return df
    df = ensure_participant_id(df)
    return df


def prepare_wcst_trials(data_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load WCST trials from overall-complete data.

    Returns a dict compatible with legacy scripts:
        - wcst: trial-level dataframe
        - rt_col: column name for RT (ms)
        - trial_col: column name for trial order
        - rule_col: column name for rule at that time
    """
    if data_dir is None:
        data_dir = get_results_dir("overall")

    wcst = _read_trials_csv(data_dir)
    if wcst.empty:
        return {"wcst": wcst, "rt_col": None, "trial_col": None, "rule_col": None}

    rt_col = None
    for candidate in ["rt_ms", "reactionTimeMs", "resp_time_ms"]:
        if candidate in wcst.columns:
            rt_col = candidate
            break

    trial_col = None
    for candidate in ["trial_index", "trialIndex", "trial"]:
        if candidate in wcst.columns:
            trial_col = candidate
            break

    rule_col = None
    for candidate in ["ruleAtThatTime", "rule_at_that_time", "rule"]:
        if candidate in wcst.columns:
            rule_col = candidate
            break

    for col in ["correct", "isPE", "isNPE", "timeout", "is_rt_valid"]:
        if col in wcst.columns:
            wcst[col] = _coerce_bool_series(wcst[col]).astype(bool)

    return {
        "wcst": wcst,
        "rt_col": rt_col,
        "trial_col": trial_col,
        "rule_col": rule_col,
    }


def label_wcst_phases(
    wcst: pd.DataFrame,
    rule_col: str,
    trial_col: str,
    confirm_len: int = 3,
    n_categories: int = 6,
    phase_order: Sequence[str] = ("exploration", "confirmation", "exploitation"),
) -> pd.DataFrame:
    """
    Assign WCST phases within each rule block.

    exploration: category onset -> first correct
    confirmation: first correct -> confirm_len consecutive correct
    exploitation: after confirm_len consecutive correct
    """
    df = wcst.sort_values(["participant_id", trial_col]).copy()
    df["category_num"] = np.nan
    df["phase"] = pd.NA

    for _, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col).copy()
        idxs = grp_sorted.index.to_list()
        rules = (
            grp_sorted[rule_col].astype(str).str.lower().replace({"color": "colour"}).to_numpy()
        )
        correct = grp_sorted["correct"].astype(bool).to_numpy()

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]

        for cat_idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > n_categories:
                break
            if start >= end:
                continue

            first_correct = None
            for j in range(start, end):
                if correct[j]:
                    first_correct = j
                    break

            reacq_idx = None
            if confirm_len >= 1:
                for j in range(start, end - (confirm_len - 1)):
                    if np.all(correct[j : j + confirm_len]):
                        reacq_idx = j + confirm_len - 1
                        break

            for i in range(start, end):
                row_idx = idxs[i]
                df.at[row_idx, "category_num"] = float(cat_idx)
                if first_correct is None:
                    df.at[row_idx, "phase"] = "exploration"
                elif i < first_correct:
                    df.at[row_idx, "phase"] = "exploration"
                elif reacq_idx is None or i <= reacq_idx:
                    df.at[row_idx, "phase"] = "confirmation"
                else:
                    df.at[row_idx, "phase"] = "exploitation"

    df["phase"] = pd.Categorical(df["phase"], categories=list(phase_order))
    return df
