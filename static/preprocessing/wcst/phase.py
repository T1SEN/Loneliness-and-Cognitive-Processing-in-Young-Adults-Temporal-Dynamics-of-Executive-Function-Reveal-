"""
WCST phase labeling.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


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
