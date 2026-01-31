"""WCST feature derivation (manuscript outcomes + supplementary phases)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..constants import WCST_RT_MIN, WCST_RT_MAX
from ._shared import prepare_wcst_trials

N_CATEGORIES = 6
PHASE_ORDER = ["exploration", "confirmation", "exploitation"]


def _label_wcst_phases(
    wcst: pd.DataFrame,
    rule_col: str,
    trial_col: str,
    confirm_len: int = 3,
) -> pd.DataFrame:
    df = wcst.sort_values(["participant_id", trial_col]).copy()
    df["category_num"] = np.nan
    df["phase"] = pd.NA

    for pid, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col).copy()
        idxs = grp_sorted.index.to_list()
        rules = (
            grp_sorted[rule_col]
            .astype(str)
            .str.lower()
            .replace({"color": "colour"})
            .to_numpy()
        )
        correct = grp_sorted["correct"].astype(bool).to_numpy()

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]

        for cat_idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > N_CATEGORIES:
                break
            if start >= end:
                continue

            reacq_start = None
            reacq_idx = None
            if confirm_len >= 1:
                for j in range(start, end - (confirm_len - 1)):
                    if np.all(correct[j : j + confirm_len]):
                        reacq_start = j
                        reacq_idx = j + confirm_len - 1
                        break

            for i in range(start, end):
                row_idx = idxs[i]
                df.at[row_idx, "category_num"] = float(cat_idx)
                if reacq_start is None:
                    df.at[row_idx, "phase"] = "exploration"
                elif i < reacq_start:
                    df.at[row_idx, "phase"] = "exploration"
                elif i <= reacq_idx:
                    df.at[row_idx, "phase"] = "confirmation"
                else:
                    df.at[row_idx, "phase"] = "exploitation"

    df["phase"] = pd.Categorical(df["phase"], categories=PHASE_ORDER)
    return df


def _valid_rt(series: pd.Series, rt_max: float | None) -> pd.Series:
    valid = series.notna() & (series >= WCST_RT_MIN)
    if rt_max is not None:
        valid &= series <= rt_max
    return valid


def derive_wcst_features(
    data_dir: None | str | Path = None,
    confirm_len: int = 3,
    rt_max: float | None = None,
) -> pd.DataFrame:
    prepared = prepare_wcst_trials(data_dir=data_dir, filter_rt=False)
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]

    if not isinstance(wcst, pd.DataFrame) or wcst.empty:
        return pd.DataFrame(columns=["participant_id"])
    if rt_col is None or trial_col is None or rule_col is None:
        return pd.DataFrame(columns=["participant_id"])
    if "correct" not in wcst.columns:
        return pd.DataFrame(columns=["participant_id"])

    wcst = wcst.copy()
    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")

    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]
    else:
        rt_max_val = WCST_RT_MAX if rt_max is None else rt_max
        wcst = wcst[_valid_rt(wcst["rt_ms"], rt_max_val)]

    wcst["rule_norm"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = _label_wcst_phases(wcst, rule_col="rule_norm", trial_col=trial_col, confirm_len=confirm_len)

    phase_means = (
        wcst.groupby(["participant_id", "phase"], observed=False)["rt_ms"]
        .mean()
        .unstack()
    )
    phase_means.index = phase_means.index.astype(str)

    out = pd.DataFrame(index=phase_means.index)
    out.index.name = "participant_id"
    out["wcst_exploration_rt"] = phase_means.get("exploration")
    out["wcst_confirmation_rt"] = phase_means.get("confirmation")
    out["wcst_exploitation_rt"] = phase_means.get("exploitation")
    out["wcst_confirmation_minus_exploitation_rt"] = (
        out["wcst_confirmation_rt"] - out["wcst_exploitation_rt"]
    )

    pre_map = wcst["phase"].map({"exploration": "pre_exploitation", "confirmation": "pre_exploitation", "exploitation": "exploitation"})
    wcst_pre = wcst.copy()
    wcst_pre["phase_pre"] = pre_map
    pre_means = (
        wcst_pre.groupby(["participant_id", "phase_pre"], observed=False)["rt_ms"]
        .mean()
        .unstack()
    )
    pre_means.index = pre_means.index.astype(str)
    out["wcst_pre_exploitation_rt"] = pre_means.get("pre_exploitation")
    out["wcst_pre_exploitation_minus_exploitation_rt"] = (
        out["wcst_pre_exploitation_rt"] - pre_means.get("exploitation")
    )

    return out.reset_index()
