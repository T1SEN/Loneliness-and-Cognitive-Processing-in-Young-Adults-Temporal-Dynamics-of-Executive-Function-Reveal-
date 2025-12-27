"""Shared helpers for Stroop feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..constants import STROOP_RT_MIN
from .loaders import load_stroop_trials


def _normalize_condition(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    mapping = {
        "cong": "congruent",
        "congruent": "congruent",
        "inc": "incongruent",
        "incong": "incongruent",
        "incongruent": "incongruent",
        "neutral": "neutral",
        "neut": "neutral",
        "neu": "neutral",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if "incong" in cleaned:
        return "incongruent"
    if "cong" in cleaned:
        return "congruent"
    if "neut" in cleaned:
        return "neutral"
    return None


def _run_lengths(mask: pd.Series) -> List[int]:
    lengths: List[int] = []
    count = 0
    for val in mask:
        if val:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def _ez_ddm_params(rt_correct: pd.Series, acc: float, n_trials: int, s: float = 0.1) -> Dict[str, float]:
    if n_trials < 10:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}
    if rt_correct.empty:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}
    mrt = rt_correct.mean()
    vrt = rt_correct.var(ddof=1)
    if pd.isna(mrt) or pd.isna(vrt) or vrt <= 0:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}

    eps = 1.0 / (2.0 * n_trials)
    p = float(np.clip(acc, eps, 1.0 - eps))
    s2 = s ** 2
    logit = np.log(p / (1 - p))
    x = (logit * (p ** 2 * logit - p * logit + p - 0.5)) / vrt
    if x <= 0:
        return {"v": np.nan, "a": np.nan, "t0": np.nan}
    v = np.sign(p - 0.5) * s * (x ** 0.25)
    a = (s2 * logit) / v
    y = -v * a / s2
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    t0 = mrt - mdt
    return {"v": float(v), "a": float(a), "t0": float(t0)}


def prepare_stroop_trials(
    rt_min: float = STROOP_RT_MIN,
    rt_max: float | None = None,
    data_dir: None | str | Path = None,
) -> Dict[str, object]:
    stroop, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=True,
    )
    stroop_raw, _ = load_stroop_trials(
        data_dir=data_dir,
        rt_min=rt_min,
        rt_max=rt_max,
        apply_trial_filters=False,
    )

    for df in (stroop, stroop_raw):
        if "rt" in df.columns:
            df["rt_ms"] = pd.to_numeric(df["rt"], errors="coerce")

    trial_col = None
    for cand in ("trial", "trialIndex", "idx", "trial_index"):
        if cand in stroop.columns or cand in stroop_raw.columns:
            trial_col = cand
            break
    if trial_col:
        stroop["trial_order"] = pd.to_numeric(stroop[trial_col], errors="coerce")
        stroop_raw["trial_order"] = pd.to_numeric(stroop_raw[trial_col], errors="coerce")
        stroop = stroop.sort_values(["participant_id", "trial_order"])
        stroop_raw = stroop_raw.sort_values(["participant_id", "trial_order"])

    cond_col = None
    for cand in ("type", "condition", "cond", "congruency"):
        if cand in stroop.columns or cand in stroop_raw.columns:
            cond_col = cand
            break
    if cond_col:
        stroop["condition_norm"] = stroop[cond_col].apply(_normalize_condition)
        stroop_raw["condition_norm"] = stroop_raw[cond_col].apply(_normalize_condition)

    stroop_acc = stroop_raw.copy()
    if "timeout" in stroop_acc.columns:
        stroop_acc = stroop_acc[stroop_acc["timeout"] == False]

    return {
        "stroop": stroop,
        "stroop_raw": stroop_raw,
        "stroop_acc": stroop_acc,
        "cond_col": cond_col,
        "trial_col": trial_col,
    }
