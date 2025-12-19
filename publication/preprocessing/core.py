"""
Core helpers for preprocessing.
"""

from __future__ import annotations

import re
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from .constants import (
    PARTICIPANT_ID_ALIASES,
    MALE_TOKENS_EXACT,
    FEMALE_TOKENS_EXACT,
    MALE_TOKENS_CONTAINS,
    FEMALE_TOKENS_CONTAINS,
)


def ensure_participant_id(df: pd.DataFrame, warn_threshold: float = 1.0) -> pd.DataFrame:
    """
    Ensure there is exactly one 'participant_id' column.
    Prefers an existing participant_id column, otherwise renames common aliases.
    """
    canonical = "participant_id"
    if canonical not in df.columns:
        for col in df.columns:
            if col in PARTICIPANT_ID_ALIASES and col != canonical:
                df = df.rename(columns={col: canonical})
                break
    if canonical not in df.columns:
        raise KeyError("No participant id column found in dataframe.")

    missing_count = df[canonical].isna().sum()
    missing_pct = missing_count / len(df) * 100 if len(df) > 0 else 0

    if missing_pct > warn_threshold:
        warnings.warn(
            f"participant_id column has {missing_pct:.1f}% missing values ({missing_count}/{len(df)} rows). "
            "This may cause silent data loss in downstream analyses.",
            UserWarning,
        )

    aliases = [col for col in df.columns if col in PARTICIPANT_ID_ALIASES and col != canonical]
    if aliases:
        df = df.drop(columns=aliases)
    return df


def _normalize_gender_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^a-zㄱ-ㆎ가-힣]", "", cleaned)
    return cleaned


def normalize_gender_value(value: object) -> Optional[str]:
    """
    Normalize arbitrary gender text (Korean/English) to 'male'/'female'.
    Returns None if the value cannot be mapped.
    """
    token = _normalize_gender_string(value)
    if not token:
        return None
    if token in FEMALE_TOKENS_EXACT or any(t and t in token for t in FEMALE_TOKENS_CONTAINS):
        return "female"
    if token in MALE_TOKENS_EXACT or any(t and t in token for t in MALE_TOKENS_CONTAINS):
        return "male"
    return None


def normalize_gender_series(series: pd.Series) -> pd.Series:
    mapped = series.apply(normalize_gender_value)
    return pd.Series(mapped, index=series.index, dtype="object")


def coefficient_of_variation(series: pd.Series, min_n: int = 3) -> float:
    """Compute coefficient of variation (CV = std/mean)."""
    if len(series) < min_n or series.mean() == 0:
        return np.nan
    return series.std(ddof=1) / series.mean()


def compute_post_error_slowing(
    df: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    participant_col: str = "participant_id",
    order_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute post-error slowing (PES) for each participant.

    PES = mean(RT after error) - mean(RT after correct)
    """
    records = []
    preferred_order_cols = []
    if order_col:
        preferred_order_cols.append(order_col)
    preferred_order_cols.extend(["trial_index", "trialIndex", "trial", "idx", "order"])

    for pid, grp in df.groupby(participant_col):
        grp = grp.copy()
        for candidate in preferred_order_cols:
            if candidate in grp.columns:
                grp = grp.sort_values(candidate)
                break
        else:
            grp = grp.sort_index()
        grp["prev_correct"] = grp[correct_col].shift(1)

        post_error = grp[grp["prev_correct"] == False]
        post_correct = grp[grp["prev_correct"] == True]

        post_error_mean = post_error[rt_col].mean() if len(post_error) > 0 else np.nan
        post_correct_mean = post_correct[rt_col].mean() if len(post_correct) > 0 else np.nan

        pes = (
            post_error_mean - post_correct_mean
            if pd.notna(post_error_mean) and pd.notna(post_correct_mean)
            else np.nan
        )

        records.append({
            participant_col: pid,
            "pes": pes,
            "post_error_rt": post_error_mean,
            "post_correct_rt": post_correct_mean,
        })

    return pd.DataFrame(records)
