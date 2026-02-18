"""
Overall summary loader (public-only).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import get_public_file
from .core import ensure_participant_id


def load_overall_summary(data_dir: Path) -> pd.DataFrame:
    """
    Load per-participant summary metrics required by downstream analyses.

    In the public-only runtime, these summary metrics are sourced from
    `features_public.csv`.
    """
    _ = data_dir  # kept for API compatibility

    features_path = get_public_file("features")
    if not features_path.exists():
        return pd.DataFrame(columns=["participant_id"])

    df = pd.read_csv(features_path, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame(columns=["participant_id"])
    df = ensure_participant_id(df)

    keep_cols = [
        "participant_id",
        "stroop_interference",
        "wcst_perseverative_error_rate",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    if "participant_id" not in keep_cols:
        return pd.DataFrame(columns=["participant_id"])
    return df[keep_cols].drop_duplicates(subset=["participant_id"], keep="last").copy()
