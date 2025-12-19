"""Overall feature derivation (merged trial features)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..prp.features import derive_prp_features
from ..stroop.features import derive_stroop_features
from ..wcst.features import derive_wcst_features


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Merge trial-derived features from PRP, Stroop, and WCST.
    """
    prp_features = derive_prp_features(data_dir=data_dir)
    stroop_features = derive_stroop_features(data_dir=data_dir)
    wcst_features = derive_wcst_features(data_dir=data_dir)

    merged = prp_features.merge(stroop_features, on="participant_id", how="inner")
    merged = merged.merge(wcst_features, on="participant_id", how="inner")
    return merged
