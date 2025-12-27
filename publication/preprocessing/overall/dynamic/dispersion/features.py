"""Overall dispersion feature derivation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....prp.dynamic.dispersion.features import derive_prp_dispersion_features
from ....stroop.dynamic.dispersion.features import derive_stroop_dispersion_features
from ....wcst.dynamic.dispersion.features import derive_wcst_dispersion_features


def derive_overall_dispersion_features(data_dir: Path | None = None) -> pd.DataFrame:
    prp_features = derive_prp_dispersion_features(data_dir=data_dir)
    stroop_features = derive_stroop_dispersion_features(data_dir=data_dir)
    wcst_features = derive_wcst_dispersion_features(data_dir=data_dir)

    merged = prp_features.merge(stroop_features, on="participant_id", how="inner")
    merged = merged.merge(wcst_features, on="participant_id", how="inner")
    return merged
