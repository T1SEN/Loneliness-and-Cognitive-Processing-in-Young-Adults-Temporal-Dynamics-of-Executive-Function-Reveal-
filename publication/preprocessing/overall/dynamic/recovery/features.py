"""Overall recovery feature derivation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....prp.dynamic.recovery.features import derive_prp_recovery_features
from ....stroop.dynamic.recovery.features import derive_stroop_recovery_features
from ....wcst.dynamic.recovery.features import derive_wcst_recovery_features


def derive_overall_recovery_features(data_dir: Path | None = None) -> pd.DataFrame:
    prp_features = derive_prp_recovery_features(data_dir=data_dir)
    stroop_features = derive_stroop_recovery_features(data_dir=data_dir)
    wcst_features = derive_wcst_recovery_features(data_dir=data_dir)

    merged = prp_features.merge(stroop_features, on="participant_id", how="inner")
    merged = merged.merge(wcst_features, on="participant_id", how="inner")
    return merged
