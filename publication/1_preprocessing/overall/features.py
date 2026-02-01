"""Overall feature loader (precomputed)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import get_results_dir
from ..core import ensure_participant_id


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")

    features_path = data_dir / "5_overall_features.csv"
    if not features_path.exists():
        return pd.DataFrame()

    features = pd.read_csv(features_path, encoding="utf-8-sig")
    features = ensure_participant_id(features)
    return features
