"""Overall feature derivation (manuscript core only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import get_results_dir
from ..stroop.features import derive_stroop_features
from ..wcst.features import derive_wcst_features
from ._shared import get_overall_participant_ids, filter_to_overall_ids


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")
    overall_ids = get_overall_participant_ids(data_dir)

    stroop_features = derive_stroop_features(data_dir=get_results_dir("stroop"))
    wcst_features = derive_wcst_features(data_dir=get_results_dir("wcst"))

    if overall_ids:
        stroop_features = filter_to_overall_ids(stroop_features, overall_ids)
        wcst_features = filter_to_overall_ids(wcst_features, overall_ids)

    if stroop_features.empty or wcst_features.empty:
        return pd.DataFrame()

    merged = stroop_features.merge(wcst_features, on="participant_id", how="inner")
    return merged
