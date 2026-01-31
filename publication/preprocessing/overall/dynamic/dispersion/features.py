"""Overall dispersion feature derivation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....constants import get_results_dir
from ....stroop.dynamic.dispersion.features import derive_stroop_dispersion_features
from ....wcst.dynamic.dispersion.features import derive_wcst_dispersion_features
from ..._shared import get_overall_participant_ids, filter_to_overall_ids


def derive_overall_dispersion_features(data_dir: Path | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")
    overall_ids = get_overall_participant_ids(data_dir)

    stroop_features = derive_stroop_dispersion_features(data_dir=get_results_dir("stroop"))
    wcst_features = derive_wcst_dispersion_features(data_dir=get_results_dir("wcst"))

    if overall_ids:
        stroop_features = filter_to_overall_ids(stroop_features, overall_ids)
        wcst_features = filter_to_overall_ids(wcst_features, overall_ids)

    merged = stroop_features.merge(wcst_features, on="participant_id", how="inner")
    return merged
