"""Overall drift feature derivation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....constants import get_results_dir
from ....prp.dynamic.drift.features import derive_prp_drift_features
from ....stroop.dynamic.drift.features import derive_stroop_drift_features
from ....wcst.dynamic.drift.features import derive_wcst_drift_features
from ..._shared import get_overall_participant_ids, filter_to_overall_ids


def derive_overall_drift_features(data_dir: Path | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")
    overall_ids = get_overall_participant_ids(data_dir)

    prp_features = derive_prp_drift_features(data_dir=get_results_dir("prp"))
    stroop_features = derive_stroop_drift_features(data_dir=get_results_dir("stroop"))
    wcst_features = derive_wcst_drift_features(data_dir=get_results_dir("wcst"))

    if overall_ids:
        prp_features = filter_to_overall_ids(prp_features, overall_ids)
        stroop_features = filter_to_overall_ids(stroop_features, overall_ids)
        wcst_features = filter_to_overall_ids(wcst_features, overall_ids)

    merged = prp_features.merge(stroop_features, on="participant_id", how="inner")
    merged = merged.merge(wcst_features, on="participant_id", how="inner")
    return merged
