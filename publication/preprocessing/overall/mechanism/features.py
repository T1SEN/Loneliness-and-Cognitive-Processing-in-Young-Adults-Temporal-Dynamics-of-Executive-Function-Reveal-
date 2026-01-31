"""Overall mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...constants import get_results_dir
from ...stroop.mechanism.exgaussian import load_or_compute_stroop_mechanism_features
from ...stroop.mechanism.hmm_event import load_or_compute_stroop_hmm_event_features
from ...wcst.mechanism.bayesianrl import load_or_compute_wcst_bayesianrl_mechanism_features
from ...wcst.mechanism.hmm import load_or_compute_wcst_hmm_mechanism_features
from ...wcst.mechanism.rl import load_or_compute_wcst_rl_mechanism_features
from ...wcst.mechanism.wsls import load_or_compute_wcst_wsls_mechanism_features
from .._shared import get_overall_participant_ids, filter_to_overall_ids


def _merge_feature_frames(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra.empty:
        return base
    if base.empty:
        return extra
    overlap = [c for c in extra.columns if c != "participant_id" and c in base.columns]
    if overlap:
        base = base.drop(columns=overlap)
    return base.merge(extra, on="participant_id", how="left")


def derive_overall_mechanism_features(data_dir: Path | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")
    overall_ids = get_overall_participant_ids(data_dir)

    features_df = pd.DataFrame()

    for part in (
        load_or_compute_stroop_mechanism_features(data_dir=get_results_dir("stroop")),
        load_or_compute_stroop_hmm_event_features(data_dir=get_results_dir("stroop")),
        load_or_compute_wcst_hmm_mechanism_features(data_dir=get_results_dir("wcst")),
        load_or_compute_wcst_rl_mechanism_features(data_dir=get_results_dir("wcst")),
        load_or_compute_wcst_wsls_mechanism_features(data_dir=get_results_dir("wcst")),
        load_or_compute_wcst_bayesianrl_mechanism_features(data_dir=get_results_dir("wcst")),
    ):
        if overall_ids:
            part = filter_to_overall_ids(part, overall_ids)
        features_df = _merge_feature_frames(features_df, part)

    return features_df
