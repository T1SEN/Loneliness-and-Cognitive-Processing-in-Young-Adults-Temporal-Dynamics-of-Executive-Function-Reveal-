"""Overall mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...prp.mechanism.bottleneck import load_or_compute_prp_bottleneck_mechanism_features
from ...prp.mechanism.exgaussian import load_or_compute_prp_mechanism_features
from ...prp.mechanism.hmm_event import load_or_compute_prp_hmm_event_features
from ...stroop.mechanism.exgaussian import load_or_compute_stroop_mechanism_features
from ...stroop.mechanism.hmm_event import load_or_compute_stroop_hmm_event_features
from ...stroop.mechanism.lba import load_or_compute_stroop_lba_mechanism_features
from ...wcst.mechanism.bayesianrl import load_or_compute_wcst_bayesianrl_mechanism_features
from ...wcst.mechanism.hmm import load_or_compute_wcst_hmm_mechanism_features
from ...wcst.mechanism.rl import load_or_compute_wcst_rl_mechanism_features
from ...wcst.mechanism.wsls import load_or_compute_wcst_wsls_mechanism_features


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
    features_df = pd.DataFrame()

    for part in (
        load_or_compute_prp_mechanism_features(data_dir=data_dir),
        load_or_compute_prp_hmm_event_features(data_dir=data_dir),
        load_or_compute_prp_bottleneck_mechanism_features(data_dir=data_dir),
        load_or_compute_stroop_mechanism_features(data_dir=data_dir),
        load_or_compute_stroop_hmm_event_features(data_dir=data_dir),
        load_or_compute_stroop_lba_mechanism_features(data_dir=data_dir),
        load_or_compute_wcst_hmm_mechanism_features(data_dir=data_dir),
        load_or_compute_wcst_rl_mechanism_features(data_dir=data_dir),
        load_or_compute_wcst_wsls_mechanism_features(data_dir=data_dir),
        load_or_compute_wcst_bayesianrl_mechanism_features(data_dir=data_dir),
    ):
        features_df = _merge_feature_frames(features_df, part)

    return features_df
