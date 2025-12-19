"""WCST preprocessing."""

from pathlib import Path

import pandas as pd

from ..constants import get_results_dir
from .loaders import load_wcst_trials, load_wcst_summary
from .filters import (
    WCSTQCCriteria,
    clean_wcst_trials,
    filter_wcst_rt_trials,
    compute_wcst_qc_stats,
    get_wcst_valid_participants,
)
from .features import derive_wcst_features
from .hmm_mechanism import (
    compute_wcst_hmm_features,
    load_or_compute_wcst_hmm_mechanism_features,
)
from .rl_mechanism import (
    compute_wcst_rl_features,
    load_or_compute_wcst_rl_mechanism_features,
)
from .wsls_mechanism import (
    compute_wcst_wsls_features,
    load_or_compute_wcst_wsls_mechanism_features,
)
from .bayesianrl_mechanism import (
    compute_wcst_bayesianrl_features,
    load_or_compute_wcst_bayesianrl_mechanism_features,
)
from .dataset import build_wcst_dataset, get_wcst_complete_participants

MECHANISM_FILENAME = "5_wcst_mechanism_features.csv"


def compute_wcst_mechanism_features(data_dir: Path | None = None) -> pd.DataFrame:
    hmm_df = compute_wcst_hmm_features(data_dir=data_dir)
    rl_df = compute_wcst_rl_features(data_dir=data_dir)
    wsls_df = compute_wcst_wsls_features(data_dir=data_dir)
    brl_df = compute_wcst_bayesianrl_features(data_dir=data_dir)

    mech_frames = [df for df in (hmm_df, rl_df, wsls_df, brl_df) if not df.empty]
    if not mech_frames:
        return pd.DataFrame()
    combined = mech_frames[0]
    for extra_df in mech_frames[1:]:
        overlap = [c for c in extra_df.columns if c != "participant_id" and c in combined.columns]
        if overlap:
            extra_df = extra_df.drop(columns=overlap)
        combined = combined.merge(extra_df, on="participant_id", how="outer")
    return combined


def load_or_compute_wcst_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("wcst")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_wcst_mechanism_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] WCST mechanism features saved: {output_path}")
    return features

__all__ = [
    "load_wcst_trials",
    "load_wcst_summary",
    "WCSTQCCriteria",
    "clean_wcst_trials",
    "filter_wcst_rt_trials",
    "compute_wcst_qc_stats",
    "get_wcst_valid_participants",
    "derive_wcst_features",
    "compute_wcst_hmm_features",
    "load_or_compute_wcst_hmm_mechanism_features",
    "compute_wcst_rl_features",
    "load_or_compute_wcst_rl_mechanism_features",
    "compute_wcst_wsls_features",
    "load_or_compute_wcst_wsls_mechanism_features",
    "compute_wcst_bayesianrl_features",
    "load_or_compute_wcst_bayesianrl_mechanism_features",
    "compute_wcst_mechanism_features",
    "load_or_compute_wcst_mechanism_features",
    "build_wcst_dataset",
    "get_wcst_complete_participants",
]
