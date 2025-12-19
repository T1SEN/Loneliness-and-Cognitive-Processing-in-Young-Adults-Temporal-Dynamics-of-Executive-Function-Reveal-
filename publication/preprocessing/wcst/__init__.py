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
from .dataset import build_wcst_dataset, get_wcst_complete_participants

MECHANISM_FILENAME = "5_wcst_mechanism_features.csv"


def compute_wcst_mechanism_features(data_dir: Path | None = None) -> pd.DataFrame:
    hmm_df = compute_wcst_hmm_features(data_dir=data_dir)
    rl_df = compute_wcst_rl_features(data_dir=data_dir)

    if hmm_df.empty and rl_df.empty:
        return pd.DataFrame()
    if hmm_df.empty:
        return rl_df
    if rl_df.empty:
        return hmm_df

    overlap = [c for c in rl_df.columns if c != "participant_id" and c in hmm_df.columns]
    if overlap:
        rl_df = rl_df.drop(columns=overlap)

    return hmm_df.merge(rl_df, on="participant_id", how="outer")


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
    "compute_wcst_mechanism_features",
    "load_or_compute_wcst_mechanism_features",
    "build_wcst_dataset",
    "get_wcst_complete_participants",
]
