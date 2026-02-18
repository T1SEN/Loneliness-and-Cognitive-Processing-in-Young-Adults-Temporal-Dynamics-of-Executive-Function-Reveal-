"""
Overall feature loader and builder (task-specific QC).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import get_public_file, get_results_dir
from .core import ensure_participant_id
from .public_validate import get_common_public_ids
from .stroop.qc import prepare_stroop_trials
from .stroop.features import build_stroop_features
from .wcst.qc import prepare_wcst_trials
from .wcst.features import build_wcst_features


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    _ = data_dir  # public-only runtime
    features_path = get_public_file("features")
    if not features_path.exists():
        return pd.DataFrame()

    features = pd.read_csv(features_path, encoding="utf-8-sig")
    features = ensure_participant_id(features)
    return features


def build_overall_features(
    data_dir: Path | None = None,
    qc_ids: set[str] | None = None,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")

    stroop = prepare_stroop_trials(data_dir)
    wcst = prepare_wcst_trials(data_dir)

    if qc_ids is None:
        qc_ids = get_common_public_ids(validate=True)

    if qc_ids:
        stroop = stroop[stroop["participant_id"].isin(qc_ids)].copy()
        wcst = wcst[wcst["participant_id"].isin(qc_ids)].copy()

    if verbose:
        print(f"  [FEATURES] Stroop trials: {len(stroop)}")
        print(f"  [FEATURES] WCST trials: {len(wcst)}")

    if qc_ids:
        participant_ids = sorted(qc_ids)
    else:
        participant_ids = sorted(set(stroop["participant_id"]) | set(wcst["participant_id"]))
    features = pd.DataFrame({"participant_id": participant_ids})

    stroop_features = build_stroop_features(stroop)
    if not stroop_features.empty:
        features = features.merge(stroop_features, on="participant_id", how="left")

    wcst_features = build_wcst_features(wcst)
    if not wcst_features.empty:
        features = features.merge(wcst_features, on="participant_id", how="left")

    if save:
        output_path = get_public_file("features")
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  [FEATURES] Saved: {output_path}")

    return features
