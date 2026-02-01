"""
Overall feature loader and builder (task-specific QC).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import get_results_dir
from .core import ensure_participant_id
from .stroop.qc import prepare_stroop_trials
from .stroop.features import build_stroop_features
from .wcst.qc import prepare_wcst_trials
from .wcst.features import build_wcst_features


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")

    features_path = data_dir / "5_overall_features.csv"
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
        ids_path = data_dir / "filtered_participant_ids.csv"
        if ids_path.exists():
            ids_df = pd.read_csv(ids_path, encoding="utf-8-sig")
            ids_df = ensure_participant_id(ids_df)
            qc_ids = set(ids_df["participant_id"].dropna().astype(str))

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
        output_path = data_dir / "5_overall_features.csv"
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  [FEATURES] Saved: {output_path}")

    return features
