"""PRP QC filters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

from ..constants import RAW_DIR, DEFAULT_RT_MIN, PRP_RT_MAX, DEFAULT_SOA_SHORT, DEFAULT_SOA_LONG


@dataclass
class PRPQCCriteria:
    n_trials_required: int = 120
    min_accuracy: Optional[float] = None
    require_valid_trials: bool = True
    require_metrics: bool = True


def _get_prp_valid_trial_participants(data_dir: Path, verbose: bool = False) -> Tuple[Set[str], Set[str]]:
    prp_path = data_dir / "4a_prp_trials.csv"
    if not prp_path.exists():
        if verbose:
            print(f"[WARN] PRP trials file not found: {prp_path}")
        return set(), set()

    prp_trials = pd.read_csv(prp_path, encoding="utf-8-sig")

    prp_trials["t1_correct"] = prp_trials["t1_correct"].fillna(False).astype(bool)
    prp_trials["t2_correct"] = prp_trials["t2_correct"].fillna(False).astype(bool)
    if "t2_timeout" in prp_trials.columns:
        prp_trials["t2_timeout"] = prp_trials["t2_timeout"].fillna(False).astype(bool)
    else:
        prp_trials["t2_timeout"] = False

    rt_col = "t2_rt_ms" if "t2_rt_ms" in prp_trials.columns else (
        "t2_rt" if "t2_rt" in prp_trials.columns else None
    )
    soa_col = "soa_nominal_ms" if "soa_nominal_ms" in prp_trials.columns else (
        "soa" if "soa" in prp_trials.columns else None
    )

    if rt_col is None or soa_col is None:
        return set(), set(prp_trials["participantId"].unique())

    valid = prp_trials[
        (prp_trials["t1_correct"] == True)
        & (prp_trials["t2_correct"] == True)
        & (prp_trials["t2_timeout"] == False)
        & (prp_trials[rt_col] > DEFAULT_RT_MIN)
        & (prp_trials[rt_col] < PRP_RT_MAX)
    ].copy()

    def bin_soa(soa):
        if pd.isna(soa):
            return "other"
        if soa <= DEFAULT_SOA_SHORT:
            return "short"
        if soa >= DEFAULT_SOA_LONG:
            return "long"
        return "other"

    valid["soa_bin"] = valid[soa_col].apply(bin_soa)
    valid = valid[valid["soa_bin"].isin(["short", "long"])]

    valid_ids = set(valid["participantId"].unique())
    all_ids = set(prp_trials["participantId"].unique())
    no_valid_ids = all_ids - valid_ids

    return valid_ids, no_valid_ids


def get_prp_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[PRPQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = PRPQCCriteria()

    summary_path = data_dir / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] cognitive summary file not found: {summary_path}")
        return set()

    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary_df["testName"] = summary_df["testName"].str.lower()

    prp_df = summary_df[summary_df["testName"] == "prp"].copy()
    if prp_df.empty:
        return set()

    if criteria.n_trials_required > 0:
        prp_df = prp_df[prp_df["n_trials"].fillna(0).astype(int) == criteria.n_trials_required]

    if criteria.require_metrics:
        mask = (
            prp_df["mrt_t2"].notna()
            & prp_df["rt2_soa_50"].notna()
            & prp_df["rt2_soa_1200"].notna()
        )
        prp_df = prp_df[mask]

    valid_ids = set(prp_df["participantId"].unique())

    if criteria.min_accuracy is not None:
        acc_threshold = criteria.min_accuracy * 100
        if "acc_t2" in prp_df.columns:
            prp_df_acc = prp_df[prp_df["acc_t2"].fillna(0) >= acc_threshold]
            valid_ids = set(prp_df_acc["participantId"].unique())

    if criteria.require_valid_trials:
        _, trial_invalid = _get_prp_valid_trial_participants(data_dir, verbose)
        excluded = valid_ids & trial_invalid
        if excluded and verbose:
            print(f"  [INFO] PRP excluded for 0 valid trials: {len(excluded)}")
        valid_ids -= trial_invalid

    if verbose:
        all_prp = set(summary_df[summary_df["testName"] == "prp"]["participantId"].unique())
        excluded = all_prp - valid_ids
        if excluded:
            print(f"  [INFO] PRP QC failed: {len(excluded)}")

    return valid_ids
