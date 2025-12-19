"""WCST QC filters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd

from ..constants import (
    RAW_DIR,
    WCST_RT_MIN,
    WCST_RT_MAX,
    WCST_VALID_CONDS,
    WCST_VALID_CARDS,
    WCST_MIN_TRIALS,
    WCST_MIN_MEDIAN_RT,
    WCST_MAX_SINGLE_CHOICE,
)
from ..core import ensure_participant_id


@dataclass
class WCSTQCCriteria:
    max_total_errors: int = 60
    max_perseverative_responses: int = 60
    min_completed_categories: int = 1
    require_metrics: bool = True
    min_trials: int = WCST_MIN_TRIALS
    min_median_rt: float = WCST_MIN_MEDIAN_RT
    max_single_choice_ratio: float = WCST_MAX_SINGLE_CHOICE
    rt_min: float = WCST_RT_MIN
    rt_max: float = WCST_RT_MAX


def clean_wcst_trials(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = ensure_participant_id(df)

    if "trial_index" not in df.columns and "trialIndex" in df.columns:
        df = df.rename(columns={"trialIndex": "trial_index"})
    if "cond" not in df.columns and "ruleAtThatTime" in df.columns:
        df["cond"] = df["ruleAtThatTime"]
    if "chosenCard" not in df.columns:
        for cand in ("chosen_card", "chosen", "cardChoice"):
            if cand in df.columns:
                df["chosenCard"] = df[cand]
                break
    if "rt_ms" not in df.columns:
        for cand in ("reactionTimeMs", "resp_time_ms"):
            if cand in df.columns:
                df["rt_ms"] = df[cand]
                break

    if "trial_index" in df.columns:
        sort_cols = ["participant_id", "trial_index"]
        if "timestamp" in df.columns:
            sort_cols.append("timestamp")
        df = df.sort_values(sort_cols)
        before = len(df)
        df = df.drop_duplicates(subset=["participant_id", "trial_index"], keep="first")
        dup_removed = before - len(df)
    else:
        dup_removed = 0

    required_cols = ["correct", "cond", "chosenCard", "rt_ms"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise KeyError(f"WCST trials missing required columns: {missing_required}")

    before = len(df)
    df = df.dropna(subset=required_cols)
    missing_removed = before - len(df)

    df["cond"] = df["cond"].astype(str).str.lower().str.strip()
    df["cond"] = df["cond"].replace({"color": "colour"})
    df = df[df["cond"].isin(WCST_VALID_CONDS)]

    df["chosenCard"] = df["chosenCard"].astype(str).str.strip().str.lower()
    df = df[df["chosenCard"].isin(WCST_VALID_CARDS)]

    df["rt_ms"] = pd.to_numeric(df["rt_ms"], errors="coerce")
    df = df[df["rt_ms"].notna()]
    df = df[df["rt_ms"] >= 0]

    stats = {
        "duplicates_removed": dup_removed,
        "missing_removed": missing_removed,
    }
    return df, stats


def filter_wcst_rt_trials(
    df: pd.DataFrame,
    rt_min: float = WCST_RT_MIN,
    rt_max: float = WCST_RT_MAX,
) -> pd.DataFrame:
    df = df.copy()
    df["rt_ms"] = pd.to_numeric(df["rt_ms"], errors="coerce")
    df = df[df["rt_ms"].notna()]
    df = df[(df["rt_ms"] > rt_min) & (df["rt_ms"] <= rt_max)]
    return df


def compute_wcst_qc_stats(
    trials_df: pd.DataFrame,
    criteria: Optional[WCSTQCCriteria] = None,
) -> pd.DataFrame:
    if criteria is None:
        criteria = WCSTQCCriteria()

    base = ensure_participant_id(trials_df.copy())

    n_trials = base.groupby("participant_id").size().rename("n_trials")
    median_rt = base.groupby("participant_id")["rt_ms"].median().rename("median_rt")

    choice_ratio = (
        base.groupby(["participant_id", "chosenCard"]).size()
        .groupby(level=0)
        .apply(lambda s: (s / s.sum()).max())
        .rename("max_choice_ratio")
    )

    qc = pd.concat([n_trials, median_rt, choice_ratio], axis=1).fillna(0).reset_index()
    qc["qc_passed"] = (
        (qc["n_trials"] >= criteria.min_trials)
        & (qc["median_rt"] >= criteria.min_median_rt)
        & (qc["max_choice_ratio"] <= criteria.max_single_choice_ratio)
    )
    return qc


def get_wcst_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[WCSTQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = WCSTQCCriteria()

    trials_path = data_dir / "4b_wcst_trials.csv"
    if not trials_path.exists():
        if verbose:
            print(f"[WARN] WCST trials file not found: {trials_path}")
        return set()

    trials = pd.read_csv(trials_path, encoding="utf-8-sig")
    trials = ensure_participant_id(trials)
    raw_counts = trials.groupby("participant_id").size()
    cleaned, _ = clean_wcst_trials(trials)
    qc = compute_wcst_qc_stats(cleaned, criteria)

    summary_path = data_dir / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] cognitive summary file not found: {summary_path}")
        summary_valid = set(qc[qc["qc_passed"]]["participant_id"].unique())
        return summary_valid

    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary_df["testName"] = summary_df["testName"].str.lower()

    wcst_df = summary_df[summary_df["testName"] == "wcst"].copy()
    if wcst_df.empty:
        return set()

    if criteria.require_metrics:
        mask = (
            (wcst_df["totalTrialCount"] >= criteria.min_trials)
            & (wcst_df["completedCategories"] >= criteria.min_completed_categories)
            & (wcst_df["perseverativeErrorCount"].notna())
        )
        wcst_df = wcst_df[mask]

    if criteria.max_total_errors > 0 and "totalErrorCount" in wcst_df.columns:
        wcst_df = wcst_df[wcst_df["totalErrorCount"] < criteria.max_total_errors]

    if criteria.max_perseverative_responses > 0 and "perseverativeResponses" in wcst_df.columns:
        wcst_df = wcst_df[wcst_df["perseverativeResponses"] < criteria.max_perseverative_responses]

    summary_valid = set(wcst_df["participantId"].unique())
    mismatch_ids = set()
    if "totalTrialCount" in wcst_df.columns:
        total_counts = pd.to_numeric(wcst_df["totalTrialCount"], errors="coerce")
        for pid, total in zip(wcst_df["participantId"], total_counts):
            if pd.isna(total):
                continue
            raw_count = raw_counts.get(pid)
            if raw_count is not None and int(raw_count) != int(total):
                mismatch_ids.add(pid)
        summary_valid -= mismatch_ids
    trial_valid = set(qc[qc["qc_passed"]]["participant_id"].unique())
    valid_ids = summary_valid & trial_valid

    if verbose:
        all_wcst = set(summary_df[summary_df["testName"] == "wcst"]["participantId"].unique())
        excluded = all_wcst - valid_ids
        if excluded:
            print(f"  [INFO] WCST QC failed: {len(excluded)}")
        if mismatch_ids:
            print(f"  [INFO] WCST trial count mismatch: {len(mismatch_ids)}")

    return valid_ids
