"""Overall dataset builder (survey + completed tasks)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from ..constants import RAW_DIR, COMPLETE_OVERALL_DIR
from ..surveys import get_survey_valid_participants, SurveyQCCriteria
from .qc import (
    clean_stroop_trials,
    clean_wcst_trials,
    compute_stroop_qc_ids,
    compute_wcst_qc_ids,
    prepare_stroop_trials,
    prepare_wcst_trials,
)

TASK_FILES = [
    "1_participants_info.csv",
    "2_surveys_results.csv",
    "3_cognitive_tests_summary.csv",
    "4b_wcst_trials.csv",
    "4c_stroop_trials.csv",
]


def _ensure_participant_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "participantId" in df.columns:
        return df
    if "participant_id" in df.columns:
        return df.rename(columns={"participant_id": "participantId"})
    return df


def _get_completed_task_participants(data_dir: Path, verbose: bool) -> Set[str]:
    summary_path = data_dir / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] cognitive summary not found: {summary_path}")
        return set()

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary = _ensure_participant_id_column(summary)
    if "participantId" not in summary.columns or "testName" not in summary.columns:
        if verbose:
            print("[WARN] cognitive summary missing participantId or testName")
        return set()

    summary["testName"] = summary["testName"].astype(str).str.lower()
    summary = summary[summary["testName"].isin({"stroop", "wcst"})]
    if summary.empty:
        return set()

    counts = summary.groupby(["participantId", "testName"]).size().unstack(fill_value=0)
    valid_mask = (counts.get("stroop", 0) > 0) & (counts.get("wcst", 0) > 0)
    return set(counts[valid_mask].index.astype(str))


def get_overall_complete_participants(
    data_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    stroop_criteria: Optional[object] = None,
    wcst_criteria: Optional[object] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    # Criteria placeholders kept for backward compatibility in overall-only mode.
    _ = (stroop_criteria, wcst_criteria)

    survey_valid = get_survey_valid_participants(data_dir, survey_criteria, verbose)
    task_valid = _get_completed_task_participants(data_dir, verbose)

    stroop_trials = prepare_stroop_trials(data_dir)
    wcst_trials = prepare_wcst_trials(data_dir)
    stroop_valid = compute_stroop_qc_ids(stroop_trials)
    wcst_valid = compute_wcst_qc_ids(wcst_trials)

    if verbose:
        if stroop_trials.empty:
            print("[WARN] Stroop trials missing; skipping Stroop QC.")
        else:
            print(f"Stroop QC valid: {len(stroop_valid)}")
        if wcst_trials.empty:
            print("[WARN] WCST trials missing; skipping WCST QC.")
        else:
            print(f"WCST QC valid: {len(wcst_valid)}")

    valid_ids = survey_valid & task_valid
    if not stroop_trials.empty:
        valid_ids = valid_ids & stroop_valid
    if not wcst_trials.empty:
        valid_ids = valid_ids & wcst_valid

    return valid_ids


def build_overall_dataset(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    stroop_criteria: Optional[object] = None,
    wcst_criteria: Optional[object] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    if data_dir is None:
        data_dir = RAW_DIR
    if output_dir is None:
        output_dir = COMPLETE_OVERALL_DIR
    # Criteria placeholders kept for backward compatibility in overall-only mode.
    _ = (stroop_criteria, wcst_criteria)

    if verbose:
        print("=" * 60)
        print("Overall dataset build")
        print("=" * 60)

    valid_ids = get_overall_complete_participants(
        data_dir=data_dir,
        survey_criteria=survey_criteria,
        stroop_criteria=stroop_criteria,
        wcst_criteria=wcst_criteria,
        verbose=verbose,
    )

    if not valid_ids:
        if verbose:
            print("[WARN] no valid overall participants")
        return {}

    if verbose:
        print(f"\nValid participants: {len(valid_ids)}")

    if save:
        os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}

    for filename in TASK_FILES:
        input_path = data_dir / filename
        if not input_path.exists():
            if verbose:
                print(f"  [SKIP] {filename} not found")
            continue

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        original_count = len(df)

        df = _ensure_participant_id_column(df)
        if "participantId" not in df.columns:
            if verbose:
                print(f"  [ERROR] {filename} missing participantId")
            continue

        df_filtered = df[df["participantId"].isin(valid_ids)].copy()
        if filename == "3_cognitive_tests_summary.csv" and "testName" in df_filtered.columns:
            df_filtered["testName"] = df_filtered["testName"].str.lower()
            df_filtered = df_filtered[df_filtered["testName"].isin({"wcst", "stroop"})]
        if filename == "4b_wcst_trials.csv":
            df_filtered = clean_wcst_trials(df_filtered)
        if filename == "4c_stroop_trials.csv":
            df_filtered = clean_stroop_trials(df_filtered)
        results[filename] = df_filtered

        if save:
            output_path = output_dir / filename
            df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

        if verbose:
            print(f"  [OK] {filename}: {original_count} -> {len(df_filtered)} rows")

    if verbose:
        print(f"\nDone: '{output_dir}'")

    if save:
        ids_path = output_dir / "filtered_participant_ids.csv"
        pd.DataFrame({"participantId": sorted(valid_ids)}).to_csv(ids_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  [OK] participant ids: {ids_path}")

    return results
