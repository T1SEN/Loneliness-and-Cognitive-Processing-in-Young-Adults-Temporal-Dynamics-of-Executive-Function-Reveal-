"""Overall dataset builder (survey + date + all tasks)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from ..constants import RAW_DIR, COMPLETE_OVERALL_DIR
from ..surveys import get_survey_valid_participants, SurveyQCCriteria
from ..prp.filters import get_prp_valid_participants, PRPQCCriteria
from ..stroop.filters import get_stroop_valid_participants, StroopQCCriteria
from ..wcst.filters import get_wcst_valid_participants, WCSTQCCriteria, clean_wcst_trials

TASK_FILES = [
    "1_participants_info.csv",
    "2_surveys_results.csv",
    "3_cognitive_tests_summary.csv",
    "4a_prp_trials.csv",
    "4b_wcst_trials.csv",
    "4c_stroop_trials.csv",
]


def get_overall_complete_participants(
    data_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    prp_criteria: Optional[PRPQCCriteria] = None,
    stroop_criteria: Optional[StroopQCCriteria] = None,
    wcst_criteria: Optional[WCSTQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR

    survey_valid = get_survey_valid_participants(data_dir, survey_criteria, verbose)
    prp_valid = get_prp_valid_participants(data_dir, prp_criteria, verbose)
    stroop_valid = get_stroop_valid_participants(data_dir, stroop_criteria, verbose)
    wcst_valid = get_wcst_valid_participants(data_dir, wcst_criteria, verbose)

    return survey_valid & prp_valid & stroop_valid & wcst_valid


def build_overall_dataset(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    prp_criteria: Optional[PRPQCCriteria] = None,
    stroop_criteria: Optional[StroopQCCriteria] = None,
    wcst_criteria: Optional[WCSTQCCriteria] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    if data_dir is None:
        data_dir = RAW_DIR
    if output_dir is None:
        output_dir = COMPLETE_OVERALL_DIR

    if verbose:
        print("=" * 60)
        print("Overall dataset build")
        print("=" * 60)

    valid_ids = get_overall_complete_participants(
        data_dir=data_dir,
        survey_criteria=survey_criteria,
        prp_criteria=prp_criteria,
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

        if "participantId" not in df.columns:
            if verbose:
                print(f"  [ERROR] {filename} missing participantId")
            continue

        df_filtered = df[df["participantId"].isin(valid_ids)].copy()
        if filename == "3_cognitive_tests_summary.csv" and "testName" in df_filtered.columns:
            df_filtered["testName"] = df_filtered["testName"].str.lower()
            df_filtered = df_filtered[df_filtered["testName"].isin({"prp", "wcst", "stroop"})]
        if filename == "4b_wcst_trials.csv":
            df_filtered, _ = clean_wcst_trials(df_filtered)

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
