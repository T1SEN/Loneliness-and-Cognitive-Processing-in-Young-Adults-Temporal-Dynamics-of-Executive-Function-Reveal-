"""Stroop dataset builder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from ..constants import DEFAULT_RT_MIN, RAW_DIR, get_results_dir
from ..surveys import get_survey_valid_participants, SurveyQCCriteria
from .filters import get_stroop_valid_participants, StroopQCCriteria

TASK_FILES = [
    "1_participants_info.csv",
    "2_surveys_results.csv",
    "3_cognitive_tests_summary.csv",
    "4c_stroop_trials.csv",
]


def get_stroop_complete_participants(
    data_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    task_criteria: Optional[StroopQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR

    survey_valid = get_survey_valid_participants(data_dir, survey_criteria, verbose)
    stroop_valid = get_stroop_valid_participants(data_dir, task_criteria, verbose)
    return survey_valid & stroop_valid


def build_stroop_dataset(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    task_criteria: Optional[StroopQCCriteria] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    if data_dir is None:
        data_dir = RAW_DIR
    if output_dir is None:
        output_dir = get_results_dir("stroop")

    if verbose:
        print("=" * 60)
        print("Stroop dataset build")
        print("=" * 60)

    valid_ids = get_stroop_complete_participants(
        data_dir=data_dir,
        survey_criteria=survey_criteria,
        task_criteria=task_criteria,
        verbose=verbose,
    )

    if not valid_ids:
        if verbose:
            print("[WARN] no valid Stroop participants")
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
            df_filtered = df_filtered[df_filtered["testName"] == "stroop"]
        if filename == "4c_stroop_trials.csv":
            rt_col = "rt_ms" if "rt_ms" in df_filtered.columns else "rt" if "rt" in df_filtered.columns else None
            if rt_col is None:
                if verbose:
                    print("  [ERROR] Stroop trials missing RT column")
                continue
            df_filtered[rt_col] = pd.to_numeric(df_filtered[rt_col], errors="coerce")
            if "timeout" in df_filtered.columns:
                timeout = df_filtered["timeout"]
                if timeout.dtype != bool:
                    timeout = timeout.astype(str).str.strip().str.lower().map(
                        {"true": True, "1": True, "false": False, "0": False}
                    )
                    timeout = timeout.fillna(False)
                df_filtered["timeout"] = timeout.astype(bool)
                df_filtered = df_filtered[df_filtered["timeout"] == False]
            df_filtered = df_filtered[df_filtered[rt_col].notna()]
            df_filtered = df_filtered[df_filtered[rt_col] >= DEFAULT_RT_MIN]

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
