"""Overall dataset builder (survey + completed tasks)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from ..constants import STROOP_RT_MIN, STROOP_RT_MAX, RAW_DIR, COMPLETE_OVERALL_DIR, WCST_RT_MIN, WCST_RT_MAX
from ..surveys import get_survey_valid_participants, SurveyQCCriteria

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

    return survey_valid & task_valid


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
            rt_col = None
            for candidate in ["rt_ms", "reactionTimeMs", "resp_time_ms"]:
                if candidate in df_filtered.columns:
                    rt_col = candidate
                    break
            if rt_col is None:
                if verbose:
                    print("  [ERROR] WCST trials missing RT column")
                continue

            df_filtered["rt_ms"] = pd.to_numeric(df_filtered[rt_col], errors="coerce")
            df_filtered = df_filtered[df_filtered["rt_ms"].notna()]
            df_filtered = df_filtered[df_filtered["rt_ms"] >= WCST_RT_MIN]
            df_filtered["is_rt_valid"] = df_filtered["rt_ms"].between(WCST_RT_MIN, WCST_RT_MAX)
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
            else:
                df_filtered["timeout"] = False

            if "correct" in df_filtered.columns:
                correct = df_filtered["correct"]
                if correct.dtype != bool:
                    correct = correct.astype(str).str.strip().str.lower().map(
                        {"true": True, "1": True, "false": False, "0": False}
                    )
                    correct = correct.fillna(False)
                df_filtered["correct"] = correct.astype(bool)
            else:
                df_filtered["correct"] = False

            df_filtered["correct"] = df_filtered["correct"] & (~df_filtered["timeout"])
            df_filtered["is_timeout"] = df_filtered["timeout"]
            df_filtered["is_rt_valid"] = df_filtered[rt_col].between(STROOP_RT_MIN, STROOP_RT_MAX)
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
