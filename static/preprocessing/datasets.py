"""
Overall dataset builder and master dataset loader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from .constants import (
    STANDARDIZE_COLS,
    VALID_TASKS,
    get_results_dir,
    RAW_DIR,
    COMPLETE_OVERALL_DIR,
)
from .core import normalize_gender_series
from .surveys import (
    SurveyQCCriteria,
    get_survey_valid_participants,
    load_participants,
    load_ucla_scores,
    load_dass_scores,
    load_survey_items,
)
from .stroop.qc import clean_stroop_trials, compute_stroop_qc_ids, prepare_stroop_trials
from .wcst.qc import clean_wcst_trials, compute_wcst_qc_ids, prepare_wcst_trials
from .summary import load_overall_summary
from .features import derive_overall_features


TASK_FILES = [
    "1_participants_info.csv",
    "2_surveys_results.csv",
    "3_cognitive_tests_summary.csv",
    "4b_wcst_trials.csv",
    "4a_stroop_trials.csv",
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
        if filename == "4a_stroop_trials.csv":
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


def load_master_dataset(
    task: str,
    add_standardized: bool = True,
    merge_cognitive_summary: bool = False,
    merge_trial_features: bool = True,
) -> pd.DataFrame:
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")

    data_dir = get_results_dir(task)
    def _merge_trial_features_if_needed(master_df: pd.DataFrame, log: bool = False) -> pd.DataFrame:
        if not merge_trial_features:
            return master_df
        try:
            trial_features = derive_overall_features(data_dir=data_dir)
        except Exception as exc:  # pragma: no cover
            print(f"  Warning: Failed to load trial features ({exc})")
            return master_df

        if trial_features.empty:
            return master_df

        trial_cols = [c for c in trial_features.columns if c != "participant_id"]
        overlap = [c for c in trial_cols if c in master_df.columns]
        if overlap:
            master_df = master_df.drop(columns=overlap)

        before = len(master_df)
        merged = master_df.merge(trial_features, on="participant_id", how="left")
        if log:
            print(f"  Merge master + trial_features: {before} -> {len(merged)} rows (left join)")
        return merged

    participants = load_participants(data_dir)
    participants["gender_normalized"] = normalize_gender_series(participants["gender"])
    participants["gender_male"] = participants["gender_normalized"].map({"male": 1, "female": 0})

    def _log_merge(df: pd.DataFrame, df_name: str, merge_df: pd.DataFrame, merge_name: str, key: str = "participant_id") -> pd.DataFrame:
        before = len(df)
        merged = df.merge(merge_df, on=key, how="left")
        after = len(merged)
        print(f"  Merge {df_name} + {merge_name}: {before} -> {after} rows (left join on '{key}')")
        return merged

    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    if "ucla_total" not in ucla.columns and "ucla_score" in ucla.columns:
        ucla["ucla_total"] = ucla["ucla_score"]

    dass = load_dass_scores(data_dir)
    survey_items = load_survey_items(data_dir)

    summary = load_overall_summary(data_dir)

    master = participants.merge(ucla, on="participant_id", how="inner")
    print(f"  Merge participants + UCLA (inner): {len(participants)} -> {len(master)} rows")
    master = _log_merge(master, "master", dass, "dass")
    if not survey_items.empty:
        master = _log_merge(master, "master", survey_items, "survey_items")
    master = _log_merge(master, "master", summary, task)

    master = _merge_trial_features_if_needed(master, log=True)

    if merge_cognitive_summary:
        print("  Skipping cognitive summary merge (overall already includes task summary).")

    if add_standardized:
        for col in STANDARDIZE_COLS:
            if col in master.columns:
                std_val = master[col].std()
                master[f"z_{col}"] = (master[col] - master[col].mean()) / std_val if std_val else 0
        if "z_ucla_score" in master.columns and "z_ucla" not in master.columns:
            master["z_ucla"] = master["z_ucla_score"]

    print(
        "  Master dataset built: N={}, male={}, female={}, unknown={}".format(
            len(master),
            int(master["gender_male"].fillna(0).sum()),
            int((master["gender_male"] == 0).sum()),
            int(master["gender_male"].isna().sum()),
        )
    )

    return master


def build_all_datasets(
    data_dir: Optional[Path] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    results = {}

    results["overall"] = build_overall_dataset(data_dir=data_dir, save=save, verbose=verbose)

    return results


def get_dataset_info(task: Optional[str] = None) -> Dict:
    if task is not None and task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")

    tasks = [task] if task else sorted(VALID_TASKS)
    info = {}

    for t in tasks:
        task_dir = get_results_dir(t)
        participants_path = task_dir / "1_participants_info.csv"

        task_info = {
            "path": str(task_dir),
            "exists": task_dir.exists(),
            "n_participants": 0,
            "files": [],
        }

        if task_dir.exists():
            task_info["files"] = [f.name for f in task_dir.glob("*.csv")]
            if participants_path.exists():
                df = pd.read_csv(participants_path, encoding="utf-8-sig")
                task_info["n_participants"] = len(df)

        info[t] = task_info

    return info


def print_dataset_summary() -> None:
    print("=" * 60)
    print("Dataset summary")
    print("=" * 60)

    info = get_dataset_info()

    for task, task_info in info.items():
        status = "OK" if task_info["exists"] else "NO"
        n = task_info["n_participants"]
        print(f"  [{status}] {task.upper():8} - N={n:3} ({task_info['path']})")

    raw_participants_path = RAW_DIR / "1_participants_info.csv"
    if raw_participants_path.exists():
        raw_df = pd.read_csv(raw_participants_path, encoding="utf-8-sig")
        print(f"\n  [Raw] N={len(raw_df)} ({RAW_DIR})")
