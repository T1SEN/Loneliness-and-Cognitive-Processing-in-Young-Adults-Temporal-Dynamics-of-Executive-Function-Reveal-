"""Public-only dataset loader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from .constants import PUBLIC_DIR, STANDARDIZE_COLS, VALID_TASKS, get_public_file, get_results_dir
from .core import ensure_participant_id, normalize_gender_series
from .features import derive_overall_features
from .public_validate import get_common_public_ids, validate_public_bundle
from .summary import load_overall_summary
from .surveys import load_dass_scores, load_participants, load_survey_items, load_ucla_scores


PUBLIC_FILES = [
    "demographics_public.csv",
    "surveys_public.csv",
    "features_public.csv",
    "stroop_trials_public.csv",
    "wcst_trials_public.csv",
]


def get_overall_complete_participants(
    data_dir: Optional[Path] = None,
    survey_criteria: Optional[object] = None,
    stroop_criteria: Optional[object] = None,
    wcst_criteria: Optional[object] = None,
    verbose: bool = False,
) -> Set[str]:
    _ = (data_dir, survey_criteria, stroop_criteria, wcst_criteria)
    validate_public_bundle(raise_on_error=True)
    ids = get_common_public_ids(validate=False)
    if verbose:
        print(f"Public common sample: N={len(ids)}")
    return ids


def build_overall_dataset(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    survey_criteria: Optional[object] = None,
    stroop_criteria: Optional[object] = None,
    wcst_criteria: Optional[object] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    _ = (data_dir, output_dir, survey_criteria, stroop_criteria, wcst_criteria, save)
    validate_public_bundle(raise_on_error=True)

    ids = get_common_public_ids(validate=False)
    results: Dict[str, pd.DataFrame] = {}

    for filename in PUBLIC_FILES:
        path = PUBLIC_DIR / filename
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = ensure_participant_id(df)
        if "participant_id" in df.columns:
            df = df[df["participant_id"].astype(str).isin(ids)].copy()
        results[filename] = df

    if verbose:
        print("=" * 60)
        print("Public-only dataset contract")
        print("=" * 60)
        print(f"Path: {PUBLIC_DIR}")
        print(f"Common public_id count: {len(ids)}")
        for filename, df in results.items():
            print(f"  [OK] {filename}: {len(df)} rows")

    return results


def load_master_dataset(
    task: str,
    add_standardized: bool = True,
    merge_cognitive_summary: bool = False,
    merge_trial_features: bool = True,
) -> pd.DataFrame:
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")

    validate_public_bundle(raise_on_error=True)
    data_dir = get_results_dir(task)

    participants = load_participants(data_dir)
    participants["gender_normalized"] = normalize_gender_series(participants["gender"])
    participants["gender_male"] = participants["gender_normalized"].map({"male": 1, "female": 0})

    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    if "ucla_total" not in ucla.columns and "ucla_score" in ucla.columns:
        ucla["ucla_total"] = ucla["ucla_score"]

    dass = load_dass_scores(data_dir)
    survey_items = load_survey_items(data_dir)
    summary = load_overall_summary(data_dir)

    master = participants.merge(ucla, on="participant_id", how="inner")
    master = master.merge(dass, on="participant_id", how="left")
    if not survey_items.empty:
        master = master.merge(survey_items, on="participant_id", how="left")
    master = master.merge(summary, on="participant_id", how="left")

    if merge_trial_features:
        trial_features = derive_overall_features(data_dir=data_dir)
        if not trial_features.empty:
            trial_cols = [c for c in trial_features.columns if c != "participant_id"]
            overlap = [c for c in trial_cols if c in master.columns]
            if overlap:
                master = master.drop(columns=overlap)
            master = master.merge(trial_features, on="participant_id", how="left")

    if merge_cognitive_summary:
        print("  [INFO] merge_cognitive_summary ignored in public-only mode.")

    if add_standardized:
        for col in STANDARDIZE_COLS:
            if col in master.columns:
                std_val = master[col].std()
                master[f"z_{col}"] = (master[col] - master[col].mean()) / std_val if std_val else 0
        if "z_ucla_score" in master.columns and "z_ucla" not in master.columns:
            master["z_ucla"] = master["z_ucla_score"]

    return master


def build_all_datasets(
    data_dir: Optional[Path] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    _ = data_dir
    return {"overall": build_overall_dataset(save=save, verbose=verbose)}


def get_dataset_info(task: Optional[str] = None) -> Dict:
    if task is not None and task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")

    tasks = [task] if task else sorted(VALID_TASKS)
    info = {}

    for t in tasks:
        task_dir = get_results_dir(t)
        n_participants = 0
        demo_path = get_public_file("demographics")
        if demo_path.exists():
            demo = pd.read_csv(demo_path, encoding="utf-8-sig")
            if "public_id" in demo.columns:
                n_participants = int(demo["public_id"].dropna().astype(str).nunique())

        info[t] = {
            "path": str(task_dir),
            "exists": task_dir.exists(),
            "n_participants": n_participants,
            "files": [f.name for f in task_dir.glob("*.csv")],
        }

    return info


def print_dataset_summary() -> None:
    print("=" * 60)
    print("Dataset summary (public-only)")
    print("=" * 60)

    info = get_dataset_info()
    for task, task_info in info.items():
        status = "OK" if task_info["exists"] else "NO"
        n = task_info["n_participants"]
        print(f"  [{status}] {task.upper():8} - N={n:3} ({task_info['path']})")
