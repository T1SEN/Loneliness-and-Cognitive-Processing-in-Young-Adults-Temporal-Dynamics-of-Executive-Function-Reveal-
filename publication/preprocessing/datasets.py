"""
Task-specific dataset builders and master dataset loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .constants import STANDARDIZE_COLS, VALID_TASKS, get_results_dir, RAW_DIR
from .core import ensure_participant_id, normalize_gender_series
from .surveys import load_participants, load_ucla_scores, load_dass_scores, load_survey_items
from .stroop.dataset import build_stroop_dataset
from .wcst.dataset import build_wcst_dataset
from .stroop.features import derive_stroop_features
from .wcst.features import derive_wcst_features
from .stroop.loaders import load_stroop_summary
from .wcst.loaders import load_wcst_summary
from .overall.loaders import load_overall_summary
from .overall.features import derive_overall_features


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
            if task == "overall":
                trial_features = derive_overall_features(data_dir=data_dir)
            elif task == "stroop":
                trial_features = derive_stroop_features(data_dir=data_dir)
            else:
                trial_features = derive_wcst_features(data_dir=data_dir)
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

    if task == "overall":
        summary = load_overall_summary(data_dir)
    elif task == "stroop":
        summary = load_stroop_summary(data_dir)
    else:
        summary = load_wcst_summary(data_dir)

    master = participants.merge(ucla, on="participant_id", how="inner")
    print(f"  Merge participants + UCLA (inner): {len(participants)} -> {len(master)} rows")
    master = _log_merge(master, "master", dass, "dass")
    if not survey_items.empty:
        master = _log_merge(master, "master", survey_items, "survey_items")
    master = _log_merge(master, "master", summary, task)

    master = _merge_trial_features_if_needed(master, log=True)

    if merge_cognitive_summary:
        summary_path = data_dir / "3_cognitive_tests_summary.csv"
        if summary_path.exists() and task != "overall":
            cognitive = pd.read_csv(summary_path, encoding="utf-8")
            cognitive = ensure_participant_id(cognitive)
            if "testName" in cognitive.columns:
                cognitive["testName"] = cognitive["testName"].str.lower()
                cognitive = cognitive[cognitive["testName"] == task]
            cognitive = cognitive.sort_values("participant_id").drop_duplicates(subset=["participant_id"])
            master = _log_merge(master, "master", cognitive, "cognitive_summary")
        elif summary_path.exists() and task == "overall":
            print("  Skipping cognitive summary merge for overall (preprocessing-only default)")

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

    results["stroop"] = build_stroop_dataset(data_dir=data_dir, save=save, verbose=verbose)
    if verbose:
        print()
    results["wcst"] = build_wcst_dataset(data_dir=data_dir, save=save, verbose=verbose)

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
