"""
Preprocessing CLI for building task-specific datasets and feature sets.

Usage:
    python -m publication.preprocessing --build stroop
    python -m publication.preprocessing --build all
    python -m publication.preprocessing --features stroop --feature-set traditional
    python -m publication.preprocessing --features all --feature-set dispersion
    python -m publication.preprocessing --list
    python -m publication.preprocessing --info stroop
"""

import argparse
import sys

import pandas as pd

from .constants import VALID_TASKS, get_results_dir
from .datasets import build_all_datasets, get_dataset_info, print_dataset_summary
from .overall import (
    derive_overall_features,
    derive_overall_traditional_features,
    derive_overall_dispersion_features,
    derive_overall_drift_features,
    derive_overall_recovery_features,
    derive_overall_mechanism_features,
)
from .prp import (
    derive_prp_features,
    derive_prp_traditional_features,
    derive_prp_dispersion_features,
    derive_prp_drift_features,
    derive_prp_recovery_features,
)
from .prp.mechanism import (
    load_or_compute_prp_mechanism_features,
    load_or_compute_prp_hmm_event_features,
    load_or_compute_prp_bottleneck_mechanism_features,
)
from .prp.trial_level_dataset import build_prp_dataset
from .stroop import (
    derive_stroop_features,
    derive_stroop_traditional_features,
    derive_stroop_dispersion_features,
    derive_stroop_drift_features,
    derive_stroop_recovery_features,
)
from .stroop.mechanism import (
    load_or_compute_stroop_mechanism_features,
    load_or_compute_stroop_hmm_event_features,
    load_or_compute_stroop_lba_mechanism_features,
)
from .stroop.dataset import build_stroop_dataset
from .wcst import (
    derive_wcst_features,
    derive_wcst_traditional_features,
    derive_wcst_dispersion_features,
    derive_wcst_drift_features,
    derive_wcst_recovery_features,
)
from .wcst.mechanism import (
    load_or_compute_wcst_hmm_mechanism_features,
    load_or_compute_wcst_rl_mechanism_features,
    load_or_compute_wcst_wsls_mechanism_features,
    load_or_compute_wcst_bayesianrl_mechanism_features,
)
from .wcst.dataset import build_wcst_dataset


if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


FEATURE_SETS = ("all", "traditional", "dispersion", "drift", "recovery", "mechanism")


def _merge_feature_frames(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra.empty:
        return base
    if base.empty:
        return extra
    overlap = [c for c in extra.columns if c != "participant_id" and c in base.columns]
    if overlap:
        base = base.drop(columns=overlap)
    return base.merge(extra, on="participant_id", how="left")


def _derive_prp_mechanism_features(data_dir):
    features_df = pd.DataFrame()
    for part in (
        load_or_compute_prp_mechanism_features,
        load_or_compute_prp_hmm_event_features,
        load_or_compute_prp_bottleneck_mechanism_features,
    ):
        df = part(data_dir=data_dir)
        features_df = _merge_feature_frames(features_df, df)
    return features_df


def _derive_stroop_mechanism_features(data_dir):
    features_df = pd.DataFrame()
    for part in (
        load_or_compute_stroop_mechanism_features,
        load_or_compute_stroop_hmm_event_features,
        load_or_compute_stroop_lba_mechanism_features,
    ):
        df = part(data_dir=data_dir)
        features_df = _merge_feature_frames(features_df, df)
    return features_df


def _derive_wcst_mechanism_features(data_dir):
    features_df = pd.DataFrame()
    for part in (
        load_or_compute_wcst_hmm_mechanism_features,
        load_or_compute_wcst_rl_mechanism_features,
        load_or_compute_wcst_wsls_mechanism_features,
        load_or_compute_wcst_bayesianrl_mechanism_features,
    ):
        df = part(data_dir=data_dir)
        features_df = _merge_feature_frames(features_df, df)
    return features_df


def _feature_filename(task: str, feature_set: str) -> str:
    if feature_set == "all":
        return f"5_{task}_features.csv"
    if feature_set == "traditional":
        return f"5_{task}_traditional_features.csv"
    if feature_set == "dispersion":
        return f"5_{task}_dynamic_dispersion_features.csv"
    if feature_set == "drift":
        return f"5_{task}_dynamic_drift_features.csv"
    if feature_set == "recovery":
        return f"5_{task}_dynamic_recovery_features.csv"
    if feature_set == "mechanism":
        return f"5_{task}_mechanism_merged_features.csv"
    raise ValueError(f"Unknown feature set: {feature_set}")


def _build_feature_set(task: str, feature_set: str, save: bool, verbose: bool) -> None:
    data_dir = get_results_dir(task)

    if task == "overall":
        builders = {
            "all": derive_overall_features,
            "traditional": derive_overall_traditional_features,
            "dispersion": derive_overall_dispersion_features,
            "drift": derive_overall_drift_features,
            "recovery": derive_overall_recovery_features,
            "mechanism": derive_overall_mechanism_features,
        }
    elif task == "prp":
        builders = {
            "all": derive_prp_features,
            "traditional": derive_prp_traditional_features,
            "dispersion": derive_prp_dispersion_features,
            "drift": derive_prp_drift_features,
            "recovery": derive_prp_recovery_features,
            "mechanism": _derive_prp_mechanism_features,
        }
    elif task == "stroop":
        builders = {
            "all": derive_stroop_features,
            "traditional": derive_stroop_traditional_features,
            "dispersion": derive_stroop_dispersion_features,
            "drift": derive_stroop_drift_features,
            "recovery": derive_stroop_recovery_features,
            "mechanism": _derive_stroop_mechanism_features,
        }
    else:
        builders = {
            "all": derive_wcst_features,
            "traditional": derive_wcst_traditional_features,
            "dispersion": derive_wcst_dispersion_features,
            "drift": derive_wcst_drift_features,
            "recovery": derive_wcst_recovery_features,
            "mechanism": _derive_wcst_mechanism_features,
        }

    builder = builders[feature_set]
    features = builder(data_dir=data_dir)
    if features.empty:
        if verbose:
            print(f"[WARN] {task} {feature_set} features: empty")
        return

    output_path = data_dir / _feature_filename(task, feature_set)
    if save:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] {task} {feature_set} features saved: {output_path}")
    else:
        print(f"[INFO] {task} {feature_set} features: {len(features)} rows, {len(features.columns)} cols")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing CLI for building task-specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m publication.preprocessing --build stroop
    python -m publication.preprocessing --build all
    python -m publication.preprocessing --features stroop --feature-set traditional
    python -m publication.preprocessing --features all --feature-set dispersion
    python -m publication.preprocessing --list
    python -m publication.preprocessing --info stroop
        """,
    )

    parser.add_argument(
        "--build",
        choices=list(VALID_TASKS) + ["all"],
        help="Build task-specific dataset (stroop, prp, wcst, overall) or all datasets",
    )
    parser.add_argument(
        "--features",
        choices=list(VALID_TASKS) + ["all"],
        help="Compute feature sets for a task (stroop, prp, wcst, overall) or all tasks",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="all",
        help="Feature set to compute (all, traditional, dispersion, drift, recovery, mechanism)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and their status",
    )
    parser.add_argument(
        "--info",
        choices=list(VALID_TASKS),
        help="Show detailed info for a specific dataset",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Build datasets or features without saving to disk",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if not any([args.build, args.features, args.list, args.info]):
        print_dataset_summary()
        return

    if args.list:
        print_dataset_summary()
        return

    if args.info:
        info = get_dataset_info(args.info)
        task_info = info[args.info]
        print(f"\n{args.info.upper()} Dataset Info:")
        print(f"  Path: {task_info['path']}")
        print(f"  Exists: {task_info['exists']}")
        print(f"  N participants: {task_info['n_participants']}")
        print(f"  Files: {', '.join(task_info['files']) if task_info['files'] else 'None'}")
        return

    if args.features:
        save = not args.no_save
        verbose = not args.quiet
        tasks = list(VALID_TASKS) if args.features == "all" else [args.features]
        for task in tasks:
            _build_feature_set(task, args.feature_set, save=save, verbose=verbose)
        return

    if args.build:
        save = not args.no_save
        verbose = not args.quiet

        if args.build == "all":
            build_all_datasets(save=save, verbose=verbose)
        elif args.build == "prp":
            build_prp_dataset(save=save, verbose=verbose)
        elif args.build == "stroop":
            build_stroop_dataset(save=save, verbose=verbose)
        elif args.build == "wcst":
            build_wcst_dataset(save=save, verbose=verbose)
        elif args.build == "overall":
            from .overall import build_overall_dataset

            build_overall_dataset(save=save, verbose=verbose)


if __name__ == "__main__":
    main()
