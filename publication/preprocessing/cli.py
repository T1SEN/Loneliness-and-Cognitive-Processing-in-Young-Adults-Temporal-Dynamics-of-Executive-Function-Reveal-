"""
Preprocessing CLI for building task-specific datasets and core feature sets.

Usage:
    python -m publication.preprocessing --build stroop
    python -m publication.preprocessing --build all
    python -m publication.preprocessing --features stroop
    python -m publication.preprocessing --list
    python -m publication.preprocessing --info stroop
"""

import argparse
import sys

from .constants import VALID_TASKS, get_results_dir
from .datasets import build_all_datasets, get_dataset_info, print_dataset_summary
from .overall import derive_overall_features
from .stroop import derive_stroop_features
from .wcst import derive_wcst_features
from .stroop.dataset import build_stroop_dataset
from .wcst.dataset import build_wcst_dataset


if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _save_features(task: str, features, save: bool, verbose: bool) -> None:
    if features.empty:
        if verbose:
            print(f"[WARN] {task} features: empty")
        return
    output_dir = get_results_dir(task)
    output_path = output_dir / f"5_{task}_features.csv"
    if save:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] {task} features saved: {output_path}")
    else:
        print(f"[INFO] {task} features: {len(features)} rows, {len(features.columns)} cols")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing CLI for building task-specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m publication.preprocessing --build stroop
    python -m publication.preprocessing --build all
    python -m publication.preprocessing --features stroop
    python -m publication.preprocessing --list
    python -m publication.preprocessing --info stroop
        """,
    )

    parser.add_argument(
        "--build",
        choices=list(VALID_TASKS) + ["all"],
        help="Build task-specific dataset (stroop, wcst, overall) or all datasets",
    )
    parser.add_argument(
        "--features",
        choices=list(VALID_TASKS) + ["all"],
        help="Compute core feature set for a task (stroop, wcst, overall) or all tasks",
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
        print(info.get(args.info, {}))
        return

    save = not args.no_save
    verbose = not args.quiet

    if args.build:
        if args.build == "all":
            build_all_datasets(save=save, verbose=verbose)
        elif args.build == "stroop":
            build_stroop_dataset(save=save, verbose=verbose)
        elif args.build == "wcst":
            build_wcst_dataset(save=save, verbose=verbose)
        else:
            print("[WARN] overall dataset build is handled by upstream scripts")

    if args.features:
        tasks = list(VALID_TASKS) if args.features == "all" else [args.features]
        for task in tasks:
            if task == "stroop":
                features = derive_stroop_features(data_dir=get_results_dir(task))
            elif task == "wcst":
                features = derive_wcst_features(data_dir=get_results_dir(task))
            else:
                features = derive_overall_features(data_dir=get_results_dir(task))
            _save_features(task, features, save=save, verbose=verbose)


if __name__ == "__main__":
    main()
