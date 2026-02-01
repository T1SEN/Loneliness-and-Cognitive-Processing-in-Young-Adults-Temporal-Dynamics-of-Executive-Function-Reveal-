"""
Preprocessing CLI for building overall datasets and core feature sets.

Usage:
    python -m static.preprocessing --build overall
    python -m static.preprocessing --build all
    python -m static.preprocessing --features overall
    python -m static.preprocessing --list
    python -m static.preprocessing --info overall
"""

import argparse
import sys

import pandas as pd

from .constants import VALID_TASKS, get_results_dir
from .datasets import build_all_datasets, build_overall_dataset, get_dataset_info, print_dataset_summary
from .features import build_overall_features


if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _report_features(task: str, features: pd.DataFrame, verbose: bool) -> None:
    if features.empty:
        if verbose:
            print(f"[WARN] {task} features: empty")
        return
    if verbose:
        print(f"[INFO] {task} features: {len(features)} rows, {len(features.columns)} cols")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing CLI for building task-specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m static.preprocessing --build overall
    python -m static.preprocessing --build all
    python -m static.preprocessing --features overall
    python -m static.preprocessing --list
    python -m static.preprocessing --info overall
        """,
    )

    parser.add_argument(
        "--build",
        choices=list(VALID_TASKS) + ["all"],
        help="Build overall dataset or all datasets",
    )
    parser.add_argument(
        "--features",
        choices=list(VALID_TASKS) + ["all"],
        help="Compute core feature set for overall or all tasks",
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
        else:
            build_overall_dataset(save=save, verbose=verbose)

    if args.features:
        tasks = list(VALID_TASKS) if args.features == "all" else [args.features]
        for task in tasks:
            features = build_overall_features(
                data_dir=get_results_dir(task),
                save=save,
                verbose=verbose,
            )
            if not save:
                _report_features(task, features, verbose=verbose)


if __name__ == "__main__":
    main()
