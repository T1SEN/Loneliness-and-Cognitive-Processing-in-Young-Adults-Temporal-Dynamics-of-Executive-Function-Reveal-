"""
Preprocessing CLI for building task-specific datasets.

Usage:
    python -m publication.preprocessing --build stroop
    python -m publication.preprocessing --build all
    python -m publication.preprocessing --list
    python -m publication.preprocessing --info stroop
"""

import argparse
import sys

from .constants import VALID_TASKS
from .datasets import build_all_datasets, get_dataset_info, print_dataset_summary
from .prp.dataset import build_prp_dataset
from .stroop.dataset import build_stroop_dataset
from .wcst.dataset import build_wcst_dataset

# Windows Unicode 출력 지원
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing CLI for building task-specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m publication.preprocessing --build stroop
    python -m publication.preprocessing --build all
    python -m publication.preprocessing --list
    python -m publication.preprocessing --info stroop
        """
    )

    parser.add_argument(
        '--build',
        choices=list(VALID_TASKS) + ['all'],
        help='Build task-specific dataset (stroop, prp, wcst) or all datasets',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and their status',
    )
    parser.add_argument(
        '--info',
        choices=list(VALID_TASKS),
        help='Show detailed info for a specific dataset',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Build dataset without saving to disk',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output',
    )

    args = parser.parse_args()

    # Default action: list datasets
    if not any([args.build, args.list, args.info]):
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

    if args.build:
        save = not args.no_save
        verbose = not args.quiet

        if args.build == 'all':
            build_all_datasets(save=save, verbose=verbose)
        elif args.build == 'prp':
            build_prp_dataset(save=save, verbose=verbose)
        elif args.build == 'stroop':
            build_stroop_dataset(save=save, verbose=verbose)
        elif args.build == 'wcst':
            build_wcst_dataset(save=save, verbose=verbose)


if __name__ == '__main__':
    main()
