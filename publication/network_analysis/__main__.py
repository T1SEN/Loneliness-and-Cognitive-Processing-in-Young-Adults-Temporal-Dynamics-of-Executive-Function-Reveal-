"""
CLI entry point for `python -m publication.network_analysis`.
"""

from __future__ import annotations

import argparse
import sys

from . import AVAILABLE_ANALYSES, list_analyses, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication Network Analysis CLI")
    parser.add_argument("--analysis", "-a", type=str, default="domain_gender", help="Analysis key.")
    parser.add_argument("--variable-set", "-v", type=str, default="domain_level", help="Variable set name.")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations per subset.")
    parser.add_argument("--bootstrap-frac", type=float, default=0.8, help="Bootstrap sample fraction.")
    parser.add_argument("--permutations", type=int, default=500, help="Gender permutation iterations.")
    parser.add_argument("--alpha", type=float, default=None, help="Manual GraphicalLasso alpha (skip CV).")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds when enabled.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of master dataset.")
    parser.add_argument(
        "--correlation",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation metric before GraphicalLasso (pearson or spearman).",
    )
    parser.add_argument("--list", action="store_true", help="List available analyses and exit.")

    args = parser.parse_args()

    if args.list:
        list_analyses()
        sys.exit(0)

    if args.analysis not in AVAILABLE_ANALYSES:
        raise SystemExit(f"Unknown analysis '{args.analysis}'. Use --list to inspect options.")

    run(
        analysis=args.analysis,
        variable_set=args.variable_set,
        bootstrap_iter=args.bootstrap,
        bootstrap_fraction=args.bootstrap_frac,
        gender_permutations=args.permutations,
        alpha=args.alpha,
        use_cv=not args.no_cv and args.alpha is None,
        cv_folds=args.cv_folds,
        force_rebuild=args.force_rebuild,
        verbose=True,
        correlation=args.correlation,
    )


if __name__ == "__main__":
    main()
