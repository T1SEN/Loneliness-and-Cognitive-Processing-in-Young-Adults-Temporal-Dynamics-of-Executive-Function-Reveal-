"""
Unified Analysis Runner
=======================

Command-line interface for running analysis suites.

Usage:
    # List available suites
    python -m analysis --list

    # Run Gold Standard pipeline
    python -m analysis --suite gold_standard

    # Run specific exploratory analysis
    python -m analysis --suite exploratory.wcst --analysis learning_trajectory

    # Run all suites
    python -m analysis --all

    # Force rebuild cached data
    python -m analysis --suite gold_standard --force-rebuild

Examples:
    python -m analysis --list
    python -m analysis -s gold_standard
    python -m analysis -s exploratory.prp -a bottleneck_shape
    python -m analysis --all --verbose
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
from typing import Optional


def main(args: Optional[list] = None) -> int:
    """Main entry point for analysis CLI."""

    parser = argparse.ArgumentParser(
        prog='python -m analysis',
        description='UCLA × Executive Function Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m analysis --list                    # List all available suites
  python -m analysis -s gold_standard          # Run Gold Standard pipeline
  python -m analysis -s exploratory.wcst       # Run WCST exploratory suite
  python -m analysis -s validation -a cross_validation  # Run specific analysis
  python -m analysis --all                     # Run all suites

Available Suites:
  gold_standard           Publication-ready confirmatory analyses (DASS-controlled)
  exploratory.prp         PRP task exploratory analyses
  exploratory.stroop      Stroop task exploratory analyses
  exploratory.wcst        WCST task exploratory analyses
  exploratory.cross_task  Cross-task integration analyses
  mediation               Mediation pathway analyses (DASS as mediator)
  validation              Methodological validation (CV, robustness, etc.)
  synthesis               Integration and summary analyses
  advanced.mechanistic    Mechanistic decomposition (tier 1/2)
  advanced.latent         Latent variable and SEM analyses
  advanced.clustering     Clustering and vulnerability profiling
        """
    )

    parser.add_argument(
        '--suite', '-s',
        type=str,
        metavar='NAME',
        help='Suite to run (e.g., gold_standard, exploratory.wcst)'
    )

    parser.add_argument(
        '--analysis', '-a',
        type=str,
        metavar='NAME',
        help='Specific analysis within suite'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available suites and analyses'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all suites in recommended order'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of cached data'
    )

    parser.add_argument(
        '--skip-on-error',
        action='store_true',
        default=True,
        help='Continue running other suites if one fails (default: True)'
    )

    parsed = parser.parse_args(args)

    # Determine verbosity
    verbose = not parsed.quiet

    # Import run module (delayed to speed up --help)
    from analysis.run import run_suite, run_all_suites, list_suites

    # Handle --list
    if parsed.list:
        list_suites(show_analyses=True)
        return 0

    # Handle --all
    if parsed.all:
        print("\n" + "=" * 80)
        print("RUNNING ALL ANALYSIS SUITES")
        print("=" * 80)

        results = run_all_suites(
            verbose=verbose,
            force_rebuild=parsed.force_rebuild,
            skip_on_error=parsed.skip_on_error,
        )

        # Summary
        successful = sum(1 for v in results.values() if v is not None)
        failed = sum(1 for v in results.values() if v is None)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        return 0 if failed == 0 else 1

    # Handle --suite
    if parsed.suite:
        try:
            result = run_suite(
                parsed.suite,
                analysis=parsed.analysis,
                verbose=verbose,
                force_rebuild=parsed.force_rebuild,
            )
            return 0
        except Exception as e:
            print(f"Error running {parsed.suite}: {e}")
            return 1

    # No action specified - show help
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
