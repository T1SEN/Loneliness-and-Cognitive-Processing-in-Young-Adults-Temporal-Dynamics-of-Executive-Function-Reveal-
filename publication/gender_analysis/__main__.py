"""
Gender Analysis CLI
===================

Command-line interface for the publication gender analysis suite.

Usage:
    python -m publication.gender_analysis              # Run all analyses
    python -m publication.gender_analysis --list       # List available analyses
    python -m publication.gender_analysis -a male_vulnerability  # Run specific
    python -m publication.gender_analysis --quiet      # Suppress output
"""

import argparse
from . import run, list_analyses


def main():
    parser = argparse.ArgumentParser(
        description="Publication Gender Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m publication.gender_analysis --list
    python -m publication.gender_analysis --all
    python -m publication.gender_analysis -a male_vulnerability
    python -m publication.gender_analysis -a ddm_gender --quiet
        """
    )

    parser.add_argument(
        '--analysis', '-a',
        type=str,
        default=None,
        help='Specific analysis to run (use --list to see options)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available analyses'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all analyses (default if no analysis specified)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)


if __name__ == "__main__":
    main()
