"""
Mechanism Analysis CLI
======================

Command-line interface for the publication mechanism analysis suite.

Usage:
    python -m publication.mechanism_analysis              # Run all analyses
    python -m publication.mechanism_analysis --list       # List available analyses
    python -m publication.mechanism_analysis -a wcst_rl_modeling_dass  # Run specific suite
    python -m publication.mechanism_analysis -a wcst_rl_modeling_dass --sub rescorla_wagner  # Run sub-analysis
    python -m publication.mechanism_analysis --quiet      # Suppress output
"""

import argparse
from . import run, list_analyses


def main():
    parser = argparse.ArgumentParser(
        description="Publication Mechanism Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m publication.mechanism_analysis --list
    python -m publication.mechanism_analysis -a wcst_rl_modeling_dass
    python -m publication.mechanism_analysis -a wcst_rl_modeling_dass --sub rescorla_wagner
    python -m publication.mechanism_analysis -a wcst_hmm_modeling_dass --sub dass_controlled_hmm --quiet
        """
    )

    parser.add_argument(
        '--analysis', '-a',
        type=str,
        default=None,
        help='Suite to run (e.g., wcst_rl_modeling_dass, wcst_rl_path, wcst_hmm_path)'
    )

    parser.add_argument(
        '--sub', '-s',
        type=str,
        default=None,
        help='Sub-analysis within suite (use --list to see options)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available analyses'
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
        run(
            analysis=args.analysis,
            sub_analysis=args.sub,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
