"""
Cross-Task Analysis Suite
=========================

Unified cross-task analyses for UCLA × Executive Function research.

Consolidates:
- cross_task_consistency.py
- cross_task_integration.py
- cross_task_meta_control.py
- cross_task_order_effects.py
- task_order_effects.py
- age_gam_developmental_windows.py
- age_gender_ucla_threeway.py
- dass_anxiety_mask_hypothesis.py
- dose_response_threshold_analysis.py
- extreme_group_analysis.py
- hidden_patterns_analysis.py
- nonlinear_gender_effects.py
- nonlinear_threshold_analysis.py
- residual_ucla_analysis.py
- temporal_context_effects.py
- ucla_nonlinear_effects.py

Usage:
    python -m analysis.exploratory.cross_task                    # Run all
    python -m analysis.exploratory.cross_task --analysis consistency
    python -m analysis.exploratory.cross_task --list

    from analysis.exploratory import cross_task
    cross_task.run('consistency')
    cross_task.run()  # All analyses

NOTE: These are EXPLORATORY analyses. For confirmatory results, use:
    analysis/gold_standard/pipeline.py

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
from typing import Dict, Optional
import pandas as pd

from analysis.exploratory.cross_task._common import OUTPUT_DIR, AnalysisSpec

# Import analysis registries from submodules
from analysis.exploratory.cross_task.consistency import ANALYSES as CONSISTENCY_ANALYSES
from analysis.exploratory.cross_task.age_gender import ANALYSES as AGE_GENDER_ANALYSES
from analysis.exploratory.cross_task.nonlinear import ANALYSES as NONLINEAR_ANALYSES
from analysis.exploratory.cross_task.residual_temporal import ANALYSES as RESIDUAL_TEMPORAL_ANALYSES

# Combined registry
ANALYSES: Dict[str, AnalysisSpec] = {}
ANALYSES.update(CONSISTENCY_ANALYSES)
ANALYSES.update(AGE_GENDER_ANALYSES)
ANALYSES.update(NONLINEAR_ANALYSES)
ANALYSES.update(RESIDUAL_TEMPORAL_ANALYSES)


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run cross-task analyses.

    Parameters
    ----------
    analysis : str, optional
        Specific analysis to run. If None, runs all analyses.
    verbose : bool, default True
        Print progress output.

    Returns
    -------
    dict
        Dictionary mapping analysis names to result DataFrames.
    """
    if verbose:
        print("=" * 70)
        print("CROSS-TASK ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Running: {spec.name}")
            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-TASK SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Cross-Task Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")
        print(f"    Source: {spec.source_script}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Task Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                        help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
