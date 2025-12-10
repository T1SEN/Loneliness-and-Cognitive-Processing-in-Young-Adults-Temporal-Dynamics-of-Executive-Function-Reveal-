"""
Publication Gender Analysis Suite
=================================

Comprehensive gender-specific analyses for UCLA-EF relationships.

This package consolidates all gender-related analyses for publication,
including male vulnerability patterns, gender-stratified analyses,
and UCLA x Gender interactions.

Usage:
    python -m publication.gender_analysis --list
    python -m publication.gender_analysis --all
    python -m publication.gender_analysis -a male_vulnerability

Key Findings:
- WCST PE: UCLA x Gender interaction (p=0.025) - Males show significant UCLA effect
- DDM Drift: UCLA effect significant in FEMALES (p=0.021), not males
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Dict, Optional, Callable, List

# Constants
from ._constants import (
    MIN_SAMPLE_STRATIFIED,
    MIN_SAMPLE_INTERACTION,
    EF_OUTCOMES,
    PRIMARY_OUTCOMES,
    DDM_PARAMS,
    NETWORK_VARS,
    DASS_COVARIATES,
    STRATIFIED_FORMULA,
    INTERACTION_FORMULA,
)

# Utilities
from ._utils import (
    RESULTS_DIR,
    OUTPUT_DIR,
    load_gender_data,
    run_gender_stratified_regression,
    test_gender_interaction,
    run_all_gender_interactions,
    cohens_d,
    partial_eta_squared,
    compute_gender_effect_sizes,
    fisher_z_test,
    compare_correlations_by_gender,
)

# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

ANALYSES: Dict[str, Callable] = {}


def _lazy_load_analyses() -> Dict[str, Callable]:
    """Lazily load analysis functions to avoid circular imports."""
    global ANALYSES

    if ANALYSES:
        return ANALYSES

    # Vulnerability analyses
    from .vulnerability.male_vulnerability import run as run_male_vulnerability
    from .vulnerability.double_dissociation import run as run_double_dissociation

    # Stratified analyses
    from .stratified.ddm_gender import run as run_ddm_gender
    from .stratified.stroop_gender import run as run_stroop_gender
    from .stratified.wcst_gender import run as run_wcst_gender

    # Interaction analyses
    from .interactions.ucla_gender import run as run_ucla_gender
    from .interactions.synthesis import run as run_synthesis

    ANALYSES = {
        # Vulnerability
        'male_vulnerability': run_male_vulnerability,
        'double_dissociation': run_double_dissociation,
        # Stratified
        'ddm_gender': run_ddm_gender,
        'stroop_gender': run_stroop_gender,
        'wcst_gender': run_wcst_gender,
        # Interactions
        'ucla_gender': run_ucla_gender,
        'synthesis': run_synthesis,
    }

    return ANALYSES


# =============================================================================
# RUNNER FUNCTIONS
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict:
    """
    Run gender analysis suite.

    Parameters
    ----------
    analysis : str, optional
        Specific analysis to run. If None, runs all.
    verbose : bool
        Print progress and results

    Returns
    -------
    dict
        Results from all analyses
    """
    analyses = _lazy_load_analyses()

    if verbose:
        print("=" * 70)
        print("PUBLICATION GENDER ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in analyses:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(analyses.keys())}")
        results[analysis] = analyses[analysis](verbose=verbose)
    else:
        # Run all analyses in order
        analysis_order = [
            'male_vulnerability',
            'double_dissociation',
            'ddm_gender',
            'stroop_gender',
            'wcst_gender',
            'ucla_gender',
            'synthesis',
        ]

        for name in analysis_order:
            if name in analyses:
                try:
                    results[name] = analyses[name](verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")
                    import traceback
                    traceback.print_exc()

    if verbose:
        print("\n" + "=" * 70)
        print("GENDER ANALYSIS SUITE COMPLETE")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses() -> None:
    """Print available analyses."""
    analyses = _lazy_load_analyses()

    print("\nAvailable Gender Analyses:")
    print("-" * 40)

    descriptions = {
        'male_vulnerability': 'Comprehensive male vulnerability patterns',
        'double_dissociation': 'Task-specific gender dissociation tests',
        'ddm_gender': 'Gender-stratified DDM analysis',
        'stroop_gender': 'Gender-stratified Stroop decomposition',
        'wcst_gender': 'Gender-stratified WCST error analysis',
        'ucla_gender': 'UCLA x Gender interactions across outcomes',
        'synthesis': 'Gender-specific synthesis and summary',
    }

    for name in analyses:
        desc = descriptions.get(name, '')
        print(f"  {name}: {desc}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'MIN_SAMPLE_STRATIFIED',
    'MIN_SAMPLE_INTERACTION',
    'EF_OUTCOMES',
    'PRIMARY_OUTCOMES',
    'DDM_PARAMS',
    'NETWORK_VARS',
    'DASS_COVARIATES',
    'STRATIFIED_FORMULA',
    'INTERACTION_FORMULA',
    # Utilities
    'RESULTS_DIR',
    'OUTPUT_DIR',
    'load_gender_data',
    'run_gender_stratified_regression',
    'test_gender_interaction',
    'run_all_gender_interactions',
    'cohens_d',
    'partial_eta_squared',
    'compute_gender_effect_sizes',
    'fisher_z_test',
    'compare_correlations_by_gender',
    # Runner
    'ANALYSES',
    'run',
    'list_analyses',
]
