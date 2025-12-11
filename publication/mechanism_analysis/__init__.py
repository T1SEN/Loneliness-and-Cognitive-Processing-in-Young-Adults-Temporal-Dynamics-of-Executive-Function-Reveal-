"""
Publication Mechanism Analysis Package
======================================

Computational modeling of cognitive processes using reinforcement learning,
Hidden Markov Models, and Ex-Gaussian RT decomposition.

This package contains paired suites for each mechanism:
- wcst_rl_modeling_dass: Reinforcement learning models (Rescorla-Wagner variants) with DASS control
- wcst_rl_path: RL parameter path analysis (UCLA↔DASS)
- wcst_hmm_modeling_dass: Hidden Markov Model lapse analysis with DASS control
- wcst_hmm_path: HMM lapse path analysis
- stroop_exgaussian_dass: Stroop ex-Gaussian decomposition (DASS-controlled)
- stroop_exgaussian_path: Stroop ex-Gaussian path analysis
- prp_exgaussian_dass: PRP ex-Gaussian decomposition (DASS-controlled)
- prp_exgaussian_path: PRP ex-Gaussian path analysis

Usage:
    python -m publication.mechanism_analysis --list
    python -m publication.mechanism_analysis -a wcst_rl_modeling_dass
    python -m publication.mechanism_analysis -a wcst_rl_modeling_dass --sub rescorla_wagner
    python -m publication.mechanism_analysis -a wcst_rl_path
    python -m publication.mechanism_analysis -a wcst_hmm_modeling_dass --sub dass_controlled_hmm
    python -m publication.mechanism_analysis -a wcst_hmm_path
    python -m publication.mechanism_analysis -a stroop_exgaussian_dass
    python -m publication.mechanism_analysis -a stroop_exgaussian_path
    python -m publication.mechanism_analysis -a prp_exgaussian_dass --sub bottleneck_effects
    python -m publication.mechanism_analysis -a prp_exgaussian_path

Key Findings Context:
- UCLA main effects are non-significant after DASS control
- UCLA x Gender interactions may remain significant for specific mechanisms
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Dict, Optional, Callable

from ._utils import BASE_OUTPUT

# =============================================================================
# ANALYSIS REGISTRY (Lazy Loading)
# =============================================================================

ANALYSES: Dict[str, Callable] = {}

# Sub-analysis descriptions for list_analyses()
SUB_ANALYSES = {
    'wcst_rl_modeling_dass': {
        'rescorla_wagner': 'Basic Rescorla-Wagner model fitting',
        'asymmetric_learning': 'Asymmetric learning rates (positive/negative)',
        'model_comparison': 'Compare RW model variants (BIC/AIC)',
        'parameter_recovery': 'Validate parameter recovery via simulation',
        'ucla_relationship': 'UCLA effects on RL parameters (DASS-controlled)',
    },
    'wcst_rl_path': {
        'path_models': 'Loneliness↔DASS path analysis for RL parameters',
    },
    'wcst_hmm_modeling_dass': {
        'dass_controlled_hmm': 'HMM state analysis with DASS control',
        'gender_stratified': 'Separate HMM analysis for males/females',
        'mediation_pathway': 'UCLA -> DASS -> Lapse -> PE mediation',
        'state_characteristics': 'Detailed state-specific analysis',
        'recovery_dynamics': 'Lapse recovery patterns by UCLA',
    },
    'wcst_hmm_path': {
        'path_models': 'Loneliness↔DASS path analysis for HMM lapse metrics',
    },
    'stroop_exgaussian_dass': {
        'basic_exgaussian': 'Fit ex-Gaussian to Stroop RTs by condition',
        'interference_effects': 'Compute Stroop interference on ex-Gaussian',
        'ucla_relationship': 'UCLA effects on ex-Gaussian (DASS-controlled + gender-stratified)',
    },
    'stroop_exgaussian_path': {
        'path_models': 'Loneliness↔DASS path analysis for Stroop ex-Gaussian',
    },
    'prp_exgaussian_dass': {
        'basic_exgaussian': 'Fit ex-Gaussian to PRP T2 RTs by SOA (short/long only)',
        'bottleneck_effects': 'Compute bottleneck effects on ex-Gaussian',
        'ucla_relationship': 'UCLA effects on ex-Gaussian (DASS-controlled + gender-stratified)',
    },
    'prp_exgaussian_path': {
        'path_models': 'Loneliness↔DASS path analysis for PRP ex-Gaussian',
    },
}


def _lazy_load_analyses() -> Dict[str, Callable]:
    """Lazily load analysis modules to avoid circular imports."""
    global ANALYSES

    if ANALYSES:
        return ANALYSES

    from .wcst_rl_modeling_dass import run as run_rl_dass
    from .wcst_rl_path import run as run_rl_path
    from .wcst_hmm_modeling_dass import run as run_hmm_dass
    from .wcst_hmm_path import run as run_hmm_path
    from .stroop_exgaussian_dass import run as run_stroop_dass
    from .stroop_exgaussian_path import run as run_stroop_path
    from .prp_exgaussian_dass import run as run_prp_dass
    from .prp_exgaussian_path import run as run_prp_path

    ANALYSES = {
        'wcst_rl_modeling_dass': run_rl_dass,
        'wcst_rl_path': run_rl_path,
        'wcst_hmm_modeling_dass': run_hmm_dass,
        'wcst_hmm_path': run_hmm_path,
        'stroop_exgaussian_dass': run_stroop_dass,
        'stroop_exgaussian_path': run_stroop_path,
        'prp_exgaussian_dass': run_prp_dass,
        'prp_exgaussian_path': run_prp_path,
    }

    return ANALYSES


# =============================================================================
# RUNNER FUNCTIONS
# =============================================================================

def run(
    analysis: Optional[str] = None,
    sub_analysis: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Run mechanism analyses.

    Parameters
    ----------
    analysis : str, optional
        Suite to run (e.g., 'wcst_rl_modeling_dass', 'wcst_rl_path').
        If None, runs all suites.
    sub_analysis : str, optional
        Specific sub-analysis within suite.
        If None, runs all sub-analyses in the suite.
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict
        Results from all analyses.

    Examples
    --------
    >>> from publication.mechanism_analysis import run
    >>> run('wcst_rl_modeling_dass')  # Run all DASS RL analyses
    >>> run('wcst_rl_modeling_dass', 'rescorla_wagner')  # Run specific
    """
    analyses = _lazy_load_analyses()
    results = {}

    if sub_analysis and not analysis:
        raise ValueError(
            f"--sub '{sub_analysis}' requires specifying --analysis/-a first.\n"
            f"Example: python -m publication.mechanism_analysis -a wcst_rl_modeling_dass --sub {sub_analysis}"
        )

    if verbose:
        print("=" * 70)
        print("PUBLICATION MECHANISM ANALYSIS SUITE")
        print("=" * 70)

    if analysis:
        if analysis not in analyses:
            raise ValueError(
                f"Unknown analysis: {analysis}. "
                f"Available: {list(analyses.keys())}"
            )
        # Pass sub_analysis to the suite's run() function
        results[analysis] = analyses[analysis](
            analysis=sub_analysis,
            verbose=verbose
        )
    else:
        # Run all suites
        for name, func in analyses.items():
            try:
                results[name] = func(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()

    if verbose:
        print("\n" + "=" * 70)
        print("MECHANISM ANALYSIS SUITE COMPLETE")
        print(f"Output: {BASE_OUTPUT}")
        print("=" * 70)

    return results


def list_analyses() -> None:
    """Print available analyses with descriptions."""
    print("\n" + "=" * 60)
    print("Available Mechanism Analyses")
    print("=" * 60)

    for suite, sub_analyses in SUB_ANALYSES.items():
        print(f"\n{suite}:")
        print("-" * 50)
        for name, desc in sub_analyses.items():
            print(f"  {name}: {desc}")

    print("\n" + "-" * 60)
    print("Usage examples:")
    print("  python -m publication.mechanism_analysis -a wcst_rl_modeling_dass")
    print("  python -m publication.mechanism_analysis -a wcst_rl_modeling_dass --sub rescorla_wagner")
    print("  python -m publication.mechanism_analysis -a wcst_rl_path")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'BASE_OUTPUT',
    'SUB_ANALYSES',
    # Registry
    'ANALYSES',
    # Functions
    'run',
    'list_analyses',
]
