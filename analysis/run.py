"""
Suite Runner Module
==================

Provides programmatic access to all analysis suites.

Usage:
    from analysis.run import run_suite, list_suites

    # Run a specific suite
    run_suite('gold_standard')

    # Run with specific analysis
    run_suite('exploratory.wcst', analysis='learning_trajectory')

    # List available suites
    list_suites()
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Dict, List, Optional, Any
import importlib
from pathlib import Path

# Registry of available suites
SUITE_REGISTRY: Dict[str, str] = {
    # Gold Standard (confirmatory)
    'gold_standard': 'analysis.gold_standard.pipeline',

    # Exploratory (task-specific)
    'exploratory.prp': 'analysis.exploratory.prp_suite',
    'exploratory.stroop': 'analysis.exploratory.stroop_suite',
    'exploratory.wcst': 'analysis.exploratory.wcst_suite',
    'exploratory.cross_task': 'analysis.exploratory.cross_task_suite',
    'exploratory.fatigue': 'analysis.exploratory.fatigue_suite',

    # Mediation
    'mediation': 'analysis.mediation.mediation_suite',

    # Validation
    'validation': 'analysis.validation.validation_suite',
    'validation.bayesian_equivalence': 'analysis.validation.bayesian_equivalence_suite',
    'validation.power': 'analysis.validation.power_suite',

    # Synthesis
    'synthesis': 'analysis.synthesis.synthesis_suite',
    'synthesis.dass_specificity': 'analysis.synthesis.dass_specificity_suite',

    # Advanced
    'advanced.mechanistic': 'analysis.advanced.mechanistic_suite',
    'advanced.latent': 'analysis.advanced.latent_suite',
    'advanced.clustering': 'analysis.advanced.clustering_suite',
    'advanced.developmental_window': 'analysis.advanced.developmental_window_suite',
    'advanced.hmm_deep': 'analysis.advanced.hmm_deep_suite',
    'advanced.pure_ucla': 'analysis.advanced.pure_ucla_suite',
    'advanced.rule_learning': 'analysis.advanced.rule_learning_suite',
    'advanced.sequential_dynamics': 'analysis.advanced.sequential_dynamics_suite',
    'advanced.wcst_mechanism': 'analysis.advanced.wcst_mechanism_deep_suite',
    'advanced.normative': 'analysis.advanced.normative_modeling_suite',
    'advanced.temporal': 'analysis.advanced.temporal_dynamics_suite',
    'advanced.hmm_mechanism': 'analysis.advanced.hmm_mechanism_suite',
    'advanced.bayesian': 'analysis.advanced.bayesian_sem_suite',
    'advanced.causal': 'analysis.advanced.causal_inference_suite',

    # Computational Modeling (new)
    'advanced.ddm': 'analysis.advanced.ddm_suite',
    'advanced.reinforcement_learning': 'analysis.advanced.reinforcement_learning_suite',
    'advanced.attention_depletion': 'analysis.advanced.attention_depletion_suite',
    'advanced.error_monitoring': 'analysis.advanced.error_monitoring_suite',
    'advanced.control_strategy': 'analysis.advanced.control_strategy_suite',
    'advanced.integration': 'analysis.advanced.integration_suite',
}

# Suite categories for organized listing
SUITE_CATEGORIES = {
    'Gold Standard': ['gold_standard'],
    'Exploratory': ['exploratory.prp', 'exploratory.stroop', 'exploratory.wcst', 'exploratory.cross_task', 'exploratory.fatigue'],
    'Mediation': ['mediation'],
    'Validation': ['validation', 'validation.bayesian_equivalence', 'validation.power'],
    'Synthesis': ['synthesis', 'synthesis.dass_specificity'],
    'Advanced': ['advanced.mechanistic', 'advanced.latent', 'advanced.clustering', 'advanced.developmental_window', 'advanced.hmm_deep', 'advanced.pure_ucla', 'advanced.rule_learning', 'advanced.sequential_dynamics', 'advanced.wcst_mechanism', 'advanced.normative', 'advanced.temporal', 'advanced.hmm_mechanism', 'advanced.bayesian', 'advanced.causal', 'advanced.ddm', 'advanced.reinforcement_learning', 'advanced.attention_depletion', 'advanced.error_monitoring', 'advanced.control_strategy', 'advanced.integration'],
}


def run_suite(
    suite_name: str,
    analysis: Optional[str] = None,
    verbose: bool = True,
    force_rebuild: bool = False,
    **kwargs,
) -> Any:
    """
    Run a specific suite or analysis.

    Parameters
    ----------
    suite_name : str
        Suite identifier (e.g., 'gold_standard', 'exploratory.wcst')
    analysis : str, optional
        Specific analysis within suite to run
    verbose : bool
        Print progress messages
    force_rebuild : bool
        Force rebuild of cached data

    Returns
    -------
    Any
        Suite-specific return value (usually dict of results)
    """
    if suite_name not in SUITE_REGISTRY:
        available = ', '.join(sorted(SUITE_REGISTRY.keys()))
        raise ValueError(f"Unknown suite: {suite_name}. Available: {available}")

    module_path = SUITE_REGISTRY[suite_name]

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import suite {suite_name} from {module_path}: {e}")

    # Build kwargs for run function
    run_kwargs = {'verbose': verbose}

    if analysis:
        run_kwargs['analysis'] = analysis

    if force_rebuild:
        run_kwargs['force_rebuild'] = force_rebuild

    run_kwargs.update(kwargs)

    # Find and call run function
    if hasattr(module, 'run'):
        return module.run(**run_kwargs)
    elif hasattr(module, 'main'):
        return module.main()
    else:
        raise AttributeError(f"Suite {suite_name} has no run() or main() function")


def run_all_suites(
    verbose: bool = True,
    force_rebuild: bool = False,
    skip_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Run all suites in recommended order.

    Parameters
    ----------
    verbose : bool
        Print progress messages
    force_rebuild : bool
        Force rebuild of cached data
    skip_on_error : bool
        Continue to next suite if one fails

    Returns
    -------
    dict
        Results from each suite (None if failed)
    """
    # Recommended execution order
    order = [
        'gold_standard',
        'mediation',
        'validation',
        'synthesis',
        'exploratory.prp',
        'exploratory.stroop',
        'exploratory.wcst',
        'exploratory.cross_task',
        'advanced.mechanistic',
        'advanced.latent',
        'advanced.clustering',
    ]

    results = {}

    for suite_name in order:
        if suite_name not in SUITE_REGISTRY:
            continue

        if verbose:
            print()
            print("=" * 80)
            print(f"RUNNING: {suite_name}")
            print("=" * 80)

        try:
            results[suite_name] = run_suite(
                suite_name,
                verbose=verbose,
                force_rebuild=force_rebuild,
            )
            if verbose:
                print(f"[OK] {suite_name} completed")
        except Exception as e:
            print(f"[ERROR] {suite_name}: {e}")
            results[suite_name] = None
            if not skip_on_error:
                raise

    return results


def list_suites(show_analyses: bool = True) -> None:
    """
    Print available suites and their analyses.

    Parameters
    ----------
    show_analyses : bool
        If True, also list analyses within each suite
    """
    print("\n" + "=" * 60)
    print("AVAILABLE ANALYSIS SUITES")
    print("=" * 60)

    for category, suite_names in SUITE_CATEGORIES.items():
        print(f"\n{category}:")
        print("-" * 40)

        for name in suite_names:
            if name not in SUITE_REGISTRY:
                continue

            module_path = SUITE_REGISTRY[name]
            print(f"  {name}")

            if show_analyses:
                try:
                    module = importlib.import_module(module_path)
                    if hasattr(module, 'ANALYSES'):
                        for analysis_name in module.ANALYSES.keys():
                            print(f"    - {analysis_name}")
                    elif hasattr(module, 'ANALYSIS_REGISTRY'):
                        for analysis_name in module.ANALYSIS_REGISTRY.keys():
                            print(f"    - {analysis_name}")
                except ImportError:
                    print(f"    (module not available)")

    print("\n" + "=" * 60)
    print("\nUsage examples:")
    print("  python -m analysis --suite gold_standard")
    print("  python -m analysis --suite exploratory.wcst --analysis learning_trajectory")
    print("  python -m analysis --all")
    print()


def get_suite_info(suite_name: str) -> Dict[str, Any]:
    """
    Get information about a specific suite.

    Returns
    -------
    dict
        Suite metadata including available analyses
    """
    if suite_name not in SUITE_REGISTRY:
        raise ValueError(f"Unknown suite: {suite_name}")

    module_path = SUITE_REGISTRY[suite_name]
    info = {
        'name': suite_name,
        'module': module_path,
        'analyses': [],
        'docstring': None,
    }

    try:
        module = importlib.import_module(module_path)
        info['docstring'] = module.__doc__

        if hasattr(module, 'ANALYSES'):
            info['analyses'] = list(module.ANALYSES.keys())
        elif hasattr(module, 'ANALYSIS_REGISTRY'):
            info['analyses'] = list(module.ANALYSIS_REGISTRY.keys())

    except ImportError as e:
        info['error'] = str(e)

    return info


if __name__ == '__main__':
    list_suites()
