"""
Exploratory Analysis Suite
==========================

Exploratory analyses for hypothesis generation.

IMPORTANT: These analyses are NOT confirmatory. Results should be interpreted
with caution and NOT cited as primary evidence in publications.

All exploratory analyses still control for DASS-21 subscales for consistency.

Available suites:
- prp_suite: PRP task analyses (bottleneck, response order, ex-Gaussian, etc.)
- stroop_suite: Stroop task analyses (conflict adaptation, neutral baseline, etc.)
- wcst_suite: WCST task analyses (learning trajectory, error decomposition, etc.)
- cross_task: Cross-task analyses (consistency, correlations, nonlinear effects)
- fatigue_suite: Fatigue and time-on-task effects
- extreme_group_ancova: Extreme group comparisons (standalone script)

Usage:
    from analysis.exploratory import prp_suite
    prp_suite.run('bottleneck_shape')  # Run specific analysis
    prp_suite.run()  # Run all PRP analyses

    from analysis.exploratory import cross_task
    cross_task.run('consistency')  # Run specific cross-task analysis
"""

__all__ = ['prp_suite', 'stroop_suite', 'wcst_suite', 'cross_task', 'fatigue_suite', 'extreme_group_ancova']
