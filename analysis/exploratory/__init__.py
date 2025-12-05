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
- post_error_suite: Post-error slowing across tasks
- tier1_suite: Tier 1 mechanistic analyses
- tier2_suite: Tier 2 advanced analyses

Usage:
    from analysis.exploratory import prp_suite
    prp_suite.run('bottleneck_shape')  # Run specific analysis
    prp_suite.run()  # Run all PRP analyses
"""

__all__ = ['prp_suite', 'stroop_suite', 'wcst_suite', 'post_error_suite', 'cross_task_suite']
