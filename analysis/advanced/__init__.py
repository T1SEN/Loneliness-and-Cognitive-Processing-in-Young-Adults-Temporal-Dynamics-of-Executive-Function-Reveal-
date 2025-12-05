"""
Advanced Analysis Suite
=======================

Advanced mechanistic and methodological analyses for UCLA × Executive Function research.

Modules:
- mechanistic_suite: Tier 1 & 2 mechanistic decomposition
- latent_suite: SEM, network psychometrics, mixture models
- clustering_suite: Subgroup/vulnerability clustering
- hmm_deep_suite: Deep HMM attentional state analysis
- sequential_dynamics_suite: Trial-level sequential dynamics
- pure_ucla_suite: DASS-independent "pure" UCLA effects
- wcst_mechanism_deep_suite: WCST PE gender-specific mechanism analysis
- normative_modeling_suite: Personalized deviation analysis (Gaussian Process)
- temporal_dynamics_suite: Trial-level time series analysis (ACF, DFA, change points)

Computational Modeling (new):
- ddm_suite: Drift-Diffusion Model (EZ-DDM) for Stroop
- reinforcement_learning_suite: Rescorla-Wagner models for WCST
- attention_depletion_suite: Fatigue trajectory and tau accumulation
- error_monitoring_suite: PES decomposition, error cascade, conflict detection
- control_strategy_suite: Proactive vs Reactive control (DMC framework)
- integration_suite: Cross-suite integration and effect size summary

Usage:
    from analysis.advanced import mechanistic_suite, hmm_deep_suite
    mechanistic_suite.run()  # Run all
    hmm_deep_suite.run('model_comparison')  # Specific analysis

    # Computational modeling suites
    from analysis.advanced import ddm_suite, reinforcement_learning_suite
    ddm_suite.run()  # DDM parameter estimation
    reinforcement_learning_suite.run()  # RL model fitting
"""

__all__ = [
    'mechanistic_suite',
    'latent_suite',
    'clustering_suite',
    'hmm_deep_suite',
    'sequential_dynamics_suite',
    'pure_ucla_suite',
    'wcst_mechanism_deep_suite',
    'normative_modeling_suite',
    'temporal_dynamics_suite',
    'hmm_mechanism_suite',
    'bayesian_sem_suite',
    'causal_inference_suite',
    # Computational Modeling (new)
    'ddm_suite',
    'reinforcement_learning_suite',
    'attention_depletion_suite',
    'error_monitoring_suite',
    'control_strategy_suite',
    'integration_suite',
]
