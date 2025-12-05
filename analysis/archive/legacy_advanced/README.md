# Legacy Advanced Scripts (DEPRECATED)

> **WARNING**: These scripts have been migrated to production suites in `analysis/advanced/`.
> They are retained for reference only. Use the production suites instead.

## Migration Status

All 43 legacy scripts have been migrated to enhanced production suites with improved statistical rigor:

| Production Suite | Legacy Scripts | Improvements |
|-----------------|----------------|--------------|
| `analysis/advanced/mechanistic_suite.py` | `exgaussian_rt_analysis.py`, `stroop_exgaussian_decomposition.py`, `prp_exgaussian_*.py`, `iiv_decomposition_analysis.py` | FDR correction (9 tests), Bootstrap SE, HC3 robust errors |
| `analysis/advanced/sequential_dynamics_suite.py` | `adaptive_recovery_dynamics.py`, `sequential_dynamics_analysis.py`, `post_error_slowing_*.py` | Exponential tau fitting, Bootstrap CI, Proactive/reactive indices |
| `analysis/advanced/clustering_suite.py` | `ef_vulnerability_clustering.py`, `composite_vulnerability_indices.py`, `attentional_lapse_mixture.py`, `multivariate_ef_analysis.py` | MANOVA assumption checks (Shapiro-Wilk, Box's M), Bootstrap cluster stability (ARI), FDR for post-hoc ANOVAs |
| `analysis/advanced/latent_suite.py` | `network_psychometrics*.py`, `framework3_latent_factors_sem.py`, `latent_metacontrol_sem.py` | GraphicalLASSO regularization, Network Comparison Test (NCT), Bootstrap edge stability |
| `analysis/advanced/hmm_deep_suite.py` | `attentional_lapse_mixture.py`, `trial_level_mvpa_vulnerability.py` | 4 chains x 2000 draws, HMM state identification |
| `analysis/advanced/causal_inference_suite.py` | `framework4_causal_dag_simulation.py`, `causal_dag_extended.py` | Improved DAG specification |

## Key Statistical Improvements

### 1. Multiple Comparison Correction
- **Before**: No correction for 9+ simultaneous tests
- **After**: Benjamini-Hochberg FDR applied to all analyses

### 2. Parameter Uncertainty
- **Before**: Point estimates only
- **After**: Bootstrap SE for exponential tau, ex-Gaussian parameters

### 3. Regularization
- **Before**: `np.linalg.inv(corr_matrix)` for partial correlations (unstable)
- **After**: GraphicalLASSO with cross-validated alpha

### 4. Network Comparison
- **Before**: No formal gender network comparison
- **After**: NCT with 1000 permutations for global strength and edge differences

### 5. Cluster Stability
- **Before**: K-means without validation
- **After**: Bootstrap ARI stability check (100 iterations)

### 6. MANOVA Assumptions
- **Before**: MANOVA without assumption checks
- **After**: Shapiro-Wilk per DV, Levene's test, Box's M approximation

### 7. Bayesian Sampling
- **Before**: 2 chains x 1000 draws (insufficient for convergence diagnostics)
- **After**: 4 chains x 2000 draws with R-hat monitoring

## How to Use Production Suites

```python
# Instead of running legacy scripts directly:
# python analysis/archive/legacy_advanced/adaptive_recovery_dynamics.py  # DON'T

# Use the production suite:
python -m analysis.advanced.sequential_dynamics_suite --analysis adaptive_recovery

# Or programmatically:
from analysis.advanced import sequential_dynamics_suite
results = sequential_dynamics_suite.run(analysis='adaptive_recovery')
```

## Full Legacy-to-Suite Mapping

| Legacy Script | Production Suite | Analysis Name |
|--------------|-----------------|---------------|
| `adaptive_recovery_dynamics.py` | `sequential_dynamics_suite` | `adaptive_recovery` |
| `attentional_lapse_mixture.py` | `clustering_suite` | `gmm_profiles` |
| `causal_dag_extended.py` | `causal_inference_suite` | `extended_dag` |
| `composite_vulnerability_indices.py` | `clustering_suite` | `composite_index` |
| `dass_ef_hier_bayes.py` | `latent_suite` | `hierarchical_bayes` |
| `dass_ef_specificity.py` | `synthesis.dass_specificity_suite` | (direct) |
| `ef_vulnerability_clustering.py` | `clustering_suite` | `vulnerability` |
| `error_burst_clustering.py` | `clustering_suite` | `error_burst` |
| `exgaussian_rt_analysis.py` | `mechanistic_suite` | `exgaussian` |
| `framework1_regression_mixtures.py` | `clustering_suite` | `cross_task_profile` |
| `framework2_normative_modeling.py` | `normative_modeling_suite` | (direct) |
| `framework3_latent_factors_sem.py` | `latent_suite` | `sem_factors` |
| `framework4_causal_dag_simulation.py` | `causal_inference_suite` | `dag_simulation` |
| `gendered_temporal_vulnerability.py` | `clustering_suite` | `gendered_vulnerability` |
| `iiv_decomposition_analysis.py` | `mechanistic_suite` | `lapse_decomposition` |
| `latent_metacontrol_sem.py` | `latent_suite` | `metacontrol` |
| `multivariate_ef_analysis.py` | `clustering_suite` | `manova_validation` |
| `network_psychometrics.py` | `latent_suite` | `network_extended` |
| `network_psychometrics_extended.py` | `latent_suite` | `network_extended` |
| `perseveration_momentum_analysis.py` | `sequential_dynamics_suite` | `momentum` |
| `post_error_slowing_gender_moderation.py` | `sequential_dynamics_suite` | `adaptive_recovery` |
| `post_error_slowing_integrated.py` | `sequential_dynamics_suite` | `recovery_trajectory` |
| `proactive_reactive_control.py` | `sequential_dynamics_suite` | `adaptive_recovery` |
| `prp_exgaussian_*.py` | `mechanistic_suite` | `exgaussian` |
| `prp_post_error_adjustments.py` | `sequential_dynamics_suite` | `recovery_trajectory` |
| `prp_response_order_analysis.py` | `exploratory.prp_suite` | `response_order` |
| `prp_rt_variability_extended.py` | `mechanistic_suite` | `lapse_decomposition` |
| `rt_percentile_group_comparison.py` | `mechanistic_suite` | `exgaussian` |
| `sequential_dynamics_analysis.py` | `sequential_dynamics_suite` | `error_cascade` |
| `stroop_exgaussian_decomposition.py` | `mechanistic_suite` | `exgaussian` |
| `stroop_post_error_adjustments.py` | `sequential_dynamics_suite` | `recovery_trajectory` |
| `stroop_rt_variability_extended.py` | `mechanistic_suite` | `lapse_decomposition` |
| `trial_level_bayesian.py` | `hmm_deep_suite` | `bayesian_states` |
| `trial_level_cascade_glmm.py` | `sequential_dynamics_suite` | `resilience` |
| `trial_level_mixed_effects.py` | `sequential_dynamics_suite` | `volatility` |
| `trial_level_mvpa_vulnerability.py` | `clustering_suite` | `vulnerability` |
| `ucla_dass_moderation_commonality.py` | `synthesis_suite` | `commonality` |
| `wcst_error_type_decomposition.py` | `wcst_mechanism_deep_suite` | `error_types` |
| `wcst_mechanism_comprehensive.py` | `wcst_mechanism_deep_suite` | (multiple) |
| `wcst_post_error_adaptation_quick.py` | `sequential_dynamics_suite` | `recovery_trajectory` |
| `wcst_switching_dynamics_quick.py` | `sequential_dynamics_suite` | `error_cascade` |

## Contact

For questions about migration, see `CLAUDE.md` or run:
```bash
python -m analysis --list
```

---
*Last updated: 2025-12*
