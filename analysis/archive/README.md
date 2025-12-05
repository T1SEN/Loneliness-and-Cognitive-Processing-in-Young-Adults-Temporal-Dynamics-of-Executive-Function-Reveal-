# Archived Analysis Scripts

This directory contains legacy scripts that have been superseded by consolidated suite files or the Gold Standard pipeline.

**Archive Date:** 2025-12-03
**Last Updated:** 2025-12-04

## Migration Status Summary

| Category | Migrated | Reference Only | Total |
|----------|----------|----------------|-------|
| `legacy_confirmatory/` | ✅ 4/4 (100%) | - | 4 |
| `legacy_synthesis/` | ✅ 16/16 (100%) | - | 16 |
| `legacy_validation/` | ✅ 18/18 (100%) | - | 18 |
| `legacy_mediation/` | ✅ 6/6 (100%) | - | 6 |
| `legacy_mechanistic/` | ✅ 9/9 (100%) | - | 9 |
| `legacy_exploratory/` | ✅ 21/21 (100%) | - | 21 |
| `legacy_advanced/` | ✅ 4 | 📚 36 | 40 |
| `legacy_utils/` | ✅ 3/3 (100%) | - | 3 |
| **Total** | **81** | **36** | **117** |

---

## Directory Structure

### `legacy_confirmatory/`
Scripts superseded by `gold_standard/pipeline.py`

| Script | Superseded By | Notes |
|--------|---------------|-------|
| `master_dass_controlled_analysis.py` | `gold_standard/pipeline.py` | Original DASS-controlled confirmatory analysis |
| `loneliness_exec_models.py` | `gold_standard/pipeline.py` | Early implementation of core regression |
| `prp_comprehensive_dass_controlled.py` | `gold_standard/pipeline.py` | PRP-specific confirmatory (now in pipeline) |
| `dass_exec_models.py` | `gold_standard/pipeline.py` | DASS predictor models (exploratory) |

### `legacy_synthesis/`
Scripts consolidated into `synthesis/synthesis_suite.py`

| Script | Superseded By |
|--------|---------------|
| `synthesis_1_group_comparisons.py` | `synthesis_suite.py::group_comparisons` |
| `synthesis_2_forest_plot.py` | `synthesis_suite.py::forest_plot` |
| `synthesis_3_reliability_analysis.py` | `synthesis_suite.py::reliability` |
| `synthesis_4_gender_stratified_regressions.py` | `synthesis_suite.py::gender_stratified` |
| `synthesis_5_ucla_dass_correlation_by_gender.py` | `synthesis_suite.py::ucla_dass_correlation` |
| `synthesis_6_variance_tests.py` | `synthesis_suite.py::variance_tests` |
| `synthesis_7_power_analysis.py` | `validation/power_suite.py` |
| `synthesis_8_robust_sensitivity.py` | `synthesis_suite.py::robust_sensitivity` |
| `effect_size_meta_summary.py` | `synthesis_suite.py::effect_size_summary` |
| `advanced_analyses_summary.py` | Summary generation |
| `comprehensive_precision_review.py` | Precision review |
| `create_integrated_comparison.py` | Integrated comparisons |
| `create_integrated_figure.py` | Figure generation |
| `create_visualizations.py` | Visualization scripts |
| `generate_comprehensive_final_report.py` | Report generation |
| `generate_publication_figures.py` | Publication figures |

### `legacy_validation/`
Scripts consolidated into `validation/validation_suite.py`

| Script | Superseded By |
|--------|---------------|
| `validation_1_cross_validation.py` | `validation_suite.py::cross_validation` |
| `validation_2_robust_quantile.py` | `validation_suite.py::robust_quantile` |
| `validation_3_type_ms_simulation.py` | `validation_suite.py::type_ms_simulation` |
| `replication_verification_corrected.py` | `validation_suite.py::replication` |
| `split_half_internal_replication.py` | `validation_suite.py::split_half` |
| `specification_curve_analysis.py` | `validation_suite.py::specification_curve` |
| `statistical_robustness.py` | `validation_suite.py::robustness` |
| `bonferroni_correction_analysis.py` | Multiple comparison correction |
| `equiv_loo_tost.py` | Equivalence testing |
| `equivalence_and_invariance.py` | Measurement invariance |
| `measurement_invariance_full.py` | Full invariance testing |
| `power_reliability_adjusted.py` | Power analysis |
| `power_sensitivity_analysis.py` | Sensitivity analysis |
| `reliability_corrected_analysis.py` | Reliability correction |
| `reliability_corrected_effects.py` | Corrected effect sizes |
| `reliability_enhancement_composites.py` | Composite reliability |
| `rope_loo_summary.py` | ROPE/LOO summary |
| `ucla_item_factor_analysis.py` | Factor analysis |

### `legacy_mediation/`
Scripts consolidated into `mediation/mediation_suite.py`

| Script | Superseded By |
|--------|---------------|
| `dass_mediation_bootstrapped.py` | `mediation_suite.py::bootstrap_mediation` |
| `mechanism_mediation_analysis.py` | `mediation_suite.py::mechanism_pathways` |
| `mediation_dass_complete.py` | `mediation_suite.py::dass_complete` |
| `mediation_gender_pathways.py` | `mediation_suite.py::gender_pathways` |
| `tau_moderated_mediation.py` | `mediation_suite.py::tau_moderated` |
| `exgaussian_mediation_analysis.py` | Ex-Gaussian mediation (exploratory) |

### `legacy_mechanistic/`
Scripts consolidated into `advanced/mechanistic_suite.py`

| Script | Superseded By |
|--------|---------------|
| `tier1_autocorrelation.py` | `mechanistic_suite.py::autocorrelation` |
| `tier1_exgaussian_cross_task.py` | `mechanistic_suite.py::exgaussian_cross_task` |
| `tier1_fatigue_analysis.py` | `mechanistic_suite.py::fatigue` |
| `tier1_lapse_decomposition.py` | `mechanistic_suite.py::lapse_decomposition` |
| `tier1_speed_accuracy_tradeoff.py` | `mechanistic_suite.py::speed_accuracy` |
| `tier2_hmm_attentional_states.py` | `mechanistic_suite.py::hmm_states` |
| `tier2_meta_analysis.py` | `mechanistic_suite.py::meta_analysis` |
| `tier2_pre_error_trajectories.py` | `mechanistic_suite.py::pre_error` |
| `tier2_prp_coupling.py` | `mechanistic_suite.py::prp_coupling` |

### `legacy_exploratory/`
Scripts consolidated into `exploratory/*_suite.py` files ✅ **100% Migrated**

| Script | Superseded By |
|--------|---------------|
| `cross_task_consistency.py` | `cross_task_suite.py::consistency` |
| `cross_task_integration.py` | `cross_task_suite.py::integration` |
| `cross_task_meta_control.py` | `cross_task_suite.py::meta_control` |
| `cross_task_order_effects.py` | `cross_task_suite.py::order_effects` |
| `task_order_effects.py` | `cross_task_suite.py::order_effects` (duplicate) |
| `age_gam_developmental_windows.py` | `cross_task_suite.py::age_gam` |
| `age_gender_ucla_threeway.py` | `cross_task_suite.py::threeway_interaction` |
| `dass_anxiety_mask_hypothesis.py` | `cross_task_suite.py::anxiety_mask` |
| `dose_response_threshold_analysis.py` | `cross_task_suite.py::dose_response` |
| `extreme_group_analysis.py` | `cross_task_suite.py::extreme_groups` |
| `hidden_patterns_analysis.py` | `cross_task_suite.py::hidden_patterns` |
| `nonlinear_gender_effects.py` | `cross_task_suite.py::nonlinear_gender` |
| `nonlinear_threshold_analysis.py` | `cross_task_suite.py::nonlinear_threshold` |
| `prp_bottleneck_shape.py` | `prp_suite.py::bottleneck_shape` |
| `residual_ucla_analysis.py` | `cross_task_suite.py::residual_ucla` |
| `stroop_conflict_adaptation.py` | `stroop_suite.py::conflict_adaptation` |
| `stroop_cse_conflict_adaptation.py` | `stroop_suite.py::cse` |
| `stroop_neutral_baseline.py` | `stroop_suite.py::neutral_baseline` |
| `temporal_context_effects.py` | `cross_task_suite.py::temporal_context` |
| `ucla_nonlinear_effects.py` | `cross_task_suite.py::nonlinear_ucla` |
| `wcst_learning_trajectory.py` | `wcst_suite.py::learning_trajectory` |

### `legacy_advanced/`
Advanced analysis scripts - **4 Migrated**, 36 Reference Only

#### Migrated to Suite Files ✅

| Script | Superseded By |
|--------|---------------|
| `stroop_exgaussian_decomposition.py` | `stroop_suite.py::exgaussian` |
| `prp_exgaussian_decomposition.py` | `prp_suite.py::exgaussian` |
| `wcst_switching_dynamics_quick.py` | `wcst_suite.py::switching_dynamics` |
| `wcst_mechanism_comprehensive.py` | `wcst_suite.py::mechanism` |

#### Reference Only 📚 (36 scripts - kept for future reference)

| Script | Category |
|--------|----------|
| `adaptive_recovery_dynamics.py` | Temporal dynamics |
| `attentional_lapse_mixture.py` | Mixture modeling |
| `causal_dag_extended.py` | Causal inference |
| `composite_vulnerability_indices.py` | Clustering |
| `dass_ef_hier_bayes.py` | Bayesian hierarchical |
| `dass_ef_specificity.py` | DASS specificity |
| `ef_vulnerability_clustering.py` | Clustering |
| `error_burst_clustering.py` | Error patterns |
| `exgaussian_rt_analysis.py` | Ex-Gaussian fitting |
| `framework1_regression_mixtures.py` | Regression mixtures |
| `framework2_normative_modeling.py` | Normative modeling |
| `framework3_latent_factors_sem.py` | SEM/latent factors |
| `framework4_causal_dag_simulation.py` | Causal simulation |
| `gendered_temporal_vulnerability.py` | Gender × temporal |
| `iiv_decomposition_analysis.py` | Intra-individual variability |
| `latent_metacontrol_sem.py` | SEM meta-control |
| `multivariate_ef_analysis.py` | Multivariate EF |
| `network_psychometrics.py` | Network analysis |
| `network_psychometrics_extended.py` | Extended network |
| `perseveration_momentum_analysis.py` | Perseveration dynamics |
| `post_error_slowing_gender_moderation.py` | PES × gender |
| `post_error_slowing_integrated.py` | Integrated PES |
| `proactive_reactive_control.py` | Control modes |
| `prp_exgaussian_dass_controlled.py` | PRP ex-Gaussian |
| `prp_post_error_adjustments.py` | PRP post-error |
| `prp_response_order_analysis.py` | Response order |
| `prp_rt_variability_extended.py` | RT variability |
| `sequential_dynamics_analysis.py` | Sequential effects |
| `stroop_post_error_adjustments.py` | Stroop post-error |
| `stroop_rt_variability_extended.py` | RT variability |
| `trial_level_bayesian.py` | Bayesian trials |
| `trial_level_cascade_glmm.py` | Cascade GLMM |
| `trial_level_mvpa_vulnerability.py` | MVPA |
| `ucla_dass_moderation_commonality.py` | Moderation analysis |
| `wcst_error_type_decomposition.py` | Error decomposition |
| `wcst_post_error_adaptation_quick.py` | WCST post-error |

### `legacy_utils/`
Deprecated utility scripts

| Script | Status |
|--------|--------|
| `statistical_utils.py` | Functions moved to `utils/modeling.py` |
| `statistical_methodological_suite.py` | Superseded by individual suite files |
| `derive_additional_trial_features.py` | Merged into `utils/trial_features.py` |

---

## Usage

These scripts are kept for reference only. For active analysis, use:

```bash
# Gold Standard confirmatory analyses
python -m analysis.gold_standard.pipeline

# Task-specific exploratory
python -m analysis.exploratory.wcst_suite

# Validation suite
python -m analysis.validation.validation_suite

# Full pipeline
python -m analysis --all
```

---

## Restoration

If you need to restore a script temporarily:

```bash
# Example: restore master_dass_controlled_analysis.py
cp analysis/archive/legacy_confirmatory/master_dass_controlled_analysis.py analysis/
```

**Note:** Restored scripts may have broken imports. Update imports to use `analysis.utils.*` modules.
