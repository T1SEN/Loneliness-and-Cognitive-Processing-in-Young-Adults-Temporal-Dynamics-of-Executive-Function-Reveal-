# Analysis Script Classification

This document categorizes all analysis scripts by their rigor level and intended use.

**Review Date**: 2025-12-03

---

## Gold Standard (Publication-Ready)

These scripts implement full DASS-21 covariate control and are suitable for confirmatory analysis in publications.

| Script | Description | DASS Control | Notes |
|--------|-------------|--------------|-------|
| `master_dass_controlled_analysis.py` | **PRIMARY** - Main confirmatory analysis | Full | All UCLA models control for DASS-dep/anx/str + age |
| `prp_comprehensive_dass_controlled.py` | PRP analysis with DASS | Full | Lines 412, 433 |
| `loneliness_exec_models.py` | Hierarchical regression | Full | Line 202 |
| `trial_level_mixed_effects.py` | Mixed-effects models | Full | Lines 90-91, 135-136, 172-173 |
| `prp_exgaussian_dass_controlled.py` | Ex-Gaussian decomposition | Full | Line 256 |

---

## Exploratory Analysis (Hypothesis Generation)

These scripts are for exploratory analysis. Results should NOT be cited as confirmatory without noting limitations.

| Script | Description | Warning |
|--------|-------------|---------|
| `extreme_group_analysis.py` | High/Low UCLA comparison | Explicitly labeled as DASS-missing |
| `dass_anxiety_mask_hypothesis.py` | Anxiety moderation | **FIXED**: Now includes DASS dep/str control |
| `nonlinear_gender_effects.py` | Quadratic effects | DASS-controlled but exploratory |
| `ucla_nonlinear_effects.py` | Threshold analysis | Exploratory |
| `temporal_context_effects.py` | Time-of-day effects | Exploratory |

---

## Mediation Analysis

DASS is the mediator (not covariate) in these scripts - appropriate methodology.

| Script | Description | Notes |
|--------|-------------|-------|
| `dass_mediation_bootstrapped.py` | Bootstrap mediation | DASS as mediator |
| `mechanism_mediation_analysis.py` | Pathway analysis | DASS as mediator |
| `mediation_dass_complete.py` | Complete mediation | DASS as mediator |

---

## Utility/Data Processing

No hypothesis testing - used for data preparation and feature derivation.

| Script | Description |
|--------|-------------|
| `utils/data_loader_utils.py` | Central data loading module |
| `utils/trial_data_loader.py` | Trial-level data processing |
| `utils/publication_helpers.py` | Formatting utilities |
| `derive_trial_features.py` | Feature engineering |
| `ml_nested_tuned.py` | Machine learning pipeline |

---

## Newly Created Scripts (2025-12-03)

| Script | Category | DASS Control |
|--------|----------|--------------|
| `ucla_item_factor_analysis.py` | Exploratory | DASS-controlled regressions |
| `stroop_conflict_adaptation.py` | Exploratory | DASS-controlled |
| `wcst_learning_trajectory.py` | Exploratory | DASS-controlled |
| `prp_bottleneck_shape.py` | Exploratory | DASS-controlled |
| `post_error_slowing_integrated.py` | Exploratory | DASS-controlled |
| `dass_ef_specificity.py` | Exploratory | DASS subscale analysis |
| `wcst_error_type_decomposition.py` | Exploratory | DASS-controlled |
| `prp_response_order_analysis.py` | Exploratory | DASS-controlled |
| `stroop_neutral_baseline.py` | Exploratory | DASS-controlled |
| `network_psychometrics.py` | Exploratory | Network analysis |
| `reliability_corrected_effects.py` | Utility | Reliability estimation |

---

## Key Findings Summary

### Confirmatory (DASS-Controlled) Results:
- **UCLA Main Effects**: ALL non-significant after DASS control (p > 0.35)
- **UCLA × Gender**: WCST PE interaction p = 0.025
- **HMM Lapse Recovery**: Only significant UCLA effect (p = 0.0169)

### Interpretation:
- Loneliness effects on executive function are **confounded with depression/anxiety**
- Only gender-specific vulnerability is independent of mood symptoms
- Results should emphasize null findings with appropriate effect sizes

---

## Usage Guidelines

1. **For publication**: Use Gold Standard scripts only
2. **For exploration**: Clearly label as exploratory, no confirmatory claims
3. **Formula template**:
```python
smf.ols("y ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=df)
```

---

## Revision History

- 2025-12-03: Initial classification document created
- 2025-12-03: DASS control fixes applied to `dass_anxiety_mask_hypothesis.py`, `post_error_slowing_gender_moderation.py`
