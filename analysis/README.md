# Analysis Scripts

This directory contains all statistical analysis scripts for the UCLA Loneliness × Executive Function study.

## Directory Structure

```
analysis/
├── __main__.py             # Unified CLI entry point
├── run.py                  # Suite runner module
│
├── gold_standard/          # Publication-ready confirmatory analyses
│   ├── pipeline.py         # Unified runner for all Gold Standard analyses
│   └── analyses.yml        # Analysis configuration
│
├── exploratory/            # Exploratory analyses (hypothesis generation)
│   ├── prp_suite.py        # PRP task analyses
│   ├── stroop_suite.py     # Stroop task analyses
│   ├── wcst_suite.py       # WCST task analyses
│   └── cross_task_suite.py # Cross-task integration analyses
│
├── mediation/              # Mediation analyses (DASS as mediator)
│   └── mediation_suite.py  # Bootstrap mediation analyses
│
├── validation/             # Methodological validation
│   └── validation_suite.py # CV, robustness, Type M/S error
│
├── synthesis/              # Integration and summary analyses
│   └── synthesis_suite.py  # Group comparisons, forest plots
│
├── advanced/               # Advanced mechanistic analyses
│   ├── mechanistic_suite.py  # Tier 1 & 2 decomposition
│   ├── latent_suite.py       # SEM, network, factor analysis
│   └── clustering_suite.py   # Vulnerability clustering
│
├── ml/                     # Machine learning analyses
│   ├── nested_cv.py        # Nested CV with hyperparameter tuning
│   ├── classification.py   # Loneliness classification
│   └── feature_selection.py # Recursive feature elimination
│
├── utils/                  # Shared utility modules
│   ├── data_loader_utils.py    # Data loading and preprocessing
│   ├── trial_data_loader.py    # Trial-level data loading
│   ├── modeling.py             # Regression utilities
│   ├── plotting.py             # Visualization utilities
│   ├── publication_helpers.py  # Formatting for publications
│   ├── trial_features.py       # Trial-level feature derivation
│   ├── exgaussian.py           # Ex-Gaussian RT fitting
│   └── post_error.py           # Post-error slowing computation
│
└── archive/                # Deprecated/superseded scripts (reference only)
    ├── README.md           # Archive documentation
    ├── legacy_confirmatory/
    ├── legacy_exploratory/
    ├── legacy_mediation/
    ├── legacy_validation/
    ├── legacy_synthesis/
    ├── legacy_mechanistic/
    └── legacy_utils/
```

## Quick Start

### Unified CLI (Recommended)

```bash
# List all available suites
python -m analysis --list

# Run Gold Standard (confirmatory analyses)
python -m analysis --suite gold_standard

# Run specific exploratory suite
python -m analysis --suite exploratory.wcst

# Run specific analysis within a suite
python -m analysis --suite exploratory.wcst --analysis learning_trajectory

# Run all suites
python -m analysis --all
```

### Direct Suite Execution

```bash
# Gold Standard confirmatory analyses
python -m analysis.gold_standard.pipeline

# Exploratory suites
python -m analysis.exploratory.prp_suite
python -m analysis.exploratory.stroop_suite
python -m analysis.exploratory.wcst_suite

# Mediation
python -m analysis.mediation.mediation_suite

# Validation
python -m analysis.validation.validation_suite

# Synthesis
python -m analysis.synthesis.synthesis_suite

# Advanced
python -m analysis.advanced.mechanistic_suite
python -m analysis.advanced.latent_suite
python -m analysis.advanced.clustering_suite

# Machine Learning
python -m analysis.ml.nested_cv --task classification --features demo_dass
```

## DASS-21 Covariate Control (CRITICAL)

**ALL confirmatory analyses MUST control for DASS-21 subscales.**

### Required Formula Template

```python
smf.ols("{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=df)
```

### Required Covariates

| Variable | Description |
|----------|-------------|
| `z_dass_dep` | DASS-21 Depression (standardized) |
| `z_dass_anx` | DASS-21 Anxiety (standardized) |
| `z_dass_str` | DASS-21 Stress (standardized) |
| `z_age` | Age (standardized) |
| `C(gender_male)` | Gender (categorical) |

### Verify DASS Control

```bash
python scripts/check_dass_control.py --gold-standard-only
```

## Script Categories

### Gold Standard (Publication-ready)

| Script | Description |
|--------|-------------|
| `gold_standard/pipeline.py` | Unified hierarchical regression with DASS control |

**Note:** Legacy scripts (`master_dass_controlled_analysis.py`, etc.) have been moved to `archive/legacy_confirmatory/`.

### Exploratory Suites

| Suite | Analyses |
|-------|----------|
| `prp_suite.py` | bottleneck_shape, response_order, rt_variability, post_error |
| `stroop_suite.py` | conflict_adaptation, neutral_baseline, post_error |
| `wcst_suite.py` | learning_trajectory, error_decomposition, post_error |
| `cross_task_suite.py` | consistency, correlations, meta_control, order_effects |

### Mediation

| Suite | Analyses |
|-------|----------|
| `mediation_suite.py` | dass_bootstrap, gender_stratified, moderated_mediation |

### Validation

| Suite | Analyses |
|-------|----------|
| `validation_suite.py` | cross_validation, robust_regression, quantile_regression, type_ms_simulation, split_half_replication |

### Synthesis

| Suite | Analyses |
|-------|----------|
| `synthesis_suite.py` | group_comparisons, forest_plot, gender_stratified, ucla_dass_correlations, variance_tests |

### Advanced

| Suite | Analyses |
|-------|----------|
| `mechanistic_suite.py` | fatigue, speed_accuracy, autocorrelation, pre_error_trajectories, cross_task_coupling |
| `latent_suite.py` | network, factor_analysis, measurement_invariance |
| `clustering_suite.py` | vulnerability, composite_index, gmm_profiles |

## Output Locations

| Category | Output Directory |
|----------|------------------|
| Gold Standard | `results/gold_standard/` |
| Exploratory | `results/analysis_outputs/{prp,stroop,wcst,cross_task}_suite/` |
| Mediation | `results/analysis_outputs/mediation_suite/` |
| Validation | `results/analysis_outputs/validation_suite/` |
| Synthesis | `results/analysis_outputs/synthesis_suite/` |
| Advanced | `results/analysis_outputs/{mechanistic,latent,clustering}_suite/` |

## Key Findings

Based on the master DASS-controlled analysis:

- **UCLA Main Effects**: Non-significant after DASS control (all p > 0.35)
- **UCLA × Gender Interaction**: Significant for WCST PE (p = 0.025)
- **Interpretation**: Loneliness effects are confounded with mood; only gender-specific vulnerability is independent

## Utility Modules

### `utils/modeling.py`

```python
from analysis.utils.modeling import (
    DASS_CONTROL_FORMULA,        # Standard formula template
    fit_dass_controlled_model,   # Fit model with DASS control
    extract_coefficients,        # Extract coefficients from model
    verify_dass_control          # Verify formula has DASS terms
)
```

### `utils/plotting.py`

```python
from analysis.utils.plotting import (
    set_publication_style,           # Configure matplotlib
    create_model_comparison_plot,    # R² comparison bars
    create_forest_plot,              # Effect sizes with CIs
    create_scatter_with_regression   # Scatter + regression lines
)
```

## Development

### Adding New Analyses

1. For confirmatory analyses:
   - Add to `gold_standard/analyses.yml`
   - Ensure DASS control is included
   - Run `scripts/check_dass_control.py` to verify

2. For exploratory analyses:
   - Add to appropriate suite (`prp_suite.py`, etc.)
   - Follow existing pattern with `@register_analysis` decorator
   - Still include DASS control for consistency

### Running Tests

```bash
# Verify DASS control in all scripts
python scripts/check_dass_control.py

# Run Gold Standard pipeline
python -m analysis.gold_standard.pipeline
```
