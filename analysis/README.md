# Analysis Scripts

This directory contains all statistical analysis scripts for the UCLA Loneliness × Executive Function study.

## Directory Structure

```
analysis/
├── gold_standard/          # Publication-ready confirmatory analyses
│   ├── pipeline.py         # Unified runner for all Gold Standard analyses
│   └── analyses.yml        # Analysis configuration
│
├── exploratory/            # Exploratory analyses (hypothesis generation)
│   ├── prp_suite.py        # PRP task analyses
│   ├── stroop_suite.py     # Stroop task analyses
│   └── wcst_suite.py       # WCST task analyses
│
├── mediation/              # Mediation analyses (DASS as mediator)
│   └── mediation_suite.py  # Bootstrap mediation analyses
│
├── utils/                  # Shared utility modules
│   ├── data_loader_utils.py    # Data loading and preprocessing
│   ├── trial_data_loader.py    # Trial-level data loading
│   ├── modeling.py             # Regression utilities
│   ├── plotting.py             # Visualization utilities
│   └── publication_helpers.py  # Formatting for publications
│
└── [legacy scripts]        # Individual analysis scripts (being consolidated)
```

## Quick Start

### Run Gold Standard Analyses (Publication-ready)

```bash
# Run all confirmatory analyses
.\venv\Scripts\python.exe -m analysis.gold_standard.pipeline

# Results saved to: results/gold_standard/
```

### Run Exploratory Analyses

```bash
# PRP analyses
.\venv\Scripts\python.exe -m analysis.exploratory.prp_suite
.\venv\Scripts\python.exe -m analysis.exploratory.prp_suite --analysis bottleneck_shape

# Stroop analyses
.\venv\Scripts\python.exe -m analysis.exploratory.stroop_suite

# WCST analyses
.\venv\Scripts\python.exe -m analysis.exploratory.wcst_suite
```

### Run Mediation Analyses

```bash
.\venv\Scripts\python.exe -m analysis.mediation.mediation_suite
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
| `master_dass_controlled_analysis.py` | Legacy: Primary confirmatory analysis |
| `prp_comprehensive_dass_controlled.py` | Legacy: PRP with full DASS control |

### Exploratory Suites

| Suite | Analyses |
|-------|----------|
| `prp_suite.py` | bottleneck_shape, response_order, rt_variability, post_error |
| `stroop_suite.py` | conflict_adaptation, neutral_baseline, post_error |
| `wcst_suite.py` | learning_trajectory, error_decomposition, post_error |

### Mediation

| Suite | Analyses |
|-------|----------|
| `mediation_suite.py` | dass_bootstrap, gender_stratified, moderated_mediation |

## Output Locations

| Category | Output Directory |
|----------|------------------|
| Gold Standard | `results/gold_standard/` |
| Exploratory | `results/analysis_outputs/{suite_name}/` |
| Mediation | `results/analysis_outputs/mediation_suite/` |

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
