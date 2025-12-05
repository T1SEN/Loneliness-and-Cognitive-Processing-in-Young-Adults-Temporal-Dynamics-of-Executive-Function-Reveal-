# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research data analysis pipeline for a psychology study examining the relationship between loneliness (UCLA Loneliness Scale) and executive function (EF) across three cognitive tasks: Stroop (interference control), WCST (set-shifting), and PRP (dual-task coordination).

**Two main components:**
1. **Data Export**: Firebase → CSV extraction (`export_alldata.py`)
2. **Statistical Analysis**: Suite-based analysis pipeline (`analysis/`)

## Data Flow

```
Firebase (Firestore) → export_alldata.py → results/*.csv → python -m analysis → results/gold_standard/ & results/analysis_outputs/
```

### Key Data Files (`results/`)
| File | Contents |
|------|----------|
| `1_participants_info.csv` | Demographics (age, gender, education) |
| `2_surveys_results.csv` | UCLA Loneliness & DASS-21 responses |
| `3_cognitive_tests_summary.csv` | Aggregate task metrics |
| `4a_prp_trials.csv` | Trial-level PRP data |
| `4b_wcst_trials.csv` | Trial-level WCST data |
| `4c_stroop_trials.csv` | Trial-level Stroop data |

## Essential Commands

```bash
# Activate venv (Windows)
.\venv\Scripts\activate

# Run unified CLI
python -m analysis --list                     # List available suites
python -m analysis --suite gold_standard      # Run Gold Standard confirmatory
python -m analysis --suite exploratory.wcst   # Run WCST exploratory suite
python -m analysis --all                      # Run all suites

# Run individual suite modules
python -m analysis.gold_standard.pipeline
python -m analysis.exploratory.prp_suite
python -m analysis.mediation.mediation_suite
python -m analysis.validation.validation_suite

# Machine learning
python -m analysis.ml.nested_cv --task classification --features demo_dass

# Export data from Firebase (requires serviceAccountKey.json)
PYTHONIOENCODING=utf-8 .\venv\Scripts\python.exe export_alldata.py
```

## Analysis Architecture

```
analysis/
├── __main__.py             # Unified CLI entry point
├── run.py                  # Suite runner
├── preprocessing/          # Data loading and cleaning
│   ├── loaders.py          # load_master_dataset, load_*_scores
│   ├── trial_loaders.py    # load_prp_trials, load_stroop_trials, load_wcst_trials
│   ├── standardization.py  # safe_zscore, standardize_predictors, prepare_gender_variable
│   ├── features.py         # derive_all_features, derive_*_features
│   └── constants.py        # RT thresholds, SOA constants
├── statistics/             # Statistical utilities
│   ├── exgaussian.py       # Ex-Gaussian RT fitting
│   └── post_error.py       # Post-error slowing computation
├── visualization/          # Plotting utilities
│   ├── plotting.py         # set_publication_style, forest plots
│   └── publication.py      # APA formatting, effect sizes
├── utils/                  # Modeling only
│   └── modeling.py         # DASS_CONTROL_FORMULA, fit_dass_controlled_model
├── gold_standard/          # Confirmatory analyses (DASS-controlled)
│   ├── pipeline.py
│   └── analyses.yml        # Analysis configuration
├── exploratory/            # Hypothesis generation
│   ├── prp_suite.py
│   ├── stroop_suite.py
│   ├── wcst_suite.py
│   └── cross_task/         # Cross-task analyses (split into modules)
│       ├── consistency.py
│       ├── age_gender.py
│       ├── nonlinear.py
│       └── residual_temporal.py
├── mediation/              # DASS as mediator (not covariate)
├── validation/             # CV, robustness, Type M/S error
├── synthesis/              # Integration and summary
├── advanced/               # Mechanistic, latent, clustering (enhanced)
│   ├── mechanistic_suite.py    # Ex-Gaussian, fatigue, autocorrelation (FDR-corrected)
│   ├── sequential_dynamics_suite.py  # Adaptive recovery, error cascade
│   ├── clustering_suite.py     # MANOVA validation, GMM profiles
│   ├── latent_suite.py         # Network analysis (GraphicalLASSO, NCT)
│   └── ...                     # See run.py for full list
├── ml/                     # Machine learning pipelines
└── archive/                # Legacy scripts (DEPRECATED - see README.md)
```

## ⚠️ CRITICAL: DASS-21 Covariate Control

**ALL confirmatory analyses testing UCLA effects MUST control for DASS-21 subscales.**

UCLA loneliness and DASS (depression/anxiety/stress) correlate r ~ 0.5-0.7. Without DASS control, "loneliness effects" confound with general emotional distress. Master analysis showed ALL UCLA main effects disappear when DASS is controlled; only UCLA × Gender interactions survive.

### Required Formula Template
```python
from analysis.utils.modeling import DASS_CONTROL_FORMULA

# Standard formula:
smf.ols("{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=df)
```

**Exception:** Mediation analyses where DASS is the mediator (not a covariate).

## Shared Utility Modules

### `analysis/preprocessing/` - Data Loading & Cleaning
```python
from analysis.preprocessing import (
    # Data loaders
    load_master_dataset,      # Cached unified dataset (master_dataset.parquet)
    load_participants, load_ucla_scores, load_dass_scores,
    ensure_participant_id,    # Normalize participant ID column
    normalize_gender_value,   # Map Korean/English gender to 'male'/'female'
    # Trial loaders
    load_prp_trials, load_stroop_trials, load_wcst_trials,
    # Standardization
    safe_zscore, standardize_predictors, prepare_gender_variable,
    apply_fdr_correction, find_interaction_term,
    # Constants
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, PRP_RT_MAX, STROOP_RT_MAX,
)
```

### `analysis/utils/modeling.py` - Regression Templates
```python
from analysis.utils.modeling import (
    DASS_CONTROL_FORMULA,        # "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    fit_dass_controlled_model,   # Fit OLS with HC3 robust SE
    verify_dass_control,         # Verify formula has required terms
)
```

### `analysis/statistics/` - Statistical Utilities
```python
from analysis.statistics import (
    fit_exgaussian, fit_exgaussian_by_condition,  # Ex-Gaussian RT fitting
    compute_pes, compute_all_task_pes,            # Post-error slowing
)
```

### `analysis/visualization/` - Plotting
```python
from analysis.visualization import (
    set_publication_style, create_forest_plot,    # Plotting
    bootstrap_ci, cohens_d, format_pvalue,        # Publication helpers
)
```

### RT Filtering Constants
```python
from analysis.preprocessing import (
    DEFAULT_RT_MIN,       # 100 ms; drop anticipations
    PRP_RT_MAX,           # 3000 ms; PRP task timeout
    STROOP_RT_MAX,        # 3000 ms; Stroop task timeout
    DEFAULT_SOA_SHORT,    # 150 ms; PRP short bin upper bound
    DEFAULT_SOA_LONG,     # 1200 ms; PRP long bin lower bound
)
```

## Implementation Details

### Unicode Handling (Windows)
```python
import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
```
Save CSVs with `encoding='utf-8-sig'` for Excel compatibility. Korean text is present in comments/prints.

### WCST Extra Field Parsing
The `wcst_trials.csv` `extra` column contains stringified dicts:
```python
import ast
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str): return {}
    try: return ast.literal_eval(extra_str)
    except: return {}
```

### Column Naming Inconsistencies
- `participantId` vs `participant_id` - use `ensure_participant_id()` to normalize
- Survey names may be `surveyName` or `survey`

### PRP SOA Binning
- **short**: ≤150ms
- **medium**: 300-600ms
- **long**: ≥1200ms

## Output Locations

| Category | Directory |
|----------|-----------|
| Gold Standard | `results/gold_standard/` |
| Exploratory | `results/analysis_outputs/{prp,stroop,wcst,cross_task}_suite/` |
| Other suites | `results/analysis_outputs/{suite_name}/` |

## Results Recording

분석 스크립트 실행 후 **p < 0.05 유의한 결과**가 나오면 `Results.md` (프로젝트 루트)에 기록한다.

### 기록 형식
| 날짜 | 분석명 | 결과변수 | 효과 | β/r | p-value | 효과크기 |
|------|--------|----------|------|-----|---------|----------|

### 기록 예시
| 2025-01-16 | WCST PE regression | pe_rate | UCLA × Gender | β=0.15 | p=0.025 | η²=0.04 |

## Advanced Suite Statistical Enhancements

The `analysis/advanced/` suites have been enhanced with rigorous statistical methods:

### 1. FDR Correction (Benjamini-Hochberg)
- **`mechanistic_suite.py`**: 9 tests (3 tasks × 3 ex-Gaussian parameters)
- **`sequential_dynamics_suite.py`**: Adaptive recovery outcomes
- **`clustering_suite.py`**: Post-hoc ANOVAs

### 2. Network Analysis (`latent_suite.py`)
```python
# GraphicalLASSO for regularized partial correlations
from sklearn.covariance import GraphicalLassoCV

# Network Comparison Test (NCT) for gender differences
# - 1000 permutations for global strength/edge differences
# - Bootstrap edge stability (500 iterations)
```

### 3. Exponential Recovery Fitting (`sequential_dynamics_suite.py`)
```python
# RT(t) = baseline + delta * exp(-t/tau)
# tau: recovery time constant (higher = slower recovery)
# Bootstrap SE for tau (200 iterations)
```

### 4. MANOVA Assumption Checks (`clustering_suite.py`)
- Shapiro-Wilk normality per DV per cluster
- Levene's test for homogeneity
- Box's M approximation for covariance homogeneity
- Bootstrap cluster stability (ARI)

### 5. Bayesian Analysis
- 4 chains × 2000 draws (improved from 2 × 1000)
- ROPE interval: [-0.1, 0.1] for practical equivalence

## Key Findings

- **UCLA Main Effects**: Non-significant after DASS control (all p > 0.35)
- **UCLA × Gender Interaction**: Significant for WCST PE (p = 0.025)
- **Interpretation**: Loneliness effects confound with mood; only gender-specific vulnerability is independent

## Key Libraries
pandas, numpy, scipy, statsmodels, scikit-learn, pymc, arviz, matplotlib, seaborn, firebase-admin

## Notes

- **Platform**: Windows (path separators, encoding)
- **Firebase credentials**: `serviceAccountKey.json` required but not committed
- **Trial filtering**: Always filter `timeout == False` and `rt_ms > DEFAULT_RT_MIN`
- **Archive**: Legacy scripts in `analysis/archive/legacy_advanced/` are DEPRECATED - see `analysis/archive/legacy_advanced/README.md` for migration mapping to production suites
