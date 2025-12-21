# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research data analysis pipeline for a psychology study examining the relationship between loneliness (UCLA Loneliness Scale) and executive function (EF) across three cognitive tasks: Stroop (interference control), WCST (set-shifting), and PRP (dual-task coordination).

**Three main components:**
1. **Data Collection**: Flutter mobile app for cognitive tasks (`lib/`)
2. **Data Export**: Firebase → CSV extraction (`export_alldata.py`)
3. **Statistical Analysis**: Suite-based analysis pipeline (`analysis/`)

## Data Flow

```
Firebase (Firestore) → export_alldata.py → publication/data/raw/
                                              ↓
                       python -m publication.preprocessing --build all
                                              ↓
                       ├── publication/data/complete_stroop/  (N ~ 200)
                       ├── publication/data/complete_prp/     (N ~ 195)
                       └── publication/data/complete_wcst/    (N ~ 190)
                                              ↓
                       python -m analysis → results/gold_standard/ & results/analysis_outputs/
                       python -m publication.* → results/publication/
```

### Key Data Files (`publication/data/`)
| Directory | Contents |
|-----------|----------|
| `raw/` | Raw exported data (all participants, N=251) |
| `complete_stroop/` | Stroop + 설문 완료자 (N ~ 200) |
| `complete_prp/` | PRP + 설문 완료자 (N ~ 195) |
| `complete_wcst/` | WCST + 설문 완료자 (N ~ 190) |
| `complete_overall/` | 모든 과제 완료자 통합 (N ~ 180) |
| `outputs/` | Generated outputs (master_dataset.csv, analysis results) |

**Data file structure (same across raw/complete_*):**
| File | Contents |
|------|----------|
| `1_participants_info.csv` | Demographics (age, gender, education) |
| `2_surveys_results.csv` | UCLA Loneliness & DASS-21 responses |
| `3_cognitive_tests_summary.csv` | Aggregate task metrics |
| `4a_prp_trials.csv` | Trial-level PRP data |
| `4b_wcst_trials.csv` | Trial-level WCST data |
| `4c_stroop_trials.csv` | Trial-level Stroop data |

Note: 각 task별 complete_* 디렉토리는 해당 task의 trial 파일만 포함합니다.

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

# Publication Package
python -m publication.basic_analysis.descriptive_statistics      # Descriptive statistics
python -m publication.basic_analysis.correlation_analysis        # Correlation analysis
python -m publication.basic_analysis.hierarchical_regression     # Hierarchical regression (DASS-controlled)

python -m publication.advanced_analysis.mediation_suite          # UCLA → DASS → EF 매개분석
python -m publication.advanced_analysis.bayesian_suite           # 베이지안 SEM

python -m publication.path_analysis.path_depression --task overall  # Path analysis (Depression)
python -m publication.path_analysis.path_anxiety --task overall     # Path analysis (Anxiety)
python -m publication.path_analysis.path_stress --task overall      # Path analysis (Stress)

python -m publication.validity_reliability.reliability_suite     # Cronbach's alpha, split-half
python -m publication.validity_reliability.validity_suite        # Factor analysis
python -m publication.validity_reliability.data_quality_suite    # Response validation

python -m publication.gender_analysis --list                     # List gender analyses
python -m publication.gender_analysis --all                      # Run all gender analyses
python -m publication.gender_analysis -a male_vulnerability      # Run specific analysis

# Machine learning
python -m analysis.ml.nested_cv --task classification --features demo_dass

# Export data from Firebase (requires serviceAccountKey.json)
PYTHONIOENCODING=utf-8 .\venv\Scripts\python.exe export_alldata.py

# Preprocessing: Build task-specific datasets
python -m publication.preprocessing --build stroop   # Stroop 완료자 데이터셋
python -m publication.preprocessing --build prp      # PRP 완료자 데이터셋
python -m publication.preprocessing --build wcst     # WCST 완료자 데이터셋
python -m publication.preprocessing --build overall  # 모든 과제 완료자 통합 데이터셋
python -m publication.preprocessing --build all      # 모든 task 데이터셋 빌드
python -m publication.preprocessing --list           # 데이터셋 현황 조회
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
│   ├── ddm_suite.py            # Drift-diffusion modeling
│   ├── intervention_subgroups_suite.py  # High-risk subgroup identification
│   ├── male_vulnerability_suite.py      # Gender-specific effects
│   └── ...                     # ~45 suites total; see run.py SUITE_REGISTRY
├── ml/                     # Machine learning pipelines
└── archive/                # Legacy scripts (DEPRECATED - see README.md)
```

## Publication Package Structure

Publication analysis package layout (task-specific preprocessing).

```
publication/
|-- data/
|   |-- raw/
|   |-- complete_prp/
|   |-- complete_stroop/
|   |-- complete_wcst/
|   |-- outputs/
|   `-- export_alldata.py
|-- preprocessing/
|   |-- __init__.py
|   |-- __main__.py
|   |-- cli.py
|   |-- constants.py
|   |-- core.py
|   |-- surveys.py
|   |-- datasets.py
|   |-- standardization.py
|   |-- prp/
|   |   |-- __init__.py
|   |   |-- loaders.py
|   |   |-- filters.py
|   |   |-- features.py
|   |   `-- dataset.py
|   |-- stroop/
|   |   |-- __init__.py
|   |   |-- loaders.py
|   |   |-- filters.py
|   |   |-- features.py
|   |   `-- dataset.py
|   |-- wcst/
|   |   |-- __init__.py
|   |   |-- loaders.py
|   |   |-- filters.py
|   |   |-- features.py
|   |   `-- dataset.py
|   `-- overall/
|       |-- __init__.py
|       `-- dataset.py
|-- basic_analysis/
|-- advanced_analysis/
|-- path_analysis/
|-- validity_reliability/
`-- gender_analysis/
```

**Output directory:** `results/publication/{basic_analysis,advanced_analysis,validity_reliability,gender_analysis}/`

### 사용법
```python
from publication.gender_analysis import (
    load_gender_data,                  # 성별 변수 준비된 마스터 데이터
    run_gender_stratified_regression,  # 성별별 회귀분석
    run_all_gender_interactions,       # UCLA × Gender 상호작용 검정
    fisher_z_test,                     # 성별 간 상관 비교
)

from publication.advanced_analysis import (
    bootstrap_mediation,               # Bootstrap 매개분석
    sobel_test,                        # Sobel 검정
    fit_path_model_semopy,             # SEM 경로모형
)
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

### RT Filtering & QC Constants
```python
from publication.preprocessing import (
    # RT Filtering
    DEFAULT_RT_MIN,       # 100 ms; drop anticipations
    PRP_RT_MAX,           # 3000 ms; PRP task timeout
    STROOP_RT_MAX,        # 3000 ms; Stroop task timeout
    WCST_RT_MIN,          # 100 ms; WCST anticipation cutoff
    # PRP SOA Binning
    DEFAULT_SOA_SHORT,    # 150 ms; short bin upper bound
    DEFAULT_SOA_LONG,     # 1200 ms; long bin lower bound
    # WCST QC Thresholds
    WCST_MIN_TRIALS,      # 60; minimum trials
    WCST_MIN_MEDIAN_RT,   # 300 ms; random clicking threshold
    WCST_MAX_SINGLE_CHOICE,  # 0.85; max single card choice ratio
    # QC Criteria Classes
    PRPQCCriteria, StroopQCCriteria, WCSTQCCriteria, SurveyQCCriteria,
)
```

### `publication/preprocessing/` - Publication Data Loading
```python
from publication.preprocessing import (
    # Task-specific datasets (recommended)
    load_master_dataset,  # task='stroop', 'prp', 'wcst', 'overall'
    # Task builders
    build_prp_dataset, build_stroop_dataset, build_wcst_dataset,
    build_overall_dataset, build_all_datasets,
    # Trial loaders
    load_prp_trials, load_stroop_trials, load_wcst_trials,
    # Trial-derived features
    derive_prp_features, derive_stroop_features, derive_wcst_features,
    derive_overall_features,
    # Mechanism features (Ex-Gaussian, HMM, RL)
    compute_prp_exgaussian_features, compute_stroop_exgaussian_features,
    compute_wcst_hmm_features, compute_wcst_rl_features,
    load_or_compute_prp_mechanism_features,
    load_or_compute_stroop_mechanism_features,
    load_or_compute_wcst_mechanism_features,
    # Path utilities
    get_results_dir,
    RAW_DIR, DATA_DIR, VALID_TASKS,
)

# Task-specific dataset usage:
df_stroop = load_master_dataset(task='stroop')   # N ~ 200
df_prp = load_master_dataset(task='prp')         # N ~ 195
df_wcst = load_master_dataset(task='wcst')       # N ~ 190
df_overall = load_master_dataset(task='overall') # N ~ 180 (all tasks complete)
```
Note: Each task has its own complete dataset. Each complete_* contains only its task trial file. The `overall` dataset requires all three tasks completed.

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
| **Publication** | `results/publication/{basic_analysis,advanced_analysis,validity_reliability,gender_analysis}/` |
| Publication outputs | `publication/data/outputs/{basic_analysis,mechanism_analysis,validity_reliability,network_analysis}/` |

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

## Key Findings (N=185 Quality-Controlled)

- **UCLA Main Effects After DASS Control**: 3 variables significant
  - PRP Delay Effect (ΔR²=2.38%, p<.05)
  - WCST Post-Error Slowing (ΔR²=2.53%, p<.05)
  - Stroop Incongruent RT Slope (ΔR²=3.74%, p<.05)
- **UCLA × Gender Interactions**: None significant after DASS control
- **Male-Specific Vulnerability** (gender-stratified analysis):
  - PRP: UCLA correlates with Ex-Gaussian parameters (mu, sigma) in males only (r=0.26-0.31, p<.03)
  - WCST HMM: UCLA→Lapse occupancy significant in males only (β=5.36, p=.024; r=0.34, p=.004)
- **Network Analysis**: 4 edges show significant gender differences (p=.001 via permutation test)

## Key Libraries
pandas, numpy, scipy, statsmodels, scikit-learn, pymc, arviz, matplotlib, seaborn, firebase-admin

## Flutter Data Collection App (`lib/`)

The Flutter mobile app collects experimental data:
- **Tasks**: `prp_page.dart`, `stroop_page.dart`, `wcst_page.dart`
- **Surveys**: `ucla_page.dart` (UCLA Loneliness), `dass_page.dart` (DASS-21)
- **Flow**: `test_sequencer_page.dart` orchestrates task order

Data flows to Firebase Firestore, then exported via `export_alldata.py`.

## Notes

- **Platform**: Windows (path separators, encoding)
- **Firebase credentials**: `serviceAccountKey.json` required but not committed
- **Trial filtering**: Always filter `timeout == False` and `rt_ms > DEFAULT_RT_MIN`
- **Archive**: Legacy scripts in `analysis/archive/legacy_advanced/` are DEPRECATED - see `analysis/archive/legacy_advanced/README.md` for migration mapping to production suites
