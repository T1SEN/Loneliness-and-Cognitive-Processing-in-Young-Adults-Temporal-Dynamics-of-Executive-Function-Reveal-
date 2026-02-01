# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## User Preferences

- **Explanations**: Always in Korean (한국어)
- **Code**: Always in English
- **Regression**: Use OLS (not HC3), always control for DASS
- **PRP task**: Analyzed but will NOT be included in the paper

## Project Overview

Research data analysis pipeline for a psychology study examining loneliness (UCLA Loneliness Scale) and executive function (EF) across three cognitive tasks: Stroop (interference control), WCST (set-shifting), and PRP (dual-task coordination).

**Components:**
1. **Data Collection**: Flutter mobile app (`lib/`)
2. **Data Export**: Firebase → CSV (`export_alldata.py`)
3. **Statistical Analysis**: Publication package (`publication/`)

## Data Flow

```
Firebase (Firestore) → export_alldata.py → publication/data/raw/
                                              ↓
                       python -m publication.preprocessing --build all
                                              ↓
                       ├── publication/data/complete_prp/     (N ~ 195)
                       ├── publication/data/complete_overall/ (N ~ 212)
                                              ↓
                       python -m publication.* → results/publication/
```

## Essential Commands

```bash
# Activate venv (Windows)
.\venv\Scripts\activate

# Preprocessing: Build overall dataset
python -m publication.preprocessing --build overall
python -m publication.preprocessing --build all
python -m publication.preprocessing --list

# Core Analyses
python -m publication.analysis.descriptive_statistics
python -m publication.analysis.correlation_analysis
python -m publication.analysis.hierarchical_regression

# Advanced Analyses
python -m publication.advanced_analysis.mediation_suite
python -m publication.path_analysis.path_depression --task overall

# Validity & Reliability
python -m publication.validity_reliability.reliability_suite
python -m publication.validity_reliability.validity_suite

# Gender Analysis
python -m publication.gender_analysis --all
python -m publication.gender_analysis -a male_vulnerability

# Export data from Firebase
PYTHONIOENCODING=utf-8 .\venv\Scripts\python.exe export_alldata.py

# Run specific WCST phase analysis scripts
python publication/4_wcst_phase/run_wcst_segment_regressions.py
python publication/4_wcst_phase/run_wcst_segment_error_regressions.py
python publication/4_wcst_phase/run_wcst_segment_delta_regressions.py
python publication/4_wcst_phase/run_wcst_post_shift_error_log_ols.py
python publication/4_wcst_phase/run_wcst_post_error_log_ols.py
python publication/4_wcst_phase/run_wcst_post_error_rt_ols_overall.py
python publication/4_wcst_phase/compute_wcst_switching_features.py
python publication/4_wcst_phase/compute_wcst_pre_switch_features.py
python publication/4_wcst_phase/generate_wcst_segment_rt_trend.py
python publication/4_wcst_phase/generate_wcst_category_segment_rt_trend.py
python publication/4_wcst_phase/generate_wcst_category_cycle_rt_trend.py
python publication/4_wcst_phase/plot_wcst_segment_regression_ols.py
python publication/4_wcst_phase/plot_wcst_segment_regression_ols_timeseries.py
```

## Key Data Files

| Directory | Contents |
|-----------|----------|
| `publication/data/raw/` | Raw exported data (N=251) |
| `publication/data/complete_overall/` | All tasks complete (N ~ 212) |
| `publication/data/outputs/` | Generated results |

**CSV files in each directory:**
- `1_participants_info.csv` - Demographics
- `2_surveys_results.csv` - UCLA & DASS-21
- `3_cognitive_tests_summary.csv` - Aggregate metrics
- `4b_wcst_trials.csv` / `4c_stroop_trials.csv` - Trial-level data

## Package Architecture

```
publication/
├── preprocessing/          # Data loading & feature extraction
│   ├── __init__.py        # Main exports (load_master_dataset, etc.)
│   ├── constants.py       # RT thresholds, QC criteria
│   ├── overall/            # Overall-only task module
│   └── standardization.py # z-scoring utilities
├── analysis/              # Core statistical analyses
│   ├── utils.py           # Shared: get_analysis_data(), run_ucla_regression()
│   ├── hierarchical_regression.py
│   └── correlation_analysis.py
├── scripts/               # Ad-hoc analysis scripts
├── 4_wcst_phase/            # WCST phase-related scripts & logic
├── 3_stroop_lmm/            # Stroop linear mixed model scripts
├── advanced_analysis/     # Mediation, Bayesian
└── gender_analysis/       # Gender stratified analyses
```

## ⚠️ CRITICAL: DASS-21 Covariate Control

**ALL confirmatory analyses testing UCLA effects MUST control for DASS-21 subscales.**

UCLA and DASS correlate r ~ 0.5-0.7. Without control, "loneliness effects" confound with general distress.

### Standard Regression Formula
```python
from publication.analysis.utils import get_analysis_data, run_ucla_regression

df = get_analysis_data("overall")

# run_ucla_regression uses this formula internally:
# {outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)

result = run_ucla_regression(df, "wcst_pes", cov_type="nonrobust")  # Use OLS, not HC3
```

### Manual OLS with DASS Control
```python
import statsmodels.formula.api as smf

formula = f"{outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
model = smf.ols(formula, data=df).fit(cov_type="nonrobust")  # OLS, not HC3
```

## Data Loading API

```python
from publication.preprocessing import (
    load_master_dataset,      # Main entry point
    get_results_dir,          # Output paths
    VALID_TASKS,              # {'overall'}
)

# Load task-specific master dataset
df_overall = load_master_dataset(task='overall') # N ~ 212

# For analysis scripts, use utils wrapper:
from publication.analysis.utils import get_analysis_data
df = get_analysis_data("overall")  # Includes QC filter
```

## RT Filtering Constants

```python
from publication.preprocessing import (
    DEFAULT_RT_MIN,           # 100 ms (drop anticipations)
    WCST_RT_MIN,              # 100 ms
    STROOP_RT_MAX,            # 3000 ms
    WCST_MIN_TRIALS,          # 60 minimum trials
    WCST_MIN_MEDIAN_RT,       # 300 ms (random clicking threshold)
    WCST_MAX_SINGLE_CHOICE,   # 0.85 max single card ratio
)
```

## Implementation Notes

### Unicode Handling (Windows)
```python
import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
```
Save CSVs with `encoding='utf-8-sig'` for Excel compatibility.

### WCST Extra Field Parsing
```python
import ast
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str): return {}
    try: return ast.literal_eval(extra_str)
    except: return {}
```

### Column Naming
- `participantId` vs `participant_id` - use `ensure_participant_id()` to normalize
- Survey names: `surveyName` or `survey`

## Output Locations

| Category | Directory |
|----------|-----------|
| Publication results | `results/publication/` |
| Analysis outputs | `publication/data/outputs/analysis/{task}/` |
| Paper tables | `publication/data/outputs/paper_tables/` |

## Results Recording

Record significant results (p < 0.05) in `Results.md`:

| Date | Analysis | Outcome | Effect | Beta | p-value | Effect size |
|------|----------|---------|--------|------|---------|-------------|

## Key Findings Summary

- **UCLA Main Effects After DASS Control**: Limited (3 variables significant)
- **Male-Specific Vulnerability**: UCLA correlates with Ex-Gaussian parameters and HMM lapse in males only
- **WCST Segment Effects**: UCLA predicts RT in confirmation phase

## Key Libraries

pandas, numpy, scipy, statsmodels, scikit-learn, pymc, arviz, matplotlib, seaborn, firebase-admin

## Flutter App (`lib/`)

Data collection app structure:
- Tasks: `pages/ncft/{wcst,stroop,prp}_page.dart`
- Surveys: `pages/survey/{ucla,dass}_page.dart`
- Flow: `pages/intro/test_sequencer_page.dart`

Data flows to Firebase Firestore, exported via `export_alldata.py`.
