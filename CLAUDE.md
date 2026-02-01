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
2. **Statistical Analysis**: Static package (`static/`)

## Data Flow

```
data/raw/
       ->
python -m static.preprocessing --build all
       ->
data/complete_overall/ (N ~ 212)
       ->
python -m static.* -> outputs/
```



## Essential Commands

```bash
# Activate venv (Windows)
.\.venv\Scripts\activate

# Preprocessing: Build overall dataset
python -m static.preprocessing --build overall
python -m static.preprocessing --build all
python -m static.preprocessing --list

# Core Analyses
python -m static.analysis.descriptive_statistics
python -m static.analysis.correlation_analysis
python -m static.analysis.hierarchical_regression

# Advanced Analyses
python -m static.advanced_analysis.mediation_suite
python -m static.path_analysis.path_depression --task overall

# Validity & Reliability
python -m static.validity_reliability.reliability_suite
python -m static.validity_reliability.validity_suite

# Gender Analysis
python -m static.gender_analysis --all
python -m static.gender_analysis -a male_vulnerability

# Run specific WCST phase analysis scripts
python static/wcst_phase/compute_wcst_pre_switch_features.py
python static/wcst_phase/generate_wcst_category_cycle_rt_trend.py
```

## Key Data Files

| Directory | Contents |
|-----------|----------|
| `data/raw/` | Raw exported data (N=251) |
| `data/complete_overall/` | All tasks complete (N ~ 212) |
| `outputs/` | Generated results |

**CSV files in each directory:**
- `1_participants_info.csv` - Demographics
- `2_surveys_results.csv` - UCLA & DASS-21
- `3_cognitive_tests_summary.csv` - Aggregate metrics
- `4b_wcst_trials.csv` / `4c_stroop_trials.csv` - Trial-level data

## Package Architecture

```
static/
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
├── wcst_phase/            # WCST phase-related scripts & logic
├── stroop_lmm/            # Stroop linear mixed model scripts
├── figures_tables/            # Figure & table generation scripts
├── advanced_analysis/     # Mediation, Bayesian
└── gender_analysis/       # Gender stratified analyses
```

## ⚠️ CRITICAL: DASS-21 Covariate Control

**ALL confirmatory analyses testing UCLA effects MUST control for DASS-21 subscales.**

UCLA and DASS correlate r ~ 0.5-0.7. Without control, "loneliness effects" confound with general distress.

### Standard Regression Formula
```python
from static.analysis.utils import get_analysis_data, run_ucla_regression

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
from static.preprocessing import (
    load_master_dataset,      # Main entry point
    get_results_dir,          # Output paths
    VALID_TASKS,              # {'overall'}
)

# Load task-specific master dataset
df_overall = load_master_dataset(task='overall') # N ~ 212

# For analysis scripts, use utils wrapper:
from static.analysis.utils import get_analysis_data
df = get_analysis_data("overall")  # Includes QC filter
```

## RT Filtering Constants

```python
from static.preprocessing import (
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
| Publication results | `outputs/` |
| Analysis outputs | `outputs/stats/analysis/{task}/` |
| Paper tables | `outputs/tables/` |
| Paper figures | `outputs/figures/` |

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
