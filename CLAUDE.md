# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research data analysis pipeline for a psychology/cognitive neuroscience study examining the relationship between loneliness (UCLA Loneliness Scale) and executive function (EF) performance across three cognitive tasks: Stroop (interference control), WCST (set-shifting), and PRP (dual-task coordination).

The project has two main components:
1. **Data Export**: Firebase → CSV extraction scripts
2. **Statistical Analysis**: Multiple analysis scripts covering correlation, regression, Bayesian modeling, and machine learning approaches

## Data Pipeline Architecture

### Data Flow
```
Firebase (Firestore) → export_alldata.py → results/*.csv → analysis/*.py → results/analysis_outputs/*.csv
```

### Key Data Files (in `results/`)
- `1_participants_info.csv` - Demographics (age, gender, education, etc.)
- `2_surveys_results.csv` - UCLA Loneliness & DASS-21 (depression/anxiety/stress) responses
- `3_cognitive_tests_summary.csv` - Aggregate task performance metrics
- `4a_prp_trials.csv` - Trial-level PRP (Psychological Refractory Period) data
- `4b_wcst_trials.csv` - Trial-level WCST (Wisconsin Card Sorting Test) data
- `4c_stroop_trials.csv` - Trial-level Stroop task data

### Executive Function Metrics
The analysis derives three primary EF measures:
1. **Stroop interference**: Incongruent RT - Congruent RT (inhibitory control)
2. **WCST perseverative error rate**: % trials with perseverative errors (set-shifting)
3. **PRP bottleneck effect**: T2 RT at short SOA - T2 RT at long SOA (dual-task coordination)

## Essential Commands

### Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate

# View installed packages (no requirements.txt; packages are pre-installed in venv)
.\venv\Scripts\pip.exe list
```

### Data Export
```bash
# Export all Firebase data to CSV
# Requires serviceAccountKey.json in root directory
PYTHONIOENCODING=utf-8 .\venv\Scripts\python.exe export_alldata.py

# Export only participant info (simpler version)
PYTHONIOENCODING=utf-8 .\venv\Scripts\python.exe export_data.py
```

### Running Analysis Scripts
```bash
# Run any analysis script (general pattern)
.\venv\Scripts\python.exe analysis\<script_name>.py

# PRIMARY: DASS-controlled confirmatory analysis (gold standard)
.\venv\Scripts\python.exe analysis\master_dass_controlled_analysis.py

# Machine learning nested CV with hyperparameter tuning
.\venv\Scripts\python.exe analysis\ml_nested_tuned.py --task classification --features demo_dass
.\venv\Scripts\python.exe analysis\ml_nested_tuned.py --task regression --features ef_demo_dass

# Generate trial-level features (CV, post-error slowing, RT slopes)
.\venv\Scripts\python.exe analysis\derive_trial_features.py
```

## Critical Implementation Details

### Unicode Handling (Windows-specific)
Scripts use `PYTHONIOENCODING=utf-8` and include:
```python
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
```
Korean text is present in comments/prints. Always save CSVs with `encoding='utf-8-sig'` for Excel compatibility.

### WCST Extra Field Parsing
The `wcst_trials.csv` contains an `extra` column with stringified dicts that must be parsed:
```python
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}
```
This extracts `isPE` (is perseverative error) flags.

### Column Naming Inconsistencies
- Some CSVs use `participantId`, others use `participant_id`
- Survey names may be `surveyName` or stored as lowercase `survey`
- Always normalize column names: `.rename(columns={'participantId': 'participant_id'})` or `.columns.str.lower()`

### PRP SOA Binning
SOA (Stimulus Onset Asynchrony) values are binned consistently across scripts:
- **short**: ≤150ms
- **medium**: 300-600ms
- **long**: ≥1200ms

### Master Dataset Construction
Most analysis scripts merge data using this pattern:
1. Load participants, surveys, cognitive summaries
2. Compute task-specific metrics from trial data
3. Merge on `participant_id`
4. Drop rows with missing values in key columns
5. Check for minimum N (usually ≥20) before proceeding

## Statistical Analysis Approach

The codebase implements a multi-method validation strategy:

1. **Frequentist**: Pearson correlations, hierarchical regression (controlling for DASS-21)
2. **Bayesian**: PyMC models with ROPE (Region of Practical Equivalence) + LOO-CV
3. **Machine Learning**: Nested CV with GridSearchCV, permutation importance, partial dependence plots
4. **Dimensionality Reduction**: PCA to extract "meta-control" factor across three EF tasks

Key hypothesis: UCLA loneliness predicts EF impairment beyond mood/anxiety (DASS-21 as covariates).

### ⚠️ CRITICAL: DASS-21 Covariate Control Requirement

**ALL confirmatory analyses testing UCLA effects MUST control for DASS-21 subscales.**

**Why this is mandatory:**
1. UCLA loneliness and DASS (depression/anxiety/stress) are highly correlated (r ~ 0.5-0.7)
2. Without DASS control, "loneliness effects" confound with general emotional distress
3. Master analysis (2025-01-16) showed: ALL UCLA main effects disappear when DASS is controlled
4. Only UCLA × Gender interactions survive DASS control

**Implementation:**
```python
# WRONG - No DASS control:
model = smf.ols("pe_rate ~ z_ucla * C(gender_male)", data=df).fit()

# CORRECT - With DASS control:
model = smf.ols("pe_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                data=df).fit()
```

**Required covariates in ALL regression models:**
- `dass_depression` (or `z_dass_dep` if standardized)
- `dass_anxiety` (or `z_dass_anx` if standardized)
- `dass_stress` (or `z_dass_str` if standardized)
- `age` (or `z_age` if standardized)

**Exception:** Mediation analyses where DASS is the mediator (not a covariate).

**Reference analysis:**
- Script: `analysis/master_dass_controlled_analysis.py`
- Results: `results/analysis_outputs/master_dass_controlled/`
- Report: `CORRECTED_FINAL_INTERPRETATION.txt`

**Key finding:**
- UCLA main effects: ALL p > 0.05 after DASS control (no pure loneliness effect)
- UCLA × Gender interaction: p = 0.025 for WCST PE (survives DASS control)
- Conclusion: Only gender-specific vulnerability is independent of mood

### Analysis Script Classification (by DASS-21 Control Rigor)

| Category | Scripts | Notes |
|----------|---------|-------|
| ⭐ **Gold Standard** | `master_dass_controlled_analysis.py` (PRIMARY), `loneliness_exec_models.py`, `trial_level_mixed_effects.py`, `prp_comprehensive_dass_controlled.py` | Full DASS control; citable in publications |
| 🔬 **Mediation** | `dass_mediation_bootstrapped.py`, `mechanism_mediation_analysis.py` | DASS is mediator, not covariate (appropriate) |
| ⚠️ **Exploratory** | `extreme_group_analysis.py` | No DASS control; hypothesis generation only |
| 🧰 **Utility** | `data_loader_utils.py`, `derive_trial_features.py`, `ml_nested_tuned.py` | No hypothesis testing |

**Formula template for confirmatory analyses:**
```python
smf.ols("y ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=df)
```

## Output Location

All analysis outputs go to `results/analysis_outputs/`:
- CSV files with metrics, coefficients, p-values, predictions
- PNG files for PDPs (partial dependence plots)
- `analysis_log.txt` for some verbose outputs

## Shared Utility Modules

### `analysis/utils/data_loader_utils.py`
Central data loading module (note: `analysis/data_loader_utils.py` is a backward-compat shim that re-exports from here):
- `load_master_dataset()` - Builds/caches unified dataset (cached at `results/analysis_outputs/master_dataset.parquet`)
- `load_participants()`, `load_ucla_scores()`, `load_dass_scores()` - Individual loaders
- `load_wcst_summary()`, `load_prp_summary()`, `load_stroop_summary()` - Task metrics
- `ensure_participant_id()` - Normalizes participant ID column names
- `normalize_gender_series()` - Maps Korean/English gender to 'male'/'female'

**RT Filtering Constants** (use these consistently):
```python
DEFAULT_RT_MIN = 100      # ms; drop anticipations
PRP_RT_MAX = 3000         # ms; PRP task timeout
STROOP_RT_MAX = 3000      # ms; Stroop task timeout
DEFAULT_RT_MAX = 5000     # ms; legacy upper bound (WCST etc.)
DEFAULT_SOA_SHORT = 150   # ms; PRP short bin upper bound
DEFAULT_SOA_LONG = 1200   # ms; PRP long bin lower bound
```

### `analysis/utils/trial_data_loader.py`
Trial-level data loaders with standardized preprocessing (cached at `results/analysis_outputs/trial_cache/`):
- `load_prp_trials()` - PRP trial data with RT filtering and SOA binning
- `load_stroop_trials()` - Stroop trial data with congruency flags
- `load_wcst_trials()` - WCST trial data with perseverative error parsing

### `analysis/utils/publication_helpers.py`
Helper functions for publication-quality outputs:
- `bootstrap_ci()` - Bootstrap confidence intervals
- `cohens_d()`, `eta_squared()` - Effect size calculations
- `format_pvalue()`, `format_ci()` - APA-style formatting
- `set_publication_style()` - Configure matplotlib for publication figures

## Key Libraries (pre-installed in venv)
- **Data**: pandas, numpy
- **Stats**: scipy, statsmodels
- **ML**: scikit-learn
- **Bayesian**: pymc, pytensor, arviz
- **Viz**: matplotlib, seaborn
- **Firebase**: firebase-admin, google-cloud-firestore

## Common Development Patterns

### Adding New Analysis Scripts
1. Load base CSVs from `RESULTS_DIR = Path("results")`
2. Create output directory: `output_dir.mkdir(exist_ok=True)`
3. Normalize participant IDs across dataframes
4. Handle missing values explicitly (`.dropna()` or imputation)
5. Save outputs to `results/analysis_outputs/` with descriptive filenames
6. Use `encoding='utf-8-sig'` for CSV exports

### Debugging Data Issues
```bash
# Check row counts for all CSVs
.\venv\Scripts\python.exe -c "import pandas as pd; import os; files = sorted([f for f in os.listdir('results') if f.endswith('.csv')]); [print(f'{f}: {len(pd.read_csv(os.path.join(\"results\", f)))} rows') for f in files]"
```

## Important Notes

- **Firebase credentials**: `serviceAccountKey.json` must be present but is NOT committed to git
- **Platform**: Developed on Windows (note path separators and encoding issues)
- **Language**: Mixed Korean/English comments; analysis outputs are primarily English
- **Null handling**: Both `np.nan` and `pd.NA` appear; be consistent within each script
- **Trial filtering**: Always filter timeouts (`timeout == False` or `t2_timeout == False`) and invalid RTs (`rt_ms > 0`) before analysis
