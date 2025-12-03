"""
Validation Analysis #1: k-Fold Cross-Validation
================================================
CRITICAL: All models control for DASS-21 subscales + age

Purpose:
--------
Compare out-of-sample predictive performance between:
- Base model: DASS + Age + Gender (no UCLA)
- Full model: DASS + Age + Gender + UCLA + UCLA×Gender

Key Question:
-------------
Does the UCLA × Gender interaction term provide genuine predictive utility
beyond in-sample statistical significance?

Outcomes Tested:
----------------
1. WCST perseverative error rate (PRIMARY - significant interaction found)
2. Ex-Gaussian tau (incongruent Stroop trials)
3. PRP bottleneck effect
4. Stroop interference

Method:
-------
- 5-fold cross-validation (shuffle=True, random_state=42)
- Metrics: RMSE, R², MAE (mean ± SD across folds)
- Statistical test: Paired t-test for ΔMetric (Full - Base)
- Interpretation: Negative ΔRMSE/ΔMAE = Full model better

Author: Research Team
Date: 2025-01-17
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
from analysis.utils.data_loader_utils import load_master_dataset
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/validation_analyses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42

print("=" * 80)
print("VALIDATION ANALYSIS #1: k-FOLD CROSS-VALIDATION")
print("=" * 80)
print(f"Configuration:")
print(f"  - Folds: {N_FOLDS}")
print(f"  - Random seed: {RANDOM_STATE}")
print(f"  - Metrics: RMSE, R², MAE")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")

# Load master dataset
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
master['gender_male'] = (master['gender'] == 'male').astype(int)
print(f"  Loaded master dataset: N = {len(master)} participants")

# Load ex-Gaussian tau parameters
exgauss_path = RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/exgaussian_parameters.csv"
if exgauss_path.exists():
    exgauss = pd.read_csv(exgauss_path, encoding='utf-8-sig')
    master = master.merge(exgauss[['participant_id', 'incongruent_tau']],
                          on='participant_id', how='left')
    master = master.rename(columns={'incongruent_tau': 'tau'})
    print(f"  Merged ex-Gaussian tau: N with tau = {master['tau'].notna().sum()}")
else:
    print(f"  WARNING: Ex-Gaussian parameters not found at {exgauss_path}")
    master['tau'] = np.nan

# Normalize column names
if 'participantId' in master.columns:
    master = master.rename(columns={'participantId': 'participant_id'})

# Rename WCST PE if needed
if 'pe_rate' in master.columns:
    master = master.rename(columns={'pe_rate': 'pe_rate'})

# Gender mapping
gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Standardize predictors
print(f"  Standardizing predictors...")
scaler = StandardScaler()

# Check which columns exist
required_cols = ['age', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
missing_cols = [col for col in required_cols if col not in master.columns]
if missing_cols:
    print(f"  ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)

master['z_age'] = scaler.fit_transform(master[['age']])
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_dass_dep'] = scaler.fit_transform(master[['dass_depression']])
master['z_dass_anx'] = scaler.fit_transform(master[['dass_anxiety']])
master['z_dass_str'] = scaler.fit_transform(master[['dass_stress']])

print(f"  Standardization complete.")

# ============================================================================
# DEFINE OUTCOMES
# ============================================================================
outcomes = []

# Check which DVs are available
if 'pe_rate' in master.columns:
    outcomes.append(('pe_rate', 'WCST Perseverative Error Rate'))
if 'tau' in master.columns and master['tau'].notna().sum() > 0:
    outcomes.append(('tau', 'Ex-Gaussian Tau (Stroop Incongruent)'))
if 'prp_bottleneck' in master.columns:
    outcomes.append(('prp_bottleneck', 'PRP Bottleneck Effect'))
if 'stroop_interference' in master.columns:
    outcomes.append(('stroop_interference', 'Stroop Interference'))

print(f"\n[2/4] Outcomes to test: {len(outcomes)}")
for i, (col, label) in enumerate(outcomes, 1):
    n_valid = master[col].notna().sum()
    print(f"  {i}. {label} (N = {n_valid})")

if len(outcomes) == 0:
    print("ERROR: No valid outcome variables found!")
    sys.exit(1)

# ============================================================================
# CROSS-VALIDATION FUNCTION
# ============================================================================
def run_cv_comparison(df, outcome, n_folds=5):
    """
    Run k-fold CV comparing Base vs Full model.

    Returns:
    --------
    fold_results: DataFrame with per-fold metrics
    summary: Dict with mean metrics and statistical tests
    """
    # Drop missing values
    analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                     'z_ucla', 'gender_male', outcome]
    df_clean = df[analysis_cols].dropna()

    if len(df_clean) < 20:
        print(f"    WARNING: Only {len(df_clean)} complete cases. Skipping.")
        return None, None

    # Prepare data
    y = df_clean[outcome].values
    X_base = df_clean[['z_age', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str']].values
    X_full = df_clean[['z_age', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                        'z_ucla']].copy()
    # Add interaction term
    X_full['ucla_x_gender'] = df_clean['z_ucla'] * df_clean['gender_male']
    X_full = X_full.values

    # Initialize CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_base)):
        # Split data
        df_train = df_clean.iloc[train_idx]
        df_test = df_clean.iloc[test_idx]

        # Base model (no UCLA)
        formula_base = f"{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str"
        model_base = smf.ols(formula_base, data=df_train).fit()
        pred_base = model_base.predict(df_test)

        # Full model (with UCLA × Gender)
        formula_full = f"{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
        model_full = smf.ols(formula_full, data=df_train).fit()
        pred_full = model_full.predict(df_test)

        # Compute metrics
        y_test = df_test[outcome].values

        # Base metrics
        rmse_base = np.sqrt(mean_squared_error(y_test, pred_base))
        r2_base = r2_score(y_test, pred_base)
        mae_base = mean_absolute_error(y_test, pred_base)

        # Full metrics
        rmse_full = np.sqrt(mean_squared_error(y_test, pred_full))
        r2_full = r2_score(y_test, pred_full)
        mae_full = mean_absolute_error(y_test, pred_full)

        # Store results
        fold_results.append({
            'fold': fold_idx + 1,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'base_rmse': rmse_base,
            'base_r2': r2_base,
            'base_mae': mae_base,
            'full_rmse': rmse_full,
            'full_r2': r2_full,
            'full_mae': mae_full,
            'delta_rmse': rmse_full - rmse_base,  # Negative = Full better
            'delta_r2': r2_full - r2_base,        # Positive = Full better
            'delta_mae': mae_full - mae_base      # Negative = Full better
        })

    fold_df = pd.DataFrame(fold_results)

    # Compute summary statistics
    summary = {
        'n_total': len(df_clean),
        'n_folds': n_folds,
        'base_rmse_mean': fold_df['base_rmse'].mean(),
        'base_rmse_sd': fold_df['base_rmse'].std(),
        'base_r2_mean': fold_df['base_r2'].mean(),
        'base_r2_sd': fold_df['base_r2'].std(),
        'base_mae_mean': fold_df['base_mae'].mean(),
        'base_mae_sd': fold_df['base_mae'].std(),
        'full_rmse_mean': fold_df['full_rmse'].mean(),
        'full_rmse_sd': fold_df['full_rmse'].std(),
        'full_r2_mean': fold_df['full_r2'].mean(),
        'full_r2_sd': fold_df['full_r2'].std(),
        'full_mae_mean': fold_df['full_mae'].mean(),
        'full_mae_sd': fold_df['full_mae'].std(),
        'delta_rmse_mean': fold_df['delta_rmse'].mean(),
        'delta_rmse_sd': fold_df['delta_rmse'].std(),
        'delta_r2_mean': fold_df['delta_r2'].mean(),
        'delta_r2_sd': fold_df['delta_r2'].std(),
        'delta_mae_mean': fold_df['delta_mae'].mean(),
        'delta_mae_sd': fold_df['delta_mae'].std()
    }

    # Paired t-tests (is delta significantly different from 0?)
    t_rmse, p_rmse = stats.ttest_1samp(fold_df['delta_rmse'], 0)
    t_r2, p_r2 = stats.ttest_1samp(fold_df['delta_r2'], 0)
    t_mae, p_mae = stats.ttest_1samp(fold_df['delta_mae'], 0)

    summary['delta_rmse_t'] = t_rmse
    summary['delta_rmse_p'] = p_rmse
    summary['delta_r2_t'] = t_r2
    summary['delta_r2_p'] = p_r2
    summary['delta_mae_t'] = t_mae
    summary['delta_mae_p'] = p_mae

    return fold_df, summary

# ============================================================================
# RUN CV FOR ALL OUTCOMES
# ============================================================================
print(f"\n[3/4] Running {N_FOLDS}-fold cross-validation...")

all_fold_results = []
all_summaries = []

for outcome_col, outcome_label in outcomes:
    print(f"\n  Analyzing: {outcome_label}")
    print(f"  " + "-" * 60)

    fold_df, summary = run_cv_comparison(master, outcome_col, n_folds=N_FOLDS)

    if fold_df is None:
        continue

    # Add outcome info to results
    fold_df['outcome'] = outcome_col
    fold_df['outcome_label'] = outcome_label
    all_fold_results.append(fold_df)

    summary['outcome'] = outcome_col
    summary['outcome_label'] = outcome_label
    all_summaries.append(summary)

    # Print summary
    print(f"    N = {summary['n_total']}")
    print(f"    Base Model:")
    print(f"      RMSE: {summary['base_rmse_mean']:.4f} ± {summary['base_rmse_sd']:.4f}")
    print(f"      R²:   {summary['base_r2_mean']:.4f} ± {summary['base_r2_sd']:.4f}")
    print(f"      MAE:  {summary['base_mae_mean']:.4f} ± {summary['base_mae_sd']:.4f}")
    print(f"    Full Model (+ UCLA × Gender):")
    print(f"      RMSE: {summary['full_rmse_mean']:.4f} ± {summary['full_rmse_sd']:.4f}")
    print(f"      R²:   {summary['full_r2_mean']:.4f} ± {summary['full_r2_sd']:.4f}")
    print(f"      MAE:  {summary['full_mae_mean']:.4f} ± {summary['full_mae_sd']:.4f}")
    print(f"    Delta (Full - Base):")
    print(f"      ΔRMSE: {summary['delta_rmse_mean']:.4f} ± {summary['delta_rmse_sd']:.4f}, t={summary['delta_rmse_t']:.3f}, p={summary['delta_rmse_p']:.4f}")
    print(f"      ΔR²:   {summary['delta_r2_mean']:.4f} ± {summary['delta_r2_sd']:.4f}, t={summary['delta_r2_t']:.3f}, p={summary['delta_r2_p']:.4f}")
    print(f"      ΔMAE:  {summary['delta_mae_mean']:.4f} ± {summary['delta_mae_sd']:.4f}, t={summary['delta_mae_t']:.3f}, p={summary['delta_mae_p']:.4f}")

    # Interpretation
    if summary['delta_rmse_p'] < 0.05:
        if summary['delta_rmse_mean'] < 0:
            print(f"    ✓ Full model has significantly BETTER predictive performance (lower RMSE)")
        else:
            print(f"    ✗ Full model has significantly WORSE predictive performance (higher RMSE)")
    else:
        print(f"    ~ No significant difference in predictive performance")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\n[4/4] Saving results...")

if len(all_fold_results) > 0:
    # Fold-level results
    fold_results_df = pd.concat(all_fold_results, ignore_index=True)
    fold_results_path = OUTPUT_DIR / "cv_fold_results.csv"
    fold_results_df.to_csv(fold_results_path, index=False, encoding='utf-8-sig')
    print(f"  Saved fold-level results: {fold_results_path}")

    # Summary statistics
    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / "cv_performance_comparison.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  Saved summary statistics: {summary_path}")

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey Findings:")
    print(f"-" * 80)

    for summary in all_summaries:
        print(f"\n{summary['outcome_label']}:")
        if summary['delta_rmse_p'] < 0.05 and summary['delta_rmse_mean'] < 0:
            print(f"  ✓ Full model provides significant predictive improvement")
            print(f"    ΔRMSE = {summary['delta_rmse_mean']:.4f} (p = {summary['delta_rmse_p']:.4f})")
            print(f"    ΔR² = {summary['delta_r2_mean']:.4f} (p = {summary['delta_r2_p']:.4f})")
        elif summary['delta_rmse_p'] < 0.05 and summary['delta_rmse_mean'] > 0:
            print(f"  ✗ Full model has WORSE predictive performance")
            print(f"    ΔRMSE = {summary['delta_rmse_mean']:.4f} (p = {summary['delta_rmse_p']:.4f})")
        else:
            print(f"  ~ No significant predictive difference")
            print(f"    ΔRMSE = {summary['delta_rmse_mean']:.4f} (p = {summary['delta_rmse_p']:.4f})")

    print("\n" + "=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

else:
    print("  ERROR: No valid results to save!")
    sys.exit(1)
