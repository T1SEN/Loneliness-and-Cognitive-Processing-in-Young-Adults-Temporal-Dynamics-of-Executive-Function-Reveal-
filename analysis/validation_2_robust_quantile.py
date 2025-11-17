"""
Validation Analysis #2: Robust & Quantile Regression
=====================================================
CRITICAL: All models control for DASS-21 subscales + age

Purpose:
--------
Test robustness of UCLA × Gender interaction to:
1. Outliers (Huber M-estimator)
2. Distributional tail effects (quantile regression at τ=0.25, 0.5, 0.75, 0.9)

Key Questions:
--------------
- Does the interaction survive robust regression (outlier-resistant)?
- Is the interaction stronger at specific quantiles (e.g., high performers)?

Outcomes Tested:
----------------
1. WCST perseverative error rate (PRIMARY)
2. Ex-Gaussian tau (incongruent Stroop trials)
3. PRP bottleneck effect
4. Stroop interference

Methods:
--------
Part A - Huber Robust Regression:
  - M-estimator with Huber loss (less sensitive to outliers than OLS)
  - Compare coefficients, SEs, t-values with OLS

Part B - Quantile Regression:
  - τ = 0.25 (lower quartile - better performers)
  - τ = 0.50 (median)
  - τ = 0.75 (upper quartile - worse performers)
  - τ = 0.90 (top decile - most impaired)
  - Test if UCLA × Gender effect varies across distribution

Interpretation:
---------------
- If Huber coef ≈ OLS coef: Effect is robust to outliers
- If effect strongest at τ=0.75, 0.90: "Vulnerable tail" hypothesis
  (loneliness hurts most among those already struggling)

Author: Research Team
Date: 2025-01-17
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/validation_analyses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.25, 0.50, 0.75, 0.90]

print("=" * 80)
print("VALIDATION ANALYSIS #2: ROBUST & QUANTILE REGRESSION")
print("=" * 80)
print(f"Configuration:")
print(f"  - Robust method: Huber M-estimator")
print(f"  - Quantiles: {QUANTILES}")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Load master dataset
master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
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
    print(f"  WARNING: Ex-Gaussian parameters not found")
    master['tau'] = np.nan

# Normalize column names
if 'participantId' in master.columns:
    master = master.rename(columns={'participantId': 'participant_id'})

# Rename WCST PE if needed
if 'perseverative_error_rate' in master.columns:
    master = master.rename(columns={'perseverative_error_rate': 'pe_rate'})

# Gender mapping
gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Standardize predictors
print(f"  Standardizing predictors...")
scaler = StandardScaler()

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

if 'pe_rate' in master.columns:
    outcomes.append(('pe_rate', 'WCST Perseverative Error Rate'))
if 'tau' in master.columns and master['tau'].notna().sum() > 0:
    outcomes.append(('tau', 'Ex-Gaussian Tau (Stroop Incongruent)'))
if 'prp_bottleneck' in master.columns:
    outcomes.append(('prp_bottleneck', 'PRP Bottleneck Effect'))
if 'stroop_interference' in master.columns:
    outcomes.append(('stroop_interference', 'Stroop Interference'))

print(f"\n[2/5] Outcomes to test: {len(outcomes)}")
for i, (col, label) in enumerate(outcomes, 1):
    n_valid = master[col].notna().sum()
    print(f"  {i}. {label} (N = {n_valid})")

if len(outcomes) == 0:
    print("ERROR: No valid outcome variables found!")
    sys.exit(1)

# ============================================================================
# PART A: HUBER ROBUST REGRESSION
# ============================================================================
print(f"\n[3/5] PART A: Huber Robust Regression vs OLS...")

huber_results = []

for outcome_col, outcome_label in outcomes:
    print(f"\n  Analyzing: {outcome_label}")
    print(f"  " + "-" * 60)

    # Prepare data
    analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                     'z_ucla', 'gender_male', outcome_col]
    df_clean = master[analysis_cols].dropna()

    if len(df_clean) < 20:
        print(f"    WARNING: Only {len(df_clean)} complete cases. Skipping.")
        continue

    print(f"    N = {len(df_clean)}")

    # OLS regression (for comparison)
    formula = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
    ols_model = smf.ols(formula, data=df_clean).fit()

    # Extract OLS results for UCLA × Gender interaction
    interaction_term = 'z_ucla:C(gender_male)[T.1]'
    if interaction_term in ols_model.params.index:
        ols_beta = ols_model.params[interaction_term]
        ols_se = ols_model.bse[interaction_term]
        ols_t = ols_model.tvalues[interaction_term]
        ols_p = ols_model.pvalues[interaction_term]
        ols_ci_lower = ols_model.conf_int().loc[interaction_term, 0]
        ols_ci_upper = ols_model.conf_int().loc[interaction_term, 1]
    else:
        print(f"    WARNING: Interaction term not found in OLS model")
        continue

    # Construct design matrix for RLM
    # RLM doesn't accept formulas, need manual design matrix
    X_design = pd.DataFrame({
        'const': 1,
        'z_age': df_clean['z_age'],
        'gender_male': df_clean['gender_male'],
        'z_dass_dep': df_clean['z_dass_dep'],
        'z_dass_anx': df_clean['z_dass_anx'],
        'z_dass_str': df_clean['z_dass_str'],
        'z_ucla': df_clean['z_ucla'],
        'ucla_x_gender': df_clean['z_ucla'] * df_clean['gender_male']
    })

    y = df_clean[outcome_col].values

    # Fit Huber robust regression
    huber_model = RLM(y, X_design, M=HuberT()).fit()

    # Extract Huber results for interaction
    huber_beta = huber_model.params['ucla_x_gender']
    huber_se = huber_model.bse['ucla_x_gender']
    huber_t = huber_model.tvalues['ucla_x_gender']
    huber_p = huber_model.pvalues['ucla_x_gender']
    huber_ci_lower = huber_model.conf_int().loc['ucla_x_gender', 0]
    huber_ci_upper = huber_model.conf_int().loc['ucla_x_gender', 1]

    # Compare
    beta_diff = huber_beta - ols_beta
    beta_diff_pct = (beta_diff / ols_beta * 100) if ols_beta != 0 else np.nan

    huber_results.append({
        'outcome': outcome_col,
        'outcome_label': outcome_label,
        'n': len(df_clean),
        'ols_beta': ols_beta,
        'ols_se': ols_se,
        'ols_t': ols_t,
        'ols_p': ols_p,
        'ols_ci_lower': ols_ci_lower,
        'ols_ci_upper': ols_ci_upper,
        'huber_beta': huber_beta,
        'huber_se': huber_se,
        'huber_t': huber_t,
        'huber_p': huber_p,
        'huber_ci_lower': huber_ci_lower,
        'huber_ci_upper': huber_ci_upper,
        'beta_diff': beta_diff,
        'beta_diff_pct': beta_diff_pct
    })

    print(f"    OLS:")
    print(f"      β = {ols_beta:.4f}, SE = {ols_se:.4f}, t = {ols_t:.3f}, p = {ols_p:.4f}")
    print(f"      95% CI: [{ols_ci_lower:.4f}, {ols_ci_upper:.4f}]")
    print(f"    Huber:")
    print(f"      β = {huber_beta:.4f}, SE = {huber_se:.4f}, t = {huber_t:.3f}, p = {huber_p:.4f}")
    print(f"      95% CI: [{huber_ci_lower:.4f}, {huber_ci_upper:.4f}]")
    print(f"    Difference:")
    print(f"      Δβ = {beta_diff:.4f} ({beta_diff_pct:.1f}% change)")

    if abs(beta_diff_pct) < 10:
        print(f"      ✓ Robust estimate very similar to OLS (< 10% change)")
    elif abs(beta_diff_pct) < 25:
        print(f"      ~ Moderate difference from OLS (10-25% change)")
    else:
        print(f"      ⚠ Large difference from OLS (> 25% change) - outliers influential")

# ============================================================================
# PART B: QUANTILE REGRESSION
# ============================================================================
print(f"\n[4/5] PART B: Quantile Regression (τ = {QUANTILES})...")

quantile_results = []

for outcome_col, outcome_label in outcomes:
    print(f"\n  Analyzing: {outcome_label}")
    print(f"  " + "-" * 60)

    # Prepare data
    analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                     'z_ucla', 'gender_male', outcome_col]
    df_clean = master[analysis_cols].dropna()

    if len(df_clean) < 20:
        print(f"    WARNING: Only {len(df_clean)} complete cases. Skipping.")
        continue

    print(f"    N = {len(df_clean)}")

    # Prepare design matrix
    X = pd.DataFrame({
        'z_age': df_clean['z_age'],
        'gender_male': df_clean['gender_male'],
        'z_dass_dep': df_clean['z_dass_dep'],
        'z_dass_anx': df_clean['z_dass_anx'],
        'z_dass_str': df_clean['z_dass_str'],
        'z_ucla': df_clean['z_ucla'],
        'ucla_x_gender': df_clean['z_ucla'] * df_clean['gender_male']
    })

    y = df_clean[outcome_col].values

    # Run quantile regression for each τ
    print(f"    Quantile regression coefficients (UCLA × Gender):")
    print(f"    {'τ':>6}  {'β':>8}  {'Sig':>5}")
    print(f"    " + "-" * 25)

    for quantile in QUANTILES:
        qr = QuantileRegressor(quantile=quantile, solver='highs', alpha=0)
        qr.fit(X, y)

        # Extract interaction coefficient (last column)
        coef_interaction = qr.coef_[-1]

        # Note: QuantileRegressor doesn't provide p-values directly
        # We just report the coefficient
        quantile_results.append({
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': len(df_clean),
            'quantile': quantile,
            'beta_interaction': coef_interaction
        })

        print(f"    {quantile:>6.2f}  {coef_interaction:>8.4f}  {'--':>5}")

    # Compare quantiles
    outcome_qr = [r for r in quantile_results if r['outcome'] == outcome_col]
    betas = [r['beta_interaction'] for r in outcome_qr]

    print(f"    Range: [{min(betas):.4f}, {max(betas):.4f}]")

    # Check if effect stronger at high quantiles (vulnerable tail hypothesis)
    beta_low = outcome_qr[0]['beta_interaction']  # τ=0.25
    beta_high = outcome_qr[-1]['beta_interaction']  # τ=0.90

    if abs(beta_high) > abs(beta_low) * 1.5:
        print(f"    ⚠ Effect much stronger at τ=0.90 (vulnerable tail)")
    elif abs(beta_low) > abs(beta_high) * 1.5:
        print(f"    ⚠ Effect much stronger at τ=0.25 (resilient floor)")
    else:
        print(f"    ~ Effect relatively consistent across quantiles")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\n[5/5] Saving results...")

if len(huber_results) > 0:
    huber_df = pd.DataFrame(huber_results)
    huber_path = OUTPUT_DIR / "robust_huber_results.csv"
    huber_df.to_csv(huber_path, index=False, encoding='utf-8-sig')
    print(f"  Saved Huber results: {huber_path}")

if len(quantile_results) > 0:
    quantile_df = pd.DataFrame(quantile_results)
    quantile_path = OUTPUT_DIR / "quantile_regression_results.csv"
    quantile_df.to_csv(quantile_path, index=False, encoding='utf-8-sig')
    print(f"  Saved quantile results: {quantile_path}")

print("\n" + "=" * 80)
print("ROBUST & QUANTILE REGRESSION ANALYSIS COMPLETE")
print("=" * 80)

if len(huber_results) > 0:
    print(f"\nPART A: Robustness to Outliers (Huber vs OLS):")
    print(f"-" * 80)
    for res in huber_results:
        print(f"\n{res['outcome_label']}:")
        print(f"  OLS β = {res['ols_beta']:.4f} (p = {res['ols_p']:.4f})")
        print(f"  Huber β = {res['huber_beta']:.4f} (p = {res['huber_p']:.4f})")
        print(f"  Change: {res['beta_diff_pct']:.1f}%")

        if abs(res['beta_diff_pct']) < 10:
            print(f"  ✓ Robust to outliers")
        elif abs(res['beta_diff_pct']) < 25:
            print(f"  ~ Moderate outlier influence")
        else:
            print(f"  ⚠ High outlier influence")

if len(quantile_results) > 0:
    print(f"\nPART B: Distributional Effects (Quantile Regression):")
    print(f"-" * 80)

    for outcome in outcomes:
        outcome_col, outcome_label = outcome
        outcome_qr = [r for r in quantile_results if r['outcome'] == outcome_col]

        if len(outcome_qr) > 0:
            print(f"\n{outcome_label}:")
            print(f"  {'Quantile':>10}  {'β (UCLA×Gender)':>18}")
            print(f"  " + "-" * 30)
            for res in outcome_qr:
                print(f"  {res['quantile']:>10.2f}  {res['beta_interaction']:>18.4f}")

            betas = [r['beta_interaction'] for r in outcome_qr]
            beta_range = max(betas) - min(betas)
            print(f"  Range: {beta_range:.4f}")

            # Interpretation
            beta_low = outcome_qr[0]['beta_interaction']
            beta_high = outcome_qr[-1]['beta_interaction']

            if abs(beta_high) > abs(beta_low) * 1.5:
                print(f"  → Effect strongest in vulnerable tail (high PE/RT)")
            elif abs(beta_low) > abs(beta_high) * 1.5:
                print(f"  → Effect strongest in resilient group (low PE/RT)")
            else:
                print(f"  → Effect consistent across distribution")

print("\n" + "=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)
