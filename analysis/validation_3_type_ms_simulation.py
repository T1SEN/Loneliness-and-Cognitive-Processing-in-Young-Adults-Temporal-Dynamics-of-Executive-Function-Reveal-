"""
Validation Analysis #3: Simulation-Based Type M/S Error
========================================================
CRITICAL: All models control for DASS-21 subscales + age

Purpose:
--------
Quantify the risk of effect size overestimation (Type M error) and
sign error (Type S error) for the observed UCLA × Gender interaction effects.

Key Questions:
--------------
- If the true effect is smaller than observed, how often do we get
  significant results with exaggerated effect sizes? (Type M error)
- How often do we get the wrong sign? (Type S error)
- What is our statistical power?

Background:
-----------
Gelman & Carlin (2014): "Beyond Power Calculations: Assessing Type S
(Sign) and Type M (Magnitude) Errors"

Type M error = E[|β_hat| / |β_true| | significant]
Type S error = P(sign(β_hat) ≠ sign(β_true) | significant)

Method:
-------
1. Extract observed β and SE from master_dass_controlled analysis
2. Set conservative true β (e.g., 70% of observed β)
3. Simulate 1000 datasets with N = actual sample size:
   - Generate predictors (age, gender, DASS, UCLA)
   - Generate outcome: y = X @ β_true + noise
   - Fit full regression model
   - Record β_hat, SE, p-value
4. Calculate:
   - Power = P(p < 0.05)
   - Type M = mean(|β_hat| / |β_true|) for significant results
   - Type S = P(sign error) for significant results

Outcomes Tested:
----------------
1. WCST perseverative error rate (PRIMARY - p=0.025 in original analysis)
2. Ex-Gaussian tau (incongruent Stroop trials)
3. PRP bottleneck effect
4. Stroop interference

Interpretation:
---------------
- Type M > 2.0: Significant results exaggerate true effect by 2×
- Type S > 0.05: Non-trivial risk of getting wrong sign
- Power < 0.80: Underpowered (increases Type M/S errors)

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
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/validation_analyses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 1000
TRUE_EFFECT_MULTIPLIER = 0.70  # Conservative: true β = 70% of observed β
ALPHA = 0.05

print("=" * 80)
print("VALIDATION ANALYSIS #3: TYPE M/S ERROR SIMULATION")
print("=" * 80)
print(f"Configuration:")
print(f"  - Simulations: {N_SIMULATIONS}")
print(f"  - True effect: {TRUE_EFFECT_MULTIPLIER*100:.0f}% of observed")
print(f"  - Alpha: {ALPHA}")
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
# EXTRACT OBSERVED EFFECTS
# ============================================================================
print(f"\n[3/5] Extracting observed effects from data...")

observed_effects = []

for outcome_col, outcome_label in outcomes:
    print(f"\n  {outcome_label}")
    print(f"  " + "-" * 60)

    # Prepare data
    analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                     'z_ucla', 'gender_male', outcome_col]
    df_clean = master[analysis_cols].dropna()

    if len(df_clean) < 20:
        print(f"    WARNING: Only {len(df_clean)} complete cases. Skipping.")
        continue

    n = len(df_clean)
    print(f"    N = {n}")

    # Fit full model
    formula = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
    model = smf.ols(formula, data=df_clean).fit()

    # Extract interaction effect
    interaction_term = 'z_ucla:C(gender_male)[T.1]'
    if interaction_term not in model.params.index:
        print(f"    WARNING: Interaction term not found. Skipping.")
        continue

    observed_beta = model.params[interaction_term]
    observed_se = model.bse[interaction_term]
    observed_p = model.pvalues[interaction_term]
    residual_std = np.sqrt(model.mse_resid)  # Residual standard error

    print(f"    Observed β = {observed_beta:.4f}, SE = {observed_se:.4f}, p = {observed_p:.4f}")
    print(f"    Residual SD = {residual_std:.4f}")

    # Store for simulation
    observed_effects.append({
        'outcome': outcome_col,
        'outcome_label': outcome_label,
        'n': n,
        'observed_beta': observed_beta,
        'observed_se': observed_se,
        'observed_p': observed_p,
        'residual_std': residual_std,
        'df_clean': df_clean
    })

# ============================================================================
# RUN SIMULATIONS
# ============================================================================
print(f"\n[4/5] Running {N_SIMULATIONS} simulations per outcome...")

simulation_results = []

for obs_effect in observed_effects:
    outcome_col = obs_effect['outcome']
    outcome_label = obs_effect['outcome_label']
    n = obs_effect['n']
    observed_beta = obs_effect['observed_beta']
    residual_std = obs_effect['residual_std']
    df_clean = obs_effect['df_clean']

    print(f"\n  Simulating: {outcome_label}")
    print(f"  " + "-" * 60)
    print(f"    N = {n}, Observed β = {observed_beta:.4f}")

    # Set conservative true effect
    true_beta_interaction = observed_beta * TRUE_EFFECT_MULTIPLIER
    print(f"    True β (conservative) = {true_beta_interaction:.4f}")

    # Extract other coefficients from observed model to use as true values
    formula = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
    obs_model = smf.ols(formula, data=df_clean).fit()

    # True coefficients (using observed as proxy, except interaction)
    true_intercept = obs_model.params['Intercept']
    true_age = obs_model.params['z_age']
    true_gender = obs_model.params['C(gender_male)[T.1]']
    true_dass_dep = obs_model.params['z_dass_dep']
    true_dass_anx = obs_model.params['z_dass_anx']
    true_dass_str = obs_model.params['z_dass_str']
    true_ucla = obs_model.params['z_ucla']

    # Run simulations
    sim_results = []

    for sim_idx in range(N_SIMULATIONS):
        # Use actual predictor values from data (to preserve correlations)
        X = df_clean[['z_age', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_ucla']].values

        # Generate outcome with true effects
        y_true = (true_intercept +
                  true_age * X[:, 0] +
                  true_gender * X[:, 1] +
                  true_dass_dep * X[:, 2] +
                  true_dass_anx * X[:, 3] +
                  true_dass_str * X[:, 4] +
                  true_ucla * X[:, 5] +
                  true_beta_interaction * X[:, 5] * X[:, 1] +  # UCLA × Gender
                  np.random.normal(0, residual_std, n))

        # Create simulation dataframe with interaction term
        sim_df = pd.DataFrame({
            'y': y_true,
            'z_age': X[:, 0],
            'gender_male': X[:, 1],
            'z_dass_dep': X[:, 2],
            'z_dass_anx': X[:, 3],
            'z_dass_str': X[:, 4],
            'z_ucla': X[:, 5],
            'ucla_x_gender': X[:, 5] * X[:, 1]  # Manually create interaction
        })

        # Fit model on simulated data (using numeric coding for simplicity)
        sim_formula = "y ~ z_age + gender_male + z_dass_dep + z_dass_anx + z_dass_str + z_ucla + ucla_x_gender"
        try:
            sim_model = smf.ols(sim_formula, data=sim_df).fit()

            # Extract interaction coefficient
            interaction_term = 'ucla_x_gender'
            beta_hat = sim_model.params[interaction_term]
            se_hat = sim_model.bse[interaction_term]
            p_hat = sim_model.pvalues[interaction_term]

            sim_results.append({
                'sim_idx': sim_idx,
                'beta_hat': beta_hat,
                'se_hat': se_hat,
                'p_value': p_hat,
                'significant': p_hat < ALPHA,
                'sign_correct': np.sign(beta_hat) == np.sign(true_beta_interaction)
            })

        except Exception as e:
            # Debug: print first error
            if sim_idx == 0:
                print(f"      DEBUG - First simulation error: {e}")
            continue

    sim_df_results = pd.DataFrame(sim_results)

    # Check if we have valid simulations
    if len(sim_df_results) == 0:
        print(f"    ERROR: All simulations failed. Skipping.")
        continue

    # Calculate Type M/S errors
    sig_results = sim_df_results[sim_df_results['significant']]
    n_significant = len(sig_results)
    power = n_significant / len(sim_df_results)

    if n_significant > 0:
        # Type M error: exaggeration ratio (for significant results)
        type_m_error = sig_results['beta_hat'].abs().mean() / abs(true_beta_interaction)

        # Type S error: sign error rate (for significant results)
        type_s_error = (~sig_results['sign_correct']).mean()

        # Distribution of significant estimates
        beta_hat_mean = sig_results['beta_hat'].mean()
        beta_hat_median = sig_results['beta_hat'].median()
        beta_hat_sd = sig_results['beta_hat'].std()
    else:
        type_m_error = np.nan
        type_s_error = np.nan
        beta_hat_mean = np.nan
        beta_hat_median = np.nan
        beta_hat_sd = np.nan

    print(f"    Power: {power:.3f}")
    print(f"    Type M error: {type_m_error:.3f} (exaggeration ratio)")
    print(f"    Type S error: {type_s_error:.3f} (sign error rate)")

    if not np.isnan(type_m_error):
        if type_m_error < 1.2:
            print(f"    ✓ Minimal exaggeration (< 20%)")
        elif type_m_error < 2.0:
            print(f"    ~ Moderate exaggeration (20-100%)")
        else:
            print(f"    ⚠ High exaggeration (> 2×)")

    if not np.isnan(type_s_error):
        if type_s_error < 0.05:
            print(f"    ✓ Sign error risk negligible (< 5%)")
        elif type_s_error < 0.20:
            print(f"    ~ Moderate sign error risk (5-20%)")
        else:
            print(f"    ⚠ High sign error risk (> 20%)")

    # Store results
    simulation_results.append({
        'outcome': outcome_col,
        'outcome_label': outcome_label,
        'n': n,
        'n_simulations': len(sim_df_results),
        'true_beta': true_beta_interaction,
        'observed_beta': observed_beta,
        'true_effect_multiplier': TRUE_EFFECT_MULTIPLIER,
        'power': power,
        'n_significant': n_significant,
        'type_m_error': type_m_error,
        'type_s_error': type_s_error,
        'beta_hat_mean_sig': beta_hat_mean,
        'beta_hat_median_sig': beta_hat_median,
        'beta_hat_sd_sig': beta_hat_sd
    })

    # Save detailed simulation results
    sim_detail_path = OUTPUT_DIR / f"type_ms_simulation_detail_{outcome_col}.csv"
    sim_df_results['outcome'] = outcome_col
    sim_df_results['true_beta'] = true_beta_interaction
    sim_df_results.to_csv(sim_detail_path, index=False, encoding='utf-8-sig')

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\n[5/5] Saving results...")

if len(simulation_results) > 0:
    results_df = pd.DataFrame(simulation_results)
    results_path = OUTPUT_DIR / "type_ms_simulation_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"  Saved summary results: {results_path}")
    print(f"  Saved detailed results: {OUTPUT_DIR / 'type_ms_simulation_detail_*.csv'}")

    print("\n" + "=" * 80)
    print("TYPE M/S ERROR SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nSummary (True β = {TRUE_EFFECT_MULTIPLIER*100:.0f}% of observed):")
    print(f"-" * 80)

    for res in simulation_results:
        print(f"\n{res['outcome_label']}:")
        print(f"  N = {res['n']}, Simulations = {res['n_simulations']}")
        print(f"  Observed β = {res['observed_beta']:.4f}")
        print(f"  True β (conservative) = {res['true_beta']:.4f}")
        print(f"  Power = {res['power']:.3f}")
        print(f"  Type M error = {res['type_m_error']:.3f} ({res['type_m_error']*100:.0f}% exaggeration)")
        print(f"  Type S error = {res['type_s_error']:.3f} ({res['type_s_error']*100:.1f}% sign error)")

        if not np.isnan(res['type_m_error']):
            if res['type_m_error'] < 1.2:
                print(f"  ✓ Exaggeration risk LOW")
            elif res['type_m_error'] < 2.0:
                print(f"  ~ Exaggeration risk MODERATE")
            else:
                print(f"  ⚠ Exaggeration risk HIGH (> 2×)")

        if not np.isnan(res['type_s_error']):
            if res['type_s_error'] < 0.05:
                print(f"  ✓ Sign error risk LOW")
            elif res['type_s_error'] < 0.20:
                print(f"  ~ Sign error risk MODERATE")
            else:
                print(f"  ⚠ Sign error risk HIGH")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("-" * 80)
    print("Type M Error (Exaggeration Ratio):")
    print("  1.0-1.2: Minimal exaggeration (< 20%)")
    print("  1.2-2.0: Moderate exaggeration (20-100%)")
    print("  > 2.0:   High exaggeration (> 2×)")
    print("")
    print("Type S Error (Sign Error Rate):")
    print("  < 0.05:  Negligible risk")
    print("  0.05-0.20: Moderate risk")
    print("  > 0.20:  High risk")
    print("")
    print("Power:")
    print("  < 0.50:  Very low (increases Type M/S errors)")
    print("  0.50-0.80: Moderate")
    print("  > 0.80:  Adequate")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

else:
    print("  ERROR: No valid results to save!")
    sys.exit(1)
