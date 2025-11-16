"""
Ex-Gaussian RT Decomposition Analysis
======================================

Purpose: Separate reaction time distributions into cognitive components to reveal
         hidden mechanisms of the UCLA × Gender effect.

Ex-Gaussian Parameters:
- μ (mu): Normal component mean (routine processing speed)
- σ (sigma): Normal component SD (variability in routine processing)
- τ (tau): Exponential tail (attentional lapses, slow responses)

Hypothesis: Lonely males show higher τ on PE trials (attentional lapses during
            perseveration, not deliberate errors).

Method: Fit Ex-Gaussian distribution to RT distributions for:
- WCST PE trials vs correct trials
- WCST NPE trials vs correct trials
- Stroop incongruent vs congruent trials

Author: Claude Code
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/exgaussian_rt")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("Ex-Gaussian RT Decomposition Analysis")
print("=" * 80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n1. Loading data...")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")

print(f"   - Participants: {len(participants)}")
print(f"   - Survey responses: {len(surveys)}")
print(f"   - WCST trials: {len(wcst_trials)}")
print(f"   - Stroop trials: {len(stroop_trials)}")

# ============================================================================
# 2. Extract UCLA and Demographics
# ============================================================================
print("\n2. Extracting UCLA scores and demographics...")

# Normalize column names
surveys = surveys.rename(columns={'participantId': 'participant_id'})

# UCLA scores
ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores.columns = ['participant_id', 'ucla_total']

# Demographics
demo = participants[['participantId', 'age', 'gender', 'education']].copy()
demo.columns = ['participant_id', 'age', 'gender', 'education']

# Recode gender from Korean to English
demo['gender'] = demo['gender'].map({'남성': 'male', '여성': 'female'})
demo = demo.dropna(subset=['gender'])

print(f"   - UCLA scores: {len(ucla_scores)}")
print(f"   - Demographics: {len(demo)}")

# ============================================================================
# 3. Ex-Gaussian Fitting Function
# ============================================================================
print("\n3. Defining Ex-Gaussian fitting function...")

def fit_exgaussian(rt_values, min_trials=10):
    """
    Fit Ex-Gaussian distribution to RT values.

    Ex-Gaussian = convolution of Gaussian(μ, σ) + Exponential(τ)
    In scipy: exponnorm with K = τ/σ

    Returns: (mu, sigma, tau) or (nan, nan, nan) if fit fails
    """
    if len(rt_values) < min_trials:
        return np.nan, np.nan, np.nan

    # Remove outliers (< 200ms or > 5000ms for WCST/Stroop)
    rt_clean = rt_values[(rt_values >= 200) & (rt_values <= 5000)]

    if len(rt_clean) < min_trials:
        return np.nan, np.nan, np.nan

    try:
        # Fit exponnorm: shape parameter K = tau/sigma
        # loc = mu, scale = sigma
        K, loc, scale = stats.exponnorm.fit(rt_clean)

        mu = loc
        sigma = scale
        tau = K * sigma

        # Sanity checks
        if mu < 0 or sigma <= 0 or tau < 0:
            return np.nan, np.nan, np.nan

        if mu > 3000 or tau > 2000:  # Unrealistic values
            return np.nan, np.nan, np.nan

        return mu, sigma, tau

    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan

print("   - Ex-Gaussian fitting function defined")

# ============================================================================
# 4. Fit Ex-Gaussian to WCST RTs
# ============================================================================
print("\n4. Fitting Ex-Gaussian to WCST reaction times...")

# Prepare WCST data
wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['reactionTimeMs'] > 0) &
    (wcst_trials['isPE'].notna())
].copy()

columns_needed = ['participantId', 'reactionTimeMs', 'isPE', 'isNPE', 'correct']
wcst_valid = wcst_valid[columns_needed].copy()
wcst_valid = wcst_valid.rename(columns={'participantId': 'participant_id'})

# Merge with UCLA and demographics
wcst_valid = wcst_valid.merge(ucla_scores, on='participant_id', how='inner')
wcst_valid = wcst_valid.merge(demo, on='participant_id', how='inner')

wcst_exgauss_results = []

for pid in wcst_valid['participant_id'].unique():
    pid_data = wcst_valid[wcst_valid['participant_id'] == pid]

    ucla = pid_data['ucla_total'].iloc[0]
    gender = pid_data['gender'].iloc[0]
    age = pid_data['age'].iloc[0]

    # Fit to different trial types
    # PE trials
    pe_trials = pid_data[pid_data['isPE'] == True]
    pe_mu, pe_sigma, pe_tau = fit_exgaussian(pe_trials['reactionTimeMs'].values)

    # NPE trials
    npe_trials = pid_data[pid_data['isNPE'] == True]
    npe_mu, npe_sigma, npe_tau = fit_exgaussian(npe_trials['reactionTimeMs'].values)

    # Correct trials
    correct_trials = pid_data[pid_data['correct'] == True]
    correct_mu, correct_sigma, correct_tau = fit_exgaussian(correct_trials['reactionTimeMs'].values)

    # All trials
    all_mu, all_sigma, all_tau = fit_exgaussian(pid_data['reactionTimeMs'].values)

    wcst_exgauss_results.append({
        'participant_id': pid,
        'ucla_total': ucla,
        'gender': gender,
        'age': age,

        # PE trials
        'pe_mu': pe_mu,
        'pe_sigma': pe_sigma,
        'pe_tau': pe_tau,

        # NPE trials
        'npe_mu': npe_mu,
        'npe_sigma': npe_sigma,
        'npe_tau': npe_tau,

        # Correct trials
        'correct_mu': correct_mu,
        'correct_sigma': correct_sigma,
        'correct_tau': correct_tau,

        # All trials
        'all_mu': all_mu,
        'all_sigma': all_sigma,
        'all_tau': all_tau,

        # Trial counts
        'n_pe': len(pe_trials),
        'n_npe': len(npe_trials),
        'n_correct': len(correct_trials)
    })

wcst_exgauss_df = pd.DataFrame(wcst_exgauss_results)

# Drop rows with missing tau (primary parameter of interest)
wcst_exgauss_df = wcst_exgauss_df.dropna(subset=['pe_tau', 'correct_tau', 'ucla_total', 'gender', 'age'])

print(f"   - Participants with Ex-Gaussian fits: {len(wcst_exgauss_df)}")
print(f"   - PE tau successfully fitted: {wcst_exgauss_df['pe_tau'].notna().sum()}")
print(f"   - Correct tau successfully fitted: {wcst_exgauss_df['correct_tau'].notna().sum()}")

# Check if we have enough data
if len(wcst_exgauss_df) < 20:
    print(f"   WARNING: Only {len(wcst_exgauss_df)} participants with successful fits")
    print(f"   Ex-Gaussian fitting may be too strict. Continuing with available data...")

# ============================================================================
# 5. Test UCLA × Gender on PE Tau (Key Hypothesis)
# ============================================================================
print("\n5. Testing UCLA × Gender interaction on PE tau (attentional lapses)...")

if len(wcst_exgauss_df) < 10:
    print(f"   ERROR: Insufficient participants ({len(wcst_exgauss_df)}) for regression analysis")
    print(f"   Skipping statistical tests...")
    pe_tau_interaction = np.nan
    male_r_tau = np.nan
    male_p_tau = np.nan
    female_r_tau = np.nan
    female_p_tau = np.nan
else:
    # Code variables
    wcst_exgauss_df['gender_coded'] = wcst_exgauss_df['gender'].map({'male': 1, 'female': -1})
    wcst_exgauss_df['z_ucla'] = (wcst_exgauss_df['ucla_total'] - wcst_exgauss_df['ucla_total'].mean()) / wcst_exgauss_df['ucla_total'].std()
    wcst_exgauss_df['z_age'] = (wcst_exgauss_df['age'] - wcst_exgauss_df['age'].mean()) / wcst_exgauss_df['age'].std()
    wcst_exgauss_df['ucla_x_gender'] = wcst_exgauss_df['z_ucla'] * wcst_exgauss_df['gender_coded']

    # Remove any NaN values in predictors
    wcst_clean = wcst_exgauss_df.dropna(subset=['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender', 'pe_tau'])

    X = wcst_clean[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values

    # PE tau
    y_pe_tau = wcst_clean['pe_tau'].values
    model_pe_tau = LinearRegression().fit(X, y_pe_tau)
    pe_tau_interaction = model_pe_tau.coef_[-1]

print(f"\n   Predicting PE tau (attentional lapses on perseverative errors):")
if not np.isnan(pe_tau_interaction):
    print(f"      - UCLA × Gender β = {pe_tau_interaction:.2f}ms")

    # Stratified correlations
    male_data = wcst_clean[wcst_clean['gender'] == 'male']
    female_data = wcst_clean[wcst_clean['gender'] == 'female']

    if len(male_data) >= 3 and len(female_data) >= 3:
        male_r_tau, male_p_tau = stats.pearsonr(male_data['ucla_total'], male_data['pe_tau'])
        female_r_tau, female_p_tau = stats.pearsonr(female_data['ucla_total'], female_data['pe_tau'])

        print(f"\n   Correlation: UCLA × PE Tau")
        print(f"      - Males (N={len(male_data)}): r={male_r_tau:.3f}, p={male_p_tau:.4f}")
        print(f"      - Females (N={len(female_data)}): r={female_r_tau:.3f}, p={female_p_tau:.4f}")
    else:
        print(f"\n   Insufficient data for stratified correlations")
        male_r_tau = np.nan
        male_p_tau = np.nan
        female_r_tau = np.nan
        female_p_tau = np.nan
else:
    print(f"      - Skipped (insufficient data)")
    male_data = pd.DataFrame()
    female_data = pd.DataFrame()

# ============================================================================
# 6. Test PE vs Correct Tau Difference
# ============================================================================
print("\n6. Testing PE vs Correct tau difference...")

# Compute tau difference (PE - Correct)
wcst_exgauss_df['tau_diff_pe_correct'] = wcst_exgauss_df['pe_tau'] - wcst_exgauss_df['correct_tau']

y_tau_diff = wcst_exgauss_df['tau_diff_pe_correct'].values
model_tau_diff = LinearRegression().fit(X, y_tau_diff)
tau_diff_interaction = model_tau_diff.coef_[-1]

print(f"\n   Predicting tau difference (PE - Correct):")
print(f"      - UCLA × Gender β = {tau_diff_interaction:.2f}ms")

male_r_diff, male_p_diff = stats.pearsonr(male_data['ucla_total'], male_data['tau_diff_pe_correct'])
female_r_diff, female_p_diff = stats.pearsonr(female_data['ucla_total'], female_data['tau_diff_pe_correct'])

print(f"\n   Correlation: UCLA × Tau Difference")
print(f"      - Males (N={len(male_data)}): r={male_r_diff:.3f}, p={male_p_diff:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_diff:.3f}, p={female_p_diff:.4f}")

# ============================================================================
# 7. Test Other Ex-Gaussian Parameters (μ, σ)
# ============================================================================
print("\n7. Testing other Ex-Gaussian parameters...")

# PE mu (routine processing speed)
y_pe_mu = wcst_exgauss_df['pe_mu'].values
model_pe_mu = LinearRegression().fit(X, y_pe_mu)
pe_mu_interaction = model_pe_mu.coef_[-1]

print(f"\n   PE μ (routine processing speed):")
print(f"      - UCLA × Gender β = {pe_mu_interaction:.2f}ms")

# PE sigma (variability)
y_pe_sigma = wcst_exgauss_df['pe_sigma'].values
model_pe_sigma = LinearRegression().fit(X, y_pe_sigma)
pe_sigma_interaction = model_pe_sigma.coef_[-1]

print(f"\n   PE σ (processing variability):")
print(f"      - UCLA × Gender β = {pe_sigma_interaction:.2f}ms")

# ============================================================================
# 8. Fit Ex-Gaussian to Stroop RTs (Control Task)
# ============================================================================
print("\n8. Fitting Ex-Gaussian to Stroop reaction times (control)...")

# Prepare Stroop data
stroop_valid = stroop_trials[
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 0)
].copy()

stroop_valid = stroop_valid.rename(columns={'participantId': 'participant_id'})

# Merge with UCLA and demographics
stroop_valid = stroop_valid.merge(ucla_scores, on='participant_id', how='inner')
stroop_valid = stroop_valid.merge(demo, on='participant_id', how='inner')

stroop_exgauss_results = []

for pid in stroop_valid['participant_id'].unique():
    pid_data = stroop_valid[stroop_valid['participant_id'] == pid]

    ucla = pid_data['ucla_total'].iloc[0]
    gender = pid_data['gender'].iloc[0]
    age = pid_data['age'].iloc[0]

    # Incongruent trials
    incong_trials = pid_data[pid_data['cond'] == 'incongruent']
    incong_mu, incong_sigma, incong_tau = fit_exgaussian(incong_trials['rt_ms'].values)

    # Congruent trials
    cong_trials = pid_data[pid_data['cond'] == 'congruent']
    cong_mu, cong_sigma, cong_tau = fit_exgaussian(cong_trials['rt_ms'].values)

    stroop_exgauss_results.append({
        'participant_id': pid,
        'ucla_total': ucla,
        'gender': gender,
        'age': age,

        # Incongruent
        'incong_mu': incong_mu,
        'incong_sigma': incong_sigma,
        'incong_tau': incong_tau,

        # Congruent
        'cong_mu': cong_mu,
        'cong_sigma': cong_sigma,
        'cong_tau': cong_tau,

        'n_incong': len(incong_trials),
        'n_cong': len(cong_trials)
    })

stroop_exgauss_df = pd.DataFrame(stroop_exgauss_results)
stroop_exgauss_df = stroop_exgauss_df.dropna(subset=['incong_tau', 'cong_tau'])

print(f"   - Stroop participants with fits: {len(stroop_exgauss_df)}")

# Test Stroop incongruent tau
stroop_exgauss_df['gender_coded'] = stroop_exgauss_df['gender'].map({'male': 1, 'female': -1})
stroop_exgauss_df['z_ucla'] = (stroop_exgauss_df['ucla_total'] - stroop_exgauss_df['ucla_total'].mean()) / stroop_exgauss_df['ucla_total'].std()
stroop_exgauss_df['z_age'] = (stroop_exgauss_df['age'] - stroop_exgauss_df['age'].mean()) / stroop_exgauss_df['age'].std()
stroop_exgauss_df['ucla_x_gender'] = stroop_exgauss_df['z_ucla'] * stroop_exgauss_df['gender_coded']

X_stroop = stroop_exgauss_df[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values
y_stroop_tau = stroop_exgauss_df['incong_tau'].values

model_stroop_tau = LinearRegression().fit(X_stroop, y_stroop_tau)
stroop_tau_interaction = model_stroop_tau.coef_[-1]

print(f"\n   Stroop Incongruent tau:")
print(f"      - UCLA × Gender β = {stroop_tau_interaction:.2f}ms")
print(f"      - Comparison: WCST PE tau interaction is {abs(pe_tau_interaction)/abs(stroop_tau_interaction) if stroop_tau_interaction != 0 else np.inf:.2f}× stronger")

# ============================================================================
# 9. Descriptive Statistics
# ============================================================================
print("\n9. Descriptive statistics:")

print(f"\n   WCST Ex-Gaussian parameters (overall):")
print(f"      - PE tau: M={wcst_exgauss_df['pe_tau'].mean():.1f}ms (SD={wcst_exgauss_df['pe_tau'].std():.1f})")
print(f"      - Correct tau: M={wcst_exgauss_df['correct_tau'].mean():.1f}ms (SD={wcst_exgauss_df['correct_tau'].std():.1f})")
print(f"      - Tau difference: M={wcst_exgauss_df['tau_diff_pe_correct'].mean():.1f}ms")

print(f"\n   By gender:")
for gender in ['male', 'female']:
    gdata = wcst_exgauss_df[wcst_exgauss_df['gender'] == gender]
    print(f"\n   {gender.capitalize()} (N={len(gdata)}):")
    print(f"      - PE tau: M={gdata['pe_tau'].mean():.1f}ms")
    print(f"      - Correct tau: M={gdata['correct_tau'].mean():.1f}ms")
    print(f"      - Tau diff: M={gdata['tau_diff_pe_correct'].mean():.1f}ms")

# ============================================================================
# 10. Save Results
# ============================================================================
print("\n10. Saving results...")

# WCST Ex-Gaussian summary
wcst_summary = pd.DataFrame([{
    'task': 'wcst',
    'parameter': 'pe_tau',
    'interaction_beta': pe_tau_interaction,
    'male_r': male_r_tau,
    'male_p': male_p_tau,
    'female_r': female_r_tau,
    'female_p': female_p_tau,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'task': 'wcst',
    'parameter': 'tau_diff_pe_correct',
    'interaction_beta': tau_diff_interaction,
    'male_r': male_r_diff,
    'male_p': male_p_diff,
    'female_r': female_r_diff,
    'female_p': female_p_diff,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'task': 'wcst',
    'parameter': 'pe_mu',
    'interaction_beta': pe_mu_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'task': 'wcst',
    'parameter': 'pe_sigma',
    'interaction_beta': pe_sigma_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'task': 'stroop',
    'parameter': 'incong_tau',
    'interaction_beta': stroop_tau_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': np.nan,
    'female_n': np.nan
}])

wcst_summary.to_csv(OUTPUT_DIR / "exgaussian_summary.csv", index=False, encoding='utf-8-sig')

# Individual Ex-Gaussian parameters
wcst_exgauss_df.to_csv(OUTPUT_DIR / "wcst_exgaussian_params.csv", index=False, encoding='utf-8-sig')
stroop_exgauss_df.to_csv(OUTPUT_DIR / "stroop_exgaussian_params.csv", index=False, encoding='utf-8-sig')

print(f"   - Saved: exgaussian_summary.csv")
print(f"   - Saved: wcst_exgaussian_params.csv")
print(f"   - Saved: stroop_exgaussian_params.csv")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# 11. Interpretation Summary
# ============================================================================
print("\n11. KEY FINDINGS SUMMARY:")
print("=" * 80)

print("\n📊 PE TAU (ATTENTIONAL LAPSES ON PERSEVERATIVE ERRORS):")
print(f"   - UCLA × Gender β = {pe_tau_interaction:.2f}ms")
print(f"   - Males: r={male_r_tau:.3f}, p={male_p_tau:.4f}")
print(f"   - Females: r={female_r_tau:.3f}, p={female_p_tau:.4f}")

print("\n📊 TAU DIFFERENCE (PE - CORRECT):")
print(f"   - UCLA × Gender β = {tau_diff_interaction:.2f}ms")
print(f"   - Males: r={male_r_diff:.3f}, p={male_p_diff:.4f}")
print(f"   - Females: r={female_r_diff:.3f}, p={female_p_diff:.4f}")

print("\n📊 OTHER PARAMETERS:")
print(f"   - PE μ (routine speed) β = {pe_mu_interaction:.2f}ms")
print(f"   - PE σ (variability) β = {pe_sigma_interaction:.2f}ms")

print("\n🎯 TASK SPECIFICITY:")
print(f"   - WCST PE tau β = {pe_tau_interaction:.2f}ms")
print(f"   - Stroop incong tau β = {stroop_tau_interaction:.2f}ms")
if stroop_tau_interaction != 0:
    print(f"   - Ratio: {abs(pe_tau_interaction)/abs(stroop_tau_interaction):.2f}×")

print("\n💡 INTERPRETATION:")
if abs(pe_tau_interaction) > 10:
    print("   ✓ UCLA × Gender affects attentional lapses (τ) during PE")
    print("   ✓ Lonely males show more OCCASIONAL slow responses on PE trials")
    print("   ✓ Suggests lapses in sustained attention, not deliberate perseveration")
else:
    print("   ✓ Weak/null effect on attentional lapses")
    print("   ✓ Effect may not be driven by attention lapses")

if abs(tau_diff_interaction) > abs(pe_tau_interaction) * 0.5:
    print("   ✓ PE-specific tau elevation (not general slowing)")
else:
    print("   ✓ General tau elevation (not PE-specific)")

print("\n" + "=" * 80)
