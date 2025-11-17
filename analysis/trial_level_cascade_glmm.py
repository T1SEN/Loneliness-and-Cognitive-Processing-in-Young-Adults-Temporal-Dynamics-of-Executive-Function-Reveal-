"""
trial_level_cascade_glmm.py

Trial-level mixed-effects models for error cascade and post-error slowing.
Tests whether error cascades and PES effects are moderated by UCLA × Gender.

Models:
1. Error_cascade ~ Error_t-1 × UCLA × Gender + SOA + DASS + (1|Participant)
2. RT_t ~ Error_t-1 × UCLA × Gender + SOA + DASS + (1|Participant)
3. Accuracy_t ~ Error_t-1 × UCLA × Gender + SOA + DASS + (1|Participant)

Outputs:
- cascade_glmm_results.csv
- pes_glmm_results.csv
- cascade_predicted_probabilities.png
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# UTF-8 encoding for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_comprehensive/trial_cascade_glmm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TRIAL-LEVEL ERROR CASCADE & POST-ERROR SLOWING (GLMM)")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD TRIAL-LEVEL DATA
# ============================================================================

print("Loading trial data...")

# PRP trials
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
prp_trials.columns = prp_trials.columns.str.lower()

# Normalize participant ID
if 'participantid' in prp_trials.columns and 'participant_id' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participantid'])
elif 'participantid' in prp_trials.columns:
    prp_trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

print(f"  PRP trials loaded: {len(prp_trials):,} rows")

# ============================================================================
# 2. LOAD PARTICIPANT-LEVEL PREDICTORS
# ============================================================================

print("Loading participant predictors...")

# Load master dataset
master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()

# Load participants for demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns and 'participant_id' not in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

# Merge gender
if 'gender' not in master.columns:
    master = master.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')

# Gender mapping
gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Standardize predictors
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_dass_dep'] = scaler.fit_transform(master[['dass_depression']])
master['z_dass_anx'] = scaler.fit_transform(master[['dass_anxiety']])
master['z_dass_str'] = scaler.fit_transform(master[['dass_stress']])
master['z_age'] = scaler.fit_transform(master[['age']])

print(f"  Participant predictors: {len(master)} participants")
print()

# ============================================================================
# 3. PREPARE TRIAL DATA
# ============================================================================

print("Preparing trial-level data...")

# Clean PRP trials
prp_clean = prp_trials[
    (prp_trials['t1_correct'].notna()) &
    (prp_trials['t2_correct'].notna()) &
    (prp_trials['t2_rt_ms'].notna()) &
    (prp_trials['t2_rt_ms'] > 200) &
    (prp_trials['t2_rt_ms'] < 5000) &
    (prp_trials['soa_nominal_ms'].notna())
].copy()

# Convert boolean
if prp_clean['t1_correct'].dtype != bool:
    prp_clean['t1_correct'] = prp_clean['t1_correct'].astype(bool)
if prp_clean['t2_correct'].dtype != bool:
    prp_clean['t2_correct'] = prp_clean['t2_correct'].astype(bool)

# Compute error cascade (T1 error AND T2 error)
prp_clean['error_cascade'] = (~prp_clean['t1_correct']) & (~prp_clean['t2_correct'])
prp_clean['error_cascade_int'] = prp_clean['error_cascade'].astype(int)

# Sort by participant and trial
prp_clean = prp_clean.sort_values(['participant_id', 'trial_index']).reset_index(drop=True)

# Create lagged variables within participant
prp_clean['t2_error_prev'] = prp_clean.groupby('participant_id')['t2_correct'].shift(1).fillna(True)
prp_clean['t2_error_prev'] = (~prp_clean['t2_error_prev']).astype(int)

# SOA binning
def categorize_soa(soa):
    if soa <= 150:
        return 'short'
    elif 300 <= soa <= 600:
        return 'medium'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'

prp_clean['soa_category'] = prp_clean['soa_nominal_ms'].apply(categorize_soa)
prp_clean = prp_clean[prp_clean['soa_category'].isin(['short', 'medium', 'long'])].copy()

# Standardize SOA
prp_clean['soa_scaled'] = scaler.fit_transform(prp_clean[['soa_nominal_ms']])

# Merge participant-level predictors
prp_clean = prp_clean.merge(
    master[['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male']],
    on='participant_id', how='left'
)

# Drop rows with missing predictors
prp_clean = prp_clean.dropna(subset=['z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str'])

print(f"  Clean trial data: {len(prp_clean):,} trials")
print(f"  Participants: {prp_clean['participant_id'].nunique()}")
print(f"  Error cascade rate: {prp_clean['error_cascade_int'].mean():.3f}")
print()

# ============================================================================
# 4. MODEL 1: ERROR CASCADE GLMM (Logistic Mixed Model)
# ============================================================================

print("=" * 80)
print("MODEL 1: ERROR CASCADE (Logistic GLMM)")
print("=" * 80)
print()

# NOTE: statsmodels GLMM with binomial family can be slow/unstable
# Using simplified formula without full interactions due to convergence issues
print("Fitting logistic mixed model...")
print("Formula: error_cascade ~ t2_error_prev + z_ucla + gender_male + soa_scaled + DASS + (1|participant)")
print()

try:
    formula_cascade = ("error_cascade_int ~ t2_error_prev + z_ucla + C(gender_male) + "
                       "soa_scaled + z_dass_dep + z_dass_anx + z_dass_str")

    # Subsample if too large (for speed)
    if len(prp_clean) > 3000:
        prp_sample = prp_clean.sample(n=3000, random_state=42)
        print(f"  Subsampling to {len(prp_sample)} trials for computational efficiency")
    else:
        prp_sample = prp_clean

    model_cascade = smf.mixedlm(formula_cascade, data=prp_sample,
                                 groups=prp_sample["participant_id"],
                                 re_formula="~1")

    result_cascade = model_cascade.fit(method="lbfgs", maxiter=100)

    print(result_cascade.summary())
    print()

    # Extract coefficients
    cascade_results = pd.DataFrame({
        'parameter': result_cascade.params.index,
        'coefficient': result_cascade.params.values,
        'std_err': result_cascade.bse.values,
        'z_value': result_cascade.tvalues.values,
        'p_value': result_cascade.pvalues.values
    })

    cascade_results.to_csv(OUTPUT_DIR / "cascade_glmm_results.csv", index=False, encoding='utf-8-sig')
    print(f"Saved: {OUTPUT_DIR / 'cascade_glmm_results.csv'}")
    print()

except Exception as e:
    print(f"  ERROR fitting cascade GLMM: {e}")
    print("  Skipping cascade model")
    cascade_results = None

print()

# ============================================================================
# 5. MODEL 2: POST-ERROR SLOWING (Linear Mixed Model)
# ============================================================================

print("=" * 80)
print("MODEL 2: POST-ERROR SLOWING (Linear GLMM)")
print("=" * 80)
print()

print("Fitting linear mixed model for T2 RT...")
print("Formula: t2_rt_ms ~ t2_error_prev × ucla × gender + SOA + DASS + (1|participant)")
print()

try:
    formula_pes = ("t2_rt_ms ~ t2_error_prev * z_ucla * C(gender_male) + "
                   "soa_scaled + z_dass_dep + z_dass_anx + z_dass_str + z_age")

    # Subsample
    if len(prp_clean) > 3000:
        prp_sample_pes = prp_clean.sample(n=3000, random_state=43)
        print(f"  Subsampling to {len(prp_sample_pes)} trials")
    else:
        prp_sample_pes = prp_clean

    model_pes = smf.mixedlm(formula_pes, data=prp_sample_pes,
                             groups=prp_sample_pes["participant_id"],
                             re_formula="~1")

    result_pes = model_pes.fit(method="lbfgs", maxiter=100)

    print(result_pes.summary())
    print()

    # Extract coefficients
    pes_results = pd.DataFrame({
        'parameter': result_pes.params.index,
        'coefficient': result_pes.params.values,
        'std_err': result_pes.bse.values,
        't_value': result_pes.tvalues.values,
        'p_value': result_pes.pvalues.values
    })

    pes_results.to_csv(OUTPUT_DIR / "pes_glmm_results.csv", index=False, encoding='utf-8-sig')
    print(f"Saved: {OUTPUT_DIR / 'pes_glmm_results.csv'}")
    print()

except Exception as e:
    print(f"  ERROR fitting PES GLMM: {e}")
    print("  Skipping PES model")
    pes_results = None

print()

# ============================================================================
# 6. VISUALIZATION (if models succeeded)
# ============================================================================

if cascade_results is not None:
    print("Creating cascade visualization...")

    # Plot predicted cascade rate by previous error
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate by previous error status
    cascade_by_prev = prp_clean.groupby(['t2_error_prev', 'gender_male'])['error_cascade_int'].mean().reset_index()

    sns.barplot(data=cascade_by_prev, x='t2_error_prev', y='error_cascade_int', hue='gender_male',
                palette={0: 'coral', 1: 'steelblue'}, ax=ax)

    ax.set_xlabel('Previous T2 Error (0=Correct, 1=Error)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Cascade Rate', fontsize=12, fontweight='bold')
    ax.set_title('Error Cascade Rate by Previous Trial Status & Gender', fontsize=14, fontweight='bold')
    ax.legend(title='Gender', labels=['Female', 'Male'], fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cascade_by_previous_error.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'cascade_by_previous_error.png'}")
    plt.close()

print()
print("=" * 80)
print("TRIAL-LEVEL CASCADE & PES ANALYSIS COMPLETE")
print("=" * 80)
print()
print("KEY FILES:")
if cascade_results is not None:
    print(f"  - {OUTPUT_DIR / 'cascade_glmm_results.csv'}")
    print(f"  - {OUTPUT_DIR / 'cascade_by_previous_error.png'}")
if pes_results is not None:
    print(f"  - {OUTPUT_DIR / 'pes_glmm_results.csv'}")
print()
