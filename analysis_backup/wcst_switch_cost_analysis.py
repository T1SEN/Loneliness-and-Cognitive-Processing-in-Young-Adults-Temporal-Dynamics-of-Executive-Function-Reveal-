"""
WCST Switch Cost Analysis
=========================

Purpose: Test whether the UCLA × Gender interaction is specific to switch trials
         (requiring cognitive flexibility) vs. repeat trials (maintaining set).

Hypothesis: Lonely males show impaired performance specifically on switch trials,
            indicating a cognitive flexibility deficit rather than general impairment.

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
import ast

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("WCST Switch Cost Analysis")
print("=" * 80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n1. Loading data...")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

print(f"   - Participants: {len(participants)}")
print(f"   - Survey responses: {len(surveys)}")
print(f"   - WCST trials: {len(wcst_trials)}")

# ============================================================================
# 2. Extract UCLA and Demographics
# ============================================================================
print("\n2. Extracting UCLA scores and demographics...")

# Normalize survey column names
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
print(f"   - Gender distribution: {demo['gender'].value_counts().to_dict()}")

# ============================================================================
# 3. Identify Switch vs. Repeat Trials
# ============================================================================
print("\n3. Identifying switch vs. repeat trials...")

# Filter valid trials (non-timeout, valid RT)
# Note: isPE and isNPE columns already exist in the data
wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['reactionTimeMs'] > 0)
].copy()

print(f"   - Valid WCST trials: {len(wcst_valid)} (filtered from {len(wcst_trials)})")

# Sort by participant and trial index to identify switches
wcst_valid = wcst_valid.sort_values(['participantId', 'trialIndex'])

# Identify switch trials using the ruleAtThatTime field
# When the rule changes from the previous trial, it's a switch trial
wcst_valid['prev_rule'] = wcst_valid.groupby('participantId')['ruleAtThatTime'].shift(1)
wcst_valid['is_switch'] = wcst_valid['ruleAtThatTime'] != wcst_valid['prev_rule']

# First trial for each participant is not a switch (no previous context)
wcst_valid.loc[wcst_valid.groupby('participantId').head(1).index, 'is_switch'] = False

wcst_valid['is_repeat'] = ~wcst_valid['is_switch']

print(f"   - Switch trials: {wcst_valid['is_switch'].sum()} ({wcst_valid['is_switch'].mean()*100:.1f}%)")
print(f"   - Repeat trials: {wcst_valid['is_repeat'].sum()} ({wcst_valid['is_repeat'].mean()*100:.1f}%)")

# ============================================================================
# 4. Compute Switch vs. Repeat Performance per Participant
# ============================================================================
print("\n4. Computing switch vs. repeat performance...")

# For each participant, compute:
# - Accuracy on switch trials
# - Accuracy on repeat trials
# - RT on switch trials
# - RT on repeat trials
# - Switch cost = switch_metric - repeat_metric

perf_summary = []

for pid in wcst_valid['participantId'].unique():
    pid_data = wcst_valid[wcst_valid['participantId'] == pid]

    switch_data = pid_data[pid_data['is_switch']]
    repeat_data = pid_data[pid_data['is_repeat']]

    if len(switch_data) < 3 or len(repeat_data) < 3:
        continue  # Need minimum trials for reliable estimates

    switch_acc = switch_data['correct'].mean()
    repeat_acc = repeat_data['correct'].mean()
    switch_rt = switch_data['reactionTimeMs'].mean()
    repeat_rt = repeat_data['reactionTimeMs'].mean()

    # Switch costs
    acc_switch_cost = repeat_acc - switch_acc  # Positive = worse on switch
    rt_switch_cost = switch_rt - repeat_rt      # Positive = slower on switch

    perf_summary.append({
        'participant_id': pid,
        'switch_accuracy': switch_acc,
        'repeat_accuracy': repeat_acc,
        'switch_rt': switch_rt,
        'repeat_rt': repeat_rt,
        'accuracy_switch_cost': acc_switch_cost,
        'rt_switch_cost': rt_switch_cost,
        'n_switch_trials': len(switch_data),
        'n_repeat_trials': len(repeat_data)
    })

perf_df = pd.DataFrame(perf_summary)
print(f"   - Participants with sufficient data: {len(perf_df)}")

# ============================================================================
# 5. Merge with UCLA and Demographics
# ============================================================================
print("\n5. Merging with UCLA and demographics...")

master = perf_df.merge(ucla_scores, on='participant_id', how='inner')
master = master.merge(demo, on='participant_id', how='inner')

# Drop missing values
master = master.dropna(subset=['ucla_total', 'age', 'gender', 'accuracy_switch_cost'])

print(f"   - Final sample size: {len(master)}")
print(f"   - Gender distribution: {master['gender'].value_counts().to_dict()}")

# ============================================================================
# 6. Descriptive Statistics
# ============================================================================
print("\n6. Descriptive statistics:")
print("\nOverall switch cost metrics:")
print(f"   - Accuracy switch cost: M={master['accuracy_switch_cost'].mean():.3f}, SD={master['accuracy_switch_cost'].std():.3f}")
print(f"   - RT switch cost: M={master['rt_switch_cost'].mean():.1f}ms, SD={master['rt_switch_cost'].std():.1f}ms")

print("\nBy gender:")
for gender in ['male', 'female']:
    gender_data = master[master['gender'] == gender]
    print(f"\n   {gender.capitalize()} (N={len(gender_data)}):")
    print(f"      - Accuracy switch cost: M={gender_data['accuracy_switch_cost'].mean():.3f}")
    print(f"      - RT switch cost: M={gender_data['rt_switch_cost'].mean():.1f}ms")
    print(f"      - UCLA: M={gender_data['ucla_total'].mean():.1f}, SD={gender_data['ucla_total'].std():.1f}")

# ============================================================================
# 7. Test UCLA × Gender × Trial Type Interaction
# ============================================================================
print("\n7. Testing UCLA × Gender × Trial Type interaction...")

# We need to reshape to long format for trial type analysis
# Each participant contributes 2 rows (switch, repeat)

long_data = []
for _, row in master.iterrows():
    # Switch trial row
    long_data.append({
        'participant_id': row['participant_id'],
        'trial_type': 'switch',
        'accuracy': row['switch_accuracy'],
        'rt': row['switch_rt'],
        'ucla_total': row['ucla_total'],
        'gender': row['gender'],
        'age': row['age']
    })
    # Repeat trial row
    long_data.append({
        'participant_id': row['participant_id'],
        'trial_type': 'repeat',
        'accuracy': row['repeat_accuracy'],
        'rt': row['repeat_rt'],
        'ucla_total': row['ucla_total'],
        'gender': row['gender'],
        'age': row['age']
    })

long_df = pd.DataFrame(long_data)

# Code variables
long_df['gender_coded'] = long_df['gender'].map({'male': 1, 'female': -1})
long_df['trial_type_coded'] = long_df['trial_type'].map({'switch': 1, 'repeat': -1})

# Standardize continuous predictors
long_df['z_ucla'] = (long_df['ucla_total'] - long_df['ucla_total'].mean()) / long_df['ucla_total'].std()
long_df['z_age'] = (long_df['age'] - long_df['age'].mean()) / long_df['age'].std()

# Create interaction terms
long_df['ucla_x_gender'] = long_df['z_ucla'] * long_df['gender_coded']
long_df['ucla_x_trial'] = long_df['z_ucla'] * long_df['trial_type_coded']
long_df['gender_x_trial'] = long_df['gender_coded'] * long_df['trial_type_coded']
long_df['ucla_x_gender_x_trial'] = long_df['z_ucla'] * long_df['gender_coded'] * long_df['trial_type_coded']

# Test three-way interaction on accuracy
from sklearn.linear_model import LinearRegression

# Model without three-way interaction
X_reduced = long_df[['z_ucla', 'gender_coded', 'trial_type_coded', 'z_age',
                      'ucla_x_gender', 'ucla_x_trial', 'gender_x_trial']].values
y = long_df['accuracy'].values

model_reduced = LinearRegression().fit(X_reduced, y)
r2_reduced = model_reduced.score(X_reduced, y)

# Model with three-way interaction
X_full = long_df[['z_ucla', 'gender_coded', 'trial_type_coded', 'z_age',
                   'ucla_x_gender', 'ucla_x_trial', 'gender_x_trial',
                   'ucla_x_gender_x_trial']].values

model_full = LinearRegression().fit(X_full, y)
r2_full = model_full.score(X_full, y)

# F-test for three-way interaction
n = len(y)
k_reduced = X_reduced.shape[1]
k_full = X_full.shape[1]

delta_r2 = r2_full - r2_reduced
f_stat = (delta_r2 / (k_full - k_reduced)) / ((1 - r2_full) / (n - k_full - 1))
p_value = 1 - stats.f.cdf(f_stat, k_full - k_reduced, n - k_full - 1)

print(f"\n   Three-way interaction (UCLA × Gender × Trial Type) on Accuracy:")
print(f"      - ΔR² = {delta_r2:.4f}")
print(f"      - F({k_full - k_reduced}, {n - k_full - 1}) = {f_stat:.3f}")
print(f"      - p = {p_value:.4f}")
print(f"      - Three-way β = {model_full.coef_[-1]:.4f}")

# ============================================================================
# 8. Stratified Analysis: Switch Cost by Gender × UCLA
# ============================================================================
print("\n8. Stratified analysis of switch cost...")

# Standardize UCLA
master['z_ucla'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()
master['gender_coded'] = master['gender'].map({'male': 1, 'female': -1})
master['z_age'] = (master['age'] - master['age'].mean()) / master['age'].std()

# Test UCLA × Gender interaction on accuracy switch cost
master['ucla_x_gender'] = master['z_ucla'] * master['gender_coded']

from sklearn.linear_model import LinearRegression

X = master[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values
y_acc_cost = master['accuracy_switch_cost'].values

model = LinearRegression().fit(X, y_acc_cost)
interaction_beta = model.coef_[-1]

print(f"\n   Predicting accuracy switch cost:")
print(f"      - UCLA × Gender β = {interaction_beta:.4f}")

# Correlations stratified by gender
male_data = master[master['gender'] == 'male']
female_data = master[master['gender'] == 'female']

male_r, male_p = stats.pearsonr(male_data['ucla_total'], male_data['accuracy_switch_cost'])
female_r, female_p = stats.pearsonr(female_data['ucla_total'], female_data['accuracy_switch_cost'])

print(f"\n   Correlation: UCLA × Accuracy Switch Cost")
print(f"      - Males (N={len(male_data)}): r={male_r:.3f}, p={male_p:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r:.3f}, p={female_p:.4f}")

# RT switch cost
y_rt_cost = master['rt_switch_cost'].values
model_rt = LinearRegression().fit(X, y_rt_cost)
rt_interaction_beta = model_rt.coef_[-1]

print(f"\n   Predicting RT switch cost:")
print(f"      - UCLA × Gender β = {rt_interaction_beta:.1f}")

male_r_rt, male_p_rt = stats.pearsonr(male_data['ucla_total'], male_data['rt_switch_cost'])
female_r_rt, female_p_rt = stats.pearsonr(female_data['ucla_total'], female_data['rt_switch_cost'])

print(f"\n   Correlation: UCLA × RT Switch Cost")
print(f"      - Males (N={len(male_data)}): r={male_r_rt:.3f}, p={male_p_rt:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_rt:.3f}, p={female_p_rt:.4f}")

# ============================================================================
# 9. Compare Switch Cost Effect vs. Overall Perseverative Error Effect
# ============================================================================
print("\n9. Comparing switch cost effect to perseverative error effect...")

# Load previous PE analysis results for comparison
try:
    pe_results = pd.read_csv(OUTPUT_DIR / "error_types" / "error_type_summary.csv")
    pe_interaction = pe_results[pe_results['error_type'] == 'PE']['interaction_beta'].iloc[0]
    npe_interaction = pe_results[pe_results['error_type'] == 'NPE']['interaction_beta'].iloc[0]

    print(f"\n   Previous findings:")
    print(f"      - Perseverative Error rate × Gender β = {pe_interaction:.3f}")
    print(f"      - Non-Perseverative Error rate × Gender β = {npe_interaction:.3f}")
    print(f"      - Accuracy Switch Cost × Gender β = {interaction_beta:.4f}")
    print(f"\n   Interpretation:")

    ratio = abs(interaction_beta) / abs(pe_interaction)
    print(f"      Switch cost interaction is {ratio:.1%} the magnitude of PE interaction.")

    if ratio < 0.1:
        print(f"      This suggests the gender moderation effect is NOT primarily about")
        print(f"      cognitive flexibility in switching between rules, but rather about")
        print(f"      perseverative rigidity (continuing incorrect response patterns).")
        print(f"\n      Key finding: The effect is specific to PERSEVERATION, not SWITCHING.")
    else:
        print(f"      Switch cost shows substantial gender moderation, suggesting")
        print(f"      cognitive flexibility deficits contribute to the effect.")

except FileNotFoundError:
    print("   (Previous PE analysis results not found for comparison)")

# ============================================================================
# 10. Save Results
# ============================================================================
print("\n10. Saving results...")

# Summary statistics
summary_output = pd.DataFrame([{
    'metric': 'accuracy_switch_cost',
    'three_way_beta': model_full.coef_[-1],
    'three_way_p': p_value,
    'interaction_beta': interaction_beta,
    'male_r': male_r,
    'male_p': male_p,
    'female_r': female_r,
    'female_p': female_p,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'rt_switch_cost',
    'three_way_beta': np.nan,
    'three_way_p': np.nan,
    'interaction_beta': rt_interaction_beta,
    'male_r': male_r_rt,
    'male_p': male_p_rt,
    'female_r': female_r_rt,
    'female_p': female_p_rt,
    'male_n': len(male_data),
    'female_n': len(female_data)
}])

summary_output.to_csv(OUTPUT_DIR / "wcst_switch_cost_summary.csv", index=False, encoding='utf-8-sig')

# Detailed participant-level data
master.to_csv(OUTPUT_DIR / "wcst_switch_cost_data.csv", index=False, encoding='utf-8-sig')

print(f"   - Saved: wcst_switch_cost_summary.csv")
print(f"   - Saved: wcst_switch_cost_data.csv")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
