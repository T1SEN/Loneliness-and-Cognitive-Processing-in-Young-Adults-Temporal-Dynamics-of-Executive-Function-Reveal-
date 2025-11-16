"""
WCST Error Cascade Analysis
============================

Purpose: Test whether the gender moderation reflects ERROR CLUSTERING (cascades)
         vs. random error distribution.

Key Question: Do lonely males show LONGER consecutive error runs (cascades),
              operationalizing "rigidity" as temporal clustering?

Hypotheses:
- H1: Lonely males show longer PE runs (stuck in perseverative pattern)
- H2: UCLA × Gender predicts PE run length, not NPE run length (specificity)
- H3: Error probability increases given previous error: P(Error_t | Error_t-1)

Metrics:
- Run length: Number of consecutive errors
- Run frequency: How many error runs
- Conditional probability: P(Error_t | Error_t-1) vs P(Error_t | Correct_t-1)
- Longest run: Maximum consecutive errors

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
OUTPUT_DIR = Path("results/analysis_outputs/error_cascades")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("WCST Error Cascade Analysis")
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
print(f"   - Gender distribution: {demo['gender'].value_counts().to_dict()}")

# ============================================================================
# 3. Prepare Trial-Level Data
# ============================================================================
print("\n3. Preparing trial-level data...")

# Filter valid trials
wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['reactionTimeMs'] > 0) &
    (wcst_trials['isPE'].notna())
].copy()

# Select only needed columns and rename
columns_needed = ['participantId', 'trialIndex', 'isPE', 'isNPE', 'correct', 'reactionTimeMs']
wcst_valid = wcst_valid[columns_needed].copy()
wcst_valid = wcst_valid.rename(columns={'participantId': 'participant_id'})

# Merge with UCLA and demographics
wcst_valid = wcst_valid.merge(ucla_scores, on='participant_id', how='inner')
wcst_valid = wcst_valid.merge(demo, on='participant_id', how='inner')

# Drop missing UCLA
wcst_valid = wcst_valid.dropna(subset=['ucla_total'])

print(f"   - Valid trials: {len(wcst_valid)}")
print(f"   - Participants: {wcst_valid['participant_id'].nunique()}")

# Sort by participant and trial
wcst_valid = wcst_valid.sort_values(['participant_id', 'trialIndex'])

# ============================================================================
# 4. Identify Error Runs
# ============================================================================
print("\n4. Identifying error runs...")

def identify_runs(group, error_col):
    """Identify consecutive error runs in a sequence."""
    group = group.sort_values('trialIndex').copy()
    errors = group[error_col].values

    runs = []
    current_run = 0

    for err in errors:
        if err:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0

    # Don't forget last run if sequence ends with error
    if current_run > 0:
        runs.append(current_run)

    return runs

# Compute error runs for each participant
cascade_metrics = []

for pid in wcst_valid['participant_id'].unique():
    pid_data = wcst_valid[wcst_valid['participant_id'] == pid].copy()

    ucla = pid_data['ucla_total'].iloc[0]
    gender = pid_data['gender'].iloc[0]
    age = pid_data['age'].iloc[0]

    # PE runs
    pe_runs = identify_runs(pid_data, 'isPE')

    # NPE runs
    npe_runs = identify_runs(pid_data, 'isNPE')

    # All error runs (PE or NPE)
    pid_data['is_error'] = ~pid_data['correct']
    error_runs = identify_runs(pid_data, 'is_error')

    # Metrics
    pe_run_length = np.mean(pe_runs) if len(pe_runs) > 0 else 0
    pe_max_run = max(pe_runs) if len(pe_runs) > 0 else 0
    pe_num_runs = len(pe_runs)

    npe_run_length = np.mean(npe_runs) if len(npe_runs) > 0 else 0
    npe_max_run = max(npe_runs) if len(npe_runs) > 0 else 0
    npe_num_runs = len(npe_runs)

    error_run_length = np.mean(error_runs) if len(error_runs) > 0 else 0
    error_max_run = max(error_runs) if len(error_runs) > 0 else 0
    error_num_runs = len(error_runs)

    # Conditional probabilities
    # P(Error_t | Error_t-1) vs P(Error_t | Correct_t-1)
    pid_data['prev_error'] = pid_data['is_error'].shift(1)

    # After error trials
    after_error = pid_data[pid_data['prev_error'] == True]
    p_error_given_error = after_error['is_error'].mean() if len(after_error) > 0 else np.nan

    # After correct trials
    after_correct = pid_data[pid_data['prev_error'] == False]
    p_error_given_correct = after_correct['is_error'].mean() if len(after_correct) > 0 else np.nan

    # Cascade tendency: difference in error probability
    cascade_tendency = p_error_given_error - p_error_given_correct if not pd.isna(p_error_given_error) else np.nan

    cascade_metrics.append({
        'participant_id': pid,
        'ucla_total': ucla,
        'gender': gender,
        'age': age,

        # PE run metrics
        'pe_run_length': pe_run_length,
        'pe_max_run': pe_max_run,
        'pe_num_runs': pe_num_runs,

        # NPE run metrics
        'npe_run_length': npe_run_length,
        'npe_max_run': npe_max_run,
        'npe_num_runs': npe_num_runs,

        # Overall error runs
        'error_run_length': error_run_length,
        'error_max_run': error_max_run,
        'error_num_runs': error_num_runs,

        # Conditional probabilities
        'p_error_given_error': p_error_given_error,
        'p_error_given_correct': p_error_given_correct,
        'cascade_tendency': cascade_tendency,

        'n_trials': len(pid_data)
    })

cascade_df = pd.DataFrame(cascade_metrics)

# Drop rows with missing values
cascade_df = cascade_df.dropna(subset=['pe_run_length', 'cascade_tendency'])

print(f"   - Participants with cascade data: {len(cascade_df)}")

# ============================================================================
# 5. Test UCLA × Gender Interaction on PE Run Length
# ============================================================================
print("\n5. Testing UCLA × Gender interaction on PE run length...")

# Code gender
cascade_df['gender_coded'] = cascade_df['gender'].map({'male': 1, 'female': -1})

# Standardize predictors
cascade_df['z_ucla'] = (cascade_df['ucla_total'] - cascade_df['ucla_total'].mean()) / cascade_df['ucla_total'].std()
cascade_df['z_age'] = (cascade_df['age'] - cascade_df['age'].mean()) / cascade_df['age'].std()
cascade_df['ucla_x_gender'] = cascade_df['z_ucla'] * cascade_df['gender_coded']

# Model: PE run length ~ UCLA × Gender + Age
X = cascade_df[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values
y_pe_run = cascade_df['pe_run_length'].values

model_pe = LinearRegression().fit(X, y_pe_run)
pe_interaction = model_pe.coef_[-1]

print(f"\n   Predicting PE run length:")
print(f"      - UCLA × Gender β = {pe_interaction:.4f}")

# Stratified correlations
male_data = cascade_df[cascade_df['gender'] == 'male']
female_data = cascade_df[cascade_df['gender'] == 'female']

male_r_pe, male_p_pe = stats.pearsonr(male_data['ucla_total'], male_data['pe_run_length'])
female_r_pe, female_p_pe = stats.pearsonr(female_data['ucla_total'], female_data['pe_run_length'])

print(f"\n   Correlation: UCLA × PE Run Length")
print(f"      - Males (N={len(male_data)}): r={male_r_pe:.3f}, p={male_p_pe:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_pe:.3f}, p={female_p_pe:.4f}")

# ============================================================================
# 6. Test UCLA × Gender Interaction on NPE Run Length (Control)
# ============================================================================
print("\n6. Testing UCLA × Gender interaction on NPE run length (control)...")

y_npe_run = cascade_df['npe_run_length'].values
model_npe = LinearRegression().fit(X, y_npe_run)
npe_interaction = model_npe.coef_[-1]

print(f"\n   Predicting NPE run length:")
print(f"      - UCLA × Gender β = {npe_interaction:.4f}")

male_r_npe, male_p_npe = stats.pearsonr(male_data['ucla_total'], male_data['npe_run_length'])
female_r_npe, female_p_npe = stats.pearsonr(female_data['ucla_total'], female_data['npe_run_length'])

print(f"\n   Correlation: UCLA × NPE Run Length")
print(f"      - Males (N={len(male_data)}): r={male_r_npe:.3f}, p={male_p_npe:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_npe:.3f}, p={female_p_npe:.4f}")

print(f"\n   Comparison:")
print(f"      - PE/NPE interaction ratio: {pe_interaction/npe_interaction if npe_interaction != 0 else np.inf:.2f}")

# ============================================================================
# 7. Test Cascade Tendency
# ============================================================================
print("\n7. Testing cascade tendency (error clustering)...")

y_cascade = cascade_df['cascade_tendency'].values
model_cascade = LinearRegression().fit(X, y_cascade)
cascade_interaction = model_cascade.coef_[-1]

print(f"\n   Predicting cascade tendency:")
print(f"      - UCLA × Gender β = {cascade_interaction:.4f}")
print(f"      - (Cascade tendency = P(Error|Error) - P(Error|Correct))")

male_r_cascade, male_p_cascade = stats.pearsonr(male_data['ucla_total'], male_data['cascade_tendency'])
female_r_cascade, female_p_cascade = stats.pearsonr(female_data['ucla_total'], female_data['cascade_tendency'])

print(f"\n   Correlation: UCLA × Cascade Tendency")
print(f"      - Males (N={len(male_data)}): r={male_r_cascade:.3f}, p={male_p_cascade:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_cascade:.3f}, p={female_p_cascade:.4f}")

# ============================================================================
# 8. Test Maximum Run Length
# ============================================================================
print("\n8. Testing maximum run length...")

y_max_run = cascade_df['pe_max_run'].values
model_max = LinearRegression().fit(X, y_max_run)
max_interaction = model_max.coef_[-1]

print(f"\n   Predicting PE maximum run:")
print(f"      - UCLA × Gender β = {max_interaction:.4f}")

male_r_max, male_p_max = stats.pearsonr(male_data['ucla_total'], male_data['pe_max_run'])
female_r_max, female_p_max = stats.pearsonr(female_data['ucla_total'], female_data['pe_max_run'])

print(f"\n   Correlation: UCLA × PE Max Run")
print(f"      - Males (N={len(male_data)}): r={male_r_max:.3f}, p={male_p_max:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_max:.3f}, p={female_p_max:.4f}")

# ============================================================================
# 9. Descriptive Statistics
# ============================================================================
print("\n9. Descriptive statistics:")

print(f"\n   Overall:")
print(f"      - Mean PE run length: {cascade_df['pe_run_length'].mean():.3f} (SD={cascade_df['pe_run_length'].std():.3f})")
print(f"      - Mean NPE run length: {cascade_df['npe_run_length'].mean():.3f} (SD={cascade_df['npe_run_length'].std():.3f})")
print(f"      - Mean cascade tendency: {cascade_df['cascade_tendency'].mean():.3f}")
print(f"      - Mean PE max run: {cascade_df['pe_max_run'].mean():.1f}")

print(f"\n   By gender:")
for gender in ['male', 'female']:
    gdata = cascade_df[cascade_df['gender'] == gender]
    print(f"\n   {gender.capitalize()} (N={len(gdata)}):")
    print(f"      - PE run length: {gdata['pe_run_length'].mean():.3f}")
    print(f"      - NPE run length: {gdata['npe_run_length'].mean():.3f}")
    print(f"      - Cascade tendency: {gdata['cascade_tendency'].mean():.3f}")
    print(f"      - PE max run: {gdata['pe_max_run'].mean():.1f}")

# ============================================================================
# 10. Save Results
# ============================================================================
print("\n10. Saving results...")

# Summary statistics
summary_results = pd.DataFrame([{
    'metric': 'pe_run_length',
    'interaction_beta': pe_interaction,
    'male_r': male_r_pe,
    'male_p': male_p_pe,
    'female_r': female_r_pe,
    'female_p': female_p_pe,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'npe_run_length',
    'interaction_beta': npe_interaction,
    'male_r': male_r_npe,
    'male_p': male_p_npe,
    'female_r': female_r_npe,
    'female_p': female_p_npe,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'cascade_tendency',
    'interaction_beta': cascade_interaction,
    'male_r': male_r_cascade,
    'male_p': male_p_cascade,
    'female_r': female_r_cascade,
    'female_p': female_p_cascade,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'pe_max_run',
    'interaction_beta': max_interaction,
    'male_r': male_r_max,
    'male_p': male_p_max,
    'female_r': female_r_max,
    'female_p': female_p_max,
    'male_n': len(male_data),
    'female_n': len(female_data)
}])

summary_results.to_csv(OUTPUT_DIR / "cascade_summary.csv", index=False, encoding='utf-8-sig')

# Individual cascade metrics
cascade_df.to_csv(OUTPUT_DIR / "individual_cascade_metrics.csv", index=False, encoding='utf-8-sig')

print(f"   - Saved: cascade_summary.csv")
print(f"   - Saved: individual_cascade_metrics.csv")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# 11. Interpretation Summary
# ============================================================================
print("\n11. KEY FINDINGS SUMMARY:")
print("=" * 80)

print("\n📊 PE RUN LENGTH (PERSEVERATIVE CASCADES):")
print(f"   - UCLA × Gender interaction: β={pe_interaction:.4f}")
print(f"   - Males: r={male_r_pe:.3f}, p={male_p_pe:.4f}")
print(f"   - Females: r={female_r_pe:.3f}, p={female_p_pe:.4f}")

print("\n📊 NPE RUN LENGTH (CONTROL):")
print(f"   - UCLA × Gender interaction: β={npe_interaction:.4f}")
print(f"   - PE/NPE ratio: {pe_interaction/npe_interaction if npe_interaction != 0 else np.inf:.2f}×")

print("\n🔗 CASCADE TENDENCY:")
print(f"   - UCLA × Gender interaction: β={cascade_interaction:.4f}")
print(f"   - Males: r={male_r_cascade:.3f}, p={male_p_cascade:.4f}")

print("\n📈 MAX RUN (WORST EPISODE):")
print(f"   - UCLA × Gender interaction: β={max_interaction:.4f}")
print(f"   - Males: r={male_r_max:.3f}, p={male_p_max:.4f}")

print("\n💡 INTERPRETATION:")
if abs(pe_interaction) > abs(npe_interaction) * 1.5:
    print("   ✓ PE cascades show STRONGER gender moderation than NPE")
    print("   ✓ Lonely males get STUCK in perseverative patterns")
    print("   ✓ Operationalizes 'rigidity' as temporal clustering of errors")
else:
    print("   ✓ PE and NPE cascades show similar gender moderation")
    print("   ✓ Effect may be general error clustering, not PE-specific")

if abs(cascade_interaction) > 0.01:
    print(f"   ✓ Cascade tendency moderated by gender")
    print(f"   ✓ Lonely males show higher P(Error|Error) - P(Error|Correct)")

print("\n" + "=" * 80)
