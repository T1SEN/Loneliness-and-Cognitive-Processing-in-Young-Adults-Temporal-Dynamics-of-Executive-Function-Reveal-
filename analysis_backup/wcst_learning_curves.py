"""
WCST Learning Curves Analysis
==============================

Purpose: Test whether the UCLA × Gender interaction reflects a LEARNING DEFICIT
         (flat learning curves) vs. TRAIT RIGIDITY (constant across session).

Key Questions:
1. Do lonely males show FLATTER learning curves (less improvement over time)?
2. Is the gender effect stronger EARLY (initial rigidity) or LATE (fatigue)?
3. Do lonely males fail to adapt after feedback across consecutive blocks?

Hypotheses:
- H1: Lonely males show FLAT PE rate trajectories (no learning)
- H2: UCLA × Gender × TrialNumber interaction (moderation varies by time)
- H3: PE rate INCREASES in lonely males across blocks (worsening rigidity)

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
OUTPUT_DIR = Path("results/analysis_outputs/learning_curves")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("WCST Learning Curves Analysis")
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
columns_needed = ['participantId', 'trialIndex', 'blockIndex', 'isPE', 'correct', 'reactionTimeMs']
wcst_valid = wcst_valid[columns_needed].copy()
wcst_valid = wcst_valid.rename(columns={'participantId': 'participant_id'})

# Merge with UCLA and demographics
wcst_valid = wcst_valid.merge(ucla_scores, on='participant_id', how='inner')
wcst_valid = wcst_valid.merge(demo, on='participant_id', how='inner')

# Drop missing UCLA
wcst_valid = wcst_valid.dropna(subset=['ucla_total'])

print(f"   - Valid trials: {len(wcst_valid)}")
print(f"   - Participants: {wcst_valid['participant_id'].nunique()}")
print(f"   - Gender distribution: {wcst_valid['gender'].value_counts().to_dict()}")

# ============================================================================
# 4. Compute Learning Trajectory Metrics
# ============================================================================
print("\n4. Computing learning trajectories...")

# Sort by participant and trial
wcst_valid = wcst_valid.sort_values(['participant_id', 'trialIndex'])

# Normalize trial index within participant (0-1 scale)
def normalize_trial_index(group):
    group = group.copy()
    min_trial = group['trialIndex'].min()
    max_trial = group['trialIndex'].max()
    if max_trial > min_trial:
        group['trial_norm'] = (group['trialIndex'] - min_trial) / (max_trial - min_trial)
    else:
        group['trial_norm'] = 0.5  # Single trial edge case
    return group

wcst_valid = wcst_valid.groupby('participant_id', group_keys=False).apply(normalize_trial_index)

# Create time bins (quartiles)
wcst_valid['quartile'] = pd.cut(wcst_valid['trial_norm'],
                                  bins=[0, 0.25, 0.5, 0.75, 1.0],
                                  labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                  include_lowest=True)

print(f"   - Trials per quartile:")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    n_trials = (wcst_valid['quartile'] == q).sum()
    print(f"     {q}: {n_trials} trials")

# ============================================================================
# 5. Individual Learning Slopes
# ============================================================================
print("\n5. Computing individual learning slopes...")

learning_slopes = []

for pid in wcst_valid['participant_id'].unique():
    pid_data = wcst_valid[wcst_valid['participant_id'] == pid].copy()

    if len(pid_data) < 10:  # Need minimum trials
        continue

    # Get UCLA, gender
    ucla = pid_data['ucla_total'].iloc[0]
    gender = pid_data['gender'].iloc[0]
    age = pid_data['age'].iloc[0]

    # Compute PE rate over time (moving window)
    # Split into 4 bins for stability
    pe_by_quartile = pid_data.groupby('quartile')['isPE'].agg(['mean', 'count']).reset_index()

    if len(pe_by_quartile) < 3:  # Need at least 3 time points
        continue

    # Linear slope: PE ~ TrialBin
    quartile_nums = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    pe_by_quartile['quartile_num'] = pe_by_quartile['quartile'].map(quartile_nums)

    if pe_by_quartile['quartile_num'].notna().sum() >= 3:
        X = pe_by_quartile['quartile_num'].values.reshape(-1, 1)
        y = pe_by_quartile['mean'].values

        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Early vs late PE rate
        q1_pe = pe_by_quartile[pe_by_quartile['quartile'] == 'Q1']['mean'].values
        q4_pe = pe_by_quartile[pe_by_quartile['quartile'] == 'Q4']['mean'].values

        early_pe = q1_pe[0] if len(q1_pe) > 0 else np.nan
        late_pe = q4_pe[0] if len(q4_pe) > 0 else np.nan

        learning_slopes.append({
            'participant_id': pid,
            'ucla_total': ucla,
            'gender': gender,
            'age': age,
            'pe_slope': slope,
            'pe_intercept': intercept,
            'early_pe_rate': early_pe,
            'late_pe_rate': late_pe,
            'pe_change': late_pe - early_pe,  # Positive = worsening
            'n_trials': len(pid_data)
        })

slopes_df = pd.DataFrame(learning_slopes)
print(f"   - Participants with sufficient data: {len(slopes_df)}")

# ============================================================================
# 6. Test UCLA × Gender Interaction on Learning Slopes
# ============================================================================
print("\n6. Testing UCLA × Gender interaction on learning slopes...")

# Code gender
slopes_df['gender_coded'] = slopes_df['gender'].map({'male': 1, 'female': -1})

# Standardize predictors
slopes_df['z_ucla'] = (slopes_df['ucla_total'] - slopes_df['ucla_total'].mean()) / slopes_df['ucla_total'].std()
slopes_df['z_age'] = (slopes_df['age'] - slopes_df['age'].mean()) / slopes_df['age'].std()
slopes_df['ucla_x_gender'] = slopes_df['z_ucla'] * slopes_df['gender_coded']

# Model: PE slope ~ UCLA × Gender + Age
X = slopes_df[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values
y_slope = slopes_df['pe_slope'].values

model = LinearRegression().fit(X, y_slope)
interaction_beta = model.coef_[-1]

print(f"\n   Predicting PE learning slope:")
print(f"      - UCLA × Gender β = {interaction_beta:.4f}")
print(f"      - Interpretation: {'Lonely males show FLATTER slopes (less learning)' if interaction_beta > 0 else 'Lonely males show STEEPER slopes (more learning)'}")

# Stratified correlations
male_data = slopes_df[slopes_df['gender'] == 'male']
female_data = slopes_df[slopes_df['gender'] == 'female']

male_r, male_p = stats.pearsonr(male_data['ucla_total'], male_data['pe_slope'])
female_r, female_p = stats.pearsonr(female_data['ucla_total'], female_data['pe_slope'])

print(f"\n   Correlation: UCLA × PE Slope")
print(f"      - Males (N={len(male_data)}): r={male_r:.3f}, p={male_p:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r:.3f}, p={female_p:.4f}")

# ============================================================================
# 7. Test Early vs Late Performance
# ============================================================================
print("\n7. Testing early vs late session performance...")

# Model: Early PE ~ UCLA × Gender
X_early = slopes_df[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values
y_early = slopes_df['early_pe_rate'].values

model_early = LinearRegression().fit(X_early, y_early)
early_interaction = model_early.coef_[-1]

# Model: Late PE ~ UCLA × Gender
y_late = slopes_df['late_pe_rate'].values
model_late = LinearRegression().fit(X_early, y_late)
late_interaction = model_late.coef_[-1]

print(f"\n   Predicting Early PE rate (Q1):")
print(f"      - UCLA × Gender β = {early_interaction:.4f}")

print(f"\n   Predicting Late PE rate (Q4):")
print(f"      - UCLA × Gender β = {late_interaction:.4f}")

print(f"\n   Comparison:")
print(f"      - Late/Early interaction ratio: {late_interaction/early_interaction:.2f}")
if late_interaction > early_interaction * 1.2:
    print(f"      → Effect STRENGTHENS over time (fatigue/depletion)")
elif early_interaction > late_interaction * 1.2:
    print(f"      → Effect WEAKENS over time (adaptation despite loneliness)")
else:
    print(f"      → Effect is STABLE across session (trait rigidity)")

# ============================================================================
# 8. Block-Level Analysis
# ============================================================================
print("\n8. Analyzing performance across WCST blocks...")

# WCST blocks are defined by rule changes
# Compute PE rate for each block (defined by blockIndex)
wcst_valid['block'] = wcst_valid['blockIndex']

block_summary = []

for pid in wcst_valid['participant_id'].unique():
    pid_data = wcst_valid[wcst_valid['participant_id'] == pid]

    ucla = pid_data['ucla_total'].iloc[0]
    gender = pid_data['gender'].iloc[0]
    age = pid_data['age'].iloc[0]

    # PE rate by block
    block_pe = pid_data.groupby('block')['isPE'].mean().reset_index()
    block_pe.columns = ['block', 'pe_rate']

    for _, row in block_pe.iterrows():
        block_summary.append({
            'participant_id': pid,
            'block': row['block'],
            'pe_rate': row['pe_rate'],
            'ucla_total': ucla,
            'gender': gender,
            'age': age
        })

block_df = pd.DataFrame(block_summary)

print(f"\n   Block-level data: {len(block_df)} rows")
print(f"   Columns: {block_df.columns.tolist()}")

if len(block_df) == 0 or 'gender' not in block_df.columns:
    print("   WARNING: Block-level analysis failed - insufficient data")
    three_way_beta = np.nan
else:
    # Drop rows with missing gender (shouldn't happen but safety check)
    block_df = block_df.dropna(subset=['gender'])

    # Test if UCLA × Gender predicts PE rate differently across blocks
    # Interaction with block number (linear trend)

    block_df['gender_coded'] = block_df['gender'].map({'male': 1, 'female': -1})
    block_df['z_ucla'] = (block_df['ucla_total'] - block_df['ucla_total'].mean()) / block_df['ucla_total'].std()
    block_df['z_block'] = (block_df['block'] - block_df['block'].mean()) / block_df['block'].std()
    block_df['ucla_x_gender'] = block_df['z_ucla'] * block_df['gender_coded']
    block_df['ucla_x_block'] = block_df['z_ucla'] * block_df['z_block']
    block_df['gender_x_block'] = block_df['gender_coded'] * block_df['z_block']
    block_df['three_way'] = block_df['z_ucla'] * block_df['gender_coded'] * block_df['z_block']

    # Model with three-way interaction
    X_block = block_df[['z_ucla', 'gender_coded', 'z_block',
                         'ucla_x_gender', 'ucla_x_block', 'gender_x_block',
                         'three_way']].values
    y_block = block_df['pe_rate'].values

    model_block = LinearRegression().fit(X_block, y_block)
    three_way_beta = model_block.coef_[-1]

    print(f"\n   Three-way interaction (UCLA × Gender × Block):")
    print(f"      - β = {three_way_beta:.4f}")
    print(f"      - Interpretation:")
    if three_way_beta > 0.01:
        print(f"        Lonely males show INCREASING PE across blocks (worsening rigidity)")
    elif three_way_beta < -0.01:
        print(f"        Lonely males show DECREASING PE across blocks (delayed learning)")
    else:
        print(f"        No differential block trajectory by gender")

# ============================================================================
# 9. Descriptive Statistics
# ============================================================================
print("\n9. Descriptive statistics:")

print(f"\n   Overall learning slopes:")
print(f"      - Mean PE slope: {slopes_df['pe_slope'].mean():.4f} (SD={slopes_df['pe_slope'].std():.4f})")
print(f"      - Mean early PE: {slopes_df['early_pe_rate'].mean():.3f}")
print(f"      - Mean late PE: {slopes_df['late_pe_rate'].mean():.3f}")
print(f"      - Mean PE change: {slopes_df['pe_change'].mean():.3f}")

print(f"\n   By gender:")
for gender in ['male', 'female']:
    gdata = slopes_df[slopes_df['gender'] == gender]
    print(f"\n   {gender.capitalize()} (N={len(gdata)}):")
    print(f"      - Mean PE slope: {gdata['pe_slope'].mean():.4f}")
    print(f"      - Early PE: {gdata['early_pe_rate'].mean():.3f}")
    print(f"      - Late PE: {gdata['late_pe_rate'].mean():.3f}")
    print(f"      - PE change: {gdata['pe_change'].mean():.3f}")

# ============================================================================
# 10. Save Results
# ============================================================================
print("\n10. Saving results...")

# Summary statistics
summary_results = pd.DataFrame([{
    'metric': 'pe_slope',
    'interaction_beta': interaction_beta,
    'male_r': male_r,
    'male_p': male_p,
    'female_r': female_r,
    'female_p': female_p,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'early_pe_rate',
    'interaction_beta': early_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'late_pe_rate',
    'interaction_beta': late_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'block_three_way',
    'interaction_beta': three_way_beta,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': len(male_data),
    'female_n': len(female_data)
}])

summary_results.to_csv(OUTPUT_DIR / "learning_slopes_summary.csv", index=False, encoding='utf-8-sig')

# Individual learning slopes
slopes_df.to_csv(OUTPUT_DIR / "individual_learning_slopes.csv", index=False, encoding='utf-8-sig')

# Block-level data
block_df.to_csv(OUTPUT_DIR / "block_level_pe_rates.csv", index=False, encoding='utf-8-sig')

print(f"   - Saved: learning_slopes_summary.csv")
print(f"   - Saved: individual_learning_slopes.csv")
print(f"   - Saved: block_level_pe_rates.csv")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# 11. Interpretation Summary
# ============================================================================
print("\n11. KEY FINDINGS SUMMARY:")
print("=" * 80)

print("\n📊 LEARNING SLOPE ANALYSIS:")
print(f"   - UCLA × Gender interaction on PE slope: β={interaction_beta:.4f}")
print(f"   - Males: r={male_r:.3f}, p={male_p:.4f}")
print(f"   - Females: r={female_r:.3f}, p={female_p:.4f}")

print("\n⏱️  EARLY VS LATE SESSION:")
print(f"   - Early (Q1) interaction: β={early_interaction:.4f}")
print(f"   - Late (Q4) interaction: β={late_interaction:.4f}")
print(f"   - Ratio: {late_interaction/early_interaction:.2f}×")

print("\n📈 BLOCK-LEVEL TRAJECTORY:")
print(f"   - UCLA × Gender × Block three-way: β={three_way_beta:.4f}")

print("\n💡 INTERPRETATION:")
if abs(interaction_beta) > 0.01:
    if interaction_beta > 0:
        print("   ✓ Lonely males show FLATTER learning curves (LESS improvement)")
        print("   ✓ This suggests a LEARNING DEFICIT, not just trait rigidity")
    else:
        print("   ✓ Lonely males show STEEPER learning curves (MORE improvement)")
        print("   ✓ Initial rigidity but capacity for adaptation")
else:
    print("   ✓ No differential learning trajectories by gender")
    print("   ✓ Effect is STABLE across session (trait rigidity)")

if late_interaction > early_interaction * 1.2:
    print("   ✓ Effect STRENGTHENS late in session → Fatigue/depletion mechanism")
elif early_interaction > late_interaction * 1.2:
    print("   ✓ Effect WEAKENS late in session → Initial rigidity overcome")

print("\n" + "=" * 80)
