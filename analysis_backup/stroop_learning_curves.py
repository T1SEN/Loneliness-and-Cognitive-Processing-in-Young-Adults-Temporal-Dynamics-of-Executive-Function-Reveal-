"""
Stroop Learning Curves Analysis
================================
Tests whether UCLA × Gender effects on Stroop interference change across the session.

Analyses:
1. Interference effect (Incong RT - Cong RT) by quartile (Q1-Q4)
2. Individual learning slopes
3. UCLA × Gender × Quartile interaction
4. Tests: State (fatigue) vs Trait (stable interference)

Hypotheses:
- H1: Late-session fatigue (like WCST 3.14× effect)
- H2: Or practice-induced reduction (adaptation)
- H3: Gender-specific temporal patterns
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

# UTF-8 encoding for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_deep_dive/learning_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STROOP LEARNING CURVES ANALYSIS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1] Loading data...")

# Trials
trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
trials.columns = trials.columns.str.lower()

# Handle duplicate participant_id columns
if 'participantid' in trials.columns and 'participant_id' in trials.columns:
    # Drop participantid if both exist
    trials = trials.drop(columns=['participantid'])
elif 'participantid' in trials.columns:
    trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

# Master dataset for UCLA, gender, DASS
master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()
if 'participantid' in master.columns:
    master.rename(columns={'participantid': 'participant_id'}, inplace=True)

# Load participant info for demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns and 'participant_id' in participants.columns:
    participants = participants.drop(columns=['participantid'])
elif 'participantid' in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

# Merge demographics into master
master = master.merge(
    participants[['participant_id', 'gender', 'age']],
    on='participant_id',
    how='left'
)

n_participants_trials = trials['participant_id'].nunique()
n_participants_master = master['participant_id'].nunique()
print(f"  Trials: {len(trials)} rows, {n_participants_trials} participants")
print(f"  Master: {n_participants_master} participants")

# ============================================================================
# Clean Trials
# ============================================================================
print("\n[2] Cleaning trials...")

# Use rt_ms if rt is mostly null
rt_col = 'rt_ms' if trials['rt'].isnull().sum() > len(trials) * 0.5 else 'rt'
print(f"  Using RT column: {rt_col}")

# Filter valid trials
trials_clean = trials[
    (trials['is_timeout'] == False) &
    (trials[rt_col].notna()) &
    (trials[rt_col] > 0) &
    (trials[rt_col] < 10000) &
    (trials['type'].isin(['congruent', 'incongruent']))
].copy()

# Rename rt_ms to rt for consistency
if rt_col == 'rt_ms':
    trials_clean['rt'] = trials_clean['rt_ms']

print(f"  Valid trials: {len(trials_clean)}")

# ============================================================================
# Compute Quartiles
# ============================================================================
print("\n[3] Computing quartiles...")

def assign_quartile(group):
    """Assign quartile based on trial order within participant."""
    n = len(group)
    group = group.sort_values('trial')
    group['quartile'] = pd.cut(range(n), bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    return group

trials_clean = trials_clean.groupby('participant_id', group_keys=False).apply(assign_quartile)

print(f"  Quartiles assigned")
print(trials_clean.groupby('quartile').size())

# ============================================================================
# Compute Interference by Quartile
# ============================================================================
print("\n[4] Computing interference by quartile...")

# Mean RT by participant × quartile × condition
rt_summary = trials_clean.groupby(
    ['participant_id', 'quartile', 'type'], as_index=False
)['rt'].mean()

# Pivot to wide format
rt_wide = rt_summary.pivot_table(
    index=['participant_id', 'quartile'],
    columns='type',
    values='rt'
).reset_index()

# Interference = Incong - Cong
rt_wide['interference'] = rt_wide['incongruent'] - rt_wide['congruent']

print(f"  Interference computed for {len(rt_wide)} participant-quartile combinations")

# Merge with master
rt_wide = rt_wide.merge(
    master[['participant_id', 'ucla_total', 'gender', 'age',
            'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

final_n = rt_wide['participant_id'].nunique()
print(f"  Final N={final_n} participants")

# Save
rt_wide.to_csv(OUTPUT_DIR / "interference_by_quartile.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Quartile Summary Stats
# ============================================================================
print("\n[5] Quartile summary statistics...")

quartile_summary = rt_wide.groupby('quartile').agg({
    'congruent': ['mean', 'std'],
    'incongruent': ['mean', 'std'],
    'interference': ['mean', 'std', 'sem']
}).round(2)

print(quartile_summary)

quartile_summary.to_csv(OUTPUT_DIR / "quartile_summary.csv", encoding='utf-8-sig')

# ============================================================================
# Individual Learning Slopes
# ============================================================================
print("\n[6] Computing individual learning slopes...")

# Map quartile to numeric
quartile_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
rt_wide['quartile_num'] = rt_wide['quartile'].map(quartile_map)

# Compute slope for each participant
slopes = []
for pid in rt_wide['participant_id'].unique():
    data = rt_wide[rt_wide['participant_id'] == pid]
    if len(data) >= 3:  # Need at least 3 quartiles
        slope, intercept, r, p, se = stats.linregress(
            data['quartile_num'], data['interference']
        )
        slopes.append({
            'participant_id': pid,
            'slope': slope,
            'intercept': intercept,
            'r': r,
            'p': p,
            'se': se
        })

slopes_df = pd.DataFrame(slopes)
slopes_df = slopes_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'age',
            'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Slopes computed for N={len(slopes_df)} participants")
print(f"  Mean slope: {slopes_df['slope'].mean():.3f} ms/quartile")

slopes_df.to_csv(OUTPUT_DIR / "individual_slopes.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# UCLA × Gender × Quartile Interaction
# ============================================================================
print("\n[7] UCLA × Gender × Quartile interaction...")

# Create gender binary
rt_wide['gender_male'] = (rt_wide['gender'] == 'male').astype(int)

# Z-score predictors
for col in ['ucla_total', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']:
    rt_wide[f'z_{col}'] = (rt_wide[col] - rt_wide[col].mean()) / rt_wide[col].std()

# Model: Interference ~ z_ucla * C(quartile) * C(gender_male) + covariates
model_full = smf.mixedlm(
    "interference ~ z_ucla_total * C(quartile) * C(gender_male) + z_age + z_dass_depression + z_dass_anxiety + z_dass_stress",
    data=rt_wide,
    groups=rt_wide['participant_id']
).fit(reml=False)

print("\n" + "="*80)
print("FULL MODEL: UCLA × Quartile × Gender")
print("="*80)
print(model_full.summary())

# Save model results
model_results = pd.DataFrame({
    'term': model_full.params.index,
    'coef': model_full.params.values,
    'se': model_full.bse.values,
    'z': model_full.tvalues.values,
    'p': model_full.pvalues.values,
    'ci_lower': model_full.conf_int()[0].values,
    'ci_upper': model_full.conf_int()[1].values
})

model_results.to_csv(OUTPUT_DIR / "ucla_gender_quartile_interaction.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Gender-Stratified Slopes
# ============================================================================
print("\n[8] Gender-stratified slope analysis...")

# Debug: check if gender is in slopes_df
print(f"  Slopes_df columns: {slopes_df.columns.tolist()}")
print(f"  Slopes_df shape: {slopes_df.shape}")
if 'gender' in slopes_df.columns:
    print(f"  Gender distribution: {slopes_df['gender'].value_counts()}")

# Map Korean gender to English
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
slopes_df['gender'] = slopes_df['gender'].map(gender_map)

# Correlate UCLA with slope separately by gender
gender_slope_corr = []

for gender in ['male', 'female']:
    if 'gender' not in slopes_df.columns:
        print(f"  WARNING: gender column not found in slopes_df!")
        break
    data = slopes_df[slopes_df['gender'] == gender]
    print(f"  {gender}: N={len(data)}")
    if len(data) >= 5:  # Lowered threshold from 10
        r, p = pearsonr(data['ucla_total'], data['slope'])
        gender_slope_corr.append({
            'gender': gender,
            'n': len(data),
            'r': r,
            'p': p,
            'mean_slope': data['slope'].mean(),
            'sd_slope': data['slope'].std()
        })

gender_slope_df = pd.DataFrame(gender_slope_corr)
print(gender_slope_df)

gender_slope_df.to_csv(OUTPUT_DIR / "gender_stratified_slopes.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Early vs Late Effect Sizes
# ============================================================================
print("\n[9] Early (Q1) vs Late (Q4) effect sizes...")

# Debug: check gender in rt_wide
print(f"  rt_wide columns: {rt_wide.columns.tolist()}")
if 'gender' in rt_wide.columns:
    print(f"  Gender distribution: {rt_wide['gender'].value_counts()}")

# Map Korean gender to English
rt_wide['gender'] = rt_wide['gender'].map(gender_map)

early_late = []
for gender in ['male', 'female']:
    for quartile in ['Q1', 'Q4']:
        if 'gender' not in rt_wide.columns:
            print(f"  WARNING: gender column not found in rt_wide!")
            break
        data = rt_wide[(rt_wide['gender'] == gender) & (rt_wide['quartile'] == quartile)]
        print(f"  {gender} {quartile}: N={len(data)}")
        if len(data) >= 5:  # Lowered threshold
            r, p = pearsonr(data['ucla_total'], data['interference'])

            # Simple regression for beta
            X = data[['ucla_total']].values
            y = data['interference'].values
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0]

            early_late.append({
                'gender': gender,
                'quartile': quartile,
                'n': len(data),
                'r': r,
                'p': p,
                'beta': beta,
                'mean_interference': data['interference'].mean(),
                'sd_interference': data['interference'].std()
            })

early_late_df = pd.DataFrame(early_late)
print(early_late_df)

early_late_df.to_csv(OUTPUT_DIR / "early_vs_late_effects.csv", index=False, encoding='utf-8-sig')

# Compute ratio (like WCST 3.14×)
male_q1 = early_late_df[(early_late_df['gender'] == 'male') & (early_late_df['quartile'] == 'Q1')]['beta'].values
male_q4 = early_late_df[(early_late_df['gender'] == 'male') & (early_late_df['quartile'] == 'Q4')]['beta'].values

if len(male_q1) > 0 and len(male_q4) > 0:
    if male_q1[0] != 0:
        ratio = male_q4[0] / male_q1[0]
        print(f"\n⭐ Male Q4/Q1 ratio: {ratio:.2f}× (WCST was 3.14×)")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*80)
print("STROOP LEARNING CURVES - KEY FINDINGS")
print("="*80)

print(f"""
1. Overall Pattern:
   - Q1 (Early): {quartile_summary.loc['Q1', ('interference', 'mean')]:.1f} ms
   - Q4 (Late):  {quartile_summary.loc['Q4', ('interference', 'mean')]:.1f} ms
   - Change: {quartile_summary.loc['Q4', ('interference', 'mean')] - quartile_summary.loc['Q1', ('interference', 'mean')]:.1f} ms

2. Individual Slopes:
   - Mean: {slopes_df['slope'].mean():.3f} ms/quartile
   - SD: {slopes_df['slope'].std():.3f}
   - Range: [{slopes_df['slope'].min():.3f}, {slopes_df['slope'].max():.3f}]

3. Gender Patterns:
   - Males: Mean slope = {gender_slope_df[gender_slope_df['gender']=='male']['mean_slope'].values[0]:.3f} (r={gender_slope_df[gender_slope_df['gender']=='male']['r'].values[0]:.3f}, p={gender_slope_df[gender_slope_df['gender']=='male']['p'].values[0]:.3f})
   - Females: Mean slope = {gender_slope_df[gender_slope_df['gender']=='female']['mean_slope'].values[0]:.3f} (r={gender_slope_df[gender_slope_df['gender']=='female']['r'].values[0]:.3f}, p={gender_slope_df[gender_slope_df['gender']=='female']['p'].values[0]:.3f})

4. Files Generated:
   - interference_by_quartile.csv
   - quartile_summary.csv
   - individual_slopes.csv
   - ucla_gender_quartile_interaction.csv
   - gender_stratified_slopes.csv
   - early_vs_late_effects.csv
""")

print("\n✅ Stroop learning curves analysis complete!")
print("="*80)
