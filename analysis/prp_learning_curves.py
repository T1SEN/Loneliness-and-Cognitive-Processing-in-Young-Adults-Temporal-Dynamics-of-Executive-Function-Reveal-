"""
PRP Learning Curves Analysis
=============================
Tests whether UCLA × Gender effects on PRP bottleneck change across the session.

Analyses:
1. Bottleneck effect (T2_RT_short - T2_RT_long) by quartile (Q1-Q4)
2. Individual learning slopes
3. UCLA × Gender × Quartile interaction
4. SOA-specific learning curves

Hypotheses:
- H1: Bottleneck should decrease with practice (learning)
- H2: Lonely individuals may show flat curves (no learning)
- H3: Gender-specific temporal patterns
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

# UTF-8 encoding
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/prp_deep_dive/learning_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRP LEARNING CURVES ANALYSIS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1] Loading data...")

# Trials
trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
trials.columns = trials.columns.str.lower()
if 'participantid' in trials.columns and 'participant_id' in trials.columns:
    trials = trials.drop(columns=['participantid'])
elif 'participantid' in trials.columns:
    trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

# Master
master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()

# Demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns and 'participant_id' in participants.columns:
    participants = participants.drop(columns=['participantid'])
elif 'participantid' in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

# Gender mapping
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
master['gender'] = master['gender'].map(gender_map)

print(f"  Trials: {len(trials)} rows, {trials['participant_id'].nunique()} participants")
print(f"  Master: {len(master)} participants")

# ============================================================================
# Clean Trials
# ============================================================================
print("\n[2] Cleaning trials...")

# Use t2_rt_ms if available, otherwise t2_rt
rt_col = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 't2_rt'
print(f"  Using RT column: {rt_col}")

# Filter valid trials (dual-correct, check timeout if available)
filter_conditions = [
    (trials['t1_correct'] == True),
    (trials['t2_correct'] == True),
    (trials[rt_col].notna()),
    (trials[rt_col] > 0),
    (trials[rt_col] < 10000)
]

# Add timeout filters only if columns exist and are not all NaN
if 't1_timeout' in trials.columns and trials['t1_timeout'].notna().any():
    filter_conditions.append(trials['t1_timeout'] == False)
if 't2_timeout' in trials.columns and trials['t2_timeout'].notna().any():
    filter_conditions.append(trials['t2_timeout'] == False)

# Combine all conditions
from functools import reduce
trials_clean = trials[reduce(lambda x, y: x & y, filter_conditions)].copy()

# Rename RT column
trials_clean['t2_rt'] = trials_clean[rt_col]

print(f"  Valid dual-correct trials: {len(trials_clean)}")

# ============================================================================
# SOA Categorization
# ============================================================================
print("\n[3] Categorizing SOA...")

# Use soa_nominal_ms if available
soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in trials_clean.columns else 'soa'
print(f"  Using SOA column: {soa_col}")

# Debug: Check SOA distribution
if soa_col in trials_clean.columns:
    print(f"  SOA values in clean data:")
    print(trials_clean[soa_col].value_counts().sort_index())
else:
    print(f"  ERROR: {soa_col} not in columns: {trials_clean.columns.tolist()}")

# Categorize
trials_clean['soa_cat'] = pd.cut(
    trials_clean[soa_col],
    bins=[-np.inf, 200, 800, np.inf],
    labels=['short', 'medium', 'long']
)

print(f"  SOA categories:")
print(trials_clean.groupby('soa_cat', observed=True).size())

# ============================================================================
# Compute Quartiles
# ============================================================================
print("\n[4] Computing quartiles...")

def assign_quartile(group):
    n = len(group)
    if n < 4:
        # If too few trials, assign Q1 to all
        group = group.copy()
        group['quartile'] = 'Q1'
        return group
    group = group.copy()
    idx_col = 'idx' if 'idx' in group.columns else 'trial_index'
    group = group.sort_values(idx_col)
    # Assign quartile based on position
    group['quartile'] = pd.cut(range(n), bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    return group

trials_clean = trials_clean.groupby('participant_id', group_keys=False).apply(assign_quartile)

print(f"  Quartiles assigned")
print(f"  Columns after quartile assignment: {trials_clean.columns.tolist()[:10]}...")
if 'quartile' in trials_clean.columns:
    print(trials_clean.groupby('quartile', observed=True).size())

# ============================================================================
# Compute Bottleneck by Quartile
# ============================================================================
print("\n[5] Computing bottleneck by quartile...")

# Mean T2 RT by participant × quartile × SOA
rt_summary = trials_clean.groupby(
    ['participant_id', 'quartile', 'soa_cat'], as_index=False, observed=True
)['t2_rt'].mean()

# Pivot
rt_wide = rt_summary.pivot_table(
    index=['participant_id', 'quartile'],
    columns='soa_cat',
    values='t2_rt',
    observed=True
).reset_index()

# Bottleneck = T2_RT_short - T2_RT_long
if 'short' in rt_wide.columns and 'long' in rt_wide.columns:
    rt_wide['bottleneck'] = rt_wide['short'] - rt_wide['long']

print(f"  Bottleneck computed for {len(rt_wide)} participant-quartile combinations")

# Merge demographics
rt_wide = rt_wide.merge(
    master[['participant_id', 'ucla_total', 'gender', 'age',
            'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Final N={rt_wide['participant_id'].nunique()} participants")

# Save
rt_wide.to_csv(OUTPUT_DIR / "bottleneck_by_quartile.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Quartile Summary
# ============================================================================
print("\n[6] Quartile summary statistics...")

quartile_summary = rt_wide.groupby('quartile', observed=True).agg({
    'short': ['mean', 'std'],
    'long': ['mean', 'std'],
    'bottleneck': ['mean', 'std', 'sem']
}).round(2)

print(quartile_summary)

quartile_summary.to_csv(OUTPUT_DIR / "quartile_summary.csv", encoding='utf-8-sig')

# ============================================================================
# Individual Learning Slopes
# ============================================================================
print("\n[7] Computing individual learning slopes...")

quartile_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
rt_wide['quartile_num'] = rt_wide['quartile'].map(quartile_map)

slopes = []
for pid in rt_wide['participant_id'].unique():
    data = rt_wide[rt_wide['participant_id'] == pid]
    if len(data) >= 3:
        slope, intercept, r, p, se = stats.linregress(
            data['quartile_num'], data['bottleneck']
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
print("\n[8] UCLA × Gender × Quartile interaction...")

rt_wide['gender_male'] = (rt_wide['gender'] == 'male').astype(int)

# Z-score
for col in ['ucla_total', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']:
    rt_wide[f'z_{col}'] = (rt_wide[col] - rt_wide[col].mean()) / rt_wide[col].std()

# Model
model_full = smf.mixedlm(
    "bottleneck ~ z_ucla_total * C(quartile) * C(gender_male) + z_age + z_dass_depression + z_dass_anxiety + z_dass_stress",
    data=rt_wide,
    groups=rt_wide['participant_id']
).fit(reml=False)

print("\n" + "="*80)
print("FULL MODEL: UCLA × Quartile × Gender")
print("="*80)
print(model_full.summary())

# Save
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
print("\n[9] Gender-stratified slope analysis...")

gender_slope_corr = []
for gender in ['male', 'female']:
    data = slopes_df[slopes_df['gender'] == gender]
    if len(data) >= 5:
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
# Early vs Late
# ============================================================================
print("\n[10] Early (Q1) vs Late (Q4) effect sizes...")

early_late = []
for gender in ['male', 'female']:
    for quartile in ['Q1', 'Q4']:
        data = rt_wide[(rt_wide['gender'] == gender) & (rt_wide['quartile'] == quartile)]
        if len(data) >= 5:
            r, p = pearsonr(data['ucla_total'], data['bottleneck'])

            from sklearn.linear_model import LinearRegression
            X = data[['ucla_total']].values
            y = data['bottleneck'].values
            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0]

            early_late.append({
                'gender': gender,
                'quartile': quartile,
                'n': len(data),
                'r': r,
                'p': p,
                'beta': beta,
                'mean_bottleneck': data['bottleneck'].mean(),
                'sd_bottleneck': data['bottleneck'].std()
            })

early_late_df = pd.DataFrame(early_late)
print(early_late_df)

early_late_df.to_csv(OUTPUT_DIR / "early_vs_late_effects.csv", index=False, encoding='utf-8-sig')

# Ratio
male_q1 = early_late_df[(early_late_df['gender'] == 'male') & (early_late_df['quartile'] == 'Q1')]['beta'].values
male_q4 = early_late_df[(early_late_df['gender'] == 'male') & (early_late_df['quartile'] == 'Q4')]['beta'].values

if len(male_q1) > 0 and len(male_q4) > 0 and male_q1[0] != 0:
    ratio = male_q4[0] / male_q1[0]
    print(f"\n⭐ Male Q4/Q1 ratio: {ratio:.2f}× (WCST was 3.14×, Stroop was -0.40×)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("PRP LEARNING CURVES - KEY FINDINGS")
print("="*80)

print(f"""
1. Overall Pattern:
   - Q1 (Early): {quartile_summary.loc['Q1', ('bottleneck', 'mean')]:.1f} ms
   - Q4 (Late):  {quartile_summary.loc['Q4', ('bottleneck', 'mean')]:.1f} ms
   - Change: {quartile_summary.loc['Q4', ('bottleneck', 'mean')] - quartile_summary.loc['Q1', ('bottleneck', 'mean')]:.1f} ms

2. Individual Slopes:
   - Mean: {slopes_df['slope'].mean():.3f} ms/quartile
   - SD: {slopes_df['slope'].std():.3f}

3. Gender Patterns:
   {gender_slope_df.to_string(index=False)}

4. Files Generated:
   - bottleneck_by_quartile.csv
   - quartile_summary.csv
   - individual_slopes.csv
   - ucla_gender_quartile_interaction.csv
   - gender_stratified_slopes.csv
   - early_vs_late_effects.csv
""")

print("\n✅ PRP learning curves analysis complete!")
print("="*80)
