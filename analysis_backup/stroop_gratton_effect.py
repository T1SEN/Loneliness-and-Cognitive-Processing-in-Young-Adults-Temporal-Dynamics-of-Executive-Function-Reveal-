"""
Stroop Gratton Effect Analysis
===============================
Sequential congruency effects: conflict adaptation.

Analyses:
1. cI, cC, iI, iC trial types
2. Adaptation magnitude: (iC - iI)
3. UCLA × Gender on adaptation
4. Comparison to WCST perseveration

Hypothesis:
- Lonely males show reduced conflict adaptation (like WCST perseveration)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_deep_dive/gratton_effect")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STROOP GRATTON EFFECT ANALYSIS")
print("=" * 80)

# Load
print("\n[1] Loading data...")
trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
trials.columns = trials.columns.str.lower()
if 'participantid' in trials.columns and 'participant_id' in trials.columns:
    trials = trials.drop(columns=['participantid'])
elif 'participantid' in trials.columns:
    trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns and 'participant_id' in participants.columns:
    participants = participants.drop(columns=['participantid'])
elif 'participantid' in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
master['gender'] = master['gender'].map(gender_map)

# Clean
print("\n[2] Cleaning trials...")
rt_col = 'rt_ms' if trials['rt'].isnull().sum() > len(trials) * 0.5 else 'rt'
trials_clean = trials[
    (trials['is_timeout'] == False) &
    (trials[rt_col].notna()) &
    (trials[rt_col] > 0) &
    (trials[rt_col] < 10000) &
    (trials['type'].isin(['congruent', 'incongruent']))
].copy()
trials_clean['rt'] = trials_clean[rt_col]

print(f"  Valid trials: {len(trials_clean)}")

# Sequential congruency
print("\n[3] Computing sequential congruency effects...")

def compute_sequential_effects(group):
    """cI, cC, iI, iC."""
    group = group.sort_values('trial').reset_index(drop=True)

    # Lag congruency
    group['prev_type'] = group['type'].shift(1)

    # Define trial types
    ci = group[(group['type'] == 'congruent') & (group['prev_type'] == 'incongruent')]['rt']
    cc = group[(group['type'] == 'congruent') & (group['prev_type'] == 'congruent')]['rt']
    ii = group[(group['type'] == 'incongruent') & (group['prev_type'] == 'incongruent')]['rt']
    ic = group[(group['type'] == 'incongruent') & (group['prev_type'] == 'congruent')]['rt']

    metrics = {}
    if len(ci) >= 5: metrics['rt_ci'] = ci.mean()
    else: metrics['rt_ci'] = np.nan

    if len(cc) >= 5: metrics['rt_cc'] = cc.mean()
    else: metrics['rt_cc'] = np.nan

    if len(ii) >= 5: metrics['rt_ii'] = ii.mean()
    else: metrics['rt_ii'] = np.nan

    if len(ic) >= 5: metrics['rt_ic'] = ic.mean()
    else: metrics['rt_ic'] = np.nan

    return pd.Series(metrics)

sequential_df = trials_clean.groupby('participant_id').apply(compute_sequential_effects).reset_index()

# Adaptation magnitude
sequential_df['adaptation_magnitude'] = sequential_df['rt_ic'] - sequential_df['rt_ii']
sequential_df['congruent_facilitation'] = sequential_df['rt_cc'] - sequential_df['rt_ci']

sequential_df = sequential_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Sequential effects computed for N={len(sequential_df)} participants")
print(f"  Mean adaptation magnitude: {sequential_df['adaptation_magnitude'].mean():.1f} ms")

sequential_df.to_csv(OUTPUT_DIR / "sequential_congruency.csv", index=False, encoding='utf-8-sig')

# Gender-stratified correlations
print("\n[4] Gender-stratified correlations...")

results = []
for gender in ['male', 'female']:
    data = sequential_df[sequential_df['gender'] == gender].dropna(subset=['adaptation_magnitude', 'ucla_total'])
    if len(data) < 10:
        results.append({
            'gender': gender,
            'n': len(data),
            'adaptation_magnitude_r': np.nan,
            'adaptation_magnitude_p': np.nan
        })
        continue

    r, p = pearsonr(data['ucla_total'], data['adaptation_magnitude'])

    results.append({
        'gender': gender,
        'n': len(data),
        'adaptation_magnitude_r': r,
        'adaptation_magnitude_p': p,
        'mean_adaptation': data['adaptation_magnitude'].mean(),
        'sd_adaptation': data['adaptation_magnitude'].std()
    })

gender_corr_df = pd.DataFrame(results)
print(gender_corr_df)

gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

# Regression
print("\n[5] UCLA × Gender regression...")

valid_data = sequential_df.dropna(subset=['adaptation_magnitude'])
valid_data['gender_male'] = (valid_data['gender'] == 'male').astype(int)
for col in ['ucla_total', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']:
    valid_data[f'z_{col}'] = (valid_data[col] - valid_data[col].mean()) / valid_data[col].std()

formula = "adaptation_magnitude ~ z_ucla_total * C(gender_male) + z_age + z_dass_depression + z_dass_anxiety + z_dass_stress"
model = smf.ols(formula, data=valid_data).fit()

print(model.summary())

model_results = pd.DataFrame({
    'term': model.params.index,
    'coef': model.params.values,
    'se': model.bse.values,
    't': model.tvalues.values,
    'p': model.pvalues.values
})

model_results.to_csv(OUTPUT_DIR / "ucla_gender_model.csv", index=False, encoding='utf-8-sig')

# Summary
print("\n" + "="*80)
print("STROOP GRATTON EFFECT - KEY FINDINGS")
print("="*80)

print(f"""
1. Sequential Congruency Effects:
   - cI (congruent after incongruent): {sequential_df['rt_ci'].mean():.1f} ms
   - cC (congruent after congruent): {sequential_df['rt_cc'].mean():.1f} ms
   - iI (incongruent after incongruent): {sequential_df['rt_ii'].mean():.1f} ms
   - iC (incongruent after congruent): {sequential_df['rt_ic'].mean():.1f} ms

2. Adaptation Magnitude (iC - iI):
   - Mean: {sequential_df['adaptation_magnitude'].mean():.1f} ms
   - Males: {sequential_df[sequential_df['gender']=='male']['adaptation_magnitude'].mean():.1f} ms
   - Females: {sequential_df[sequential_df['gender']=='female']['adaptation_magnitude'].mean():.1f} ms

3. Gender Correlations:
{gender_corr_df.to_string(index=False)}

4. Regression Results:
   - UCLA × Gender interaction: p={model.pvalues['z_ucla_total:C(gender_male)[T.1]']:.3f}

5. Comparison to WCST:
   - WCST: Perseveration-specific, males vulnerable
   - Stroop: Adaptation magnitude UCLA × Gender p={model.pvalues['z_ucla_total:C(gender_male)[T.1]']:.3f}

6. Files Generated:
   - sequential_congruency.csv
   - gender_stratified_correlations.csv
   - ucla_gender_model.csv
""")

print("\n✅ Stroop Gratton effect analysis complete!")
print("="*80)
