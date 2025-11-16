"""
Stroop Post-Error Adjustments Analysis
=======================================
Tests adaptive control after errors.

Analyses:
1. Post-error slowing (RT_n+1 after error vs correct)
2. Post-error accuracy
3. Post-error interference (does interference change after errors?)

WCST finding: Post-error accuracy β=-7.682, p=0.069
Tests if Stroop shows similar deficits.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel
import statsmodels.formula.api as smf

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_deep_dive/post_error_adjustments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STROOP POST-ERROR ADJUSTMENTS")
print("=" * 80)

# Load
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
rt_col = 'rt_ms' if trials['rt'].isnull().sum() > len(trials) * 0.5 else 'rt'
trials_clean = trials[
    (trials['is_timeout'] == False) &
    (trials[rt_col].notna()) &
    (trials[rt_col] > 0) &
    (trials[rt_col] < 10000) &
    (trials['type'].isin(['congruent', 'incongruent']))
].copy()
trials_clean['rt'] = trials_clean[rt_col]

print(f"\n[1] Valid trials: {len(trials_clean)}")

# Post-error metrics
def compute_post_error_metrics(group):
    group = group.sort_values('trial').reset_index(drop=True)

    # Lag variables
    group['prev_correct'] = group['correct'].shift(1)
    group['prev_type'] = group['type'].shift(1)

    # Post-error trials
    post_error = group[group['prev_correct'] == False]
    post_correct = group[group['prev_correct'] == True]

    metrics = {}

    # Post-error slowing
    if len(post_error) >= 5 and len(post_correct) >= 5:
        metrics['rt_post_error'] = post_error['rt'].mean()
        metrics['rt_post_correct'] = post_correct['rt'].mean()
        metrics['post_error_slowing'] = post_error['rt'].mean() - post_correct['rt'].mean()
    else:
        metrics['rt_post_error'] = np.nan
        metrics['rt_post_correct'] = np.nan
        metrics['post_error_slowing'] = np.nan

    # Post-error accuracy
    if len(post_error) >= 5:
        metrics['accuracy_post_error'] = post_error['correct'].mean()
    else:
        metrics['accuracy_post_error'] = np.nan

    if len(post_correct) >= 5:
        metrics['accuracy_post_correct'] = post_correct['correct'].mean()
    else:
        metrics['accuracy_post_correct'] = np.nan

    # Post-error interference (incongruent - congruent after errors)
    post_error_incong = group[(group['prev_correct'] == False) & (group['type'] == 'incongruent')]
    post_error_cong = group[(group['prev_correct'] == False) & (group['type'] == 'congruent')]

    if len(post_error_incong) >= 3 and len(post_error_cong) >= 3:
        metrics['interference_post_error'] = post_error_incong['rt'].mean() - post_error_cong['rt'].mean()
    else:
        metrics['interference_post_error'] = np.nan

    return pd.Series(metrics)

post_error_df = trials_clean.groupby('participant_id').apply(compute_post_error_metrics).reset_index()
post_error_df = post_error_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"\n[2] Post-error metrics computed for N={len(post_error_df)}")

post_error_df.to_csv(OUTPUT_DIR / "post_error_metrics.csv", index=False, encoding='utf-8-sig')

# Gender-stratified correlations
results = []
for gender in ['male', 'female']:
    data = post_error_df[post_error_df['gender'] == gender]

    for metric in ['post_error_slowing', 'accuracy_post_error', 'interference_post_error']:
        valid = data.dropna(subset=['ucla_total', metric])
        if len(valid) >= 10:
            r, p = pearsonr(valid['ucla_total'], valid[metric])
            results.append({
                'gender': gender,
                'metric': metric,
                'n': len(valid),
                'r': r,
                'p': p
            })

gender_corr_df = pd.DataFrame(results)
print(f"\n[3] Gender correlations:")
print(gender_corr_df)

gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

# Summary
print("\n" + "="*80)
print("STROOP POST-ERROR ADJUSTMENTS - KEY FINDINGS")
print("="*80)

print(f"""
1. Post-Error Slowing:
   - Mean: {post_error_df['post_error_slowing'].mean():.1f} ms
   - Males: {post_error_df[post_error_df['gender']=='male']['post_error_slowing'].mean():.1f} ms
   - Females: {post_error_df[post_error_df['gender']=='female']['post_error_slowing'].mean():.1f} ms

2. Post-Error Accuracy:
   - Mean: {post_error_df['accuracy_post_error'].mean():.3f}
   - Post-Correct Accuracy: {post_error_df['accuracy_post_correct'].mean():.3f}

3. Gender Correlations:
{gender_corr_df.to_string(index=False)}

4. Files Generated:
   - post_error_metrics.csv
   - gender_stratified_correlations.csv
""")

print("\n✅ Stroop post-error adjustments complete!")
print("="*80)
