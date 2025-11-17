"""
PRP Post-Error Adjustments Analysis
====================================
Tests adaptive control after dual-task errors.

Analyses:
1. Post-T1-error effects on T2
2. Post-T2-error slowing on next trial
3. Post-dual-error adjustments
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/prp_deep_dive/post_error_adjustments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRP POST-ERROR ADJUSTMENTS")
print("=" * 80)

# Load
trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
trials.columns = trials.columns.str.lower()
if 'participantid' in trials.columns and 'participant_id' in trials.columns:
    trials = trials.drop(columns=['participantid'])
elif 'participantid' in trials.columns:
    trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()
# Map Korean gender values to English
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
master['gender'] = master['gender'].map(gender_map)

# Clean
rt_col_t1 = 't1_rt_ms' if 't1_rt_ms' in trials.columns else 't1_rt'
rt_col_t2 = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 't2_rt'

trials_clean = trials[
    (trials['t1_correct'].notna()) &
    (trials['t2_correct'].notna()) &
    (trials[rt_col_t1].notna()) &
    (trials[rt_col_t2].notna()) &
    (trials[rt_col_t1] > 0) &
    (trials[rt_col_t2] > 0)
].copy()

trials_clean['t1_rt'] = trials_clean[rt_col_t1]
trials_clean['t2_rt'] = trials_clean[rt_col_t2]

print(f"\n[1] Valid trials: {len(trials_clean)}")

# Post-error metrics
def compute_post_error_metrics(group):
    group = group.sort_values('idx' if 'idx' in group.columns else 'trial_index').reset_index(drop=True)

    # Lag
    group['prev_t1_correct'] = group['t1_correct'].shift(1)
    group['prev_t2_correct'] = group['t2_correct'].shift(1)

    metrics = {}

    # Post-T1-error effect on current T2
    t1_error_trials = group[group['prev_t1_correct'] == False]
    t1_correct_trials = group[group['prev_t1_correct'] == True]

    if len(t1_error_trials) >= 5 and len(t1_correct_trials) >= 5:
        metrics['t2_rt_after_t1_error'] = t1_error_trials['t2_rt'].mean()
        metrics['t2_rt_after_t1_correct'] = t1_correct_trials['t2_rt'].mean()
        metrics['t2_slowing_after_t1_error'] = t1_error_trials['t2_rt'].mean() - t1_correct_trials['t2_rt'].mean()
    else:
        metrics['t2_rt_after_t1_error'] = np.nan
        metrics['t2_rt_after_t1_correct'] = np.nan
        metrics['t2_slowing_after_t1_error'] = np.nan

    # Post-T2-error slowing
    t2_error_trials = group[group['prev_t2_correct'] == False]
    t2_correct_trials = group[group['prev_t2_correct'] == True]

    if len(t2_error_trials) >= 5 and len(t2_correct_trials) >= 5:
        metrics['t2_rt_after_t2_error'] = t2_error_trials['t2_rt'].mean()
        metrics['t2_rt_after_t2_correct'] = t2_correct_trials['t2_rt'].mean()
        metrics['t2_slowing_after_t2_error'] = t2_error_trials['t2_rt'].mean() - t2_correct_trials['t2_rt'].mean()
    else:
        metrics['t2_rt_after_t2_error'] = np.nan
        metrics['t2_rt_after_t2_correct'] = np.nan
        metrics['t2_slowing_after_t2_error'] = np.nan

    # Dual-error
    dual_error_trials = group[(group['prev_t1_correct'] == False) & (group['prev_t2_correct'] == False)]
    if len(dual_error_trials) >= 3:
        metrics['t2_rt_after_dual_error'] = dual_error_trials['t2_rt'].mean()
        metrics['n_dual_errors'] = len(dual_error_trials)
    else:
        metrics['t2_rt_after_dual_error'] = np.nan
        metrics['n_dual_errors'] = 0

    return pd.Series(metrics)

post_error_df = trials_clean.groupby('participant_id').apply(compute_post_error_metrics).reset_index()
post_error_df = post_error_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"\n[2] Post-error metrics computed for N={len(post_error_df)}")

post_error_df.to_csv(OUTPUT_DIR / "post_error_metrics.csv", index=False, encoding='utf-8-sig')

# Gender correlations
results = []
for gender in ['male', 'female']:
    data = post_error_df[post_error_df['gender'] == gender]

    for metric in ['t2_slowing_after_t1_error', 't2_slowing_after_t2_error']:
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
print("PRP POST-ERROR ADJUSTMENTS - KEY FINDINGS")
print("="*80)

print(f"""
1. Post-T1-Error Effects on T2:
   - Mean slowing: {post_error_df['t2_slowing_after_t1_error'].mean():.1f} ms

2. Post-T2-Error Slowing:
   - Mean slowing: {post_error_df['t2_slowing_after_t2_error'].mean():.1f} ms

3. Gender Correlations:
{gender_corr_df.to_string(index=False)}

4. Files Generated:
   - post_error_metrics.csv
   - gender_stratified_correlations.csv
""")

print("\n✅ PRP post-error adjustments complete!")
print("="*80)
