"""
Stroop Phase 1 Comprehensive Analysis
======================================
Combines remaining Phase 1 analyses:
- Block fatigue (1-2)
- Timeout/slow trial patterns (1-3)
- Ratio interference (1-4)

Goal: Find overlooked effects not revealed by traditional interference metrics.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_phase1_comprehensive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STROOP PHASE 1 COMPREHENSIVE ANALYSIS")
print("="*80)

# ======================================================================================
# LOAD DATA
# ======================================================================================

stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')

# Normalize participant_id
if 'participantId' in participants.columns:
    if 'participant_id' in participants.columns:
        participants = participants.drop(columns=['participant_id'])
    participants = participants.rename(columns={'participantId': 'participant_id'})

if 'participantId' in surveys.columns:
    if 'participant_id' in surveys.columns:
        surveys = surveys.drop(columns=['participant_id'])
    surveys = surveys.rename(columns={'participantId': 'participant_id'})

# UCLA
ucla_df = surveys[surveys['surveyName'].str.contains('UCLA', case=False, na=False)].copy()
ucla_df = ucla_df.rename(columns={'score': 'ucla_score'})
ucla_df = ucla_df[['participant_id', 'ucla_score']]

# DASS
dass_df = surveys[surveys['surveyName'].str.contains('DASS', na=False)].copy()
dass_pivot = dass_df.pivot_table(
    index='participant_id',
    columns='surveyName',
    values='score',
    aggfunc='first'
).reset_index()
dass_pivot.columns.name = None
if 'dass' in dass_pivot.columns:
    dass_pivot = dass_pivot.rename(columns={'dass': 'dass_total'})

# Master
master = participants[['participant_id', 'age', 'gender']].copy()
master = master.merge(ucla_df, on='participant_id', how='inner')
master = master.merge(dass_pivot, on='participant_id', how='left')

# Filter valid trials
stroop_valid = stroop_trials[
    (stroop_trials['cond'].notna()) &
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 0) &
    (stroop_trials['rt_ms'] < 5000)
].copy()

print(f"\nValid Stroop trials: {len(stroop_valid)}")
print(f"Participants with trials: {stroop_valid['participant_id'].nunique()}")

# ======================================================================================
# PHASE 1-2: BLOCK FATIGUE ANALYSIS
# ======================================================================================

print("\n" + "="*80)
print("PHASE 1-2: BLOCK FATIGUE ANALYSIS")
print("="*80)

# Create blocks from trial_index if block_index not available
if 'block_index' not in stroop_valid.columns or stroop_valid['block_index'].isna().all():
    print("\n⚠ block_index missing, creating blocks from trial_index...")
    # Divide trials into quartiles (Q1-Q4)
    stroop_valid = stroop_valid.sort_values(['participant_id', 'trial_index'])
    stroop_valid['block'] = stroop_valid.groupby('participant_id')['trial_index'].transform(
        lambda x: pd.qcut(x, q=4, labels=[1, 2, 3, 4], duplicates='drop') if len(x) > 4 else 1
    )
else:
    stroop_valid['block'] = stroop_valid['block_index']

print(f"\nBlock distribution:\n{stroop_valid['block'].value_counts().sort_index()}")

# Calculate interference per block
block_interference = []

for pid in stroop_valid['participant_id'].unique():
    pid_data = stroop_valid[stroop_valid['participant_id'] == pid]

    for block in pid_data['block'].dropna().unique():
        block_data = pid_data[pid_data['block'] == block]

        # Condition RTs
        cong_rt = block_data[block_data['cond'] == 'congruent']['rt_ms'].mean()
        incong_rt = block_data[block_data['cond'] == 'incongruent']['rt_ms'].mean()

        if pd.notna(cong_rt) and pd.notna(incong_rt):
            block_interference.append({
                'participant_id': pid,
                'block': block,
                'interference': incong_rt - cong_rt,
                'congruent_rt': cong_rt,
                'incongruent_rt': incong_rt
            })

block_df = pd.DataFrame(block_interference)
print(f"\nBlock-level data: {len(block_df)} rows")

if len(block_df) > 0:
    # Merge with master
    block_df = block_df.merge(master, on='participant_id', how='left')

    # Early vs Late blocks
    median_block = block_df['block'].median()
    block_df['phase'] = block_df['block'].apply(lambda x: 'Late' if x > median_block else 'Early')

    # Z-score UCLA
    block_df['z_ucla'] = (block_df['ucla_score'] - block_df['ucla_score'].mean()) / block_df['ucla_score'].std()

    # Test: UCLA × Gender × Phase
    formula = 'interference ~ z_ucla * C(gender) * C(phase)'
    try:
        model_block = smf.ols(formula, data=block_df.dropna(subset=['interference', 'gender', 'ucla_score'])).fit()
        print(f"\n{model_block.summary()}")

        # Save
        block_summary = pd.DataFrame({
            'term': model_block.params.index,
            'coef': model_block.params.values,
            'p': model_block.pvalues.values
        })
        block_summary.to_csv(OUTPUT_DIR / "block_fatigue_model.csv", index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"⚠ Block model failed: {e}")

    # Gender-stratified slopes
    gender_slopes = []
    for gender in block_df['gender'].unique():
        subset = block_df[block_df['gender'] == gender]
        if len(subset) > 10:
            r, p = stats.pearsonr(subset['block'], subset['interference'])
            gender_slopes.append({'gender': gender, 'r': r, 'p': p, 'N': len(subset)})
            print(f"{gender} slope: r={r:.3f}, p={p:.3f}")

    pd.DataFrame(gender_slopes).to_csv(OUTPUT_DIR / "block_slopes_by_gender.csv", index=False, encoding='utf-8-sig')

else:
    print("\n⚠ WARNING: block_index not found in data!")

# ======================================================================================
# PHASE 1-3: TIMEOUT/SLOW TRIAL ANALYSIS
# ======================================================================================

print("\n" + "="*80)
print("PHASE 1-3: TIMEOUT/SLOW TRIAL ANALYSIS")
print("="*80)

# Define near-timeout threshold (2000ms or 2SD above mean)
rt_mean = stroop_valid['rt_ms'].mean()
rt_sd = stroop_valid['rt_ms'].std()
slow_threshold = max(2000, rt_mean + 2*rt_sd)

print(f"\nSlow trial threshold: {slow_threshold:.0f} ms")

# Slow trial frequency per participant × condition
slow_freq = []

for pid in stroop_valid['participant_id'].unique():
    pid_data = stroop_valid[stroop_valid['participant_id'] == pid]

    for cond in ['congruent', 'incongruent']:
        cond_data = pid_data[pid_data['cond'] == cond]
        if len(cond_data) > 0:
            n_slow = (cond_data['rt_ms'] > slow_threshold).sum()
            slow_rate = n_slow / len(cond_data) * 100

            slow_freq.append({
                'participant_id': pid,
                'condition': cond,
                'slow_rate': slow_rate,
                'n_slow': n_slow,
                'n_total': len(cond_data)
            })

slow_df = pd.DataFrame(slow_freq)
slow_pivot = slow_df.pivot_table(index='participant_id', columns='condition', values='slow_rate').reset_index()
slow_pivot.columns.name = None
slow_pivot['slow_interference'] = slow_pivot['incongruent'] - slow_pivot.get('congruent', 0)

# Merge
slow_pivot = slow_pivot.merge(master, on='participant_id', how='left')
slow_pivot['z_ucla'] = (slow_pivot['ucla_score'] - slow_pivot['ucla_score'].mean()) / slow_pivot['ucla_score'].std()

print(f"\nSlow trial data: {len(slow_pivot)} participants")
print(f"Mean slow rate (incongruent): {slow_pivot['incongruent'].mean():.2f}%")

# Test
formula = 'incongruent ~ z_ucla * C(gender)'
try:
    model_slow = smf.ols(formula, data=slow_pivot.dropna()).fit()
    print(f"\n{model_slow.summary()}")

    slow_summary = pd.DataFrame({
        'term': model_slow.params.index,
        'coef': model_slow.params.values,
        'p': model_slow.pvalues.values
    })
    slow_summary.to_csv(OUTPUT_DIR / "slow_trial_model.csv", index=False, encoding='utf-8-sig')

except Exception as e:
    print(f"⚠ Slow trial model failed: {e}")

# Gender correlations
for gender in slow_pivot['gender'].unique():
    subset = slow_pivot[slow_pivot['gender'] == gender]
    if len(subset) > 5 and subset['incongruent'].std() > 0:
        r, p = stats.pearsonr(subset['ucla_score'], subset['incongruent'])
        print(f"{gender} - slow incongruent: r={r:.3f}, p={p:.3f}")

# ======================================================================================
# PHASE 1-4: RATIO INTERFERENCE
# ======================================================================================

print("\n" + "="*80)
print("PHASE 1-4: RATIO INTERFERENCE")
print("="*80)

# Calculate condition RTs
cond_rts = stroop_valid.groupby(['participant_id', 'cond'])['rt_ms'].mean().unstack(fill_value=np.nan).reset_index()

# Ratio metrics
cond_rts['ratio_interference'] = cond_rts['incongruent'] / cond_rts['congruent']
cond_rts['log_ratio'] = np.log(cond_rts['incongruent'] / cond_rts['congruent'])

# Merge
cond_rts = cond_rts.merge(master, on='participant_id', how='left')
cond_rts['z_ucla'] = (cond_rts['ucla_score'] - cond_rts['ucla_score'].mean()) / cond_rts['ucla_score'].std()

print(f"\nRatio interference data: {len(cond_rts)} participants")
print(f"Mean ratio: {cond_rts['ratio_interference'].mean():.3f}")
print(f"Mean log ratio: {cond_rts['log_ratio'].mean():.3f}")

# Test ratio
formula = 'ratio_interference ~ z_ucla * C(gender)'
try:
    model_ratio = smf.ols(formula, data=cond_rts.dropna()).fit()
    print(f"\n{model_ratio.summary()}")

    ratio_summary = pd.DataFrame({
        'term': model_ratio.params.index,
        'coef': model_ratio.params.values,
        'p': model_ratio.pvalues.values
    })
    ratio_summary.to_csv(OUTPUT_DIR / "ratio_interference_model.csv", index=False, encoding='utf-8-sig')

except Exception as e:
    print(f"⚠ Ratio model failed: {e}")

# Test log ratio
formula = 'log_ratio ~ z_ucla * C(gender)'
try:
    model_log = smf.ols(formula, data=cond_rts.dropna()).fit()
    print(f"\nLog Ratio Model:")
    print(f"{model_log.summary()}")

except Exception as e:
    print(f"⚠ Log ratio model failed: {e}")

# Gender correlations
for gender in cond_rts['gender'].unique():
    subset = cond_rts[cond_rts['gender'] == gender]
    if len(subset) > 5:
        r_ratio, p_ratio = stats.pearsonr(subset['ucla_score'], subset['ratio_interference'])
        r_log, p_log = stats.pearsonr(subset['ucla_score'], subset['log_ratio'])
        print(f"{gender} - ratio: r={r_ratio:.3f}, p={p_ratio:.3f}")
        print(f"{gender} - log_ratio: r={r_log:.3f}, p={p_log:.3f}")

# ======================================================================================
# SUMMARY
# ======================================================================================

print("\n" + "="*80)
print("PHASE 1 COMPREHENSIVE SUMMARY")
print("="*80)

print("\n✓ All Phase 1 analyses complete!")
print(f"✓ Results saved to: {OUTPUT_DIR}")
