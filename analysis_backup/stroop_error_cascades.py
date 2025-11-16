"""
Stroop Error Cascades Analysis
===============================
Tests whether lonely individuals (especially males) get "stuck" in error/slow response runs.

Analyses:
1. Consecutive error run lengths
2. Slow response runs (RT > median + 1SD)
3. Error clustering: P(Error_t | Error_t-1)
4. Gender-specific patterns

Hypothesis:
- WCST females showed SHORTER PE runs (r=-0.376, p=0.01) - compensation
- Test if Stroop shows same pattern
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# UTF-8
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_deep_dive/error_cascades")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STROOP ERROR CASCADES ANALYSIS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
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

print(f"  Trials: {len(trials)}, Master: {len(master)}")

# ============================================================================
# Clean Trials
# ============================================================================
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

# ============================================================================
# Compute Error Runs
# ============================================================================
print("\n[3] Computing error runs...")

def compute_error_runs(group):
    """Compute consecutive error run lengths."""
    group = group.sort_values('trial')
    errors = (~group['correct']).astype(int)

    # Find runs
    runs = []
    current_run = 0
    for err in errors:
        if err == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    if len(runs) == 0:
        return {
            'mean_error_run': 0,
            'max_error_run': 0,
            'n_error_runs': 0,
            'total_errors': 0
        }

    return {
        'mean_error_run': np.mean(runs),
        'max_error_run': np.max(runs),
        'n_error_runs': len(runs),
        'total_errors': errors.sum()
    }

error_runs_list = []
for pid in trials_clean['participant_id'].unique():
    data = trials_clean[trials_clean['participant_id'] == pid]
    runs = compute_error_runs(data)
    runs['participant_id'] = pid
    error_runs_list.append(runs)

error_runs_df = pd.DataFrame(error_runs_list)
error_runs_df = error_runs_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Error runs computed for N={len(error_runs_df)} participants")
print(f"  Mean max_error_run: {error_runs_df['max_error_run'].mean():.2f}")

error_runs_df.to_csv(OUTPUT_DIR / "error_runs.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Slow Response Runs
# ============================================================================
print("\n[4] Computing slow response runs...")

# Define "slow" as RT > median + 1SD
median_rt = trials_clean['rt'].median()
sd_rt = trials_clean['rt'].std()
slow_threshold = median_rt + sd_rt

print(f"  Slow threshold: {slow_threshold:.1f} ms (median={median_rt:.1f}, SD={sd_rt:.1f})")

def compute_slow_runs(group):
    """Compute consecutive slow response run lengths."""
    group = group.sort_values('trial')
    slow = (group['rt'] > slow_threshold).astype(int)

    runs = []
    current_run = 0
    for s in slow:
        if s == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    if len(runs) == 0:
        return {
            'mean_slow_run': 0,
            'max_slow_run': 0,
            'n_slow_runs': 0,
            'total_slow': 0
        }

    return {
        'mean_slow_run': np.mean(runs),
        'max_slow_run': np.max(runs),
        'n_slow_runs': len(runs),
        'total_slow': slow.sum()
    }

slow_runs_list = []
for pid in trials_clean['participant_id'].unique():
    data = trials_clean[trials_clean['participant_id'] == pid]
    runs = compute_slow_runs(data)
    runs['participant_id'] = pid
    slow_runs_list.append(runs)

slow_runs_df = pd.DataFrame(slow_runs_list)
slow_runs_df = slow_runs_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Slow runs computed for N={len(slow_runs_df)} participants")
print(f"  Mean max_slow_run: {slow_runs_df['max_slow_run'].mean():.2f}")

slow_runs_df.to_csv(OUTPUT_DIR / "slow_runs.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Error Clustering Tendency
# ============================================================================
print("\n[5] Computing error clustering tendency...")

def compute_error_clustering(group):
    """P(Error_t | Error_t-1)"""
    group = group.sort_values('trial')
    errors = (~group['correct']).astype(int).values

    if len(errors) < 2:
        return {'p_error_after_error': np.nan, 'n_errors': 0}

    # Find trials after errors
    error_indices = np.where(errors == 1)[0]
    error_indices_next = error_indices + 1
    error_indices_next = error_indices_next[error_indices_next < len(errors)]

    if len(error_indices_next) == 0:
        return {'p_error_after_error': np.nan, 'n_errors': errors.sum()}

    # P(Error_t | Error_t-1)
    p = errors[error_indices_next].mean()

    return {'p_error_after_error': p, 'n_errors': errors.sum()}

clustering_list = []
for pid in trials_clean['participant_id'].unique():
    data = trials_clean[trials_clean['participant_id'] == pid]
    clust = compute_error_clustering(data)
    clust['participant_id'] = pid
    clustering_list.append(clust)

clustering_df = pd.DataFrame(clustering_list)
clustering_df = clustering_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Clustering computed for N={len(clustering_df)} participants")
print(f"  Mean P(Error|Error): {clustering_df['p_error_after_error'].mean():.3f}")

clustering_df.to_csv(OUTPUT_DIR / "error_clustering.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Gender-Stratified Correlations
# ============================================================================
print("\n[6] Gender-stratified correlations...")

results = []
for gender in ['male', 'female']:
    # Error runs
    data_err = error_runs_df[error_runs_df['gender'] == gender]
    if len(data_err) >= 10:
        r_err, p_err = pearsonr(data_err['ucla_total'], data_err['max_error_run'])
    else:
        r_err, p_err = np.nan, np.nan

    # Slow runs
    data_slow = slow_runs_df[slow_runs_df['gender'] == gender]
    if len(data_slow) >= 10:
        r_slow, p_slow = pearsonr(data_slow['ucla_total'], data_slow['max_slow_run'])
    else:
        r_slow, p_slow = np.nan, np.nan

    # Clustering
    data_clust = clustering_df[clustering_df['gender'] == gender].dropna(subset=['p_error_after_error'])
    if len(data_clust) >= 10:
        r_clust, p_clust = pearsonr(data_clust['ucla_total'], data_clust['p_error_after_error'])
    else:
        r_clust, p_clust = np.nan, np.nan

    results.append({
        'gender': gender,
        'n': len(data_err),
        'max_error_run_r': r_err,
        'max_error_run_p': p_err,
        'max_slow_run_r': r_slow,
        'max_slow_run_p': p_slow,
        'p_error_after_error_r': r_clust,
        'p_error_after_error_p': p_clust
    })

gender_corr_df = pd.DataFrame(results)
print(gender_corr_df)

gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("STROOP ERROR CASCADES - KEY FINDINGS")
print("="*80)

print(f"""
1. Error Runs:
   - Mean max run length: {error_runs_df['max_error_run'].mean():.2f} trials
   - Males: {error_runs_df[error_runs_df['gender']=='male']['max_error_run'].mean():.2f}
   - Females: {error_runs_df[error_runs_df['gender']=='female']['max_error_run'].mean():.2f}

2. Slow Response Runs:
   - Mean max run length: {slow_runs_df['max_slow_run'].mean():.2f} trials
   - Males: {slow_runs_df[slow_runs_df['gender']=='male']['max_slow_run'].mean():.2f}
   - Females: {slow_runs_df[slow_runs_df['gender']=='female']['max_slow_run'].mean():.2f}

3. Error Clustering:
   - Mean P(Error|Error): {clustering_df['p_error_after_error'].mean():.3f}

4. Gender Correlations:
{gender_corr_df.to_string(index=False)}

5. Comparison to WCST:
   - WCST females: r=-0.376, p=0.01 (SHORTER PE runs) ⭐
   - Stroop females: r={gender_corr_df[gender_corr_df['gender']=='female']['max_error_run_r'].values[0]:.3f}, p={gender_corr_df[gender_corr_df['gender']=='female']['max_error_run_p'].values[0]:.3f}

6. Files Generated:
   - error_runs.csv
   - slow_runs.csv
   - error_clustering.csv
   - gender_stratified_correlations.csv
""")

print("\n✅ Stroop error cascades analysis complete!")
print("="*80)
