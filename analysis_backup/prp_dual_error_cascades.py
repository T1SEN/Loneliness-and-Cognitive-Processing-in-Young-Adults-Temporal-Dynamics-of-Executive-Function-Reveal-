"""
PRP Dual-Error Cascades Analysis
=================================
Tests coordination breakdowns in dual-task performance.

Analyses:
1. Dual-error runs (both T1 and T2 incorrect)
2. T2-specific error runs
3. Cross-task error correlation: P(T2_error | T1_error)
4. Post-dual-error recovery

Hypothesis:
- Dual-task coordination failures cluster together
- Lonely males show longer dual-error runs
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
OUTPUT_DIR = Path("results/analysis_outputs/prp_deep_dive/error_cascades")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRP DUAL-ERROR CASCADES ANALYSIS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1] Loading data...")

trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
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

rt_col = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 't2_rt'
trials_clean = trials[
    (trials['t1_correct'].notna()) &
    (trials['t2_correct'].notna()) &
    (trials[rt_col].notna()) &
    (trials[rt_col] > 0) &
    (trials[rt_col] < 10000)
].copy()

trials_clean['t2_rt'] = trials_clean[rt_col]

print(f"  Valid trials: {len(trials_clean)}")

# ============================================================================
# Dual-Error Runs
# ============================================================================
print("\n[3] Computing dual-error runs...")

def compute_dual_error_runs(group):
    """Consecutive trials where BOTH T1 and T2 are incorrect."""
    group = group.sort_values('idx' if 'idx' in group.columns else 'trial_index')
    dual_errors = ((group['t1_correct'] == False) & (group['t2_correct'] == False)).astype(int)

    runs = []
    current_run = 0
    for err in dual_errors:
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
            'mean_dual_error_run': 0,
            'max_dual_error_run': 0,
            'n_dual_error_runs': 0,
            'total_dual_errors': 0
        }

    return {
        'mean_dual_error_run': np.mean(runs),
        'max_dual_error_run': np.max(runs),
        'n_dual_error_runs': len(runs),
        'total_dual_errors': dual_errors.sum()
    }

dual_error_list = []
for pid in trials_clean['participant_id'].unique():
    data = trials_clean[trials_clean['participant_id'] == pid]
    runs = compute_dual_error_runs(data)
    runs['participant_id'] = pid
    dual_error_list.append(runs)

dual_error_df = pd.DataFrame(dual_error_list)
dual_error_df = dual_error_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Dual-error runs computed for N={len(dual_error_df)} participants")
print(f"  Mean max_dual_error_run: {dual_error_df['max_dual_error_run'].mean():.2f}")

dual_error_df.to_csv(OUTPUT_DIR / "dual_error_runs.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# T2-Specific Error Runs
# ============================================================================
print("\n[4] Computing T2-specific error runs...")

def compute_t2_error_runs(group):
    """Consecutive T2 errors."""
    group = group.sort_values('idx' if 'idx' in group.columns else 'trial_index')
    t2_errors = (group['t2_correct'] == False).astype(int)

    runs = []
    current_run = 0
    for err in t2_errors:
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
            'mean_t2_error_run': 0,
            'max_t2_error_run': 0,
            'n_t2_error_runs': 0,
            'total_t2_errors': 0
        }

    return {
        'mean_t2_error_run': np.mean(runs),
        'max_t2_error_run': np.max(runs),
        'n_t2_error_runs': len(runs),
        'total_t2_errors': t2_errors.sum()
    }

t2_error_list = []
for pid in trials_clean['participant_id'].unique():
    data = trials_clean[trials_clean['participant_id'] == pid]
    runs = compute_t2_error_runs(data)
    runs['participant_id'] = pid
    t2_error_list.append(runs)

t2_error_df = pd.DataFrame(t2_error_list)
t2_error_df = t2_error_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  T2-error runs computed for N={len(t2_error_df)} participants")
print(f"  Mean max_t2_error_run: {t2_error_df['max_t2_error_run'].mean():.2f}")

t2_error_df.to_csv(OUTPUT_DIR / "t2_error_runs.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Cross-Task Error Correlation
# ============================================================================
print("\n[5] Computing cross-task error correlation...")

def compute_cross_task_correlation(group):
    """P(T2_error | T1_error) vs P(T2_error | T1_correct)"""
    group = group.sort_values('idx' if 'idx' in group.columns else 'trial_index')

    t1_errors = group[group['t1_correct'] == False]
    t1_corrects = group[group['t1_correct'] == True]

    if len(t1_errors) == 0 or len(t1_corrects) == 0:
        return {
            'p_t2_error_given_t1_error': np.nan,
            'p_t2_error_given_t1_correct': np.nan,
            'cross_task_dependency': np.nan
        }

    p_t2_err_given_t1_err = (~t1_errors['t2_correct']).mean()
    p_t2_err_given_t1_corr = (~t1_corrects['t2_correct']).mean()

    return {
        'p_t2_error_given_t1_error': p_t2_err_given_t1_err,
        'p_t2_error_given_t1_correct': p_t2_err_given_t1_corr,
        'cross_task_dependency': p_t2_err_given_t1_err - p_t2_err_given_t1_corr
    }

cross_task_list = []
for pid in trials_clean['participant_id'].unique():
    data = trials_clean[trials_clean['participant_id'] == pid]
    corr = compute_cross_task_correlation(data)
    corr['participant_id'] = pid
    cross_task_list.append(corr)

cross_task_df = pd.DataFrame(cross_task_list)
cross_task_df = cross_task_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Cross-task correlation computed for N={len(cross_task_df)} participants")
print(f"  Mean cross-task dependency: {cross_task_df['cross_task_dependency'].mean():.3f}")

cross_task_df.to_csv(OUTPUT_DIR / "cross_task_error_correlation.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Gender-Stratified Correlations
# ============================================================================
print("\n[6] Gender-stratified correlations...")

results = []
for gender in ['male', 'female']:
    # Dual errors
    data_dual = dual_error_df[dual_error_df['gender'] == gender]
    if len(data_dual) >= 10:
        r_dual, p_dual = pearsonr(data_dual['ucla_total'], data_dual['max_dual_error_run'])
    else:
        r_dual, p_dual = np.nan, np.nan

    # T2 errors
    data_t2 = t2_error_df[t2_error_df['gender'] == gender]
    if len(data_t2) >= 10:
        r_t2, p_t2 = pearsonr(data_t2['ucla_total'], data_t2['max_t2_error_run'])
    else:
        r_t2, p_t2 = np.nan, np.nan

    # Cross-task dependency
    data_cross = cross_task_df[cross_task_df['gender'] == gender].dropna(subset=['cross_task_dependency'])
    if len(data_cross) >= 10:
        r_cross, p_cross = pearsonr(data_cross['ucla_total'], data_cross['cross_task_dependency'])
    else:
        r_cross, p_cross = np.nan, np.nan

    results.append({
        'gender': gender,
        'n': len(data_dual),
        'max_dual_error_run_r': r_dual,
        'max_dual_error_run_p': p_dual,
        'max_t2_error_run_r': r_t2,
        'max_t2_error_run_p': p_t2,
        'cross_task_dependency_r': r_cross,
        'cross_task_dependency_p': p_cross
    })

gender_corr_df = pd.DataFrame(results)
print(gender_corr_df)

gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("PRP DUAL-ERROR CASCADES - KEY FINDINGS")
print("="*80)

print(f"""
1. Dual-Error Runs (both T1 & T2 incorrect):
   - Mean max run: {dual_error_df['max_dual_error_run'].mean():.2f} trials
   - Males: {dual_error_df[dual_error_df['gender']=='male']['max_dual_error_run'].mean():.2f}
   - Females: {dual_error_df[dual_error_df['gender']=='female']['max_dual_error_run'].mean():.2f}

2. T2-Specific Error Runs:
   - Mean max run: {t2_error_df['max_t2_error_run'].mean():.2f} trials
   - Males: {t2_error_df[t2_error_df['gender']=='male']['max_t2_error_run'].mean():.2f}
   - Females: {t2_error_df[t2_error_df['gender']=='female']['max_t2_error_run'].mean():.2f}

3. Cross-Task Error Dependency:
   - Mean dependency: {cross_task_df['cross_task_dependency'].mean():.3f}
   - (Positive = T2 errors more likely after T1 errors)

4. Gender Correlations:
{gender_corr_df.to_string(index=False)}

5. Files Generated:
   - dual_error_runs.csv
   - t2_error_runs.csv
   - cross_task_error_correlation.csv
   - gender_stratified_correlations.csv
""")

print("\n✅ PRP dual-error cascades analysis complete!")
print("="*80)
