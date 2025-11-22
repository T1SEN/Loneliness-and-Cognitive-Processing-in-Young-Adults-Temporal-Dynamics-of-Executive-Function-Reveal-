"""
derive_additional_trial_features.py

Computes additional trial-level features needed for advanced analyses:
1. PRP error cascade metrics (cascade rate, cascade inflation)
2. PRP post-error slowing (PES)
3. WCST post-error slowing (PES)
4. WCST post-switch error rate

Outputs: master_comprehensive_features.csv (extends master_expanded_metrics.csv)
"""

import sys
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# UTF-8 encoding for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_comprehensive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DERIVE ADDITIONAL TRIAL-LEVEL FEATURES")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD TRIAL DATA
# ============================================================================

print("Loading trial-level data...")

# PRP trials
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
prp_trials.columns = prp_trials.columns.str.lower()
if 'participantid' in prp_trials.columns and 'participant_id' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participantid'])
elif 'participantid' in prp_trials.columns:
    prp_trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

# WCST trials
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
wcst_trials.columns = wcst_trials.columns.str.lower()
if 'participantid' in wcst_trials.columns and 'participant_id' in wcst_trials.columns:
    wcst_trials = wcst_trials.drop(columns=['participantid'])
elif 'participantid' in wcst_trials.columns:
    wcst_trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

print(f"  PRP trials: {len(prp_trials):,} rows")
print(f"  WCST trials: {len(wcst_trials):,} rows")
print()

# ============================================================================
# 2. PRP ERROR CASCADE METRICS
# ============================================================================

print("Computing PRP error cascade metrics...")

# Clean PRP trials
prp_clean = prp_trials[
    (prp_trials['t1_correct'].notna()) &
    (prp_trials['t2_correct'].notna()) &
    (prp_trials['t2_rt_ms'].notna()) &
    (prp_trials['t2_rt_ms'] > 200) &
    (prp_trials['t2_rt_ms'] < 5000)
].copy()

# Convert boolean columns if they're strings
if prp_clean['t1_correct'].dtype == 'object':
    prp_clean['t1_correct'] = prp_clean['t1_correct'].map({True: True, 'True': True, False: False, 'False': False, 1: True, 0: False})
if prp_clean['t2_correct'].dtype == 'object':
    prp_clean['t2_correct'] = prp_clean['t2_correct'].map({True: True, 'True': True, False: False, 'False': False, 1: True, 0: False})

# Ensure boolean type
prp_clean['t1_correct'] = prp_clean['t1_correct'].astype(bool)
prp_clean['t2_correct'] = prp_clean['t2_correct'].astype(bool)

# Error cascade: T1 error AND T2 error on same trial
prp_clean['error_cascade'] = (~prp_clean['t1_correct']) & (~prp_clean['t2_correct'])

# Cascade inflation: P(T2_error | T1_error) - P(T2_error | T1_correct)
def compute_cascade_inflation(group):
    """Compute cascade inflation for a participant."""
    t1_err = group[~group['t1_correct']]
    t1_cor = group[group['t1_correct']]

    if len(t1_err) < 3 or len(t1_cor) < 3:
        return np.nan

    p_t2err_given_t1err = (~t1_err['t2_correct']).mean()
    p_t2err_given_t1cor = (~t1_cor['t2_correct']).mean()

    return p_t2err_given_t1err - p_t2err_given_t1cor

# Aggregate cascade metrics by participant
cascade_metrics = prp_clean.groupby('participant_id').agg({
    'error_cascade': 'mean',  # Cascade rate
    't1_correct': 'mean',      # T1 accuracy
    't2_correct': 'mean'       # T2 accuracy
}).rename(columns={
    'error_cascade': 'prp_cascade_rate',
    't1_correct': 'prp_t1_acc',
    't2_correct': 'prp_t2_acc'
}).reset_index()

# Cascade inflation
cascade_inflation = prp_clean.groupby('participant_id').apply(
    compute_cascade_inflation
).reset_index(name='prp_cascade_inflation')

cascade_metrics = cascade_metrics.merge(cascade_inflation, on='participant_id', how='left')

print(f"  Cascade metrics computed for {len(cascade_metrics)} participants")
print(f"  Mean cascade rate: {cascade_metrics['prp_cascade_rate'].mean():.3f}")
print(f"  Mean cascade inflation: {cascade_metrics['prp_cascade_inflation'].mean():.3f}")
print()

# ============================================================================
# 3. PRP POST-ERROR SLOWING (PES)
# ============================================================================

print("Computing PRP post-error slowing...")

# Sort trials by participant and trial index
prp_clean = prp_clean.sort_values(['participant_id', 'trial_index'])

# For each participant, compute PES for T2
def compute_prp_pes(group):
    """Compute post-error slowing for PRP T2 trials."""
    group = group.copy().sort_values('trial_index').reset_index(drop=True)

    if len(group) < 10:
        return pd.Series({'prp_t2_pes': np.nan, 'prp_t2_post_error_rt': np.nan,
                          'prp_t2_post_correct_rt': np.nan})

    # Shift to get previous trial's T2 correctness
    group['prev_t2_correct'] = group['t2_correct'].shift(1)

    # Remove first trial (no previous) and ensure boolean type
    group = group[group['prev_t2_correct'].notna()].copy()
    group['prev_t2_correct'] = group['prev_t2_correct'].astype(bool)

    if len(group) < 5:
        return pd.Series({'prp_t2_pes': np.nan, 'prp_t2_post_error_rt': np.nan,
                          'prp_t2_post_correct_rt': np.nan})

    # RT after error vs after correct
    post_error_mask = group['prev_t2_correct'] == False
    post_correct_mask = group['prev_t2_correct'] == True

    post_error = group.loc[post_error_mask, 't2_rt_ms']
    post_correct = group.loc[post_correct_mask, 't2_rt_ms']

    if len(post_error) < 3 or len(post_correct) < 3:
        return pd.Series({'prp_t2_pes': np.nan, 'prp_t2_post_error_rt': np.nan,
                          'prp_t2_post_correct_rt': np.nan})

    pes = post_error.mean() - post_correct.mean()

    return pd.Series({
        'prp_t2_pes': pes,
        'prp_t2_post_error_rt': post_error.mean(),
        'prp_t2_post_correct_rt': post_correct.mean()
    })

prp_pes = prp_clean.groupby('participant_id').apply(compute_prp_pes).reset_index()

print(f"  PRP PES computed for {prp_pes['prp_t2_pes'].notna().sum()} participants")
print(f"  Mean PRP T2 PES: {prp_pes['prp_t2_pes'].mean():.1f} ms")
print()

# ============================================================================
# 4. WCST POST-ERROR SLOWING (PES)
# ============================================================================

print("Computing WCST post-error slowing...")

# Clean WCST trials
wcst_clean = wcst_trials[
    (wcst_trials['correct'].notna()) &
    (wcst_trials['reactiontimems'].notna()) &
    (wcst_trials['reactiontimems'] > 200) &
    (wcst_trials['reactiontimems'] < 10000)
].copy()

# Rename RT column if needed
if 'reactiontimems' in wcst_clean.columns:
    wcst_clean.rename(columns={'reactiontimems': 'rt_ms'}, inplace=True)
elif 'rt_ms' not in wcst_clean.columns and 'rt' in wcst_clean.columns:
    wcst_clean['rt_ms'] = wcst_clean['rt'] * 1000

# Convert boolean if needed
if wcst_clean['correct'].dtype == 'object':
    wcst_clean['correct'] = wcst_clean['correct'].map({True: True, 'True': True, False: False, 'False': False, 1: True, 0: False})

# Ensure boolean type
wcst_clean['correct'] = wcst_clean['correct'].astype(bool)

# Sort trials
wcst_clean = wcst_clean.sort_values(['participant_id', 'trialindex'])

def compute_wcst_pes(group):
    """Compute post-error slowing for WCST."""
    group = group.copy().sort_values('trialindex').reset_index(drop=True)

    if len(group) < 20:
        return pd.Series({'wcst_pes': np.nan, 'wcst_post_error_rt': np.nan,
                          'wcst_post_correct_rt': np.nan})

    # Shift to get previous trial's correctness
    group['prev_correct'] = group['correct'].shift(1)

    # Remove first trial and ensure boolean type
    group = group[group['prev_correct'].notna()].copy()
    group['prev_correct'] = group['prev_correct'].astype(bool)

    if len(group) < 10:
        return pd.Series({'wcst_pes': np.nan, 'wcst_post_error_rt': np.nan,
                          'wcst_post_correct_rt': np.nan})

    # RT after error vs after correct
    post_error_mask = group['prev_correct'] == False
    post_correct_mask = group['prev_correct'] == True

    post_error = group.loc[post_error_mask, 'rt_ms']
    post_correct = group.loc[post_correct_mask, 'rt_ms']

    if len(post_error) < 3 or len(post_correct) < 3:
        return pd.Series({'wcst_pes': np.nan, 'wcst_post_error_rt': np.nan,
                          'wcst_post_correct_rt': np.nan})

    pes = post_error.mean() - post_correct.mean()

    return pd.Series({
        'wcst_pes': pes,
        'wcst_post_error_rt': post_error.mean(),
        'wcst_post_correct_rt': post_correct.mean()
    })

wcst_pes = wcst_clean.groupby('participant_id').apply(compute_wcst_pes).reset_index()

try:
    if 'wcst_pes' in wcst_pes.columns:
        valid_pes = wcst_pes['wcst_pes'].dropna()
        print(f"  WCST PES computed for {len(valid_pes)} participants")
        if len(valid_pes) > 0:
            mean_pes = pd.to_numeric(valid_pes, errors='coerce').mean()
            if not np.isnan(mean_pes):
                print(f"  Mean WCST PES: {mean_pes:.1f} ms")
    else:
        print("  WCST PES: Column not found")
except Exception as e:
    print(f"  WCST PES: Computation issue - {e}")
print()

# ============================================================================
# 5. WCST POST-SWITCH ERROR RATE
# ============================================================================

print("Computing WCST post-switch error rate...")

# Identify rule switches
def compute_post_switch_errors(group):
    """Compute error rate in first N trials after rule switch."""
    group = group.sort_values('trialindex').reset_index(drop=True)

    if len(group) < 20 or 'ruleatthattime' not in group.columns:
        return pd.Series({'wcst_post_switch_error_rate': np.nan,
                          'wcst_n_switches': 0})

    # Detect rule changes
    group['rule_changed'] = group['ruleatthattime'] != group['ruleatthattime'].shift(1)
    group.loc[0, 'rule_changed'] = False  # First trial is not a switch

    # Extract post-switch trials (first 3 trials after switch)
    post_switch_trials = []

    for idx in group[group['rule_changed']].index:
        # Get next 3 trials
        next_trials = group.loc[idx:idx+2]
        post_switch_trials.append(next_trials)

    if len(post_switch_trials) == 0:
        return pd.Series({'wcst_post_switch_error_rate': np.nan,
                          'wcst_n_switches': 0})

    post_switch = pd.concat(post_switch_trials)

    error_rate = (~post_switch['correct']).mean()
    n_switches = len(post_switch_trials)

    return pd.Series({
        'wcst_post_switch_error_rate': error_rate,
        'wcst_n_switches': n_switches
    })

wcst_post_switch = wcst_clean.groupby('participant_id').apply(
    compute_post_switch_errors
).reset_index()

try:
    if 'wcst_post_switch_error_rate' in wcst_post_switch.columns:
        valid_count = wcst_post_switch['wcst_post_switch_error_rate'].notna().sum()
        print(f"  Post-switch errors computed for {valid_count} participants")
        if valid_count > 0:
            mean_rate = pd.to_numeric(wcst_post_switch['wcst_post_switch_error_rate'], errors='coerce').mean()
            if not np.isnan(mean_rate):
                print(f"  Mean post-switch error rate: {mean_rate:.3f}")
except Exception as e:
    print(f"  Post-switch errors: Computation issue - {e}")
print()

# ============================================================================
# 6. MERGE WITH MASTER DATASET
# ============================================================================

print("Merging with master dataset...")

master = load_master_dataset(use_cache=True)
master.columns = master.columns.str.lower()

# Normalize participant ID
if 'participantid' in master.columns and 'participant_id' not in master.columns:
    master.rename(columns={'participantid': 'participant_id'}, inplace=True)

print(f"  Master dataset: {len(master)} participants")

# Merge all new features
master_comprehensive = master.copy()

# Merge PRP cascade metrics
master_comprehensive = master_comprehensive.merge(
    cascade_metrics, on='participant_id', how='left'
)

# Merge PRP PES
master_comprehensive = master_comprehensive.merge(
    prp_pes, on='participant_id', how='left'
)

# Merge WCST PES
master_comprehensive = master_comprehensive.merge(
    wcst_pes, on='participant_id', how='left'
)

# Merge WCST post-switch errors
master_comprehensive = master_comprehensive.merge(
    wcst_post_switch, on='participant_id', how='left'
)

print(f"  Comprehensive dataset: {len(master_comprehensive)} participants")
print(f"  Total variables: {len(master_comprehensive.columns)}")
print()

# ============================================================================
# 7. SAVE OUTPUT
# ============================================================================

print("Saving comprehensive dataset...")

output_file = RESULTS_DIR / "analysis_outputs/master_comprehensive_features.csv"
master_comprehensive.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"  Saved to: {output_file}")
print()

# Summary of new features
print("=" * 80)
print("NEW FEATURES SUMMARY")
print("=" * 80)
print()

new_features = [
    'prp_cascade_rate', 'prp_cascade_inflation',
    'prp_t2_pes', 'prp_t2_post_error_rt', 'prp_t2_post_correct_rt',
    'wcst_pes', 'wcst_post_error_rt', 'wcst_post_correct_rt',
    'wcst_post_switch_error_rate', 'wcst_n_switches'
]

for feature in new_features:
    if feature in master_comprehensive.columns:
        try:
            valid_data = pd.to_numeric(master_comprehensive[feature], errors='coerce').dropna()
            valid_n = len(valid_data)
            mean_val = valid_data.mean() if valid_n > 0 else np.nan
            std_val = valid_data.std() if valid_n > 0 else np.nan
            print(f"{feature:35s}: N={valid_n:3d}, M={mean_val:8.3f}, SD={std_val:8.3f}")
        except Exception as e:
            print(f"{feature:35s}: Error - {e}")

print()
print("=" * 80)
print("FEATURE DERIVATION COMPLETE")
print("=" * 80)
print()
print(f"Output file: {output_file.name}")
print(f"Total participants: {len(master_comprehensive)}")
print(f"Total variables: {len(master_comprehensive.columns)}")
print()
