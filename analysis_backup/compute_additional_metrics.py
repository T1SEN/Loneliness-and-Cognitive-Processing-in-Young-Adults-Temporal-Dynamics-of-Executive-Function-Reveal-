"""
추가 집행기능 메트릭 계산
Compute Additional Executive Function Metrics

This script computes expanded metrics from trial-level data to supplement
the master dataset for comprehensive gender moderation analysis.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")

print("="*80)
print("추가 집행기능 메트릭 계산")
print("Computing Additional Executive Function Metrics")
print("="*80)
print()

# Load master dataset
master = pd.read_csv(OUTPUT_DIR / "master_dataset.csv")
print(f"Master dataset loaded: {len(master)} participants")
print()

# Load trial-level data
print("Loading trial-level data...")
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")

# Normalize participant ID column
for df in [stroop_trials, wcst_trials, prp_trials]:
    if 'participantId' in df.columns:
        if 'participant_id' in df.columns:
            df.drop(columns=['participantId'], inplace=True)
        else:
            df.rename(columns={'participantId': 'participant_id'}, inplace=True)

print(f"  Stroop trials: {len(stroop_trials)}")
print(f"  WCST trials: {len(wcst_trials)}")
print(f"  PRP trials: {len(prp_trials)}")
print()

# ============================================================================
# STROOP ADDITIONAL METRICS
# ============================================================================

print("Computing Stroop additional metrics...")
stroop_add = []

for pid in master['participant_id'].unique():
    trials = stroop_trials[stroop_trials['participant_id'] == pid].copy()

    # Filter valid trials
    trials = trials[(trials['timeout'] == False) & (trials['rt_ms'] > 0)].copy()

    if len(trials) < 10:
        continue

    metrics = {'participant_id': pid}

    # Accuracy by condition
    cong = trials[trials['type'] == 'congruent']
    incong = trials[trials['type'] == 'incongruent']

    metrics['stroop_cong_acc'] = (cong['correct'].sum() / len(cong) * 100) if len(cong) > 0 else np.nan
    metrics['stroop_incong_acc'] = (incong['correct'].sum() / len(incong) * 100) if len(incong) > 0 else np.nan

    # RT variability by condition
    metrics['stroop_cong_cv'] = (cong['rt_ms'].std() / cong['rt_ms'].mean()) if len(cong) > 0 and cong['rt_ms'].mean() > 0 else np.nan
    metrics['stroop_incong_cv'] = (incong['rt_ms'].std() / incong['rt_ms'].mean()) if len(incong) > 0 and incong['rt_ms'].mean() > 0 else np.nan

    stroop_add.append(metrics)

stroop_add_df = pd.DataFrame(stroop_add)
print(f"  ✓ Stroop metrics computed for {len(stroop_add_df)} participants")
print()

# ============================================================================
# WCST ADDITIONAL METRICS
# ============================================================================

print("Computing WCST additional metrics...")
wcst_add = []

for pid in master['participant_id'].unique():
    trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()

    # Filter valid trials
    trials = trials[trials['timeout'] == False].copy()

    if len(trials) < 10:
        continue

    metrics = {'participant_id': pid}

    # Non-perseverative error rate
    if 'isNPE' in trials.columns:
        npe_count = trials['isNPE'].sum()
        metrics['wcst_npe_rate'] = (npe_count / len(trials) * 100)
    else:
        metrics['wcst_npe_rate'] = np.nan

    # Accuracy
    metrics['wcst_accuracy'] = (trials['correct'].sum() / len(trials) * 100) if len(trials) > 0 else np.nan

    # RT variability
    metrics['wcst_rt_cv'] = (trials['rt_ms'].std() / trials['rt_ms'].mean()) if trials['rt_ms'].mean() > 0 else np.nan

    wcst_add.append(metrics)

wcst_add_df = pd.DataFrame(wcst_add)
print(f"  ✓ WCST metrics computed for {len(wcst_add_df)} participants")
print()

# ============================================================================
# PRP ADDITIONAL METRICS
# ============================================================================

print("Computing PRP additional metrics...")
prp_add = []

for pid in master['participant_id'].unique():
    trials = prp_trials[prp_trials['participant_id'] == pid].copy()

    # Filter valid trials
    trials = trials[(trials['t1_timeout'] == False) &
                    (trials['t2_timeout'] == False) &
                    (trials['t1_rt_ms'] > 0) &
                    (trials['t2_rt_ms'] > 0)].copy()

    if len(trials) < 10:
        continue

    metrics = {'participant_id': pid}

    # SOA binning
    def bin_soa(soa):
        if soa <= 150:
            return 'short'
        elif soa >= 1200:
            return 'long'
        else:
            return 'other'

    trials['soa_bin'] = trials['soa_nominal_ms'].apply(bin_soa)

    short = trials[trials['soa_bin'] == 'short']
    long = trials[trials['soa_bin'] == 'long']

    # SOA-specific accuracy
    metrics['prp_short_t2_acc'] = (short['t2_correct'].sum() / len(short) * 100) if len(short) > 0 else np.nan
    metrics['prp_long_t2_acc'] = (long['t2_correct'].sum() / len(long) * 100) if len(long) > 0 else np.nan

    # T1 slowing at short SOA
    short_t1_rt = short['t1_rt_ms'].mean() if len(short) > 0 else np.nan
    long_t1_rt = long['t1_rt_ms'].mean() if len(long) > 0 else np.nan
    metrics['prp_t1_slowing'] = short_t1_rt - long_t1_rt if not (np.isnan(short_t1_rt) or np.isnan(long_t1_rt)) else np.nan

    # PRP slope
    if len(trials) > 20 and trials['soa_nominal_ms'].nunique() >= 3:
        slope, _, _, _, _ = stats.linregress(trials['soa_nominal_ms'], trials['t2_rt_ms'])
        metrics['prp_slope'] = slope
    else:
        metrics['prp_slope'] = np.nan

    prp_add.append(metrics)

prp_add_df = pd.DataFrame(prp_add)
print(f"  ✓ PRP metrics computed for {len(prp_add_df)} participants")
print()

# ============================================================================
# MERGE WITH MASTER AND SAVE
# ============================================================================

print("Merging additional metrics with master dataset...")
master_expanded = master.copy()
master_expanded = master_expanded.merge(stroop_add_df, on='participant_id', how='left')
master_expanded = master_expanded.merge(wcst_add_df, on='participant_id', how='left')
master_expanded = master_expanded.merge(prp_add_df, on='participant_id', how='left')

# Save
output_file = OUTPUT_DIR / "master_expanded_metrics.csv"
master_expanded.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✓ Saved: {output_file}")
print(f"  Rows: {len(master_expanded)}")
print(f"  Columns: {len(master_expanded.columns)}")
print()

# Print summary statistics
print("="*80)
print("SUMMARY OF ADDITIONAL METRICS")
print("="*80)
print()

new_cols = [col for col in master_expanded.columns if col not in master.columns]
print(f"New metrics added: {len(new_cols)}")
for col in new_cols:
    n_valid = master_expanded[col].notna().sum()
    print(f"  {col}: {n_valid}/{len(master_expanded)} participants")

print()
print("="*80)
print("COMPLETE")
print("="*80)
