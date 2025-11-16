"""
Perseveration Momentum Analysis - Consecutive PE Runs
======================================================
Distinguishes true perseverative rigidity (stuck in mental set) from random errors

Research Questions:
1. Do lonely males show longer runs of consecutive PEs?
2. Is there autocorrelation in PE sequences (PE[t] → PE[t+1])?
3. Which metric is stronger: run length vs PE rate?
4. Does UCLA predict run clustering better than overall PE rate?

Expected: Vulnerable males show high autocorrelation (true rigidity)
          vs random PE distribution (general inattention)
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

from data_loader_utils import load_master_dataset

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/perseveration_momentum")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("PERSEVERATION MOMENTUM ANALYSIS")
print("=" * 80)
print("\nResearch Question: Do PEs cluster into consecutive runs (true rigidity)?")
print("  - Analyze run lengths of consecutive PEs")
print("  - Compute lag-1 autocorrelation")
print("  - Compare run metrics vs PE rate\n")

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load trial-level WCST data
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

# Normalize participant ID
if 'participant_id' in wcst_trials.columns:
    wcst_trials = wcst_trials.drop(columns=['participant_id'])
if 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

# Parse extra field for PE
import ast
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_dict'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False)).astype(int)

print(f"  Loaded {len(wcst_trials)} WCST trials")

# Load master dataset
master = load_master_dataset()

# Merge
wcst_trials = wcst_trials.merge(master[['participant_id', 'ucla_total', 'gender_male', 'age']],
                                  on='participant_id', how='left')

wcst_trials = wcst_trials.dropna(subset=['ucla_total', 'gender_male']).copy()

print(f"  After merging: {len(wcst_trials)} trials from {wcst_trials['participant_id'].nunique()} participants\n")

# ============================================================================
# 1. Run Length Analysis
# ============================================================================

print("=" * 80)
print("ANALYSIS 1: PE RUN LENGTHS")
print("=" * 80)

def compute_run_lengths(pe_sequence):
    """Compute lengths of consecutive PE runs"""
    if len(pe_sequence) == 0:
        return []

    runs = []
    current_run = 0

    for is_pe in pe_sequence:
        if is_pe == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0

    # Add final run if sequence ends with PE
    if current_run > 0:
        runs.append(current_run)

    return runs

# Compute run metrics for each participant
run_metrics = []

for pid in wcst_trials['participant_id'].unique():
    trials_p = wcst_trials[wcst_trials['participant_id'] == pid].sort_values('trialIndex')

    pe_sequence = trials_p['is_pe'].values
    runs = compute_run_lengths(pe_sequence)

    if len(trials_p) >= 20:
        metrics = {
            'participant_id': pid,
            'n_trials': len(trials_p),
            'n_pe': pe_sequence.sum(),
            'pe_rate': (pe_sequence.sum() / len(trials_p)) * 100,
            'n_runs': len(runs),
            'max_run': max(runs) if runs else 0,
            'mean_run': np.mean(runs) if runs else 0,
            'pct_trials_in_runs_2plus': (sum([r for r in runs if r >= 2]) / pe_sequence.sum() * 100) if pe_sequence.sum() > 0 else 0
        }

        # Add participant-level data
        p_info = master[master['participant_id'] == pid]
        if len(p_info) > 0:
            metrics['ucla_total'] = p_info['ucla_total'].values[0]
            metrics['gender_male'] = p_info['gender_male'].values[0]

        run_metrics.append(metrics)

run_df = pd.DataFrame(run_metrics)

print(f"\nComputed run metrics for {len(run_df)} participants\n")

# Group comparison
print("Run Metrics by Gender × UCLA:")
print("-" * 60)

ucla_median = run_df['ucla_total'].median()

for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
    print(f"\n{gender_name.upper()}:")

    for ucla_group in ['High UCLA', 'Low UCLA']:
        if ucla_group == 'High UCLA':
            subset = run_df[(run_df['gender_male'] == gender_val) &
                            (run_df['ucla_total'] > ucla_median)]
        else:
            subset = run_df[(run_df['gender_male'] == gender_val) &
                            (run_df['ucla_total'] <= ucla_median)]

        if len(subset) >= 3:
            print(f"  {ucla_group:12s}: N={len(subset):2d}, "
                  f"Max run={subset['max_run'].mean():.1f}, "
                  f"Mean run={subset['mean_run'].mean():.2f}, "
                  f"% in runs≥2: {subset['pct_trials_in_runs_2plus'].mean():.1f}%")

# Statistical test: Lonely vs Non-lonely Males
lonely_males = run_df[(run_df['gender_male'] == 1) & (run_df['ucla_total'] > ucla_median)]
nonlonely_males = run_df[(run_df['gender_male'] == 1) & (run_df['ucla_total'] <= ucla_median)]

if len(lonely_males) >= 3 and len(nonlonely_males) >= 3:
    print(f"\n\nStatistical Tests (Lonely vs Non-lonely Males):")
    print("-" * 60)

    # Max run
    t_max, p_max = stats.ttest_ind(lonely_males['max_run'], nonlonely_males['max_run'])
    print(f"  Max run length: t = {t_max:.3f}, p = {p_max:.4f}{'  ⭐' if p_max < 0.05 else ''}")

    # Mean run
    t_mean, p_mean = stats.ttest_ind(lonely_males['mean_run'], nonlonely_males['mean_run'])
    print(f"  Mean run length: t = {t_mean:.3f}, p = {p_mean:.4f}{'  ⭐' if p_mean < 0.05 else ''}")

    # % in runs ≥2
    t_pct, p_pct = stats.ttest_ind(lonely_males['pct_trials_in_runs_2plus'],
                                    nonlonely_males['pct_trials_in_runs_2plus'])
    print(f"  % PEs in runs ≥2: t = {t_pct:.3f}, p = {p_pct:.4f}{'  ⭐' if p_pct < 0.05 else ''}")

# Save run metrics
run_file = OUTPUT_DIR / "pe_run_metrics.csv"
run_df.to_csv(run_file, index=False, encoding='utf-8-sig')
print(f"\n\nSaved run metrics: {run_file}")

# ============================================================================
# 2. Autocorrelation Analysis
# ============================================================================

print("\n\n" + "=" * 80)
print("ANALYSIS 2: PE AUTOCORRELATION")
print("=" * 80)

def compute_lag1_autocorr(pe_sequence):
    """Compute lag-1 autocorrelation of PE sequence"""
    if len(pe_sequence) < 10:
        return np.nan

    pe_t = pe_sequence[:-1]
    pe_t1 = pe_sequence[1:]

    if np.std(pe_t) == 0 or np.std(pe_t1) == 0:
        return np.nan

    r, p = stats.pearsonr(pe_t, pe_t1)
    return r

# Compute autocorrelation for each participant
autocorr_list = []

for pid in wcst_trials['participant_id'].unique():
    trials_p = wcst_trials[wcst_trials['participant_id'] == pid].sort_values('trialIndex')

    pe_sequence = trials_p['is_pe'].values

    if len(trials_p) >= 20:
        autocorr = compute_lag1_autocorr(pe_sequence)

        p_info = master[master['participant_id'] == pid]
        if len(p_info) > 0:
            autocorr_list.append({
                'participant_id': pid,
                'autocorr_lag1': autocorr,
                'ucla_total': p_info['ucla_total'].values[0],
                'gender_male': p_info['gender_male'].values[0]
            })

autocorr_df = pd.DataFrame(autocorr_list)
autocorr_df = autocorr_df.dropna(subset=['autocorr_lag1'])

print(f"\nComputed autocorrelation for {len(autocorr_df)} participants\n")

# Group comparison
print("Autocorrelation by Gender × UCLA:")
print("-" * 60)

for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
    print(f"\n{gender_name.upper()}:")

    for ucla_group in ['High UCLA', 'Low UCLA']:
        if ucla_group == 'High UCLA':
            subset = autocorr_df[(autocorr_df['gender_male'] == gender_val) &
                                 (autocorr_df['ucla_total'] > ucla_median)]
        else:
            subset = autocorr_df[(autocorr_df['gender_male'] == gender_val) &
                                 (autocorr_df['ucla_total'] <= ucla_median)]

        if len(subset) >= 3:
            mean_r = subset['autocorr_lag1'].mean()
            sem_r = subset['autocorr_lag1'].sem()
            print(f"  {ucla_group:12s}: N={len(subset):2d}, r_lag1 = {mean_r:+.3f} (SEM = {sem_r:.3f})")

# Statistical test
lonely_males_ac = autocorr_df[(autocorr_df['gender_male'] == 1) &
                               (autocorr_df['ucla_total'] > ucla_median)]
nonlonely_males_ac = autocorr_df[(autocorr_df['gender_male'] == 1) &
                                  (autocorr_df['ucla_total'] <= ucla_median)]

if len(lonely_males_ac) >= 3 and len(nonlonely_males_ac) >= 3:
    t_ac, p_ac = stats.ttest_ind(lonely_males_ac['autocorr_lag1'],
                                  nonlonely_males_ac['autocorr_lag1'])
    print(f"\n\nStatistical Test (Lonely vs Non-lonely Males):")
    print(f"  Autocorrelation: t = {t_ac:.3f}, p = {p_ac:.4f}{'  ⭐' if p_ac < 0.05 else ''}")

    if p_ac < 0.05:
        print("\n  → Lonely males show HIGHER autocorrelation (true perseverative rigidity)")

# Save autocorrelation
autocorr_file = OUTPUT_DIR / "pe_autocorrelation.csv"
autocorr_df.to_csv(autocorr_file, index=False, encoding='utf-8-sig')
print(f"\n\nSaved autocorrelation: {autocorr_file}")

# ============================================================================
# 3. UCLA Correlation: Run Length vs PE Rate
# ============================================================================

print("\n\n" + "=" * 80)
print("ANALYSIS 3: COMPARE PREDICTIVE POWER")
print("=" * 80)

# Merge run metrics with autocorrelation
combined = run_df.merge(autocorr_df[['participant_id', 'autocorr_lag1']],
                        on='participant_id', how='left')

print("\nMales: UCLA correlations")
print("-" * 60)

males_combined = combined[combined['gender_male'] == 1].copy()

if len(males_combined) >= 10:
    # PE rate
    r_rate, p_rate = stats.pearsonr(males_combined['ucla_total'], males_combined['pe_rate'])
    print(f"  UCLA × PE rate:        r = {r_rate:+.3f}, p = {p_rate:.4f}")

    # Max run
    r_max, p_max = stats.pearsonr(males_combined['ucla_total'], males_combined['max_run'])
    print(f"  UCLA × Max run:        r = {r_max:+.3f}, p = {p_max:.4f}")

    # Mean run
    r_mean, p_mean = stats.pearsonr(males_combined['ucla_total'], males_combined['mean_run'])
    print(f"  UCLA × Mean run:       r = {r_mean:+.3f}, p = {p_mean:.4f}")

    # Autocorrelation
    males_ac = males_combined.dropna(subset=['autocorr_lag1'])
    if len(males_ac) >= 10:
        r_ac, p_ac = stats.pearsonr(males_ac['ucla_total'], males_ac['autocorr_lag1'])
        print(f"  UCLA × Autocorr:       r = {r_ac:+.3f}, p = {p_ac:.4f}")

    # Determine strongest predictor
    correlations = {
        'PE rate': (r_rate, p_rate),
        'Max run': (r_max, p_max),
        'Mean run': (r_mean, p_mean)
    }

    if len(males_ac) >= 10:
        correlations['Autocorr'] = (r_ac, p_ac)

    strongest = max(correlations.items(), key=lambda x: abs(x[1][0]))

    print(f"\n  ✓ Strongest predictor: {strongest[0]} (r = {strongest[1][0]:+.3f}, p = {strongest[1][1]:.4f})")

# Save combined metrics
combined_file = OUTPUT_DIR / "combined_momentum_metrics.csv"
combined.to_csv(combined_file, index=False, encoding='utf-8-sig')
print(f"\n\nSaved combined metrics: {combined_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("PERSEVERATION MOMENTUM ANALYSIS COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:")
print("-" * 80)

if len(lonely_males) >= 3 and len(nonlonely_males) >= 3:
    lonely_max = lonely_males['max_run'].mean()
    nonlonely_max = nonlonely_males['max_run'].mean()

    print(f"\n✓ Max run length:")
    print(f"  Lonely males: {lonely_max:.1f}")
    print(f"  Non-lonely males: {nonlonely_max:.1f}")
    print(f"  Difference: {lonely_max - nonlonely_max:+.1f}")

if len(lonely_males_ac) >= 3 and len(nonlonely_males_ac) >= 3:
    lonely_ac = lonely_males_ac['autocorr_lag1'].mean()
    nonlonely_ac = nonlonely_males_ac['autocorr_lag1'].mean()

    print(f"\n✓ Autocorrelation:")
    print(f"  Lonely males: {lonely_ac:+.3f}")
    print(f"  Non-lonely males: {nonlonely_ac:+.3f}")
    print(f"  Difference: {lonely_ac - nonlonely_ac:+.3f}")

print("\n" + "=" * 80)
