"""
Temporal Fatigue Analysis - Trial-by-Trial PE Dynamics
=======================================================
Tests whether lonely males show accelerated cognitive fatigue (vigilance decrement)

Research Questions:
1. Does PE rate increase across trials in lonely males?
2. Is there a UCLA × Gender × Trial interaction?
3. At which trial # do lonely males diverge from controls?
4. Does fatigue slope correlate with PRP τ (attention depletion)?

Expected: Vulnerable males show steeper fatigue slope (vigilance decrement)
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

from data_loader_utils import load_master_dataset, load_exgaussian_params

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/temporal_fatigue")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("TEMPORAL FATIGUE ANALYSIS")
print("=" * 80)
print("\nResearch Question: Do lonely males show accelerated cognitive fatigue?")
print("  - Analyze trial-by-trial PE rate changes")
print("  - Test UCLA × Gender × Trial interaction")
print("  - Identify fatigue breakpoint\n")

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
wcst_trials['is_pe'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

print(f"  Loaded {len(wcst_trials)} WCST trials")

# Load master dataset for UCLA, gender
master = load_master_dataset()

# Merge trial data with participant-level UCLA/gender
wcst_trials = wcst_trials.merge(master[['participant_id', 'ucla_total', 'gender_male', 'age']],
                                  on='participant_id', how='left')

# Drop missing
wcst_trials = wcst_trials.dropna(subset=['ucla_total', 'gender_male', 'is_pe']).copy()

print(f"  After merging: {len(wcst_trials)} trials from {wcst_trials['participant_id'].nunique()} participants")
print(f"    Males: {wcst_trials[wcst_trials['gender_male']==1]['participant_id'].nunique()}")
print(f"    Females: {wcst_trials[wcst_trials['gender_male']==0]['participant_id'].nunique()}\n")

# Add trial index within participant
wcst_trials = wcst_trials.sort_values(['participant_id', 'trialIndex'])
wcst_trials['trial_number'] = wcst_trials.groupby('participant_id').cumcount() + 1

# Create trial blocks (quartiles)
wcst_trials['trial_block'] = pd.cut(wcst_trials['trial_number'],
                                     bins=[0, 20, 40, 60, 200],
                                     labels=['Early (1-20)', 'Mid1 (21-40)', 'Mid2 (41-60)', 'Late (60+)'])

# ============================================================================
# 1. Aggregate Analysis: PE Rate by Block
# ============================================================================

print("=" * 80)
print("ANALYSIS 1: PE RATE BY TRIAL BLOCK")
print("=" * 80)

# Group by participant × block
block_summary = wcst_trials.groupby(['participant_id', 'trial_block', 'gender_male', 'ucla_total']).agg(
    pe_rate=('is_pe', lambda x: (x.sum() / len(x)) * 100),
    n_trials=('is_pe', 'count')
).reset_index()

# Drop blocks with <5 trials
block_summary = block_summary[block_summary['n_trials'] >= 5].copy()

print(f"\nBlock summary: {len(block_summary)} participant-block combinations\n")

# Test interaction: PE ~ ucla * gender_male * trial_block
# Convert trial_block to numeric for interaction
block_mapping = {'Early (1-20)': 1, 'Mid1 (21-40)': 2, 'Mid2 (41-60)': 3, 'Late (60+)': 4}
block_summary['block_num'] = block_summary['trial_block'].map(block_mapping)

# Standardize UCLA
block_summary['z_ucla'] = (block_summary['ucla_total'] - block_summary['ucla_total'].mean()) / block_summary['ucla_total'].std()

# Mixed model (treating participant as random effect is complex, use OLS for now)
try:
    model = smf.ols('pe_rate ~ z_ucla * gender_male * block_num + age', data=block_summary).fit()
    print("Linear Model: PE ~ UCLA × Gender × Block")
    print("-" * 60)
    print(model.summary().tables[1])

    # Extract 3-way interaction
    params = model.params
    if 'z_ucla:gender_male:block_num' in params:
        beta_3way = params['z_ucla:gender_male:block_num']
        p_3way = model.pvalues['z_ucla:gender_male:block_num']
        print(f"\n3-way interaction: β = {beta_3way:.4f}, p = {p_3way:.4f}")
        if p_3way < 0.05:
            print("  ⭐ SIGNIFICANT: Lonely males show different fatigue trajectory!")
    else:
        print("\n  3-way interaction term not found in model")

except Exception as e:
    print(f"Model fitting error: {e}")

# ============================================================================
# 2. Stratified Analysis: Fatigue Slopes by UCLA/Gender
# ============================================================================

print("\n\n" + "=" * 80)
print("ANALYSIS 2: FATIGUE SLOPES (STRATIFIED)")
print("=" * 80)

# Define groups
master_summary = master[['participant_id', 'ucla_total', 'gender_male']].copy()
ucla_median = master_summary['ucla_total'].median()

master_summary['group'] = 'Other'
master_summary.loc[(master_summary['gender_male'] == 1) & (master_summary['ucla_total'] > ucla_median), 'group'] = 'Lonely Males'
master_summary.loc[(master_summary['gender_male'] == 1) & (master_summary['ucla_total'] <= ucla_median), 'group'] = 'Non-lonely Males'
master_summary.loc[(master_summary['gender_male'] == 0) & (master_summary['ucla_total'] > ucla_median), 'group'] = 'Lonely Females'
master_summary.loc[(master_summary['gender_male'] == 0) & (master_summary['ucla_total'] <= ucla_median), 'group'] = 'Non-lonely Females'

# Compute individual slopes
slopes_list = []

for pid in wcst_trials['participant_id'].unique():
    trials_p = wcst_trials[wcst_trials['participant_id'] == pid].copy()

    if len(trials_p) >= 20:  # Minimum 20 trials
        # Linear regression: PE ~ trial_number
        X = trials_p['trial_number'].values
        Y = trials_p['is_pe'].astype(int).values

        if len(X) > 0 and np.std(X) > 0:
            slope, intercept, r, p, se = stats.linregress(X, Y)

            group_info = master_summary[master_summary['participant_id'] == pid]
            if len(group_info) > 0:
                slopes_list.append({
                    'participant_id': pid,
                    'slope': slope,
                    'intercept': intercept,
                    'r': r,
                    'p': p,
                    'n_trials': len(trials_p),
                    'group': group_info['group'].values[0],
                    'ucla_total': group_info['ucla_total'].values[0],
                    'gender_male': group_info['gender_male'].values[0]
                })

slopes_df = pd.DataFrame(slopes_list)

print(f"\nComputed slopes for {len(slopes_df)} participants")

# Group comparison
print("\nMean Fatigue Slopes by Group:")
print("-" * 60)

for group in ['Lonely Males', 'Non-lonely Males', 'Lonely Females', 'Non-lonely Females']:
    group_data = slopes_df[slopes_df['group'] == group]
    if len(group_data) >= 3:
        mean_slope = group_data['slope'].mean()
        sem_slope = group_data['slope'].sem()
        print(f"{group:20s}: slope = {mean_slope:.6f} (SEM = {sem_slope:.6f}), N = {len(group_data)}")

# Statistical test: Lonely vs Non-lonely Males
lonely_males = slopes_df[(slopes_df['group'] == 'Lonely Males')]['slope'].values
nonlonely_males = slopes_df[(slopes_df['group'] == 'Non-lonely Males')]['slope'].values

if len(lonely_males) >= 3 and len(nonlonely_males) >= 3:
    t_stat, p_val = stats.ttest_ind(lonely_males, nonlonely_males)
    print(f"\nLonely vs Non-lonely Males: t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("  ⭐ SIGNIFICANT DIFFERENCE")

# Save slopes
slopes_file = OUTPUT_DIR / "individual_fatigue_slopes.csv"
slopes_df.to_csv(slopes_file, index=False, encoding='utf-8-sig')
print(f"\nSaved slopes: {slopes_file}")

# ============================================================================
# 3. Moving Window Analysis: Identify Breakpoint
# ============================================================================

print("\n\n" + "=" * 80)
print("ANALYSIS 3: MOVING WINDOW (BREAKPOINT DETECTION)")
print("=" * 80)

# Compute PE rate in 10-trial windows
window_size = 10
window_results = []

for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
    for ucla_group in ['High UCLA', 'Low UCLA']:

        if ucla_group == 'High UCLA':
            subset = wcst_trials[(wcst_trials['gender_male'] == gender_val) &
                                 (wcst_trials['ucla_total'] > ucla_median)]
        else:
            subset = wcst_trials[(wcst_trials['gender_male'] == gender_val) &
                                 (wcst_trials['ucla_total'] <= ucla_median)]

        if len(subset) == 0:
            continue

        # Compute PE rate for each window
        for window_start in range(1, 81, 10):  # 1-10, 11-20, 21-30, ... 71-80
            window_end = window_start + window_size - 1
            window_trials = subset[(subset['trial_number'] >= window_start) &
                                    (subset['trial_number'] <= window_end)]

            if len(window_trials) >= 20:  # At least 20 trials in window
                pe_rate_window = (window_trials['is_pe'].sum() / len(window_trials)) * 100

                window_results.append({
                    'gender': gender_name,
                    'ucla_group': ucla_group,
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_mid': (window_start + window_end) / 2,
                    'pe_rate': pe_rate_window,
                    'n_trials': len(window_trials)
                })

window_df = pd.DataFrame(window_results)

print(f"\nComputed PE rates for {len(window_df)} windows\n")

# Print results
print("PE Rate by Window (Males):")
print("-" * 60)

for window_mid in sorted(window_df['window_mid'].unique()):
    lonely = window_df[(window_df['gender'] == 'Male') &
                       (window_df['ucla_group'] == 'High UCLA') &
                       (window_df['window_mid'] == window_mid)]
    nonlonely = window_df[(window_df['gender'] == 'Male') &
                          (window_df['ucla_group'] == 'Low UCLA') &
                          (window_df['window_mid'] == window_mid)]

    if len(lonely) > 0 and len(nonlonely) > 0:
        pe_lonely = lonely['pe_rate'].values[0]
        pe_nonlonely = nonlonely['pe_rate'].values[0]
        diff = pe_lonely - pe_nonlonely

        print(f"Trials {int(window_mid)-4:.0f}-{int(window_mid)+5:.0f}:  Lonely = {pe_lonely:.1f}%,  Non-lonely = {pe_nonlonely:.1f}%,  Diff = {diff:+.1f}%")

# Save window analysis
window_file = OUTPUT_DIR / "moving_window_pe_rates.csv"
window_df.to_csv(window_file, index=False, encoding='utf-8-sig')
print(f"\nSaved window analysis: {window_file}")

# ============================================================================
# 4. Correlation with PRP τ
# ============================================================================

print("\n\n" + "=" * 80)
print("ANALYSIS 4: FATIGUE SLOPE × PRP τ CORRELATION")
print("=" * 80)

# Load PRP Ex-Gaussian parameters
try:
    prp_exg = load_exgaussian_params('prp')

    # Merge with slopes
    slopes_with_tau = slopes_df.merge(prp_exg[['participant_id', 'long_tau']],
                                       on='participant_id', how='left')

    # Correlation (males only)
    males_tau = slopes_with_tau[(slopes_with_tau['gender_male'] == 1)].dropna(subset=['slope', 'long_tau'])

    if len(males_tau) >= 10:
        r, p = stats.pearsonr(males_tau['slope'], males_tau['long_tau'])
        print(f"\nMales: Fatigue Slope × PRP τ (long)")
        print(f"  r = {r:.3f}, p = {p:.4f}, N = {len(males_tau)}")
        if p < 0.05:
            print("  ⭐ SIGNIFICANT: Attention lapses (τ) predict faster fatigue!")

        # Save
        corr_file = OUTPUT_DIR / "fatigue_tau_correlation.csv"
        pd.DataFrame([{
            'gender': 'Male',
            'n': len(males_tau),
            'r': r,
            'p': p
        }]).to_csv(corr_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved correlation: {corr_file}")

except Exception as e:
    print(f"\nCould not load PRP Ex-Gaussian data: {e}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("TEMPORAL FATIGUE ANALYSIS COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:")
print("-" * 80)

# Summarize slope differences
if len(slopes_df) > 0:
    lonely_m = slopes_df[slopes_df['group'] == 'Lonely Males']['slope'].mean()
    nonlonely_m = slopes_df[slopes_df['group'] == 'Non-lonely Males']['slope'].mean()

    print(f"\n✓ Fatigue Slopes (Males):")
    print(f"  Lonely: {lonely_m:.6f} (PE increases {lonely_m*100:.4f}% per trial)")
    print(f"  Non-lonely: {nonlonely_m:.6f} (PE increases {nonlonely_m*100:.4f}% per trial)")

print("\n" + "=" * 80)
