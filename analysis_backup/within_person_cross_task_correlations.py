"""
Within-Person Cross-Task Correlations Analysis
================================================
Tests whether WCST vulnerability predicts PRP vulnerability within individuals

Key questions:
1. Do males with high WCST PE also show high PRP τ (attentional lapses)?
2. Do males with high WCST PE also show high PRP σ (variability)?
3. Is WCST PE correlated with Stroop interference (cross-task generalization)?
4. Are these correlations specific to males (vulnerability) vs females (compensation)?

Method: Pearson/Spearman correlations, partial correlations controlling UCLA
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

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/cross_task_correlations")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("WITHIN-PERSON CROSS-TASK CORRELATIONS ANALYSIS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading data...")

# Participant info and surveys
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
surveys = surveys.rename(columns={'participantId': 'participant_id'})

# UCLA scores
if 'surveyName' in surveys.columns:
    ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
elif 'survey' in surveys.columns:
    ucla_data = surveys[surveys['survey'].str.lower() == 'ucla'].copy()
else:
    raise KeyError("No survey name column found")
ucla_scores = ucla_data.groupby('participant_id')['score'].sum().reset_index()
ucla_scores.columns = ['participant_id', 'ucla_total']

# Merge with demographics
master = participants[['participant_id', 'age', 'gender', 'education']].merge(
    ucla_scores, on='participant_id', how='inner'
)

# WCST data for PE rate
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
if 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

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

wcst_summary = wcst_trials.groupby('participant_id').agg(
    pe_count=('is_pe', 'sum'),
    total_trials=('is_pe', 'count'),
    wcst_accuracy=('correct', lambda x: (x.sum() / len(x)) * 100),
    wcst_mean_rt=('rt', 'mean'),
    wcst_sd_rt=('rt', 'std')
).reset_index()
wcst_summary['pe_rate'] = (wcst_summary['pe_count'] / wcst_summary['total_trials']) * 100

# PRP Ex-Gaussian
prp_exg = pd.read_csv("results/analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv")
if prp_exg['participant_id'].dtype == 'O' and prp_exg['participant_id'].iloc[0].startswith('\ufeff'):
    prp_exg['participant_id'] = prp_exg['participant_id'].str.replace('\ufeff', '')

prp_exg['prp_tau_long'] = prp_exg['long_tau']
prp_exg['prp_sigma_long'] = prp_exg['long_sigma']
prp_exg['prp_mu_long'] = prp_exg['long_mu']
prp_exg['prp_tau_short'] = prp_exg['short_tau']
prp_exg['prp_sigma_short'] = prp_exg['short_sigma']
prp_exg['prp_mu_short'] = prp_exg['short_mu']
prp_exg['prp_tau_bottleneck'] = prp_exg['short_tau'] - prp_exg['long_tau']
prp_exg['prp_sigma_bottleneck'] = prp_exg['short_sigma'] - prp_exg['long_sigma']
prp_exg['prp_mu_bottleneck'] = prp_exg['short_mu'] - prp_exg['long_mu']

# PRP bottleneck effect
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")
if 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.rename(columns={'participantId': 'participant_id'})

prp_trials = prp_trials[
    (prp_trials['t1_timeout'] == False) &
    (prp_trials['t2_timeout'] == False) &
    (prp_trials['t2_rt'] > 0)
].copy()

def bin_soa(soa):
    if soa <= 150:
        return 'short'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'

prp_trials['soa_bin'] = prp_trials['soa'].apply(bin_soa)
prp_trials = prp_trials[prp_trials['soa_bin'].isin(['short', 'long'])].copy()

prp_rt_summary = prp_trials.groupby(['participant_id', 'soa_bin']).agg(
    t2_rt_mean=('t2_rt', 'mean'),
    t2_rt_sd=('t2_rt', 'std')
).reset_index()

prp_wide = prp_rt_summary.pivot(index='participant_id', columns='soa_bin',
                                  values=['t2_rt_mean', 't2_rt_sd']).reset_index()
prp_wide.columns = ['_'.join(col).strip('_') for col in prp_wide.columns.values]
prp_wide.columns = ['participant_id', 't2_rt_long', 't2_rt_short', 't2_sd_long', 't2_sd_short']
prp_wide['prp_bottleneck'] = prp_wide['t2_rt_short'] - prp_wide['t2_rt_long']

# Stroop Ex-Gaussian and interference
stroop_exg = pd.read_csv("results/analysis_outputs/mechanism_analysis/exgaussian/exgaussian_parameters.csv")
if stroop_exg['participant_id'].dtype == 'O' and stroop_exg['participant_id'].iloc[0].startswith('\ufeff'):
    stroop_exg['participant_id'] = stroop_exg['participant_id'].str.replace('\ufeff', '')

stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
if 'participantId' in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})

stroop_summary = stroop_trials[stroop_trials['timeout'] == False].groupby(['participant_id', 'condition']).agg(
    rt_mean=('rt', 'mean'),
    accuracy=('correct', 'mean')
).reset_index()

stroop_wide = stroop_summary.pivot(index='participant_id', columns='condition',
                                    values=['rt_mean', 'accuracy']).reset_index()
stroop_wide.columns = ['_'.join(col).strip('_') for col in stroop_wide.columns.values]
stroop_wide.columns = [c.replace('rt_mean_', '').replace('accuracy_', 'acc_') for c in stroop_wide.columns]

if 'incongruent' in stroop_wide.columns and 'congruent' in stroop_wide.columns:
    stroop_wide['stroop_interference'] = stroop_wide['incongruent'] - stroop_wide['congruent']

# Merge all
master = master.merge(wcst_summary[['participant_id', 'pe_rate', 'wcst_accuracy', 'wcst_sd_rt']],
                      on='participant_id', how='left')
master = master.merge(prp_exg[['participant_id', 'prp_tau_long', 'prp_sigma_long', 'prp_mu_long',
                                 'prp_tau_short', 'prp_sigma_short', 'prp_mu_short',
                                 'prp_tau_bottleneck', 'prp_sigma_bottleneck', 'prp_mu_bottleneck']],
                      on='participant_id', how='left')
master = master.merge(prp_wide[['participant_id', 'prp_bottleneck', 't2_sd_long', 't2_sd_short']],
                      on='participant_id', how='left')
master = master.merge(stroop_exg[['participant_id', 'tau', 'sigma', 'mu']],
                      on='participant_id', how='left', suffixes=('', '_stroop'))

if 'stroop_interference' in stroop_wide.columns:
    master = master.merge(stroop_wide[['participant_id', 'stroop_interference']],
                          on='participant_id', how='left')

# Create gender dummy
master['gender_male'] = (master['gender'].str.lower() == 'male').astype(int)

# Drop missing
master = master.dropna(subset=['ucla_total', 'gender_male', 'pe_rate']).copy()

print(f"  Loaded {len(master)} participants with complete data")
print(f"  Males: {master['gender_male'].sum()}, Females: {(1-master['gender_male']).sum()}\n")

# ============================================================================
# Cross-Task Correlations
# ============================================================================

print("=" * 80)
print("CROSS-TASK CORRELATIONS")
print("=" * 80)

correlation_pairs = [
    ('pe_rate', 'prp_tau_long', 'WCST PE × PRP τ (long SOA)'),
    ('pe_rate', 'prp_sigma_long', 'WCST PE × PRP σ (long SOA)'),
    ('pe_rate', 'prp_tau_bottleneck', 'WCST PE × PRP τ bottleneck'),
    ('pe_rate', 'prp_sigma_bottleneck', 'WCST PE × PRP σ bottleneck'),
    ('pe_rate', 'prp_bottleneck', 'WCST PE × PRP Bottleneck effect'),
    ('pe_rate', 't2_sd_long', 'WCST PE × PRP T2 SD (long SOA)'),
    ('pe_rate', 'wcst_sd_rt', 'WCST PE × WCST RT SD (within-task)'),
    ('pe_rate', 'tau', 'WCST PE × Stroop τ'),
    ('pe_rate', 'sigma', 'WCST PE × Stroop σ'),
    ('pe_rate', 'stroop_interference', 'WCST PE × Stroop Interference'),
    ('prp_tau_long', 'tau', 'PRP τ × Stroop τ'),
    ('prp_sigma_long', 'sigma', 'PRP σ × Stroop σ'),
]

results_list = []

for var1, var2, label in correlation_pairs:
    if var1 not in master.columns or var2 not in master.columns:
        continue

    print(f"\n{label}")
    print("   " + "-" * 60)

    for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
        df_sub = master[master['gender_male'] == gender_val].copy()

        # Drop missing for this pair
        df_clean = df_sub[[var1, var2, 'ucla_total']].dropna()

        if len(df_clean) < 10:
            print(f"   {gender_name}: Insufficient data (N={len(df_clean)})")
            continue

        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(df_clean[var1], df_clean[var2])

        # Spearman correlation (robust to outliers)
        r_spearman, p_spearman = stats.spearmanr(df_clean[var1], df_clean[var2])

        # Partial correlation controlling UCLA
        from scipy.stats import pearsonr

        def partial_corr(x, y, z):
            """Partial correlation of x and y controlling for z"""
            # Residualize x and y on z
            rx = x - np.polyval(np.polyfit(z, x, 1), z)
            ry = y - np.polyval(np.polyfit(z, y, 1), z)
            return pearsonr(rx, ry)

        r_partial, p_partial = partial_corr(df_clean[var1].values, df_clean[var2].values,
                                              df_clean['ucla_total'].values)

        print(f"   {gender_name} (N={len(df_clean)}):")
        print(f"      Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")
        print(f"      Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")
        print(f"      Partial r (control UCLA) = {r_partial:.3f}, p = {p_partial:.4f}")

        results_list.append({
            'variable_1': var1,
            'variable_2': var2,
            'label': label,
            'gender': gender_name,
            'n': len(df_clean),
            'r_pearson': r_pearson,
            'p_pearson': p_pearson,
            'r_spearman': r_spearman,
            'p_spearman': p_spearman,
            'r_partial': r_partial,
            'p_partial': p_partial
        })

results_df = pd.DataFrame(results_list)

# ============================================================================
# Identify Multi-Task Vulnerability Profile
# ============================================================================

print("\n\n" + "=" * 80)
print("MULTI-TASK VULNERABILITY PROFILE")
print("=" * 80)

# Define vulnerable individuals (males only)
males = master[master['gender_male'] == 1].copy()

if 'prp_tau_long' in males.columns and len(males) >= 10:
    # Z-score PE and tau
    males['z_pe'] = (males['pe_rate'] - males['pe_rate'].mean()) / males['pe_rate'].std()
    males['z_tau'] = (males['prp_tau_long'] - males['prp_tau_long'].mean()) / males['prp_tau_long'].std()

    # Multi-task vulnerable: High on both PE and tau
    males['multi_task_vulnerable'] = ((males['z_pe'] > 0.5) & (males['z_tau'] > 0.5)).astype(int)

    # Task-specific vulnerable
    males['wcst_only_vulnerable'] = ((males['z_pe'] > 0.5) & (males['z_tau'] <= 0.5)).astype(int)
    males['prp_only_vulnerable'] = ((males['z_pe'] <= 0.5) & (males['z_tau'] > 0.5)).astype(int)

    # Counts
    n_multi = males['multi_task_vulnerable'].sum()
    n_wcst_only = males['wcst_only_vulnerable'].sum()
    n_prp_only = males['prp_only_vulnerable'].sum()
    n_neither = ((males['z_pe'] <= 0.5) & (males['z_tau'] <= 0.5)).sum()

    print(f"\nMale vulnerability profiles (N={len(males)}):")
    print(f"  Multi-task vulnerable (high PE & high τ): {n_multi} ({n_multi/len(males):.1%})")
    print(f"  WCST-only vulnerable (high PE, low τ): {n_wcst_only} ({n_wcst_only/len(males):.1%})")
    print(f"  PRP-only vulnerable (low PE, high τ): {n_prp_only} ({n_prp_only/len(males):.1%})")
    print(f"  Neither: {n_neither} ({n_neither/len(males):.1%})")

    # Compare UCLA across profiles
    print("\nUCLA scores by profile:")
    for profile, col in [('Multi-task', 'multi_task_vulnerable'),
                         ('WCST-only', 'wcst_only_vulnerable'),
                         ('PRP-only', 'prp_only_vulnerable')]:
        profile_data = males[males[col] == 1]['ucla_total']
        if len(profile_data) > 0:
            print(f"  {profile}: M = {profile_data.mean():.1f}, SD = {profile_data.std():.1f}")

    # Save profile assignments
    profile_file = OUTPUT_DIR / "multi_task_vulnerability_profiles.csv"
    males[['participant_id', 'ucla_total', 'pe_rate', 'prp_tau_long',
           'z_pe', 'z_tau', 'multi_task_vulnerable', 'wcst_only_vulnerable',
           'prp_only_vulnerable']].to_csv(profile_file, index=False, encoding='utf-8-sig')
    print(f"\nProfile assignments saved: {profile_file}")

# ============================================================================
# Save Results
# ============================================================================

print("\n\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Correlation results
corr_file = OUTPUT_DIR / "cross_task_correlations_detailed.csv"
results_df.to_csv(corr_file, index=False, encoding='utf-8-sig')
print(f"\nDetailed correlations: {corr_file}")

# Summary (significant correlations only)
sig_corr = results_df[results_df['p_pearson'] < 0.05].copy()
sig_corr = sig_corr.sort_values('p_pearson')

summary_file = OUTPUT_DIR / "cross_task_correlations_significant.csv"
sig_corr.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f"Significant correlations: {summary_file}")

# Key findings
print("\n\n" + "=" * 80)
print("KEY FINDINGS: Significant Cross-Task Correlations (p < 0.05)")
print("=" * 80)

if len(sig_corr) > 0:
    for _, row in sig_corr.iterrows():
        print(f"\n{row['label']} ({row['gender']}):")
        print(f"  r = {row['r_pearson']:.3f}, p = {row['p_pearson']:.4f}")
        print(f"  Partial r (UCLA) = {row['r_partial']:.3f}, p = {row['p_partial']:.4f}")
else:
    print("\nNo significant correlations at p < 0.05")

# Trend-level findings
trend_corr = results_df[(results_df['p_pearson'] >= 0.05) & (results_df['p_pearson'] < 0.10)]
if len(trend_corr) > 0:
    print("\n\nTrend-level correlations (0.05 ≤ p < 0.10):")
    for _, row in trend_corr.iterrows():
        print(f"  {row['label']} ({row['gender']}): r = {row['r_pearson']:.3f}, p = {row['p_pearson']:.4f}")

print("\n" + "=" * 80)
print("WITHIN-PERSON CROSS-TASK CORRELATIONS ANALYSIS COMPLETE")
print("=" * 80)
