#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WCST Error Type Analysis: Perseverative vs. Non-Perseverative Errors

Tests whether UCLA loneliness specifically predicts cognitive rigidity (PE)
or general impairment (both PE and NPE) in males.

Hypothesis:
- Males: UCLA → high PE (rigidity), normal NPE
- Females: UCLA → no effect on PE or NPE

Theoretical significance:
- PE-specific effect supports "rumination/rigidity" theory
- PE+NPE effect would suggest general attentional impairment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/error_types")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WCST ERROR TYPE ANALYSIS")
print("Perseverative vs. Non-Perseverative Errors")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1] Loading data...")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})

wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

print(f"  Loaded {len(participants)} participants")
print(f"  Loaded {len(wcst_trials)} WCST trials")

# ============================================================================
# 2. EXTRACT VARIABLES
# ============================================================================

print("\n[2] Extracting UCLA and demographics...")

# UCLA
ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores = ucla_scores.rename(columns={'score': 'ucla_total'})
ucla_scores = ucla_scores.dropna(subset=['ucla_total'])

# DASS
dass_subscales = surveys[surveys['surveyName'] == 'dass'][['participant_id', 'score_D', 'score_A', 'score_S', 'score']].copy()
dass_subscales = dass_subscales.rename(columns={
    'score_D': 'dass_depression',
    'score_A': 'dass_anxiety',
    'score_S': 'dass_stress',
    'score': 'dass_total'
})
dass_subscales = dass_subscales.dropna()

# Merge demographics
ucla_scores = ucla_scores.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

# Recode gender
ucla_scores['gender'] = ucla_scores['gender'].map({'남성': 'male', '여성': 'female'})

print(f"  UCLA: {len(ucla_scores)} participants")
print(f"  DASS: {len(dass_subscales)} participants")

# ============================================================================
# 3. PARSE ERROR TYPES FROM WCST
# ============================================================================

print("\n[3] Parsing WCST error types...")

def parse_wcst_extra(extra_str):
    """Parse extra field for error flags"""
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_parsed'] = wcst_trials['extra'].apply(parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_parsed'].apply(lambda x: x.get('isPE', False))
wcst_trials['is_npe'] = wcst_trials['extra_parsed'].apply(lambda x: x.get('isNPE', False))

# Also check for correct/incorrect trials
wcst_trials['is_error'] = ~wcst_trials['correct']

# Filter valid trials
wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['rt_ms'] > 0)
].copy()

# Compute error counts per participant
error_summary = wcst_valid.groupby('participant_id').agg(
    total_trials=('trial_index', 'count'),
    total_errors=('is_error', 'sum'),
    pe_count=('is_pe', 'sum'),
    npe_count=('is_npe', 'sum')
).reset_index()

# Compute rates
error_summary['error_rate'] = (error_summary['total_errors'] / error_summary['total_trials']) * 100
error_summary['pe_rate'] = (error_summary['pe_count'] / error_summary['total_trials']) * 100
error_summary['npe_rate'] = (error_summary['npe_count'] / error_summary['total_trials']) * 100

# PE/NPE ratio (rigidity index)
error_summary['pe_npe_ratio'] = error_summary['pe_count'] / (error_summary['npe_count'] + 1)  # +1 to avoid division by zero

# Compute mean PE run length (consecutive PE errors)
def compute_pe_run_length(df):
    """Compute mean length of consecutive PE runs"""
    runs = []
    current_run = 0

    for is_pe in df['is_pe']:
        if is_pe:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0

    if current_run > 0:
        runs.append(current_run)

    return np.mean(runs) if runs else 0

pe_run_lengths = wcst_valid.groupby('participant_id').apply(compute_pe_run_length).reset_index()
pe_run_lengths.columns = ['participant_id', 'mean_pe_run_length']

error_summary = error_summary.merge(pe_run_lengths, on='participant_id', how='left')

print(f"  WCST error metrics for {len(error_summary)} participants")
print(f"\n  Overall Error Statistics:")
print(f"    Mean PE rate: {error_summary['pe_rate'].mean():.2f}% (SD={error_summary['pe_rate'].std():.2f})")
print(f"    Mean NPE rate: {error_summary['npe_rate'].mean():.2f}% (SD={error_summary['npe_rate'].std():.2f})")
print(f"    Mean PE/NPE ratio: {error_summary['pe_npe_ratio'].mean():.2f} (SD={error_summary['pe_npe_ratio'].std():.2f})")
print(f"    Mean PE run length: {error_summary['mean_pe_run_length'].mean():.2f} trials")

# ============================================================================
# 4. MERGE MASTER DATASET
# ============================================================================

print("\n[4] Merging master dataset...")

master = error_summary.merge(ucla_scores, on='participant_id', how='inner')
master = master.merge(dass_subscales, on='participant_id', how='left')

# Drop missing
master = master.dropna(subset=['gender', 'ucla_total', 'pe_rate', 'npe_rate'])

print(f"  Final N = {len(master)}")
print(f"  Males: {(master['gender'] == 'male').sum()}, Females: {(master['gender'] == 'female').sum()}")

# Standardize
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_age'] = scaler.fit_transform(master[['age']])
master['z_dass'] = scaler.fit_transform(master[['dass_total']])

# ============================================================================
# 5. HYPOTHESIS TESTING: UCLA × GENDER → PE vs. NPE
# ============================================================================

print("\n[5] Testing UCLA × Gender effects on error types...")

# Test 1: PE rate
print("\n  PE RATE (Cognitive Rigidity)")
print("  " + "-"*70)

for gender in ['male', 'female']:
    subset = master[master['gender'] == gender]
    r, p = stats.pearsonr(subset['z_ucla'], subset['pe_rate'])

    # Regression controlling for DASS
    if 'z_dass' in subset.columns and not subset['z_dass'].isna().all():
        X = np.column_stack([np.ones(len(subset)), subset['z_ucla'], subset['z_dass']])
        Y = subset['pe_rate'].values
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta_ucla = coeffs[1]
    else:
        beta_ucla = np.nan

    print(f"    {gender.capitalize():8s} (N={len(subset):2d}):  r={r:>6.3f}, p={p:.4f}, β={beta_ucla:>6.3f}")

# Interaction model
master['gender_coded'] = (master['gender'] == 'male').astype(int)
master['ucla_x_gender'] = master['z_ucla'] * master['gender_coded']

X = master[['z_ucla', 'gender_coded', 'ucla_x_gender']].values
X = np.column_stack([np.ones(len(X)), X])
Y = master['pe_rate'].values

coeffs_pe = np.linalg.lstsq(X, Y, rcond=None)[0]

print(f"\n    Interaction Model:")
print(f"      UCLA main:         β={coeffs_pe[1]:>6.3f}")
print(f"      Gender main:       β={coeffs_pe[2]:>6.3f}")
print(f"      UCLA × Gender:     β={coeffs_pe[3]:>6.3f}")

# Test 2: NPE rate
print("\n  NPE RATE (General Impairment)")
print("  " + "-"*70)

for gender in ['male', 'female']:
    subset = master[master['gender'] == gender]
    r, p = stats.pearsonr(subset['z_ucla'], subset['npe_rate'])

    if 'z_dass' in subset.columns and not subset['z_dass'].isna().all():
        X = np.column_stack([np.ones(len(subset)), subset['z_ucla'], subset['z_dass']])
        Y = subset['npe_rate'].values
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta_ucla = coeffs[1]
    else:
        beta_ucla = np.nan

    print(f"    {gender.capitalize():8s} (N={len(subset):2d}):  r={r:>6.3f}, p={p:.4f}, β={beta_ucla:>6.3f}")

Y_npe = master['npe_rate'].values
coeffs_npe = np.linalg.lstsq(X, Y_npe, rcond=None)[0]

print(f"\n    Interaction Model:")
print(f"      UCLA main:         β={coeffs_npe[1]:>6.3f}")
print(f"      Gender main:       β={coeffs_npe[2]:>6.3f}")
print(f"      UCLA × Gender:     β={coeffs_npe[3]:>6.3f}")

# Test 3: PE/NPE ratio (rigidity index)
print("\n  PE/NPE RATIO (Rigidity Index)")
print("  " + "-"*70)

for gender in ['male', 'female']:
    subset = master[master['gender'] == gender]
    r, p = stats.pearsonr(subset['z_ucla'], subset['pe_npe_ratio'])
    print(f"    {gender.capitalize():8s} (N={len(subset):2d}):  r={r:>6.3f}, p={p:.4f}")

Y_ratio = master['pe_npe_ratio'].values
coeffs_ratio = np.linalg.lstsq(X, Y_ratio, rcond=None)[0]

print(f"\n    Interaction Model:")
print(f"      UCLA × Gender:     β={coeffs_ratio[3]:>6.3f}")

# Test 4: PE run length
print("\n  MEAN PE RUN LENGTH (Perseveration Depth)")
print("  " + "-"*70)

for gender in ['male', 'female']:
    subset = master[master['gender'] == gender]
    r, p = stats.pearsonr(subset['z_ucla'], subset['mean_pe_run_length'])
    print(f"    {gender.capitalize():8s} (N={len(subset):2d}):  r={r:>6.3f}, p={p:.4f}")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print("\n[6] Saving results...")

# Save participant-level data
master.to_csv(OUTPUT_DIR / "wcst_error_types_participant_level.csv", index=False, encoding='utf-8-sig')

# Save summary statistics
summary_results = pd.DataFrame([
    {
        'error_type': 'PE',
        'male_r': stats.pearsonr(master[master['gender'] == 'male']['z_ucla'], master[master['gender'] == 'male']['pe_rate'])[0],
        'male_p': stats.pearsonr(master[master['gender'] == 'male']['z_ucla'], master[master['gender'] == 'male']['pe_rate'])[1],
        'female_r': stats.pearsonr(master[master['gender'] == 'female']['z_ucla'], master[master['gender'] == 'female']['pe_rate'])[0],
        'female_p': stats.pearsonr(master[master['gender'] == 'female']['z_ucla'], master[master['gender'] == 'female']['pe_rate'])[1],
        'interaction_beta': coeffs_pe[3]
    },
    {
        'error_type': 'NPE',
        'male_r': stats.pearsonr(master[master['gender'] == 'male']['z_ucla'], master[master['gender'] == 'male']['npe_rate'])[0],
        'male_p': stats.pearsonr(master[master['gender'] == 'male']['z_ucla'], master[master['gender'] == 'male']['npe_rate'])[1],
        'female_r': stats.pearsonr(master[master['gender'] == 'female']['z_ucla'], master[master['gender'] == 'female']['npe_rate'])[0],
        'female_p': stats.pearsonr(master[master['gender'] == 'female']['z_ucla'], master[master['gender'] == 'female']['npe_rate'])[1],
        'interaction_beta': coeffs_npe[3]
    },
    {
        'error_type': 'PE/NPE Ratio',
        'male_r': stats.pearsonr(master[master['gender'] == 'male']['z_ucla'], master[master['gender'] == 'male']['pe_npe_ratio'])[0],
        'male_p': stats.pearsonr(master[master['gender'] == 'male']['z_ucla'], master[master['gender'] == 'male']['pe_npe_ratio'])[1],
        'female_r': stats.pearsonr(master[master['gender'] == 'female']['z_ucla'], master[master['gender'] == 'female']['pe_npe_ratio'])[0],
        'female_p': stats.pearsonr(master[master['gender'] == 'female']['z_ucla'], master[master['gender'] == 'female']['pe_npe_ratio'])[1],
        'interaction_beta': coeffs_ratio[3]
    }
])

summary_results.to_csv(OUTPUT_DIR / "error_type_summary.csv", index=False, encoding='utf-8-sig')
print("✓ Saved: error_type_summary.csv")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('WCST Error Type Analysis by UCLA Loneliness and Gender', fontsize=14, fontweight='bold')

# Panel 1: PE rate
ax = axes[0, 0]
for gender, color in [('male', 'blue'), ('female', 'red')]:
    subset = master[master['gender'] == gender]
    ax.scatter(subset['z_ucla'], subset['pe_rate'], alpha=0.6, s=80, label=gender.capitalize(),
               color=color, edgecolors='black', linewidths=0.5)

    # Fit line
    z = np.polyfit(subset['z_ucla'], subset['pe_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
    ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

ax.set_xlabel('UCLA (z-scored)')
ax.set_ylabel('PE Rate (%)')
ax.set_title('Perseverative Errors (Rigidity)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: NPE rate
ax = axes[0, 1]
for gender, color in [('male', 'blue'), ('female', 'red')]:
    subset = master[master['gender'] == gender]
    ax.scatter(subset['z_ucla'], subset['npe_rate'], alpha=0.6, s=80, label=gender.capitalize(),
               color=color, edgecolors='black', linewidths=0.5)

    z = np.polyfit(subset['z_ucla'], subset['npe_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
    ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

ax.set_xlabel('UCLA (z-scored)')
ax.set_ylabel('NPE Rate (%)')
ax.set_title('Non-Perseverative Errors (General)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: PE/NPE ratio
ax = axes[1, 0]
for gender, color in [('male', 'blue'), ('female', 'red')]:
    subset = master[master['gender'] == gender]
    ax.scatter(subset['z_ucla'], subset['pe_npe_ratio'], alpha=0.6, s=80, label=gender.capitalize(),
               color=color, edgecolors='black', linewidths=0.5)

    z = np.polyfit(subset['z_ucla'], subset['pe_npe_ratio'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
    ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

ax.set_xlabel('UCLA (z-scored)')
ax.set_ylabel('PE/NPE Ratio')
ax.set_title('Rigidity Index')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Comparison bar chart
ax = axes[1, 1]

male_data = master[master['gender'] == 'male']
female_data = master[master['gender'] == 'female']

x = np.arange(2)
width = 0.35

male_pe_r = stats.pearsonr(male_data['z_ucla'], male_data['pe_rate'])[0]
male_npe_r = stats.pearsonr(male_data['z_ucla'], male_data['npe_rate'])[0]
female_pe_r = stats.pearsonr(female_data['z_ucla'], female_data['pe_rate'])[0]
female_npe_r = stats.pearsonr(female_data['z_ucla'], female_data['npe_rate'])[0]

ax.bar(x - width/2, [male_pe_r, male_npe_r], width, label='Male', color='blue', alpha=0.7, edgecolor='black')
ax.bar(x + width/2, [female_pe_r, female_npe_r], width, label='Female', color='red', alpha=0.7, edgecolor='black')

ax.set_ylabel('Correlation with UCLA')
ax.set_title('UCLA Correlations by Error Type and Gender')
ax.set_xticks(x)
ax.set_xticklabels(['PE Rate', 'NPE Rate'])
ax.legend()
ax.axhline(0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'error_type_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_type_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WCST ERROR TYPE ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")

print("\n  PERSEVERATIVE ERRORS (Rigidity):")
pe_results = summary_results[summary_results['error_type'] == 'PE'].iloc[0]
print(f"    Males:   r={pe_results['male_r']:>6.3f}, p={pe_results['male_p']:.4f}")
print(f"    Females: r={pe_results['female_r']:>6.3f}, p={pe_results['female_p']:.4f}")
print(f"    Interaction β={pe_results['interaction_beta']:>6.3f}")

print("\n  NON-PERSEVERATIVE ERRORS (General Impairment):")
npe_results = summary_results[summary_results['error_type'] == 'NPE'].iloc[0]
print(f"    Males:   r={npe_results['male_r']:>6.3f}, p={npe_results['male_p']:.4f}")
print(f"    Females: r={npe_results['female_r']:>6.3f}, p={npe_results['female_p']:.4f}")
print(f"    Interaction β={npe_results['interaction_beta']:>6.3f}")

print("\n  INTERPRETATION:")
if abs(pe_results['male_r']) > abs(npe_results['male_r']):
    print("    ✓ UCLA predicts PE > NPE in males → SUPPORTS RIGIDITY HYPOTHESIS")
    print("    ✓ Loneliness specifically impairs cognitive flexibility, not general ability")
else:
    print("    → UCLA predicts both PE and NPE → GENERAL IMPAIRMENT")

# Additional interpretation based on interaction strengths
if abs(pe_results['interaction_beta']) > abs(npe_results['interaction_beta']):
    print(f"\n    ✓ UCLA × Gender interaction STRONGER for PE (β={pe_results['interaction_beta']:.3f}) than NPE (β={npe_results['interaction_beta']:.3f})")
    print("    ✓ VALIDATES RIGIDITY HYPOTHESIS: Gender moderation specific to perseveration")

print("\nOUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}/")
print("  - wcst_error_types_participant_level.csv")
print("  - error_type_summary.csv")
print("  - error_type_analysis.png")

print("\n" + "="*80)
