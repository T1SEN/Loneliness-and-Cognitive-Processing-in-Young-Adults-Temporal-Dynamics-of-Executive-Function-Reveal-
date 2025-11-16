"""
Trial-Level Error Cascade Analysis

Research Question:
Do errors cluster in temporal chains, and does UCLA loneliness predict
cascade severity across tasks (WCST, Stroop, PRP)?

Hypothesis:
Lonely individuals with EF impairment may show "error cascades" where
failures cluster together due to reduced error monitoring and recovery.

Methods:
- WCST: Consecutive perseverative error chains
- Stroop: Consecutive incorrect response chains
- PRP: Consecutive dual-task error chains
- Metrics: Max chain length, total chains, cascade proportion
- Tests: UCLA × cascade correlations, gender moderation
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/error_cascades")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("TRIAL-LEVEL ERROR CASCADE ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

# Demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)
participants['gender_male'] = (participants['gender'] == 'male').astype(int)
participants = participants.rename(columns={'participantId': 'participant_id'})

# UCLA
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})
ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()
ucla_data['ucla_total'] = pd.to_numeric(ucla_data['score'], errors='coerce')
ucla_data = ucla_data[['participant_id', 'ucla_total']].dropna()

# DASS
dass_data = surveys[surveys['surveyName'] == 'dass'].copy()
dass_data['score_A'] = pd.to_numeric(dass_data['score_A'], errors='coerce')
dass_data['score_S'] = pd.to_numeric(dass_data['score_S'], errors='coerce')
dass_data['score_D'] = pd.to_numeric(dass_data['score_D'], errors='coerce')
dass_data['dass_total'] = dass_data[['score_A', 'score_S', 'score_D']].sum(axis=1)
dass_pivot = dass_data[['participant_id', 'dass_total']].dropna()

# Trial-level data
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')

# Normalize column names
for df in [wcst_trials, stroop_trials, prp_trials]:
    if 'participantId' in df.columns and 'participant_id' not in df.columns:
        df.rename(columns={'participantId': 'participant_id'}, inplace=True)
    elif 'participantId' in df.columns and 'participant_id' in df.columns:
        df.drop(columns=['participantId'], inplace=True)

print(f"  WCST trials: {len(wcst_trials):,}")
print(f"  Stroop trials: {len(stroop_trials):,}")
print(f"  PRP trials: {len(prp_trials):,}")

# ============================================================================
# 2. WCST ERROR CASCADES
# ============================================================================
print("\n[2/7] Analyzing WCST error cascades...")

def _parse_wcst_extra(extra_str):
    """Parse extra field to extract isPE"""
    if not isinstance(extra_str, str):
        return False
    try:
        extra_dict = ast.literal_eval(extra_str)
        return extra_dict.get('isPE', False)
    except (ValueError, SyntaxError):
        return False

# Extract perseverative errors
wcst_trials['is_pe'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials = wcst_trials.sort_values(['participant_id', 'trialIndex'])

# Identify error chains
wcst_cascades = []
for pid, grp in wcst_trials.groupby('participant_id'):
    errors = grp['is_pe'].values.astype(int)

    # Find consecutive error sequences
    chains = []
    current_chain = 0
    for is_error in errors:
        if is_error:
            current_chain += 1
        else:
            if current_chain > 0:
                chains.append(current_chain)
            current_chain = 0
    if current_chain > 0:
        chains.append(current_chain)

    # Cascade metrics
    max_chain = max(chains) if chains else 0
    num_chains = len(chains)
    total_errors = sum(errors)
    cascade_prop = sum([c for c in chains if c > 1]) / total_errors if total_errors > 0 else 0

    wcst_cascades.append({
        'participant_id': pid,
        'wcst_max_chain': max_chain,
        'wcst_num_chains': num_chains,
        'wcst_total_pe': total_errors,
        'wcst_cascade_prop': cascade_prop
    })

wcst_cascade_df = pd.DataFrame(wcst_cascades)
print(f"  Max chain length range: {wcst_cascade_df['wcst_max_chain'].min()}-{wcst_cascade_df['wcst_max_chain'].max()}")
print(f"  Mean cascade proportion: {wcst_cascade_df['wcst_cascade_prop'].mean():.2%}")

# ============================================================================
# 3. STROOP ERROR CASCADES
# ============================================================================
print("\n[3/7] Analyzing Stroop error cascades...")

stroop_trials['is_error'] = (stroop_trials['correct'] == False).astype(int)
stroop_trials = stroop_trials.sort_values(['participant_id', 'trial'])

stroop_cascades = []
for pid, grp in stroop_trials.groupby('participant_id'):
    errors = grp['is_error'].values

    # Find consecutive error sequences
    chains = []
    current_chain = 0
    for is_error in errors:
        if is_error:
            current_chain += 1
        else:
            if current_chain > 0:
                chains.append(current_chain)
            current_chain = 0
    if current_chain > 0:
        chains.append(current_chain)

    # Cascade metrics
    max_chain = max(chains) if chains else 0
    num_chains = len(chains)
    total_errors = sum(errors)
    cascade_prop = sum([c for c in chains if c > 1]) / total_errors if total_errors > 0 else 0

    stroop_cascades.append({
        'participant_id': pid,
        'stroop_max_chain': max_chain,
        'stroop_num_chains': num_chains,
        'stroop_total_errors': total_errors,
        'stroop_cascade_prop': cascade_prop
    })

stroop_cascade_df = pd.DataFrame(stroop_cascades)
print(f"  Max chain length range: {stroop_cascade_df['stroop_max_chain'].min()}-{stroop_cascade_df['stroop_max_chain'].max()}")
print(f"  Mean cascade proportion: {stroop_cascade_df['stroop_cascade_prop'].mean():.2%}")

# ============================================================================
# 4. PRP ERROR CASCADES
# ============================================================================
print("\n[4/7] Analyzing PRP error cascades...")

# Define PRP error as either T1 or T2 incorrect
prp_trials['is_error'] = ((prp_trials['t1_correct'] == False) |
                           (prp_trials['t2_correct'] == False)).astype(int)
prp_trials = prp_trials.sort_values(['participant_id', 'idx'])

prp_cascades = []
for pid, grp in prp_trials.groupby('participant_id'):
    errors = grp['is_error'].values

    # Find consecutive error sequences
    chains = []
    current_chain = 0
    for is_error in errors:
        if is_error:
            current_chain += 1
        else:
            if current_chain > 0:
                chains.append(current_chain)
            current_chain = 0
    if current_chain > 0:
        chains.append(current_chain)

    # Cascade metrics
    max_chain = max(chains) if chains else 0
    num_chains = len(chains)
    total_errors = sum(errors)
    cascade_prop = sum([c for c in chains if c > 1]) / total_errors if total_errors > 0 else 0

    prp_cascades.append({
        'participant_id': pid,
        'prp_max_chain': max_chain,
        'prp_num_chains': num_chains,
        'prp_total_errors': total_errors,
        'prp_cascade_prop': cascade_prop
    })

prp_cascade_df = pd.DataFrame(prp_cascades)
print(f"  Max chain length range: {prp_cascade_df['prp_max_chain'].min()}-{prp_cascade_df['prp_max_chain'].max()}")
print(f"  Mean cascade proportion: {prp_cascade_df['prp_cascade_prop'].mean():.2%}")

# ============================================================================
# 5. MERGE AND TEST UCLA CORRELATIONS
# ============================================================================
print("\n[5/7] Testing UCLA × cascade correlations...")

# Debug: Check participant counts
print(f"\n  Debug - Participants per dataset:")
print(f"    WCST: {len(wcst_cascade_df)}")
print(f"    Stroop: {len(stroop_cascade_df)}")
print(f"    PRP: {len(prp_cascade_df)}")
print(f"    UCLA: {len(ucla_data)}")
print(f"    Demographics: {len(participants)}")

# Merge all data
master = wcst_cascade_df.merge(stroop_cascade_df, on='participant_id', how='outer')
print(f"  After WCST+Stroop merge: {len(master)}")
master = master.merge(prp_cascade_df, on='participant_id', how='outer')
print(f"  After +PRP merge: {len(master)}")
master = master.merge(ucla_data, on='participant_id', how='left')
print(f"  After +UCLA merge: {len(master)}, with UCLA: {master['ucla_total'].notna().sum()}")
master = master.merge(participants[['participant_id', 'gender_male', 'age']], on='participant_id', how='left')
print(f"  After +demographics merge: {len(master)}")
master = master.merge(dass_pivot[['participant_id', 'dass_total']], on='participant_id', how='left')
master = master.dropna(subset=['ucla_total'])

print(f"\n  Complete cases: {len(master)}")

# Correlations
cascade_vars = [
    'wcst_max_chain', 'wcst_cascade_prop',
    'stroop_max_chain', 'stroop_cascade_prop',
    'prp_max_chain', 'prp_cascade_prop'
]

correlations = []
for var in cascade_vars:
    if var in master.columns:
        valid = master[[var, 'ucla_total']].dropna()
        if len(valid) >= 20:
            r, p = stats.pearsonr(valid['ucla_total'], valid[var])
            correlations.append({
                'Variable': var,
                'r': r,
                'p': p,
                'N': len(valid),
                'Sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            })

corr_df = pd.DataFrame(correlations)
print("\n  UCLA × Cascade Correlations:")
print(corr_df.to_string(index=False))

# ============================================================================
# 6. GENDER-STRATIFIED ANALYSES
# ============================================================================
print("\n[6/7] Testing gender moderation...")

gender_results = []
for var in cascade_vars:
    if var in master.columns:
        for gender, label in [(0, 'Female'), (1, 'Male')]:
            subset = master[(master['gender_male'] == gender) &
                          master[[var, 'ucla_total']].notna().all(axis=1)]
            if len(subset) >= 10:
                r, p = stats.pearsonr(subset['ucla_total'], subset[var])
                gender_results.append({
                    'Variable': var,
                    'Gender': label,
                    'r': r,
                    'p': p,
                    'N': len(subset)
                })

gender_df = pd.DataFrame(gender_results)
print("\n  Gender-Stratified Correlations:")
for var in cascade_vars:
    var_data = gender_df[gender_df['Variable'] == var]
    if not var_data.empty:
        print(f"\n  {var}:")
        for _, row in var_data.iterrows():
            sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
            print(f"    {row['Gender']}: r={row['r']:.3f}, p={row['p']:.3f} {sig} (N={row['N']})")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[7/7] Creating visualizations...")

# Plot 1: UCLA × WCST Max Chain
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# WCST
valid = master[['ucla_total', 'wcst_max_chain', 'gender_male']].dropna()
if len(valid) >= 20:
    for gender, label, marker, color in [(0, 'Female', 'o', '#E74C3C'), (1, 'Male', 's', '#3498DB')]:
        subset = valid[valid['gender_male'] == gender]
        axes[0].scatter(subset['ucla_total'], subset['wcst_max_chain'],
                       alpha=0.6, label=label, marker=marker, s=80, color=color)
    axes[0].set_xlabel('UCLA Loneliness')
    axes[0].set_ylabel('WCST Max Error Chain')
    axes[0].set_title('WCST Error Cascades')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

# Stroop
valid = master[['ucla_total', 'stroop_max_chain', 'gender_male']].dropna()
if len(valid) >= 20:
    for gender, label, marker, color in [(0, 'Female', 'o', '#E74C3C'), (1, 'Male', 's', '#3498DB')]:
        subset = valid[valid['gender_male'] == gender]
        axes[1].scatter(subset['ucla_total'], subset['stroop_max_chain'],
                       alpha=0.6, label=label, marker=marker, s=80, color=color)
    axes[1].set_xlabel('UCLA Loneliness')
    axes[1].set_ylabel('Stroop Max Error Chain')
    axes[1].set_title('Stroop Error Cascades')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

# PRP
valid = master[['ucla_total', 'prp_max_chain', 'gender_male']].dropna()
if len(valid) >= 20:
    for gender, label, marker, color in [(0, 'Female', 'o', '#E74C3C'), (1, 'Male', 's', '#3498DB')]:
        subset = valid[valid['gender_male'] == gender]
        axes[2].scatter(subset['ucla_total'], subset['prp_max_chain'],
                       alpha=0.6, label=label, marker=marker, s=80, color=color)
    axes[2].set_xlabel('UCLA Loneliness')
    axes[2].set_ylabel('PRP Max Error Chain')
    axes[2].set_title('PRP Error Cascades')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_cascade_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Cascade proportion distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (var, title) in enumerate([
    ('wcst_cascade_prop', 'WCST'),
    ('stroop_cascade_prop', 'Stroop'),
    ('prp_cascade_prop', 'PRP')
]):
    if var in master.columns:
        data = master[var].dropna()
        axes[idx].hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {data.mean():.2%}')
        axes[idx].set_xlabel('Cascade Proportion')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{title} Cascade Proportion')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cascade_prop_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\nSaving results...")

# Save master dataset
master.to_csv(OUTPUT_DIR / "error_cascade_master.csv", index=False, encoding='utf-8-sig')

# Save correlations
corr_df.to_csv(OUTPUT_DIR / "ucla_cascade_correlations.csv", index=False, encoding='utf-8-sig')
gender_df.to_csv(OUTPUT_DIR / "gender_cascade_correlations.csv", index=False, encoding='utf-8-sig')

# Summary report
with open(OUTPUT_DIR / "ERROR_CASCADE_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TRIAL-LEVEL ERROR CASCADE ANALYSIS - SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-"*80 + "\n")
    f.write("Do errors cluster in temporal chains, and does UCLA loneliness predict\n")
    f.write("cascade severity across WCST, Stroop, and PRP tasks?\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-"*80 + "\n\n")

    f.write("CASCADE PREVALENCE:\n")
    f.write(f"  WCST: {wcst_cascade_df['wcst_cascade_prop'].mean():.1%} of errors occur in chains\n")
    f.write(f"  Stroop: {stroop_cascade_df['stroop_cascade_prop'].mean():.1%} of errors occur in chains\n")
    f.write(f"  PRP: {prp_cascade_df['prp_cascade_prop'].mean():.1%} of errors occur in chains\n\n")

    f.write("UCLA × CASCADE CORRELATIONS:\n")
    f.write("-"*80 + "\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\n")

    f.write("GENDER-STRATIFIED CORRELATIONS:\n")
    f.write("-"*80 + "\n")
    for var in cascade_vars:
        var_data = gender_df[gender_df['Variable'] == var]
        if not var_data.empty:
            f.write(f"\n{var}:\n")
            for _, row in var_data.iterrows():
                sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
                f.write(f"  {row['Gender']}: r={row['r']:.3f}, p={row['p']:.3f} {sig} (N={int(row['N'])})\n")

    f.write("\n" + "="*80 + "\n")
    f.write("Full results saved to: " + str(OUTPUT_DIR) + "\n")

print("\n" + "="*80)
print("✓ Error Cascade Analysis complete!")
print("="*80)
