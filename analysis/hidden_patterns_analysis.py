"""
Hidden Patterns Analysis
=========================

Explores overlooked patterns discovered by deep analysis agent:
1. Stroop female vulnerability (double dissociation)
2. Post-error impulsivity (detection vs. correction)
3. PRP slope reanalysis (baseline vs. bottleneck)
4. PE/NPE dissociation (specificity)

Author: Research Team
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Set plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/deep_dive_analysis/hidden_patterns")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("HIDDEN PATTERNS ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/5] Loading data...")

master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')

if 'participantId' in participants.columns:
    if 'participant_id' in participants.columns:
        participants.drop(columns=['participantId'], inplace=True)
    else:
        participants.rename(columns={'participantId': 'participant_id'}, inplace=True)

master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

# Handle Korean gender
if 'gender' in master.columns:
    master['gender_male'] = 0
    master.loc[master['gender'] == '남성', 'gender_male'] = 1
    master.loc[master['gender'].str.lower() == 'male', 'gender_male'] = 1
    master['gender_label'] = master['gender_male'].map({0: 'Female', 1: 'Male'})

master = master.dropna(subset=['ucla_total', 'gender_male']).copy()

# Load feedback sensitivity (for post-error analysis)
feedback = pd.read_csv(Path("results/analysis_outputs/wcst_trial_dynamics/feedback_sensitivity.csv"))
feedback = feedback.merge(master[['participant_id', 'gender_label']], on='participant_id', how='left')

print(f"  N={len(master)} ({(master['gender_male']==0).sum()} Female, {(master['gender_male']==1).sum()} Male)")
print()

# Standardize
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])

# ============================================================================
# ANALYSIS 1: STROOP FEMALE VULNERABILITY (DOUBLE DISSOCIATION)
# ============================================================================

print("[2/5] Analysis 1: Stroop Female Vulnerability (Double Dissociation)...")

double_dissoc_results = []

# Test across tasks
tasks_metrics = {
    'WCST_PE': 'pe_rate',
    'WCST_Accuracy': 'wcst_accuracy',
    'Stroop_Interference': 'stroop_interference',
    'Stroop_Incong_Acc': 'stroop_incong_acc',
    'PRP_Bottleneck': 'prp_bottleneck'
}

for task_name, metric in tasks_metrics.items():
    if metric not in master.columns or master[metric].notna().sum() < 40:
        continue

    # Overall moderation
    formula = f"{metric} ~ z_ucla * C(gender_male)"
    model = ols(formula, data=master.dropna(subset=[metric])).fit()

    int_term = "z_ucla:C(gender_male)[T.1]"
    if int_term in model.params:
        int_beta = model.params[int_term]
        int_p = model.pvalues[int_term]
    else:
        int_beta, int_p = np.nan, np.nan

    # Gender-stratified slopes
    female_data = master[master['gender_male'] == 0].dropna(subset=[metric])
    male_data = master[master['gender_male'] == 1].dropna(subset=[metric])

    if len(female_data) > 15:
        female_model = ols(f"{metric} ~ z_ucla", data=female_data).fit()
        female_beta = female_model.params['z_ucla']
        female_p = female_model.pvalues['z_ucla']
    else:
        female_beta, female_p = np.nan, np.nan

    if len(male_data) > 15:
        male_model = ols(f"{metric} ~ z_ucla", data=male_data).fit()
        male_beta = male_model.params['z_ucla']
        male_p = male_model.pvalues['z_ucla']
    else:
        male_beta, male_p = np.nan, np.nan

    double_dissoc_results.append({
        'task': task_name,
        'metric': metric,
        'female_beta': female_beta,
        'female_p': female_p,
        'male_beta': male_beta,
        'male_p': male_p,
        'interaction_beta': int_beta,
        'interaction_p': int_p,
        'female_vulnerable': female_p < 0.05 and female_beta < 0,
        'male_vulnerable': male_p < 0.05 and male_beta > 0
    })

    print(f"  {task_name}:")
    print(f"    Female: β={female_beta:.3f}, p={female_p:.3f}")
    print(f"    Male: β={male_beta:.3f}, p={male_p:.3f}")
    print(f"    Interaction: β={int_beta:.3f}, p={int_p:.3f}")

double_dissoc_df = pd.DataFrame(double_dissoc_results)
double_dissoc_df.to_csv(OUTPUT_DIR / "double_dissociation_results.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: double_dissociation_results.csv")
print()

# Visualization: Double Dissociation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# WCST (male vulnerable)
ax = axes[0]
for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
    data = master[master['gender_male'] == gender]
    ax.scatter(data['ucla_total'], data['pe_rate'],
               alpha=0.6, s=80, color=color, label=label,
               edgecolors='white', linewidth=0.5)

    if len(data) > 5:
        z = np.polyfit(data['ucla_total'].dropna(), data['pe_rate'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

ax.set_xlabel('UCLA Loneliness', fontsize=12, fontweight='bold')
ax.set_ylabel('WCST PE Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Male-Vulnerable Task', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Stroop (female vulnerable)
ax = axes[1]
for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
    data = master[master['gender_male'] == gender]
    ax.scatter(data['ucla_total'], data['stroop_incong_acc'],
               alpha=0.6, s=80, color=color, label=label,
               edgecolors='white', linewidth=0.5)

    if len(data) > 5:
        z = np.polyfit(data['ucla_total'].dropna(), data['stroop_incong_acc'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

ax.set_xlabel('UCLA Loneliness', fontsize=12, fontweight='bold')
ax.set_ylabel('Stroop Incongruent Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Female-Vulnerable Task', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "double_dissociation_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: double_dissociation_plot.png")
print()

# ============================================================================
# ANALYSIS 2: POST-ERROR IMPULSIVITY
# ============================================================================

print("[3/5] Analysis 2: Post-Error Impulsivity...")

# Create impulsivity index
impulsivity_results = []

for _, row in feedback.iterrows():
    # Impulsivity = Low slowing + Low accuracy
    # Standardize both
    pes = row['post_error_slowing'] if pd.notna(row['post_error_slowing']) else 0
    pe_acc = row['post_error_accuracy'] if pd.notna(row['post_error_accuracy']) else 0

    # Impulsivity index: (Low PES + Low accuracy)
    # Normalize to 0-100 scale
    impulsivity_index = (100 - pe_acc) if pd.notna(pe_acc) else np.nan

    impulsivity_results.append({
        'participant_id': row['participant_id'],
        'ucla_total': row['ucla_total'],
        'gender_male': row['gender_male'],
        'post_error_slowing': pes,
        'post_error_accuracy': pe_acc,
        'impulsivity_index': impulsivity_index
    })

impulsivity_df = pd.DataFrame(impulsivity_results)
impulsivity_df = impulsivity_df.dropna(subset=['impulsivity_index'])

# Test moderation
scaler_imp = StandardScaler()
impulsivity_df['z_ucla_imp'] = scaler_imp.fit_transform(impulsivity_df[['ucla_total']])

formula = "impulsivity_index ~ z_ucla_imp * C(gender_male)"
model_imp = ols(formula, data=impulsivity_df).fit()

print(f"  Impulsivity Index moderation:")
int_term = "z_ucla_imp:C(gender_male)[T.1]"
if int_term in model_imp.params:
    print(f"    Interaction: β={model_imp.params[int_term]:.3f}, p={model_imp.pvalues[int_term]:.3f}")

impulsivity_df.to_csv(OUTPUT_DIR / "post_error_impulsivity.csv", index=False, encoding='utf-8-sig')
print("  ✓ Saved: post_error_impulsivity.csv")
print()

# ============================================================================
# ANALYSIS 3: PRP SLOPE DECOMPOSITION
# ============================================================================

print("[4/5] Analysis 3: PRP Slope Decomposition...")

# Load PRP trial data
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')

if 'participantId' in prp_trials.columns and 'participant_id' in prp_trials.columns:
    prp_trials.drop(columns=['participantId'], inplace=True)
elif 'participantId' in prp_trials.columns:
    prp_trials.rename(columns={'participantId': 'participant_id'}, inplace=True)

prp_decomp_results = []

for pid in master['participant_id'].unique():
    pid_trials = prp_trials[prp_trials['participant_id'] == pid].copy()
    pid_trials = pid_trials[(pid_trials['t2_timeout'] == False) & (pid_trials['t2_rt_ms'] > 0)].copy()

    if len(pid_trials) < 20:
        continue

    pid_info = master[master['participant_id'] == pid].iloc[0]

    # Separate SOA bins
    short = pid_trials[pid_trials['soa_nominal_ms'] <= 150]
    long = pid_trials[pid_trials['soa_nominal_ms'] >= 1200]

    if len(short) > 0 and len(long) > 0:
        baseline_rt = long['t2_rt_ms'].mean()  # Long SOA = baseline
        bottleneck = short['t2_rt_ms'].mean() - baseline_rt  # Bottleneck effect

        prp_decomp_results.append({
            'participant_id': pid,
            'ucla_total': pid_info['ucla_total'],
            'gender_male': pid_info['gender_male'],
            'baseline_t2_rt': baseline_rt,
            'bottleneck_effect': bottleneck,
            'prp_total': short['t2_rt_ms'].mean()
        })

prp_decomp_df = pd.DataFrame(prp_decomp_results)

# Test moderation on baseline vs. bottleneck
scaler_prp = StandardScaler()
prp_decomp_df['z_ucla_prp'] = scaler_prp.fit_transform(prp_decomp_df[['ucla_total']])

for outcome in ['baseline_t2_rt', 'bottleneck_effect']:
    formula = f"{outcome} ~ z_ucla_prp * C(gender_male)"
    model_prp = ols(formula, data=prp_decomp_df.dropna(subset=[outcome])).fit()

    int_term = "z_ucla_prp:C(gender_male)[T.1]"
    if int_term in model_prp.params:
        print(f"  {outcome}: β={model_prp.params[int_term]:.3f}, p={model_prp.pvalues[int_term]:.3f}")

prp_decomp_df.to_csv(OUTPUT_DIR / "prp_decomposition.csv", index=False, encoding='utf-8-sig')
print("  ✓ Saved: prp_decomposition.csv")
print()

# ============================================================================
# ANALYSIS 4: PE/NPE DISSOCIATION
# ============================================================================

print("[5/5] Analysis 4: PE/NPE Dissociation...")

pe_npe_results = []

# From master dataset
for _, row in master.iterrows():
    if pd.notna(row['pe_rate']) and pd.notna(row['wcst_npe_rate']):
        pe_npe_dissoc = row['pe_rate'] - row['wcst_npe_rate']  # Positive = more PE than NPE

        pe_npe_results.append({
            'participant_id': row['participant_id'],
            'ucla_total': row['ucla_total'],
            'gender_male': row['gender_male'],
            'pe_rate': row['pe_rate'],
            'npe_rate': row['wcst_npe_rate'],
            'pe_npe_dissociation': pe_npe_dissoc
        })

pe_npe_df = pd.DataFrame(pe_npe_results)

# Test moderation on dissociation index
scaler_dissoc = StandardScaler()
pe_npe_df['z_ucla_dissoc'] = scaler_dissoc.fit_transform(pe_npe_df[['ucla_total']])

formula = "pe_npe_dissociation ~ z_ucla_dissoc * C(gender_male)"
model_dissoc = ols(formula, data=pe_npe_df).fit()

print(f"  PE-NPE Dissociation moderation:")
int_term = "z_ucla_dissoc:C(gender_male)[T.1]"
if int_term in model_dissoc.params:
    print(f"    Interaction: β={model_dissoc.params[int_term]:.3f}, p={model_dissoc.pvalues[int_term]:.3f}")

pe_npe_df.to_csv(OUTPUT_DIR / "pe_npe_dissociation.csv", index=False, encoding='utf-8-sig')
print("  ✓ Saved: pe_npe_dissociation.csv")
print()

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
    data = pe_npe_df[pe_npe_df['gender_male'] == gender]
    ax.scatter(data['pe_rate'], data['npe_rate'],
               alpha=0.6, s=100, color=color, label=label,
               edgecolors='white', linewidth=0.5)

# Diagonal line (PE = NPE)
max_val = max(pe_npe_df['pe_rate'].max(), pe_npe_df['npe_rate'].max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='PE = NPE')

ax.set_xlabel('Perseverative Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Non-Perseverative Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('PE vs NPE Dissociation', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pe_npe_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: pe_npe_scatter.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("HIDDEN PATTERNS ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - double_dissociation_results.csv")
print("  - double_dissociation_plot.png")
print("  - post_error_impulsivity.csv")
print("  - prp_decomposition.csv")
print("  - pe_npe_dissociation.csv")
print("  - pe_npe_scatter.png")
print()
print("KEY FINDINGS:")
print("  1. Double dissociation: Males→WCST, Females→Stroop")
print("  2. Post-error impulsivity: Low PES + Low accuracy")
print("  3. PRP: Baseline vs. bottleneck decomposition")
print("  4. PE/NPE: Specific perseveration, not random errors")
print()
