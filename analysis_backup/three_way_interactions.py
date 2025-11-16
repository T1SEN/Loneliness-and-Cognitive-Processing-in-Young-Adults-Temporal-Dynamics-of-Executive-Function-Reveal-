#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Way Interaction Analysis

Tests complex moderation patterns:
1. UCLA × Gender × Age → WCST PE (developmental moderation)
2. UCLA × Gender × Education → WCST PE (cognitive reserve buffer)
3. UCLA × Gender × DASS → WCST PE (affective state synergy)

Rationale:
- Known: Gender moderates UCLA → WCST
- Question: Is this gender effect age-dependent? education-dependent?
- E.g., gender norms may be stronger in younger adults
- E.g., education may buffer loneliness effects more in females
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
OUTPUT_DIR = Path("results/analysis_outputs/three_way_interactions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("THREE-WAY INTERACTION ANALYSIS")
print("UCLA × Gender × Age/Education/DASS → WCST Perseverative Errors")
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
# Already has participant_id column

print(f"  Loaded {len(participants)} participants")
print(f"  Loaded {len(wcst_trials)} WCST trials")

# ============================================================================
# 2. EXTRACT VARIABLES
# ============================================================================

print("\n[2] Extracting variables...")

# --- UCLA ---
ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores = ucla_scores.rename(columns={'score': 'ucla_total'})
ucla_scores = ucla_scores.dropna(subset=['ucla_total'])

print(f"  UCLA scores: {len(ucla_scores)} participants")

# --- DASS subscales ---
dass_subscales = surveys[surveys['surveyName'] == 'dass'][['participant_id', 'score_D', 'score_A', 'score_S', 'score']].copy()
dass_subscales = dass_subscales.rename(columns={
    'score_D': 'dass_depression',
    'score_A': 'dass_anxiety',
    'score_S': 'dass_stress',
    'score': 'dass_total'
})
dass_subscales = dass_subscales.dropna(subset=['dass_depression', 'dass_anxiety', 'dass_stress'])

print(f"  DASS subscales: {len(dass_subscales)} participants")

# --- WCST PE rate ---
def parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_parsed'] = wcst_trials['extra'].apply(parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_parsed'].apply(lambda x: x.get('isPE', False))

wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['rt_ms'] > 0)
].copy()

wcst_pe_rate = (
    wcst_valid.groupby('participant_id')
    .agg(
        total_trials=('trial_index', 'count'),
        pe_count=('is_pe', 'sum')
    )
    .reset_index()
)
wcst_pe_rate['pe_rate'] = (wcst_pe_rate['pe_count'] / wcst_pe_rate['total_trials']) * 100

print(f"  WCST PE rate: {len(wcst_pe_rate)} participants")

# ============================================================================
# 3. MERGE MASTER DATASET
# ============================================================================

print("\n[3] Merging master dataset...")

master = participants[['participant_id', 'age', 'gender', 'education']].copy()
master = master.merge(ucla_scores, on='participant_id', how='inner')
master = master.merge(dass_subscales, on='participant_id', how='inner')
master = master.merge(wcst_pe_rate[['participant_id', 'pe_rate']], on='participant_id', how='inner')

# Drop missing
master = master.dropna(subset=['age', 'gender', 'ucla_total', 'pe_rate'])

# Recode gender from Korean to English
master['gender'] = master['gender'].map({'남성': 'male', '여성': 'female'})
master = master.dropna(subset=['gender'])

print(f"  Final N = {len(master)}")
print(f"  Males: {(master['gender'] == 'male').sum()}, Females: {(master['gender'] == 'female').sum()}")

# Standardize continuous variables
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_age'] = scaler.fit_transform(master[['age']])
master['z_dass_total'] = scaler.fit_transform(master[['dass_total']])

# Create education groups (if enough variation)
if master['education'].nunique() > 2:
    master['education_group'] = pd.cut(master['education'], bins=2, labels=['Lower', 'Higher'])
else:
    master['education_group'] = master['education']

# Binary gender coding
master['gender_coded'] = (master['gender'] == 'male').astype(int)

# Age groups (median split or tertiles)
master['age_group'] = pd.cut(master['age'], bins=2, labels=['Younger', 'Older'])

print(f"  Age groups: {master['age_group'].value_counts().to_dict()}")
print(f"  Education groups: {master['education_group'].value_counts().to_dict()}")

# ============================================================================
# 4. THREE-WAY INTERACTION: UCLA × Gender × Age
# ============================================================================

print("\n[4] Testing UCLA × Gender × Age → WCST PE...")

# Create interaction terms
master['ucla_x_gender'] = master['z_ucla'] * master['gender_coded']
master['age_coded'] = (master['age_group'] == 'Older').astype(int)
master['ucla_x_age'] = master['z_ucla'] * master['age_coded']
master['gender_x_age'] = master['gender_coded'] * master['age_coded']
master['ucla_x_gender_x_age'] = master['z_ucla'] * master['gender_coded'] * master['age_coded']

# Regression: PE ~ UCLA + Gender + Age + UCLA×Gender + UCLA×Age + Gender×Age + UCLA×Gender×Age
X = master[['z_ucla', 'gender_coded', 'age_coded', 'ucla_x_gender', 'ucla_x_age', 'gender_x_age', 'ucla_x_gender_x_age']].values
X = np.column_stack([np.ones(len(X)), X])
Y = master['pe_rate'].values

coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]

# Compute R-squared
y_pred = X @ coeffs
ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - Y.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Compare to model without 3-way interaction
X_reduced = master[['z_ucla', 'gender_coded', 'age_coded', 'ucla_x_gender', 'ucla_x_age', 'gender_x_age']].values
X_reduced = np.column_stack([np.ones(len(X_reduced)), X_reduced])
coeffs_reduced = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]
y_pred_reduced = X_reduced @ coeffs_reduced
ss_res_reduced = np.sum((Y - y_pred_reduced) ** 2)
r2_reduced = 1 - (ss_res_reduced / ss_tot)

delta_r2 = r2 - r2_reduced

# F-test for 3-way interaction
f_stat = ((ss_res_reduced - ss_res) / 1) / (ss_res / (len(Y) - X.shape[1]))
p_value = 1 - stats.f.cdf(f_stat, 1, len(Y) - X.shape[1])

print(f"\n  Full Model (with 3-way interaction):")
print(f"    R² = {r2:.4f}")
print(f"    Three-way coefficient: β = {coeffs[7]:.4f}")
print(f"\n  Reduced Model (without 3-way):")
print(f"    R² = {r2_reduced:.4f}")
print(f"\n  Incremental Test:")
print(f"    ΔR² = {delta_r2:.4f}")
print(f"    F({1}, {len(Y) - X.shape[1]}) = {f_stat:.4f}, p = {p_value:.4f} {'***' if p_value < 0.05 else 'ns'}")

if p_value < 0.05:
    print(f"\n    *** SIGNIFICANT THREE-WAY INTERACTION ***")
    print(f"    Interpretation: Gender moderation of UCLA→WCST differs by age")

# Simple slopes analysis (stratified by age and gender)
print(f"\n  Simple Slopes (UCLA → WCST PE):")
print("  " + "-"*70)

for age_grp in ['Younger', 'Older']:
    for gender in ['male', 'female']:
        subset = master[(master['age_group'] == age_grp) & (master['gender'] == gender)]
        if len(subset) > 5:
            r, p = stats.pearsonr(subset['z_ucla'], subset['pe_rate'])

            # Linear regression for coefficient
            X_simple = np.column_stack([np.ones(len(subset)), subset['z_ucla']])
            Y_simple = subset['pe_rate'].values
            coeffs_simple = np.linalg.lstsq(X_simple, Y_simple, rcond=None)[0]
            beta = coeffs_simple[1]

            print(f"    {age_grp:8s} × {gender.capitalize():8s} (N={len(subset):2d}):  β={beta:>6.3f}, r={r:>6.3f}, p={p:.4f}")

# Save results
age_interaction_results = {
    'model': 'UCLA × Gender × Age',
    'r2_full': r2,
    'r2_reduced': r2_reduced,
    'delta_r2': delta_r2,
    'three_way_beta': coeffs[7],
    'f_statistic': f_stat,
    'p_value': p_value,
    'significant': p_value < 0.05
}

# ============================================================================
# 5. THREE-WAY INTERACTION: UCLA × Gender × Education
# ============================================================================

print("\n[5] Testing UCLA × Gender × Education → WCST PE...")

# Filter to participants with education data
master_edu = master[master['education_group'].notna()].copy()

if len(master_edu) > 20:
    master_edu['education_coded'] = (master_edu['education_group'] == 'Higher').astype(int)
    master_edu['ucla_x_education'] = master_edu['z_ucla'] * master_edu['education_coded']
    master_edu['gender_x_education'] = master_edu['gender_coded'] * master_edu['education_coded']
    master_edu['ucla_x_gender_x_education'] = master_edu['z_ucla'] * master_edu['gender_coded'] * master_edu['education_coded']

    X_edu = master_edu[['z_ucla', 'gender_coded', 'education_coded', 'ucla_x_gender', 'ucla_x_education', 'gender_x_education', 'ucla_x_gender_x_education']].values
    X_edu = np.column_stack([np.ones(len(X_edu)), X_edu])
    Y_edu = master_edu['pe_rate'].values

    coeffs_edu = np.linalg.lstsq(X_edu, Y_edu, rcond=None)[0]

    y_pred_edu = X_edu @ coeffs_edu
    ss_res_edu = np.sum((Y_edu - y_pred_edu) ** 2)
    ss_tot_edu = np.sum((Y_edu - Y_edu.mean()) ** 2)
    r2_edu = 1 - (ss_res_edu / ss_tot_edu)

    # Reduced model
    X_edu_reduced = master_edu[['z_ucla', 'gender_coded', 'education_coded', 'ucla_x_gender', 'ucla_x_education', 'gender_x_education']].values
    X_edu_reduced = np.column_stack([np.ones(len(X_edu_reduced)), X_edu_reduced])
    coeffs_edu_reduced = np.linalg.lstsq(X_edu_reduced, Y_edu, rcond=None)[0]
    y_pred_edu_reduced = X_edu_reduced @ coeffs_edu_reduced
    ss_res_edu_reduced = np.sum((Y_edu - y_pred_edu_reduced) ** 2)
    r2_edu_reduced = 1 - (ss_res_edu_reduced / ss_tot_edu)

    delta_r2_edu = r2_edu - r2_edu_reduced

    f_stat_edu = ((ss_res_edu_reduced - ss_res_edu) / 1) / (ss_res_edu / (len(Y_edu) - X_edu.shape[1]))
    p_value_edu = 1 - stats.f.cdf(f_stat_edu, 1, len(Y_edu) - X_edu.shape[1])

    print(f"\n  N = {len(master_edu)}")
    print(f"  Full Model R² = {r2_edu:.4f}")
    print(f"  Three-way coefficient: β = {coeffs_edu[7]:.4f}")
    print(f"  ΔR² = {delta_r2_edu:.4f}")
    print(f"  F({1}, {len(Y_edu) - X_edu.shape[1]}) = {f_stat_edu:.4f}, p = {p_value_edu:.4f} {'***' if p_value_edu < 0.05 else 'ns'}")

    if p_value_edu < 0.05:
        print(f"\n    *** SIGNIFICANT THREE-WAY INTERACTION ***")
        print(f"    Interpretation: Gender moderation differs by education level")

    # Simple slopes
    print(f"\n  Simple Slopes (UCLA → WCST PE):")
    print("  " + "-"*70)

    for edu_grp in ['Lower', 'Higher']:
        for gender in ['male', 'female']:
            subset = master_edu[(master_edu['education_group'] == edu_grp) & (master_edu['gender'] == gender)]
            if len(subset) > 5:
                r, p = stats.pearsonr(subset['z_ucla'], subset['pe_rate'])
                X_simple = np.column_stack([np.ones(len(subset)), subset['z_ucla']])
                Y_simple = subset['pe_rate'].values
                coeffs_simple = np.linalg.lstsq(X_simple, Y_simple, rcond=None)[0]
                beta = coeffs_simple[1]
                print(f"    {edu_grp:8s} × {gender.capitalize():8s} (N={len(subset):2d}):  β={beta:>6.3f}, r={r:>6.3f}, p={p:.4f}")

    education_interaction_results = {
        'model': 'UCLA × Gender × Education',
        'r2_full': r2_edu,
        'r2_reduced': r2_edu_reduced,
        'delta_r2': delta_r2_edu,
        'three_way_beta': coeffs_edu[7],
        'f_statistic': f_stat_edu,
        'p_value': p_value_edu,
        'significant': p_value_edu < 0.05
    }
else:
    print(f"  Insufficient data (N={len(master_edu)})")
    education_interaction_results = None

# ============================================================================
# 6. THREE-WAY INTERACTION: UCLA × Gender × DASS
# ============================================================================

print("\n[6] Testing UCLA × Gender × DASS → WCST PE...")

# Create DASS high/low split
master['dass_group'] = pd.cut(master['z_dass_total'], bins=[-np.inf, 0, np.inf], labels=['Low', 'High'])
master['dass_coded'] = (master['dass_group'] == 'High').astype(int)
master['ucla_x_dass'] = master['z_ucla'] * master['dass_coded']
master['gender_x_dass'] = master['gender_coded'] * master['dass_coded']
master['ucla_x_gender_x_dass'] = master['z_ucla'] * master['gender_coded'] * master['dass_coded']

X_dass = master[['z_ucla', 'gender_coded', 'dass_coded', 'ucla_x_gender', 'ucla_x_dass', 'gender_x_dass', 'ucla_x_gender_x_dass']].values
X_dass = np.column_stack([np.ones(len(X_dass)), X_dass])
Y_dass = master['pe_rate'].values

coeffs_dass = np.linalg.lstsq(X_dass, Y_dass, rcond=None)[0]

y_pred_dass = X_dass @ coeffs_dass
ss_res_dass = np.sum((Y_dass - y_pred_dass) ** 2)
ss_tot_dass = np.sum((Y_dass - Y_dass.mean()) ** 2)
r2_dass = 1 - (ss_res_dass / ss_tot_dass)

# Reduced model
X_dass_reduced = master[['z_ucla', 'gender_coded', 'dass_coded', 'ucla_x_gender', 'ucla_x_dass', 'gender_x_dass']].values
X_dass_reduced = np.column_stack([np.ones(len(X_dass_reduced)), X_dass_reduced])
coeffs_dass_reduced = np.linalg.lstsq(X_dass_reduced, Y_dass, rcond=None)[0]
y_pred_dass_reduced = X_dass_reduced @ coeffs_dass_reduced
ss_res_dass_reduced = np.sum((Y_dass - y_pred_dass_reduced) ** 2)
r2_dass_reduced = 1 - (ss_res_dass_reduced / ss_tot_dass)

delta_r2_dass = r2_dass - r2_dass_reduced

f_stat_dass = ((ss_res_dass_reduced - ss_res_dass) / 1) / (ss_res_dass / (len(Y_dass) - X_dass.shape[1]))
p_value_dass = 1 - stats.f.cdf(f_stat_dass, 1, len(Y_dass) - X_dass.shape[1])

print(f"\n  N = {len(master)}")
print(f"  Full Model R² = {r2_dass:.4f}")
print(f"  Three-way coefficient: β = {coeffs_dass[7]:.4f}")
print(f"  ΔR² = {delta_r2_dass:.4f}")
print(f"  F({1}, {len(Y_dass) - X_dass.shape[1]}) = {f_stat_dass:.4f}, p = {p_value_dass:.4f} {'***' if p_value_dass < 0.05 else 'ns'}")

if p_value_dass < 0.05:
    print(f"\n    *** SIGNIFICANT THREE-WAY INTERACTION ***")
    print(f"    Interpretation: Gender moderation synergizes with DASS")

# Simple slopes
print(f"\n  Simple Slopes (UCLA → WCST PE):")
print("  " + "-"*70)

for dass_grp in ['Low', 'High']:
    for gender in ['male', 'female']:
        subset = master[(master['dass_group'] == dass_grp) & (master['gender'] == gender)]
        if len(subset) > 5:
            r, p = stats.pearsonr(subset['z_ucla'], subset['pe_rate'])
            X_simple = np.column_stack([np.ones(len(subset)), subset['z_ucla']])
            Y_simple = subset['pe_rate'].values
            coeffs_simple = np.linalg.lstsq(X_simple, Y_simple, rcond=None)[0]
            beta = coeffs_simple[1]
            print(f"    {dass_grp:8s} DASS × {gender.capitalize():8s} (N={len(subset):2d}):  β={beta:>6.3f}, r={r:>6.3f}, p={p:.4f}")

dass_interaction_results = {
    'model': 'UCLA × Gender × DASS',
    'r2_full': r2_dass,
    'r2_reduced': r2_dass_reduced,
    'delta_r2': delta_r2_dass,
    'three_way_beta': coeffs_dass[7],
    'f_statistic': f_stat_dass,
    'p_value': p_value_dass,
    'significant': p_value_dass < 0.05
}

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print("\n[7] Saving results...")

results_list = [age_interaction_results, dass_interaction_results]
if education_interaction_results:
    results_list.append(education_interaction_results)

results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_DIR / "three_way_interaction_summary.csv", index=False, encoding='utf-8-sig')
print("✓ Saved: three_way_interaction_summary.csv")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n[8] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Three-Way Interactions: UCLA × Gender × Moderator → WCST PE', fontsize=16, fontweight='bold')

# --- Age × Gender ---
for idx, (age_grp, label) in enumerate([('Younger', 'Younger Adults'), ('Older', 'Older Adults')]):
    ax = axes[0, idx]

    for gender in ['male', 'female']:
        subset = master[(master['age_group'] == age_grp) & (master['gender'] == gender)]
        if len(subset) > 5:
            ax.scatter(subset['z_ucla'], subset['pe_rate'], alpha=0.6, s=80, label=gender.capitalize(), edgecolors='black', linewidths=0.5)

            # Fit line
            z = np.polyfit(subset['z_ucla'], subset['pe_rate'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
            ax.plot(x_line, p(x_line), linestyle='--', linewidth=2)

    ax.set_xlabel('UCLA (z-scored)')
    ax.set_ylabel('WCST PE Rate (%)')
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Panel 3: Summary of age interaction
ax = axes[0, 2]
age_summary_data = []
for age_grp in ['Younger', 'Older']:
    for gender in ['male', 'female']:
        subset = master[(master['age_group'] == age_grp) & (master['gender'] == gender)]
        if len(subset) > 5:
            r, _ = stats.pearsonr(subset['z_ucla'], subset['pe_rate'])
            age_summary_data.append({'age': age_grp, 'gender': gender, 'correlation': r})

if age_summary_data:
    age_summary_df = pd.DataFrame(age_summary_data)
    age_pivot = age_summary_df.pivot(index='gender', columns='age', values='correlation')
    sns.heatmap(age_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('UCLA-WCST Correlations')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Gender')

# --- DASS × Gender ---
for idx, (dass_grp, label) in enumerate([('Low', 'Low DASS'), ('High', 'High DASS')]):
    ax = axes[1, idx]

    for gender in ['male', 'female']:
        subset = master[(master['dass_group'] == dass_grp) & (master['gender'] == gender)]
        if len(subset) > 5:
            ax.scatter(subset['z_ucla'], subset['pe_rate'], alpha=0.6, s=80, label=gender.capitalize(), edgecolors='black', linewidths=0.5)

            z = np.polyfit(subset['z_ucla'], subset['pe_rate'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
            ax.plot(x_line, p(x_line), linestyle='--', linewidth=2)

    ax.set_xlabel('UCLA (z-scored)')
    ax.set_ylabel('WCST PE Rate (%)')
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Panel 6: Summary comparison
ax = axes[1, 2]
models = []
delta_r2s = []
p_values = []

for result in results_list:
    models.append(result['model'].split('×')[2].strip())
    delta_r2s.append(result['delta_r2'])
    p_values.append(result['p_value'])

colors = ['red' if p < 0.05 else 'gray' for p in p_values]
ax.bar(models, delta_r2s, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('ΔR² (3-way interaction)')
ax.set_title('Three-Way Interaction Strengths')
ax.set_xlabel('Moderator')
ax.grid(True, alpha=0.3, axis='y')

# Add significance markers
for i, (model, dr2, p) in enumerate(zip(models, delta_r2s, p_values)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(i, dr2 + 0.001, sig, ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'three_way_interactions_comprehensive.png', dpi=300, bbox_inches='tight')
print("✓ Saved: three_way_interactions_comprehensive.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("THREE-WAY INTERACTION ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")

for result in results_list:
    print(f"\n  {result['model']}:")
    print(f"    ΔR² = {result['delta_r2']:.4f}")
    print(f"    F-statistic = {result['f_statistic']:.4f}, p = {result['p_value']:.4f} {'***' if result['p_value'] < 0.05 else 'ns'}")
    print(f"    Three-way β = {result['three_way_beta']:.4f}")

    if result['significant']:
        print(f"    *** SIGNIFICANT THREE-WAY INTERACTION ***")

# Strongest interaction
if len(results_list) > 0:
    strongest = max(results_list, key=lambda x: abs(x['delta_r2']))
    print(f"\n  STRONGEST INTERACTION: {strongest['model']}")
    print(f"    ΔR² = {strongest['delta_r2']:.4f}, p = {strongest['p_value']:.4f}")

print("\nOUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}/")
print("  - three_way_interaction_summary.csv")
print("  - three_way_interactions_comprehensive.png")

print("\n" + "="*80)
