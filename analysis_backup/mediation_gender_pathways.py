#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mediation Analysis: Gender-Specific Pathways from UCLA Loneliness to WCST Performance

Tests whether DASS subscales (depression, anxiety, stress) mediate the
UCLA → WCST perseverative errors relationship, and whether mediation
differs by gender (explaining the gender moderation effect).

Theoretical mechanisms:
1. Rumination pathway: UCLA → Depression → Perseverative Errors
2. Stress reactivity: UCLA → Stress → Cognitive Flexibility Impairment
3. Anxiety pathway: UCLA → Anxiety → Set-Shifting Deficit
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
OUTPUT_DIR = Path("results/analysis_outputs/mediation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MEDIATION ANALYSIS: Gender-Specific Pathways")
print("UCLA Loneliness → DASS Subscales → WCST Perseverative Errors")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1] Loading data...")

# Load participants
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

# Load surveys
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})

# Load WCST trials
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
# Already has participant_id column

print(f"  Loaded {len(participants)} participants")
print(f"  Loaded {len(surveys)} survey responses")
print(f"  Loaded {len(wcst_trials)} WCST trials")

# ============================================================================
# 2. EXTRACT UCLA LONELINESS
# ============================================================================

print("\n[2] Extracting UCLA loneliness scores...")

ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores = ucla_scores.rename(columns={'score': 'ucla_total'})
ucla_scores = ucla_scores.dropna(subset=['ucla_total'])

print(f"  UCLA scores for {len(ucla_scores)} participants")
print(f"  Mean: {ucla_scores['ucla_total'].mean():.2f}, SD: {ucla_scores['ucla_total'].std():.2f}")

# ============================================================================
# 3. EXTRACT DASS-21 SUBSCALES
# ============================================================================

print("\n[3] Extracting DASS-21 subscales...")

dass_subscales = surveys[surveys['surveyName'] == 'dass'][['participant_id', 'score_D', 'score_A', 'score_S', 'score']].copy()
dass_subscales = dass_subscales.rename(columns={
    'score_D': 'dass_depression',
    'score_A': 'dass_anxiety',
    'score_S': 'dass_stress',
    'score': 'dass_total'
})
dass_subscales = dass_subscales.dropna(subset=['dass_depression', 'dass_anxiety', 'dass_stress'])

print(f"  DASS subscales for {len(dass_subscales)} participants")
print(f"  Depression - Mean: {dass_subscales['dass_depression'].mean():.2f}, SD: {dass_subscales['dass_depression'].std():.2f}")
print(f"  Anxiety - Mean: {dass_subscales['dass_anxiety'].mean():.2f}, SD: {dass_subscales['dass_anxiety'].std():.2f}")
print(f"  Stress - Mean: {dass_subscales['dass_stress'].mean():.2f}, SD: {dass_subscales['dass_stress'].std():.2f}")

# ============================================================================
# 4. COMPUTE WCST PERSEVERATIVE ERROR RATE
# ============================================================================

print("\n[4] Computing WCST perseverative error rate...")

def parse_wcst_extra(extra_str):
    """Parse WCST extra field for isPE flag"""
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_parsed'] = wcst_trials['extra'].apply(parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_parsed'].apply(lambda x: x.get('isPE', False))

# Filter valid trials
wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['rt_ms'] > 0)
].copy()

# Compute perseverative error rate per participant
wcst_pe_rate = (
    wcst_valid.groupby('participant_id')
    .agg(
        total_trials=('trial_index', 'count'),
        pe_count=('is_pe', 'sum')
    )
    .reset_index()
)
wcst_pe_rate['pe_rate'] = (wcst_pe_rate['pe_count'] / wcst_pe_rate['total_trials']) * 100

print(f"  WCST PE rate for {len(wcst_pe_rate)} participants")
print(f"  Mean PE rate: {wcst_pe_rate['pe_rate'].mean():.2f}%, SD: {wcst_pe_rate['pe_rate'].std():.2f}%")

# ============================================================================
# 5. MERGE MASTER DATASET
# ============================================================================

print("\n[5] Merging master dataset...")

master = participants[['participant_id', 'age', 'gender']].copy()
master = master.merge(ucla_scores, on='participant_id', how='inner')
master = master.merge(dass_subscales, on='participant_id', how='inner')
master = master.merge(wcst_pe_rate[['participant_id', 'pe_rate']], on='participant_id', how='inner')

# Drop missing values
master = master.dropna(subset=['age', 'gender', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'pe_rate'])

# Recode gender from Korean to English
master['gender'] = master['gender'].map({'남성': 'male', '여성': 'female'})
master = master.dropna(subset=['gender'])

print(f"  Final N = {len(master)}")
print(f"  Males: {(master['gender'] == 'male').sum()}, Females: {(master['gender'] == 'female').sum()}")

# Standardize variables
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_dass_dep'] = scaler.fit_transform(master[['dass_depression']])
master['z_dass_anx'] = scaler.fit_transform(master[['dass_anxiety']])
master['z_dass_str'] = scaler.fit_transform(master[['dass_stress']])
master['z_age'] = scaler.fit_transform(master[['age']])

# ============================================================================
# 6. BOOTSTRAP MEDIATION FUNCTION
# ============================================================================

def bootstrap_mediation(X, M, Y, n_bootstrap=5000):
    """
    Bootstrap mediation analysis

    X: Independent variable (UCLA)
    M: Mediator (DASS subscale)
    Y: Dependent variable (WCST PE rate)

    Returns:
    - a_path: X → M coefficient
    - b_path: M → Y coefficient (controlling X)
    - c_path: X → Y total effect
    - c_prime: X → Y direct effect (controlling M)
    - indirect: a*b (mediated effect)
    - 95% CI for indirect effect
    """

    # Original coefficients
    # a path: X → M
    a_path = np.corrcoef(X, M)[0, 1] * (np.std(M) / np.std(X))

    # b path: M → Y (controlling X)
    # Use multiple regression: Y ~ X + M
    X_mat = np.column_stack([X, M])
    X_mat = np.column_stack([np.ones(len(X)), X_mat])  # Add intercept
    coeffs = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    b_path = coeffs[2]  # M → Y coefficient

    # c path: X → Y (total effect)
    X_simple = np.column_stack([np.ones(len(X)), X])
    c_path = np.linalg.lstsq(X_simple, Y, rcond=None)[0][1]

    # c' path: X → Y (direct effect, controlling M)
    c_prime = coeffs[1]  # X → Y coefficient from Y ~ X + M

    # Indirect effect
    indirect = a_path * b_path

    # Bootstrap
    indirect_boot = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        M_boot = M[indices]
        Y_boot = Y[indices]

        # a path
        a_boot = np.corrcoef(X_boot, M_boot)[0, 1] * (np.std(M_boot) / np.std(X_boot))

        # b path
        X_mat_boot = np.column_stack([np.ones(len(X_boot)), X_boot, M_boot])
        coeffs_boot = np.linalg.lstsq(X_mat_boot, Y_boot, rcond=None)[0]
        b_boot = coeffs_boot[2]

        indirect_boot.append(a_boot * b_boot)

    # 95% CI
    ci_lower = np.percentile(indirect_boot, 2.5)
    ci_upper = np.percentile(indirect_boot, 97.5)

    # Proportion mediated
    if c_path != 0:
        prop_mediated = indirect / c_path
    else:
        prop_mediated = np.nan

    return {
        'a_path': a_path,
        'b_path': b_path,
        'c_path': c_path,
        'c_prime': c_prime,
        'indirect': indirect,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'prop_mediated': prop_mediated,
        'significant': (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
    }

# ============================================================================
# 7. MEDIATION ANALYSES BY GENDER
# ============================================================================

print("\n[7] Running mediation analyses...")

mediators = ['z_dass_dep', 'z_dass_anx', 'z_dass_str']
mediator_labels = ['Depression', 'Anxiety', 'Stress']

results_list = []

for gender in ['male', 'female']:
    print(f"\n  {gender.upper()} PARTICIPANTS (N={len(master[master['gender'] == gender])})")
    print("  " + "-"*70)

    subset = master[master['gender'] == gender].copy()

    X = subset['z_ucla'].values
    Y = subset['pe_rate'].values

    for mediator, label in zip(mediators, mediator_labels):
        M = subset[mediator].values

        result = bootstrap_mediation(X, M, Y, n_bootstrap=5000)

        print(f"\n  {label} Mediation:")
        print(f"    a path (UCLA → {label}):         {result['a_path']:>6.3f}")
        print(f"    b path ({label} → WCST PE):      {result['b_path']:>6.3f}")
        print(f"    c path (Total effect):            {result['c_path']:>6.3f}")
        print(f"    c' path (Direct effect):          {result['c_prime']:>6.3f}")
        print(f"    Indirect effect (a*b):            {result['indirect']:>6.3f}")
        print(f"    95% CI:                           [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
        print(f"    Proportion mediated:              {result['prop_mediated']:>6.2%}")
        print(f"    Significant mediation:            {'YES' if result['significant'] else 'NO'}")

        results_list.append({
            'gender': gender,
            'mediator': label,
            'N': len(subset),
            'a_path': result['a_path'],
            'b_path': result['b_path'],
            'c_path': result['c_path'],
            'c_prime': result['c_prime'],
            'indirect_effect': result['indirect'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'proportion_mediated': result['prop_mediated'],
            'significant': result['significant']
        })

results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_DIR / "mediation_results.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: mediation_results.csv")

# ============================================================================
# 8. VISUALIZATION: PATH DIAGRAMS
# ============================================================================

print("\n[8] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Mediation Pathways: UCLA → DASS → WCST PE (by Gender)', fontsize=16, fontweight='bold')

for idx, (gender, label) in enumerate([(g, l) for g in ['male', 'female'] for l in mediator_labels]):
    row = 0 if gender == 'male' else 1
    col = mediator_labels.index(label)
    ax = axes[row, col]

    subset = results_df[(results_df['gender'] == gender) & (results_df['mediator'] == label)]

    if len(subset) == 0:
        continue

    result = subset.iloc[0]

    # Path diagram coordinates
    ucla_pos = (0.2, 0.5)
    mediator_pos = (0.5, 0.8)
    wcst_pos = (0.8, 0.5)

    # Draw nodes
    ax.scatter(*ucla_pos, s=3000, c='lightblue', edgecolors='black', linewidths=2, zorder=3)
    ax.scatter(*mediator_pos, s=3000, c='lightcoral', edgecolors='black', linewidths=2, zorder=3)
    ax.scatter(*wcst_pos, s=3000, c='lightgreen', edgecolors='black', linewidths=2, zorder=3)

    ax.text(ucla_pos[0], ucla_pos[1], 'UCLA', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(mediator_pos[0], mediator_pos[1], label, ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(wcst_pos[0], wcst_pos[1], 'WCST PE', ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw paths
    # a path: UCLA → Mediator
    ax.annotate('', xy=mediator_pos, xytext=ucla_pos,
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(0.35, 0.68, f'a={result["a_path"]:.3f}', fontsize=10, color='blue', fontweight='bold')

    # b path: Mediator → WCST
    ax.annotate('', xy=wcst_pos, xytext=mediator_pos,
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(0.68, 0.68, f'b={result["b_path"]:.3f}', fontsize=10, color='red', fontweight='bold')

    # c' path: UCLA → WCST (direct)
    ax.annotate('', xy=wcst_pos, xytext=ucla_pos,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', linestyle='dashed'))
    ax.text(0.5, 0.42, f"c'={result['c_prime']:.3f}", fontsize=10, color='gray')

    # Indirect effect text
    sig_marker = '***' if result['significant'] else 'ns'
    ax.text(0.5, 0.15, f"Indirect: {result['indirect_effect']:.3f} {sig_marker}",
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow' if result['significant'] else 'white', alpha=0.7))
    ax.text(0.5, 0.05, f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]",
            ha='center', fontsize=9)

    # Title
    ax.set_title(f"{gender.capitalize()} (N={result['N']})", fontsize=12, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mediation_path_diagrams.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: mediation_path_diagrams.png")

# ============================================================================
# 9. COMPARISON: MALE VS FEMALE MEDIATION STRENGTH
# ============================================================================

print("\n[9] Comparing mediation strength across genders...")

comparison_data = []

for mediator, label in zip(mediators, mediator_labels):
    male_result = results_df[(results_df['gender'] == 'male') & (results_df['mediator'] == label)].iloc[0]
    female_result = results_df[(results_df['gender'] == 'female') & (results_df['mediator'] == label)].iloc[0]

    comparison_data.append({
        'mediator': label,
        'male_indirect': male_result['indirect_effect'],
        'female_indirect': female_result['indirect_effect'],
        'difference': male_result['indirect_effect'] - female_result['indirect_effect'],
        'male_significant': male_result['significant'],
        'female_significant': female_result['significant']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(OUTPUT_DIR / "mediation_gender_comparison.csv", index=False, encoding='utf-8-sig')

print("\n  Indirect Effect Comparison (Male - Female):")
print("  " + "-"*70)
for _, row in comparison_df.iterrows():
    print(f"  {row['mediator']:12s}:  Male={row['male_indirect']:>6.3f}, Female={row['female_indirect']:>6.3f}, Diff={row['difference']:>6.3f}")

# ============================================================================
# 10. MODERATED MEDIATION INDEX
# ============================================================================

print("\n[10] Computing moderated mediation indices...")

# For each mediator, compute index of moderated mediation
# This tests whether the indirect effect differs significantly by gender

moderated_mediation_results = []

for mediator, label in zip(mediators, mediator_labels):
    # Full sample with gender as moderator
    master['gender_coded'] = (master['gender'] == 'male').astype(int)

    # Interaction terms
    master[f'{mediator}_x_gender'] = master[mediator] * master['gender_coded']
    master['ucla_x_gender'] = master['z_ucla'] * master['gender_coded']

    # Path a: UCLA → Mediator (moderated by gender)
    # Mediator ~ UCLA + Gender + UCLA*Gender
    X_a = master[['z_ucla', 'gender_coded', 'ucla_x_gender']].values
    X_a = np.column_stack([np.ones(len(X_a)), X_a])
    Y_a = master[mediator].values
    coeffs_a = np.linalg.lstsq(X_a, Y_a, rcond=None)[0]
    a_interaction = coeffs_a[3]  # UCLA*Gender coefficient

    # Path b: Mediator → WCST PE (moderated by gender)
    # PE ~ UCLA + Mediator + Gender + Mediator*Gender + UCLA*Gender
    X_b = master[['z_ucla', mediator, 'gender_coded', f'{mediator}_x_gender', 'ucla_x_gender']].values
    X_b = np.column_stack([np.ones(len(X_b)), X_b])
    Y_b = master['pe_rate'].values
    coeffs_b = np.linalg.lstsq(X_b, Y_b, rcond=None)[0]
    b_interaction = coeffs_b[4]  # Mediator*Gender coefficient

    moderated_mediation_results.append({
        'mediator': label,
        'a_path_moderation': a_interaction,
        'b_path_moderation': b_interaction,
        'interpretation': 'Gender moderates UCLA→Mediator' if abs(a_interaction) > 0.1 else 'Gender moderates Mediator→WCST' if abs(b_interaction) > 0.1 else 'Minimal moderation'
    })

moderated_df = pd.DataFrame(moderated_mediation_results)
moderated_df.to_csv(OUTPUT_DIR / "moderated_mediation_indices.csv", index=False, encoding='utf-8-sig')

print("\n  Moderated Mediation Indices:")
print("  " + "-"*70)
for _, row in moderated_df.iterrows():
    print(f"  {row['mediator']:12s}:  a_mod={row['a_path_moderation']:>6.3f}, b_mod={row['b_path_moderation']:>6.3f}")
    print(f"                 → {row['interpretation']}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MEDIATION ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")
sig_mediations = results_df[results_df['significant'] == True]
if len(sig_mediations) > 0:
    print("\n  SIGNIFICANT MEDIATION PATHWAYS:")
    for _, row in sig_mediations.iterrows():
        print(f"    • {row['gender'].capitalize()} - {row['mediator']}: Indirect = {row['indirect_effect']:.3f}, 95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
else:
    print("  No significant mediation pathways detected (all 95% CIs include zero)")

print("\n  STRONGEST INDIRECT EFFECTS (regardless of significance):")
top_effects = results_df.nlargest(3, 'indirect_effect', keep='all')
for _, row in top_effects.iterrows():
    print(f"    • {row['gender'].capitalize()} - {row['mediator']}: Indirect = {row['indirect_effect']:.3f}")

print("\nOUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}/")
print("  - mediation_results.csv")
print("  - mediation_path_diagrams.png")
print("  - mediation_gender_comparison.csv")
print("  - moderated_mediation_indices.csv")

print("\n" + "="*80)
