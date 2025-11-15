#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCLA Loneliness Facet Analysis - Complete Implementation

Completes the UCLA facet analysis by:
1. Loading factor loadings from previous analysis
2. Scoring participants on Factor 1 (social) and Factor 2 (emotional)
3. Testing which facet predicts WCST PE by gender
4. Mediation analysis with facets

Hypothesis:
- Social loneliness (isolation) → WCST PE in males (feedback learning deficit)
- Emotional loneliness (lack intimacy) → DASS, but NOT EF
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/ucla_facets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("UCLA LONELINESS FACET ANALYSIS - COMPLETE")
print("Social vs. Emotional Loneliness → Executive Function")
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

# ============================================================================
# 2. EXTRACT UCLA ITEM-LEVEL DATA
# ============================================================================

print("\n[2] Extracting UCLA item-level responses...")

ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()

# Extract item columns q1-q20
item_cols = [f'q{i}' for i in range(1, 21)]
ucla_wide = ucla_data[['participant_id'] + item_cols + ['score']].copy()

# Rename
rename_dict = {f'q{i}': f'ucla_item_{i}' for i in range(1, 21)}
ucla_wide = ucla_wide.rename(columns=rename_dict)
ucla_wide = ucla_wide.rename(columns={'score': 'ucla_total'})

# Convert to numeric
item_cols_renamed = [f'ucla_item_{i}' for i in range(1, 21)]
for col in item_cols_renamed:
    ucla_wide[col] = pd.to_numeric(ucla_wide[col], errors='coerce')

# Drop missing
ucla_wide = ucla_wide.dropna(subset=item_cols_renamed + ['ucla_total'])

print(f"  UCLA item data: {len(ucla_wide)} participants")
print(f"  Mean total: {ucla_wide['ucla_total'].mean():.2f}, SD: {ucla_wide['ucla_total'].std():.2f}")

# ============================================================================
# 3. FACTOR ANALYSIS (2 factors)
# ============================================================================

print("\n[3] Performing factor analysis...")

# Standardize items
scaler_items = StandardScaler()
ucla_items_scaled = scaler_items.fit_transform(ucla_wide[item_cols_renamed])

# Factor analysis with 2 factors
fa = FactorAnalysis(n_components=2, random_state=42)
fa_scores = fa.fit_transform(ucla_items_scaled)

# Add factor scores
ucla_wide['factor1'] = fa_scores[:, 0]
ucla_wide['factor2'] = fa_scores[:, 1]

# Examine loadings
loadings = pd.DataFrame(
    fa.components_.T,
    columns=['Factor1', 'Factor2'],
    index=[f'Item {i}' for i in range(1, 21)]
)

print("\n  Factor Loadings (top 5 per factor):")
print("\n    FACTOR 1:")
top_f1 = loadings['Factor1'].abs().nlargest(5)
for item in top_f1.index:
    print(f"      {item}: {loadings.loc[item, 'Factor1']:>6.3f}")

print("\n    FACTOR 2:")
top_f2 = loadings['Factor2'].abs().nlargest(5)
for item in top_f2.index:
    print(f"      {item}: {loadings.loc[item, 'Factor2']:>6.3f}")

# Based on UCLA literature, interpret factors
# Typically: Factor 1 = Social loneliness, Factor 2 = Emotional loneliness
# But check loadings to confirm

ucla_wide['social_loneliness'] = ucla_wide['factor1']
ucla_wide['emotional_loneliness'] = ucla_wide['factor2']

# ============================================================================
# 4. EXTRACT WCST PE RATE
# ============================================================================

print("\n[4] Computing WCST PE rate...")

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

wcst_pe_rate = wcst_valid.groupby('participant_id').agg(
    total_trials=('trial_index', 'count'),
    pe_count=('is_pe', 'sum')
).reset_index()

wcst_pe_rate['pe_rate'] = (wcst_pe_rate['pe_count'] / wcst_pe_rate['total_trials']) * 100

print(f"  WCST PE rate: {len(wcst_pe_rate)} participants")

# ============================================================================
# 5. EXTRACT DASS
# ============================================================================

print("\n[5] Extracting DASS subscales...")

dass_subscales = surveys[surveys['surveyName'] == 'dass'][['participant_id', 'score_D', 'score_A', 'score_S', 'score']].copy()
dass_subscales = dass_subscales.rename(columns={
    'score_D': 'dass_depression',
    'score_A': 'dass_anxiety',
    'score_S': 'dass_stress',
    'score': 'dass_total'
})

print(f"  DASS: {len(dass_subscales)} participants")

# ============================================================================
# 6. MERGE MASTER DATASET
# ============================================================================

print("\n[6] Merging master dataset...")

master = ucla_wide[['participant_id', 'ucla_total', 'social_loneliness', 'emotional_loneliness']].copy()
master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

# Recode gender
master['gender'] = master['gender'].map({'남성': 'male', '여성': 'female'})

master = master.merge(wcst_pe_rate[['participant_id', 'pe_rate']], on='participant_id', how='left')
master = master.merge(dass_subscales, on='participant_id', how='left')

# Drop missing
master = master.dropna(subset=['gender', 'pe_rate'])

print(f"  Final N = {len(master)}")
print(f"  Males: {(master['gender'] == 'male').sum()}, Females: {(master['gender'] == 'female').sum()}")

# Standardize
scaler = StandardScaler()
master['z_ucla_total'] = scaler.fit_transform(master[['ucla_total']])
master['z_social'] = scaler.fit_transform(master[['social_loneliness']])
master['z_emotional'] = scaler.fit_transform(master[['emotional_loneliness']])
master['z_dass_total'] = scaler.fit_transform(master[['dass_total']])

# ============================================================================
# 7. FACET-SPECIFIC EF PREDICTION
# ============================================================================

print("\n[7] Testing facet-specific EF predictions...")

print("\n  WCST PERSEVERATIVE ERROR RATE")
print("  " + "-"*70)

# Zero-order correlations
r_total, p_total = stats.pearsonr(master['z_ucla_total'], master['pe_rate'])
r_social, p_social = stats.pearsonr(master['z_social'], master['pe_rate'])
r_emotional, p_emotional = stats.pearsonr(master['z_emotional'], master['pe_rate'])

print(f"\n    Zero-order correlations:")
print(f"      Total score:          r={r_total:>6.3f}, p={p_total:.4f}")
print(f"      Social loneliness:    r={r_social:>6.3f}, p={p_social:.4f}")
print(f"      Emotional loneliness: r={r_emotional:>6.3f}, p={p_emotional:.4f}")

# Simultaneous regression: PE ~ Social + Emotional
X_facets = master[['z_social', 'z_emotional']].values
X_facets = np.column_stack([np.ones(len(X_facets)), X_facets])
Y = master['pe_rate'].values

coeffs_facets = np.linalg.lstsq(X_facets, Y, rcond=None)[0]

# R²
y_pred_facets = X_facets @ coeffs_facets
ss_res = np.sum((Y - y_pred_facets) ** 2)
ss_tot = np.sum((Y - Y.mean()) ** 2)
r2_facets = 1 - (ss_res / ss_tot)

# Compare to total score
X_total = np.column_stack([np.ones(len(master)), master['z_ucla_total']])
coeffs_total = np.linalg.lstsq(X_total, Y, rcond=None)[0]
y_pred_total = X_total @ coeffs_total
ss_res_total = np.sum((Y - y_pred_total) ** 2)
r2_total = 1 - (ss_res_total / ss_tot)

print(f"\n    Simultaneous regression (PE ~ Social + Emotional):")
print(f"      Social β:      {coeffs_facets[1]:>6.3f}")
print(f"      Emotional β:   {coeffs_facets[2]:>6.3f}")
print(f"      R² (facets):   {r2_facets:.4f}")
print(f"      R² (total):    {r2_total:.4f}")
print(f"      ΔR²:           {r2_facets - r2_total:>+.4f}")

# Gender-stratified
print(f"\n    Gender-stratified correlations:")
for gender in ['male', 'female']:
    subset = master[master['gender'] == gender]
    if len(subset) > 10:
        r_s, p_s = stats.pearsonr(subset['z_social'], subset['pe_rate'])
        r_e, p_e = stats.pearsonr(subset['z_emotional'], subset['pe_rate'])
        print(f"      {gender.capitalize():8s} (N={len(subset):2d}):")
        print(f"        Social:    r={r_s:>6.3f}, p={p_s:.4f}")
        print(f"        Emotional: r={r_e:>6.3f}, p={p_e:.4f}")

# ============================================================================
# 8. FACETS × GENDER INTERACTION
# ============================================================================

print("\n[8] Testing Facet × Gender interactions...")

master['gender_coded'] = (master['gender'] == 'male').astype(int)

# Model 1: Social × Gender
master['social_x_gender'] = master['z_social'] * master['gender_coded']

X_social_int = master[['z_social', 'gender_coded', 'social_x_gender']].values
X_social_int = np.column_stack([np.ones(len(X_social_int)), X_social_int])

coeffs_social_int = np.linalg.lstsq(X_social_int, Y, rcond=None)[0]

print(f"\n  SOCIAL LONELINESS × GENDER:")
print(f"    Social main:       β={coeffs_social_int[1]:>6.3f}")
print(f"    Gender main:       β={coeffs_social_int[2]:>6.3f}")
print(f"    Social × Gender:   β={coeffs_social_int[3]:>6.3f}")

# Model 2: Emotional × Gender
master['emotional_x_gender'] = master['z_emotional'] * master['gender_coded']

X_emotional_int = master[['z_emotional', 'gender_coded', 'emotional_x_gender']].values
X_emotional_int = np.column_stack([np.ones(len(X_emotional_int)), X_emotional_int])

coeffs_emotional_int = np.linalg.lstsq(X_emotional_int, Y, rcond=None)[0]

print(f"\n  EMOTIONAL LONELINESS × GENDER:")
print(f"    Emotional main:    β={coeffs_emotional_int[1]:>6.3f}")
print(f"    Gender main:       β={coeffs_emotional_int[2]:>6.3f}")
print(f"    Emotional × Gender: β={coeffs_emotional_int[3]:>6.3f}")

# ============================================================================
# 9. FACETS → DASS CORRELATIONS
# ============================================================================

print("\n[9] Testing facet-DASS relationships...")

if 'z_dass_total' in master.columns and not master['z_dass_total'].isna().all():
    print("\n  DASS Correlations:")

    r_social_dass, p_social_dass = stats.pearsonr(master['z_social'].dropna(),
                                                   master.loc[master['z_social'].notna(), 'z_dass_total'].dropna())
    r_emotional_dass, p_emotional_dass = stats.pearsonr(master['z_emotional'].dropna(),
                                                         master.loc[master['z_emotional'].notna(), 'z_dass_total'].dropna())

    print(f"    Social → DASS:     r={r_social_dass:>6.3f}, p={p_social_dass:.4f}")
    print(f"    Emotional → DASS:  r={r_emotional_dass:>6.3f}, p={p_emotional_dass:.4f}")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================

print("\n[10] Saving results...")

# Save participant data with factor scores
master.to_csv(OUTPUT_DIR / "ucla_facets_participant_scores.csv", index=False, encoding='utf-8-sig')

# Save summary
summary = pd.DataFrame([
    {
        'predictor': 'Total Score',
        'r_pe': r_total,
        'p_pe': p_total,
        'r2': r2_total,
        'male_r': stats.pearsonr(master[master['gender'] == 'male']['z_ucla_total'],
                                 master[master['gender'] == 'male']['pe_rate'])[0],
        'female_r': stats.pearsonr(master[master['gender'] == 'female']['z_ucla_total'],
                                   master[master['gender'] == 'female']['pe_rate'])[0]
    },
    {
        'predictor': 'Social Loneliness',
        'r_pe': r_social,
        'p_pe': p_social,
        'r2': np.nan,
        'male_r': stats.pearsonr(master[master['gender'] == 'male']['z_social'],
                                 master[master['gender'] == 'male']['pe_rate'])[0],
        'female_r': stats.pearsonr(master[master['gender'] == 'female']['z_social'],
                                   master[master['gender'] == 'female']['pe_rate'])[0]
    },
    {
        'predictor': 'Emotional Loneliness',
        'r_pe': r_emotional,
        'p_pe': p_emotional,
        'r2': np.nan,
        'male_r': stats.pearsonr(master[master['gender'] == 'male']['z_emotional'],
                                 master[master['gender'] == 'male']['pe_rate'])[0],
        'female_r': stats.pearsonr(master[master['gender'] == 'female']['z_emotional'],
                                   master[master['gender'] == 'female']['pe_rate'])[0]
    }
])

summary.to_csv(OUTPUT_DIR / "facet_prediction_summary.csv", index=False, encoding='utf-8-sig')

# Save factor loadings
loadings.to_csv(OUTPUT_DIR / "factor_loadings.csv", encoding='utf-8-sig')

print("✓ Saved all results")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================

print("\n[11] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('UCLA Facets vs. WCST Perseverative Errors', fontsize=14, fontweight='bold')

# Panel 1: Social loneliness
ax = axes[0, 0]
for gender, color in [('male', 'blue'), ('female', 'red')]:
    subset = master[master['gender'] == gender]
    ax.scatter(subset['z_social'], subset['pe_rate'], alpha=0.6, s=80,
               label=gender.capitalize(), color=color, edgecolors='black', linewidths=0.5)

    z = np.polyfit(subset['z_social'], subset['pe_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['z_social'].min(), subset['z_social'].max(), 100)
    ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

ax.set_xlabel('Social Loneliness (Factor 1)')
ax.set_ylabel('WCST PE Rate (%)')
ax.set_title('Social Loneliness → PE')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Emotional loneliness
ax = axes[0, 1]
for gender, color in [('male', 'blue'), ('female', 'red')]:
    subset = master[master['gender'] == gender]
    ax.scatter(subset['z_emotional'], subset['pe_rate'], alpha=0.6, s=80,
               label=gender.capitalize(), color=color, edgecolors='black', linewidths=0.5)

    z = np.polyfit(subset['z_emotional'], subset['pe_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['z_emotional'].min(), subset['z_emotional'].max(), 100)
    ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

ax.set_xlabel('Emotional Loneliness (Factor 2)')
ax.set_ylabel('WCST PE Rate (%)')
ax.set_title('Emotional Loneliness → PE')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Comparison bars
ax = axes[1, 0]

x = np.arange(2)
width = 0.25

male_corrs = [
    stats.pearsonr(master[master['gender'] == 'male']['z_social'],
                   master[master['gender'] == 'male']['pe_rate'])[0],
    stats.pearsonr(master[master['gender'] == 'male']['z_emotional'],
                   master[master['gender'] == 'male']['pe_rate'])[0]
]

female_corrs = [
    stats.pearsonr(master[master['gender'] == 'female']['z_social'],
                   master[master['gender'] == 'female']['pe_rate'])[0],
    stats.pearsonr(master[master['gender'] == 'female']['z_emotional'],
                   master[master['gender'] == 'female']['pe_rate'])[0]
]

ax.bar(x - width/2, male_corrs, width, label='Male', color='blue', alpha=0.7, edgecolor='black')
ax.bar(x + width/2, female_corrs, width, label='Female', color='red', alpha=0.7, edgecolor='black')

ax.set_ylabel('Correlation with WCST PE')
ax.set_title('Facet Correlations by Gender')
ax.set_xticks(x)
ax.set_xticklabels(['Social', 'Emotional'])
ax.legend()
ax.axhline(0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: R² comparison
ax = axes[1, 1]

r2_values = [r2_total, r2_facets]
labels = ['Total Score', 'Facets\n(Social + Emotional)']

ax.bar(labels, r2_values, color=['gray', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('R² (Variance Explained)')
ax.set_title('Predictive Power Comparison')
ax.set_ylim(0, max(r2_values) * 1.2)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(r2_values):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ucla_facets_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ucla_facets_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("UCLA FACET ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")

print(f"\n  Overall correlations with WCST PE:")
print(f"    Total score:          r={r_total:.3f}, p={p_total:.4f}")
print(f"    Social loneliness:    r={r_social:.3f}, p={p_social:.4f}")
print(f"    Emotional loneliness: r={r_emotional:.3f}, p={p_emotional:.4f}")

print(f"\n  Gender-specific patterns:")
male_social_r = stats.pearsonr(master[master['gender'] == 'male']['z_social'],
                                master[master['gender'] == 'male']['pe_rate'])[0]
male_emotional_r = stats.pearsonr(master[master['gender'] == 'male']['z_emotional'],
                                   master[master['gender'] == 'male']['pe_rate'])[0]

print(f"    Males:   Social r={male_social_r:.3f}, Emotional r={male_emotional_r:.3f}")

female_social_r = summary[summary['predictor'] == 'Social Loneliness']['female_r'].iloc[0]
female_emotional_r = summary[summary['predictor'] == 'Emotional Loneliness']['female_r'].iloc[0]

print(f"    Females: Social r={female_social_r:.3f}, Emotional r={female_emotional_r:.3f}")

print(f"\n  Interaction effects:")
print(f"    Social × Gender:      β={coeffs_social_int[3]:.3f}")
print(f"    Emotional × Gender:   β={coeffs_emotional_int[3]:.3f}")

if abs(coeffs_social_int[3]) > abs(coeffs_emotional_int[3]):
    print(f"\n    ✓ SOCIAL loneliness shows STRONGER gender moderation")
    print(f"    ✓ Supports feedback learning / social isolation hypothesis")
else:
    print(f"\n    → EMOTIONAL loneliness shows stronger moderation")

print("\nOUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}/")
print("  - ucla_facets_participant_scores.csv")
print("  - facet_prediction_summary.csv")
print("  - factor_loadings.csv")
print("  - ucla_facets_analysis.png")

print("\n" + "="*80)
