"""
DASS 2×2×2 Stratification Analysis
====================================

Uses median splits (Low/High) instead of tertiles to avoid sparse cells.
Tests anxiety buffering hypothesis with more statistical power.

Design: Depression (Low/High) × Anxiety (Low/High) × Stress (Low/High) = 8 cells

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

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/deep_dive_analysis/dass_2x2x2")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("DASS 2×2×2 STRATIFICATION ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/6] Loading data...")

master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))
participants = pd.read_csv(Path("results/1_participants_info.csv"), encoding='utf-8-sig')

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

# Handle gender
if 'gender' in master.columns:
    master['gender_male'] = 0
    master.loc[master['gender'] == '남성', 'gender_male'] = 1
    master.loc[master['gender'].str.lower() == 'male', 'gender_male'] = 1

# Filter complete cases
required_cols = ['ucla_total', 'pe_rate', 'wcst_accuracy', 'gender_male',
                 'dass_depression', 'dass_anxiety', 'dass_stress']
master = master.dropna(subset=required_cols).copy()

print(f"  N={len(master)} ({(master['gender_male']==0).sum()} Female, {(master['gender_male']==1).sum()} Male)")
print()

# ============================================================================
# CREATE MEDIAN SPLITS
# ============================================================================

print("[2/6] Creating median splits for DASS dimensions...")

for dass_var in ['dass_depression', 'dass_anxiety', 'dass_stress']:
    median_val = master[dass_var].median()

    master[f'{dass_var}_binary'] = 'High'
    master.loc[master[dass_var] <= median_val, f'{dass_var}_binary'] = 'Low'

    print(f"  {dass_var}:")
    print(f"    Median: {median_val:.1f}")
    print(f"    Distribution: {master[f'{dass_var}_binary'].value_counts().to_dict()}")

print()

# ============================================================================
# 2×2×2 STRATIFICATION ANALYSIS
# ============================================================================

print("[3/6] Testing gender moderation in 8 cells (2×2×2)...")

outcomes = ['pe_rate', 'wcst_accuracy']
stratification_results = []

dep_levels = ['Low', 'High']
anx_levels = ['Low', 'High']
stress_levels = ['Low', 'High']

cell_count = 0

for dep in dep_levels:
    for anx in anx_levels:
        for stress in stress_levels:
            cell_count += 1

            # Filter to this cell
            cell_data = master[
                (master['dass_depression_binary'] == dep) &
                (master['dass_anxiety_binary'] == anx) &
                (master['dass_stress_binary'] == stress)
            ].copy()

            n_female = (cell_data['gender_male'] == 0).sum()
            n_male = (cell_data['gender_male'] == 1).sum()

            print(f"\n[{cell_count}/8] {dep} Dep × {anx} Anx × {stress} Stress (N={len(cell_data)}, {n_female}F/{n_male}M)")

            if len(cell_data) < 10:
                print(f"  Skipped - too small")
                continue

            # Standardize UCLA
            scaler = StandardScaler()
            cell_data['z_ucla'] = scaler.fit_transform(cell_data[['ucla_total']])

            for outcome in outcomes:
                if outcome not in cell_data.columns:
                    continue

                # Test moderation
                formula = f"{outcome} ~ z_ucla * C(gender_male)"
                try:
                    model = ols(formula, data=cell_data.dropna(subset=[outcome])).fit()
                    interaction_term = "z_ucla:C(gender_male)[T.1]"

                    if interaction_term in model.params:
                        beta = model.params[interaction_term]
                        pval = model.pvalues[interaction_term]

                        # Gender-stratified correlations
                        female_data = cell_data[cell_data['gender_male'] == 0]
                        male_data = cell_data[cell_data['gender_male'] == 1]

                        female_corr, female_p = np.nan, np.nan
                        male_corr, male_p = np.nan, np.nan

                        if len(female_data) >= 5:
                            female_corr, female_p = stats.pearsonr(
                                female_data['ucla_total'].dropna(),
                                female_data[outcome].dropna()
                            )

                        if len(male_data) >= 5:
                            male_corr, male_p = stats.pearsonr(
                                male_data['ucla_total'].dropna(),
                                male_data[outcome].dropna()
                            )
                    else:
                        beta, pval = np.nan, np.nan
                        female_corr, male_corr = np.nan, np.nan
                        female_p, male_p = np.nan, np.nan

                    stratification_results.append({
                        'depression': dep,
                        'anxiety': anx,
                        'stress': stress,
                        'cell_label': f"{dep}Dep×{anx}Anx×{stress}Stress",
                        'outcome': outcome,
                        'n': len(cell_data.dropna(subset=[outcome])),
                        'n_female': n_female,
                        'n_male': n_male,
                        'interaction_beta': beta,
                        'interaction_pval': pval,
                        'female_corr': female_corr,
                        'female_p': female_p,
                        'male_corr': male_corr,
                        'male_p': male_p
                    })

                    sig_marker = " ***" if pval < 0.001 else " **" if pval < 0.01 else " *" if pval < 0.05 else ""
                    print(f"  {outcome}: β={beta:.3f}, p={pval:.4f}{sig_marker}")
                    print(f"    Female: r={female_corr:.3f}, p={female_p:.4f}")
                    print(f"    Male: r={male_corr:.3f}, p={male_p:.4f}")

                except Exception as e:
                    print(f"  {outcome}: ERROR - {str(e)}")
                    continue

print()

# Save results
strat_df = pd.DataFrame(stratification_results)
strat_df.to_csv(OUTPUT_DIR / "dass_2x2x2_stratification.csv", index=False, encoding='utf-8-sig')
print(f"✓ Saved: dass_2x2x2_stratification.csv ({len(strat_df)} tests)")
print()

# ============================================================================
# ANXIETY BUFFERING HYPOTHESIS
# ============================================================================

print("[4/6] Testing anxiety buffering hypothesis...")
print()

buffering_tests = []

# Focus on "pure" profiles
print("PURE PROFILES COMPARISON:")
print()

# Pure low (Low on all 3)
pure_low = master[
    (master['dass_depression_binary'] == 'Low') &
    (master['dass_anxiety_binary'] == 'Low') &
    (master['dass_stress_binary'] == 'Low')
].copy()

# Pure high (High on all 3)
pure_high = master[
    (master['dass_depression_binary'] == 'High') &
    (master['dass_anxiety_binary'] == 'High') &
    (master['dass_stress_binary'] == 'High')
].copy()

for profile_name, profile_data in [('Pure_Low', pure_low), ('Pure_High', pure_high)]:
    if len(profile_data) < 10:
        print(f"{profile_name}: N={len(profile_data)} (too small)")
        continue

    scaler = StandardScaler()
    profile_data['z_ucla'] = scaler.fit_transform(profile_data[['ucla_total']])

    print(f"{profile_name} (N={len(profile_data)}, {(profile_data['gender_male']==0).sum()}F/{(profile_data['gender_male']==1).sum()}M):")

    for outcome in outcomes:
        try:
            model = ols(f"{outcome} ~ z_ucla * C(gender_male)", data=profile_data).fit()
            if "z_ucla:C(gender_male)[T.1]" in model.params:
                beta = model.params["z_ucla:C(gender_male)[T.1]"]
                pval = model.pvalues["z_ucla:C(gender_male)[T.1]"]

                buffering_tests.append({
                    'profile': profile_name,
                    'outcome': outcome,
                    'n': len(profile_data),
                    'beta': beta,
                    'pval': pval
                })

                sig_marker = " ***" if pval < 0.001 else " **" if pval < 0.01 else " *" if pval < 0.05 else ""
                print(f"  {outcome}: β={beta:.3f}, p={pval:.4f}{sig_marker}")
        except:
            print(f"  {outcome}: Model failed")
    print()

# Compare Low vs High Anxiety (collapsed across Dep/Stress)
print("ANXIETY-SPECIFIC COMPARISON (collapsed across Dep/Stress):")
print()

low_anx_all = master[master['dass_anxiety_binary'] == 'Low'].copy()
high_anx_all = master[master['dass_anxiety_binary'] == 'High'].copy()

for anx_label, anx_data in [('Low_Anxiety', low_anx_all), ('High_Anxiety', high_anx_all)]:
    scaler = StandardScaler()
    anx_data['z_ucla'] = scaler.fit_transform(anx_data[['ucla_total']])

    print(f"{anx_label} (N={len(anx_data)}, {(anx_data['gender_male']==0).sum()}F/{(anx_data['gender_male']==1).sum()}M):")

    for outcome in outcomes:
        try:
            model = ols(f"{outcome} ~ z_ucla * C(gender_male)", data=anx_data).fit()
            if "z_ucla:C(gender_male)[T.1]" in model.params:
                beta = model.params["z_ucla:C(gender_male)[T.1]"]
                pval = model.pvalues["z_ucla:C(gender_male)[T.1]"]

                buffering_tests.append({
                    'profile': anx_label,
                    'outcome': outcome,
                    'n': len(anx_data),
                    'beta': beta,
                    'pval': pval
                })

                sig_marker = " ***" if pval < 0.001 else " **" if pval < 0.01 else " *" if pval < 0.05 else ""
                print(f"  {outcome}: β={beta:.3f}, p={pval:.4f}{sig_marker}")
        except:
            print(f"  {outcome}: Model failed")
    print()

buffering_df = pd.DataFrame(buffering_tests)
buffering_df.to_csv(OUTPUT_DIR / "anxiety_buffering_profiles.csv", index=False, encoding='utf-8-sig')
print(f"✓ Saved: anxiety_buffering_profiles.csv")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("[5/6] Creating visualizations...")

# Heatmap of effect sizes (Anxiety × Depression, collapsing Stress)
pe_results = strat_df[strat_df['outcome'] == 'pe_rate'].copy()

# Average across stress levels
heatmap_data = pe_results.groupby(['depression', 'anxiety']).agg({
    'interaction_beta': 'mean',
    'interaction_pval': lambda x: x.min() if len(x) > 0 else np.nan,
    'n': 'sum'
}).reset_index()

# Create pivot tables
pivot_beta = heatmap_data.pivot(index='anxiety', columns='depression', values='interaction_beta')
pivot_pval = heatmap_data.pivot(index='anxiety', columns='depression', values='interaction_pval')
pivot_n = heatmap_data.pivot(index='anxiety', columns='depression', values='n')

# Reorder
pivot_beta = pivot_beta.reindex(['Low', 'High'], axis=0)
pivot_beta = pivot_beta.reindex(['Low', 'High'], axis=1)
pivot_pval = pivot_pval.reindex(['Low', 'High'], axis=0)
pivot_pval = pivot_pval.reindex(['Low', 'High'], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Effect sizes
ax = axes[0]
sns.heatmap(pivot_beta, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-5, vmax=5, cbar_kws={'label': 'Interaction β'},
            ax=ax, linewidths=2, linecolor='black', annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('Gender × UCLA Interaction (PE Rate)', fontsize=14, fontweight='bold')
ax.set_xlabel('Depression', fontsize=12)
ax.set_ylabel('Anxiety', fontsize=12)

# P-values
ax = axes[1]
annot_labels = pivot_pval.map(lambda x: f'{x:.3f}' if pd.notna(x) else '')
sns.heatmap(pivot_pval, annot=annot_labels, fmt='', cmap='RdYlGn',
            vmin=0, vmax=0.15, cbar_kws={'label': 'p-value'},
            ax=ax, linewidths=2, linecolor='black', annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('P-values (PE Rate)', fontsize=14, fontweight='bold')
ax.set_xlabel('Depression', fontsize=12)
ax.set_ylabel('Anxiety', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vulnerability_map_2x2.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: vulnerability_map_2x2.png")
print()

# ============================================================================
# SUMMARY & INTERPRETATION
# ============================================================================

print("[6/6] Summary and interpretation...")
print()
print("="*80)
print("DASS 2×2×2 STRATIFICATION COMPLETE")
print("="*80)
print()

# Count significant effects
sig_cells = strat_df[strat_df['interaction_pval'] < 0.05]
print(f"Significant cells (p<0.05): {len(sig_cells)}/{len(strat_df)}")

if len(sig_cells) > 0:
    print()
    print("SIGNIFICANT CELLS:")
    for _, row in sig_cells.iterrows():
        print(f"  {row['cell_label']} | {row['outcome']}: β={row['interaction_beta']:.3f}, p={row['interaction_pval']:.4f}")
        print(f"    Female: r={row['female_corr']:.3f}, p={row['female_p']:.4f}")
        print(f"    Male: r={row['male_corr']:.3f}, p={row['male_p']:.4f}")

print()
print("ANXIETY BUFFERING TEST:")

low_anx_pe = buffering_df[(buffering_df['profile'] == 'Low_Anxiety') & (buffering_df['outcome'] == 'pe_rate')]
high_anx_pe = buffering_df[(buffering_df['profile'] == 'High_Anxiety') & (buffering_df['outcome'] == 'pe_rate')]

if len(low_anx_pe) > 0 and len(high_anx_pe) > 0:
    low_p = low_anx_pe.iloc[0]['pval']
    high_p = high_anx_pe.iloc[0]['pval']

    print(f"  Low Anxiety: p={low_p:.4f}")
    print(f"  High Anxiety: p={high_p:.4f}")

    if low_p < 0.05 and high_p > 0.10:
        print("  *** ANXIETY BUFFERING CONFIRMED ***")
        print("  Effect present in Low Anxiety, absent in High Anxiety")
    elif low_p > 0.10 and high_p < 0.05:
        print("  *** REVERSE PATTERN ***")
        print("  Effect present in High Anxiety, absent in Low Anxiety")
    else:
        print("  No clear buffering pattern")

print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - dass_2x2x2_stratification.csv")
print("  - anxiety_buffering_profiles.csv")
print("  - vulnerability_map_2x2.png")
print()
