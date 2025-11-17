"""
Synthesis Analysis 4: Gender-Stratified Regression Comparison
===============================================================

Purpose:
--------
Direct comparison of UCLA loneliness effects on executive function
across male vs female groups, with full DASS-21 covariate control.

Statistical Approach:
--------------------
- Separate regressions for males and females:
  Formula: EF_outcome ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age
- Formal interaction test using full sample (UCLA × Gender)
- Side-by-side coefficient visualization with error bars
- Effect size comparison across genders

Output:
-------
- gender_stratified_coefficients.csv: Regression coefficients by gender
- gender_interaction_tests.csv: Formal interaction statistics
- figure_gender_stratified_regressions.png: Side-by-side coefficient plots

Author: Research Team
Date: 2025-01-17
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/synthesis_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SYNTHESIS ANALYSIS 4: GENDER-STRATIFIED REGRESSIONS")
print("=" * 80)
print()
print("Comparing UCLA effects on EF across male vs female groups")
print("All models control for DASS-21 (depression, anxiety, stress) + age")
print()

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("[1/4] Loading data...")

master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if not master_path.exists():
    print("ERROR: master_dataset.csv not found")
    sys.exit(1)

master = pd.read_csv(master_path, encoding='utf-8-sig')

# Normalize columns
if 'participantId' in master.columns:
    master = master.rename(columns={'participantId': 'participant_id'})

# Gender coding
if 'gender' not in master.columns and 'gender_male' in master.columns:
    master['gender'] = master['gender_male'].map({1: 'male', 0: 'female'})
elif 'gender' in master.columns:
    gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
    master['gender'] = master['gender'].map(gender_map)
    master['gender_male'] = (master['gender'] == 'male').astype(int)

# Rename PE if needed
for col in ['pe_rate', 'perseverative_error_rate']:
    if col in master.columns and col != 'pe_rate':
        master = master.rename(columns={col: 'pe_rate'})
        break

# Required columns
required_cols = ['participant_id', 'ucla_total', 'gender_male', 'age',
                'dass_depression', 'dass_anxiety', 'dass_stress']

master_clean = master.dropna(subset=required_cols).copy()

# Standardize predictors
scaler = StandardScaler()
master_clean['z_age'] = scaler.fit_transform(master_clean[['age']])
master_clean['z_ucla'] = scaler.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_str'] = scaler.fit_transform(master_clean[['dass_stress']])

# Split by gender
females = master_clean[master_clean['gender_male'] == 0].copy()
males = master_clean[master_clean['gender_male'] == 1].copy()

print(f"  Total N = {len(master_clean)}")
print(f"    Females: {len(females)}")
print(f"    Males: {len(males)}")
print()

# ============================================================================
# DEFINE OUTCOMES
# ============================================================================

outcomes = [
    ('pe_rate', 'WCST PE Rate (%)'),
    ('wcst_accuracy', 'WCST Accuracy (%)'),
    ('stroop_interference', 'Stroop Interference (ms)'),
    ('prp_bottleneck', 'PRP Bottleneck (ms)')
]

# ============================================================================
# GENDER-STRATIFIED REGRESSIONS
# ============================================================================

print("[2/4] Running gender-stratified regressions...")

stratified_results = []

for outcome_col, outcome_label in outcomes:
    if outcome_col not in master_clean.columns:
        print(f"  ⚠ Skipping {outcome_label} (column not found)")
        continue

    print(f"\n  {outcome_label}:")

    # Regression formula (DASS-controlled)
    formula = f"{outcome_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"

    # Female model
    df_female = females.dropna(subset=[outcome_col]).copy()
    if len(df_female) >= 15:
        try:
            model_female = smf.ols(formula, data=df_female).fit()

            stratified_results.append({
                'outcome': outcome_label,
                'outcome_col': outcome_col,
                'gender': 'Female',
                'n': len(df_female),
                'beta_ucla': model_female.params['z_ucla'],
                'se_ucla': model_female.bse['z_ucla'],
                'p_ucla': model_female.pvalues['z_ucla'],
                'ci_lower': model_female.conf_int().loc['z_ucla', 0],
                'ci_upper': model_female.conf_int().loc['z_ucla', 1],
                'beta_dass_dep': model_female.params['z_dass_dep'],
                'p_dass_dep': model_female.pvalues['z_dass_dep'],
                'beta_dass_anx': model_female.params['z_dass_anx'],
                'p_dass_anx': model_female.pvalues['z_dass_anx'],
                'beta_dass_str': model_female.params['z_dass_str'],
                'p_dass_str': model_female.pvalues['z_dass_str'],
                'beta_age': model_female.params['z_age'],
                'p_age': model_female.pvalues['z_age'],
                'r2': model_female.rsquared,
                'r2_adj': model_female.rsquared_adj
            })

            print(f"    Female (N={len(df_female)}): β_UCLA={model_female.params['z_ucla']:.3f}, " +
                 f"p={model_female.pvalues['z_ucla']:.4f}, R²={model_female.rsquared:.3f}")

        except Exception as e:
            print(f"    ⚠ Error fitting female model: {e}")
    else:
        print(f"    ⚠ Insufficient female data (N={len(df_female)})")

    # Male model
    df_male = males.dropna(subset=[outcome_col]).copy()
    if len(df_male) >= 15:
        try:
            model_male = smf.ols(formula, data=df_male).fit()

            stratified_results.append({
                'outcome': outcome_label,
                'outcome_col': outcome_col,
                'gender': 'Male',
                'n': len(df_male),
                'beta_ucla': model_male.params['z_ucla'],
                'se_ucla': model_male.bse['z_ucla'],
                'p_ucla': model_male.pvalues['z_ucla'],
                'ci_lower': model_male.conf_int().loc['z_ucla', 0],
                'ci_upper': model_male.conf_int().loc['z_ucla', 1],
                'beta_dass_dep': model_male.params['z_dass_dep'],
                'p_dass_dep': model_male.pvalues['z_dass_dep'],
                'beta_dass_anx': model_male.params['z_dass_anx'],
                'p_dass_anx': model_male.pvalues['z_dass_anx'],
                'beta_dass_str': model_male.params['z_dass_str'],
                'p_dass_str': model_male.pvalues['z_dass_str'],
                'beta_age': model_male.params['z_age'],
                'p_age': model_male.pvalues['z_age'],
                'r2': model_male.rsquared,
                'r2_adj': model_male.rsquared_adj
            })

            print(f"    Male (N={len(df_male)}): β_UCLA={model_male.params['z_ucla']:.3f}, " +
                 f"p={model_male.pvalues['z_ucla']:.4f}, R²={model_male.rsquared:.3f}")

        except Exception as e:
            print(f"    ⚠ Error fitting male model: {e}")
    else:
        print(f"    ⚠ Insufficient male data (N={len(df_male)})")

# Save stratified results
stratified_df = pd.DataFrame(stratified_results)
stratified_df.to_csv(OUTPUT_DIR / "gender_stratified_coefficients.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: gender_stratified_coefficients.csv")

# ============================================================================
# FORMAL INTERACTION TESTS
# ============================================================================

print("\n[3/4] Testing formal UCLA × Gender interactions...")

interaction_results = []

for outcome_col, outcome_label in outcomes:
    if outcome_col not in master_clean.columns:
        continue

    df = master_clean.dropna(subset=[outcome_col]).copy()

    if len(df) < 30:
        continue

    # Full model with interaction
    formula_int = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

    try:
        model_int = smf.ols(formula_int, data=df).fit()

        int_term = "z_ucla:C(gender_male)[T.1]"
        if int_term in model_int.params:
            beta_int = model_int.params[int_term]
            se_int = model_int.bse[int_term]
            p_int = model_int.pvalues[int_term]
            ci_lower = model_int.conf_int().loc[int_term, 0]
            ci_upper = model_int.conf_int().loc[int_term, 1]

            interaction_results.append({
                'outcome': outcome_label,
                'n': len(df),
                'interaction_beta': beta_int,
                'interaction_se': se_int,
                'interaction_p': p_int,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_int < 0.05,
                'r2_full_model': model_int.rsquared
            })

            sig_marker = '***' if p_int < 0.001 else '**' if p_int < 0.01 else '*' if p_int < 0.05 else 'ns'
            print(f"  {outcome_label}: β_interaction={beta_int:.3f}, p={p_int:.4f} {sig_marker}")

    except Exception as e:
        print(f"  ⚠ Error testing interaction for {outcome_label}: {e}")

# Save interaction results
if len(interaction_results) > 0:
    interaction_df = pd.DataFrame(interaction_results)
    interaction_df.to_csv(OUTPUT_DIR / "gender_interaction_tests.csv", index=False, encoding='utf-8-sig')
    print(f"\n  ✓ Saved: gender_interaction_tests.csv")

# ============================================================================
# VISUALIZATION: SIDE-BY-SIDE COEFFICIENT PLOTS
# ============================================================================

print("\n[4/4] Creating visualization...")

# Filter to UCLA coefficients only
ucla_coefs = stratified_df[['outcome', 'gender', 'n', 'beta_ucla', 'se_ucla', 'ci_lower', 'ci_upper', 'p_ucla']].copy()

# Pivot to wide format for plotting
unique_outcomes = ucla_coefs['outcome'].unique()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, outcome in enumerate(unique_outcomes):
    if idx >= 4:
        break

    ax = axes[idx]
    outcome_data = ucla_coefs[ucla_coefs['outcome'] == outcome]

    if len(outcome_data) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        ax.set_title(outcome, fontweight='bold')
        continue

    # X positions
    x_pos = np.arange(len(outcome_data))
    colors = ['#DE8F05' if g == 'Female' else '#0173B2' for g in outcome_data['gender']]

    # Bar plot
    bars = ax.bar(x_pos, outcome_data['beta_ucla'], width=0.6,
                 color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Error bars (95% CI)
    errors = np.array([
        outcome_data['beta_ucla'].values - outcome_data['ci_lower'].values,
        outcome_data['ci_upper'].values - outcome_data['beta_ucla'].values
    ])
    ax.errorbar(x_pos, outcome_data['beta_ucla'], yerr=errors,
               fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=2)

    # Significance markers
    for i, (_, row) in enumerate(outcome_data.iterrows()):
        sig_marker = '***' if row['p_ucla'] < 0.001 else \
                    '**' if row['p_ucla'] < 0.01 else \
                    '*' if row['p_ucla'] < 0.05 else 'ns'

        y_pos = row['ci_upper'] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(i, y_pos, sig_marker, ha='center', fontsize=14, fontweight='bold')

    # Null reference line
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['gender']}\n(N={row['n']:.0f})"
                       for _, row in outcome_data.iterrows()], fontsize=11)
    ax.set_ylabel('UCLA β (DASS-adjusted)', fontweight='bold', fontsize=11)
    ax.set_title(outcome, fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Remove empty subplots
for idx in range(len(unique_outcomes), 4):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_gender_stratified_regressions.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure_gender_stratified_regressions.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SYNTHESIS ANALYSIS 4 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - gender_stratified_coefficients.csv")
print("  - gender_interaction_tests.csv")
print("  - figure_gender_stratified_regressions.png")
print()

# Key findings summary
print("KEY FINDINGS:")
print()

if len(stratified_df) > 0:
    sig_effects = stratified_df[stratified_df['p_ucla'] < 0.05]
    print(f"Significant UCLA effects (p < 0.05): {len(sig_effects)}/{len(stratified_df)}")

    if len(sig_effects) > 0:
        print("\nSignificant effects by gender:")
        for gender in ['Female', 'Male']:
            gender_sig = sig_effects[sig_effects['gender'] == gender]
            if len(gender_sig) > 0:
                print(f"\n  {gender}:")
                for _, row in gender_sig.iterrows():
                    print(f"    {row['outcome']}: β={row['beta_ucla']:.3f}, p={row['p_ucla']:.4f}")

if len(interaction_results) > 0:
    sig_interactions = [r for r in interaction_results if r['significant']]
    print(f"\n\nSignificant UCLA × Gender interactions: {len(sig_interactions)}/{len(interaction_results)}")

    if len(sig_interactions) > 0:
        for result in sig_interactions:
            print(f"  {result['outcome']}: β={result['interaction_beta']:.3f}, p={result['interaction_p']:.4f}")

print()
