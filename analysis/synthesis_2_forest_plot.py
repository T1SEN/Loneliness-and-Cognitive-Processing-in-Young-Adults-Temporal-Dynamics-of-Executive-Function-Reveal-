"""
Synthesis Analysis 2: Forest Plot of Key Effects
==================================================

Purpose:
--------
Visual summary of all major effect sizes with 95% confidence intervals.
Extracts standardized β coefficients from DASS-controlled hierarchical regressions
and gender-stratified correlations.

Effects Included:
-----------------
1. UCLA × Gender interactions for each EF outcome (DASS-controlled)
2. Gender-stratified UCLA effects (males vs females, DASS-controlled)
3. Gender-stratified Pearson correlations

Statistical Approach:
--------------------
- Hierarchical regression: Model 3 (UCLA × Gender + DASS + age)
- Standardized predictors (z-scores) for effect size interpretability
- 95% CI from model standard errors (t-distribution)
- Fisher's z-transformation for correlation CIs

Output:
-------
- forest_plot_effect_sizes.csv: All effect sizes with CIs
- figure_forest_plot.png: Visual summary

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
print("SYNTHESIS ANALYSIS 2: FOREST PLOT OF KEY EFFECTS")
print("=" * 80)
print()
print("Extracting DASS-controlled effect sizes from hierarchical regressions")
print("and gender-stratified correlations")
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

print(f"  Total N = {len(master_clean)}")
print(f"    Males: {(master_clean['gender_male'] == 1).sum()}")
print(f"    Females: {(master_clean['gender_male'] == 0).sum()}")
print()

# ============================================================================
# DEFINE OUTCOMES TO TEST
# ============================================================================

outcomes = [
    ('pe_rate', 'WCST PE Rate'),
    ('wcst_accuracy', 'WCST Accuracy'),
    ('stroop_interference', 'Stroop Interference'),
    ('prp_bottleneck', 'PRP Bottleneck')
]

# ============================================================================
# EXTRACT EFFECT SIZES FROM HIERARCHICAL REGRESSIONS
# ============================================================================

print("[2/4] Extracting effect sizes from hierarchical regressions...")

all_effects = []

for outcome_col, outcome_label in outcomes:
    if outcome_col not in master_clean.columns:
        print(f"  ⚠ Skipping {outcome_label} (column not found)")
        continue

    # Drop missing for this outcome
    df = master_clean.dropna(subset=[outcome_col]).copy()

    if len(df) < 30:
        print(f"  ⚠ Skipping {outcome_label}: insufficient data (N={len(df)})")
        continue

    print(f"\n  {outcome_label} (N={len(df)}):")

    # ========================================================================
    # FULL MODEL: UCLA × Gender interaction (DASS-controlled)
    # ========================================================================
    formula_full = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

    try:
        model_full = smf.ols(formula_full, data=df).fit()

        # Extract interaction term
        int_term = "z_ucla:C(gender_male)[T.1]"
        if int_term in model_full.params:
            beta_int = model_full.params[int_term]
            se_int = model_full.bse[int_term]
            p_int = model_full.pvalues[int_term]

            # 95% CI using t-distribution
            df_resid = model_full.df_resid
            t_crit = stats.t.ppf(0.975, df_resid)
            ci_lower = beta_int - t_crit * se_int
            ci_upper = beta_int + t_crit * se_int

            all_effects.append({
                'outcome': outcome_label,
                'effect_type': 'Interaction',
                'effect_label': f'{outcome_label}: UCLA × Gender',
                'analysis': 'Hierarchical Regression (DASS-controlled)',
                'beta': beta_int,
                'se': se_int,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_int,
                'n': len(df),
                'r2': model_full.rsquared
            })

            print(f"    Interaction: β={beta_int:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], p={p_int:.4f}")

    except Exception as e:
        print(f"    ⚠ Error fitting interaction model: {e}")

    # ========================================================================
    # GENDER-STRATIFIED MODELS
    # ========================================================================
    females = df[df['gender_male'] == 0]
    males = df[df['gender_male'] == 1]

    for gender_val, gender_label, subset in [(0, 'Female', females), (1, 'Male', males)]:
        if len(subset) < 15:
            print(f"    ⚠ Skipping {gender_label}: N={len(subset)} too small")
            continue

        # Regression: UCLA effect controlling for DASS
        formula_gender = f"{outcome_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"

        try:
            model_gender = smf.ols(formula_gender, data=subset).fit()

            beta_ucla = model_gender.params['z_ucla']
            se_ucla = model_gender.bse['z_ucla']
            p_ucla = model_gender.pvalues['z_ucla']

            # 95% CI
            df_resid = model_gender.df_resid
            t_crit = stats.t.ppf(0.975, df_resid)
            ci_lower = beta_ucla - t_crit * se_ucla
            ci_upper = beta_ucla + t_crit * se_ucla

            all_effects.append({
                'outcome': outcome_label,
                'effect_type': f'{gender_label} Simple Slope',
                'effect_label': f'{outcome_label}: {gender_label} UCLA Effect',
                'analysis': 'Stratified Regression (DASS-controlled)',
                'beta': beta_ucla,
                'se': se_ucla,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_ucla,
                'n': len(subset),
                'r2': model_gender.rsquared
            })

            print(f"    {gender_label}: β={beta_ucla:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], p={p_ucla:.4f}")

        except Exception as e:
            print(f"    ⚠ Error fitting {gender_label} model: {e}")

        # Pearson correlation (for comparison)
        if len(subset) >= 15:
            valid = subset.dropna(subset=['ucla_total', outcome_col])
            if len(valid) >= 15:
                r, p_r = stats.pearsonr(valid['ucla_total'], valid[outcome_col])

                # Fisher's z transformation for CI
                z = np.arctanh(r)
                se_z = 1 / np.sqrt(len(valid) - 3)
                z_crit = stats.norm.ppf(0.975)
                z_lower = z - z_crit * se_z
                z_upper = z + z_crit * se_z
                r_lower = np.tanh(z_lower)
                r_upper = np.tanh(z_upper)

                all_effects.append({
                    'outcome': outcome_label,
                    'effect_type': f'{gender_label} Correlation',
                    'effect_label': f'{outcome_label}: {gender_label} r(UCLA)',
                    'analysis': 'Pearson Correlation (Unadjusted)',
                    'beta': r,
                    'se': np.nan,
                    'ci_lower': r_lower,
                    'ci_upper': r_upper,
                    'p_value': p_r,
                    'n': len(valid),
                    'r2': r**2
                })

# ============================================================================
# SAVE EFFECT SIZES
# ============================================================================

effects_df = pd.DataFrame(all_effects)
effects_df = effects_df.sort_values(['outcome', 'effect_type'])
effects_df.to_csv(OUTPUT_DIR / "forest_plot_effect_sizes.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: forest_plot_effect_sizes.csv")
print(f"    Total effects extracted: {len(effects_df)}")

# ============================================================================
# CREATE FOREST PLOT
# ============================================================================

print("\n[3/4] Creating forest plot...")

# Filter to only include DASS-controlled regression effects (exclude unadjusted correlations for clarity)
plot_effects = effects_df[effects_df['analysis'].str.contains('Regression')].copy()

if len(plot_effects) == 0:
    print("  ⚠ No effects to plot")
    sys.exit(0)

# Create figure
fig, ax = plt.subplots(figsize=(12, max(8, len(plot_effects) * 0.4)))

# Sort by outcome and effect type
plot_effects = plot_effects.sort_values(['outcome', 'effect_type'], ascending=[True, False])

y_positions = np.arange(len(plot_effects))

# Color by significance
colors = ['#2ECC71' if p < 0.05 else '#95A5A6' for p in plot_effects['p_value']]

# Plot CIs and point estimates
for i, (idx, row) in enumerate(plot_effects.iterrows()):
    # CI line
    ax.plot([row['ci_lower'], row['ci_upper']],
           [y_positions[i], y_positions[i]],
           'k-', linewidth=2, zorder=1)

    # Point estimate
    ax.scatter(row['beta'], y_positions[i],
              s=150, color=colors[i], edgecolor='black', linewidth=2,
              zorder=3, marker='o')

    # Significance marker
    sig_marker = '***' if row['p_value'] < 0.001 else \
                '**' if row['p_value'] < 0.01 else \
                '*' if row['p_value'] < 0.05 else 'ns'

    # Add beta value and p-value as text
    ax.text(row['ci_upper'] + 0.05, y_positions[i],
           f"β={row['beta']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}] {sig_marker}",
           va='center', fontsize=9)

# Null reference line
ax.axvline(0, color='black', linestyle='--', linewidth=1.5, zorder=2)

# Y-axis labels
y_labels = [f"{row['effect_label']}\n(N={row['n']:.0f})"
           for _, row in plot_effects.iterrows()]
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=10)

# X-axis
ax.set_xlabel('Standardized Effect Size (β) with 95% CI', fontweight='bold', fontsize=12)
ax.set_title('Forest Plot: Key UCLA Effects on Executive Function\n(DASS-21 Controlled)',
            fontweight='bold', fontsize=14)

# Grid
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ECC71', edgecolor='black', label='p < 0.05 (Significant)'),
    Patch(facecolor='#95A5A6', edgecolor='black', label='p ≥ 0.05 (Not Significant)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_forest_plot.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure_forest_plot.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n[4/4] Summary statistics:")

sig_effects = effects_df[effects_df['p_value'] < 0.05]
print(f"\n  Total effects analyzed: {len(effects_df)}")
print(f"  Significant effects (p < 0.05): {len(sig_effects)}")

if len(sig_effects) > 0:
    print(f"\n  Significant effects by type:")
    for etype in sig_effects['effect_type'].unique():
        count = len(sig_effects[sig_effects['effect_type'] == etype])
        print(f"    {etype}: {count}")

    print(f"\n  Top 5 strongest effects (by |beta|):")
    top5 = effects_df.nlargest(5, 'beta', keep='all')[['effect_label', 'beta', 'p_value']]
    for _, row in top5.iterrows():
        print(f"    {row['effect_label']}: β={row['beta']:.3f}, p={row['p_value']:.4f}")

# ============================================================================
# DONE
# ============================================================================

print()
print("=" * 80)
print("SYNTHESIS ANALYSIS 2 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - forest_plot_effect_sizes.csv")
print("  - figure_forest_plot.png")
print()
