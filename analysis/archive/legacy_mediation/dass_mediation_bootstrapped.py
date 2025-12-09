"""
DASS Mediation Analysis (Bootstrapped)
========================================
CRITICAL MECHANISM TEST: Does DASS mediate UCLA→WCST PE effect?

Research Question:
- Path a: UCLA → DASS (anxiety/depression/stress)
- Path b: DASS → WCST_PE (controlling for UCLA)
- Path c': UCLA → WCST_PE (direct effect, controlling for DASS)
- Indirect effect: a × b (with 10,000 bootstrap CIs)

Moderated Mediation:
- Test if mediation differs by Gender
- Males may show different mediation pathway than Females

Author: Advanced Analysis Suite
Date: 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample

from analysis.utils.data_loader_utils import load_master_dataset, normalize_gender_series

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/dass_mediation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("DASS MEDIATION ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/6] Loading data...")

master = load_master_dataset()
if 'pe_rate' in master.columns and 'pe_rate' not in master.columns:
    master = master.rename(columns={'pe_rate': 'pe_rate'})

if 'dass_total' not in master.columns and all(col in master.columns for col in ['dass_depression', 'dass_anxiety', 'dass_stress']):
    master['dass_total'] = master[['dass_depression', 'dass_anxiety', 'dass_stress']].sum(axis=1)

master['age'] = pd.to_numeric(master['age'], errors='coerce')
master['gender'] = normalize_gender_series(master['gender'])

print(f"  Loaded master dataset: {len(master)} participants")

# Filter complete cases
master = master.dropna(subset=['ucla_total', 'dass_total', 'dass_anxiety', 'dass_stress',
                               'dass_depression', 'pe_rate', 'gender', 'age'])
print(f"  Complete cases: {len(master)} participants")

# Create gender dummy
master['gender'] = normalize_gender_series(master['gender'])
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Standardize for easier interpretation
master['z_ucla'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()
master['z_dass_total'] = (master['dass_total'] - master['dass_total'].mean()) / master['dass_total'].std()
master['z_dass_anxiety'] = (master['dass_anxiety'] - master['dass_anxiety'].mean()) / master['dass_anxiety'].std()
master['z_dass_stress'] = (master['dass_stress'] - master['dass_stress'].mean()) / master['dass_stress'].std()
master['z_dass_depression'] = (master['dass_depression'] - master['dass_depression'].mean()) / master['dass_depression'].std()

print(f"\n  Descriptives:")
print(f"    UCLA: M={master['ucla_total'].mean():.2f}, SD={master['ucla_total'].std():.2f}")
print(f"    DASS Total: M={master['dass_total'].mean():.2f}, SD={master['dass_total'].std():.2f}")
print(f"    WCST PE Rate: M={master['pe_rate'].mean():.2f}%, SD={master['pe_rate'].std():.2f}%")
print(f"    Gender: {master['gender_male'].sum()} male, {len(master) - master['gender_male'].sum()} female")

# ============================================================================
# 2. SIMPLE MEDIATION (Whole Sample)
# ============================================================================
print("\n[2/6] Testing simple mediation (whole sample)...")

def bootstrap_mediation(data, x_col, m_col, y_col, covariates=None, n_boot=10000):
    """
    Bootstrap mediation analysis.
    Returns: indirect effect, direct effect, total effect, and their CIs
    """
    results = []

    for i in range(n_boot):
        # Resample
        boot_data = resample(data, replace=True, random_state=i)

        # Path a: X → M
        if covariates:
            formula_a = f"{m_col} ~ {x_col} + {' + '.join(covariates)}"
        else:
            formula_a = f"{m_col} ~ {x_col}"
        model_a = smf.ols(formula_a, data=boot_data).fit()
        a = model_a.params[x_col]

        # Path b: M → Y (controlling X)
        if covariates:
            formula_b = f"{y_col} ~ {m_col} + {x_col} + {' + '.join(covariates)}"
        else:
            formula_b = f"{y_col} ~ {m_col} + {x_col}"
        model_b = smf.ols(formula_b, data=boot_data).fit()
        b = model_b.params[m_col]
        c_prime = model_b.params[x_col]  # Direct effect

        # Path c: X → Y (total effect, without M)
        if covariates:
            formula_c = f"{y_col} ~ {x_col} + {' + '.join(covariates)}"
        else:
            formula_c = f"{y_col} ~ {x_col}"
        model_c = smf.ols(formula_c, data=boot_data).fit()
        c = model_c.params[x_col]

        # Indirect effect
        indirect = a * b

        results.append({
            'indirect': indirect,
            'direct': c_prime,
            'total': c,
            'path_a': a,
            'path_b': b
        })

    return pd.DataFrame(results)

# Test DASS Total as mediator
print("\n  [2a] DASS Total as mediator")
boot_results = bootstrap_mediation(master, 'z_ucla', 'z_dass_total', 'pe_rate',
                                     covariates=['age'], n_boot=10000)

indirect_ci = boot_results['indirect'].quantile([0.025, 0.975])
direct_ci = boot_results['direct'].quantile([0.025, 0.975])
total_ci = boot_results['total'].quantile([0.025, 0.975])

print(f"    Indirect effect (a×b): {boot_results['indirect'].mean():.4f} [95% CI: {indirect_ci.iloc[0]:.4f}, {indirect_ci.iloc[1]:.4f}]")
print(f"    Direct effect (c'):    {boot_results['direct'].mean():.4f} [95% CI: {direct_ci.iloc[0]:.4f}, {direct_ci.iloc[1]:.4f}]")
print(f"    Total effect (c):      {boot_results['total'].mean():.4f} [95% CI: {total_ci.iloc[0]:.4f}, {total_ci.iloc[1]:.4f}]")

if indirect_ci.iloc[0] > 0 or indirect_ci.iloc[1] < 0:
    print(f"    ✓ Indirect effect is significant (CI does not include 0)")
    proportion_mediated = boot_results['indirect'].mean() / boot_results['total'].mean()
    print(f"    Proportion mediated: {proportion_mediated:.1%}")
else:
    print(f"    ✗ Indirect effect is NOT significant (CI includes 0)")

# Test DASS subscales
mediation_results = []

for subscale, col in [('Anxiety', 'z_dass_anxiety'),
                      ('Stress', 'z_dass_stress'),
                      ('Depression', 'z_dass_depression')]:
    print(f"\n  [2b] DASS {subscale} as mediator")
    boot_results = bootstrap_mediation(master, 'z_ucla', col, 'pe_rate',
                                        covariates=['age'], n_boot=10000)

    indirect_ci = boot_results['indirect'].quantile([0.025, 0.975])
    indirect_sig = 'Yes' if (indirect_ci.iloc[0] > 0 or indirect_ci.iloc[1] < 0) else 'No'

    print(f"    Indirect: {boot_results['indirect'].mean():.4f} [{indirect_ci.iloc[0]:.4f}, {indirect_ci.iloc[1]:.4f}] (Sig: {indirect_sig})")
    print(f"    Path a (UCLA→{subscale}): {boot_results['path_a'].mean():.4f}")
    print(f"    Path b ({subscale}→PE): {boot_results['path_b'].mean():.4f}")

    mediation_results.append({
        'Mediator': f'DASS_{subscale}',
        'Indirect_Effect': boot_results['indirect'].mean(),
        'Indirect_CI_Lower': indirect_ci.iloc[0],
        'Indirect_CI_Upper': indirect_ci.iloc[1],
        'Direct_Effect': boot_results['direct'].mean(),
        'Total_Effect': boot_results['total'].mean(),
        'Path_a': boot_results['path_a'].mean(),
        'Path_b': boot_results['path_b'].mean(),
        'Significant': indirect_sig
    })

mediation_df = pd.DataFrame(mediation_results)

# ============================================================================
# 3. MODERATED MEDIATION (Gender as Moderator)
# ============================================================================
print("\n[3/6] Testing moderated mediation (Gender as moderator)...")

# Separate by gender
males = master[master['gender_male'] == 1].copy()
females = master[master['gender_male'] == 0].copy()

print(f"  Males: N={len(males)}, Females: N={len(females)}")

moderated_results = []

for subscale, col in [('Total', 'z_dass_total'),
                      ('Anxiety', 'z_dass_anxiety'),
                      ('Stress', 'z_dass_stress'),
                      ('Depression', 'z_dass_depression')]:

    print(f"\n  [{subscale}] Moderated mediation")

    # Males
    boot_male = bootstrap_mediation(males, 'z_ucla', col, 'pe_rate',
                                     covariates=['age'], n_boot=10000)
    indirect_male = boot_male['indirect'].mean()
    ci_male = boot_male['indirect'].quantile([0.025, 0.975])
    sig_male = 'Yes' if (ci_male.iloc[0] > 0 or ci_male.iloc[1] < 0) else 'No'

    # Females
    boot_female = bootstrap_mediation(females, 'z_ucla', col, 'pe_rate',
                                       covariates=['age'], n_boot=10000)
    indirect_female = boot_female['indirect'].mean()
    ci_female = boot_female['indirect'].quantile([0.025, 0.975])
    sig_female = 'Yes' if (ci_female.iloc[0] > 0 or ci_female.iloc[1] < 0) else 'No'

    print(f"    Males - Indirect: {indirect_male:.4f} [{ci_male.iloc[0]:.4f}, {ci_male.iloc[1]:.4f}] (Sig: {sig_male})")
    print(f"    Females - Indirect: {indirect_female:.4f} [{ci_female.iloc[0]:.4f}, {ci_female.iloc[1]:.4f}] (Sig: {sig_female})")

    # Test difference (index of moderated mediation)
    diff = indirect_male - indirect_female
    print(f"    Difference (M - F): {diff:.4f}")

    moderated_results.append({
        'Mediator': f'DASS_{subscale}',
        'Male_Indirect': indirect_male,
        'Male_CI_Lower': ci_male.iloc[0],
        'Male_CI_Upper': ci_male.iloc[1],
        'Male_Significant': sig_male,
        'Female_Indirect': indirect_female,
        'Female_CI_Lower': ci_female.iloc[0],
        'Female_CI_Upper': ci_female.iloc[1],
        'Female_Significant': sig_female,
        'Difference_M_minus_F': diff
    })

moderated_df = pd.DataFrame(moderated_results)

# ============================================================================
# 4. FORMAL MODERATED MEDIATION MODEL
# ============================================================================
print("\n[4/6] Formal moderated mediation (interaction terms)...")

# Model: Test if gender moderates Path a and Path b

for subscale, col in [('Anxiety', 'z_dass_anxiety')]:  # Focus on anxiety (most relevant)
    print(f"\n  Testing {subscale}...")

    # Path a: UCLA × Gender → DASS
    model_a = smf.ols(f'{col} ~ z_ucla * gender_male + age', data=master).fit()
    print(f"\n    Path a (UCLA × Gender → DASS_{subscale}):")
    print(f"      UCLA main: β={model_a.params['z_ucla']:.4f}, p={model_a.pvalues['z_ucla']:.4f}")
    print(f"      Gender main: β={model_a.params['gender_male']:.4f}, p={model_a.pvalues['gender_male']:.4f}")
    if 'z_ucla:gender_male' in model_a.params:
        print(f"      UCLA × Gender: β={model_a.params['z_ucla:gender_male']:.4f}, p={model_a.pvalues['z_ucla:gender_male']:.4f}")

    # Path b: DASS × Gender → WCST_PE (controlling UCLA)
    model_b = smf.ols(f'pe_rate ~ {col} * gender_male + z_ucla + age', data=master).fit()
    print(f"\n    Path b (DASS_{subscale} × Gender → PE, controlling UCLA):")
    print(f"      DASS main: β={model_b.params[col]:.4f}, p={model_b.pvalues[col]:.4f}")
    if f'{col}:gender_male' in model_b.params:
        print(f"      DASS × Gender: β={model_b.params[f'{col}:gender_male']:.4f}, p={model_b.pvalues[f'{col}:gender_male']:.4f}")

    # Direct effect: UCLA × Gender → PE (controlling DASS)
    print(f"\n    Direct effect (UCLA × Gender → PE, controlling DASS_{subscale}):")
    print(f"      UCLA × Gender: β={model_b.params.get('z_ucla:gender_male', np.nan):.4f}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/6] Creating visualizations...")

# Plot 1: Mediation diagram with estimates
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Draw paths
from matplotlib.patches import FancyArrowPatch

# Nodes
node_positions = {
    'UCLA': (0.1, 0.5),
    'DASS': (0.5, 0.8),
    'PE': (0.9, 0.5)
}

# Draw arrows
arrows = [
    ('UCLA', 'DASS', 'a'),
    ('DASS', 'PE', 'b'),
    ('UCLA', 'PE', "c'")
]

for start, end, label in arrows:
    x1, y1 = node_positions[start]
    x2, y2 = node_positions[end]

    if label == 'a':
        # Curved arrow
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', lw=2,
                                connectionstyle="arc3,rad=0.3", color='blue')
    elif label == 'b':
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', lw=2,
                                connectionstyle="arc3,rad=-0.3", color='green')
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', lw=2,
                                color='red', linestyle='--')
    ax.add_patch(arrow)

# Add node labels
for node, (x, y) in node_positions.items():
    ax.text(x, y, node, fontsize=14, weight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', lw=2))

# Add path labels
ax.text(0.3, 0.7, 'Path a\n(UCLA→DASS)', fontsize=10, ha='center', color='blue')
ax.text(0.7, 0.7, 'Path b\n(DASS→PE)', fontsize=10, ha='center', color='green')
ax.text(0.5, 0.4, "Path c' (Direct)", fontsize=10, ha='center', color='red')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.title('Mediation Model: UCLA → DASS → WCST PE', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mediation_diagram.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Forest plot of indirect effects
fig, ax = plt.subplots(figsize=(10, 6))

mediators = mediation_df['Mediator'].tolist()
y_pos = np.arange(len(mediators))

ax.errorbar(mediation_df['Indirect_Effect'], y_pos,
            xerr=[mediation_df['Indirect_Effect'] - mediation_df['Indirect_CI_Lower'],
                  mediation_df['Indirect_CI_Upper'] - mediation_df['Indirect_Effect']],
            fmt='o', markersize=8, capsize=5)

ax.axvline(0, color='red', linestyle='--', label='No effect')
ax.set_yticks(y_pos)
ax.set_yticklabels(mediators)
ax.set_xlabel('Indirect Effect (Bootstrap 95% CI)')
ax.set_title('DASS Mediation: Indirect Effects')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "indirect_effects_forest.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Moderated mediation comparison
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(moderated_df))
width = 0.35

ax.bar(x - width/2, moderated_df['Male_Indirect'], width, label='Males',
       yerr=[moderated_df['Male_Indirect'] - moderated_df['Male_CI_Lower'],
             moderated_df['Male_CI_Upper'] - moderated_df['Male_Indirect']],
       capsize=5)

ax.bar(x + width/2, moderated_df['Female_Indirect'], width, label='Females',
       yerr=[moderated_df['Female_Indirect'] - moderated_df['Female_CI_Lower'],
             moderated_df['Female_CI_Upper'] - moderated_df['Female_Indirect']],
       capsize=5)

ax.axhline(0, color='red', linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(moderated_df['Mediator'], rotation=45, ha='right')
ax.set_ylabel('Indirect Effect')
ax.set_title('Moderated Mediation: Gender Differences in DASS Mediation')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "moderated_mediation_gender.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

mediation_df.to_csv(OUTPUT_DIR / "simple_mediation_results.csv", index=False, encoding='utf-8-sig')
moderated_df.to_csv(OUTPUT_DIR / "moderated_mediation_results.csv", index=False, encoding='utf-8-sig')

# Summary report
summary_lines = []
summary_lines.append("DASS MEDIATION ANALYSIS - SUMMARY REPORT\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("RESEARCH QUESTION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Does DASS (anxiety/depression/stress) mediate the UCLA→WCST PE relationship?\n")
summary_lines.append("Does this mediation differ by gender (moderated mediation)?\n\n")

summary_lines.append("KEY FINDINGS\n")
summary_lines.append("-" * 80 + "\n\n")

# Check for significant mediation
sig_mediators = mediation_df[mediation_df['Significant'] == 'Yes']
if not sig_mediators.empty:
    summary_lines.append(f"✓ {len(sig_mediators)} significant mediator(s) found:\n\n")
    for _, row in sig_mediators.iterrows():
        summary_lines.append(f"  • {row['Mediator']}:\n")
        summary_lines.append(f"    - Indirect effect: {row['Indirect_Effect']:.4f} [95% CI: {row['Indirect_CI_Lower']:.4f}, {row['Indirect_CI_Upper']:.4f}]\n")
        summary_lines.append(f"    - Path a (UCLA→DASS): {row['Path_a']:.4f}\n")
        summary_lines.append(f"    - Path b (DASS→PE): {row['Path_b']:.4f}\n")
        summary_lines.append(f"    - Direct effect (c'): {row['Direct_Effect']:.4f}\n\n")
else:
    summary_lines.append("✗ No significant mediation detected\n")
    summary_lines.append("  DASS does not mediate UCLA→WCST PE relationship\n\n")

summary_lines.append("\nMODERATED MEDIATION (Gender)\n")
summary_lines.append("-" * 80 + "\n")
for _, row in moderated_df.iterrows():
    summary_lines.append(f"\n{row['Mediator']}:\n")
    summary_lines.append(f"  Males:   {row['Male_Indirect']:.4f} [{row['Male_CI_Lower']:.4f}, {row['Male_CI_Upper']:.4f}] (Sig: {row['Male_Significant']})\n")
    summary_lines.append(f"  Females: {row['Female_Indirect']:.4f} [{row['Female_CI_Lower']:.4f}, {row['Female_CI_Upper']:.4f}] (Sig: {row['Female_Significant']})\n")
    summary_lines.append(f"  Difference: {row['Difference_M_minus_F']:.4f}\n")

summary_lines.append("\n" + "="*80 + "\n")
summary_lines.append(f"Results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print("\n" + summary_text)

with open(OUTPUT_DIR / "MEDIATION_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ DASS Mediation Analysis complete!")
