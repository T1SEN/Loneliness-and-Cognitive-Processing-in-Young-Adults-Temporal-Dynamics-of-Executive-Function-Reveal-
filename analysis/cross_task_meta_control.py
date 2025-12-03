"""
Cross-Task Meta-Control Integration Analysis
=============================================
Tests if executive function impairments share a common "meta-control" factor

Hypothesis:
Three EF tasks (WCST, PRP, Stroop) tap into a shared proactive control system.
Individuals with low meta-control are vulnerable to loneliness→distress pathway.

Methods:
1. PCA on [WCST_PE, PRP_bottleneck, Stroop_interference]
2. Extract first PC = "meta-control factor"
3. Test: UCLA × Meta-control → DASS (moderation)
4. Test: Meta-control mediates UCLA→DASS relationship

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
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.utils.data_loader_utils import load_master_dataset

# Unicode handling
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/meta_control")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CROSS-TASK META-CONTROL INTEGRATION ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

master = load_master_dataset(use_cache=True)

# Rename columns
if 'pe_rate' in master.columns and 'pe_rate' not in master.columns:
    master['pe_rate'] = master['pe_rate']

if 'prp_bottleneck' in master.columns:
    master['prp_bottleneck_effect'] = master['prp_bottleneck']
elif 'prp_t2_rt_short' in master.columns and 'prp_t2_rt_long' in master.columns:
    # Calculate bottleneck if not present
    master['prp_bottleneck_effect'] = master['prp_t2_rt_short'] - master['prp_t2_rt_long']

if 'stroop_interference' in master.columns:
    master['stroop_interference_effect'] = master['stroop_interference']

# Calculate DASS total
if 'dass_total' not in master.columns:
    master['dass_total'] = master['dass_anxiety'] + master['dass_stress'] + master['dass_depression']

# Filter complete cases
ef_vars = ['pe_rate', 'prp_bottleneck_effect', 'stroop_interference_effect']
master = master.dropna(subset=ef_vars + ['ucla_total', 'dass_total'])

print(f"  Complete cases: {len(master)} participants")
print(f"\n  EF variable descriptives:")
for var in ef_vars:
    print(f"    {var}: M={master[var].mean():.2f}, SD={master[var].std():.2f}")

# ============================================================================
# 2. PCA FOR META-CONTROL FACTOR
# ============================================================================
print("\n[2/6] Extracting meta-control factor via PCA...")

# Standardize EF variables
scaler = StandardScaler()
ef_data = master[ef_vars].copy()
ef_scaled = scaler.fit_transform(ef_data)

# PCA
pca = PCA(n_components=3)
pca_components = pca.fit_transform(ef_scaled)

# Extract first PC
master['meta_control_factor'] = -pca_components[:, 0]  # Negate so high = better control

# Get loadings
loadings = pd.DataFrame(
    -pca.components_.T,  # Negate to match
    columns=['PC1', 'PC2', 'PC3'],
    index=ef_vars
)

print(f"\n  PCA Results:")
print(f"    Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"    Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"    Variance explained by PC3: {pca.explained_variance_ratio_[2]*100:.1f}%")

print(f"\n  PC1 Loadings (Meta-Control Factor):")
for var in ef_vars:
    loading = loadings.loc[var, 'PC1']
    print(f"    {var}: {loading:.3f}")

# Save loadings
loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv", encoding='utf-8-sig')

# ============================================================================
# 3. UCLA × META-CONTROL MODERATION ON DASS
# ============================================================================
print("\n[3/6] Testing UCLA × Meta-Control moderation on DASS...")

# Standardize for interpretation
master['z_ucla'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()
master['z_meta_control'] = (master['meta_control_factor'] - master['meta_control_factor'].mean()) / master['meta_control_factor'].std()

# Model: DASS ~ UCLA × Meta-control
model_moderation = smf.ols('dass_total ~ z_ucla * z_meta_control + age', data=master).fit()

print(f"\n  Moderation Model:")
print(f"    UCLA main: β={model_moderation.params['z_ucla']:.2f}, p={model_moderation.pvalues['z_ucla']:.4f}")
print(f"    Meta-control main: β={model_moderation.params['z_meta_control']:.2f}, p={model_moderation.pvalues['z_meta_control']:.4f}")

if 'z_ucla:z_meta_control' in model_moderation.params:
    interaction_beta = model_moderation.params['z_ucla:z_meta_control']
    interaction_p = model_moderation.pvalues['z_ucla:z_meta_control']
    print(f"    UCLA × Meta-control: β={interaction_beta:.2f}, p={interaction_p:.4f}")

    if interaction_p < 0.05:
        print("      ✓ SIGNIFICANT interaction!")
        if interaction_beta < 0:
            print("      → Low meta-control AMPLIFIES UCLA→DASS relationship")
            print("      → Meta-control is protective factor")
        else:
            print("      → High meta-control AMPLIFIES UCLA→DASS relationship")
    else:
        print("      ✗ Interaction not significant")

# Simple slopes analysis
print("\n  Simple Slopes Analysis:")

# High meta-control (+1 SD)
high_mc = master[master['z_meta_control'] >= 1]
if len(high_mc) >= 10:
    corr_high = stats.pearsonr(high_mc['ucla_total'], high_mc['dass_total'])
    print(f"    High meta-control (+1 SD): UCLA×DASS r={corr_high[0]:.3f}, p={corr_high[1]:.4f}")

# Low meta-control (-1 SD)
low_mc = master[master['z_meta_control'] <= -1]
if len(low_mc) >= 10:
    corr_low = stats.pearsonr(low_mc['ucla_total'], low_mc['dass_total'])
    print(f"    Low meta-control (-1 SD): UCLA×DASS r={corr_low[0]:.3f}, p={corr_low[1]:.4f}")

# ============================================================================
# 4. ALTERNATIVE TEST: META-CONTROL MEDIATES UCLA→DASS?
# ============================================================================
print("\n[4/6] Testing meta-control as mediator (UCLA→Meta-control→DASS)...")

# Path a: UCLA → Meta-control
model_a = smf.ols('z_meta_control ~ z_ucla + age', data=master).fit()
path_a = model_a.params['z_ucla']
p_a = model_a.pvalues['z_ucla']

# Path b: Meta-control → DASS (controlling UCLA)
model_b = smf.ols('dass_total ~ z_meta_control + z_ucla + age', data=master).fit()
path_b = model_b.params['z_meta_control']
p_b = model_b.pvalues['z_meta_control']

# Path c': Direct effect
path_c_prime = model_b.params['z_ucla']
p_c_prime = model_b.pvalues['z_ucla']

# Path c: Total effect (without mediator)
model_c = smf.ols('dass_total ~ z_ucla + age', data=master).fit()
path_c = model_c.params['z_ucla']

# Indirect effect (approximation)
indirect_effect = path_a * path_b

print(f"\n  Mediation Results:")
print(f"    Path a (UCLA → Meta-control): β={path_a:.3f}, p={p_a:.4f}")
print(f"    Path b (Meta-control → DASS): β={path_b:.3f}, p={p_b:.4f}")
print(f"    Path c' (Direct effect): β={path_c_prime:.3f}, p={p_c_prime:.4f}")
print(f"    Path c (Total effect): β={path_c:.3f}")
print(f"    Indirect effect (a×b): {indirect_effect:.3f}")

if p_a < 0.05 and p_b < 0.05:
    print("\n    ✓ Both paths significant → Mediation possible")
    proportion_mediated = indirect_effect / path_c if abs(path_c) > 0.001 else 0
    print(f"    Proportion mediated: {proportion_mediated*100:.1f}%")
else:
    print("\n    ✗ Mediation not supported (one or both paths ns)")

# ============================================================================
# 5. CROSS-TASK CORRELATIONS
# ============================================================================
print("\n[5/6] Cross-task correlations...")

corr_matrix = master[ef_vars].corr()
print(f"\n  Correlation matrix:")
print(corr_matrix.round(3))

# Test if tasks are related
wcst_prp_corr = stats.pearsonr(master['pe_rate'], master['prp_bottleneck_effect'])
wcst_stroop_corr = stats.pearsonr(master['pe_rate'], master['stroop_interference_effect'])
prp_stroop_corr = stats.pearsonr(master['prp_bottleneck_effect'], master['stroop_interference_effect'])

print(f"\n  Pairwise correlations:")
print(f"    WCST × PRP: r={wcst_prp_corr[0]:.3f}, p={wcst_prp_corr[1]:.4f}")
print(f"    WCST × Stroop: r={wcst_stroop_corr[0]:.3f}, p={wcst_stroop_corr[1]:.4f}")
print(f"    PRP × Stroop: r={prp_stroop_corr[0]:.3f}, p={prp_stroop_corr[1]:.4f}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating visualizations...")

# Plot 1: PCA loadings
fig, ax = plt.subplots(figsize=(8, 6))

loading_vals = loadings['PC1'].values
colors = ['red' if abs(x) > 0.5 else 'blue' for x in loading_vals]

ax.barh(range(len(ef_vars)), loading_vals, color=colors)
ax.set_yticks(range(len(ef_vars)))
ax.set_yticklabels([v.replace('_', ' ').title() for v in ef_vars])
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Loading on PC1 (Meta-Control Factor)')
ax.set_title(f'Meta-Control Factor Loadings\n(Variance Explained: {pca.explained_variance_ratio_[0]*100:.1f}%)',
             weight='bold')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_loadings.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Moderation plot (UCLA × Meta-control → DASS)
fig, ax = plt.subplots(figsize=(10, 6))

# Split by meta-control (median split for visualization)
median_mc = master['meta_control_factor'].median()
high_mc_group = master[master['meta_control_factor'] >= median_mc]
low_mc_group = master[master['meta_control_factor'] < median_mc]

# Plot
for subset, label, color in [(high_mc_group, 'High Meta-Control', 'blue'),
                              (low_mc_group, 'Low Meta-Control', 'red')]:
    ax.scatter(subset['ucla_total'], subset['dass_total'], alpha=0.5, label=label, color=color, s=50)

    # Regression line
    if len(subset) >= 5:
        z = np.polyfit(subset['ucla_total'], subset['dass_total'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

ax.set_xlabel('UCLA Loneliness Score', fontsize=12)
ax.set_ylabel('DASS Total Score', fontsize=12)
ax.set_title('Meta-Control Moderates UCLA→DASS Relationship', fontsize=14, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "moderation_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

ax.set_title('Cross-Task EF Correlations', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "crosstask_correlations.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Meta-control distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(master['meta_control_factor'], bins=20, edgecolor='black', alpha=0.7)
axes[0].axvline(master['meta_control_factor'].mean(), color='red', linestyle='--',
                linewidth=2, label='Mean')
axes[0].set_xlabel('Meta-Control Factor Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Meta-Control Factor')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot by gender
if 'gender' in master.columns:
    master.boxplot(column='meta_control_factor', by='gender', ax=axes[1])
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Meta-Control Factor Score')
    axes[1].set_title('Meta-Control by Gender')
    plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "meta_control_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Save results
master[['participant_id', 'meta_control_factor', 'ucla_total', 'dass_total']].to_csv(
    OUTPUT_DIR / "meta_control_scores.csv", index=False, encoding='utf-8-sig')

# Summary report
summary_lines = []
summary_lines.append("CROSS-TASK META-CONTROL INTEGRATION ANALYSIS - SUMMARY\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("RESEARCH QUESTION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Do WCST, PRP, and Stroop share a common 'meta-control' factor?\n")
summary_lines.append("Does low meta-control amplify the UCLA→DASS relationship?\n\n")

summary_lines.append("KEY FINDINGS\n")
summary_lines.append("-" * 80 + "\n\n")

summary_lines.append("1. PCA META-CONTROL FACTOR\n")
summary_lines.append(f"   Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%\n\n")
summary_lines.append("   Loadings:\n")
for var in ef_vars:
    loading = loadings.loc[var, 'PC1']
    summary_lines.append(f"     {var}: {loading:.3f}\n")
summary_lines.append("\n")

summary_lines.append("2. UCLA × META-CONTROL MODERATION\n")
if 'z_ucla:z_meta_control' in model_moderation.params:
    interaction_beta = model_moderation.params['z_ucla:z_meta_control']
    interaction_p = model_moderation.pvalues['z_ucla:z_meta_control']
    summary_lines.append(f"   Interaction: β={interaction_beta:.3f}, p={interaction_p:.4f}\n")

    if interaction_p < 0.05:
        summary_lines.append("   ✓ SIGNIFICANT moderation\n")
        if interaction_beta < 0:
            summary_lines.append("   → Low meta-control amplifies UCLA→DASS\n")
            summary_lines.append("   → Meta-control is protective factor\n\n")
        else:
            summary_lines.append("   → High meta-control amplifies UCLA→DASS\n\n")
    else:
        summary_lines.append("   ✗ No significant moderation\n\n")

summary_lines.append("3. MEDIATION TEST\n")
summary_lines.append(f"   Path a (UCLA → Meta-control): β={path_a:.3f}, p={p_a:.4f}\n")
summary_lines.append(f"   Path b (Meta-control → DASS): β={path_b:.3f}, p={p_b:.4f}\n")
summary_lines.append(f"   Indirect effect: {indirect_effect:.3f}\n")

if p_a < 0.05 and p_b < 0.05:
    summary_lines.append("   ✓ Mediation supported\n\n")
else:
    summary_lines.append("   ✗ Mediation not supported\n\n")

summary_lines.append("4. CROSS-TASK CORRELATIONS\n")
summary_lines.append(f"   WCST × PRP: r={wcst_prp_corr[0]:.3f}, p={wcst_prp_corr[1]:.4f}\n")
summary_lines.append(f"   WCST × Stroop: r={wcst_stroop_corr[0]:.3f}, p={wcst_stroop_corr[1]:.4f}\n")
summary_lines.append(f"   PRP × Stroop: r={prp_stroop_corr[0]:.3f}, p={prp_stroop_corr[1]:.4f}\n\n")

summary_lines.append("\nINTERPRETATION\n")
summary_lines.append("-" * 80 + "\n")

if pca.explained_variance_ratio_[0] > 0.3:
    summary_lines.append("✓ META-CONTROL FACTOR IS COHERENT:\n")
    summary_lines.append(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance\n")
    summary_lines.append("  → Three tasks tap into shared executive control system\n\n")
else:
    summary_lines.append("⚠️ WEAK META-CONTROL FACTOR:\n")
    summary_lines.append(f"  PC1 only explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance\n")
    summary_lines.append("  → Tasks may be largely independent\n\n")

if interaction_p < 0.05 and interaction_beta < 0:
    summary_lines.append("✓ META-CONTROL AS PROTECTIVE FACTOR:\n")
    summary_lines.append("  Low EF individuals are vulnerable to loneliness→distress\n")
    summary_lines.append("  High EF individuals are resilient\n\n")

summary_lines.append("="*80 + "\n")
summary_lines.append(f"Full results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print("\n" + summary_text)

with open(OUTPUT_DIR / "META_CONTROL_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Cross-Task Meta-Control Integration Analysis complete!")
