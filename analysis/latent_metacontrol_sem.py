"""
latent_metacontrol_sem.py

Creates latent meta-control factor using PCA and tests path analysis:
UCLA → Meta-Control → EF outcomes

Outputs:
- sem_factor_loadings.csv
- sem_path_analysis.csv
- meta_control_factor_by_gender.png
"""

import sys
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_comprehensive/latent_sem")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LATENT META-CONTROL FACTOR & PATH ANALYSIS")
print("=" * 80)
print()

# Load data via shared master
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
master = master.rename(columns={'gender_normalized': 'gender'})
master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
master['gender_male'] = (master['gender'] == 'male').astype(int)

print(f"Sample: N={len(master)}")
print()

# ============================================================================
# PCA: Extract Meta-Control Factor
# ============================================================================

print("EXTRACTING LATENT META-CONTROL FACTOR (PCA)")
print("-" * 80)

# Variables (reverse-code so higher = better control)
ef_vars = ['perseverative_error_rate', 'stroop_interference', 'prp_bottleneck']
master_clean = master.dropna(subset=ef_vars).copy()

# Reverse code (multiply by -1 so higher = better)
master_clean['wcst_control'] = -master_clean['perseverative_error_rate']
master_clean['stroop_control'] = -master_clean['stroop_interference']
master_clean['prp_control'] = -master_clean['prp_bottleneck']

control_vars = ['wcst_control', 'stroop_control', 'prp_control']

# Standardize
scaler = StandardScaler()
control_scaled = scaler.fit_transform(master_clean[control_vars])

# PCA
pca = PCA(n_components=3)
components = pca.fit_transform(control_scaled)

print("Explained variance by component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
print()

# Loadings
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=control_vars
)
print("Factor loadings:")
print(loadings_df)
print()

loadings_df.to_csv(OUTPUT_DIR / "sem_factor_loadings.csv", encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'sem_factor_loadings.csv'}")
print()

# Use PC1 as Meta-Control Factor
master_clean['meta_control_factor'] = components[:, 0]

# ============================================================================
# PATH ANALYSIS: UCLA → Meta-Control → WCST PE
# ============================================================================

print("=" * 80)
print("PATH ANALYSIS")
print("=" * 80)
print()

# Standardize predictors
scaler2 = StandardScaler()
master_clean['z_ucla'] = scaler2.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler2.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler2.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_str'] = scaler2.fit_transform(master_clean[['dass_stress']])
master_clean['z_age'] = scaler2.fit_transform(master_clean[['age']])

# Path a: UCLA → Meta-Control
print("Path a: UCLA → Meta-Control Factor")
model_a = smf.ols("meta_control_factor ~ z_ucla + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                   data=master_clean).fit()
path_a = model_a.params['z_ucla']
path_a_p = model_a.pvalues['z_ucla']
print(f"  β = {path_a:.3f}, p = {path_a_p:.4f}")
print()

# Path b: Meta-Control → WCST PE (controlling for UCLA)
print("Path b: Meta-Control → WCST PE (independent of UCLA)")
model_b = smf.ols("perseverative_error_rate ~ meta_control_factor",
                   data=master_clean).fit()
path_b = model_b.params['meta_control_factor']
path_b_p = model_b.pvalues['meta_control_factor']
print(f"  β = {path_b:.3f}, p = {path_b_p:.4f}")
print()

# Path c': UCLA → WCST PE (direct effect, controlling for Meta-Control)
print("Path c': UCLA → WCST PE (controlling for Meta-Control)")
model_c_prime = smf.ols("perseverative_error_rate ~ z_ucla + meta_control_factor + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                         data=master_clean).fit()
path_c_prime = model_c_prime.params['z_ucla']
path_c_prime_p = model_c_prime.pvalues['z_ucla']
print(f"  β = {path_c_prime:.3f}, p = {path_c_prime_p:.4f}")
print()

# Indirect effect (a × b)
indirect_effect = path_a * path_b
print(f"Indirect effect (a × b): {indirect_effect:.4f}")
print()

# Save path coefficients
path_df = pd.DataFrame({
    'path': ['a: UCLA → Meta-Control', 'b: Meta-Control → PE', 'c\': UCLA → PE (direct)', 'Indirect (a×b)'],
    'coefficient': [path_a, path_b, path_c_prime, indirect_effect],
    'p_value': [path_a_p, path_b_p, path_c_prime_p, np.nan]
})
path_df.to_csv(OUTPUT_DIR / "sem_path_analysis.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'sem_path_analysis.csv'}")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Meta-Control Factor by Gender & UCLA
for gender, color, label in [('male', 'steelblue', 'Male'), ('female', 'coral', 'Female')]:
    subset = master_clean[master_clean['gender'] == gender]
    axes[0].scatter(subset['ucla_total'], subset['meta_control_factor'],
                    color=color, alpha=0.6, s=60, label=label)

    if len(subset) > 5:
        z = np.polyfit(subset['ucla_total'].values, subset['meta_control_factor'].values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
        axes[0].plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--')

axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_xlabel('UCLA Loneliness', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Meta-Control Factor (PC1)', fontsize=12, fontweight='bold')
axes[0].set_title('Latent Meta-Control vs UCLA', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Meta-Control Factor vs WCST PE
axes[1].scatter(master_clean['meta_control_factor'], master_clean['perseverative_error_rate'],
                color='gray', alpha=0.5, s=50)

z = np.polyfit(master_clean['meta_control_factor'].values, master_clean['perseverative_error_rate'].values, 1)
p = np.poly1d(z)
x_line = np.linspace(master_clean['meta_control_factor'].min(), master_clean['meta_control_factor'].max(), 100)
axes[1].plot(x_line, p(x_line), color='black', linewidth=2)

axes[1].set_xlabel('Meta-Control Factor', fontsize=12, fontweight='bold')
axes[1].set_ylabel('WCST PE Rate', fontsize=12, fontweight='bold')
axes[1].set_title(f'Meta-Control → PE (β={path_b:.3f}, p={path_b_p:.3f})', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "meta_control_factor_by_gender.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'meta_control_factor_by_gender.png'}")
plt.close()

print()
print("=" * 80)
print("LATENT META-CONTROL SEM COMPLETE")
print("=" * 80)
print()
