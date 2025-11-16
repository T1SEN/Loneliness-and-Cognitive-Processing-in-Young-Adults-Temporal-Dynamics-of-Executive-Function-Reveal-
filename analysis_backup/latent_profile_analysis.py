"""
Latent Profile Analysis (LPA)
===============================
Person-centered analysis to identify subgroups with distinct UCLA×EF profiles

Research Question:
Are there latent classes of individuals with different vulnerability patterns?
- Class 1: High UCLA + High PE (vulnerable males)
- Class 2: Moderate UCLA + Low PE (resilient)
- Class 3: Low UCLA (healthy controls)

Methods:
- Gaussian Mixture Models (GMM) on [UCLA, WCST_PE, Gender, Age]
- BIC for optimal class selection (2-5 classes)
- Characterize classes by demographics and outcomes

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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Unicode handling
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/latent_profiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("LATENT PROFILE ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

master = pd.read_csv(RESULTS_DIR / "analysis_outputs" / "master_expanded_metrics.csv")

# Add demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants['participant_id'] = participants.get('participantId', participants.get('participant_id'))

gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)
participants['age'] = pd.to_numeric(participants['age'], errors='coerce')

master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

# Rename columns
if 'pe_rate' in master.columns:
    master['wcst_pe_rate'] = master['pe_rate']

# Calculate DASS total
if 'dass_total' not in master.columns:
    master['dass_total'] = master['dass_anxiety'] + master['dass_stress'] + master['dass_depression']

# Create gender numeric
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Filter complete cases
lpa_vars = ['ucla_total', 'wcst_pe_rate', 'gender_male', 'age']
master = master.dropna(subset=lpa_vars)

print(f"  Complete cases: {len(master)} participants")

# ============================================================================
# 2. PREPARE DATA FOR LPA
# ============================================================================
print("\n[2/6] Preparing data for LPA...")

# Standardize variables (important for GMM)
scaler = StandardScaler()
lpa_data = master[lpa_vars].copy()
lpa_scaled = scaler.fit_transform(lpa_data)

print(f"\n  Variables included:")
for i, var in enumerate(lpa_vars):
    print(f"    {var}: M={master[var].mean():.2f}, SD={master[var].std():.2f}")

# ============================================================================
# 3. FIT GMM MODELS (2-5 CLASSES)
# ============================================================================
print("\n[3/6] Fitting Gaussian Mixture Models...")

model_fits = []

for n_classes in range(2, 6):
    print(f"\n  Fitting {n_classes}-class model...")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_classes, covariance_type='full',
                          random_state=42, n_init=10)
    gmm.fit(lpa_scaled)

    # Get assignments
    class_labels = gmm.predict(lpa_scaled)

    # Calculate fit indices
    bic = gmm.bic(lpa_scaled)
    aic = gmm.aic(lpa_scaled)
    log_likelihood = gmm.score(lpa_scaled) * len(lpa_scaled)

    # Entropy (classification uncertainty)
    probs = gmm.predict_proba(lpa_scaled)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1).mean()

    # Class sizes
    class_counts = pd.Series(class_labels).value_counts().sort_index()

    model_fits.append({
        'N_Classes': n_classes,
        'BIC': bic,
        'AIC': aic,
        'Log_Likelihood': log_likelihood,
        'Entropy': entropy,
        'Class_Sizes': class_counts.tolist(),
        'Min_Class_Size': class_counts.min(),
        'Model': gmm
    })

    print(f"    BIC: {bic:.2f}")
    print(f"    AIC: {aic:.2f}")
    print(f"    Entropy: {entropy:.3f}")
    print(f"    Class sizes: {class_counts.tolist()}")

# Select best model (lowest BIC)
fit_df = pd.DataFrame([{k: v for k, v in m.items() if k != 'Model'} for m in model_fits])
best_model_idx = fit_df['BIC'].idxmin()
best_n_classes = fit_df.loc[best_model_idx, 'N_Classes']
best_gmm = model_fits[best_model_idx]['Model']

print(f"\n  Best model: {best_n_classes} classes (BIC={fit_df.loc[best_model_idx, 'BIC']:.2f})")

# ============================================================================
# 4. CHARACTERIZE CLASSES
# ============================================================================
print("\n[4/6] Characterizing latent classes...")

# Get class assignments from best model
master['latent_class'] = best_gmm.predict(lpa_scaled)

# Class profiles
print(f"\n  Class profiles:")
for class_id in range(best_n_classes):
    subset = master[master['latent_class'] == class_id]

    print(f"\n  Class {class_id + 1} (N={len(subset)}):")
    print(f"    UCLA: M={subset['ucla_total'].mean():.2f}, SD={subset['ucla_total'].std():.2f}")
    print(f"    WCST PE: M={subset['wcst_pe_rate'].mean():.2f}, SD={subset['wcst_pe_rate'].std():.2f}")
    print(f"    Age: M={subset['age'].mean():.2f}, SD={subset['age'].std():.2f}")
    print(f"    Gender: {subset['gender_male'].sum()} male, {len(subset) - subset['gender_male'].sum()} female")

    # DASS scores (outcome)
    if 'dass_total' in subset.columns:
        print(f"    DASS Total: M={subset['dass_total'].mean():.2f}, SD={subset['dass_total'].std():.2f}")

# Compare classes on outcomes
print(f"\n  Class differences (ANOVA):")

# DASS
if 'dass_total' in master.columns:
    dass_by_class = [master[master['latent_class'] == i]['dass_total'].dropna()
                      for i in range(best_n_classes)]
    f_stat, p_value = stats.f_oneway(*dass_by_class)
    print(f"    DASS Total: F={f_stat:.2f}, p={p_value:.4f}")

# Age
age_by_class = [master[master['latent_class'] == i]['age'].dropna()
                 for i in range(best_n_classes)]
f_stat, p_value = stats.f_oneway(*age_by_class)
print(f"    Age: F={f_stat:.2f}, p={p_value:.4f}")

# Gender (chi-square)
contingency = pd.crosstab(master['latent_class'], master['gender_male'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"    Gender: χ²={chi2:.2f}, p={p_value:.4f}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/6] Creating visualizations...")

# Plot 1: BIC comparison
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fit_df['N_Classes'], fit_df['BIC'], 'o-', linewidth=2, markersize=10, label='BIC')
ax.plot(fit_df['N_Classes'], fit_df['AIC'], 's--', linewidth=2, markersize=8, label='AIC')

# Mark best model
ax.axvline(best_n_classes, color='red', linestyle='--', linewidth=2, label=f'Best model ({best_n_classes} classes)')

ax.set_xlabel('Number of Classes', fontsize=12)
ax.set_ylabel('Information Criterion', fontsize=12)
ax.set_title('Model Selection: BIC and AIC', fontsize=14, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_selection_bic.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Class profiles (radar plot or heatmap)
class_profiles = []
for class_id in range(best_n_classes):
    subset = master[master['latent_class'] == class_id]
    class_profiles.append({
        'Class': f'Class {class_id + 1}',
        'UCLA_Mean': subset['ucla_total'].mean(),
        'WCST_PE_Mean': subset['wcst_pe_rate'].mean(),
        'Age_Mean': subset['age'].mean(),
        'Percent_Male': subset['gender_male'].mean() * 100,
        'N': len(subset)
    })

profile_df = pd.DataFrame(class_profiles)

# Normalize for heatmap
profile_normalized = profile_df.copy()
col_mapping = {
    'UCLA_Mean': 'ucla_total',
    'WCST_PE_Mean': 'wcst_pe_rate',
    'Age_Mean': 'age',
    'Percent_Male': 'gender_male'
}
for col in ['UCLA_Mean', 'WCST_PE_Mean', 'Age_Mean', 'Percent_Male']:
    master_col = col_mapping[col]
    profile_normalized[col] = (profile_normalized[col] - master[master_col].mean()) / master[master_col].std()

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(profile_normalized[['UCLA_Mean', 'WCST_PE_Mean', 'Age_Mean', 'Percent_Male']].T,
            annot=profile_df[['UCLA_Mean', 'WCST_PE_Mean', 'Age_Mean', 'Percent_Male']].T.values,
            fmt='.1f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Z-score'},
            xticklabels=profile_df['Class'], ax=ax)

ax.set_yticklabels(['UCLA Score', 'WCST PE Rate', 'Age', '% Male'], rotation=0)
ax.set_title('Latent Class Profiles (Standardized)', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_profiles_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Scatter plot with class assignment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# UCLA × WCST PE
for class_id in range(best_n_classes):
    subset = master[master['latent_class'] == class_id]
    axes[0].scatter(subset['ucla_total'], subset['wcst_pe_rate'],
                   label=f'Class {class_id + 1} (N={len(subset)})', alpha=0.7, s=60)

axes[0].set_xlabel('UCLA Loneliness Score')
axes[0].set_ylabel('WCST Perseverative Error Rate (%)')
axes[0].set_title('Latent Classes: UCLA × WCST PE', weight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Age × Gender
for class_id in range(best_n_classes):
    subset = master[master['latent_class'] == class_id]
    axes[1].scatter(subset['age'], subset['gender_male'],
                   label=f'Class {class_id + 1}', alpha=0.7, s=60)

axes[1].set_xlabel('Age')
axes[1].set_ylabel('Gender (0=Female, 1=Male)')
axes[1].set_title('Latent Classes: Age × Gender', weight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_scatter_plots.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Outcome comparison (DASS by class)
if 'dass_total' in master.columns:
    fig, ax = plt.subplots(figsize=(8, 6))

    class_labels_str = [f'Class {i+1}' for i in range(best_n_classes)]
    dass_data = [master[master['latent_class'] == i]['dass_total'].dropna()
                  for i in range(best_n_classes)]

    bp = ax.boxplot(dass_data, labels=class_labels_str, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Latent Class')
    ax.set_ylabel('DASS Total Score')
    ax.set_title('Mental Health Outcomes by Latent Class', weight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outcome_by_class.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

# Save fit indices
fit_df.to_csv(OUTPUT_DIR / "model_fit_indices.csv", index=False, encoding='utf-8-sig')

# Save class assignments
master[['participant_id', 'latent_class', 'ucla_total', 'wcst_pe_rate', 'age', 'gender']].to_csv(
    OUTPUT_DIR / "class_assignments.csv", index=False, encoding='utf-8-sig')

# Save class profiles
profile_df.to_csv(OUTPUT_DIR / "class_profiles.csv", index=False, encoding='utf-8-sig')

# Summary report
summary_lines = []
summary_lines.append("LATENT PROFILE ANALYSIS - SUMMARY\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("RESEARCH QUESTION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Are there latent subgroups with distinct UCLA×EF vulnerability profiles?\n\n")

summary_lines.append("KEY FINDINGS\n")
summary_lines.append("-" * 80 + "\n\n")

summary_lines.append(f"OPTIMAL MODEL: {best_n_classes} classes\n")
summary_lines.append(f"  BIC = {fit_df.loc[best_model_idx, 'BIC']:.2f}\n")
summary_lines.append(f"  Entropy = {fit_df.loc[best_model_idx, 'Entropy']:.3f}\n\n")

summary_lines.append("CLASS PROFILES:\n")
summary_lines.append("-" * 80 + "\n")

for _, row in profile_df.iterrows():
    summary_lines.append(f"\n{row['Class']} (N={row['N']}):\n")
    summary_lines.append(f"  UCLA: M={row['UCLA_Mean']:.2f}\n")
    summary_lines.append(f"  WCST PE: M={row['WCST_PE_Mean']:.2f}%\n")
    summary_lines.append(f"  Age: M={row['Age_Mean']:.2f}\n")
    summary_lines.append(f"  % Male: {row['Percent_Male']:.1f}%\n")

summary_lines.append("\nCLASS DIFFERENCES (OUTCOMES):\n")
summary_lines.append("-" * 80 + "\n")

if 'dass_total' in master.columns:
    dass_by_class = [master[master['latent_class'] == i]['dass_total'].dropna()
                      for i in range(best_n_classes)]
    f_stat, p_value = stats.f_oneway(*dass_by_class)
    summary_lines.append(f"DASS Total: F={f_stat:.2f}, p={p_value:.4f}\n")

    if p_value < 0.05:
        summary_lines.append("  ✓ Significant class differences in mental health outcomes\n")

summary_lines.append("\nINTERPRETATION:\n")
summary_lines.append("-" * 80 + "\n")

# Identify vulnerable vs resilient classes
if best_n_classes >= 2:
    # Find class with highest UCLA + highest PE
    vulnerable_class = profile_df.loc[(profile_df['UCLA_Mean'] + profile_df['WCST_PE_Mean']).idxmax()]
    resilient_class = profile_df.loc[(profile_df['UCLA_Mean'] + profile_df['WCST_PE_Mean']).idxmin()]

    summary_lines.append(f"VULNERABLE GROUP: {vulnerable_class['Class']}\n")
    summary_lines.append(f"  High UCLA ({vulnerable_class['UCLA_Mean']:.1f}) + High PE ({vulnerable_class['WCST_PE_Mean']:.1f}%)\n")
    summary_lines.append(f"  {vulnerable_class['Percent_Male']:.0f}% male, Age {vulnerable_class['Age_Mean']:.1f}\n\n")

    summary_lines.append(f"RESILIENT GROUP: {resilient_class['Class']}\n")
    summary_lines.append(f"  Low UCLA ({resilient_class['UCLA_Mean']:.1f}) + Low PE ({resilient_class['WCST_PE_Mean']:.1f}%)\n")
    summary_lines.append(f"  {resilient_class['Percent_Male']:.0f}% male, Age {resilient_class['Age_Mean']:.1f}\n\n")

summary_lines.append("="*80 + "\n")
summary_lines.append(f"Full results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print("\n" + summary_text)

with open(OUTPUT_DIR / "LPA_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Latent Profile Analysis complete!")
