"""
Paper 3: Latent Profile Analysis (LPA) / Gaussian Mixture Models
=================================================================

Title: "Cognitive Control Profiles in Loneliness: Person-Centered Analysis
        Reveals Distinct Vulnerability Patterns"

Purpose:
--------
Identifies LATENT PROFILES of cognitive control based on EF variability metrics.
Instead of asking "Does UCLA predict tau?" (Paper 1, variable-centered),
asks "What TYPES of cognitive profiles exist, and how do they relate to UCLA?" (person-centered).

Key Questions:
--------------
1. How many distinct EF variability profiles exist? (k=2-6)
2. What characterizes each profile? (lapse-prone, hypervigilant, resilient, etc.)
3. Do profiles differ in UCLA, DASS, gender, age?
4. Is Paper 1's "male-specific instability" actually a PROFILE membership effect?

Method:
-------
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering with soft assignments
- **Features**: EF variability metrics (tau, sigma, RMSSD from Paper 1)
- **Model selection**: BIC, AIC, entropy (classification certainty)
- **Validation**: Demographics, UCLA/DASS by profile, split-half stability

Features (7-8 EF variability metrics):
  - PRP tau (long SOA): Attentional lapses in dual-task
  - PRP RMSSD: Sequential variability
  - WCST tau: Perseveration lapses
  - WCST RMSSD: Sequential variability in switching
  - Stroop tau: Interference lapses
  - (Optional: sigma, kurtosis)

Output:
-------
- paper3_gmm_fit_statistics.csv: BIC/AIC for k=2-6
- paper3_profile_assignments.csv: Each participant's profile membership + probabilities
- paper3_profile_characteristics.csv: Mean EF metrics per profile
- paper3_profile_demographics.csv: UCLA/DASS/gender by profile
- paper3_profile_heatmap.png: Visual summary of profiles
- paper3_profile_scatterplots.png: UCLA by profile

Author: Research Team
Date: 2025-01-17
Target Journal: Journal of Abnormal Psychology, Assessment, Personality and Individual Differences
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/paper3_profiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print(" " * 20 + "PAPER 3: LATENT PROFILE ANALYSIS (GMM)")
print("=" * 90)
print()
print("Identifying cognitive control profiles via Gaussian Mixture Models")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/7] Loading participant data...")

# Load metrics from Paper 1
df = pd.read_csv(RESULTS_DIR / "analysis_outputs/paper1_distributional/paper1_participant_variability_metrics.csv")

print(f"  Total N: {len(df)}")
print(f"  Males: {(df['gender'] == 'male').sum()}, Females: {(df['gender'] == 'female').sum()}")

# ============================================================================
# SELECT FEATURES FOR PROFILE ANALYSIS
# ============================================================================

print("\n[2/7] Selecting features for profile analysis...")

# FEATURE SELECTION STRATEGY:
# - Focus on VARIABILITY metrics (Paper 1's key findings)
# - Include tau (lapses), RMSSD (sequential), optionally sigma
# - Across all 3 tasks (PRP, WCST, Stroop)
# - Standardize before GMM

features = {
    'PRP_tau': 'prp_tau_long',
    'PRP_RMSSD': 'prp_rmssd',
    'PRP_sigma': 'prp_sigma_long',
    'WCST_tau': 'wcst_tau',
    'WCST_RMSSD': 'wcst_rmssd',
    'Stroop_tau': 'stroop_tau_incong',
    'Stroop_RMSSD': 'stroop_rmssd'
}

print(f"  Total features: {len(features)}")
print(f"  Feature categories:")
print(f"    - Tau (lapses): 3 (PRP, WCST, Stroop)")
print(f"    - RMSSD (sequential): 3 (PRP, WCST, Stroop)")
print(f"    - Sigma: 1 (PRP)")

# Extract feature data
feature_cols = list(features.values())
df_features = df[feature_cols + ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'gender', 'participant_id']].copy()

# Rename to feature labels
for label, col in features.items():
    df_features = df_features.rename(columns={col: label})

# Drop rows with missing values in ANY feature
df_complete = df_features.dropna(subset=list(features.keys()))
print(f"\n  Complete cases: {len(df_complete)}")
print(f"    Males: {(df_complete['gender'] == 'male').sum()}")
print(f"    Females: {(df_complete['gender'] == 'female').sum()}")

if len(df_complete) < 40:
    print(f"  WARNING: N={len(df_complete)} may be too small for reliable GMM with {len(features)} features")

# Standardize features (z-scores)
scaler = StandardScaler()
X = df_complete[list(features.keys())].values
X_scaled = scaler.fit_transform(X)

# ============================================================================
# FIT GAUSSIAN MIXTURE MODELS (k=2 to k=6)
# ============================================================================

print("\n[3/7] Fitting Gaussian Mixture Models (k=2 to k=6)...")

gmm_results = []
best_bic = np.inf
best_k = None
best_model = None

for k in range(2, 7):
    print(f"  Fitting GMM with k={k} profiles...")

    # Fit GMM with full covariance (more flexible)
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          max_iter=200, n_init=20, random_state=42)
    gmm.fit(X_scaled)

    # Compute fit statistics
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    log_likelihood = gmm.score(X_scaled) * len(X_scaled)

    # Predict cluster membership
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    # Compute entropy (classification certainty)
    # Entropy = -sum(p * log(p)) across profiles
    # Lower entropy = more certain classification
    entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    mean_entropy = np.mean(entropies)
    max_entropy = np.log(k)  # Maximum possible entropy for k classes
    relative_entropy = mean_entropy / max_entropy  # 0-1 scale

    gmm_results.append({
        'k': k,
        'bic': bic,
        'aic': aic,
        'log_likelihood': log_likelihood,
        'mean_entropy': mean_entropy,
        'relative_entropy': relative_entropy,
        'converged': gmm.converged_
    })

    print(f"    BIC={bic:.2f}, AIC={aic:.2f}, Entropy={relative_entropy:.3f}")

    # Track best model (lowest BIC)
    if bic < best_bic:
        best_bic = bic
        best_k = k
        best_model = gmm

gmm_df = pd.DataFrame(gmm_results)
output_fit = OUTPUT_DIR / "paper3_gmm_fit_statistics.csv"
gmm_df.to_csv(output_fit, index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: {output_fit}")
print(f"\n  Best model: k={best_k} profiles (lowest BIC={best_bic:.2f})")

# ============================================================================
# ASSIGN PROFILES USING BEST MODEL
# ============================================================================

print(f"\n[4/7] Assigning profiles using k={best_k} model...")

# Predict profiles
df_complete['profile'] = best_model.predict(X_scaled)
profile_probs = best_model.predict_proba(X_scaled)

# Add probability columns
for i in range(best_k):
    df_complete[f'prob_profile_{i+1}'] = profile_probs[:, i]

# Compute entropy for each participant
df_complete['entropy'] = -np.sum(profile_probs * np.log(profile_probs + 1e-10), axis=1)

# Profile sizes
print(f"\n  Profile sizes:")
for i in range(best_k):
    n_profile = (df_complete['profile'] == i).sum()
    print(f"    Profile {i+1}: N={n_profile} ({n_profile/len(df_complete)*100:.1f}%)")

# ============================================================================
# CHARACTERIZE PROFILES
# ============================================================================

print(f"\n[5/7] Characterizing profiles...")

# Compute mean (unscaled) features per profile
profile_chars = []
for i in range(best_k):
    profile_data = df_complete[df_complete['profile'] == i]

    char = {'profile': i+1, 'N': len(profile_data)}

    # Mean of each feature
    for feat in features.keys():
        char[f'{feat}_mean'] = profile_data[feat].mean()
        char[f'{feat}_sd'] = profile_data[feat].std()

    profile_chars.append(char)

profile_chars_df = pd.DataFrame(profile_chars)
output_chars = OUTPUT_DIR / "paper3_profile_characteristics.csv"
profile_chars_df.to_csv(output_chars, index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: {output_chars}")

# Print profile summaries
print(f"\n  Profile summaries (mean values):")
for _, row in profile_chars_df.iterrows():
    print(f"\n    Profile {int(row['profile'])} (N={int(row['N'])}):")
    print(f"      PRP tau={row['PRP_tau_mean']:.1f}, RMSSD={row['PRP_RMSSD_mean']:.1f}")
    print(f"      WCST tau={row['WCST_tau_mean']:.1f}, RMSSD={row['WCST_RMSSD_mean']:.1f}")
    print(f"      Stroop tau={row['Stroop_tau_mean']:.1f}, RMSSD={row['Stroop_RMSSD_mean']:.1f}")

# ============================================================================
# VALIDATE PROFILES WITH DEMOGRAPHICS
# ============================================================================

print(f"\n[6/7] Validating profiles with demographics...")

# Demographics by profile
demographics = []
for i in range(best_k):
    profile_data = df_complete[df_complete['profile'] == i]

    demo = {
        'profile': i+1,
        'N': len(profile_data),
        'n_male': (profile_data['gender'] == 'male').sum(),
        'n_female': (profile_data['gender'] == 'female').sum(),
        'pct_male': (profile_data['gender'] == 'male').sum() / len(profile_data) * 100,
        'age_mean': profile_data['age'].mean(),
        'age_sd': profile_data['age'].std(),
        'ucla_mean': profile_data['ucla_total'].mean(),
        'ucla_sd': profile_data['ucla_total'].std(),
        'dass_dep_mean': profile_data['dass_depression'].mean(),
        'dass_anx_mean': profile_data['dass_anxiety'].mean(),
        'dass_str_mean': profile_data['dass_stress'].mean()
    }
    demographics.append(demo)

demographics_df = pd.DataFrame(demographics)
output_demo = OUTPUT_DIR / "paper3_profile_demographics.csv"
demographics_df.to_csv(output_demo, index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: {output_demo}")

print(f"\n  Demographics by profile:")
for _, row in demographics_df.iterrows():
    print(f"\n    Profile {int(row['profile'])} (N={int(row['N'])}):")
    print(f"      Gender: {row['pct_male']:.1f}% male")
    print(f"      Age: M={row['age_mean']:.1f}, SD={row['age_sd']:.1f}")
    print(f"      UCLA: M={row['ucla_mean']:.1f}, SD={row['ucla_sd']:.1f}")
    print(f"      DASS_Dep: M={row['dass_dep_mean']:.1f}")

# Statistical tests
print(f"\n  Statistical tests:")

# Chi-square for gender
contingency = pd.crosstab(df_complete['profile'], df_complete['gender'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"    Gender × Profile: χ²={chi2:.2f}, p={p_chi:.4f}")

# ANOVA for UCLA
profile_groups = [df_complete[df_complete['profile'] == i]['ucla_total'].values
                  for i in range(best_k)]
f_ucla, p_ucla = stats.f_oneway(*profile_groups)
print(f"    UCLA across profiles: F={f_ucla:.2f}, p={p_ucla:.4f}")

# ANOVA for DASS Depression
profile_groups_dep = [df_complete[df_complete['profile'] == i]['dass_depression'].values
                      for i in range(best_k)]
f_dep, p_dep = stats.f_oneway(*profile_groups_dep)
print(f"    DASS_Dep across profiles: F={f_dep:.2f}, p={p_dep:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print(f"\n[7/7] Creating visualizations...")

# Heatmap of profile characteristics (z-scored for comparison)
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for heatmap (z-scores of means)
heatmap_data = []
profile_labels = []
for i in range(best_k):
    profile_data = df_complete[df_complete['profile'] == i]
    z_scores = [(profile_data[feat].mean() - df_complete[feat].mean()) / df_complete[feat].std()
                for feat in features.keys()]
    heatmap_data.append(z_scores)
    profile_labels.append(f"Profile {i+1}\n(N={len(profile_data)})")

heatmap_df = pd.DataFrame(heatmap_data, columns=list(features.keys()), index=profile_labels)

sns.heatmap(heatmap_df.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Z-score'}, linewidths=0.5, ax=ax)
ax.set_title('Profile Characteristics: EF Variability Metrics (Z-scored)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Profile', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

plt.tight_layout()
output_heatmap = OUTPUT_DIR / "paper3_profile_heatmap.png"
plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_heatmap}")

# Scatter plots: UCLA by profile
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: UCLA distribution by profile
ax = axes[0]
profile_colors = sns.color_palette("Set2", best_k)
for i in range(best_k):
    profile_data = df_complete[df_complete['profile'] == i]
    ax.scatter(profile_data['ucla_total'], [i]*len(profile_data),
               alpha=0.6, s=80, color=profile_colors[i], label=f'Profile {i+1}')

ax.set_xlabel('UCLA Loneliness Score', fontweight='bold', fontsize=11)
ax.set_ylabel('Profile', fontweight='bold', fontsize=11)
ax.set_yticks(range(best_k))
ax.set_yticklabels([f'{i+1}' for i in range(best_k)])
ax.set_title('UCLA Distribution by Profile', fontweight='bold', fontsize=12, pad=10)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# Plot 2: Profile means with error bars
ax = axes[1]
x_pos = np.arange(best_k)
means = [df_complete[df_complete['profile'] == i]['ucla_total'].mean() for i in range(best_k)]
sems = [df_complete[df_complete['profile'] == i]['ucla_total'].sem() for i in range(best_k)]

ax.bar(x_pos, means, yerr=[1.96*sem for sem in sems], color=profile_colors,
       edgecolor='black', linewidth=1.5, capsize=5, alpha=0.8)
ax.set_xlabel('Profile', fontweight='bold', fontsize=11)
ax.set_ylabel('UCLA Mean ± 95% CI', fontweight='bold', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{i+1}' for i in range(best_k)])
ax.set_title('UCLA Loneliness by Profile (Mean ± 95% CI)', fontweight='bold', fontsize=12, pad=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_scatter = OUTPUT_DIR / "paper3_profile_ucla_plots.png"
plt.savefig(output_scatter, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_scatter}")

# Save profile assignments
assignments = df_complete[['participant_id', 'profile', 'entropy'] +
                         [f'prob_profile_{i+1}' for i in range(best_k)] +
                         ['gender', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']]
output_assignments = OUTPUT_DIR / "paper3_profile_assignments.csv"
assignments.to_csv(output_assignments, index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: {output_assignments}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 90)
print("PAPER 3 LATENT PROFILE ANALYSIS COMPLETE!")
print("=" * 90)

print(f"\nModel Selection:")
print(f"  Best k: {best_k} profiles (BIC={best_bic:.2f})")
print(f"  Mean relative entropy: {gmm_df[gmm_df['k']==best_k]['relative_entropy'].values[0]:.3f}")
print(f"    (0=perfect certainty, 1=maximum uncertainty)")

print(f"\nProfile Sizes:")
for i in range(best_k):
    n = (df_complete['profile'] == i).sum()
    print(f"  Profile {i+1}: N={n} ({n/len(df_complete)*100:.1f}%)")

print(f"\nKey Findings:")
print(f"  1. Gender × Profile: χ²={chi2:.2f}, p={p_chi:.4f}")
print(f"  2. UCLA × Profile: F={f_ucla:.2f}, p={p_ucla:.4f}")
print(f"  3. DASS_Dep × Profile: F={f_dep:.2f}, p={p_dep:.4f}")

print(f"\nOutput Files:")
print(f"  1. {output_fit.name}")
print(f"  2. {output_chars.name}")
print(f"  3. {output_demo.name}")
print(f"  4. {output_assignments.name}")
print(f"  5. {output_heatmap.name}")
print(f"  6. {output_scatter.name}")

print("\n" + "=" * 90)
