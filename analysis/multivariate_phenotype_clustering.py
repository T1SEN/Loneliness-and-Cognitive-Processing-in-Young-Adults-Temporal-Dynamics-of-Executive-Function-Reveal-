"""
Multivariate Phenotype Clustering
==================================
Identifies distinct cognitive profiles across WCST, PRP, and Stroop

Research Questions:
1. How many distinct EF phenotypes exist?
2. Do lonely males cluster into specific phenotypes?
3. What are the DASS profiles of each phenotype?
4. Can phenotypes guide personalized interventions?

Expected: 3-4 phenotypes (global deficit, WCST-specific, compensatory, healthy)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from data_loader_utils import load_master_dataset, load_exgaussian_params

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("results/analysis_outputs/ef_phenotypes")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("MULTIVARIATE PHENOTYPE CLUSTERING")
print("=" * 80)
print("\nResearch Question: What distinct cognitive profiles exist?")
print("  - Integrate WCST, PRP, Stroop metrics")
print("  - Identify phenotypes via K-means clustering")
print("  - Characterize UCLA, DASS profiles\n")

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load master dataset
master = load_master_dataset()

# Load Ex-Gaussian parameters
try:
    prp_exg = load_exgaussian_params('prp')
    master = master.merge(prp_exg[['participant_id', 'long_tau', 'long_sigma', 'short_tau', 'short_sigma']],
                          on='participant_id', how='left')
    master = master.rename(columns={
        'long_tau': 'prp_tau_long',
        'long_sigma': 'prp_sigma_long',
        'short_tau': 'prp_tau_short',
        'short_sigma': 'prp_sigma_short'
    })
    print("  Loaded PRP Ex-Gaussian parameters")
except Exception as e:
    print(f"  Warning: Could not load PRP Ex-Gaussian: {e}")

try:
    stroop_exg = load_exgaussian_params('stroop')
    master = master.merge(stroop_exg[['participant_id', 'tau', 'sigma', 'mu']],
                          on='participant_id', how='left')
    master = master.rename(columns={
        'tau': 'stroop_tau',
        'sigma': 'stroop_sigma',
        'mu': 'stroop_mu'
    })
    print("  Loaded Stroop Ex-Gaussian parameters")
except Exception as e:
    print(f"  Warning: Could not load Stroop Ex-Gaussian: {e}")

# Select clustering features - use core features that we know exist
# Start with basic features from master dataset
feature_cols_basic = [
    'pe_rate',                    # WCST: perseverative errors
    'wcst_sd_rt',                 # WCST: RT variability
]

# Optional features (may not be in master)
feature_cols_optional = [
    'stroop_interference',        # Stroop: interference effect (if available)
    'wcst_accuracy',              # WCST: accuracy
]

# Add Ex-Gaussian features if available
feature_cols_exg = [
    'stroop_sigma',               # Stroop: RT variability
    'stroop_tau',                 # Stroop: slow tail
    'prp_tau_long',               # PRP: attention lapses (long SOA)
    'prp_sigma_long'              # PRP: RT variability (long SOA)
]

# Check which features are available
available_features_core = [f for f in feature_cols_basic if f in master.columns and master[f].notna().sum() > 50]
available_features_opt = [f for f in feature_cols_optional if f in master.columns and master[f].notna().sum() > 50]
available_features_exg = [f for f in feature_cols_exg if f in master.columns and master[f].notna().sum() > 50]

print(f"\nAvailable CORE features (required, {len(available_features_core)}):")
for f in available_features_core:
    non_null = master[f].notna().sum()
    print(f"  - {f}: {non_null} non-null")

if available_features_opt:
    print(f"\nAvailable OPTIONAL features ({len(available_features_opt)}):")
    for f in available_features_opt:
        non_null = master[f].notna().sum()
        print(f"  - {f}: {non_null} non-null")

if available_features_exg:
    print(f"\nAvailable Ex-Gaussian features ({len(available_features_exg)}):")
    for f in available_features_exg:
        non_null = master[f].notna().sum()
        print(f"  - {f}: {non_null} non-null")

# Use all available features
all_features = available_features_core + available_features_opt + available_features_exg

# Create clustering dataset
cluster_data = master[['participant_id', 'gender_male', 'ucla_total',
                        'dass_depression', 'dass_anxiety', 'dass_stress'] +
                       all_features].copy()

# Drop rows with missing values in CORE features only (others will be imputed)
cluster_data = cluster_data.dropna(subset=['participant_id', 'gender_male', 'ucla_total'] +
                                           available_features_core)

print(f"\nClustering dataset: N={len(cluster_data)}")
print(f"  Males: {cluster_data['gender_male'].sum()}")
print(f"  Females: {(1-cluster_data['gender_male']).sum()}\n")

# ============================================================================
# K-Means Clustering
# ============================================================================

print("=" * 80)
print("K-MEANS CLUSTERING")
print("=" * 80)

# Determine which features to actually use for clustering
# Use only features with sufficient data
final_features = []
for feat in all_features:
    if feat in cluster_data.columns:
        non_null_count = cluster_data[feat].notna().sum()
        if non_null_count >= len(cluster_data) * 0.5:  # At least 50% non-null
            final_features.append(feat)

print(f"Features for clustering ({len(final_features)}):")
for f in final_features:
    print(f"  - {f}")

if len(final_features) == 0:
    print("\nERROR: No features with sufficient data!")
    sys.exit(1)

# Prepare feature matrix - impute missing values with median
from sklearn.impute import SimpleImputer
X = cluster_data[final_features].values

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Test different K values
silhouette_scores = []
inertias = []

print("\nTesting different cluster numbers...")
print("-" * 60)

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    inertias.append(kmeans.inertia_)

    print(f"K={k}: Silhouette = {sil_score:.3f}, Inertia = {kmeans.inertia_:.1f}")

# Choose optimal K (highest silhouette)
optimal_k = np.argmax(silhouette_scores) + 2
print(f"\n✓ Optimal K = {optimal_k} (highest silhouette score)\n")

# Fit final model
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=100)
cluster_data['cluster'] = kmeans_final.fit_predict(X_scaled)

# Get cluster centers (in original scale)
centers_scaled = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

# Save cluster centers
centers_df = pd.DataFrame(centers_original, columns=final_features)
centers_df['cluster'] = range(optimal_k)

centers_file = OUTPUT_DIR / "cluster_centers.csv"
centers_df.to_csv(centers_file, index=False, encoding='utf-8-sig')
print(f"Saved cluster centers: {centers_file}")

# ============================================================================
# Characterize Clusters
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER CHARACTERIZATION")
print("=" * 80)

# Compute cluster statistics
cluster_profiles = []

for c in range(optimal_k):
    cluster_members = cluster_data[cluster_data['cluster'] == c]

    profile = {
        'cluster': c,
        'n': len(cluster_members),
        'n_male': cluster_members['gender_male'].sum(),
        'n_female': (1-cluster_members['gender_male']).sum(),
        'pct_male': (cluster_members['gender_male'].sum() / len(cluster_members)) * 100,
        'ucla_mean': cluster_members['ucla_total'].mean(),
        'ucla_sd': cluster_members['ucla_total'].std(),
        'dass_dep_mean': cluster_members['dass_depression'].mean(),
        'dass_anx_mean': cluster_members['dass_anxiety'].mean(),
        'dass_stress_mean': cluster_members['dass_stress'].mean()
    }

    # Add feature means
    for feat in final_features:
        if feat in cluster_members.columns:
            profile[f'{feat}_mean'] = cluster_members[feat].mean()
            profile[f'{feat}_sd'] = cluster_members[feat].std()

    cluster_profiles.append(profile)

profiles_df = pd.DataFrame(cluster_profiles)

# Print profiles
print(f"\n{optimal_k} Clusters Identified:")
print("=" * 80)

for _, row in profiles_df.iterrows():
    c = int(row['cluster'])
    print(f"\nCLUSTER {c} (N={row['n']:.0f}, {row['pct_male']:.0f}% male)")
    print("-" * 60)
    print(f"  UCLA:  {row['ucla_mean']:.1f} ± {row['ucla_sd']:.1f}")
    print(f"  DASS:  Dep={row['dass_dep_mean']:.1f}, Anx={row['dass_anx_mean']:.1f}, Stress={row['dass_stress_mean']:.1f}")
    print(f"  EF Metrics:")
    for feat in final_features:
        mean_key = f'{feat}_mean'
        if mean_key in row:
            print(f"    {feat}: {row[mean_key]:.2f}")

# Save profiles
profiles_file = OUTPUT_DIR / "cluster_profiles.csv"
profiles_df.to_csv(profiles_file, index=False, encoding='utf-8-sig')
print(f"\nSaved cluster profiles: {profiles_file}")

# ============================================================================
# Cluster × UCLA/Gender Distribution
# ============================================================================

print("\n\n" + "=" * 80)
print("CLUSTER × UCLA/GENDER DISTRIBUTION")
print("=" * 80)

# Create UCLA groups
cluster_data['ucla_group'] = 'Medium UCLA'
ucla_median = cluster_data['ucla_total'].median()
cluster_data.loc[cluster_data['ucla_total'] > ucla_median, 'ucla_group'] = 'High UCLA'
cluster_data.loc[cluster_data['ucla_total'] < ucla_median, 'ucla_group'] = 'Low UCLA'

# Contingency table: Cluster × UCLA Group (males only)
males = cluster_data[cluster_data['gender_male'] == 1]

print("\nMales: Cluster × UCLA Group")
print("-" * 60)

crosstab = pd.crosstab(males['cluster'], males['ucla_group'], margins=True)
print(crosstab)

# Chi-square test
if len(males) >= 20:
    chi2, p, dof, expected = stats.chi2_contingency(crosstab.iloc[:-1, :-1])
    print(f"\nChi-square test: χ²={chi2:.2f}, p={p:.4f}")
    if p < 0.05:
        print("  ⭐ SIGNIFICANT: Lonely males cluster into specific phenotypes!")

# Contingency table: Cluster × UCLA Group (females)
females = cluster_data[cluster_data['gender_male'] == 0]

print("\n\nFemales: Cluster × UCLA Group")
print("-" * 60)

crosstab_f = pd.crosstab(females['cluster'], females['ucla_group'], margins=True)
print(crosstab_f)

# ============================================================================
# Label Clusters (Interpretive Names)
# ============================================================================

print("\n\n" + "=" * 80)
print("CLUSTER LABELING (Interpretive)")
print("=" * 80)

# Assign interpretive labels based on profiles
# (This is done manually based on inspection)

cluster_labels = {}

for c in range(optimal_k):
    profile = profiles_df[profiles_df['cluster'] == c].iloc[0]

    pe = profile['pe_rate_mean'] if 'pe_rate_mean' in profile else 0
    sigma = profile['stroop_sigma_mean'] if 'stroop_sigma_mean' in profile else 0
    tau = profile['prp_tau_long_mean'] if 'prp_tau_long_mean' in profile else 0
    ucla = profile['ucla_mean']

    # Simple heuristic labeling
    if pe > 12:
        label = "High PE (Vulnerable)"
    elif tau > 200:
        label = "High τ (Attention Lapses)"
    elif sigma > 100:
        label = "High σ (Variable RT)"
    elif ucla < 35:
        label = "Healthy Controls"
    else:
        label = f"Cluster {c}"

    cluster_labels[c] = label
    print(f"Cluster {c}: {label}")

# Add labels to data
cluster_data['cluster_label'] = cluster_data['cluster'].map(cluster_labels)

# ============================================================================
# Save Final Results
# ============================================================================

# Save participant-level cluster assignments
output_file = OUTPUT_DIR / "participant_cluster_assignments.csv"
cluster_data.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n\nSaved participant assignments: {output_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("PHENOTYPE CLUSTERING COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:")
print("-" * 80)

print(f"\n✓ Identified {optimal_k} distinct cognitive phenotypes")

# Find cluster with most lonely males
lonely_males_by_cluster = males[males['ucla_group'] == 'High UCLA'].groupby('cluster').size()
if len(lonely_males_by_cluster) > 0:
    most_vulnerable_cluster = lonely_males_by_cluster.idxmax()
    n_vulnerable = lonely_males_by_cluster.max()
    total_lonely_males = males[males['ucla_group'] == 'High UCLA'].shape[0]
    pct = (n_vulnerable / total_lonely_males) * 100 if total_lonely_males > 0 else 0

    print(f"\n✓ Lonely males concentrated in Cluster {most_vulnerable_cluster}")
    print(f"  {n_vulnerable}/{total_lonely_males} ({pct:.0f}%) of lonely males")
    print(f"  Label: {cluster_labels.get(most_vulnerable_cluster, 'Unknown')}")

print("\n" + "=" * 80)
