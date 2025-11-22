"""
Paper 3 Reanalysis: GMM with k=3-4 to investigate gender subgroups

RATIONALE:
- Paper 1: Strong gender moderation (males r=0.566, females r=-0.078)
- Paper 3 k=2: NO gender differences (χ²=1.02, p=.31)
- Hypothesis: k=2 merges "male lapse-prone" with other high-variability cases
- Solution: Fit k=3-4 to separate gender-specific subtypes

ANALYSIS PLAN:
1. Fit k=3 and k=4 GMM models
2. Examine profile characteristics by gender
3. Test for gender × profile associations
4. Compare to k=2 solution
"""

import sys
from pathlib import Path
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/paper3_profiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*100)
print("PAPER 3 REANALYSIS: GMM k=3-4 FOR GENDER SUBGROUPS")
print("="*100)

# === LOAD DATA ===
print("\n[1] Loading data...")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]
participants = master[['participant_id','gender_normalized','age']].rename(columns={'gender_normalized':'gender'})
participants['gender'] = participants['gender'].fillna("").astype(str).str.strip().str.lower()

# Surveys not needed; use master for UCLA/DASS
surveys = master[['participant_id', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']].dropna()

# Load Paper 1 EF variability metrics
paper1_dir = Path("results/analysis_outputs/paper1_distributional")
paper1_results = pd.read_csv(paper1_dir / "paper1_participant_variability_metrics.csv", encoding='utf-8-sig')

# Merge
df = participants[['participant_id', 'age', 'gender']].copy()

# UCLA/DASS from master
df = df.merge(surveys, on='participant_id', how='inner')

# Paper 1 EF metrics
df = df.merge(paper1_results, on='participant_id', how='inner')

# Drop missing values and check
print(f"After merge: N={len(df)}")
print(f"Columns: {list(df.columns)}")

# Resolve duplicate columns from merge (use _x versions, which come from participants table)
if 'gender_x' in df.columns:
    df['gender'] = df['gender_x']
if 'age_x' in df.columns:
    df['age'] = df['age_x']
if 'ucla_total_x' in df.columns:
    df['ucla_total'] = df['ucla_total_x']
if 'dass_depression_x' in df.columns:
    df['dass_depression'] = df['dass_depression_x']
    df['dass_anxiety'] = df['dass_anxiety_x']
    df['dass_stress'] = df['dass_stress_x']

# Drop duplicate columns
cols_to_drop = [c for c in df.columns if c.endswith('_y') or (c.endswith('_x') and c.replace('_x', '') in df.columns)]
df = df.drop(columns=cols_to_drop)

# Gender binary
df = df.dropna(subset=['gender'])
df['gender_male'] = (df['gender'] == 'male').astype(int)

print(f"Final sample: N={len(df)} ({df['gender_male'].sum()} males, {len(df) - df['gender_male'].sum()} females)")

# === PREPARE FEATURES ===
print("\n[2] Preparing features...")
features_dict = {
    'PRP_tau': 'prp_tau_long',
    'PRP_RMSSD': 'prp_rmssd',
    'PRP_sigma': 'prp_sigma_long',
    'WCST_tau': 'wcst_tau',
    'WCST_RMSSD': 'wcst_rmssd',
    'Stroop_tau': 'stroop_tau_incong',
    'Stroop_RMSSD': 'stroop_rmssd'
}

# Drop rows with missing feature values
feature_cols = [features_dict[k] for k in features_dict.keys()]
df_complete = df.dropna(subset=feature_cols).copy()
print(f"Complete cases: N={len(df_complete)} (dropped {len(df) - len(df_complete)} rows with missing features)")

X = df_complete[[features_dict[k] for k in features_dict.keys()]].values

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === FIT K=3 AND K=4 MODELS ===
print("\n[3] Fitting GMM models...")

results = []

for k in [3, 4]:
    print(f"\n  Fitting k={k}...")
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='full',
        max_iter=200,
        n_init=20,
        random_state=42
    )
    gmm.fit(X_scaled)

    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    # Compute entropy (classification certainty)
    entropy = -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)

    results.append({
        'k': k,
        'BIC': bic,
        'AIC': aic,
        'Entropy': entropy,
        'model': gmm,
        'labels': labels,
        'probs': probs
    })

    print(f"    BIC={bic:.2f}, AIC={aic:.2f}, Entropy={entropy:.3f}")

# === ANALYZE K=3 PROFILES ===
print("\n" + "="*100)
print("K=3 PROFILE ANALYSIS")
print("="*100)

k3_result = [r for r in results if r['k'] == 3][0]
df_complete['profile_k3'] = k3_result['labels']
df = df_complete  # Use complete cases for all subsequent analyses

# Profile characteristics
print("\n[4.1] Profile characteristics (k=3):")
print("-"*100)

for profile_id in range(3):
    subset = df[df['profile_k3'] == profile_id]
    n_total = len(subset)
    n_male = subset['gender_male'].sum()
    pct_male = 100 * n_male / n_total

    print(f"\nProfile {profile_id+1} (N={n_total}, {pct_male:.1f}% male):")
    print(f"  Gender: {n_male} males, {n_total - n_male} females")
    print(f"  Age: M={subset['age'].mean():.1f}, SD={subset['age'].std():.1f}")
    print(f"  UCLA: M={subset['ucla_total'].mean():.1f}, SD={subset['ucla_total'].std():.1f}")
    print(f"  DASS Dep: M={subset['dass_depression'].mean():.1f}")
    print(f"  PRP tau: M={subset['prp_tau_long'].mean():.1f} ms")
    print(f"  WCST tau: M={subset['wcst_tau'].mean():.1f} ms")
    print(f"  Stroop tau: M={subset['stroop_tau_incong'].mean():.1f} ms")

# Gender distribution test
print("\n[4.2] Gender × Profile association (k=3):")
print("-"*100)

contingency_k3 = pd.crosstab(df['gender'], df['profile_k3'])
print("\nContingency table:")
print(contingency_k3)

chi2_k3, p_k3, dof_k3, expected_k3 = stats.chi2_contingency(contingency_k3)
print(f"\nχ²({dof_k3}) = {chi2_k3:.3f}, p = {p_k3:.4f}")

# Effect size
n = len(df)
cramer_v_k3 = np.sqrt(chi2_k3 / (n * (min(contingency_k3.shape) - 1)))
print(f"Cramér's V = {cramer_v_k3:.3f}")

if p_k3 < 0.05:
    print("VERDICT: [SIGNIFICANT] - Gender distribution differs across profiles")
else:
    print("VERDICT: [NOT SIGNIFICANT] - No gender differences")

# UCLA differences
print("\n[4.3] UCLA × Profile ANOVA (k=3):")
print("-"*100)

groups_k3 = [df[df['profile_k3'] == i]['ucla_total'].values for i in range(3)]
f_k3, p_ucla_k3 = stats.f_oneway(*groups_k3)
print(f"F({2},{len(df)-3}) = {f_k3:.3f}, p = {p_ucla_k3:.4f}")

# === ANALYZE K=4 PROFILES ===
print("\n" + "="*100)
print("K=4 PROFILE ANALYSIS")
print("="*100)

k4_result = [r for r in results if r['k'] == 4][0]
df['profile_k4'] = k4_result['labels']

# Profile characteristics
print("\n[5.1] Profile characteristics (k=4):")
print("-"*100)

for profile_id in range(4):
    subset = df[df['profile_k4'] == profile_id]
    n_total = len(subset)
    n_male = subset['gender_male'].sum()
    pct_male = 100 * n_male / n_total if n_total > 0 else 0

    print(f"\nProfile {profile_id+1} (N={n_total}, {pct_male:.1f}% male):")
    print(f"  Gender: {n_male} males, {n_total - n_male} females")
    print(f"  Age: M={subset['age'].mean():.1f}, SD={subset['age'].std():.1f}")
    print(f"  UCLA: M={subset['ucla_total'].mean():.1f}, SD={subset['ucla_total'].std():.1f}")
    print(f"  DASS Dep: M={subset['dass_depression'].mean():.1f}")
    print(f"  PRP tau: M={subset['prp_tau_long'].mean():.1f} ms")
    print(f"  WCST tau: M={subset['wcst_tau'].mean():.1f} ms")
    print(f"  Stroop tau: M={subset['stroop_tau_incong'].mean():.1f} ms")

# Gender distribution test
print("\n[5.2] Gender × Profile association (k=4):")
print("-"*100)

contingency_k4 = pd.crosstab(df['gender'], df['profile_k4'])
print("\nContingency table:")
print(contingency_k4)

chi2_k4, p_k4, dof_k4, expected_k4 = stats.chi2_contingency(contingency_k4)
print(f"\nχ²({dof_k4}) = {chi2_k4:.3f}, p = {p_k4:.4f}")

# Effect size
cramer_v_k4 = np.sqrt(chi2_k4 / (n * (min(contingency_k4.shape) - 1)))
print(f"Cramér's V = {cramer_v_k4:.3f}")

if p_k4 < 0.05:
    print("VERDICT: [SIGNIFICANT] - Gender distribution differs across profiles")
else:
    print("VERDICT: [NOT SIGNIFICANT] - No gender differences")

# UCLA differences
print("\n[5.3] UCLA × Profile ANOVA (k=4):")
print("-"*100)

groups_k4 = [df[df['profile_k4'] == i]['ucla_total'].values for i in range(4)]
f_k4, p_ucla_k4 = stats.f_oneway(*groups_k4)
print(f"F({3},{len(df)-4}) = {f_k4:.3f}, p = {p_ucla_k4:.4f}")

# === TARGETED ANALYSIS: IDENTIFY "MALE LAPSE-PRONE" PROFILE ===
print("\n" + "="*100)
print("TARGETED ANALYSIS: MALE LAPSE-PRONE PROFILE")
print("="*100)

print("\n[6] Searching for gender-specific high-variability subgroup...")
print("-"*100)

# For k=3
print("\nK=3 Profile Gender Proportions:")
for profile_id in range(3):
    subset = df[df['profile_k3'] == profile_id]
    pct_male = 100 * subset['gender_male'].sum() / len(subset)
    mean_tau = subset['prp_tau_long'].mean()
    print(f"  Profile {profile_id+1}: {pct_male:.1f}% male, PRP tau M={mean_tau:.1f} ms")

# Identify high-tau profiles
k3_profiles_sorted = df.groupby('profile_k3')['prp_tau_long'].mean().sort_values(ascending=False)
print(f"\nK=3 Profiles ranked by PRP tau:")
for idx, (profile_id, mean_tau) in enumerate(k3_profiles_sorted.items()):
    subset = df[df['profile_k3'] == profile_id]
    pct_male = 100 * subset['gender_male'].sum() / len(subset)
    print(f"  #{idx+1}: Profile {profile_id+1} (tau={mean_tau:.1f} ms, {pct_male:.1f}% male)")

# For k=4
print("\nK=4 Profile Gender Proportions:")
for profile_id in range(4):
    subset = df[df['profile_k4'] == profile_id]
    pct_male = 100 * subset['gender_male'].sum() / len(subset) if len(subset) > 0 else 0
    mean_tau = subset['prp_tau_long'].mean()
    print(f"  Profile {profile_id+1}: {pct_male:.1f}% male, PRP tau M={mean_tau:.1f} ms")

k4_profiles_sorted = df.groupby('profile_k4')['prp_tau_long'].mean().sort_values(ascending=False)
print(f"\nK=4 Profiles ranked by PRP tau:")
for idx, (profile_id, mean_tau) in enumerate(k4_profiles_sorted.items()):
    subset = df[df['profile_k4'] == profile_id]
    pct_male = 100 * subset['gender_male'].sum() / len(subset) if len(subset) > 0 else 0
    print(f"  #{idx+1}: Profile {profile_id+1} (tau={mean_tau:.1f} ms, {pct_male:.1f}% male)")

# === WITHIN-PROFILE GENDER EFFECTS ===
print("\n" + "="*100)
print("WITHIN-PROFILE GENDER EFFECTS")
print("="*100)

print("\n[7] Testing UCLA × Gender within high-tau profiles...")
print("-"*100)

# For k=3: test within highest-tau profile
highest_tau_k3 = k3_profiles_sorted.index[0]
subset_k3 = df[df['profile_k3'] == highest_tau_k3].copy()

males_k3 = subset_k3[subset_k3['gender_male'] == 1]
females_k3 = subset_k3[subset_k3['gender_male'] == 0]

if len(males_k3) >= 5 and len(females_k3) >= 5:
    r_male_k3, p_male_k3 = stats.pearsonr(males_k3['ucla_total'], males_k3['prp_tau_long'])
    r_female_k3, p_female_k3 = stats.pearsonr(females_k3['ucla_total'], females_k3['prp_tau_long'])

    print(f"\nK=3 Profile {highest_tau_k3+1} (highest tau):")
    print(f"  Males (N={len(males_k3)}): r={r_male_k3:.3f}, p={p_male_k3:.4f}")
    print(f"  Females (N={len(females_k3)}): r={r_female_k3:.3f}, p={p_female_k3:.4f}")
else:
    print(f"\nK=3 Profile {highest_tau_k3+1}: INSUFFICIENT DATA (males N={len(males_k3)}, females N={len(females_k3)})")

# For k=4: test within highest-tau profile
highest_tau_k4 = k4_profiles_sorted.index[0]
subset_k4 = df[df['profile_k4'] == highest_tau_k4].copy()

males_k4 = subset_k4[subset_k4['gender_male'] == 1]
females_k4 = subset_k4[subset_k4['gender_male'] == 0]

if len(males_k4) >= 5 and len(females_k4) >= 5:
    r_male_k4, p_male_k4 = stats.pearsonr(males_k4['ucla_total'], males_k4['prp_tau_long'])
    r_female_k4, p_female_k4 = stats.pearsonr(females_k4['ucla_total'], females_k4['prp_tau_long'])

    print(f"\nK=4 Profile {highest_tau_k4+1} (highest tau):")
    print(f"  Males (N={len(males_k4)}): r={r_male_k4:.3f}, p={p_male_k4:.4f}")
    print(f"  Females (N={len(females_k4)}): r={r_female_k4:.3f}, p={p_female_k4:.4f}")
else:
    print(f"\nK=4 Profile {highest_tau_k4+1}: INSUFFICIENT DATA (males N={len(males_k4)}, females N={len(females_k4)})")

# === COMPARISON TO K=2 ===
print("\n" + "="*100)
print("COMPARISON: K=2 vs K=3 vs K=4")
print("="*100)

# Load k=2 results
k2_assignments = pd.read_csv(OUTPUT_DIR / "paper3_profile_assignments.csv", encoding='utf-8-sig')
df = df.merge(k2_assignments[['participant_id', 'profile']].rename(columns={'profile': 'profile_k2'}), on='participant_id', how='left')

print("\n[8] Model comparison:")
print("-"*100)

comparison_df = pd.DataFrame({
    'k': [2, 3, 4],
    'BIC': [1438.96, results[0]['BIC'], results[1]['BIC']],
    'AIC': [1275.37, results[0]['AIC'], results[1]['AIC']],
    'Entropy': [0.049, results[0]['Entropy'], results[1]['Entropy']],
    'Gender_chi2': [1.02, chi2_k3, chi2_k4],
    'Gender_p': [0.311, p_k3, p_k4],
    'UCLA_F': [1.85, f_k3, f_k4],
    'UCLA_p': [0.178, p_ucla_k3, p_ucla_k4]
})

print("\n" + comparison_df.to_string(index=False))

print("\n[9] Interpretation:")
print("-"*100)

# BIC winner
best_bic_k = comparison_df.loc[comparison_df['BIC'].idxmin(), 'k']
print(f"\nBIC best model: k={int(best_bic_k)}")

# AIC winner
best_aic_k = comparison_df.loc[comparison_df['AIC'].idxmin(), 'k']
print(f"AIC best model: k={int(best_aic_k)}")

# Gender effects
sig_gender = comparison_df[comparison_df['Gender_p'] < 0.05]
if len(sig_gender) > 0:
    print(f"\nSignificant gender effects: k={sig_gender['k'].values}")
else:
    print(f"\nSignificant gender effects: NONE")

# === SAVE RESULTS ===
print("\n" + "="*100)
print("SAVING RESULTS")
print("="*100)

# Save k=3 and k=4 assignments
df[['participant_id', 'profile_k3', 'profile_k4']].to_csv(
    OUTPUT_DIR / "paper3_gmm_k34_assignments.csv",
    index=False,
    encoding='utf-8-sig'
)

# Save comparison table
comparison_df.to_csv(
    OUTPUT_DIR / "paper3_gmm_k234_comparison.csv",
    index=False,
    encoding='utf-8-sig'
)

print("\nSaved:")
print("  - paper3_gmm_k34_assignments.csv")
print("  - paper3_gmm_k234_comparison.csv")

print("\n" + "="*100)
print("REANALYSIS COMPLETE")
print("="*100)
