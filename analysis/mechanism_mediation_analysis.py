"""
Mechanism & Mediation Analysis
===============================

Explores the pathways through which loneliness affects WCST performance,
with focus on DASS as mediator and gender as moderator.

Analyses:
1. Mediation analysis (UCLA → DASS → WCST) by gender
2. DASS stratification (Low vs High DASS)
3. Male vulnerability profiling (cluster analysis)

Author: Research Team
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/mechanism_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("MECHANISM & MEDIATION ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/4] Loading data...")

master = load_master_dataset(use_cache=True)
master = load_master_dataset(use_cache=True)
participants = master[['participant_id','gender_normalized','age']].rename(columns={'gender_normalized':'gender'})

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

# Handle Korean gender
if 'gender' in master.columns:
    master['gender_male'] = 0
    master.loc[master['gender'] == '남성', 'gender_male'] = 1
    master.loc[master['gender'].str.lower() == 'male', 'gender_male'] = 1

# Filter complete cases (include age for covariate control per CLAUDE.md)
required_cols = ['ucla_total', 'pe_rate', 'gender_male', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']
master = master.dropna(subset=required_cols).copy()

print(f"  N={len(master)} ({(master['gender_male']==0).sum()} Female, {(master['gender_male']==1).sum()} Male)")
print()

# ============================================================================
# ANALYSIS 1: MEDIATION ANALYSIS (BY GENDER)
# ============================================================================

print("[2/4] Mediation analysis (UCLA → DASS → WCST)...")

def mediation_analysis(data, mediator_col, outcome_col, n_boot=5000):
    """
    Mediation with age control: UCLA → mediator → outcome (controlling age)
    Returns direct, indirect, total effects with bootstrap CIs

    NOTE: DASS is the MEDIATOR, so we don't control for it.
    We control for age in all paths.
    """
    # Path a: UCLA → Mediator (controlling age)
    model_a = ols(f"{mediator_col} ~ ucla_total + age", data=data).fit()
    a = model_a.params['ucla_total']

    # Path b & c': Mediator → Outcome (controlling UCLA + age)
    model_bc = ols(f"{outcome_col} ~ {mediator_col} + ucla_total + age", data=data).fit()
    b = model_bc.params[mediator_col]
    c_prime = model_bc.params['ucla_total']  # Direct effect

    # Total effect: UCLA → Outcome (controlling age, no mediator)
    model_c = ols(f"{outcome_col} ~ ucla_total + age", data=data).fit()
    c = model_c.params['ucla_total']  # Total effect

    # Indirect effect
    indirect = a * b

    # Bootstrap CI for indirect effect
    boot_indirect = []
    for _ in range(n_boot):
        boot_data = resample(data)
        try:
            boot_a = ols(f"{mediator_col} ~ ucla_total + age", data=boot_data).fit().params['ucla_total']
            boot_b = ols(f"{outcome_col} ~ {mediator_col} + ucla_total + age", data=boot_data).fit().params[mediator_col]
            boot_indirect.append(boot_a * boot_b)
        except:
            continue

    if len(boot_indirect) > 0:
        indirect_ci_lower = np.percentile(boot_indirect, 2.5)
        indirect_ci_upper = np.percentile(boot_indirect, 97.5)
        indirect_sig = (indirect_ci_lower > 0 and indirect_ci_upper > 0) or (indirect_ci_lower < 0 and indirect_ci_upper < 0)
    else:
        indirect_ci_lower, indirect_ci_upper, indirect_sig = np.nan, np.nan, False

    return {
        'a': a,  # UCLA → Mediator
        'b': b,  # Mediator → Outcome (controlling UCLA)
        'c_prime': c_prime,  # Direct: UCLA → Outcome (controlling mediator)
        'c': c,  # Total: UCLA → Outcome
        'indirect': indirect,  # a × b
        'indirect_ci_lower': indirect_ci_lower,
        'indirect_ci_upper': indirect_ci_upper,
        'indirect_significant': indirect_sig,
        'proportion_mediated': indirect / c if c != 0 else np.nan
    }


mediation_results = []

# Gender-stratified mediation
for gender, gender_label in [(0, 'Female'), (1, 'Male')]:
    gender_data = master[master['gender_male'] == gender].copy()

    if len(gender_data) < 20:
        continue

    for mediator in ['dass_depression', 'dass_anxiety', 'dass_stress']:
        for outcome in ['pe_rate', 'wcst_accuracy']:
            if outcome not in gender_data.columns:
                continue

            result = mediation_analysis(gender_data, mediator, outcome, n_boot=1000)
            result['gender'] = gender_label
            result['mediator'] = mediator
            result['outcome'] = outcome
            result['n'] = len(gender_data)
            mediation_results.append(result)

            print(f"  {gender_label} | {mediator} → {outcome}:")
            print(f"    Indirect: {result['indirect']:.4f}, 95%CI=[{result['indirect_ci_lower']:.4f}, {result['indirect_ci_upper']:.4f}], Sig={result['indirect_significant']}")
            print(f"    Direct: {result['c_prime']:.4f}, Total: {result['c']:.4f}")

mediation_df = pd.DataFrame(mediation_results)
mediation_df.to_csv(OUTPUT_DIR / "mediation_by_gender.csv", index=False, encoding='utf-8-sig')
print()

# ============================================================================
# ANALYSIS 2: DASS STRATIFICATION
# ============================================================================

print("[3/4] DASS stratification analysis...")

# Test gender moderation separately in Low vs High DASS groups
dass_strat_results = []

for dass_measure in ['dass_depression', 'dass_anxiety', 'dass_stress']:
    # Median split
    median_dass = master[dass_measure].median()

    low_dass = master[master[dass_measure] <= median_dass].copy()
    high_dass = master[master[dass_measure] > median_dass].copy()

    for stratum, stratum_data in [('Low', low_dass), ('High', high_dass)]:
        if len(stratum_data) < 30:
            continue

        # Standardize UCLA
        scaler = StandardScaler()
        stratum_data['z_ucla'] = scaler.fit_transform(stratum_data[['ucla_total']])

        for outcome in ['pe_rate', 'wcst_accuracy']:
            if outcome not in stratum_data.columns:
                continue

            # Note: DASS is used for stratification, not as covariate (stratification approach)
            # However, we add age control per CLAUDE.md best practices
            # Within DASS strata, we're testing if UCLA × Gender effect differs by DASS level
            if 'age' in stratum_data.columns:
                formula = f"{outcome} ~ z_ucla * C(gender_male) + age"
            else:
                formula = f"{outcome} ~ z_ucla * C(gender_male)"
            try:
                model = ols(formula, data=stratum_data.dropna(subset=[outcome])).fit()
                interaction_term = "z_ucla:C(gender_male)[T.1]"

                if interaction_term in model.params:
                    beta = model.params[interaction_term]
                    pval = model.pvalues[interaction_term]
                else:
                    beta, pval = np.nan, np.nan

                dass_strat_results.append({
                    'dass_measure': dass_measure,
                    'stratum': stratum,
                    'outcome': outcome,
                    'interaction_beta': beta,
                    'interaction_pval': pval,
                    'n': len(stratum_data.dropna(subset=[outcome])),
                    'n_female': (stratum_data['gender_male'] == 0).sum(),
                    'n_male': (stratum_data['gender_male'] == 1).sum()
                })

                print(f"  {dass_measure} {stratum} | {outcome}: β={beta:.3f}, p={pval:.4f} (N={len(stratum_data)})")
            except:
                continue

dass_strat_df = pd.DataFrame(dass_strat_results)
dass_strat_df.to_csv(OUTPUT_DIR / "dass_stratified_moderation.csv", index=False, encoding='utf-8-sig')
print()

# ============================================================================
# ANALYSIS 3: MALE VULNERABILITY PROFILING
# ============================================================================

print("[4/4] Male vulnerability profiling...")

# Cluster analysis on males only
males = master[master['gender_male'] == 1].copy()

if len(males) >= 15:
    # Variables for clustering
    cluster_vars = ['ucla_total', 'pe_rate', 'dass_depression', 'dass_anxiety', 'dass_stress']
    cluster_data = males[cluster_vars].dropna()

    if len(cluster_data) >= 15:
        # Standardize
        scaler = StandardScaler()
        cluster_data_std = scaler.fit_transform(cluster_data)

        # Try k=2 and k=3
        profile_results = []

        for k in [2, 3]:
            # Gaussian Mixture Model
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
            labels = gmm.fit_predict(cluster_data_std)

            # Add labels to data
            cluster_data['profile'] = labels

            # Calculate profile characteristics
            for profile_id in range(k):
                profile_subset = cluster_data[cluster_data['profile'] == profile_id]

                profile_results.append({
                    'k': k,
                    'profile': profile_id,
                    'n': len(profile_subset),
                    'mean_ucla': profile_subset['ucla_total'].mean(),
                    'mean_pe_rate': profile_subset['pe_rate'].mean(),
                    'mean_dass_dep': profile_subset['dass_depression'].mean(),
                    'mean_dass_anx': profile_subset['dass_anxiety'].mean(),
                    'mean_dass_stress': profile_subset['dass_stress'].mean()
                })

            print(f"  k={k} clusters:")
            for profile_id in range(k):
                profile_subset = cluster_data[cluster_data['profile'] == profile_id]
                print(f"    Profile {profile_id}: N={len(profile_subset)}, UCLA={profile_subset['ucla_total'].mean():.1f}, PE={profile_subset['pe_rate'].mean():.1f}%")

        profile_df = pd.DataFrame(profile_results)
        profile_df.to_csv(OUTPUT_DIR / "male_vulnerability_profiles.csv", index=False, encoding='utf-8-sig')

        # Save individual assignments (k=2)
        gmm_final = GaussianMixture(n_components=2, random_state=42, n_init=10)
        final_labels = gmm_final.fit_predict(cluster_data_std)
        cluster_data['profile_k2'] = final_labels

        cluster_data_full = males.merge(
            cluster_data[['profile_k2']],
            left_index=True,
            right_index=True,
            how='left'
        )

        cluster_data_full[['participant_id', 'ucla_total', 'pe_rate', 'dass_depression', 'profile_k2']].to_csv(
            OUTPUT_DIR / "male_profile_assignments.csv", index=False, encoding='utf-8-sig'
        )

        print()
    else:
        print("  Insufficient data for clustering")
else:
    print("  Insufficient male participants for profiling")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("MECHANISM ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - mediation_by_gender.csv")
print("  - dass_stratified_moderation.csv")
print("  - male_vulnerability_profiles.csv")
print("  - male_profile_assignments.csv")
print()

# Summary of significant mediations
sig_mediations = mediation_df[mediation_df['indirect_significant'] == True]
print(f"Significant mediation effects: {len(sig_mediations)}/{len(mediation_df)}")
if len(sig_mediations) > 0:
    for _, row in sig_mediations.iterrows():
        print(f"  - {row['gender']} | {row['mediator']} → {row['outcome']}: indirect={row['indirect']:.4f}")

# Summary of DASS stratification
sig_strat = dass_strat_df[dass_strat_df['interaction_pval'] < 0.05]
print(f"\nSignificant gender moderation in DASS strata: {len(sig_strat)}/{len(dass_strat_df)}")
if len(sig_strat) > 0:
    for _, row in sig_strat.iterrows():
        print(f"  - {row['dass_measure']} {row['stratum']} | {row['outcome']}: β={row['interaction_beta']:.3f}, p={row['interaction_pval']:.4f}")

print()
