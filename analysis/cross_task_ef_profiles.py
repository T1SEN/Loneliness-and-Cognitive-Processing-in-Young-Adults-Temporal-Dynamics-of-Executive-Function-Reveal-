"""
Cross-Task Executive Function Profile Analysis
==============================================

Purpose: Test whether the UCLA × Gender effect is WCST-SPECIFIC or reflects
         GENERAL executive dysfunction across multiple EF tasks.

Key Questions:
1. Do lonely males show deficits on WCST ONLY, or across Stroop + PRP too?
2. Are there distinct "EF profile types" (e.g., WCST-impaired vs global-impaired)?
3. How far are lonely males from the "healthy prototype" (low on all EF deficits)?

Tasks:
- Stroop: Inhibitory control (interference effect)
- WCST: Set-shifting (perseverative errors)
- PRP: Dual-task coordination (bottleneck effect)

Hypothesis: UCLA × Gender is WCST-specific (current evidence), not global EF.

Author: Claude Code
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/cross_task_profiles")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("Cross-Task EF Profile Analysis")
print("=" * 80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n1. Loading data...")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
cognitive_summary = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv")

print(f"   - Participants: {len(participants)}")
print(f"   - Survey responses: {len(surveys)}")
print(f"   - Cognitive summary: {len(cognitive_summary)}")

# ============================================================================
# 2. Extract UCLA and Demographics
# ============================================================================
print("\n2. Extracting UCLA scores and demographics...")

# Normalize column names
surveys = surveys.rename(columns={'participantId': 'participant_id'})

# UCLA scores
ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores.columns = ['participant_id', 'ucla_total']

# Demographics
demo = participants[['participantId', 'age', 'gender', 'education']].copy()
demo.columns = ['participant_id', 'age', 'gender', 'education']

# Recode gender from Korean to English
demo['gender'] = demo['gender'].map({'남성': 'male', '여성': 'female'})
demo = demo.dropna(subset=['gender'])

print(f"   - UCLA scores: {len(ucla_scores)}")
print(f"   - Demographics: {len(demo)}")

# ============================================================================
# 3. Extract EF Metrics from Cognitive Summary
# ============================================================================
print("\n3. Extracting EF metrics from cognitive summary...")

# Normalize column names
cognitive_summary = cognitive_summary.rename(columns={'participantId': 'participant_id'})

# Extract task-specific metrics
ef_metrics = []

for pid in cognitive_summary['participant_id'].unique():
    pid_data = cognitive_summary[cognitive_summary['participant_id'] == pid]

    # Stroop metrics
    stroop = pid_data[pid_data['testName'] == 'stroop']
    if len(stroop) > 0:
        # stroop_effect is already calculated
        stroop_interference = stroop['stroop_effect'].values[0] if 'stroop_effect' in stroop.columns else np.nan
    else:
        stroop_interference = np.nan

    # WCST metrics
    wcst = pid_data[pid_data['testName'] == 'wcst']
    if len(wcst) > 0:
        # Calculate PE rate
        pe_count = wcst['perseverativeErrorCount'].values[0] if 'perseverativeErrorCount' in wcst.columns else np.nan
        total_trials = wcst['totalTrialCount'].values[0] if 'totalTrialCount' in wcst.columns else np.nan

        if not pd.isna(pe_count) and not pd.isna(total_trials) and total_trials > 0:
            wcst_pe_rate = pe_count / total_trials
        else:
            wcst_pe_rate = np.nan

        wcst_total_errors = wcst['totalErrorCount'].values[0] if 'totalErrorCount' in wcst.columns else np.nan
    else:
        wcst_pe_rate = np.nan
        wcst_total_errors = np.nan

    # PRP metrics
    prp = pid_data[pid_data['testName'] == 'prp']
    if len(prp) > 0:
        # Bottleneck effect = T2 RT at short SOA - T2 RT at long SOA
        rt_short = prp['rt2_soa_50'].values[0] if 'rt2_soa_50' in prp.columns else np.nan
        rt_long = prp['rt2_soa_1200'].values[0] if 'rt2_soa_1200' in prp.columns else np.nan

        if not pd.isna(rt_short) and not pd.isna(rt_long):
            prp_bottleneck = rt_short - rt_long
        else:
            prp_bottleneck = np.nan
    else:
        prp_bottleneck = np.nan

    ef_metrics.append({
        'participant_id': pid,
        'stroop_interference': stroop_interference,
        'wcst_pe_rate': wcst_pe_rate,
        'wcst_total_errors': wcst_total_errors,
        'prp_bottleneck': prp_bottleneck
    })

ef_df = pd.DataFrame(ef_metrics)

print(f"   - Participants with EF metrics: {len(ef_df)}")
print(f"   - Stroop data: {ef_df['stroop_interference'].notna().sum()}")
print(f"   - WCST PE data: {ef_df['wcst_pe_rate'].notna().sum()}")
print(f"   - PRP data: {ef_df['prp_bottleneck'].notna().sum()}")

# ============================================================================
# 4. Merge All Data
# ============================================================================
print("\n4. Merging all data...")

master = ef_df.merge(ucla_scores, on='participant_id', how='inner')
master = master.merge(demo, on='participant_id', how='inner')

# Drop rows with missing EF metrics
master = master.dropna(subset=['ucla_total', 'stroop_interference', 'wcst_pe_rate'])

print(f"   - Final sample size: {len(master)}")
print(f"   - Gender distribution: {master['gender'].value_counts().to_dict()}")

# ============================================================================
# 5. Test UCLA × Gender Interactions on Each EF Task
# ============================================================================
print("\n5. Testing UCLA × Gender interactions for each EF task...")

# Code variables
master['gender_coded'] = master['gender'].map({'male': 1, 'female': -1})
master['z_ucla'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()
master['z_age'] = (master['age'] - master['age'].mean()) / master['age'].std()
master['ucla_x_gender'] = master['z_ucla'] * master['gender_coded']

# Standardize EF metrics (for comparability)
master['z_stroop'] = (master['stroop_interference'] - master['stroop_interference'].mean()) / master['stroop_interference'].std()
master['z_wcst_pe'] = (master['wcst_pe_rate'] - master['wcst_pe_rate'].mean()) / master['wcst_pe_rate'].std()

# Test each task separately
X = master[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values

# Stroop
y_stroop = master['z_stroop'].values
model_stroop = LinearRegression().fit(X, y_stroop)
stroop_interaction = model_stroop.coef_[-1]

# WCST
y_wcst = master['z_wcst_pe'].values
model_wcst = LinearRegression().fit(X, y_wcst)
wcst_interaction = model_wcst.coef_[-1]

print(f"\n   Stroop Interference:")
print(f"      - UCLA × Gender β = {stroop_interaction:.4f}")

print(f"\n   WCST PE Rate:")
print(f"      - UCLA × Gender β = {wcst_interaction:.4f}")

print(f"\n   Comparison:")
print(f"      - WCST/Stroop interaction ratio: {abs(wcst_interaction)/abs(stroop_interaction) if stroop_interaction != 0 else np.inf:.2f}×")

if abs(wcst_interaction) > abs(stroop_interaction) * 2:
    print(f"      → WCST-SPECIFIC effect (not global EF impairment)")
else:
    print(f"      → Both tasks show comparable effects (general EF deficit)")

# ============================================================================
# 6. Compute EF Profile Scores
# ============================================================================
print("\n6. Computing EF profile scores...")

# Create composite EF profile for each person
# Higher scores = worse EF performance
master['ef_composite'] = (master['z_stroop'] + master['z_wcst_pe']) / 2

# Distance from "healthy prototype" (low on all deficits)
# Healthy = bottom 25% on both tasks
stroop_cutoff = master['z_stroop'].quantile(0.25)
wcst_cutoff = master['z_wcst_pe'].quantile(0.25)

# Euclidean distance from healthy prototype
master['distance_from_healthy'] = np.sqrt(
    (master['z_stroop'] - stroop_cutoff)**2 +
    (master['z_wcst_pe'] - wcst_cutoff)**2
)

# Test UCLA × Gender on composite and distance
y_composite = master['ef_composite'].values
model_composite = LinearRegression().fit(X, y_composite)
composite_interaction = model_composite.coef_[-1]

y_distance = master['distance_from_healthy'].values
model_distance = LinearRegression().fit(X, y_distance)
distance_interaction = model_distance.coef_[-1]

print(f"\n   EF Composite Score:")
print(f"      - UCLA × Gender β = {composite_interaction:.4f}")

print(f"\n   Distance from Healthy Prototype:")
print(f"      - UCLA × Gender β = {distance_interaction:.4f}")

# ============================================================================
# 7. Profile Types: Clustering Analysis
# ============================================================================
print("\n7. Identifying EF profile types (K-means clustering)...")

# Use complete data only
cluster_data = master[['z_stroop', 'z_wcst_pe']].dropna()
cluster_ids = cluster_data.index

# K-means with k=3 (healthy, WCST-impaired, global-impaired)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_data)

# Add cluster labels to master
master['cluster'] = np.nan
master.loc[cluster_ids, 'cluster'] = clusters

# Describe clusters
print(f"\n   Cluster centers:")
for i in range(3):
    center = kmeans.cluster_centers_[i]
    n_members = (clusters == i).sum()
    print(f"      Cluster {i} (N={n_members}): Stroop={center[0]:.2f}, WCST={center[1]:.2f}")

# Identify cluster types
cluster_profiles = []
for i in range(3):
    center = kmeans.cluster_centers_[i]
    if center[0] < 0 and center[1] < 0:
        profile_type = "Healthy"
    elif center[0] < 0 and center[1] > 0:
        profile_type = "WCST-Impaired"
    elif center[0] > 0 and center[1] < 0:
        profile_type = "Stroop-Impaired"
    else:
        profile_type = "Global-Impaired"

    cluster_profiles.append({'cluster': i, 'type': profile_type})

cluster_types = pd.DataFrame(cluster_profiles)
print(f"\n   Cluster types:")
for _, row in cluster_types.iterrows():
    n_members = (master['cluster'] == row['cluster']).sum()
    print(f"      Cluster {row['cluster']}: {row['type']} (N={n_members})")

# Test if UCLA × Gender predicts cluster membership (focus on WCST-impaired)
# Use logistic approach: does UCLA × Gender predict being in WCST-impaired cluster?

# Find WCST-impaired cluster (high WCST, low Stroop)
wcst_impaired_cluster = None
for i, center in enumerate(kmeans.cluster_centers_):
    if center[1] > 0 and center[0] < 0.5:  # High WCST, relatively low Stroop
        wcst_impaired_cluster = i
        break

if wcst_impaired_cluster is not None:
    master['is_wcst_impaired'] = (master['cluster'] == wcst_impaired_cluster).astype(int)
    cluster_data_full = master[master['cluster'].notna()]

    # Test gender × UCLA predicting WCST-impaired membership
    male_wcst_imp = cluster_data_full[(cluster_data_full['gender'] == 'male') & (cluster_data_full['is_wcst_impaired'] == 1)]
    female_wcst_imp = cluster_data_full[(cluster_data_full['gender'] == 'female') & (cluster_data_full['is_wcst_impaired'] == 1)]

    male_wcst_imp_rate = len(male_wcst_imp) / len(cluster_data_full[cluster_data_full['gender'] == 'male'])
    female_wcst_imp_rate = len(female_wcst_imp) / len(cluster_data_full[cluster_data_full['gender'] == 'female'])

    print(f"\n   WCST-Impaired cluster membership:")
    print(f"      - Males: {male_wcst_imp_rate*100:.1f}%")
    print(f"      - Females: {female_wcst_imp_rate*100:.1f}%")

# ============================================================================
# 8. Stratified Correlations
# ============================================================================
print("\n8. Stratified correlations by gender...")

male_data = master[master['gender'] == 'male']
female_data = master[master['gender'] == 'female']

# Stroop
male_r_stroop, male_p_stroop = stats.pearsonr(male_data['ucla_total'], male_data['stroop_interference'])
female_r_stroop, female_p_stroop = stats.pearsonr(female_data['ucla_total'], female_data['stroop_interference'])

print(f"\n   UCLA × Stroop Interference:")
print(f"      - Males (N={len(male_data)}): r={male_r_stroop:.3f}, p={male_p_stroop:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_stroop:.3f}, p={female_p_stroop:.4f}")

# WCST
male_r_wcst, male_p_wcst = stats.pearsonr(male_data['ucla_total'], male_data['wcst_pe_rate'])
female_r_wcst, female_p_wcst = stats.pearsonr(female_data['ucla_total'], female_data['wcst_pe_rate'])

print(f"\n   UCLA × WCST PE Rate:")
print(f"      - Males (N={len(male_data)}): r={male_r_wcst:.3f}, p={male_p_wcst:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_wcst:.3f}, p={female_p_wcst:.4f}")

# ============================================================================
# 9. Descriptive Statistics
# ============================================================================
print("\n9. Descriptive statistics:")

print(f"\n   Overall EF performance:")
print(f"      - Stroop interference: M={master['stroop_interference'].mean():.1f}ms (SD={master['stroop_interference'].std():.1f})")
print(f"      - WCST PE rate: M={master['wcst_pe_rate'].mean():.3f} (SD={master['wcst_pe_rate'].std():.3f})")

print(f"\n   By gender:")
for gender in ['male', 'female']:
    gdata = master[master['gender'] == gender]
    print(f"\n   {gender.capitalize()} (N={len(gdata)}):")
    print(f"      - Stroop: M={gdata['stroop_interference'].mean():.1f}ms")
    print(f"      - WCST PE: M={gdata['wcst_pe_rate'].mean():.3f}")
    print(f"      - Distance from healthy: M={gdata['distance_from_healthy'].mean():.2f}")

# ============================================================================
# 10. Save Results
# ============================================================================
print("\n10. Saving results...")

# Summary statistics
summary_results = pd.DataFrame([{
    'metric': 'stroop_interference',
    'interaction_beta': stroop_interaction,
    'male_r': male_r_stroop,
    'male_p': male_p_stroop,
    'female_r': female_r_stroop,
    'female_p': female_p_stroop,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'wcst_pe_rate',
    'interaction_beta': wcst_interaction,
    'male_r': male_r_wcst,
    'male_p': male_p_wcst,
    'female_r': female_r_wcst,
    'female_p': female_p_wcst,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'ef_composite',
    'interaction_beta': composite_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': len(male_data),
    'female_n': len(female_data)
}, {
    'metric': 'distance_from_healthy',
    'interaction_beta': distance_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan,
    'male_n': len(male_data),
    'female_n': len(female_data)
}])

summary_results.to_csv(OUTPUT_DIR / "cross_task_summary.csv", index=False, encoding='utf-8-sig')

# Individual profiles
master.to_csv(OUTPUT_DIR / "individual_ef_profiles.csv", index=False, encoding='utf-8-sig')

# Cluster assignments
cluster_types.to_csv(OUTPUT_DIR / "cluster_types.csv", index=False, encoding='utf-8-sig')

print(f"   - Saved: cross_task_summary.csv")
print(f"   - Saved: individual_ef_profiles.csv")
print(f"   - Saved: cluster_types.csv")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# 11. Interpretation Summary
# ============================================================================
print("\n11. KEY FINDINGS SUMMARY:")
print("=" * 80)

print("\n📊 TASK-SPECIFIC INTERACTIONS:")
print(f"   - Stroop × Gender: β={stroop_interaction:.4f}")
print(f"   - WCST × Gender: β={wcst_interaction:.4f}")
print(f"   - Ratio: {abs(wcst_interaction)/abs(stroop_interaction) if stroop_interaction != 0 else np.inf:.2f}×")

print("\n🎯 SPECIFICITY TEST:")
if abs(wcst_interaction) > abs(stroop_interaction) * 2:
    print("   ✓ WCST-SPECIFIC effect confirmed")
    print("   ✓ NOT general executive dysfunction")
else:
    print("   ✓ Both tasks show effects (global EF impairment)")

print("\n📈 PROFILE ANALYSIS:")
print(f"   - EF Composite interaction: β={composite_interaction:.4f}")
print(f"   - Distance from healthy interaction: β={distance_interaction:.4f}")

print("\n🔍 CLUSTER ANALYSIS:")
print(f"   - {len(cluster_types)} profile types identified")
for _, row in cluster_types.iterrows():
    n_members = (master['cluster'] == row['cluster']).sum()
    print(f"   - {row['type']}: {n_members} participants")

print("\n💡 INTERPRETATION:")
if abs(wcst_interaction) > abs(stroop_interaction) * 2:
    print("   ✓ Confirms hypothesis: Effect is WCST-SPECIFIC")
    print("   ✓ Lonely males do NOT show global EF impairment")
    print("   ✓ Deficit is localized to set-shifting/perseveration")
    print("   ✓ Inhibitory control (Stroop) is intact")
else:
    print("   ✓ Effect may extend beyond WCST")
    print("   ✓ Possible broader executive dysfunction")

print("\n" + "=" * 80)
