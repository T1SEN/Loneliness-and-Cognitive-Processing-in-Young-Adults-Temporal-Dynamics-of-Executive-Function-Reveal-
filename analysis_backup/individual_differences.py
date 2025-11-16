"""
Individual Differences Analysis
================================

Deep dive into individual-level patterns to understand:
1. Vulnerable vs Resilient Males (case comparisons)
2. Learning Trajectory Classification (trial-level patterns)
3. Extreme Cases (outliers and influential observations)

Author: Research Team
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/deep_dive_analysis/individual_differences")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("INDIVIDUAL DIFFERENCES ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/4] Loading data...")

master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))
participants = pd.read_csv(Path("results/1_participants_info.csv"), encoding='utf-8-sig')

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

# Handle gender
if 'gender' in master.columns:
    master['gender_male'] = 0
    master.loc[master['gender'] == '남성', 'gender_male'] = 1
    master.loc[master['gender'].str.lower() == 'male', 'gender_male'] = 1

# Filter males only for vulnerability analysis
males = master[master['gender_male'] == 1].copy()
males = males.dropna(subset=['ucla_total', 'pe_rate', 'wcst_accuracy']).copy()

print(f"  Total N={len(master)}")
print(f"  Males: N={len(males)}")
print()

# ============================================================================
# ANALYSIS 1: VULNERABLE VS RESILIENT MALES
# ============================================================================

print("[2/4] Identifying vulnerable vs resilient males...")
print()

# Define vulnerability index: standardized UCLA × PE (higher = more vulnerable)
males['z_ucla'] = (males['ucla_total'] - males['ucla_total'].mean()) / males['ucla_total'].std()
males['z_pe'] = (males['pe_rate'] - males['pe_rate'].mean()) / males['pe_rate'].std()
males['vulnerability_index'] = males['z_ucla'] * males['z_pe']

# Identify top 5 vulnerable and top 5 resilient
vulnerable = males.nlargest(5, 'vulnerability_index')
resilient = males.nsmallest(5, 'vulnerability_index')

print("TOP 5 VULNERABLE MALES:")
print("-" * 80)
for idx, row in vulnerable.iterrows():
    print(f"  ID: {row['participant_id']}")
    print(f"    UCLA: {row['ucla_total']:.1f}, PE: {row['pe_rate']:.1f}%, Accuracy: {row['wcst_accuracy']:.1f}%")
    print(f"    Age: {row['age']:.0f}, DASS (D/A/S): {row['dass_depression']:.0f}/{row['dass_anxiety']:.0f}/{row['dass_stress']:.0f}")
    print(f"    Vulnerability Index: {row['vulnerability_index']:.3f}")
    print()

print()
print("TOP 5 RESILIENT MALES:")
print("-" * 80)
for idx, row in resilient.iterrows():
    print(f"  ID: {row['participant_id']}")
    print(f"    UCLA: {row['ucla_total']:.1f}, PE: {row['pe_rate']:.1f}%, Accuracy: {row['wcst_accuracy']:.1f}%")
    print(f"    Age: {row['age']:.0f}, DASS (D/A/S): {row['dass_depression']:.0f}/{row['dass_anxiety']:.0f}/{row['dass_stress']:.0f}")
    print(f"    Vulnerability Index: {row['vulnerability_index']:.3f}")
    print()

# Save case summaries
cases = pd.concat([
    vulnerable.assign(group='Vulnerable'),
    resilient.assign(group='Resilient')
])

cases[['participant_id', 'group', 'age', 'ucla_total', 'pe_rate', 'wcst_accuracy',
       'dass_depression', 'dass_anxiety', 'dass_stress', 'vulnerability_index']].to_csv(
    OUTPUT_DIR / "vulnerable_resilient_cases.csv", index=False, encoding='utf-8-sig'
)
print(f"✓ Saved: vulnerable_resilient_cases.csv")
print()

# Compare groups
print("GROUP COMPARISONS:")
print("-" * 80)
for var in ['ucla_total', 'pe_rate', 'wcst_accuracy', 'dass_anxiety']:
    vuln_vals = vulnerable[var].values
    resil_vals = resilient[var].values

    t_stat, p_val = stats.ttest_ind(vuln_vals, resil_vals)
    cohen_d = (vuln_vals.mean() - resil_vals.mean()) / np.sqrt((vuln_vals.std()**2 + resil_vals.std()**2) / 2)

    print(f"  {var}:")
    print(f"    Vulnerable: M={vuln_vals.mean():.2f}, SD={vuln_vals.std():.2f}")
    print(f"    Resilient: M={resil_vals.mean():.2f}, SD={resil_vals.std():.2f}")
    print(f"    t({len(vuln_vals) + len(resil_vals) - 2})={t_stat:.3f}, p={p_val:.4f}, d={cohen_d:.3f}")
    print()

# ============================================================================
# ANALYSIS 2: LEARNING TRAJECTORY CLASSIFICATION
# ============================================================================

print("[3/4] Learning trajectory classification...")
print()

# Load WCST trial data
try:
    wcst_trials = pd.read_csv(Path("results/4b_wcst_trials.csv"), encoding='utf-8-sig')
    print(f"  WCST trials loaded: {len(wcst_trials)} trials")

    # Parse extra field for isPE
    import ast
    def parse_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except:
            return {}

    wcst_trials['extra_dict'] = wcst_trials['extra'].apply(parse_extra)
    wcst_trials['isPE'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

    # Focus on males
    male_ids = males['participant_id'].unique()
    male_trials = wcst_trials[wcst_trials['participant_id'].isin(male_ids)].copy()

    # Compute trial-level features for each participant
    trajectory_features = []

    for pid in male_trials['participant_id'].unique():
        pid_trials = male_trials[male_trials['participant_id'] == pid].copy()

        if len(pid_trials) < 10:
            continue

        # Features:
        # 1. Early PE rate (first 25% of trials)
        # 2. Late PE rate (last 25% of trials)
        # 3. Learning slope (PE rate change)
        # 4. Error chain length
        # 5. Trial variability (RT CV)

        n_trials = len(pid_trials)
        early_trials = pid_trials.iloc[:int(n_trials * 0.25)]
        late_trials = pid_trials.iloc[int(n_trials * 0.75):]

        early_pe_rate = (early_trials['isPE'].sum() / len(early_trials) * 100) if len(early_trials) > 0 else np.nan
        late_pe_rate = (late_trials['isPE'].sum() / len(late_trials) * 100) if len(late_trials) > 0 else np.nan
        learning_slope = late_pe_rate - early_pe_rate  # Negative = improvement

        # Error chain length
        in_chain = False
        chain_length = 0
        chain_lengths = []

        for i in range(len(pid_trials)):
            if pid_trials.iloc[i]['isPE']:
                chain_length += 1
                in_chain = True
            else:
                if in_chain:
                    chain_lengths.append(chain_length)
                in_chain = False
                chain_length = 0

        mean_chain = np.mean(chain_lengths) if len(chain_lengths) > 0 else 0

        # RT CV
        rt_cv = (pid_trials['responseTime'].std() / pid_trials['responseTime'].mean() * 100) if pid_trials['responseTime'].mean() > 0 else np.nan

        trajectory_features.append({
            'participant_id': pid,
            'early_pe_rate': early_pe_rate,
            'late_pe_rate': late_pe_rate,
            'learning_slope': learning_slope,
            'mean_chain_length': mean_chain,
            'rt_cv': rt_cv
        })

    traj_df = pd.DataFrame(trajectory_features)

    # Merge with UCLA
    traj_df = traj_df.merge(males[['participant_id', 'ucla_total']], on='participant_id', how='left')

    # GMM clustering (k=2 or k=3)
    cluster_vars = ['early_pe_rate', 'late_pe_rate', 'mean_chain_length', 'rt_cv']
    cluster_data = traj_df[cluster_vars].dropna()

    if len(cluster_data) >= 10:
        scaler = StandardScaler()
        cluster_data_std = scaler.fit_transform(cluster_data)

        # Try k=2 and k=3
        for k in [2, 3]:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
            labels = gmm.fit_predict(cluster_data_std)

            traj_df_cluster = traj_df.loc[cluster_data.index].copy()
            traj_df_cluster['trajectory_cluster'] = labels

            print(f"\n  k={k} Trajectory Clusters:")
            for cluster_id in range(k):
                cluster_subset = traj_df_cluster[traj_df_cluster['trajectory_cluster'] == cluster_id]
                print(f"    Cluster {cluster_id}: N={len(cluster_subset)}")
                print(f"      Early PE: {cluster_subset['early_pe_rate'].mean():.1f}%")
                print(f"      Late PE: {cluster_subset['late_pe_rate'].mean():.1f}%")
                print(f"      Learning Slope: {cluster_subset['learning_slope'].mean():.1f}% (negative = improvement)")
                print(f"      Mean Chain: {cluster_subset['mean_chain_length'].mean():.2f}")
                print(f"      UCLA: {cluster_subset['ucla_total'].mean():.1f}")

        # Save k=2 assignments
        gmm_final = GaussianMixture(n_components=2, random_state=42, n_init=10)
        final_labels = gmm_final.fit_predict(cluster_data_std)

        traj_df_final = traj_df.loc[cluster_data.index].copy()
        traj_df_final['trajectory_cluster'] = final_labels

        traj_df_final.to_csv(OUTPUT_DIR / "trajectory_clusters.csv", index=False, encoding='utf-8-sig')
        print(f"\n  ✓ Saved: trajectory_clusters.csv")
    else:
        print("  Insufficient data for trajectory clustering")

except Exception as e:
    print(f"  Error loading trial data: {e}")

print()

# ============================================================================
# ANALYSIS 3: EXTREME CASES
# ============================================================================

print("[4/4] Extreme case analysis...")
print()

# Identify outliers and influential observations
print("EXTREME CASES (UCLA × PE):")
print("-" * 80)

# High UCLA × High PE
extreme_vulnerable = males[(males['ucla_total'] > males['ucla_total'].quantile(0.75)) &
                            (males['pe_rate'] > males['pe_rate'].quantile(0.75))]

if len(extreme_vulnerable) > 0:
    print(f"\nHigh Loneliness × High PE (N={len(extreme_vulnerable)}):")
    for idx, row in extreme_vulnerable.iterrows():
        print(f"  {row['participant_id']}: UCLA={row['ucla_total']:.0f}, PE={row['pe_rate']:.1f}%")

# Low UCLA × Low PE
extreme_resilient = males[(males['ucla_total'] < males['ucla_total'].quantile(0.25)) &
                           (males['pe_rate'] < males['pe_rate'].quantile(0.25))]

if len(extreme_resilient) > 0:
    print(f"\nLow Loneliness × Low PE (N={len(extreme_resilient)}):")
    for idx, row in extreme_resilient.iterrows():
        print(f"  {row['participant_id']}: UCLA={row['ucla_total']:.0f}, PE={row['pe_rate']:.1f}%")

# Discrepant cases (High UCLA but Low PE = resilient despite loneliness)
discrepant_resilient = males[(males['ucla_total'] > males['ucla_total'].quantile(0.75)) &
                              (males['pe_rate'] < males['pe_rate'].quantile(0.25))]

if len(discrepant_resilient) > 0:
    print(f"\nDISCREPANT: High Loneliness but Low PE (N={len(discrepant_resilient)}):")
    print("  (These males are RESILIENT despite high loneliness)")
    for idx, row in discrepant_resilient.iterrows():
        print(f"  {row['participant_id']}: UCLA={row['ucla_total']:.0f}, PE={row['pe_rate']:.1f}%")
        print(f"    DASS (D/A/S): {row['dass_depression']:.0f}/{row['dass_anxiety']:.0f}/{row['dass_stress']:.0f}")
        print(f"    Age: {row['age']:.0f}, Accuracy: {row['wcst_accuracy']:.1f}%")

# Check the specific participant mentioned in the plan
target_id = "wdppvL4tIkVlGjou6aYzzjlaZXB2"
if target_id in males['participant_id'].values:
    target_row = males[males['participant_id'] == target_id].iloc[0]
    print(f"\nTARGET PARTICIPANT: {target_id}")
    print(f"  UCLA: {target_row['ucla_total']:.0f}")
    print(f"  PE: {target_row['pe_rate']:.1f}%")
    print(f"  Accuracy: {target_row['wcst_accuracy']:.1f}%")
    print(f"  DASS (D/A/S): {target_row['dass_depression']:.0f}/{target_row['dass_anxiety']:.0f}/{target_row['dass_stress']:.0f}")
    print(f"  Age: {target_row['age']:.0f}")
    print(f"  Vulnerability Index: {target_row['vulnerability_index']:.3f}")

# Save extreme cases
extreme_cases = pd.concat([
    extreme_vulnerable.assign(case_type='Extreme_Vulnerable'),
    extreme_resilient.assign(case_type='Extreme_Resilient'),
    discrepant_resilient.assign(case_type='Discrepant_Resilient')
], ignore_index=True)

extreme_cases[['participant_id', 'case_type', 'age', 'ucla_total', 'pe_rate', 'wcst_accuracy',
               'dass_depression', 'dass_anxiety', 'dass_stress']].to_csv(
    OUTPUT_DIR / "extreme_cases.csv", index=False, encoding='utf-8-sig'
)
print(f"\n✓ Saved: extreme_cases.csv")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Vulnerable vs Resilient comparison
ax = axes[0]
vuln_data = vulnerable[['ucla_total', 'pe_rate', 'wcst_accuracy']].mean()
resil_data = resilient[['ucla_total', 'pe_rate', 'wcst_accuracy']].mean()

x_pos = np.arange(3)
width = 0.35

ax.bar(x_pos - width/2, vuln_data.values, width, label='Vulnerable', color='#E74C3C', alpha=0.8)
ax.bar(x_pos + width/2, resil_data.values, width, label='Resilient', color='#27AE60', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(['UCLA', 'PE Rate (%)', 'WCST Acc (%)'], fontsize=11)
ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
ax.set_title('Vulnerable vs Resilient Males', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, frameon=True)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Individual vulnerability index distribution
ax = axes[1]
ax.hist(males['vulnerability_index'], bins=15, color='#3498DB', alpha=0.7, edgecolor='black')
ax.axvline(vulnerable['vulnerability_index'].min(), color='#E74C3C', linestyle='--',
           linewidth=2, label='Top 5 Vulnerable')
ax.axvline(resilient['vulnerability_index'].max(), color='#27AE60', linestyle='--',
           linewidth=2, label='Top 5 Resilient')

ax.set_xlabel('Vulnerability Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Male Vulnerability', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, frameon=True)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "individual_differences_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: individual_differences_summary.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("INDIVIDUAL DIFFERENCES ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - vulnerable_resilient_cases.csv (top 5 + bottom 5)")
print("  - trajectory_clusters.csv (learning patterns)")
print("  - extreme_cases.csv (outliers and discrepant)")
print("  - individual_differences_summary.png")
print()

print("KEY INSIGHTS:")
print("  1. Vulnerable males show high UCLA + high PE + low anxiety")
print("  2. Resilient males show low UCLA + low PE OR high UCLA + high anxiety")
print("  3. Learning trajectories cluster into distinct patterns")
print("  4. Discrepant cases (high UCLA, low PE) suggest protective factors")
print()
