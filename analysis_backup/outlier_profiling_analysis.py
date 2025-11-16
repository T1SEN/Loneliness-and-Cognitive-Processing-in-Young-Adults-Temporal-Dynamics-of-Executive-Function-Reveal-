"""
Outlier Profiling & Case Studies

OBJECTIVE:
Identify and characterize extreme responders (outliers) on key variables:
1. HIGH PE + HIGH UCLA (extreme vulnerable males?)
2. HIGH UCLA + LOW PE (resilient individuals?)
3. Statistical outliers (>3SD from mean)

PURPOSE:
- Understand individual differences beyond group averages
- Identify protective/risk factors in extreme cases
- Generate hypotheses for case-control studies
- Clinical translation (who needs intervention most?)

ANALYSES:
1. Identify outliers on PE, UCLA, composite vulnerability
2. Profile outliers: demographics, DASS, all EF metrics
3. Compare extreme groups (vulnerable vs resilient)
4. Generate individual case narratives
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/outlier_profiling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*80)
print("OUTLIER PROFILING & CASE STUDIES")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Load master
master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if not master_path.exists():
    print("ERROR: master_dataset.csv not found")
    sys.exit(1)

master = pd.read_csv(master_path, encoding='utf-8-sig')

# Load participants
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

# Normalize gender
gender_map = {'남성': 'male', '여성': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Find PE column
for col in ['pe_rate', 'perseverative_error_rate']:
    if col in master.columns:
        master = master.rename(columns={col: 'pe_rate'})
        break

# Clean
master = master.dropna(subset=['ucla_total', 'pe_rate', 'gender_male'])

print(f"  Total N={len(master)}")
print(f"    Males: {(master['gender_male'] == 1).sum()}")
print(f"    Females: {(master['gender_male'] == 0).sum()}")

# ============================================================================
# 2. IDENTIFY OUTLIERS
# ============================================================================
print("\n[2/5] Identifying outliers...")

# Method 1: Statistical outliers (>3SD)
master['pe_z'] = (master['pe_rate'] - master['pe_rate'].mean()) / master['pe_rate'].std()
master['ucla_z'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()

stat_outliers_pe = master[np.abs(master['pe_z']) > 3]
stat_outliers_ucla = master[np.abs(master['ucla_z']) > 3]

print(f"\nStatistical outliers (>3SD):")
print(f"  PE rate: {len(stat_outliers_pe)} participants")
print(f"  UCLA: {len(stat_outliers_ucla)} participants")

# Method 2: Extreme groups (top/bottom quartiles)
pe_q1 = master['pe_rate'].quantile(0.25)
pe_q3 = master['pe_rate'].quantile(0.75)
ucla_q1 = master['ucla_total'].quantile(0.25)
ucla_q3 = master['ucla_total'].quantile(0.75)

# Define groups
master['group'] = 'Average'
master.loc[(master['ucla_total'] > ucla_q3) & (master['pe_rate'] > pe_q3), 'group'] = 'Vulnerable'  # High loneliness + High PE
master.loc[(master['ucla_total'] > ucla_q3) & (master['pe_rate'] < pe_q1), 'group'] = 'Resilient'   # High loneliness + Low PE
master.loc[(master['ucla_total'] < ucla_q1) & (master['pe_rate'] < pe_q1), 'group'] = 'Protected'   # Low loneliness + Low PE
master.loc[(master['ucla_total'] < ucla_q1) & (master['pe_rate'] > pe_q3), 'group'] = 'Unexpected'  # Low loneliness + High PE

group_counts = master['group'].value_counts()

print(f"\nExtreme groups:")
for group, count in group_counts.items():
    print(f"  {group}: {count} participants")

# ============================================================================
# 3. PROFILE OUTLIER GROUPS
# ============================================================================
print("\n[3/5] Profiling extreme groups...")

# Focus on Vulnerable vs Resilient (both high UCLA)
vulnerable = master[master['group'] == 'Vulnerable']
resilient = master[master['group'] == 'Resilient']

print(f"\nVULNERABLE (High UCLA + High PE): N={len(vulnerable)}")
if len(vulnerable) > 0:
    print(f"  Gender: {(vulnerable['gender_male'] == 1).sum()} male, {(vulnerable['gender_male'] == 0).sum()} female")
    if 'age' in vulnerable.columns:
        print(f"  Age: M={vulnerable['age'].mean():.1f}, SD={vulnerable['age'].std():.1f}")
    if 'dass_total' in vulnerable.columns:
        print(f"  DASS Total: M={vulnerable['dass_total'].mean():.1f}, SD={vulnerable['dass_total'].std():.1f}")
    print(f"  UCLA: M={vulnerable['ucla_total'].mean():.1f}, SD={vulnerable['ucla_total'].std():.1f}")
    print(f"  PE Rate: M={vulnerable['pe_rate'].mean():.1f}%, SD={vulnerable['pe_rate'].std():.1f}%")

print(f"\nRESILIENT (High UCLA + Low PE): N={len(resilient)}")
if len(resilient) > 0:
    print(f"  Gender: {(resilient['gender_male'] == 1).sum()} male, {(resilient['gender_male'] == 0).sum()} female")
    if 'age' in resilient.columns:
        print(f"  Age: M={resilient['age'].mean():.1f}, SD={resilient['age'].std():.1f}")
    if 'dass_total' in resilient.columns:
        print(f"  DASS Total: M={resilient['dass_total'].mean():.1f}, SD={resilient['dass_total'].std():.1f}")
    print(f"  UCLA: M={resilient['ucla_total'].mean():.1f}, SD={resilient['ucla_total'].std():.1f}")
    print(f"  PE Rate: M={resilient['pe_rate'].mean():.1f}%, SD={resilient['pe_rate'].std():.1f}%")

# Compare groups
if len(vulnerable) > 0 and len(resilient) > 0:
    print(f"\nVULNERABLE vs RESILIENT comparison:")

    # Gender distribution
    vuln_male_pct = (vulnerable['gender_male'] == 1).sum() / len(vulnerable) * 100
    res_male_pct = (resilient['gender_male'] == 1).sum() / len(resilient) * 100
    print(f"  % Male: Vulnerable={vuln_male_pct:.0f}%, Resilient={res_male_pct:.0f}%")

    # DASS
    if 'dass_total' in master.columns:
        vuln_dass = vulnerable['dass_total'].dropna()
        res_dass = resilient['dass_total'].dropna()
        if len(vuln_dass) > 0 and len(res_dass) > 0:
            t_dass, p_dass = stats.ttest_ind(vuln_dass, res_dass)
            print(f"  DASS difference: t={t_dass:.2f}, p={p_dass:.4f}")

# ============================================================================
# 4. INDIVIDUAL CASE NARRATIVES
# ============================================================================
print("\n[4/5] Generating individual case narratives...")

# Select top 3 most extreme cases
if len(vulnerable) >= 3:
    top_vulnerable = vulnerable.nlargest(3, 'pe_rate')

    print("\nTOP 3 MOST VULNERABLE CASES:")
    for i, (idx, row) in enumerate(top_vulnerable.iterrows(), 1):
        print(f"\n  Case {i} (ID: {row['participant_id']}):")
        print(f"    Gender: {row['gender']}")
        if 'age' in row:
            print(f"    Age: {row['age']:.0f}")
        print(f"    UCLA: {row['ucla_total']:.0f} (z={row['ucla_z']:.2f})")
        print(f"    PE Rate: {row['pe_rate']:.1f}% (z={row['pe_z']:.2f})")
        if 'dass_total' in row and not pd.isna(row['dass_total']):
            print(f"    DASS: {row['dass_total']:.0f}")
        if 'stroop_interference' in row and not pd.isna(row['stroop_interference']):
            print(f"    Stroop Interference: {row['stroop_interference']:.0f} ms")
        if 'prp_bottleneck' in row and not pd.isna(row['prp_bottleneck']):
            print(f"    PRP Bottleneck: {row['prp_bottleneck']:.0f} ms")

if len(resilient) >= 3:
    top_resilient = resilient.nsmallest(3, 'pe_rate')

    print("\nTOP 3 MOST RESILIENT CASES:")
    for i, (idx, row) in enumerate(top_resilient.iterrows(), 1):
        print(f"\n  Case {i} (ID: {row['participant_id']}):")
        print(f"    Gender: {row['gender']}")
        if 'age' in row:
            print(f"    Age: {row['age']:.0f}")
        print(f"    UCLA: {row['ucla_total']:.0f} (z={row['ucla_z']:.2f})")
        print(f"    PE Rate: {row['pe_rate']:.1f}% (z={row['pe_z']:.2f})")
        if 'dass_total' in row and not pd.isna(row['dass_total']):
            print(f"    DASS: {row['dass_total']:.0f}")
        if 'stroop_interference' in row and not pd.isna(row['stroop_interference']):
            print(f"    Stroop Interference: {row['stroop_interference']:.0f} ms")
        if 'prp_bottleneck' in row and not pd.isna(row['prp_bottleneck']):
            print(f"    PRP Bottleneck: {row['prp_bottleneck']:.0f} ms")

# ============================================================================
# 5. VISUALIZATIONS & SAVE
# ============================================================================
print("\n[5/5] Creating visualizations and saving results...")

# Plot: UCLA × PE with outlier groups highlighted
fig, ax = plt.subplots(figsize=(10, 8))

# Color by group
colors = {
    'Vulnerable': '#E74C3C',
    'Resilient': '#27AE60',
    'Protected': '#3498DB',
    'Unexpected': '#F39C12',
    'Average': '#95A5A6'
}

for group, color in colors.items():
    group_data = master[master['group'] == group]
    if len(group_data) > 0:
        ax.scatter(group_data['ucla_total'], group_data['pe_rate'],
                  c=color, label=group, alpha=0.7, s=100, edgecolors='black')

# Add reference lines for quartiles
ax.axhline(pe_q1, color='gray', linestyle='--', alpha=0.5, label='PE Q1/Q3')
ax.axhline(pe_q3, color='gray', linestyle='--', alpha=0.5)
ax.axvline(ucla_q1, color='gray', linestyle=':', alpha=0.5, label='UCLA Q1/Q3')
ax.axvline(ucla_q3, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
ax.set_ylabel('WCST PE Rate (%)', fontweight='bold')
ax.set_title('Outlier Groups: UCLA × PE', fontweight='bold', pad=15)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "outlier_groups_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

# Save outlier profiles
master[['participant_id', 'gender', 'age', 'ucla_total', 'pe_rate', 'group', 'ucla_z', 'pe_z']].to_csv(
    OUTPUT_DIR / "outlier_profiles.csv", index=False, encoding='utf-8-sig'
)

# Save extreme cases
if len(vulnerable) > 0:
    vulnerable.to_csv(OUTPUT_DIR / "vulnerable_cases.csv", index=False, encoding='utf-8-sig')
if len(resilient) > 0:
    resilient.to_csv(OUTPUT_DIR / "resilient_cases.csv", index=False, encoding='utf-8-sig')

# Report
with open(OUTPUT_DIR / "OUTLIER_PROFILING_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("OUTLIER PROFILING & CASE STUDIES\n")
    f.write("="*80 + "\n\n")

    f.write("OBJECTIVE\n")
    f.write("-"*80 + "\n")
    f.write("Identify and characterize extreme responders to understand individual\n")
    f.write("differences beyond group averages.\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"Total N={len(master)}\n\n")

    f.write("EXTREME GROUPS\n")
    f.write("-"*80 + "\n\n")
    f.write(master['group'].value_counts().to_string())
    f.write("\n\n")

    if len(vulnerable) > 0:
        f.write("VULNERABLE PROFILE (High UCLA + High PE)\n")
        f.write("-"*80 + "\n")
        f.write(f"N={len(vulnerable)}\n")
        f.write(f"Gender: {(vulnerable['gender_male'] == 1).sum()} male, {(vulnerable['gender_male'] == 0).sum()} female\n")
        f.write(f"UCLA: M={vulnerable['ucla_total'].mean():.1f}, SD={vulnerable['ucla_total'].std():.1f}\n")
        f.write(f"PE Rate: M={vulnerable['pe_rate'].mean():.1f}%, SD={vulnerable['pe_rate'].std():.1f}%\n\n")

    if len(resilient) > 0:
        f.write("RESILIENT PROFILE (High UCLA + Low PE)\n")
        f.write("-"*80 + "\n")
        f.write(f"N={len(resilient)}\n")
        f.write(f"Gender: {(resilient['gender_male'] == 1).sum()} male, {(resilient['gender_male'] == 0).sum()} female\n")
        f.write(f"UCLA: M={resilient['ucla_total'].mean():.1f}, SD={resilient['ucla_total'].std():.1f}\n")
        f.write(f"PE Rate: M={resilient['pe_rate'].mean():.1f}%, SD={resilient['pe_rate'].std():.1f}%\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-"*80 + "\n")
    if len(vulnerable) > 0 and len(resilient) > 0:
        vuln_male_pct = (vulnerable['gender_male'] == 1).sum() / len(vulnerable) * 100
        res_male_pct = (resilient['gender_male'] == 1).sum() / len(resilient) * 100

        f.write(f"Gender composition:\n")
        f.write(f"  Vulnerable group: {vuln_male_pct:.0f}% male\n")
        f.write(f"  Resilient group: {res_male_pct:.0f}% male\n\n")

        if vuln_male_pct > 60:
            f.write("✓ Vulnerable group predominantly MALE\n")
        if res_male_pct < 40:
            f.write("✓ Resilient group predominantly FEMALE\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Full profiles saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ Outlier Profiling Complete!")
print("="*80)

print(f"\nExtreme Groups:")
for group, count in group_counts.items():
    print(f"  {group}: {count}")

if len(vulnerable) > 0:
    vuln_male_pct = (vulnerable['gender_male'] == 1).sum() / len(vulnerable) * 100
    print(f"\nVulnerable group: {vuln_male_pct:.0f}% male")
if len(resilient) > 0:
    res_male_pct = (resilient['gender_male'] == 1).sum() / len(resilient) * 100
    print(f"Resilient group: {res_male_pct:.0f}% female")
