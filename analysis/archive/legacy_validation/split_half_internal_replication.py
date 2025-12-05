"""
Split-Half Internal Replication: UCLA × Gender → WCST PE

PURPOSE:
Address potential concerns about p-hacking and small sample sizes by demonstrating
that the main UCLA × Gender interaction effect on WCST perseverative errors
replicates in independent subsamples of the dataset.

DESIGN:
1. Randomly split N=88 into Discovery (N=44) and Validation (N=44)
2. Test UCLA × Gender → PE in both samples
3. Report: Do both show p<0.05? Are effect sizes similar?
4. Repeat with multiple random splits to assess robustness
5. Meta-analyze across splits for overall evidence

HYPOTHESIS:
The UCLA × Gender interaction will replicate (p<0.05) in both discovery
and validation samples, with consistent effect sizes (β ≈ 2-3).

This demonstrates the effect is not a statistical fluke or artifact of
specific participant combinations.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/robustness/split_half_replication")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
np.random.seed(42)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("SPLIT-HALF INTERNAL REPLICATION: UCLA × GENDER → WCST PE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

# Demographics
master = load_master_dataset(use_cache=True)
participants = master[['participant_id','gender_normalized','age']].rename(columns={'gender_normalized':'gender'})
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)
participants['gender_male'] = (participants['gender'] == 'male').astype(int)
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

ucla_data = master[['participant_id']].copy()
ucla_data['ucla_total'] = master['ucla_score'] if 'ucla_score' in master.columns else master.get('ucla_total')

# DASS scores (for covariate control) - MUST include individual subscales per CLAUDE.md
dass_data = master[['participant_id']].copy()
for src, dst in [('dass_anxiety', 'dass_anx'), ('dass_stress', 'dass_str'), ('dass_depression', 'dass_dep')]:
    if src in master.columns:
        dass_data[dst] = master[src]
# Compute z-scores for DASS subscales (required for DASS-controlled analysis)
for col in ['dass_dep', 'dass_anx', 'dass_str']:
    if col in dass_data.columns:
        dass_data[f'z_{col}'] = (dass_data[col] - dass_data[col].mean()) / dass_data[col].std()
dass_data = dass_data.dropna(subset=['dass_dep', 'dass_anx', 'dass_str'])

# WCST perseverative errors
wcst_data = master[['participant_id']].copy()
pe_col = None
for col in ['pe_rate', 'pe_rate', 'pe_rate', 'pe_rate_percent']:
    if col in master.columns:
        pe_col = col
        wcst_data[pe_col] = master[col]
        break

if pe_col is None:
    print(f"ERROR: No PE column found in master dataset. Columns: {master.columns.tolist()}")
    sys.exit(1)

wcst_data = wcst_data[['participant_id', pe_col]].copy()
wcst_data.columns = ['participant_id', 'pe_rate']

# ============================================================================
# 2. MERGE MASTER DATASET
# ============================================================================
print("\n[2/6] Merging datasets...")

master = ucla_data.merge(participants[['participant_id', 'gender_male', 'age']], on='participant_id', how='inner')
master = master.merge(wcst_data, on='participant_id', how='inner')
master = master.merge(dass_data, on='participant_id', how='left')

# Drop missing on key variables (including DASS controls per CLAUDE.md requirement)
master = master.dropna(subset=['ucla_total', 'pe_rate', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str'])

n_total = len(master)
n_male = (master['gender_male'] == 1).sum()
n_female = (master['gender_male'] == 0).sum()

print(f"  Total N = {n_total}")
print(f"    Males: {n_male}")
print(f"    Females: {n_female}")

if n_total < 40:
    print("ERROR: Sample too small for split-half replication")
    sys.exit(1)

# ============================================================================
# 3. SINGLE SPLIT-HALF TEST
# ============================================================================
print("\n[3/6] Running single split-half replication...")

def test_interaction(data, sample_name=""):
    """Test UCLA × Gender interaction in a sample with DASS-21 controls (per CLAUDE.md)"""

    # Model: PE ~ UCLA × Gender + DASS controls (+ age as covariate)
    # CRITICAL: Must include DASS-21 subscales to control for mood/anxiety confounding
    if 'age' in data.columns and 'z_dass_dep' in data.columns:
        formula = 'pe_rate ~ ucla_total * gender_male + z_dass_dep + z_dass_anx + z_dass_str + age'
    elif 'z_dass_dep' in data.columns:
        formula = 'pe_rate ~ ucla_total * gender_male + z_dass_dep + z_dass_anx + z_dass_str'
    elif 'age' in data.columns:
        formula = 'pe_rate ~ ucla_total * gender_male + age'
    else:
        formula = 'pe_rate ~ ucla_total * gender_male'

    model = smf.ols(formula, data=data).fit()

    # Extract interaction term
    interaction_coef = model.params['ucla_total:gender_male']
    interaction_se = model.bse['ucla_total:gender_male']
    interaction_t = model.tvalues['ucla_total:gender_male']
    interaction_p = model.pvalues['ucla_total:gender_male']

    # Effect size (partial eta squared for interaction)
    r_squared = model.rsquared

    # Simple slopes (with DASS controls per CLAUDE.md)
    males = data[data['gender_male'] == 1]
    females = data[data['gender_male'] == 0]

    # Determine simple slope formula based on available covariates
    if 'z_dass_dep' in data.columns:
        simple_formula = 'pe_rate ~ ucla_total + z_dass_dep + z_dass_anx + z_dass_str'
    else:
        simple_formula = 'pe_rate ~ ucla_total'

    if len(males) > 5:
        male_model = smf.ols(simple_formula, data=males).fit()
        male_slope = male_model.params['ucla_total']
        male_p = male_model.pvalues['ucla_total']
    else:
        male_slope, male_p = np.nan, np.nan

    if len(females) > 5:
        female_model = smf.ols(simple_formula, data=females).fit()
        female_slope = female_model.params['ucla_total']
        female_p = female_model.pvalues['ucla_total']
    else:
        female_slope, female_p = np.nan, np.nan

    result = {
        'sample': sample_name,
        'n_total': len(data),
        'n_male': (data['gender_male'] == 1).sum(),
        'n_female': (data['gender_male'] == 0).sum(),
        'interaction_beta': interaction_coef,
        'interaction_se': interaction_se,
        'interaction_t': interaction_t,
        'interaction_p': interaction_p,
        'r_squared': r_squared,
        'male_slope': male_slope,
        'male_p': male_p,
        'female_slope': female_slope,
        'female_p': female_p,
        'significant': 'Yes' if interaction_p < 0.05 else 'No'
    }

    return result

# Random split
np.random.seed(42)
shuffled_indices = np.random.permutation(len(master))
split_point = len(master) // 2

discovery_idx = shuffled_indices[:split_point]
validation_idx = shuffled_indices[split_point:]

discovery_sample = master.iloc[discovery_idx].copy()
validation_sample = master.iloc[validation_idx].copy()

print(f"\n  Discovery sample: N={len(discovery_sample)}")
print(f"    Males: {(discovery_sample['gender_male'] == 1).sum()}")
print(f"    Females: {(discovery_sample['gender_male'] == 0).sum()}")

print(f"\n  Validation sample: N={len(validation_sample)}")
print(f"    Males: {(validation_sample['gender_male'] == 1).sum()}")
print(f"    Females: {(validation_sample['gender_male'] == 0).sum()}")

# Test both samples
discovery_result = test_interaction(discovery_sample, "Discovery")
validation_result = test_interaction(validation_sample, "Validation")

print(f"\n  DISCOVERY:")
print(f"    Interaction: β={discovery_result['interaction_beta']:.3f}, SE={discovery_result['interaction_se']:.3f}")
print(f"    p={discovery_result['interaction_p']:.4f} ({discovery_result['significant']})")
print(f"    Male slope: β={discovery_result['male_slope']:.3f}, p={discovery_result['male_p']:.4f}")
print(f"    Female slope: β={discovery_result['female_slope']:.3f}, p={discovery_result['female_p']:.4f}")

print(f"\n  VALIDATION:")
print(f"    Interaction: β={validation_result['interaction_beta']:.3f}, SE={validation_result['interaction_se']:.3f}")
print(f"    p={validation_result['interaction_p']:.4f} ({validation_result['significant']})")
print(f"    Male slope: β={validation_result['male_slope']:.3f}, p={validation_result['male_p']:.4f}")
print(f"    Female slope: β={validation_result['female_slope']:.3f}, p={validation_result['female_p']:.4f}")

# Check if both replicated
both_sig = discovery_result['significant'] == 'Yes' and validation_result['significant'] == 'Yes'
print(f"\n  ✓ REPLICATION SUCCESS: {both_sig}")

# ============================================================================
# 4. MULTIPLE RANDOM SPLITS (ROBUSTNESS CHECK)
# ============================================================================
print("\n[4/6] Running multiple random splits (N=100)...")

n_splits = 100
split_results = []

for split_num in range(n_splits):
    np.random.seed(split_num)  # Different seed each time

    shuffled = np.random.permutation(len(master))
    split_pt = len(master) // 2

    disc_idx = shuffled[:split_pt]
    val_idx = shuffled[split_pt:]

    disc_data = master.iloc[disc_idx].copy()
    val_data = master.iloc[val_idx].copy()

    disc_res = test_interaction(disc_data, f"Discovery_{split_num}")
    val_res = test_interaction(val_data, f"Validation_{split_num}")

    split_results.append({
        'split': split_num,
        'discovery_beta': disc_res['interaction_beta'],
        'discovery_p': disc_res['interaction_p'],
        'discovery_sig': disc_res['significant'],
        'validation_beta': val_res['interaction_beta'],
        'validation_p': val_res['interaction_p'],
        'validation_sig': val_res['significant'],
        'both_sig': disc_res['significant'] == 'Yes' and val_res['significant'] == 'Yes',
        'beta_consistent': np.sign(disc_res['interaction_beta']) == np.sign(val_res['interaction_beta'])
    })

split_df = pd.DataFrame(split_results)

# Summary statistics
replication_rate = (split_df['both_sig'].sum() / n_splits) * 100
direction_consistency = (split_df['beta_consistent'].sum() / n_splits) * 100
discovery_sig_rate = (split_df['discovery_sig'] == 'Yes').sum() / n_splits * 100
validation_sig_rate = (split_df['validation_sig'] == 'Yes').sum() / n_splits * 100

print(f"\n  Replication Rate (both p<0.05): {replication_rate:.1f}%")
print(f"  Direction Consistency (same sign): {direction_consistency:.1f}%")
print(f"  Discovery significant: {discovery_sig_rate:.1f}%")
print(f"  Validation significant: {validation_sig_rate:.1f}%")

print(f"\n  Discovery β range: [{split_df['discovery_beta'].min():.3f}, {split_df['discovery_beta'].max():.3f}]")
print(f"  Validation β range: [{split_df['validation_beta'].min():.3f}, {split_df['validation_beta'].max():.3f}]")
print(f"  Discovery β mean: {split_df['discovery_beta'].mean():.3f} (SD={split_df['discovery_beta'].std():.3f})")
print(f"  Validation β mean: {split_df['validation_beta'].mean():.3f} (SD={split_df['validation_beta'].std():.3f})")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/6] Creating visualizations...")

# Plot 1: Distribution of p-values across splits
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Discovery p-values
axes[0, 0].hist(split_df['discovery_p'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
axes[0, 0].set_xlabel('p-value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title(f'Discovery Sample p-values (N splits={n_splits})')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Validation p-values
axes[0, 1].hist(split_df['validation_p'], bins=20, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
axes[0, 1].set_xlabel('p-value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Validation Sample p-values (N splits={n_splits})')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Beta coefficients scatter
axes[1, 0].scatter(split_df['discovery_beta'], split_df['validation_beta'],
                   alpha=0.6, s=50, c=split_df['both_sig'].map({True: 'green', False: 'gray'}))
axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].plot([split_df['discovery_beta'].min(), split_df['discovery_beta'].max()],
                [split_df['discovery_beta'].min(), split_df['discovery_beta'].max()],
                'r--', label='Perfect agreement')
axes[1, 0].set_xlabel('Discovery β')
axes[1, 0].set_ylabel('Validation β')
axes[1, 0].set_title('Effect Size Consistency Across Splits')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Replication rate bar chart
categories = ['Both Sig', 'Disc Only', 'Val Only', 'Neither']
counts = [
    (split_df['both_sig']).sum(),
    ((split_df['discovery_sig'] == 'Yes') & (split_df['validation_sig'] == 'No')).sum(),
    ((split_df['discovery_sig'] == 'No') & (split_df['validation_sig'] == 'Yes')).sum(),
    ((split_df['discovery_sig'] == 'No') & (split_df['validation_sig'] == 'No')).sum()
]
colors = ['green', 'steelblue', 'coral', 'gray']

axes[1, 1].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Number of Splits')
axes[1, 1].set_title('Replication Outcomes')
axes[1, 1].grid(axis='y', alpha=0.3)

for i, (cat, count) in enumerate(zip(categories, counts)):
    axes[1, 1].text(i, count + 1, f'{count}\n({count/n_splits*100:.0f}%)',
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "split_half_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Forest plot of effect sizes
fig, ax = plt.subplots(figsize=(10, 8))

# Show first 20 splits for clarity
n_show = min(20, n_splits)
y_positions = np.arange(n_show)

for i, row in split_df.head(n_show).iterrows():
    # Discovery
    ax.scatter(row['discovery_beta'], i - 0.1, color='steelblue', s=80, marker='s',
               alpha=0.7 if row['discovery_sig'] == 'Yes' else 0.3)
    # Validation
    ax.scatter(row['validation_beta'], i + 0.1, color='coral', s=80, marker='o',
               alpha=0.7 if row['validation_sig'] == 'Yes' else 0.3)

# Mean lines
ax.axvline(split_df['discovery_beta'].mean(), color='steelblue', linestyle='--',
           linewidth=2, label=f'Discovery mean: {split_df["discovery_beta"].mean():.3f}')
ax.axvline(split_df['validation_beta'].mean(), color='coral', linestyle='--',
           linewidth=2, label=f'Validation mean: {split_df["validation_beta"].mean():.3f}')
ax.axvline(0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_positions)
ax.set_yticklabels([f'Split {i+1}' for i in range(n_show)])
ax.set_xlabel('Interaction β (UCLA × Gender → PE)')
ax.set_title(f'Effect Size Consistency Across Random Splits (First {n_show} of {n_splits})')
ax.legend()
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "split_half_forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

# Save split results
split_df.to_csv(OUTPUT_DIR / "split_half_100_iterations.csv", index=False, encoding='utf-8-sig')

# Save main split
main_results = pd.DataFrame([discovery_result, validation_result])
main_results.to_csv(OUTPUT_DIR / "split_half_main_result.csv", index=False, encoding='utf-8-sig')

# Summary report
with open(OUTPUT_DIR / "SPLIT_HALF_REPLICATION_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SPLIT-HALF INTERNAL REPLICATION\n")
    f.write("UCLA × Gender → WCST Perseverative Errors\n")
    f.write("="*80 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-"*80 + "\n")
    f.write("Demonstrate that the UCLA × Gender interaction effect replicates in\n")
    f.write("independent subsamples, addressing concerns about p-hacking and small N.\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"Total N = {n_total}\n")
    f.write(f"  Males: {n_male}\n")
    f.write(f"  Females: {n_female}\n\n")

    f.write("MAIN SPLIT-HALF RESULT (Seed=42)\n")
    f.write("-"*80 + "\n\n")
    f.write(main_results.to_string(index=False))
    f.write("\n\n")

    f.write(f"Replication Success: {both_sig}\n")
    f.write(f"  Discovery p={discovery_result['interaction_p']:.4f} ({discovery_result['significant']})\n")
    f.write(f"  Validation p={validation_result['interaction_p']:.4f} ({validation_result['significant']})\n\n")

    f.write("ROBUSTNESS ACROSS 100 RANDOM SPLITS\n")
    f.write("-"*80 + "\n")
    f.write(f"Replication rate (both p<0.05): {replication_rate:.1f}%\n")
    f.write(f"Direction consistency (same sign): {direction_consistency:.1f}%\n")
    f.write(f"Discovery significant: {discovery_sig_rate:.1f}%\n")
    f.write(f"Validation significant: {validation_sig_rate:.1f}%\n\n")

    f.write("EFFECT SIZE CONSISTENCY\n")
    f.write("-"*80 + "\n")
    f.write(f"Discovery β: M={split_df['discovery_beta'].mean():.3f}, SD={split_df['discovery_beta'].std():.3f}\n")
    f.write(f"  Range: [{split_df['discovery_beta'].min():.3f}, {split_df['discovery_beta'].max():.3f}]\n")
    f.write(f"Validation β: M={split_df['validation_beta'].mean():.3f}, SD={split_df['validation_beta'].std():.3f}\n")
    f.write(f"  Range: [{split_df['validation_beta'].min():.3f}, {split_df['validation_beta'].max():.3f}]\n\n")

    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")

    if replication_rate >= 60:
        f.write(f"✓ STRONG REPLICATION: {replication_rate:.0f}% of splits showed p<0.05 in BOTH samples.\n")
        f.write("  This demonstrates the effect is robust and not due to chance sampling.\n\n")
    elif replication_rate >= 40:
        f.write(f"~ MODERATE REPLICATION: {replication_rate:.0f}% of splits showed p<0.05 in BOTH samples.\n")
        f.write("  Effect is present but sensitive to specific participant combinations.\n\n")
    else:
        f.write(f"✗ WEAK REPLICATION: Only {replication_rate:.0f}% of splits showed p<0.05 in BOTH samples.\n")
        f.write("  Effect may not be robust to subsample variation.\n\n")

    if direction_consistency >= 90:
        f.write(f"✓ EFFECT DIRECTION HIGHLY CONSISTENT: {direction_consistency:.0f}% same sign.\n\n")

    f.write("CONCLUSION\n")
    f.write("-"*80 + "\n")
    f.write("The UCLA × Gender → WCST PE interaction shows ")

    if replication_rate >= 60 and direction_consistency >= 90:
        f.write("STRONG internal replication.\n")
        f.write("This finding is unlikely to be a statistical artifact of the full sample.\n")
    elif replication_rate >= 40:
        f.write("MODERATE internal replication.\n")
        f.write("The effect is present but shows some sensitivity to sampling variation.\n")
    else:
        f.write("LIMITED internal replication.\n")
        f.write("Caution warranted in interpreting this effect.\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Full results saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ Split-Half Replication Analysis Complete!")
print("="*80)
print(f"\nKey Finding: {replication_rate:.0f}% replication rate across {n_splits} random splits")
print(f"  Effect direction consistent: {direction_consistency:.0f}%")
print(f"  Main result: Discovery p={discovery_result['interaction_p']:.4f}, Validation p={validation_result['interaction_p']:.4f}")
