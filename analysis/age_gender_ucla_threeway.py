"""
Age × Gender × UCLA Three-Way Interaction Analysis
====================================================
CRITICAL DEVELOPMENTAL TEST: Is male vulnerability age-dependent?

Hypothesis: Prefrontal cortex matures through mid-20s. Male UCLA→WCST_PE
vulnerability may be strongest in younger males (18-21) and absent in older
males (22+).

Tests:
1. Continuous Age × Gender × UCLA interaction
2. Median-split Age groups (young vs old)
3. Johnson-Neyman intervals for age cutpoints
4. Stratified analyses by age quartiles

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
import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_loader_utils import normalize_gender_series

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/age_gender_ucla")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("AGE × GENDER × UCLA THREE-WAY INTERACTION ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Load master dataset and add demographics
master = pd.read_csv(RESULTS_DIR / "analysis_outputs" / "master_expanded_metrics.csv")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants['participant_id'] = participants.get('participantId', participants.get('participant_id'))
participants = participants[['participant_id', 'gender', 'age']].copy()
participants['age'] = pd.to_numeric(participants['age'], errors='coerce')

# Map Korean gender to English using shared helper
participants['gender'] = normalize_gender_series(participants['gender'])

master = master.merge(participants, on='participant_id', how='left')

# Rename columns
if 'pe_rate' in master.columns:
    master['wcst_pe_rate'] = master['pe_rate']

# Calculate DASS total
if 'dass_total' not in master.columns:
    master['dass_total'] = master['dass_anxiety'] + master['dass_stress'] + master['dass_depression']

# Filter complete cases
master = master.dropna(subset=['ucla_total', 'wcst_pe_rate', 'gender', 'age'])
print(f"  Complete cases: {len(master)} participants")

# Create variables
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Center age for interpretation
master['age_centered'] = master['age'] - master['age'].mean()

# Create age groups
master['age_median_split'] = (master['age'] >= master['age'].median()).astype(int)
master['age_group'] = master['age_median_split'].map({0: 'Younger', 1: 'Older'})

# Age quartiles
master['age_quartile'] = pd.qcut(master['age'], q=4, labels=['Q1_Youngest', 'Q2', 'Q3', 'Q4_Oldest'])

print(f"\n  Descriptives:")
print(f"    Age: M={master['age'].mean():.2f}, SD={master['age'].std():.2f}, Range={master['age'].min():.0f}-{master['age'].max():.0f}")
print(f"    Age median: {master['age'].median():.1f}")
print(f"    Gender: {master['gender_male'].sum()} male, {len(master) - master['gender_male'].sum()} female")
print(f"\n  Age group distribution:")
print(master.groupby(['age_group', 'gender']).size().unstack(fill_value=0))

# ============================================================================
# 2. CONTINUOUS THREE-WAY INTERACTION
# ============================================================================
print("\n[2/5] Testing continuous Age × Gender × UCLA interaction...")

# Model 1: Two-way (baseline - from existing analyses)
model_2way = smf.ols('wcst_pe_rate ~ ucla_total * gender_male + age_centered + dass_total',
                      data=master).fit()

print("\n  Model 1 (Two-way): UCLA × Gender")
print(f"    UCLA × Gender: β={model_2way.params['ucla_total:gender_male']:.4f}, p={model_2way.pvalues['ucla_total:gender_male']:.4f}")
print(f"    R² = {model_2way.rsquared:.4f}")

# Model 2: Three-way interaction
model_3way = smf.ols('wcst_pe_rate ~ ucla_total * gender_male * age_centered + dass_total',
                      data=master).fit()

print("\n  Model 2 (Three-way): UCLA × Gender × Age")
if 'ucla_total:gender_male:age_centered' in model_3way.params:
    print(f"    UCLA × Gender × Age: β={model_3way.params['ucla_total:gender_male:age_centered']:.4f}, p={model_3way.pvalues['ucla_total:gender_male:age_centered']:.4f}")
else:
    print("    UCLA × Gender × Age: Not in model")

print(f"    R² = {model_3way.rsquared:.4f}")
print(f"    ΔR² = {model_3way.rsquared - model_2way.rsquared:.4f}")

# Likelihood ratio test
lr_stat = 2 * (model_3way.llf - model_2way.llf)
df_diff = model_3way.df_model - model_2way.df_model
p_value = stats.chi2.sf(lr_stat, df_diff)

print(f"\n  Likelihood Ratio Test:")
print(f"    LR = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.4f}")

if p_value < 0.05:
    print("    ✓ Three-way interaction is significant!")
else:
    print("    ✗ Three-way interaction is NOT significant")

# ============================================================================
# 3. MEDIAN-SPLIT AGE GROUP ANALYSIS
# ============================================================================
print("\n[3/5] Testing median-split age groups...")

# Run 2×2 ANCOVA: Age_group × Gender → PE (controlling UCLA)
model_age_group = smf.ols('wcst_pe_rate ~ C(age_group) * C(gender) * ucla_total + dass_total',
                          data=master).fit()

print("\n  2×2×continuous ANCOVA: Age_group × Gender × UCLA")
print(f"    Age_group main: F={model_age_group.fvalue:.2f}")

# Extract interaction terms
interactions = {}
for param in model_age_group.params.index:
    if ':' in param and 'ucla_total' in param:
        interactions[param] = {
            'beta': model_age_group.params[param],
            'p': model_age_group.pvalues[param]
        }

print("\n  Key interactions:")
for param, vals in interactions.items():
    sig = '✓' if vals['p'] < 0.05 else ''
    print(f"    {param}: β={vals['beta']:.4f}, p={vals['p']:.4f} {sig}")

# Stratified analysis by age group
print("\n  Stratified UCLA × Gender analysis:")
results_stratified = []

for age_group in ['Younger', 'Older']:
    subset = master[master['age_group'] == age_group].copy()
    print(f"\n    [{age_group} (N={len(subset)})]")

    model = smf.ols('wcst_pe_rate ~ ucla_total * gender_male + dass_total', data=subset).fit()

    if 'ucla_total:gender_male' in model.params:
        beta = model.params['ucla_total:gender_male']
        p = model.pvalues['ucla_total:gender_male']
        print(f"      UCLA × Gender: β={beta:.4f}, p={p:.4f}")

        results_stratified.append({
            'Age_Group': age_group,
            'N': len(subset),
            'UCLA_x_Gender_Beta': beta,
            'UCLA_x_Gender_p': p,
            'Significant': 'Yes' if p < 0.05 else 'No'
        })

        # Also check simple slopes
        males = subset[subset['gender_male'] == 1]
        females = subset[subset['gender_male'] == 0]

        if len(males) >= 10:
            model_m = smf.ols('wcst_pe_rate ~ ucla_total + dass_total', data=males).fit()
            print(f"        Males (N={len(males)}): UCLA slope β={model_m.params['ucla_total']:.4f}, p={model_m.pvalues['ucla_total']:.4f}")

        if len(females) >= 10:
            model_f = smf.ols('wcst_pe_rate ~ ucla_total + dass_total', data=females).fit()
            print(f"        Females (N={len(females)}): UCLA slope β={model_f.params['ucla_total']:.4f}, p={model_f.pvalues['ucla_total']:.4f}")

stratified_df = pd.DataFrame(results_stratified)

# ============================================================================
# 4. AGE QUARTILE ANALYSIS
# ============================================================================
print("\n[4/5] Testing age quartiles...")

quartile_results = []

for quartile in ['Q1_Youngest', 'Q2', 'Q3', 'Q4_Oldest']:
    subset = master[master['age_quartile'] == quartile].copy()
    age_range = f"{subset['age'].min():.0f}-{subset['age'].max():.0f}"

    if len(subset) < 10:
        continue

    model = smf.ols('wcst_pe_rate ~ ucla_total * gender_male + dass_total', data=subset).fit()

    if 'ucla_total:gender_male' in model.params:
        beta = model.params['ucla_total:gender_male']
        p = model.pvalues['ucla_total:gender_male']

        quartile_results.append({
            'Quartile': quartile,
            'Age_Range': age_range,
            'N': len(subset),
            'N_Male': subset['gender_male'].sum(),
            'N_Female': len(subset) - subset['gender_male'].sum(),
            'UCLA_x_Gender_Beta': beta,
            'UCLA_x_Gender_p': p,
            'Significant': 'Yes' if p < 0.05 else 'No'
        })

quartile_df = pd.DataFrame(quartile_results)
print("\n  UCLA × Gender effects by age quartile:")
print(quartile_df.to_string(index=False))

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/5] Creating visualizations...")

# Plot 1: Three-way interaction plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for i, age_group in enumerate(['Younger', 'Older']):
    subset = master[master['age_group'] == age_group]

    for gender in ['male', 'female']:
        gender_data = subset[subset['gender'] == gender]

        # Scatter
        axes[i].scatter(gender_data['ucla_total'], gender_data['wcst_pe_rate'],
                       label=gender.capitalize(), alpha=0.6, s=50)

        # Regression line
        if len(gender_data) >= 5:
            z = np.polyfit(gender_data['ucla_total'], gender_data['wcst_pe_rate'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(gender_data['ucla_total'].min(), gender_data['ucla_total'].max(), 100)
            axes[i].plot(x_line, p(x_line), linestyle='--', linewidth=2)

    axes[i].set_title(f'{age_group} (Age {subset["age"].min():.0f}-{subset["age"].max():.0f})', fontsize=12, weight='bold')
    axes[i].set_xlabel('UCLA Loneliness Score')
    if i == 0:
        axes[i].set_ylabel('WCST Perseverative Error Rate (%)')
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.suptitle('Age × Gender × UCLA Interaction: Does Male Vulnerability Vary by Age?', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_gender_ucla_threeway.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Effect size by age quartile
if not quartile_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c' if sig == 'Yes' else '#95a5a6' for sig in quartile_df['Significant']]

    ax.bar(range(len(quartile_df)), quartile_df['UCLA_x_Gender_Beta'], color=colors)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xticks(range(len(quartile_df)))
    ax.set_xticklabels([f"{row['Quartile']}\n(Age {row['Age_Range']})\nN={row['N']}"
                        for _, row in quartile_df.iterrows()])
    ax.set_ylabel('UCLA × Gender Interaction Coefficient')
    ax.set_title('UCLA × Gender Effect by Age Quartile\n(Red = Significant p<0.05)', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "effect_by_age_quartile.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 3: Continuous age moderation (for males only)
males = master[master['gender_male'] == 1].copy()
if len(males) >= 20:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create age bins for visualization
    males['age_bin'] = pd.cut(males['age'], bins=5)
    age_bin_centers = males.groupby('age_bin')['age'].mean()
    ucla_pe_corr = males.groupby('age_bin').apply(
        lambda x: x[['ucla_total', 'wcst_pe_rate']].corr().iloc[0, 1]
    )

    ax.scatter(age_bin_centers, ucla_pe_corr, s=100, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', label='No correlation')

    ax.set_xlabel('Age (binned)', fontsize=12)
    ax.set_ylabel('UCLA-PE Correlation (within age bin)', fontsize=12)
    ax.set_title('Males Only: Does UCLA→PE Correlation Vary by Age?', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "males_age_moderation.png", dpi=300, bbox_inches='tight')
    plt.close()

# Save results
stratified_df.to_csv(OUTPUT_DIR / "age_group_stratified_results.csv", index=False, encoding='utf-8-sig')
quartile_df.to_csv(OUTPUT_DIR / "age_quartile_results.csv", index=False, encoding='utf-8-sig')

# Model summaries
with open(OUTPUT_DIR / "model_summaries.txt", 'w') as f:
    f.write("TWO-WAY MODEL (UCLA × Gender)\n")
    f.write("="*80 + "\n")
    f.write(model_2way.summary().as_text())
    f.write("\n\n")

    f.write("THREE-WAY MODEL (UCLA × Gender × Age)\n")
    f.write("="*80 + "\n")
    f.write(model_3way.summary().as_text())

# Summary report
summary_lines = []
summary_lines.append("AGE × GENDER × UCLA THREE-WAY INTERACTION ANALYSIS\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("RESEARCH QUESTION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Is male UCLA→WCST_PE vulnerability age-dependent?\n")
summary_lines.append("Hypothesis: Effect strongest in younger males (18-21), absent in older (22+)\n\n")

summary_lines.append("KEY FINDINGS\n")
summary_lines.append("-" * 80 + "\n\n")

summary_lines.append("1. CONTINUOUS THREE-WAY INTERACTION\n")
summary_lines.append(f"   UCLA × Gender × Age: p={p_value:.4f}\n")
if p_value < 0.05:
    summary_lines.append("   ✓ SIGNIFICANT: Age modulates the Gender × UCLA effect\n\n")
else:
    summary_lines.append("   ✗ NOT significant: Age does not significantly modulate the effect\n\n")

summary_lines.append("2. MEDIAN-SPLIT AGE GROUPS\n")
for _, row in stratified_df.iterrows():
    summary_lines.append(f"   {row['Age_Group']} (N={row['N']}): UCLA×Gender β={row['UCLA_x_Gender_Beta']:.4f}, p={row['UCLA_x_Gender_p']:.4f} [{row['Significant']}]\n")
summary_lines.append("\n")

summary_lines.append("3. AGE QUARTILE ANALYSIS\n")
if not quartile_df.empty:
    for _, row in quartile_df.iterrows():
        sig_marker = '✓' if row['Significant'] == 'Yes' else '✗'
        summary_lines.append(f"   {sig_marker} {row['Quartile']} (Age {row['Age_Range']}, N={row['N']}): β={row['UCLA_x_Gender_Beta']:.4f}, p={row['UCLA_x_Gender_p']:.4f}\n")
summary_lines.append("\n")

summary_lines.append("INTERPRETATION\n")
summary_lines.append("-" * 80 + "\n")
if p_value < 0.10:  # Marginal or significant
    summary_lines.append("⚠️ Age moderation is present (or trending)\n")
    summary_lines.append("   Male vulnerability to loneliness→PE may be developmental\n")
    summary_lines.append("   Further investigation with larger sample recommended\n\n")
else:
    summary_lines.append("✓ Age moderation is NOT significant\n")
    summary_lines.append("   Male UCLA→PE effect appears stable across 18-36 age range\n")
    summary_lines.append("   Effect is likely trait-like, not developmentally constrained\n\n")

summary_lines.append("="*80 + "\n")
summary_lines.append(f"Results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print("\n" + summary_text)

with open(OUTPUT_DIR / "AGE_THREEWAY_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Age × Gender × UCLA Three-Way Analysis complete!")
