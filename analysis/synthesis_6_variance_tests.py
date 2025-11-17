"""
Synthesis Analysis 6: Variance Homogeneity Tests
==================================================

Purpose:
--------
Validate the statistical assumptions for UCLA × Gender interaction tests.
Unequal group sizes (Female N=46, Male N=30) require verification that
variance homogeneity assumption is not violated.

Statistical Approach:
--------------------
- Levene's test for equality of variances (robust to non-normality)
- Brown-Forsythe test (more robust alternative)
- Variance ratio (F-max test)
- Visual inspection via box plots and variance ratio plots

If violated: Recommend Welch correction or bootstrap methods

Output:
-------
- variance_homogeneity_tests.csv: Test statistics and p-values
- figure_variance_diagnostics.png: Box plots by gender

Author: Research Team
Date: 2025-01-17
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
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/synthesis_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SYNTHESIS ANALYSIS 6: VARIANCE HOMOGENEITY TESTS")
print("=" * 80)
print()
print("Validating assumptions for UCLA × Gender interaction tests")
print()

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("[1/4] Loading data...")

master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if not master_path.exists():
    print("ERROR: master_dataset.csv not found")
    sys.exit(1)

master = pd.read_csv(master_path, encoding='utf-8-sig')

# Normalize columns
if 'participantId' in master.columns:
    master = master.rename(columns={'participantId': 'participant_id'})

# Gender coding
if 'gender' not in master.columns and 'gender_male' in master.columns:
    master['gender'] = master['gender_male'].map({1: 'male', 0: 'female'})
elif 'gender' in master.columns:
    gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
    master['gender'] = master['gender'].map(gender_map)
    master['gender_male'] = (master['gender'] == 'male').astype(int)

# Rename PE if needed
for col in ['pe_rate', 'perseverative_error_rate']:
    if col in master.columns and col != 'pe_rate':
        master = master.rename(columns={col: 'pe_rate'})
        break

# Required columns
required_cols = ['participant_id', 'gender_male']
master_clean = master.dropna(subset=required_cols).copy()

# Split by gender
females = master_clean[master_clean['gender_male'] == 0].copy()
males = master_clean[master_clean['gender_male'] == 1].copy()

print(f"  Total N = {len(master_clean)}")
print(f"    Females: {len(females)}")
print(f"    Males: {len(males)}")
print()

# ============================================================================
# VARIANCE HOMOGENEITY TESTS
# ============================================================================

print("[2/4] Testing variance homogeneity...")

# Outcomes to test
outcomes = [
    ('pe_rate', 'WCST PE Rate'),
    ('stroop_interference', 'Stroop Interference'),
    ('prp_bottleneck', 'PRP Bottleneck'),
    ('ucla_total', 'UCLA Loneliness'),
]

variance_results = []

for outcome_col, outcome_label in outcomes:
    if outcome_col not in master_clean.columns:
        print(f"  ⚠ Skipping {outcome_label} (column not found)")
        continue

    # Drop missing for this outcome
    df = master_clean.dropna(subset=[outcome_col]).copy()
    female_vals = df[df['gender_male'] == 0][outcome_col].values
    male_vals = df[df['gender_male'] == 1][outcome_col].values

    if len(female_vals) < 10 or len(male_vals) < 10:
        print(f"  ⚠ Skipping {outcome_label}: insufficient data")
        continue

    print(f"\n  {outcome_label}:")
    print(f"    N: Female={len(female_vals)}, Male={len(male_vals)}")

    # Descriptive statistics
    female_mean = np.mean(female_vals)
    female_sd = np.std(female_vals, ddof=1)
    male_mean = np.mean(male_vals)
    male_sd = np.std(male_vals, ddof=1)

    print(f"    Female: M={female_mean:.2f}, SD={female_sd:.2f}")
    print(f"    Male:   M={male_mean:.2f}, SD={male_sd:.2f}")

    # Variance ratio
    var_ratio = (female_sd ** 2) / (male_sd ** 2)
    print(f"    Variance ratio (F/M): {var_ratio:.3f}")

    # Levene's test (median-based, robust to non-normality)
    levene_stat, levene_p = stats.levene(female_vals, male_vals, center='median')
    print(f"    Levene's test: F={levene_stat:.3f}, p={levene_p:.4f}")

    # Brown-Forsythe test (trimmed mean, more robust)
    bf_stat, bf_p = stats.levene(female_vals, male_vals, center='trimmed')
    print(f"    Brown-Forsythe: F={bf_stat:.3f}, p={bf_p:.4f}")

    # F-max test (variance ratio test)
    # Critical value: F(df1, df2, α=0.05)
    df_female = len(female_vals) - 1
    df_male = len(male_vals) - 1
    f_critical = stats.f.ppf(0.975, df_female, df_male)  # Two-tailed at α=0.05

    f_max = max(var_ratio, 1/var_ratio)
    f_max_p = 2 * min(stats.f.cdf(var_ratio, df_female, df_male),
                     1 - stats.f.cdf(var_ratio, df_female, df_male))

    print(f"    F-max test: F={f_max:.3f}, p={f_max_p:.4f}")

    # Interpretation
    if levene_p < 0.05 or bf_p < 0.05:
        print(f"    ⚠ WARNING: Variance heterogeneity detected (p<0.05)")
        print(f"       → Interaction tests may have inflated Type I error")
        print(f"       → Recommend: Welch correction or bootstrap methods")
        homogeneous = False
    else:
        print(f"    ✓ Variance homogeneity assumption satisfied")
        homogeneous = True

    # Store results
    variance_results.append({
        'outcome': outcome_label,
        'outcome_col': outcome_col,
        'n_female': len(female_vals),
        'n_male': len(male_vals),
        'female_mean': female_mean,
        'female_sd': female_sd,
        'male_mean': male_mean,
        'male_sd': male_sd,
        'variance_ratio': var_ratio,
        'levene_stat': levene_stat,
        'levene_p': levene_p,
        'brown_forsythe_stat': bf_stat,
        'brown_forsythe_p': bf_p,
        'f_max': f_max,
        'f_max_p': f_max_p,
        'f_critical_005': f_critical,
        'homogeneous': homogeneous
    })

# Save results
var_df = pd.DataFrame(variance_results)
var_df.to_csv(OUTPUT_DIR / "variance_homogeneity_tests.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: variance_homogeneity_tests.csv")

# ============================================================================
# VISUALIZATION: BOX PLOTS
# ============================================================================

print("\n[3/4] Creating diagnostic plots...")

# Create subplots
n_outcomes = len(variance_results)
fig, axes = plt.subplots(1, n_outcomes, figsize=(5*n_outcomes, 6))

if n_outcomes == 1:
    axes = [axes]

for idx, result in enumerate(variance_results):
    ax = axes[idx]
    outcome_col = result['outcome_col']
    outcome_label = result['outcome']

    # Prepare data
    df_plot = master_clean.dropna(subset=[outcome_col]).copy()
    df_plot['Gender'] = df_plot['gender_male'].map({0: 'Female', 1: 'Male'})

    # Box plot
    bp = ax.boxplot(
        [df_plot[df_plot['Gender'] == 'Female'][outcome_col].values,
         df_plot[df_plot['Gender'] == 'Male'][outcome_col].values],
        labels=['Female', 'Male'],
        patch_artist=True,
        widths=0.6
    )

    # Color boxes
    colors = ['#DE8F05', '#0173B2']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points (jittered)
    for gender_idx, gender in enumerate(['Female', 'Male']):
        gender_data = df_plot[df_plot['Gender'] == gender][outcome_col].values
        y = gender_data
        x = np.random.normal(gender_idx + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.3, s=30, color=colors[gender_idx], edgecolor='black', linewidth=0.5)

    # Styling
    ax.set_ylabel(outcome_label, fontweight='bold', fontsize=11)
    ax.set_title(f"{outcome_label}\n" +
                f"Levene p={result['levene_p']:.3f} " +
                f"({'Homogeneous' if result['homogeneous'] else 'Heterogeneous'})",
                fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add variance info
    var_text = f"SD_F={result['female_sd']:.2f}\nSD_M={result['male_sd']:.2f}\nRatio={result['variance_ratio']:.2f}"
    ax.text(0.05, 0.95, var_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_variance_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure_variance_diagnostics.png")
plt.close()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n[4/4] Summary and recommendations...")

violations = var_df[var_df['homogeneous'] == False]

print(f"\n  Total outcomes tested: {len(var_df)}")
print(f"  Variance homogeneity violations: {len(violations)}/{len(var_df)}")

if len(violations) > 0:
    print("\n  OUTCOMES WITH HETEROGENEOUS VARIANCES:")
    for _, row in violations.iterrows():
        print(f"    • {row['outcome']}:")
        print(f"        Levene p = {row['levene_p']:.4f}")
        print(f"        Variance ratio = {row['variance_ratio']:.3f}")
        print(f"        Recommendation: Use Welch-corrected t-test or bootstrap")
else:
    print("\n  ✓ ALL OUTCOMES: Variance homogeneity assumptions satisfied")
    print("    → Standard interaction tests are valid")

# Specific recommendations for interaction tests
print("\n  RECOMMENDATIONS FOR INTERACTION TESTS:")

for _, row in var_df.iterrows():
    if row['outcome_col'] in ['pe_rate', 'prp_bottleneck', 'stroop_interference']:
        if row['homogeneous']:
            print(f"    {row['outcome']}: Standard OLS regression ✓")
        else:
            print(f"    {row['outcome']}: Use robust standard errors (HC3) or bootstrap ⚠")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 80)
print("SYNTHESIS ANALYSIS 6 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - variance_homogeneity_tests.csv")
print("  - figure_variance_diagnostics.png")
print()
