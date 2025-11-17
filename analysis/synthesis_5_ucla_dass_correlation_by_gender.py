"""
Synthesis Analysis 5: UCLA-DASS Correlation by Gender
=======================================================

Purpose:
--------
Test "Hypothesis 2" from the comprehensive synthesis report:
Female loneliness may be more tightly coupled with depression/anxiety,
explaining why DASS control eliminates UCLA effects in females but not males.

Hypotheses:
-----------
H1: r(UCLA, DASS) is higher in females than males
H2: Higher correlation explains differential DASS control efficacy

Statistical Approach:
--------------------
- Pearson correlations for UCLA × DASS subscales by gender
- Fisher's z-transformation test for correlation differences
- Bootstrap 95% CI for correlations
- Visualization: Side-by-side correlation comparison

Output:
-------
- ucla_dass_correlations_by_gender.csv: Correlation coefficients and tests
- figure_ucla_dass_correlations.png: Bar plot comparison

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
print("SYNTHESIS ANALYSIS 5: UCLA-DASS CORRELATION BY GENDER")
print("=" * 80)
print()
print("Testing Hypothesis 2: Female UCLA-DASS correlation > Male")
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

# Required columns
required_cols = ['participant_id', 'ucla_total', 'gender_male',
                'dass_depression', 'dass_anxiety', 'dass_stress']

master_clean = master.dropna(subset=required_cols).copy()

# Compute DASS total
master_clean['dass_total'] = (master_clean['dass_depression'] +
                              master_clean['dass_anxiety'] +
                              master_clean['dass_stress'])

# Split by gender
females = master_clean[master_clean['gender_male'] == 0].copy()
males = master_clean[master_clean['gender_male'] == 1].copy()

print(f"  Total N = {len(master_clean)}")
print(f"    Females: {len(females)}")
print(f"    Males: {len(males)}")
print()

# ============================================================================
# CORRELATION ANALYSIS BY GENDER
# ============================================================================

print("[2/4] Computing correlations by gender...")

def fisher_z_test(r1, n1, r2, n2):
    """
    Test difference between two correlation coefficients using Fisher's z-transformation

    Returns:
        z_stat: z-statistic for difference
        p_value: two-tailed p-value
    """
    # Fisher's z-transformation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Standard error of difference
    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))

    # Z-statistic
    z_stat = (z1 - z2) / se_diff

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


def bootstrap_correlation_ci(x, y, n_bootstrap=10000, confidence_level=0.95):
    """Bootstrap confidence interval for Pearson correlation"""
    rng = np.random.default_rng(42)
    n = len(x)

    boot_corrs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        r_boot = np.corrcoef(x[indices], y[indices])[0, 1]
        boot_corrs.append(r_boot)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_corrs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_corrs, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


# DASS variables to test
dass_vars = [
    ('dass_depression', 'DASS Depression'),
    ('dass_anxiety', 'DASS Anxiety'),
    ('dass_stress', 'DASS Stress'),
    ('dass_total', 'DASS Total')
]

correlation_results = []

for dass_col, dass_label in dass_vars:
    print(f"\n  {dass_label}:")

    # Female correlation
    r_female, p_female = stats.pearsonr(females['ucla_total'], females[dass_col])
    ci_lower_f, ci_upper_f = bootstrap_correlation_ci(
        females['ucla_total'].values,
        females[dass_col].values
    )

    print(f"    Female: r={r_female:.3f}, p={p_female:.4f}, 95% CI=[{ci_lower_f:.3f}, {ci_upper_f:.3f}]")

    # Male correlation
    r_male, p_male = stats.pearsonr(males['ucla_total'], males[dass_col])
    ci_lower_m, ci_upper_m = bootstrap_correlation_ci(
        males['ucla_total'].values,
        males[dass_col].values
    )

    print(f"    Male:   r={r_male:.3f}, p={p_male:.4f}, 95% CI=[{ci_lower_m:.3f}, {ci_upper_m:.3f}]")

    # Fisher's z-test for difference
    z_stat, p_diff = fisher_z_test(r_female, len(females), r_male, len(males))

    # Interpretation
    if p_diff < 0.05:
        direction = "HIGHER" if r_female > r_male else "LOWER"
        print(f"    Difference: z={z_stat:.3f}, p={p_diff:.4f} * ({direction} in females)")
    else:
        print(f"    Difference: z={z_stat:.3f}, p={p_diff:.4f} (ns)")

    # Store results
    correlation_results.append({
        'dass_variable': dass_label,
        'female_r': r_female,
        'female_p': p_female,
        'female_n': len(females),
        'female_ci_lower': ci_lower_f,
        'female_ci_upper': ci_upper_f,
        'male_r': r_male,
        'male_p': p_male,
        'male_n': len(males),
        'male_ci_lower': ci_lower_m,
        'male_ci_upper': ci_upper_m,
        'fisher_z': z_stat,
        'difference_p': p_diff,
        'significant_difference': p_diff < 0.05,
        'higher_in_females': r_female > r_male
    })

# Save results
corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv(OUTPUT_DIR / "ucla_dass_correlations_by_gender.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: ucla_dass_correlations_by_gender.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[3/4] Creating visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(dass_vars))
width = 0.35

# Plot bars
female_bars = ax.bar(x_pos - width/2, corr_df['female_r'], width,
                     label='Female', color='#DE8F05', alpha=0.7,
                     edgecolor='black', linewidth=1.5)

male_bars = ax.bar(x_pos + width/2, corr_df['male_r'], width,
                   label='Male', color='#0173B2', alpha=0.7,
                   edgecolor='black', linewidth=1.5)

# Error bars (95% CI)
for i, row in corr_df.iterrows():
    # Female
    ax.errorbar(i - width/2, row['female_r'],
               yerr=[[row['female_r'] - row['female_ci_lower']],
                     [row['female_ci_upper'] - row['female_r']]],
               fmt='none', ecolor='black', capsize=5, capthick=2)

    # Male
    ax.errorbar(i + width/2, row['male_r'],
               yerr=[[row['male_r'] - row['male_ci_lower']],
                     [row['male_ci_upper'] - row['male_r']]],
               fmt='none', ecolor='black', capsize=5, capthick=2)

    # Significance markers for individual correlations
    # Female
    if row['female_p'] < 0.001:
        sig_f = '***'
    elif row['female_p'] < 0.01:
        sig_f = '**'
    elif row['female_p'] < 0.05:
        sig_f = '*'
    else:
        sig_f = ''

    if sig_f:
        ax.text(i - width/2, row['female_ci_upper'] + 0.03, sig_f,
               ha='center', fontsize=12, fontweight='bold')

    # Male
    if row['male_p'] < 0.001:
        sig_m = '***'
    elif row['male_p'] < 0.01:
        sig_m = '**'
    elif row['male_p'] < 0.05:
        sig_m = '*'
    else:
        sig_m = ''

    if sig_m:
        ax.text(i + width/2, row['male_ci_upper'] + 0.03, sig_m,
               ha='center', fontsize=12, fontweight='bold')

    # Difference significance (bracket)
    if row['significant_difference']:
        y_bracket = max(row['female_ci_upper'], row['male_ci_upper']) + 0.1
        ax.plot([i - width/2, i + width/2], [y_bracket, y_bracket],
               'k-', linewidth=2)
        ax.text(i, y_bracket + 0.05, '§',
               ha='center', fontsize=14, fontweight='bold')

# Styling
ax.set_xlabel('DASS Subscale', fontweight='bold', fontsize=12)
ax.set_ylabel('Pearson Correlation with UCLA Loneliness', fontweight='bold', fontsize=12)
ax.set_title('UCLA-DASS Correlations by Gender\n(Testing Hypothesis 2: Female > Male?)',
            fontweight='bold', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels([label for _, label in dass_vars], fontsize=11)
ax.legend(fontsize=11, loc='upper left', frameon=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add note
note_text = "* p<0.05, ** p<0.01, *** p<0.001 (individual correlations)\n§ Significant gender difference (Fisher's z-test, p<0.05)"
ax.text(0.98, 0.02, note_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_ucla_dass_correlations.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure_ucla_dass_correlations.png")
plt.close()

# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================

print("\n[4/4] Testing Hypothesis 2...")

# Test: Are female UCLA-DASS correlations consistently higher?
higher_in_females = corr_df['higher_in_females'].sum()
total_tests = len(corr_df)

print(f"\n  Correlations higher in females: {higher_in_females}/{total_tests}")

# Statistical test: Sign test (binomial)
# Null: Equal probability of being higher in females vs males
result = stats.binomtest(higher_in_females, total_tests, 0.5, alternative='greater')
p_sign_test = result.pvalue
print(f"  Sign test: p = {p_sign_test:.4f}")

# Count significant differences
sig_diffs = corr_df['significant_difference'].sum()
print(f"  Significant differences (Fisher's z): {sig_diffs}/{total_tests}")

# Interpretation
print("\n  HYPOTHESIS 2 VERDICT:")
if higher_in_females >= 3 and p_sign_test < 0.05:
    print("  ✓ SUPPORTED: Female UCLA-DASS correlations are consistently higher")
    print("    → This partially explains why DASS control eliminates UCLA effects in females")
elif higher_in_females >= 3:
    print("  ~ PARTIAL SUPPORT: Trend toward higher female correlations (not significant)")
    print("    → Consistent pattern but needs larger sample for confirmation")
else:
    print("  ✗ NOT SUPPORTED: No consistent pattern of higher female correlations")
    print("    → Hypothesis 2 does not explain gender moderation")

# Mean correlation difference
mean_female = corr_df['female_r'].mean()
mean_male = corr_df['male_r'].mean()
print(f"\n  Mean UCLA-DASS correlation:")
print(f"    Female: r = {mean_female:.3f}")
print(f"    Male:   r = {mean_male:.3f}")
print(f"    Difference: Δr = {mean_female - mean_male:.3f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SYNTHESIS ANALYSIS 5 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - ucla_dass_correlations_by_gender.csv")
print("  - figure_ucla_dass_correlations.png")
print()

# Print key finding
if sig_diffs > 0:
    print("KEY FINDING:")
    for _, row in corr_df[corr_df['significant_difference'] == True].iterrows():
        direction = "higher" if row['higher_in_females'] else "lower"
        print(f"  {row['dass_variable']}: Female r={row['female_r']:.3f} vs Male r={row['male_r']:.3f}")
        print(f"    → Significantly {direction} in females (p={row['difference_p']:.4f})")
else:
    print("KEY FINDING:")
    print("  No significant gender differences in UCLA-DASS correlations")
    print("  Hypothesis 2 (differential DASS coupling) NOT supported")
print()
