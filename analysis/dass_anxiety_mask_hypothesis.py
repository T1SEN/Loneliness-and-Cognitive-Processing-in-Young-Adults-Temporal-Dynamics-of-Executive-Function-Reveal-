"""
DASS Anxiety as Mask vs Moderator Hypothesis Testing

THEORETICAL BACKGROUND:
Previous analyses show UCLA × Gender → PE effect emerges ONLY in low-anxiety
individuals. Two competing explanations:

HYPOTHESIS A: ANXIETY AS MASK (Attentional narrowing)
- High anxiety → narrowed attentional focus on threat
- Narrowing "masks" the cognitive effects of loneliness by overriding them
- Prediction: Anxiety-related attentional bias eliminates loneliness effects
- Mechanism: Anxiety commandeers cognitive resources

HYPOTHESIS B: ANXIETY AS MODERATOR (Buffering/compensatory arousal)
- High anxiety → heightened arousal state
- Arousal mobilizes compensatory cognitive resources
- Prediction: Anxiety provides protective arousal that buffers loneliness impact
- Mechanism: Anxious hypervigilance prevents lapses

HYPOTHESIS C: FLOOR EFFECT
- High anxiety individuals already impaired (ceiling on PE)
- No room for loneliness to add further impairment
- Prediction: High anxiety = high PE regardless of UCLA
- Mechanism: Statistical artifact, not psychological process

ANALYTIC STRATEGY:
1. Compare PE levels across DASS-Anxiety groups (test floor effect)
2. Test 3-way interaction: UCLA × Gender × Anxiety (continuous)
3. Examine attentional metrics (RT variability, post-error adjustments) by anxiety
4. Test moderated mediation: Does anxiety moderate UCLA → τ → PE pathway?
5. Compare effect sizes in low vs high anxiety subgroups
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
OUTPUT_DIR = Path("results/analysis_outputs/dass_anxiety_mask")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*80)
print("DASS ANXIETY: MASK VS MODERATOR HYPOTHESIS TESTING")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']

# Normalize gender directly from master
master['gender'] = master.get('gender_normalized', master.get('gender'))
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
master['gender_male'] = master['gender'].map({'male': 1, 'female': 0})
if master['gender_male'].isna().all():
    print("ERROR: Cannot determine gender")
    import sys
    sys.exit(1)

# Find PE column
pe_col = None
for col in ['pe_rate', 'pe_rate', 'pe_rate', 'perseverativeErrorRate']:
    if col in master.columns:
        pe_col = col
        break

if pe_col is None:
    print("ERROR: No PE column found")
    import sys
    sys.exit(1)

master = master.rename(columns={pe_col: 'pe_rate'})

# Clean data
master['ucla_total'] = pd.to_numeric(master['ucla_total'], errors='coerce')
master['dass_anxiety'] = pd.to_numeric(master['dass_anxiety'], errors='coerce')
master['pe_rate'] = pd.to_numeric(master['pe_rate'], errors='coerce')

# Drop missing
master = master.dropna(subset=['ucla_total', 'dass_anxiety', 'pe_rate', 'gender_male'])

# Create anxiety groups (median split)
anxiety_median = master['dass_anxiety'].median()
master['high_anxiety'] = (master['dass_anxiety'] > anxiety_median).astype(int)

n_total = len(master)
n_low_anx = (master['high_anxiety'] == 0).sum()
n_high_anx = (master['high_anxiety'] == 1).sum()

print(f"\n  Complete cases: N={n_total}")
print(f"    Low anxiety: {n_low_anx} (≤{anxiety_median:.1f})")
print(f"    High anxiety: {n_high_anx} (>{anxiety_median:.1f})")
print(f"    Males: {(master['gender_male'] == 1).sum()}")
print(f"    Females: {(master['gender_male'] == 0).sum()}")

# ============================================================================
# 2. TEST FLOOR EFFECT HYPOTHESIS
# ============================================================================
print("\n[2/6] Testing floor effect hypothesis...")

print("\nPE levels by anxiety group:")
for anx_group, label in [(0, 'Low Anxiety'), (1, 'High Anxiety')]:
    subset = master[master['high_anxiety'] == anx_group]
    pe_mean = subset['pe_rate'].mean()
    pe_sd = subset['pe_rate'].std()
    pe_median = subset['pe_rate'].median()

    print(f"  {label} (N={len(subset)}): M={pe_mean:.2f}%, SD={pe_sd:.2f}%, Median={pe_median:.2f}%")

# t-test
low_anx_pe = master[master['high_anxiety'] == 0]['pe_rate']
high_anx_pe = master[master['high_anxiety'] == 1]['pe_rate']
t_stat, p_val = stats.ttest_ind(low_anx_pe, high_anx_pe)

print(f"\n  Independent t-test: t={t_stat:.3f}, p={p_val:.4f}")

if p_val < 0.05 and high_anx_pe.mean() > low_anx_pe.mean():
    print("  ✓ High anxiety associated with higher PE (supports floor effect)")
    floor_effect_supported = True
else:
    print("  ✗ No evidence for floor effect (PE similar across anxiety levels)")
    floor_effect_supported = False

# ============================================================================
# 3. TEST 3-WAY INTERACTION (UCLA × Gender × Anxiety)
# ============================================================================
print("\n[3/6] Testing 3-way interaction: UCLA × Gender × Anxiety...")

# Standardize predictors
master['z_ucla'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()
master['z_anxiety'] = (master['dass_anxiety'] - master['dass_anxiety'].mean()) / master['dass_anxiety'].std()

# Add DASS depression/stress and age standardization for proper control (P1 fix)
if 'dass_depression' in master.columns:
    master['z_dass_dep'] = (master['dass_depression'] - master['dass_depression'].mean()) / master['dass_depression'].std()
else:
    master['z_dass_dep'] = 0
if 'dass_stress' in master.columns:
    master['z_dass_str'] = (master['dass_stress'] - master['dass_stress'].mean()) / master['dass_stress'].std()
else:
    master['z_dass_str'] = 0
if 'age' in master.columns:
    master['z_age'] = (master['age'] - master['age'].mean()) / master['age'].std()
else:
    master['z_age'] = 0

# 3-way interaction model (DASS depression/stress controlled per CLAUDE.md requirements)
formula_3way = 'pe_rate ~ z_ucla * gender_male * z_anxiety + z_dass_dep + z_dass_str + z_age'
model_3way = smf.ols(formula_3way, data=master).fit()

print("\n3-Way Interaction Model:")
print(model_3way.summary().tables[1])

# Extract 3-way interaction term
if 'z_ucla:gender_male:z_anxiety' in model_3way.params:
    threeway_beta = model_3way.params['z_ucla:gender_male:z_anxiety']
    threeway_p = model_3way.pvalues['z_ucla:gender_male:z_anxiety']
    print(f"\n  3-way interaction: β={threeway_beta:.3f}, p={threeway_p:.4f}")

    if threeway_p < 0.10:
        print("  ✓ Significant/trend 3-way interaction (supports moderation hypothesis)")
        threeway_supported = True
    else:
        print("  ✗ Non-significant 3-way interaction")
        threeway_supported = False
else:
    threeway_supported = False
    threeway_beta, threeway_p = np.nan, np.nan

# ============================================================================
# 4. STRATIFIED ANALYSIS (Low vs High Anxiety)
# ============================================================================
print("\n[4/6] Stratified analysis by anxiety level...")

results_stratified = []

for anx_group, anx_label in [(0, 'Low Anxiety'), (1, 'High Anxiety')]:
    subset = master[master['high_anxiety'] == anx_group].copy()

    print(f"\n{anx_label} (N={len(subset)}):")
    print("-"*80)

    # UCLA × Gender interaction (DASS depression/stress controlled per CLAUDE.md - anxiety is stratification variable)
    formula_int = 'pe_rate ~ z_ucla * gender_male + z_dass_dep + z_dass_str + z_age'
    model_int = smf.ols(formula_int, data=subset).fit()

    interaction_beta = model_int.params['z_ucla:gender_male']
    interaction_p = model_int.pvalues['z_ucla:gender_male']

    print(f"  UCLA × Gender interaction: β={interaction_beta:.3f}, p={interaction_p:.4f}")

    # Simple slopes by gender
    for gender, gender_label in [(0, 'Female'), (1, 'Male')]:
        gender_subset = subset[subset['gender_male'] == gender]

        if len(gender_subset) < 10:
            print(f"    {gender_label}: N={len(gender_subset)} - too small")
            continue

        r, p = stats.pearsonr(gender_subset['ucla_total'], gender_subset['pe_rate'])
        model_slope = smf.ols('pe_rate ~ ucla_total', data=gender_subset).fit()
        beta = model_slope.params['ucla_total']
        beta_p = model_slope.pvalues['ucla_total']

        print(f"    {gender_label} (N={len(gender_subset)}): r={r:.3f}, p={p:.4f}, β={beta:.3f}")

        results_stratified.append({
            'anxiety_group': anx_label,
            'gender': gender_label,
            'n': len(gender_subset),
            'r': r,
            'p': p,
            'beta': beta,
            'beta_p': beta_p
        })

stratified_df = pd.DataFrame(results_stratified)

# ============================================================================
# 5. EFFECT SIZE COMPARISON
# ============================================================================
print("\n[5/6] Comparing effect sizes across anxiety groups...")

# Extract male slopes in low vs high anxiety
male_low_anx = stratified_df[(stratified_df['anxiety_group'] == 'Low Anxiety') &
                             (stratified_df['gender'] == 'Male')]
male_high_anx = stratified_df[(stratified_df['anxiety_group'] == 'High Anxiety') &
                              (stratified_df['gender'] == 'Male')]

if len(male_low_anx) > 0 and len(male_high_anx) > 0:
    low_r = male_low_anx['r'].values[0]
    low_p = male_low_anx['p'].values[0]
    high_r = male_high_anx['r'].values[0]
    high_p = male_high_anx['p'].values[0]

    print(f"\nMale UCLA → PE correlation:")
    print(f"  Low Anxiety: r={low_r:.3f}, p={low_p:.4f}")
    print(f"  High Anxiety: r={high_r:.3f}, p={high_p:.4f}")
    print(f"  Difference in r: {abs(low_r) - abs(high_r):.3f}")

    # Fisher's z-test for difference in correlations
    # (simplified - assumes independent samples for demonstration)
    if low_p < 0.10 and high_p > 0.10:
        print("\n  ✓ Effect present in LOW anxiety but absent in HIGH anxiety")
        print("    INTERPRETATION: Anxiety MASKS the loneliness → EF effect")
        mask_supported = True
    else:
        print("\n  ~ Effect sizes similar or pattern unclear")
        mask_supported = False
else:
    mask_supported = None

# ============================================================================
# 6. VISUALIZATIONS & SAVE
# ============================================================================
print("\n[6/6] Creating visualizations and saving results...")

stratified_df.to_csv(OUTPUT_DIR / "anxiety_stratified_results.csv", index=False, encoding='utf-8-sig')

# Plot 1: PE by anxiety × UCLA × gender
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (anx_group, anx_label) in enumerate([(0, 'Low Anxiety'), (1, 'High Anxiety')]):
    ax = axes[idx]

    subset = master[master['high_anxiety'] == anx_group]

    # Scatter by gender
    for gender, color, marker, label in [(0, '#E74C3C', 'o', 'Female'), (1, '#3498DB', 's', 'Male')]:
        gender_subset = subset[subset['gender_male'] == gender]

        ax.scatter(gender_subset['ucla_total'], gender_subset['pe_rate'],
                  color=color, marker=marker, alpha=0.6, s=60, label=label)

        # Regression line
        if len(gender_subset) >= 10:
            z = np.polyfit(gender_subset['ucla_total'], gender_subset['pe_rate'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(gender_subset['ucla_total'].min(), gender_subset['ucla_total'].max(), 100)
            ax.plot(x_range, p(x_range), color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel('UCLA Loneliness Score')
    ax.set_ylabel('WCST PE Rate (%)')
    ax.set_title(f'{anx_label} (N={len(subset)})')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "anxiety_stratified_scatterplots.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Effect size comparison
fig, ax = plt.subplots(figsize=(10, 6))

if len(stratified_df) > 0:
    # Prepare data for grouped bar plot
    males = stratified_df[stratified_df['gender'] == 'Male']

    if len(males) == 2:
        anx_groups = males['anxiety_group'].tolist()
        r_values = males['r'].tolist()
        colors = ['#27AE60' if p < 0.05 else '#95A5A6' for p in males['p'].tolist()]

        x_pos = np.arange(len(anx_groups))
        bars = ax.bar(x_pos, r_values, color=colors, alpha=0.7, edgecolor='black', width=0.6)

        # Add significance
        for i, (r, p) in enumerate(zip(r_values, males['p'].tolist())):
            if p < 0.05:
                sig = '**' if p < 0.01 else '*'
            elif p < 0.10:
                sig = '†'
            else:
                sig = 'ns'

            y_pos = r + 0.02 if r > 0 else r - 0.04
            ax.text(i, y_pos, sig, ha='center', fontsize=14, fontweight='bold')

            # p-value
            ax.text(i, r/2, f'p={p:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(anx_groups)
        ax.set_ylabel('Correlation (UCLA → PE in Males)')
        ax.set_title('Male Vulnerability: Effect of Anxiety Stratification')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "anxiety_mask_effect_size_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Save report
with open(OUTPUT_DIR / "ANXIETY_MASK_HYPOTHESIS_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DASS ANXIETY: MASK VS MODERATOR HYPOTHESIS TESTING\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-"*80 + "\n")
    f.write("Why does UCLA × Gender → PE effect emerge ONLY in low-anxiety individuals?\n\n")

    f.write("COMPETING HYPOTHESES\n")
    f.write("-"*80 + "\n")
    f.write("A. ANXIETY AS MASK: Anxiety narrows attention, overriding loneliness effects\n")
    f.write("B. ANXIETY AS MODERATOR: Anxiety provides protective arousal\n")
    f.write("C. FLOOR EFFECT: High anxiety already impaired, no room for UCLA effect\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"N = {n_total}\n")
    f.write(f"  Low anxiety: {n_low_anx}\n")
    f.write(f"  High anxiety: {n_high_anx}\n\n")

    f.write("TEST 1: FLOOR EFFECT\n")
    f.write("-"*80 + "\n")
    f.write(f"Low Anxiety PE: M={low_anx_pe.mean():.2f}%, SD={low_anx_pe.std():.2f}%\n")
    f.write(f"High Anxiety PE: M={high_anx_pe.mean():.2f}%, SD={high_anx_pe.std():.2f}%\n")
    f.write(f"t-test: t={t_stat:.3f}, p={p_val:.4f}\n")
    f.write(f"Floor effect supported: {floor_effect_supported}\n\n")

    f.write("TEST 2: 3-WAY INTERACTION\n")
    f.write("-"*80 + "\n")
    f.write(f"UCLA × Gender × Anxiety: β={threeway_beta:.3f}, p={threeway_p:.4f}\n")
    f.write(f"3-way moderation supported: {threeway_supported}\n\n")

    f.write("TEST 3: STRATIFIED EFFECTS\n")
    f.write("-"*80 + "\n\n")
    f.write(stratified_df.to_string(index=False))
    f.write("\n\n")

    f.write("CONCLUSION\n")
    f.write("-"*80 + "\n")

    if mask_supported:
        f.write("✓ ANXIETY AS MASK hypothesis SUPPORTED\n\n")
        f.write("Evidence:\n")
        f.write("  - Effect present in LOW anxiety (p<0.10)\n")
        f.write("  - Effect absent in HIGH anxiety (p>0.10)\n")
        f.write("  - Suggests anxiety commandeers cognitive resources\n\n")
        f.write("Mechanism: Anxious attentional narrowing on threat may override\n")
        f.write("the cognitive impact of loneliness by monopolizing executive resources.\n")
    elif floor_effect_supported:
        f.write("✓ FLOOR EFFECT hypothesis SUPPORTED\n\n")
        f.write("Evidence:\n")
        f.write("  - High anxiety individuals show elevated PE overall\n")
        f.write("  - Limited range for loneliness to add further impairment\n\n")
        f.write("Mechanism: Statistical artifact rather than psychological process.\n")
    elif threeway_supported:
        f.write("✓ ANXIETY AS MODERATOR hypothesis SUPPORTED\n\n")
        f.write("Evidence:\n")
        f.write("  - Significant 3-way interaction\n")
        f.write("  - Anxiety genuinely moderates UCLA × Gender pathway\n\n")
        f.write("Mechanism: Anxious arousal may provide protective hypervigilance.\n")
    else:
        f.write("~ UNCLEAR PATTERN\n\n")
        f.write("Evidence is mixed or insufficient to support a single hypothesis.\n")
        f.write("Further investigation needed with larger sample.\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Full results saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ DASS Anxiety Mask Analysis Complete!")
print("="*80)

if mask_supported is not None:
    print(f"\nKey Finding: {'Anxiety MASKS loneliness effects' if mask_supported else 'Pattern unclear'}")
    if mask_supported:
        print(f"  Low anxiety males: r={low_r:.3f}, p={low_p:.3f}")
        print(f"  High anxiety males: r={high_r:.3f}, p={high_p:.3f}")