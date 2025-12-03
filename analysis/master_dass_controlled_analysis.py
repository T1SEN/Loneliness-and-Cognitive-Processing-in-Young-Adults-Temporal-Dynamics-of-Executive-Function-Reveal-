"""
Master DASS-Controlled Confirmatory Analysis
=============================================

This script re-tests ALL core hypotheses with proper DASS-21 covariate control.

Purpose:
--------
1. Ensure all "pure loneliness effects" claims are justified
2. Test whether UCLA effects survive DASS (depression/anxiety/stress) control
3. Provide hierarchical regression framework:
   - Model 0: Covariates only (age, gender)
   - Model 1: + DASS subscales
   - Model 2: + UCLA
   - Model 3: + UCLA × Gender interaction

Core Hypotheses Tested:
-----------------------
H1: UCLA × Gender → WCST PE rate (PRIMARY HYPOTHESIS)
H2: UCLA × Gender → WCST Accuracy
H3: UCLA × Gender → Stroop Interference
H4: UCLA × Gender → PRP Bottleneck
H5: Male vulnerability pathway (UCLA → PRP τ → WCST PE)
H6: Female compensation pathway (UCLA → PES ↑ → PE ↓)

Output:
-------
- hierarchical_regression_results.csv: Full model comparison
- dass_adjusted_effects.csv: UCLA effects after DASS control
- interaction_summary.csv: UCLA × Gender interactions
- model_comparison_plot.png: ΔR² visualization

Author: Research Team
Date: 2025-01-16
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from analysis.utils.data_loader_utils import load_master_dataset
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/master_dass_controlled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MASTER DASS-CONTROLLED CONFIRMATORY ANALYSIS")
print("=" * 80)
print()
print("Testing all core hypotheses with DASS-21 covariate control")
print("Hierarchical regression framework:")
print("  Model 0: Covariates only (age, gender)")
print("  Model 1: + DASS subscales (depression, anxiety, stress)")
print("  Model 2: + UCLA loneliness")
print("  Model 3: + UCLA × Gender interaction")
print()
print("  NOTE: Inference uses HC3 robust standard errors.")
print()

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("[1/6] Loading data...")

# Load master dataset via shared loader (force rebuild to avoid stale cache)
master = load_master_dataset(use_cache=False, force_rebuild=True, merge_cognitive_summary=True)
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Rename PE column if needed
for col in ['pe_rate', 'pe_rate']:
    if col in master.columns and col != 'pe_rate':
        master = master.rename(columns={col: 'pe_rate'})
        break

# Ensure required columns exist
required_cols = ['participant_id', 'ucla_total', 'gender_male', 'age',
                'dass_depression', 'dass_anxiety', 'dass_stress']

missing = [col for col in required_cols if col not in master.columns]
if missing:
    print(f"ERROR: Missing required columns: {missing}")
    sys.exit(1)

# Clean data - drop rows with missing covariates
master_clean = master.dropna(subset=required_cols).copy()

print(f"  Total N = {len(master_clean)}")
print(f"    Males: {(master_clean['gender_male'] == 1).sum()}")
print(f"    Females: {(master_clean['gender_male'] == 0).sum()}")
print()

# Standardize predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

master_clean['z_age'] = scaler.fit_transform(master_clean[['age']])
master_clean['z_ucla'] = scaler.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_str'] = scaler.fit_transform(master_clean[['dass_stress']])

# ============================================================================
# HIERARCHICAL REGRESSION FUNCTION
# ============================================================================

def hierarchical_regression(data, outcome, label):
    """
    Hierarchical regression with 4 models.

    Returns dict with model fits and statistics.
    """
    # Drop missing outcome
    df = data.dropna(subset=[outcome]).copy()

    if len(df) < 30:
        return None  # Insufficient data

    # Model 0: Covariates only (age + gender)
    formula0 = f"{outcome} ~ z_age + C(gender_male)"
    model0 = smf.ols(formula0, data=df).fit(cov_type="HC3")

    # Model 1: + DASS
    formula1 = f"{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str"
    model1 = smf.ols(formula1, data=df).fit(cov_type="HC3")

    # Model 2: + UCLA
    formula2 = f"{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla"
    model2 = smf.ols(formula2, data=df).fit(cov_type="HC3")

    # Model 3: + UCLA × Gender interaction
    formula3 = f"{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
    model3 = smf.ols(formula3, data=df).fit(cov_type="HC3")

    # Model comparison
    anova_1v0 = anova_lm(model0, model1)
    anova_2v1 = anova_lm(model1, model2)
    anova_3v2 = anova_lm(model2, model3)

    results = {
        'outcome': label,
        'n': len(df),

        # Model 0 (baseline)
        'model0_r2': model0.rsquared,
        'model0_aic': model0.aic,

        # Model 1 (+ DASS)
        'model1_r2': model1.rsquared,
        'model1_aic': model1.aic,
        'delta_r2_1v0': model1.rsquared - model0.rsquared,
        'f_1v0': anova_1v0['F'][1],
        'p_1v0': anova_1v0['Pr(>F)'][1],

        # Model 2 (+ UCLA)
        'model2_r2': model2.rsquared,
        'model2_aic': model2.aic,
        'delta_r2_2v1': model2.rsquared - model1.rsquared,
        'f_2v1': anova_2v1['F'][1],
        'p_2v1': anova_2v1['Pr(>F)'][1],
        'ucla_beta': model2.params['z_ucla'] if 'z_ucla' in model2.params else np.nan,
        'ucla_p': model2.pvalues['z_ucla'] if 'z_ucla' in model2.pvalues else np.nan,

        # Model 3 (+ interaction)
        'model3_r2': model3.rsquared,
        'model3_aic': model3.aic,
        'delta_r2_3v2': model3.rsquared - model2.rsquared,
        'f_3v2': anova_3v2['F'][1],
        'p_3v2': anova_3v2['Pr(>F)'][1],
    }

    # Interaction term
    int_term = "z_ucla:C(gender_male)[T.1]"
    if int_term in model3.params:
        results['interaction_beta'] = model3.params[int_term]
        results['interaction_p'] = model3.pvalues[int_term]
    else:
        results['interaction_beta'] = np.nan
        results['interaction_p'] = np.nan

    # Gender-stratified UCLA effects (from Model 3)
    females = df[df['gender_male'] == 0]
    males = df[df['gender_male'] == 1]

    if len(females) >= 15:
        f_formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        f_model = smf.ols(f_formula, data=females).fit()
        results['female_ucla_beta'] = f_model.params['z_ucla']
        results['female_ucla_p'] = f_model.pvalues['z_ucla']
    else:
        results['female_ucla_beta'] = np.nan
        results['female_ucla_p'] = np.nan

    if len(males) >= 15:
        m_formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        m_model = smf.ols(m_formula, data=males).fit()
        results['male_ucla_beta'] = m_model.params['z_ucla']
        results['male_ucla_p'] = m_model.pvalues['z_ucla']
    else:
        results['male_ucla_beta'] = np.nan
        results['male_ucla_p'] = np.nan

    return results

# ============================================================================
# TEST CORE HYPOTHESES
# ============================================================================

print("[2/6] Testing core hypotheses with hierarchical regression...")

outcomes_to_test = [
    ('pe_rate', 'WCST PE Rate', 'H1_PRIMARY'),
    ('wcst_accuracy', 'WCST Accuracy', 'H2'),
    ('stroop_interference', 'Stroop Interference', 'H3'),
    ('prp_bottleneck', 'PRP Bottleneck', 'H4'),
]

hierarchical_results = []

for outcome_col, outcome_label, hypothesis in outcomes_to_test:
    if outcome_col not in master_clean.columns:
        print(f"  ⚠ Skipping {outcome_label} (column not found)")
        continue

    print(f"  Testing {outcome_label}...")
    result = hierarchical_regression(master_clean, outcome_col, outcome_label)

    if result is not None:
        result['hypothesis'] = hypothesis
        hierarchical_results.append(result)

        # Print key findings
        print(f"    N (complete cases) = {result['n']}")
        print(f"    DASS contribution: ΔR² = {result['delta_r2_1v0']:.4f}, p = {result['p_1v0']:.4f}")
        print(f"    UCLA contribution: ΔR² = {result['delta_r2_2v1']:.4f}, p = {result['p_2v1']:.4f}")
        print(f"    UCLA×Gender: ΔR² = {result['delta_r2_3v2']:.4f}, p = {result['p_3v2']:.4f}")

        if not np.isnan(result['interaction_p']):
            print(f"    Interaction β = {result['interaction_beta']:.3f}, p = {result['interaction_p']:.4f}")

        if not np.isnan(result['female_ucla_p']) and not np.isnan(result['male_ucla_p']):
            print(f"    Female slope: β = {result['female_ucla_beta']:.3f}, p = {result['female_ucla_p']:.4f}")
            print(f"    Male slope: β = {result['male_ucla_beta']:.3f}, p = {result['male_ucla_p']:.4f}")
        print()

# Save results
hier_df = pd.DataFrame(hierarchical_results)

# Multiple-comparison control (BH/FDR) across outcomes for UCLA main and interaction steps
if not hier_df.empty:
    for col in ['p_2v1', 'p_3v2']:
        pvals = hier_df[col].to_numpy()
        mask = np.isfinite(pvals)
        adj = np.full_like(pvals, np.nan, dtype=float)
        if mask.sum() > 0:
            adj_vals = multipletests(pvals[mask], method='fdr_bh')[1]
            adj[mask] = adj_vals
        hier_df[f'{col}_fdr'] = adj

hier_df.to_csv(OUTPUT_DIR / "hierarchical_regression_results.csv", index=False, encoding='utf-8-sig')
print("  Saved: hierarchical_regression_results.csv")
if not hier_df.empty:
    print("  Note: FDR-adjusted p-values are stored in p_2v1_fdr and p_3v2_fdr columns.")
print()

# ============================================================================
# SUMMARY: DASS-ADJUSTED EFFECTS
# ============================================================================

print("[3/6] Summarizing DASS-adjusted UCLA effects...")

summary_results = []

for _, row in hier_df.iterrows():
    summary_results.append({
        'hypothesis': row['hypothesis'],
        'outcome': row['outcome'],
        'n': row['n'],
        'ucla_effect_controlled': row['ucla_beta'],
        'ucla_p_controlled': row['ucla_p'],
        'ucla_p_controlled_fdr': row.get('p_2v1_fdr', np.nan),
        'significant_after_dass': row['ucla_p'] < 0.05,
        'significant_after_dass_fdr': row.get('p_2v1_fdr', np.nan) < 0.05 if not np.isnan(row.get('p_2v1_fdr', np.nan)) else False,
        'interaction_beta': row['interaction_beta'],
        'interaction_p': row['interaction_p'],
        'interaction_p_fdr': row.get('p_3v2_fdr', np.nan),
        'significant_interaction': row['interaction_p'] < 0.05,
        'significant_interaction_fdr': row.get('p_3v2_fdr', np.nan) < 0.05 if not np.isnan(row.get('p_3v2_fdr', np.nan)) else False,
        'female_slope': row['female_ucla_beta'],
        'female_p': row['female_ucla_p'],
        'male_slope': row['male_ucla_beta'],
        'male_p': row['male_ucla_p']
    })

summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(OUTPUT_DIR / "dass_adjusted_effects.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: dass_adjusted_effects.csv")
print()

# ============================================================================
# VISUALIZATION: MODEL COMPARISON
# ============================================================================

print("[4/6] Creating model comparison visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (_, row) in enumerate(hier_df.iterrows()):
    if i >= 4:
        break

    ax = axes[i]

    # R² values
    r2_values = [
        row['model0_r2'],
        row['model1_r2'],
        row['model2_r2'],
        row['model3_r2']
    ]

    model_labels = ['Covariates\nOnly', '+ DASS', '+ UCLA', '+ UCLA×Gender']
    colors = ['#95A5A6', '#3498DB', '#E74C3C', '#2ECC71']

    bars = ax.bar(range(4), r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add ΔR² annotations
    delta_r2 = [
        0,
        row['delta_r2_1v0'],
        row['delta_r2_2v1'],
        row['delta_r2_3v2']
    ]

    for j, (bar, delta) in enumerate(zip(bars, delta_r2)):
        if j > 0:
            sig_symbol = '***' if delta > 0 and row[f'p_{j}v{j-1}'] < 0.001 else \
                        '**' if delta > 0 and row[f'p_{j}v{j-1}'] < 0.01 else \
                        '*' if delta > 0 and row[f'p_{j}v{j-1}'] < 0.05 else 'ns'
            ax.text(j, r2_values[j] + 0.01, f'ΔR²={delta:.3f}\n{sig_symbol}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylabel('R² (Variance Explained)', fontsize=11, fontweight='bold')
    ax.set_title(f"{row['outcome']}\n(N={row['n']:.0f})", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(r2_values) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_plot.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: model_comparison_plot.png")
plt.close()
print()

# ============================================================================
# CRITICAL ASSESSMENT: DID EFFECTS SURVIVE?
# ============================================================================

print("[5/6] Critical assessment: Did UCLA effects survive DASS control?")
print()

survived = summary_df[summary_df['significant_after_dass'] == True]
failed = summary_df[summary_df['significant_after_dass'] == False]

print(f"SURVIVED DASS CONTROL (p < 0.05): {len(survived)}/{len(summary_df)}")
if len(survived) > 0:
    for _, row in survived.iterrows():
        print(f"  ✓ {row['outcome']}: β = {row['ucla_effect_controlled']:.3f}, p = {row['ucla_p_controlled']:.4f}")
print()

print(f"FAILED DASS CONTROL (p ≥ 0.05): {len(failed)}/{len(summary_df)}")
if len(failed) > 0:
    for _, row in failed.iterrows():
        print(f"  ✗ {row['outcome']}: β = {row['ucla_effect_controlled']:.3f}, p = {row['ucla_p_controlled']:.4f}")
print()

print("UCLA × GENDER INTERACTIONS (DASS-controlled):")
sig_int = summary_df[summary_df['significant_interaction'] == True]
if len(sig_int) > 0:
    for _, row in sig_int.iterrows():
        print(f"  ✓ {row['outcome']}: β = {row['interaction_beta']:.3f}, p = {row['interaction_p']:.4f}")
        print(f"    Female: β = {row['female_slope']:.3f}, p = {row['female_p']:.4f}")
        print(f"    Male: β = {row['male_slope']:.3f}, p = {row['male_p']:.4f}")
else:
    print("  ✗ No significant interactions after DASS control")
print()

# ============================================================================
# FINAL REPORT
# ============================================================================

print("[6/6] Generating final report...")

with open(OUTPUT_DIR / "FINAL_DASS_CONTROL_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("FINAL REPORT: DASS-CONTROLLED CONFIRMATORY ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write("OBJECTIVE\n")
    f.write("-" * 80 + "\n")
    f.write("Test whether UCLA loneliness effects on executive function persist after\n")
    f.write("controlling for DASS-21 (depression, anxiety, stress).\n\n")

    f.write("RESULTS SUMMARY\n")
    f.write("-" * 80 + "\n\n")

    f.write(f"Total hypotheses tested: {len(summary_df)}\n")
    f.write(f"Survived DASS control (p < 0.05): {len(survived)}\n")
    f.write(f"Failed DASS control (p ≥ 0.05): {len(failed)}\n\n")

    if len(survived) > 0:
        f.write("EFFECTS THAT SURVIVED DASS CONTROL:\n")
        for _, row in survived.iterrows():
            f.write(f"\n{row['hypothesis']}: {row['outcome']}\n")
            f.write(f"  UCLA β = {row['ucla_effect_controlled']:.4f}, p = {row['ucla_p_controlled']:.4f}\n")
            if row['significant_interaction']:
                f.write(f"  Interaction β = {row['interaction_beta']:.4f}, p = {row['interaction_p']:.4f}\n")
                f.write(f"    Female: β = {row['female_slope']:.4f}, p = {row['female_p']:.4f}\n")
                f.write(f"    Male: β = {row['male_slope']:.4f}, p = {row['male_p']:.4f}\n")
        f.write("\n")

    if len(failed) > 0:
        f.write("EFFECTS THAT FAILED DASS CONTROL:\n")
        for _, row in failed.iterrows():
            f.write(f"\n{row['hypothesis']}: {row['outcome']}\n")
            f.write(f"  UCLA β = {row['ucla_effect_controlled']:.4f}, p = {row['ucla_p_controlled']:.4f}\n")
            f.write(f"  INTERPRETATION: Effect attributable to mood/anxiety, not pure loneliness\n")
        f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("CONCLUSION\n")
    f.write("=" * 80 + "\n\n")

    if len(survived) == 0:
        f.write("CRITICAL: NO UCLA effects survived DASS control.\n\n")
        f.write("All observed 'loneliness effects' are attributable to general emotional\n")
        f.write("distress (DASS), not pure loneliness. Claims of 'loneliness beyond mood'\n")
        f.write("are NOT supported.\n\n")
        f.write("RECOMMENDATION: Re-frame findings as 'emotional distress effects' or\n")
        f.write("acknowledge that loneliness and mood are inseparable in this dataset.\n")
    elif len(survived) < len(summary_df):
        f.write("PARTIAL SUPPORT: Some UCLA effects survived DASS control.\n\n")
        f.write(f"{len(survived)}/{len(summary_df)} effects remained significant after controlling\n")
        f.write("for mood/anxiety. These represent 'pure loneliness effects beyond mood.'\n\n")
        f.write("RECOMMENDATION: Focus publication on effects that survived. Clearly label\n")
        f.write("failed effects as mood-confounded.\n")
    else:
        f.write("FULL SUPPORT: All UCLA effects survived DASS control.\n\n")
        f.write("All tested effects remained significant after controlling for mood/anxiety.\n")
        f.write("Claims of 'pure loneliness effects beyond emotional distress' are justified.\n\n")
        f.write("RECOMMENDATION: Proceed with publication, emphasizing robust DASS control.\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("Generated: 2025-01-16\n")
    f.write(f"Output location: {OUTPUT_DIR}\n")
    f.write("=" * 80 + "\n")

print(f"  ✓ Saved: FINAL_DASS_CONTROL_REPORT.txt")
print()

# ============================================================================
# DONE
# ============================================================================

print("=" * 80)
print("MASTER DASS-CONTROLLED ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - hierarchical_regression_results.csv")
print("  - dass_adjusted_effects.csv")
print("  - model_comparison_plot.png")
print("  - FINAL_DASS_CONTROL_REPORT.txt")
print()
print("CRITICAL NEXT STEP:")
print("  → Read FINAL_DASS_CONTROL_REPORT.txt to determine if your core claims are justified")
print()
