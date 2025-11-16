"""
Ex-Gaussian Gender Pathways Analysis
======================================
Tests gender-specific mechanisms linking UCLA to RT components

Background:
Ex-Gaussian distribution decomposes RT into:
- μ (mu): Normal component mean (core processing speed)
- σ (sigma): Normal component SD (processing variability)
- τ (tau): Exponential tail (attentional lapses)

Hypotheses:
- Males: UCLA → τ↑ (attentional lapses/mind-wandering)
- Females: UCLA → σ↑ (compensatory hyperarousal/variability)

Methods:
- UCLA × Gender interactions for each Ex-Gaussian parameter
- Stratified analyses by gender
- Correlations with WCST PE to test mechanism

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
import statsmodels.formula.api as smf

from data_loader_utils import load_master_dataset

# Unicode handling
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/exgaussian_pathways")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EX-GAUSSIAN GENDER PATHWAYS ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading Ex-Gaussian parameters...")

# Check for existing Ex-Gaussian data
exg_files = list(RESULTS_DIR.glob("analysis_outputs/mechanism_analysis/exgaussian/*parameters*.csv"))

if not exg_files:
    print("  ERROR: No Ex-Gaussian parameter files found!")
    print("  Looking for files in: results/analysis_outputs/mechanism_analysis/exgaussian/")
    sys.exit(1)

# Load Ex-Gaussian parameters
exg_file = exg_files[0]
print(f"  Loading: {exg_file.name}")

exg_data = pd.read_csv(exg_file)
print(f"  Loaded {len(exg_data)} rows")
print(f"  Columns: {exg_data.columns.tolist()}")

master = load_master_dataset()
master = master.rename(columns={"gender_normalized": "gender"})
if "dass_total" not in master.columns:
    dass_cols = [c for c in ["dass_depression", "dass_anxiety", "dass_stress"] if c in master.columns]
    master["dass_total"] = master[dass_cols].sum(axis=1, min_count=1)

if "participant_id" not in exg_data.columns:
    if "participantId" in exg_data.columns:
        exg_data = exg_data.rename(columns={"participantId": "participant_id"})
    else:
        raise KeyError("Ex-Gaussian parameter file must include participant_id.")

merge_cols = ["participant_id", "ucla_total", "age", "gender", "dass_total"]
if "pe_rate" in master.columns:
    merge_cols.append("pe_rate")

exg_data = exg_data.merge(master[merge_cols], on="participant_id", how="left")

if "pe_rate" in exg_data.columns and "wcst_pe_rate" not in exg_data.columns:
    exg_data = exg_data.rename(columns={"pe_rate": "wcst_pe_rate"})

exg_data["gender_male"] = exg_data["gender"].map({"male": 1, "female": 0})

exg_data = exg_data.dropna(subset=["ucla_total", "gender_male"])

print(f"\n  Complete cases: {len(exg_data)} observations")
print(f"  Unique participants: {exg_data['participant_id'].nunique()}")
print(f"  Gender distribution: {exg_data['gender'].value_counts().to_dict()}")

# Identify parameter columns (mu, sigma, tau variants)
param_cols = [col for col in exg_data.columns if any(x in col.lower() for x in ['mu', 'sigma', 'tau'])]
print(f"\n  Ex-Gaussian parameter columns found: {param_cols}")

# ============================================================================
# 2. UCLA × GENDER INTERACTIONS FOR EACH PARAMETER
# ============================================================================
print("\n[2/5] Testing UCLA × Gender interactions for Ex-Gaussian parameters...")

results = []

for param in param_cols:
    # Skip if all NaN
    if exg_data[param].isna().all():
        continue

    # Filter complete for this parameter
    subset = exg_data[[param, 'ucla_total', 'gender_male', 'age', 'dass_total']].dropna()

    if len(subset) < 10:
        continue

    # Model: Parameter ~ UCLA × Gender
    try:
        model = smf.ols(f'{param} ~ ucla_total * gender_male + age + dass_total', data=subset).fit()

        # Extract coefficients
        ucla_main = model.params.get('ucla_total', np.nan)
        gender_main = model.params.get('gender_male', np.nan)
        interaction = model.params.get('ucla_total:gender_male', np.nan)

        ucla_p = model.pvalues.get('ucla_total', np.nan)
        gender_p = model.pvalues.get('gender_male', np.nan)
        interaction_p = model.pvalues.get('ucla_total:gender_male', np.nan)

        results.append({
            'Parameter': param,
            'N': len(subset),
            'UCLA_main_β': ucla_main,
            'UCLA_main_p': ucla_p,
            'Gender_main_β': gender_main,
            'Gender_main_p': gender_p,
            'UCLA×Gender_β': interaction,
            'UCLA×Gender_p': interaction_p,
            'Significant_interaction': 'Yes' if interaction_p < 0.05 else 'No',
            'R²': model.rsquared
        })

        sig_marker = '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else ''
        print(f"\n  [{param}]")
        print(f"    UCLA main: β={ucla_main:.4f}, p={ucla_p:.4f}")
        print(f"    UCLA × Gender: β={interaction:.4f}, p={interaction_p:.4f} {sig_marker}")

    except Exception as e:
        print(f"\n  [{param}] Error: {str(e)}")
        continue

results_df = pd.DataFrame(results)

# ============================================================================
# 3. GENDER-STRATIFIED CORRELATIONS
# ============================================================================
print("\n[3/5] Gender-stratified correlations...")

gender_results = []

for param in param_cols:
    if exg_data[param].isna().all():
        continue

    for gender, label in [(0, 'Female'), (1, 'Male')]:
        subset = exg_data[(exg_data['gender_male'] == gender) & exg_data[[param, 'ucla_total']].notna().all(axis=1)]

        if len(subset) >= 10:
            corr, p_val = stats.pearsonr(subset['ucla_total'], subset[param])

            gender_results.append({
                'Parameter': param,
                'Gender': label,
                'N': len(subset),
                'r': corr,
                'p': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })

            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  {param} × UCLA ({label}): r={corr:.3f}, p={p_val:.4f} {sig_marker}")

gender_results_df = pd.DataFrame(gender_results)

# ============================================================================
# 4. MECHANISM TEST: Ex-Gaussian → WCST PE
# ============================================================================
print("\n[4/5] Testing mechanism: Ex-Gaussian parameters → WCST PE...")

if 'wcst_pe_rate' in exg_data.columns:
    mechanism_results = []

    for param in param_cols:
        subset = exg_data[[param, 'wcst_pe_rate', 'ucla_total', 'gender_male']].dropna()

        if len(subset) >= 10:
            # Correlation
            corr, p_val = stats.pearsonr(subset[param], subset['wcst_pe_rate'])

            # Partial correlation (controlling UCLA)
            from scipy.stats import pearsonr
            # Simple approach: residualize
            param_resid = smf.ols(f'{param} ~ ucla_total', data=subset).fit().resid
            pe_resid = smf.ols('wcst_pe_rate ~ ucla_total', data=subset).fit().resid
            partial_corr, partial_p = pearsonr(param_resid, pe_resid)

            mechanism_results.append({
                'Parameter': param,
                'N': len(subset),
                'r_with_PE': corr,
                'p_with_PE': p_val,
                'Partial_r': partial_corr,
                'Partial_p': partial_p,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })

            print(f"  {param} → PE: r={corr:.3f}, p={p_val:.4f}; partial r={partial_corr:.3f}, p={partial_p:.4f}")

    mechanism_df = pd.DataFrame(mechanism_results)
else:
    print("  WCST PE data not available, skipping mechanism test")
    mechanism_df = None

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/5] Creating visualizations...")

# Plot 1: UCLA × Gender interactions for key parameters
# Identify tau and sigma parameters
tau_params = [p for p in param_cols if 'tau' in p.lower()]
sigma_params = [p for p in param_cols if 'sigma' in p.lower()]

if tau_params and sigma_params:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot tau (attentional lapses - expected male effect)
    if tau_params:
        param = tau_params[0]
        for gender, label, color in [(0, 'Female', 'blue'), (1, 'Male', 'red')]:
            subset = exg_data[(exg_data['gender_male'] == gender) & exg_data[[param, 'ucla_total']].notna().all(axis=1)]
            if len(subset) >= 5:
                axes[0].scatter(subset['ucla_total'], subset[param], alpha=0.5, label=label, color=color)

                # Regression line
                z = np.polyfit(subset['ucla_total'], subset[param], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
                axes[0].plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

        axes[0].set_xlabel('UCLA Loneliness Score')
        axes[0].set_ylabel('τ (Tau - Attentional Lapses)')
        axes[0].set_title('Male Vulnerability: UCLA → Attentional Lapses', weight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # Plot sigma (variability - expected female effect)
    if sigma_params:
        param = sigma_params[0]
        for gender, label, color in [(0, 'Female', 'blue'), (1, 'Male', 'red')]:
            subset = exg_data[(exg_data['gender_male'] == gender) & exg_data[[param, 'ucla_total']].notna().all(axis=1)]
            if len(subset) >= 5:
                axes[1].scatter(subset['ucla_total'], subset[param], alpha=0.5, label=label, color=color)

                # Regression line
                z = np.polyfit(subset['ucla_total'], subset[param], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
                axes[1].plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2)

        axes[1].set_xlabel('UCLA Loneliness Score')
        axes[1].set_ylabel('σ (Sigma - RT Variability)')
        axes[1].set_title('Female Pattern: UCLA → RT Variability', weight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gender_specific_pathways.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 2: Forest plot of UCLA × Gender interactions
if not results_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by interaction strength
    results_sorted = results_df.sort_values('UCLA×Gender_β', ascending=False)

    y_pos = np.arange(len(results_sorted))
    colors = ['red' if sig == 'Yes' else 'gray' for sig in results_sorted['Significant_interaction']]

    ax.barh(y_pos, results_sorted['UCLA×Gender_β'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['Parameter'])
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('UCLA × Gender Interaction Coefficient')
    ax.set_title('Gender-Specific UCLA Effects on Ex-Gaussian Parameters\n(Red = Significant p<0.05)', weight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interaction_forest_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

# Save results
results_df.to_csv(OUTPUT_DIR / "ucla_gender_interactions.csv", index=False, encoding='utf-8-sig')
gender_results_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

if mechanism_df is not None:
    mechanism_df.to_csv(OUTPUT_DIR / "exgaussian_pe_mechanism.csv", index=False, encoding='utf-8-sig')

# Summary report
summary_lines = []
summary_lines.append("EX-GAUSSIAN GENDER PATHWAYS ANALYSIS - SUMMARY\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("RESEARCH QUESTION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Do males and females show different RT mechanisms linking UCLA to EF?\n")
summary_lines.append("  • Males: UCLA → τ↑ (attentional lapses)\n")
summary_lines.append("  • Females: UCLA → σ↑ (compensatory hyperarousal)\n\n")

summary_lines.append("KEY FINDINGS\n")
summary_lines.append("-" * 80 + "\n\n")

# Significant interactions
if not results_df.empty and 'Significant_interaction' in results_df.columns:
    sig_interactions = results_df[results_df['Significant_interaction'] == 'Yes']
    if not sig_interactions.empty:
        summary_lines.append(f"✓ {len(sig_interactions)} SIGNIFICANT UCLA×Gender interactions found:\n\n")
        for _, row in sig_interactions.iterrows():
            summary_lines.append(f"  • {row['Parameter']}: β={row['UCLA×Gender_β']:.4f}, p={row['UCLA×Gender_p']:.4f}\n")
        summary_lines.append("\n")
    else:
        summary_lines.append("✗ No significant UCLA×Gender interactions detected\n\n")
else:
    summary_lines.append("⚠️ UCLA×Gender interaction tests could not be performed\n\n")

# Gender-specific correlations
summary_lines.append("GENDER-STRATIFIED PATTERNS:\n")
summary_lines.append("-" * 80 + "\n")

# Males
male_sig = gender_results_df[(gender_results_df['Gender'] == 'Male') & (gender_results_df['Significant'] == 'Yes')]
if not male_sig.empty:
    summary_lines.append("Males:\n")
    for _, row in male_sig.iterrows():
        summary_lines.append(f"  • UCLA → {row['Parameter']}: r={row['r']:.3f}, p={row['p']:.4f}\n")
    summary_lines.append("\n")

# Females
female_sig = gender_results_df[(gender_results_df['Gender'] == 'Female') & (gender_results_df['Significant'] == 'Yes')]
if not female_sig.empty:
    summary_lines.append("Females:\n")
    for _, row in female_sig.iterrows():
        summary_lines.append(f"  • UCLA → {row['Parameter']}: r={row['r']:.3f}, p={row['p']:.4f}\n")
    summary_lines.append("\n")

summary_lines.append("\nINTERPRETATION\n")
summary_lines.append("-" * 80 + "\n")

# Check for tau vs sigma patterns
male_tau_sig = gender_results_df[(gender_results_df['Gender'] == 'Male') &
                                  (gender_results_df['Parameter'].str.contains('tau', case=False)) &
                                  (gender_results_df['Significant'] == 'Yes')]

female_sigma_sig = gender_results_df[(gender_results_df['Gender'] == 'Female') &
                                      (gender_results_df['Parameter'].str.contains('sigma', case=False)) &
                                      (gender_results_df['Significant'] == 'Yes')]

if not male_tau_sig.empty:
    summary_lines.append("✓ MALE LAPSE PATHWAY CONFIRMED:\n")
    summary_lines.append("  Lonely males show increased attentional lapses (τ)\n")
    summary_lines.append("  → Mind-wandering, social rumination depletes attention\n\n")

if not female_sigma_sig.empty:
    summary_lines.append("✓ FEMALE HYPERAROUSAL PATHWAY CONFIRMED:\n")
    summary_lines.append("  Lonely females show increased RT variability (σ)\n")
    summary_lines.append("  → Compensatory over-arousal, heightened vigilance\n\n")

if male_tau_sig.empty and female_sigma_sig.empty:
    summary_lines.append("⚠️ HYPOTHESIZED PATHWAYS NOT CONFIRMED\n")
    summary_lines.append("  May need larger sample or different task conditions\n\n")

summary_lines.append("="*80 + "\n")
summary_lines.append(f"Full results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print("\n" + summary_text)

with open(OUTPUT_DIR / "EXGAUSSIAN_PATHWAYS_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Ex-Gaussian Gender Pathways Analysis complete!")
