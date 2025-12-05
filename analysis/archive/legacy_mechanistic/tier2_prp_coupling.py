"""
Tier 2.5: PRP Temporal Coupling Analysis

Tests whether UCLA loneliness impairs dual-task coordination by examining
the correlation between T1 and T2 reaction times (temporal coupling).

Hypothesis: UCLA disrupts T1-T2 coupling at short SOA (central bottleneck breakdown)

Author: Claude Code
Date: 2025-01-16
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials

# Output directory
OUTPUT_DIR = Path("results/analysis_outputs/prp_temporal_coupling")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("TIER 2.5: PRP TEMPORAL COUPLING ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading data...")

# Load master dataset (demographics, UCLA, DASS)
df_master = load_master_dataset()
print(f"  Master dataset: {len(df_master)} participants")

# Load PRP trial data - IMPORTANT: Set enforce_short_long_only=False to keep medium SOA
prp_df, prp_info = load_prp_trials(
    enforce_short_long_only=False,  # Keep ALL SOA bins including medium
    require_t1_correct=False,  # Allow all trials (we'll filter if needed)
    require_t2_correct_for_rt=False,  # Allow all trials
    drop_timeouts=True  # Still drop timeouts
)
print(f"  PRP trial data: {len(prp_df)} trials from {prp_df['participant_id'].nunique()} participants")

# Ensure t1_rt exists (may need to use t1_rt_ms)
if 't1_rt' not in prp_df.columns and 't1_rt_ms' in prp_df.columns:
    prp_df['t1_rt'] = prp_df['t1_rt_ms']
elif 't1_rt' in prp_df.columns:
    # If t1_rt exists but has NaNs, fill from t1_rt_ms
    if prp_df['t1_rt'].isna().any() and 't1_rt_ms' in prp_df.columns:
        prp_df['t1_rt'] = prp_df['t1_rt'].fillna(prp_df['t1_rt_ms'])

# Check we have required columns
if 't1_rt' not in prp_df.columns or 't2_rt' not in prp_df.columns:
    print(f"  ERROR: Missing required columns. Available: {prp_df.columns.tolist()}")
    sys.exit(1)

# ============================================================================
# STEP 2: Bin SOA into Short/Medium/Long
# ============================================================================
print("\nSTEP 2: Binning SOA conditions...")

def bin_soa(soa_ms):
    """Bin SOA into short/medium/long"""
    if soa_ms <= 150:
        return 'short'
    elif 300 <= soa_ms <= 600:
        return 'medium'
    elif soa_ms >= 1200:
        return 'long'
    else:
        return 'other'

prp_df['soa_bin'] = prp_df['soa'].apply(bin_soa)

# Filter to only short/medium/long trials
prp_df = prp_df[prp_df['soa_bin'].isin(['short', 'medium', 'long'])].copy()
print(f"  After filtering to short/medium/long SOA: {len(prp_df)} trials")

# SOA distribution
soa_counts = prp_df.groupby('soa_bin').size()
print("\n  SOA distribution:")
for soa_bin in ['short', 'medium', 'long']:
    if soa_bin in soa_counts.index:
        print(f"    {soa_bin}: {soa_counts[soa_bin]} trials")

# ============================================================================
# STEP 3: Compute T1-T2 Coupling per Participant × SOA
# ============================================================================
print("\nSTEP 3: Computing T1-T2 temporal coupling...")

coupling_results = []

for pid in prp_df['participant_id'].unique():
    pid_data = prp_df[prp_df['participant_id'] == pid]

    for soa_bin in ['short', 'medium', 'long']:
        soa_data = pid_data[pid_data['soa_bin'] == soa_bin]

        if len(soa_data) < 5:  # Need at least 5 trials for meaningful correlation
            continue

        t1_rt = soa_data['t1_rt'].values
        t2_rt = soa_data['t2_rt'].values

        # Remove any NaN values
        valid_mask = ~(np.isnan(t1_rt) | np.isnan(t2_rt))
        t1_rt = t1_rt[valid_mask]
        t2_rt = t2_rt[valid_mask]

        if len(t1_rt) < 5:
            continue

        # Compute Pearson correlation (temporal coupling)
        r_pearson, p_pearson = pearsonr(t1_rt, t2_rt)

        # Compute Spearman correlation (robust to outliers)
        r_spearman, p_spearman = spearmanr(t1_rt, t2_rt)

        coupling_results.append({
            'participant_id': pid,
            'soa_bin': soa_bin,
            'n_trials': len(t1_rt),
            'coupling_pearson': r_pearson,
            'coupling_spearman': r_spearman,
            'p_pearson': p_pearson,
            'p_spearman': p_spearman,
            't1_mean': np.mean(t1_rt),
            't1_sd': np.std(t1_rt),
            't2_mean': np.mean(t2_rt),
            't2_sd': np.std(t2_rt)
        })

coupling_df = pd.DataFrame(coupling_results)
print(f"  Computed coupling for {len(coupling_df)} participant × SOA combinations")

# Save coupling metrics
coupling_df.to_csv(OUTPUT_DIR / "coupling_metrics_raw.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STEP 4: Merge with Master Dataset
# ============================================================================
print("\nSTEP 4: Merging with demographics and UCLA/DASS...")

# Merge coupling metrics with master dataset
df = coupling_df.merge(df_master, on='participant_id', how='inner')
print(f"  Merged dataset: {len(df)} rows")

# Ensure necessary columns exist
required_cols = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'gender']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"  WARNING: Missing columns: {missing_cols}")
    sys.exit(1)

# Add gender_male binary
if 'gender_male' not in df.columns:
    df['gender_male'] = (df['gender'].str.lower() == 'male').astype(int)

# Standardize predictors
for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()

# ============================================================================
# STEP 5: Descriptive Statistics
# ============================================================================
print("\nSTEP 5: Descriptive statistics...")

desc_stats = df.groupby('soa_bin')[['coupling_pearson', 'coupling_spearman']].agg(['mean', 'std', 'count'])
print("\nCoupling by SOA condition:")
print(desc_stats.round(3))

# Save descriptive stats
desc_stats.to_csv(OUTPUT_DIR / "coupling_descriptive_stats.csv", encoding='utf-8-sig')

# ============================================================================
# STEP 6: Regression Analysis - Pearson Coupling
# ============================================================================
print("\nSTEP 6: Regression analysis - Pearson coupling...")

# Formula: coupling ~ UCLA × SOA_bin + Gender + DASS + age
# Note: SOA_bin is categorical with 3 levels (short, medium, long)
formula_pearson = "coupling_pearson ~ z_ucla_total * C(soa_bin, Treatment('long')) + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

try:
    model_pearson = smf.ols(formula_pearson, data=df).fit()
    print("\nPearson Coupling ~ UCLA × SOA_bin (DASS-controlled):")
    print(model_pearson.summary())

    # Save model summary
    with open(OUTPUT_DIR / "regression_pearson_coupling.txt", 'w', encoding='utf-8') as f:
        f.write(str(model_pearson.summary()))

    # Extract key coefficients
    pearson_results = []
    for param in model_pearson.params.index:
        if 'ucla' in param.lower() or 'soa_bin' in param.lower():
            pearson_results.append({
                'parameter': param,
                'coef': model_pearson.params[param],
                'se': model_pearson.bse[param],
                'tvalue': model_pearson.tvalues[param],
                'pvalue': model_pearson.pvalues[param],
                'ci_lower': model_pearson.conf_int().loc[param, 0],
                'ci_upper': model_pearson.conf_int().loc[param, 1]
            })

    pearson_results_df = pd.DataFrame(pearson_results)
    pearson_results_df.to_csv(OUTPUT_DIR / "regression_pearson_coefficients.csv", index=False, encoding='utf-8-sig')

    print("\nKey Pearson Coupling Results:")
    print(pearson_results_df.round(4))

except Exception as e:
    print(f"  ERROR in Pearson regression: {e}")
    pearson_results_df = pd.DataFrame()

# ============================================================================
# STEP 7: Regression Analysis - Spearman Coupling
# ============================================================================
print("\nSTEP 7: Regression analysis - Spearman coupling...")

formula_spearman = "coupling_spearman ~ z_ucla_total * C(soa_bin, Treatment('long')) + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

try:
    model_spearman = smf.ols(formula_spearman, data=df).fit()
    print("\nSpearman Coupling ~ UCLA × SOA_bin (DASS-controlled):")
    print(model_spearman.summary())

    # Save model summary
    with open(OUTPUT_DIR / "regression_spearman_coupling.txt", 'w', encoding='utf-8') as f:
        f.write(str(model_spearman.summary()))

    # Extract key coefficients
    spearman_results = []
    for param in model_spearman.params.index:
        if 'ucla' in param.lower() or 'soa_bin' in param.lower():
            spearman_results.append({
                'parameter': param,
                'coef': model_spearman.params[param],
                'se': model_spearman.bse[param],
                'tvalue': model_spearman.tvalues[param],
                'pvalue': model_spearman.pvalues[param],
                'ci_lower': model_spearman.conf_int().loc[param, 0],
                'ci_upper': model_spearman.conf_int().loc[param, 1]
            })

    spearman_results_df = pd.DataFrame(spearman_results)
    spearman_results_df.to_csv(OUTPUT_DIR / "regression_spearman_coefficients.csv", index=False, encoding='utf-8-sig')

    print("\nKey Spearman Coupling Results:")
    print(spearman_results_df.round(4))

except Exception as e:
    print(f"  ERROR in Spearman regression: {e}")
    spearman_results_df = pd.DataFrame()

# ============================================================================
# STEP 8: Visualization - T1 vs T2 Scatter by UCLA Tertile × SOA
# ============================================================================
print("\nSTEP 8: Creating visualizations...")

# Create UCLA tertiles for visualization
df_master_vis = df_master.copy()
df_master_vis['ucla_tertile'] = pd.qcut(df_master_vis['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

# Merge tertiles with trial data
prp_vis = prp_df.merge(df_master_vis[['participant_id', 'ucla_tertile']], on='participant_id', how='inner')

# Remove outliers (extreme RTs) for cleaner visualization
prp_vis = prp_vis[(prp_vis['t1_rt'] < 2000) & (prp_vis['t2_rt'] < 3000)].copy()

# Create faceted scatter plot
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('PRP T1-T2 Temporal Coupling by UCLA Tertile × SOA', fontsize=16, fontweight='bold')

soa_bins = ['short', 'medium', 'long']
ucla_tertiles = ['Low', 'Medium', 'High']
colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

for i, soa_bin in enumerate(soa_bins):
    for j, ucla_tertile in enumerate(ucla_tertiles):
        ax = axes[i, j]

        subset = prp_vis[(prp_vis['soa_bin'] == soa_bin) & (prp_vis['ucla_tertile'] == ucla_tertile)]

        if len(subset) > 0:
            ax.scatter(subset['t1_rt'], subset['t2_rt'], alpha=0.3, s=20, color=colors[ucla_tertile])

            # Add regression line
            try:
                z = np.polyfit(subset['t1_rt'], subset['t2_rt'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset['t1_rt'].min(), subset['t1_rt'].max(), 100)
                ax.plot(x_line, p(x_line), color=colors[ucla_tertile], linewidth=2, linestyle='--')

                # Compute and display correlation
                r, p_val = pearsonr(subset['t1_rt'], subset['t2_rt'])
                ax.text(0.05, 0.95, f'r={r:.3f}\np={p_val:.3f}\nn={len(subset)}',
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            except:
                pass

        ax.set_xlabel('T1 RT (ms)', fontsize=10)
        ax.set_ylabel('T2 RT (ms)', fontsize=10)
        ax.set_title(f'{soa_bin.upper()} SOA, UCLA {ucla_tertile}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "t1_t2_scatter_by_ucla_soa.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: t1_t2_scatter_by_ucla_soa.png")

# ============================================================================
# STEP 9: Visualization - Coupling by SOA and UCLA Quartile
# ============================================================================

# Add UCLA quartiles
df['ucla_quartile'] = pd.qcut(df['ucla_total'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('T1-T2 Temporal Coupling by SOA Condition and UCLA Loneliness', fontsize=14, fontweight='bold')

# Pearson coupling
ax1 = axes[0]
sns.boxplot(data=df, x='soa_bin', y='coupling_pearson', hue='ucla_quartile',
            order=['short', 'medium', 'long'], palette='RdYlGn_r', ax=ax1)
ax1.set_xlabel('SOA Condition', fontsize=12)
ax1.set_ylabel('T1-T2 Coupling (Pearson r)', fontsize=12)
ax1.set_title('Pearson Correlation', fontsize=13, fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.legend(title='UCLA Quartile', loc='upper right')
ax1.grid(True, alpha=0.3)

# Spearman coupling
ax2 = axes[1]
sns.boxplot(data=df, x='soa_bin', y='coupling_spearman', hue='ucla_quartile',
            order=['short', 'medium', 'long'], palette='RdYlGn_r', ax=ax2)
ax2.set_xlabel('SOA Condition', fontsize=12)
ax2.set_ylabel('T1-T2 Coupling (Spearman ρ)', fontsize=12)
ax2.set_title('Spearman Correlation', fontsize=13, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.legend(title='UCLA Quartile', loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "coupling_by_soa_ucla.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: coupling_by_soa_ucla.png")

# ============================================================================
# STEP 10: Summary Report
# ============================================================================
print("\nSTEP 10: Generating summary report...")

summary_lines = [
    "=" * 80,
    "TIER 2.5: PRP TEMPORAL COUPLING ANALYSIS - SUMMARY REPORT",
    "=" * 80,
    "",
    "RESEARCH QUESTION:",
    "Does UCLA loneliness impair dual-task coordination (T1-T2 temporal coupling)?",
    "Hypothesis: UCLA disrupts coupling at short SOA (central bottleneck breakdown)",
    "",
    "=" * 80,
    "DATA SUMMARY",
    "=" * 80,
    f"Total participants: {df['participant_id'].nunique()}",
    f"Total participant × SOA combinations: {len(df)}",
    f"  - Short SOA: {len(df[df['soa_bin'] == 'short'])} combinations",
    f"  - Medium SOA: {len(df[df['soa_bin'] == 'medium'])} combinations",
    f"  - Long SOA: {len(df[df['soa_bin'] == 'long'])} combinations",
    "",
    "=" * 80,
    "DESCRIPTIVE STATISTICS - COUPLING BY SOA",
    "=" * 80,
]

# Add descriptive stats
summary_lines.append("\nPearson Coupling:")
for soa_bin in ['short', 'medium', 'long']:
    soa_data = df[df['soa_bin'] == soa_bin]['coupling_pearson']
    summary_lines.append(f"  {soa_bin.upper()}: M={soa_data.mean():.3f}, SD={soa_data.std():.3f}, N={len(soa_data)}")

summary_lines.append("\nSpearman Coupling:")
for soa_bin in ['short', 'medium', 'long']:
    soa_data = df[df['soa_bin'] == soa_bin]['coupling_spearman']
    summary_lines.append(f"  {soa_bin.upper()}: M={soa_data.mean():.3f}, SD={soa_data.std():.3f}, N={len(soa_data)}")

summary_lines.extend([
    "",
    "=" * 80,
    "REGRESSION RESULTS - PEARSON COUPLING",
    "=" * 80,
])

if len(pearson_results_df) > 0:
    summary_lines.append("\nKey findings (Pearson coupling ~ UCLA × SOA + covariates):")
    for _, row in pearson_results_df.iterrows():
        sig = "***" if row['pvalue'] < 0.001 else "**" if row['pvalue'] < 0.01 else "*" if row['pvalue'] < 0.05 else ""
        summary_lines.append(f"  {row['parameter']}: β={row['coef']:.4f}, p={row['pvalue']:.4f} {sig}")
else:
    summary_lines.append("  No results available (model failed)")

summary_lines.extend([
    "",
    "=" * 80,
    "REGRESSION RESULTS - SPEARMAN COUPLING",
    "=" * 80,
])

if len(spearman_results_df) > 0:
    summary_lines.append("\nKey findings (Spearman coupling ~ UCLA × SOA + covariates):")
    for _, row in spearman_results_df.iterrows():
        sig = "***" if row['pvalue'] < 0.001 else "**" if row['pvalue'] < 0.01 else "*" if row['pvalue'] < 0.05 else ""
        summary_lines.append(f"  {row['parameter']}: β={row['coef']:.4f}, p={row['pvalue']:.4f} {sig}")
else:
    summary_lines.append("  No results available (model failed)")

summary_lines.extend([
    "",
    "=" * 80,
    "INTERPRETATION",
    "=" * 80,
    "",
    "The temporal coupling analysis examines whether T1 and T2 reaction times are",
    "correlated within participants, and whether this coupling is moderated by UCLA",
    "loneliness and SOA condition.",
    "",
    "Key Hypotheses:",
    "1. If UCLA → LOWER coupling at SHORT SOA:",
    "   → Loneliness disrupts central bottleneck coordination",
    "   → Impaired parallel processing capacity",
    "",
    "2. If UCLA → NO change in coupling:",
    "   → Loneliness affects individual task performance, not dual-task coordination",
    "",
    "3. If coupling is STRONGER at short SOA (regardless of UCLA):",
    "   → Evidence for central bottleneck (T1 delays cascade to T2)",
    "",
    "=" * 80,
    "FILES GENERATED",
    "=" * 80,
    "CSV Files:",
    "  - coupling_metrics_raw.csv (per-participant × SOA coupling values)",
    "  - coupling_descriptive_stats.csv (summary by SOA condition)",
    "  - regression_pearson_coefficients.csv (Pearson regression results)",
    "  - regression_spearman_coefficients.csv (Spearman regression results)",
    "",
    "Text Files:",
    "  - regression_pearson_coupling.txt (full regression output)",
    "  - regression_spearman_coupling.txt (full regression output)",
    "",
    "Figures:",
    "  - t1_t2_scatter_by_ucla_soa.png (3×3 faceted scatter plots)",
    "  - coupling_by_soa_ucla.png (boxplots by SOA and UCLA quartile)",
    "",
    "=" * 80,
    "END OF REPORT",
    "=" * 80
])

summary_text = "\n".join(summary_lines)
print(summary_text)

# Save summary
with open(OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nTier 2.5 PRP Temporal Coupling Analysis COMPLETE!")
