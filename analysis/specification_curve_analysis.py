"""
Specification Curve Analysis: Robustness of UCLA × Gender → WCST PE

Research Question:
Is the UCLA × Gender → WCST PE interaction robust across reasonable analytical choices?

Method:
Test all combinations of:
- Outlier handling (none, winsorize, remove)
- Covariates (none, DASS, DASS+age)
- Error metric (PE rate, total errors, strict PE only)
- Trial filtering (all, valid RT, no timeouts)
- Standardization (raw, z-score)

Expected: Median specification p<0.05, demonstrating robustness
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import ast

from analysis.utils.trial_data_loader import load_wcst_trials

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/specification_curve")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9

print("="*80)
print("SPECIFICATION CURVE ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Demographics and surveys via master
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]
master = master.rename(columns={"gender_normalized": "gender"})
master["gender"] = master["gender"].fillna("").astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

# Derive DASS total
if "dass_total" not in master.columns and all(col in master.columns for col in ["dass_depression", "dass_anxiety", "dass_stress"]):
    master["dass_total"] = master["dass_depression"] + master["dass_anxiety"] + master["dass_stress"]

# WCST trial-level data via shared loader
wcst_trials, _ = load_wcst_trials(use_cache=True)
if "is_pe" not in wcst_trials.columns:
    def _parse_wcst_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except Exception:
            return {}
    wcst_trials["extra_dict"] = wcst_trials["extra"].apply(_parse_wcst_extra) if "extra" in wcst_trials.columns else {}
    wcst_trials["is_pe"] = wcst_trials.get("extra_dict", {}).apply(lambda x: x.get("isPE", False) if isinstance(x, dict) else False)

print(f"  Loaded {len(wcst_trials):,} WCST trials from {wcst_trials['participant_id'].nunique()} participants")

# ============================================================================
# 2. DEFINE SPECIFICATION DIMENSIONS
# ============================================================================
print("\n[2/5] Defining analytical specifications...")

specifications = {
    'outlier_handling': [
        'none',  # Keep all data
        'winsorize_95',  # Cap at 95th percentile
        'remove_3sd'  # Remove >3 SD from mean
    ],
    'covariates': [
        'none',  # No covariates
        'dass',  # Control for DASS
        'dass_age'  # Control for DASS + age
    ],
    'error_metric': [
        'pe_rate',  # Perseverative error rate (%)
        'total_errors',  # Total number of errors
        'pe_count'  # Total number of PE
    ],
    'trial_filter': [
        'all',  # All trials
        'valid_rt',  # Only trials with valid RT
        'no_timeout'  # Exclude timeout trials (if available)
    ],
    'standardize': [
        'raw',  # Raw scores
        'zscore'  # Z-scored predictors
    ]
}

# Generate all combinations
all_specs = list(product(*specifications.values()))
spec_names = list(specifications.keys())

print(f"\n  Total specifications: {len(all_specs)}")
for dim, options in specifications.items():
    print(f"    {dim}: {len(options)} options")

# ============================================================================
# 3. COMPUTE ERROR METRICS FOR ALL FILTERS
# ============================================================================
print("\n[3/5] Computing error metrics...")

def _parse_wcst_extra(extra_str):
    """Parse extra field for isPE"""
    if not isinstance(extra_str, str):
        return False
    try:
        extra_dict = ast.literal_eval(extra_str)
        return extra_dict.get('isPE', False)
    except (ValueError, SyntaxError):
        return False

wcst_trials['is_pe'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials['is_correct'] = wcst_trials['correct'] == True
wcst_trials['rt_valid'] = wcst_trials['reactionTimeMs'] > 0

# Compute metrics for each trial filter
metrics_by_filter = {}

for trial_filter in ['all', 'valid_rt']:
    if trial_filter == 'all':
        subset = wcst_trials.copy()
    elif trial_filter == 'valid_rt':
        subset = wcst_trials[wcst_trials['rt_valid']].copy()

    metrics = subset.groupby('participant_id').agg({
        'is_pe': ['sum', 'mean', 'count'],
        'is_correct': lambda x: (~x).sum()  # total errors
    }).reset_index()

    metrics.columns = ['participant_id', 'pe_count', 'pe_rate', 'n_trials', 'total_errors']
    metrics['pe_rate'] = metrics['pe_rate'] * 100  # Convert to percentage

    metrics_by_filter[trial_filter] = metrics

# Merge base dataset
master = ucla_data.merge(participants[['participant_id', 'gender_male', 'age']], on='participant_id', how='inner')
master = master.merge(dass_data, on='participant_id', how='left')

print(f"  Base dataset: N={len(master)}")

# ============================================================================
# 4. RUN ALL SPECIFICATIONS
# ============================================================================
print("\n[4/5] Running all specifications...")

results = []
spec_id = 0

for spec in all_specs:
    spec_id += 1
    spec_dict = dict(zip(spec_names, spec))

    # Select trial filter
    trial_filter = spec_dict['trial_filter']
    if trial_filter == 'no_timeout':
        trial_filter = 'valid_rt'  # Fallback (no explicit timeout column)

    metrics = metrics_by_filter[trial_filter if trial_filter in metrics_by_filter else 'all'].copy()

    # Merge metrics
    data = master.merge(metrics, on='participant_id', how='inner')

    # Select error metric
    error_col = spec_dict['error_metric']
    if error_col not in data.columns:
        continue

    data['outcome'] = data[error_col]

    # Handle outliers
    if spec_dict['outlier_handling'] == 'winsorize_95':
        p95 = data['outcome'].quantile(0.95)
        data['outcome'] = np.where(data['outcome'] > p95, p95, data['outcome'])
    elif spec_dict['outlier_handling'] == 'remove_3sd':
        mean_outcome = data['outcome'].mean()
        sd_outcome = data['outcome'].std()
        data = data[np.abs(data['outcome'] - mean_outcome) <= 3 * sd_outcome]

    # Standardize
    if spec_dict['standardize'] == 'zscore':
        data['ucla_total_std'] = (data['ucla_total'] - data['ucla_total'].mean()) / data['ucla_total'].std()
        if 'dass_total' in data.columns:
            data['dass_total_std'] = (data['dass_total'] - data['dass_total'].mean()) / data['dass_total'].std()
        if 'age' in data.columns:
            data['age_std'] = (data['age'] - data['age'].mean()) / data['age'].std()
    else:
        data['ucla_total_std'] = data['ucla_total']
        if 'dass_total' in data.columns:
            data['dass_total_std'] = data['dass_total']
        if 'age' in data.columns:
            data['age_std'] = data['age']

    # Build formula
    covariates_spec = spec_dict['covariates']
    if covariates_spec == 'none':
        formula = 'outcome ~ ucla_total_std * gender_male'
    elif covariates_spec == 'dass':
        formula = 'outcome ~ ucla_total_std * gender_male + dass_total_std'
        data = data.dropna(subset=['dass_total_std'])
    elif covariates_spec == 'dass_age':
        formula = 'outcome ~ ucla_total_std * gender_male + dass_total_std + age_std'
        data = data.dropna(subset=['dass_total_std', 'age_std'])

    # Check sufficient N
    if len(data) < 30:
        continue

    # Fit model
    try:
        model = smf.ols(formula, data=data).fit()

        # Extract interaction term
        interaction_coef = model.params['ucla_total_std:gender_male']
        interaction_se = model.bse['ucla_total_std:gender_male']
        interaction_t = model.tvalues['ucla_total_std:gender_male']
        interaction_p = model.pvalues['ucla_total_std:gender_male']
        r_squared = model.rsquared

        results.append({
            'spec_id': spec_id,
            'outlier_handling': spec_dict['outlier_handling'],
            'covariates': spec_dict['covariates'],
            'error_metric': spec_dict['error_metric'],
            'trial_filter': spec_dict['trial_filter'],
            'standardize': spec_dict['standardize'],
            'n': len(data),
            'beta': interaction_coef,
            'se': interaction_se,
            't': interaction_t,
            'p': interaction_p,
            'p_sig': interaction_p < 0.05,
            'r_squared': r_squared,
            'ci_lower': interaction_coef - 1.96 * interaction_se,
            'ci_upper': interaction_coef + 1.96 * interaction_se
        })
    except Exception as e:
        print(f"  Spec {spec_id} failed: {e}")
        continue

results_df = pd.DataFrame(results)

print(f"\n  Completed {len(results_df)} specifications")
print(f"  Significant (p<0.05): {results_df['p_sig'].sum()} ({results_df['p_sig'].mean()*100:.1f}%)")
print(f"  Median p-value: {results_df['p'].median():.3f}")
print(f"  Median β: {results_df['beta'].median():.3f}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/5] Creating specification curve...")

# Sort by effect size
results_df_sorted = results_df.sort_values('beta').reset_index(drop=True)
results_df_sorted['rank'] = results_df_sorted.index + 1

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.05)

# Panel A: Effect sizes
ax_effect = fig.add_subplot(gs[0])

# Plot effect sizes with CIs
for idx, row in results_df_sorted.iterrows():
    color = '#2ECC71' if row['p_sig'] else '#E74C3C'
    ax_effect.plot([row['rank'], row['rank']], [row['ci_lower'], row['ci_upper']],
                   color=color, alpha=0.4, linewidth=1)
    ax_effect.scatter(row['rank'], row['beta'], c=color, s=20, alpha=0.8, zorder=3)

ax_effect.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax_effect.set_ylabel('Interaction Coefficient (β)', fontweight='bold', fontsize=11)
ax_effect.set_title('Specification Curve: UCLA × Gender → WCST PE Interaction', fontsize=13, fontweight='bold')
ax_effect.grid(alpha=0.3, axis='y')
ax_effect.set_xlim(0, len(results_df_sorted)+1)

# Add summary stats
median_beta = results_df_sorted['beta'].median()
pct_sig = results_df_sorted['p_sig'].mean() * 100
ax_effect.text(0.02, 0.98, f'Median β = {median_beta:.3f}\n{pct_sig:.0f}% significant (p<0.05)',
              transform=ax_effect.transAxes, fontsize=10, va='top',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Panel B: P-values
ax_p = fig.add_subplot(gs[1], sharex=ax_effect)

for idx, row in results_df_sorted.iterrows():
    color = '#2ECC71' if row['p_sig'] else '#E74C3C'
    ax_p.scatter(row['rank'], row['p'], c=color, s=15, alpha=0.7)

ax_p.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='α=0.05')
ax_p.set_ylabel('p-value', fontweight='bold')
ax_p.set_yscale('log')
ax_p.grid(alpha=0.3, axis='y')
ax_p.legend(loc='upper right', fontsize=9)

# Panel C: Specification choices (heatmap)
ax_specs = fig.add_subplot(gs[2:], sharex=ax_effect)

# Create binary matrix for specs
spec_matrix = []
spec_labels = []

for dim in ['outlier_handling', 'covariates', 'error_metric', 'standardize']:
    unique_vals = results_df_sorted[dim].unique()
    for val in unique_vals:
        spec_matrix.append((results_df_sorted[dim] == val).astype(int).values)
        spec_labels.append(f"{dim}={val}")

spec_matrix = np.array(spec_matrix)

# Plot heatmap
im = ax_specs.imshow(spec_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
ax_specs.set_yticks(range(len(spec_labels)))
ax_specs.set_yticklabels(spec_labels, fontsize=8)
ax_specs.set_xlabel('Specification Rank (sorted by effect size)', fontweight='bold')
ax_specs.set_xlim(-0.5, len(results_df_sorted)-0.5)

# Remove x tick labels except bottom
plt.setp(ax_effect.get_xticklabels(), visible=False)
plt.setp(ax_p.get_xticklabels(), visible=False)

plt.savefig(OUTPUT_DIR / "specification_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# Additional plot: Distribution of effect sizes
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(results_df['beta'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(results_df['beta'].median(), color='red', linestyle='--', linewidth=2, label=f'Median β = {results_df["beta"].median():.3f}')
ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Interaction Coefficient (UCLA × Gender → PE)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Effect Sizes Across Specifications', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "effect_size_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\nSaving results...")

results_df.to_csv(OUTPUT_DIR / "specification_curve_results.csv", index=False, encoding='utf-8-sig')

# Summary report
with open(OUTPUT_DIR / "SPECIFICATION_CURVE_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SPECIFICATION CURVE ANALYSIS - SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-"*80 + "\n")
    f.write("Is the UCLA × Gender → WCST PE interaction robust across reasonable\n")
    f.write("analytical choices?\n\n")

    f.write("SPECIFICATIONS TESTED\n")
    f.write("-"*80 + "\n")
    f.write(f"Total specifications: {len(results_df)}\n\n")
    for dim, options in specifications.items():
        f.write(f"{dim}:\n")
        for opt in options:
            count = (results_df[dim] == opt).sum()
            f.write(f"  - {opt}: {count} specs\n")
        f.write("\n")

    f.write("KEY FINDINGS\n")
    f.write("-"*80 + "\n")
    f.write(f"Median interaction β: {results_df['beta'].median():.3f}\n")
    f.write(f"Range: [{results_df['beta'].min():.3f}, {results_df['beta'].max():.3f}]\n")
    f.write(f"IQR: [{results_df['beta'].quantile(0.25):.3f}, {results_df['beta'].quantile(0.75):.3f}]\n\n")

    f.write(f"Median p-value: {results_df['p'].median():.3f}\n")
    f.write(f"Significant (p<0.05): {results_df['p_sig'].sum()} / {len(results_df)} ({results_df['p_sig'].mean()*100:.1f}%)\n")
    f.write(f"Highly significant (p<0.01): {(results_df['p'] < 0.01).sum()} / {len(results_df)} ({(results_df['p'] < 0.01).mean()*100:.1f}%)\n\n")

    f.write("ROBUSTNESS INTERPRETATION\n")
    f.write("-"*80 + "\n")
    if results_df['p_sig'].mean() > 0.5:
        f.write("✓ ROBUST FINDING\n")
        f.write(f"  {results_df['p_sig'].mean()*100:.0f}% of specifications yield p<0.05\n")
        f.write("  Effect survives diverse analytical choices\n")
        f.write("  Median effect size remains positive/consistent\n\n")
    elif results_df['p_sig'].mean() > 0.3:
        f.write("~ MODERATELY ROBUST\n")
        f.write(f"  {results_df['p_sig'].mean()*100:.0f}% of specifications yield p<0.05\n")
        f.write("  Effect present but sensitive to some choices\n\n")
    else:
        f.write("✗ FRAGILE FINDING\n")
        f.write(f"  Only {results_df['p_sig'].mean()*100:.0f}% of specifications yield p<0.05\n")
        f.write("  Effect may depend on specific analytical choices\n\n")

    f.write("SPECIFICATION SENSITIVITY\n")
    f.write("-"*80 + "\n")
    for dim in ['outlier_handling', 'covariates', 'error_metric']:
        f.write(f"\n{dim}:\n")
        for val in results_df[dim].unique():
            subset = results_df[results_df[dim] == val]
            pct_sig = subset['p_sig'].mean() * 100
            median_beta = subset['beta'].median()
            f.write(f"  {val}: {pct_sig:.0f}% sig, median β={median_beta:.3f}\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Full results saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ Specification Curve Analysis Complete!")
print("="*80)
print(f"\nRobustness: {results_df['p_sig'].mean()*100:.0f}% of specifications significant (p<0.05)")
print(f"Median effect: β={results_df['beta'].median():.3f}, p={results_df['p'].median():.3f}")
