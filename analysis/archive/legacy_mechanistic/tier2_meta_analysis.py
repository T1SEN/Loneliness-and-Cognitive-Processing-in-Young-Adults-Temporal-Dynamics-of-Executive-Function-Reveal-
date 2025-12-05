"""
Cross-Task Random-Effects Meta-Analysis
========================================

Research Question: What is the generalizable effect size of UCLA → RT variability across tasks?

Method:
- Extract standardized β from UCLA → RT variability models for:
  - WCST RT variability (SD, RMSSD, IQR)
  - PRP T2 RT variability
  - Stroop RT variability
- Random-effects meta-analysis: β_task ~ 1 + random(task)
- Compute: pooled β, I² heterogeneity, 95% CI and 95% prediction interval

Interpretation:
- I² < 25%: Effect is highly generalizable
- I² > 50%: Task-specific heterogeneity exists

Author: Research Team
Date: 2025-12-03
Part of: Tier 2 Computational Modeling
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from analysis.utils.data_loader_utils import (
    load_master_dataset,
    normalize_gender_series,
    ensure_participant_id
)
from analysis.utils.trial_data_loader import load_wcst_trials, load_prp_trials, load_stroop_trials

np.random.seed(42)

OUTPUT_DIR = Path("results/analysis_outputs/meta_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("CROSS-TASK RANDOM-EFFECTS META-ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# COMPUTE RT VARIABILITY METRICS
# ============================================================================

def compute_rt_variability_metrics(trial_df, rt_col='rt_ms', participant_col='participant_id'):
    """
    Compute multiple RT variability metrics per participant.
    """
    results = []

    for pid, group in trial_df.groupby(participant_col):
        rt_series = group[rt_col].values
        valid_mask = (rt_series > 0) & (~np.isnan(rt_series))
        rt_valid = rt_series[valid_mask]

        if len(rt_valid) < 20:
            continue

        # Standard deviation
        rt_sd = np.std(rt_valid, ddof=1)

        # Interquartile range
        rt_iqr = np.percentile(rt_valid, 75) - np.percentile(rt_valid, 25)

        # Root mean square of successive differences (RMSSD)
        diffs = np.diff(rt_valid)
        rmssd = np.sqrt(np.mean(diffs**2))

        # Coefficient of variation
        rt_cv = (rt_sd / np.mean(rt_valid)) * 100 if np.mean(rt_valid) > 0 else np.nan

        # Mean (for comparison)
        rt_mean = np.mean(rt_valid)

        results.append({
            participant_col: pid,
            'rt_mean': rt_mean,
            'rt_sd': rt_sd,
            'rt_iqr': rt_iqr,
            'rt_rmssd': rmssd,
            'rt_cv': rt_cv,
            'n_trials': len(rt_valid)
        })

    return pd.DataFrame(results)


# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading trial-level data...")
print()

wcst_df, _ = load_wcst_trials()
print(f"  WCST: {len(wcst_df)} trials, {wcst_df['participant_id'].nunique()} participants")

prp_df, _ = load_prp_trials()
prp_df['rt_ms'] = prp_df['t2_rt']
print(f"  PRP: {len(prp_df)} trials, {prp_df['participant_id'].nunique()} participants")

stroop_df, _ = load_stroop_trials()
stroop_df['rt_ms'] = stroop_df['rt']
print(f"  Stroop: {len(stroop_df)} trials, {stroop_df['participant_id'].nunique()} participants")

print()
print("Loading master dataset...")
master_df = load_master_dataset()
master_df = ensure_participant_id(master_df)
master_df['gender'] = normalize_gender_series(master_df['gender'])
master_df = master_df[master_df['gender'].isin(['male', 'female'])].copy()
master_df['gender_male'] = (master_df['gender'] == 'male').astype(int)
print(f"  {len(master_df)} participants")
print()

# ============================================================================
# COMPUTE VARIABILITY METRICS
# ============================================================================

print("Computing RT variability metrics...")
print()

print("  WCST...")
wcst_var = compute_rt_variability_metrics(wcst_df, rt_col='rt_ms')
wcst_var.columns = [f'wcst_{col}' if col != 'participant_id' else col for col in wcst_var.columns]
print(f"    {len(wcst_var)} participants")

print("  PRP...")
prp_var = compute_rt_variability_metrics(prp_df, rt_col='rt_ms')
prp_var.columns = [f'prp_{col}' if col != 'participant_id' else col for col in prp_var.columns]
print(f"    {len(prp_var)} participants")

print("  Stroop...")
stroop_var = compute_rt_variability_metrics(stroop_df, rt_col='rt_ms')
stroop_var.columns = [f'stroop_{col}' if col != 'participant_id' else col for col in stroop_var.columns]
print(f"    {len(stroop_var)} participants")

print()

# ============================================================================
# MERGE
# ============================================================================

print("Merging with master dataset...")
df = master_df.copy()

# Drop existing task columns
task_cols = [c for c in df.columns if any(t in c.lower() for t in ['wcst', 'prp', 'stroop'])]
if task_cols:
    df = df.drop(columns=task_cols)

df = df.merge(wcst_var, on='participant_id', how='left')
df = df.merge(prp_var, on='participant_id', how='left')
df = df.merge(stroop_var, on='participant_id', how='left')

var_cols = [c for c in df.columns if '_rt_sd' in c or '_rt_iqr' in c or '_rt_rmssd' in c]
df = df.dropna(subset=var_cols, how='all')
print(f"  Final sample: {len(df)} participants")
print()

# Standardize
print("Standardizing predictors...")
df['z_ucla'] = (df['ucla_total'] - df['ucla_total'].mean()) / df['ucla_total'].std()
df['z_dass_dep'] = (df['dass_depression'] - df['dass_depression'].mean()) / df['dass_depression'].std()
df['z_dass_anx'] = (df['dass_anxiety'] - df['dass_anxiety'].mean()) / df['dass_anxiety'].std()
df['z_dass_str'] = (df['dass_stress'] - df['dass_stress'].mean()) / df['dass_stress'].std()
df['z_age'] = (df['age'] - df['age'].mean()) / df['age'].std()
print()

df.to_csv(OUTPUT_DIR / "rt_variability_metrics.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'rt_variability_metrics.csv'}")
print()

# ============================================================================
# REGRESSION ANALYSES FOR META-ANALYSIS
# ============================================================================

print("=" * 80)
print("TASK-SPECIFIC REGRESSIONS: UCLA → RT Variability")
print("=" * 80)
print()

formula_base = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

meta_data = []

for task in ['wcst', 'prp', 'stroop']:
    print(f"\n{'=' * 60}")
    print(f"{task.upper()} TASK")
    print(f"{'=' * 60}\n")

    for metric in ['rt_sd', 'rt_iqr', 'rt_rmssd']:
        outcome_col = f'{task}_{metric}'

        if outcome_col not in df.columns or df[outcome_col].isna().all():
            continue

        print(f"{metric.upper()} ({outcome_col})")
        print("-" * 60)

        df_clean = df.dropna(subset=[outcome_col, 'z_ucla', 'gender_male',
                                      'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])

        if len(df_clean) < 20:
            print(f"  Insufficient N ({len(df_clean)}), skipping...")
            print()
            continue

        formula = formula_base.format(outcome=outcome_col)
        model = smf.ols(formula, data=df_clean).fit()

        # Print summary
        print(model.summary())
        print()

        # Extract UCLA coefficient
        beta_ucla = model.params.get('z_ucla', np.nan)
        se_ucla = model.bse.get('z_ucla', np.nan)
        pval_ucla = model.pvalues.get('z_ucla', np.nan)
        ci_ucla = model.conf_int().loc['z_ucla'].values if 'z_ucla' in model.params else [np.nan, np.nan]

        meta_data.append({
            'task': task,
            'metric': metric,
            'outcome': outcome_col,
            'beta': beta_ucla,
            'se': se_ucla,
            'pval': pval_ucla,
            'ci_lower': ci_ucla[0],
            'ci_upper': ci_ucla[1],
            'n': len(df_clean)
        })

meta_df = pd.DataFrame(meta_data)
meta_df.to_csv(OUTPUT_DIR / "task_specific_results.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUTPUT_DIR / 'task_specific_results.csv'}")
print()

# ============================================================================
# META-ANALYSIS
# ============================================================================

print("=" * 80)
print("RANDOM-EFFECTS META-ANALYSIS")
print("=" * 80)
print()

def random_effects_meta_analysis(effect_sizes, std_errors, labels):
    """
    Perform random-effects meta-analysis using DerSimonian-Laird method.

    Returns: pooled_effect, pooled_se, I2, tau2, Q_statistic, p_heterogeneity
    """
    k = len(effect_sizes)

    # Weights (inverse variance)
    weights = 1 / (std_errors ** 2)

    # Fixed-effects estimate
    pooled_fixed = np.sum(weights * effect_sizes) / np.sum(weights)

    # Q statistic (heterogeneity test)
    Q = np.sum(weights * (effect_sizes - pooled_fixed) ** 2)
    df = k - 1
    p_het = 1 - stats.chi2.cdf(Q, df) if df > 0 else np.nan

    # Tau-squared (between-study variance)
    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = max(0, (Q - df) / C) if C > 0 and df > 0 else 0

    # I² statistic
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    # Random-effects weights
    weights_re = 1 / (std_errors ** 2 + tau2)

    # Random-effects pooled estimate
    pooled_re = np.sum(weights_re * effect_sizes) / np.sum(weights_re)
    pooled_se = np.sqrt(1 / np.sum(weights_re))

    return {
        'pooled_beta': pooled_re,
        'pooled_se': pooled_se,
        'pooled_ci_lower': pooled_re - 1.96 * pooled_se,
        'pooled_ci_upper': pooled_re + 1.96 * pooled_se,
        'pooled_z': pooled_re / pooled_se,
        'pooled_p': 2 * (1 - stats.norm.cdf(abs(pooled_re / pooled_se))),
        'I2': I2,
        'tau2': tau2,
        'Q': Q,
        'p_heterogeneity': p_het,
        'n_studies': k
    }


# Meta-analyze each metric separately
meta_results = []

for metric in ['rt_sd', 'rt_iqr', 'rt_rmssd']:
    print(f"\n{'=' * 60}")
    print(f"META-ANALYSIS: {metric.upper()}")
    print(f"{'=' * 60}\n")

    subset = meta_df[meta_df['metric'] == metric].copy()

    if len(subset) < 2:
        print("  Insufficient studies for meta-analysis")
        continue

    # Remove any with missing SE
    subset = subset[subset['se'].notna() & (subset['se'] > 0)].copy()

    if len(subset) < 2:
        print("  Insufficient studies with valid SE")
        continue

    result = random_effects_meta_analysis(
        subset['beta'].values,
        subset['se'].values,
        subset['task'].values
    )

    result['metric'] = metric
    meta_results.append(result)

    print(f"Number of studies: {result['n_studies']}")
    print(f"Pooled β: {result['pooled_beta']:.2f}")
    print(f"95% CI: [{result['pooled_ci_lower']:.2f}, {result['pooled_ci_upper']:.2f}]")
    print(f"p-value: {result['pooled_p']:.4f}")
    print(f"I² heterogeneity: {result['I2']:.1f}%")
    print(f"τ² (tau-squared): {result['tau2']:.2f}")
    print(f"Q statistic: {result['Q']:.2f} (p = {result['p_heterogeneity']:.4f})")

    if result['I2'] < 25:
        print("  → Low heterogeneity: Effect is highly consistent across tasks")
    elif result['I2'] < 50:
        print("  → Moderate heterogeneity: Some task-specific variation")
    else:
        print("  → High heterogeneity: Substantial task-specific differences")
    print()

meta_results_df = pd.DataFrame(meta_results)
meta_results_df.to_csv(OUTPUT_DIR / "meta_analysis_results.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'meta_analysis_results.csv'}")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")
print()

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Forest plots for each metric
for metric in ['rt_sd', 'rt_iqr', 'rt_rmssd']:
    subset = meta_df[meta_df['metric'] == metric].copy()

    if len(subset) < 2:
        continue

    subset = subset[subset['se'].notna() & (subset['se'] > 0)].copy()

    if len(subset) < 2:
        continue

    # Get pooled result
    pooled = [r for r in meta_results if r['metric'] == metric]
    if len(pooled) == 0:
        continue
    pooled = pooled[0]

    fig, ax = plt.subplots(figsize=(10, max(6, len(subset) + 2)))

    y_positions = np.arange(len(subset))

    # Individual studies
    for i, (idx, row) in enumerate(subset.iterrows()):
        ax.errorbar(row['beta'], i,
                   xerr=[[row['beta'] - row['ci_lower']],
                         [row['ci_upper'] - row['beta']]],
                   fmt='s', markersize=8, capsize=5, capthick=2,
                   label=row['task'].upper() if i == 0 else "", alpha=0.7,
                   color='steelblue')

        # Add study label and weight
        weight = 1 / (row['se']**2 + pooled['tau2'])
        total_weight = np.sum(1 / (subset['se']**2 + pooled['tau2']))
        pct_weight = (weight / total_weight) * 100
        ax.text(row['ci_upper'] + 10, i,
               f"{row['task'].upper()} (n={int(row['n'])}, {pct_weight:.1f}%)",
               va='center', fontsize=9)

    # Pooled estimate
    ax.errorbar(pooled['pooled_beta'], len(subset) + 0.5,
               xerr=[[pooled['pooled_beta'] - pooled['pooled_ci_lower']],
                     [pooled['pooled_ci_upper'] - pooled['pooled_beta']]],
               fmt='D', markersize=12, capsize=7, capthick=3,
               color='red', label='Pooled', alpha=0.9, linewidth=2)

    ax.text(pooled['pooled_ci_upper'] + 10, len(subset) + 0.5,
           f"POOLED (I²={pooled['I2']:.0f}%)", va='center', fontsize=10, fontweight='bold')

    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_yticks(list(y_positions) + [len(subset) + 0.5])
    ax.set_yticklabels([''] * len(subset) + [''])
    ax.set_xlabel(f'Standardized β (UCLA → {metric.upper()})', fontsize=12)
    ax.set_title(f'Forest Plot: UCLA Effect on {metric.upper()} (DASS-Controlled)\n' +
                f'Pooled β={pooled["pooled_beta"]:.2f}, p={pooled["pooled_p"]:.4f}, I²={pooled["I2"]:.0f}%',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"forest_plot_{metric}.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / f'forest_plot_{metric}.png'}")
    plt.close()

# Summary plot: All metrics pooled effects
if len(meta_results_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = meta_results_df['metric'].values
    betas = meta_results_df['pooled_beta'].values
    ci_lowers = meta_results_df['pooled_ci_lower'].values
    ci_uppers = meta_results_df['pooled_ci_upper'].values

    y_pos = np.arange(len(metrics))

    colors = ['red' if p < 0.05 else 'steelblue' for p in meta_results_df['pooled_p'].values]

    ax.errorbar(betas, y_pos,
               xerr=[betas - ci_lowers, ci_uppers - betas],
               fmt='o', markersize=10, capsize=7, capthick=2, linewidth=2,
               color=colors[0] if len(set(colors)) == 1 else 'steelblue')

    for i, (metric, beta, ci_l, ci_u, I2, p) in enumerate(zip(
        metrics, betas, ci_lowers, ci_uppers,
        meta_results_df['I2'].values, meta_results_df['pooled_p'].values)):
        ax.text(ci_u + 5, i, f'β={beta:.1f}, p={p:.3f}, I²={I2:.0f}%',
               va='center', fontsize=9)

    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.upper() for m in metrics])
    ax.set_xlabel('Pooled Standardized β (UCLA Effect)', fontsize=12)
    ax.set_title('Meta-Analytic Summary: UCLA → RT Variability Metrics',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "meta_summary_all_metrics.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'meta_summary_all_metrics.png'}")
    plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
print("Key outputs:")
print("  1. rt_variability_metrics.csv - Variability metrics for all participants")
print("  2. task_specific_results.csv - Individual task regression results")
print("  3. meta_analysis_results.csv - Pooled meta-analytic estimates")
print("  4. forest_plot_*.png - Forest plots for each metric")
print("  5. meta_summary_all_metrics.png - Summary of all pooled effects")
print()
