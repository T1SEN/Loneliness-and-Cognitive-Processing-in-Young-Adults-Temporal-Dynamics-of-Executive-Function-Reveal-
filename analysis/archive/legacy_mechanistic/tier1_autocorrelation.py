"""
RT Autocorrelation Analysis (Drift vs Noise)
============================================

Research Question: Is UCLA-related variance due to DRIFT (high AC1) or RANDOM NOISE (low AC1)?

Method:
- For each participant: compute lag-1 autocorrelation (AC1) on RT residuals after detrending
- Regression: AC1 ~ UCLA × Gender + DASS + age
- Alternative: ARIMA(1,0,0) model per participant, extract AR coefficient

Interpretation:
- High UCLA → lower AC1: More random lapses (white noise)
- High UCLA → higher AC1: Slower drift/fatigue (temporal dependency)

Author: Research Team
Date: 2025-12-03
Part of: Tier 1 Mechanistic Analyses
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf
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

OUTPUT_DIR = Path("results/analysis_outputs/autocorrelation")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("RT AUTOCORRELATION ANALYSIS (Drift vs Noise)")
print("=" * 80)
print()

# ============================================================================
# AUTOCORRELATION COMPUTATION
# ============================================================================

def compute_autocorrelation_metrics(trial_df, rt_col='rt_ms', participant_col='participant_id'):
    """
    Compute lag-1 autocorrelation on detrended RT for each participant.
    """
    results = []

    for pid, group in trial_df.groupby(participant_col):
        group = group.sort_values('trial_index').reset_index(drop=True)
        rt_series = group[rt_col].values

        # Remove invalid RTs
        valid_mask = (rt_series > 0) & (~np.isnan(rt_series))
        if valid_mask.sum() < 30:  # Need sufficient trials
            continue

        rt_valid = rt_series[valid_mask]
        n_trials = len(rt_valid)

        # Detrend: remove linear trend
        trial_nums = np.arange(n_trials)
        slope, intercept = np.polyfit(trial_nums, rt_valid, 1)
        rt_detrended = rt_valid - (slope * trial_nums + intercept)

        # Compute lag-1 autocorrelation
        try:
            acf_vals = acf(rt_detrended, nlags=5, fft=False)
            ac1 = acf_vals[1]  # Lag-1
            ac2 = acf_vals[2]  # Lag-2
            ac5 = acf_vals[5]  # Lag-5
        except:
            ac1 = np.nan
            ac2 = np.nan
            ac5 = np.nan

        # Also compute simple correlation-based AC1
        ac1_simple = np.corrcoef(rt_detrended[:-1], rt_detrended[1:])[0, 1] if len(rt_detrended) > 1 else np.nan

        # RT variability metrics
        rt_sd = np.std(rt_valid)
        rt_iqr = np.percentile(rt_valid, 75) - np.percentile(rt_valid, 25)

        results.append({
            participant_col: pid,
            'ac1': ac1,
            'ac1_simple': ac1_simple,
            'ac2': ac2,
            'ac5': ac5,
            'rt_sd': rt_sd,
            'rt_iqr': rt_iqr,
            'n_trials': n_trials
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
# COMPUTE AUTOCORRELATION FOR EACH TASK
# ============================================================================

print("Computing autocorrelation metrics...")
print()

print("  WCST...")
wcst_ac = compute_autocorrelation_metrics(wcst_df, rt_col='rt_ms')
wcst_ac.columns = [f'wcst_{col}' if col != 'participant_id' else col for col in wcst_ac.columns]
print(f"    {len(wcst_ac)} participants")

print("  PRP...")
prp_ac = compute_autocorrelation_metrics(prp_df, rt_col='rt_ms')
prp_ac.columns = [f'prp_{col}' if col != 'participant_id' else col for col in prp_ac.columns]
print(f"    {len(prp_ac)} participants")

print("  Stroop...")
stroop_ac = compute_autocorrelation_metrics(stroop_df, rt_col='rt_ms')
stroop_ac.columns = [f'stroop_{col}' if col != 'participant_id' else col for col in stroop_ac.columns]
print(f"    {len(stroop_ac)} participants")

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

df = df.merge(wcst_ac, on='participant_id', how='left')
df = df.merge(prp_ac, on='participant_id', how='left')
df = df.merge(stroop_ac, on='participant_id', how='left')

ac_cols = [c for c in df.columns if '_ac1' in c]
df = df.dropna(subset=ac_cols, how='all')
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

df.to_csv(OUTPUT_DIR / "autocorrelation_metrics.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'autocorrelation_metrics.csv'}")
print()

# ============================================================================
# REGRESSION ANALYSES
# ============================================================================

print("=" * 80)
print("REGRESSION ANALYSES: UCLA × Gender → AC1")
print("=" * 80)
print()

formula_base = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
results_summary = []

for task in ['wcst', 'prp', 'stroop']:
    print(f"\n{'=' * 60}")
    print(f"{task.upper()} TASK")
    print(f"{'=' * 60}\n")

    ac1_col = f'{task}_ac1'

    if ac1_col not in df.columns or df[ac1_col].isna().all():
        print(f"  No data, skipping...")
        continue

    print(f"LAG-1 AUTOCORRELATION ({ac1_col})")
    print("-" * 60)

    df_clean = df.dropna(subset=[ac1_col, 'z_ucla', 'gender_male',
                                  'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])

    if len(df_clean) < 20:
        print(f"  Insufficient N ({len(df_clean)}), skipping...")
        continue

    formula = formula_base.format(outcome=ac1_col)
    model = smf.ols(formula, data=df_clean).fit()
    print(model.summary())
    print()

    beta_ucla = model.params.get('z_ucla', np.nan)
    pval_ucla = model.pvalues.get('z_ucla', np.nan)
    ci_ucla = model.conf_int().loc['z_ucla'].values if 'z_ucla' in model.params else [np.nan, np.nan]

    results_summary.append({
        'task': task,
        'outcome': 'ac1',
        'beta_ucla': beta_ucla,
        'pval_ucla': pval_ucla,
        'ci_lower': ci_ucla[0],
        'ci_upper': ci_ucla[1],
        'N': len(df_clean)
    })

results_df = pd.DataFrame(results_summary)
results_df.to_csv(OUTPUT_DIR / "regression_results_summary.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUTPUT_DIR / 'regression_results_summary.csv'}")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")
print()

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Scatter: UCLA vs AC1 by gender
for task in ['wcst', 'prp', 'stroop']:
    ac1_col = f'{task}_ac1'

    df_plot = df.dropna(subset=[ac1_col, 'ucla_total', 'gender'])
    if len(df_plot) < 10:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))

    for gender, marker in [('male', 'o'), ('female', 's')]:
        subset = df_plot[df_plot['gender'] == gender]
        ax.scatter(subset['ucla_total'], subset[ac1_col],
                  label=gender.capitalize(), alpha=0.6, s=50, marker=marker)

    ax.set_xlabel('UCLA Loneliness Score', fontsize=12)
    ax.set_ylabel('Lag-1 Autocorrelation (AC1)', fontsize=12)
    ax.set_title(f'{task.upper()}: UCLA vs RT Autocorrelation', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{task}_ucla_ac1_scatter.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / f'{task}_ucla_ac1_scatter.png'}")
    plt.close()

# 2. Bar plot: β coefficients
if len(results_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = results_df['task'].values
    betas = results_df['beta_ucla'].values
    colors = ['red' if p < 0.05 else 'steelblue' for p in results_df['pval_ucla'].values]

    ax.bar(tasks, betas, color=colors, alpha=0.8)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Standardized β (UCLA → AC1)', fontsize=12)
    ax.set_title('UCLA Effect on RT Autocorrelation (DASS-Controlled)', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (task, beta, pval) in enumerate(zip(tasks, betas, results_df['pval_ucla'].values)):
        ax.text(i, beta + (0.01 * np.sign(beta) if beta != 0 else 0.01),
               f'p={pval:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "beta_comparison_ac1.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'beta_comparison_ac1.png'}")
    plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
print("Key outputs:")
print("  1. autocorrelation_metrics.csv - AC1 for all participants")
print("  2. regression_results_summary.csv - UCLA → AC1 effects")
print("  3. *_ucla_ac1_scatter.png - Scatter plots")
print("  4. beta_comparison_ac1.png - Bar plot of effects")
print()

if len(results_df) > 0:
    mean_beta = results_df['beta_ucla'].mean()
    print("INTERPRETATION:")
    if mean_beta < -0.05:
        print("  → UCLA → LOWER AC1 (negative β)")
        print("    This suggests MORE RANDOM LAPSES (white noise pattern)")
    elif mean_beta > 0.05:
        print("  → UCLA → HIGHER AC1 (positive β)")
        print("    This suggests DRIFT/FATIGUE (temporal dependency)")
    else:
        print("  → UCLA → No systematic AC1 effect")
        print("    Variance is neither drift nor purely random")
print()
