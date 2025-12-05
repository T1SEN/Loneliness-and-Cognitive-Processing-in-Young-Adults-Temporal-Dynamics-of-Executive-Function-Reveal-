"""
Speed-Accuracy Tradeoff Analysis
=================================

Research Question: Do high-UCLA individuals trade SPEED for ACCURACY (compensatory strategy)?

Method:
- Compute per-participant: mean_RT, error_rate
- Regression: mean_RT ~ UCLA × error_rate + DASS + age
- Test interaction: Does UCLA × error_rate predict mean_RT?
- Alternative: Inverse efficiency score (IES = mean_RT / accuracy) ~ UCLA

Interpretation:
- If UCLA × error_rate interaction significant:
    High UCLA slows down to maintain accuracy (strategic compensation)
- If null:
    No compensatory strategy; genuine control failure

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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from analysis.utils.data_loader_utils import (
    load_master_dataset,
    normalize_gender_series,
    ensure_participant_id
)
from analysis.utils.trial_data_loader import load_wcst_trials, load_prp_trials, load_stroop_trials

# Set random seed
np.random.seed(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/speed_accuracy")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("SPEED-ACCURACY TRADEOFF ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading trial-level data...")
print()

# Load WCST
wcst_df, _ = load_wcst_trials()
print(f"  WCST: {len(wcst_df)} trials, {wcst_df['participant_id'].nunique()} participants")

# Load PRP
prp_df, _ = load_prp_trials()
prp_df['rt_ms'] = prp_df['t2_rt']
prp_df['correct'] = prp_df['t2_correct']
print(f"  PRP: {len(prp_df)} trials, {prp_df['participant_id'].nunique()} participants")

# Load Stroop
stroop_df, _ = load_stroop_trials()
stroop_df['rt_ms'] = stroop_df['rt']
print(f"  Stroop: {len(stroop_df)} trials, {stroop_df['participant_id'].nunique()} participants")

print()
print("Loading master dataset (UCLA, DASS, demographics)...")
master_df = load_master_dataset()
master_df = ensure_participant_id(master_df)
master_df['gender'] = normalize_gender_series(master_df['gender'])
master_df = master_df[master_df['gender'].isin(['male', 'female'])].copy()
master_df['gender_male'] = (master_df['gender'] == 'male').astype(int)
print(f"  {len(master_df)} participants")
print()

# ============================================================================
# COMPUTE SPEED-ACCURACY METRICS PER PARTICIPANT
# ============================================================================

print("Computing speed-accuracy metrics for each task...")
print()

def compute_speed_accuracy_metrics(trial_df, rt_col='rt_ms', acc_col='correct', participant_col='participant_id'):
    """
    Compute mean RT, error rate, and inverse efficiency score per participant.
    """
    results = []

    for pid, group in trial_df.groupby(participant_col):
        rt_series = group[rt_col]
        acc_series = group[acc_col]

        # Remove invalid trials
        valid_mask = (rt_series > 0) & (~rt_series.isna()) & (~acc_series.isna())
        rt_valid = rt_series[valid_mask]
        acc_valid = acc_series[valid_mask]

        if len(rt_valid) < 10:
            continue

        # Metrics
        mean_rt = rt_valid.mean()
        error_rate = (1 - acc_valid.mean()) * 100  # Percentage
        accuracy = acc_valid.mean() * 100  # Percentage

        # Inverse efficiency score (IES = mean RT / accuracy)
        # Higher IES = worse performance (slow and inaccurate)
        ies = mean_rt / (accuracy / 100) if accuracy > 0 else np.nan

        results.append({
            participant_col: pid,
            'mean_rt': mean_rt,
            'error_rate': error_rate,
            'accuracy': accuracy,
            'ies': ies,
            'n_trials': len(rt_valid)
        })

    return pd.DataFrame(results)


print("  WCST...")
wcst_metrics = compute_speed_accuracy_metrics(wcst_df, rt_col='rt_ms', acc_col='correct')
wcst_metrics.columns = [f'wcst_{col}' if col != 'participant_id' else col for col in wcst_metrics.columns]
print(f"    {len(wcst_metrics)} participants")

print("  PRP (T2)...")
prp_metrics = compute_speed_accuracy_metrics(prp_df, rt_col='rt_ms', acc_col='correct')
prp_metrics.columns = [f'prp_{col}' if col != 'participant_id' else col for col in prp_metrics.columns]
print(f"    {len(prp_metrics)} participants")

print("  Stroop...")
stroop_metrics = compute_speed_accuracy_metrics(stroop_df, rt_col='rt_ms', acc_col='correct')
stroop_metrics.columns = [f'stroop_{col}' if col != 'participant_id' else col for col in stroop_metrics.columns]
print(f"    {len(stroop_metrics)} participants")

print()

# ============================================================================
# MERGE WITH MASTER DATASET
# ============================================================================

print("Merging speed-accuracy metrics with UCLA, DASS, demographics...")
df = master_df.copy()

# Drop existing task metrics to avoid conflicts
task_cols_to_drop = [c for c in df.columns if any(task in c.lower() for task in ['wcst', 'prp', 'stroop'])]
if len(task_cols_to_drop) > 0:
    print(f"  Dropping existing task columns: {len(task_cols_to_drop)} columns")
    df = df.drop(columns=task_cols_to_drop)

df = df.merge(wcst_metrics, on='participant_id', how='left')
df = df.merge(prp_metrics, on='participant_id', how='left')
df = df.merge(stroop_metrics, on='participant_id', how='left')

# Check which columns exist before dropping
mean_rt_cols = [col for col in ['wcst_mean_rt', 'prp_mean_rt', 'stroop_mean_rt'] if col in df.columns]
if len(mean_rt_cols) > 0:
    df = df.dropna(subset=mean_rt_cols, how='all')
print(f"  Final sample: {len(df)} participants")
print()

# Standardize predictors
print("Standardizing predictors (UCLA, DASS, age)...")
df['z_ucla'] = (df['ucla_total'] - df['ucla_total'].mean()) / df['ucla_total'].std()
df['z_dass_dep'] = (df['dass_depression'] - df['dass_depression'].mean()) / df['dass_depression'].std()
df['z_dass_anx'] = (df['dass_anxiety'] - df['dass_anxiety'].mean()) / df['dass_anxiety'].std()
df['z_dass_str'] = (df['dass_stress'] - df['dass_stress'].mean()) / df['dass_stress'].std()
df['z_age'] = (df['age'] - df['age'].mean()) / df['age'].std()
print()

# Save merged dataset
df.to_csv(OUTPUT_DIR / "speed_accuracy_metrics.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'speed_accuracy_metrics.csv'}")
print()

# ============================================================================
# REGRESSION ANALYSES
# ============================================================================

print("=" * 80)
print("REGRESSION ANALYSES: Speed-Accuracy Tradeoff")
print("=" * 80)
print()

results_summary = []

for task in ['wcst', 'prp', 'stroop']:
    print(f"\n{'=' * 60}")
    print(f"{task.upper()} TASK")
    print(f"{'=' * 60}\n")

    mean_rt_col = f'{task}_mean_rt'
    error_rate_col = f'{task}_error_rate'
    ies_col = f'{task}_ies'

    # Skip if columns don't exist or insufficient data
    if mean_rt_col not in df.columns or error_rate_col not in df.columns:
        print(f"  Columns not found, skipping...")
        continue

    if df[mean_rt_col].isna().all() or df[error_rate_col].isna().all():
        print(f"  Insufficient data, skipping...")
        continue

    # Standardize error_rate for interaction term
    df[f'z_{error_rate_col}'] = (df[error_rate_col] - df[error_rate_col].mean()) / df[error_rate_col].std()

    # ========================================================================
    # Test 1: Mean RT ~ UCLA × error_rate (INTERACTION TEST)
    # ========================================================================
    print("TEST 1: Mean RT ~ UCLA × Error Rate (Compensation Hypothesis)")
    print("-" * 60)

    df_clean = df.dropna(subset=[mean_rt_col, f'z_{error_rate_col}', 'z_ucla', 'gender_male',
                                  'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])

    if len(df_clean) < 20:
        print(f"  Insufficient N ({len(df_clean)}), skipping...")
        print()
        continue

    formula = f"{mean_rt_col} ~ z_ucla * z_{error_rate_col} + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_interaction = smf.ols(formula, data=df_clean).fit()
    print(model_interaction.summary())
    print()

    # Extract interaction coefficient
    beta_interaction = model_interaction.params.get(f'z_ucla:z_{error_rate_col}', np.nan)
    pval_interaction = model_interaction.pvalues.get(f'z_ucla:z_{error_rate_col}', np.nan)
    ci_interaction = model_interaction.conf_int().loc[f'z_ucla:z_{error_rate_col}'].values if f'z_ucla:z_{error_rate_col}' in model_interaction.params else [np.nan, np.nan]

    results_summary.append({
        'task': task,
        'test': 'RT ~ UCLA × error_rate',
        'beta_interaction': beta_interaction,
        'pval_interaction': pval_interaction,
        'ci_lower': ci_interaction[0],
        'ci_upper': ci_interaction[1],
        'N': len(df_clean)
    })

    # ========================================================================
    # Test 2: Inverse Efficiency Score ~ UCLA
    # ========================================================================
    print("\nTEST 2: Inverse Efficiency Score (IES) ~ UCLA")
    print("-" * 60)

    df_ies = df.dropna(subset=[ies_col, 'z_ucla', 'gender_male',
                                'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])

    if len(df_ies) < 20:
        print(f"  Insufficient N ({len(df_ies)}), skipping...")
        print()
        continue

    formula_ies = f"{ies_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_ies = smf.ols(formula_ies, data=df_ies).fit()
    print(model_ies.summary())
    print()

    # Extract UCLA coefficient
    beta_ucla_ies = model_ies.params.get('z_ucla', np.nan)
    pval_ucla_ies = model_ies.pvalues.get('z_ucla', np.nan)
    ci_ucla_ies = model_ies.conf_int().loc['z_ucla'].values if 'z_ucla' in model_ies.params else [np.nan, np.nan]

    results_summary.append({
        'task': task,
        'test': 'IES ~ UCLA',
        'beta_ucla': beta_ucla_ies,
        'pval_ucla': pval_ucla_ies,
        'ci_lower': ci_ucla_ies[0],
        'ci_upper': ci_ucla_ies[1],
        'N': len(df_ies)
    })

# Save results
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

# 1. Scatter: Mean RT vs Error Rate (by UCLA tertiles)
for task in ['wcst', 'prp', 'stroop']:
    mean_rt_col = f'{task}_mean_rt'
    error_rate_col = f'{task}_error_rate'

    df_plot = df.dropna(subset=[mean_rt_col, error_rate_col, 'ucla_total'])
    if len(df_plot) < 10:
        continue

    # Create UCLA tertiles
    df_plot['ucla_tertile'] = pd.qcut(df_plot['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

    fig, ax = plt.subplots(figsize=(10, 6))

    for tertile, color in zip(['Low', 'Medium', 'High'], ['green', 'orange', 'red']):
        subset = df_plot[df_plot['ucla_tertile'] == tertile]
        ax.scatter(subset[error_rate_col], subset[mean_rt_col],
                  label=f'UCLA {tertile}', alpha=0.6, s=50, color=color)

        # Add regression line
        if len(subset) > 3:
            try:
                z = np.polyfit(subset[error_rate_col], subset[mean_rt_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset[error_rate_col].min(), subset[error_rate_col].max(), 100)
                ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2, alpha=0.7)
            except:
                pass  # Skip regression line if polyfit fails

    ax.set_xlabel('Error Rate (%)', fontsize=12)
    ax.set_ylabel('Mean RT (ms)', fontsize=12)
    ax.set_title(f'{task.upper()}: Speed-Accuracy Relationship by UCLA Level',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{task}_speed_accuracy_scatter.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / f'{task}_speed_accuracy_scatter.png'}")
    plt.close()

# 2. Bar plot: UCLA × error_rate interaction coefficients
interaction_results = results_df[results_df['test'] == 'RT ~ UCLA × error_rate'].copy()

if len(interaction_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = interaction_results['task'].values
    betas = interaction_results['beta_interaction'].values
    colors = ['red' if p < 0.05 else 'steelblue' for p in interaction_results['pval_interaction'].values]

    ax.bar(tasks, betas, color=colors, alpha=0.8)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('β (UCLA × Error Rate)', fontsize=12)
    ax.set_title('UCLA × Error Rate Interaction on Mean RT (DASS-Controlled)',
                fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add p-value annotations
    for i, (task, beta, pval) in enumerate(zip(tasks, betas, interaction_results['pval_interaction'].values)):
        ax.text(i, beta + (0.05 * abs(betas).max() * np.sign(beta) if beta != 0 else 10),
               f'p={pval:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interaction_coefficients.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'interaction_coefficients.png'}")
    plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
print("Key outputs:")
print("  1. speed_accuracy_metrics.csv - Metrics for all participants")
print("  2. regression_results_summary.csv - Interaction and IES results")
print("  3. *_speed_accuracy_scatter.png - RT vs error rate by UCLA level")
print("  4. interaction_coefficients.png - UCLA × error_rate interactions")
print()
print("INTERPRETATION:")
if len(interaction_results) > 0 and (interaction_results['pval_interaction'] < 0.05).any():
    print("  → Significant UCLA × error_rate interaction detected")
    print("    High UCLA individuals show COMPENSATORY SLOWING")
else:
    print("  → No significant UCLA × error_rate interactions")
    print("    No evidence of compensatory speed-accuracy tradeoff")
print()
