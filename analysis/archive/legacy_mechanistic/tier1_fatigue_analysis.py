"""
Time-on-Task Fatigue Analysis
==============================

Research Question: Does UCLA predict vigilance decrement (RT slope) rather than baseline impairment (RT intercept)?

Method:
- Parse WCST timestamp column to compute trial-by-trial time elapsed
- Within-participant regression: RT ~ trial_number (extract slope = fatigue rate)
- Between-participant analysis: slope ~ UCLA × Gender + DASS + age
- Compare: UCLA → RT slope vs UCLA → RT intercept

Interpretation:
- If UCLA → slope (not intercept): Resource depletion / motivational deficit
- If UCLA → intercept (not slope): Capacity / baseline processing deficit

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
from datetime import datetime
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
from analysis.utils.trial_data_loader import load_wcst_trials

# Set random seed
np.random.seed(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/fatigue_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("TIME-ON-TASK FATIGUE ANALYSIS (WCST TIMESTAMPS)")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading WCST trial data with timestamps...")
wcst_df, wcst_info = load_wcst_trials()
print(f"  Loaded {len(wcst_df)} trials from {wcst_df['participant_id'].nunique()} participants")
print()

# Check if timestamp column exists
if 'timestamp' not in wcst_df.columns:
    print("ERROR: 'timestamp' column not found in WCST data!")
    print(f"Available columns: {wcst_df.columns.tolist()}")
    sys.exit(1)

print("Loading master dataset (UCLA, DASS, demographics)...")
master_df = load_master_dataset()
master_df = ensure_participant_id(master_df)
master_df['gender'] = normalize_gender_series(master_df['gender'])
master_df = master_df[master_df['gender'].isin(['male', 'female'])].copy()
master_df['gender_male'] = (master_df['gender'] == 'male').astype(int)
print(f"  Master dataset: {len(master_df)} participants")
print()

# ============================================================================
# COMPUTE FATIGUE METRICS PER PARTICIPANT
# ============================================================================

print("Computing fatigue slopes for each participant...")
print()

fatigue_results = []

for pid, group in wcst_df.groupby('participant_id'):
    # Sort by trial index to ensure correct temporal order
    group = group.sort_values('trial_index').reset_index(drop=True)

    # Skip if too few trials
    if len(group) < 20:
        continue

    # Extract RT and trial number
    rt_series = group['rt_ms'].values
    trial_nums = np.arange(len(rt_series))  # 0, 1, 2, ...

    # Remove invalid RTs
    valid_mask = (rt_series > 0) & (~np.isnan(rt_series))
    if valid_mask.sum() < 20:
        continue

    rt_valid = rt_series[valid_mask]
    trials_valid = trial_nums[valid_mask]

    # Within-participant regression: RT ~ trial_number
    # Using numpy polyfit (faster than statsmodels for simple linear regression)
    slope, intercept = np.polyfit(trials_valid, rt_valid, deg=1)

    # Compute residuals for variance estimation
    rt_pred = slope * trials_valid + intercept
    residuals = rt_valid - rt_pred
    residual_sd = np.std(residuals)

    # Mean RT (for comparison)
    mean_rt = np.mean(rt_valid)

    # Session duration (if timestamps are available and parseable)
    session_duration_min = np.nan
    if group['timestamp'].notna().any():
        try:
            # Parse timestamps
            timestamps = pd.to_datetime(group['timestamp'], errors='coerce')
            timestamps_valid = timestamps[valid_mask].dropna()
            if len(timestamps_valid) >= 2:
                duration_sec = (timestamps_valid.iloc[-1] - timestamps_valid.iloc[0]).total_seconds()
                session_duration_min = duration_sec / 60.0
        except Exception as e:
            pass  # Keep as NaN if parsing fails

    fatigue_results.append({
        'participant_id': pid,
        'slope': slope,  # ms per trial (fatigue rate)
        'intercept': intercept,  # ms (baseline RT)
        'mean_rt': mean_rt,  # ms
        'residual_sd': residual_sd,  # RT variability after detrending
        'n_trials': len(rt_valid),
        'session_duration_min': session_duration_min
    })

fatigue_df = pd.DataFrame(fatigue_results)
print(f"Computed fatigue metrics for {len(fatigue_df)} participants")
print()
print("Descriptive statistics:")
print(fatigue_df[['slope', 'intercept', 'mean_rt', 'residual_sd', 'session_duration_min']].describe())
print()

# ============================================================================
# MERGE WITH MASTER DATASET
# ============================================================================

print("Merging fatigue metrics with UCLA, DASS, demographics...")
df = master_df.merge(fatigue_df, on='participant_id', how='inner')
print(f"  Final sample: {len(df)} participants with fatigue metrics")
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
df.to_csv(OUTPUT_DIR / "fatigue_metrics_by_participant.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'fatigue_metrics_by_participant.csv'}")
print()

# ============================================================================
# REGRESSION ANALYSES
# ============================================================================

print("=" * 80)
print("REGRESSION ANALYSES: UCLA × Gender → Fatigue Slope vs Intercept")
print("=" * 80)
print()

# Formula with DASS control
formula_base = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

results_summary = []

# 1. Fatigue Slope (critical test)
print("1. FATIGUE SLOPE (ms/trial)")
print("-" * 60)
df_clean = df.dropna(subset=['slope', 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])
formula_slope = formula_base.format(outcome='slope')
model_slope = smf.ols(formula_slope, data=df_clean).fit()
print(model_slope.summary())
print()

beta_ucla_slope = model_slope.params.get('z_ucla', np.nan)
pval_ucla_slope = model_slope.pvalues.get('z_ucla', np.nan)
ci_ucla_slope = model_slope.conf_int().loc['z_ucla'].values if 'z_ucla' in model_slope.params else [np.nan, np.nan]

results_summary.append({
    'outcome': 'slope',
    'beta_ucla': beta_ucla_slope,
    'pval_ucla': pval_ucla_slope,
    'ci_lower': ci_ucla_slope[0],
    'ci_upper': ci_ucla_slope[1],
    'N': len(df_clean)
})

# 2. Baseline RT (intercept)
print("\n2. BASELINE RT (INTERCEPT, ms)")
print("-" * 60)
formula_intercept = formula_base.format(outcome='intercept')
model_intercept = smf.ols(formula_intercept, data=df_clean).fit()
print(model_intercept.summary())
print()

beta_ucla_intercept = model_intercept.params.get('z_ucla', np.nan)
pval_ucla_intercept = model_intercept.pvalues.get('z_ucla', np.nan)
ci_ucla_intercept = model_intercept.conf_int().loc['z_ucla'].values if 'z_ucla' in model_intercept.params else [np.nan, np.nan]

results_summary.append({
    'outcome': 'intercept',
    'beta_ucla': beta_ucla_intercept,
    'pval_ucla': pval_ucla_intercept,
    'ci_lower': ci_ucla_intercept[0],
    'ci_upper': ci_ucla_intercept[1],
    'N': len(df_clean)
})

# 3. Mean RT (for comparison)
print("\n3. MEAN RT (ms, for comparison)")
print("-" * 60)
formula_mean = formula_base.format(outcome='mean_rt')
model_mean = smf.ols(formula_mean, data=df_clean).fit()
print(model_mean.summary())
print()

beta_ucla_mean = model_mean.params.get('z_ucla', np.nan)
pval_ucla_mean = model_mean.pvalues.get('z_ucla', np.nan)
ci_ucla_mean = model_mean.conf_int().loc['z_ucla'].values if 'z_ucla' in model_mean.params else [np.nan, np.nan]

results_summary.append({
    'outcome': 'mean_rt',
    'beta_ucla': beta_ucla_mean,
    'pval_ucla': pval_ucla_mean,
    'ci_lower': ci_ucla_mean[0],
    'ci_upper': ci_ucla_mean[1],
    'N': len(df_clean)
})

# 4. Residual SD (variability after detrending)
print("\n4. RESIDUAL SD (RT variability after removing trend)")
print("-" * 60)
formula_resid = formula_base.format(outcome='residual_sd')
model_resid = smf.ols(formula_resid, data=df_clean).fit()
print(model_resid.summary())
print()

beta_ucla_resid = model_resid.params.get('z_ucla', np.nan)
pval_ucla_resid = model_resid.pvalues.get('z_ucla', np.nan)
ci_ucla_resid = model_resid.conf_int().loc['z_ucla'].values if 'z_ucla' in model_resid.params else [np.nan, np.nan]

results_summary.append({
    'outcome': 'residual_sd',
    'beta_ucla': beta_ucla_resid,
    'pval_ucla': pval_ucla_resid,
    'ci_lower': ci_ucla_resid[0],
    'ci_upper': ci_ucla_resid[1],
    'N': len(df_clean)
})

# Save results
results_df = pd.DataFrame(results_summary)
results_df.to_csv(OUTPUT_DIR / "regression_results_summary.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUTPUT_DIR / 'regression_results_summary.csv'}")
print()

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("=" * 80)
print("COMPARISON: Slope vs Intercept vs Mean RT")
print("=" * 80)
print()
print(results_df[['outcome', 'beta_ucla', 'pval_ucla']])
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")
print()

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Spaghetti plot: RT trajectories by UCLA quartile
print("  Creating RT trajectory spaghetti plot...")
df['ucla_quartile'] = pd.qcut(df['ucla_total'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

# Sample participants for visualization (too many lines = unreadable)
np.random.seed(42)
sample_pids = df.groupby('ucla_quartile')['participant_id'].apply(
    lambda x: x.sample(min(10, len(x)), random_state=42)
).values

# Get trial data for sampled participants
wcst_sample = wcst_df[wcst_df['participant_id'].isin(sample_pids)].copy()
wcst_sample = wcst_sample.merge(df[['participant_id', 'ucla_quartile']], on='participant_id')

fig, ax = plt.subplots(figsize=(12, 6))
for quartile, color in zip(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], ['blue', 'green', 'orange', 'red']):
    subset = wcst_sample[wcst_sample['ucla_quartile'] == quartile]
    for pid in subset['participant_id'].unique():
        pid_data = subset[subset['participant_id'] == pid].sort_values('trial_index')
        ax.plot(range(len(pid_data)), pid_data['rt_ms'].values,
               alpha=0.3, color=color, linewidth=0.8)

# Add quartile-level trend lines
for quartile, color in zip(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], ['blue', 'green', 'orange', 'red']):
    subset = wcst_sample[wcst_sample['ucla_quartile'] == quartile]
    trial_means = subset.groupby('trial_index')['rt_ms'].mean()
    ax.plot(trial_means.index, trial_means.values,
           color=color, linewidth=2.5, label=quartile, alpha=0.9)

ax.set_xlabel('Trial Number', fontsize=12)
ax.set_ylabel('Reaction Time (ms)', fontsize=12)
ax.set_title('RT Trajectories by UCLA Loneliness Quartile (WCST)', fontsize=14, fontweight='bold')
ax.legend(title='UCLA Quartile', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rt_trajectories_by_ucla.png", dpi=150, bbox_inches='tight')
print(f"    Saved: {OUTPUT_DIR / 'rt_trajectories_by_ucla.png'}")
plt.close()

# 2. Scatter: UCLA vs Slope (by gender)
fig, ax = plt.subplots(figsize=(10, 6))
for gender, marker in [('male', 'o'), ('female', 's')]:
    subset = df[df['gender'] == gender]
    ax.scatter(subset['ucla_total'], subset['slope'],
              label=gender.capitalize(), alpha=0.6, s=50, marker=marker)

ax.set_xlabel('UCLA Loneliness Score', fontsize=12)
ax.set_ylabel('Fatigue Slope (ms/trial)', fontsize=12)
ax.set_title('UCLA vs Fatigue Slope (WCST)', fontsize=14, fontweight='bold')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ucla_vs_slope.png", dpi=150, bbox_inches='tight')
print(f"    Saved: {OUTPUT_DIR / 'ucla_vs_slope.png'}")
plt.close()

# 3. Bar plot: β coefficients comparison
fig, ax = plt.subplots(figsize=(10, 6))
outcomes = results_df['outcome'].values
betas = results_df['beta_ucla'].values
colors = ['red' if p < 0.05 else 'steelblue' for p in results_df['pval_ucla'].values]

ax.bar(outcomes, betas, color=colors, alpha=0.8)
ax.set_xlabel('Outcome Measure', fontsize=12)
ax.set_ylabel('Standardized β (UCLA)', fontsize=12)
ax.set_title('UCLA Effect on Fatigue Metrics (DASS-Controlled)', fontsize=14, fontweight='bold')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

# Add p-value annotations
for i, (outcome, beta, pval) in enumerate(zip(outcomes, betas, results_df['pval_ucla'].values)):
    ax.text(i, beta + (0.05 * abs(betas).max() * np.sign(beta) if beta != 0 else 0.01),
           f'p={pval:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beta_comparison_fatigue.png", dpi=150, bbox_inches='tight')
print(f"    Saved: {OUTPUT_DIR / 'beta_comparison_fatigue.png'}")
plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
print("Key outputs:")
print("  1. fatigue_metrics_by_participant.csv - Fatigue slopes/intercepts for all participants")
print("  2. regression_results_summary.csv - β coefficients for slope vs intercept")
print("  3. rt_trajectories_by_ucla.png - Spaghetti plot of RT over time")
print("  4. ucla_vs_slope.png - Scatter plot of UCLA vs fatigue slope")
print("  5. beta_comparison_fatigue.png - Bar plot comparing β coefficients")
print()
print("INTERPRETATION:")
if abs(beta_ucla_slope) > abs(beta_ucla_intercept):
    print("  → UCLA effect is stronger on SLOPE than INTERCEPT")
    print("    This suggests a VIGILANCE DECREMENT / MOTIVATIONAL deficit")
else:
    print("  → UCLA effect is stronger on INTERCEPT than SLOPE")
    print("    This suggests a BASELINE CAPACITY deficit")
print()
