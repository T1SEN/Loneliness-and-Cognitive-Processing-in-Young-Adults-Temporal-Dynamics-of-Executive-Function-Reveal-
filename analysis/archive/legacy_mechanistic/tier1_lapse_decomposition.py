"""
Lapse Frequency vs Magnitude Decomposition Analysis
====================================================

Research Question: Is the UCLA → RT variability effect driven by MORE lapses or BIGGER lapses?

Method:
- Classify trials as "normal" (RT < median + 2.5×MAD) vs "lapse" (RT ≥ threshold)
- Compute per participant: lapse_frequency (% lapse trials), lapse_magnitude (mean lapse RT - mean normal RT)
- Regression: Both metrics ~ UCLA × Gender + DASS + age
- Compare standardized β coefficients

Interpretation:
- If lapse_frequency β > lapse_magnitude β: Loneliness impairs sustained attention (more frequent lapses)
- If lapse_magnitude β > lapse_frequency β: Loneliness increases distraction depth (bigger lapses)

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
from scipy import stats
from scipy.stats import median_abs_deviation
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
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/lapse_decomposition")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("LAPSE FREQUENCY VS MAGNITUDE DECOMPOSITION ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# LAPSE CLASSIFICATION FUNCTION
# ============================================================================

def classify_lapses_mad(rt_series, threshold_multiplier=2.5):
    """
    Classify trials as normal or lapse based on RT using MAD (Median Absolute Deviation).

    Lapse threshold = median + threshold_multiplier × MAD

    Returns:
    - is_lapse: Boolean array indicating lapse trials
    - threshold: The RT threshold used
    """
    rt_clean = rt_series.dropna()
    if len(rt_clean) < 10:
        return pd.Series([False] * len(rt_series), index=rt_series.index), np.nan

    median_rt = rt_clean.median()
    mad_rt = median_abs_deviation(rt_clean, nan_policy='omit')

    # Avoid division by zero
    if mad_rt == 0:
        mad_rt = rt_clean.std() / 1.4826  # Fallback to SD-based estimate

    threshold = median_rt + threshold_multiplier * mad_rt
    is_lapse = rt_series > threshold

    return is_lapse, threshold


def compute_lapse_metrics(trial_df, rt_col='rt_ms', participant_col='participant_id'):
    """
    Compute lapse frequency and magnitude for each participant.

    Parameters:
    - trial_df: DataFrame with trial-level data
    - rt_col: Column name for reaction time
    - participant_col: Column name for participant ID

    Returns:
    - DataFrame with columns: participant_id, lapse_frequency, lapse_magnitude,
                              mean_normal_rt, mean_lapse_rt, n_trials, n_lapses
    """
    results = []

    for pid, group in trial_df.groupby(participant_col):
        rt_series = group[rt_col]

        # Skip if too few trials
        if len(rt_series) < 10 or rt_series.isna().all():
            continue

        # Classify lapses
        is_lapse, threshold = classify_lapses_mad(rt_series)

        # Separate normal and lapse RTs
        normal_rts = rt_series[~is_lapse]
        lapse_rts = rt_series[is_lapse]

        n_trials = len(rt_series)
        n_lapses = is_lapse.sum()

        # Lapse frequency (percentage)
        lapse_freq = (n_lapses / n_trials) * 100 if n_trials > 0 else np.nan

        # Mean RTs
        mean_normal_rt = normal_rts.mean() if len(normal_rts) > 0 else np.nan
        mean_lapse_rt = lapse_rts.mean() if len(lapse_rts) > 0 else np.nan

        # Lapse magnitude (difference)
        lapse_mag = mean_lapse_rt - mean_normal_rt if (len(lapse_rts) > 0 and len(normal_rts) > 0) else np.nan

        results.append({
            participant_col: pid,
            'lapse_frequency': lapse_freq,
            'lapse_magnitude': lapse_mag,
            'mean_normal_rt': mean_normal_rt,
            'mean_lapse_rt': mean_lapse_rt,
            'threshold': threshold,
            'n_trials': n_trials,
            'n_lapses': n_lapses
        })

    return pd.DataFrame(results)


# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading trial-level data...")
print()

# Load WCST trials
print("  Loading WCST trials...")
wcst_df, wcst_info = load_wcst_trials()
# Note: already has participant_id from loader, already filtered for valid RTs
print(f"    WCST: {len(wcst_df)} trials, {wcst_df['participant_id'].nunique()} participants")

# Load PRP trials
print("  Loading PRP trials...")
prp_df, prp_info = load_prp_trials()
# Use T2 RT for PRP (secondary task RT, more sensitive to dual-task demands)
prp_df['rt_ms'] = prp_df['t2_rt']  # Already renamed by loader
print(f"    PRP: {len(prp_df)} trials, {prp_df['participant_id'].nunique()} participants")

# Load Stroop trials
print("  Loading Stroop trials...")
stroop_df, stroop_info = load_stroop_trials()
stroop_df['rt_ms'] = stroop_df['rt']  # Stroop uses 'rt' column (renamed by loader)
print(f"    Stroop: {len(stroop_df)} trials, {stroop_df['participant_id'].nunique()} participants")

print()
print("Loading master dataset (UCLA, DASS, demographics)...")
master_df = load_master_dataset()
master_df = ensure_participant_id(master_df)

# Ensure gender is normalized
master_df['gender'] = normalize_gender_series(master_df['gender'])
master_df = master_df[master_df['gender'].isin(['male', 'female'])].copy()
master_df['gender_male'] = (master_df['gender'] == 'male').astype(int)

print(f"  Master dataset: {len(master_df)} participants")
print()

# ============================================================================
# COMPUTE LAPSE METRICS FOR EACH TASK
# ============================================================================

print("Computing lapse metrics for each task...")
print()

print("  WCST...")
wcst_lapse = compute_lapse_metrics(wcst_df, rt_col='rt_ms', participant_col='participant_id')
wcst_lapse.columns = [f'wcst_{col}' if col != 'participant_id' else col for col in wcst_lapse.columns]
print(f"    Computed metrics for {len(wcst_lapse)} participants")
print(f"    Mean lapse frequency: {wcst_lapse['wcst_lapse_frequency'].mean():.2f}%")
print(f"    Mean lapse magnitude: {wcst_lapse['wcst_lapse_magnitude'].mean():.2f} ms")

print("  PRP (T2)...")
prp_lapse = compute_lapse_metrics(prp_df, rt_col='rt_ms', participant_col='participant_id')
prp_lapse.columns = [f'prp_{col}' if col != 'participant_id' else col for col in prp_lapse.columns]
print(f"    Computed metrics for {len(prp_lapse)} participants")
print(f"    Mean lapse frequency: {prp_lapse['prp_lapse_frequency'].mean():.2f}%")
print(f"    Mean lapse magnitude: {prp_lapse['prp_lapse_magnitude'].mean():.2f} ms")

print("  Stroop...")
stroop_lapse = compute_lapse_metrics(stroop_df, rt_col='rt_ms', participant_col='participant_id')
stroop_lapse.columns = [f'stroop_{col}' if col != 'participant_id' else col for col in stroop_lapse.columns]
print(f"    Computed metrics for {len(stroop_lapse)} participants")
print(f"    Mean lapse frequency: {stroop_lapse['stroop_lapse_frequency'].mean():.2f}%")
print(f"    Mean lapse magnitude: {stroop_lapse['stroop_lapse_magnitude'].mean():.2f} ms")

print()

# ============================================================================
# MERGE WITH MASTER DATASET
# ============================================================================

print("Merging lapse metrics with UCLA, DASS, demographics...")
df = master_df.copy()
df = df.merge(wcst_lapse, on='participant_id', how='left')
df = df.merge(prp_lapse, on='participant_id', how='left')
df = df.merge(stroop_lapse, on='participant_id', how='left')

# Keep only participants with at least one task's lapse metrics
df = df.dropna(subset=['wcst_lapse_frequency', 'prp_lapse_frequency', 'stroop_lapse_frequency'], how='all')

print(f"  Final sample: {len(df)} participants with lapse metrics")
print()

# Standardize predictors for regression
print("Standardizing predictors (UCLA, DASS, age)...")
df['z_ucla'] = (df['ucla_total'] - df['ucla_total'].mean()) / df['ucla_total'].std()
df['z_dass_dep'] = (df['dass_depression'] - df['dass_depression'].mean()) / df['dass_depression'].std()
df['z_dass_anx'] = (df['dass_anxiety'] - df['dass_anxiety'].mean()) / df['dass_anxiety'].std()
df['z_dass_str'] = (df['dass_stress'] - df['dass_stress'].mean()) / df['dass_stress'].std()
df['z_age'] = (df['age'] - df['age'].mean()) / df['age'].std()
print()

# Save merged dataset
df.to_csv(OUTPUT_DIR / "lapse_metrics_by_participant.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'lapse_metrics_by_participant.csv'}")
print()

# ============================================================================
# REGRESSION ANALYSES
# ============================================================================

print("=" * 80)
print("REGRESSION ANALYSES: UCLA × Gender → Lapse Frequency vs Magnitude")
print("=" * 80)
print()

# Formula with DASS control
formula_base = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

results_summary = []

for task in ['wcst', 'prp', 'stroop']:
    print(f"\n{'=' * 60}")
    print(f"{task.upper()} TASK")
    print(f"{'=' * 60}\n")

    freq_col = f'{task}_lapse_frequency'
    mag_col = f'{task}_lapse_magnitude'

    # Check if sufficient data
    if df[freq_col].isna().all() or df[mag_col].isna().all():
        print(f"  Insufficient data for {task.upper()}, skipping...")
        continue

    # Lapse Frequency regression
    print(f"1. LAPSE FREQUENCY ({freq_col})")
    print("-" * 60)
    df_freq = df.dropna(subset=[freq_col, 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])
    if len(df_freq) < 20:
        print(f"  Insufficient N ({len(df_freq)}), skipping...")
    else:
        formula_freq = formula_base.format(outcome=freq_col)
        model_freq = smf.ols(formula_freq, data=df_freq).fit()
        print(model_freq.summary())
        print()

        # Extract key coefficients
        beta_ucla_freq = model_freq.params.get('z_ucla', np.nan)
        pval_ucla_freq = model_freq.pvalues.get('z_ucla', np.nan)
        ci_ucla_freq = model_freq.conf_int().loc['z_ucla'].values if 'z_ucla' in model_freq.params else [np.nan, np.nan]

        results_summary.append({
            'task': task,
            'outcome': 'lapse_frequency',
            'beta_ucla': beta_ucla_freq,
            'pval_ucla': pval_ucla_freq,
            'ci_lower': ci_ucla_freq[0],
            'ci_upper': ci_ucla_freq[1],
            'N': len(df_freq)
        })

    # Lapse Magnitude regression
    print(f"\n2. LAPSE MAGNITUDE ({mag_col})")
    print("-" * 60)
    df_mag = df.dropna(subset=[mag_col, 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])
    if len(df_mag) < 20:
        print(f"  Insufficient N ({len(df_mag)}), skipping...")
    else:
        formula_mag = formula_base.format(outcome=mag_col)
        model_mag = smf.ols(formula_mag, data=df_mag).fit()
        print(model_mag.summary())
        print()

        # Extract key coefficients
        beta_ucla_mag = model_mag.params.get('z_ucla', np.nan)
        pval_ucla_mag = model_mag.pvalues.get('z_ucla', np.nan)
        ci_ucla_mag = model_mag.conf_int().loc['z_ucla'].values if 'z_ucla' in model_mag.params else [np.nan, np.nan]

        results_summary.append({
            'task': task,
            'outcome': 'lapse_magnitude',
            'beta_ucla': beta_ucla_mag,
            'pval_ucla': pval_ucla_mag,
            'ci_lower': ci_ucla_mag[0],
            'ci_upper': ci_ucla_mag[1],
            'N': len(df_mag)
        })

# Save results summary
results_df = pd.DataFrame(results_summary)
results_df.to_csv(OUTPUT_DIR / "regression_results_summary.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUTPUT_DIR / 'regression_results_summary.csv'}")
print()

# ============================================================================
# COMPARISON: FREQUENCY VS MAGNITUDE
# ============================================================================

print("=" * 80)
print("COMPARISON: Lapse Frequency β vs Lapse Magnitude β")
print("=" * 80)
print()

comparison = results_df.pivot(index='task', columns='outcome', values='beta_ucla')
print(comparison)
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")
print()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Scatter plots: UCLA vs Lapse Frequency/Magnitude by Gender
for task in ['wcst', 'prp', 'stroop']:
    freq_col = f'{task}_lapse_frequency'
    mag_col = f'{task}_lapse_magnitude'

    df_plot = df.dropna(subset=[freq_col, mag_col, 'ucla_total', 'gender'])
    if len(df_plot) < 10:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Frequency plot
    for gender, marker in [('male', 'o'), ('female', 's')]:
        subset = df_plot[df_plot['gender'] == gender]
        axes[0].scatter(subset['ucla_total'], subset[freq_col],
                       label=gender.capitalize(), alpha=0.6, s=50, marker=marker)
    axes[0].set_xlabel('UCLA Loneliness Score', fontsize=12)
    axes[0].set_ylabel('Lapse Frequency (%)', fontsize=12)
    axes[0].set_title(f'{task.upper()}: UCLA vs Lapse Frequency', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Magnitude plot
    for gender, marker in [('male', 'o'), ('female', 's')]:
        subset = df_plot[df_plot['gender'] == gender]
        axes[1].scatter(subset['ucla_total'], subset[mag_col],
                       label=gender.capitalize(), alpha=0.6, s=50, marker=marker)
    axes[1].set_xlabel('UCLA Loneliness Score', fontsize=12)
    axes[1].set_ylabel('Lapse Magnitude (ms)', fontsize=12)
    axes[1].set_title(f'{task.upper()}: UCLA vs Lapse Magnitude', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{task}_ucla_lapse_scatter.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / f'{task}_ucla_lapse_scatter.png'}")
    plt.close()

# 2. Bar plot: β coefficients comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df['task'].unique()))
width = 0.35

freq_data = results_df[results_df['outcome'] == 'lapse_frequency']
mag_data = results_df[results_df['outcome'] == 'lapse_magnitude']

ax.bar(x - width/2, freq_data['beta_ucla'].values, width, label='Lapse Frequency', alpha=0.8)
ax.bar(x + width/2, mag_data['beta_ucla'].values, width, label='Lapse Magnitude', alpha=0.8)

ax.set_xlabel('Task', fontsize=12)
ax.set_ylabel('Standardized β (UCLA)', fontsize=12)
ax.set_title('UCLA Effect: Lapse Frequency vs Magnitude (DASS-Controlled)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([t.upper() for t in results_df['task'].unique()])
ax.legend()
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beta_comparison_freq_vs_mag.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'beta_comparison_freq_vs_mag.png'}")
plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
print("Key outputs:")
print("  1. lapse_metrics_by_participant.csv - Lapse metrics for all participants")
print("  2. regression_results_summary.csv - β coefficients for frequency vs magnitude")
print("  3. *_ucla_lapse_scatter.png - Scatter plots by task")
print("  4. beta_comparison_freq_vs_mag.png - Bar plot comparing β coefficients")
print()
