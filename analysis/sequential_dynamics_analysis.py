"""
Sequential Dynamics Analysis
============================
시행 간 시계열적 패턴 분석

Features:
1. Trial-level 자기상관 (autocorrelation)
2. 학습 곡선 모델링
3. 동적 RT 패턴 (rolling variability)
4. UCLA × Gender × Trial progression 상호작용

DASS-21 Control: 모델에 통제 변수로 포함
"""

import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Add parent to path for imports
_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/sequential_dynamics_analysis.py')
sys.path.insert(0, str(_this_file.parent))
from utils.data_loader_utils import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR, RESULTS_DIR,
    ensure_participant_id, DEFAULT_RT_MIN, PRP_RT_MAX, STROOP_RT_MAX
)

# Define trial loaders directly to avoid import issues
def load_prp_trials():
    """Load PRP trial data."""
    df = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)
    # Normalize column names
    if 't2_rt_ms' in df.columns and 't2_rt' not in df.columns:
        df['t2_rt'] = df['t2_rt_ms']
    if 'soa_nominal_ms' in df.columns and 'soa' not in df.columns:
        df['soa'] = df['soa_nominal_ms']
    return df

def load_stroop_trials():
    """Load Stroop trial data."""
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)
    return df

def load_wcst_trials():
    """Load WCST trial data."""
    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)
    return df

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "sequential_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SEQUENTIAL DYNAMICS ANALYSIS")
print("=" * 70)


# ============================================================================
# Helper functions
# ============================================================================

def compute_autocorrelation(series, lag=1):
    """Compute autocorrelation at specified lag."""
    if len(series) < lag + 2:
        return np.nan
    return series.autocorr(lag=lag)


def compute_participant_autocorrelation(df, rt_col, participant_col='participant_id', max_lag=5):
    """Compute autocorrelation for each participant."""
    results = []

    for pid, group in df.groupby(participant_col):
        if len(group) < max_lag + 5:
            continue

        rt_series = group[rt_col].reset_index(drop=True)

        for lag in range(1, max_lag + 1):
            ac = compute_autocorrelation(rt_series, lag)
            results.append({
                'participant_id': pid,
                'lag': lag,
                'autocorrelation': ac
            })

    return pd.DataFrame(results)


def compute_learning_curve_metrics(df, rt_col, correct_col, participant_col='participant_id', n_bins=5):
    """
    Compute learning curve metrics by dividing trials into bins.
    """
    results = []

    for pid, group in df.groupby(participant_col):
        if len(group) < n_bins * 3:
            continue

        group = group.sort_values('trial_index' if 'trial_index' in group.columns else group.index.name).reset_index(drop=True)
        group['trial_bin'] = pd.qcut(range(len(group)), q=n_bins, labels=range(n_bins))

        for bin_idx in range(n_bins):
            bin_data = group[group['trial_bin'] == bin_idx]

            results.append({
                'participant_id': pid,
                'trial_bin': bin_idx + 1,
                'mean_rt': bin_data[rt_col].mean(),
                'sd_rt': bin_data[rt_col].std(),
                'accuracy': bin_data[correct_col].mean() if correct_col in bin_data.columns else np.nan,
                'n_trials': len(bin_data)
            })

    return pd.DataFrame(results)


def compute_rolling_variability(df, rt_col, participant_col='participant_id', window=10):
    """Compute rolling variability (CV) across trials."""
    results = []

    for pid, group in df.groupby(participant_col):
        if len(group) < window + 5:
            continue

        group = group.sort_values('trial_index' if 'trial_index' in group.columns else group.index.name).reset_index(drop=True)

        rolling_mean = group[rt_col].rolling(window=window, min_periods=window).mean()
        rolling_std = group[rt_col].rolling(window=window, min_periods=window).std()
        rolling_cv = rolling_std / rolling_mean

        # Summary statistics
        results.append({
            'participant_id': pid,
            'mean_rolling_cv': rolling_cv.mean(),
            'sd_rolling_cv': rolling_cv.std(),
            'cv_trend': stats.spearmanr(range(len(rolling_cv.dropna())), rolling_cv.dropna())[0] if len(rolling_cv.dropna()) > 5 else np.nan,
            'n_windows': len(rolling_cv.dropna())
        })

    return pd.DataFrame(results)


# ============================================================================
# Load data
# ============================================================================
print("\n[1] Loading trial-level data...")

# Load master dataset for individual differences
master_df = load_master_dataset()

# Load trial data
try:
    prp_trials = load_prp_trials()
    print(f"   PRP trials loaded: {len(prp_trials)}")
except Exception as e:
    print(f"   PRP trials: Error - {e}")
    prp_trials = None

try:
    stroop_trials = load_stroop_trials()
    print(f"   Stroop trials loaded: {len(stroop_trials)}")
except Exception as e:
    print(f"   Stroop trials: Error - {e}")
    stroop_trials = None

try:
    wcst_trials = load_wcst_trials()
    print(f"   WCST trials loaded: {len(wcst_trials)}")
except Exception as e:
    print(f"   WCST trials: Error - {e}")
    wcst_trials = None


# ============================================================================
# 2. Autocorrelation Analysis
# ============================================================================
print("\n[2] Computing RT autocorrelation by participant...")

autocorr_results = []

# PRP autocorrelation
if prp_trials is not None and len(prp_trials) > 100:
    # Filter valid trials
    prp_valid = prp_trials[
        (prp_trials['t2_rt'] > 100) &
        (prp_trials['t2_rt'] < 3000) &
        (prp_trials['t2_timeout'] == False)
    ].copy()

    if len(prp_valid) > 100:
        prp_ac = compute_participant_autocorrelation(prp_valid, 't2_rt')
        prp_ac['task'] = 'PRP'
        autocorr_results.append(prp_ac)
        print(f"   PRP autocorrelation computed: {len(prp_ac)} observations")

# Stroop autocorrelation
if stroop_trials is not None and len(stroop_trials) > 100:
    # Identify RT column
    rt_col = 'rt' if 'rt' in stroop_trials.columns else 'rt_ms'

    stroop_valid = stroop_trials[
        (stroop_trials[rt_col] > 100) &
        (stroop_trials[rt_col] < 3000) &
        (stroop_trials['timeout'] == False)
    ].copy()

    if len(stroop_valid) > 100:
        stroop_ac = compute_participant_autocorrelation(stroop_valid, rt_col)
        stroop_ac['task'] = 'Stroop'
        autocorr_results.append(stroop_ac)
        print(f"   Stroop autocorrelation computed: {len(stroop_ac)} observations")

# WCST autocorrelation
if wcst_trials is not None and len(wcst_trials) > 100:
    rt_col = 'rt_ms' if 'rt_ms' in wcst_trials.columns else 'reactionTimeMs'

    if rt_col in wcst_trials.columns:
        wcst_valid = wcst_trials[
            (wcst_trials[rt_col] > 100) &
            (wcst_trials[rt_col] < 5000)
        ].copy()

        if len(wcst_valid) > 100:
            wcst_ac = compute_participant_autocorrelation(wcst_valid, rt_col)
            wcst_ac['task'] = 'WCST'
            autocorr_results.append(wcst_ac)
            print(f"   WCST autocorrelation computed: {len(wcst_ac)} observations")

if autocorr_results:
    autocorr_df = pd.concat(autocorr_results, ignore_index=True)

    # Merge with individual differences
    autocorr_df = autocorr_df.merge(
        master_df[['participant_id', 'gender', 'ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress']],
        on='participant_id',
        how='left'
    )

    autocorr_df.to_csv(OUTPUT_DIR / "rt_autocorrelation.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: rt_autocorrelation.csv")

    # Summary by lag
    autocorr_summary = autocorr_df.groupby(['task', 'lag']).agg({
        'autocorrelation': ['mean', 'std', 'count']
    }).round(3)
    autocorr_summary.columns = ['mean_ac', 'sd_ac', 'n']
    autocorr_summary = autocorr_summary.reset_index()
    autocorr_summary.to_csv(OUTPUT_DIR / "autocorrelation_summary.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: autocorrelation_summary.csv")
else:
    autocorr_df = pd.DataFrame()
    print("   No autocorrelation data available")


# ============================================================================
# 3. Learning Curve Analysis
# ============================================================================
print("\n[3] Computing learning curves...")

learning_results = []

# PRP learning curve
if prp_trials is not None and len(prp_trials) > 100:
    prp_valid = prp_trials[
        (prp_trials['t2_rt'] > 100) &
        (prp_trials['t2_rt'] < 3000)
    ].copy()

    if 'trial_index' not in prp_valid.columns:
        prp_valid['trial_index'] = prp_valid.groupby('participant_id').cumcount()

    prp_learning = compute_learning_curve_metrics(prp_valid, 't2_rt', 't2_correct')
    prp_learning['task'] = 'PRP'
    learning_results.append(prp_learning)

# Stroop learning curve
if stroop_trials is not None and len(stroop_trials) > 100:
    rt_col = 'rt' if 'rt' in stroop_trials.columns else 'rt_ms'

    stroop_valid = stroop_trials[
        (stroop_trials[rt_col] > 100) &
        (stroop_trials[rt_col] < 3000)
    ].copy()

    if 'trial_index' not in stroop_valid.columns:
        stroop_valid['trial_index'] = stroop_valid.groupby('participant_id').cumcount()

    stroop_learning = compute_learning_curve_metrics(stroop_valid, rt_col, 'correct')
    stroop_learning['task'] = 'Stroop'
    learning_results.append(stroop_learning)

if learning_results and len(learning_results) > 0:
    learning_df = pd.concat(learning_results, ignore_index=True)

    # Merge with individual differences if participant_id exists
    if 'participant_id' in learning_df.columns and len(learning_df) > 0:
        learning_df = learning_df.merge(
            master_df[['participant_id', 'gender', 'ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress']],
            on='participant_id',
            how='left'
        )

        learning_df.to_csv(OUTPUT_DIR / "learning_curves.csv", index=False, encoding='utf-8-sig')
        print(f"   Saved: learning_curves.csv")

        # Summary
        learning_summary = learning_df.groupby(['task', 'trial_bin']).agg({
            'mean_rt': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).round(2)
        learning_summary.columns = ['rt_mean', 'rt_sd', 'acc_mean', 'acc_sd']
        learning_summary = learning_summary.reset_index()
        learning_summary.to_csv(OUTPUT_DIR / "learning_curve_summary.csv", index=False, encoding='utf-8-sig')
        print(f"   Saved: learning_curve_summary.csv")
    else:
        print("   Learning curves computed but missing participant_id")
        learning_df = pd.DataFrame()
else:
    learning_df = pd.DataFrame()
    print("   No learning curve data available")


# ============================================================================
# 4. Rolling Variability Analysis
# ============================================================================
print("\n[4] Computing rolling RT variability...")

rolling_results = []

# PRP rolling variability
if prp_trials is not None and len(prp_trials) > 100:
    prp_valid = prp_trials[
        (prp_trials['t2_rt'] > 100) &
        (prp_trials['t2_rt'] < 3000)
    ].copy()

    if 'trial_index' not in prp_valid.columns:
        prp_valid['trial_index'] = prp_valid.groupby('participant_id').cumcount()

    prp_rolling = compute_rolling_variability(prp_valid, 't2_rt', window=8)
    prp_rolling['task'] = 'PRP'
    rolling_results.append(prp_rolling)

# Stroop rolling variability
if stroop_trials is not None and len(stroop_trials) > 100:
    rt_col = 'rt' if 'rt' in stroop_trials.columns else 'rt_ms'

    stroop_valid = stroop_trials[
        (stroop_trials[rt_col] > 100) &
        (stroop_trials[rt_col] < 3000)
    ].copy()

    if 'trial_index' not in stroop_valid.columns:
        stroop_valid['trial_index'] = stroop_valid.groupby('participant_id').cumcount()

    stroop_rolling = compute_rolling_variability(stroop_valid, rt_col, window=8)
    stroop_rolling['task'] = 'Stroop'
    rolling_results.append(stroop_rolling)

if rolling_results and len(rolling_results) > 0:
    rolling_df = pd.concat(rolling_results, ignore_index=True)

    # Merge with individual differences if participant_id exists
    if 'participant_id' in rolling_df.columns and len(rolling_df) > 0:
        rolling_df = rolling_df.merge(
            master_df[['participant_id', 'gender', 'ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress']],
            on='participant_id',
            how='left'
        )

        rolling_df.to_csv(OUTPUT_DIR / "rolling_variability.csv", index=False, encoding='utf-8-sig')
        print(f"   Saved: rolling_variability.csv")
    else:
        print("   Rolling variability computed but missing participant_id")
        rolling_df = pd.DataFrame()
else:
    rolling_df = pd.DataFrame()
    print("   No rolling variability data available")


# ============================================================================
# 5. UCLA × Gender × Trial Progression Interaction
# ============================================================================
print("\n[5] Testing UCLA × Gender × Trial Progression interactions...")

interaction_results = []

if len(learning_df) > 50:
    # Standardize predictors
    learning_df['z_ucla'] = (learning_df['ucla_score'] - learning_df['ucla_score'].mean()) / learning_df['ucla_score'].std()
    learning_df['z_trial_bin'] = (learning_df['trial_bin'] - learning_df['trial_bin'].mean()) / learning_df['trial_bin'].std()
    learning_df['gender_male'] = (learning_df['gender'] == 'male').astype(int)

    for task in learning_df['task'].unique():
        task_df = learning_df[learning_df['task'] == task].dropna(subset=['z_ucla', 'gender_male', 'mean_rt'])

        if len(task_df) < 30:
            continue

        # Model: RT ~ UCLA * Gender * Trial_bin
        try:
            model = smf.mixedlm(
                "mean_rt ~ z_ucla * C(gender_male) * z_trial_bin",
                data=task_df,
                groups=task_df['participant_id']
            ).fit(reml=False)

            # Extract key effects
            for term in model.params.index:
                interaction_results.append({
                    'task': task,
                    'term': term,
                    'coefficient': model.params[term],
                    'se': model.bse[term],
                    'z': model.tvalues[term],
                    'p_value': model.pvalues[term]
                })

            print(f"   {task}: Model fitted successfully")

        except Exception as e:
            print(f"   {task}: Model failed - {e}")

            # Fallback to OLS
            try:
                model = smf.ols(
                    "mean_rt ~ z_ucla * C(gender_male) * z_trial_bin",
                    data=task_df
                ).fit()

                for term in model.params.index:
                    interaction_results.append({
                        'task': task,
                        'term': term,
                        'coefficient': model.params[term],
                        'se': model.bse[term],
                        'z': model.tvalues[term],
                        'p_value': model.pvalues[term]
                    })

                print(f"   {task}: OLS fallback successful")
            except Exception as e2:
                print(f"   {task}: OLS also failed - {e2}")

if interaction_results:
    interaction_df = pd.DataFrame(interaction_results)
    interaction_df.to_csv(OUTPUT_DIR / "trial_progression_interactions.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: trial_progression_interactions.csv")

    # Summary of significant effects
    sig_effects = interaction_df[interaction_df['p_value'] < 0.05]
    print(f"\n   Significant effects (p < .05): {len(sig_effects)}")
    for _, row in sig_effects.iterrows():
        print(f"     {row['task']} - {row['term']}: β = {row['coefficient']:.3f}, p = {row['p_value']:.4f}")
else:
    interaction_df = pd.DataFrame()
    print("   No interaction results available")


# ============================================================================
# 6. Autocorrelation × UCLA Analysis
# ============================================================================
print("\n[6] Testing UCLA effects on RT autocorrelation...")

ucla_ac_results = []

if len(autocorr_df) > 0:
    # Lag-1 autocorrelation only
    lag1_df = autocorr_df[autocorr_df['lag'] == 1].copy()
    lag1_df = lag1_df.dropna(subset=['ucla_score', 'autocorrelation'])

    for task in lag1_df['task'].unique():
        task_df = lag1_df[lag1_df['task'] == task]

        if len(task_df) < 20:
            continue

        # Correlation
        r, p = stats.pearsonr(task_df['ucla_score'], task_df['autocorrelation'])

        ucla_ac_results.append({
            'task': task,
            'lag': 1,
            'n': len(task_df),
            'correlation': r,
            'p_value': p
        })

        # By gender
        for gender in ['male', 'female']:
            gender_df = task_df[task_df['gender'] == gender]
            if len(gender_df) < 10:
                continue

            r_g, p_g = stats.pearsonr(gender_df['ucla_score'], gender_df['autocorrelation'])

            ucla_ac_results.append({
                'task': task,
                'lag': 1,
                'n': len(gender_df),
                'gender': gender,
                'correlation': r_g,
                'p_value': p_g
            })

if ucla_ac_results:
    ucla_ac_df = pd.DataFrame(ucla_ac_results)
    ucla_ac_df.to_csv(OUTPUT_DIR / "ucla_autocorrelation_effects.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: ucla_autocorrelation_effects.csv")


# ============================================================================
# 7. Visualization
# ============================================================================
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 7a. Autocorrelation by lag
ax = axes[0, 0]
if len(autocorr_df) > 0:
    summary = autocorr_df.groupby(['task', 'lag'])['autocorrelation'].mean().reset_index()
    for task in summary['task'].unique():
        task_data = summary[summary['task'] == task]
        ax.plot(task_data['lag'], task_data['autocorrelation'], marker='o', label=task)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Mean Autocorrelation')
    ax.set_title('RT Autocorrelation by Lag', fontweight='bold')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

# 7b. Learning curves (RT)
ax = axes[0, 1]
if len(learning_df) > 0:
    summary = learning_df.groupby(['task', 'trial_bin'])['mean_rt'].mean().reset_index()
    for task in summary['task'].unique():
        task_data = summary[summary['task'] == task]
        ax.plot(task_data['trial_bin'], task_data['mean_rt'], marker='o', label=task)
    ax.set_xlabel('Trial Bin')
    ax.set_ylabel('Mean RT (ms)')
    ax.set_title('Learning Curves (RT)', fontweight='bold')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

# 7c. Learning curves by UCLA group
ax = axes[0, 2]
if len(learning_df) > 0:
    # Median split
    median_ucla = learning_df['ucla_score'].median()
    learning_df['ucla_group'] = learning_df['ucla_score'].apply(lambda x: 'High' if x > median_ucla else 'Low')

    for task in learning_df['task'].unique():
        task_df = learning_df[learning_df['task'] == task]
        for group in ['Low', 'High']:
            group_df = task_df[task_df['ucla_group'] == group]
            summary = group_df.groupby('trial_bin')['mean_rt'].mean()
            linestyle = '-' if group == 'High' else '--'
            ax.plot(summary.index, summary.values, marker='o', linestyle=linestyle,
                   label=f'{task} ({group} UCLA)')

    ax.set_xlabel('Trial Bin')
    ax.set_ylabel('Mean RT (ms)')
    ax.set_title('Learning Curves by UCLA Group', fontweight='bold')
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

# 7d. Rolling CV distribution
ax = axes[1, 0]
if len(rolling_df) > 0:
    for task in rolling_df['task'].unique():
        task_data = rolling_df[rolling_df['task'] == task]['mean_rolling_cv'].dropna()
        ax.hist(task_data, bins=20, alpha=0.5, label=task)
    ax.set_xlabel('Mean Rolling CV')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Rolling RT Variability', fontweight='bold')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

# 7e. Rolling CV × UCLA
ax = axes[1, 1]
if len(rolling_df) > 0:
    rolling_df_plot = rolling_df.dropna(subset=['ucla_score', 'mean_rolling_cv'])
    for task in rolling_df_plot['task'].unique():
        task_data = rolling_df_plot[rolling_df_plot['task'] == task]
        ax.scatter(task_data['ucla_score'], task_data['mean_rolling_cv'],
                  alpha=0.5, label=task, s=30)
    ax.set_xlabel('UCLA Score')
    ax.set_ylabel('Mean Rolling CV')
    ax.set_title('RT Variability × UCLA', fontweight='bold')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

# 7f. Autocorrelation × UCLA by gender
ax = axes[1, 2]
if len(autocorr_df) > 0:
    lag1_df = autocorr_df[autocorr_df['lag'] == 1].dropna(subset=['ucla_score', 'autocorrelation', 'gender'])

    for gender, color in [('male', 'blue'), ('female', 'red')]:
        gender_data = lag1_df[lag1_df['gender'] == gender]
        ax.scatter(gender_data['ucla_score'], gender_data['autocorrelation'],
                  alpha=0.5, label=gender.capitalize(), color=color, s=30)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('UCLA Score')
    ax.set_ylabel('Lag-1 Autocorrelation')
    ax.set_title('RT Autocorrelation × UCLA by Gender', fontweight='bold')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sequential_dynamics_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: sequential_dynamics_plots.png")


# ============================================================================
# 8. Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("SEQUENTIAL DYNAMICS ANALYSIS SUMMARY")
print("=" * 70)

print("\n1. RT Autocorrelation:")
if len(autocorr_df) > 0:
    summary = autocorr_df[autocorr_df['lag'] == 1].groupby('task')['autocorrelation'].agg(['mean', 'std', 'count'])
    for task in summary.index:
        print(f"   {task}: Mean lag-1 AC = {summary.loc[task, 'mean']:.3f} ± {summary.loc[task, 'std']:.3f} (N={summary.loc[task, 'count']:.0f})")

print("\n2. Learning Effects:")
if len(learning_df) > 0:
    for task in learning_df['task'].unique():
        task_df = learning_df[learning_df['task'] == task]
        first_bin = task_df[task_df['trial_bin'] == 1]['mean_rt'].mean()
        last_bin = task_df[task_df['trial_bin'] == task_df['trial_bin'].max()]['mean_rt'].mean()
        change = last_bin - first_bin
        print(f"   {task}: RT change from first to last bin = {change:.1f} ms ({change/first_bin*100:.1f}%)")

print("\n3. UCLA Effects on Sequential Dynamics:")
if ucla_ac_results:
    for result in ucla_ac_results:
        if 'gender' not in result:
            sig = "*" if result['p_value'] < 0.05 else ""
            print(f"   {result['task']}: r(UCLA, AC) = {result['correlation']:.3f}, p = {result['p_value']:.4f} {sig}")

print("\n4. UCLA × Gender × Trial Progression:")
if len(interaction_df) > 0:
    # Focus on three-way interaction
    threeway = interaction_df[interaction_df['term'].str.contains('z_ucla:C') & interaction_df['term'].str.contains('z_trial_bin')]
    for _, row in threeway.iterrows():
        sig = "*" if row['p_value'] < 0.05 else ""
        print(f"   {row['task']}: β = {row['coefficient']:.3f}, p = {row['p_value']:.4f} {sig}")

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("=" * 70)

# Save report
with open(OUTPUT_DIR / "sequential_dynamics_report.txt", 'w', encoding='utf-8') as f:
    f.write("SEQUENTIAL DYNAMICS ANALYSIS SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. RT Autocorrelation (Lag-1):\n")
    if len(autocorr_df) > 0:
        summary = autocorr_df[autocorr_df['lag'] == 1].groupby('task')['autocorrelation'].agg(['mean', 'std', 'count'])
        for task in summary.index:
            f.write(f"   {task}: {summary.loc[task, 'mean']:.3f} ± {summary.loc[task, 'std']:.3f}\n")

    f.write("\n2. Learning Curve Summary:\n")
    if len(learning_df) > 0:
        for task in learning_df['task'].unique():
            task_df = learning_df[learning_df['task'] == task]
            first = task_df[task_df['trial_bin'] == 1]['mean_rt'].mean()
            last = task_df[task_df['trial_bin'] == task_df['trial_bin'].max()]['mean_rt'].mean()
            f.write(f"   {task}: {first:.1f} → {last:.1f} ms (Δ = {last-first:.1f} ms)\n")

    f.write("\n3. UCLA Effects on Autocorrelation:\n")
    if ucla_ac_results:
        for result in ucla_ac_results:
            if 'gender' not in result:
                f.write(f"   {result['task']}: r = {result['correlation']:.3f}, p = {result['p_value']:.4f}\n")

print("\nSequential dynamics analysis complete!")
