#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-Session Dynamics Analysis

Tests whether loneliness predicts temporal patterns across trials:
1. Fatigue: Steeper RT increases over trials (cognitive resource depletion)
2. Learning deficit: Slower improvement/adaptation
3. Task-specific dynamics (Stroop, WCST, PRP)

Uses mixed-effects models with trial_number × UCLA interactions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/dynamics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WITHIN-SESSION DYNAMICS ANALYSIS")
print("Trial-Level Fatigue, Learning, and Performance Trajectories")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1] Loading data...")

# Load participants
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

# Load surveys
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})

# Load trial data (all already have participant_id column)
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")

wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")

print(f"  Loaded {len(participants)} participants")
print(f"  Loaded {len(stroop_trials)} Stroop trials")
print(f"  Loaded {len(wcst_trials)} WCST trials")
print(f"  Loaded {len(prp_trials)} PRP trials")

# ============================================================================
# 2. EXTRACT UCLA SCORES
# ============================================================================

print("\n[2] Extracting UCLA loneliness scores...")

ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores = ucla_scores.rename(columns={'score': 'ucla_total'})
ucla_scores = ucla_scores.dropna(subset=['ucla_total'])

# Standardize
scaler = StandardScaler()
ucla_scores['z_ucla'] = scaler.fit_transform(ucla_scores[['ucla_total']])

print(f"  UCLA scores for {len(ucla_scores)} participants")

# ============================================================================
# 3. PREPARE TRIAL-LEVEL DATA
# ============================================================================

print("\n[3] Preparing trial-level datasets...")

# --- STROOP ---
stroop_valid = stroop_trials[
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 200) &
    (stroop_trials['rt_ms'] < 5000)
].copy()

# Add trial number within participant
stroop_valid = stroop_valid.sort_values(['participant_id', 'trial'])
stroop_valid['trial_number'] = stroop_valid['trial'].astype(int)
stroop_valid['trial_number_normalized'] = stroop_valid.groupby('participant_id')['trial_number'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

# Merge UCLA
stroop_valid = stroop_valid.merge(ucla_scores[['participant_id', 'z_ucla']], on='participant_id', how='left')

print(f"  Stroop: {len(stroop_valid)} valid trials from {stroop_valid['participant_id'].nunique()} participants")

# --- WCST ---
def parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_parsed'] = wcst_trials['extra'].apply(parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_parsed'].apply(lambda x: x.get('isPE', False))

wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['rt_ms'] > 200) &
    (wcst_trials['rt_ms'] < 10000)
].copy()

wcst_valid = wcst_valid.sort_values(['participant_id', 'trial_index'])
wcst_valid['trial_number'] = wcst_valid['trial_index'].astype(int)
wcst_valid['trial_number_normalized'] = wcst_valid.groupby('participant_id')['trial_number'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

wcst_valid = wcst_valid.merge(ucla_scores[['participant_id', 'z_ucla']], on='participant_id', how='left')

print(f"  WCST: {len(wcst_valid)} valid trials from {wcst_valid['participant_id'].nunique()} participants")

# --- PRP ---
prp_valid = prp_trials[
    (prp_trials['t1_timeout'] == False) &
    (prp_trials['t2_timeout'] == False) &
    (prp_trials['t1_rt_ms'] > 200) &
    (prp_trials['t2_rt_ms'] > 200)
].copy()

prp_valid = prp_valid.sort_values(['participant_id', 'trial_index'])
prp_valid['trial_number'] = prp_valid['trial_index'].astype(int)
prp_valid['trial_number_normalized'] = prp_valid.groupby('participant_id')['trial_number'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

prp_valid = prp_valid.merge(ucla_scores[['participant_id', 'z_ucla']], on='participant_id', how='left')

print(f"  PRP: {len(prp_valid)} valid trials from {prp_valid['participant_id'].nunique()} participants")

# ============================================================================
# 4. MIXED-EFFECTS MODELS: RT ~ trial_number * UCLA
# ============================================================================

print("\n[4] Fitting mixed-effects models...")
print("  Note: Using linear regression with participant-level aggregation")
print("  (Mixed-effects models require statsmodels or R integration)")

def analyze_trial_dynamics(df, rt_col, task_name):
    """
    Analyze trial dynamics using participant-level slopes

    For each participant:
    1. Fit RT ~ trial_number
    2. Extract slope (fatigue/learning)
    3. Test: slope ~ UCLA
    """

    print(f"\n  {task_name.upper()} TASK")
    print("  " + "-"*70)

    # Compute per-participant slopes
    participant_slopes = []

    for pid, group in df.groupby('participant_id'):
        if len(group) < 10:  # Need minimum trials
            continue

        X = group['trial_number'].values
        Y = group[rt_col].values

        # Fit linear trend: RT ~ trial_number
        X_mat = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
            slope = coeffs[1]  # ms per trial
            intercept = coeffs[0]

            # Get UCLA
            ucla = group['z_ucla'].iloc[0]

            participant_slopes.append({
                'participant_id': pid,
                'slope': slope,
                'intercept': intercept,
                'z_ucla': ucla,
                'n_trials': len(group)
            })
        except:
            continue

    slopes_df = pd.DataFrame(participant_slopes)

    # Test: slope ~ UCLA
    if len(slopes_df) > 0:
        r, p = stats.pearsonr(slopes_df['z_ucla'], slopes_df['slope'])

        # Also run regression for coefficient
        X_reg = np.column_stack([np.ones(len(slopes_df)), slopes_df['z_ucla']])
        Y_reg = slopes_df['slope']
        coeffs_reg = np.linalg.lstsq(X_reg, Y_reg, rcond=None)[0]
        beta = coeffs_reg[1]  # UCLA → slope coefficient

        print(f"    N participants: {len(slopes_df)}")
        print(f"    Mean slope: {slopes_df['slope'].mean():.4f} ms/trial (SD={slopes_df['slope'].std():.4f})")
        print(f"    Slope range: [{slopes_df['slope'].min():.4f}, {slopes_df['slope'].max():.4f}]")
        print(f"")
        print(f"    UCLA → Slope correlation:  r = {r:.3f}, p = {p:.4f}")
        print(f"    Regression coefficient:    β = {beta:.4f} ms/trial per SD of UCLA")

        if p < 0.05:
            direction = "FATIGUE" if beta > 0 else "LEARNING/ADAPTATION"
            print(f"    *** SIGNIFICANT {direction} EFFECT ***")
        else:
            print(f"    No significant relationship")

        return slopes_df, r, p, beta
    else:
        print("    Insufficient data")
        return None, None, None, None

# Analyze each task
stroop_slopes, stroop_r, stroop_p, stroop_beta = analyze_trial_dynamics(
    stroop_valid, 'rt_ms', 'Stroop'
)

wcst_slopes, wcst_r, wcst_p, wcst_beta = analyze_trial_dynamics(
    wcst_valid, 'rt_ms', 'WCST'
)

prp_t2_slopes, prp_r, prp_p, prp_beta = analyze_trial_dynamics(
    prp_valid, 't2_rt_ms', 'PRP (T2)'
)

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

print("\n[5] Saving results...")

summary_results = pd.DataFrame([
    {
        'task': 'Stroop',
        'rt_variable': 'rt_ms',
        'n_participants': len(stroop_slopes) if stroop_slopes is not None else 0,
        'mean_slope': stroop_slopes['slope'].mean() if stroop_slopes is not None else np.nan,
        'sd_slope': stroop_slopes['slope'].std() if stroop_slopes is not None else np.nan,
        'ucla_slope_r': stroop_r,
        'ucla_slope_p': stroop_p,
        'ucla_slope_beta': stroop_beta,
        'interpretation': 'Fatigue' if stroop_beta and stroop_beta > 0 else 'Learning' if stroop_beta and stroop_beta < 0 else 'None'
    },
    {
        'task': 'WCST',
        'rt_variable': 'rt_ms',
        'n_participants': len(wcst_slopes) if wcst_slopes is not None else 0,
        'mean_slope': wcst_slopes['slope'].mean() if wcst_slopes is not None else np.nan,
        'sd_slope': wcst_slopes['slope'].std() if wcst_slopes is not None else np.nan,
        'ucla_slope_r': wcst_r,
        'ucla_slope_p': wcst_p,
        'ucla_slope_beta': wcst_beta,
        'interpretation': 'Fatigue' if wcst_beta and wcst_beta > 0 else 'Learning' if wcst_beta and wcst_beta < 0 else 'None'
    },
    {
        'task': 'PRP_T2',
        'rt_variable': 't2_rt_ms',
        'n_participants': len(prp_t2_slopes) if prp_t2_slopes is not None else 0,
        'mean_slope': prp_t2_slopes['slope'].mean() if prp_t2_slopes is not None else np.nan,
        'sd_slope': prp_t2_slopes['slope'].std() if prp_t2_slopes is not None else np.nan,
        'ucla_slope_r': prp_r,
        'ucla_slope_p': prp_p,
        'ucla_slope_beta': prp_beta,
        'interpretation': 'Fatigue' if prp_beta and prp_beta > 0 else 'Learning' if prp_beta and prp_beta < 0 else 'None'
    }
])

summary_results.to_csv(OUTPUT_DIR / "trial_dynamics_summary.csv", index=False, encoding='utf-8-sig')
print("✓ Saved: trial_dynamics_summary.csv")

# Save individual participant slopes
if stroop_slopes is not None:
    stroop_slopes.to_csv(OUTPUT_DIR / "stroop_participant_slopes.csv", index=False, encoding='utf-8-sig')
if wcst_slopes is not None:
    wcst_slopes.to_csv(OUTPUT_DIR / "wcst_participant_slopes.csv", index=False, encoding='utf-8-sig')
if prp_t2_slopes is not None:
    prp_t2_slopes.to_csv(OUTPUT_DIR / "prp_participant_slopes.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# 6. VISUALIZATION: RT TRAJECTORIES BY UCLA QUARTILE
# ============================================================================

print("\n[6] Creating visualizations...")

def plot_trajectories(df, rt_col, task_name, slopes_df, output_name):
    """Plot RT trajectories by UCLA quartile"""

    if slopes_df is None or len(slopes_df) == 0:
        return

    # Merge slopes to get quartiles
    df = df.merge(slopes_df[['participant_id', 'z_ucla']], on='participant_id', how='inner', suffixes=('', '_from_slopes'))
    df['ucla_quartile'] = pd.qcut(df['z_ucla_from_slopes'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

    # Compute mean RT by trial number and quartile
    trajectory_data = (
        df.groupby(['trial_number', 'ucla_quartile'])[rt_col]
        .mean()
        .reset_index()
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{task_name} Task: Within-Session Dynamics by UCLA Loneliness', fontsize=14, fontweight='bold')

    # Panel 1: Trajectories
    for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        subset = trajectory_data[trajectory_data['ucla_quartile'] == quartile]
        ax1.plot(subset['trial_number'], subset[rt_col], marker='o', label=quartile, alpha=0.7, linewidth=2)

    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Mean RT (ms)', fontsize=12)
    ax1.set_title('RT Trajectories Across Trials', fontsize=12, fontweight='bold')
    ax1.legend(title='UCLA Quartile', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Slopes vs UCLA scatterplot
    ax2.scatter(slopes_df['z_ucla'], slopes_df['slope'], alpha=0.6, s=100, edgecolors='black', linewidths=0.5)

    # Fit line
    z = np.polyfit(slopes_df['z_ucla'], slopes_df['slope'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(slopes_df['z_ucla'].min(), slopes_df['z_ucla'].max(), 100)
    ax2.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit line (β={z[0]:.4f})')

    # Add horizontal line at y=0 (no change)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('UCLA Loneliness (z-scored)', fontsize=12)
    ax2.set_ylabel('RT Slope (ms/trial)', fontsize=12)
    ax2.set_title(f'Individual Slopes vs. UCLA\n(r={slopes_df["z_ucla"].corr(slopes_df["slope"]):.3f}, p={summary_results[summary_results["task"]==task_name.split()[0]]["ucla_slope_p"].iloc[0]:.4f})',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_name}")

# Generate plots
plot_trajectories(stroop_valid, 'rt_ms', 'Stroop', stroop_slopes, 'stroop_trajectories.png')
plot_trajectories(wcst_valid, 'rt_ms', 'WCST', wcst_slopes, 'wcst_trajectories.png')
plot_trajectories(prp_valid, 't2_rt_ms', 'PRP T2', prp_t2_slopes, 'prp_trajectories.png')

# ============================================================================
# 7. ACCURACY TRAJECTORIES (for tasks with correctness)
# ============================================================================

print("\n[7] Analyzing accuracy trajectories...")

def analyze_accuracy_dynamics(df, task_name):
    """Analyze accuracy changes across trials"""

    if 'correct' not in df.columns:
        print(f"  {task_name}: No accuracy data available")
        return None, None, None

    print(f"\n  {task_name.upper()} TASK - Accuracy Dynamics")
    print("  " + "-"*70)

    # Compute per-participant accuracy slopes
    participant_acc_slopes = []

    for pid, group in df.groupby('participant_id'):
        if len(group) < 10:
            continue

        X = group['trial_number'].values
        Y = group['correct'].astype(float).values

        # Fit: accuracy ~ trial_number
        X_mat = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
            slope = coeffs[1]  # change in accuracy per trial

            ucla = group['z_ucla'].iloc[0]

            participant_acc_slopes.append({
                'participant_id': pid,
                'accuracy_slope': slope,
                'z_ucla': ucla
            })
        except:
            continue

    acc_slopes_df = pd.DataFrame(participant_acc_slopes)

    if len(acc_slopes_df) > 0:
        r, p = stats.pearsonr(acc_slopes_df['z_ucla'], acc_slopes_df['accuracy_slope'])

        print(f"    N participants: {len(acc_slopes_df)}")
        print(f"    Mean accuracy slope: {acc_slopes_df['accuracy_slope'].mean():.6f} per trial")
        print(f"    UCLA → Accuracy slope:  r = {r:.3f}, p = {p:.4f}")

        if p < 0.05:
            direction = "IMPROVEMENT" if r < 0 else "DECLINE"
            print(f"    *** SIGNIFICANT {direction} EFFECT ***")

        return acc_slopes_df, r, p
    else:
        return None, None, None

stroop_acc_slopes, stroop_acc_r, stroop_acc_p = analyze_accuracy_dynamics(stroop_valid, 'Stroop')
wcst_acc_slopes, wcst_acc_r, wcst_acc_p = analyze_accuracy_dynamics(wcst_valid, 'WCST')

# ============================================================================
# 8. ERROR RATE TRAJECTORIES (WCST perseverative errors)
# ============================================================================

print("\n[8] Analyzing WCST perseverative error trajectories...")

if 'is_pe' in wcst_valid.columns:
    # Bin trials into blocks
    wcst_valid['trial_block'] = (wcst_valid['trial_number'] - 1) // 10  # Blocks of 10 trials

    # Compute PE rate by block and UCLA quartile
    wcst_valid['ucla_quartile'] = pd.qcut(wcst_valid['z_ucla'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

    pe_trajectory = (
        wcst_valid.groupby(['trial_block', 'ucla_quartile'])['is_pe']
        .mean()
        .reset_index()
        .rename(columns={'is_pe': 'pe_rate'})
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = pe_trajectory[pe_trajectory['ucla_quartile'] == quartile]
        ax.plot(subset['trial_block'], subset['pe_rate'], marker='o', label=f'UCLA {quartile}', linewidth=2)

    ax.set_xlabel('Trial Block (10 trials each)', fontsize=12)
    ax.set_ylabel('Perseverative Error Rate', fontsize=12)
    ax.set_title('WCST Perseverative Error Rate Across Trial Blocks by UCLA', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'wcst_pe_trajectories.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: wcst_pe_trajectories.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WITHIN-SESSION DYNAMICS ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")

# RT slopes
print("\n  RT SLOPE FINDINGS (Fatigue/Learning):")
for _, row in summary_results.iterrows():
    if not pd.isna(row['ucla_slope_p']):
        sig = "***" if row['ucla_slope_p'] < 0.05 else "ns"
        print(f"    • {row['task']:10s}: β={row['ucla_slope_beta']:>7.4f} ms/trial, r={row['ucla_slope_r']:>6.3f}, p={row['ucla_slope_p']:.4f} {sig}")
        if row['ucla_slope_p'] < 0.05:
            interpretation = "Lonelier individuals show FASTER FATIGUE" if row['ucla_slope_beta'] > 0 else "Lonelier individuals show SLOWER LEARNING"
            print(f"                 → {interpretation}")

# Accuracy slopes
if stroop_acc_r is not None:
    print(f"\n  ACCURACY SLOPE FINDINGS:")
    print(f"    • Stroop:  r={stroop_acc_r:.3f}, p={stroop_acc_p:.4f}")
if wcst_acc_r is not None:
    print(f"    • WCST:    r={wcst_acc_r:.3f}, p={wcst_acc_p:.4f}")

print("\nOUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}/")
print("  - trial_dynamics_summary.csv")
print("  - *_participant_slopes.csv (individual slopes)")
print("  - *_trajectories.png (visualizations)")

print("\n" + "="*80)
