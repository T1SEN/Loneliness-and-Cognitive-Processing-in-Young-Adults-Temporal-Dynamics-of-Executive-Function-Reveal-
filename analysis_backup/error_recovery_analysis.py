#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Recovery Analysis

Tests whether errors disrupt subsequent performance differently for lonely individuals:
1. Post-error accuracy (not just RT slowing)
2. Post-error RT variability (does control destabilize?)
3. Error cascades (do errors cluster?)
4. Gender moderation of error recovery
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

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/error_recovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ERROR RECOVERY ANALYSIS")
print("Post-Error Accuracy, RT Variability, and Control Destabilization")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1] Loading data...")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})

stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
# Already has participant_id column

wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
# Already has participant_id column

print(f"  Loaded {len(participants)} participants")
print(f"  Loaded {len(stroop_trials)} Stroop trials")
print(f"  Loaded {len(wcst_trials)} WCST trials")

# ============================================================================
# 2. EXTRACT UCLA & DEMOGRAPHICS
# ============================================================================

print("\n[2] Extracting UCLA loneliness scores...")

ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores = ucla_scores.rename(columns={'score': 'ucla_total'})
ucla_scores = ucla_scores.dropna(subset=['ucla_total'])

# Merge gender
ucla_scores = ucla_scores.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

# Recode gender from Korean to English
ucla_scores['gender'] = ucla_scores['gender'].map({'남성': 'male', '여성': 'female'})

# Standardize
scaler = StandardScaler()
ucla_scores['z_ucla'] = scaler.fit_transform(ucla_scores[['ucla_total']])

print(f"  UCLA scores for {len(ucla_scores)} participants")

# ============================================================================
# 3. COMPUTE POST-ERROR METRICS
# ============================================================================

print("\n[3] Computing post-error metrics...")

def compute_post_error_metrics(df, task_name):
    """
    Compute post-error accuracy, RT, and RT variability

    For each trial, flag if previous trial was an error
    """

    print(f"\n  {task_name.upper()} TASK")
    print("  " + "-"*70)

    # Filter valid trials
    if task_name == 'Stroop':
        df_valid = df[
            (df['timeout'] == False) &
            (df['rt_ms'] > 200) &
            (df['rt_ms'] < 5000) &
            (df['correct'].notna())
        ].copy()
    elif task_name == 'WCST':
        df_valid = df[
            (df['timeout'] == False) &
            (df['rt_ms'] > 200) &
            (df['rt_ms'] < 10000) &
            (df['correct'].notna())
        ].copy()
    else:
        return None

    # Sort by participant and trial
    trial_col = 'trial' if task_name == 'Stroop' else 'trial_index'
    df_valid = df_valid.sort_values(['participant_id', trial_col]).reset_index(drop=True)

    # Create previous trial error flag
    df_valid['prev_error'] = False
    df_valid['prev_correct'] = False

    for pid in df_valid['participant_id'].unique():
        mask = df_valid['participant_id'] == pid
        correctness = df_valid.loc[mask, 'correct'].values

        # Shift by 1 to get previous trial
        prev_correctness = np.roll(correctness, 1)
        prev_correctness[0] = np.nan  # First trial has no previous

        df_valid.loc[mask, 'prev_error'] = (prev_correctness == False)
        df_valid.loc[mask, 'prev_correct'] = (prev_correctness == True)

    # Remove first trial of each participant (no previous trial)
    df_valid = df_valid[~df_valid['prev_error'].isna()].copy()

    print(f"    Valid trials: {len(df_valid)}")
    print(f"    Post-error trials: {df_valid['prev_error'].sum()}")
    print(f"    Post-correct trials: {df_valid['prev_correct'].sum()}")

    # Merge UCLA & gender
    df_valid = df_valid.merge(ucla_scores[['participant_id', 'z_ucla', 'gender']], on='participant_id', how='left')

    return df_valid

stroop_post_error = compute_post_error_metrics(stroop_trials, 'Stroop')
wcst_post_error = compute_post_error_metrics(wcst_trials, 'WCST')

# ============================================================================
# 4. ANALYSIS: POST-ERROR ACCURACY ~ UCLA × GENDER
# ============================================================================

print("\n[4] Analyzing post-error accuracy...")

def analyze_post_error_accuracy(df, task_name):
    """Test if errors disrupt accuracy differently for lonely individuals"""

    if df is None or len(df) == 0:
        return None

    print(f"\n  {task_name.upper()} - Post-Error Accuracy")
    print("  " + "-"*70)

    # Aggregate per participant
    participant_metrics = []

    for pid, group in df.groupby('participant_id'):
        post_error_trials = group[group['prev_error'] == True]
        post_correct_trials = group[group['prev_correct'] == True]

        if len(post_error_trials) < 3 or len(post_correct_trials) < 3:
            continue

        post_error_acc = post_error_trials['correct'].mean()
        post_correct_acc = post_correct_trials['correct'].mean()
        accuracy_drop = post_correct_acc - post_error_acc  # Positive = worse after errors

        ucla = group['z_ucla'].iloc[0]
        gender = group['gender'].iloc[0]

        participant_metrics.append({
            'participant_id': pid,
            'post_error_accuracy': post_error_acc,
            'post_correct_accuracy': post_correct_acc,
            'accuracy_drop': accuracy_drop,
            'z_ucla': ucla,
            'gender': gender
        })

    metrics_df = pd.DataFrame(participant_metrics)

    # Overall correlation
    r_overall, p_overall = stats.pearsonr(metrics_df['z_ucla'], metrics_df['accuracy_drop'])

    print(f"    N participants: {len(metrics_df)}")
    print(f"    Mean accuracy drop: {metrics_df['accuracy_drop'].mean():.4f} (SD={metrics_df['accuracy_drop'].std():.4f})")
    print(f"    UCLA → Accuracy drop:  r={r_overall:.3f}, p={p_overall:.4f}")

    if p_overall < 0.05:
        interpretation = "Lonelier individuals show GREATER post-error disruption" if r_overall > 0 else "Lonelier individuals show BETTER error recovery"
        print(f"    *** SIGNIFICANT: {interpretation} ***")

    # Gender stratified
    print(f"\n    Gender-Stratified Analysis:")
    for gender in ['male', 'female']:
        subset = metrics_df[metrics_df['gender'] == gender]
        if len(subset) > 10:
            r, p = stats.pearsonr(subset['z_ucla'], subset['accuracy_drop'])
            print(f"      {gender.capitalize():8s} (N={len(subset):2d}):  r={r:>6.3f}, p={p:.4f}")

    # Regression: accuracy_drop ~ UCLA * Gender
    metrics_df['gender_coded'] = (metrics_df['gender'] == 'male').astype(int)
    metrics_df['ucla_x_gender'] = metrics_df['z_ucla'] * metrics_df['gender_coded']

    X = metrics_df[['z_ucla', 'gender_coded', 'ucla_x_gender']].values
    X = np.column_stack([np.ones(len(X)), X])
    Y = metrics_df['accuracy_drop'].values

    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]

    print(f"\n    Regression: Accuracy Drop ~ UCLA × Gender")
    print(f"      UCLA main effect:       β={coeffs[1]:.4f}")
    print(f"      Gender main effect:     β={coeffs[2]:.4f}")
    print(f"      UCLA × Gender:          β={coeffs[3]:.4f}")

    return metrics_df

stroop_error_acc = analyze_post_error_accuracy(stroop_post_error, 'Stroop')
wcst_error_acc = analyze_post_error_accuracy(wcst_post_error, 'WCST')

# ============================================================================
# 5. ANALYSIS: POST-ERROR RT VARIABILITY
# ============================================================================

print("\n[5] Analyzing post-error RT variability...")

def analyze_post_error_rt_variability(df, task_name):
    """Test if errors increase RT variability for lonely individuals"""

    if df is None or len(df) == 0:
        return None

    print(f"\n  {task_name.upper()} - Post-Error RT Variability")
    print("  " + "-"*70)

    participant_metrics = []

    for pid, group in df.groupby('participant_id'):
        post_error_trials = group[group['prev_error'] == True]
        post_correct_trials = group[group['prev_correct'] == True]

        if len(post_error_trials) < 3 or len(post_correct_trials) < 3:
            continue

        post_error_rt_sd = post_error_trials['rt_ms'].std()
        post_correct_rt_sd = post_correct_trials['rt_ms'].std()
        variability_increase = post_error_rt_sd - post_correct_rt_sd

        ucla = group['z_ucla'].iloc[0]
        gender = group['gender'].iloc[0]

        participant_metrics.append({
            'participant_id': pid,
            'post_error_rt_sd': post_error_rt_sd,
            'post_correct_rt_sd': post_correct_rt_sd,
            'variability_increase': variability_increase,
            'z_ucla': ucla,
            'gender': gender
        })

    metrics_df = pd.DataFrame(participant_metrics)

    r_overall, p_overall = stats.pearsonr(metrics_df['z_ucla'], metrics_df['variability_increase'])

    print(f"    N participants: {len(metrics_df)}")
    print(f"    Mean variability increase: {metrics_df['variability_increase'].mean():.2f} ms (SD={metrics_df['variability_increase'].std():.2f})")
    print(f"    UCLA → Variability increase:  r={r_overall:.3f}, p={p_overall:.4f}")

    if p_overall < 0.05:
        print(f"    *** SIGNIFICANT: Lonelier individuals show {'GREATER' if r_overall > 0 else 'LESS'} RT destabilization after errors ***")

    # Gender stratified
    print(f"\n    Gender-Stratified:")
    for gender in ['male', 'female']:
        subset = metrics_df[metrics_df['gender'] == gender]
        if len(subset) > 10:
            r, p = stats.pearsonr(subset['z_ucla'], subset['variability_increase'])
            print(f"      {gender.capitalize():8s} (N={len(subset):2d}):  r={r:>6.3f}, p={p:.4f}")

    return metrics_df

stroop_error_var = analyze_post_error_rt_variability(stroop_post_error, 'Stroop')
wcst_error_var = analyze_post_error_rt_variability(wcst_post_error, 'WCST')

# ============================================================================
# 6. ERROR CASCADES: Do errors cluster?
# ============================================================================

print("\n[6] Analyzing error cascades...")

def analyze_error_cascades(df, task_name):
    """Test if errors are followed by more errors (loss of control)"""

    if df is None or len(df) == 0:
        return None

    print(f"\n  {task_name.upper()} - Error Cascades")
    print("  " + "-"*70)

    participant_metrics = []

    for pid, group in df.groupby('participant_id'):
        post_error_trials = group[group['prev_error'] == True]
        post_correct_trials = group[group['prev_correct'] == True]

        if len(post_error_trials) < 3 or len(post_correct_trials) < 3:
            continue

        # Error rate after errors vs. after correct
        error_after_error = (post_error_trials['correct'] == False).mean()
        error_after_correct = (post_correct_trials['correct'] == False).mean()
        cascade_effect = error_after_error - error_after_correct

        ucla = group['z_ucla'].iloc[0]
        gender = group['gender'].iloc[0]

        participant_metrics.append({
            'participant_id': pid,
            'error_after_error': error_after_error,
            'error_after_correct': error_after_correct,
            'cascade_effect': cascade_effect,
            'z_ucla': ucla,
            'gender': gender
        })

    metrics_df = pd.DataFrame(participant_metrics)

    r_overall, p_overall = stats.pearsonr(metrics_df['z_ucla'], metrics_df['cascade_effect'])

    print(f"    N participants: {len(metrics_df)}")
    print(f"    Mean cascade effect: {metrics_df['cascade_effect'].mean():.4f} (SD={metrics_df['cascade_effect'].std():.4f})")
    print(f"    UCLA → Cascade effect:  r={r_overall:.3f}, p={p_overall:.4f}")

    if p_overall < 0.05:
        print(f"    *** SIGNIFICANT: Lonelier individuals show {'GREATER' if r_overall > 0 else 'LESS'} error clustering ***")

    # Gender stratified
    print(f"\n    Gender-Stratified:")
    for gender in ['male', 'female']:
        subset = metrics_df[metrics_df['gender'] == gender]
        if len(subset) > 10:
            r, p = stats.pearsonr(subset['z_ucla'], subset['cascade_effect'])
            print(f"      {gender.capitalize():8s} (N={len(subset):2d}):  r={r:>6.3f}, p={p:.4f}")

    return metrics_df

stroop_cascades = analyze_error_cascades(stroop_post_error, 'Stroop')
wcst_cascades = analyze_error_cascades(wcst_post_error, 'WCST')

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print("\n[7] Saving results...")

if stroop_error_acc is not None:
    stroop_error_acc.to_csv(OUTPUT_DIR / "stroop_post_error_accuracy.csv", index=False, encoding='utf-8-sig')
if wcst_error_acc is not None:
    wcst_error_acc.to_csv(OUTPUT_DIR / "wcst_post_error_accuracy.csv", index=False, encoding='utf-8-sig')

if stroop_error_var is not None:
    stroop_error_var.to_csv(OUTPUT_DIR / "stroop_post_error_variability.csv", index=False, encoding='utf-8-sig')
if wcst_error_var is not None:
    wcst_error_var.to_csv(OUTPUT_DIR / "wcst_post_error_variability.csv", index=False, encoding='utf-8-sig')

if stroop_cascades is not None:
    stroop_cascades.to_csv(OUTPUT_DIR / "stroop_error_cascades.csv", index=False, encoding='utf-8-sig')
if wcst_cascades is not None:
    wcst_cascades.to_csv(OUTPUT_DIR / "wcst_error_cascades.csv", index=False, encoding='utf-8-sig')

print("✓ Saved all CSV files")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n[8] Creating visualizations...")

def plot_error_recovery_comparison(acc_df, var_df, cascade_df, task_name):
    """Create comprehensive error recovery visualization"""

    if acc_df is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{task_name} Task: Error Recovery Patterns by UCLA Loneliness', fontsize=16, fontweight='bold')

    # Row 1: By gender
    # Panel 1: Accuracy drop by gender
    ax = axes[0, 0]
    for gender in ['male', 'female']:
        subset = acc_df[acc_df['gender'] == gender]
        ax.scatter(subset['z_ucla'], subset['accuracy_drop'], alpha=0.6, s=80, label=gender.capitalize(), edgecolors='black', linewidths=0.5)

        # Fit line
        if len(subset) > 5:
            z = np.polyfit(subset['z_ucla'], subset['accuracy_drop'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
            ax.plot(x_line, p(x_line), linestyle='--', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('UCLA (z-scored)')
    ax.set_ylabel('Accuracy Drop (Post-Correct - Post-Error)')
    ax.set_title('Post-Error Accuracy Disruption')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: RT variability increase
    if var_df is not None:
        ax = axes[0, 1]
        for gender in ['male', 'female']:
            subset = var_df[var_df['gender'] == gender]
            ax.scatter(subset['z_ucla'], subset['variability_increase'], alpha=0.6, s=80, label=gender.capitalize(), edgecolors='black', linewidths=0.5)

            if len(subset) > 5:
                z = np.polyfit(subset['z_ucla'], subset['variability_increase'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
                ax.plot(x_line, p(x_line), linestyle='--', linewidth=2)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('UCLA (z-scored)')
        ax.set_ylabel('RT Variability Increase (ms SD)')
        ax.set_title('Post-Error RT Destabilization')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: Error cascades
    if cascade_df is not None:
        ax = axes[0, 2]
        for gender in ['male', 'female']:
            subset = cascade_df[cascade_df['gender'] == gender]
            ax.scatter(subset['z_ucla'], subset['cascade_effect'], alpha=0.6, s=80, label=gender.capitalize(), edgecolors='black', linewidths=0.5)

            if len(subset) > 5:
                z = np.polyfit(subset['z_ucla'], subset['cascade_effect'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset['z_ucla'].min(), subset['z_ucla'].max(), 100)
                ax.plot(x_line, p(x_line), linestyle='--', linewidth=2)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('UCLA (z-scored)')
        ax.set_ylabel('Cascade Effect (Error Rate Diff)')
        ax.set_title('Error Clustering')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Distributions
    # Panel 4: Accuracy drop distribution
    ax = axes[1, 0]
    acc_df['ucla_group'] = pd.cut(acc_df['z_ucla'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=['Low', 'Medium', 'High'])
    acc_df.boxplot(column='accuracy_drop', by='ucla_group', ax=ax)
    ax.set_xlabel('UCLA Group')
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('Accuracy Drop by UCLA Group')
    plt.sca(ax)
    plt.xticks(rotation=0)

    # Panel 5: Variability increase distribution
    if var_df is not None:
        ax = axes[1, 1]
        var_df['ucla_group'] = pd.cut(var_df['z_ucla'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=['Low', 'Medium', 'High'])
        var_df.boxplot(column='variability_increase', by='ucla_group', ax=ax)
        ax.set_xlabel('UCLA Group')
        ax.set_ylabel('RT Variability Increase (ms)')
        ax.set_title('RT Variability by UCLA Group')
        plt.sca(ax)
        plt.xticks(rotation=0)

    # Panel 6: Summary bars
    if cascade_df is not None:
        ax = axes[1, 2]

        # Compute correlations
        r_acc, _ = stats.pearsonr(acc_df['z_ucla'], acc_df['accuracy_drop'])
        r_var, _ = stats.pearsonr(var_df['z_ucla'], var_df['variability_increase']) if var_df is not None else (0, 1)
        r_casc, _ = stats.pearsonr(cascade_df['z_ucla'], cascade_df['cascade_effect']) if cascade_df is not None else (0, 1)

        metrics = ['Accuracy\nDrop', 'RT\nVariability', 'Error\nCascade']
        correlations = [r_acc, r_var, r_casc]

        colors = ['red' if r > 0 else 'blue' for r in correlations]
        ax.bar(metrics, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Correlation with UCLA')
        ax.set_title('Summary: UCLA Correlations')
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{task_name.lower()}_error_recovery_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {task_name.lower()}_error_recovery_comprehensive.png")

plot_error_recovery_comparison(stroop_error_acc, stroop_error_var, stroop_cascades, 'Stroop')
plot_error_recovery_comparison(wcst_error_acc, wcst_error_var, wcst_cascades, 'WCST')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ERROR RECOVERY ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")

# Compile summary
summary_data = []

for task, acc_df, var_df, casc_df in [
    ('Stroop', stroop_error_acc, stroop_error_var, stroop_cascades),
    ('WCST', wcst_error_acc, wcst_error_var, wcst_cascades)
]:
    if acc_df is not None:
        r_acc, p_acc = stats.pearsonr(acc_df['z_ucla'], acc_df['accuracy_drop'])
        r_var, p_var = stats.pearsonr(var_df['z_ucla'], var_df['variability_increase']) if var_df is not None else (np.nan, np.nan)
        r_casc, p_casc = stats.pearsonr(casc_df['z_ucla'], casc_df['cascade_effect']) if casc_df is not None else (np.nan, np.nan)

        print(f"\n  {task.upper()}:")
        print(f"    Post-Error Accuracy Drop:     r={r_acc:>6.3f}, p={p_acc:.4f} {'***' if p_acc < 0.05 else 'ns'}")
        print(f"    Post-Error RT Variability:    r={r_var:>6.3f}, p={p_var:.4f} {'***' if p_var < 0.05 else 'ns'}")
        print(f"    Error Cascade Effect:         r={r_casc:>6.3f}, p={p_casc:.4f} {'***' if p_casc < 0.05 else 'ns'}")

        summary_data.append({
            'task': task,
            'accuracy_drop_r': r_acc,
            'accuracy_drop_p': p_acc,
            'rt_variability_r': r_var,
            'rt_variability_p': p_var,
            'cascade_r': r_casc,
            'cascade_p': p_casc
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(OUTPUT_DIR / "error_recovery_summary.csv", index=False, encoding='utf-8-sig')

print("\nOUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}/")
print("  - *_post_error_accuracy.csv")
print("  - *_post_error_variability.csv")
print("  - *_error_cascades.csv")
print("  - *_error_recovery_comprehensive.png")
print("  - error_recovery_summary.csv")

print("\n" + "="*80)
