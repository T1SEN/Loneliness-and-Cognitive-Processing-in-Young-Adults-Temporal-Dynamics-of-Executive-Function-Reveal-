"""
WCST Post-Error Adaptation (Quick Analysis)

Research Question:
Do lonely males show impaired post-error adaptation (reduced slowing, more errors after errors)?

Metrics:
- Post-error slowing: RT(n+1|error_n) - RT(n+1|correct_n)
- Post-error accuracy: Accuracy on trial n+1 after error vs correct
- UCLA × Gender moderation
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
import ast

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/post_error_adaptation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WCST POST-ERROR ADAPTATION")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)
participants['gender_male'] = (participants['gender'] == 'male').astype(int)
participants = participants.rename(columns={'participantId': 'participant_id'})

surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})
ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()
ucla_data['ucla_total'] = pd.to_numeric(ucla_data['score'], errors='coerce')
ucla_data = ucla_data[['participant_id', 'ucla_total']].dropna()

wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
if 'participantId' in wcst_trials.columns and 'participant_id' not in wcst_trials.columns:
    wcst_trials.rename(columns={'participantId': 'participant_id'}, inplace=True)
elif 'participantId' in wcst_trials.columns and 'participant_id' in wcst_trials.columns:
    wcst_trials.drop(columns=['participantId'], inplace=True)

wcst_trials['rt_valid'] = wcst_trials['reactionTimeMs'] > 0
wcst_trials = wcst_trials.sort_values(['participant_id', 'trialIndex'])

print(f"  Loaded {len(wcst_trials):,} trials")

# Compute post-error metrics
print("\n[2/4] Computing post-error adaptation...")

adaptation_metrics = []

for pid, grp in wcst_trials.groupby('participant_id'):
    grp = grp.reset_index(drop=True)
    is_correct = grp['correct'].values
    rts = grp['reactionTimeMs'].values
    rt_valid = grp['rt_valid'].values

    # Identify post-error and post-correct trials
    post_error_rts = []
    post_correct_rts = []
    post_error_accuracy = []
    post_correct_accuracy = []

    for i in range(len(grp) - 1):
        if not rt_valid[i] or not rt_valid[i+1]:
            continue

        if is_correct[i] == False:  # Error on trial i
            post_error_rts.append(rts[i+1])
            post_error_accuracy.append(is_correct[i+1])
        elif is_correct[i] == True:  # Correct on trial i
            post_correct_rts.append(rts[i+1])
            post_correct_accuracy.append(is_correct[i+1])

    # Compute metrics
    if len(post_error_rts) >= 3 and len(post_correct_rts) >= 3:
        post_error_slowing = np.mean(post_error_rts) - np.mean(post_correct_rts)
        post_error_acc = np.mean(post_error_accuracy) if post_error_accuracy else np.nan
        post_correct_acc = np.mean(post_correct_accuracy) if post_correct_accuracy else np.nan

        adaptation_metrics.append({
            'participant_id': pid,
            'post_error_slowing': post_error_slowing,
            'post_error_accuracy': post_error_acc,
            'post_correct_accuracy': post_correct_acc,
            'post_error_n': len(post_error_rts),
            'post_correct_n': len(post_correct_rts)
        })

adaptation_df = pd.DataFrame(adaptation_metrics)
print(f"  Computed adaptation metrics for {len(adaptation_df)} participants")

# Merge
master = adaptation_df.merge(ucla_data, on='participant_id', how='inner')
master = master.merge(participants[['participant_id', 'gender_male']], on='participant_id', how='inner')

print(f"\n  Complete cases: N={len(master)}")

# Test correlations
print("\n[3/4] Testing UCLA × adaptation correlations...")

for gender, label in [(1, 'Male'), (0, 'Female')]:
    subset = master[master['gender_male'] == gender]
    print(f"\n  {label} (N={len(subset)}):")

    if len(subset) >= 10:
        for metric in ['post_error_slowing', 'post_error_accuracy']:
            valid = subset[['ucla_total', metric]].dropna()
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid['ucla_total'], valid[metric])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"    UCLA × {metric}: r={r:.3f}, p={p:.3f} {sig}")

# Visualization
print("\n[4/4] Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for gender, label, marker, color in [(1, 'Male', 's', '#3498DB'), (0, 'Female', 'o', '#E74C3C')]:
    subset = master[master['gender_male'] == gender]

    # Post-error slowing
    axes[0].scatter(subset['ucla_total'], subset['post_error_slowing'],
                   alpha=0.6, label=label, marker=marker, s=80, color=color)

    # Post-error accuracy
    axes[1].scatter(subset['ucla_total'], subset['post_error_accuracy'] * 100,
                   alpha=0.6, label=label, marker=marker, s=80, color=color)

axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='No slowing')
axes[0].set_xlabel('UCLA Loneliness')
axes[0].set_ylabel('Post-Error Slowing (ms)')
axes[0].set_title('Post-Error RT Adaptation')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('UCLA Loneliness')
axes[1].set_ylabel('Post-Error Accuracy (%)')
axes[1].set_title('Accuracy After Errors')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "post_error_adaptation.png", dpi=300, bbox_inches='tight')
plt.close()

# Save results
master.to_csv(OUTPUT_DIR / "post_error_metrics.csv", index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print("✓ WCST Post-Error Adaptation Complete!")
print("="*80)
