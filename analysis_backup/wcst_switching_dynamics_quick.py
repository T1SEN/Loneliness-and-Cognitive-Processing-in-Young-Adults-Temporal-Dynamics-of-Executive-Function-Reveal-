"""
WCST Rule-Switching Dynamics (Quick Analysis)

Research Question:
Do lonely males show slower learning after rule changes (delayed switching)?

Metrics:
- Trials-to-criterion after each rule change
- Post-switch error rate (first 5 trials vs last 5 trials before switch)
- UCLA × Gender moderation of learning rate
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
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/wcst_switching")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WCST RULE-SWITCHING DYNAMICS")
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

# Parse extra for PE
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return False
    try:
        return ast.literal_eval(extra_str).get('isPE', False)
    except:
        return False

wcst_trials['is_pe'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials = wcst_trials.sort_values(['participant_id', 'trialIndex'])

print(f"  Loaded {len(wcst_trials):,} trials")

# Detect rule changes
print("\n[2/4] Detecting rule changes...")

switching_metrics = []

for pid, grp in wcst_trials.groupby('participant_id'):
    rules = grp['ruleAtThatTime'].values
    is_correct = grp['correct'].values
    is_pe = grp['is_pe'].values

    # Find rule change points
    rule_changes = [0]  # Start of task
    for i in range(1, len(rules)):
        if rules[i] != rules[i-1]:
            rule_changes.append(i)

    if len(rule_changes) < 2:
        continue

    # Compute post-switch metrics
    post_switch_errors = []
    trials_to_criterion = []

    for change_idx in rule_changes[1:]:  # Skip first (task start)
        # Post-switch window (next 10 trials)
        window_end = min(change_idx + 10, len(grp))
        post_switch_trials = is_correct[change_idx:window_end]

        if len(post_switch_trials) >= 5:
            post_switch_errors.append(1 - post_switch_trials[:5].mean())  # Error rate in first 5

            # Trials to reach 5 consecutive correct
            criterion_met = False
            for j in range(change_idx, len(is_correct) - 4):
                if is_correct[j:j+5].sum() == 5:
                    trials_to_criterion.append(j - change_idx)
                    criterion_met = True
                    break

            if not criterion_met:
                trials_to_criterion.append(np.nan)

    # Average across switches
    switching_metrics.append({
        'participant_id': pid,
        'n_rule_changes': len(rule_changes) - 1,
        'post_switch_error_rate': np.nanmean(post_switch_errors) if post_switch_errors else np.nan,
        'avg_trials_to_criterion': np.nanmean(trials_to_criterion) if trials_to_criterion else np.nan
    })

switching_df = pd.DataFrame(switching_metrics)
print(f"  Computed switching metrics for {len(switching_df)} participants")

# Merge with UCLA and gender
master = switching_df.merge(ucla_data, on='participant_id', how='inner')
master = master.merge(participants[['participant_id', 'gender_male']], on='participant_id', how='inner')
master = master.dropna(subset=['post_switch_error_rate'])

print(f"\n  Complete cases: N={len(master)}")

# Test correlations
print("\n[3/4] Testing UCLA × switching correlations...")

for gender, label in [(1, 'Male'), (0, 'Female')]:
    subset = master[master['gender_male'] == gender]
    print(f"\n  {label} (N={len(subset)}):")

    if len(subset) >= 10:
        for metric in ['post_switch_error_rate', 'avg_trials_to_criterion']:
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

    # Post-switch errors
    axes[0].scatter(subset['ucla_total'], subset['post_switch_error_rate'],
                   alpha=0.6, label=label, marker=marker, s=80, color=color)

    # Trials to criterion
    valid = subset.dropna(subset=['avg_trials_to_criterion'])
    axes[1].scatter(valid['ucla_total'], valid['avg_trials_to_criterion'],
                   alpha=0.6, label=label, marker=marker, s=80, color=color)

axes[0].set_xlabel('UCLA Loneliness')
axes[0].set_ylabel('Post-Switch Error Rate')
axes[0].set_title('Errors After Rule Changes')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('UCLA Loneliness')
axes[1].set_ylabel('Trials to Criterion')
axes[1].set_title('Learning Speed After Rule Changes')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "switching_dynamics.png", dpi=300, bbox_inches='tight')
plt.close()

# Save results
master.to_csv(OUTPUT_DIR / "switching_metrics.csv", index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print("✓ WCST Switching Dynamics Complete!")
print("="*80)
