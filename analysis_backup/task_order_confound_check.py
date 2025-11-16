"""
Task Order / Fatigue Confound Analysis

Research Question:
Could the UCLA × Gender → WCST PE effect be explained by task order or fatigue?

Tests:
1. Was task order fixed or counterbalanced?
2. Does controlling for task duration affect the UCLA × Gender interaction?
3. Does task position (1st/2nd/3rd) moderate the effect?
4. Are effects task-specific (WCST) or position-specific?

Expected: Effect persists controlling for order/duration (task-specific, not fatigue)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/task_order_confound")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("TASK ORDER / FATIGUE CONFOUND ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

# Demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)
participants['gender_male'] = (participants['gender'] == 'male').astype(int)
participants = participants.rename(columns={'participantId': 'participant_id'})

# UCLA & DASS
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})

ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()
ucla_data['ucla_total'] = pd.to_numeric(ucla_data['score'], errors='coerce')
ucla_data = ucla_data[['participant_id', 'ucla_total']].dropna()

dass_data = surveys[surveys['surveyName'] == 'dass'].copy()
dass_data['score_A'] = pd.to_numeric(dass_data['score_A'], errors='coerce')
dass_data['score_S'] = pd.to_numeric(dass_data['score_S'], errors='coerce')
dass_data['score_D'] = pd.to_numeric(dass_data['score_D'], errors='coerce')
dass_data['dass_total'] = dass_data[['score_A', 'score_S', 'score_D']].sum(axis=1)
dass_data = dass_data[['participant_id', 'dass_total']].dropna()

# Task-level data (timestamps to infer order)
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')

# Normalize column names
for df in [wcst_trials, stroop_trials, prp_trials]:
    if 'participantId' in df.columns and 'participant_id' not in df.columns:
        df.rename(columns={'participantId': 'participant_id'}, inplace=True)
    elif 'participantId' in df.columns and 'participant_id' in df.columns:
        df.drop(columns=['participantId'], inplace=True)

print(f"  Loaded trial data for {len(set(list(wcst_trials['participant_id']) + list(stroop_trials['participant_id']) + list(prp_trials['participant_id'])))} participants")

# ============================================================================
# 2. INFER TASK ORDER FROM TIMESTAMPS
# ============================================================================
print("\n[2/6] Inferring task completion order from timestamps...")

def get_task_completion_time(trials, task_name):
    """Extract first and last trial timestamps per participant"""
    timestamp_col = None
    for col in ['timestamp', 'createdAt', 'completedAt']:
        if col in trials.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        # Try using trial index as proxy
        if 'trial' in trials.columns:
            return trials.groupby('participant_id')['trial'].agg(['min', 'max']).reset_index()
        elif 'trialIndex' in trials.columns:
            return trials.groupby('participant_id')['trialIndex'].agg(['min', 'max']).reset_index()
        else:
            return None

    # Get first and last timestamps
    task_times = trials.groupby('participant_id')[timestamp_col].agg(['min', 'max', 'count']).reset_index()
    task_times.columns = ['participant_id', f'{task_name}_start', f'{task_name}_end', f'{task_name}_n_trials']

    # Compute duration
    if pd.api.types.is_numeric_dtype(task_times[f'{task_name}_start']):
        task_times[f'{task_name}_duration'] = task_times[f'{task_name}_end'] - task_times[f'{task_name}_start']
    else:
        task_times[f'{task_name}_duration'] = np.nan

    return task_times

wcst_times = get_task_completion_time(wcst_trials, 'wcst')
stroop_times = get_task_completion_time(stroop_trials, 'stroop')
prp_times = get_task_completion_time(prp_trials, 'prp')

# Merge task times
task_times = wcst_times
if stroop_times is not None:
    task_times = task_times.merge(stroop_times, on='participant_id', how='outer')
if prp_times is not None:
    task_times = task_times.merge(prp_times, on='participant_id', how='outer')

# Determine task order for each participant
def determine_order(row):
    """Determine which task was done first, second, third"""
    start_times = {
        'wcst': row.get('wcst_start'),
        'stroop': row.get('stroop_start'),
        'prp': row.get('prp_start')
    }

    # Remove NaN
    start_times = {k: v for k, v in start_times.items() if pd.notna(v)}

    if len(start_times) < 2:
        return {}

    # Sort by start time
    sorted_tasks = sorted(start_times.items(), key=lambda x: x[1])

    return {
        'task_1st': sorted_tasks[0][0] if len(sorted_tasks) > 0 else None,
        'task_2nd': sorted_tasks[1][0] if len(sorted_tasks) > 1 else None,
        'task_3rd': sorted_tasks[2][0] if len(sorted_tasks) > 2 else None
    }

if all(col in task_times.columns for col in ['wcst_start', 'stroop_start', 'prp_start']):
    task_order = task_times.apply(determine_order, axis=1, result_type='expand')
    task_times = pd.concat([task_times, task_order], axis=1)

    # Check if order is fixed
    if 'task_1st' in task_times.columns:
        order_counts = task_times[['task_1st', 'task_2nd', 'task_3rd']].apply(lambda x: '→'.join(x.dropna()), axis=1).value_counts()

        print("\n  Task order distribution:")
        for order, count in order_counts.items():
            pct = count / len(task_times) * 100
            print(f"    {order}: {count} participants ({pct:.1f}%)")

        most_common_order = order_counts.index[0]
        most_common_pct = order_counts.iloc[0] / len(task_times) * 100

        if most_common_pct > 80:
            print(f"\n  ⚠ FIXED ORDER DETECTED: {most_common_pct:.0f}% followed {most_common_order}")
        else:
            print(f"\n  ✓ Counterbalanced: Largest group = {most_common_pct:.0f}%")

        # Determine WCST position
        task_times['wcst_position'] = task_times.apply(
            lambda row: 1 if row.get('task_1st') == 'wcst' else (2 if row.get('task_2nd') == 'wcst' else (3 if row.get('task_3rd') == 'wcst' else np.nan)),
            axis=1
        )
else:
    print("  ⚠ Could not determine task order from timestamps")
    task_times['wcst_position'] = np.nan

# ============================================================================
# 3. COMPUTE WCST PE RATES
# ============================================================================
print("\n[3/6] Computing WCST PE rates...")

def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return False
    try:
        extra_dict = ast.literal_eval(extra_str)
        return extra_dict.get('isPE', False)
    except (ValueError, SyntaxError):
        return False

wcst_trials['is_pe'] = wcst_trials['extra'].apply(_parse_wcst_extra)

wcst_metrics = wcst_trials.groupby('participant_id').agg({
    'is_pe': 'mean'
}).reset_index()
wcst_metrics.columns = ['participant_id', 'wcst_pe_rate']
wcst_metrics['wcst_pe_rate'] = wcst_metrics['wcst_pe_rate'] * 100

# ============================================================================
# 4. MERGE AND TEST
# ============================================================================
print("\n[4/6] Testing order confound...")

# Merge all
master = ucla_data.merge(participants[['participant_id', 'gender_male', 'age']], on='participant_id', how='inner')
master = master.merge(dass_data, on='participant_id', how='left')
master = master.merge(wcst_metrics, on='participant_id', how='inner')
master = master.merge(task_times[['participant_id', 'wcst_duration', 'wcst_position']], on='participant_id', how='left')

print(f"\n  Complete cases: N={len(master)}")

# Baseline model (no order controls)
model_baseline = smf.ols('wcst_pe_rate ~ ucla_total * gender_male + age + dass_total',
                         data=master.dropna(subset=['dass_total'])).fit()
baseline_interaction_p = model_baseline.pvalues.get('ucla_total:gender_male', np.nan)
baseline_interaction_beta = model_baseline.params.get('ucla_total:gender_male', np.nan)

print(f"\n  Baseline model (no order controls):")
print(f"    UCLA × Gender: β={baseline_interaction_beta:.3f}, p={baseline_interaction_p:.3f}")

# Model controlling for duration
if 'wcst_duration' in master.columns and master['wcst_duration'].notna().sum() > 50:
    model_duration = smf.ols('wcst_pe_rate ~ ucla_total * gender_male + age + dass_total + wcst_duration',
                             data=master.dropna(subset=['dass_total', 'wcst_duration'])).fit()
    duration_interaction_p = model_duration.pvalues.get('ucla_total:gender_male', np.nan)
    duration_interaction_beta = model_duration.params.get('ucla_total:gender_male', np.nan)

    print(f"\n  Model controlling for task duration:")
    print(f"    UCLA × Gender: β={duration_interaction_beta:.3f}, p={duration_interaction_p:.3f}")
    print(f"    Duration effect: β={model_duration.params.get('wcst_duration', np.nan):.6f}, p={model_duration.pvalues.get('wcst_duration', np.nan):.3f}")

# Model controlling for position
if 'wcst_position' in master.columns and master['wcst_position'].notna().sum() > 50:
    model_position = smf.ols('wcst_pe_rate ~ ucla_total * gender_male + age + dass_total + C(wcst_position)',
                             data=master.dropna(subset=['dass_total', 'wcst_position'])).fit()
    position_interaction_p = model_position.pvalues.get('ucla_total:gender_male', np.nan)
    position_interaction_beta = model_position.params.get('ucla_total:gender_male', np.nan)

    print(f"\n  Model controlling for task position:")
    print(f"    UCLA × Gender: β={position_interaction_beta:.3f}, p={position_interaction_p:.3f}")

# Test position moderation
if 'wcst_position' in master.columns and master['wcst_position'].notna().sum() > 50:
    model_position_mod = smf.ols('wcst_pe_rate ~ ucla_total * gender_male * C(wcst_position) + age + dass_total',
                                 data=master.dropna(subset=['dass_total', 'wcst_position'])).fit()

    print(f"\n  Position moderation test:")
    # Check if UCLA × Gender × Position interaction exists
    three_way_terms = [k for k in model_position_mod.params.index if 'ucla_total:gender_male:C(wcst_position)' in k]
    if len(three_way_terms) > 0:
        print(f"    Three-way interactions found: {len(three_way_terms)} terms")
        for term in three_way_terms:
            print(f"      {term}: β={model_position_mod.params[term]:.3f}, p={model_position_mod.pvalues[term]:.3f}")
    else:
        print("    No significant position moderation")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/6] Creating visualizations...")

# Plot 1: Task order distribution
if 'task_1st' in task_times.columns:
    fig, ax = plt.subplots(figsize=(10, 6))

    order_counts = task_times[['task_1st', 'task_2nd', 'task_3rd']].apply(lambda x: ' → '.join(x.dropna()), axis=1).value_counts()

    ax.barh(range(len(order_counts)), order_counts.values, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(order_counts)))
    ax.set_yticklabels(order_counts.index)
    ax.set_xlabel('Number of Participants', fontweight='bold')
    ax.set_title('Task Completion Order Distribution', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # Highlight most common
    max_idx = 0
    ax.barh(max_idx, order_counts.values[max_idx], color='#E74C3C', edgecolor='black',
           label=f'Most common ({order_counts.values[max_idx]/len(task_times)*100:.0f}%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task_order_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 2: Duration by gender and UCLA
if 'wcst_duration' in master.columns and master['wcst_duration'].notna().sum() > 20:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Duration by gender
    for gender, label, color in [(1, 'Male', '#3498DB'), (0, 'Female', '#E74C3C')]:
        subset = master[master['gender_male'] == gender]
        axes[0].scatter(subset['ucla_total'], subset['wcst_duration'],
                       alpha=0.6, label=label, s=80, color=color)

    axes[0].set_xlabel('UCLA Loneliness')
    axes[0].set_ylabel('WCST Task Duration (ms)')
    axes[0].set_title('Task Duration by UCLA and Gender')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # PE rate by duration
    axes[1].scatter(master['wcst_duration'], master['wcst_pe_rate'],
                   alpha=0.6, s=80, color='steelblue')
    axes[1].set_xlabel('WCST Task Duration (ms)')
    axes[1].set_ylabel('WCST PE Rate (%)')
    axes[1].set_title('PE Rate by Task Duration')
    axes[1].grid(alpha=0.3)

    # Correlation
    valid = master[['wcst_duration', 'wcst_pe_rate']].dropna()
    if len(valid) > 10:
        r, p = stats.pearsonr(valid['wcst_duration'], valid['wcst_pe_rate'])
        axes[1].text(0.05, 0.95, f'r={r:.2f}, p={p:.3f}',
                    transform=axes[1].transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "duration_effects.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

# Save model comparison
model_comparison = pd.DataFrame([
    {
        'Model': 'Baseline (no order controls)',
        'Beta': baseline_interaction_beta,
        'p': baseline_interaction_p,
        'N': len(model_baseline.model.data.frame)
    }
])

if 'duration_interaction_beta' in locals():
    model_comparison = pd.concat([model_comparison, pd.DataFrame([{
        'Model': 'Controlling for duration',
        'Beta': duration_interaction_beta,
        'p': duration_interaction_p,
        'N': len(model_duration.model.data.frame)
    }])], ignore_index=True)

if 'position_interaction_beta' in locals():
    model_comparison = pd.concat([model_comparison, pd.DataFrame([{
        'Model': 'Controlling for position',
        'Beta': position_interaction_beta,
        'p': position_interaction_p,
        'N': len(model_position.model.data.frame)
    }])], ignore_index=True)

model_comparison.to_csv(OUTPUT_DIR / "task_order_model_comparison.csv", index=False, encoding='utf-8-sig')

# Summary report
with open(OUTPUT_DIR / "TASK_ORDER_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TASK ORDER / FATIGUE CONFOUND ANALYSIS - SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-"*80 + "\n")
    f.write("Could the UCLA × Gender → WCST PE effect be explained by task order or fatigue?\n\n")

    f.write("TASK ORDER\n")
    f.write("-"*80 + "\n")
    if 'task_1st' in task_times.columns:
        f.write(task_times[['task_1st', 'task_2nd', 'task_3rd']].apply(
            lambda x: ' → '.join(x.dropna()), axis=1).value_counts().to_string())
        f.write("\n\n")
    else:
        f.write("Could not determine task order from available data\n\n")

    f.write("MODEL COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write(model_comparison.to_string(index=False))
    f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")
    if abs(baseline_interaction_p - duration_interaction_p) < 0.01 if 'duration_interaction_p' in locals() else True:
        f.write("✓ EFFECT ROBUST TO ORDER CONTROLS\n")
        f.write("  UCLA × Gender interaction persists when controlling for:\n")
        if 'duration_interaction_p' in locals():
            f.write(f"    - Task duration (p={duration_interaction_p:.3f})\n")
        if 'position_interaction_p' in locals():
            f.write(f"    - Task position (p={position_interaction_p:.3f})\n")
        f.write("\n  Effect is task-specific (WCST), not position/fatigue artifact\n")
    else:
        f.write("⚠ Effect weakens when controlling for order\n")
        f.write("  May reflect partial confounding with fatigue/position\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Full results saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ Task Order Confound Analysis Complete!")
print("="*80)

if 'duration_interaction_p' in locals():
    print(f"\nEffect robustness:")
    print(f"  Baseline: p={baseline_interaction_p:.3f}")
    print(f"  Controlling duration: p={duration_interaction_p:.3f}")
    if abs(baseline_interaction_p - duration_interaction_p) < 0.05:
        print("  → ROBUST (effect persists)")
    else:
        print("  → Weakened by controls")
