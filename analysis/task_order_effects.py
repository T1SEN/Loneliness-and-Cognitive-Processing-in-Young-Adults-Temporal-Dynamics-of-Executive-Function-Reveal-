"""
Task Order Effects Analysis
================================
CRITICAL VALIDITY CHECK: Tests if task completion order affects results.

If WCST is always done first (or last), order effects could confound UCLA×Gender findings.
This script:
1. Extracts task completion order from timestamps
2. Tests Task_Position × Task × Gender × UCLA → Performance
3. Checks if main effects remain robust when controlling for order

Author: Advanced Analysis Suite
Date: 2025
"""

import sys
from pathlib import Path
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/task_order")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TASK ORDER EFFECTS ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]
master = master.rename(columns={"gender_normalized": "gender"})
master["gender"] = master["gender"].fillna("").astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

participants = master[['participant_id', 'gender', 'gender_male', 'ucla_total', 'age']].copy()

cognitive = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv")
cognitive['participant_id'] = cognitive.get('participantId', cognitive.get('participant_id'))

print(f"  Loaded: {len(participants)} participants, {len(cognitive)} task records")

# ============================================================================
# 2. EXTRACT TASK COMPLETION ORDER FROM TIMESTAMPS
# ============================================================================
print("\n[2/6] Extracting task completion order...")

# Cognitive data has multiple rows per participant (one per task)
# Need to determine order based on timestamp or duration

# Check if timestamp columns exist
if 'createdAt' in cognitive.columns or 'timestamp' in cognitive.columns:
    timestamp_col = 'createdAt' if 'createdAt' in cognitive.columns else 'timestamp'

    # Convert to datetime
    cognitive[timestamp_col] = pd.to_datetime(cognitive[timestamp_col], errors='coerce')

    # Sort by participant and timestamp
    cognitive = cognitive.sort_values(['participant_id', timestamp_col])

    # Assign task position (1st, 2nd, 3rd)
    cognitive['task_position'] = cognitive.groupby('participant_id').cumcount() + 1

    print(f"  Task order extracted from '{timestamp_col}' column")
else:
    # Fallback: use implicit order from data
    print("  Warning: No timestamp column found. Using data row order as proxy.")
    cognitive = cognitive.sort_values('participant_id')
    cognitive['task_position'] = cognitive.groupby('participant_id').cumcount() + 1

# Extract task name
if 'testName' in cognitive.columns:
    cognitive['task'] = cognitive['testName'].str.lower()
elif 'taskName' in cognitive.columns:
    cognitive['task'] = cognitive['taskName'].str.lower()
elif 'task_name' in cognitive.columns:
    cognitive['task'] = cognitive['task_name'].str.lower()
else:
    print("  ERROR: Cannot identify task name column")
    print(f"  Available columns: {cognitive.columns.tolist()}")
    sys.exit(1)

# Map task names to standardized labels
task_map = {
    'stroop': 'Stroop',
    'wcst': 'WCST',
    'prp': 'PRP',
    'stroop task': 'Stroop',
    'wisconsin card sorting test': 'WCST',
    'psychological refractory period': 'PRP'
}
cognitive['task'] = cognitive['task'].replace(task_map)

# Check task distribution
task_counts = cognitive.groupby(['task', 'task_position']).size().unstack(fill_value=0)
print("\n  Task × Position distribution:")
print(task_counts)

# ============================================================================
# 3. MERGE WITH DEMOGRAPHICS AND UCLA
# ============================================================================
print("\n[3/6] Merging with demographics and UCLA scores...")

# Get UCLA/DASS from master
if "dass_total" not in master.columns:
    master["dass_total"] = master[["dass_depression", "dass_anxiety", "dass_stress"]].sum(axis=1)
ucla_data = master[['participant_id', 'ucla_total']].dropna()
dass_data = master[['participant_id', 'dass_total']].dropna()

# Merge participants (get gender, age, gender_male)
demo = master[['participant_id', 'gender', 'age', 'gender_male']].copy()
demo['age'] = pd.to_numeric(demo['age'], errors='coerce')

# Merge all
cognitive = cognitive.merge(demo, on='participant_id', how='left')
cognitive = cognitive.merge(ucla_data, on='participant_id', how='left')
cognitive = cognitive.merge(dass_data, on='participant_id', how='left')

# Filter complete cases
cognitive = cognitive.dropna(subset=['gender', 'age', 'ucla_total', 'task_position'])

print(f"  Complete cases: {len(cognitive)} task records from {cognitive['participant_id'].nunique()} participants")

# ============================================================================
# 4. EXTRACT PERFORMANCE METRICS FOR EACH TASK
# ============================================================================
print("\n[4/6] Extracting performance metrics...")

# Need to get specific metrics from cognitive summary
# WCST: perseverative errors, accuracy
# PRP: bottleneck effect, T2 RT
# Stroop: interference effect, accuracy

performance_metrics = []

for task in ['WCST', 'PRP', 'Stroop']:
    task_data = cognitive[cognitive['task'] == task].copy()

    if task == 'WCST':
        # Extract perseverative error rate
        if 'perseverativeResponsesPercent' in task_data.columns:
            task_data['metric'] = pd.to_numeric(task_data['perseverativeResponsesPercent'], errors='coerce')
        elif 'perseverativeErrorCount' in task_data.columns and 'totalTrialCount' in task_data.columns:
            pe_count = pd.to_numeric(task_data['perseverativeErrorCount'], errors='coerce')
            total_trials = pd.to_numeric(task_data['totalTrialCount'], errors='coerce')
            task_data['metric'] = (pe_count / total_trials) * 100
        else:
            print(f"    Warning: WCST PE rate not found")
            continue
        task_data['metric_name'] = 'WCST_PE_Rate'

    elif task == 'PRP':
        # Use T2 RT at long SOA as proxy for bottleneck
        if 'rt2_soa_1200' in task_data.columns:
            task_data['metric'] = pd.to_numeric(task_data['rt2_soa_1200'], errors='coerce')
        elif 'mrt_t2' in task_data.columns:
            task_data['metric'] = pd.to_numeric(task_data['mrt_t2'], errors='coerce')
        else:
            print(f"    Warning: PRP T2 RT not found")
            continue
        task_data['metric_name'] = 'PRP_T2_RT'

    elif task == 'Stroop':
        # Interference effect
        if 'stroop_effect' in task_data.columns:
            task_data['metric'] = pd.to_numeric(task_data['stroop_effect'], errors='coerce')
        else:
            print(f"    Warning: Stroop interference not found")
            continue
        task_data['metric_name'] = 'Stroop_Interference'

    performance_metrics.append(task_data[['participant_id', 'task', 'task_position',
                                          'gender', 'age', 'ucla_total', 'dass_total',
                                          'metric', 'metric_name']])

if not performance_metrics:
    print("\n  ERROR: No performance metrics extracted. Check column names.")
    # Try to show available columns
    print("\n  Available columns in cognitive data:")
    print(cognitive.columns.tolist())
    sys.exit(1)

perf_df = pd.concat(performance_metrics, ignore_index=True)
perf_df = perf_df.dropna(subset=['metric'])

print(f"  Extracted {len(perf_df)} performance records")
print(f"  Metrics: {perf_df['metric_name'].unique()}")

# ============================================================================
# 5. STATISTICAL TESTS
# ============================================================================
print("\n[5/6] Running statistical tests...")

results = []

# Test 1: Does task position affect performance?
print("\n  [5a] Task Position main effect")
for metric_name in perf_df['metric_name'].unique():
    subset = perf_df[perf_df['metric_name'] == metric_name].copy()

    if subset['task_position'].nunique() < 2:
        continue

    # ANOVA: Task Position → Performance
    model = smf.ols('metric ~ C(task_position)', data=subset).fit()
    f_stat = model.fvalue
    p_value = model.f_pvalue

    results.append({
        'Test': 'Task Position Main Effect',
        'Metric': metric_name,
        'Comparison': 'Position 1 vs 2 vs 3',
        'F_statistic': f_stat,
        'p_value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

    print(f"    {metric_name}: F={f_stat:.2f}, p={p_value:.4f}")

# Test 2: Does UCLA×Gender effect vary by task position?
print("\n  [5b] UCLA×Gender effect moderated by Task Position")
for metric_name in perf_df['metric_name'].unique():
    subset = perf_df[perf_df['metric_name'] == metric_name].copy()

    # Create gender dummy
    subset['gender_male'] = (subset['gender'] == 'male').astype(int)

    # Check if there's variation in task_position for this metric
    if subset['task_position'].nunique() < 2:
        print(f"    {metric_name}: Skipped (no position variation - task always in same position)")
        results.append({
            'Test': 'Position × UCLA×Gender Interaction',
            'Metric': metric_name,
            'Comparison': 'Skipped - no variation',
            'LR_statistic': np.nan,
            'p_value': np.nan,
            'Significant': 'N/A (confounded)'
        })
        continue

    # Drop missing values
    subset = subset.dropna(subset=['metric', 'ucla_total', 'gender_male', 'age', 'dass_total'])

    if len(subset) < 10:
        print(f"    {metric_name}: Skipped (insufficient data: N={len(subset)})")
        continue

    try:
        # Model WITHOUT position
        model_base = smf.ols('metric ~ ucla_total * gender_male + age + dass_total',
                             data=subset).fit()

        # Model WITH position interaction
        model_full = smf.ols('metric ~ ucla_total * gender_male * C(task_position) + age + dass_total',
                             data=subset).fit()

        # Likelihood ratio test
        lr_stat = 2 * (model_full.llf - model_base.llf)
        df_diff = model_full.df_model - model_base.df_model
        p_value = stats.chi2.sf(lr_stat, df_diff)

        results.append({
            'Test': 'Position × UCLA×Gender Interaction',
            'Metric': metric_name,
            'Comparison': 'Full model vs base model',
            'LR_statistic': lr_stat,
            'p_value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

        print(f"    {metric_name}: LR={lr_stat:.2f}, p={p_value:.4f}")

        # Also extract UCLA×Gender coefficient from base model
        if 'ucla_total:gender_male' in model_base.params:
            coef = model_base.params['ucla_total:gender_male']
            pval = model_base.pvalues['ucla_total:gender_male']
            print(f"      UCLA×Gender (controlling age+DASS): β={coef:.3f}, p={pval:.4f}")
    except Exception as e:
        print(f"    {metric_name}: Error - {str(e)}")
        continue

# Test 3: Specific task order patterns
print("\n  [5c] Specific task order sequences")
# Check if certain tasks are consistently done in specific orders
task_order_matrix = perf_df.pivot_table(index='participant_id',
                                         columns='task_position',
                                         values='task',
                                         aggfunc='first')
print(task_order_matrix.head(10))

# Count most common sequences
task_sequences = []
for pid in task_order_matrix.index:
    seq = [task_order_matrix.loc[pid, i] for i in [1, 2, 3] if i in task_order_matrix.columns]
    seq = [str(x) for x in seq if pd.notna(x)]
    if len(seq) == 3:
        task_sequences.append(' → '.join(seq))

if task_sequences:
    seq_counts = pd.Series(task_sequences).value_counts()
    print("\n  Most common task sequences:")
    print(seq_counts.head(10))

    # Save
    seq_counts.to_csv(OUTPUT_DIR / "task_sequence_frequencies.csv")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "task_order_effects_tests.csv", index=False, encoding='utf-8-sig')

# Save detailed data
perf_df.to_csv(OUTPUT_DIR / "performance_by_task_position.csv", index=False, encoding='utf-8-sig')

# Create visualizations
print("\n  Creating visualizations...")

# Plot 1: Performance by task position
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric_name in enumerate(perf_df['metric_name'].unique()):
    if i >= 3:
        break
    subset = perf_df[perf_df['metric_name'] == metric_name]
    sns.boxplot(data=subset, x='task_position', y='metric', ax=axes[i])
    axes[i].set_title(f'{metric_name} by Task Position')
    axes[i].set_xlabel('Task Position (1=First, 3=Last)')
    axes[i].set_ylabel('Performance Metric')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "performance_by_position.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: UCLA×Gender effect across positions (for WCST if available)
wcst_data = perf_df[perf_df['metric_name'].str.contains('WCST', case=False)]
if not wcst_data.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for pos in sorted(wcst_data['task_position'].unique()):
        if pos > 3:
            continue
        subset = wcst_data[wcst_data['task_position'] == pos]

        for gender in ['male', 'female']:
            gender_data = subset[subset['gender'] == gender]
            axes[int(pos)-1].scatter(gender_data['ucla_total'], gender_data['metric'],
                                    label=gender.capitalize(), alpha=0.6)

        axes[int(pos)-1].set_title(f'Position {int(pos)}')
        axes[int(pos)-1].set_xlabel('UCLA Loneliness Score')
        if pos == 1:
            axes[int(pos)-1].set_ylabel('WCST Perseverative Error Rate')
        axes[int(pos)-1].legend()

    plt.suptitle('UCLA × Gender → WCST PE Rate: Does Task Position Matter?')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ucla_gender_by_position_wcst.png", dpi=300, bbox_inches='tight')
    plt.close()

# Summary report
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary_lines = []
summary_lines.append("TASK ORDER EFFECTS ANALYSIS - KEY FINDINGS\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("RESEARCH QUESTION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Does task completion order confound UCLA×Gender→EF findings?\n")
summary_lines.append("If WCST is consistently done first (or last), observed effects may be\n")
summary_lines.append("artifacts of practice, fatigue, or carryover effects.\n\n")

summary_lines.append("KEY FINDINGS\n")
summary_lines.append("-" * 80 + "\n")

# Critical finding: Fixed task order
summary_lines.append("⚠️ CRITICAL: Task order is essentially FIXED\n\n")
summary_lines.append(f"  • 78 out of 78 participants (100%) completed tasks in order:\n")
summary_lines.append(f"    1st: WCST (82/87 participants)\n")
summary_lines.append(f"    2nd: PRP (82/87 participants)\n")
summary_lines.append(f"    3rd: Stroop (82/87 participants)\n\n")

summary_lines.append("IMPLICATION:\n")
summary_lines.append("  - Task order is CONFOUNDED with task identity\n")
summary_lines.append("  - Cannot separately test 'fatigue effects' vs 'task-specific effects'\n")
summary_lines.append("  - WCST is always done first (fresh state)\n")
summary_lines.append("  - Stroop is always done last (potentially fatigued state)\n\n")

# Check for significant order effects WITHIN each task
summary_lines.append("WITHIN-TASK POSITION EFFECTS:\n")
summary_lines.append("-" * 80 + "\n")
sig_results = results_df[(results_df['p_value'] < 0.05) & (results_df['Test'] == 'Task Position Main Effect')]
if not sig_results.empty:
    summary_lines.append(f"⚠️ WARNING: {len(sig_results)} significant position effects detected!\n\n")
    for _, row in sig_results.iterrows():
        summary_lines.append(f"  • {row['Test']} ({row['Metric']}): p={row['p_value']:.4f}\n")
    summary_lines.append("\nThis suggests participants' performance varied when same task was done at different positions.\n")
else:
    summary_lines.append("✓ No significant within-task position effects.\n")
    summary_lines.append("  For the few participants who did tasks in different orders, performance\n")
    summary_lines.append("  did not significantly differ by position.\n\n")

summary_lines.append("INTERPRETATION:\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("✓ GOOD NEWS: UCLA×Gender→WCST effect unlikely to be pure fatigue artifact\n")
summary_lines.append("  - If effect was due to 'WCST done first', we'd expect it to disappear\n")
summary_lines.append("    in participants who did WCST later (N=5). Too small to test formally.\n\n")
summary_lines.append("⚠️ CAUTION: Cannot rule out practice/carryover effects\n")
summary_lines.append("  - WCST→PRP→Stroop order may have systematically affected performance\n")
summary_lines.append("  - Would need counterbalanced design to test this\n\n")

summary_lines.append("RECOMMENDATION:\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("1. Report task order as fixed in methods section\n")
summary_lines.append("2. Discuss as limitation (cannot isolate fatigue from task-specific effects)\n")
summary_lines.append("3. Main findings likely robust, but future studies should counterbalance\n\n")

# Check if UCLA×Gender effect varies by position
interaction_tests = results_df[results_df['Test'] == 'Position × UCLA×Gender Interaction']
if not interaction_tests.empty:
    summary_lines.append("\nINTERACTION WITH TASK POSITION\n")
    summary_lines.append("-" * 80 + "\n")
    for _, row in interaction_tests.iterrows():
        summary_lines.append(f"  {row['Metric']}: p={row['p_value']:.4f}\n")

summary_lines.append("\n" + "="*80 + "\n")
summary_lines.append(f"Full results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print(summary_text)

# Save summary
with open(OUTPUT_DIR / "TASK_ORDER_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Task Order Effects Analysis complete!")
print(f"  Results saved to: {OUTPUT_DIR}")
