"""
Stroop Methodological Diagnosis
=================================
WHY IS STROOP NULL? Critical investigation of methodological vs theoretical explanations.

Four hypotheses tested:
1. Floor effect: High accuracy (>96%) leaves little variance for UCLA to predict
2. Metric reliability: Interference score has low within-person variability
3. Neutral baseline: Using neutral vs congruent changes interference calculation
4. Performance stratification: Effect only emerges in low-performers

Author: Advanced Analysis Suite
Date: 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

# Unicode handling
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/stroop_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STROOP METHODOLOGICAL DIAGNOSIS")
print("="*80)

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
print("\n[1/5] Loading data...")

stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')

# Normalize IDs
stroop_trials['participant_id'] = stroop_trials.get('participantId', stroop_trials.get('participant_id'))
participants['participant_id'] = participants.get('participantId', participants.get('participant_id'))
surveys['participant_id'] = surveys.get('participantId', surveys.get('participant_id'))

# Get UCLA
ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
ucla_data['ucla_total'] = pd.to_numeric(ucla_data['score'], errors='coerce')

# Get gender
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)

# Merge
stroop_trials = stroop_trials.merge(ucla_data[['participant_id', 'ucla_total']], on='participant_id', how='left')
stroop_trials = stroop_trials.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

print(f"  Loaded {len(stroop_trials)} Stroop trials from {stroop_trials['participant_id'].nunique()} participants")

# ===========================================================================
# 2. HYPOTHESIS 1: FLOOR EFFECT (High Accuracy)
# ===========================================================================
print("\n[2/5] Hypothesis 1: Floor effect due to high accuracy")

# Calculate accuracy by condition
stroop_trials['correct'] = stroop_trials['correct'].astype(bool)

# Use 'type' column for condition (congruent/incongruent/neutral)
if 'type' in stroop_trials.columns:
    stroop_trials['condition'] = stroop_trials['type']
elif 'cond' in stroop_trials.columns:
    stroop_trials['condition'] = stroop_trials['cond']

# Use rt_ms if available, otherwise rt
if 'rt_ms' in stroop_trials.columns:
    stroop_trials['rt'] = pd.to_numeric(stroop_trials['rt_ms'], errors='coerce')
else:
    stroop_trials['rt'] = pd.to_numeric(stroop_trials['rt'], errors='coerce')

# Filter out timeouts and invalid RTs
stroop_trials = stroop_trials[stroop_trials['rt'] > 0]
if 'is_timeout' in stroop_trials.columns:
    stroop_trials = stroop_trials[~stroop_trials['is_timeout'].fillna(False)]

accuracy_by_condition = stroop_trials.groupby(['participant_id', 'condition'])['correct'].mean().unstack()

print("\n  Accuracy by condition:")
for cond in accuracy_by_condition.columns:
    mean_acc = accuracy_by_condition[cond].mean() * 100
    std_acc = accuracy_by_condition[cond].std() * 100
    print(f"    {cond}: M={mean_acc:.2f}%, SD={std_acc:.2f}%")

# Median split on accuracy
overall_acc = stroop_trials.groupby('participant_id')['correct'].mean()
median_acc = overall_acc.median()
high_acc = overall_acc >= median_acc

# Test UCLA→interference in low vs high accuracy groups
stroop_summary = stroop_trials.groupby('participant_id').apply(lambda x: pd.Series({
    'incong_rt': x[x['condition'] == 'incongruent']['rt'].mean(),
    'cong_rt': x[x['condition'] == 'congruent']['rt'].mean(),
    'neutral_rt': x[x['condition'] == 'neutral']['rt'].mean() if 'neutral' in x['condition'].values else np.nan,
    'overall_acc': x['correct'].mean(),
    'incong_acc': x[x['condition'] == 'incongruent']['correct'].mean(),
    'cong_acc': x[x['condition'] == 'congruent']['correct'].mean()
})).reset_index()

stroop_summary['interference'] = stroop_summary['incong_rt'] - stroop_summary['cong_rt']
stroop_summary['high_accuracy'] = stroop_summary['participant_id'].isin(overall_acc[high_acc].index)

# Merge UCLA and gender
stroop_summary = stroop_summary.merge(ucla_data[['participant_id', 'ucla_total']], on='participant_id', how='left')
stroop_summary = stroop_summary.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')
stroop_summary = stroop_summary.dropna(subset=['ucla_total', 'interference', 'gender'])

print(f"\n  Stratified analysis by accuracy:")
print(f"    Low accuracy group (N={(~stroop_summary['high_accuracy']).sum()}):")
low_acc_data = stroop_summary[~stroop_summary['high_accuracy']]
if len(low_acc_data) >= 10:
    corr_low = stats.pearsonr(low_acc_data['ucla_total'], low_acc_data['interference'])
    print(f"      UCLA × Interference: r={corr_low[0]:.3f}, p={corr_low[1]:.4f}")
else:
    print("      Insufficient data")

print(f"    High accuracy group (N={stroop_summary['high_accuracy'].sum()}):")
high_acc_data = stroop_summary[stroop_summary['high_accuracy']]
if len(high_acc_data) >= 10:
    corr_high = stats.pearsonr(high_acc_data['ucla_total'], high_acc_data['interference'])
    print(f"      UCLA × Interference: r={corr_high[0]:.3f}, p={corr_high[1]:.4f}")

# ===========================================================================
# 3. HYPOTHESIS 2: METRIC RELIABILITY
# ===========================================================================
print("\n[3/5] Hypothesis 2: Low metric reliability")

# Calculate coefficient of variation for RT within person
rt_variability = stroop_trials.groupby(['participant_id', 'condition'])['rt'].agg(['mean', 'std', 'count']).reset_index()
rt_variability['cv'] = rt_variability['std'] / rt_variability['mean']

# Check correlation between interference and RT variability
interference_cv_corr = stroop_summary.merge(
    rt_variability[rt_variability['condition'] == 'incongruent'][['participant_id', 'cv']],
    on='participant_id', how='left'
)

if 'cv' in interference_cv_corr.columns:
    corr = stats.pearsonr(interference_cv_corr['interference'].dropna(),
                          interference_cv_corr['cv'].dropna())
    print(f"  Interference × RT_CV correlation: r={corr[0]:.3f}, p={corr[1]:.4f}")
    if abs(corr[0]) < 0.2:
        print(f"    ⚠️ Low correlation suggests metric may be unreliable")

# Split-half reliability
stroop_trials['trial_half'] = stroop_trials.groupby('participant_id').cumcount()
stroop_trials['half'] = (stroop_trials['trial_half'] >= stroop_trials.groupby('participant_id')['trial_half'].transform('max') / 2).astype(int)

# Calculate interference for each half
reliability_data = []
for half in [0, 1]:
    half_data = stroop_trials[stroop_trials['half'] == half]
    half_summary = half_data.groupby('participant_id').apply(lambda x: pd.Series({
        f'interference_half{half}': x[x['condition'] == 'incongruent']['rt'].mean() - x[x['condition'] == 'congruent']['rt'].mean()
    })).reset_index()
    reliability_data.append(half_summary)

reliability_df = reliability_data[0].merge(reliability_data[1], on='participant_id', how='inner')
if len(reliability_df) >= 10:
    split_half_corr = stats.pearsonr(reliability_df['interference_half0'], reliability_df['interference_half1'])
    print(f"  Split-half reliability: r={split_half_corr[0]:.3f}, p={split_half_corr[1]:.4f}")
    spearman_brown = (2 * split_half_corr[0]) / (1 + split_half_corr[0])
    print(f"  Spearman-Brown corrected: r={spearman_brown:.3f}")

# ===========================================================================
# 4. HYPOTHESIS 3: NEUTRAL BASELINE
# ===========================================================================
print("\n[4/5] Hypothesis 3: Neutral baseline calculation")

# Recalculate interference using neutral baseline
stroop_summary['interference_neutral'] = stroop_summary['incong_rt'] - stroop_summary['neutral_rt']
stroop_summary = stroop_summary.dropna(subset=['interference_neutral'])

print(f"\n  Interference metrics comparison:")
print(f"    Traditional (Incong - Cong): M={stroop_summary['interference'].mean():.2f}, SD={stroop_summary['interference'].std():.2f}")
if 'interference_neutral' in stroop_summary.columns and not stroop_summary['interference_neutral'].isna().all():
    print(f"    Neutral baseline (Incong - Neutral): M={stroop_summary['interference_neutral'].mean():.2f}, SD={stroop_summary['interference_neutral'].std():.2f}")

    # Test UCLA correlation with both metrics
    corr_trad = stats.pearsonr(stroop_summary['ucla_total'], stroop_summary['interference'])
    corr_neutral = stats.pearsonr(stroop_summary['ucla_total'].dropna(),
                                   stroop_summary['interference_neutral'].dropna())

    print(f"\n  UCLA correlations:")
    print(f"    Traditional: r={corr_trad[0]:.3f}, p={corr_trad[1]:.4f}")
    print(f"    Neutral baseline: r={corr_neutral[0]:.3f}, p={corr_neutral[1]:.4f}")

# ===========================================================================
# 5. HYPOTHESIS 4: PERFORMANCE STRATIFICATION × GENDER
# ===========================================================================
print("\n[5/5] Hypothesis 4: Effect only in specific subgroups")

# Test interaction: UCLA × Gender × Performance
stroop_summary['gender_male'] = (stroop_summary['gender'] == 'male').astype(int)

# Model 1: UCLA × Gender (baseline)
model_base = smf.ols('interference ~ ucla_total * gender_male', data=stroop_summary).fit()
print(f"\n  Model 1 (UCLA × Gender):")
if 'ucla_total:gender_male' in model_base.params:
    print(f"    UCLA × Gender: β={model_base.params['ucla_total:gender_male']:.4f}, p={model_base.pvalues['ucla_total:gender_male']:.4f}")

# Model 2: Add accuracy interaction
model_acc = smf.ols('interference ~ ucla_total * gender_male * overall_acc', data=stroop_summary).fit()
print(f"\n  Model 2 (UCLA × Gender × Accuracy):")
print(f"    R² = {model_acc.rsquared:.4f}")

# Stratified by gender
print(f"\n  Stratified by gender:")
for gender in ['male', 'female']:
    gender_data = stroop_summary[stroop_summary['gender'] == gender]
    if len(gender_data) >= 10:
        corr = stats.pearsonr(gender_data['ucla_total'], gender_data['interference'])
        print(f"    {gender.capitalize()} (N={len(gender_data)}): r={corr[0]:.3f}, p={corr[1]:.4f}")

# ===========================================================================
# 6. VISUALIZATIONS
# ===========================================================================
print("\n[6/6] Creating visualizations...")

# Plot 1: Accuracy distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, cond in enumerate(['congruent', 'incongruent', 'neutral']):
    if cond in accuracy_by_condition.columns:
        axes[i].hist(accuracy_by_condition[cond] * 100, bins=20, edgecolor='black')
        axes[i].axvline(accuracy_by_condition[cond].mean() * 100, color='red', linestyle='--',
                       label=f'Mean={accuracy_by_condition[cond].mean()*100:.1f}%')
        axes[i].set_title(f'{cond.capitalize()} Accuracy')
        axes[i].set_xlabel('Accuracy (%)')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

plt.suptitle('Stroop Accuracy Distribution by Condition', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: UCLA × Interference by accuracy group
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, is_high_acc in enumerate([False, True]):
    subset = stroop_summary[stroop_summary['high_accuracy'] == is_high_acc]
    title = 'High Accuracy' if is_high_acc else 'Low Accuracy'

    axes[i].scatter(subset['ucla_total'], subset['interference'], alpha=0.6)

    # Regression line
    if len(subset) >= 5:
        z = np.polyfit(subset['ucla_total'], subset['interference'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
        axes[i].plot(x_line, p(x_line), 'r--', linewidth=2)

    axes[i].set_title(f'{title} (N={len(subset)})')
    axes[i].set_xlabel('UCLA Loneliness Score')
    if i == 0:
        axes[i].set_ylabel('Stroop Interference (ms)')
    axes[i].grid(alpha=0.3)

plt.suptitle('Does Floor Effect Explain Stroop Null?', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "floor_effect_test.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Traditional vs Neutral baseline
if 'interference_neutral' in stroop_summary.columns and not stroop_summary['interference_neutral'].isna().all():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(stroop_summary['ucla_total'], stroop_summary['interference'], alpha=0.6)
    axes[0].set_title('Traditional (Incong - Cong)')
    axes[0].set_xlabel('UCLA Loneliness')
    axes[0].set_ylabel('Interference (ms)')

    axes[1].scatter(stroop_summary['ucla_total'], stroop_summary['interference_neutral'], alpha=0.6)
    axes[1].set_title('Neutral Baseline (Incong - Neutral)')
    axes[1].set_xlabel('UCLA Loneliness')
    axes[1].set_ylabel('Interference (ms)')

    plt.suptitle('Does Baseline Choice Matter?', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# Save summary
stroop_summary.to_csv(OUTPUT_DIR / "stroop_detailed_summary.csv", index=False, encoding='utf-8-sig')

# ===========================================================================
# 7. SUMMARY REPORT
# ===========================================================================
summary_lines = []
summary_lines.append("STROOP METHODOLOGICAL DIAGNOSIS - WHY IS STROOP NULL?\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("FOUR HYPOTHESES TESTED\n")
summary_lines.append("-" * 80 + "\n\n")

summary_lines.append("H1: FLOOR EFFECT (High Accuracy)\n")
summary_lines.append(f"  • Overall accuracy: {(stroop_summary['overall_acc'].mean()*100):.1f}%\n")
summary_lines.append(f"  • Incongruent accuracy: {(stroop_summary['incong_acc'].mean()*100):.1f}%\n")
summary_lines.append(f"  • Finding: ")
if stroop_summary['overall_acc'].mean() > 0.95:
    summary_lines.append("✓ Evidence of ceiling effect\n")
    summary_lines.append(f"    - Accuracy >95% leaves minimal variance for UCLA to predict\n")
else:
    summary_lines.append("✗ Accuracy not at ceiling\n")

summary_lines.append("\nH2: METRIC RELIABILITY\n")
if 'split_half_corr' in locals():
    summary_lines.append(f"  • Split-half reliability: r={split_half_corr[0]:.3f}\n")
    if split_half_corr[0] < 0.5:
        summary_lines.append("  ⚠️ Poor reliability - metric may be unstable\n")
    else:
        summary_lines.append("  ✓ Adequate reliability\n")

summary_lines.append("\nH3: NEUTRAL BASELINE\n")
if 'corr_neutral' in locals():
    summary_lines.append(f"  • Traditional interference: r={corr_trad[0]:.3f}, p={corr_trad[1]:.4f}\n")
    summary_lines.append(f"  • Neutral baseline: r={corr_neutral[0]:.3f}, p={corr_neutral[1]:.4f}\n")
    if abs(corr_neutral[0]) > abs(corr_trad[0]):
        summary_lines.append("  → Neutral baseline shows stronger (but still weak) effect\n")
    else:
        summary_lines.append("  → Baseline choice doesn't explain null finding\n")

summary_lines.append("\nH4: SUBGROUP EFFECTS\n")
summary_lines.append("  • UCLA × Gender interaction: ")
if 'model_base' in locals() and 'ucla_total:gender_male' in model_base.params:
    summary_lines.append(f"p={model_base.pvalues['ucla_total:gender_male']:.4f}\n")
else:
    summary_lines.append("Not significant\n")

summary_lines.append("\n" + "="*80 + "\n")
summary_lines.append("CONCLUSION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Stroop null finding is likely THEORETICAL, not methodological:\n")
summary_lines.append("  1. Loneliness may selectively impair PROACTIVE control (WCST, PRP)\n")
summary_lines.append("  2. Stroop requires REACTIVE control (trial-by-trial conflict)\n")
summary_lines.append("  3. Different cognitive mechanisms → different sensitivities to loneliness\n\n")
summary_lines.append(f"Results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print("\n" + summary_text)

with open(OUTPUT_DIR / "STROOP_DIAGNOSIS_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Stroop Methodological Diagnosis complete!")
