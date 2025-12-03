"""
Paper 4 (Revised): Mean RT vs SD RT Analysis

RESEARCH QUESTION:
  Does loneliness affect RT LOCATION (mean) or SCALE (SD)?

RATIONALE:
  - Paper 1 focused on TAU (exponential tail, lapses)
  - This analysis separates MEAN (overall speed) from SD (consistency)
  - Complements distributional approach with simpler metrics

THEORETICAL PREDICTIONS:
  - Mean RT ~ UCLA: Overall slowing (general impairment)
  - SD RT ~ UCLA: Inconsistency (variability)
  - Tau ~ UCLA: Lapses (Paper 1 finding)

  Hypothesis: SD and Tau more sensitive than Mean (variability > speed)

METHOD:
  - Hierarchical regression with DASS control
  - Gender stratification
  - Compare effect sizes across Mean, SD, Tau
"""

import sys
from pathlib import Path
import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_stroop_trials, load_prp_trials, load_wcst_trials
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/paper4_mean_sd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*100)
print("PAPER 4 (REVISED): MEAN RT vs SD RT ANALYSIS")
print("="*100)

# === LOAD DATA ===
print("\n[1] Loading data...")

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']

demo = master[['participant_id', 'age', 'gender', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']].copy()
demo = demo.dropna(subset=['gender'])
demo['gender_male'] = (demo['gender'] == 'male').astype(int)

print(f"Demographics: N={len(demo)} ({demo['gender_male'].sum()} males, {len(demo) - demo['gender_male'].sum()} females)")

# Load Paper 1 tau estimates
paper1_dir = Path("results/analysis_outputs/paper1_distributional")
paper1_metrics = pd.read_csv(paper1_dir / "paper1_participant_variability_metrics.csv", encoding='utf-8-sig')

# === COMPUTE MEAN AND SD RT ===
print("\n[2] Computing Mean RT and SD RT for each task...")

rt_metrics = []

# === STROOP ===
print("\n[2.1] Stroop task...")
stroop_trials, _ = load_stroop_trials(use_cache=True)
rt_col_stroop = 'rt'
if rt_col_stroop not in stroop_trials.columns and 'rt_ms' in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'rt_ms': 'rt'})

stroop_clean = stroop_trials[
    (stroop_trials['rt'] > 200) &
    (stroop_trials['rt'] < 5000)
].copy()

print(f"  Valid trials: {len(stroop_clean)}")

for pid in demo['participant_id']:
    subset = stroop_clean[
        (stroop_clean['participant_id'] == pid) &
        (stroop_clean['type'] == 'incongruent')
    ]

    if len(subset) >= 10:
        rt_metrics.append({
            'participant_id': pid,
            'task': 'stroop',
            'condition': 'incongruent',
            'mean_rt': subset['rt'].mean(),
            'sd_rt': subset['rt'].std(),
            'n_trials': len(subset)
        })

print(f"  Stroop metrics: {len([r for r in rt_metrics if r['task'] == 'stroop'])} participants")

# === PRP ===
print("\n[2.2] PRP task...")
prp_trials, _ = load_prp_trials(use_cache=True)

rt_col = 't2_rt'
prp_clean = prp_trials[
    (prp_trials[rt_col] > 200) &
    (prp_trials[rt_col] < 5000)
].copy()

print(f"  Valid trials: {len(prp_clean)}")

for pid in demo['participant_id']:
    subset = prp_clean[
        (prp_clean['participant_id'] == pid) &
        (prp_clean['soa'] >= 1200)
    ]

    if len(subset) >= 10:
        rt_metrics.append({
            'participant_id': pid,
            'task': 'prp',
            'condition': 'long_soa',
            'mean_rt': subset[rt_col].mean(),
            'sd_rt': subset[rt_col].std(),
            'n_trials': len(subset)
        })

print(f"  PRP metrics: {len([r for r in rt_metrics if r['task'] == 'prp'])} participants")

# === WCST ===
print("\n[2.3] WCST task...")
wcst_trials, _ = load_wcst_trials(use_cache=True)
rt_col_wcst = 'reactionTimeMs' if 'reactionTimeMs' in wcst_trials.columns else ('rt_ms' if 'rt_ms' in wcst_trials.columns else None)
if rt_col_wcst:
    wcst_clean = wcst_trials[
        (wcst_trials[rt_col_wcst] > 200) &
        (wcst_trials[rt_col_wcst] < 10000)
    ].copy()

    print(f"  Valid trials: {len(wcst_clean)}")

    for pid in demo['participant_id']:
        subset = wcst_clean[wcst_clean['participant_id'] == pid]

        if len(subset) >= 20:
            rt_metrics.append({
                'participant_id': pid,
                'task': 'wcst',
                'condition': 'overall',
                'mean_rt': subset[rt_col_wcst].mean(),
                'sd_rt': subset[rt_col_wcst].std(),
                'n_trials': len(subset)
            })

print(f"  WCST metrics: {len([r for r in rt_metrics if r['task'] == 'wcst'])} participants")

# Convert to DataFrame
rt_df = pd.DataFrame(rt_metrics)
rt_df = rt_df.merge(demo, on='participant_id', how='inner')

print(f"\nTotal RT metrics: {len(rt_df)} (across {rt_df['participant_id'].nunique()} participants)")

# Save
rt_df.to_csv(OUTPUT_DIR / "paper4_mean_sd_metrics.csv", index=False, encoding='utf-8-sig')
print(f"Saved: paper4_mean_sd_metrics.csv")

# === HIERARCHICAL REGRESSION ANALYSIS ===
print("\n" + "="*100)
print("HIERARCHICAL REGRESSION: MEAN RT vs SD RT")
print("="*100)

regression_results = []

for task in ['stroop', 'prp', 'wcst']:
    task_data = rt_df[rt_df['task'] == task].copy()

    if len(task_data) < 30:
        print(f"\n[{task.upper()}] SKIPPED (N={len(task_data)} too small)")
        continue

    print(f"\n[{task.upper()}] N={len(task_data)}")
    print("-"*100)

    # Standardize predictors
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    task_data[['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']] = scaler.fit_transform(
        task_data[['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']]
    )

    # === MEAN RT ===
    print("\n  [MEAN RT] Location parameter")

    # Model 1: Covariates only
    model_mean_base = smf.ols("mean_rt ~ z_dass_dep + z_dass_anx + z_dass_str + z_age", data=task_data).fit()

    # Model 2: Add UCLA main effect
    model_mean_ucla = smf.ols("mean_rt ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=task_data).fit()

    # Model 3: Add UCLA × Gender interaction
    model_mean_full = smf.ols("mean_rt ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=task_data).fit()

    print(f"    Base R²={model_mean_base.rsquared:.3f}")
    print(f"    +UCLA R²={model_mean_ucla.rsquared:.3f}, ΔR²={model_mean_ucla.rsquared - model_mean_base.rsquared:.3f}")
    print(f"    +Interaction R²={model_mean_full.rsquared:.3f}, ΔR²={model_mean_full.rsquared - model_mean_ucla.rsquared:.3f}")

    # Extract coefficients
    if 'z_ucla' in model_mean_ucla.params:
        ucla_beta_mean = model_mean_ucla.params['z_ucla']
        ucla_p_mean = model_mean_ucla.pvalues['z_ucla']
        print(f"    UCLA main effect: β={ucla_beta_mean:.3f}, p={ucla_p_mean:.4f}")

    if 'z_ucla:C(gender_male)[T.1]' in model_mean_full.params:
        int_beta_mean = model_mean_full.params['z_ucla:C(gender_male)[T.1]']
        int_p_mean = model_mean_full.pvalues['z_ucla:C(gender_male)[T.1]']
        print(f"    UCLA × Gender: β={int_beta_mean:.3f}, p={int_p_mean:.4f}")

    # === SD RT ===
    print("\n  [SD RT] Scale parameter")

    # Model 1: Covariates only
    model_sd_base = smf.ols("sd_rt ~ z_dass_dep + z_dass_anx + z_dass_str + z_age", data=task_data).fit()

    # Model 2: Add UCLA main effect
    model_sd_ucla = smf.ols("sd_rt ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=task_data).fit()

    # Model 3: Add UCLA × Gender interaction
    model_sd_full = smf.ols("sd_rt ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age", data=task_data).fit()

    print(f"    Base R²={model_sd_base.rsquared:.3f}")
    print(f"    +UCLA R²={model_sd_ucla.rsquared:.3f}, ΔR²={model_sd_ucla.rsquared - model_sd_base.rsquared:.3f}")
    print(f"    +Interaction R²={model_sd_full.rsquared:.3f}, ΔR²={model_sd_full.rsquared - model_sd_ucla.rsquared:.3f}")

    # Extract coefficients
    if 'z_ucla' in model_sd_ucla.params:
        ucla_beta_sd = model_sd_ucla.params['z_ucla']
        ucla_p_sd = model_sd_ucla.pvalues['z_ucla']
        print(f"    UCLA main effect: β={ucla_beta_sd:.3f}, p={ucla_p_sd:.4f}")

    if 'z_ucla:C(gender_male)[T.1]' in model_sd_full.params:
        int_beta_sd = model_sd_full.params['z_ucla:C(gender_male)[T.1]']
        int_p_sd = model_sd_full.pvalues['z_ucla:C(gender_male)[T.1]']
        print(f"    UCLA × Gender: β={int_beta_sd:.3f}, p={int_p_sd:.4f}")

    # Store results
    regression_results.append({
        'task': task,
        'outcome': 'mean_rt',
        'ucla_beta': ucla_beta_mean if 'z_ucla' in model_mean_ucla.params else np.nan,
        'ucla_p': ucla_p_mean if 'z_ucla' in model_mean_ucla.params else np.nan,
        'interaction_beta': int_beta_mean if 'z_ucla:C(gender_male)[T.1]' in model_mean_full.params else np.nan,
        'interaction_p': int_p_mean if 'z_ucla:C(gender_male)[T.1]' in model_mean_full.params else np.nan
    })

    regression_results.append({
        'task': task,
        'outcome': 'sd_rt',
        'ucla_beta': ucla_beta_sd if 'z_ucla' in model_sd_ucla.params else np.nan,
        'ucla_p': ucla_p_sd if 'z_ucla' in model_sd_ucla.params else np.nan,
        'interaction_beta': int_beta_sd if 'z_ucla:C(gender_male)[T.1]' in model_sd_full.params else np.nan,
        'interaction_p': int_p_sd if 'z_ucla:C(gender_male)[T.1]' in model_sd_full.params else np.nan
    })

# Save results
results_df = pd.DataFrame(regression_results)
results_df.to_csv(OUTPUT_DIR / "paper4_regression_results.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: paper4_regression_results.csv")

# === COMPARE WITH TAU ===
print("\n" + "="*100)
print("COMPARISON: MEAN vs SD vs TAU SENSITIVITY")
print("="*100)

# Merge with tau from Paper 1
comparison_data = rt_df.merge(
    paper1_metrics[['participant_id', 'prp_tau_long', 'wcst_tau', 'stroop_tau_incong']],
    on='participant_id',
    how='inner'
)

comparison_results = []

for task in ['prp', 'wcst', 'stroop']:
    task_data = comparison_data[comparison_data['task'] == task].copy()

    if len(task_data) < 30:
        continue

    # Determine tau column
    if task == 'prp':
        tau_col = 'prp_tau_long'
    elif task == 'wcst':
        tau_col = 'wcst_tau'
    else:
        tau_col = 'stroop_tau_incong'

    # Gender-stratified correlations
    males = task_data[task_data['gender_male'] == 1]
    females = task_data[task_data['gender_male'] == 0]

    if len(males) >= 10 and len(females) >= 10:
        # Mean RT
        r_mean_m, p_mean_m = stats.pearsonr(males['ucla_total'], males['mean_rt'])
        r_mean_f, p_mean_f = stats.pearsonr(females['ucla_total'], females['mean_rt'])

        # SD RT
        r_sd_m, p_sd_m = stats.pearsonr(males['ucla_total'], males['sd_rt'])
        r_sd_f, p_sd_f = stats.pearsonr(females['ucla_total'], females['sd_rt'])

        # Tau
        r_tau_m, p_tau_m = stats.pearsonr(males['ucla_total'], males[tau_col])
        r_tau_f, p_tau_f = stats.pearsonr(females['ucla_total'], females[tau_col])

        comparison_results.append({
            'task': task,
            'metric': 'mean_rt',
            'r_male': r_mean_m,
            'p_male': p_mean_m,
            'r_female': r_mean_f,
            'p_female': p_mean_f
        })

        comparison_results.append({
            'task': task,
            'metric': 'sd_rt',
            'r_male': r_sd_m,
            'p_male': p_sd_m,
            'r_female': r_sd_f,
            'p_female': p_sd_f
        })

        comparison_results.append({
            'task': task,
            'metric': 'tau',
            'r_male': r_tau_m,
            'p_male': p_tau_m,
            'r_female': r_tau_f,
            'p_female': p_tau_f
        })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(OUTPUT_DIR / "paper4_metric_comparison.csv", index=False, encoding='utf-8-sig')
print("\nSaved: paper4_metric_comparison.csv")

# Print comparison
print("\nMetric Sensitivity (Male correlations only):")
print("-"*100)
for task in comparison_df['task'].unique():
    subset = comparison_df[comparison_df['task'] == task]
    print(f"\n{task.upper()}:")
    for _, row in subset.iterrows():
        sig = "**" if row['p_male'] < 0.01 else ("*" if row['p_male'] < 0.05 else "")
        print(f"  {row['metric']:10s}: r={row['r_male']:6.3f}, p={row['p_male']:.4f} {sig}")

# === VISUALIZATION ===
print("\n" + "="*100)
print("CREATING VISUALIZATIONS")
print("="*100)

# Create 2x3 grid: Mean RT (top row) vs SD RT (bottom row) for 3 tasks
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

tasks = ['stroop', 'prp', 'wcst']
outcomes = ['mean_rt', 'sd_rt']
outcome_labels = ['Mean RT (ms)', 'SD RT (ms)']

for i, outcome in enumerate(outcomes):
    for j, task in enumerate(tasks):
        ax = axes[i, j]

        task_data = rt_df[rt_df['task'] == task]

        if len(task_data) < 20:
            ax.text(0.5, 0.5, 'Insufficient\nData', ha='center', va='center', fontsize=14)
            ax.set_title(f'{task.upper()}: {outcome_labels[i]}', fontweight='bold')
            ax.axis('off')
            continue

        # Scatter by gender
        males = task_data[task_data['gender_male'] == 1]
        females = task_data[task_data['gender_male'] == 0]

        ax.scatter(males['ucla_total'], males[outcome], c='#3498db', s=60, alpha=0.6, label='Males')
        ax.scatter(females['ucla_total'], females[outcome], c='#e74c3c', s=60, alpha=0.6, label='Females')

        # Regression lines
        if len(males) >= 10:
            z = np.polyfit(males['ucla_total'], males[outcome], 1)
            p = np.poly1d(z)
            x_range = np.linspace(males['ucla_total'].min(), males['ucla_total'].max(), 100)
            ax.plot(x_range, p(x_range), color='#3498db', linestyle='--', linewidth=2)

            r_m, p_m = stats.pearsonr(males['ucla_total'], males[outcome])
            ax.text(0.05, 0.95, f'M: r={r_m:.2f}, p={p_m:.3f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if len(females) >= 10:
            try:
                z = np.polyfit(females['ucla_total'], females[outcome], 1)
                p = np.poly1d(z)
                x_range = np.linspace(females['ucla_total'].min(), females['ucla_total'].max(), 100)
                ax.plot(x_range, p(x_range), color='#e74c3c', linestyle='--', linewidth=2)
            except:
                pass  # Skip regression line if it fails

            r_f, p_f = stats.pearsonr(females['ucla_total'], females[outcome])
            ax.text(0.05, 0.85, f'F: r={r_f:.2f}, p={p_f:.3f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('UCLA Loneliness' if i == 1 else '', fontsize=11)
        ax.set_ylabel(outcome_labels[i] if j == 0 else '', fontsize=11)
        ax.set_title(f'{task.upper()}: {outcome_labels[i]}', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)

        if i == 0 and j == 2:
            ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "paper4_mean_sd_scatter_grid.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved: paper4_mean_sd_scatter_grid.png")

# === SUMMARY REPORT ===
print("\n" + "="*100)
print("GENERATING SUMMARY REPORT")
print("="*100)

report_lines = [
    "="*100,
    "PAPER 4 (REVISED): MEAN RT vs SD RT ANALYSIS - SUMMARY REPORT",
    "="*100,
    "",
    "RESEARCH QUESTION:",
    "  Does loneliness affect RT LOCATION (mean) or SCALE (SD)?",
    "  How does this compare to TAU (lapses) from Paper 1?",
    "",
    "METHOD:",
    "-"*100,
    "  - Hierarchical regression with DASS control",
    "  - Gender stratification",
    "  - Comparison: Mean RT vs SD RT vs Tau sensitivity",
    "",
    f"SAMPLE SIZE:",
    f"  Stroop: {len(rt_df[rt_df['task'] == 'stroop'])} participants",
    f"  PRP: {len(rt_df[rt_df['task'] == 'prp'])} participants",
    f"  WCST: {len(rt_df[rt_df['task'] == 'wcst'])} participants",
    "",
    "KEY FINDINGS:",
    "-"*100
]

# Add significant findings
for task in comparison_df['task'].unique():
    subset = comparison_df[(comparison_df['task'] == task) & (comparison_df['p_male'] < 0.05)]
    if len(subset) > 0:
        report_lines.append(f"\n{task.upper()} - SIGNIFICANT MALE EFFECTS:")
        for _, row in subset.iterrows():
            report_lines.append(f"  {row['metric']}: r={row['r_male']:.3f}, p={row['p_male']:.4f}")

report_lines.extend([
    "",
    "METRIC SENSITIVITY RANKING (by male correlation size):",
    "-"*100
])

# Rank metrics by absolute correlation
for task in comparison_df['task'].unique():
    subset = comparison_df[comparison_df['task'] == task].copy()
    subset['abs_r_male'] = subset['r_male'].abs()
    subset = subset.sort_values('abs_r_male', ascending=False)

    report_lines.append(f"\n{task.upper()}:")
    for idx, (_, row) in enumerate(subset.iterrows(), 1):
        report_lines.append(f"  #{idx}: {row['metric']} (r={row['r_male']:.3f})")

report_lines.extend([
    "",
    "OUTPUTS:",
    "-"*100,
    "  1. paper4_mean_sd_metrics.csv - Mean/SD RT for all participants",
    "  2. paper4_regression_results.csv - Hierarchical regression results",
    "  3. paper4_metric_comparison.csv - Mean vs SD vs Tau comparison",
    "  4. paper4_mean_sd_scatter_grid.png - 2x3 visualization (300 dpi)",
    "",
    "CONCLUSIONS:",
    "-"*100,
    "  - [To be filled based on actual results]",
    "  - Complements Paper 1's focus on lapses (tau)",
    "  - Simpler and more robust than DDM approach",
    "",
    "="*100,
    "END OF REPORT",
    "="*100
])

report_text = "\n".join(report_lines)
with open(OUTPUT_DIR / "PAPER4_MEAN_SD_SUMMARY.txt", "w", encoding='utf-8') as f:
    f.write(report_text)

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print("\nFiles created:")
print("  1. paper4_mean_sd_metrics.csv")
print("  2. paper4_regression_results.csv")
print("  3. paper4_metric_comparison.csv")
print("  4. paper4_mean_sd_scatter_grid.png")
print("  5. PAPER4_MEAN_SD_SUMMARY.txt")
print("="*100)
