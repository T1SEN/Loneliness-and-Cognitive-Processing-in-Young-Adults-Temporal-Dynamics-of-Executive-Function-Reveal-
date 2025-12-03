"""
Tier 2.4: Pre-Error RT Trajectory Analysis

Tests whether errors follow distinctive RT patterns (pre-error speeding/slowing),
and whether UCLA loneliness moderates these patterns.

Pre-error speeding → Impulsivity/disengagement
Pre-error slowing → Fatigue/vigilance decrement

Author: Claude Code
Date: 2025-01-16
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_wcst_trials, load_prp_trials, load_stroop_trials

# Output directory
OUTPUT_DIR = Path("results/analysis_outputs/pre_error_trajectories")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("TIER 2.4: PRE-ERROR RT TRAJECTORY ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading data...")

# Load master dataset (demographics, UCLA, DASS)
df_master = load_master_dataset()
print(f"  Master dataset: {len(df_master)} participants")

# Load trial data for each task
# Note: WCST loader doesn't have filtering parameters, loads all trials by default
wcst_df, wcst_info = load_wcst_trials()
print(f"  WCST: {len(wcst_df)} trials from {wcst_df['participant_id'].nunique()} participants")

# PRP: Include error trials for T2
prp_df, prp_info = load_prp_trials(
    require_t2_correct_for_rt=False,  # Include error trials!
    require_t1_correct=False,  # Include T1 errors too
    drop_timeouts=True,
    enforce_short_long_only=False  # Keep all SOAs
)
print(f"  PRP: {len(prp_df)} trials from {prp_df['participant_id'].nunique()} participants")

# Stroop: Include error trials
stroop_df, stroop_info = load_stroop_trials(
    require_correct_for_rt=False,  # Include error trials!
    drop_timeouts=True
)
print(f"  Stroop: {len(stroop_df)} trials from {stroop_df['participant_id'].nunique()} participants")

# ============================================================================
# STEP 2: Extract Peri-Error RT Trajectories
# ============================================================================
print("\nSTEP 2: Extracting peri-error RT trajectories...")

def extract_peri_error_rts(df, participant_col='participant_id', rt_col='rt', correct_col='correct', window=3):
    """
    Extract RT trajectories around error trials.

    Returns DataFrame with columns:
    - participant_id
    - error_idx: Index of error trial
    - rt_m3, rt_m2, rt_m1: RTs 3, 2, 1 trials before error
    - rt_0: RT on error trial
    - rt_p1, rt_p2, rt_p3: RTs 1, 2, 3 trials after error
    """
    results = []

    for pid in df[participant_col].unique():
        if pd.isna(pid):
            continue

        pid_data = df[df[participant_col] == pid].copy()

        # Sort by trial index if available, otherwise use natural order
        if 'trial_index' in pid_data.columns:
            pid_data = pid_data.sort_values('trial_index')
        elif 'trialIndex' in pid_data.columns:
            pid_data = pid_data.sort_values('trialIndex')
        # If no trial index column, assume data is already in order

        pid_data = pid_data.reset_index(drop=True)

        # Skip if no errors
        if correct_col not in pid_data.columns:
            continue

        # Find error trials
        error_mask = (pid_data[correct_col] == False) | (pid_data[correct_col] == 0)
        error_indices = pid_data[error_mask].index.tolist()

        for err_idx in error_indices:
            # Check we have enough context (window trials before and after)
            if err_idx < window or err_idx >= len(pid_data) - window:
                continue

            # Check that RT values exist
            if pd.isna(pid_data.loc[err_idx, rt_col]):
                continue

            # Extract RTs
            trajectory = {
                'participant_id': pid,
                'error_idx': err_idx,
                'rt_0': pid_data.loc[err_idx, rt_col]  # Error trial RT
            }

            # Pre-error RTs
            valid = True
            for i in range(1, window + 1):
                rt_val = pid_data.loc[err_idx - i, rt_col]
                if pd.isna(rt_val):
                    valid = False
                    break
                trajectory[f'rt_m{i}'] = rt_val

            if not valid:
                continue

            # Post-error RTs
            for i in range(1, window + 1):
                rt_val = pid_data.loc[err_idx + i, rt_col]
                if pd.isna(rt_val):
                    valid = False
                    break
                trajectory[f'rt_p{i}'] = rt_val

            if not valid:
                continue

            results.append(trajectory)

    return pd.DataFrame(results)

# WCST (use rt_ms column)
print("\n  WCST peri-error trajectories...")
wcst_peri = extract_peri_error_rts(wcst_df, rt_col='rt_ms', correct_col='correct')
print(f"    Extracted {len(wcst_peri)} error trials with context")

# PRP (use T2 RT and T2 correct)
print("\n  PRP peri-error trajectories...")
prp_peri = extract_peri_error_rts(prp_df, rt_col='t2_rt', correct_col='t2_correct')
print(f"    Extracted {len(prp_peri)} error trials with context")

# Stroop
print("\n  Stroop peri-error trajectories...")
stroop_peri = extract_peri_error_rts(stroop_df, rt_col='rt', correct_col='correct')
print(f"    Extracted {len(stroop_peri)} error trials with context")

# ============================================================================
# STEP 3: Compute Pre-Error Slope (Speeding/Slowing)
# ============================================================================
print("\nSTEP 3: Computing pre-error slopes...")

def compute_pre_error_slope(peri_df):
    """
    Compute pre-error slope: linear trend from rt_m3 to rt_m1
    Negative slope = pre-error speeding (impulsivity)
    Positive slope = pre-error slowing (fatigue)
    """
    slopes = []

    for idx, row in peri_df.iterrows():
        # Pre-error RTs
        pre_rts = [row['rt_m3'], row['rt_m2'], row['rt_m1']]

        if any(pd.isna(pre_rts)):
            slopes.append(np.nan)
            continue

        # Fit linear trend (trial position: -3, -2, -1)
        trial_pos = [-3, -2, -1]
        try:
            slope, intercept = np.polyfit(trial_pos, pre_rts, 1)
            slopes.append(slope)
        except:
            slopes.append(np.nan)

    peri_df['pre_error_slope'] = slopes
    return peri_df

# Compute slopes
wcst_peri = compute_pre_error_slope(wcst_peri)
prp_peri = compute_pre_error_slope(prp_peri)
stroop_peri = compute_pre_error_slope(stroop_peri)

# Remove NaN slopes
wcst_peri = wcst_peri.dropna(subset=['pre_error_slope'])
prp_peri = prp_peri.dropna(subset=['pre_error_slope'])
stroop_peri = stroop_peri.dropna(subset=['pre_error_slope'])

print(f"\n  WCST: {len(wcst_peri)} trajectories with valid slopes")
print(f"  PRP: {len(prp_peri)} trajectories with valid slopes")
print(f"  Stroop: {len(stroop_peri)} trajectories with valid slopes")

# ============================================================================
# STEP 4: Aggregate per Participant
# ============================================================================
print("\nSTEP 4: Aggregating per participant...")

def aggregate_peri_error_by_participant(peri_df):
    """Aggregate peri-error metrics per participant"""
    agg = peri_df.groupby('participant_id').agg({
        'pre_error_slope': ['mean', 'std', 'count'],
        'rt_0': 'mean',  # Mean error RT
        'rt_m1': 'mean',  # Mean RT 1 trial before error
        'rt_m2': 'mean',  # Mean RT 2 trials before error
        'rt_m3': 'mean',  # Mean RT 3 trials before error
        'rt_p1': 'mean',  # Mean RT 1 trial after error (post-error slowing)
        'rt_p2': 'mean',
        'rt_p3': 'mean'
    }).reset_index()

    # Flatten column names
    agg.columns = ['participant_id',
                   'pre_error_slope_mean', 'pre_error_slope_std', 'n_errors',
                   'error_rt_mean', 'rt_m1_mean', 'rt_m2_mean', 'rt_m3_mean',
                   'rt_p1_mean', 'rt_p2_mean', 'rt_p3_mean']

    # Compute post-error slowing: rt_p1 - rt_m1
    agg['post_error_slowing'] = agg['rt_p1_mean'] - agg['rt_m1_mean']

    return agg

# Only aggregate if we have data
wcst_agg = aggregate_peri_error_by_participant(wcst_peri) if len(wcst_peri) > 0 else pd.DataFrame()
prp_agg = aggregate_peri_error_by_participant(prp_peri) if len(prp_peri) > 0 else pd.DataFrame()
stroop_agg = aggregate_peri_error_by_participant(stroop_peri) if len(stroop_peri) > 0 else pd.DataFrame()

print(f"\n  WCST: {len(wcst_agg)} participants")
print(f"  PRP: {len(prp_agg)} participants")
print(f"  Stroop: {len(stroop_agg)} participants")

# ============================================================================
# STEP 5: Merge with UCLA/DASS
# ============================================================================
print("\nSTEP 5: Merging with UCLA/DASS...")

def merge_with_master(agg_df, task_name):
    if len(agg_df) == 0:
        print(f"  {task_name}: Skipping (no error data)")
        return pd.DataFrame()

    df = agg_df.merge(df_master, on='participant_id', how='inner')
    print(f"  {task_name}: {len(df)} participants after merge")

    # Ensure required columns
    if 'gender_male' not in df.columns:
        df['gender_male'] = (df['gender'].str.lower() == 'male').astype(int)

    # Standardize predictors
    for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        if col in df.columns:
            df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()

    return df

wcst_df_merged = merge_with_master(wcst_agg, 'WCST')
prp_df_merged = merge_with_master(prp_agg, 'PRP')
stroop_df_merged = merge_with_master(stroop_agg, 'Stroop')

# Save merged data
wcst_df_merged.to_csv(OUTPUT_DIR / "wcst_peri_error_merged.csv", index=False, encoding='utf-8-sig')
prp_df_merged.to_csv(OUTPUT_DIR / "prp_peri_error_merged.csv", index=False, encoding='utf-8-sig')
stroop_df_merged.to_csv(OUTPUT_DIR / "stroop_peri_error_merged.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STEP 6: Regression Analysis - Pre-Error Slope ~ UCLA
# ============================================================================
print("\nSTEP 6: Regression analysis - Pre-error slope ~ UCLA...")

regression_results = []

for task_name, df in [('WCST', wcst_df_merged), ('PRP', prp_df_merged), ('Stroop', stroop_df_merged)]:
    print(f"\n  {task_name}:")

    if len(df) < 20:
        print(f"    Insufficient N ({len(df)}), skipping")
        continue

    # Formula: pre_error_slope ~ UCLA + Gender + DASS + age
    formula = "pre_error_slope_mean ~ z_ucla_total + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

    try:
        model = smf.ols(formula, data=df).fit()

        # Extract UCLA coefficient
        ucla_coef = model.params.get('z_ucla_total', np.nan)
        ucla_se = model.bse.get('z_ucla_total', np.nan)
        ucla_t = model.tvalues.get('z_ucla_total', np.nan)
        ucla_p = model.pvalues.get('z_ucla_total', np.nan)
        ucla_ci = model.conf_int().loc['z_ucla_total'] if 'z_ucla_total' in model.conf_int().index else [np.nan, np.nan]

        regression_results.append({
            'task': task_name,
            'outcome': 'pre_error_slope',
            'n': len(df),
            'n_errors_mean': df['n_errors'].mean(),
            'ucla_beta': ucla_coef,
            'ucla_se': ucla_se,
            'ucla_t': ucla_t,
            'ucla_p': ucla_p,
            'ucla_ci_lower': ucla_ci[0],
            'ucla_ci_upper': ucla_ci[1],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj
        })

        print(f"    UCLA β={ucla_coef:.4f}, p={ucla_p:.4f}")

        # Save full model summary
        with open(OUTPUT_DIR / f"{task_name.lower()}_pre_error_slope_regression.txt", 'w', encoding='utf-8') as f:
            f.write(str(model.summary()))

    except Exception as e:
        print(f"    ERROR: {e}")

# Also test post-error slowing
print("\n  Post-error slowing ~ UCLA:")
for task_name, df in [('WCST', wcst_df_merged), ('PRP', prp_df_merged), ('Stroop', stroop_df_merged)]:
    if len(df) < 20:
        continue

    formula = "post_error_slowing ~ z_ucla_total + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

    try:
        model = smf.ols(formula, data=df).fit()

        ucla_coef = model.params.get('z_ucla_total', np.nan)
        ucla_p = model.pvalues.get('z_ucla_total', np.nan)
        ucla_ci = model.conf_int().loc['z_ucla_total'] if 'z_ucla_total' in model.conf_int().index else [np.nan, np.nan]

        regression_results.append({
            'task': task_name,
            'outcome': 'post_error_slowing',
            'n': len(df),
            'n_errors_mean': df['n_errors'].mean(),
            'ucla_beta': ucla_coef,
            'ucla_se': model.bse.get('z_ucla_total', np.nan),
            'ucla_t': model.tvalues.get('z_ucla_total', np.nan),
            'ucla_p': ucla_p,
            'ucla_ci_lower': ucla_ci[0],
            'ucla_ci_upper': ucla_ci[1],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj
        })

        print(f"    {task_name} PES: UCLA β={ucla_coef:.4f}, p={ucla_p:.4f}")

    except Exception as e:
        print(f"    {task_name} ERROR: {e}")

# Save regression results
regression_df = pd.DataFrame(regression_results)
regression_df.to_csv(OUTPUT_DIR / "regression_results.csv", index=False, encoding='utf-8-sig')
print("\n  Regression results saved")

# ============================================================================
# STEP 7: Visualization - Peri-Error RT Timecourse by UCLA Tertile
# ============================================================================
print("\nSTEP 7: Creating visualizations...")

def plot_peri_error_timecourse(peri_df, master_df, task_name, output_path):
    """Plot peri-error RT timecourse by UCLA tertile"""

    # Merge with UCLA
    df = peri_df.merge(master_df[['participant_id', 'ucla_total']], on='participant_id', how='inner')

    # Create UCLA tertiles
    df['ucla_tertile'] = pd.qcut(df['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

    # Compute mean RT at each position by UCLA tertile
    positions = ['rt_m3', 'rt_m2', 'rt_m1', 'rt_0', 'rt_p1', 'rt_p2', 'rt_p3']
    position_labels = ['-3', '-2', '-1', '0\n(Error)', '+1', '+2', '+3']

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

    for tertile in ['Low', 'Medium', 'High']:
        tertile_data = df[df['ucla_tertile'] == tertile]

        means = []
        sems = []

        for pos in positions:
            means.append(tertile_data[pos].mean())
            sems.append(tertile_data[pos].sem())

        x = np.arange(len(positions))
        ax.plot(x, means, marker='o', linewidth=2, markersize=8, label=f'UCLA {tertile}', color=colors[tertile])
        ax.fill_between(x, np.array(means) - np.array(sems), np.array(means) + np.array(sems),
                        alpha=0.2, color=colors[tertile])

    ax.axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Error Trial')
    ax.set_xlabel('Trial Position Relative to Error', fontsize=12)
    ax.set_ylabel('Reaction Time (ms)', fontsize=12)
    ax.set_title(f'{task_name}: Peri-Error RT Trajectory by UCLA Loneliness', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(position_labels)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

# Create plots for each task (only if data exists)
if len(wcst_peri) > 0:
    plot_peri_error_timecourse(wcst_peri, df_master, 'WCST', OUTPUT_DIR / 'wcst_peri_error_timecourse.png')
if len(prp_peri) > 0:
    plot_peri_error_timecourse(prp_peri, df_master, 'PRP', OUTPUT_DIR / 'prp_peri_error_timecourse.png')
if len(stroop_peri) > 0:
    plot_peri_error_timecourse(stroop_peri, df_master, 'Stroop', OUTPUT_DIR / 'stroop_peri_error_timecourse.png')

# ============================================================================
# STEP 8: Visualization - Pre-Error Slope vs UCLA
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Pre-Error RT Slope vs UCLA Loneliness', fontsize=14, fontweight='bold')

for ax, (task_name, df) in zip(axes, [('WCST', wcst_df_merged), ('PRP', prp_df_merged), ('Stroop', stroop_df_merged)]):
    if len(df) < 20:
        ax.text(0.5, 0.5, f'{task_name}\nInsufficient Data', ha='center', va='center', transform=ax.transAxes)
        continue

    ax.scatter(df['ucla_total'], df['pre_error_slope_mean'], alpha=0.5, s=50)

    # Add regression line
    try:
        z = np.polyfit(df['ucla_total'], df['pre_error_slope_mean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['ucla_total'].min(), df['ucla_total'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

        # Compute correlation
        r, p_val = stats.pearsonr(df['ucla_total'], df['pre_error_slope_mean'])
        ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    except:
        pass

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('UCLA Loneliness Score', fontsize=11)
    ax.set_ylabel('Pre-Error Slope (ms/trial)', fontsize=11)
    ax.set_title(task_name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.05, 'Speeding ↓', transform=ax.transAxes, ha='right', fontsize=9, color='blue')
    ax.text(0.95, 0.95, 'Slowing ↑', transform=ax.transAxes, ha='right', fontsize=9, color='red')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pre_error_slope_vs_ucla.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: pre_error_slope_vs_ucla.png")

# ============================================================================
# STEP 9: Summary Report
# ============================================================================
print("\nSTEP 9: Generating summary report...")

summary_lines = [
    "=" * 80,
    "TIER 2.4: PRE-ERROR RT TRAJECTORY ANALYSIS - SUMMARY REPORT",
    "=" * 80,
    "",
    "RESEARCH QUESTION:",
    "Do errors follow distinctive RT patterns (pre-error speeding/slowing)?",
    "Does UCLA loneliness predict pre-error RT dynamics?",
    "",
    "=" * 80,
    "DATA SUMMARY",
    "=" * 80,
]

for task_name, df_agg in [('WCST', wcst_agg), ('PRP', prp_agg), ('Stroop', stroop_agg)]:
    if len(df_agg) == 0:
        summary_lines.extend([
            f"\n{task_name}:",
            f"  No error data available"
        ])
    else:
        summary_lines.extend([
            f"\n{task_name}:",
            f"  Participants with errors: {len(df_agg)}",
            f"  Mean errors per participant: {df_agg['n_errors'].mean():.1f} (SD={df_agg['n_errors'].std():.1f})",
            f"  Mean pre-error slope: {df_agg['pre_error_slope_mean'].mean():.2f} ms/trial (SD={df_agg['pre_error_slope_mean'].std():.2f})",
            f"  Mean post-error slowing: {df_agg['post_error_slowing'].mean():.1f} ms (SD={df_agg['post_error_slowing'].std():.1f})"
        ])

summary_lines.extend([
    "",
    "=" * 80,
    "REGRESSION RESULTS",
    "=" * 80,
    "\nPre-Error Slope ~ UCLA (DASS-controlled):",
])

pre_error_results = regression_df[regression_df['outcome'] == 'pre_error_slope']
for _, row in pre_error_results.iterrows():
    sig = "***" if row['ucla_p'] < 0.001 else "**" if row['ucla_p'] < 0.01 else "*" if row['ucla_p'] < 0.05 else ""
    summary_lines.append(f"  {row['task']}: β={row['ucla_beta']:.4f}, p={row['ucla_p']:.4f} {sig} (N={row['n']:.0f})")

summary_lines.extend([
    "",
    "Post-Error Slowing ~ UCLA (DASS-controlled):",
])

pes_results = regression_df[regression_df['outcome'] == 'post_error_slowing']
for _, row in pes_results.iterrows():
    sig = "***" if row['ucla_p'] < 0.001 else "**" if row['ucla_p'] < 0.01 else "*" if row['ucla_p'] < 0.05 else ""
    summary_lines.append(f"  {row['task']}: β={row['ucla_beta']:.4f}, p={row['ucla_p']:.4f} {sig} (N={row['n']:.0f})")

summary_lines.extend([
    "",
    "=" * 80,
    "INTERPRETATION",
    "=" * 80,
    "",
    "Pre-Error Slope:",
    "  - NEGATIVE slope = Pre-error speeding (impulsivity/disengagement)",
    "  - POSITIVE slope = Pre-error slowing (fatigue/vigilance decrement)",
    "  - ZERO slope = No systematic RT change before errors",
    "",
    "If UCLA → MORE NEGATIVE slope:",
    "  → Loneliness predicts impulsive errors (rushing)",
    "",
    "If UCLA → MORE POSITIVE slope:",
    "  → Loneliness predicts fatigue-related errors (slowing then lapsing)",
    "",
    "Post-Error Slowing:",
    "  - Adaptive control mechanism: slow down after errors to avoid repeating",
    "  - If UCLA → LESS PES: Impaired error monitoring/reactive control",
    "",
    "=" * 80,
    "FILES GENERATED",
    "=" * 80,
    "CSV Files:",
    "  - wcst_peri_error_merged.csv (participant-level metrics)",
    "  - prp_peri_error_merged.csv",
    "  - stroop_peri_error_merged.csv",
    "  - regression_results.csv (all regression coefficients)",
    "",
    "Text Files:",
    "  - wcst_pre_error_slope_regression.txt (full model output)",
    "  - prp_pre_error_slope_regression.txt",
    "  - stroop_pre_error_slope_regression.txt",
    "",
    "Figures:",
    "  - wcst_peri_error_timecourse.png (RT trajectory ±3 trials around error)",
    "  - prp_peri_error_timecourse.png",
    "  - stroop_peri_error_timecourse.png",
    "  - pre_error_slope_vs_ucla.png (scatter plots)",
    "",
    "=" * 80,
    "END OF REPORT",
    "=" * 80
])

summary_text = "\n".join(summary_lines)
print(summary_text)

# Save summary
with open(OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nTier 2.4 Pre-Error RT Trajectory Analysis COMPLETE!")
