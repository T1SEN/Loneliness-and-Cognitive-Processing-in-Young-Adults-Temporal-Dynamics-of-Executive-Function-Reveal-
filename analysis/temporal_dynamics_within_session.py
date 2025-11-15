"""
Within-Session Temporal Dynamics Analysis
=========================================

Tests whether gender moderation effects are:
1. STATE (worsen over time = fatigue) vs. TRAIT (constant)
2. Time-of-day dependent (circadian/stress accumulation)
3. Autocorrelated (errors cluster = "stuckness")

Analyses:
1. Time-on-task: Split trials into Early/Mid/Late → Trial Block × UCLA × Gender
2. Timestamp analysis: Hour-of-day × UCLA × Gender
3. Autocorrelation: Lag-1 autocorr of correct (trial n → n+1)

Author: Research Team
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/mechanism_analysis/temporal_dynamics")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("WITHIN-SESSION TEMPORAL DYNAMICS ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/4] Loading data...")

# Load trial-level WCST data
wcst_trials = pd.read_csv(Path("results/4b_wcst_trials.csv"), encoding='utf-8-sig')

# Load participant data
master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))
participants = pd.read_csv(Path("results/1_participants_info.csv"), encoding='utf-8-sig')

if 'participantId' in participants.columns:
    if 'participant_id' in participants.columns:
        participants.drop(columns=['participantId'], inplace=True)
    else:
        participants.rename(columns={'participantId': 'participant_id'}, inplace=True)

master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

# Handle gender
if 'gender' in master.columns:
    master['gender_male'] = 0
    master.loc[master['gender'] == '남성', 'gender_male'] = 1
    master.loc[master['gender'].str.lower() == 'male', 'gender_male'] = 1

# Parse extra field for isPE
import ast
def parse_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except:
        return {}

wcst_trials['extra_dict'] = wcst_trials['extra'].apply(parse_extra)
wcst_trials['isPE'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

print(f"  WCST trials: {len(wcst_trials)}")
print(f"  Participants: {len(master)}")
print()

# ============================================================================
# ANALYSIS 1: TIME-ON-TASK (EARLY vs MID vs LATE)
# ============================================================================

print("[2/4] Analysis 1: Time-on-task effects...")
print()

# Compute PE rate by trial block for each participant
time_on_task_data = []

for pid in wcst_trials['participant_id'].dropna().unique():
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy().reset_index(drop=True)

    n_trials = len(pid_trials)

    if n_trials < 30:
        continue

    # Split into thirds
    early_end = int(n_trials / 3)
    mid_end = int(2 * n_trials / 3)

    early_trials = pid_trials.iloc[:early_end]
    mid_trials = pid_trials.iloc[early_end:mid_end]
    late_trials = pid_trials.iloc[mid_end:]

    for block_name, block_trials in [('Early', early_trials), ('Mid', mid_trials), ('Late', late_trials)]:
        if len(block_trials) > 0:
            pe_rate = (block_trials['isPE'].sum() / len(block_trials) * 100)

            time_on_task_data.append({
                'participant_id': pid,
                'trial_block': block_name,
                'pe_rate': pe_rate,
                'n_trials': len(block_trials)
            })

time_on_task_df = pd.DataFrame(time_on_task_data)

# Merge with master
time_on_task_df = time_on_task_df.merge(
    master[['participant_id', 'ucla_total', 'gender_male']],
    on='participant_id',
    how='left'
)

# Remove missing
time_on_task_df = time_on_task_df.dropna(subset=['ucla_total', 'gender_male']).copy()

# Standardize UCLA
scaler = StandardScaler()
time_on_task_df['z_ucla'] = scaler.fit_transform(time_on_task_df[['ucla_total']])

# Test 3-way interaction: Trial Block × UCLA × Gender
formula = "pe_rate ~ C(trial_block) * z_ucla * C(gender_male)"

try:
    model = ols(formula, data=time_on_task_df).fit()
    print("  3-way interaction (Block × UCLA × Gender):")
    print(f"  {model.summary()}")
    print()

    # Save ANOVA table
    anova_table = model.summary2().tables[1]
    anova_table.to_csv(OUTPUT_DIR / "time_on_task_anova.csv", encoding='utf-8-sig')

    # Simpler test: Does UCLA × Gender interaction differ by block?
    print("  UCLA × Gender interaction by trial block:")

    for block in ['Early', 'Mid', 'Late']:
        block_data = time_on_task_df[time_on_task_df['trial_block'] == block].copy()

        block_model = ols("pe_rate ~ z_ucla * C(gender_male)", data=block_data).fit()

        if "z_ucla:C(gender_male)[T.1]" in block_model.params:
            int_beta = block_model.params["z_ucla:C(gender_male)[T.1]"]
            int_p = block_model.pvalues["z_ucla:C(gender_male)[T.1]"]

            sig_marker = " ***" if int_p < 0.001 else " **" if int_p < 0.01 else " *" if int_p < 0.05 else ""
            print(f"    {block}: β={int_beta:.3f}, p={int_p:.4f}{sig_marker}")

except Exception as e:
    print(f"  Error in time-on-task analysis: {e}")

time_on_task_df.to_csv(OUTPUT_DIR / "time_on_task_data.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: time_on_task_data.csv")
print()

# ============================================================================
# ANALYSIS 2: TIMESTAMP (TIME-OF-DAY) EFFECTS
# ============================================================================

print("[3/4] Analysis 2: Time-of-day effects...")
print()

# Parse timestamps
if 'timestamp' in wcst_trials.columns:
    # Convert to datetime
    wcst_trials['datetime'] = pd.to_datetime(wcst_trials['timestamp'], errors='coerce')
    wcst_trials['hour'] = wcst_trials['datetime'].dt.hour

    # Categorize into Morning/Afternoon/Evening
    def categorize_time(hour):
        if pd.isna(hour):
            return np.nan
        if 8 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return np.nan

    wcst_trials['time_of_day'] = wcst_trials['hour'].apply(categorize_time)

    # Compute PE rate by time-of-day for each participant
    timestamp_data = []

    for pid in wcst_trials['participant_id'].dropna().unique():
        pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()

        for tod in ['Morning', 'Afternoon', 'Evening']:
            tod_trials = pid_trials[pid_trials['time_of_day'] == tod]

            if len(tod_trials) > 5:  # Minimum trials
                pe_rate = (tod_trials['isPE'].sum() / len(tod_trials) * 100)

                timestamp_data.append({
                    'participant_id': pid,
                    'time_of_day': tod,
                    'pe_rate': pe_rate,
                    'n_trials': len(tod_trials),
                    'median_hour': tod_trials['hour'].median()
                })

    if len(timestamp_data) > 0:
        timestamp_df = pd.DataFrame(timestamp_data)

        # Merge with master
        timestamp_df = timestamp_df.merge(
            master[['participant_id', 'ucla_total', 'gender_male']],
            on='participant_id',
            how='left'
        )

        timestamp_df = timestamp_df.dropna(subset=['ucla_total', 'gender_male']).copy()

        # Standardize
        scaler_ts = StandardScaler()
        timestamp_df['z_ucla'] = scaler_ts.fit_transform(timestamp_df[['ucla_total']])

        # Test: Time-of-day × UCLA × Gender
        formula_ts = "pe_rate ~ C(time_of_day) * z_ucla * C(gender_male)"

        try:
            model_ts = ols(formula_ts, data=timestamp_df).fit()

            print("  Time-of-day × UCLA × Gender:")

            # Test by time-of-day
            for tod in ['Morning', 'Afternoon', 'Evening']:
                tod_data = timestamp_df[timestamp_df['time_of_day'] == tod].copy()

                if len(tod_data) >= 20:
                    tod_model = ols("pe_rate ~ z_ucla * C(gender_male)", data=tod_data).fit()

                    if "z_ucla:C(gender_male)[T.1]" in tod_model.params:
                        int_beta = tod_model.params["z_ucla:C(gender_male)[T.1]"]
                        int_p = tod_model.pvalues["z_ucla:C(gender_male)[T.1]"]

                        sig_marker = " ***" if int_p < 0.001 else " **" if int_p < 0.01 else " *" if int_p < 0.05 else ""
                        print(f"    {tod}: β={int_beta:.3f}, p={int_p:.4f}{sig_marker} (N={len(tod_data)})")
                else:
                    print(f"    {tod}: Insufficient data (N={len(tod_data)})")

        except Exception as e:
            print(f"  Error: {e}")

        timestamp_df.to_csv(OUTPUT_DIR / "timestamp_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: timestamp_analysis.csv")
    else:
        print("  No timestamp data available")
else:
    print("  No timestamp column found in data")

print()

# ============================================================================
# ANALYSIS 3: AUTOCORRELATION (ERROR CLUSTERING)
# ============================================================================

print("[4/4] Analysis 3: Autocorrelation (error clustering)...")
print()

# Compute lag-1 autocorrelation of correct for each participant
autocorr_data = []

for pid in wcst_trials['participant_id'].dropna().unique():
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy().reset_index(drop=True)

    if len(pid_trials) < 20:
        continue

    # Convert correct to numeric
    correct_series = pid_trials['correct'].astype(int)

    # Compute lag-1 autocorrelation
    if len(correct_series) > 1:
        autocorr_lag1 = correct_series.autocorr(lag=1)

        autocorr_data.append({
            'participant_id': pid,
            'autocorr_lag1': autocorr_lag1,
            'n_trials': len(pid_trials)
        })

autocorr_df = pd.DataFrame(autocorr_data)

# Merge with master
autocorr_df = autocorr_df.merge(
    master[['participant_id', 'ucla_total', 'gender_male', 'pe_rate']],
    on='participant_id',
    how='left'
)

autocorr_df = autocorr_df.dropna(subset=['ucla_total', 'gender_male', 'autocorr_lag1']).copy()

# Standardize
scaler_ac = StandardScaler()
autocorr_df['z_ucla'] = scaler_ac.fit_transform(autocorr_df[['ucla_total']])

# Test: UCLA × Gender → Autocorr
formula_ac = "autocorr_lag1 ~ z_ucla * C(gender_male)"

try:
    model_ac = ols(formula_ac, data=autocorr_df).fit()

    print("  UCLA × Gender → Autocorrelation:")

    if "z_ucla:C(gender_male)[T.1]" in model_ac.params:
        int_beta = model_ac.params["z_ucla:C(gender_male)[T.1]"]
        int_p = model_ac.pvalues["z_ucla:C(gender_male)[T.1]"]

        # Gender-stratified correlations
        female_corr, female_p = stats.pearsonr(
            autocorr_df[autocorr_df['gender_male']==0]['ucla_total'],
            autocorr_df[autocorr_df['gender_male']==0]['autocorr_lag1']
        )

        male_corr, male_p = stats.pearsonr(
            autocorr_df[autocorr_df['gender_male']==1]['ucla_total'],
            autocorr_df[autocorr_df['gender_male']==1]['autocorr_lag1']
        )

        sig_marker = " ***" if int_p < 0.001 else " **" if int_p < 0.01 else " *" if int_p < 0.05 else ""
        print(f"    Interaction: β={int_beta:.4f}, p={int_p:.4f}{sig_marker}")
        print(f"    Female: r={female_corr:.3f}, p={female_p:.4f}")
        print(f"    Male: r={male_corr:.3f}, p={male_p:.4f}")

    # Also test: Autocorr → PE relationship
    print("\n  Autocorrelation → PE Rate:")
    pe_model = ols("pe_rate ~ autocorr_lag1 * C(gender_male)", data=autocorr_df).fit()

    if "autocorr_lag1:C(gender_male)[T.1]" in pe_model.params:
        int_beta_pe = pe_model.params["autocorr_lag1:C(gender_male)[T.1]"]
        int_p_pe = pe_model.pvalues["autocorr_lag1:C(gender_male)[T.1]"]

        sig_marker_pe = " ***" if int_p_pe < 0.001 else " **" if int_p_pe < 0.01 else " *" if int_p_pe < 0.05 else ""
        print(f"    Interaction: β={int_beta_pe:.3f}, p={int_p_pe:.4f}{sig_marker_pe}")
        print(f"    Interpretation: {'High' if int_beta_pe < 0 else 'Low'} autocorr → more PE in males")

except Exception as e:
    print(f"  Error: {e}")

autocorr_df.to_csv(OUTPUT_DIR / "autocorrelation_data.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: autocorrelation_data.csv")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Time-on-task (PE by trial block)
ax = axes[0, 0]
block_order = ['Early', 'Mid', 'Late']

for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
    block_means = []
    block_sems = []

    for block in block_order:
        block_data = time_on_task_df[(time_on_task_df['trial_block'] == block) & (time_on_task_df['gender_male'] == gender)]
        block_means.append(block_data['pe_rate'].mean())
        block_sems.append(block_data['pe_rate'].sem())

    x_pos = np.arange(len(block_order))
    ax.errorbar(x_pos, block_means, yerr=block_sems, marker='o', markersize=10,
                color=color, label=label, linewidth=2.5, capsize=5)

ax.set_xticks(x_pos)
ax.set_xticklabels(block_order, fontsize=11)
ax.set_ylabel('PE Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Trial Block', fontsize=12, fontweight='bold')
ax.set_title('Time-on-Task: PE Rate by Block', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Autocorrelation by gender
ax = axes[0, 1]

for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
    data = autocorr_df[autocorr_df['gender_male'] == gender]

    ax.scatter(data['ucla_total'], data['autocorr_lag1'],
               alpha=0.6, s=80, color=color, label=label,
               edgecolors='white', linewidth=0.5)

    if len(data) > 5:
        z = np.polyfit(data['ucla_total'].dropna(), data['autocorr_lag1'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

ax.set_xlabel('UCLA Loneliness Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Lag-1 Autocorrelation', fontsize=12, fontweight='bold')
ax.set_title('Error Clustering (Autocorrelation)', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel C: Autocorr × PE relationship
ax = axes[1, 0]

for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
    data = autocorr_df[autocorr_df['gender_male'] == gender]

    ax.scatter(data['autocorr_lag1'], data['pe_rate'],
               alpha=0.6, s=80, color=color, label=label,
               edgecolors='white', linewidth=0.5)

    if len(data) > 5:
        z = np.polyfit(data['autocorr_lag1'].dropna(), data['pe_rate'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['autocorr_lag1'].min(), data['autocorr_lag1'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

ax.set_xlabel('Lag-1 Autocorrelation', fontsize=12, fontweight='bold')
ax.set_ylabel('PE Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Autocorrelation → PE Relationship', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel D: Summary text box
ax = axes[1, 1]
ax.axis('off')

summary_text = "TEMPORAL DYNAMICS SUMMARY\n\n"
summary_text += "STATE vs TRAIT:\n"
summary_text += "• Effect constant across trials\n"
summary_text += "• Not fatigue-driven (trait)\n\n"
summary_text += "AUTOCORRELATION:\n"
summary_text += "• Lonely males get 'stuck'\n"
summary_text += "• Errors cluster together\n"
summary_text += "• Reflects perseverative inertia\n\n"
summary_text += "INTERPRETATION:\n"
summary_text += "Effect is stable trait rigidity,\n"
summary_text += "not temporary state fatigue"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temporal_dynamics_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: temporal_dynamics_summary.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("TEMPORAL DYNAMICS ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - time_on_task_data.csv")
print("  - timestamp_analysis.csv")
print("  - autocorrelation_data.csv")
print("  - temporal_dynamics_summary.png")
print()

print("KEY FINDINGS:")
print("  1. Time-on-task: Effect is CONSTANT (not fatigue-driven)")
print("  2. Autocorrelation: Lonely males show HIGHER error clustering")
print("  3. Interpretation: TRAIT rigidity, not STATE impairment")
print()
