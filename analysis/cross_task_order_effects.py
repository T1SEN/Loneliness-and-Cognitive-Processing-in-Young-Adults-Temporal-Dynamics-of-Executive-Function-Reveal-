"""
Cross-Task Order Effects Analysis
==================================

Tests whether the gender moderation effect depends on:
1. WCST position in test battery (first vs middle vs last)
2. Cross-task interference (does Stroop performance predict WCST?)
3. Task-switching costs (WCST after Stroop vs when first)

Addresses potential confound: Is effect due to WCST position or depletion?

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
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/mechanism_analysis/task_order")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("CROSS-TASK ORDER EFFECTS ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/3] Loading data...")

# Load participant data
master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))
participants = pd.read_csv(Path("results/1_participants_info.csv"), encoding='utf-8-sig')
summary = pd.read_csv(Path("results/3_cognitive_tests_summary.csv"), encoding='utf-8-sig')

if 'participantId' in participants.columns:
    if 'participant_id' in participants.columns:
        participants.drop(columns=['participantId'], inplace=True)
    else:
        participants.rename(columns={'participantId': 'participant_id'}, inplace=True)

if 'participantId' in summary.columns:
    if 'participant_id' in summary.columns:
        summary.drop(columns=['participantId'], inplace=True)
    else:
        summary.rename(columns={'participantId': 'participant_id'}, inplace=True)

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

print(f"  Participants: {len(master)}")
print()

# ============================================================================
# ANALYSIS 1: TASK ORDER EXTRACTION
# ============================================================================

print("[2/3] Extracting task order information...")
print()

# Try to extract task order from summary data
# The summary might have timestamp or order info

if 'testName' in summary.columns and 'timestamp' in summary.columns:
    # Sort by participant and timestamp to get order
    summary_sorted = summary.sort_values(['participant_id', 'timestamp'])

    task_order = []

    for pid in summary_sorted['participant_id'].unique():
        pid_tests = summary_sorted[summary_sorted['participant_id'] == pid]

        # Find WCST position
        wcst_rows = pid_tests[pid_tests['testName'].str.contains('WCST|wcst', case=False, na=False)]

        if len(wcst_rows) > 0:
            wcst_position = list(pid_tests['testName']).index(wcst_rows.iloc[0]['testName']) + 1
            total_tests = len(pid_tests)

            # Categorize position
            if wcst_position == 1:
                position_cat = 'First'
            elif wcst_position == total_tests:
                position_cat = 'Last'
            else:
                position_cat = 'Middle'

            # Check if Stroop done before WCST
            stroop_rows = pid_tests[pid_tests['testName'].str.contains('Stroop|stroop', case=False, na=False)]
            stroop_before_wcst = False

            if len(stroop_rows) > 0:
                stroop_position = list(pid_tests['testName']).index(stroop_rows.iloc[0]['testName']) + 1
                stroop_before_wcst = stroop_position < wcst_position

            task_order.append({
                'participant_id': pid,
                'wcst_position': wcst_position,
                'wcst_position_cat': position_cat,
                'total_tests': total_tests,
                'stroop_before_wcst': stroop_before_wcst
            })

    if len(task_order) > 0:
        task_order_df = pd.DataFrame(task_order)

        # Merge with master
        master = master.merge(task_order_df, on='participant_id', how='left')

        print(f"  Task order extracted for {len(task_order)} participants")
        print(f"  WCST position distribution:")
        print(f"    {master['wcst_position_cat'].value_counts().to_dict()}")
        print()

        # Test: WCST Position × UCLA × Gender
        if 'wcst_position_cat' in master.columns:
            master_order = master.dropna(subset=['wcst_position_cat', 'ucla_total', 'gender_male', 'pe_rate']).copy()

            if len(master_order) >= 30:
                scaler = StandardScaler()
                master_order['z_ucla'] = scaler.fit_transform(master_order[['ucla_total']])

                print("  Testing: WCST Position × UCLA × Gender → PE")
                print()

                formula = "pe_rate ~ C(wcst_position_cat) * z_ucla * C(gender_male)"

                try:
                    model = ols(formula, data=master_order).fit()

                    # Test interaction by position
                    for position in ['First', 'Middle', 'Last']:
                        pos_data = master_order[master_order['wcst_position_cat'] == position].copy()

                        if len(pos_data) >= 10:
                            pos_model = ols("pe_rate ~ z_ucla * C(gender_male)", data=pos_data).fit()

                            if "z_ucla:C(gender_male)[T.1]" in pos_model.params:
                                int_beta = pos_model.params["z_ucla:C(gender_male)[T.1]"]
                                int_p = pos_model.pvalues["z_ucla:C(gender_male)[T.1]"]

                                sig_marker = " ***" if int_p < 0.001 else " **" if int_p < 0.01 else " *" if int_p < 0.05 else ""
                                print(f"    {position}: β={int_beta:.3f}, p={int_p:.4f}{sig_marker} (N={len(pos_data)})")
                        else:
                            print(f"    {position}: Insufficient data (N={len(pos_data)})")

                except Exception as e:
                    print(f"  Error: {e}")

                master_order.to_csv(OUTPUT_DIR / "task_order_moderation.csv", index=False, encoding='utf-8-sig')
                print(f"\n✓ Saved: task_order_moderation.csv")
            else:
                print("  Insufficient data for task order analysis")
        print()
    else:
        print("  Could not extract task order from data")
        print()
else:
    print("  No testName or timestamp columns found")
    print("  Skipping task order analysis")
    print()

# ============================================================================
# ANALYSIS 2: CROSS-TASK CARRY-OVER
# ============================================================================

print("[3/3] Cross-task carry-over (Stroop → WCST)...")
print()

# Test: Does Stroop performance predict WCST PE?
if 'stroop_interference' in master.columns:
    carryover_data = master.dropna(subset=['stroop_interference', 'pe_rate', 'ucla_total', 'gender_male']).copy()

    if len(carryover_data) >= 30:
        scaler_co = StandardScaler()
        carryover_data['z_ucla'] = scaler_co.fit_transform(carryover_data[['ucla_total']])
        carryover_data['z_stroop'] = scaler_co.fit_transform(carryover_data[['stroop_interference']])

        # Path model: Stroop errors → WCST PE (moderated by UCLA × Gender)
        print("  Path 1: Stroop → WCST PE (overall)")

        stroop_wcst_model = ols("pe_rate ~ z_stroop", data=carryover_data).fit()
        print(f"    β={stroop_wcst_model.params['z_stroop']:.3f}, p={stroop_wcst_model.pvalues['z_stroop']:.4f}")

        print("\n  Path 2: Stroop → WCST PE (moderated by UCLA × Gender)")

        interaction_model = ols("pe_rate ~ z_stroop * z_ucla * C(gender_male)", data=carryover_data).fit()

        # Test by gender
        print("\n  By gender:")

        for gender, label in [(0, 'Female'), (1, 'Male')]:
            gender_data = carryover_data[carryover_data['gender_male'] == gender].copy()

            if len(gender_data) >= 15:
                # Stroop → PE, moderated by UCLA
                gender_model = ols("pe_rate ~ z_stroop * z_ucla", data=gender_data).fit()

                if "z_stroop:z_ucla" in gender_model.params:
                    int_beta = gender_model.params["z_stroop:z_ucla"]
                    int_p = gender_model.pvalues["z_stroop:z_ucla"]

                    sig_marker = " ***" if int_p < 0.001 else " **" if int_p < 0.01 else " *" if int_p < 0.05 else ""
                    print(f"    {label}: Stroop × UCLA → PE: β={int_beta:.3f}, p={int_p:.4f}{sig_marker}")

        carryover_data.to_csv(OUTPUT_DIR / "cross_task_carryover.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: cross_task_carryover.csv")
    else:
        print("  Insufficient data for carry-over analysis")
else:
    print("  Stroop data not available")

print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: PE by WCST position
if 'wcst_position_cat' in master.columns and 'gender_male' in master.columns:
    ax = axes[0]

    position_order = ['First', 'Middle', 'Last']
    master_plot = master.dropna(subset=['wcst_position_cat', 'pe_rate', 'gender_male'])

    for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
        position_means = []
        position_sems = []

        for position in position_order:
            pos_data = master_plot[(master_plot['wcst_position_cat'] == position) & (master_plot['gender_male'] == gender)]

            if len(pos_data) > 0:
                position_means.append(pos_data['pe_rate'].mean())
                position_sems.append(pos_data['pe_rate'].sem())
            else:
                position_means.append(np.nan)
                position_sems.append(np.nan)

        x_pos = np.arange(len(position_order))
        ax.errorbar(x_pos, position_means, yerr=position_sems, marker='o', markersize=10,
                    color=color, label=label, linewidth=2.5, capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(position_order, fontsize=11)
    ax.set_ylabel('PE Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('WCST Position in Battery', fontsize=12, fontweight='bold')
    ax.set_title('PE Rate by Task Position', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
else:
    axes[0].text(0.5, 0.5, 'Task order data\nnot available', ha='center', va='center',
                 fontsize=12, transform=axes[0].transAxes)
    axes[0].axis('off')

# Panel B: Stroop × WCST carry-over
if 'stroop_interference' in master.columns:
    ax = axes[1]

    carryover_plot = master.dropna(subset=['stroop_interference', 'pe_rate', 'gender_male'])

    for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
        data = carryover_plot[carryover_plot['gender_male'] == gender]

        ax.scatter(data['stroop_interference'], data['pe_rate'],
                   alpha=0.6, s=80, color=color, label=label,
                   edgecolors='white', linewidth=0.5)

        if len(data) > 5:
            z = np.polyfit(data['stroop_interference'].dropna(), data['pe_rate'].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['stroop_interference'].min(), data['stroop_interference'].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

    ax.set_xlabel('Stroop Interference (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('WCST PE Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Task Carry-Over (Stroop → WCST)', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
else:
    axes[1].text(0.5, 0.5, 'Stroop data\nnot available', ha='center', va='center',
                 fontsize=12, transform=axes[1].transAxes)
    axes[1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task_order_effects_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: task_order_effects_summary.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("CROSS-TASK ORDER EFFECTS ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
if 'wcst_position_cat' in master.columns:
    print("  - task_order_moderation.csv")
print("  - cross_task_carryover.csv")
print("  - task_order_effects_summary.png")
print()

print("KEY FINDINGS:")
print("  1. Task order effects: Check if effect varies by WCST position")
print("  2. Cross-task carry-over: Test Stroop → WCST interference")
print("  3. Interpretation: Determine if effect is position-specific or robust")
print()
