"""
WCST Trial-Level Dynamics Analysis
===================================

Deep dive into WCST performance dynamics to understand the gender × loneliness
interaction mechanism at the trial level.

Analyses:
1. Learning curves across categories
2. Post-shift error patterns
3. Error chain analysis
4. Feedback sensitivity
5. Rule-specific perseveration

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
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/wcst_trial_dynamics")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("WCST TRIAL-LEVEL DYNAMICS ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/6] Loading data...")

# Master dataset with gender
master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')

if 'participantId' in participants.columns:
    if 'participant_id' in participants.columns:
        participants.drop(columns=['participantId'], inplace=True)
    else:
        participants.rename(columns={'participantId': 'participant_id'}, inplace=True)

# Merge demographics
master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

# Handle Korean gender values
if 'gender' in master.columns:
    master['gender_male'] = 0
    master.loc[master['gender'] == '남성', 'gender_male'] = 1
    master.loc[master['gender'].str.lower() == 'male', 'gender_male'] = 1

# Load trial-level WCST data
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')

# Handle participant_id column
if 'participantId' in wcst_trials.columns and 'participant_id' in wcst_trials.columns:
    wcst_trials.drop(columns=['participantId'], inplace=True)
elif 'participantId' in wcst_trials.columns:
    wcst_trials.rename(columns={'participantId': 'participant_id'}, inplace=True)

print(f"  Master: {len(master)} participants")
print(f"  WCST trials: {len(wcst_trials)}")
print()

# ============================================================================
# ANALYSIS 1: LEARNING CURVES ACROSS CATEGORIES
# ============================================================================

print("[2/6] Analyzing learning curves across categories...")

learning_curves_list = []

for pid in master['participant_id'].unique():
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()
    pid_trials = pid_trials[pid_trials['timeout'] == False].copy()

    if len(pid_trials) < 20:
        continue

    # Get participant info
    pid_data = master[master['participant_id'] == pid].iloc[0]
    ucla = pid_data['ucla_total']
    gender_male = pid_data['gender_male']

    # Split into sextiles (6 categories approximate)
    n_trials = len(pid_trials)
    segment_size = n_trials // 6

    for seg in range(1, 7):
        start_idx = (seg - 1) * segment_size
        end_idx = seg * segment_size if seg < 6 else n_trials

        segment_trials = pid_trials.iloc[start_idx:end_idx]

        if len(segment_trials) > 0 and 'isPE' in segment_trials.columns:
            pe_rate = (segment_trials['isPE'].sum() / len(segment_trials)) * 100
        else:
            pe_rate = np.nan

        learning_curves_list.append({
            'participant_id': pid,
            'ucla_total': ucla,
            'gender_male': gender_male,
            'category_segment': seg,
            'pe_rate': pe_rate,
            'n_trials': len(segment_trials)
        })

learning_curves_df = pd.DataFrame(learning_curves_list)
learning_curves_df.to_csv(OUTPUT_DIR / "learning_curves.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Learning curves: {len(learning_curves_df)} data points")

# Test category × gender × UCLA interaction
if len(learning_curves_df) > 100:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    learning_curves_df['z_ucla'] = scaler.fit_transform(learning_curves_df[['ucla_total']])

    formula = "pe_rate ~ category_segment * z_ucla * C(gender_male)"
    try:
        model = ols(formula, data=learning_curves_df.dropna()).fit()
        interaction_term = "category_segment:z_ucla:C(gender_male)[T.1]"
        if interaction_term in model.params:
            print(f"    3-way interaction: β={model.params[interaction_term]:.3f}, p={model.pvalues[interaction_term]:.4f}")
    except:
        print("    Could not fit 3-way interaction model")

print()

# ============================================================================
# ANALYSIS 2: POST-SHIFT ERROR PATTERNS
# ============================================================================

print("[3/6] Analyzing post-shift error patterns...")

# This requires detecting rule changes - use trial-level rule information
post_shift_list = []

for pid in master['participant_id'].unique():
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()
    pid_trials = pid_trials[pid_trials['timeout'] == False].copy()

    if len(pid_trials) < 20 or 'ruleAtThatTime' not in pid_trials.columns:
        continue

    pid_trials = pid_trials.sort_values('trial_index').reset_index(drop=True)

    # Get participant info
    pid_data = master[master['participant_id'] == pid].iloc[0]
    ucla = pid_data['ucla_total']
    gender_male = pid_data['gender_male']

    # Detect rule changes
    rule_changes = []
    for i in range(1, len(pid_trials)):
        if pid_trials.loc[i, 'ruleAtThatTime'] != pid_trials.loc[i-1, 'ruleAtThatTime']:
            rule_changes.append(i)

    # Analyze first 5 trials after each rule change
    post_shift_pe_rates = []
    for change_idx in rule_changes:
        if change_idx + 5 <= len(pid_trials):
            post_shift_trials = pid_trials.iloc[change_idx:change_idx+5]
            if 'isPE' in post_shift_trials.columns:
                pe_rate = (post_shift_trials['isPE'].sum() / 5) * 100
                post_shift_pe_rates.append(pe_rate)

    if len(post_shift_pe_rates) > 0:
        post_shift_list.append({
            'participant_id': pid,
            'ucla_total': ucla,
            'gender_male': gender_male,
            'mean_post_shift_pe_rate': np.mean(post_shift_pe_rates),
            'n_shifts': len(post_shift_pe_rates)
        })

post_shift_df = pd.DataFrame(post_shift_list)
post_shift_df.to_csv(OUTPUT_DIR / "post_shift_errors.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Post-shift errors: {len(post_shift_df)} participants")

# Test gender moderation
if len(post_shift_df) > 30:
    scaler = StandardScaler()
    post_shift_df['z_ucla'] = scaler.fit_transform(post_shift_df[['ucla_total']])
    formula = "mean_post_shift_pe_rate ~ z_ucla * C(gender_male)"
    try:
        model = ols(formula, data=post_shift_df.dropna()).fit()
        interaction_term = "z_ucla:C(gender_male)[T.1]"
        if interaction_term in model.params:
            print(f"    Gender moderation: β={model.params[interaction_term]:.3f}, p={model.pvalues[interaction_term]:.4f}")
    except:
        print("    Could not fit model")

print()

# ============================================================================
# ANALYSIS 3: ERROR CHAIN ANALYSIS
# ============================================================================

print("[4/6] Analyzing error chains...")

error_chains_list = []

for pid in master['participant_id'].unique():
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()
    pid_trials = pid_trials[pid_trials['timeout'] == False].copy()

    if len(pid_trials) < 20 or 'isPE' not in pid_trials.columns:
        continue

    pid_trials = pid_trials.sort_values('trial_index').reset_index(drop=True)

    # Get participant info
    pid_data = master[master['participant_id'] == pid].iloc[0]
    ucla = pid_data['ucla_total']
    gender_male = pid_data['gender_male']

    # Find error chains
    in_chain = False
    chain_length = 0
    chain_lengths = []

    for i in range(len(pid_trials)):
        if pid_trials.loc[i, 'isPE']:
            if in_chain:
                chain_length += 1
            else:
                in_chain = True
                chain_length = 1
        else:
            if in_chain:
                chain_lengths.append(chain_length)
                in_chain = False
                chain_length = 0

    # Add last chain if still in one
    if in_chain:
        chain_lengths.append(chain_length)

    if len(chain_lengths) > 0:
        error_chains_list.append({
            'participant_id': pid,
            'ucla_total': ucla,
            'gender_male': gender_male,
            'mean_chain_length': np.mean(chain_lengths),
            'max_chain_length': np.max(chain_lengths),
            'n_chains': len(chain_lengths),
            'total_pe_count': np.sum(chain_lengths)
        })

error_chains_df = pd.DataFrame(error_chains_list)
error_chains_df.to_csv(OUTPUT_DIR / "error_chains.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Error chains: {len(error_chains_df)} participants")

# Test gender moderation on chain length
if len(error_chains_df) > 30:
    scaler = StandardScaler()
    error_chains_df['z_ucla'] = scaler.fit_transform(error_chains_df[['ucla_total']])

    for outcome in ['mean_chain_length', 'max_chain_length']:
        formula = f"{outcome} ~ z_ucla * C(gender_male)"
        try:
            model = ols(formula, data=error_chains_df.dropna()).fit()
            interaction_term = "z_ucla:C(gender_male)[T.1]"
            if interaction_term in model.params:
                print(f"    {outcome}: β={model.params[interaction_term]:.3f}, p={model.pvalues[interaction_term]:.4f}")
        except:
            continue

print()

# ============================================================================
# ANALYSIS 4: FEEDBACK SENSITIVITY
# ============================================================================

print("[5/6] Analyzing feedback sensitivity...")

feedback_sensitivity_list = []

for pid in master['participant_id'].unique():
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()
    pid_trials = pid_trials[pid_trials['timeout'] == False].copy()

    if len(pid_trials) < 30:
        continue

    pid_trials = pid_trials.sort_values('trial_index').reset_index(drop=True)

    # Get participant info
    pid_data = master[master['participant_id'] == pid].iloc[0]
    ucla = pid_data['ucla_total']
    gender_male = pid_data['gender_male']

    # Post-error slowing (RT)
    post_error_rts = []
    post_correct_rts = []

    # Post-error accuracy
    post_error_correct = []
    post_correct_correct = []

    for i in range(len(pid_trials) - 1):
        current_trial = pid_trials.iloc[i]
        next_trial = pid_trials.iloc[i + 1]

        # RT analysis
        if 'rt_ms' in pid_trials.columns and next_trial['rt_ms'] > 0:
            if not current_trial['correct']:
                post_error_rts.append(next_trial['rt_ms'])
            else:
                post_correct_rts.append(next_trial['rt_ms'])

        # Accuracy analysis
        if not current_trial['correct']:
            post_error_correct.append(int(next_trial['correct']))
        else:
            post_correct_correct.append(int(next_trial['correct']))

    # Calculate metrics
    pes = np.nan
    if len(post_error_rts) > 0 and len(post_correct_rts) > 0:
        pes = np.mean(post_error_rts) - np.mean(post_correct_rts)

    post_error_acc = np.mean(post_error_correct) * 100 if len(post_error_correct) > 0 else np.nan
    post_correct_acc = np.mean(post_correct_correct) * 100 if len(post_correct_correct) > 0 else np.nan

    feedback_sensitivity_list.append({
        'participant_id': pid,
        'ucla_total': ucla,
        'gender_male': gender_male,
        'post_error_slowing': pes,
        'post_error_accuracy': post_error_acc,
        'post_correct_accuracy': post_correct_acc,
        'n_post_error_trials': len(post_error_rts)
    })

feedback_df = pd.DataFrame(feedback_sensitivity_list)
feedback_df.to_csv(OUTPUT_DIR / "feedback_sensitivity.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Feedback sensitivity: {len(feedback_df)} participants")

# Test gender moderation
if len(feedback_df) > 30:
    scaler = StandardScaler()
    feedback_df['z_ucla'] = scaler.fit_transform(feedback_df[['ucla_total']])

    for outcome in ['post_error_slowing', 'post_error_accuracy']:
        if feedback_df[outcome].notna().sum() > 30:
            formula = f"{outcome} ~ z_ucla * C(gender_male)"
            try:
                model = ols(formula, data=feedback_df.dropna(subset=[outcome])).fit()
                interaction_term = "z_ucla:C(gender_male)[T.1]"
                if interaction_term in model.params:
                    print(f"    {outcome}: β={model.params[interaction_term]:.3f}, p={model.pvalues[interaction_term]:.4f}")
            except:
                continue

print()

# ============================================================================
# ANALYSIS 5: RULE-SPECIFIC PERSEVERATION
# ============================================================================

print("[6/6] Analyzing rule-specific perseveration...")

rule_specific_list = []

for pid in master['participant_id'].unique():
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()
    pid_trials = pid_trials[pid_trials['timeout'] == False].copy()

    if len(pid_trials) < 20 or 'ruleAtThatTime' not in pid_trials.columns or 'isPE' not in pid_trials.columns:
        continue

    # Get participant info
    pid_data = master[master['participant_id'] == pid].iloc[0]
    ucla = pid_data['ucla_total']
    gender_male = pid_data['gender_male']

    # Calculate PE rate for each rule
    for rule in ['colour', 'shape', 'number']:
        rule_trials = pid_trials[pid_trials['ruleAtThatTime'] == rule]
        if len(rule_trials) > 0:
            pe_rate = (rule_trials['isPE'].sum() / len(rule_trials)) * 100
        else:
            pe_rate = np.nan

        rule_specific_list.append({
            'participant_id': pid,
            'ucla_total': ucla,
            'gender_male': gender_male,
            'rule': rule,
            'pe_rate': pe_rate,
            'n_trials': len(rule_trials)
        })

rule_specific_df = pd.DataFrame(rule_specific_list)
rule_specific_df.to_csv(OUTPUT_DIR / "rule_specific_pe.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Rule-specific PE: {len(rule_specific_df)} data points")

# Test if effect varies by rule
if len(rule_specific_df) > 100:
    scaler = StandardScaler()
    rule_specific_df['z_ucla'] = scaler.fit_transform(rule_specific_df[['ucla_total']])

    # Test shape vs color (shape is more abstract)
    shape_df = rule_specific_df[rule_specific_df['rule'] == 'shape'].dropna()
    colour_df = rule_specific_df[rule_specific_df['rule'] == 'colour'].dropna()

    for rule_name, rule_df in [('shape', shape_df), ('colour', colour_df)]:
        if len(rule_df) > 30:
            formula = "pe_rate ~ z_ucla * C(gender_male)"
            try:
                model = ols(formula, data=rule_df).fit()
                interaction_term = "z_ucla:C(gender_male)[T.1]"
                if interaction_term in model.params:
                    print(f"    {rule_name} rule: β={model.params[interaction_term]:.3f}, p={model.pvalues[interaction_term]:.4f}")
            except:
                continue

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("WCST TRIAL-LEVEL ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - learning_curves.csv")
print("  - post_shift_errors.csv")
print("  - error_chains.csv")
print("  - feedback_sensitivity.csv")
print("  - rule_specific_pe.csv")
print()
print("Key findings:")
print(f"  - Learning curves analyzed across {len(learning_curves_df)//6} participants")
print(f"  - Post-shift errors: {len(post_shift_df)} participants")
print(f"  - Error chains: {len(error_chains_df)} participants")
print(f"  - Feedback sensitivity: {len(feedback_df)} participants")
print(f"  - Rule-specific PE: {len(rule_specific_df)//3} participants")
print()
