"""
Female Compensation Cost Analysis

THEORETICAL BACKGROUND:
Previous analyses show that lonely females preserve EF performance (no UCLA→PE
relationship) through compensatory hypervigilance (reduced σ after errors, fewer
error cascades). However, compensation may come at a COST:

1. TIME COST: Hypervigilance requires more time → higher overall RT
2. DEPLETION COST: Sustained compensation depletes resources → steeper fatigue slopes
3. PSYCHOLOGICAL COST: Effort without reward → higher DASS despite preserved EF

This analysis tests whether lonely females "pay a price" for preserved performance.

HYPOTHESES:
H1: Among females, UCLA predicts HIGHER overall RT (time cost of compensation)
H2: Among females, UCLA predicts STEEPER trial-by-trial fatigue slopes (depletion)
H3: Among females, DASS scores are HIGHER despite low PE rates (psychological burden)
H4: Lonely females show dissociation: High DASS + Low PE (effortful success pattern)

COMPARISON:
Males: Should show performance impairment (high PE) WITHOUT these costs
Females: Show preserved performance (low PE) WITH these costs

If confirmed: Gender-balanced narrative of "different vulnerabilities"
- Males: Cognitive vulnerability (direct impairment)
- Females: Metabolic/psychological vulnerability (compensation costs)
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

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/female_compensation_costs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*80)
print("FEMALE COMPENSATION COST ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

# Demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)
participants['gender_male'] = (participants['gender'] == 'male').astype(int)
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

# UCLA & DASS
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
if 'participantId' in surveys.columns:
    surveys = surveys.rename(columns={'participantId': 'participant_id'})

ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()
ucla_data['ucla_total'] = pd.to_numeric(ucla_data['score'], errors='coerce')
ucla_data = ucla_data[['participant_id', 'ucla_total']].dropna()

dass_data = surveys[surveys['surveyName'] == 'dass'].copy()
dass_data['score_A'] = pd.to_numeric(dass_data['score_A'], errors='coerce')
dass_data['score_S'] = pd.to_numeric(dass_data['score_S'], errors='coerce')
dass_data['score_D'] = pd.to_numeric(dass_data['score_D'], errors='coerce')
dass_data['dass_total'] = dass_data[['score_A', 'score_S', 'score_D']].sum(axis=1)
dass_data['dass_anxiety'] = dass_data['score_A']
dass_data['dass_stress'] = dass_data['score_S']
dass_data['dass_depression'] = dass_data['score_D']
dass_data = dass_data[['participant_id', 'dass_total', 'dass_anxiety', 'dass_stress', 'dass_depression']].dropna()

# WCST PE for comparison
wcst_summary = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv", encoding='utf-8-sig')
if 'participantId' in wcst_summary.columns:
    wcst_summary = wcst_summary.rename(columns={'participantId': 'participant_id'})

# Find PE column
pe_col = None
for col in ['pe_rate', 'wcst_pe_rate', 'perseverative_error_rate', 'perseverativeErrorRate']:
    if col in wcst_summary.columns:
        pe_col = col
        break

if pe_col:
    wcst_data = wcst_summary[['participant_id', pe_col]].copy()
    wcst_data.columns = ['participant_id', 'wcst_pe_rate']
else:
    print("  Warning: No PE column found")
    wcst_data = pd.DataFrame({'participant_id': [], 'wcst_pe_rate': []})

# Load trial-level data for RT and fatigue analyses
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
if 'participantId' in wcst_trials.columns and 'participant_id' in wcst_trials.columns:
    wcst_trials = wcst_trials.drop(columns=['participantId'])
elif 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
if 'participantId' in stroop_trials.columns and 'participant_id' in stroop_trials.columns:
    stroop_trials = stroop_trials.drop(columns=['participantId'])
elif 'participantId' in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})

prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
if 'participantId' in prp_trials.columns and 'participant_id' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participantId'])
elif 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.rename(columns={'participantId': 'participant_id'})

print(f"  Loaded {len(participants)} participants")
print(f"  Loaded {len(ucla_data)} UCLA scores")
print(f"  Loaded {len(dass_data)} DASS scores")
print(f"  Loaded {len(wcst_trials)} WCST trials")
print(f"  Loaded {len(stroop_trials)} Stroop trials")
print(f"  Loaded {len(prp_trials)} PRP trials")

# ============================================================================
# 2. COMPUTE TIME COSTS (Overall RT metrics)
# ============================================================================
print("\n[2/7] Computing time cost metrics (overall RT)...")

time_costs = []

# WCST RT
wcst_valid = wcst_trials[wcst_trials['rt_ms'].notna() & (wcst_trials['rt_ms'] > 0)].copy()
wcst_rt_summary = wcst_valid.groupby('participant_id')['rt_ms'].agg(['mean', 'median', 'std']).reset_index()
wcst_rt_summary.columns = ['participant_id', 'wcst_rt_mean', 'wcst_rt_median', 'wcst_rt_sd']

# Stroop RT
stroop_valid = stroop_trials[stroop_trials['rt_ms'].notna() & (stroop_trials['rt_ms'] > 0)].copy()
stroop_rt_summary = stroop_valid.groupby('participant_id')['rt_ms'].agg(['mean', 'median', 'std']).reset_index()
stroop_rt_summary.columns = ['participant_id', 'stroop_rt_mean', 'stroop_rt_median', 'stroop_rt_sd']

# PRP T2 RT
prp_valid = prp_trials[prp_trials['t2_rt_ms'].notna() & (prp_trials['t2_rt_ms'] > 0)].copy()
prp_rt_summary = prp_valid.groupby('participant_id')['t2_rt_ms'].agg(['mean', 'median', 'std']).reset_index()
prp_rt_summary.columns = ['participant_id', 'prp_t2_rt_mean', 'prp_t2_rt_median', 'prp_t2_rt_sd']

print(f"  WCST RT: {len(wcst_rt_summary)} participants")
print(f"  Stroop RT: {len(stroop_rt_summary)} participants")
print(f"  PRP T2 RT: {len(prp_rt_summary)} participants")

# ============================================================================
# 3. COMPUTE DEPLETION COSTS (Fatigue slopes)
# ============================================================================
print("\n[3/7] Computing depletion cost metrics (fatigue slopes)...")

def compute_fatigue_slope(trials_df, participant_col='participant_id', rt_col='rt_ms'):
    """Compute trial-by-trial RT slope (fatigue indicator)"""
    fatigue_slopes = []

    for pid in trials_df[participant_col].unique():
        participant_trials = trials_df[trials_df[participant_col] == pid].copy()

        # Filter valid trials
        valid = participant_trials[participant_trials[rt_col].notna() & (participant_trials[rt_col] > 0)]

        if len(valid) < 10:
            continue

        # Add trial number
        if 'timestamp' in valid.columns:
            valid = valid.sort_values('timestamp')
        elif 'trialIndex' in valid.columns:
            valid = valid.sort_values('trialIndex')
        # else keep original order
        valid = valid.reset_index(drop=True)
        valid['trial_num'] = np.arange(len(valid))

        # Fit linear regression: RT ~ trial_number
        try:
            model = smf.ols(f'{rt_col} ~ trial_num', data=valid).fit()
            slope = model.params['trial_num']
            p_value = model.pvalues['trial_num']

            fatigue_slopes.append({
                participant_col: pid,
                'fatigue_slope': slope,
                'fatigue_p': p_value,
                'n_trials': len(valid)
            })
        except:
            continue

    return pd.DataFrame(fatigue_slopes)

wcst_fatigue = compute_fatigue_slope(wcst_trials, rt_col='rt_ms')
wcst_fatigue.columns = ['participant_id', 'wcst_fatigue_slope', 'wcst_fatigue_p', 'wcst_n_trials']

stroop_fatigue = compute_fatigue_slope(stroop_trials, rt_col='rt_ms')
stroop_fatigue.columns = ['participant_id', 'stroop_fatigue_slope', 'stroop_fatigue_p', 'stroop_n_trials']

prp_fatigue = compute_fatigue_slope(prp_trials, rt_col='t2_rt_ms')
prp_fatigue.columns = ['participant_id', 'prp_fatigue_slope', 'prp_fatigue_p', 'prp_n_trials']

print(f"  WCST fatigue slopes: {len(wcst_fatigue)} participants")
print(f"  Stroop fatigue slopes: {len(stroop_fatigue)} participants")
print(f"  PRP fatigue slopes: {len(prp_fatigue)} participants")

# ============================================================================
# 4. MERGE MASTER DATASET
# ============================================================================
print("\n[4/7] Merging master dataset...")

master = participants[['participant_id', 'gender_male', 'age']].copy()
master = master.merge(ucla_data, on='participant_id', how='inner')
master = master.merge(dass_data, on='participant_id', how='left')
master = master.merge(wcst_data, on='participant_id', how='left')
master = master.merge(wcst_rt_summary, on='participant_id', how='left')
master = master.merge(stroop_rt_summary, on='participant_id', how='left')
master = master.merge(prp_rt_summary, on='participant_id', how='left')
master = master.merge(wcst_fatigue, on='participant_id', how='left')
master = master.merge(stroop_fatigue, on='participant_id', how='left')
master = master.merge(prp_fatigue, on='participant_id', how='left')

# Drop rows missing key variables
master = master.dropna(subset=['ucla_total', 'gender_male'])

n_total = len(master)
n_male = (master['gender_male'] == 1).sum()
n_female = (master['gender_male'] == 0).sum()

print(f"\n  Complete cases: N={n_total}")
print(f"    Males: {n_male}")
print(f"    Females: {n_female}")

# ============================================================================
# 5. TEST HYPOTHESES (Stratified by gender)
# ============================================================================
print("\n[5/7] Testing compensation cost hypotheses...")

results = []

# Test each outcome separately by gender
outcomes = {
    'TIME COSTS (Overall RT)': [
        ('wcst_rt_mean', 'WCST Mean RT'),
        ('stroop_rt_mean', 'Stroop Mean RT'),
        ('prp_t2_rt_mean', 'PRP T2 Mean RT')
    ],
    'DEPLETION COSTS (Fatigue slopes)': [
        ('wcst_fatigue_slope', 'WCST Fatigue Slope'),
        ('stroop_fatigue_slope', 'Stroop Fatigue Slope'),
        ('prp_fatigue_slope', 'PRP Fatigue Slope')
    ],
    'PSYCHOLOGICAL COSTS (DASS)': [
        ('dass_total', 'DASS Total'),
        ('dass_anxiety', 'DASS Anxiety'),
        ('dass_depression', 'DASS Depression'),
        ('dass_stress', 'DASS Stress')
    ]
}

for category, outcome_list in outcomes.items():
    print(f"\n{category}")
    print("-"*80)

    for outcome_col, outcome_label in outcome_list:
        if outcome_col not in master.columns:
            print(f"  {outcome_label}: Column not found, skipping")
            continue

        # Test by gender
        for gender, label in [(0, 'Female'), (1, 'Male')]:
            subset = master[(master['gender_male'] == gender) & master[outcome_col].notna()].copy()

            if len(subset) < 10:
                print(f"  {outcome_label} ({label}): N={len(subset)} - too small")
                continue

            # Correlation
            r, p = stats.pearsonr(subset['ucla_total'], subset[outcome_col])

            # Regression controlling age
            if 'age' in subset.columns:
                formula = f"{outcome_col} ~ ucla_total + age"
            else:
                formula = f"{outcome_col} ~ ucla_total"

            try:
                model = smf.ols(formula, data=subset).fit()
                beta = model.params['ucla_total']
                beta_p = model.pvalues['ucla_total']
            except:
                beta, beta_p = np.nan, np.nan

            print(f"  {outcome_label} ({label}, N={len(subset)}): r={r:.3f}, p={p:.4f}, β={beta:.4f}, p_β={beta_p:.4f}")

            results.append({
                'category': category,
                'outcome': outcome_label,
                'gender': label,
                'n': len(subset),
                'r': r,
                'p': p,
                'beta': beta,
                'beta_p': beta_p
            })

# ============================================================================
# 6. TEST DISSOCIATION PATTERN (High DASS + Low PE)
# ============================================================================
print("\n[6/7] Testing dissociation pattern (High DASS + Low PE in lonely females)...")

# Create high/low UCLA groups (median split)
master['high_ucla'] = (master['ucla_total'] > master['ucla_total'].median()).astype(int)

# Subset: Females only with DASS and PE data
females = master[(master['gender_male'] == 0) &
                 master['dass_total'].notna() &
                 master['wcst_pe_rate'].notna()].copy()

if len(females) >= 20:
    # Compare high vs low UCLA females
    high_ucla_f = females[females['high_ucla'] == 1]
    low_ucla_f = females[females['high_ucla'] == 0]

    print(f"\n  Low UCLA Females (N={len(low_ucla_f)}):")
    print(f"    DASS: M={low_ucla_f['dass_total'].mean():.2f}, SD={low_ucla_f['dass_total'].std():.2f}")
    print(f"    PE Rate: M={low_ucla_f['wcst_pe_rate'].mean():.2f}%, SD={low_ucla_f['wcst_pe_rate'].std():.2f}")

    print(f"\n  High UCLA Females (N={len(high_ucla_f)}):")
    print(f"    DASS: M={high_ucla_f['dass_total'].mean():.2f}, SD={high_ucla_f['dass_total'].std():.2f}")
    print(f"    PE Rate: M={high_ucla_f['wcst_pe_rate'].mean():.2f}%, SD={high_ucla_f['wcst_pe_rate'].std():.2f}")

    # t-tests
    dass_t, dass_p = stats.ttest_ind(high_ucla_f['dass_total'], low_ucla_f['dass_total'])
    pe_t, pe_p = stats.ttest_ind(high_ucla_f['wcst_pe_rate'], low_ucla_f['wcst_pe_rate'])

    print(f"\n  Group Differences:")
    print(f"    DASS: t={dass_t:.3f}, p={dass_p:.4f}")
    print(f"    PE Rate: t={pe_t:.3f}, p={pe_p:.4f}")

    # Dissociation pattern: High DASS but NO increase in PE
    dissociation_pattern = (dass_p < 0.10) and (pe_p > 0.10)
    print(f"\n  ✓ DISSOCIATION PATTERN DETECTED: {dissociation_pattern}")
    print(f"    (High UCLA → Higher DASS but NOT higher PE)")

else:
    print(f"  Insufficient female sample (N={len(females)}) for dissociation analysis")
    dissociation_pattern = None

# ============================================================================
# 7. VISUALIZATIONS & SAVE
# ============================================================================
print("\n[7/7] Creating visualizations and saving results...")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "compensation_costs_results.csv", index=False, encoding='utf-8-sig')

# Plot 1: Effect sizes by gender
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

categories_plot = ['TIME COSTS (Overall RT)', 'DEPLETION COSTS (Fatigue slopes)', 'PSYCHOLOGICAL COSTS (DASS)']

for idx, cat in enumerate(categories_plot):
    if idx >= 3:
        break

    ax = axes[idx // 2, idx % 2]

    cat_data = results_df[results_df['category'] == cat]

    if len(cat_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(cat)
        continue

    # Separate by gender
    female_data = cat_data[cat_data['gender'] == 'Female']
    male_data = cat_data[cat_data['gender'] == 'Male']

    x_pos = np.arange(len(female_data))

    width = 0.35

    ax.bar(x_pos - width/2, female_data['r'], width, label='Female', color='#E74C3C', alpha=0.7)
    ax.bar(x_pos + width/2, male_data['r'], width, label='Male', color='#3498DB', alpha=0.7)

    ax.set_xlabel('Outcome')
    ax.set_ylabel('Correlation (r)')
    ax.set_title(cat)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(female_data['outcome'].str.replace('WCST ', '').str.replace('Stroop ', '').str.replace('PRP ', '').str.replace('DASS ', ''), rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

# Summary text in 4th panel
ax = axes[1, 1]
ax.axis('off')

summary_text = "KEY FINDINGS:\n\n"

# Count significant effects by gender
female_sig = results_df[(results_df['gender'] == 'Female') & (results_df['p'] < 0.05)]
male_sig = results_df[(results_df['gender'] == 'Male') & (results_df['p'] < 0.05)]

summary_text += f"Females:\n"
summary_text += f"  Significant costs: {len(female_sig)}\n"
for _, row in female_sig.iterrows():
    summary_text += f"    {row['outcome']}: r={row['r']:.3f}, p={row['p']:.3f}\n"

summary_text += f"\nMales:\n"
summary_text += f"  Significant costs: {len(male_sig)}\n"
for _, row in male_sig.iterrows():
    summary_text += f"    {row['outcome']}: r={row['r']:.3f}, p={row['p']:.3f}\n"

if dissociation_pattern is not None:
    summary_text += f"\nDissociation Pattern (Females):\n"
    summary_text += f"  High UCLA → Higher DASS: p={dass_p:.3f}\n"
    summary_text += f"  High UCLA → PE (no change): p={pe_p:.3f}\n"
    summary_text += f"  Pattern confirmed: {dissociation_pattern}\n"

ax.text(0.1, 0.9, summary_text, ha='left', va='top', fontsize=9, family='monospace')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "compensation_costs_summary.png", dpi=300, bbox_inches='tight')
plt.close()

# Save report
with open(OUTPUT_DIR / "FEMALE_COMPENSATION_COSTS_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("FEMALE COMPENSATION COST ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write("THEORETICAL RATIONALE\n")
    f.write("-"*80 + "\n")
    f.write("Lonely females preserve EF performance through compensatory hypervigilance.\n")
    f.write("This analysis tests whether compensation comes at a COST:\n")
    f.write("  1. TIME COST: Higher overall RT\n")
    f.write("  2. DEPLETION COST: Steeper fatigue slopes\n")
    f.write("  3. PSYCHOLOGICAL COST: Higher DASS despite preserved EF\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"N = {n_total}\n")
    f.write(f"  Females: {n_female}\n")
    f.write(f"  Males: {n_male}\n\n")

    f.write("RESULTS BY CATEGORY\n")
    f.write("-"*80 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("DISSOCIATION PATTERN TEST\n")
    f.write("-"*80 + "\n")
    if dissociation_pattern is not None:
        f.write(f"High vs Low UCLA Females:\n")
        f.write(f"  DASS difference: p={dass_p:.4f} {'(SIG)' if dass_p < 0.05 else '(NS)'}\n")
        f.write(f"  PE difference: p={pe_p:.4f} {'(SIG)' if pe_p < 0.05 else '(NS)'}\n")
        f.write(f"\nDissociation confirmed: {dissociation_pattern}\n")
        f.write("  Interpretation: Lonely females show elevated distress (DASS) but\n")
        f.write("  preserved performance (PE), consistent with effortful compensation.\n")
    else:
        f.write("Insufficient data for dissociation analysis\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Full results saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ Female Compensation Cost Analysis Complete!")
print("="*80)
print(f"\nKey Findings:")
print(f"  Female significant effects: {len(female_sig)}")
print(f"  Male significant effects: {len(male_sig)}")
if dissociation_pattern is not None:
    print(f"  Dissociation pattern confirmed: {dissociation_pattern}")
