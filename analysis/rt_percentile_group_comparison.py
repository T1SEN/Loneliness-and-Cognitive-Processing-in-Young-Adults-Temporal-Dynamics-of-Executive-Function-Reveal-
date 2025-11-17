"""
RT Percentile Group Comparison Analysis
========================================
Tests UCLA × Gender effects on participant-level RT percentiles (10th, 50th, 90th).

⚠️ IMPORTANT: This is NOT conditional quantile regression (QuantReg).

Method:
1. Compute each participant's RT percentiles from trial data (q = 0.10, 0.25, 0.50, 0.75, 0.90)
2. Use OLS to test group differences: percentile ~ UCLA × Gender + DASS + Age
3. Compare effects across percentiles

This tests: "Do high-lonely males have higher 90th percentile RTs (more lapses)?"
NOT: "At the 90th quantile of the conditional RT distribution, is the UCLA effect stronger?"

The distinction:
- This approach: Unconditional percentiles → group comparison (easier to interpret)
- True quantile regression: Conditional percentiles → effect heterogeneity across distribution

Hypothesis: If effects are driven by attentional lapses (tau),
we should see STRONGER effects at q=0.90 (slow tail) than q=0.50 (median).

For true quantile regression, see: true_quantile_regression_analysis.py

CRITICAL: All models control for DASS-21 subscales.

Author: Research Team
Date: 2025-01-16
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Constants
MIN_N_REGRESSION = 30  # Minimum sample size for regression models

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/rt_percentile_group_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("RT PERCENTILE GROUP COMPARISON ANALYSIS")
print("=" * 80)
print("\nPurpose: Test UCLA × Gender effects on participant RT percentiles")
print("Method: Compute percentiles → OLS group comparison")
print("Note: This is NOT conditional quantile regression\n")

# Load trial-level data
print("Loading trial-level data...")

# Try to load PRP trials (primary analysis)
try:
    prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig', index_col=None)
    prp_trials.columns = prp_trials.columns.str.lower()
    # Handle potential duplicate participant_id columns (participantid + participant_id)
    if 'participantid' in prp_trials.columns and 'participant_id' in prp_trials.columns:
        prp_trials.drop(columns=['participantid'], inplace=True)
    elif 'participantid' in prp_trials.columns:
        prp_trials.rename(columns={'participantid': 'participant_id'}, inplace=True)
    if prp_trials.index.name == 'participant_id':
        prp_trials = prp_trials.reset_index()
    print(f"  Loaded PRP trials: {len(prp_trials)} trials")
    has_prp = True
except FileNotFoundError:
    print("  PRP trials not found")
    has_prp = False

# Try to load WCST trials
try:
    wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig', index_col=None)
    wcst_trials.columns = wcst_trials.columns.str.lower()
    # Handle potential duplicate participant_id columns (participantid + participant_id)
    if 'participantid' in wcst_trials.columns and 'participant_id' in wcst_trials.columns:
        wcst_trials.drop(columns=['participantid'], inplace=True)
    elif 'participantid' in wcst_trials.columns:
        wcst_trials.rename(columns={'participantid': 'participant_id'}, inplace=True)
    # Ensure clean index (no participant_id as index)
    if wcst_trials.index.name == 'participant_id':
        wcst_trials = wcst_trials.reset_index()
    print(f"  Loaded WCST trials: {len(wcst_trials)} trials")
    has_wcst = True
except FileNotFoundError:
    print("  WCST trials not found")
    has_wcst = False

# Try to load Stroop trials
try:
    stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig', index_col=None)
    stroop_trials.columns = stroop_trials.columns.str.lower()
    # Handle potential duplicate participant_id columns (participantid + participant_id)
    if 'participantid' in stroop_trials.columns and 'participant_id' in stroop_trials.columns:
        stroop_trials.drop(columns=['participantid'], inplace=True)
    elif 'participantid' in stroop_trials.columns:
        stroop_trials.rename(columns={'participantid': 'participant_id'}, inplace=True)
    if stroop_trials.index.name == 'participant_id':
        stroop_trials = stroop_trials.reset_index()
    print(f"  Loaded Stroop trials: {len(stroop_trials)} trials")
    has_stroop = True
except FileNotFoundError:
    print("  Stroop trials not found")
    has_stroop = False

if not (has_prp or has_wcst or has_stroop):
    print("ERROR: No trial-level data found.")
    sys.exit(1)

# Load participant-level covariates
print("\nLoading participant covariates...")
try:
    participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
    participants.columns = participants.columns.str.lower()
    if 'participantid' in participants.columns:
        participants.rename(columns={'participantid': 'participant_id'}, inplace=True)
except FileNotFoundError:
    print("ERROR: 1_participants_info.csv not found")
    sys.exit(1)

try:
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
    surveys.columns = surveys.columns.str.lower()
    if 'participantid' in surveys.columns:
        surveys.rename(columns={'participantid': 'participant_id'}, inplace=True)

    # Extract UCLA and DASS
    # Assuming there's a master dataset with these already computed
    # If not, we'll need to parse surveys
    master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
    master.columns = master.columns.str.lower()
    if 'participantid' in master.columns:
        master.rename(columns={'participantid': 'participant_id'}, inplace=True)

    covariates = master[['participant_id', 'ucla_total', 'gender', 'age',
                        'dass_depression', 'dass_anxiety', 'dass_stress']].copy()
except FileNotFoundError:
    print("WARNING: Could not load master dataset. Using participants only.")
    covariates = participants[['participant_id', 'gender', 'age']].copy()
    # Won't be able to control for UCLA/DASS

# Ensure gender coding
gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female', 'M': 'male', 'F': 'female'}
if 'gender' in covariates.columns:
    covariates['gender'] = covariates['gender'].map(gender_map).fillna(covariates['gender'])
    covariates['gender_male'] = (covariates['gender'] == 'male').astype(int)

print(f"  Loaded covariates for {len(covariates)} participants")

# Check required columns
required_cols = ['ucla_total', 'gender_male', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']
missing = [col for col in required_cols if col not in covariates.columns]
if missing:
    print(f"ERROR: Missing covariate columns: {missing}")
    sys.exit(1)

# Standardize covariates
scaler = StandardScaler()
covariates['z_age'] = scaler.fit_transform(covariates[['age']])
covariates['z_ucla'] = scaler.fit_transform(covariates[['ucla_total']])
covariates['z_dass_dep'] = scaler.fit_transform(covariates[['dass_depression']])
covariates['z_dass_anx'] = scaler.fit_transform(covariates[['dass_anxiety']])
covariates['z_dass_str'] = scaler.fit_transform(covariates[['dass_stress']])

print("\n" + "=" * 80)
print("STEP 1: Compute Participant-Level RT Quantiles")
print("=" * 80)

quantiles_to_compute = [0.10, 0.25, 0.50, 0.75, 0.90]

all_quantile_data = []

# PRP: T2_RT by SOA condition
if has_prp:
    print("\nComputing PRP T2_RT quantiles...")

    # Filter valid trials
    prp_clean = prp_trials[
        (prp_trials['t2_rt'].notna()) &
        (prp_trials['t2_rt'] > 200) &
        (prp_trials['t2_rt'] < 5000)
    ].copy()

    # Categorize SOA
    def categorize_soa(soa):
        if pd.isna(soa):
            return 'other'
        if soa <= 150:
            return 'short'
        elif 300 <= soa <= 600:
            return 'medium'
        elif soa >= 1200:
            return 'long'
        else:
            return 'other'

    prp_clean['soa_cat'] = prp_clean['soa'].apply(categorize_soa)

    for soa_condition in ['short', 'long']:
        prp_soa = prp_clean[prp_clean['soa_cat'] == soa_condition]

        if len(prp_soa) < 100:
            print(f"  Skipping PRP {soa_condition} SOA (insufficient trials)")
            continue

        # Compute quantiles by participant
        quantile_df = prp_soa.groupby('participant_id')['t2_rt'].quantile(quantiles_to_compute).unstack()
        quantile_df = quantile_df.reset_index()
        quantile_df.columns = ['participant_id'] + [f'q_{int(q*100)}' for q in quantiles_to_compute]

        # Merge covariates
        quantile_df = quantile_df.merge(covariates, on='participant_id', how='inner')

        # Add metadata
        quantile_df['task'] = 'PRP'
        quantile_df['condition'] = soa_condition

        all_quantile_data.append(quantile_df)
        print(f"  PRP {soa_condition} SOA: {len(quantile_df)} participants")

# WCST: Overall RT
if has_wcst:
    print("\nComputing WCST RT quantiles...")

    wcst_clean = wcst_trials[
        (wcst_trials['reactiontimems'].notna()) &
        (wcst_trials['reactiontimems'] > 200) &
        (wcst_trials['reactiontimems'] < 5000)
    ].copy()

    if len(wcst_clean) >= 100:
        quantile_df = wcst_clean.groupby('participant_id')['reactiontimems'].quantile(quantiles_to_compute).unstack()
        quantile_df = quantile_df.reset_index()
        quantile_df.columns = ['participant_id'] + [f'q_{int(q*100)}' for q in quantiles_to_compute]

        quantile_df = quantile_df.merge(covariates, on='participant_id', how='inner')
        quantile_df['task'] = 'WCST'
        quantile_df['condition'] = 'overall'

        all_quantile_data.append(quantile_df)
        print(f"  WCST: {len(quantile_df)} participants")

# Stroop: Incongruent trials only
if has_stroop:
    print("\nComputing Stroop RT quantiles (incongruent trials)...")

    stroop_clean = stroop_trials[
        (stroop_trials['type'].str.lower() == 'incongruent') &
        (stroop_trials['rt_ms'].notna()) &
        (stroop_trials['rt_ms'] > 200) &
        (stroop_trials['rt_ms'] < 5000)
    ].copy()

    if len(stroop_clean) >= 100:
        quantile_df = stroop_clean.groupby('participant_id')['rt_ms'].quantile(quantiles_to_compute).unstack()
        quantile_df = quantile_df.reset_index()
        quantile_df.columns = ['participant_id'] + [f'q_{int(q*100)}' for q in quantiles_to_compute]

        quantile_df = quantile_df.merge(covariates, on='participant_id', how='inner')
        quantile_df['task'] = 'Stroop'
        quantile_df['condition'] = 'incongruent'

        all_quantile_data.append(quantile_df)
        print(f"  Stroop incongruent: {len(quantile_df)} participants")

if not all_quantile_data:
    print("\nERROR: No quantile data computed. Check trial data.")
    sys.exit(1)

# Combine all datasets
combined_df = pd.concat(all_quantile_data, ignore_index=True)
print(f"\nTotal datasets: {len(all_quantile_data)}")
print(f"Combined: {len(combined_df)} participant×task rows")

print("\n" + "=" * 80)
print("STEP 2: Quantile Regression Models")
print("=" * 80)

quantile_results = []

for task_name in combined_df['task'].unique():
    for condition in combined_df[combined_df['task'] == task_name]['condition'].unique():

        subset = combined_df[(combined_df['task'] == task_name) &
                            (combined_df['condition'] == condition)].copy()

        if len(subset) < MIN_N_REGRESSION:
            continue

        print(f"\n{task_name} - {condition}:")
        print("-" * 60)
        print(f"  N = {len(subset)}")

        for q in quantiles_to_compute:
            q_col = f'q_{int(q*100)}'

            # Drop missing
            df_q = subset.dropna(subset=[q_col, 'z_ucla', 'gender_male',
                                        'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])

            if len(df_q) < MIN_N_REGRESSION:
                continue

            # Standardize outcome for effect size comparison
            df_q['z_rt'] = scaler.fit_transform(df_q[[q_col]])

            # Test group differences on this percentile using OLS
            # NOTE: This is NOT quantile regression (QuantReg)
            # We are testing: Do groups differ in their AVERAGE percentile value?
            # Not: Does the effect vary at different points of the conditional distribution?
            formula = "z_rt ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

            try:
                # Use OLS to compare groups on participant-level percentiles
                # For true quantile regression, see: true_quantile_regression_analysis.py
                model = smf.ols(formula, data=df_q).fit()

                # Extract coefficients
                ucla_main = model.params.get('z_ucla', np.nan)
                ucla_main_p = model.pvalues.get('z_ucla', np.nan)

                int_term = 'z_ucla:C(gender_male)[T.1]'
                if int_term in model.params:
                    interaction = model.params[int_term]
                    interaction_p = model.pvalues[int_term]
                    interaction_se = model.bse[int_term]
                else:
                    interaction = np.nan
                    interaction_p = np.nan
                    interaction_se = np.nan

                quantile_results.append({
                    'task': task_name,
                    'condition': condition,
                    'quantile': q,
                    'quantile_label': q_col,
                    'n': len(df_q),
                    'ucla_main_beta': ucla_main,
                    'ucla_main_p': ucla_main_p,
                    'interaction_beta': interaction,
                    'interaction_p': interaction_p,
                    'interaction_se': interaction_se,
                    'r_squared': model.rsquared
                })

                sig_marker = '*' if interaction_p < 0.05 else ''
                print(f"    q={q:.2f}: UCLA×Gender β={interaction:.4f}, p={interaction_p:.4f} {sig_marker}")

            except Exception as e:
                print(f"    q={q:.2f}: Model failed ({e})")

# Convert to DataFrame
results_df = pd.DataFrame(quantile_results)

if len(results_df) == 0:
    print("\nERROR: No quantile regression results. Check data quality.")
    sys.exit(1)

# Save results
results_df.to_csv(OUTPUT_DIR / "quantile_coefficients.csv",
                  index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: quantile_coefficients.csv ({len(results_df)} models)")

print("\n" + "=" * 80)
print("STEP 3: Visualizations")
print("=" * 80)

# 3A: Interaction beta across quantiles (line plot)
print("\nCreating quantile effects plot...")

fig, axes = plt.subplots(1, len(results_df['task'].unique()), figsize=(5*len(results_df['task'].unique()), 5))

if len(results_df['task'].unique()) == 1:
    axes = [axes]

for idx, task_name in enumerate(sorted(results_df['task'].unique())):
    ax = axes[idx]

    task_data = results_df[results_df['task'] == task_name]

    for condition in task_data['condition'].unique():
        cond_data = task_data[task_data['condition'] == condition].sort_values('quantile')

        ax.plot(cond_data['quantile'], cond_data['interaction_beta'],
                marker='o', linewidth=2, markersize=8, label=condition)

        # Add significance markers
        for _, row in cond_data.iterrows():
            if row['interaction_p'] < 0.05:
                ax.scatter(row['quantile'], row['interaction_beta'],
                          s=200, marker='*', color='red', zorder=5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Quantile', fontsize=11)
    ax.set_ylabel('UCLA × Gender Interaction β', fontsize=11)
    ax.set_title(f'{task_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "quantile_effects_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: quantile_effects_plot.png")

# 3B: Heatmap (task × quantile, interaction beta)
print("\nCreating quantile heatmap...")

# Pivot for heatmap
heatmap_data = results_df.pivot_table(
    index=['task', 'condition'],
    columns='quantile',
    values='interaction_beta'
)

fig, ax = plt.subplots(figsize=(10, len(heatmap_data) * 0.5 + 2))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, cbar_kws={'label': 'UCLA × Gender β'},
            linewidths=0.5, ax=ax)
ax.set_xlabel('Quantile', fontsize=11)
ax.set_ylabel('Task - Condition', fontsize=11)
ax.set_title('UCLA × Gender Interaction Across RT Quantiles', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "quantile_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: quantile_heatmap.png")

print("\n" + "=" * 80)
print("STEP 4: Compare with Ex-Gaussian Tau (if available)")
print("=" * 80)

# Try to load Ex-Gaussian results for comparison
exgaussian_file = RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv"
if exgaussian_file.exists():
    print("\nLoading Ex-Gaussian tau results...")
    exg_df = pd.read_csv(exgaussian_file, encoding='utf-8-sig')

    # Merge tau with quantile results
    # This is task-specific - we'll just note tau effect direction

    print("  Ex-Gaussian tau effects found (see separate analysis)")
    print("  Compare: If tau correlates with UCLA×Gender, q=0.90 should show strongest effect")

    comparison_summary = []
    for task_name in results_df['task'].unique():
        task_data = results_df[results_df['task'] == task_name]

        q90_data = task_data[task_data['quantile'] == 0.90]
        q50_data = task_data[task_data['quantile'] == 0.50]

        if len(q90_data) > 0 and len(q50_data) > 0:
            for condition in task_data['condition'].unique():
                q90_beta = q90_data[q90_data['condition'] == condition]['interaction_beta'].values
                q50_beta = q50_data[q50_data['condition'] == condition]['interaction_beta'].values

                if len(q90_beta) > 0 and len(q50_beta) > 0:
                    tail_stronger = abs(q90_beta[0]) > abs(q50_beta[0])

                    comparison_summary.append({
                        'task': task_name,
                        'condition': condition,
                        'q50_beta': q50_beta[0],
                        'q90_beta': q90_beta[0],
                        'tail_effect_stronger': tail_stronger
                    })

    if comparison_summary:
        comp_df = pd.DataFrame(comparison_summary)
        comp_df.to_csv(OUTPUT_DIR / "tail_vs_center_comparison.csv",
                      index=False, encoding='utf-8-sig')
        print(f"✓ Saved: tail_vs_center_comparison.csv")

        print("\nTail vs Center comparison:")
        for _, row in comp_df.iterrows():
            status = "TAIL STRONGER" if row['tail_effect_stronger'] else "Center stronger"
            print(f"  {row['task']} {row['condition']}: {status} (q50={row['q50_beta']:.3f}, q90={row['q90_beta']:.3f})")

else:
    print("\nEx-Gaussian results not found. Skipping comparison.")

print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

# Create comprehensive report
report_path = OUTPUT_DIR / "QUANTILE_REGRESSION_REPORT.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("QUANTILE REGRESSION ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 80 + "\n")
    f.write("Test whether UCLA × Gender effects are concentrated in RT distribution tails\n")
    f.write("(90th percentile = slow lapses) rather than central tendency (median).\n\n")

    f.write("HYPOTHESIS\n")
    f.write("-" * 80 + "\n")
    f.write("If loneliness affects attentional lapses (tau in Ex-Gaussian), we expect:\n")
    f.write("  - STRONG effects at q=0.90 (slow tail)\n")
    f.write("  - WEAK effects at q=0.50 (median)\n")
    f.write("This would converge with Ex-Gaussian tau findings.\n\n")

    f.write("METHOD\n")
    f.write("-" * 80 + "\n")
    f.write(f"Quantiles: {quantiles_to_compute}\n")
    f.write(f"Models tested: {len(results_df)}\n")
    f.write(f"Formula: RT(quantile) ~ UCLA × Gender + DASS(3) + Age\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-" * 80 + "\n\n")

    for task_name in results_df['task'].unique():
        task_data = results_df[results_df['task'] == task_name]
        f.write(f"{task_name}:\n")

        for condition in task_data['condition'].unique():
            cond_data = task_data[task_data['condition'] == condition].sort_values('quantile')

            f.write(f"  {condition}:\n")
            for _, row in cond_data.iterrows():
                sig = '*' if row['interaction_p'] < 0.05 else ''
                f.write(f"    q={row['quantile']:.2f}: β={row['interaction_beta']:.4f}, p={row['interaction_p']:.4f} {sig}\n")
            f.write("\n")

    f.write("INTERPRETATION\n")
    f.write("-" * 80 + "\n")

    # Check if tail effects are systematically stronger
    if 'tail_vs_center_comparison.csv' in [f.name for f in OUTPUT_DIR.iterdir()]:
        comp_df = pd.read_csv(OUTPUT_DIR / "tail_vs_center_comparison.csv")
        n_tail_stronger = sum(comp_df['tail_effect_stronger'])
        n_total = len(comp_df)

        if n_tail_stronger > n_total / 2:
            f.write(f"✓ {n_tail_stronger}/{n_total} tasks show STRONGER effects in tail (q=0.90) than center (q=0.50)\n")
            f.write("  → CONSISTENT with lapse hypothesis (tau-driven effects)\n")
            f.write("  → UCLA × Gender impairment concentrated in slowest trials\n")
            f.write("  → Convergent evidence with Ex-Gaussian tau analysis\n\n")
        else:
            f.write(f"✗ Only {n_tail_stronger}/{n_total} tasks show stronger tail effects\n")
            f.write("  → Effects NOT primarily driven by lapses\n")
            f.write("  → May reflect general slowing (mu) rather than tail (tau)\n\n")
    else:
        f.write("Effects pattern varies across tasks. Check quantile_effects_plot.png for visual summary.\n\n")

    f.write("THEORETICAL IMPLICATIONS\n")
    f.write("-" * 80 + "\n")
    f.write("If q=0.90 >> q=0.50:\n")
    f.write("  → Loneliness × Gender affects LAPSES/attentional failures (not sustained slowing)\n")
    f.write("  → Supports reactive control deficit (occasional breakdowns)\n")
    f.write("  → Aligns with dual mechanisms of control (DMC) framework\n\n")

    f.write("If q=0.90 ≈ q=0.50:\n")
    f.write("  → Loneliness × Gender affects ENTIRE RT distribution uniformly\n")
    f.write("  → Supports general slowing (mu/sigma) rather than lapse-specific (tau)\n")
    f.write("  → May reflect energetic/motivational depletion\n\n")

    f.write("NEXT STEPS\n")
    f.write("-" * 80 + "\n")
    f.write("- Compare with Ex-Gaussian tau results for convergent validity\n")
    f.write("- If tail-dominant: Emphasize lapse-vulnerability in Discussion\n")
    f.write("- If uniform: Emphasize general slowing mechanism\n")

print(f"✓ Saved: QUANTILE_REGRESSION_REPORT.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("\nKey outputs:")
print("  1. quantile_coefficients.csv - Full regression results by quantile")
print("  2. quantile_effects_plot.png - Line plot showing effect across quantiles")
print("  3. quantile_heatmap.png - Heatmap of interaction betas")
print("  4. tail_vs_center_comparison.csv - q=0.90 vs q=0.50 comparison")
print("  5. QUANTILE_REGRESSION_REPORT.txt - Interpretation summary")
print("\nKey question: Are effects stronger at q=0.90 (tail) than q=0.50 (median)?")
print("→ If YES: Supports lapse hypothesis (tau-driven)")
print("→ If NO: General slowing (mu-driven)")
