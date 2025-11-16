"""
Stroop Neutral Baseline Re-Analysis
====================================
Recalculates interference using neutral trials as baseline,
separating pure interference from facilitation effects.

Traditional: Interference = Incongruent RT - Congruent RT
New approach:
  - Pure Interference = Incongruent RT - Neutral RT
  - Facilitation = Neutral RT - Congruent RT

Tests UCLA × Gender moderation for both metrics.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_neutral_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STROOP NEUTRAL BASELINE RE-ANALYSIS")
print("="*80)

# Load data
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')

# Normalize to participant_id (drop existing one if conflict)
if 'participantId' in participants.columns:
    if 'participant_id' in participants.columns:
        participants = participants.drop(columns=['participant_id'])
    participants = participants.rename(columns={'participantId': 'participant_id'})

if 'participantId' in surveys.columns:
    if 'participant_id' in surveys.columns:
        surveys = surveys.drop(columns=['participant_id'])
    surveys = surveys.rename(columns={'participantId': 'participant_id'})

# Extract UCLA scores
print(f"\nSurveys data: {len(surveys)} rows")
print(f"Survey names: {surveys['surveyName'].unique()}")

ucla_df = surveys[surveys['surveyName'] == 'UCLA Loneliness Scale'].copy()
print(f"UCLA rows found: {len(ucla_df)}")

if len(ucla_df) == 0:
    print("⚠ WARNING: No UCLA rows! Trying alternative...")
    # Try case-insensitive
    ucla_df = surveys[surveys['surveyName'].str.contains('UCLA', case=False, na=False)].copy()
    print(f"UCLA rows (case-insensitive): {len(ucla_df)}")

ucla_df = ucla_df.rename(columns={'score': 'ucla_score'})
ucla_df = ucla_df[['participant_id', 'ucla_score']]
print(f"UCLA final rows: {len(ucla_df)}, unique participants: {ucla_df['participant_id'].nunique()}")
print(f"Sample UCLA IDs: {ucla_df['participant_id'].head(5).tolist()}")

# DASS subscales
dass_df = surveys[surveys['surveyName'].str.contains('DASS', na=False)].copy()
dass_pivot = dass_df.pivot_table(
    index='participant_id',
    columns='surveyName',
    values='score',
    aggfunc='first'
).reset_index()
dass_pivot.columns.name = None
if 'DASS-21-Depression' in dass_pivot.columns:
    dass_pivot = dass_pivot.rename(columns={
        'DASS-21-Depression': 'dass_dep',
        'DASS-21-Anxiety': 'dass_anx',
        'DASS-21-Stress': 'dass_stress'
    })

# Merge
print(f"\nParticipants data: {len(participants)} rows, {participants['participant_id'].nunique()} unique")
print(f"Sample participant IDs: {participants['participant_id'].head(5).tolist()}")

master = participants[['participant_id', 'age', 'gender']].copy()
print(f"Master after selecting columns: {len(master)} rows")

master = master.merge(ucla_df, on='participant_id', how='inner')
print(f"Master after UCLA merge: {len(master)} rows")

master = master.merge(dass_pivot, on='participant_id', how='left')
print(f"Master after DASS merge: {len(master)} rows")

# Filter valid Stroop trials
stroop_valid = stroop_trials[
    (stroop_trials['cond'].notna()) &
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 0) &
    (stroop_trials['rt_ms'] < 5000)
].copy()

print(f"\nTotal valid Stroop trials: {len(stroop_valid)}")
print(f"Participants with trials: {stroop_valid['participant_id'].nunique()}")

# Identify trial types from 'cond' column
# Expected: 'congruent', 'incongruent', 'neutral'
print("\nTrial type distribution:")
print(stroop_valid['cond'].value_counts())

# Calculate condition-specific RTs per participant
print(f"\nGrouping by participant_id (sample values): {stroop_valid['participant_id'].head(10).tolist()}")

condition_rts = stroop_valid.groupby(['participant_id', 'cond'])['rt_ms'].mean().unstack(fill_value=np.nan)
condition_rts.columns = [f'{col}_rt' for col in condition_rts.columns]
condition_rts = condition_rts.reset_index()

print("\nCondition RT columns created:")
print(condition_rts.columns.tolist())
print(f"Condition RT rows: {len(condition_rts)}")
print(f"Sample condition_rts:\n{condition_rts.head()}")

print(f"\nMaster before merge: {len(master)} rows, {master['participant_id'].nunique()} unique IDs")
print(f"Sample master participant_ids: {master['participant_id'].head(10).tolist()}")

# Merge into master
master = master.merge(condition_rts, on='participant_id', how='left')

print(f"Master after merge: {len(master)} rows")
print(f"Non-null congruent_rt: {master['congruent_rt'].notna().sum()}")
print(f"Non-null incongruent_rt: {master['incongruent_rt'].notna().sum()}")
print(f"Non-null neutral_rt: {master['neutral_rt'].notna().sum()}")

# Calculate NEW interference metrics
if 'neutral_rt' in master.columns:
    # Pure interference (Incong - Neutral)
    master['pure_interference'] = master['incongruent_rt'] - master['neutral_rt']

    # Facilitation (Neutral - Cong)
    master['facilitation'] = master['neutral_rt'] - master['congruent_rt']

    # Traditional (for comparison)
    master['traditional_interference'] = master['incongruent_rt'] - master['congruent_rt']

    print("\n✓ Neutral baseline metrics calculated:")
    print(f"  - Pure Interference: mean={master['pure_interference'].mean():.2f}, SD={master['pure_interference'].std():.2f}")
    print(f"  - Facilitation: mean={master['facilitation'].mean():.2f}, SD={master['facilitation'].std():.2f}")
    print(f"  - Traditional: mean={master['traditional_interference'].mean():.2f}, SD={master['traditional_interference'].std():.2f}")
else:
    print("\n⚠ WARNING: 'neutral' condition not found in data!")
    master['pure_interference'] = np.nan
    master['facilitation'] = np.nan
    master['traditional_interference'] = master['incongruent_rt'] - master['congruent_rt']

# Gender encoding
master['gender_male'] = (master['gender'] == 'Male').astype(int)

# Z-score predictors
for col in ['ucla_score', 'age']:
    if col in master.columns:
        master[f'z_{col}'] = (master[col] - master[col].mean()) / master[col].std()

# DASS z-scores (handle missing)
for col in ['dass_dep', 'dass_anx', 'dass_stress']:
    if col in master.columns and master[col].notna().sum() > 0:
        master[f'z_{col}'] = (master[col] - master[col].mean()) / master[col].std()
    else:
        master[f'z_{col}'] = 0  # Default to 0 if missing

# Drop missing
analysis_df = master.dropna(subset=['ucla_score', 'gender', 'pure_interference', 'facilitation'])
print(f"\nAfter dropna: {len(analysis_df)} rows")
print(f"Sample data:\n{analysis_df[['participant_id', 'gender', 'ucla_score', 'pure_interference', 'facilitation']].head()}")
print(f"\nFinal analysis N: {len(analysis_df)}")

# Save metrics
metrics_summary = analysis_df[['participant_id', 'gender', 'ucla_score',
                                'pure_interference', 'facilitation', 'traditional_interference']].copy()
metrics_summary.to_csv(OUTPUT_DIR / "neutral_baseline_metrics.csv", index=False, encoding='utf-8-sig')

# ==================== STATISTICAL TESTS ====================

results = []

# Check if we have DASS data (not all zeros)
has_dass = (analysis_df['z_dass_dep'].std() > 0 and
            analysis_df['z_dass_anx'].std() > 0 and
            analysis_df['z_dass_stress'].std() > 0)

if has_dass:
    covariate_formula = '+ z_age + z_dass_dep + z_dass_anx + z_dass_stress'
else:
    covariate_formula = '+ z_age'
    print("\n⚠ WARNING: DASS data missing or all zeros, using age only as covariate\n")

# 1. Pure Interference (Incong - Neutral)
print("\n" + "="*80)
print("1. PURE INTERFERENCE (Incongruent - Neutral)")
print("="*80)

formula = f'pure_interference ~ z_ucla_score * C(gender) {covariate_formula}'
print(f"Formula: {formula}\n")
model_pure = smf.ols(formula, data=analysis_df).fit()
print(model_pure.summary())

# Extract interaction term coefficient
interaction_cols = [c for c in model_pure.params.index if 'z_ucla_score:C(gender)' in c or 'C(gender)' in c and 'z_ucla_score' in c]
inter_beta = model_pure.params[interaction_cols[0]] if interaction_cols else np.nan
inter_p = model_pure.pvalues[interaction_cols[0]] if interaction_cols else np.nan

results.append({
    'metric': 'pure_interference',
    'ucla_beta': model_pure.params.get('z_ucla_score', np.nan),
    'ucla_p': model_pure.pvalues.get('z_ucla_score', np.nan),
    'interaction_beta': inter_beta,
    'interaction_p': inter_p
})

# 2. Facilitation (Neutral - Cong)
print("\n" + "="*80)
print("2. FACILITATION (Neutral - Congruent)")
print("="*80)

formula = f'facilitation ~ z_ucla_score * C(gender) {covariate_formula}'
print(f"Formula: {formula}\n")
model_fac = smf.ols(formula, data=analysis_df).fit()
print(model_fac.summary())

interaction_cols = [c for c in model_fac.params.index if 'z_ucla_score:C(gender)' in c or ('C(gender)' in c and 'z_ucla_score' in c)]
inter_beta = model_fac.params[interaction_cols[0]] if interaction_cols else np.nan
inter_p = model_fac.pvalues[interaction_cols[0]] if interaction_cols else np.nan

results.append({
    'metric': 'facilitation',
    'ucla_beta': model_fac.params.get('z_ucla_score', np.nan),
    'ucla_p': model_fac.pvalues.get('z_ucla_score', np.nan),
    'interaction_beta': inter_beta,
    'interaction_p': inter_p
})

# 3. Traditional Interference (comparison)
print("\n" + "="*80)
print("3. TRADITIONAL INTERFERENCE (Incongruent - Congruent)")
print("="*80)

formula = f'traditional_interference ~ z_ucla_score * C(gender) {covariate_formula}'
print(f"Formula: {formula}\n")
model_trad = smf.ols(formula, data=analysis_df).fit()
print(model_trad.summary())

interaction_cols = [c for c in model_trad.params.index if 'z_ucla_score:C(gender)' in c or ('C(gender)' in c and 'z_ucla_score' in c)]
inter_beta = model_trad.params[interaction_cols[0]] if interaction_cols else np.nan
inter_p = model_trad.pvalues[interaction_cols[0]] if interaction_cols else np.nan

results.append({
    'metric': 'traditional_interference',
    'ucla_beta': model_trad.params.get('z_ucla_score', np.nan),
    'ucla_p': model_trad.pvalues.get('z_ucla_score', np.nan),
    'interaction_beta': inter_beta,
    'interaction_p': inter_p
})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "regression_results.csv", index=False, encoding='utf-8-sig')

# ==================== GENDER-STRATIFIED CORRELATIONS ====================

print("\n" + "="*80)
print("GENDER-STRATIFIED CORRELATIONS")
print("="*80)

gender_corrs = []

for gender_label in analysis_df['gender'].unique():
    subset = analysis_df[analysis_df['gender'] == gender_label]

    for metric in ['pure_interference', 'facilitation', 'traditional_interference']:
        if metric in subset.columns and subset[metric].notna().sum() > 5:
            r, p = stats.pearsonr(subset['ucla_score'], subset[metric])
            gender_corrs.append({
                'gender': gender_label,
                'metric': metric,
                'N': len(subset),
                'r': r,
                'p': p
            })
            print(f"{gender_label} - {metric}: r={r:.3f}, p={p:.3f}, N={len(subset)}")

gender_corrs_df = pd.DataFrame(gender_corrs)
gender_corrs_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

# ==================== SUMMARY ====================

print("\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

print("\nRegression Interaction Effects:")
for _, row in results_df.iterrows():
    sig = "***" if row['interaction_p'] < 0.001 else "**" if row['interaction_p'] < 0.01 else "*" if row['interaction_p'] < 0.05 else ""
    print(f"  {row['metric']}: β={row['interaction_beta']:.3f}, p={row['interaction_p']:.3f} {sig}")

print("\nGender-Stratified Correlations:")
for gender_label in gender_corrs_df['gender'].unique():
    print(f"\n{gender_label}:")
    subset = gender_corrs_df[gender_corrs_df['gender'] == gender_label]
    for _, row in subset.iterrows():
        sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
        print(f"  {row['metric']}: r={row['r']:.3f}, p={row['p']:.3f} {sig}")

print("\n✓ Analysis complete!")
print(f"✓ Results saved to: {OUTPUT_DIR}")
