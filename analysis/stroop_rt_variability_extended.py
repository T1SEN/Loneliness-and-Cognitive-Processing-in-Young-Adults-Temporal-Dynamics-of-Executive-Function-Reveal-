"""
Stroop RT Variability Extended Analysis
========================================
Extends basic IIV analysis to condition-specific variability.

Analyses:
1. Congruent trial variability
2. Incongruent trial variability
3. Variability ratio (Incong/Cong)
4. UCLA × Gender on each variability metric

Hypothesis:
- WCST showed 7.78× task-specific variability
- Test if Stroop incongruent trials show higher UCLA effects
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_deep_dive/rt_variability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STROOP RT VARIABILITY EXTENDED ANALYSIS")
print("=" * 80)

# Load
print("\n[1] Loading data...")
trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
trials.columns = trials.columns.str.lower()
if 'participantid' in trials.columns and 'participant_id' in trials.columns:
    trials = trials.drop(columns=['participantid'])
elif 'participantid' in trials.columns:
    trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()
# Map Korean gender values to English
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
master['gender'] = master['gender'].map(gender_map)

# Clean
print("\n[2] Cleaning trials...")
rt_col = 'rt_ms' if trials['rt'].isnull().sum() > len(trials) * 0.5 else 'rt'
trials_clean = trials[
    (trials['is_timeout'] == False) &
    (trials[rt_col].notna()) &
    (trials[rt_col] > 0) &
    (trials[rt_col] < 10000) &
    (trials['type'].isin(['congruent', 'incongruent']))
].copy()
trials_clean['rt'] = trials_clean[rt_col]

print(f"  Valid trials: {len(trials_clean)}")

# Condition-specific variability
print("\n[3] Computing condition-specific variability...")

def compute_variability(group):
    """IIV metrics by condition."""
    metrics = {}
    for cond in ['congruent', 'incongruent']:
        data = group[group['type'] == cond]['rt']
        if len(data) >= 10:
            metrics[f'{cond}_rt_sd'] = data.std()
            metrics[f'{cond}_rt_cv'] = data.std() / data.mean() if data.mean() > 0 else np.nan
            metrics[f'{cond}_rt_iqr'] = data.quantile(0.75) - data.quantile(0.25)
        else:
            metrics[f'{cond}_rt_sd'] = np.nan
            metrics[f'{cond}_rt_cv'] = np.nan
            metrics[f'{cond}_rt_iqr'] = np.nan
    return pd.Series(metrics)

variability_df = trials_clean.groupby('participant_id').apply(compute_variability).reset_index()

# Variability ratio
variability_df['variability_ratio_sd'] = variability_df['incongruent_rt_sd'] / variability_df['congruent_rt_sd']
variability_df['variability_ratio_cv'] = variability_df['incongruent_rt_cv'] / variability_df['congruent_rt_cv']

variability_df = variability_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Variability computed for N={len(variability_df)} participants")

variability_df.to_csv(OUTPUT_DIR / "condition_specific_variability.csv", index=False, encoding='utf-8-sig')

# Gender-stratified correlations
print("\n[4] Gender-stratified correlations...")

results = []
for gender in ['male', 'female']:
    data = variability_df[variability_df['gender'] == gender]
    if len(data) < 10:
        continue

    metrics = {
        'gender': gender,
        'n': len(data)
    }

    for col in ['congruent_rt_sd', 'incongruent_rt_sd', 'variability_ratio_sd',
                'congruent_rt_cv', 'incongruent_rt_cv', 'variability_ratio_cv']:
        valid = data.dropna(subset=['ucla_total', col])
        if len(valid) >= 10:
            r, p = pearsonr(valid['ucla_total'], valid[col])
            metrics[f'{col}_r'] = r
            metrics[f'{col}_p'] = p
        else:
            metrics[f'{col}_r'] = np.nan
            metrics[f'{col}_p'] = np.nan

    results.append(metrics)

gender_corr_df = pd.DataFrame(results)
print(gender_corr_df[['gender', 'n', 'congruent_rt_sd_r', 'incongruent_rt_sd_r', 'variability_ratio_sd_r']])

gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

# Regression models
print("\n[5] UCLA × Gender regression models...")

variability_df['gender_male'] = (variability_df['gender'] == 'male').astype(int)
for col in ['ucla_total', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']:
    variability_df[f'z_{col}'] = (variability_df[col] - variability_df[col].mean()) / variability_df[col].std()

model_results = []
for outcome in ['incongruent_rt_sd', 'congruent_rt_sd', 'variability_ratio_sd']:
    valid_data = variability_df.dropna(subset=[outcome])
    if len(valid_data) < 30:
        continue

    formula = f"{outcome} ~ z_ucla_total * C(gender_male) + z_age + z_dass_depression + z_dass_anxiety + z_dass_stress"
    model = smf.ols(formula, data=valid_data).fit()

    for term in model.params.index:
        model_results.append({
            'outcome': outcome,
            'term': term,
            'coef': model.params[term],
            'se': model.bse[term],
            't': model.tvalues[term],
            'p': model.pvalues[term]
        })

model_results_df = pd.DataFrame(model_results)
print(model_results_df[model_results_df['term'].str.contains('ucla')])

model_results_df.to_csv(OUTPUT_DIR / "ucla_gender_models.csv", index=False, encoding='utf-8-sig')

# Summary
print("\n" + "="*80)
print("STROOP RT VARIABILITY - KEY FINDINGS")
print("="*80)

print(f"""
1. Condition-Specific Variability:
   - Congruent SD: {variability_df['congruent_rt_sd'].mean():.1f} ms
   - Incongruent SD: {variability_df['incongruent_rt_sd'].mean():.1f} ms
   - Ratio: {variability_df['variability_ratio_sd'].mean():.2f}

2. Gender Correlations (incongruent SD):
   - Males: r={gender_corr_df[gender_corr_df['gender']=='male']['incongruent_rt_sd_r'].values[0]:.3f}, p={gender_corr_df[gender_corr_df['gender']=='male']['incongruent_rt_sd_p'].values[0]:.3f}
   - Females: r={gender_corr_df[gender_corr_df['gender']=='female']['incongruent_rt_sd_r'].values[0]:.3f}, p={gender_corr_df[gender_corr_df['gender']=='female']['incongruent_rt_sd_p'].values[0]:.3f}

3. Files Generated:
   - condition_specific_variability.csv
   - gender_stratified_correlations.csv
   - ucla_gender_models.csv
""")

print("\n✅ Stroop RT variability extended analysis complete!")
print("="*80)
