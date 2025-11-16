"""
PRP RT Variability Extended Analysis
=====================================
SOA-specific and T1/T2 variability analysis.

Analyses:
1. T2 RT variability by SOA (short/medium/long)
2. T1 RT variability
3. Cross-task variability correlation
4. UCLA × Gender on each metric

Hypothesis:
- Short SOA (coordination difficulty) shows higher variability effects
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
OUTPUT_DIR = Path("results/analysis_outputs/prp_deep_dive/rt_variability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRP RT VARIABILITY EXTENDED ANALYSIS")
print("=" * 80)

# Load
print("\n[1] Loading data...")
trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
trials.columns = trials.columns.str.lower()
if 'participantid' in trials.columns and 'participant_id' in trials.columns:
    trials = trials.drop(columns=['participantid'])
elif 'participantid' in trials.columns:
    trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns and 'participant_id' in participants.columns:
    participants = participants.drop(columns=['participantid'])
elif 'participantid' in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
master['gender'] = master['gender'].map(gender_map)

# Clean
print("\n[2] Cleaning trials...")
rt_col_t1 = 't1_rt_ms' if 't1_rt_ms' in trials.columns else 't1_rt'
rt_col_t2 = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 't2_rt'

trials_clean = trials[
    (trials['t1_correct'] == True) &
    (trials['t2_correct'] == True) &
    (trials[rt_col_t1].notna()) &
    (trials[rt_col_t2].notna()) &
    (trials[rt_col_t1] > 0) &
    (trials[rt_col_t2] > 0) &
    (trials[rt_col_t1] < 10000) &
    (trials[rt_col_t2] < 10000)
].copy()

trials_clean['t1_rt'] = trials_clean[rt_col_t1]
trials_clean['t2_rt'] = trials_clean[rt_col_t2]

# SOA categorization
soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in trials_clean.columns else 'soa'
trials_clean['soa_cat'] = pd.cut(
    trials_clean[soa_col],
    bins=[-np.inf, 200, 800, np.inf],
    labels=['short', 'medium', 'long']
)

print(f"  Valid dual-correct trials: {len(trials_clean)}")

# Variability metrics
print("\n[3] Computing variability metrics...")

def compute_variability(group):
    """IIV for T1, T2 overall, and T2 by SOA."""
    metrics = {}

    # T1
    if len(group) >= 10:
        metrics['t1_rt_sd'] = group['t1_rt'].std()
        metrics['t1_rt_cv'] = group['t1_rt'].std() / group['t1_rt'].mean()

    # T2 overall
    metrics['t2_rt_sd'] = group['t2_rt'].std()
    metrics['t2_rt_cv'] = group['t2_rt'].std() / group['t2_rt'].mean()

    # T2 by SOA
    for soa in ['short', 'medium', 'long']:
        soa_data = group[group['soa_cat'] == soa]['t2_rt']
        if len(soa_data) >= 5:
            metrics[f't2_rt_sd_{soa}'] = soa_data.std()
            metrics[f't2_rt_cv_{soa}'] = soa_data.std() / soa_data.mean()
        else:
            metrics[f't2_rt_sd_{soa}'] = np.nan
            metrics[f't2_rt_cv_{soa}'] = np.nan

    return pd.Series(metrics)

variability_df = trials_clean.groupby('participant_id').apply(compute_variability).reset_index()

variability_df = variability_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Variability computed for N={len(variability_df)} participants")

variability_df.to_csv(OUTPUT_DIR / "soa_specific_variability.csv", index=False, encoding='utf-8-sig')

# Gender-stratified correlations
print("\n[4] Gender-stratified correlations...")

results = []
for gender in ['male', 'female']:
    data = variability_df[variability_df['gender'] == gender]
    if len(data) < 10:
        continue

    metrics = {'gender': gender, 'n': len(data)}

    for col in ['t1_rt_sd', 't2_rt_sd', 't2_rt_sd_short', 't2_rt_sd_long']:
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
print(gender_corr_df)

gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding='utf-8-sig')

# Regression
print("\n[5] UCLA × Gender regression models...")

variability_df['gender_male'] = (variability_df['gender'] == 'male').astype(int)
for col in ['ucla_total', 'age', 'dass_depression', 'dass_anxiety', 'dass_stress']:
    variability_df[f'z_{col}'] = (variability_df[col] - variability_df[col].mean()) / variability_df[col].std()

model_results = []
for outcome in ['t2_rt_sd', 't2_rt_sd_short', 't2_rt_sd_long']:
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
print("PRP RT VARIABILITY - KEY FINDINGS")
print("="*80)

print(f"""
1. Variability Metrics:
   - T1 SD: {variability_df['t1_rt_sd'].mean():.1f} ms
   - T2 SD (overall): {variability_df['t2_rt_sd'].mean():.1f} ms
   - T2 SD (short SOA): {variability_df['t2_rt_sd_short'].mean():.1f} ms
   - T2 SD (long SOA): {variability_df['t2_rt_sd_long'].mean():.1f} ms

2. Gender Correlations:
{gender_corr_df[['gender', 'n', 't2_rt_sd_short_r', 't2_rt_sd_short_p']].to_string(index=False)}

3. Files Generated:
   - soa_specific_variability.csv
   - gender_stratified_correlations.csv
   - ucla_gender_models.csv
""")

print("\n✅ PRP RT variability extended analysis complete!")
print("="*80)
