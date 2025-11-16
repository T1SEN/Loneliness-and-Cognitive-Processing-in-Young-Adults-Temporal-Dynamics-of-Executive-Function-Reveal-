#!/usr/bin/env python3
"""Loneliness x Executive Function Analysis - Fixed Version"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("Loneliness x Executive Function Analysis")
print("=" * 80)

data_dir = Path("results")
output_dir = data_dir / "analysis_outputs"
output_dir.mkdir(exist_ok=True)

# Load data
print("\n[Step 1] Loading CSV files...")
participants = pd.read_csv(data_dir / "1_participants_info.csv")
surveys_raw = pd.read_csv(data_dir / "2_surveys_results.csv")
stroop_trials = pd.read_csv(data_dir / "4c_stroop_trials.csv")
wcst_trials = pd.read_csv(data_dir / "4b_wcst_trials.csv")
prp_trials = pd.read_csv(data_dir / "4a_prp_trials.csv")

print(f"Participants: {len(participants)}")
print(f"Survey responses: {len(surveys_raw)}")

# Process surveys (convert long to wide format)
print("\n[Step 2] Processing survey data...")
ucla = surveys_raw[surveys_raw['surveyName'] == 'ucla'][['participantId', 'score']].copy()
ucla.columns = ['participant_id', 'ucla_total']

dass = surveys_raw[surveys_raw['surveyName'] == 'dass'][['participantId', 'score_D', 'score_A', 'score_S']].copy()
dass.columns = ['participant_id', 'dass_depression', 'dass_anxiety', 'dass_stress']

surveys = ucla.merge(dass, on='participant_id', how='inner')
print(f"Complete survey data: {len(surveys)} participants")

# Compute Stroop interference
print("\n[Step 3] Computing Stroop interference...")
stroop_correct = stroop_trials[stroop_trials['correct'] == True].copy()
stroop_summary = stroop_correct.groupby(['participant_id', 'cond'])['rt_ms'].mean().unstack()

if 'congruent' in stroop_summary.columns and 'incongruent' in stroop_summary.columns:
    stroop_summary['stroop_interference'] = stroop_summary['incongruent'] - stroop_summary['congruent']
    print(f"Stroop data: {len(stroop_summary)} participants")
else:
    print("ERROR: Stroop data missing expected conditions")
    stroop_summary['stroop_interference'] = np.nan

# Compute WCST perseverative errors
print("\n[Step 4] Computing WCST perseverative errors...")
wcst_pe = []
for pid, group in wcst_trials.groupby('participant_id'):
    total = len(group)
    pe_count = 0

    for extra_str in group['extra']:
        if isinstance(extra_str, str):
            try:
                extra = eval(extra_str)
                if extra.get('isPE', False):
                    pe_count += 1
            except:
                pass

    wcst_pe.append({
        'participant_id': pid,
        'pe_rate': pe_count / total * 100 if total > 0 else 0
    })

wcst_summary = pd.DataFrame(wcst_pe)
print(f"WCST data: {len(wcst_summary)} participants")

# Compute PRP bottleneck
print("\n[Step 5] Computing PRP bottleneck effect...")
prp_correct = prp_trials[
    (prp_trials['t2_correct'] == True) &
    (prp_trials['t2_rt_ms'].notna()) &
    (prp_trials['t2_rt_ms'] > 0)
].copy()

prp_by_soa = prp_correct.groupby(['participant_id', 'soa_nominal_ms'])['t2_rt_ms'].mean().unstack()

# Bottleneck = short SOA (50-150) minus long SOA (1200)
short_cols = [c for c in prp_by_soa.columns if c <= 150]
long_cols = [c for c in prp_by_soa.columns if c >= 1200]

if short_cols and long_cols:
    prp_by_soa['prp_bottleneck'] = prp_by_soa[short_cols].mean(axis=1) - prp_by_soa[long_cols].mean(axis=1)
    print(f"PRP data: {len(prp_by_soa)} participants")
else:
    print("WARNING: PRP data may not have expected SOA range")
    prp_by_soa['prp_bottleneck'] = np.nan

# Merge all datasets
print("\n[Step 6] Merging all data...")
master = surveys.copy()
master = master.merge(
    stroop_summary.reset_index()[['participant_id', 'stroop_interference']],
    on='participant_id',
    how='left'
)
master = master.merge(wcst_summary, on='participant_id', how='left')
master = master.merge(
    prp_by_soa.reset_index()[['participant_id', 'prp_bottleneck']],
    on='participant_id',
    how='left'
)

print(f"Total merged: {len(master)} participants")

master_clean = master.dropna(subset=[
    'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
    'stroop_interference', 'pe_rate', 'prp_bottleneck'
])

print(f"Complete cases (all measures): {len(master_clean)} participants")

if len(master_clean) < 20:
    print("\nWARNING: Less than 20 complete cases. Results may be unreliable.")
    print("Consider relaxing missing data requirements or collecting more data.")

master_clean.to_csv(output_dir / "master_dataset.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: {output_dir / 'master_dataset.csv'}")

# Descriptive statistics
print("\n" + "=" * 80)
print("Descriptive Statistics")
print("=" * 80)

vars_of_interest = [
    'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
    'stroop_interference', 'pe_rate', 'prp_bottleneck'
]

desc = master_clean[vars_of_interest].describe()
print("\n", desc.round(2))

desc.to_csv(output_dir / "descriptive_stats.csv", encoding='utf-8-sig')

# Correlations
print("\n" + "=" * 80)
print("Correlation Matrix (Pearson r)")
print("=" * 80)

corr = master_clean[vars_of_interest].corr()
print("\n", corr.round(3))

corr.to_csv(output_dir / "correlation_matrix.csv", encoding='utf-8-sig')

# P-values for correlations
from scipy import stats

print("\n" + "=" * 80)
print("P-values for Correlations")
print("=" * 80)

pval_matrix = pd.DataFrame(
    np.ones((len(vars_of_interest), len(vars_of_interest))),
    index=vars_of_interest,
    columns=vars_of_interest
)

for i, var1 in enumerate(vars_of_interest):
    for j, var2 in enumerate(vars_of_interest):
        if i != j:
            r, p = stats.pearsonr(
                master_clean[var1].dropna(),
                master_clean[var2].dropna()
            )
            pval_matrix.iloc[i, j] = p

print("\n", pval_matrix.round(4))
pval_matrix.to_csv(output_dir / "correlation_pvalues.csv", encoding='utf-8-sig')

# Simple correlations: UCLA vs EF measures
print("\n" + "=" * 80)
print("KEY QUESTION: Does UCLA Loneliness predict Executive Function?")
print("=" * 80)

ef_measures = ['stroop_interference', 'pe_rate', 'prp_bottleneck']
ef_labels = {
    'stroop_interference': 'Stroop Interference (ms)',
    'pe_rate': 'WCST Perseverative Error Rate (%)',
    'prp_bottleneck': 'PRP Bottleneck Effect (ms)'
}

for measure in ef_measures:
    r, p = stats.pearsonr(master_clean['ucla_total'], master_clean[measure])
    print(f"\n{ef_labels[measure]}:")
    print(f"  r = {r:.3f}, r^2 = {r**2:.3f}, p = {p:.4f}")

    if p < 0.001:
        print(f"  *** HIGHLY SIGNIFICANT (p < .001)")
    elif p < 0.01:
        print(f"  ** SIGNIFICANT (p < .01)")
    elif p < 0.05:
        print(f"  * SIGNIFICANT (p < .05)")
    else:
        print(f"  NOT significant (p >= .05)")

# Partial correlations (controlling for DASS)
print("\n" + "=" * 80)
print("ADVANCED: Partial Correlations (controlling for DASS-21)")
print("=" * 80)

from scipy.stats import pearsonr

def partial_corr(df, x, y, covariates):
    """Compute partial correlation between x and y, controlling for covariates"""
    from sklearn.linear_model import LinearRegression

    # Residualize x
    lr_x = LinearRegression()
    lr_x.fit(df[covariates], df[x])
    resid_x = df[x] - lr_x.predict(df[covariates])

    # Residualize y
    lr_y = LinearRegression()
    lr_y.fit(df[covariates], df[y])
    resid_y = df[y] - lr_y.predict(df[covariates])

    # Correlation of residuals
    return pearsonr(resid_x, resid_y)

covariates = ['dass_depression', 'dass_anxiety', 'dass_stress']

for measure in ef_measures:
    r, p = partial_corr(master_clean, 'ucla_total', measure, covariates)
    print(f"\n{ef_labels[measure]} (controlling for DASS):")
    print(f"  partial r = {r:.3f}, p = {p:.4f}")

    if p < 0.05:
        print(f"  * SIGNIFICANT! UCLA has unique effect beyond DASS")
    else:
        print(f"  NOT significant - UCLA effect explained by DASS")

# PCA for common factor
print("\n" + "=" * 80)
print("Common 'Meta-Control' Factor (PCA)")
print("=" * 80)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ef_data = master_clean[ef_measures].values
scaler = StandardScaler()
ef_scaled = scaler.fit_transform(ef_data)

pca = PCA(n_components=1)
pca.fit(ef_scaled)

print(f"\nVariance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print("\nComponent loadings:")

loadings = pd.DataFrame({
    'Task': ['Stroop Interference', 'WCST Perseverative Errors', 'PRP Bottleneck'],
    'Loading': pca.components_[0]
})
print(loadings.to_string(index=False))

loadings.to_csv(output_dir / "pca_loadings.csv", index=False, encoding='utf-8-sig')

# Compute factor scores
factor_scores = pca.transform(ef_scaled)
master_clean['meta_control_factor'] = factor_scores

# Factor vs UCLA
r, p = stats.pearsonr(master_clean['meta_control_factor'], master_clean['ucla_total'])
print(f"\nMeta-Control Factor vs UCLA Loneliness:")
print(f"  r = {r:.3f}, p = {p:.4f}")

if p < 0.05:
    print(f"  * SIGNIFICANT - Loneliness predicts common EF factor")
else:
    print(f"  NOT significant")

# Partial correlation: Factor vs UCLA (controlling for DASS)
r_partial, p_partial = partial_corr(
    master_clean,
    'ucla_total',
    'meta_control_factor',
    covariates
)

print(f"\nMeta-Control Factor vs UCLA (controlling for DASS):")
print(f"  partial r = {r_partial:.3f}, p = {p_partial:.4f}")

if p_partial < 0.05:
    print(f"  * SIGNIFICANT - Loneliness has unique effect on meta-control")
else:
    print(f"  NOT significant - Effect explained by DASS")

# Save final dataset with factor scores
master_clean.to_csv(output_dir / "master_with_factor.csv", index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\nN = {len(master_clean)} participants with complete data")
print(f"\nResults saved to: {output_dir}")
print("\nFiles created:")
print("  1. master_dataset.csv - Full merged dataset")
print("  2. master_with_factor.csv - Dataset + meta-control factor scores")
print("  3. descriptive_stats.csv")
print("  4. correlation_matrix.csv")
print("  5. correlation_pvalues.csv")
print("  6. pca_loadings.csv")

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR Q1 PAPER:")
print("=" * 80)
print("1. If partial correlations are significant -> Strong story for Q1")
print("2. If only zero-order correlations significant -> Q2 level")
print("3. Sample size N ~" + str(len(master_clean)) + ":")

if len(master_clean) >= 80:
    print("   -> Good for Q1/Q2 submission")
elif len(master_clean) >= 50:
    print("   -> Acceptable for Q2, borderline for Q1")
else:
    print("   -> Consider as pilot, collect more data for publication")

print("\n4. Suggested target journals:")
print("   - Cognition & Emotion (Q1, IF ~3.5)")
print("   - Cognitive, Affective, & Behavioral Neuroscience (Q1, IF ~3.0)")
print("   - PLOS ONE (Q2, IF ~2.9, online data welcome)")
print("   - Acta Psychologica (Q2, IF ~2.5)")

print("\n" + "=" * 80)
