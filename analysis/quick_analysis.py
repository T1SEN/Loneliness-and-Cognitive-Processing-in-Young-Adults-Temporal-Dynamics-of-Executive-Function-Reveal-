#!/usr/bin/env python3
"""Quick analysis of loneliness x executive function data"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Force UTF-8 output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Loneliness x Executive Function Analysis")
print("=" * 70)

data_dir = Path("results")
output_dir = data_dir / "analysis_outputs"
output_dir.mkdir(exist_ok=True)

# Load data
print("\n[Step 1] Loading CSV files...")
participants = pd.read_csv(data_dir / "1_participants_info.csv")
surveys = pd.read_csv(data_dir / "2_surveys_results.csv")
stroop_trials = pd.read_csv(data_dir / "4c_stroop_trials.csv")
wcst_trials = pd.read_csv(data_dir / "4b_wcst_trials.csv")
prp_trials = pd.read_csv(data_dir / "4a_prp_trials.csv")

print(f"Participants: {len(participants)}")
print(f"Survey responses: {len(surveys)}")
print(f"Stroop trials: {len(stroop_trials)}")
print(f"WCST trials: {len(wcst_trials)}")
print(f"PRP trials: {len(prp_trials)}")

# Compute Stroop interference
print("\n[Step 2] Computing Stroop interference...")
stroop_correct = stroop_trials[stroop_trials['correct'] == True].copy()
stroop_summary = stroop_correct.groupby(['participant_id', 'cond'])['rt_ms'].mean().unstack()
stroop_summary['stroop_interference'] = stroop_summary['incongruent'] - stroop_summary['congruent']

# Compute WCST perseverative errors
print("\n[Step 3] Computing WCST errors...")
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
    wcst_pe.append({'participant_id': pid, 'pe_rate': pe_count / total * 100 if total > 0 else 0})

wcst_summary = pd.DataFrame(wcst_pe)

# Compute PRP bottleneck
print("\n[Step 4] Computing PRP bottleneck...")
prp_correct = prp_trials[(prp_trials['t2_correct'] == True) & (prp_trials['t2_rt_ms'] > 0)].copy()
prp_summary = prp_correct.groupby(['participant_id', 'soa_nominal_ms'])['t2_rt_ms'].mean().unstack()

# Bottleneck = T2 RT at short SOA (50-150) - T2 RT at long SOA (1200)
short_soas = [c for c in prp_summary.columns if c <= 150]
long_soas = [c for c in prp_summary.columns if c >= 1200]

if short_soas and long_soas:
    prp_summary['prp_bottleneck'] = prp_summary[short_soas].mean(axis=1) - prp_summary[long_soas].mean(axis=1)
else:
    prp_summary['prp_bottleneck'] = np.nan

# Merge all data
print("\n[Step 5] Merging datasets...")
master = surveys[['participant_id', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']].copy()
master = master.merge(stroop_summary[['stroop_interference']], left_on='participant_id', right_index=True, how='left')
master = master.merge(wcst_summary, on='participant_id', how='left')
master = master.merge(prp_summary[['prp_bottleneck']], left_on='participant_id', right_index=True, how='left')

print(f"Total participants before cleaning: {len(master)}")
master_clean = master.dropna()
print(f"Complete cases: {len(master_clean)}")

# Save master dataset
master_clean.to_csv(output_dir / "master_dataset.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: {output_dir / 'master_dataset.csv'}")

# Descriptive statistics
print("\n" + "=" * 70)
print("Descriptive Statistics")
print("=" * 70)
desc = master_clean.describe()
print(desc.to_string())
desc.to_csv(output_dir / "descriptive_stats.csv", encoding='utf-8-sig')

# Correlations
print("\n" + "=" * 70)
print("Correlation Matrix (Pearson)")
print("=" * 70)
corr_vars = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
             'stroop_interference', 'pe_rate', 'prp_bottleneck']
corr = master_clean[corr_vars].corr()
print(corr.round(3).to_string())
corr.to_csv(output_dir / "correlations.csv", encoding='utf-8-sig')

# Simple regressions (UCLA predicting each EF measure)
print("\n" + "=" * 70)
print("UCLA Loneliness -> Executive Function")
print("=" * 70)

from scipy import stats

for outcome in ['stroop_interference', 'pe_rate', 'prp_bottleneck']:
    x = master_clean['ucla_total']
    y = master_clean[outcome]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print(f"\n{outcome}:")
    print(f"  r = {r_value:.3f}, r2 = {r_value**2:.3f}, p = {p_value:.4f}")
    print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'}")

# PCA for common factor
print("\n" + "=" * 70)
print("Common Meta-Control Factor (PCA)")
print("=" * 70)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ef_data = master_clean[['stroop_interference', 'pe_rate', 'prp_bottleneck']].values
scaler = StandardScaler()
ef_scaled = scaler.fit_transform(ef_data)

pca = PCA(n_components=1)
pca.fit(ef_scaled)

print(f"\nVariance explained: {pca.explained_variance_ratio_[0]:.2%}")
print("\nLoadings:")
loadings = pd.DataFrame({
    'Task': ['Stroop', 'WCST', 'PRP'],
    'Loading': pca.components_[0]
})
print(loadings.to_string(index=False))

# Factor scores
factor_scores = pca.transform(ef_scaled)
master_clean['meta_control'] = factor_scores

# Factor vs UCLA
r, p = stats.pearsonr(master_clean['meta_control'], master_clean['ucla_total'])
print(f"\nMeta-control factor vs UCLA: r = {r:.3f}, p = {p:.4f}")
print(f"{'SIGNIFICANT' if p < 0.05 else 'NOT significant'}")

master_clean.to_csv(output_dir / "master_with_factor.csv", index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
print(f"\nResults saved to: {output_dir}")
print("Files created:")
print("  - master_dataset.csv")
print("  - descriptive_stats.csv")
print("  - correlations.csv")
print("  - master_with_factor.csv")
