"""
DASS 3-Way Stratification Analysis
===================================

Tests the "anxiety buffering hypothesis" by examining gender moderation effects
across all combinations of Depression × Anxiety × Stress levels.

Goal: Create a vulnerability map showing where loneliness × gender effects
are strongest vs. absent.

Hypothesis: Effect is strongest in Low Anxiety group (pure loneliness),
weakest/absent in High Anxiety group (compensatory control).

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
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_loader_utils import normalize_gender_series

np.random.seed(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/deep_dive_analysis/dass_3way")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("DASS 3-WAY STRATIFICATION ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/5] Loading data...")

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

master['gender'] = normalize_gender_series(master['gender'])
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Filter complete cases
required_cols = ['ucla_total', 'pe_rate', 'wcst_accuracy', 'gender_male',
                 'dass_depression', 'dass_anxiety', 'dass_stress']
master = master.dropna(subset=required_cols).copy()

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    std = s.std(ddof=0)
    if std is None or std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std

master['z_ucla'] = zscore(master['ucla_total'])

print(f"  N={len(master)} ({(master['gender_male']==0).sum()} Female, {(master['gender_male']==1).sum()} Male)")
print()

# ============================================================================
# CREATE TERTILE SPLITS
# ============================================================================

print("[2/5] Creating tertile splits for DASS dimensions...")

for dass_var in ['dass_depression', 'dass_anxiety', 'dass_stress']:
    # Tertile cutpoints
    low_cutoff = master[dass_var].quantile(1/3)
    high_cutoff = master[dass_var].quantile(2/3)

    master[f'{dass_var}_tertile'] = 'Medium'
    master.loc[master[dass_var] <= low_cutoff, f'{dass_var}_tertile'] = 'Low'
    master.loc[master[dass_var] >= high_cutoff, f'{dass_var}_tertile'] = 'High'

    print(f"  {dass_var}:")
    print(f"    Low: ≤{low_cutoff:.1f}")
    print(f"    Medium: {low_cutoff:.1f}-{high_cutoff:.1f}")
    print(f"    High: ≥{high_cutoff:.1f}")
    print(f"    Distribution: {master[f'{dass_var}_tertile'].value_counts().to_dict()}")

print()

# ============================================================================
# 3-WAY STRATIFICATION ANALYSIS
# ============================================================================

print("[3/5] Testing gender moderation in all 27 cells (3×3×3)...")

outcomes = ['pe_rate', 'wcst_accuracy']
stratification_results = []

dep_levels = ['Low', 'Medium', 'High']
anx_levels = ['Low', 'Medium', 'High']
stress_levels = ['Low', 'Medium', 'High']

total_cells = len(dep_levels) * len(anx_levels) * len(stress_levels)
cell_count = 0

for dep in dep_levels:
    for anx in anx_levels:
        for stress in stress_levels:
            cell_count += 1

            # Filter to this cell
            cell_data = master[
                (master['dass_depression_tertile'] == dep) &
                (master['dass_anxiety_tertile'] == anx) &
                (master['dass_stress_tertile'] == stress)
            ].copy()

            if len(cell_data) < 10:  # Skip cells with too few participants
                print(f"  [{cell_count}/{total_cells}] {dep} Dep × {anx} Anx × {stress} Stress: N={len(cell_data)} (skipped - too small)")
                continue

            for outcome in outcomes:
                if outcome not in cell_data.columns:
                    continue

                # Test moderation
                formula = f"{outcome} ~ z_ucla * C(gender_male)"
                try:
                    model = ols(formula, data=cell_data.dropna(subset=[outcome])).fit()
                    interaction_term = "z_ucla:C(gender_male)[T.1]"

                    if interaction_term in model.params:
                        beta = model.params[interaction_term]
                        pval = model.pvalues[interaction_term]

                        # Gender-stratified effects
                        female_data = cell_data[cell_data['gender_male'] == 0]
                        male_data = cell_data[cell_data['gender_male'] == 1]

                        female_corr = np.nan
                        male_corr = np.nan

                        # Female correlation (with proper paired dropna)
                        female_valid = female_data[['ucla_total', outcome]].dropna()
                        if len(female_valid) >= 5:
                            female_corr, _ = stats.pearsonr(
                                female_valid['ucla_total'],
                                female_valid[outcome]
                            )

                        # Male correlation (with proper paired dropna)
                        male_valid = male_data[['ucla_total', outcome]].dropna()
                        if len(male_valid) >= 5:
                            male_corr, _ = stats.pearsonr(
                                male_valid['ucla_total'],
                                male_valid[outcome]
                            )
                    else:
                        beta, pval = np.nan, np.nan
                        female_corr, male_corr = np.nan, np.nan

                    stratification_results.append({
                        'depression': dep,
                        'anxiety': anx,
                        'stress': stress,
                        'outcome': outcome,
                        'n': len(cell_data.dropna(subset=[outcome])),
                        'n_female': (cell_data['gender_male'] == 0).sum(),
                        'n_male': (cell_data['gender_male'] == 1).sum(),
                        'interaction_beta': beta,
                        'interaction_pval': pval,
                        'female_corr': female_corr,
                        'male_corr': male_corr
                    })

                    sig_marker = " ***" if pval < 0.001 else " **" if pval < 0.01 else " *" if pval < 0.05 else ""
                    print(f"  [{cell_count}/{total_cells}] {dep} Dep × {anx} Anx × {stress} Stress | {outcome}: β={beta:.3f}, p={pval:.4f}{sig_marker} (N={len(cell_data)}, {(cell_data['gender_male']==0).sum()}F/{(cell_data['gender_male']==1).sum()}M)")

                except Exception as e:
                    print(f"  [{cell_count}/{total_cells}] {dep} Dep × {anx} Anx × {stress} Stress | {outcome}: ERROR - {str(e)}")
                    continue

print()

# Save results
strat_df = pd.DataFrame(stratification_results)
if not strat_df.empty and 'interaction_pval' in strat_df.columns:
    mask = strat_df['interaction_pval'].notna()
    if mask.any():
        strat_df.loc[mask, 'q_value'] = multipletests(
            strat_df.loc[mask, 'interaction_pval'].astype(float).values,
            method='fdr_bh'
        )[1]

strat_df.to_csv(OUTPUT_DIR / "dass_3way_stratification.csv", index=False, encoding='utf-8-sig')
print(f"✓ Saved: dass_3way_stratification.csv ({len(strat_df)} tests)")
print()

# ============================================================================
# ANXIETY BUFFERING HYPOTHESIS TEST
# ============================================================================

print("[4/5] Testing anxiety buffering hypothesis...")

# Compare Low vs. High anxiety across all dep/stress combinations
buffering_results = []

for dep in dep_levels:
    for stress in stress_levels:
        # Low anxiety cell
        low_anx_data = master[
            (master['dass_depression_tertile'] == dep) &
            (master['dass_anxiety_tertile'] == 'Low') &
            (master['dass_stress_tertile'] == stress)
        ].copy()

        # High anxiety cell
        high_anx_data = master[
            (master['dass_depression_tertile'] == dep) &
            (master['dass_anxiety_tertile'] == 'High') &
            (master['dass_stress_tertile'] == stress)
        ].copy()

        for outcome in outcomes:
            # Test in Low anxiety
            low_result = {'anxiety': 'Low', 'depression': dep, 'stress': stress,
                          'outcome': outcome, 'n': np.nan, 'beta': np.nan, 'pval': np.nan}

            if len(low_anx_data) >= 10:
                try:
                    model = ols(f"{outcome} ~ z_ucla * C(gender_male)", data=low_anx_data).fit()
                    if "z_ucla:C(gender_male)[T.1]" in model.params:
                        low_result['n'] = len(low_anx_data)
                        low_result['beta'] = model.params["z_ucla:C(gender_male)[T.1]"]
                        low_result['pval'] = model.pvalues["z_ucla:C(gender_male)[T.1]"]
                except:
                    pass

            # Test in High anxiety
            high_result = {'anxiety': 'High', 'depression': dep, 'stress': stress,
                           'outcome': outcome, 'n': np.nan, 'beta': np.nan, 'pval': np.nan}

            if len(high_anx_data) >= 10:
                try:
                    model = ols(f"{outcome} ~ z_ucla * C(gender_male)", data=high_anx_data).fit()
                    if "z_ucla:C(gender_male)[T.1]" in model.params:
                        high_result['n'] = len(high_anx_data)
                        high_result['beta'] = model.params["z_ucla:C(gender_male)[T.1]"]
                        high_result['pval'] = model.pvalues["z_ucla:C(gender_male)[T.1]"]
                except:
                    pass

            buffering_results.append(low_result)
            buffering_results.append(high_result)

            # Compare
            if not np.isnan(low_result['pval']) and not np.isnan(high_result['pval']):
                buffering = "YES" if (low_result['pval'] < 0.05 and high_result['pval'] > 0.10) else "NO"
                print(f"  {dep} Dep × {stress} Stress | {outcome}:")
                print(f"    Low Anx: β={low_result['beta']:.3f}, p={low_result['pval']:.4f}")
                print(f"    High Anx: β={high_result['beta']:.3f}, p={high_result['pval']:.4f}")
                print(f"    Buffering: {buffering}")

buffering_df = pd.DataFrame(buffering_results)
buffering_df.to_csv(OUTPUT_DIR / "anxiety_buffering_test.csv", index=False, encoding='utf-8-sig')
print()
print(f"✓ Saved: anxiety_buffering_test.csv")
print()

# ============================================================================
# VULNERABILITY MAP VISUALIZATION
# ============================================================================

print("[5/5] Creating vulnerability map...")

# Focus on PE rate for visualization
pe_results = strat_df[strat_df['outcome'] == 'pe_rate'].copy()

# Create a heatmap for each Depression level (3 separate heatmaps)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, dep_level in enumerate(['Low', 'Medium', 'High']):
    ax = axes[i]

    # Filter to this depression level
    dep_data = pe_results[pe_results['depression'] == dep_level].copy()

    # Create pivot table (Anxiety × Stress)
    pivot_beta = dep_data.pivot_table(
        values='interaction_beta',
        index='anxiety',
        columns='stress',
        aggfunc='mean'
    )

    # Reorder to Low, Medium, High
    pivot_beta = pivot_beta.reindex(['Low', 'Medium', 'High'], axis=0)
    pivot_beta = pivot_beta.reindex(['Low', 'Medium', 'High'], axis=1)

    # Heatmap
    sns.heatmap(pivot_beta, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-5, vmax=5, cbar_kws={'label': 'Interaction β'},
                ax=ax, linewidths=1, linecolor='black')

    ax.set_title(f'{dep_level} Depression', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stress Level', fontsize=12)
    ax.set_ylabel('Anxiety Level', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vulnerability_map_3way.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: vulnerability_map_3way.png")
print()

# Create p-value heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, dep_level in enumerate(['Low', 'Medium', 'High']):
    ax = axes[i]

    dep_data = pe_results[pe_results['depression'] == dep_level].copy()

    pivot_pval = dep_data.pivot_table(
        values='interaction_pval',
        index='anxiety',
        columns='stress',
        aggfunc='mean'
    )

    pivot_pval = pivot_pval.reindex(['Low', 'Medium', 'High'], axis=0)
    pivot_pval = pivot_pval.reindex(['Low', 'Medium', 'High'], axis=1)

    # Significance mask
    significance = pivot_pval.map(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')

    sns.heatmap(pivot_pval, annot=significance, fmt='', cmap='RdYlGn',
                vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'},
                ax=ax, linewidths=1, linecolor='black')

    ax.set_title(f'{dep_level} Depression (p-values)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stress Level', fontsize=12)
    ax.set_ylabel('Anxiety Level', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vulnerability_map_pvalues.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: vulnerability_map_pvalues.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("DASS 3-WAY STRATIFICATION COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - dass_3way_stratification.csv (all 27 cells)")
print("  - anxiety_buffering_test.csv (Low vs High anxiety comparison)")
print("  - vulnerability_map_3way.png (effect sizes)")
print("  - vulnerability_map_pvalues.png (significance)")
print()

# Summary statistics
sig_cells = strat_df[strat_df['interaction_pval'] < 0.05]
print(f"Significant cells (p<0.05): {len(sig_cells)}/{len(strat_df)}")

if 'q_value' in strat_df.columns:
    sig_fdr = strat_df[strat_df['q_value'].notna() & (strat_df['q_value'] < 0.05)]
    print(f"FDR q<0.05 cells: {len(sig_fdr)}/{len(strat_df)}")
print()

if len(sig_cells) > 0:
    print("Significant cells:")
    for _, row in sig_cells.iterrows():
        print(f"  {row['depression']} Dep × {row['anxiety']} Anx × {row['stress']} Stress | {row['outcome']}: β={row['interaction_beta']:.3f}, p={row['interaction_pval']:.4f}")
    print()

# Anxiety buffering summary
low_anx_sig = buffering_df[(buffering_df['anxiety'] == 'Low') & (buffering_df['pval'] < 0.05)]
high_anx_sig = buffering_df[(buffering_df['anxiety'] == 'High') & (buffering_df['pval'] < 0.05)]

print(f"Low Anxiety significant effects: {len(low_anx_sig)}")
print(f"High Anxiety significant effects: {len(high_anx_sig)}")
print()

if len(low_anx_sig) > len(high_anx_sig):
    print("*** ANXIETY BUFFERING CONFIRMED ***")
    print("Effect is stronger in Low Anxiety contexts, weaker/absent in High Anxiety.")
    print("Interpretation: Anxiety may trigger compensatory cognitive control.")
else:
    print("ANXIETY BUFFERING NOT CONFIRMED")
    print("Effect is present across anxiety levels.")

print()
