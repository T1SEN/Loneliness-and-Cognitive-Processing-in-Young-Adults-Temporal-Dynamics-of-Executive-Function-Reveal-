"""
DASS Component 3-Way Interactions Analysis
Tests UCLA × Gender × (Depression/Anxiety/Stress) on WCST PE
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

from data_loader_utils import load_master_dataset

OUTPUT_DIR = Path("results/analysis_outputs/dass_components")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("DASS COMPONENT 3-WAY INTERACTIONS")
print("=" * 80)

# Load data
print("\nLoading data...")
master = load_master_dataset()
master = master.dropna(subset=['ucla_total', 'gender_male', 'pe_rate',
                                'dass_depression', 'dass_anxiety', 'dass_stress']).copy()

print(f"  Loaded {len(master)} participants")
print(f"  Males: {master['gender_male'].sum()}, Females: {(1-master['gender_male']).sum()}\n")

# 3-Way Interaction Models
results_3way = []

print("=" * 80)
print("3-WAY INTERACTION MODELS")
print("=" * 80)

# Model 1: Depression
print("\n1. UCLA × Gender × Depression")
print("   " + "-" * 60)

formula = "pe_rate ~ z_ucla * gender_male * z_depression + z_age"
model = smf.ols(formula, data=master).fit()
print(model.summary())

results_3way.append({
    'component': 'Depression',
    'n': int(model.nobs),
    'r_squared': model.rsquared,
    '3way_coef': model.params.get('z_ucla:gender_male:z_depression', np.nan),
    '3way_pvalue': model.pvalues.get('z_ucla:gender_male:z_depression', np.nan),
    'ucla_gender_coef': model.params.get('z_ucla:gender_male', np.nan),
    'ucla_gender_pvalue': model.pvalues.get('z_ucla:gender_male', np.nan)
})

# Model 2: Anxiety
print("\n\n2. UCLA × Gender × Anxiety")
print("   " + "-" * 60)

formula = "pe_rate ~ z_ucla * gender_male * z_anxiety + z_age"
model = smf.ols(formula, data=master).fit()
print(model.summary())

results_3way.append({
    'component': 'Anxiety',
    'n': int(model.nobs),
    'r_squared': model.rsquared,
    '3way_coef': model.params.get('z_ucla:gender_male:z_anxiety', np.nan),
    '3way_pvalue': model.pvalues.get('z_ucla:gender_male:z_anxiety', np.nan),
    'ucla_gender_coef': model.params.get('z_ucla:gender_male', np.nan),
    'ucla_gender_pvalue': model.pvalues.get('z_ucla:gender_male', np.nan)
})

# Model 3: Stress
print("\n\n3. UCLA × Gender × Stress")
print("   " + "-" * 60)

formula = "pe_rate ~ z_ucla * gender_male * z_stress + z_age"
model = smf.ols(formula, data=master).fit()
print(model.summary())

results_3way.append({
    'component': 'Stress',
    'n': int(model.nobs),
    'r_squared': model.rsquared,
    '3way_coef': model.params.get('z_ucla:gender_male:z_stress', np.nan),
    '3way_pvalue': model.pvalues.get('z_ucla:gender_male:z_stress', np.nan),
    'ucla_gender_coef': model.params.get('z_ucla:gender_male', np.nan),
    'ucla_gender_pvalue': model.pvalues.get('z_ucla:gender_male', np.nan)
})

results_3way_df = pd.DataFrame(results_3way)

# Stratification Analysis
print("\n\n" + "=" * 80)
print("STRATIFICATION ANALYSIS")
print("=" * 80)

stratification_results = []

for component in ['depression', 'anxiety', 'stress']:
    print(f"\n{component.upper()} Stratification")
    print("   " + "-" * 60)

    col = f'dass_{component}'
    median_val = master[col].median()
    master[f'{component}_high'] = (master[col] > median_val).astype(int)

    for strata_level in [0, 1]:
        strata_name = 'High' if strata_level == 1 else 'Low'
        df_strata = master[master[f'{component}_high'] == strata_level].copy()

        print(f"\n   {strata_name} {component.capitalize()} (N={len(df_strata)}):")

        for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
            df_sub = df_strata[df_strata['gender_male'] == gender_val].copy()

            if len(df_sub) < 5:
                continue

            r, p = stats.pearsonr(df_sub['ucla_total'], df_sub['pe_rate'])

            formula = "pe_rate ~ z_ucla + z_age"
            model = smf.ols(formula, data=df_sub).fit()
            beta_ucla = model.params.get('z_ucla', np.nan)
            p_ucla = model.pvalues.get('z_ucla', np.nan)

            print(f"      {gender_name}: r = {r:.3f}, p = {p:.4f}, β = {beta_ucla:.3f}")

            stratification_results.append({
                'component': component.capitalize(),
                'strata': strata_name,
                'gender': gender_name,
                'n': len(df_sub),
                'ucla_pe_r': r,
                'ucla_pe_p': p,
                'ucla_beta': beta_ucla,
                'ucla_beta_p': p_ucla
            })

stratification_df = pd.DataFrame(stratification_results)

# Effect Size Comparison
print("\n\n" + "=" * 80)
print("EFFECT SIZE COMPARISON")
print("=" * 80)

print("\n3-Way Interaction Coefficients:")
print(results_3way_df[['component', '3way_coef', '3way_pvalue']].to_string(index=False))

if len(results_3way_df) > 0:
    strongest_buffer = results_3way_df.loc[results_3way_df['3way_coef'].idxmin()]
    strongest_amplify = results_3way_df.loc[results_3way_df['3way_coef'].idxmax()]

    print(f"\nStrongest buffering (most negative): {strongest_buffer['component']}")
    print(f"  β = {strongest_buffer['3way_coef']:.3f}, p = {strongest_buffer['3way_pvalue']:.4f}")

    print(f"\nStrongest amplification (most positive): {strongest_amplify['component']}")
    print(f"  β = {strongest_amplify['3way_coef']:.3f}, p = {strongest_amplify['3way_pvalue']:.4f}")

# Save Results
print("\n\nSaving results...")

threeway_file = OUTPUT_DIR / "dass_component_3way_models.csv"
results_3way_df.to_csv(threeway_file, index=False, encoding='utf-8-sig')
print(f"  {threeway_file}")

strat_file = OUTPUT_DIR / "dass_component_stratification.csv"
stratification_df.to_csv(strat_file, index=False, encoding='utf-8-sig')
print(f"  {strat_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
