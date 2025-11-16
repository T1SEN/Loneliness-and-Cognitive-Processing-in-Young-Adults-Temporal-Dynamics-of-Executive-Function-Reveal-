"""
DASS Component 3-Way Interactions Analysis
===========================================
Tests UCLA × Gender × DASS component interactions on WCST PE rate

Separate models for each DASS component:
1. PE ~ UCLA × Gender × Depression
2. PE ~ UCLA × Gender × Anxiety
3. PE ~ UCLA × Gender × Stress

Compares effect sizes to identify which DASS component shows:
- Strongest buffering effect (protective hypervigilance hypothesis)
- Strongest amplification effect (anhedonia/depletion hypothesis)

Method: OLS regression with median splits for stratification
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/dass_components")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("DASS COMPONENT 3-WAY INTERACTIONS ANALYSIS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading data...")

# Participant info and surveys
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
surveys = surveys.rename(columns={'participantId': 'participant_id'})

# UCLA scores
if 'surveyName' in surveys.columns:
    ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
elif 'survey' in surveys.columns:
    ucla_data = surveys[surveys['survey'].str.lower() == 'ucla'].copy()
else:
    raise KeyError("No survey name column found")
ucla_scores = ucla_data.groupby('participant_id')['score'].sum().reset_index()
ucla_scores.columns = ['participant_id', 'ucla_total']

# DASS scores - separate components
if 'surveyName' in surveys.columns:
    dass_data = surveys[surveys['surveyName'].str.lower().str.contains('dass')].copy()
elif 'survey' in surveys.columns:
    dass_data = surveys[surveys['survey'].str.lower().str.contains('dass')].copy()
else:
    dass_data = pd.DataFrame()
dass_scores = dass_data.groupby(['participant_id', 'questionText'])['score'].sum().unstack(fill_value=0)

# Map DASS subscales
dep_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ['meaningless', 'nothing', 'enthused', 'worth', 'positive', 'initiative', 'future'])]
anx_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ['breathing', 'trembling', 'worried', 'panic', 'heart', 'scared', 'dry'])]
stress_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ['wind down', 'over-react', 'nervous', 'agitated', 'relax', 'intolerant', 'touchy'])]

dass_summary = pd.DataFrame()
dass_summary['participant_id'] = dass_scores.index
dass_summary['dass_depression'] = dass_scores[dep_items].sum(axis=1).values if dep_items else 0
dass_summary['dass_anxiety'] = dass_scores[anx_items].sum(axis=1).values if anx_items else 0
dass_summary['dass_stress'] = dass_scores[stress_items].sum(axis=1).values if stress_items else 0
dass_summary = dass_summary.reset_index(drop=True)

# Merge with demographics
master = participants[['participant_id', 'age', 'gender', 'education']].merge(
    ucla_scores, on='participant_id', how='inner'
).merge(
    dass_summary, on='participant_id', how='left'
)

# WCST data for PE rate
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
if 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

import ast
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_dict'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

wcst_summary = wcst_trials.groupby('participant_id').agg(
    pe_count=('is_pe', 'sum'),
    total_trials=('is_pe', 'count'),
    wcst_accuracy=('correct', lambda x: (x.sum() / len(x)) * 100)
).reset_index()
wcst_summary['pe_rate'] = (wcst_summary['pe_count'] / wcst_summary['total_trials']) * 100

# Merge WCST
master = master.merge(wcst_summary[['participant_id', 'pe_rate', 'wcst_accuracy']],
                      on='participant_id', how='left')

# Create gender dummy
master['gender_male'] = (master['gender'].str.lower() == 'male').astype(int)

# Standardize predictors
master['z_ucla'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()
master['z_depression'] = (master['dass_depression'] - master['dass_depression'].mean()) / master['dass_depression'].std()
master['z_anxiety'] = (master['dass_anxiety'] - master['dass_anxiety'].mean()) / master['dass_anxiety'].std()
master['z_stress'] = (master['dass_stress'] - master['dass_stress'].mean()) / master['dass_stress'].std()
master['z_age'] = (master['age'] - master['age'].mean()) / master['age'].std()

# Drop missing
master = master.dropna(subset=['ucla_total', 'gender_male', 'pe_rate',
                                'dass_depression', 'dass_anxiety', 'dass_stress']).copy()

print(f"  Loaded {len(master)} participants with complete data")
print(f"  Males: {master['gender_male'].sum()}, Females: {(1-master['gender_male']).sum()}\n")

# ============================================================================
# 3-Way Interaction Models
# ============================================================================

print("=" * 80)
print("3-WAY INTERACTION MODELS")
print("=" * 80)

results_3way = []

# --------------------------------------------------
# Model 1: UCLA × Gender × Depression
# --------------------------------------------------
print("\n\n1. UCLA × Gender × Depression")
print("   " + "=" * 60)

formula = "pe_rate ~ z_ucla * gender_male * z_depression + z_age"
model = smf.ols(formula, data=master).fit()

print(model.summary())

# Extract key coefficients
coef_3way_dep = model.params.get('z_ucla:gender_male:z_depression', np.nan)
p_3way_dep = model.pvalues.get('z_ucla:gender_male:z_depression', np.nan)

results_3way.append({
    'model': 'Depression',
    'formula': formula,
    'n': int(model.nobs),
    'r_squared': model.rsquared,
    'adj_r_squared': model.rsquared_adj,
    'f_statistic': model.fvalue,
    'f_pvalue': model.f_pvalue,
    '3way_coef': coef_3way_dep,
    '3way_pvalue': p_3way_dep,
    'ucla_gender_coef': model.params.get('z_ucla:gender_male', np.nan),
    'ucla_gender_pvalue': model.pvalues.get('z_ucla:gender_male', np.nan)
})

# --------------------------------------------------
# Model 2: UCLA × Gender × Anxiety
# --------------------------------------------------
print("\n\n2. UCLA × Gender × Anxiety")
print("   " + "=" * 60)

formula = "pe_rate ~ z_ucla * gender_male * z_anxiety + z_age"
model = smf.ols(formula, data=master).fit()

print(model.summary())

coef_3way_anx = model.params.get('z_ucla:gender_male:z_anxiety', np.nan)
p_3way_anx = model.pvalues.get('z_ucla:gender_male:z_anxiety', np.nan)

results_3way.append({
    'model': 'Anxiety',
    'formula': formula,
    'n': int(model.nobs),
    'r_squared': model.rsquared,
    'adj_r_squared': model.rsquared_adj,
    'f_statistic': model.fvalue,
    'f_pvalue': model.f_pvalue,
    '3way_coef': coef_3way_anx,
    '3way_pvalue': p_3way_anx,
    'ucla_gender_coef': model.params.get('z_ucla:gender_male', np.nan),
    'ucla_gender_pvalue': model.pvalues.get('z_ucla:gender_male', np.nan)
})

# --------------------------------------------------
# Model 3: UCLA × Gender × Stress
# --------------------------------------------------
print("\n\n3. UCLA × Gender × Stress")
print("   " + "=" * 60)

formula = "pe_rate ~ z_ucla * gender_male * z_stress + z_age"
model = smf.ols(formula, data=master).fit()

print(model.summary())

coef_3way_str = model.params.get('z_ucla:gender_male:z_stress', np.nan)
p_3way_str = model.pvalues.get('z_ucla:gender_male:z_stress', np.nan)

results_3way.append({
    'model': 'Stress',
    'formula': formula,
    'n': int(model.nobs),
    'r_squared': model.rsquared,
    'adj_r_squared': model.rsquared_adj,
    'f_statistic': model.fvalue,
    'f_pvalue': model.f_pvalue,
    '3way_coef': coef_3way_str,
    '3way_pvalue': p_3way_str,
    'ucla_gender_coef': model.params.get('z_ucla:gender_male', np.nan),
    'ucla_gender_pvalue': model.pvalues.get('z_ucla:gender_male', np.nan)
})

results_3way_df = pd.DataFrame(results_3way)

# ============================================================================
# Stratification Analysis
# ============================================================================

print("\n\n" + "=" * 80)
print("STRATIFICATION ANALYSIS (Median Splits)")
print("=" * 80)

stratification_results = []

for component in ['depression', 'anxiety', 'stress']:
    print(f"\n\n{component.upper()} Stratification")
    print("   " + "=" * 60)

    # Median split
    median_val = master[f'dass_{component}'].median()
    master[f'{component}_high'] = (master[f'dass_{component}'] > median_val).astype(int)

    for strata_level in [0, 1]:
        strata_name = 'High' if strata_level == 1 else 'Low'
        df_strata = master[master[f'{component}_high'] == strata_level].copy()

        print(f"\n   {strata_name} {component.capitalize()} (N={len(df_strata)}):")

        # Separate by gender
        for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
            df_sub = df_strata[df_strata['gender_male'] == gender_val].copy()

            if len(df_sub) < 5:
                continue

            # UCLA → PE correlation
            r, p = stats.pearsonr(df_sub['ucla_total'], df_sub['pe_rate'])

            # OLS: PE ~ UCLA
            formula = "pe_rate ~ z_ucla + z_age"
            model = smf.ols(formula, data=df_sub).fit()
            beta_ucla = model.params.get('z_ucla', np.nan)
            p_ucla = model.pvalues.get('z_ucla', np.nan)

            print(f"      {gender_name}: r = {r:.3f}, p = {p:.4f}, β = {beta_ucla:.3f}, p = {p_ucla:.4f}")

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

# ============================================================================
# Compare Effect Sizes
# ============================================================================

print("\n\n" + "=" * 80)
print("EFFECT SIZE COMPARISON ACROSS DASS COMPONENTS")
print("=" * 80)

# Compare 3-way interaction coefficients
print("\n3-Way Interaction Coefficients (UCLA × Gender × DASS component):")
print(results_3way_df[['model', '3way_coef', '3way_pvalue']].to_string(index=False))

# Identify strongest buffering effect (negative 3-way interaction)
strongest_buffer = results_3way_df.loc[results_3way_df['3way_coef'].idxmin()]
print(f"\nStrongest buffering (most negative 3-way): {strongest_buffer['model']}")
print(f"  Coefficient: {strongest_buffer['3way_coef']:.3f}, p = {strongest_buffer['3way_pvalue']:.4f}")

# Identify strongest amplification (positive 3-way interaction)
strongest_amplify = results_3way_df.loc[results_3way_df['3way_coef'].idxmax()]
print(f"\nStrongest amplification (most positive 3-way): {strongest_amplify['model']}")
print(f"  Coefficient: {strongest_amplify['3way_coef']:.3f}, p = {strongest_amplify['3way_pvalue']:.4f}")

# Stratification comparison: Which component shows biggest effect in Low vs High strata?
print("\n\nStratification Effect Sizes (Males, Low vs High):")
for component in ['Depression', 'Anxiety', 'Stress']:
    low_male = stratification_df[
        (stratification_df['component'] == component) &
        (stratification_df['strata'] == 'Low') &
        (stratification_df['gender'] == 'Male')
    ]
    high_male = stratification_df[
        (stratification_df['component'] == component) &
        (stratification_df['strata'] == 'High') &
        (stratification_df['gender'] == 'Male')
    ]

    if len(low_male) > 0 and len(high_male) > 0:
        low_beta = low_male.iloc[0]['ucla_beta']
        high_beta = high_male.iloc[0]['ucla_beta']
        buffering_index = low_beta - high_beta  # Positive = buffering (effect stronger in Low)

        print(f"\n  {component}:")
        print(f"    Low {component} β: {low_beta:.3f}")
        print(f"    High {component} β: {high_beta:.3f}")
        print(f"    Buffering index (Low - High): {buffering_index:.3f}")

# ============================================================================
# Save Results
# ============================================================================

print("\n\nSaving results...")

# 3-way interaction results
threeway_file = OUTPUT_DIR / "dass_component_3way_models.csv"
results_3way_df.to_csv(threeway_file, index=False, encoding='utf-8-sig')
print(f"  3-way models: {threeway_file}")

# Stratification results
strat_file = OUTPUT_DIR / "dass_component_stratification.csv"
stratification_df.to_csv(strat_file, index=False, encoding='utf-8-sig')
print(f"  Stratification: {strat_file}")

# Summary
summary_text = f"""
DASS COMPONENT 3-WAY INTERACTION SUMMARY
========================================

Sample: N = {len(master)} ({master['gender_male'].sum()} males, {(1-master['gender_male']).sum()} females)

3-WAY INTERACTION EFFECTS (UCLA × Gender × DASS component):
-----------------------------------------------------------
Depression:  β = {coef_3way_dep:.3f}, p = {p_3way_dep:.4f}
Anxiety:     β = {coef_3way_anx:.3f}, p = {p_3way_anx:.4f}
Stress:      β = {coef_3way_str:.3f}, p = {p_3way_str:.4f}

INTERPRETATION:
---------------
Strongest buffering: {strongest_buffer['model']} (β = {strongest_buffer['3way_coef']:.3f})
  → High {strongest_buffer['model'].lower()} may protect against UCLA×Gender effect

Strongest amplification: {strongest_amplify['model']} (β = {strongest_amplify['3way_coef']:.3f})
  → High {strongest_amplify['model'].lower()} may worsen UCLA×Gender effect

MECHANISM HYPOTHESES:
---------------------
- Anxiety buffering: Hypervigilance compensates for loneliness-induced lapses
- Depression amplification: Anhedonia reduces motivation to adapt/learn
- Stress effects: Context-dependent (may buffer via arousal or amplify via depletion)

"""

summary_file = OUTPUT_DIR / "DASS_COMPONENT_SUMMARY.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"  Summary: {summary_file}")

print("\n" + "=" * 80)
print("DASS COMPONENT 3-WAY INTERACTIONS ANALYSIS COMPLETE")
print("=" * 80)
