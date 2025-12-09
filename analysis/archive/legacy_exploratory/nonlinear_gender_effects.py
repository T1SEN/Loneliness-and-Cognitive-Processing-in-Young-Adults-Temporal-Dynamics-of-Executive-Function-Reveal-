"""
Nonlinear Gender × Loneliness Effects Analysis
===============================================

Tests quadratic, cubic, and threshold effects in the loneliness-WCST relationship,
stratified by gender.

Analyses:
1. Quadratic UCLA × Gender interactions
2. Extreme groups comparison
3. UCLA tertile analyses
4. Spline regression (nonparametric)

Author: Research Team
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/nonlinear_effects")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("NONLINEAR GENDER × LONELINESS EFFECTS ANALYSIS")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/4] Loading data...")

master = load_master_dataset(use_cache=True)

# Load age and gender from participants
master = load_master_dataset(use_cache=True)
participants = master[['participant_id','gender_normalized','age']].rename(columns={'gender_normalized':'gender'})
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

# Merge with demographics
master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

# Handle Korean gender normalization
def normalize_gender(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip().lower()
    if '남' in val_str or val_str in ['m', 'male']:
        return 'male'
    elif '여' in val_str or val_str in ['f', 'female']:
        return 'female'
    return None

master['gender_normalized'] = master['gender'].apply(normalize_gender)
master['gender_male'] = (master['gender_normalized'] == 'male').astype(int)

# Filter complete cases (including DASS)
master = master.dropna(subset=['ucla_total', 'pe_rate', 'gender_male',
                                'dass_depression', 'dass_anxiety', 'dass_stress', 'age']).copy()

print(f"  N={len(master)} ({(master['gender_male']==0).sum()} Female, {(master['gender_male']==1).sum()} Male)")
print(f"  (with complete DASS + age data for covariate control)")
print()

# Standardize
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_ucla_sq'] = master['z_ucla'] ** 2
master['z_dass_dep'] = scaler.fit_transform(master[['dass_depression']])
master['z_dass_anx'] = scaler.fit_transform(master[['dass_anxiety']])
master['z_dass_str'] = scaler.fit_transform(master[['dass_stress']])
master['z_age'] = scaler.fit_transform(master[['age']])

# ============================================================================
# ANALYSIS 1: QUADRATIC UCLA × GENDER
# ============================================================================

print("[2/4] Testing quadratic effects...")

quadratic_results = []

# Main WCST metrics
outcomes = ['pe_rate', 'wcst_accuracy', 'wcst_npe_rate']

for outcome in outcomes:
    if outcome not in master.columns or master[outcome].notna().sum() < 40:
        continue

    # Linear model (DASS-controlled)
    formula_linear = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_linear = ols(formula_linear, data=master.dropna(subset=[outcome])).fit()

    # Quadratic model (DASS-controlled)
    formula_quad = f"{outcome} ~ (z_ucla + z_ucla_sq) * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_quad = ols(formula_quad, data=master.dropna(subset=[outcome])).fit()

    # Extract coefficients
    linear_int = model_linear.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
    linear_p = model_linear.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)

    quad_int_linear = model_quad.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
    quad_int_sq = model_quad.params.get('z_ucla_sq:C(gender_male)[T.1]', np.nan)
    quad_p_sq = model_quad.pvalues.get('z_ucla_sq:C(gender_male)[T.1]', np.nan)

    # Model comparison
    aic_linear = model_linear.aic
    aic_quad = model_quad.aic
    delta_aic = aic_linear - aic_quad

    quadratic_results.append({
        'outcome': outcome,
        'linear_interaction_beta': linear_int,
        'linear_interaction_p': linear_p,
        'quad_linear_interaction_beta': quad_int_linear,
        'quad_sq_interaction_beta': quad_int_sq,
        'quad_sq_interaction_p': quad_p_sq,
        'aic_linear': aic_linear,
        'aic_quadratic': aic_quad,
        'delta_aic': delta_aic,
        'quadratic_preferred': delta_aic > 2
    })

    print(f"  {outcome}:")
    print(f"    Linear interaction: β={linear_int:.3f}, p={linear_p:.4f}")
    print(f"    Quadratic interaction: β={quad_int_sq:.3f}, p={quad_p_sq:.4f}")
    print(f"    ΔAIC={delta_aic:.2f} ({'Quadratic better' if delta_aic > 2 else 'Linear better'})")

quadratic_df = pd.DataFrame(quadratic_results)
quadratic_df.to_csv(OUTPUT_DIR / "quadratic_effects.csv", index=False, encoding='utf-8-sig')
print()

# ============================================================================
# ANALYSIS 2: EXTREME GROUPS COMPARISON
# ============================================================================

print("[3/4] Extreme groups comparison...")

# Bottom and top tertiles
ucla_33 = master['ucla_total'].quantile(0.33)
ucla_67 = master['ucla_total'].quantile(0.67)

low_ucla = master[master['ucla_total'] <= ucla_33].copy()
high_ucla = master[master['ucla_total'] >= ucla_67].copy()

extreme_groups = pd.concat([
    low_ucla.assign(ucla_group='Low'),
    high_ucla.assign(ucla_group='High')
])

extreme_results = []

for outcome in ['pe_rate', 'wcst_accuracy']:
    if outcome not in extreme_groups.columns:
        continue

    # 4 groups: Low-Female, Low-Male, High-Female, High-Male
    groups = {
        'Low_Female': extreme_groups[(extreme_groups['ucla_group']=='Low') & (extreme_groups['gender_male']==0)][outcome].dropna(),
        'Low_Male': extreme_groups[(extreme_groups['ucla_group']=='Low') & (extreme_groups['gender_male']==1)][outcome].dropna(),
        'High_Female': extreme_groups[(extreme_groups['ucla_group']=='High') & (extreme_groups['gender_male']==0)][outcome].dropna(),
        'High_Male': extreme_groups[(extreme_groups['ucla_group']=='High') & (extreme_groups['gender_male']==1)][outcome].dropna()
    }

    # Calculate means and SDs
    stats_dict = {}
    for grp_name, grp_data in groups.items():
        if len(grp_data) > 0:
            stats_dict[f'{grp_name}_mean'] = grp_data.mean()
            stats_dict[f'{grp_name}_sd'] = grp_data.std()
            stats_dict[f'{grp_name}_n'] = len(grp_data)
        else:
            stats_dict[f'{grp_name}_mean'] = np.nan
            stats_dict[f'{grp_name}_sd'] = np.nan
            stats_dict[f'{grp_name}_n'] = 0

    # Effect sizes
    # Male: High vs Low
    if len(groups['High_Male']) > 2 and len(groups['Low_Male']) > 2:
        pooled_sd = np.sqrt(((len(groups['High_Male'])-1)*groups['High_Male'].var() +
                              (len(groups['Low_Male'])-1)*groups['Low_Male'].var()) /
                             (len(groups['High_Male']) + len(groups['Low_Male']) - 2))
        cohen_d_male = (groups['High_Male'].mean() - groups['Low_Male'].mean()) / pooled_sd if pooled_sd > 0 else np.nan

        # Bootstrap CI for Cohen's d
        boot_ds = []
        for _ in range(1000):
            boot_high = resample(groups['High_Male'])
            boot_low = resample(groups['Low_Male'])
            boot_pooled_sd = np.sqrt(((len(boot_high)-1)*boot_high.var() +
                                       (len(boot_low)-1)*boot_low.var()) /
                                      (len(boot_high) + len(boot_low) - 2))
            if boot_pooled_sd > 0:
                boot_ds.append((boot_high.mean() - boot_low.mean()) / boot_pooled_sd)

        if len(boot_ds) > 0:
            d_ci_lower = np.percentile(boot_ds, 2.5)
            d_ci_upper = np.percentile(boot_ds, 97.5)
        else:
            d_ci_lower, d_ci_upper = np.nan, np.nan
    else:
        cohen_d_male = np.nan
        d_ci_lower, d_ci_upper = np.nan, np.nan

    # Female: High vs Low
    if len(groups['High_Female']) > 2 and len(groups['Low_Female']) > 2:
        pooled_sd_f = np.sqrt(((len(groups['High_Female'])-1)*groups['High_Female'].var() +
                                (len(groups['Low_Female'])-1)*groups['Low_Female'].var()) /
                               (len(groups['High_Female']) + len(groups['Low_Female']) - 2))
        cohen_d_female = (groups['High_Female'].mean() - groups['Low_Female'].mean()) / pooled_sd_f if pooled_sd_f > 0 else np.nan
    else:
        cohen_d_female = np.nan

    stats_dict.update({
        'outcome': outcome,
        'cohen_d_male_high_vs_low': cohen_d_male,
        'cohen_d_male_95ci_lower': d_ci_lower,
        'cohen_d_male_95ci_upper': d_ci_upper,
        'cohen_d_female_high_vs_low': cohen_d_female
    })

    extreme_results.append(stats_dict)

    print(f"  {outcome}:")
    print(f"    Male (High-Low): d={cohen_d_male:.3f}, 95%CI=[{d_ci_lower:.3f}, {d_ci_upper:.3f}]")
    print(f"    Female (High-Low): d={cohen_d_female:.3f}")

extreme_df = pd.DataFrame(extreme_results)
extreme_df.to_csv(OUTPUT_DIR / "extreme_groups_effect_sizes.csv", index=False, encoding='utf-8-sig')
print()

# ============================================================================
# ANALYSIS 3: UCLA TERTILE ANOVA
# ============================================================================

print("[4/4] UCLA tertile ANOVA...")

# Create tertiles
master['ucla_tertile'] = pd.qcut(master['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

tertile_results = []

for outcome in ['pe_rate', 'wcst_accuracy']:
    if outcome not in master.columns or master[outcome].notna().sum() < 40:
        continue

    # 2-way ANOVA: Tertile × Gender (DASS-controlled)
    formula = f"{outcome} ~ C(ucla_tertile) * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    try:
        model = ols(formula, data=master.dropna(subset=[outcome])).fit()

        # ANOVA table
        from statsmodels.stats.anova import anova_lm
        anova_table = anova_lm(model, typ=2)

        tertile_main = anova_table.loc['C(ucla_tertile)', 'PR(>F)']
        gender_main = anova_table.loc['C(gender_male)', 'PR(>F)']
        interaction = anova_table.loc['C(ucla_tertile):C(gender_male)', 'PR(>F)']

        tertile_results.append({
            'outcome': outcome,
            'tertile_main_p': tertile_main,
            'gender_main_p': gender_main,
            'interaction_p': interaction,
            'model_rsquared': model.rsquared
        })

        print(f"  {outcome}:")
        print(f"    Tertile main: p={tertile_main:.4f}")
        print(f"    Gender main: p={gender_main:.4f}")
        print(f"    Interaction: p={interaction:.4f}")

    except:
        print(f"  {outcome}: Could not fit model")

tertile_df = pd.DataFrame(tertile_results)
tertile_df.to_csv(OUTPUT_DIR / "tertile_anova.csv", index=False, encoding='utf-8-sig')
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("NONLINEAR EFFECTS ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - quadratic_effects.csv")
print("  - extreme_groups_effect_sizes.csv")
print("  - tertile_anova.csv")
print()
print("Key findings:")
quad_preferred = quadratic_df['quadratic_preferred'].sum()
print(f"  - Quadratic model preferred for {quad_preferred}/{len(quadratic_df)} outcomes")
print(f"  - Extreme groups analysis: {len(extreme_df)} outcomes")
print(f"  - Tertile ANOVA: {len(tertile_df)} outcomes")
print()
