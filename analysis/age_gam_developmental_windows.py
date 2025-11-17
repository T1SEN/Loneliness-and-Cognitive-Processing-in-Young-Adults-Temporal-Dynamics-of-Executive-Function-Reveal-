"""
age_gam_developmental_windows.py

Tests age as continuous predictor using polynomial terms (GAM alternative).
Examines UCLA slope variation across age (developmental windows).

Outputs:
- age_polynomial_results.csv
- ucla_slope_by_age.png
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_comprehensive/age_gam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("AGE GAM / DEVELOPMENTAL WINDOWS ANALYSIS")
print("=" * 80)
print()

# Load data
master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

if 'gender' not in master.columns:
    master = master.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')

gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Mean-center age
master['age_mc'] = master['age'] - master['age'].mean()
master['age_mc2'] = master['age_mc'] ** 2
master['age_mc3'] = master['age_mc'] ** 3

# Standardize
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_dass_dep'] = scaler.fit_transform(master[['dass_depression']])
master['z_dass_anx'] = scaler.fit_transform(master[['dass_anxiety']])
master['z_dass_str'] = scaler.fit_transform(master[['dass_stress']])

print(f"Sample: N={len(master)}")
print(f"Age range: {master['age'].min():.0f}-{master['age'].max():.0f}")
print()

# ============================================================================
# POLYNOMIAL REGRESSION: Age × UCLA × Gender
# ============================================================================

print("POLYNOMIAL REGRESSION: PE ~ Age + Age² + UCLA × Gender × Age")
print("-" * 80)

formula_poly = ("perseverative_error_rate ~ age_mc + age_mc2 + "
                "z_ucla * C(gender_male) * age_mc + "
                "z_dass_dep + z_dass_anx + z_dass_str")

model_poly = smf.ols(formula_poly, data=master).fit()
print(model_poly.summary())
print()

# Save results
poly_results = pd.DataFrame({
    'parameter': model_poly.params.index,
    'coefficient': model_poly.params.values,
    'p_value': model_poly.pvalues.values
})
poly_results.to_csv(OUTPUT_DIR / "age_polynomial_results.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'age_polynomial_results.csv'}")
print()

# ============================================================================
# VISUALIZATION: UCLA Slope by Age (continuous)
# ============================================================================

print("Creating age-specific UCLA slope plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Male
ages_plot = np.arange(18, 37, 1)
male_slopes = []

for age in ages_plot:
    age_centered = age - master['age'].mean()

    # Marginal effect of UCLA at this age
    # Coefficient: z_ucla + z_ucla:age_mc * age_centered
    if 'z_ucla:age_mc' in model_poly.params.index:
        slope = model_poly.params['z_ucla'] + model_poly.params['z_ucla:age_mc'] * age_centered
        if 'z_ucla:C(gender_male)[T.1]:age_mc' in model_poly.params.index:
            slope += model_poly.params['z_ucla:C(gender_male)[T.1]:age_mc'] * age_centered
    else:
        slope = model_poly.params.get('z_ucla', 0)

    male_slopes.append(slope)

axes[0].plot(ages_plot, male_slopes, linewidth=3, color='steelblue')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].axvline(19, color='red', linestyle=':', alpha=0.5, label='Age 19 (Q1 cutoff)')
axes[0].set_xlabel('Age (years)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('UCLA → WCST PE Slope', fontsize=12, fontweight='bold')
axes[0].set_title('Male: UCLA Slope Across Age', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend()

# Female
female_slopes = []
for age in ages_plot:
    age_centered = age - master['age'].mean()

    if 'z_ucla:age_mc' in model_poly.params.index:
        slope = model_poly.params['z_ucla'] + model_poly.params['z_ucla:age_mc'] * age_centered
    else:
        slope = model_poly.params.get('z_ucla', 0)

    female_slopes.append(slope)

axes[1].plot(ages_plot, female_slopes, linewidth=3, color='coral')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[1].set_xlabel('Age (years)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('UCLA → WCST PE Slope', fontsize=12, fontweight='bold')
axes[1].set_title('Female: UCLA Slope Across Age', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ucla_slope_by_age.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'ucla_slope_by_age.png'}")
plt.close()

print()
print("=" * 80)
print("AGE GAM ANALYSIS COMPLETE")
print("=" * 80)
print()
