"""
ucla_dass_moderation_commonality.py

Tests UCLA × DASS (Anxiety/Stress) continuous moderation effects
and performs variance partitioning (commonality analysis).

Outputs:
- moderation_results.csv
- commonality_variance_partition.csv
- simple_slopes_plot.png
"""

import sys
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_comprehensive/dass_moderation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("UCLA × DASS MODERATION & COMMONALITY ANALYSIS")
print("=" * 80)
print()

# Load data
master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()

# Load demographics
master = load_master_dataset(use_cache=True)
participants = master[['participant_id','gender_normalized','age']].rename(columns={'gender_normalized':'gender'})
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

if 'gender' not in master.columns:

gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Standardize
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_anx'] = scaler.fit_transform(master[['dass_anxiety']])
master['z_str'] = scaler.fit_transform(master[['dass_stress']])
master['z_dep'] = scaler.fit_transform(master[['dass_depression']])
master['z_age'] = scaler.fit_transform(master[['age']])

# Mean-center for interpretability
master['anx_mc'] = master['dass_anxiety'] - master['dass_anxiety'].mean()
master['str_mc'] = master['dass_stress'] - master['dass_stress'].mean()

print(f"Sample: N={len(master)}")
print()

# ============================================================================
# MODERATION ANALYSIS: UCLA × Anxiety, UCLA × Stress
# ============================================================================

print("MODERATION: WCST PE ~ UCLA × Anxiety + UCLA × Stress")
print("-" * 80)

formula_mod = ("perseverative_error_rate ~ z_ucla * z_anx + z_ucla * z_str + "
               "C(gender_male) + z_dep + z_age")

model_mod = smf.ols(formula_mod, data=master).fit()
print(model_mod.summary())
print()

# Save results
mod_results = pd.DataFrame({
    'parameter': model_mod.params.index,
    'coefficient': model_mod.params.values,
    'p_value': model_mod.pvalues.values
})
mod_results.to_csv(OUTPUT_DIR / "moderation_results.csv", index=False, encoding='utf-8-sig')

print(f"Saved: {OUTPUT_DIR / 'moderation_results.csv'}")
print()

# ============================================================================
# COMMONALITY ANALYSIS (Variance Partitioning)
# ============================================================================

print("=" * 80)
print("COMMONALITY ANALYSIS: UCLA vs DASS Contributions")
print("=" * 80)
print()

# Clean data
master_clean = master.dropna(subset=['perseverative_error_rate', 'z_ucla', 'z_dep', 'z_anx', 'z_str', 'gender_male'])

# Model 1: UCLA only
model_ucla_only = smf.ols("perseverative_error_rate ~ z_ucla + C(gender_male) + z_age", data=master_clean).fit()
r2_ucla_only = model_ucla_only.rsquared

# Model 2: DASS only
model_dass_only = smf.ols("perseverative_error_rate ~ z_dep + z_anx + z_str + C(gender_male) + z_age", data=master_clean).fit()
r2_dass_only = model_dass_only.rsquared

# Model 3: UCLA + DASS
model_full = smf.ols("perseverative_error_rate ~ z_ucla + z_dep + z_anx + z_str + C(gender_male) + z_age", data=master_clean).fit()
r2_full = model_full.rsquared

# Model 4: Covariates only (baseline)
model_cov = smf.ols("perseverative_error_rate ~ C(gender_male) + z_age", data=master_clean).fit()
r2_cov = model_cov.rsquared

# Compute unique and common variance
unique_ucla = r2_full - r2_dass_only
unique_dass = r2_full - r2_ucla_only
common = r2_ucla_only + r2_dass_only - r2_full

print(f"R² Covariates only: {r2_cov:.3f}")
print(f"R² UCLA only (+ cov): {r2_ucla_only:.3f}")
print(f"R² DASS only (+ cov): {r2_dass_only:.3f}")
print(f"R² Full (UCLA + DASS + cov): {r2_full:.3f}")
print()
print(f"Unique UCLA: {unique_ucla:.3f} ({unique_ucla/r2_full*100:.1f}%)")
print(f"Unique DASS: {unique_dass:.3f} ({unique_dass/r2_full*100:.1f}%)")
print(f"Common UCLA-DASS: {common:.3f} ({common/r2_full*100:.1f}%)")
print()

commonality_df = pd.DataFrame({
    'component': ['Unique UCLA', 'Unique DASS', 'Common UCLA-DASS', 'Covariates'],
    'r_squared': [unique_ucla, unique_dass, common, r2_cov],
    'percent': [unique_ucla/r2_full*100 if r2_full > 0 else 0,
                unique_dass/r2_full*100 if r2_full > 0 else 0,
                common/r2_full*100 if r2_full > 0 else 0,
                r2_cov/r2_full*100 if r2_full > 0 else 0]
})

commonality_df.to_csv(OUTPUT_DIR / "commonality_variance_partition.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'commonality_variance_partition.csv'}")
print()

# ============================================================================
# VISUALIZATION: Simple Slopes
# ============================================================================

print("Creating simple slopes plot...")

# Plot UCLA slope at Low/Mean/High Anxiety
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Anxiety moderation
anx_levels = [-1, 0, 1]  # Low, Mean, High (z-scores)
for anx_level in anx_levels:
    # Predicted values
    pred_df = pd.DataFrame({
        'z_ucla': np.linspace(master['z_ucla'].min(), master['z_ucla'].max(), 100),
        'z_anx': anx_level,
        'z_str': 0,
        'z_dep': 0,
        'gender_male': 0,
        'z_age': 0
    })

    pred_df['pred_pe'] = model_mod.predict(pred_df)

    label = f"Anxiety: {'Low' if anx_level < 0 else ('Mean' if anx_level == 0 else 'High')}"
    axes[0].plot(pred_df['z_ucla'], pred_df['pred_pe'], label=label, linewidth=2)

axes[0].set_xlabel('UCLA Loneliness (z-score)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted WCST PE Rate', fontsize=12, fontweight='bold')
axes[0].set_title('UCLA × Anxiety Interaction', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Stress moderation
str_levels = [-1, 0, 1]
for str_level in str_levels:
    pred_df = pd.DataFrame({
        'z_ucla': np.linspace(master['z_ucla'].min(), master['z_ucla'].max(), 100),
        'z_anx': 0,
        'z_str': str_level,
        'z_dep': 0,
        'gender_male': 0,
        'z_age': 0
    })

    pred_df['pred_pe'] = model_mod.predict(pred_df)

    label = f"Stress: {'Low' if str_level < 0 else ('Mean' if str_level == 0 else 'High')}"
    axes[1].plot(pred_df['z_ucla'], pred_df['pred_pe'], label=label, linewidth=2)

axes[1].set_xlabel('UCLA Loneliness (z-score)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted WCST PE Rate', fontsize=12, fontweight='bold')
axes[1].set_title('UCLA × Stress Interaction', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "simple_slopes_plot.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'simple_slopes_plot.png'}")
plt.close()

print()
print("=" * 80)
print("MODERATION & COMMONALITY ANALYSIS COMPLETE")
print("=" * 80)
print()
