#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 7: 시각화 및 진단 플롯
Diagnostic Visualizations for Gender Moderation Effect
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
VERIFICATION_DIR = OUTPUT_DIR / "gender_verification"
VERIFICATION_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Step 7: 시각화 및 진단 플롯")
print("Diagnostic Visualizations")
print("=" * 80)

# Load data
master = pd.read_csv(OUTPUT_DIR / "master_dataset.csv")
master = master.rename(columns={'pe_rate': 'perseverative_error_rate'})

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

master['gender_male'] = (master['gender'] == '남성').astype(int)
master_clean = master.dropna().copy()

# Standardize
scaler = StandardScaler()
master_clean['z_ucla'] = scaler.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_stress'] = scaler.fit_transform(master_clean[['dass_stress']])
master_clean['z_age'] = scaler.fit_transform(master_clean[['age']])

# =============================================================================
# 7.1 Scatter Plot with Regression Lines
# =============================================================================

print("\n" + "=" * 80)
print("7.1 산점도 + 회귀선")
print("=" * 80)

fig, ax = plt.subplots(figsize=(10, 6))

# Separate by gender
females = master_clean[master_clean['gender_male'] == 0]
males = master_clean[master_clean['gender_male'] == 1]

# Plot scatter
ax.scatter(females['ucla_total'], females['perseverative_error_rate'],
          alpha=0.6, s=80, color='#e74c3c', label=f'Female (N={len(females)})', edgecolors='black', linewidth=0.5)
ax.scatter(males['ucla_total'], males['perseverative_error_rate'],
          alpha=0.6, s=80, color='#3498db', label=f'Male (N={len(males)})', edgecolors='black', linewidth=0.5)

# Fit regression lines
x_range = np.linspace(master_clean['ucla_total'].min(), master_clean['ucla_total'].max(), 100)

# Female regression
female_fit = np.polyfit(females['ucla_total'], females['perseverative_error_rate'], 1)
female_line = np.poly1d(female_fit)
ax.plot(x_range, female_line(x_range), color='#e74c3c', linewidth=2.5, linestyle='--',
       label=f'Female: r = -0.25, p = 0.10')

# Male regression
male_fit = np.polyfit(males['ucla_total'], males['perseverative_error_rate'], 1)
male_line = np.poly1d(male_fit)
ax.plot(x_range, male_line(x_range), color='#3498db', linewidth=2.5, linestyle='--',
       label=f'Male: r = 0.24, p = 0.23')

ax.set_xlabel('UCLA Loneliness Score', fontsize=13, fontweight='bold')
ax.set_ylabel('WCST Perseverative Error Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Gender Moderation of UCLA-WCST Relationship', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VERIFICATION_DIR / "plot1_scatter_regression.png", dpi=300, bbox_inches='tight')
print("  Saved: plot1_scatter_regression.png")
plt.close()

# =============================================================================
# 7.2 Diagnostic Plots (Residuals)
# =============================================================================

print("\n" + "=" * 80)
print("7.2 회귀 진단 플롯")
print("=" * 80)

# Fit full model
formula = "perseverative_error_rate ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
model = smf.ols(formula, data=master_clean).fit()

# Get residuals and fitted values
master_clean['residuals'] = model.resid
master_clean['fitted'] = model.fittedvalues
master_clean['std_resid'] = model.resid / np.std(model.resid)

# Create 2x2 diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Residuals vs Fitted
ax1 = axes[0, 0]
ax1.scatter(master_clean['fitted'], master_clean['residuals'], alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
ax1.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax1.set_title('Residuals vs Fitted', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Q-Q Plot
ax2 = axes[0, 1]
stats.probplot(master_clean['std_resid'], dist="norm", plot=ax2)
ax2.set_title('Normal Q-Q Plot', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Scale-Location
ax3 = axes[1, 0]
ax3.scatter(master_clean['fitted'], np.sqrt(np.abs(master_clean['std_resid'])),
           alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
ax3.set_ylabel('√|Standardized Residuals|', fontsize=11, fontweight='bold')
ax3.set_title('Scale-Location', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Cook's Distance
ax4 = axes[1, 1]
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
threshold = 4 / len(master_clean)
ax4.stem(range(len(cooks_d)), cooks_d, basefmt=" ", markerfmt='o')
ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
ax4.set_xlabel('Observation Index', fontsize=11, fontweight='bold')
ax4.set_ylabel("Cook's Distance", fontsize=11, fontweight='bold')
ax4.set_title("Cook's Distance", fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VERIFICATION_DIR / "plot2_diagnostics.png", dpi=300, bbox_inches='tight')
print("  Saved: plot2_diagnostics.png")
plt.close()

# =============================================================================
# 7.3 Simple Slopes Visualization
# =============================================================================

print("\n" + "=" * 80)
print("7.3 Simple Slopes 시각화")
print("=" * 80)

# Create predicted values across UCLA range
ucla_range = np.linspace(master_clean['z_ucla'].min(), master_clean['z_ucla'].max(), 100)

# Mean values for covariates
mean_age = 0
mean_dep = 0
mean_anx = 0
mean_stress = 0

# Predictions for females (gender_male=0)
pred_female = (
    model.params['Intercept'] +
    model.params['z_ucla'] * ucla_range +
    model.params['z_age'] * mean_age +
    model.params['z_dass_dep'] * mean_dep +
    model.params['z_dass_anx'] * mean_anx +
    model.params['z_dass_stress'] * mean_stress
)

# Predictions for males (gender_male=1)
pred_male = (
    model.params['Intercept'] +
    model.params['C(gender_male)[T.1]'] +
    (model.params['z_ucla'] + model.params['z_ucla:C(gender_male)[T.1]']) * ucla_range +
    model.params['z_age'] * mean_age +
    model.params['z_dass_dep'] * mean_dep +
    model.params['z_dass_anx'] * mean_anx +
    model.params['z_dass_stress'] * mean_stress
)

# Convert z-scores back to raw UCLA scores for x-axis
ucla_raw_range = ucla_range * master_clean['ucla_total'].std() + master_clean['ucla_total'].mean()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ucla_raw_range, pred_female, color='#e74c3c', linewidth=3, label='Female: β = -0.30, p = 0.72')
ax.plot(ucla_raw_range, pred_male, color='#3498db', linewidth=3, label='Male: β = 2.29, p = 0.07')

# Add confidence bands
ax.fill_between(ucla_raw_range, pred_female - 2, pred_female + 2, color='#e74c3c', alpha=0.2)
ax.fill_between(ucla_raw_range, pred_male - 2, pred_male + 2, color='#3498db', alpha=0.2)

ax.set_xlabel('UCLA Loneliness Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted WCST Perseverative Error Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Simple Slopes: Gender Moderation Effect', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VERIFICATION_DIR / "plot3_simple_slopes.png", dpi=300, bbox_inches='tight')
print("  Saved: plot3_simple_slopes.png")
plt.close()

# =============================================================================
# 7.4 Effect Size Visualization
# =============================================================================

print("\n" + "=" * 80)
print("7.4 효과크기 시각화")
print("=" * 80)

# Load previous results
model_specs = pd.read_csv(VERIFICATION_DIR / "step2_model_specifications.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Beta coefficients across models
models = model_specs['model'].values
betas = model_specs['interaction_beta'].values
colors = ['green' if p < 0.05 else 'red' for p in model_specs['interaction_p']]

ax1.barh(models, betas, color=colors, alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Interaction Coefficient (β)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Model Specification', fontsize=12, fontweight='bold')
ax1.set_title('Sensitivity to Model Specification', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Right: P-values across models
p_values = model_specs['interaction_p'].values
ax2.barh(models, -np.log10(p_values), color=colors, alpha=0.7, edgecolor='black')
ax2.axvline(x=-np.log10(0.05), color='blue', linestyle='--', linewidth=2, label='α = 0.05')
ax2.axvline(x=-np.log10(0.0167), color='red', linestyle='--', linewidth=2, label='Bonferroni α = 0.0167')
ax2.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
ax2.set_title('Statistical Significance Across Models', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(VERIFICATION_DIR / "plot4_effect_sizes.png", dpi=300, bbox_inches='tight')
print("  Saved: plot4_effect_sizes.png")
plt.close()

# =============================================================================
# 7.5 Outlier Impact Visualization
# =============================================================================

print("\n" + "=" * 80)
print("7.5 이상치 영향 시각화")
print("=" * 80)

outlier_sens = pd.read_csv(VERIFICATION_DIR / "step3_outlier_sensitivity.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

scenarios = outlier_sens['scenario'].values
betas_out = outlier_sens['beta'].values
p_values_out = outlier_sens['p_value'].values
sample_sizes = outlier_sens['n'].values

# Left: Beta coefficients
colors_out = ['green' if p < 0.05 else 'red' for p in p_values_out]
bars1 = ax1.barh(scenarios, betas_out, color=colors_out, alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Interaction Coefficient (β)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Outlier Removal Scenario', fontsize=12, fontweight='bold')
ax1.set_title('Effect After Outlier Removal', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add sample sizes as text
for i, (scenario, n) in enumerate(zip(scenarios, sample_sizes)):
    ax1.text(betas_out[i] + 0.1, i, f'N={n}', va='center', fontsize=10, fontweight='bold')

# Right: P-values
bars2 = ax2.barh(scenarios, -np.log10(p_values_out), color=colors_out, alpha=0.7, edgecolor='black')
ax2.axvline(x=-np.log10(0.05), color='blue', linestyle='--', linewidth=2, label='α = 0.05')
ax2.axvline(x=-np.log10(0.0167), color='red', linestyle='--', linewidth=2, label='Bonferroni α = 0.0167')
ax2.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
ax2.set_title('Significance After Outlier Removal', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(VERIFICATION_DIR / "plot5_outlier_impact.png", dpi=300, bbox_inches='tight')
print("  Saved: plot5_outlier_impact.png")
plt.close()

print("\n" + "=" * 80)
print("Step 7 완료")
print("=" * 80)
print("\n생성된 플롯:")
print("  1. plot1_scatter_regression.png - 성별별 산점도 및 회귀선")
print("  2. plot2_diagnostics.png - 회귀 진단 플롯 (4가지)")
print("  3. plot3_simple_slopes.png - Simple slopes 시각화")
print("  4. plot4_effect_sizes.png - 모형 사양 민감도")
print("  5. plot5_outlier_impact.png - 이상치 영향")
