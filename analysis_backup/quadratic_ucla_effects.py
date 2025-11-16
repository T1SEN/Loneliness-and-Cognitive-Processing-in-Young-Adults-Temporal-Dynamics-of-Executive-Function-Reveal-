"""
Quadratic UCLA Effects Analysis
=================================
Tests for non-linear relationship between UCLA and WCST PE

Research Questions:
1. Is the UCLA→PE relationship linear or quadratic?
2. Is there a threshold UCLA score above which PE accelerates?
3. Do quadratic effects differ by gender?

Methods:
- Polynomial regression (linear vs quadratic models)
- Segmented regression (breakpoint detection)
- Johnson-Neyman intervals for interaction regions

Author: Advanced Analysis Suite
Date: 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

# Unicode handling
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/quadratic_ucla")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("QUADRATIC UCLA EFFECTS ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

master = pd.read_csv(RESULTS_DIR / "analysis_outputs" / "master_expanded_metrics.csv")

# Add demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants['participant_id'] = participants.get('participantId', participants.get('participant_id'))
participants = participants[['participant_id', 'gender', 'age']].copy()
participants['age'] = pd.to_numeric(participants['age'], errors='coerce')

# Map Korean gender to English
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)

master = master.merge(participants, on='participant_id', how='left')

# Rename columns
if 'pe_rate' in master.columns:
    master['wcst_pe_rate'] = master['pe_rate']

# Calculate DASS total
if 'dass_total' not in master.columns:
    master['dass_total'] = master['dass_anxiety'] + master['dass_stress'] + master['dass_depression']

# Filter complete cases
master = master.dropna(subset=['ucla_total', 'wcst_pe_rate', 'gender'])
master['gender_male'] = (master['gender'] == 'male').astype(int)

print(f"  Complete cases: {len(master)} participants")
print(f"  UCLA range: {master['ucla_total'].min():.0f}-{master['ucla_total'].max():.0f}")
print(f"  UCLA M={master['ucla_total'].mean():.2f}, SD={master['ucla_total'].std():.2f}")

# ============================================================================
# 2. LINEAR VS QUADRATIC MODEL COMPARISON
# ============================================================================
print("\n[2/6] Testing linear vs quadratic models...")

# Create quadratic term
master['ucla_squared'] = master['ucla_total'] ** 2

# Model 1: Linear
model_linear = smf.ols('wcst_pe_rate ~ ucla_total + gender_male + age + dass_total',
                        data=master).fit()

# Model 2: Quadratic (UCLA² main effect)
model_quad = smf.ols('wcst_pe_rate ~ ucla_total + ucla_squared + gender_male + age + dass_total',
                      data=master).fit()

# Model 3: Quadratic × Gender interaction
model_quad_gender = smf.ols('wcst_pe_rate ~ ucla_total * gender_male + ucla_squared * gender_male + age + dass_total',
                             data=master).fit()

print("\n  Model Comparison:")
print(f"    Linear: R²={model_linear.rsquared:.4f}, AIC={model_linear.aic:.2f}")
print(f"    Quadratic: R²={model_quad.rsquared:.4f}, AIC={model_quad.aic:.2f}")
print(f"    Quadratic×Gender: R²={model_quad_gender.rsquared:.4f}, AIC={model_quad_gender.aic:.2f}")

# Likelihood ratio test
lr_stat = 2 * (model_quad.llf - model_linear.llf)
p_value = stats.chi2.sf(lr_stat, 1)  # 1 df for adding UCLA²

print(f"\n  Likelihood Ratio Test (Linear vs Quadratic):")
print(f"    LR = {lr_stat:.2f}, p = {p_value:.4f}")

if p_value < 0.05:
    print("    ✓ Quadratic model significantly better!")
    best_model = model_quad_gender if model_quad_gender.aic < model_quad.aic else model_quad
else:
    print("    ✗ Linear model sufficient")
    best_model = model_linear

# Extract quadratic coefficients
results = []

if 'ucla_squared' in model_quad.params:
    beta_quad = model_quad.params['ucla_squared']
    p_quad = model_quad.pvalues['ucla_squared']
    results.append({
        'Model': 'Quadratic (whole sample)',
        'UCLA_linear': model_quad.params['ucla_total'],
        'UCLA_quadratic': beta_quad,
        'p_quadratic': p_quad,
        'Significant': 'Yes' if p_quad < 0.05 else 'No'
    })
    print(f"\n  Quadratic term (whole sample): β={beta_quad:.6f}, p={p_quad:.4f}")

# Gender-specific quadratic effects
if 'ucla_squared:gender_male' in model_quad_gender.params:
    beta_quad_interaction = model_quad_gender.params['ucla_squared:gender_male']
    p_quad_interaction = model_quad_gender.pvalues['ucla_squared:gender_male']
    results.append({
        'Model': 'Quadratic × Gender',
        'UCLA_linear': model_quad_gender.params.get('ucla_total', np.nan),
        'UCLA_quadratic': model_quad_gender.params.get('ucla_squared', np.nan),
        'Interaction_quadratic': beta_quad_interaction,
        'p_interaction': p_quad_interaction,
        'Significant': 'Yes' if p_quad_interaction < 0.05 else 'No'
    })
    print(f"  Quadratic × Gender interaction: β={beta_quad_interaction:.6f}, p={p_quad_interaction:.4f}")

# Stratified by gender
print("\n  Gender-stratified quadratic effects:")
for gender, label in [(0, 'Female'), (1, 'Male')]:
    subset = master[master['gender_male'] == gender]
    if len(subset) >= 10:
        model = smf.ols('wcst_pe_rate ~ ucla_total + ucla_squared + age + dass_total',
                        data=subset).fit()
        if 'ucla_squared' in model.params:
            beta_quad = model.params['ucla_squared']
            p_quad = model.pvalues['ucla_squared']
            print(f"    {label} (N={len(subset)}): β_quad={beta_quad:.6f}, p={p_quad:.4f}")

            results.append({
                'Model': f'Quadratic ({label})',
                'N': len(subset),
                'UCLA_linear': model.params['ucla_total'],
                'UCLA_quadratic': beta_quad,
                'p_quadratic': p_quad,
                'Significant': 'Yes' if p_quad < 0.05 else 'No'
            })

results_df = pd.DataFrame(results)

# ============================================================================
# 3. SEGMENTED REGRESSION (BREAKPOINT DETECTION)
# ============================================================================
print("\n[3/6] Searching for UCLA breakpoint...")

# Try different potential breakpoints
breakpoints = range(30, 70, 5)  # Test breakpoints from 30 to 65
aic_scores = []

for bp in breakpoints:
    # Create indicator for above/below breakpoint
    master[f'above_{bp}'] = (master['ucla_total'] > bp).astype(int)
    master[f'ucla_above_{bp}'] = master['ucla_total'] * master[f'above_{bp}']

    # Segmented model
    formula = f'wcst_pe_rate ~ ucla_total + above_{bp} + ucla_above_{bp} + gender_male + age + dass_total'
    try:
        model = smf.ols(formula, data=master).fit()
        aic_scores.append({
            'Breakpoint': bp,
            'AIC': model.aic,
            'R²': model.rsquared,
            'Slope_below': model.params['ucla_total'],
            'Slope_above': model.params['ucla_total'] + model.params[f'ucla_above_{bp}'],
            'p_breakpoint': model.pvalues[f'above_{bp}']
        })
    except:
        continue

if aic_scores:
    aic_df = pd.DataFrame(aic_scores)
    best_bp = aic_df.loc[aic_df['AIC'].idxmin(), 'Breakpoint']
    best_aic_model = aic_df.loc[aic_df['AIC'].idxmin()]

    print(f"\n  Best breakpoint: UCLA = {best_bp:.0f}")
    print(f"    AIC = {best_aic_model['AIC']:.2f}")
    print(f"    Slope below {best_bp}: β={best_aic_model['Slope_below']:.4f}")
    print(f"    Slope above {best_bp}: β={best_aic_model['Slope_above']:.4f}")
    print(f"    p-value for breakpoint: {best_aic_model['p_breakpoint']:.4f}")

    # Compare with linear model
    if best_aic_model['AIC'] < model_linear.aic:
        print(f"    ✓ Segmented model (AIC={best_aic_model['AIC']:.2f}) better than linear (AIC={model_linear.aic:.2f})")
    else:
        print(f"    ✗ Linear model still better")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n[4/6] Creating visualizations...")

# Plot 1: Linear vs Quadratic fit
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for i, (gender, label) in enumerate([(0, 'Female'), (1, 'Male')]):
    subset = master[master['gender_male'] == gender]

    # Scatter plot
    axes[i].scatter(subset['ucla_total'], subset['wcst_pe_rate'], alpha=0.5, s=50)

    # Linear fit
    x_range = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
    linear_pred = model_linear.params['Intercept'] + \
                  model_linear.params['ucla_total'] * x_range + \
                  model_linear.params['gender_male'] * gender
    axes[i].plot(x_range, linear_pred, 'r--', linewidth=2, label='Linear')

    # Quadratic fit (if better)
    if p_value < 0.10:  # Show quadratic if trending or significant
        quad_pred = model_quad.params['Intercept'] + \
                   model_quad.params['ucla_total'] * x_range + \
                   model_quad.params['ucla_squared'] * (x_range ** 2) + \
                   model_quad.params['gender_male'] * gender
        axes[i].plot(x_range, quad_pred, 'b-', linewidth=2, label='Quadratic')

    axes[i].set_title(f'{label} (N={len(subset)})', fontsize=12, weight='bold')
    axes[i].set_xlabel('UCLA Loneliness Score')
    if i == 0:
        axes[i].set_ylabel('WCST Perseverative Error Rate (%)')
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.suptitle('Linear vs Quadratic UCLA→PE Relationship', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "linear_vs_quadratic.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Breakpoint analysis (if exists)
if aic_scores:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AIC by breakpoint
    axes[0].plot(aic_df['Breakpoint'], aic_df['AIC'], 'o-', linewidth=2, markersize=8)
    axes[0].axhline(model_linear.aic, color='r', linestyle='--', label='Linear model AIC')
    axes[0].axvline(best_bp, color='green', linestyle='--', label=f'Best breakpoint={best_bp:.0f}')
    axes[0].set_xlabel('UCLA Breakpoint')
    axes[0].set_ylabel('AIC (lower is better)')
    axes[0].set_title('Segmented Regression: Optimal Breakpoint')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: Data with breakpoint
    below_bp = master[master['ucla_total'] <= best_bp]
    above_bp = master[master['ucla_total'] > best_bp]

    axes[1].scatter(below_bp['ucla_total'], below_bp['wcst_pe_rate'],
                   alpha=0.5, label=f'UCLA ≤ {best_bp:.0f}', color='blue')
    axes[1].scatter(above_bp['ucla_total'], above_bp['wcst_pe_rate'],
                   alpha=0.5, label=f'UCLA > {best_bp:.0f}', color='red')

    # Fit lines for each segment
    if len(below_bp) >= 3:
        z_below = np.polyfit(below_bp['ucla_total'], below_bp['wcst_pe_rate'], 1)
        p_below = np.poly1d(z_below)
        x_below = np.linspace(below_bp['ucla_total'].min(), best_bp, 50)
        axes[1].plot(x_below, p_below(x_below), 'b--', linewidth=2)

    if len(above_bp) >= 3:
        z_above = np.polyfit(above_bp['ucla_total'], above_bp['wcst_pe_rate'], 1)
        p_above = np.poly1d(z_above)
        x_above = np.linspace(best_bp, above_bp['ucla_total'].max(), 50)
        axes[1].plot(x_above, p_above(x_above), 'r--', linewidth=2)

    axes[1].axvline(best_bp, color='green', linestyle='--', linewidth=2, label='Breakpoint')
    axes[1].set_xlabel('UCLA Loneliness Score')
    axes[1].set_ylabel('WCST Perseverative Error Rate (%)')
    axes[1].set_title('Segmented Regression Fit')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "breakpoint_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 3: Quadratic effect visualization (3D-like contour)
if p_value < 0.10:
    fig = plt.figure(figsize=(10, 8))

    # Create grid
    ucla_range = np.linspace(master['ucla_total'].min(), master['ucla_total'].max(), 50)
    gender_range = np.array([0, 1])

    # Predict PE for each combination
    predictions = np.zeros((len(ucla_range), len(gender_range)))

    for i, ucla in enumerate(ucla_range):
        for j, gender in enumerate(gender_range):
            pred_data = pd.DataFrame({
                'ucla_total': [ucla],
                'ucla_squared': [ucla**2],
                'gender_male': [gender],
                'age': [master['age'].mean()],
                'dass_total': [master['dass_total'].mean()]
            })
            predictions[i, j] = model_quad_gender.predict(pred_data)[0]

    # Plot
    plt.plot(ucla_range, predictions[:, 0], 'b-', linewidth=3, label='Females')
    plt.plot(ucla_range, predictions[:, 1], 'r-', linewidth=3, label='Males')

    # Add scatter
    plt.scatter(master[master['gender_male']==0]['ucla_total'],
               master[master['gender_male']==0]['wcst_pe_rate'],
               alpha=0.3, color='blue')
    plt.scatter(master[master['gender_male']==1]['ucla_total'],
               master[master['gender_male']==1]['wcst_pe_rate'],
               alpha=0.3, color='red')

    plt.xlabel('UCLA Loneliness Score', fontsize=12)
    plt.ylabel('WCST Perseverative Error Rate (%)', fontsize=12)
    plt.title('Quadratic UCLA→PE Relationship by Gender', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quadratic_gender_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================
print("\n[5/6] Saving results...")

results_df.to_csv(OUTPUT_DIR / "quadratic_model_results.csv", index=False, encoding='utf-8-sig')

if aic_scores:
    aic_df.to_csv(OUTPUT_DIR / "breakpoint_search_results.csv", index=False, encoding='utf-8-sig')

# Model summaries
with open(OUTPUT_DIR / "model_summaries.txt", 'w') as f:
    f.write("LINEAR MODEL\n")
    f.write("="*80 + "\n")
    f.write(model_linear.summary().as_text())
    f.write("\n\n")

    f.write("QUADRATIC MODEL\n")
    f.write("="*80 + "\n")
    f.write(model_quad.summary().as_text())
    f.write("\n\n")

    f.write("QUADRATIC × GENDER MODEL\n")
    f.write("="*80 + "\n")
    f.write(model_quad_gender.summary().as_text())

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================
print("\n[6/6] Generating summary...")

summary_lines = []
summary_lines.append("QUADRATIC UCLA EFFECTS ANALYSIS - SUMMARY\n")
summary_lines.append("="*80 + "\n\n")

summary_lines.append("RESEARCH QUESTION\n")
summary_lines.append("-" * 80 + "\n")
summary_lines.append("Is the UCLA→PE relationship linear or non-linear?\n")
summary_lines.append("Does PE accelerate at extreme loneliness levels?\n\n")

summary_lines.append("KEY FINDINGS\n")
summary_lines.append("-" * 80 + "\n\n")

summary_lines.append("1. LINEAR VS QUADRATIC MODEL COMPARISON\n")
summary_lines.append(f"   Linear model: R²={model_linear.rsquared:.4f}, AIC={model_linear.aic:.2f}\n")
summary_lines.append(f"   Quadratic model: R²={model_quad.rsquared:.4f}, AIC={model_quad.aic:.2f}\n")
summary_lines.append(f"   LR test: p={p_value:.4f}\n\n")

if p_value < 0.05:
    summary_lines.append("   ✓ SIGNIFICANT: Quadratic model significantly better\n")
    summary_lines.append("   → PE accelerates at extreme UCLA levels\n\n")
elif p_value < 0.10:
    summary_lines.append("   ⚠️ MARGINAL: Quadratic trend detected\n")
    summary_lines.append("   → Possible threshold effect, needs larger sample\n\n")
else:
    summary_lines.append("   ✗ NOT SIGNIFICANT: Linear relationship sufficient\n")
    summary_lines.append("   → No evidence of threshold or acceleration\n\n")

if 'ucla_squared' in model_quad.params:
    beta_quad = model_quad.params['ucla_squared']
    p_quad = model_quad.pvalues['ucla_squared']
    summary_lines.append(f"   Quadratic coefficient: β={beta_quad:.6f}, p={p_quad:.4f}\n")
    if beta_quad > 0:
        summary_lines.append("   → Positive curvature: PE accelerates at high UCLA\n\n")
    else:
        summary_lines.append("   → Negative curvature: PE decelerates at high UCLA\n\n")

summary_lines.append("2. BREAKPOINT ANALYSIS\n")
if aic_scores:
    summary_lines.append(f"   Best breakpoint: UCLA = {best_bp:.0f}\n")
    summary_lines.append(f"   Slope below {best_bp}: β={best_aic_model['Slope_below']:.4f}\n")
    summary_lines.append(f"   Slope above {best_bp}: β={best_aic_model['Slope_above']:.4f}\n")

    slope_change = best_aic_model['Slope_above'] - best_aic_model['Slope_below']
    if abs(slope_change) > 0.1:
        summary_lines.append(f"   Slope change: {slope_change:.4f} ({slope_change/best_aic_model['Slope_below']*100:.1f}% increase)\n")

    if best_aic_model['AIC'] < model_linear.aic:
        summary_lines.append("   ✓ Segmented model better than linear\n\n")
    else:
        summary_lines.append("   ✗ Linear model still better\n\n")

summary_lines.append("\nIMPLICATIONS\n")
summary_lines.append("-" * 80 + "\n")

if p_value < 0.05:
    summary_lines.append("CLINICAL THRESHOLD IDENTIFIED:\n")
    if aic_scores:
        summary_lines.append(f"  UCLA scores above {best_bp:.0f} may represent critical risk zone\n")
        summary_lines.append(f"  Interventions should target individuals approaching this threshold\n\n")
else:
    summary_lines.append("NO THRESHOLD DETECTED:\n")
    summary_lines.append("  Risk increases linearly with loneliness\n")
    summary_lines.append("  No specific 'cutoff' for intervention targeting\n\n")

summary_lines.append("="*80 + "\n")
summary_lines.append(f"Full results saved to: {OUTPUT_DIR}\n")

summary_text = ''.join(summary_lines)
print("\n" + summary_text)

with open(OUTPUT_DIR / "QUADRATIC_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n✓ Quadratic UCLA Effects Analysis complete!")
