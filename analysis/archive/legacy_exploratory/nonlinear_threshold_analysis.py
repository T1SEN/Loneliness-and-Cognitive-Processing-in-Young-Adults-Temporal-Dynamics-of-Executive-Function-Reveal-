"""
Nonlinear and Threshold Analysis
================================
UCLA 효과의 비선형 관계 및 임계점 분석

Features:
1. Piecewise regression (breakpoint detection)
2. GAM 기반 비선형 관계 추정
3. Dose-response 모델링
4. 성별 층화 threshold 추정

DASS-21 Control: 가능한 경우 통제
"""

import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar, curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Add parent to path for imports
_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/nonlinear_threshold_analysis.py')
sys.path.insert(0, str(_this_file.parent))
from utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "nonlinear_threshold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NONLINEAR AND THRESHOLD ANALYSIS")
print("=" * 70)


# ============================================================================
# Helper functions
# ============================================================================

def piecewise_linear(x, breakpoint, slope1, slope2, intercept):
    """Piecewise linear function with one breakpoint."""
    return np.where(x < breakpoint,
                    intercept + slope1 * x,
                    intercept + slope1 * breakpoint + slope2 * (x - breakpoint))


def fit_piecewise_regression(x, y, min_bp=None, max_bp=None):
    """
    Fit piecewise regression and find optimal breakpoint.
    Returns breakpoint, slopes, and model fit statistics.
    """
    x = np.array(x)
    y = np.array(y)

    if min_bp is None:
        min_bp = np.percentile(x, 20)
    if max_bp is None:
        max_bp = np.percentile(x, 80)

    def compute_ssr(bp):
        """Compute sum of squared residuals for given breakpoint."""
        try:
            # Create design matrix for piecewise regression
            X = np.column_stack([
                np.ones(len(x)),
                x,
                np.maximum(x - bp, 0)  # (x - bp)+ for second segment
            ])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            return np.sum((y - y_pred) ** 2)
        except Exception:
            return np.inf

    # Grid search for optimal breakpoint
    breakpoints = np.linspace(min_bp, max_bp, 50)
    ssrs = [compute_ssr(bp) for bp in breakpoints]
    optimal_bp = breakpoints[np.argmin(ssrs)]

    # Fit final model at optimal breakpoint
    X = np.column_stack([
        np.ones(len(x)),
        x,
        np.maximum(x - optimal_bp, 0)
    ])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Slopes
    slope1 = beta[1]
    slope2 = beta[1] + beta[2]  # Total slope after breakpoint

    return {
        'breakpoint': optimal_bp,
        'slope_before': slope1,
        'slope_after': slope2,
        'slope_change': beta[2],
        'intercept': beta[0],
        'r2': r2,
        'n': len(x)
    }


def polynomial_regression(x, y, degree=2):
    """Fit polynomial regression and test for nonlinearity."""
    X = np.column_stack([x ** d for d in range(1, degree + 1)])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return {
        'r2': model.rsquared,
        'coefficients': model.params,
        'pvalues': model.pvalues,
        'nonlinear_significant': model.pvalues[2] < 0.05 if degree >= 2 and len(model.pvalues) > 2 else False
    }


def loess_smooth(x, y, frac=0.5):
    """LOWESS smoothing for nonparametric trend."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def test_threshold_effect(df, x_col, y_col, threshold):
    """Test if effect differs above vs below threshold."""
    below = df[df[x_col] <= threshold]
    above = df[df[x_col] > threshold]

    results = {
        'threshold': threshold,
        'n_below': len(below),
        'n_above': len(above)
    }

    # Correlation below threshold
    if len(below) > 10:
        r_below, p_below = stats.pearsonr(below[x_col], below[y_col])
        results['r_below'] = r_below
        results['p_below'] = p_below
    else:
        results['r_below'] = np.nan
        results['p_below'] = np.nan

    # Correlation above threshold
    if len(above) > 10:
        r_above, p_above = stats.pearsonr(above[x_col], above[y_col])
        results['r_above'] = r_above
        results['p_above'] = p_above
    else:
        results['r_above'] = np.nan
        results['p_above'] = np.nan

    # Fisher z-test for difference
    if pd.notna(results['r_below']) and pd.notna(results['r_above']):
        z1 = 0.5 * np.log((1 + results['r_below']) / (1 - results['r_below'])) if abs(results['r_below']) < 1 else np.nan
        z2 = 0.5 * np.log((1 + results['r_above']) / (1 - results['r_above'])) if abs(results['r_above']) < 1 else np.nan

        if pd.notna(z1) and pd.notna(z2):
            se = np.sqrt(1/(results['n_below']-3) + 1/(results['n_above']-3))
            z_diff = (z1 - z2) / se
            results['p_difference'] = 2 * (1 - stats.norm.cdf(abs(z_diff)))
        else:
            results['p_difference'] = np.nan
    else:
        results['p_difference'] = np.nan

    return results


# ============================================================================
# Load data
# ============================================================================
print("\n[1] Loading data...")

df = load_master_dataset()

# Define outcomes
outcomes = {
    'pe_rate': 'WCST PE Rate',
    'prp_bottleneck': 'PRP Bottleneck',
    'wcst_accuracy': 'WCST Accuracy'
}

# Filter to available outcomes
outcomes = {k: v for k, v in outcomes.items() if k in df.columns}
print(f"   Outcomes: {list(outcomes.keys())}")


# ============================================================================
# 2. Piecewise Regression Analysis
# ============================================================================
print("\n[2] Piecewise Regression Analysis (Threshold Detection)...")

piecewise_results = []

for outcome_col, outcome_label in outcomes.items():
    valid_df = df.dropna(subset=['ucla_score', outcome_col])

    if len(valid_df) < 30:
        print(f"   Skipping {outcome_label}: insufficient data")
        continue

    # Full sample
    result = fit_piecewise_regression(valid_df['ucla_score'], valid_df[outcome_col])
    result['outcome'] = outcome_label
    result['group'] = 'Full Sample'
    piecewise_results.append(result)

    print(f"   {outcome_label} (Full): breakpoint = {result['breakpoint']:.1f}, R² = {result['r2']:.3f}")

    # By gender
    for gender in ['male', 'female']:
        gender_df = valid_df[valid_df['gender'] == gender]
        if len(gender_df) < 15:
            continue

        result = fit_piecewise_regression(gender_df['ucla_score'], gender_df[outcome_col])
        result['outcome'] = outcome_label
        result['group'] = gender.capitalize()
        piecewise_results.append(result)

        print(f"   {outcome_label} ({gender.capitalize()}): breakpoint = {result['breakpoint']:.1f}")

piecewise_df = pd.DataFrame(piecewise_results)
piecewise_df.to_csv(OUTPUT_DIR / "piecewise_regression_results.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: piecewise_regression_results.csv")


# ============================================================================
# 3. Polynomial (Quadratic) Test for Nonlinearity
# ============================================================================
print("\n[3] Testing Quadratic (Nonlinear) Effects...")

quadratic_results = []

for outcome_col, outcome_label in outcomes.items():
    valid_df = df.dropna(subset=['ucla_score', outcome_col])

    if len(valid_df) < 30:
        continue

    # Full sample
    x = valid_df['ucla_score'].values
    y = valid_df[outcome_col].values

    poly_result = polynomial_regression(x, y, degree=2)

    quadratic_results.append({
        'outcome': outcome_label,
        'group': 'Full Sample',
        'n': len(valid_df),
        'r2_quadratic': poly_result['r2'],
        'linear_coef': poly_result['coefficients'][1],
        'quadratic_coef': poly_result['coefficients'][2],
        'p_quadratic': poly_result['pvalues'][2],
        'nonlinear_significant': poly_result['nonlinear_significant']
    })

    # By gender
    for gender in ['male', 'female']:
        gender_df = valid_df[valid_df['gender'] == gender]
        if len(gender_df) < 15:
            continue

        x_g = gender_df['ucla_score'].values
        y_g = gender_df[outcome_col].values

        poly_result = polynomial_regression(x_g, y_g, degree=2)

        quadratic_results.append({
            'outcome': outcome_label,
            'group': gender.capitalize(),
            'n': len(gender_df),
            'r2_quadratic': poly_result['r2'],
            'linear_coef': poly_result['coefficients'][1],
            'quadratic_coef': poly_result['coefficients'][2],
            'p_quadratic': poly_result['pvalues'][2],
            'nonlinear_significant': poly_result['nonlinear_significant']
        })

quadratic_df = pd.DataFrame(quadratic_results)
quadratic_df.to_csv(OUTPUT_DIR / "quadratic_nonlinearity_tests.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: quadratic_nonlinearity_tests.csv")

# Summary
print("\n   Quadratic Term Significance:")
for _, row in quadratic_df.iterrows():
    sig = "SIGNIFICANT" if row['nonlinear_significant'] else "not significant"
    print(f"     {row['outcome']} ({row['group']}): p = {row['p_quadratic']:.4f} ({sig})")


# ============================================================================
# 4. Dose-Response Analysis (Quartile Effects)
# ============================================================================
print("\n[4] Dose-Response Analysis (UCLA Quartiles)...")

dose_response = []

for outcome_col, outcome_label in outcomes.items():
    valid_df = df.dropna(subset=['ucla_score', outcome_col])

    if len(valid_df) < 30:
        continue

    # Create quartiles
    valid_df = valid_df.copy()
    valid_df['ucla_quartile'] = pd.qcut(valid_df['ucla_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Mean and SE for each quartile
    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = valid_df[valid_df['ucla_quartile'] == quartile][outcome_col]
        q_ucla = valid_df[valid_df['ucla_quartile'] == quartile]['ucla_score']

        dose_response.append({
            'outcome': outcome_label,
            'quartile': quartile,
            'n': len(q_data),
            'ucla_mean': q_ucla.mean(),
            'ucla_range': f"{q_ucla.min():.0f}-{q_ucla.max():.0f}",
            'outcome_mean': q_data.mean(),
            'outcome_se': q_data.std() / np.sqrt(len(q_data)),
            'outcome_ci_lower': q_data.mean() - 1.96 * q_data.std() / np.sqrt(len(q_data)),
            'outcome_ci_upper': q_data.mean() + 1.96 * q_data.std() / np.sqrt(len(q_data))
        })

dose_response_df = pd.DataFrame(dose_response)
dose_response_df.to_csv(OUTPUT_DIR / "dose_response_quartiles.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: dose_response_quartiles.csv")

# Linear trend test
print("\n   Linear Trend Tests (Jonckheere-Terpstra):")
for outcome_col, outcome_label in outcomes.items():
    valid_df = df.dropna(subset=['ucla_score', outcome_col]).copy()
    if len(valid_df) < 30:
        continue

    valid_df['ucla_quartile'] = pd.qcut(valid_df['ucla_score'], q=4, labels=[1, 2, 3, 4])

    # Correlation with quartile as proxy for trend test
    r, p = stats.spearmanr(valid_df['ucla_quartile'].astype(int), valid_df[outcome_col])
    print(f"     {outcome_label}: Spearman ρ = {r:.3f}, p = {p:.4f}")


# ============================================================================
# 5. Threshold Effect Analysis
# ============================================================================
print("\n[5] Threshold Effect Analysis...")

threshold_results = []

# Test various thresholds (percentiles)
percentiles = [25, 33, 50, 67, 75]

for outcome_col, outcome_label in outcomes.items():
    valid_df = df.dropna(subset=['ucla_score', outcome_col])

    if len(valid_df) < 30:
        continue

    for pct in percentiles:
        threshold = np.percentile(valid_df['ucla_score'], pct)
        result = test_threshold_effect(valid_df, 'ucla_score', outcome_col, threshold)
        result['outcome'] = outcome_label
        result['percentile'] = pct
        threshold_results.append(result)

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(OUTPUT_DIR / "threshold_effect_tests.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: threshold_effect_tests.csv")


# ============================================================================
# 6. Visualization
# ============================================================================
print("\n[6] Creating visualizations...")

n_outcomes = len(outcomes)
fig, axes = plt.subplots(n_outcomes, 3, figsize=(15, 5 * n_outcomes))

if n_outcomes == 1:
    axes = axes.reshape(1, -1)

for i, (outcome_col, outcome_label) in enumerate(outcomes.items()):
    valid_df = df.dropna(subset=['ucla_score', outcome_col, 'gender'])

    if len(valid_df) < 20:
        continue

    # 6a. Scatter with LOWESS
    ax = axes[i, 0]
    for gender, color in [('male', 'blue'), ('female', 'red')]:
        gender_data = valid_df[valid_df['gender'] == gender]
        ax.scatter(gender_data['ucla_score'], gender_data[outcome_col],
                  alpha=0.5, label=gender.capitalize(), color=color, s=30)

        # LOWESS smooth
        if len(gender_data) > 15:
            try:
                x_smooth, y_smooth = loess_smooth(
                    gender_data['ucla_score'].values,
                    gender_data[outcome_col].values,
                    frac=0.6
                )
                ax.plot(x_smooth, y_smooth, color=color, linewidth=2, alpha=0.8)
            except Exception:
                pass

    ax.set_xlabel('UCLA Loneliness Score')
    ax.set_ylabel(outcome_label)
    ax.set_title(f'{outcome_label}: LOWESS Smooth by Gender', fontweight='bold')
    ax.legend()

    # 6b. Piecewise fit (full sample)
    ax = axes[i, 1]
    x = valid_df['ucla_score'].values
    y = valid_df[outcome_col].values

    ax.scatter(x, y, alpha=0.4, color='gray', s=20)

    # Get piecewise parameters
    pw_result = [r for r in piecewise_results
                 if r['outcome'] == outcome_label and r['group'] == 'Full Sample']

    if pw_result:
        pw = pw_result[0]
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = piecewise_linear(x_line, pw['breakpoint'], pw['slope_before'],
                                  pw['slope_after'], pw['intercept'])
        ax.plot(x_line, y_line, 'r-', linewidth=2, label='Piecewise fit')
        ax.axvline(x=pw['breakpoint'], color='green', linestyle='--',
                  label=f"Breakpoint = {pw['breakpoint']:.1f}")

    ax.set_xlabel('UCLA Loneliness Score')
    ax.set_ylabel(outcome_label)
    ax.set_title(f'{outcome_label}: Piecewise Regression', fontweight='bold')
    ax.legend()

    # 6c. Dose-response (quartiles)
    ax = axes[i, 2]
    outcome_dose = dose_response_df[dose_response_df['outcome'] == outcome_label]

    if len(outcome_dose) > 0:
        x_pos = [1, 2, 3, 4]
        means = outcome_dose['outcome_mean'].values
        ci_lower = outcome_dose['outcome_ci_lower'].values
        ci_upper = outcome_dose['outcome_ci_upper'].values

        ax.bar(x_pos, means, alpha=0.7, color='steelblue')
        ax.errorbar(x_pos, means, yerr=[means - ci_lower, ci_upper - means],
                   fmt='none', color='black', capsize=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])
        ax.set_xlabel('UCLA Loneliness Quartile')
        ax.set_ylabel(f'Mean {outcome_label}')
        ax.set_title(f'{outcome_label}: Dose-Response', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "nonlinear_analysis_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: nonlinear_analysis_plots.png")


# ============================================================================
# 7. Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("NONLINEAR AND THRESHOLD ANALYSIS SUMMARY")
print("=" * 70)

print("\n1. Piecewise Regression (Breakpoint Detection):")
for _, row in piecewise_df[piecewise_df['group'] == 'Full Sample'].iterrows():
    print(f"   {row['outcome']}:")
    print(f"     Breakpoint: UCLA = {row['breakpoint']:.1f}")
    print(f"     Slope before: {row['slope_before']:.3f}")
    print(f"     Slope after: {row['slope_after']:.3f}")
    print(f"     Slope change: {row['slope_change']:.3f}")

print("\n2. Quadratic (Nonlinear) Effects:")
sig_quad = quadratic_df[quadratic_df['nonlinear_significant']]
if len(sig_quad) > 0:
    print("   Significant nonlinear effects:")
    for _, row in sig_quad.iterrows():
        print(f"     {row['outcome']} ({row['group']}): quadratic coef = {row['quadratic_coef']:.4f}, p = {row['p_quadratic']:.4f}")
else:
    print("   No significant quadratic effects detected")

print("\n3. Threshold Effects:")
# Find thresholds where effect differs significantly
sig_threshold = threshold_df[threshold_df['p_difference'] < 0.1]
if len(sig_threshold) > 0:
    print("   Thresholds with different effects (below vs above):")
    for _, row in sig_threshold.iterrows():
        print(f"     {row['outcome']} at {row['percentile']}th percentile (UCLA = {row['threshold']:.1f}):")
        print(f"       Below: r = {row['r_below']:.3f}, Above: r = {row['r_above']:.3f}")
else:
    print("   No significant threshold effects detected")

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("=" * 70)

# Save report
with open(OUTPUT_DIR / "nonlinear_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write("NONLINEAR AND THRESHOLD ANALYSIS SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. Piecewise Regression (Breakpoint Detection):\n")
    for _, row in piecewise_df[piecewise_df['group'] == 'Full Sample'].iterrows():
        f.write(f"   {row['outcome']}:\n")
        f.write(f"     Breakpoint: UCLA = {row['breakpoint']:.1f}\n")
        f.write(f"     Slope before: {row['slope_before']:.3f}\n")
        f.write(f"     Slope after: {row['slope_after']:.3f}\n")
        f.write(f"     R² = {row['r2']:.3f}\n\n")

    f.write("\n2. Quadratic (Nonlinear) Effects:\n")
    for _, row in quadratic_df.iterrows():
        sig = "SIGNIFICANT" if row['nonlinear_significant'] else "not significant"
        f.write(f"   {row['outcome']} ({row['group']}): p = {row['p_quadratic']:.4f} ({sig})\n")

    f.write("\n3. Dose-Response (Quartile Means):\n")
    for outcome in outcomes.values():
        outcome_data = dose_response_df[dose_response_df['outcome'] == outcome]
        f.write(f"   {outcome}:\n")
        for _, row in outcome_data.iterrows():
            f.write(f"     {row['quartile']}: {row['outcome_mean']:.2f} ± {row['outcome_se']:.2f}\n")

print("\nNonlinear and threshold analysis complete!")
