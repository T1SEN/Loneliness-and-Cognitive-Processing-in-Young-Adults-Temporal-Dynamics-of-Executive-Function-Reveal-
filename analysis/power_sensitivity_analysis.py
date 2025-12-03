"""
Power & Sensitivity Analysis
============================
현재 표본 크기에서의 검정력 분석 및 민감도 분석

Features:
1. 사후 검정력 분석 (observed effect sizes)
2. 최소 탐지 효과 크기 (80% power threshold)
3. 필요 표본 크기 추정
4. 검정력 곡선 시각화
5. Interaction vs Main effect 검정력 비교

DASS-21 Control: 분석 방법론 관련이므로 N/A
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
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path for imports
_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/power_sensitivity_analysis.py')
sys.path.insert(0, str(_this_file.parent))
from utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "power_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("POWER & SENSITIVITY ANALYSIS")
print("=" * 70)


# ============================================================================
# Power calculation functions
# ============================================================================

def power_correlation(n, r, alpha=0.05):
    """Calculate power for correlation test."""
    if abs(r) >= 1 or n < 4:
        return np.nan

    # Fisher z transformation
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)

    # Critical z for two-tailed test
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Power = P(|Z| > z_crit | H1)
    power = 1 - stats.norm.cdf(z_crit - abs(z_r) / se) + stats.norm.cdf(-z_crit - abs(z_r) / se)

    return power


def power_ttest_ind(n1, n2, d, alpha=0.05):
    """Calculate power for independent t-test (Cohen's d)."""
    if n1 < 2 or n2 < 2:
        return np.nan

    # Non-centrality parameter
    se = np.sqrt(1/n1 + 1/n2)
    ncp = d / se

    # Degrees of freedom
    df = n1 + n2 - 2

    # Critical t
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Power using non-central t distribution
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    return power


def power_regression_coefficient(n, r2_full, r2_reduced, n_predictors, alpha=0.05):
    """
    Calculate power for testing a regression coefficient (F-test for R² change).
    r2_full: R² of full model
    r2_reduced: R² of model without the predictor
    n_predictors: number of predictors in full model
    """
    if n <= n_predictors + 1:
        return np.nan

    delta_r2 = r2_full - r2_reduced

    # Effect size f²
    f2 = delta_r2 / (1 - r2_full) if r2_full < 1 else np.nan

    if pd.isna(f2) or f2 < 0:
        return np.nan

    # Degrees of freedom
    df1 = 1  # testing one coefficient
    df2 = n - n_predictors - 1

    # Non-centrality parameter
    ncp = f2 * (df1 + df2 + 1)

    # Critical F
    f_crit = stats.f.ppf(1 - alpha, df1, df2)

    # Power using non-central F
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

    return power


def power_interaction(n, r2_interaction, r2_main, n_predictors, alpha=0.05):
    """
    Calculate power for interaction effect.
    Uses same logic as coefficient test but for interaction ΔR².
    """
    return power_regression_coefficient(n, r2_interaction, r2_main, n_predictors, alpha)


def sample_size_for_correlation(r, power=0.80, alpha=0.05):
    """Calculate required N for correlation with given power."""
    if abs(r) >= 1:
        return np.nan

    z_r = 0.5 * np.log((1 + abs(r)) / (1 - abs(r)))
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = ((z_alpha + z_beta) / z_r) ** 2 + 3

    return int(np.ceil(n))


def sample_size_for_ttest(d, power=0.80, alpha=0.05, ratio=1):
    """Calculate required N per group for t-test."""
    if d == 0:
        return np.nan

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) / d) ** 2 * (1 + 1/ratio)

    return int(np.ceil(n))


def minimum_detectable_effect_correlation(n, power=0.80, alpha=0.05):
    """Calculate minimum r detectable with given N and power."""
    # Binary search for r
    low, high = 0.001, 0.999

    for _ in range(50):
        mid = (low + high) / 2
        current_power = power_correlation(n, mid, alpha)

        if current_power < power:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def minimum_detectable_effect_ttest(n1, n2, power=0.80, alpha=0.05):
    """Calculate minimum Cohen's d detectable with given N and power."""
    # Binary search for d
    low, high = 0.001, 3.0

    for _ in range(50):
        mid = (low + high) / 2
        current_power = power_ttest_ind(n1, n2, mid, alpha)

        if pd.isna(current_power):
            break

        if current_power < power:
            low = mid
        else:
            high = mid

    return (low + high) / 2


# ============================================================================
# Load data and observed effect sizes
# ============================================================================
print("\n[1] Loading data and observed effect sizes...")

df = load_master_dataset()

# Sample sizes
N_total = len(df)
N_male = len(df[df['gender'] == 'male'])
N_female = len(df[df['gender'] == 'female'])

print(f"   Total N: {N_total}")
print(f"   Males: {N_male}")
print(f"   Females: {N_female}")

# Load observed effect sizes from previous analysis
effect_sizes_path = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "effect_size_summary" / "effect_size_comprehensive.csv"
if effect_sizes_path.exists():
    observed_effects = pd.read_csv(effect_sizes_path)
    print(f"   Loaded {len(observed_effects)} observed effect sizes")
else:
    observed_effects = pd.DataFrame()
    print("   No pre-computed effect sizes found, will compute directly")


# ============================================================================
# 2. Post-hoc Power Analysis
# ============================================================================
print("\n[2] Computing post-hoc power for observed effects...")

power_results = []

# Key outcomes
outcomes = {
    'pe_rate': 'WCST PE Rate',
    'wcst_accuracy': 'WCST Accuracy',
    'prp_bottleneck': 'PRP Bottleneck',
    'stroop_interference': 'Stroop Interference'
}

# 2a. Correlation power (full sample)
for col, label in outcomes.items():
    if col not in df.columns or 'ucla_score' not in df.columns:
        continue

    valid = df.dropna(subset=[col, 'ucla_score'])
    n = len(valid)
    if n < 5:
        continue

    r, p = stats.pearsonr(valid['ucla_score'], valid[col])

    power = power_correlation(n, r)

    power_results.append({
        'analysis': 'Correlation',
        'outcome': label,
        'subgroup': 'Full Sample',
        'n': n,
        'effect_size': r,
        'effect_type': 'r',
        'power': power,
        'observed_p': p,
        'significant': p < 0.05
    })

# 2b. Correlation power by gender
for gender in ['male', 'female']:
    gender_df = df[df['gender'] == gender]

    for col, label in outcomes.items():
        if col not in gender_df.columns:
            continue

        valid = gender_df.dropna(subset=[col, 'ucla_score'])
        n = len(valid)
        if n < 5:
            continue

        r, p = stats.pearsonr(valid['ucla_score'], valid[col])
        power = power_correlation(n, r)

        power_results.append({
            'analysis': 'Correlation',
            'outcome': label,
            'subgroup': gender.capitalize(),
            'n': n,
            'effect_size': r,
            'effect_type': 'r',
            'power': power,
            'observed_p': p,
            'significant': p < 0.05
        })

# 2c. Group comparison power (Cohen's d)
ucla_median = df['ucla_score'].median()
df['ucla_group'] = df['ucla_score'].apply(lambda x: 'High' if x > ucla_median else 'Low')

for col, label in outcomes.items():
    if col not in df.columns:
        continue

    valid = df.dropna(subset=[col, 'ucla_group'])
    high = valid[valid['ucla_group'] == 'High'][col]
    low = valid[valid['ucla_group'] == 'Low'][col]

    n_high, n_low = len(high), len(low)
    if n_high < 5 or n_low < 5:
        continue

    # Cohen's d
    pooled_std = np.sqrt(((n_high - 1) * high.var() + (n_low - 1) * low.var()) / (n_high + n_low - 2))
    d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0

    t_stat, p = stats.ttest_ind(high, low)
    power = power_ttest_ind(n_high, n_low, d)

    power_results.append({
        'analysis': 'Group Comparison',
        'outcome': label,
        'subgroup': 'Full Sample',
        'n': n_high + n_low,
        'effect_size': d,
        'effect_type': "Cohen's d",
        'power': power,
        'observed_p': p,
        'significant': p < 0.05
    })

# 2d. Interaction power (from hierarchical regression)
hier_path = ANALYSIS_OUTPUT_DIR / "master_dass_controlled" / "hierarchical_regression_results.csv"
if hier_path.exists():
    hier_df = pd.read_csv(hier_path)

    for _, row in hier_df.iterrows():
        if pd.notna(row.get('delta_r2_3v2')) and pd.notna(row.get('model3_r2')):
            # Full model has ~7 predictors: UCLA, Gender, DASS*3, Age, Interaction
            n_pred = 7
            n = row['n']

            power = power_interaction(
                n=n,
                r2_interaction=row['model3_r2'],
                r2_main=row['model2_r2'],
                n_predictors=n_pred
            )

            power_results.append({
                'analysis': 'UCLA × Gender Interaction',
                'outcome': row['outcome'],
                'subgroup': 'Full Sample',
                'n': n,
                'effect_size': row['delta_r2_3v2'],
                'effect_type': 'ΔR²',
                'power': power,
                'observed_p': row['p_3v2'],
                'significant': row['p_3v2'] < 0.05
            })

power_df = pd.DataFrame(power_results)
power_df.to_csv(OUTPUT_DIR / "posthoc_power_analysis.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: posthoc_power_analysis.csv ({len(power_df)} analyses)")


# ============================================================================
# 3. Minimum Detectable Effect Sizes
# ============================================================================
print("\n[3] Computing minimum detectable effect sizes...")

mde_results = []

# For correlations
for subgroup, n in [('Full Sample', N_total), ('Males', N_male), ('Females', N_female)]:
    mde_r = minimum_detectable_effect_correlation(n)
    mde_results.append({
        'analysis': 'Correlation',
        'subgroup': subgroup,
        'n': n,
        'mde_80': mde_r,
        'mde_90': minimum_detectable_effect_correlation(n, power=0.90),
        'effect_type': 'r'
    })

# For t-tests (group comparisons)
mde_d_full = minimum_detectable_effect_ttest(N_total // 2, N_total // 2)
mde_results.append({
    'analysis': 'Group Comparison',
    'subgroup': 'Full Sample',
    'n': N_total,
    'mde_80': mde_d_full,
    'mde_90': minimum_detectable_effect_ttest(N_total // 2, N_total // 2, power=0.90),
    'effect_type': "Cohen's d"
})

mde_df = pd.DataFrame(mde_results)
mde_df.to_csv(OUTPUT_DIR / "minimum_detectable_effects.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: minimum_detectable_effects.csv")


# ============================================================================
# 4. Required Sample Sizes for Future Studies
# ============================================================================
print("\n[4] Computing required sample sizes for various effect sizes...")

required_n_results = []

# Target effect sizes based on literature
target_effects = {
    'Small (r=0.10)': 0.10,
    'Small-Medium (r=0.20)': 0.20,
    'Medium (r=0.30)': 0.30,
    'Medium-Large (r=0.40)': 0.40,
    'Large (r=0.50)': 0.50
}

for label, r in target_effects.items():
    n_80 = sample_size_for_correlation(r, power=0.80)
    n_90 = sample_size_for_correlation(r, power=0.90)

    required_n_results.append({
        'effect_size_label': label,
        'r': r,
        'd_equivalent': 2 * r / np.sqrt(1 - r**2),
        'n_for_80_power': n_80,
        'n_for_90_power': n_90
    })

required_n_df = pd.DataFrame(required_n_results)
required_n_df.to_csv(OUTPUT_DIR / "required_sample_sizes.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: required_sample_sizes.csv")


# ============================================================================
# 5. Power Curves
# ============================================================================
print("\n[5] Creating power curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 5a. Power curve for correlations by sample size
ax = axes[0, 0]
ns = np.arange(20, 301, 5)
effect_sizes_r = [0.10, 0.20, 0.30, 0.40, 0.50]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(effect_sizes_r)))

for r, color in zip(effect_sizes_r, colors):
    powers = [power_correlation(n, r) for n in ns]
    ax.plot(ns, powers, label=f'r = {r}', color=color, linewidth=2)

ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80% power')
ax.axvline(x=N_total, color='green', linestyle=':', alpha=0.7, label=f'Current N = {N_total}')
ax.set_xlabel('Sample Size (N)', fontsize=11)
ax.set_ylabel('Power', fontsize=11)
ax.set_title('Power Curves for Correlation Analysis', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# 5b. Power curve for t-tests by sample size
ax = axes[0, 1]
effect_sizes_d = [0.20, 0.35, 0.50, 0.65, 0.80]
colors = plt.cm.plasma(np.linspace(0, 0.8, len(effect_sizes_d)))

for d, color in zip(effect_sizes_d, colors):
    powers = [power_ttest_ind(n // 2, n // 2, d) for n in ns]
    ax.plot(ns, powers, label=f'd = {d}', color=color, linewidth=2)

ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80% power')
ax.axvline(x=N_total, color='green', linestyle=':', alpha=0.7, label=f'Current N = {N_total}')
ax.set_xlabel('Total Sample Size (N)', fontsize=11)
ax.set_ylabel('Power', fontsize=11)
ax.set_title("Power Curves for Group Comparison (Cohen's d)", fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# 5c. Current sample power by effect size
ax = axes[1, 0]
r_range = np.linspace(0.05, 0.70, 50)
powers_full = [power_correlation(N_total, r) for r in r_range]
powers_male = [power_correlation(N_male, r) for r in r_range]
powers_female = [power_correlation(N_female, r) for r in r_range]

ax.plot(r_range, powers_full, label=f'Full Sample (N={N_total})', linewidth=2, color='blue')
ax.plot(r_range, powers_male, label=f'Males (N={N_male})', linewidth=2, color='green')
ax.plot(r_range, powers_female, label=f'Females (N={N_female})', linewidth=2, color='orange')

ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.7)
ax.fill_between(r_range, 0, powers_full, alpha=0.1, color='blue')
ax.set_xlabel('Effect Size (r)', fontsize=11)
ax.set_ylabel('Power', fontsize=11)
ax.set_title('Power at Current Sample Sizes by Effect Size', fontsize=12, fontweight='bold')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# 5d. Observed effects and their power
ax = axes[1, 1]
if len(power_df) > 0:
    corr_power = power_df[power_df['effect_type'] == 'r'].copy()
    if len(corr_power) > 0:
        corr_power = corr_power.sort_values('effect_size')

        colors = ['#2ecc71' if sig else '#e74c3c' for sig in corr_power['significant']]
        bars = ax.barh(range(len(corr_power)), corr_power['power'], color=colors, alpha=0.7)

        ax.axvline(x=0.80, color='red', linestyle='--', alpha=0.7, label='80% power threshold')
        ax.set_yticks(range(len(corr_power)))
        ax.set_yticklabels([f"{row['outcome']} ({row['subgroup']})"
                           for _, row in corr_power.iterrows()], fontsize=9)
        ax.set_xlabel('Power', fontsize=11)
        ax.set_title('Post-hoc Power for Observed Correlations\n(Green = p < .05, Red = p ≥ .05)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "power_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: power_curves.png")


# ============================================================================
# 6. Sensitivity Analysis Summary
# ============================================================================
print("\n[6] Creating sensitivity analysis summary...")

sensitivity_summary = []

# For each analysis type
for analysis_type in power_df['analysis'].unique():
    subset = power_df[power_df['analysis'] == analysis_type]

    sensitivity_summary.append({
        'Analysis Type': analysis_type,
        'N Analyses': len(subset),
        'Mean Power': subset['power'].mean(),
        'Min Power': subset['power'].min(),
        'Max Power': subset['power'].max(),
        'N Underpowered (<80%)': (subset['power'] < 0.80).sum(),
        'N Adequately Powered (≥80%)': (subset['power'] >= 0.80).sum(),
        'Significant Results': subset['significant'].sum()
    })

sensitivity_df = pd.DataFrame(sensitivity_summary)
sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: sensitivity_summary.csv")


# ============================================================================
# 7. Print Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("POWER ANALYSIS SUMMARY REPORT")
print("=" * 70)

print(f"\nSample sizes:")
print(f"  Total N: {N_total}")
print(f"  Males: {N_male}")
print(f"  Females: {N_female}")

print(f"\nMinimum detectable effect sizes (80% power, α = .05):")
print(f"  Full sample (N={N_total}): r ≥ {minimum_detectable_effect_correlation(N_total):.3f}")
print(f"  Males (N={N_male}): r ≥ {minimum_detectable_effect_correlation(N_male):.3f}")
print(f"  Females (N={N_female}): r ≥ {minimum_detectable_effect_correlation(N_female):.3f}")

print(f"\nPost-hoc power summary:")
if len(power_df) > 0:
    print(f"  Total analyses: {len(power_df)}")
    print(f"  Adequately powered (≥80%): {(power_df['power'] >= 0.80).sum()} ({(power_df['power'] >= 0.80).mean()*100:.1f}%)")
    print(f"  Underpowered (<80%): {(power_df['power'] < 0.80).sum()} ({(power_df['power'] < 0.80).mean()*100:.1f}%)")
    print(f"  Mean power: {power_df['power'].mean():.3f}")

print(f"\nKey findings:")
high_power = power_df[power_df['power'] >= 0.80].sort_values('power', ascending=False)
if len(high_power) > 0:
    print("  Adequately powered analyses:")
    for _, row in high_power.head(5).iterrows():
        sig_mark = '*' if row['significant'] else ''
        print(f"    - {row['outcome']} ({row['subgroup']}): power = {row['power']:.3f} {sig_mark}")

low_power = power_df[power_df['power'] < 0.80].sort_values('power')
if len(low_power) > 0:
    print("  Underpowered analyses (may need larger N):")
    for _, row in low_power.head(5).iterrows():
        print(f"    - {row['outcome']} ({row['subgroup']}): power = {row['power']:.3f}")

print(f"\nRequired sample sizes for future studies (80% power):")
for _, row in required_n_df.iterrows():
    print(f"  {row['effect_size_label']}: N = {row['n_for_80_power']}")

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("=" * 70)

# Save report
with open(OUTPUT_DIR / "power_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write("POWER ANALYSIS SUMMARY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Sample sizes:\n")
    f.write(f"  Total N: {N_total}\n")
    f.write(f"  Males: {N_male}\n")
    f.write(f"  Females: {N_female}\n\n")

    f.write(f"Minimum detectable effect sizes (80% power, α = .05):\n")
    f.write(f"  Full sample (N={N_total}): r ≥ {minimum_detectable_effect_correlation(N_total):.3f}\n")
    f.write(f"  Males (N={N_male}): r ≥ {minimum_detectable_effect_correlation(N_male):.3f}\n")
    f.write(f"  Females (N={N_female}): r ≥ {minimum_detectable_effect_correlation(N_female):.3f}\n\n")

    f.write(f"Post-hoc power summary:\n")
    if len(power_df) > 0:
        f.write(f"  Total analyses: {len(power_df)}\n")
        f.write(f"  Adequately powered (≥80%): {(power_df['power'] >= 0.80).sum()}\n")
        f.write(f"  Underpowered (<80%): {(power_df['power'] < 0.80).sum()}\n")
        f.write(f"  Mean power: {power_df['power'].mean():.3f}\n\n")

    f.write("Required sample sizes for future studies:\n")
    for _, row in required_n_df.iterrows():
        f.write(f"  {row['effect_size_label']}: N = {row['n_for_80_power']} (80%), N = {row['n_for_90_power']} (90%)\n")

print("\nPower analysis complete!")
