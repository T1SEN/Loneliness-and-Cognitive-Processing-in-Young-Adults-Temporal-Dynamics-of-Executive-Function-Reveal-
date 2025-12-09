"""
Synthesis Analysis 7: Power Analysis
======================================

Purpose:
--------
Justify sample size recommendations for replication study based on observed
effect sizes from current study (N=76).

Analyses:
---------
1. Post-hoc power: Power to detect observed effects with N=76
2. Prospective power: Power with proposed N=200 for replication
3. Minimum detectable effect size at 80% power for both sample sizes

Output:
-------
- power_analysis_results.csv: Power calculations for all key effects
- figure_power_curves.png: Power as function of sample size

Author: Research Team
Date: 2025-01-17
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import t, norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/synthesis_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SYNTHESIS ANALYSIS 7: POWER ANALYSIS")
print("=" * 80)
print()
print("Calculating power for observed effects and replication study (N=200)")
print()

# ============================================================================
# OBSERVED EFFECTS FROM PREVIOUS ANALYSES
# ============================================================================

print("[1/3] Loading observed effect sizes...")

# Key effects from synthesis analysis 2 (forest plot)
observed_effects = [
    {
        'effect': 'WCST PE × Gender Interaction',
        'beta': 2.517,
        'se': (4.714 - 0.320) / (2 * 1.96),  # From 95% CI
        'n': 76,
        'df': 76 - 6,  # N - number of predictors
        'outcome_sd': 4.543  # From master dataset
    },
    {
        'effect': 'WCST PE: Male Simple Slope',
        'beta': 2.900,
        'se': (5.289 - 0.511) / (2 * 1.96),
        'n': 30,
        'df': 30 - 5,
        'outcome_sd': 3.84  # Male SD from variance tests
    },
    {
        'effect': 'PRP Bottleneck × Gender',
        'beta': 68.802,
        'se': (132.981 - 4.623) / (2 * 1.96),
        'n': 76,
        'df': 76 - 6,
        'outcome_sd': 145.976  # From master dataset
    },
]

for eff in observed_effects:
    # Calculate standardized effect size (Cohen's f²)
    # f² = R² / (1 - R²)
    # For a single predictor: f² ≈ (β/SD)²
    standardized_beta = eff['beta'] / eff['outcome_sd']
    f_squared = standardized_beta ** 2
    eff['standardized_beta'] = standardized_beta
    eff['f_squared'] = f_squared
    eff['cohen_f'] = np.sqrt(f_squared)

    print(f"\n  {eff['effect']}:")
    print(f"    Unstandardized β = {eff['beta']:.3f}")
    print(f"    Standardized β (d) = {standardized_beta:.3f}")
    print(f"    Cohen's f² = {f_squared:.4f}")
    print(f"    Cohen's f = {eff['cohen_f']:.3f}")

# ============================================================================
# POWER CALCULATIONS
# ============================================================================

print("\n[2/3] Computing power...")

def power_regression(n, f_squared, k, alpha=0.05):
    """
    Calculate power for multiple regression

    Parameters:
    -----------
    n : int
        Sample size
    f_squared : float
        Effect size (Cohen's f²)
    k : int
        Number of predictors
    alpha : float
        Significance level

    Returns:
    --------
    power : float
        Statistical power (1 - β)
    """
    # Degrees of freedom
    df1 = k
    df2 = n - k - 1

    if df2 <= 0:
        return np.nan

    # Non-centrality parameter
    lambda_ncp = n * f_squared

    # Critical F-value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)

    # Power = P(F > f_crit | H1)
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_ncp)

    return power


power_results = []

for eff in observed_effects:
    # Post-hoc power (observed N)
    k_predictors = 5  # Age + Gender + DASS(3) + UCLA or interaction
    power_observed = power_regression(eff['n'], eff['f_squared'], k_predictors)

    # Prospective power (N=200)
    power_n200 = power_regression(200, eff['f_squared'], k_predictors)

    # Minimum detectable effect size (MDES) at 80% power
    # Use binary search to find f² that gives power = 0.80
    def objective(f_sq):
        return power_regression(200, f_sq, k_predictors) - 0.80

    from scipy.optimize import brentq
    try:
        mdes_f_squared = brentq(objective, 0.001, 0.5)
        mdes_d = np.sqrt(mdes_f_squared)
    except:
        mdes_f_squared = np.nan
        mdes_d = np.nan

    power_results.append({
        'effect': eff['effect'],
        'observed_n': eff['n'],
        'observed_beta': eff['beta'],
        'standardized_d': eff['standardized_beta'],
        'cohen_f_squared': eff['f_squared'],
        'power_observed_n': power_observed,
        'power_n200': power_n200,
        'mdes_d_at_n200_power80': mdes_d,
        'mdes_f_squared_at_n200_power80': mdes_f_squared
    })

    print(f"\n  {eff['effect']}:")
    print(f"    Current N={eff['n']}: Power = {power_observed:.3f} ({power_observed*100:.1f}%)")
    print(f"    Proposed N=200: Power = {power_n200:.3f} ({power_n200*100:.1f}%)")
    if not np.isnan(mdes_d):
        print(f"    MDES at N=200 (80% power): d = {mdes_d:.3f}")

# Save results
power_df = pd.DataFrame(power_results)
power_df.to_csv(OUTPUT_DIR / "power_analysis_results.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: power_analysis_results.csv")

# ============================================================================
# VISUALIZATION: POWER CURVES
# ============================================================================

print("\n[3/3] Creating power curves...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sample_sizes = np.arange(30, 300, 10)

for idx, (eff, ax) in enumerate(zip(observed_effects, axes)):
    powers = [power_regression(n, eff['f_squared'], 5) for n in sample_sizes]

    ax.plot(sample_sizes, powers, linewidth=2.5, color='#0173B2')

    # Mark current N
    current_power = power_regression(eff['n'], eff['f_squared'], 5)
    ax.scatter([eff['n']], [current_power], s=150, color='#E74C3C',
              edgecolor='black', linewidth=2, zorder=10,
              label=f'Current N={eff["n"]} (Power={current_power:.2f})')

    # Mark proposed N=200
    power_200 = power_regression(200, eff['f_squared'], 5)
    ax.scatter([200], [power_200], s=150, color='#2ECC71',
              edgecolor='black', linewidth=2, zorder=10,
              label=f'Proposed N=200 (Power={power_200:.2f})')

    # Reference lines
    ax.axhline(0.80, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
              label='80% Power')
    ax.axhline(0.95, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
              label='95% Power')

    ax.set_xlabel('Sample Size (N)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Statistical Power', fontweight='bold', fontsize=11)
    ax.set_title(f"{eff['effect']}\n(d={eff['standardized_beta']:.2f}, f²={eff['f_squared']:.3f})",
                fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='lower right', frameon=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(30, 300)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_power_curves.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure_power_curves.png")
plt.close()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SYNTHESIS ANALYSIS 7 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - power_analysis_results.csv")
print("  - figure_power_curves.png")
print()

print("SAMPLE SIZE RECOMMENDATIONS:")
print()

for _, row in power_df.iterrows():
    print(f"  {row['effect']}:")
    if row['power_observed_n'] < 0.80:
        print(f"    ⚠ Current study UNDERPOWERED (power={row['power_observed_n']:.2f})")
    else:
        print(f"    ✓ Current study adequately powered (power={row['power_observed_n']:.2f})")

    if row['power_n200'] >= 0.80:
        print(f"    ✓ N=200 provides adequate power (power={row['power_n200']:.2f})")
    else:
        print(f"    ⚠ N=200 still underpowered (power={row['power_n200']:.2f})")
        # Calculate required N for 80% power
        def find_n_for_power(target_power, f_sq):
            for n_test in range(50, 1000, 10):
                if power_regression(n_test, f_sq, 5) >= target_power:
                    return n_test
            return np.nan

        required_n = find_n_for_power(0.80, row['cohen_f_squared'])
        if not np.isnan(required_n):
            print(f"       Required N for 80% power: {required_n}")
    print()
