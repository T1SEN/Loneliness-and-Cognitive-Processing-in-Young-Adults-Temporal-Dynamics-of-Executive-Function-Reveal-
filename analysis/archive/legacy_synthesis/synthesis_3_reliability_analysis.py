"""
Synthesis Analysis 3: Comprehensive Reliability Analysis
==========================================================

Purpose:
--------
Psychometric quality assessment of three key executive function metrics:
1. PRP τ (tau) at long SOA - Ex-Gaussian exponential component
2. WCST Perseverative Error Rate - Cognitive flexibility index
3. Stroop Interference - Inhibitory control measure

Statistical Approach:
--------------------
- Split-half reliability (odd vs even trials)
- Spearman-Brown correction: r_corrected = 2r / (1+r)
- Bootstrap 95% CI for reliability coefficients (10,000 iterations)
- Internal consistency diagnostics (CV, range, trial counts)

Output:
-------
- reliability_coefficients.csv: Reliability metrics for all measures
- figure_reliability_scatter.png: Split-half consistency plots

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
from scipy.optimize import minimize
from scipy.stats import exponnorm
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
warnings.filterwarnings('ignore')

# Import local utilities
sys.path.append('analysis')
from analysis.statistical_utils import bootstrap_ci
from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials, load_wcst_trials, load_stroop_trials

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/synthesis_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SYNTHESIS ANALYSIS 3: RELIABILITY ANALYSIS")
print("=" * 80)
print()
print("Computing split-half reliability for three EF metrics:")
print("  1. PRP τ (long SOA)")
print("  2. WCST Perseverative Error Rate")
print("  3. Stroop Interference")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fit_exgaussian(rts):
    """Fit Ex-Gaussian distribution to RTs, return tau parameter"""
    if len(rts) < 15:
        return np.nan

    rts = np.array(rts)
    rts = rts[(rts > 100) & (rts < 3000)]

    if len(rts) < 15:
        return np.nan

    # Initial parameter estimates
    m = np.mean(rts)
    s = np.std(rts)
    skew = stats.skew(rts)

    tau_init = max(10, (abs(skew) / 2) ** (1/3) * s)
    mu_init = max(100, m - tau_init)
    sigma_init = max(10, np.sqrt(max(0, s**2 - tau_init**2)))

    def neg_loglik(params):
        mu, sigma, tau = params
        if sigma <= 0 or tau <= 0:
            return 1e10
        K = tau / sigma
        try:
            loglik = np.sum(exponnorm.logpdf(rts, K, loc=mu, scale=sigma))
            return -loglik
        except:
            return 1e10

    try:
        result = minimize(
            neg_loglik,
            x0=[mu_init, sigma_init, tau_init],
            method='L-BFGS-B',
            bounds=[(100, 2000), (5, 500), (5, 1000)]
        )

        if result.success:
            mu, sigma, tau = result.x
            return tau
        else:
            return np.nan
    except:
        return np.nan


def spearman_brown_correction(r):
    """Apply Spearman-Brown prophecy formula for split-half reliability"""
    if pd.isna(r) or r <= 0:
        return np.nan
    return (2 * r) / (1 + r)


def bootstrap_correlation(half1, half2, n_bootstrap=10000):
    """Bootstrap CI for split-half correlation"""
    def corr_func(indices):
        return np.corrcoef(half1[indices], half2[indices])[0, 1]

    n = len(half1)
    rng = np.random.default_rng(42)

    boot_corrs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        r_boot = corr_func(indices)
        if not np.isnan(r_boot):
            boot_corrs.append(r_boot)

    if len(boot_corrs) > 0:
        ci_lower = np.percentile(boot_corrs, 2.5)
        ci_upper = np.percentile(boot_corrs, 97.5)
        return ci_lower, ci_upper
    else:
        return np.nan, np.nan


# ============================================================================
# 1. PRP TAU RELIABILITY
# ============================================================================

print("[1/3] Computing PRP τ (long SOA) split-half reliability...")

prp_trials, _ = load_prp_trials(
    use_cache=True,
    rt_min=200,
    rt_max=5000,
    require_t1_correct=False,
    require_t2_correct_for_rt=False,
    enforce_short_long_only=False,
    drop_timeouts=True,
)
if "t2_rt" not in prp_trials.columns and "t2_rt_ms" in prp_trials.columns:
    prp_trials["t2_rt"] = prp_trials["t2_rt_ms"]

# SOA categorization
def categorize_soa(soa):
    if soa <= 150:
        return 'short'
    elif 300 <= soa <= 600:
        return 'medium'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'

soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in prp_trials.columns else 'soa'
prp_trials['soa_cat'] = prp_trials[soa_col].apply(categorize_soa)

# Focus on long SOA
prp_long = prp_trials[prp_trials['soa_cat'] == 'long'].copy()

# Remove duplicate columns if any
prp_long = prp_long.loc[:, ~prp_long.columns.duplicated()]

# Add trial index per participant
if len(prp_long) > 0:
    prp_long = prp_long.sort_values(['participant_id', 'trial_index']).reset_index(drop=True)
    prp_long['trial_idx'] = prp_long.groupby('participant_id').cumcount()

    # Split odd/even trials
    prp_odd = prp_long[prp_long['trial_idx'] % 2 == 0]
    prp_even = prp_long[prp_long['trial_idx'] % 2 == 1]
else:
    print("  ⚠ No PRP trials at long SOA found")
    prp_tau_result = None
    prp_odd = pd.DataFrame()
    prp_even = pd.DataFrame()

# Compute tau for each participant in each half
if len(prp_odd) > 0:
    print("  Fitting Ex-Gaussian to odd trials...")
    tau_odd = []
    grouped_odd = prp_odd.groupby('participant_id')
    for pid in grouped_odd.groups.keys():
        rts = prp_odd[prp_odd['participant_id'] == pid]['t2_rt'].values
        tau = fit_exgaussian(rts)
        tau_odd.append({'participant_id': pid, 'tau_odd': tau, 'n_odd': len(rts)})

    tau_odd_df = pd.DataFrame(tau_odd)
else:
    tau_odd_df = pd.DataFrame(columns=['participant_id', 'tau_odd', 'n_odd'])

if len(prp_even) > 0:
    print("  Fitting Ex-Gaussian to even trials...")
    tau_even = []
    grouped_even = prp_even.groupby('participant_id')
    for pid in grouped_even.groups.keys():
        rts = prp_even[prp_even['participant_id'] == pid]['t2_rt'].values
        tau = fit_exgaussian(rts)
        tau_even.append({'participant_id': pid, 'tau_even': tau, 'n_even': len(rts)})

    tau_even_df = pd.DataFrame(tau_even)
else:
    tau_even_df = pd.DataFrame(columns=['participant_id', 'tau_even', 'n_even'])

# Merge
tau_reliability = tau_odd_df.merge(tau_even_df, on='participant_id', how='inner')
tau_reliability = tau_reliability.dropna(subset=['tau_odd', 'tau_even'])

if len(tau_reliability) >= 10:
    r_tau = np.corrcoef(tau_reliability['tau_odd'], tau_reliability['tau_even'])[0, 1]
    r_tau_corrected = spearman_brown_correction(r_tau)

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_correlation(
        tau_reliability['tau_odd'].values,
        tau_reliability['tau_even'].values,
        n_bootstrap=10000
    )

    print(f"  N = {len(tau_reliability)}")
    print(f"  Split-half r = {r_tau:.3f}")
    print(f"  Spearman-Brown corrected r = {r_tau_corrected:.3f}")
    print(f"  95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")

    prp_tau_result = {
        'metric': 'PRP τ (long SOA)',
        'n': len(tau_reliability),
        'split_half_r': r_tau,
        'spearman_brown_r': r_tau_corrected,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_trials_per_half': (tau_reliability['n_odd'].mean() + tau_reliability['n_even'].mean()) / 2
    }
else:
    print(f"  ⚠ Insufficient data (N={len(tau_reliability)})")
    prp_tau_result = None

# ============================================================================
# 2. WCST PE RATE RELIABILITY
# ============================================================================

print("\n[2/3] Computing WCST PE rate split-half reliability...")

wcst_trials, _ = load_wcst_trials(use_cache=True)

# Ensure is_pe exists
if "is_pe" not in wcst_trials.columns:
    def _parse_wcst_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except (ValueError, SyntaxError):
            return {}
    wcst_trials["extra_dict"] = wcst_trials["extra"].apply(_parse_wcst_extra) if "extra" in wcst_trials.columns else {}
    wcst_trials["is_pe"] = wcst_trials.get("extra_dict", {}).apply(lambda x: x.get("isPE", False) if isinstance(x, dict) else False)

# Add trial index per participant
# Check for trial column name
trial_col = 'trial' if 'trial' in wcst_trials.columns else 'trialIndex' if 'trialIndex' in wcst_trials.columns else None
if trial_col is None:
    print(f"  ⚠ No trial column found. Available columns: {wcst_trials.columns.tolist()[:10]}")
    wcst_pe_result = None
elif len(wcst_trials) > 0:
    wcst_trials = wcst_trials.sort_values(['participant_id', trial_col]).reset_index(drop=True)
    wcst_trials['trial_idx'] = wcst_trials.groupby('participant_id').cumcount()

    # Split odd/even
    wcst_odd = wcst_trials[wcst_trials['trial_idx'] % 2 == 0]
    wcst_even = wcst_trials[wcst_trials['trial_idx'] % 2 == 1]

    # Compute PE rate for each half
    pe_odd = wcst_odd.groupby('participant_id').agg(
        pe_rate_odd=('is_pe', lambda x: x.sum() / len(x) * 100),
        n_odd=('is_pe', 'count')
    ).reset_index()

    pe_even = wcst_even.groupby('participant_id').agg(
        pe_rate_even=('is_pe', lambda x: x.sum() / len(x) * 100),
        n_even=('is_pe', 'count')
    ).reset_index()

    # Merge
    pe_reliability = pe_odd.merge(pe_even, on='participant_id', how='inner')

    # Filter minimum trials
    pe_reliability = pe_reliability[(pe_reliability['n_odd'] >= 10) & (pe_reliability['n_even'] >= 10)]

    if len(pe_reliability) >= 10:
        r_pe = np.corrcoef(pe_reliability['pe_rate_odd'], pe_reliability['pe_rate_even'])[0, 1]
        r_pe_corrected = spearman_brown_correction(r_pe)

        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_correlation(
            pe_reliability['pe_rate_odd'].values,
            pe_reliability['pe_rate_even'].values,
            n_bootstrap=10000
        )

        print(f"  N = {len(pe_reliability)}")
        print(f"  Split-half r = {r_pe:.3f}")
        print(f"  Spearman-Brown corrected r = {r_pe_corrected:.3f}")
        print(f"  95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")

        wcst_pe_result = {
            'metric': 'WCST PE Rate',
            'n': len(pe_reliability),
            'split_half_r': r_pe,
            'spearman_brown_r': r_pe_corrected,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_trials_per_half': (pe_reliability['n_odd'].mean() + pe_reliability['n_even'].mean()) / 2
        }
    else:
        print(f"  ⚠ Insufficient data (N={len(pe_reliability)})")
        wcst_pe_result = None
else:
    wcst_pe_result = None

# ============================================================================
# 3. STROOP INTERFERENCE RELIABILITY
# ============================================================================

print("\n[3/3] Computing Stroop interference split-half reliability...")

stroop_trials, _ = load_stroop_trials(
    use_cache=True,
    rt_min=200,
    rt_max=3000,
    drop_timeouts=True,
    require_correct_for_rt=True,
)
if "rt_ms" not in stroop_trials.columns and "rt" in stroop_trials.columns:
    stroop_trials["rt_ms"] = stroop_trials["rt"]

# Add trial index per participant
stroop_trials = stroop_trials.sort_values(['participant_id', 'trial']).reset_index(drop=True)
stroop_trials['trial_idx'] = stroop_trials.groupby('participant_id').cumcount()

# Split odd/even
stroop_odd = stroop_trials[stroop_trials['trial_idx'] % 2 == 0]
stroop_even = stroop_trials[stroop_trials['trial_idx'] % 2 == 1]

# Compute interference for each half
def compute_stroop_interference(df):
    by_type = df.groupby(['participant_id', 'type'])['rt_ms'].mean().unstack(fill_value=np.nan)
    if 'congruent' in by_type.columns and 'incongruent' in by_type.columns:
        interference = (by_type['incongruent'] - by_type['congruent']).reset_index()
        interference.columns = ['participant_id', 'interference']
        return interference
    else:
        return pd.DataFrame(columns=['participant_id', 'interference'])

stroop_odd_int = compute_stroop_interference(stroop_odd)
stroop_odd_int.columns = ['participant_id', 'interference_odd']

stroop_even_int = compute_stroop_interference(stroop_even)
stroop_even_int.columns = ['participant_id', 'interference_even']

# Merge
stroop_reliability = stroop_odd_int.merge(stroop_even_int, on='participant_id', how='inner')
stroop_reliability = stroop_reliability.dropna()

if len(stroop_reliability) >= 10:
    r_stroop = np.corrcoef(stroop_reliability['interference_odd'],
                          stroop_reliability['interference_even'])[0, 1]
    r_stroop_corrected = spearman_brown_correction(r_stroop)

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_correlation(
        stroop_reliability['interference_odd'].values,
        stroop_reliability['interference_even'].values,
        n_bootstrap=10000
    )

    print(f"  N = {len(stroop_reliability)}")
    print(f"  Split-half r = {r_stroop:.3f}")
    print(f"  Spearman-Brown corrected r = {r_stroop_corrected:.3f}")
    print(f"  95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")

    stroop_result = {
        'metric': 'Stroop Interference',
        'n': len(stroop_reliability),
        'split_half_r': r_stroop,
        'spearman_brown_r': r_stroop_corrected,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_trials_per_half': np.nan  # Not applicable for aggregated metric
    }
else:
    print(f"  ⚠ Insufficient data (N={len(stroop_reliability)})")
    stroop_result = None

# ============================================================================
# SAVE RELIABILITY COEFFICIENTS
# ============================================================================

print("\n[4/4] Saving results...")

reliability_results = []
if prp_tau_result is not None:
    reliability_results.append(prp_tau_result)
if wcst_pe_result is not None:
    reliability_results.append(wcst_pe_result)
if stroop_result is not None:
    reliability_results.append(stroop_result)

if len(reliability_results) > 0:
    reliability_df = pd.DataFrame(reliability_results)
    reliability_df.to_csv(OUTPUT_DIR / "reliability_coefficients.csv", index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved: reliability_coefficients.csv")

# ============================================================================
# VISUALIZATION: SCATTER PLOTS
# ============================================================================

print("\n[5/5] Creating scatter plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: PRP Tau
if prp_tau_result is not None and len(tau_reliability) > 0:
    ax = axes[0]
    ax.scatter(tau_reliability['tau_odd'], tau_reliability['tau_even'],
              alpha=0.6, s=80, color='#3498DB', edgecolor='black', linewidth=1)

    # Regression line
    z = np.polyfit(tau_reliability['tau_odd'], tau_reliability['tau_even'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(tau_reliability['tau_odd'].min(), tau_reliability['tau_odd'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r = {r_tau:.3f}')

    # Identity line
    max_val = max(tau_reliability['tau_odd'].max(), tau_reliability['tau_even'].max())
    min_val = min(tau_reliability['tau_odd'].min(), tau_reliability['tau_even'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel('τ Odd Trials (ms)', fontweight='bold', fontsize=11)
    ax.set_ylabel('τ Even Trials (ms)', fontweight='bold', fontsize=11)
    ax.set_title(f'PRP τ (Long SOA) Split-Half\nr = {r_tau:.3f}, SB r = {r_tau_corrected:.3f}',
                fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
    axes[0].set_title('PRP τ (Long SOA)', fontweight='bold')

# Plot 2: WCST PE Rate
if wcst_pe_result is not None and len(pe_reliability) > 0:
    ax = axes[1]
    ax.scatter(pe_reliability['pe_rate_odd'], pe_reliability['pe_rate_even'],
              alpha=0.6, s=80, color='#E74C3C', edgecolor='black', linewidth=1)

    # Regression line
    z = np.polyfit(pe_reliability['pe_rate_odd'], pe_reliability['pe_rate_even'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(pe_reliability['pe_rate_odd'].min(), pe_reliability['pe_rate_odd'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r = {r_pe:.3f}')

    # Identity line
    max_val = max(pe_reliability['pe_rate_odd'].max(), pe_reliability['pe_rate_even'].max())
    min_val = min(pe_reliability['pe_rate_odd'].min(), pe_reliability['pe_rate_even'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel('PE Rate Odd Trials (%)', fontweight='bold', fontsize=11)
    ax.set_ylabel('PE Rate Even Trials (%)', fontweight='bold', fontsize=11)
    ax.set_title(f'WCST PE Rate Split-Half\nr = {r_pe:.3f}, SB r = {r_pe_corrected:.3f}',
                fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
    axes[1].set_title('WCST PE Rate', fontweight='bold')

# Plot 3: Stroop Interference
if stroop_result is not None and len(stroop_reliability) > 0:
    ax = axes[2]
    ax.scatter(stroop_reliability['interference_odd'], stroop_reliability['interference_even'],
              alpha=0.6, s=80, color='#2ECC71', edgecolor='black', linewidth=1)

    # Regression line
    z = np.polyfit(stroop_reliability['interference_odd'], stroop_reliability['interference_even'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(stroop_reliability['interference_odd'].min(),
                        stroop_reliability['interference_odd'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r = {r_stroop:.3f}')

    # Identity line
    max_val = max(stroop_reliability['interference_odd'].max(), stroop_reliability['interference_even'].max())
    min_val = min(stroop_reliability['interference_odd'].min(), stroop_reliability['interference_even'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Interference Odd Trials (ms)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Interference Even Trials (ms)', fontweight='bold', fontsize=11)
    ax.set_title(f'Stroop Interference Split-Half\nr = {r_stroop:.3f}, SB r = {r_stroop_corrected:.3f}',
                fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
    axes[2].set_title('Stroop Interference', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_reliability_scatter.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure_reliability_scatter.png")
plt.close()

# ============================================================================
# DONE
# ============================================================================

print()
print("=" * 80)
print("SYNTHESIS ANALYSIS 3 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - reliability_coefficients.csv")
print("  - figure_reliability_scatter.png")
print()

if len(reliability_results) > 0:
    print("SUMMARY:")
    for result in reliability_results:
        print(f"  {result['metric']}:")
        print(f"    N = {result['n']}")
        print(f"    Split-half r = {result['split_half_r']:.3f}")
        print(f"    Spearman-Brown r = {result['spearman_brown_r']:.3f}")
        print(f"    95% CI = [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
        print()
