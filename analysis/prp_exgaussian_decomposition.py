"""
PRP Ex-Gaussian RT Decomposition
=================================
Decomposes T2 RT distributions into Ex-Gaussian parameters:
- μ (mu): Gaussian mean (routine processing speed)
- σ (sigma): Gaussian SD (processing variability)
- τ (tau): Exponential component (attentional lapses, slow tail)

Tests UCLA × Gender effects on each parameter across SOA conditions.

Key Question:
- Is PRP bottleneck effect (short vs long SOA) driven by μ, σ, or τ changes?
- Does loneliness amplify specific Ex-Gaussian components?
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import exponnorm, pearsonr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/mechanism_analysis/exgaussian")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRP EX-GAUSSIAN RT DECOMPOSITION")
print("=" * 80)

trials, _ = load_prp_trials(
    use_cache=True,
    rt_min=200,
    rt_max=5000,
    require_t1_correct=False,
    require_t2_correct_for_rt=False,
    enforce_short_long_only=False,
    drop_timeouts=True,
)
trials.columns = trials.columns.str.lower()
if "soa_nominal_ms" not in trials.columns and "soa" in trials.columns:
    trials["soa_nominal_ms"] = trials["soa"]
if "t2_rt_ms" not in trials.columns and "t2_rt" in trials.columns:
    trials["t2_rt_ms"] = trials["t2_rt"]

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

# Clean
rt_col_t2 = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 't2_rt'

trials_clean = trials[
    (trials['t1_correct'].notna()) &
    (trials['t2_correct'].notna()) &
    (trials[rt_col_t2].notna()) &
    (trials[rt_col_t2] > 200) &  # Remove too-fast RTs
    (trials[rt_col_t2] < 5000)  # Remove too-slow RTs
].copy()

trials_clean['t2_rt'] = trials_clean[rt_col_t2]

# SOA categorization
soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in trials_clean.columns else 'soa'

def categorize_soa(soa):
    if soa <= 150:
        return 'short'
    elif 300 <= soa <= 600:
        return 'medium'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'

trials_clean['soa_cat'] = trials_clean[soa_col].apply(categorize_soa)
trials_clean = trials_clean[trials_clean['soa_cat'] != 'other']

print(f"\n[1] Valid trials: {len(trials_clean)}")
print(f"  Short SOA: {len(trials_clean[trials_clean['soa_cat']=='short'])}")
print(f"  Medium SOA: {len(trials_clean[trials_clean['soa_cat']=='medium'])}")
print(f"  Long SOA: {len(trials_clean[trials_clean['soa_cat']=='long'])}")

# Ex-Gaussian fitting function
def fit_exgaussian(rts):
    """
    Fit Ex-Gaussian distribution to RT data.

    Ex-Gaussian = Gaussian(μ, σ) + Exponential(τ)
    scipy.stats.exponnorm uses K = τ/σ parameterization
    """
    if len(rts) < 20:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}

    rts = np.array(rts)

    # Method of moments initial estimates
    m = np.mean(rts)
    s = np.std(rts)
    skew = np.mean(((rts - m) / s) ** 3)

    # Initial tau estimate from skewness
    tau_init = max(10, (skew / 2) ** (1/3) * s)
    mu_init = m - tau_init
    sigma_init = max(10, np.sqrt(max(0, s**2 - tau_init**2)))

    # Constrain to positive values
    mu_init = max(100, mu_init)
    sigma_init = max(10, sigma_init)
    tau_init = max(10, tau_init)

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

    # Optimize
    try:
        result = minimize(
            neg_loglik,
            x0=[mu_init, sigma_init, tau_init],
            method='L-BFGS-B',
            bounds=[(100, 4000), (5, 1000), (5, 2000)]
        )

        if result.success:
            mu, sigma, tau = result.x
            return {'mu': mu, 'sigma': sigma, 'tau': tau, 'n': len(rts)}
        else:
            return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}
    except:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}

# Fit Ex-Gaussian for each participant × SOA condition
def compute_exgaussian_params(group):
    results = {}

    for soa_cat in ['short', 'medium', 'long']:
        soa_trials = group[group['soa_cat'] == soa_cat]
        params = fit_exgaussian(soa_trials['t2_rt'].values)

        results[f'{soa_cat}_mu'] = params['mu']
        results[f'{soa_cat}_sigma'] = params['sigma']
        results[f'{soa_cat}_tau'] = params['tau']
        results[f'{soa_cat}_n'] = params['n']

    return pd.Series(results)

print("\n[2] Fitting Ex-Gaussian distributions...")
exgauss_df = trials_clean.groupby('participant_id').apply(compute_exgaussian_params).reset_index()

exgauss_df = exgauss_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Fitted N={len(exgauss_df)} participants")
print(f"\n  Short SOA:")
print(f"    μ: {exgauss_df['short_mu'].mean():.1f} ms")
print(f"    σ: {exgauss_df['short_sigma'].mean():.1f} ms")
print(f"    τ: {exgauss_df['short_tau'].mean():.1f} ms")
print(f"\n  Long SOA:")
print(f"    μ: {exgauss_df['long_mu'].mean():.1f} ms")
print(f"    σ: {exgauss_df['long_sigma'].mean():.1f} ms")
print(f"    τ: {exgauss_df['long_tau'].mean():.1f} ms")

exgauss_df.to_csv(OUTPUT_DIR / "prp_exgaussian_parameters.csv", index=False, encoding='utf-8-sig')

# Gender-stratified correlations
print("\n[3] Testing UCLA × Gender effects on Ex-Gaussian parameters...")

results = []
for gender in ['male', 'female']:
    data = exgauss_df[exgauss_df['gender'] == gender]

    for soa_cat in ['short', 'medium', 'long']:
        for param in ['mu', 'sigma', 'tau']:
            col = f'{soa_cat}_{param}'
            valid = data.dropna(subset=['ucla_total', col])

            if len(valid) >= 10:
                r, p = pearsonr(valid['ucla_total'], valid[col])
                results.append({
                    'gender': gender,
                    'soa': soa_cat,
                    'parameter': param,
                    'n': len(valid),
                    'r': r,
                    'p': p,
                    'mean': valid[col].mean(),
                    'sd': valid[col].std()
                })

corr_df = pd.DataFrame(results)
print("\nSignificant effects (p < 0.10):")
print(corr_df[corr_df['p'] < 0.10])

corr_df.to_csv(OUTPUT_DIR / "prp_exgaussian_gender_correlations.csv", index=False, encoding='utf-8-sig')

# Bottleneck effects on Ex-Gaussian components
exgauss_df['mu_bottleneck'] = exgauss_df['short_mu'] - exgauss_df['long_mu']
exgauss_df['sigma_bottleneck'] = exgauss_df['short_sigma'] - exgauss_df['long_sigma']
exgauss_df['tau_bottleneck'] = exgauss_df['short_tau'] - exgauss_df['long_tau']

print("\n[4] Testing bottleneck effects on Ex-Gaussian components...")

bottleneck_results = []
for gender in ['male', 'female']:
    data = exgauss_df[exgauss_df['gender'] == gender]

    for param in ['mu', 'sigma', 'tau']:
        col = f'{param}_bottleneck'
        valid = data.dropna(subset=['ucla_total', col])

        if len(valid) >= 10:
            r, p = pearsonr(valid['ucla_total'], valid[col])
            bottleneck_results.append({
                'gender': gender,
                'parameter': f'{param}_bottleneck',
                'n': len(valid),
                'r': r,
                'p': p,
                'mean': valid[col].mean(),
                'sd': valid[col].std()
            })

bottleneck_df = pd.DataFrame(bottleneck_results)
print(bottleneck_df)

bottleneck_df.to_csv(OUTPUT_DIR / "prp_exgaussian_bottleneck_correlations.csv", index=False, encoding='utf-8-sig')

# Summary
print("\n" + "="*80)
print("PRP EX-GAUSSIAN DECOMPOSITION - KEY FINDINGS")
print("="*80)

print(f"""
1. Ex-Gaussian Parameters by SOA:

   Short SOA (≤150ms):
   - μ (routine): {exgauss_df['short_mu'].mean():.1f} ± {exgauss_df['short_mu'].std():.1f} ms
   - σ (variability): {exgauss_df['short_sigma'].mean():.1f} ± {exgauss_df['short_sigma'].std():.1f} ms
   - τ (lapses): {exgauss_df['short_tau'].mean():.1f} ± {exgauss_df['short_tau'].std():.1f} ms

   Long SOA (≥1200ms):
   - μ (routine): {exgauss_df['long_mu'].mean():.1f} ± {exgauss_df['long_mu'].std():.1f} ms
   - σ (variability): {exgauss_df['long_sigma'].mean():.1f} ± {exgauss_df['long_sigma'].std():.1f} ms
   - τ (lapses): {exgauss_df['long_tau'].mean():.1f} ± {exgauss_df['long_tau'].std():.1f} ms

2. Bottleneck Effect Decomposition:
   - μ bottleneck: {exgauss_df['mu_bottleneck'].mean():.1f} ms (routine slowing at short SOA)
   - σ bottleneck: {exgauss_df['sigma_bottleneck'].mean():.1f} ms (variability change)
   - τ bottleneck: {exgauss_df['tau_bottleneck'].mean():.1f} ms (lapse change)

3. Gender-Stratified Correlations (p < 0.10):
{corr_df[corr_df['p'] < 0.10].to_string(index=False) if len(corr_df[corr_df['p'] < 0.10]) > 0 else "   No significant effects"}

4. Bottleneck Effects on Ex-Gaussian Components:
{bottleneck_df.to_string(index=False)}

5. Interpretation:
   - μ bottleneck effects → Loneliness slows central processing
   - σ bottleneck effects → Loneliness increases dual-task variability
   - τ bottleneck effects → Loneliness increases attentional lapses under load

6. Link to Phase 3 Finding:
   - Phase 3: Males showed T2 RT SD × UCLA r=0.519, p=0.006**
   - Current: Which Ex-Gaussian component (σ or τ) drives this variability?

7. Files Generated:
   - prp_exgaussian_parameters.csv
   - prp_exgaussian_gender_correlations.csv
   - prp_exgaussian_bottleneck_correlations.csv
""")

print("\n✅ PRP Ex-Gaussian decomposition complete!")
print("="*80)
