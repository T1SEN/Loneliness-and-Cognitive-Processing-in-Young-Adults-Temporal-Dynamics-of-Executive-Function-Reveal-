"""
Stroop Ex-Gaussian RT Decomposition
====================================
Decomposes RT distributions into Ex-Gaussian parameters:
- μ (mu): Gaussian mean (routine processing speed)
- σ (sigma): Gaussian SD (processing variability)
- τ (tau): Exponential component (attentional lapses, slow tail)

Tests UCLA × Gender effects on each parameter for congruent vs incongruent trials.

Rationale:
- If loneliness affects μ → slowed routine processing
- If loneliness affects σ → increased processing inconsistency
- If loneliness affects τ → more frequent attentional lapses
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import exponnorm, pearsonr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/mechanism_analysis/exgaussian")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STROOP EX-GAUSSIAN RT DECOMPOSITION")
print("=" * 80)

# Load
trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
trials.columns = trials.columns.str.lower()
if 'participantid' in trials.columns and 'participant_id' in trials.columns:
    trials = trials.drop(columns=['participantid'])
elif 'participantid' in trials.columns:
    trials.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()
# Map Korean gender values to English
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
master['gender'] = master['gender'].map(gender_map)

# Clean
rt_col = 'rt_ms' if trials['rt'].isnull().sum() > len(trials) * 0.5 else 'rt'
trials_clean = trials[
    (trials['is_timeout'] == False) &
    (trials[rt_col].notna()) &
    (trials[rt_col] > 200) &  # Remove too-fast RTs
    (trials[rt_col] < 3000) &  # Remove too-slow RTs
    (trials['type'].isin(['congruent', 'incongruent']))
].copy()
trials_clean['rt'] = trials_clean[rt_col]

print(f"\n[1] Valid trials: {len(trials_clean)}")

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
    sigma_init = max(10, np.sqrt(s**2 - tau_init**2))

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
            bounds=[(100, 2000), (5, 500), (5, 1000)]
        )

        if result.success:
            mu, sigma, tau = result.x
            return {'mu': mu, 'sigma': sigma, 'tau': tau, 'n': len(rts)}
        else:
            return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}
    except:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}

# Fit Ex-Gaussian for each participant × condition
def compute_exgaussian_params(group):
    results = {}

    for condition in ['congruent', 'incongruent']:
        cond_trials = group[group['type'] == condition]
        params = fit_exgaussian(cond_trials['rt'].values)

        results[f'{condition}_mu'] = params['mu']
        results[f'{condition}_sigma'] = params['sigma']
        results[f'{condition}_tau'] = params['tau']
        results[f'{condition}_n'] = params['n']

    return pd.Series(results)

print("\n[2] Fitting Ex-Gaussian distributions...")
exgauss_df = trials_clean.groupby('participant_id').apply(compute_exgaussian_params).reset_index()

exgauss_df = exgauss_df.merge(
    master[['participant_id', 'ucla_total', 'gender', 'dass_depression', 'dass_anxiety', 'dass_stress']],
    on='participant_id',
    how='inner'
)

print(f"  Fitted N={len(exgauss_df)} participants")
print(f"  Mean congruent μ: {exgauss_df['congruent_mu'].mean():.1f} ms")
print(f"  Mean congruent σ: {exgauss_df['congruent_sigma'].mean():.1f} ms")
print(f"  Mean congruent τ: {exgauss_df['congruent_tau'].mean():.1f} ms")

exgauss_df.to_csv(OUTPUT_DIR / "exgaussian_parameters.csv", index=False, encoding='utf-8-sig')

# Gender-stratified correlations
print("\n[3] Testing UCLA × Gender effects on Ex-Gaussian parameters...")

results = []
for gender in ['male', 'female']:
    data = exgauss_df[exgauss_df['gender'] == gender]

    for condition in ['congruent', 'incongruent']:
        for param in ['mu', 'sigma', 'tau']:
            col = f'{condition}_{param}'
            valid = data.dropna(subset=['ucla_total', col])

            if len(valid) >= 10:
                r, p = pearsonr(valid['ucla_total'], valid[col])
                results.append({
                    'gender': gender,
                    'condition': condition,
                    'parameter': param,
                    'n': len(valid),
                    'r': r,
                    'p': p,
                    'mean': valid[col].mean(),
                    'sd': valid[col].std()
                })

corr_df = pd.DataFrame(results)
print(corr_df[corr_df['p'] < 0.10])

corr_df.to_csv(OUTPUT_DIR / "exgaussian_gender_correlations.csv", index=False, encoding='utf-8-sig')

# Interference effects on Ex-Gaussian components
exgauss_df['mu_interference'] = exgauss_df['incongruent_mu'] - exgauss_df['congruent_mu']
exgauss_df['sigma_interference'] = exgauss_df['incongruent_sigma'] - exgauss_df['congruent_sigma']
exgauss_df['tau_interference'] = exgauss_df['incongruent_tau'] - exgauss_df['congruent_tau']

print("\n[4] Testing interference effects on Ex-Gaussian components...")

interference_results = []
for gender in ['male', 'female']:
    data = exgauss_df[exgauss_df['gender'] == gender]

    for param in ['mu', 'sigma', 'tau']:
        col = f'{param}_interference'
        valid = data.dropna(subset=['ucla_total', col])

        if len(valid) >= 10:
            r, p = pearsonr(valid['ucla_total'], valid[col])
            interference_results.append({
                'gender': gender,
                'parameter': f'{param}_interference',
                'n': len(valid),
                'r': r,
                'p': p,
                'mean': valid[col].mean(),
                'sd': valid[col].std()
            })

interference_df = pd.DataFrame(interference_results)
print(interference_df)

interference_df.to_csv(OUTPUT_DIR / "exgaussian_interference_correlations.csv", index=False, encoding='utf-8-sig')

# Summary
print("\n" + "="*80)
print("STROOP EX-GAUSSIAN DECOMPOSITION - KEY FINDINGS")
print("="*80)

print(f"""
1. Ex-Gaussian Parameters (Overall):
   Congruent:
   - μ (routine): {exgauss_df['congruent_mu'].mean():.1f} ± {exgauss_df['congruent_mu'].std():.1f} ms
   - σ (variability): {exgauss_df['congruent_sigma'].mean():.1f} ± {exgauss_df['congruent_sigma'].std():.1f} ms
   - τ (lapses): {exgauss_df['congruent_tau'].mean():.1f} ± {exgauss_df['congruent_tau'].std():.1f} ms

   Incongruent:
   - μ (routine): {exgauss_df['incongruent_mu'].mean():.1f} ± {exgauss_df['incongruent_mu'].std():.1f} ms
   - σ (variability): {exgauss_df['incongruent_sigma'].mean():.1f} ± {exgauss_df['incongruent_sigma'].std():.1f} ms
   - τ (lapses): {exgauss_df['incongruent_tau'].mean():.1f} ± {exgauss_df['incongruent_tau'].std():.1f} ms

2. Gender-Stratified Correlations (p < 0.10):
{corr_df[corr_df['p'] < 0.10].to_string(index=False) if len(corr_df[corr_df['p'] < 0.10]) > 0 else "   No significant effects"}

3. Interference Effects on Ex-Gaussian Components:
{interference_df.to_string(index=False)}

4. Interpretation:
   - μ effects → Loneliness slows routine processing
   - σ effects → Loneliness increases processing variability
   - τ effects → Loneliness increases attentional lapses

5. Files Generated:
   - exgaussian_parameters.csv
   - exgaussian_gender_correlations.csv
   - exgaussian_interference_correlations.csv
""")

print("\n✅ Stroop Ex-Gaussian decomposition complete!")
print("="*80)
