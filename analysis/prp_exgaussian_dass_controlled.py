"""
PRP Ex-Gaussian Decomposition with DASS Controls
================================================
Decomposes T2 RT distributions into Ex-Gaussian parameters to identify
the mechanistic source of male RT variability:

- μ (mu): Gaussian mean (routine processing speed)
- σ (sigma): Gaussian SD (processing variability)
- τ (tau): Exponential component (attentional lapses, slow tail)

Key Question:
Is male UCLA → RT variability driven by:
  1. Increased σ (general inconsistency in processing speed)
  2. Increased τ (more frequent attentional lapses)
  3. Both?

CRITICAL: All models control for DASS-21 subscales.

Author: Automated analysis pipeline
Date: 2025-01-16
"""

import sys
import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials
import numpy as np
from pathlib import Path
from scipy.stats import exponnorm, pearsonr
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/prp_exgaussian_dass_controlled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRP EX-GAUSSIAN DECOMPOSITION - DASS CONTROLLED")
print("=" * 80)
print("\n⚠️  CRITICAL: All models control for DASS-21 subscales")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

trials, prp_summary = load_prp_trials(use_cache=True)
trials.columns = trials.columns.str.lower()

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
master['gender_male'] = (master['gender'] == 'male').astype(int)

print(f"  Master dataset: N={len(master)}")
print(f"  Trial data: {len(trials)} trials")

# ============================================================================
# CLEAN TRIAL DATA
# ============================================================================
print("\n[2] Cleaning trial data...")

rt_col_t2 = 't2_rt'
soa_col = 'soa'

trials_clean = trials[
    (trials['t2_correct'].notna()) &
    (trials[rt_col_t2].notna()) &
    (trials[rt_col_t2] > 200) &
    (trials[rt_col_t2] < 5000)
].copy()

trials_clean['t2_rt'] = trials_clean[rt_col_t2]
trials_clean['soa'] = trials_clean[soa_col]

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

trials_clean['soa_cat'] = trials_clean['soa'].apply(categorize_soa)
trials_clean = trials_clean[trials_clean['soa_cat'] != 'other']

print(f"  Valid trials: {len(trials_clean)}")

# ============================================================================
# EX-GAUSSIAN FITTING
# ============================================================================
print("\n[3] Fitting Ex-Gaussian distributions...")

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
    skew = np.mean(((rts - m) / s) ** 3) if s > 0 else 0

    # Initial tau estimate from skewness
    tau_init = max(10, (abs(skew) / 2) ** (1/3) * s)
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

    # Also fit overall (collapsed across SOA)
    params_overall = fit_exgaussian(group['t2_rt'].values)
    results['overall_mu'] = params_overall['mu']
    results['overall_sigma'] = params_overall['sigma']
    results['overall_tau'] = params_overall['tau']

    return pd.Series(results)

exgauss_df = trials_clean.groupby('participant_id').apply(compute_exgaussian_params).reset_index()

print(f"  Ex-Gaussian fitted for N={len(exgauss_df)} participants")
print(f"\n  Overall T2 RT parameters:")
print(f"    μ: {exgauss_df['overall_mu'].mean():.1f} ms (SD: {exgauss_df['overall_mu'].std():.1f})")
print(f"    σ: {exgauss_df['overall_sigma'].mean():.1f} ms (SD: {exgauss_df['overall_sigma'].std():.1f})")
print(f"    τ: {exgauss_df['overall_tau'].mean():.1f} ms (SD: {exgauss_df['overall_tau'].std():.1f})")

# Compute bottleneck effects on each parameter
exgauss_df['mu_bottleneck'] = exgauss_df['short_mu'] - exgauss_df['long_mu']
exgauss_df['sigma_bottleneck'] = exgauss_df['short_sigma'] - exgauss_df['long_sigma']
exgauss_df['tau_bottleneck'] = exgauss_df['short_tau'] - exgauss_df['long_tau']

print(f"\n  Bottleneck effects (short - long SOA):")
print(f"    Δμ: {exgauss_df['mu_bottleneck'].mean():.1f} ms")
print(f"    Δσ: {exgauss_df['sigma_bottleneck'].mean():.1f} ms")
print(f"    Δτ: {exgauss_df['tau_bottleneck'].mean():.1f} ms")

# ============================================================================
# MERGE WITH MASTER DATASET
# ============================================================================
print("\n[4] Merging with master dataset...")

analysis_df = master.merge(exgauss_df, on='participant_id', how='inner')

# Standardize variables
for var in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    if var in analysis_df.columns:
        analysis_df[f'z_{var}'] = (analysis_df[var] - analysis_df[var].mean()) / analysis_df[var].std()

print(f"  Final dataset: N={len(analysis_df)}")

exgauss_df.to_csv(OUTPUT_DIR / "prp_exgaussian_parameters.csv", index=False, encoding='utf-8-sig')
analysis_df.to_csv(OUTPUT_DIR / "prp_exgaussian_master.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STATISTICAL ANALYSIS: UCLA × GENDER EFFECTS (DASS CONTROLLED)
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS: EX-GAUSSIAN PARAMETERS")
print("=" * 80)

# Test DVs: Overall parameters + bottleneck parameters
dvs = [
    ('overall_mu', 'μ (Gaussian mean)'),
    ('overall_sigma', 'σ (Gaussian SD)'),
    ('overall_tau', 'τ (Exponential tail)'),
    ('mu_bottleneck', 'Δμ bottleneck'),
    ('sigma_bottleneck', 'Δσ bottleneck'),
    ('tau_bottleneck', 'Δτ bottleneck'),
    ('short_mu', 'μ at short SOA'),
    ('short_sigma', 'σ at short SOA'),
    ('short_tau', 'τ at short SOA'),
]

results_list = []

for dv_col, dv_name in dvs:
    if dv_col not in analysis_df.columns:
        continue

    df_clean = analysis_df.dropna(subset=[dv_col, 'z_ucla_total', 'z_dass_depression', 'z_dass_anxiety', 'z_dass_stress', 'z_age', 'gender_male'])

    if len(df_clean) < 20:
        continue

    print(f"\n{'='*80}")
    print(f"DV: {dv_name} (N={len(df_clean)})")
    print(f"{'='*80}")

    # Model: UCLA × Gender interaction (DASS controlled)
    try:
        formula = f"{dv_col} ~ z_ucla_total * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
        model = smf.ols(formula, data=df_clean).fit()

        print(f"\nUCLA × Gender Interaction (DASS controlled)")
        print(f"  UCLA β = {model.params['z_ucla_total']:.3f}, p = {model.pvalues['z_ucla_total']:.4f}")
        print(f"  Gender β = {model.params['C(gender_male)[T.1]']:.3f}, p = {model.pvalues['C(gender_male)[T.1]']:.4f}")

        interaction_term = 'z_ucla_total:C(gender_male)[T.1]'
        if interaction_term in model.params:
            print(f"  UCLA × Gender β = {model.params[interaction_term]:.3f}, p = {model.pvalues[interaction_term]:.4f}")

            results_list.append({
                'dv': dv_name,
                'n': len(df_clean),
                'beta_ucla': model.params['z_ucla_total'],
                'se_ucla': model.bse['z_ucla_total'],
                'p_ucla': model.pvalues['z_ucla_total'],
                'beta_gender': model.params['C(gender_male)[T.1]'],
                'p_gender': model.pvalues['C(gender_male)[T.1]'],
                'beta_interaction': model.params[interaction_term],
                'se_interaction': model.bse[interaction_term],
                'p_interaction': model.pvalues[interaction_term],
                'r2': model.rsquared,
                'r2_adj': model.rsquared_adj,
            })

        # Gender-stratified effects
        print(f"\nGender-Stratified Effects:")
        for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
            df_gender = df_clean[df_clean['gender_male'] == gender_val]
            if len(df_gender) >= 10:
                formula_strat = f"{dv_col} ~ z_ucla_total + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
                model_strat = smf.ols(formula_strat, data=df_gender).fit()

                print(f"  {gender_label} (N={len(df_gender)}): UCLA β = {model_strat.params['z_ucla_total']:.3f}, p = {model_strat.pvalues['z_ucla_total']:.4f}")

    except Exception as e:
        print(f"  Model failed: {e}")

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_DIR / "prp_exgaussian_dass_controlled_results.csv", index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("SUMMARY: SIGNIFICANT INTERACTION EFFECTS (p < 0.10)")
print("=" * 80)

sig_interactions = results_df[results_df['p_interaction'] < 0.10]
if len(sig_interactions) > 0:
    print(sig_interactions[['dv', 'n', 'beta_interaction', 'p_interaction', 'r2_adj']].to_string(index=False))
else:
    print("  No significant interactions")

# ============================================================================
# MECHANISTIC INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("MECHANISTIC INTERPRETATION")
print("=" * 80)

summary_report = f"""
Ex-Gaussian Decomposition Results:
==================================

Overall RT Components (N={len(analysis_df)}):
- μ (Gaussian mean): {analysis_df['overall_mu'].mean():.1f} ms (routine processing)
- σ (Gaussian SD): {analysis_df['overall_sigma'].mean():.1f} ms (trial-to-trial variability)
- τ (Exponential tail): {analysis_df['overall_tau'].mean():.1f} ms (attentional lapses)

Key Question: Which component drives male UCLA → RT variability?

Findings:
---------
"""

# Check which parameters show significant interactions
if len(sig_interactions) > 0:
    summary_report += "\nSignificant UCLA × Gender Interactions Found:\n"
    for _, row in sig_interactions.iterrows():
        summary_report += f"  - {row['dv']}: β = {row['beta_interaction']:.2f}, p = {row['p_interaction']:.4f}\n"

    if any(sig_interactions['dv'].str.contains('sigma|σ')):
        summary_report += "\n✅ FINDING: σ (Gaussian SD) shows interaction"
        summary_report += "\n   → Male variability driven by INCONSISTENT processing speed"
        summary_report += "\n   → Suggests: Inefficient resource allocation, not just lapses\n"

    if any(sig_interactions['dv'].str.contains('tau|τ')):
        summary_report += "\n✅ FINDING: τ (Exponential tail) shows interaction"
        summary_report += "\n   → Male variability driven by ATTENTIONAL LAPSES"
        summary_report += "\n   → Suggests: Sustained attention failures under dual-task load\n"

    if any(sig_interactions['dv'].str.contains('mu|μ')):
        summary_report += "\n✅ FINDING: μ (Gaussian mean) shows interaction"
        summary_report += "\n   → Males show slower routine processing with higher UCLA"
        summary_report += "\n   → Suggests: Overall speed deficit, not just variability\n"
else:
    summary_report += "\nNo significant interactions found at p < 0.10"
    summary_report += "\nPossible reasons:"
    summary_report += "\n  1. Insufficient sample size (N={})".format(len(analysis_df))
    summary_report += "\n  2. Overall RT variability (IQR) is composite of μ, σ, τ effects"
    summary_report += "\n  3. Need trial-level mixed models for more power\n"

summary_report += f"""

Cross-Reference to Previous Finding:
------------------------------------
- Previous analysis: Males show UCLA → T2 RT IQR (β = 76.98, p = 0.032*)
- Current analysis: Decomposes IQR into Ex-Gaussian components
- Interpretation depends on which parameter(s) show interaction

Theoretical Implications:
-------------------------
IF σ interaction: Males have inconsistent processing speed (reactive control failure)
IF τ interaction: Males have more attentional lapses (sustained attention failure)
IF BOTH: Males show dual deficit in speed consistency AND attention maintenance

Next Steps:
-----------
1. Trial-level mixed-effects models (more power than participant-level)
2. Larger sample size (current N={len(analysis_df)} may be underpowered for Ex-Gaussian)
3. Neural correlates: EEG markers of attention (P3) and motor preparation (CNV)
4. Functional outcomes: Does σ or τ better predict real-world dual-tasking?

Files Generated:
----------------
1. prp_exgaussian_parameters.csv - Ex-Gaussian parameters for each participant
2. prp_exgaussian_master.csv - Parameters merged with UCLA/DASS/demographics
3. prp_exgaussian_dass_controlled_results.csv - Regression results
"""

with open(OUTPUT_DIR / "EXGAUSSIAN_MECHANISTIC_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)

print("\n" + "=" * 80)
print("✅ EX-GAUSSIAN ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\n" + "=" * 80)
