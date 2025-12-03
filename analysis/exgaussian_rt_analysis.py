"""
Ex-Gaussian RT Distribution Analysis
=====================================

Decomposes RT distributions into 3 parameters:
- μ (mu): Normal component mean (routine processing speed)
- σ (sigma): Normal component SD (processing variability)
- τ (tau): Exponential tail (attentional lapses/motivation)

Tests:
1. UCLA × Gender → μ, σ, τ separately
2. τ (lapses) mediates PE effect (UCLA → τ → PE)
3. Post-error RT decomposition (does τ increase after errors?)

Critical Question: Is this CAPACITY deficit (μ/σ) or MOTIVATION deficit (τ)?

Author: Research Team
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from statsmodels.formula.api import ols
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from analysis.utils.data_loader_utils import normalize_gender_series
from analysis.utils.trial_data_loader import load_wcst_trials

np.random.seed(42)
RNG = np.random.default_rng(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/mechanism_analysis/exgaussian")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("EX-GAUSSIAN RT DISTRIBUTION ANALYSIS")
print("="*80)
print()

# ============================================================================
# EX-GAUSSIAN FITTING FUNCTIONS
# ============================================================================

def exgaussian_pdf(x, mu, sigma, tau):
    """
    Probability density function of Ex-Gaussian distribution.
    """
    if sigma <= 0 or tau <= 0:
        return np.zeros_like(x)

    z = (x - mu) / sigma
    lambda_val = 1 / tau

    term1 = lambda_val / 2
    term2 = np.exp(lambda_val / 2 * (2 * mu + lambda_val * sigma**2 - 2 * x))
    term3 = 1 - stats.norm.cdf((mu + lambda_val * sigma**2 - x) / sigma)

    pdf = term1 * term2 * term3
    return pdf

def fit_exgaussian(rt_data):
    """
    Fit Ex-Gaussian distribution to RT data using maximum likelihood.
    Returns (mu, sigma, tau) or (nan, nan, nan) if fitting fails.
    """
    rt_data = np.array(rt_data)
    rt_data = rt_data[rt_data > 0]  # Remove invalid RTs

    if len(rt_data) < 20:
        return np.nan, np.nan, np.nan

    # Initial parameter estimates
    mean_rt = np.mean(rt_data)
    std_rt = np.std(rt_data)

    # Negative log-likelihood
    def neg_log_likelihood(params):
        mu, sigma, tau = params

        if sigma <= 0 or tau <= 0 or mu <= 0:
            return 1e10

        pdf_vals = exgaussian_pdf(rt_data, mu, sigma, tau)
        pdf_vals = np.maximum(pdf_vals, 1e-10)  # Avoid log(0)

        return -np.sum(np.log(pdf_vals))

    # Try multiple random starts
    best_params = None
    best_nll = np.inf

    for _ in range(5):
        init_mu = mean_rt * np.random.uniform(0.7, 1.0)
        init_sigma = std_rt * np.random.uniform(0.3, 0.7)
        init_tau = std_rt * np.random.uniform(0.1, 0.5)

        try:
            result = minimize(
                neg_log_likelihood,
                [init_mu, init_sigma, init_tau],
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except:
            continue

    if best_params is None:
        return np.nan, np.nan, np.nan

    mu, sigma, tau = best_params
    return mu, sigma, tau

print("[1/5] Ex-Gaussian fitting functions defined")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[2/5] Loading data...")

wcst_trials, _ = load_wcst_trials(use_cache=True)

# Load participant data
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

master = master.rename(columns={"gender_normalized": "gender"})
master["gender"] = normalize_gender_series(master["gender"])
master["gender_male"] = (master["gender"] == "male").astype(int)

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    std = s.std(ddof=0)
    if std is None or std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std

master['z_ucla'] = zscore(master['ucla_total'])
master['z_dass_dep'] = zscore(master['dass_depression'])
master['z_dass_anx'] = zscore(master['dass_anxiety'])
master['z_dass_stress'] = zscore(master['dass_stress'])
master['z_age'] = zscore(master['age'])

# Ensure isPE
if "isPE" in wcst_trials.columns:
    wcst_trials["isPE"] = wcst_trials["isPE"]
elif "is_pe" in wcst_trials.columns:
    wcst_trials["isPE"] = wcst_trials["is_pe"]
else:
    import ast
    def parse_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except Exception:
            return {}
    wcst_trials["extra_dict"] = wcst_trials["extra"].apply(parse_extra) if "extra" in wcst_trials.columns else {}
    wcst_trials["isPE"] = wcst_trials.get("extra_dict", {}).apply(lambda x: x.get("isPE", False) if isinstance(x, dict) else False)

# Extract RT (prefer explicit millisecond columns when available)
if 'rt_ms' in wcst_trials.columns:
    wcst_trials['rt_ms'] = wcst_trials['rt_ms']
elif 'resp_time_ms' in wcst_trials.columns:
    wcst_trials['rt_ms'] = wcst_trials['resp_time_ms']
elif 'responseTime' in wcst_trials.columns:
    wcst_trials['rt_ms'] = wcst_trials['responseTime']
elif 'rt' in wcst_trials.columns:
    wcst_trials['rt_ms'] = wcst_trials['rt']
else:
    raise SystemExit("WCST trials missing RT column (rt_ms, resp_time_ms, responseTime, or rt).")

print(f"  WCST trials: {len(wcst_trials)}")
print(f"  Participants: {len(master)}")
print()

# ============================================================================
# FIT EX-GAUSSIAN TO ALL PARTICIPANTS
# ============================================================================

print("[3/5] Fitting Ex-Gaussian models...")
print("  (This may take several minutes...)")
print()

exgaussian_params = []

participant_ids = wcst_trials['participant_id'].dropna().unique()

for i, pid in enumerate(participant_ids):
    pid_trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()

    rt_data = pid_trials['rt_ms'].dropna()
    rt_data = rt_data[rt_data > 100]  # Remove too-fast RTs
    rt_data = rt_data[rt_data < 5000]  # Remove too-slow RTs

    if len(rt_data) < 20:
        continue

    # Fit overall
    mu, sigma, tau = fit_exgaussian(rt_data)

    if not np.isnan(mu):
        # Also fit post-error vs post-correct
        error_trials = pid_trials[pid_trials['correct'] == False]
        correct_trials = pid_trials[pid_trials['correct'] == True]

        # Post-error RT (trial after error)
        post_error_rts = []
        post_correct_rts = []

        for idx in range(len(pid_trials) - 1):
            if pid_trials.iloc[idx]['correct'] == False:
                post_error_rts.append(pid_trials.iloc[idx + 1]['rt_ms'])
            elif pid_trials.iloc[idx]['correct'] == True:
                post_correct_rts.append(pid_trials.iloc[idx + 1]['rt_ms'])

        post_error_rts = [rt for rt in post_error_rts if 100 < rt < 5000]
        post_correct_rts = [rt for rt in post_correct_rts if 100 < rt < 5000]

        # Fit post-error
        if len(post_error_rts) >= 10:
            mu_pe, sigma_pe, tau_pe = fit_exgaussian(post_error_rts)
        else:
            mu_pe, sigma_pe, tau_pe = np.nan, np.nan, np.nan

        # Fit post-correct
        if len(post_correct_rts) >= 10:
            mu_pc, sigma_pc, tau_pc = fit_exgaussian(post_correct_rts)
        else:
            mu_pc, sigma_pc, tau_pc = np.nan, np.nan, np.nan

        exgaussian_params.append({
            'participant_id': pid,
            'mu': mu,
            'sigma': sigma,
            'tau': tau,
            'mu_post_error': mu_pe,
            'sigma_post_error': sigma_pe,
            'tau_post_error': tau_pe,
            'mu_post_correct': mu_pc,
            'sigma_post_correct': sigma_pc,
            'tau_post_correct': tau_pc,
            'n_trials': len(rt_data)
        })

    if (i + 1) % 10 == 0:
        print(f"  Fitted {i+1}/{len(participant_ids)} participants...")

print()
print(f"✓ Successfully fitted Ex-Gaussian for {len(exgaussian_params)} participants")
print()

# Convert to DataFrame
exgauss_df = pd.DataFrame(exgaussian_params)

# Merge with master
exgauss_df = exgauss_df.merge(
    master[
        [
            'participant_id',
            'ucla_total',
            'z_ucla',
            'gender_male',
            'pe_rate',
            'wcst_accuracy',
            'z_dass_dep',
            'z_dass_anx',
            'z_dass_stress',
            'z_age',
        ]
    ],
    on='participant_id',
    how='left'
)

exgauss_df = exgauss_df.dropna(subset=['ucla_total', 'gender_male', 'pe_rate']).copy()

print(f"Final sample: N={len(exgauss_df)} ({(exgauss_df['gender_male']==0).sum()}F, {(exgauss_df['gender_male']==1).sum()}M)")
print()

# Save
exgauss_df.to_csv(OUTPUT_DIR / "exgaussian_parameters.csv", index=False, encoding='utf-8-sig')
print(f"✓ Saved: exgaussian_parameters.csv")
print()

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

print("[4/5] Ex-Gaussian parameter descriptive statistics...")
print()

for param in ['mu', 'sigma', 'tau']:
    print(f"{param.upper()}:")
    print(f"  Mean: {exgauss_df[param].mean():.1f} ms, SD: {exgauss_df[param].std():.1f} ms")
    print(f"  Range: [{exgauss_df[param].min():.1f}, {exgauss_df[param].max():.1f}] ms")

    # By gender
    female_vals = exgauss_df[exgauss_df['gender_male']==0][param]
    male_vals = exgauss_df[exgauss_df['gender_male']==1][param]

    print(f"  Female: M={female_vals.mean():.1f} ms, SD={female_vals.std():.1f} ms")
    print(f"  Male: M={male_vals.mean():.1f} ms, SD={male_vals.std():.1f} ms")
    print()

# ============================================================================
# TEST UCLA × GENDER → EX-GAUSSIAN PARAMETERS
# ============================================================================

print("[5/5] Testing UCLA × Gender → Ex-Gaussian parameters...")
print()

moderation_results = []

for param in ['mu', 'sigma', 'tau']:
    print(f"{param.upper()}:")

    formula = (
        f"{param} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + "
        f"z_dass_stress + z_age"
    )

    try:
        model = ols(formula, data=exgauss_df).fit()

        interaction_term = "z_ucla:C(gender_male)[T.1]"

        if interaction_term in model.params:
            int_beta = model.params[interaction_term]
            int_p = model.pvalues[interaction_term]

            # Gender-stratified correlations
            female_corr, female_p = stats.pearsonr(
                exgauss_df[exgauss_df['gender_male']==0]['ucla_total'],
                exgauss_df[exgauss_df['gender_male']==0][param]
            )

            male_corr, male_p = stats.pearsonr(
                exgauss_df[exgauss_df['gender_male']==1]['ucla_total'],
                exgauss_df[exgauss_df['gender_male']==1][param]
            )

            moderation_results.append({
                'parameter': param,
                'interaction_beta': int_beta,
                'interaction_p': int_p,
                'female_corr': female_corr,
                'female_p': female_p,
                'male_corr': male_corr,
                'male_p': male_p
            })

            sig_marker = " ***" if int_p < 0.001 else " **" if int_p < 0.01 else " *" if int_p < 0.05 else ""
            print(f"  Interaction: β={int_beta:.2f} ms, p={int_p:.4f}{sig_marker}")
            print(f"  Female: r={female_corr:.3f}, p={female_p:.4f}")
            print(f"  Male: r={male_corr:.3f}, p={male_p:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    print()

# Save moderation results
moderation_df = pd.DataFrame(moderation_results)
moderation_df.to_csv(OUTPUT_DIR / "exgauss_moderation.csv", index=False, encoding='utf-8-sig')
print(f"✓ Saved: exgauss_moderation.csv")
print()

# Test τ mediation
print("TAU MEDIATION: UCLA → τ → PE")
print()

# By gender
for gender, label in [(0, 'Female'), (1, 'Male')]:
    gender_data = exgauss_df[exgauss_df['gender_male'] == gender].copy()

    covariate_cols = [
        'tau',
        'pe_rate',
        'z_ucla',
        'z_dass_dep',
        'z_dass_anx',
        'z_dass_stress',
        'z_age',
    ]
    gender_data = gender_data.dropna(subset=covariate_cols).copy()

    if len(gender_data) < 10:
        continue

    # Path a: UCLA → τ
    model_a = ols(
        "tau ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + z_age",
        data=gender_data,
    ).fit()
    a = model_a.params['z_ucla']

    # Path b: τ → PE (controlling UCLA)
    model_b = ols(
        "pe_rate ~ tau + z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + z_age",
        data=gender_data,
    ).fit()
    b = model_b.params['tau']
    c_prime = model_b.params['z_ucla']

    # Total effect
    model_c = ols(
        "pe_rate ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + z_age",
        data=gender_data,
    ).fit()
    c = model_c.params['z_ucla']

    # Indirect
    indirect = a * b

    # Bootstrap CI
    boot_indirect = []
    for _ in range(1000):
        rand_state = int(RNG.integers(0, 2**32 - 1))
        boot_data = resample(
            gender_data,
            replace=True,
            n_samples=len(gender_data),
            random_state=rand_state,
        )
        try:
            boot_a = ols(
                "tau ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + z_age",
                data=boot_data,
            ).fit().params['z_ucla']
            boot_b = ols(
                "pe_rate ~ tau + z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + z_age",
                data=boot_data,
            ).fit().params['tau']
            boot_indirect.append(boot_a * boot_b)
        except:
            continue

    if len(boot_indirect) > 0:
        ci_lower = np.percentile(boot_indirect, 2.5)
        ci_upper = np.percentile(boot_indirect, 97.5)
        significant = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
    else:
        ci_lower, ci_upper, significant = np.nan, np.nan, False

    print(f"  {label}:")
    print(f"    Indirect effect: {indirect:.4f}, 95%CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    Significant: {significant}")
    print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row: μ, σ, τ × UCLA
for i, param in enumerate(['mu', 'sigma', 'tau']):
    ax = axes[0, i]

    for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
        data = exgauss_df[exgauss_df['gender_male'] == gender]

        ax.scatter(data['ucla_total'], data[param],
                   alpha=0.6, s=80, color=color, label=label,
                   edgecolors='white', linewidth=0.5)

        if len(data) > 5:
            z = np.polyfit(data['ucla_total'].dropna(), data[param].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

    ax.set_xlabel('UCLA Loneliness Score', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{param} (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'{param.upper()} × UCLA by Gender', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Bottom row: Parameters × PE
for i, param in enumerate(['mu', 'sigma', 'tau']):
    ax = axes[1, i]

    for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
        data = exgauss_df[exgauss_df['gender_male'] == gender]

        ax.scatter(data[param], data['pe_rate'],
                   alpha=0.6, s=80, color=color, label=label,
                   edgecolors='white', linewidth=0.5)

        if len(data) > 5:
            z = np.polyfit(data[param].dropna(), data['pe_rate'].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[param].min(), data[param].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

    ax.set_xlabel(f'{param} (ms)', fontsize=11, fontweight='bold')
    ax.set_ylabel('PE Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'PE Rate × {param.upper()} by Gender', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exgaussian_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: exgaussian_summary.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("EX-GAUSSIAN RT ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - exgaussian_parameters.csv")
print("  - exgauss_moderation.csv")
print("  - exgaussian_summary.png")
print()

print("KEY FINDINGS:")
if len(moderation_df) > 0:
    print()
    print("UCLA × Gender → Ex-Gaussian Parameters:")
    for _, row in moderation_df.iterrows():
        sig_marker = " ***" if row['interaction_p'] < 0.001 else " **" if row['interaction_p'] < 0.01 else " *" if row['interaction_p'] < 0.05 else ""
        print(f"  {row['parameter'].upper()}: β={row['interaction_beta']:.2f} ms, p={row['interaction_p']:.4f}{sig_marker}")

print()
print("INTERPRETATION:")
print("  If HIGH τ in lonely males → Attentional lapses (motivation deficit)")
print("  If NORMAL μ/σ → Processing speed/variability intact")
print("  → Effect is MOTIVATIONAL, not CAPACITY-BASED")
print()
