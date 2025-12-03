"""
Cross-Task Ex-Gaussian Decomposition Analysis
==============================================

Research Question: Does UCLA → τ (attentional lapses) generalize across all three tasks?

Method:
- Fit Ex-Gaussian (μ, σ, τ) to each participant × task
- Regression per task: τ ~ UCLA × Gender + DASS + age
- Also test μ (routine processing) and σ (variability)
- Meta-analyze: Random-effects pooling of τ effects across tasks

Ex-Gaussian Parameters:
- μ (mu): Normal component mean (routine processing speed)
- σ (sigma): Normal component SD (processing variability)
- τ (tau): Exponential tail (attentional lapses/motivation)

Interpretation:
- If UCLA → τ consistently across tasks (I² < 30%): Domain-general attentional lapse mechanism
- If task-specific heterogeneity (I² > 50%): Task-dependent effects

Author: Research Team
Date: 2025-12-03
Part of: Tier 1 Mechanistic Analyses
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from analysis.utils.data_loader_utils import (
    load_master_dataset,
    normalize_gender_series,
    ensure_participant_id
)
from analysis.utils.trial_data_loader import load_wcst_trials, load_prp_trials, load_stroop_trials

# Set random seed
np.random.seed(42)

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/exgaussian_cross_task")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("CROSS-TASK EX-GAUSSIAN DECOMPOSITION ANALYSIS")
print("=" * 80)
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

    lambda_val = 1 / tau
    z = (x - mu) / sigma

    term1 = lambda_val / 2
    term2 = np.exp(lambda_val / 2 * (2 * mu + lambda_val * sigma**2 - 2 * x))
    term3 = 1 - stats.norm.cdf((mu + lambda_val * sigma**2 - x) / sigma)

    pdf = term1 * term2 * term3
    return pdf


def fit_exgaussian_mle(rt_data, max_iter=500):
    """
    Fit Ex-Gaussian distribution to RT data using maximum likelihood.
    Returns (mu, sigma, tau) or (nan, nan, nan) if fitting fails.

    Uses robust initialization and bounds to improve convergence.
    """
    rt_data = np.array(rt_data)
    rt_data = rt_data[rt_data > 0]  # Remove invalid RTs
    rt_data = rt_data[~np.isnan(rt_data)]

    if len(rt_data) < 20:
        return np.nan, np.nan, np.nan

    # Robust parameter initialization
    mean_rt = np.mean(rt_data)
    std_rt = np.std(rt_data)
    skew_rt = stats.skew(rt_data)

    # Initial estimates based on method of moments
    if skew_rt > 0:
        tau_init = (skew_rt * std_rt / 2) ** (2/3)
        sigma_init = np.sqrt(std_rt**2 - tau_init**2) if std_rt**2 > tau_init**2 else std_rt * 0.5
        mu_init = mean_rt - tau_init
    else:
        mu_init = mean_rt * 0.7
        sigma_init = std_rt * 0.5
        tau_init = mean_rt * 0.3

    # Ensure positive initial values
    mu_init = max(mu_init, rt_data.min() * 0.5)
    sigma_init = max(sigma_init, 10)
    tau_init = max(tau_init, 10)

    # Negative log-likelihood
    def neg_log_likelihood(params):
        mu, sigma, tau = params

        if sigma <= 1 or tau <= 1 or mu <= 0:
            return 1e10

        pdf_vals = exgaussian_pdf(rt_data, mu, sigma, tau)
        pdf_vals = np.maximum(pdf_vals, 1e-10)  # Avoid log(0)

        return -np.sum(np.log(pdf_vals))

    # Parameter bounds
    bounds = [
        (rt_data.min() * 0.1, rt_data.max()),  # mu
        (1, std_rt * 2),  # sigma
        (1, mean_rt * 2)   # tau
    ]

    # Optimization
    try:
        result = minimize(
            neg_log_likelihood,
            x0=[mu_init, sigma_init, tau_init],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )

        if result.success:
            mu, sigma, tau = result.x
            return mu, sigma, tau
        else:
            return np.nan, np.nan, np.nan
    except Exception as e:
        return np.nan, np.nan, np.nan


def fit_exgaussian_per_participant(trial_df, rt_col='rt_ms', participant_col='participant_id', task_name=''):
    """
    Fit Ex-Gaussian to each participant's RT distribution.
    """
    results = []

    n_participants = trial_df[participant_col].nunique()
    print(f"  Fitting Ex-Gaussian for {n_participants} participants in {task_name}...")

    for i, (pid, group) in enumerate(trial_df.groupby(participant_col)):
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{n_participants}")

        rt_series = group[rt_col]
        rt_clean = rt_series.dropna()
        rt_clean = rt_clean[rt_clean > 0]

        if len(rt_clean) < 20:
            results.append({
                participant_col: pid,
                'mu': np.nan,
                'sigma': np.nan,
                'tau': np.nan,
                'n_trials': len(rt_clean)
            })
            continue

        # Fit Ex-Gaussian
        mu, sigma, tau = fit_exgaussian_mle(rt_clean.values)

        results.append({
            participant_col: pid,
            'mu': mu,
            'sigma': sigma,
            'tau': tau,
            'n_trials': len(rt_clean)
        })

    df_results = pd.DataFrame(results)
    print(f"  Successful fits: {df_results['tau'].notna().sum()}/{len(df_results)}")
    print()

    return df_results


# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading trial-level data...")
print()

# Load WCST
print("  WCST...")
wcst_df, _ = load_wcst_trials()
print(f"    {len(wcst_df)} trials, {wcst_df['participant_id'].nunique()} participants")

# Load PRP
print("  PRP...")
prp_df, _ = load_prp_trials()
prp_df['rt_ms'] = prp_df['t2_rt']  # Use T2 RT
print(f"    {len(prp_df)} trials, {prp_df['participant_id'].nunique()} participants")

# Load Stroop
print("  Stroop...")
stroop_df, _ = load_stroop_trials()
stroop_df['rt_ms'] = stroop_df['rt']
print(f"    {len(stroop_df)} trials, {stroop_df['participant_id'].nunique()} participants")

print()
print("Loading master dataset (UCLA, DASS, demographics)...")
master_df = load_master_dataset()
master_df = ensure_participant_id(master_df)
master_df['gender'] = normalize_gender_series(master_df['gender'])
master_df = master_df[master_df['gender'].isin(['male', 'female'])].copy()
master_df['gender_male'] = (master_df['gender'] == 'male').astype(int)
print(f"  {len(master_df)} participants")
print()

# ============================================================================
# FIT EX-GAUSSIAN FOR EACH TASK
# ============================================================================

print("=" * 80)
print("FITTING EX-GAUSSIAN DISTRIBUTIONS")
print("=" * 80)
print()

wcst_exg = fit_exgaussian_per_participant(wcst_df, rt_col='rt_ms', task_name='WCST')
wcst_exg.columns = [f'wcst_{col}' if col != 'participant_id' else col for col in wcst_exg.columns]

prp_exg = fit_exgaussian_per_participant(prp_df, rt_col='rt_ms', task_name='PRP')
prp_exg.columns = [f'prp_{col}' if col != 'participant_id' else col for col in prp_exg.columns]

stroop_exg = fit_exgaussian_per_participant(stroop_df, rt_col='rt_ms', task_name='Stroop')
stroop_exg.columns = [f'stroop_{col}' if col != 'participant_id' else col for col in stroop_exg.columns]

# ============================================================================
# MERGE WITH MASTER DATASET
# ============================================================================

print("Merging Ex-Gaussian parameters with UCLA, DASS, demographics...")
df = master_df.copy()
df = df.merge(wcst_exg, on='participant_id', how='left')
df = df.merge(prp_exg, on='participant_id', how='left')
df = df.merge(stroop_exg, on='participant_id', how='left')

# Keep only participants with at least one task's parameters
df = df.dropna(subset=['wcst_tau', 'prp_tau', 'stroop_tau'], how='all')
print(f"  Final sample: {len(df)} participants with Ex-Gaussian parameters")
print()

# Standardize predictors
print("Standardizing predictors (UCLA, DASS, age)...")
df['z_ucla'] = (df['ucla_total'] - df['ucla_total'].mean()) / df['ucla_total'].std()
df['z_dass_dep'] = (df['dass_depression'] - df['dass_depression'].mean()) / df['dass_depression'].std()
df['z_dass_anx'] = (df['dass_anxiety'] - df['dass_anxiety'].mean()) / df['dass_anxiety'].std()
df['z_dass_str'] = (df['dass_stress'] - df['dass_stress'].mean()) / df['dass_stress'].std()
df['z_age'] = (df['age'] - df['age'].mean()) / df['age'].std()
print()

# Save merged dataset
df.to_csv(OUTPUT_DIR / "exgaussian_parameters_by_participant.csv", index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DIR / 'exgaussian_parameters_by_participant.csv'}")
print()

# ============================================================================
# REGRESSION ANALYSES
# ============================================================================

print("=" * 80)
print("REGRESSION ANALYSES: UCLA × Gender → μ, σ, τ")
print("=" * 80)
print()

formula_base = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

results_summary = []

for task in ['wcst', 'prp', 'stroop']:
    print(f"\n{'=' * 60}")
    print(f"{task.upper()} TASK")
    print(f"{'=' * 60}\n")

    for param in ['mu', 'sigma', 'tau']:
        outcome_col = f'{task}_{param}'

        if df[outcome_col].isna().all():
            print(f"  {param.upper()}: No data, skipping...")
            continue

        print(f"{param.upper()} ({outcome_col})")
        print("-" * 60)

        df_clean = df.dropna(subset=[outcome_col, 'z_ucla', 'gender_male',
                                      'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])

        if len(df_clean) < 20:
            print(f"  Insufficient N ({len(df_clean)}), skipping...")
            print()
            continue

        formula = formula_base.format(outcome=outcome_col)
        model = smf.ols(formula, data=df_clean).fit()
        print(model.summary())
        print()

        # Extract coefficients
        beta_ucla = model.params.get('z_ucla', np.nan)
        pval_ucla = model.pvalues.get('z_ucla', np.nan)
        ci_ucla = model.conf_int().loc['z_ucla'].values if 'z_ucla' in model.params else [np.nan, np.nan]

        beta_interaction = model.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
        pval_interaction = model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)

        results_summary.append({
            'task': task,
            'parameter': param,
            'beta_ucla': beta_ucla,
            'pval_ucla': pval_ucla,
            'ci_lower': ci_ucla[0],
            'ci_upper': ci_ucla[1],
            'beta_interaction': beta_interaction,
            'pval_interaction': pval_interaction,
            'N': len(df_clean)
        })

# Save results
results_df = pd.DataFrame(results_summary)
results_df.to_csv(OUTPUT_DIR / "regression_results_summary.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUTPUT_DIR / 'regression_results_summary.csv'}")
print()

# ============================================================================
# META-ANALYSIS: TAU EFFECTS ACROSS TASKS
# ============================================================================

print("=" * 80)
print("META-ANALYSIS: Pooling τ (TAU) Effects Across Tasks")
print("=" * 80)
print()

tau_results = results_df[results_df['parameter'] == 'tau'].copy()

if len(tau_results) > 0:
    print("τ (Attentional Lapses) Effects:")
    print(tau_results[['task', 'beta_ucla', 'pval_ucla', 'N']])
    print()

    # Simple fixed-effects meta-analysis (inverse variance weighted)
    tau_results['se'] = (tau_results['ci_upper'] - tau_results['ci_lower']) / (2 * 1.96)
    tau_results['weight'] = 1 / (tau_results['se'] ** 2)

    pooled_beta = np.sum(tau_results['beta_ucla'] * tau_results['weight']) / np.sum(tau_results['weight'])
    pooled_se = np.sqrt(1 / np.sum(tau_results['weight']))
    pooled_ci_lower = pooled_beta - 1.96 * pooled_se
    pooled_ci_upper = pooled_beta + 1.96 * pooled_se
    pooled_z = pooled_beta / pooled_se
    pooled_p = 2 * (1 - stats.norm.cdf(abs(pooled_z)))

    print("Fixed-Effects Meta-Analysis:")
    print(f"  Pooled β: {pooled_beta:.2f}")
    print(f"  95% CI: [{pooled_ci_lower:.2f}, {pooled_ci_upper:.2f}]")
    print(f"  p-value: {pooled_p:.4f}")
    print()

    # Heterogeneity (I²)
    Q = np.sum(tau_results['weight'] * (tau_results['beta_ucla'] - pooled_beta)**2)
    df_Q = len(tau_results) - 1
    I2 = max(0, (Q - df_Q) / Q * 100) if Q > 0 else 0

    print(f"Heterogeneity I²: {I2:.1f}%")
    if I2 < 25:
        print("  → Low heterogeneity: Effect is highly consistent across tasks")
    elif I2 < 50:
        print("  → Moderate heterogeneity: Some task-specific variation")
    else:
        print("  → High heterogeneity: Substantial task-specific differences")
    print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")
print()

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Bar plot: β coefficients by parameter and task
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, param in enumerate(['mu', 'sigma', 'tau']):
    param_data = results_df[results_df['parameter'] == param]

    if len(param_data) == 0:
        continue

    tasks = param_data['task'].values
    betas = param_data['beta_ucla'].values
    colors = ['red' if p < 0.05 else 'steelblue' for p in param_data['pval_ucla'].values]

    axes[i].bar(tasks, betas, color=colors, alpha=0.8)
    axes[i].set_xlabel('Task', fontsize=11)
    axes[i].set_ylabel(f'Standardized β (UCLA)', fontsize=11)
    axes[i].set_title(f'{param.upper()} Parameter', fontsize=12, fontweight='bold')
    axes[i].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[i].grid(True, alpha=0.3, axis='y')

    # Add p-value annotations
    for j, (task, beta, pval) in enumerate(zip(tasks, betas, param_data['pval_ucla'].values)):
        axes[i].text(j, beta + (0.05 * abs(betas).max() * np.sign(beta) if beta != 0 else 10),
                    f'p={pval:.3f}', ha='center', fontsize=8)

plt.suptitle('UCLA Effects on Ex-Gaussian Parameters (DASS-Controlled)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exgaussian_beta_comparison.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'exgaussian_beta_comparison.png'}")
plt.close()

# 2. Forest plot for τ meta-analysis
if len(tau_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(tau_results))

    for i, (idx, row) in enumerate(tau_results.iterrows()):
        ax.errorbar(row['beta_ucla'], i,
                   xerr=[[row['beta_ucla'] - row['ci_lower']],
                         [row['ci_upper'] - row['beta_ucla']]],
                   fmt='s', markersize=8, capsize=5, capthick=2,
                   label=row['task'].upper(), alpha=0.7)

    # Add pooled estimate
    ax.errorbar(pooled_beta, len(tau_results) + 0.5,
               xerr=[[pooled_beta - pooled_ci_lower], [pooled_ci_upper - pooled_beta]],
               fmt='D', markersize=10, capsize=7, capthick=3,
               color='red', label='Pooled', alpha=0.9)

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(list(y_positions) + [len(tau_results) + 0.5])
    ax.set_yticklabels(list(tau_results['task'].str.upper()) + ['POOLED'])
    ax.set_xlabel('Standardized β (UCLA → τ)', fontsize=12)
    ax.set_title('Forest Plot: UCLA Effect on Attentional Lapses (τ) Across Tasks',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tau_forest_plot.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'tau_forest_plot.png'}")
    plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()
print("Key outputs:")
print("  1. exgaussian_parameters_by_participant.csv - μ, σ, τ for all participants")
print("  2. regression_results_summary.csv - β coefficients for all parameters")
print("  3. exgaussian_beta_comparison.png - Bar plots comparing parameters")
print("  4. tau_forest_plot.png - Meta-analytic forest plot for τ")
print()
