"""
Paper 1: Distributional Variance Analysis
==========================================

Title: "Loneliness Modulates Intraindividual Variability, Not Mean Performance:
       A Distributional Regression Approach"

Purpose:
--------
Comprehensive analysis of how loneliness affects the DISTRIBUTION of RT/performance,
not just mean values. Tests 4 complementary metrics of variability:

1. **Ex-Gaussian tau (τ)**: Exponential tail (attentional lapses)
2. **Ex-Gaussian sigma (σ)**: Gaussian variability (consistent noise)
3. **RMSSD**: Root Mean Square of Successive Differences (time-series variability)
4. **Kurtosis**: Heavy-tailedness beyond 3-parameter Ex-Gaussian

Key Hypothesis:
---------------
UCLA loneliness effects are PRIMARY in variance/noise structure, not mean RT/accuracy.
Gender-specific patterns:
- Males: τ↑ + σ↑ + RMSSD↑ (intermittent lapses, high sequential variability)
- Females: σ↑ with τ↓ or stable (hypervigilance, low sequential variability)

Statistical Approach:
---------------------
- **Location-Scale Models**: Model both μ and log(σ) as functions of UCLA × Gender + DASS
- **DASS Control**: All models include DASS-21 subscales (depression, anxiety, stress) + age
- **Gender Stratification**: Separate analyses for males and females
- **Effect Size Focus**: Cohen's d for distributional parameters, not just p-values

Output:
-------
- paper1_participant_variability_metrics.csv: All metrics (RMSSD, kurtosis, Ex-Gaussian)
- paper1_location_scale_results.csv: Regression results for μ and log(σ)
- paper1_variance_forest_plot.png: Forest plot of all variance effects with 95% CIs
- paper1_gender_stratified_comparisons.csv: Males vs females for all metrics
- PAPER1_SUMMARY_REPORT.txt: Interpretive summary

Author: Research Team
Date: 2025-01-17
Target Journal: Cognitive Psychology, Journal of Experimental Psychology: General
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import exponnorm, kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/paper1_distributional")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print(" " * 20 + "PAPER 1: DISTRIBUTIONAL VARIANCE ANALYSIS")
print("=" * 90)
print()
print("Focus: UCLA effects on RT VARIABILITY (not mean), with DASS control")
print("Metrics: Ex-Gaussian (τ, σ), RMSSD, Kurtosis")
print()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fit_exgaussian(rts):
    """
    Fit Ex-Gaussian distribution to RT data.

    Ex-Gaussian = Gaussian(μ, σ) + Exponential(τ)
    scipy.stats.exponnorm uses K = τ/σ parameterization

    Returns:
        dict: {'mu': float, 'sigma': float, 'tau': float, 'n': int}
    """
    if len(rts) < 20:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}

    rts = np.array(rts)

    # Method of moments initial estimates
    m = np.mean(rts)
    s = np.std(rts)
    skew_val = np.mean(((rts - m) / s) ** 3)

    # Initial tau estimate from skewness
    tau_init = max(10, (abs(skew_val) / 2) ** (1/3) * s)
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


def compute_rmssd(rts):
    """
    Root Mean Square of Successive Differences (RMSSD).

    Measures time-series variability - how much RT changes from trial to trial.
    High RMSSD = high sequential variability (unstable performance).

    RMSSD = sqrt(mean((RT[n] - RT[n-1])^2))

    Args:
        rts: array-like of reaction times in sequential order

    Returns:
        float: RMSSD value, or np.nan if < 2 trials
    """
    rts = np.array(rts)
    if len(rts) < 2:
        return np.nan

    successive_diffs = np.diff(rts)  # RT[n] - RT[n-1]
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    return rmssd


def compute_kurtosis_excess(rts):
    """
    Excess kurtosis (Fisher's definition: kurtosis - 3).

    Measures heavy-tailedness beyond normal distribution.
    - Excess kurtosis = 0: Normal distribution
    - Excess kurtosis > 0: Heavy tails (more outliers than normal)
    - Excess kurtosis < 0: Light tails (fewer outliers)

    Args:
        rts: array-like of reaction times

    Returns:
        float: Excess kurtosis, or np.nan if < 4 trials
    """
    if len(rts) < 4:
        return np.nan

    return kurtosis(rts, fisher=True)  # Fisher=True gives excess kurtosis (kurt - 3)


def categorize_soa(soa):
    """Categorize SOA into short/medium/long bins."""
    if soa <= 150:
        return 'short'
    elif 300 <= soa <= 600:
        return 'medium'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'


def parse_wcst_extra(extra_str):
    """Parse WCST 'extra' field (stringified dict) to extract isPE."""
    if not isinstance(extra_str, str):
        return {}
    try:
        import ast
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}


# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/8] Loading base datasets...")

# Load master dataset
master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')

# Normalize columns
if 'participantId' in master.columns:
    master = master.rename(columns={'participantId': 'participant_id'})

# Ensure gender coding
if 'gender' not in master.columns and 'gender_male' in master.columns:
    master['gender'] = master['gender_male'].map({1: 'male', 0: 'female'})
elif 'gender' in master.columns:
    gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female'}
    master['gender'] = master['gender'].map(gender_map)

print(f"  Master dataset: N = {len(master)}")
print(f"    - Males: {(master['gender'] == 'male').sum()}")
print(f"    - Females: {(master['gender'] == 'female').sum()}")

# Load trial-level data
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')

# Normalize participant_id - FIXED: Drop bad participant_id first, then rename participantId
# PRP
if 'participant_id' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participant_id'])  # Drop the NaN-filled column
if 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.rename(columns={'participantId': 'participant_id'})

# WCST
if 'participant_id' in wcst_trials.columns:
    wcst_trials = wcst_trials.drop(columns=['participant_id'])
if 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

# Stroop
if 'participant_id' in stroop_trials.columns:
    stroop_trials = stroop_trials.drop(columns=['participant_id'])
if 'participantId' in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})

print(f"  PRP trials: {len(prp_trials)}")
print(f"  WCST trials: {len(wcst_trials)}")
print(f"  Stroop trials: {len(stroop_trials)}")

# ============================================================================
# COMPUTE VARIABILITY METRICS FOR EACH TASK
# ============================================================================

print("\n[2/8] Computing variability metrics for PRP (T2 RT)...")

# DEBUG: Check what columns exist
print(f"  DEBUG: PRP columns with 'rt': {[c for c in prp_trials.columns if 'rt' in c.lower()]}")
print(f"  DEBUG: Has participant_id? {'participant_id' in prp_trials.columns}")
print(f"  DEBUG: participant_id nunique: {prp_trials['participant_id'].nunique() if 'participant_id' in prp_trials.columns else 'NO COLUMN'}")

# PRP: Filter valid T2 trials
print(f"  DEBUG: Total PRP trials before filtering: {len(prp_trials)}")

# Use t2_rt_ms if available (most trials), fallback to t2_rt
rt_col_t2 = 't2_rt_ms' if 't2_rt_ms' in prp_trials.columns else 't2_rt'
print(f"  DEBUG: Using RT column: {rt_col_t2}")

prp_clean = prp_trials[
    (prp_trials['t1_correct'] == True) &
    (prp_trials[rt_col_t2] > 0) &
    (prp_trials[rt_col_t2] < 5000)
].copy()

# Normalize RT column name to 't2_rt' for consistency
prp_clean['t2_rt'] = prp_clean[rt_col_t2]

print(f"  DEBUG: PRP trials after filtering: {len(prp_clean)}")
print(f"  DEBUG: participant_id in prp_clean? {'participant_id' in prp_clean.columns}")
if len(prp_clean) > 0 and 'participant_id' in prp_clean.columns:
    print(f"  DEBUG: participant_id nunique after filtering: {prp_clean['participant_id'].nunique()}")

# SOA categorization
soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in prp_clean.columns else 'soa'
prp_clean['soa_cat'] = prp_clean[soa_col].apply(categorize_soa)
prp_clean = prp_clean[prp_clean['soa_cat'] != 'other']

print(f"  DEBUG: PRP trials after SOA filtering: {len(prp_clean)}")

# Compute metrics for each participant
prp_metrics = []
for pid, group in prp_clean.groupby('participant_id'):
    # Overall metrics (all SOA)
    rts_all = group['t2_rt'].values
    exg_all = fit_exgaussian(rts_all)

    # Long SOA only (primary outcome)
    rts_long = group[group['soa_cat'] == 'long']['t2_rt'].values
    exg_long = fit_exgaussian(rts_long)

    prp_metrics.append({
        'participant_id': pid,
        'prp_mean_rt': np.mean(rts_all),
        'prp_sd_rt': np.std(rts_all),
        'prp_rmssd': compute_rmssd(rts_all),
        'prp_kurtosis': compute_kurtosis_excess(rts_all),
        'prp_mu_all': exg_all['mu'],
        'prp_sigma_all': exg_all['sigma'],
        'prp_tau_all': exg_all['tau'],
        'prp_mu_long': exg_long['mu'],
        'prp_sigma_long': exg_long['sigma'],
        'prp_tau_long': exg_long['tau'],
        'prp_n_trials': len(rts_all),
        'prp_n_long': len(rts_long)
    })

prp_metrics_df = pd.DataFrame(prp_metrics)
print(f"  PRP metrics computed for {len(prp_metrics_df)} participants")

# ============================================================================

print("\n[3/8] Computing variability metrics for WCST (RT)...")

# WCST: Filter valid trials (correct responses only for RT analysis)
wcst_clean = wcst_trials[
    (wcst_trials['rt_ms'] > 0) &
    (wcst_trials['rt_ms'] < 10000)
].copy()

wcst_metrics = []
for pid, group in wcst_clean.groupby('participant_id'):
    rts = group['rt_ms'].values
    exg = fit_exgaussian(rts)

    wcst_metrics.append({
        'participant_id': pid,
        'wcst_mean_rt': np.mean(rts),
        'wcst_sd_rt': np.std(rts),
        'wcst_rmssd': compute_rmssd(rts),
        'wcst_kurtosis': compute_kurtosis_excess(rts),
        'wcst_mu': exg['mu'],
        'wcst_sigma': exg['sigma'],
        'wcst_tau': exg['tau'],
        'wcst_n_trials': len(rts)
    })

wcst_metrics_df = pd.DataFrame(wcst_metrics)
print(f"  WCST metrics computed for {len(wcst_metrics_df)} participants")

# ============================================================================

print("\n[4/8] Computing variability metrics for Stroop (RT)...")

# Stroop: Filter valid trials
stroop_clean = stroop_trials[
    (stroop_trials['rt_ms'] > 0) &
    (stroop_trials['rt_ms'] < 3000)
].copy()

# Overall + condition-specific (incongruent trials are more demanding)
stroop_metrics = []
for pid, group in stroop_clean.groupby('participant_id'):
    rts_all = group['rt_ms'].values
    exg_all = fit_exgaussian(rts_all)

    # Incongruent trials only
    incong_trials = group[group['type'] == 'incongruent']
    if len(incong_trials) >= 20:
        rts_incong = incong_trials['rt_ms'].values
        exg_incong = fit_exgaussian(rts_incong)
    else:
        exg_incong = {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan}

    stroop_metrics.append({
        'participant_id': pid,
        'stroop_mean_rt': np.mean(rts_all),
        'stroop_sd_rt': np.std(rts_all),
        'stroop_rmssd': compute_rmssd(rts_all),
        'stroop_kurtosis': compute_kurtosis_excess(rts_all),
        'stroop_mu_all': exg_all['mu'],
        'stroop_sigma_all': exg_all['sigma'],
        'stroop_tau_all': exg_all['tau'],
        'stroop_mu_incong': exg_incong['mu'],
        'stroop_sigma_incong': exg_incong['sigma'],
        'stroop_tau_incong': exg_incong['tau'],
        'stroop_n_trials': len(rts_all),
        'stroop_n_incong': len(incong_trials)
    })

stroop_metrics_df = pd.DataFrame(stroop_metrics)
print(f"  Stroop metrics computed for {len(stroop_metrics_df)} participants")

# ============================================================================
# MERGE ALL METRICS WITH MASTER DATASET
# ============================================================================

print("\n[5/8] Merging all metrics with master dataset...")

# Merge sequentially
df = master.copy()
df = df.merge(prp_metrics_df, on='participant_id', how='left')
df = df.merge(wcst_metrics_df, on='participant_id', how='left')
df = df.merge(stroop_metrics_df, on='participant_id', how='left')

# Drop rows with missing key variables
required_cols = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'gender']
df_clean = df.dropna(subset=required_cols)

print(f"  Final N after merging: {len(df_clean)}")
print(f"    (Dropped {len(df) - len(df_clean)} participants with missing UCLA/DASS/demographics)")

# Standardize predictors
for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    df_clean[f'z_{col}'] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()

# Gender dummy
df_clean['gender_male'] = (df_clean['gender'] == 'male').astype(int)

# Save participant-level metrics
output_metrics = OUTPUT_DIR / "paper1_participant_variability_metrics.csv"
df_clean.to_csv(output_metrics, index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: {output_metrics}")

# ============================================================================
# LOCATION-SCALE REGRESSION MODELS
# ============================================================================

print("\n[6/8] Fitting location-scale regression models...")
print("  (Models both MEAN and LOG(VARIANCE) as functions of UCLA × Gender + DASS)")

location_scale_results = []

# Define outcomes to model
outcomes = {
    'PRP τ (long SOA)': 'prp_tau_long',
    'PRP σ (long SOA)': 'prp_sigma_long',
    'PRP RMSSD': 'prp_rmssd',
    'PRP Kurtosis': 'prp_kurtosis',
    'WCST τ': 'wcst_tau',
    'WCST σ': 'wcst_sigma',
    'WCST RMSSD': 'wcst_rmssd',
    'WCST Kurtosis': 'wcst_kurtosis',
    'Stroop τ (incongruent)': 'stroop_tau_incong',
    'Stroop σ (incongruent)': 'stroop_sigma_incong',
    'Stroop RMSSD': 'stroop_rmssd',
    'Stroop Kurtosis': 'stroop_kurtosis'
}

for outcome_label, outcome_var in outcomes.items():
    df_outcome = df_clean.dropna(subset=[outcome_var])

    if len(df_outcome) < 30:
        print(f"  ⚠ Skipping {outcome_label}: N = {len(df_outcome)} < 30")
        continue

    # Model 1: Mean (location) - hierarchical regression with DASS control
    formula_mean = f"{outcome_var} ~ z_ucla_total * gender_male + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

    try:
        model_mean = smf.ols(formula_mean, data=df_outcome).fit()

        location_scale_results.append({
            'outcome': outcome_label,
            'component': 'Mean (μ)',
            'N': len(df_outcome),
            'ucla_beta': model_mean.params.get('z_ucla_total', np.nan),
            'ucla_se': model_mean.bse.get('z_ucla_total', np.nan),
            'ucla_p': model_mean.pvalues.get('z_ucla_total', np.nan),
            'gender_beta': model_mean.params.get('gender_male', np.nan),
            'gender_p': model_mean.pvalues.get('gender_male', np.nan),
            'interaction_beta': model_mean.params.get('z_ucla_total:gender_male', np.nan),
            'interaction_p': model_mean.pvalues.get('z_ucla_total:gender_male', np.nan),
            'r_squared': model_mean.rsquared
        })

    except Exception as e:
        print(f"  ⚠ Error fitting mean model for {outcome_label}: {e}")

    # Model 2: Variance (scale) - compute residual variance from mean model
    # Then model log(squared residuals) as proxy for variance
    try:
        df_outcome['residual'] = model_mean.resid
        df_outcome['log_sq_residual'] = np.log(df_outcome['residual'] ** 2 + 1)  # +1 to avoid log(0)

        formula_var = "log_sq_residual ~ z_ucla_total * gender_male + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
        model_var = smf.ols(formula_var, data=df_outcome).fit()

        location_scale_results.append({
            'outcome': outcome_label,
            'component': 'Variance (log σ²)',
            'N': len(df_outcome),
            'ucla_beta': model_var.params.get('z_ucla_total', np.nan),
            'ucla_se': model_var.bse.get('z_ucla_total', np.nan),
            'ucla_p': model_var.pvalues.get('z_ucla_total', np.nan),
            'gender_beta': model_var.params.get('gender_male', np.nan),
            'gender_p': model_var.pvalues.get('gender_male', np.nan),
            'interaction_beta': model_var.params.get('z_ucla_total:gender_male', np.nan),
            'interaction_p': model_var.pvalues.get('z_ucla_total:gender_male', np.nan),
            'r_squared': model_var.rsquared
        })

    except Exception as e:
        print(f"  ⚠ Error fitting variance model for {outcome_label}: {e}")

location_scale_df = pd.DataFrame(location_scale_results)
output_ls = OUTPUT_DIR / "paper1_location_scale_results.csv"
location_scale_df.to_csv(output_ls, index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: {output_ls}")
print(f"  ✓ Fitted {len(location_scale_results)} models ({len(outcomes)} outcomes × 2 components)")

# ============================================================================
# GENDER-STRATIFIED COMPARISONS
# ============================================================================

print("\n[7/8] Computing gender-stratified correlations and comparisons...")

gender_comparisons = []

for outcome_label, outcome_var in outcomes.items():
    df_outcome = df_clean.dropna(subset=[outcome_var, 'z_ucla_total'])

    if len(df_outcome) < 30:
        continue

    # Overall correlation
    r_all, p_all = stats.pearsonr(df_outcome['z_ucla_total'], df_outcome[outcome_var])

    # Male-only correlation
    df_male = df_outcome[df_outcome['gender'] == 'male']
    if len(df_male) >= 20:
        r_male, p_male = stats.pearsonr(df_male['z_ucla_total'], df_male[outcome_var])
    else:
        r_male, p_male = np.nan, np.nan

    # Female-only correlation
    df_female = df_outcome[df_outcome['gender'] == 'female']
    if len(df_female) >= 20:
        r_female, p_female = stats.pearsonr(df_female['z_ucla_total'], df_female[outcome_var])
    else:
        r_female, p_female = np.nan, np.nan

    gender_comparisons.append({
        'outcome': outcome_label,
        'N_total': len(df_outcome),
        'N_male': len(df_male),
        'N_female': len(df_female),
        'r_all': r_all,
        'p_all': p_all,
        'r_male': r_male,
        'p_male': p_male,
        'r_female': r_female,
        'p_female': p_female,
        'fisher_z_diff': np.nan if pd.isna([r_male, r_female]).any() else
                         (np.arctanh(r_male) - np.arctanh(r_female)) /
                         np.sqrt(1/(len(df_male)-3) + 1/(len(df_female)-3))
    })

gender_comp_df = pd.DataFrame(gender_comparisons)
output_gender = OUTPUT_DIR / "paper1_gender_stratified_comparisons.csv"
gender_comp_df.to_csv(output_gender, index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: {output_gender}")

# ============================================================================
# FOREST PLOT: VARIANCE EFFECTS
# ============================================================================

print("\n[8/8] Creating forest plot of variance effects...")

# Extract variance components only (not mean)
variance_effects = location_scale_df[location_scale_df['component'] == 'Variance (log σ²)'].copy()

# Compute 95% CIs
variance_effects['ucla_ci_lower'] = variance_effects['ucla_beta'] - 1.96 * variance_effects['ucla_se']
variance_effects['ucla_ci_upper'] = variance_effects['ucla_beta'] + 1.96 * variance_effects['ucla_se']

# Sort by effect size
variance_effects = variance_effects.sort_values('ucla_beta')

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))

y_pos = np.arange(len(variance_effects))

# Plot each point individually to handle colors properly
for i, (idx, row) in enumerate(variance_effects.iterrows()):
    # Determine color based on p-value
    color = 'red' if (pd.notna(row['ucla_p']) and row['ucla_p'] < 0.05) else 'gray'

    # Error bars
    ax.errorbar(
        row['ucla_beta'],
        i,
        xerr=[[row['ucla_beta'] - row['ucla_ci_lower']],
              [row['ucla_ci_upper'] - row['ucla_beta']]],
        fmt='o',
        ecolor=color,
        elinewidth=2,
        capsize=4,
        markersize=8,
        markerfacecolor=color,
        markeredgecolor='black',
        markeredgewidth=0.5
    )

ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(variance_effects['outcome'], fontsize=10)
ax.set_xlabel('UCLA β (standardized) on log(Variance)', fontsize=12, fontweight='bold')
ax.set_title('Forest Plot: UCLA Effects on RT Variability\n(DASS-controlled, N varies by outcome)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add p-value annotations
for i, (idx, row) in enumerate(variance_effects.iterrows()):
    if pd.notna(row['ucla_p']):
        p_str = f"p={row['ucla_p']:.3f}" if row['ucla_p'] >= 0.001 else "p<.001"
        p_color = 'red' if row['ucla_p'] < 0.05 else 'gray'
    else:
        p_str = "p=N/A"
        p_color = 'gray'

    ax.text(row['ucla_ci_upper'] + 0.02, i, p_str,
            va='center', ha='left', fontsize=8,
            color=p_color)

plt.tight_layout()
output_plot = OUTPUT_DIR / "paper1_variance_forest_plot.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_plot}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 90)
print("GENERATING SUMMARY REPORT...")
print("=" * 90)

summary_lines = []
summary_lines.append("=" * 90)
summary_lines.append(" " * 20 + "PAPER 1: DISTRIBUTIONAL VARIANCE ANALYSIS")
summary_lines.append(" " * 20 + "SUMMARY REPORT")
summary_lines.append("=" * 90)
summary_lines.append("")
summary_lines.append(f"Analysis Date: 2025-01-17")
summary_lines.append(f"Final Sample Size: N = {len(df_clean)}")
summary_lines.append(f"  Males: {(df_clean['gender'] == 'male').sum()}")
summary_lines.append(f"  Females: {(df_clean['gender'] == 'female').sum()}")
summary_lines.append("")

summary_lines.append("KEY FINDINGS:")
summary_lines.append("-" * 90)

# Count significant variance effects
sig_variance = variance_effects[variance_effects['ucla_p'] < 0.05]
summary_lines.append(f"\n1. UCLA EFFECTS ON VARIANCE (DASS-controlled):")
summary_lines.append(f"   - {len(sig_variance)}/{len(variance_effects)} variance parameters show significant UCLA effects (p<.05)")

if len(sig_variance) > 0:
    summary_lines.append(f"\n   Significant effects:")
    for _, row in sig_variance.iterrows():
        summary_lines.append(f"     • {row['outcome']}: β = {row['ucla_beta']:.3f}, p = {row['ucla_p']:.3f}")

# Gender differences in correlations
sig_gender_diff = gender_comp_df[gender_comp_df['fisher_z_diff'].abs() > 1.96]  # z > 1.96 ~ p < .05
summary_lines.append(f"\n2. GENDER-SPECIFIC PATTERNS:")
summary_lines.append(f"   - {len(sig_gender_diff)}/{len(gender_comp_df)} outcomes show significant gender differences in UCLA correlation")

if len(sig_gender_diff) > 0:
    summary_lines.append(f"\n   Outcomes with gender differences:")
    for _, row in sig_gender_diff.iterrows():
        summary_lines.append(f"     • {row['outcome']}:")
        summary_lines.append(f"       Males: r = {row['r_male']:.3f}, p = {row['p_male']:.3f}")
        summary_lines.append(f"       Females: r = {row['r_female']:.3f}, p = {row['p_female']:.3f}")

summary_lines.append("\n" + "=" * 90)
summary_lines.append("OUTPUT FILES:")
summary_lines.append("-" * 90)
summary_lines.append(f"  1. {output_metrics.name}")
summary_lines.append(f"  2. {output_ls.name}")
summary_lines.append(f"  3. {output_gender.name}")
summary_lines.append(f"  4. {output_plot.name}")
summary_lines.append(f"  5. PAPER1_SUMMARY_REPORT.txt (this file)")
summary_lines.append("=" * 90)

# Write report
output_report = OUTPUT_DIR / "PAPER1_SUMMARY_REPORT.txt"
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print('\n'.join(summary_lines))
print(f"\n✓ Saved: {output_report}")

print("\n" + "=" * 90)
print("PAPER 1 ANALYSIS COMPLETE!")
print("=" * 90)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Review forest plot and identify strongest variance effects")
print("  2. Examine gender-stratified comparisons for interaction patterns")
print("  3. Consider location-scale models for key outcomes (PRP tau, WCST sigma)")
print("  4. Draft Results section focusing on 'noise structure' narrative")
