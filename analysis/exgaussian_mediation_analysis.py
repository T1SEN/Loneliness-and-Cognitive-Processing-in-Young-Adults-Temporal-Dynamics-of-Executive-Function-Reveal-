"""
Ex-Gaussian Mediation Analysis
=================================
Tests mediation pathways: UCLA → Ex-Gaussian parameters (τ, σ, μ) → Behavioral outcomes

Key hypotheses:
1. PRP: UCLA → τ (long SOA) → Bottleneck effect (males only)
2. PRP: UCLA → σ (bottleneck) → T2 RT variability (males only)
3. WCST: UCLA → Stroop RT variability → WCST PE rate (exploratory)

Method: Bootstrap mediation (10,000 iterations) with 95% CI
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For bootstrap mediation
from sklearn.utils import resample

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/mediation_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

N_BOOTSTRAP = 10000
ALPHA = 0.05

print("=" * 80)
print("EX-GAUSSIAN MEDIATION ANALYSIS")
print("=" * 80)
print(f"\nBootstrap iterations: {N_BOOTSTRAP}")
print(f"Confidence level: {(1-ALPHA)*100}%\n")

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Participant info and surveys
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
surveys = surveys.rename(columns={'participantId': 'participant_id'})

# Get UCLA scores
if 'surveyName' in surveys.columns:
    ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
elif 'survey' in surveys.columns:
    ucla_data = surveys[surveys['survey'].str.lower() == 'ucla'].copy()
else:
    raise KeyError("No survey name column found")

ucla_scores = ucla_data.groupby('participant_id')['score'].sum().reset_index()
ucla_scores.columns = ['participant_id', 'ucla_total']

# Merge with demographics
master = participants[['participant_id', 'age', 'gender', 'education']].merge(
    ucla_scores, on='participant_id', how='inner'
)

print(f"  Loaded {len(master)} participants with UCLA scores")

# Ex-Gaussian parameters - Stroop
stroop_exg = pd.read_csv("results/analysis_outputs/mechanism_analysis/exgaussian/exgaussian_parameters.csv")
if stroop_exg['participant_id'].dtype == 'O' and stroop_exg['participant_id'].iloc[0].startswith('\ufeff'):
    stroop_exg['participant_id'] = stroop_exg['participant_id'].str.replace('\ufeff', '')
stroop_exg = stroop_exg[['participant_id', 'mu', 'sigma', 'tau']].copy()
stroop_exg.columns = ['participant_id', 'stroop_mu', 'stroop_sigma', 'stroop_tau']

# Ex-Gaussian parameters - PRP
prp_exg = pd.read_csv("results/analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv")
if prp_exg['participant_id'].dtype == 'O' and prp_exg['participant_id'].iloc[0].startswith('\ufeff'):
    prp_exg['participant_id'] = prp_exg['participant_id'].str.replace('\ufeff', '')

# Calculate bottleneck parameters (short - long for τ and σ)
prp_exg['prp_tau_long'] = prp_exg['long_tau']
prp_exg['prp_sigma_long'] = prp_exg['long_sigma']
prp_exg['prp_mu_long'] = prp_exg['long_mu']
prp_exg['prp_tau_short'] = prp_exg['short_tau']
prp_exg['prp_sigma_short'] = prp_exg['short_sigma']
prp_exg['prp_mu_short'] = prp_exg['short_mu']
prp_exg['prp_tau_bottleneck'] = prp_exg['short_tau'] - prp_exg['long_tau']
prp_exg['prp_sigma_bottleneck'] = prp_exg['short_sigma'] - prp_exg['long_sigma']
prp_exg['prp_mu_bottleneck'] = prp_exg['short_mu'] - prp_exg['long_mu']

prp_exg = prp_exg[['participant_id', 'prp_tau_long', 'prp_sigma_long', 'prp_mu_long',
                    'prp_tau_short', 'prp_sigma_short', 'prp_mu_short',
                    'prp_tau_bottleneck', 'prp_sigma_bottleneck', 'prp_mu_bottleneck']].copy()

# Load WCST data for PE rate
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
if 'participant_id' not in wcst_trials.columns and 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})
elif 'participant_id' in wcst_trials.columns and 'participantId' in wcst_trials.columns:
    # Both exist - drop participantId
    wcst_trials = wcst_trials.drop(columns=['participantId'])

# Parse extra field for PE
import ast
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_dict'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

# Calculate PE rate
wcst_summary = wcst_trials.groupby('participant_id').agg(
    pe_count=('is_pe', 'sum'),
    total_trials=('is_pe', 'count'),
    mean_rt=('rt_ms', 'mean'),
    sd_rt=('rt_ms', 'std')
).reset_index()
wcst_summary['pe_rate'] = (wcst_summary['pe_count'] / wcst_summary['total_trials']) * 100

# Load PRP trials for outcome measures
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")
if 'participant_id' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participant_id'])
if 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.rename(columns={'participantId': 'participant_id'})

# Filter valid PRP trials
prp_trials = prp_trials[
    (prp_trials['t1_timeout'] == False) &
    (prp_trials['t2_timeout'] == False) &
    (prp_trials['t2_rt'] > 0)
].copy()

# Bin SOA
def bin_soa(soa):
    if soa <= 150:
        return 'short'
    elif 300 <= soa <= 600:
        return 'medium'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'

prp_trials['soa_bin'] = prp_trials['soa'].apply(bin_soa)
prp_trials = prp_trials[prp_trials['soa_bin'].isin(['short', 'long'])].copy()

# Calculate PRP bottleneck effect and T2 variability
prp_summary = prp_trials.groupby(['participant_id', 'soa_bin']).agg(
    t2_rt_mean=('t2_rt', 'mean'),
    t2_rt_sd=('t2_rt', 'std'),
    n_trials=('t2_rt', 'count')
).reset_index()

prp_wide = prp_summary.pivot(index='participant_id', columns='soa_bin',
                              values=['t2_rt_mean', 't2_rt_sd']).reset_index()
prp_wide.columns = ['_'.join(col).strip('_') for col in prp_wide.columns.values]

# Create bottleneck effect from pivoted columns
if 't2_rt_mean_short' in prp_wide.columns and 't2_rt_mean_long' in prp_wide.columns:
    prp_wide['bottleneck_effect'] = prp_wide['t2_rt_mean_short'] - prp_wide['t2_rt_mean_long']

# Merge all data
master = master.merge(stroop_exg, on='participant_id', how='left')
master = master.merge(prp_exg, on='participant_id', how='left')
master = master.merge(wcst_summary[['participant_id', 'pe_rate', 'sd_rt']], on='participant_id', how='left')

# Merge PRP wide with appropriate columns
merge_cols = ['participant_id']
if 'bottleneck_effect' in prp_wide.columns:
    merge_cols.append('bottleneck_effect')
if 't2_rt_sd_long' in prp_wide.columns:
    merge_cols.append('t2_rt_sd_long')
if 't2_rt_sd_short' in prp_wide.columns:
    merge_cols.append('t2_rt_sd_short')
master = master.merge(prp_wide[merge_cols], on='participant_id', how='left')

# Create gender dummy - more robust handling
gender_lower = master['gender'].astype(str).str.lower()
master['gender_male'] = (
    gender_lower.str.contains('male', na=False) |  # English
    gender_lower.str.contains('남', na=False) |     # Korean '남성'
    (gender_lower == 'm')                          # Short form
).astype(int)

# Drop missing
master = master.dropna(subset=['ucla_total', 'gender_male']).copy()

print(f"  Final N = {len(master)} (with complete data)")
print(f"  Males: {master['gender_male'].sum()}, Females: {(1-master['gender_male']).sum()}")

# Debug: Check available columns
print(f"\n  Available Ex-Gaussian columns:")
exg_cols = [col for col in master.columns if 'tau' in col or 'sigma' in col or 'bottleneck' in col or 't2_' in col]
print(f"    {', '.join(exg_cols)}")
print()

# ============================================================================
# Bootstrap Mediation Function
# ============================================================================

def bootstrap_mediation(X, M, Y, n_bootstrap=10000, alpha=0.05):
    """
    Bootstrap mediation analysis

    Parameters:
    -----------
    X : array-like
        Predictor (e.g., UCLA)
    M : array-like
        Mediator (e.g., tau)
    Y : array-like
        Outcome (e.g., bottleneck effect)
    n_bootstrap : int
        Number of bootstrap iterations
    alpha : float
        Significance level

    Returns:
    --------
    dict with:
        - a_path: X → M coefficient
        - b_path: M → Y coefficient (controlling X)
        - c_path: Total effect (X → Y)
        - c_prime_path: Direct effect (X → Y, controlling M)
        - indirect_effect: a * b
        - indirect_ci_lower: Bootstrap CI lower bound
        - indirect_ci_upper: Bootstrap CI upper bound
        - proportion_mediated: indirect / total effect
    """
    X = np.array(X)
    M = np.array(M)
    Y = np.array(Y)

    # Remove NaN
    valid = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
    X = X[valid]
    M = M[valid]
    Y = Y[valid]

    n = len(X)
    if n < 10:
        return None

    # Path a: X → M
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(X, M)

    # Path b: M → Y (controlling X)
    # Multiple regression: Y ~ M + X
    X_mat = np.column_stack([np.ones(n), M, X])
    try:
        beta = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
        b_path = beta[1]
        c_prime_path = beta[2]
    except:
        return None

    # Path c: X → Y (total effect)
    slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(X, Y)

    # Indirect effect
    indirect_effect = slope_a * b_path

    # Bootstrap for CI
    indirect_boots = []
    for _ in range(n_bootstrap):
        indices = resample(np.arange(n), n_samples=n, replace=True)
        X_boot = X[indices]
        M_boot = M[indices]
        Y_boot = Y[indices]

        # Path a (bootstrap)
        slope_a_boot, _, _, _, _ = stats.linregress(X_boot, M_boot)

        # Path b (bootstrap)
        X_mat_boot = np.column_stack([np.ones(n), M_boot, X_boot])
        try:
            beta_boot = np.linalg.lstsq(X_mat_boot, Y_boot, rcond=None)[0]
            b_path_boot = beta_boot[1]
            indirect_boots.append(slope_a_boot * b_path_boot)
        except:
            continue

    if len(indirect_boots) < 100:
        return None

    # Percentile CI
    ci_lower = np.percentile(indirect_boots, alpha/2 * 100)
    ci_upper = np.percentile(indirect_boots, (1 - alpha/2) * 100)

    # Proportion mediated
    if abs(slope_c) > 1e-6:
        proportion_mediated = indirect_effect / slope_c
    else:
        proportion_mediated = np.nan

    return {
        'n': n,
        'a_path': slope_a,
        'a_path_p': p_a,
        'b_path': b_path,
        'c_path': slope_c,
        'c_path_p': p_c,
        'c_prime_path': c_prime_path,
        'indirect_effect': indirect_effect,
        'indirect_ci_lower': ci_lower,
        'indirect_ci_upper': ci_upper,
        'indirect_sig': 'Yes' if (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0) else 'No',
        'proportion_mediated': proportion_mediated
    }

# ============================================================================
# Mediation Analyses
# ============================================================================

results_list = []

print("Running mediation analyses...")
print("-" * 80)

# --------------------------------------------------
# 1. PRP: UCLA → τ (long SOA) → Bottleneck effect
# --------------------------------------------------
print("\n1. PRP Mediation: UCLA → τ (long SOA) → Bottleneck Effect")
print("   " + "=" * 60)

if 'prp_tau_long' not in master.columns or 'bottleneck_effect' not in master.columns:
    print("   [SKIPPED: Required columns not available]")
else:
    for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
        df_sub = master[master['gender_male'] == gender_val].copy()
        print(f"\n   {gender_name} (N={len(df_sub)}):")

        result = bootstrap_mediation(
            df_sub['ucla_total'],
            df_sub['prp_tau_long'],
            df_sub['bottleneck_effect'],
            n_bootstrap=N_BOOTSTRAP,
            alpha=ALPHA
        )

        if result:
            print(f"      Path a (UCLA → τ_long):    β = {result['a_path']:.3f}, p = {result['a_path_p']:.4f}")
            print(f"      Path b (τ_long → Bottleneck): β = {result['b_path']:.3f}")
            print(f"      Path c (Total effect):     β = {result['c_path']:.3f}, p = {result['c_path_p']:.4f}")
            print(f"      Path c' (Direct effect):   β = {result['c_prime_path']:.3f}")
            print(f"      Indirect effect:           {result['indirect_effect']:.3f} [{result['indirect_ci_lower']:.3f}, {result['indirect_ci_upper']:.3f}]")
            print(f"      Mediation significant:     {result['indirect_sig']}")
            print(f"      Proportion mediated:       {result['proportion_mediated']:.2%}")

            results_list.append({
                'pathway': 'UCLA → τ_long → PRP Bottleneck',
                'gender': gender_name,
                **result
            })
        else:
            print("      Insufficient data")

# --------------------------------------------------
# 2. PRP: UCLA → σ (bottleneck) → T2 RT variability (long SOA)
# --------------------------------------------------
print("\n\n2. PRP Mediation: UCLA → σ (bottleneck) → T2 RT SD (long SOA)")
print("   " + "=" * 60)

if 'prp_sigma_bottleneck' not in master.columns or 't2_rt_sd_long' not in master.columns:
    print("   [SKIPPED: Required columns not available]")
else:
    for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
        df_sub = master[master['gender_male'] == gender_val].copy()
        print(f"\n   {gender_name} (N={len(df_sub)}):")

        result = bootstrap_mediation(
            df_sub['ucla_total'],
            df_sub['prp_sigma_bottleneck'],
            df_sub['t2_rt_sd_long'],
            n_bootstrap=N_BOOTSTRAP,
            alpha=ALPHA
        )

        if result:
            print(f"      Path a (UCLA → σ_bottleneck): β = {result['a_path']:.3f}, p = {result['a_path_p']:.4f}")
            print(f"      Path b (σ_bottleneck → T2_SD): β = {result['b_path']:.3f}")
            print(f"      Path c (Total effect):        β = {result['c_path']:.3f}, p = {result['c_path_p']:.4f}")
            print(f"      Path c' (Direct effect):      β = {result['c_prime_path']:.3f}")
            print(f"      Indirect effect:              {result['indirect_effect']:.3f} [{result['indirect_ci_lower']:.3f}, {result['indirect_ci_upper']:.3f}]")
            print(f"      Mediation significant:        {result['indirect_sig']}")
            print(f"      Proportion mediated:          {result['proportion_mediated']:.2%}")

            results_list.append({
                'pathway': 'UCLA → σ_bottleneck → T2 RT SD (long)',
                'gender': gender_name,
                **result
            })
        else:
            print("      Insufficient data")

# --------------------------------------------------
# 3. PRP: UCLA → τ (short SOA) → T2 RT SD (short SOA) [for females - compensatory]
# --------------------------------------------------
print("\n\n3. PRP Mediation: UCLA → τ (short SOA) → T2 RT SD (short SOA)")
print("   " + "=" * 60)

if 'prp_tau_short' not in master.columns or 't2_rt_sd_short' not in master.columns:
    print("   [SKIPPED: Required columns not available]")
else:
    for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
        df_sub = master[master['gender_male'] == gender_val].copy()
        print(f"\n   {gender_name} (N={len(df_sub)}):")

        result = bootstrap_mediation(
            df_sub['ucla_total'],
            df_sub['prp_tau_short'],
            df_sub['t2_rt_sd_short'],
            n_bootstrap=N_BOOTSTRAP,
            alpha=ALPHA
        )

        if result:
            print(f"      Path a (UCLA → τ_short):   β = {result['a_path']:.3f}, p = {result['a_path_p']:.4f}")
            print(f"      Path b (τ_short → T2_SD):  β = {result['b_path']:.3f}")
            print(f"      Path c (Total effect):     β = {result['c_path']:.3f}, p = {result['c_path_p']:.4f}")
            print(f"      Path c' (Direct effect):   β = {result['c_prime_path']:.3f}")
            print(f"      Indirect effect:           {result['indirect_effect']:.3f} [{result['indirect_ci_lower']:.3f}, {result['indirect_ci_upper']:.3f}]")
            print(f"      Mediation significant:     {result['indirect_sig']}")
            print(f"      Proportion mediated:       {result['proportion_mediated']:.2%}")

            results_list.append({
                'pathway': 'UCLA → τ_short → T2 RT SD (short)',
                'gender': gender_name,
                **result
            })
        else:
            print("      Insufficient data")

# --------------------------------------------------
# 4. WCST: UCLA → Stroop RT variability (σ) → WCST PE rate
# --------------------------------------------------
print("\n\n4. WCST Mediation: UCLA → Stroop σ → WCST PE rate")
print("   " + "=" * 60)

if 'stroop_sigma' not in master.columns or 'pe_rate' not in master.columns:
    print("   [SKIPPED: Required columns not available]")
else:
    for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
        df_sub = master[master['gender_male'] == gender_val].copy()
        print(f"\n   {gender_name} (N={len(df_sub)}):")

        result = bootstrap_mediation(
            df_sub['ucla_total'],
            df_sub['stroop_sigma'],
            df_sub['pe_rate'],
            n_bootstrap=N_BOOTSTRAP,
            alpha=ALPHA
        )

        if result:
            print(f"      Path a (UCLA → Stroop_σ): β = {result['a_path']:.3f}, p = {result['a_path_p']:.4f}")
            print(f"      Path b (Stroop_σ → PE):   β = {result['b_path']:.3f}")
            print(f"      Path c (Total effect):    β = {result['c_path']:.3f}, p = {result['c_path_p']:.4f}")
            print(f"      Path c' (Direct effect):  β = {result['c_prime_path']:.3f}")
            print(f"      Indirect effect:          {result['indirect_effect']:.3f} [{result['indirect_ci_lower']:.3f}, {result['indirect_ci_upper']:.3f}]")
            print(f"      Mediation significant:    {result['indirect_sig']}")
            print(f"      Proportion mediated:      {result['proportion_mediated']:.2%}")

            results_list.append({
                'pathway': 'UCLA → Stroop_σ → WCST PE rate',
                'gender': gender_name,
                **result
            })
        else:
            print("      Insufficient data")

# --------------------------------------------------
# 5. WCST: UCLA → WCST RT variability → WCST PE rate
# --------------------------------------------------
print("\n\n5. WCST Mediation: UCLA → WCST RT SD → WCST PE rate")
print("   " + "=" * 60)

if 'sd_rt' not in master.columns or 'pe_rate' not in master.columns:
    print("   [SKIPPED: Required columns not available]")
else:
    for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
        df_sub = master[master['gender_male'] == gender_val].copy()
        print(f"\n   {gender_name} (N={len(df_sub)}):")

        result = bootstrap_mediation(
            df_sub['ucla_total'],
            df_sub['sd_rt'],
            df_sub['pe_rate'],
            n_bootstrap=N_BOOTSTRAP,
            alpha=ALPHA
        )

        if result:
            print(f"      Path a (UCLA → WCST_SD): β = {result['a_path']:.3f}, p = {result['a_path_p']:.4f}")
            print(f"      Path b (WCST_SD → PE):   β = {result['b_path']:.3f}")
            print(f"      Path c (Total effect):   β = {result['c_path']:.3f}, p = {result['c_path_p']:.4f}")
            print(f"      Path c' (Direct effect): β = {result['c_prime_path']:.3f}")
            print(f"      Indirect effect:         {result['indirect_effect']:.3f} [{result['indirect_ci_lower']:.3f}, {result['indirect_ci_upper']:.3f}]")
            print(f"      Mediation significant:   {result['indirect_sig']}")
            print(f"      Proportion mediated:     {result['proportion_mediated']:.2%}")

            results_list.append({
                'pathway': 'UCLA → WCST_RT_SD → WCST PE rate',
                'gender': gender_name,
                **result
            })
        else:
            print("      Insufficient data")

# ============================================================================
# Save Results
# ============================================================================

if results_list:
    results_df = pd.DataFrame(results_list)

    # Save detailed results
    output_file = OUTPUT_DIR / "exgaussian_mediation_detailed.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n\nResults saved to: {output_file}")

    # Create summary table
    summary_cols = ['pathway', 'gender', 'n', 'a_path', 'a_path_p', 'b_path',
                    'c_path', 'c_path_p', 'indirect_effect', 'indirect_ci_lower',
                    'indirect_ci_upper', 'indirect_sig', 'proportion_mediated']
    summary_df = results_df[summary_cols].copy()

    summary_file = OUTPUT_DIR / "exgaussian_mediation_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"Summary saved to: {summary_file}")

print("\n" + "=" * 80)
print("MEDIATION ANALYSIS COMPLETE")
print("=" * 80)
print("\nKEY FINDINGS:")
print("-" * 80)

if results_list:
    for res in results_list:
        if res['indirect_sig'] == 'Yes':
            print(f"\n✓ SIGNIFICANT MEDIATION DETECTED:")
            print(f"  Pathway: {res['pathway']}")
            print(f"  Gender: {res['gender']}")
            print(f"  Indirect effect: {res['indirect_effect']:.3f} [{res['indirect_ci_lower']:.3f}, {res['indirect_ci_upper']:.3f}]")
            print(f"  Proportion mediated: {res['proportion_mediated']:.1%}")

print("\n" + "=" * 80)
