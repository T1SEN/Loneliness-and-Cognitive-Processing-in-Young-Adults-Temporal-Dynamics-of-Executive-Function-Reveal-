"""
Loneliness and Executive Function: DDM-Based Analysis
=====================================================
Connects UCLA loneliness to DDM parameters from Stroop and PRP tasks.

Research Question:
Does loneliness predict executive function impairments, controlling for
depression, anxiety, and stress (DASS-21)?

Key hypotheses:
H1: Loneliness -> Lower Stroop interference control (higher interference)
H2: Loneliness -> Greater PRP bottleneck effect
H3: Effects remain after controlling for DASS

This uses DDM-derived parameters instead of summary RT measures.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_style('whitegrid')

def load_and_merge_data():
    """Load and merge all data sources."""
    print("\n" + "=" * 70)
    print("LONELINESS & EXECUTIVE FUNCTION: DDM PARAMETER ANALYSIS")
    print("=" * 70)

    print("\nLoading data sources...")

    # Load surveys
    surveys = pd.read_csv('results/2_surveys_results.csv', encoding='utf-8-sig')

    # Pivot to wide format
    ucla_data = surveys[surveys['surveyName'] == 'ucla'][['participantId', 'score']].copy()
    ucla_data.columns = ['participant_id', 'ucla_total']

    dass_data = surveys[surveys['surveyName'] == 'dass'][['participantId', 'score_A', 'score_S', 'score_D']].copy()
    dass_data.columns = ['participant_id', 'dass_anxiety', 'dass_stress', 'dass_depression']

    # Merge surveys
    survey_df = ucla_data.merge(dass_data, on='participant_id', how='inner')
    print(f"  Surveys: {len(survey_df)} participants with UCLA + DASS")

    # Load Stroop DDM parameters
    stroop_params = pd.read_csv('results/analysis_outputs/stroop_ddm_parameters.csv')
    print(f"  Stroop DDM: {len(stroop_params)} participants")

    # Load PRP bottleneck parameters
    prp_params = pd.read_csv('results/analysis_outputs/prp_bottleneck_parameters.csv')
    print(f"  PRP bottleneck: {len(prp_params)} participants")

    # Merge all
    df = survey_df.merge(stroop_params, on='participant_id', how='inner')
    df = df.merge(prp_params, on='participant_id', how='inner')

    print(f"\n  Final merged dataset: {len(df)} participants")

    # Drop rows with missing values
    initial_n = len(df)
    df = df.dropna()
    if len(df) < initial_n:
        print(f"  Dropped {initial_n - len(df)} participants with missing values")

    return df

def descriptive_statistics(df):
    """Print descriptive statistics."""
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # UCLA loneliness
    print(f"\nUCLA Loneliness:")
    print(f"  Mean: {df['ucla_total'].mean():.2f} (SD: {df['ucla_total'].std():.2f})")
    print(f"  Range: {df['ucla_total'].min():.0f} - {df['ucla_total'].max():.0f}")

    # DASS
    print(f"\nDASS-21:")
    print(f"  Anxiety:     {df['dass_anxiety'].mean():.2f} (SD: {df['dass_anxiety'].std():.2f})")
    print(f"  Stress:      {df['dass_stress'].mean():.2f} (SD: {df['dass_stress'].std():.2f})")
    print(f"  Depression:  {df['dass_depression'].mean():.2f} (SD: {df['dass_depression'].std():.2f})")

    # Stroop
    print(f"\nStroop DDM Parameters:")
    print(f"  Drift (congruent):    {df['drift_congruent'].mean():.3f} (SD: {df['drift_congruent'].std():.3f})")
    print(f"  Drift (incongruent):  {df['drift_incongruent'].mean():.3f} (SD: {df['drift_incongruent'].std():.3f})")
    print(f"  Interference:         {df['stroop_interference'].mean():.3f} (SD: {df['stroop_interference'].std():.3f})")

    # PRP
    print(f"\nPRP Bottleneck:")
    print(f"  Effect (s):  {df['prp_bottleneck_effect'].mean():.3f} (SD: {df['prp_bottleneck_effect'].std():.3f})")
    print(f"  Effect (ms): {df['prp_bottleneck_effect'].mean()*1000:.1f} (SD: {df['prp_bottleneck_effect'].std()*1000:.1f})")

def simple_correlations(df):
    """Compute simple correlations."""
    print("\n" + "=" * 70)
    print("SIMPLE CORRELATIONS (Pearson r)")
    print("=" * 70)

    vars_of_interest = [
        ('ucla_total', 'UCLA Loneliness'),
        ('dass_anxiety', 'DASS Anxiety'),
        ('dass_stress', 'DASS Stress'),
        ('dass_depression', 'DASS Depression'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck_effect', 'PRP Bottleneck')
    ]

    print(f"\nLoneliness correlations:")
    for var, label in vars_of_interest[1:]:
        r, p = stats.pearsonr(df['ucla_total'], df[var])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {label:25s}: r = {r:6.3f}, p = {p:.4f} {sig}")

    return

def standardize_variables(df):
    """Z-score standardize all variables for Bayesian modeling."""
    print("\nStandardizing variables (z-scores)...")

    vars_to_standardize = [
        'ucla_total',
        'dass_anxiety', 'dass_stress', 'dass_depression',
        'stroop_interference',
        'prp_bottleneck_effect'
    ]

    for var in vars_to_standardize:
        df[f'z_{var}'] = (df[var] - df[var].mean()) / df[var].std()

    return df

def bayesian_regression_stroop(df):
    """
    Bayesian regression: UCLA -> Stroop interference, controlling for DASS.

    Model:
        stroop_interference ~ Normal(mu, sigma)
        mu = alpha + beta_ucla * UCLA + beta_dass_anx * DASS_anx +
             beta_dass_str * DASS_str + beta_dass_dep * DASS_dep
    """
    print("\n" + "=" * 70)
    print("BAYESIAN REGRESSION: UCLA -> STROOP INTERFERENCE")
    print("=" * 70)

    with pm.Model() as stroop_model:
        # Predictors
        ucla = df['z_ucla_total'].values
        dass_anx = df['z_dass_anxiety'].values
        dass_str = df['z_dass_stress'].values
        dass_dep = df['z_dass_depression'].values

        # Outcome
        stroop = df['z_stroop_interference'].values

        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta_ucla = pm.Normal('beta_ucla', mu=0, sigma=1)
        beta_dass_anx = pm.Normal('beta_dass_anx', mu=0, sigma=1)
        beta_dass_str = pm.Normal('beta_dass_str', mu=0, sigma=1)
        beta_dass_dep = pm.Normal('beta_dass_dep', mu=0, sigma=1)

        # Linear model
        mu = alpha + beta_ucla * ucla + beta_dass_anx * dass_anx + \
             beta_dass_str * dass_str + beta_dass_dep * dass_dep

        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=stroop)

    print("  Fitting model...")
    with stroop_model:
        trace_stroop = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                                return_inferencedata=True, random_seed=42)

    print("  [OK] Sampling complete")

    # Summary
    summary = az.summary(trace_stroop, var_names=['alpha', 'beta_ucla', 'beta_dass_anx',
                                                   'beta_dass_str', 'beta_dass_dep'],
                        round_to=3)
    print("\n" + summary.to_string())

    return trace_stroop

def bayesian_regression_prp(df):
    """
    Bayesian regression: UCLA -> PRP bottleneck, controlling for DASS.
    """
    print("\n" + "=" * 70)
    print("BAYESIAN REGRESSION: UCLA -> PRP BOTTLENECK")
    print("=" * 70)

    with pm.Model() as prp_model:
        # Predictors
        ucla = df['z_ucla_total'].values
        dass_anx = df['z_dass_anxiety'].values
        dass_str = df['z_dass_stress'].values
        dass_dep = df['z_dass_depression'].values

        # Outcome
        prp = df['z_prp_bottleneck_effect'].values

        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta_ucla = pm.Normal('beta_ucla', mu=0, sigma=1)
        beta_dass_anx = pm.Normal('beta_dass_anx', mu=0, sigma=1)
        beta_dass_str = pm.Normal('beta_dass_str', mu=0, sigma=1)
        beta_dass_dep = pm.Normal('beta_dass_dep', mu=0, sigma=1)

        # Linear model
        mu = alpha + beta_ucla * ucla + beta_dass_anx * dass_anx + \
             beta_dass_str * dass_str + beta_dass_dep * dass_dep

        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=prp)

    print("  Fitting model...")
    with prp_model:
        trace_prp = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                             return_inferencedata=True, random_seed=42)

    print("  [OK] Sampling complete")

    # Summary
    summary = az.summary(trace_prp, var_names=['alpha', 'beta_ucla', 'beta_dass_anx',
                                               'beta_dass_str', 'beta_dass_dep'],
                        round_to=3)
    print("\n" + summary.to_string())

    return trace_prp

def plot_results(trace_stroop, trace_prp, df, output_dir='results/analysis_outputs'):
    """Plot regression results."""
    print("\nGenerating plots...")
    output_dir = Path(output_dir)

    # Forest plot of coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stroop
    az.plot_forest(trace_stroop,
                  var_names=['beta_ucla', 'beta_dass_anx', 'beta_dass_str', 'beta_dass_dep'],
                  combined=True,
                  hdi_prob=0.95,
                  ax=axes[0])
    axes[0].set_title('Stroop Interference Predictors', fontweight='bold')
    axes[0].set_xlabel('Standardized Coefficient')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

    # PRP
    az.plot_forest(trace_prp,
                  var_names=['beta_ucla', 'beta_dass_anx', 'beta_dass_str', 'beta_dass_dep'],
                  combined=True,
                  hdi_prob=0.95,
                  ax=axes[1])
    axes[1].set_title('PRP Bottleneck Predictors', fontweight='bold')
    axes[1].set_xlabel('Standardized Coefficient')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'loneliness_ef_forest_plot.png', dpi=300, bbox_inches='tight')
    print(f"  Saved forest plot")
    plt.close()

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stroop
    axes[0].scatter(df['ucla_total'], df['stroop_interference'], alpha=0.6)
    axes[0].set_xlabel('UCLA Loneliness', fontweight='bold')
    axes[0].set_ylabel('Stroop Interference', fontweight='bold')
    axes[0].set_title('Loneliness vs Stroop Interference')

    # Add regression line
    z = np.polyfit(df['ucla_total'], df['stroop_interference'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ucla_total'].min(), df['ucla_total'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # PRP
    axes[1].scatter(df['ucla_total'], df['prp_bottleneck_effect'], alpha=0.6)
    axes[1].set_xlabel('UCLA Loneliness', fontweight='bold')
    axes[1].set_ylabel('PRP Bottleneck Effect (s)', fontweight='bold')
    axes[1].set_title('Loneliness vs PRP Bottleneck')

    # Add regression line
    z = np.polyfit(df['ucla_total'], df['prp_bottleneck_effect'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ucla_total'].min(), df['ucla_total'].max(), 100)
    axes[1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'loneliness_ef_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  Saved scatter plots")
    plt.close()

def interpret_results(trace_stroop, trace_prp):
    """Interpret and report key findings."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Stroop
    print("\n1. STROOP INTERFERENCE:")
    beta_ucla_stroop = trace_stroop.posterior['beta_ucla'].values.flatten()
    prob_positive = (beta_ucla_stroop > 0).mean()
    print(f"   Beta (UCLA): {beta_ucla_stroop.mean():.3f}")
    print(f"   95% HDI: [{np.percentile(beta_ucla_stroop, 2.5):.3f}, {np.percentile(beta_ucla_stroop, 97.5):.3f}]")
    print(f"   P(beta > 0): {prob_positive:.1%}")

    if prob_positive > 0.95:
        print(f"   ==> Loneliness INCREASES Stroop interference (supports H1)")
    elif prob_positive < 0.05:
        print(f"   ==> Loneliness DECREASES Stroop interference (opposite direction)")
    else:
        print(f"   ==> No clear evidence for loneliness effect on Stroop")

    # PRP
    print("\n2. PRP BOTTLENECK:")
    beta_ucla_prp = trace_prp.posterior['beta_ucla'].values.flatten()
    prob_positive = (beta_ucla_prp > 0).mean()
    print(f"   Beta (UCLA): {beta_ucla_prp.mean():.3f}")
    print(f"   95% HDI: [{np.percentile(beta_ucla_prp, 2.5):.3f}, {np.percentile(beta_ucla_prp, 97.5):.3f}]")
    print(f"   P(beta > 0): {prob_positive:.1%}")

    if prob_positive > 0.95:
        print(f"   ==> Loneliness INCREASES PRP bottleneck (supports H2)")
    elif prob_positive < 0.05:
        print(f"   ==> Loneliness DECREASES PRP bottleneck (opposite direction)")
    else:
        print(f"   ==> No clear evidence for loneliness effect on PRP")

    print("\n" + "=" * 70)

def main():
    """Main analysis pipeline."""

    # Load and merge data
    df = load_and_merge_data()

    # Descriptive statistics
    descriptive_statistics(df)

    # Simple correlations
    simple_correlations(df)

    # Standardize
    df = standardize_variables(df)

    # Bayesian regressions
    trace_stroop = bayesian_regression_stroop(df)
    trace_prp = bayesian_regression_prp(df)

    # Save traces
    output_dir = Path('results/analysis_outputs')
    trace_stroop.to_netcdf(output_dir / 'loneliness_stroop_trace.nc')
    trace_prp.to_netcdf(output_dir / 'loneliness_prp_trace.nc')
    print("\n[OK] Saved traces")

    # Plots
    plot_results(trace_stroop, trace_prp, df)

    # Interpret
    interpret_results(trace_stroop, trace_prp)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return df, trace_stroop, trace_prp

if __name__ == '__main__':
    df, trace_stroop, trace_prp = main()
