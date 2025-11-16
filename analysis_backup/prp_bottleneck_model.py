"""
PRP Bottleneck Model - Hierarchical Bayesian Implementation
===========================================================
Implements Pashler (1994) bottleneck model using PyMC.

Model structure:
    T2_RT = base_RT + max(0, T1_processing + bottleneck_delay - SOA) + noise

Parameters:
- T1_processing: Time to complete Task 1 response selection
- bottleneck_delay: Central bottleneck delay (CORE PARAMETER)
- base_RT: Task 2 motor/perceptual time (non-bottleneck components)

At short SOAs: T2 must wait for T1 to finish bottleneck stage
At long SOAs: T1 completes before T2 needs bottleneck, no waiting

Reference:
Pashler, H. (1994). Dual-task interference in simple tasks: Data and theory.
Psychological Bulletin, 116(2), 220-244.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

def load_preprocessed_data():
    """Load preprocessed PRP data."""
    print("\n" + "=" * 70)
    print("PRP BOTTLENECK MODEL - HIERARCHICAL BAYESIAN ANALYSIS")
    print("=" * 70)

    print("\nLoading preprocessed data...")
    df = pd.read_csv('analysis/data_prp_bottleneck.csv')
    print(f"  Loaded {len(df)} trials from {df['participant_id'].nunique()} participants")

    return df

def build_hierarchical_bottleneck_model(df):
    """
    Build hierarchical Pashler bottleneck model.

    Model:
        T2_RT ~ Normal(mu, sigma)
        mu = base_RT[subj] + slack
        slack = max(0, T1_proc[subj] + bottleneck[subj] - SOA)

    Priors:
        T1_processing ~ Normal(0.5, 0.2)  [seconds]
        bottleneck_delay ~ HalfNormal(0.3)  [seconds, positive by definition]
        base_RT ~ Normal(0.7, 0.2)  [seconds]
    """
    print("\nBuilding hierarchical bottleneck model...")

    # Prepare data arrays
    participant_idx = df['participant_idx'].values
    soa = df['soa_sec'].values
    t2_rt = df['t2_rt_sec'].values

    n_participants = df['participant_idx'].nunique()
    n_trials = len(df)

    print(f"  N participants: {n_participants}")
    print(f"  N trials: {n_trials}")

    with pm.Model() as bottleneck_model:
        # === Population-level hyperparameters ===

        # T1 processing time
        mu_t1_proc = pm.Normal('mu_t1_processing', mu=0.5, sigma=0.2)
        sigma_t1_proc = pm.HalfNormal('sigma_t1_processing', sigma=0.1)

        # Bottleneck delay (CORE PARAMETER)
        mu_bottleneck = pm.HalfNormal('mu_bottleneck_delay', sigma=0.3)
        sigma_bottleneck = pm.HalfNormal('sigma_bottleneck_delay', sigma=0.1)

        # Base RT (non-bottleneck components)
        mu_base_rt = pm.Normal('mu_base_rt', mu=0.7, sigma=0.2)
        sigma_base_rt = pm.HalfNormal('sigma_base_rt', sigma=0.1)

        # === Individual-level parameters ===

        # Each participant has their own parameters
        t1_proc = pm.Normal('t1_processing',
                           mu=mu_t1_proc,
                           sigma=sigma_t1_proc,
                           shape=n_participants)

        bottleneck = pm.TruncatedNormal(
            'bottleneck_delay',
            mu=mu_bottleneck,
            sigma=sigma_bottleneck,
            lower=0.0,
            shape=n_participants,
        )

        base_rt = pm.Normal('base_rt',
                           mu=mu_base_rt,
                           sigma=sigma_base_rt,
                           shape=n_participants)

        # === Likelihood ===

        # Calculate slack time (how much T2 must wait)
        # slack = max(0, T1_proc + bottleneck - SOA)
        required_time = t1_proc[participant_idx] + bottleneck[participant_idx]
        slack = pm.math.maximum(0, required_time - soa)

        # Predicted T2 RT
        mu_t2_rt = base_rt[participant_idx] + slack

        # Observation noise
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.2)

        # Likelihood
        pm.Normal('t2_rt_obs',
                 mu=mu_t2_rt,
                 sigma=sigma_obs,
                 observed=t2_rt)

    print("  Model structure:")
    print(f"    - Population parameters: 6")
    print(f"    - Individual parameters: {n_participants * 3}")
    print(f"    - Total free parameters: {len(bottleneck_model.free_RVs)}")

    return bottleneck_model

def fit_bottleneck_model(model, draws=2000, tune=2000, chains=4, target_accept=0.95):
    """Fit the bottleneck model using NUTS."""
    print(f"\nFitting bottleneck model...")
    print(f"  Draws: {draws}, Tune: {tune}, Chains: {chains}")
    print(f"  Target accept: {target_accept}")
    print(f"  This may take 10-20 minutes due to large model size...")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42
        )

    print("\n[OK] Sampling complete!")
    return trace

def diagnose_convergence(trace):
    """Check MCMC convergence."""
    print("\n" + "=" * 60)
    print("MCMC Convergence Diagnostics")
    print("=" * 60)

    # Focus on population-level parameters
    pop_vars = ['mu_t1_processing', 'mu_bottleneck_delay', 'mu_base_rt',
                'sigma_t1_processing', 'sigma_bottleneck_delay', 'sigma_base_rt']

    summary = az.summary(trace, var_names=pop_vars, round_to=3)

    # Check R-hat
    max_rhat = summary['r_hat'].max()
    problem_vars = summary[summary['r_hat'] > 1.01]

    print(f"\nR-hat (Gelman-Rubin statistic):")
    print(f"  Maximum: {max_rhat:.4f}")
    if len(problem_vars) > 0:
        print(f"  [WARNING] {len(problem_vars)} variables with R-hat > 1.01:")
        print(problem_vars[['mean', 'sd', 'r_hat']])
    else:
        print(f"  [OK] All variables R-hat < 1.01 (good convergence)")

    # Check ESS
    min_ess_bulk = summary['ess_bulk'].min()
    min_ess_tail = summary['ess_tail'].min()

    print(f"\nEffective Sample Size:")
    print(f"  Minimum ESS (bulk): {min_ess_bulk:.0f}")
    print(f"  Minimum ESS (tail): {min_ess_tail:.0f}")
    if min_ess_bulk < 400 or min_ess_tail < 400:
        print(f"  [WARNING] Some parameters have ESS < 400 (may need more samples)")
    else:
        print(f"  [OK] All parameters ESS > 400 (sufficient samples)")

    print("\n" + "=" * 60)

    # Full summary for all parameters
    full_summary = az.summary(trace, round_to=3)

    return summary, full_summary

def plot_posterior_distributions(trace, output_dir='results/analysis_outputs'):
    """Plot posterior distributions of population parameters."""
    print("\nGenerating posterior plots...")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot population-level parameters
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    var_names = [
        'mu_t1_processing',
        'mu_bottleneck_delay',
        'mu_base_rt',
        'sigma_t1_processing',
        'sigma_bottleneck_delay',
        'sigma_base_rt'
    ]

    titles = [
        'T1 Processing Time (s)',
        'Bottleneck Delay (s)',
        'Base RT (s)',
        'SD: T1 Processing',
        'SD: Bottleneck Delay',
        'SD: Base RT'
    ]

    for ax, var, title in zip(axes, var_names, titles):
        az.plot_posterior(trace, var_names=[var], ax=ax, hdi_prob=0.95)
        ax.set_title(title)

    plt.tight_layout()
    plot_file = output_dir / 'prp_bottleneck_posteriors.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved posterior plots to {plot_file}")
    plt.close()

    # Plot trace plots
    fig = az.plot_trace(trace, var_names=var_names, compact=True, figsize=(14, 10))
    plt.tight_layout()
    trace_file = output_dir / 'prp_bottleneck_trace.png'
    plt.savefig(trace_file, dpi=300, bbox_inches='tight')
    print(f"  Saved trace plots to {trace_file}")
    plt.close()

def plot_model_fit(trace, df, output_dir='results/analysis_outputs'):
    """
    Plot model predictions vs observed data.

    Shows T2 RT as function of SOA, with model predictions overlaid.
    """
    print("\nGenerating model fit plot...")

    # Get posterior means for population parameters
    posterior = trace.posterior
    mu_t1 = posterior['mu_t1_processing'].values.mean()
    mu_bottle = posterior['mu_bottleneck_delay'].values.mean()
    mu_base = posterior['mu_base_rt'].values.mean()

    # Calculate predicted T2 RT for range of SOAs
    soa_range = np.linspace(0.05, 1.2, 100)
    predicted_t2 = mu_base + np.maximum(0, mu_t1 + mu_bottle - soa_range)

    # Observed data
    obs_by_soa = df.groupby('soa_category')['t2_rt_sec'].agg(['mean', 'sem']).reset_index()
    soa_mapping = {'short': 0.075, 'medium_short': 0.175, 'medium_long': 0.45, 'long': 1.1}
    obs_by_soa['soa_numeric'] = obs_by_soa['soa_category'].map(soa_mapping)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Model predictions
    ax.plot(soa_range, predicted_t2, 'r-', linewidth=3, label='Model Prediction', alpha=0.7)

    # Observed data
    ax.errorbar(obs_by_soa['soa_numeric'], obs_by_soa['mean'],
               yerr=obs_by_soa['sem'], fmt='o', markersize=10,
               color='darkblue', capsize=5, linewidth=2,
               label='Observed Data')

    ax.set_xlabel('SOA (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('T2 Reaction Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('PRP Bottleneck Model Fit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add parameter annotations
    param_text = f'T1 processing: {mu_t1:.3f}s\nBottleneck delay: {mu_bottle:.3f}s\nBase RT: {mu_base:.3f}s'
    ax.text(0.70, 0.50, param_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_dir = Path(output_dir)
    plot_file = output_dir / 'prp_model_fit.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved model fit plot to {plot_file}")
    plt.close()

def extract_participant_parameters(trace, df, output_dir='results/analysis_outputs'):
    """Extract participant-level bottleneck parameters."""
    print("\nExtracting participant-level parameters...")

    # Get posterior means for each participant
    posterior = trace.posterior

    t1_proc = posterior['t1_processing'].values.mean(axis=(0, 1))
    bottleneck = posterior['bottleneck_delay'].values.mean(axis=(0, 1))
    base_rt = posterior['base_rt'].values.mean(axis=(0, 1))

    # Get participant IDs
    participant_mapping = pd.read_csv('analysis/prp_participant_mapping.csv')

    # Create dataframe
    params_df = pd.DataFrame({
        'participant_id': participant_mapping['participant_id'],
        'prp_t1_processing': t1_proc,
        'prp_bottleneck_delay': bottleneck,
        'prp_base_rt': base_rt,
        'prp_total_interference': t1_proc + bottleneck  # Total time T2 must wait at SOA=0
    })

    # Add population means for context
    params_df['pop_bottleneck_mean'] = posterior['mu_bottleneck_delay'].values.mean()
    params_df['pop_bottleneck_sd'] = posterior['mu_bottleneck_delay'].values.std()

    # Save
    output_dir = Path(output_dir)
    output_file = output_dir / 'prp_bottleneck_parameters.csv'
    params_df.to_csv(output_file, index=False)
    print(f"  Saved {len(params_df)} participant parameters to {output_file}")

    return params_df

def main():
    """Main execution pipeline."""

    # Load data
    df = load_preprocessed_data()

    # Build model
    model = build_hierarchical_bottleneck_model(df)

    # Fit model
    trace = fit_bottleneck_model(model, draws=1500, tune=1500, chains=4, target_accept=0.95)

    # Diagnostics
    pop_summary, full_summary = diagnose_convergence(trace)

    # Save trace
    trace_file = Path('results/analysis_outputs/prp_bottleneck_trace.nc')
    trace.to_netcdf(trace_file)
    print(f"\n[OK] Saved trace to {trace_file}")

    # Save summaries
    pop_summary_file = Path('results/analysis_outputs/prp_bottleneck_summary_pop.csv')
    pop_summary.to_csv(pop_summary_file)

    full_summary_file = Path('results/analysis_outputs/prp_bottleneck_summary_full.csv')
    full_summary.to_csv(full_summary_file)
    print(f"[OK] Saved summaries")

    # Plots
    plot_posterior_distributions(trace)
    plot_model_fit(trace, df)

    # Extract parameters
    params_df = extract_participant_parameters(trace, df)

    # === Report Key Findings ===
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    posterior = trace.posterior

    mu_t1 = posterior['mu_t1_processing'].values.flatten()
    mu_bottle = posterior['mu_bottleneck_delay'].values.flatten()
    mu_base = posterior['mu_base_rt'].values.flatten()

    print(f"\nPopulation-level parameters (posterior means):")
    print(f"  T1 Processing Time: {mu_t1.mean():.3f}s (95% HDI: [{np.percentile(mu_t1, 2.5):.3f}, {np.percentile(mu_t1, 97.5):.3f}])")
    print(f"  Bottleneck Delay:   {mu_bottle.mean():.3f}s (95% HDI: [{np.percentile(mu_bottle, 2.5):.3f}, {np.percentile(mu_bottle, 97.5):.3f}])")
    print(f"  Base RT:            {mu_base.mean():.3f}s (95% HDI: [{np.percentile(mu_base, 2.5):.3f}, {np.percentile(mu_base, 97.5):.3f}])")

    total_interference = mu_t1 + mu_bottle
    print(f"\n  Total Interference (at SOA=0): {total_interference.mean():.3f}s ({total_interference.mean()*1000:.1f}ms)")
    print(f"    This is the maximum delay T2 experiences due to the bottleneck.")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return trace, params_df

if __name__ == '__main__':
    trace, params = main()
