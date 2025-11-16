"""
PRP Bottleneck Model - Simplified Participant-Level Analysis
============================================================
Uses pre-computed PRP effects (short SOA - long SOA) for each participant.

This is much faster than trial-level hierarchical modeling and suitable
for initial analysis and power calculations.

For Q1 publication, we can later implement full trial-level DDM if needed.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')

def main():
    """Simplified PRP analysis using participant-level summary statistics."""

    print("\n" + "=" * 70)
    print("PRP BOTTLENECK ANALYSIS - PARTICIPANT-LEVEL MODEL")
    print("=" * 70)

    # Load pre-computed PRP effects
    print("\nLoading PRP effects...")
    prp_effects = pd.read_csv('analysis/prp_effects_summary.csv')
    print(f"  Loaded data for {len(prp_effects)} participants")

    # Statistics
    print(f"\nPRP Effect statistics:")
    print(f"  Mean: {prp_effects['prp_effect'].mean():.3f}s ({prp_effects['prp_effect'].mean()*1000:.1f}ms)")
    print(f"  SD: {prp_effects['prp_effect'].std():.3f}s ({prp_effects['prp_effect'].std()*1000:.1f}ms)")
    print(f"  Min: {prp_effects['prp_effect'].min():.3f}s")
    print(f"  Max: {prp_effects['prp_effect'].max():.3f}s")
    print(f"  Participants with positive effect: {(prp_effects['prp_effect'] > 0).sum()}/{len(prp_effects)}")

    # Simple Bayesian model on PRP effects
    print("\nBuilding Bayesian model on PRP effects...")

    prp_data = prp_effects['prp_effect'].values

    with pm.Model() as prp_model:
        # Population mean PRP effect
        mu_prp = pm.Normal('mu_prp_effect', mu=0.5, sigma=0.3)

        # Population SD
        sigma_prp = pm.HalfNormal('sigma_prp_effect', sigma=0.2)

        # Individual PRP effects
        pm.Normal('prp_effect_obs', mu=mu_prp, sigma=sigma_prp, observed=prp_data)

    print("  Model built")

    # Fit model
    print("\nFitting model (this should take < 1 minute)...")
    with prp_model:
        trace = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                         return_inferencedata=True, random_seed=42)

    print("[OK] Sampling complete!")

    # Diagnostics
    print("\n" + "=" * 60)
    print("Convergence Diagnostics")
    print("=" * 60)

    summary = az.summary(trace, round_to=3)
    print(summary)

    # Check convergence
    max_rhat = summary['r_hat'].max()
    if max_rhat < 1.01:
        print(f"\n[OK] R-hat = {max_rhat:.4f} < 1.01 (good convergence)")
    else:
        print(f"\n[WARNING] R-hat = {max_rhat:.4f} >= 1.01")

    # Save results
    output_dir = Path('results/analysis_outputs')
    output_dir.mkdir(exist_ok=True, parents=True)

    trace.to_netcdf(output_dir / 'prp_simple_trace.nc')
    summary.to_csv(output_dir / 'prp_simple_summary.csv')
    print(f"\n[OK] Saved trace and summary")

    # Plot posteriors
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    az.plot_posterior(trace, var_names=['mu_prp_effect'], ax=axes[0], hdi_prob=0.95)
    axes[0].set_title('Population Mean PRP Effect')
    axes[0].set_xlabel('PRP Effect (seconds)')

    az.plot_posterior(trace, var_names=['sigma_prp_effect'], ax=axes[1], hdi_prob=0.95)
    axes[1].set_title('Population SD PRP Effect')
    axes[1].set_xlabel('SD (seconds)')

    plt.tight_layout()
    plt.savefig(output_dir / 'prp_simple_posteriors.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved posterior plots")
    plt.close()

    # Extract individual parameters for later use
    # In this simple model, we just use the observed values
    params_df = prp_effects[['participant_id', 'prp_effect']].copy()
    params_df.columns = ['participant_id', 'prp_bottleneck_effect']

    # Add population estimates
    mu_prp_post = trace.posterior['mu_prp_effect'].values.flatten()
    params_df['pop_prp_mean'] = mu_prp_post.mean()
    params_df['pop_prp_sd'] = mu_prp_post.std()

    params_df.to_csv(output_dir / 'prp_bottleneck_parameters.csv', index=False)
    print(f"[OK] Saved participant parameters")

    # Report findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\nPopulation-level PRP Effect:")
    print(f"  Posterior mean: {mu_prp_post.mean():.3f}s ({mu_prp_post.mean()*1000:.1f}ms)")
    print(f"  95% HDI: [{np.percentile(mu_prp_post, 2.5):.3f}, {np.percentile(mu_prp_post, 97.5):.3f}]")

    # Test if effect is significantly positive
    prob_positive = (mu_prp_post > 0).mean()
    print(f"  P(PRP effect > 0): {prob_positive:.1%}")

    if prob_positive > 0.99:
        print(f"  ==> STRONG evidence for bottleneck effect")
    elif prob_positive > 0.95:
        print(f"  ==> Moderate evidence for bottleneck effect")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return trace, params_df

if __name__ == '__main__':
    trace, params = main()
