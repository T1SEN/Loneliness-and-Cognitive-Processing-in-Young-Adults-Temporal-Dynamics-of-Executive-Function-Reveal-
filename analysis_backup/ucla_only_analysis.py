"""
UCLA Loneliness Only Analysis
==============================
Test UCLA effects WITHOUT controlling for DASS first.
Then compare with DASS-controlled models.

This addresses multicollinearity concerns.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from scipy import stats

def load_data():
    """Load merged data."""
    print("\n" + "=" * 70)
    print("UCLA-ONLY ANALYSIS (NO DASS CONTROL)")
    print("=" * 70)

    # Load previous merged data
    surveys = pd.read_csv('results/2_surveys_results.csv', encoding='utf-8-sig')
    ucla_data = surveys[surveys['surveyName'] == 'ucla'][['participantId', 'score']].copy()
    ucla_data.columns = ['participant_id', 'ucla_total']
    dass_data = surveys[surveys['surveyName'] == 'dass'][['participantId', 'score_A', 'score_S', 'score_D']].copy()
    dass_data.columns = ['participant_id', 'dass_anxiety', 'dass_stress', 'dass_depression']
    survey_df = ucla_data.merge(dass_data, on='participant_id', how='inner')

    stroop = pd.read_csv('results/analysis_outputs/stroop_ddm_parameters.csv')
    prp = pd.read_csv('results/analysis_outputs/prp_bottleneck_parameters.csv')
    wcst = pd.read_csv('results/analysis_outputs/wcst_switching_parameters.csv')

    df = survey_df.merge(stroop, on='participant_id', how='inner')
    df = df.merge(prp, on='participant_id', how='inner')
    df = df.merge(wcst, on='participant_id', how='inner')
    df = df.dropna()

    print(f"\n  N = {len(df)}")

    return df

def simple_correlations(df):
    """Simple Pearson correlations."""
    print("\n" + "=" * 70)
    print("SIMPLE CORRELATIONS (Pearson r)")
    print("=" * 70)

    measures = [
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck_effect', 'PRP Bottleneck'),
        ('wcst_persev_rate', 'WCST Perseveration'),
        ('wcst_switch_cost_ms', 'WCST Switch Cost')
    ]

    print(f"\nUCLA Loneliness correlations:")
    for var, label in measures:
        r, p = stats.pearsonr(df['ucla_total'], df[var])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {label:25s}: r = {r:7.3f}, p = {p:.4f} {sig}")

def ucla_only_regressions(df):
    """
    Bayesian regressions with UCLA ONLY (no DASS).
    """
    print("\n" + "=" * 70)
    print("BAYESIAN REGRESSIONS: UCLA ONLY (NO DASS)")
    print("=" * 70)

    # Standardize
    z_ucla = (df['ucla_total'] - df['ucla_total'].mean()) / df['ucla_total'].std()

    results = {}

    # === Stroop ===
    print("\n1. Stroop Interference:")
    z_stroop = (df['stroop_interference'] - df['stroop_interference'].mean()) / df['stroop_interference'].std()

    with pm.Model() as stroop_model:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta_ucla', mu=0, sigma=1)
        mu = alpha + beta * z_ucla.values
        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=z_stroop.values)

    with stroop_model:
        trace_stroop = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                                return_inferencedata=True, random_seed=42,
                                progressbar=False)

    beta_stroop = trace_stroop.posterior['beta_ucla'].values.flatten()
    print(f"   Beta: {beta_stroop.mean():.3f}")
    print(f"   95% HDI: [{np.percentile(beta_stroop, 2.5):.3f}, {np.percentile(beta_stroop, 97.5):.3f}]")
    print(f"   P(beta > 0): {(beta_stroop > 0).mean():.1%}")

    results['stroop'] = {
        'beta_mean': beta_stroop.mean(),
        'hdi_low': np.percentile(beta_stroop, 2.5),
        'hdi_high': np.percentile(beta_stroop, 97.5),
        'prob_positive': (beta_stroop > 0).mean()
    }

    # === PRP ===
    print("\n2. PRP Bottleneck:")
    z_prp = (df['prp_bottleneck_effect'] - df['prp_bottleneck_effect'].mean()) / df['prp_bottleneck_effect'].std()

    with pm.Model() as prp_model:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta_ucla', mu=0, sigma=1)
        mu = alpha + beta * z_ucla.values
        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=z_prp.values)

    with prp_model:
        trace_prp = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                             return_inferencedata=True, random_seed=42,
                             progressbar=False)

    beta_prp = trace_prp.posterior['beta_ucla'].values.flatten()
    print(f"   Beta: {beta_prp.mean():.3f}")
    print(f"   95% HDI: [{np.percentile(beta_prp, 2.5):.3f}, {np.percentile(beta_prp, 97.5):.3f}]")
    print(f"   P(beta > 0): {(beta_prp > 0).mean():.1%}")

    results['prp'] = {
        'beta_mean': beta_prp.mean(),
        'hdi_low': np.percentile(beta_prp, 2.5),
        'hdi_high': np.percentile(beta_prp, 97.5),
        'prob_positive': (beta_prp > 0).mean()
    }

    # === WCST ===
    print("\n3. WCST Perseveration:")
    z_wcst = (df['wcst_persev_rate'] - df['wcst_persev_rate'].mean()) / df['wcst_persev_rate'].std()

    with pm.Model() as wcst_model:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta_ucla', mu=0, sigma=1)
        mu = alpha + beta * z_ucla.values
        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=z_wcst.values)

    with wcst_model:
        trace_wcst = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                              return_inferencedata=True, random_seed=42,
                              progressbar=False)

    beta_wcst = trace_wcst.posterior['beta_ucla'].values.flatten()
    print(f"   Beta: {beta_wcst.mean():.3f}")
    print(f"   95% HDI: [{np.percentile(beta_wcst, 2.5):.3f}, {np.percentile(beta_wcst, 97.5):.3f}]")
    print(f"   P(beta > 0): {(beta_wcst > 0).mean():.1%}")

    results['wcst'] = {
        'beta_mean': beta_wcst.mean(),
        'hdi_low': np.percentile(beta_wcst, 2.5),
        'hdi_high': np.percentile(beta_wcst, 97.5),
        'prob_positive': (beta_wcst > 0).mean()
    }

    return results

def compare_with_dass_models(ucla_results):
    """
    Compare UCLA-only vs UCLA+DASS models.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: UCLA-ONLY vs UCLA+DASS")
    print("=" * 70)

    print("\n                        UCLA-ONLY              UCLA+DASS")
    print("                        Beta    P(>0)          Beta    P(>0)")
    print("-" * 70)

    # Previous DASS-controlled results (from earlier analysis)
    # NOTE: These are hard-coded values from prior runs - ideally should load from file
    dass_results = {
        'stroop': {'beta': 0.134, 'prob': 0.76},
        'prp': {'beta': -0.032, 'prob': 0.44},
        'wcst': {'beta': 0.126, 'prob': 0.75}
    }

    # Print comparison table
    for task in ['stroop', 'prp', 'wcst']:
        ucla_beta = ucla_results[task]['beta_mean']
        ucla_prob = ucla_results[task]['prob_positive']
        dass_beta = dass_results[task]['beta']
        dass_prob = dass_results[task]['prob']
        print(f"  {task.upper():10s}        {ucla_beta:6.3f}  {ucla_prob:5.2%}          {dass_beta:6.3f}  {dass_prob:5.2%}")

def summary_report(results):
    """Generate summary report."""
    print("\n" + "=" * 70)
    print("SUMMARY: UCLA-ONLY EFFECTS")
    print("=" * 70)

    for task, res in results.items():
        print(f"\n{task.upper()}:")
        print(f"  Beta: {res['beta_mean']:.3f}")
        print(f"  95% HDI: [{res['hdi_low']:.3f}, {res['hdi_high']:.3f}]")
        print(f"  P(beta > 0): {res['prob_positive']:.1%}")

        if res['prob_positive'] > 0.95:
            print(f"  ==> STRONG positive effect")
        elif res['prob_positive'] < 0.05:
            print(f"  ==> STRONG negative effect")
        elif res['prob_positive'] > 0.90 or res['prob_positive'] < 0.10:
            print(f"  ==> Moderate evidence")
        else:
            print(f"  ==> No clear evidence")

    # Save results
    output_dir = Path('results/analysis_outputs')
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / 'ucla_only_results.csv')
    print(f"\n[OK] Saved results to ucla_only_results.csv")

def main():
    """Main pipeline."""

    # Load data
    df = load_data()

    # Simple correlations
    simple_correlations(df)

    # UCLA-only regressions
    results = ucla_only_regressions(df)

    # Summary
    summary_report(results)

    # Comparison note
    compare_with_dass_models(results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nKEY INSIGHT:")
    print("If UCLA-only shows effects but UCLA+DASS doesn't,")
    print("this suggests DASS mediates or overlaps with UCLA effects.")
    print("\nIf UCLA-only ALSO shows no effects,")
    print("then there's truly no loneliness-EF relationship.")

    return df, results

if __name__ == '__main__':
    df, results = main()
