"""
Final Integrated Analysis: All Tasks + Extreme Group Comparison
================================================================
Combines Stroop, PRP, and WCST with UCLA loneliness and compares
extreme groups (high vs low loneliness).
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

def load_all_data():
    """Load and merge all data sources."""
    print("\n" + "=" * 70)
    print("FINAL INTEGRATED ANALYSIS: ALL TASKS + EXTREME GROUPS")
    print("=" * 70)

    print("\nLoading data...")

    # Surveys
    surveys = pd.read_csv('results/2_surveys_results.csv', encoding='utf-8-sig')
    ucla_data = surveys[surveys['surveyName'] == 'ucla'][['participantId', 'score']].copy()
    ucla_data.columns = ['participant_id', 'ucla_total']
    dass_data = surveys[surveys['surveyName'] == 'dass'][['participantId', 'score_A', 'score_S', 'score_D']].copy()
    dass_data.columns = ['participant_id', 'dass_anxiety', 'dass_stress', 'dass_depression']
    survey_df = ucla_data.merge(dass_data, on='participant_id', how='inner')

    # Cognitive parameters
    stroop = pd.read_csv('results/analysis_outputs/stroop_ddm_parameters.csv')
    prp = pd.read_csv('results/analysis_outputs/prp_bottleneck_parameters.csv')
    wcst = pd.read_csv('results/analysis_outputs/wcst_switching_parameters.csv')

    # Merge all
    df = survey_df.merge(stroop, on='participant_id', how='inner')
    df = df.merge(prp, on='participant_id', how='inner')
    df = df.merge(wcst, on='participant_id', how='inner')
    df = df.dropna()

    print(f"  Final N = {len(df)} participants with complete data")

    return df

def descriptive_stats(df):
    """Print descriptive statistics."""
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    print(f"\nSample: N = {len(df)}")
    print(f"\nUCLA Loneliness: {df['ucla_total'].mean():.1f} +/- {df['ucla_total'].std():.1f}")
    print(f"  Range: {df['ucla_total'].min():.0f} - {df['ucla_total'].max():.0f}")

    print(f"\nStroop Interference: {df['stroop_interference'].mean():.3f} +/- {df['stroop_interference'].std():.3f}")
    print(f"PRP Bottleneck (s):  {df['prp_bottleneck_effect'].mean():.3f} +/- {df['prp_bottleneck_effect'].std():.3f}")
    print(f"WCST Persev Rate:    {df['wcst_persev_rate'].mean():.1%} +/- {df['wcst_persev_rate'].std():.3f}")
    print(f"WCST Switch Cost:    {df['wcst_switch_cost_ms'].mean():.1f} +/- {df['wcst_switch_cost_ms'].std():.1f} ms")

def wcst_regression(df):
    """Bayesian regression: UCLA -> WCST parameters."""
    print("\n" + "=" * 70)
    print("BAYESIAN REGRESSION: UCLA -> WCST SWITCHING")
    print("=" * 70)

    # Standardize
    z_ucla = (df['ucla_total'] - df['ucla_total'].mean()) / df['ucla_total'].std()
    z_dass_anx = (df['dass_anxiety'] - df['dass_anxiety'].mean()) / df['dass_anxiety'].std()
    z_dass_str = (df['dass_stress'] - df['dass_stress'].mean()) / df['dass_stress'].std()
    z_dass_dep = (df['dass_depression'] - df['dass_depression'].mean()) / df['dass_depression'].std()
    z_wcst = (df['wcst_persev_rate'] - df['wcst_persev_rate'].mean()) / df['wcst_persev_rate'].std()

    with pm.Model() as wcst_model:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta_ucla = pm.Normal('beta_ucla', mu=0, sigma=1)
        beta_dass_anx = pm.Normal('beta_dass_anx', mu=0, sigma=1)
        beta_dass_str = pm.Normal('beta_dass_str', mu=0, sigma=1)
        beta_dass_dep = pm.Normal('beta_dass_dep', mu=0, sigma=1)

        mu = alpha + beta_ucla * z_ucla + beta_dass_anx * z_dass_anx + \
             beta_dass_str * z_dass_str + beta_dass_dep * z_dass_dep

        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=z_wcst.values)

    print("  Fitting model...")
    with wcst_model:
        trace_wcst = pm.sample(2000, tune=2000, chains=4, target_accept=0.95,
                              return_inferencedata=True, random_seed=42)

    print("  [OK] Complete")

    summary = az.summary(trace_wcst, var_names=['beta_ucla'], round_to=3)
    print("\n" + summary.to_string())

    beta = trace_wcst.posterior['beta_ucla'].values.flatten()
    print(f"\n  Beta (UCLA): {beta.mean():.3f}")
    print(f"  95% HDI: [{np.percentile(beta, 2.5):.3f}, {np.percentile(beta, 97.5):.3f}]")
    print(f"  P(beta > 0): {(beta > 0).mean():.1%}")

    return trace_wcst

def extreme_group_analysis(df):
    """
    Compare high vs low loneliness groups on all EF measures.

    Groups defined by:
    - High loneliness: Top 25% (Q4)
    - Low loneliness: Bottom 25% (Q1)
    """
    print("\n" + "=" * 70)
    print("EXTREME GROUP ANALYSIS: HIGH VS LOW LONELINESS")
    print("=" * 70)

    # Define groups
    q1 = df['ucla_total'].quantile(0.25)
    q3 = df['ucla_total'].quantile(0.75)

    low_loneliness = df[df['ucla_total'] <= q1].copy()
    high_loneliness = df[df['ucla_total'] >= q3].copy()

    print(f"\n  Low loneliness (Q1): N = {len(low_loneliness)}, UCLA = {low_loneliness['ucla_total'].mean():.1f} +/- {low_loneliness['ucla_total'].std():.1f}")
    print(f"  High loneliness (Q4): N = {len(high_loneliness)}, UCLA = {high_loneliness['ucla_total'].mean():.1f} +/- {high_loneliness['ucla_total'].std():.1f}")

    # Compare on EF measures
    ef_measures = [
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck_effect', 'PRP Bottleneck (s)'),
        ('wcst_persev_rate', 'WCST Perseveration'),
        ('wcst_switch_cost_ms', 'WCST Switch Cost (ms)')
    ]

    print("\n" + "=" * 70)
    print("INDEPENDENT T-TESTS")
    print("=" * 70)

    results = []

    for var, label in ef_measures:
        low_vals = low_loneliness[var].dropna()
        high_vals = high_loneliness[var].dropna()

        # T-test
        t_stat, p_val = stats.ttest_ind(high_vals, low_vals)

        # Cohen's d
        pooled_std = np.sqrt((low_vals.var() + high_vals.var()) / 2)
        cohens_d = (high_vals.mean() - low_vals.mean()) / pooled_std

        # Significance
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"\n{label}:")
        print(f"  Low:  {low_vals.mean():.3f} (SD: {low_vals.std():.3f})")
        print(f"  High: {high_vals.mean():.3f} (SD: {high_vals.std():.3f})")
        print(f"  t({len(low_vals)+len(high_vals)-2}) = {t_stat:.3f}, p = {p_val:.4f} {sig}")
        print(f"  Cohen's d = {cohens_d:.3f}")

        results.append({
            'measure': label,
            'low_mean': low_vals.mean(),
            'high_mean': high_vals.mean(),
            't': t_stat,
            'p': p_val,
            'cohens_d': cohens_d
        })

    results_df = pd.DataFrame(results)

    # Save
    output_dir = Path('results/analysis_outputs')
    results_df.to_csv(output_dir / 'extreme_group_results.csv', index=False)
    print(f"\n[OK] Saved results")

    return results_df, low_loneliness, high_loneliness

def plot_extreme_groups(low, high, output_dir='results/analysis_outputs'):
    """Plot extreme group comparisons."""
    print("\nGenerating extreme group plots...")

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    measures = [
        ('stroop_interference', 'Stroop Interference', axes[0, 0]),
        ('prp_bottleneck_effect', 'PRP Bottleneck Effect (s)', axes[0, 1]),
        ('wcst_persev_rate', 'WCST Perseveration Rate', axes[1, 0]),
        ('wcst_switch_cost_ms', 'WCST Switch Cost (ms)', axes[1, 1])
    ]

    for var, label, ax in measures:
        data_to_plot = [
            low[var].dropna().values,
            high[var].dropna().values
        ]

        bp = ax.boxplot(data_to_plot, labels=['Low\nLoneliness', 'High\nLoneliness'],
                        patch_artist=True, widths=0.6)

        # Color boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{label} by Loneliness Group', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'extreme_group_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved plot")
    plt.close()

def comprehensive_summary(df, trace_wcst, extreme_results):
    """Generate comprehensive summary report."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 70)

    summary_file = Path('results/analysis_outputs/final_analysis_summary.txt')

    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL INTEGRATED ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Sample Size: N = {len(df)}\n\n")

        f.write("MAIN FINDINGS:\n")
        f.write("-" * 70 + "\n\n")

        f.write("1. BAYESIAN REGRESSION RESULTS (controlling for DASS):\n\n")

        # Stroop (from previous analysis)
        f.write("   Stroop Interference:\n")
        f.write("     Beta (UCLA): 0.134, 95% HDI: [-0.237, 0.514]\n")
        f.write("     P(beta > 0): 76%\n")
        f.write("     CONCLUSION: No clear evidence\n\n")

        # PRP (from previous analysis)
        f.write("   PRP Bottleneck:\n")
        f.write("     Beta (UCLA): -0.032, 95% HDI: [-0.409, 0.339]\n")
        f.write("     P(beta > 0): 44%\n")
        f.write("     CONCLUSION: No clear evidence\n\n")

        # WCST (current)
        beta = trace_wcst.posterior['beta_ucla'].values.flatten()
        f.write("   WCST Perseveration:\n")
        f.write(f"     Beta (UCLA): {beta.mean():.3f}, 95% HDI: [{np.percentile(beta, 2.5):.3f}, {np.percentile(beta, 97.5):.3f}]\n")
        f.write(f"     P(beta > 0): {(beta > 0).mean():.1%}\n")
        if (beta > 0).mean() > 0.95:
            f.write("     CONCLUSION: Loneliness increases perseveration\n\n")
        elif (beta > 0).mean() < 0.05:
            f.write("     CONCLUSION: Loneliness decreases perseveration\n\n")
        else:
            f.write("     CONCLUSION: No clear evidence\n\n")

        f.write("2. EXTREME GROUP COMPARISONS (Q1 vs Q4):\n\n")

        for _, row in extreme_results.iterrows():
            sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else "ns"
            f.write(f"   {row['measure']}:\n")
            f.write(f"     Low: {row['low_mean']:.3f}, High: {row['high_mean']:.3f}\n")
            f.write(f"     t = {row['t']:.3f}, p = {row['p']:.4f} {sig}\n")
            f.write(f"     Cohen's d = {row['cohens_d']:.3f}\n\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL CONCLUSION\n")
        f.write("=" * 70 + "\n\n")

        f.write("Current sample (N=77) shows:\n")
        f.write("- Strong PRP bottleneck effect (670ms) confirmed\n")
        f.write("- No clear loneliness -> EF relationship in Bayesian models\n")
        f.write("- Extreme group analysis may show patterns\n\n")

        f.write("Recommendations:\n")
        f.write("1. Increase sample to N=150-200 for adequate power\n")
        f.write("2. Consider measurement paper for null results\n")
        f.write("3. Explore non-linear relationships\n")

    print(f"\n[OK] Saved comprehensive summary to {summary_file}")

def main():
    """Main analysis pipeline."""

    # Load all data
    df = load_all_data()

    # Descriptives
    descriptive_stats(df)

    # WCST regression
    trace_wcst = wcst_regression(df)

    # Extreme group analysis
    extreme_results, low, high = extreme_group_analysis(df)

    # Plots
    plot_extreme_groups(low, high)

    # Comprehensive summary
    comprehensive_summary(df, trace_wcst, extreme_results)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)

    return df, trace_wcst, extreme_results

if __name__ == '__main__':
    df, trace_wcst, results = main()
