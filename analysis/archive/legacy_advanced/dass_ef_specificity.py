"""
DASS × EF Specificity Analysis
==============================
Tests whether specific DASS-21 subscales (Depression, Anxiety, Stress)
have selective effects on different EF components.

Hypotheses:
- Anxiety → Attentional processes (Stroop interference, PRP variability)
- Depression → Processing speed and set-shifting (WCST)
- Stress → General performance decrement

This analysis goes beyond using DASS as a covariate to examine
whether each subscale has unique predictive validity for specific EF.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "dass_ef_specificity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dass_subscale_effects(master_df):
    """Test each DASS subscale's unique effect on each EF outcome."""
    ef_outcomes = {
        'pe_rate': 'WCST PE Rate',
        'wcst_accuracy': 'WCST Accuracy',
        'stroop_interference': 'Stroop Interference',
        'prp_bottleneck': 'PRP Bottleneck'
    }

    dass_subscales = ['dass_depression', 'dass_anxiety', 'dass_stress']

    results = []

    for outcome, outcome_label in ef_outcomes.items():
        if outcome not in master_df.columns:
            continue

        # Prepare data
        cols = [outcome] + dass_subscales + ['age', 'gender_male', 'ucla_score']
        analysis_df = master_df[cols].dropna()

        if len(analysis_df) < 30:
            continue

        # Standardize predictors
        for col in dass_subscales + ['age', 'ucla_score']:
            analysis_df[f'z_{col}'] = (analysis_df[col] - analysis_df[col].mean()) / analysis_df[col].std()

        # Model 1: All DASS subscales together (controlling for each other)
        formula_full = f"{outcome} ~ z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        model_full = smf.ols(formula_full, data=analysis_df).fit()

        # Model 2: Add UCLA to see incremental validity
        formula_with_ucla = f"{outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        model_with_ucla = smf.ols(formula_with_ucla, data=analysis_df).fit()

        # Extract results for each subscale
        for subscale in dass_subscales:
            z_subscale = f'z_{subscale}'
            subscale_short = subscale.replace('dass_', '')

            results.append({
                'outcome': outcome,
                'outcome_label': outcome_label,
                'predictor': subscale_short,
                'n': len(analysis_df),
                # Without UCLA
                'coef_no_ucla': model_full.params.get(z_subscale, np.nan),
                'se_no_ucla': model_full.bse.get(z_subscale, np.nan),
                'p_no_ucla': model_full.pvalues.get(z_subscale, np.nan),
                # With UCLA
                'coef_with_ucla': model_with_ucla.params.get(z_subscale, np.nan),
                'se_with_ucla': model_with_ucla.bse.get(z_subscale, np.nan),
                'p_with_ucla': model_with_ucla.pvalues.get(z_subscale, np.nan),
                # Model fit
                'model_r2_no_ucla': model_full.rsquared,
                'model_r2_with_ucla': model_with_ucla.rsquared,
                'r2_increase_ucla': model_with_ucla.rsquared - model_full.rsquared
            })

    return pd.DataFrame(results)


def test_specificity_hypothesis(results_df):
    """Test pre-specified specificity hypotheses."""
    hypotheses = [
        ('stroop_interference', 'anxiety', 'Anxiety → Stroop (attentional control)'),
        ('pe_rate', 'depression', 'Depression → WCST PE (set-shifting)'),
        ('prp_bottleneck', 'stress', 'Stress → PRP (dual-task capacity)'),
    ]

    hypothesis_results = []

    for outcome, predictor, label in hypotheses:
        row = results_df[(results_df['outcome'] == outcome) &
                        (results_df['predictor'] == predictor)]

        if len(row) == 0:
            continue

        row = row.iloc[0]
        hypothesis_results.append({
            'hypothesis': label,
            'outcome': outcome,
            'predictor': predictor,
            'coef': row['coef_with_ucla'],
            'se': row['se_with_ucla'],
            'p': row['p_with_ucla'],
            'supported': row['p_with_ucla'] < 0.05
        })

    return pd.DataFrame(hypothesis_results)


def compare_subscale_effects(results_df):
    """Compare effect sizes across subscales for each outcome."""
    comparisons = []

    for outcome in results_df['outcome'].unique():
        outcome_data = results_df[results_df['outcome'] == outcome]

        # Get coefficients and SEs
        effects = {}
        for _, row in outcome_data.iterrows():
            effects[row['predictor']] = {
                'coef': row['coef_with_ucla'],
                'se': row['se_with_ucla'],
                'p': row['p_with_ucla']
            }

        # Find strongest predictor
        strongest = max(effects.keys(), key=lambda k: abs(effects[k]['coef']))

        # Z-test comparing each pair
        subscales = list(effects.keys())
        for i, s1 in enumerate(subscales):
            for s2 in subscales[i+1:]:
                diff = effects[s1]['coef'] - effects[s2]['coef']
                se_diff = np.sqrt(effects[s1]['se']**2 + effects[s2]['se']**2)
                z = diff / se_diff if se_diff > 0 else 0
                p = 2 * (1 - stats.norm.cdf(abs(z)))

                comparisons.append({
                    'outcome': outcome,
                    'comparison': f"{s1} vs {s2}",
                    'diff': diff,
                    'se_diff': se_diff,
                    'z': z,
                    'p': p,
                    'stronger': s1 if diff > 0 else s2
                })

    return pd.DataFrame(comparisons)


def analyze_dass_interactions(master_df):
    """Test DASS subscale × EF task interactions."""
    # Reshape to long format for interaction analysis
    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available_ef = [c for c in ef_cols if c in master_df.columns]

    if len(available_ef) < 2:
        return None

    # Standardize EF measures
    for col in available_ef:
        master_df[f'z_{col}'] = (master_df[col] - master_df[col].mean()) / master_df[col].std()

    # Create correlation matrix
    dass_cols = ['dass_depression', 'dass_anxiety', 'dass_stress']
    corr_matrix = master_df[dass_cols + available_ef].corr()

    # DASS-EF correlations only
    ef_dass_corr = corr_matrix.loc[dass_cols, available_ef]

    return ef_dass_corr


def main():
    print("=" * 60)
    print("DASS × EF Specificity Analysis")
    print("=" * 60)

    # Load data
    master = load_master_dataset(use_cache=True)
    print(f"Dataset loaded: N={len(master)}")

    # Main analysis
    print("\n[1] DASS Subscale Effects on EF Outcomes")
    print("-" * 40)

    results = analyze_dass_subscale_effects(master)
    results.to_csv(OUTPUT_DIR / "dass_subscale_effects.csv", index=False, encoding='utf-8-sig')

    # Summary table
    print("\nResults Summary (with UCLA control):")
    print("-" * 60)
    print(f"{'Outcome':<20} {'Predictor':<12} {'Coef':>8} {'SE':>8} {'p':>8}")
    print("-" * 60)
    for _, row in results.iterrows():
        sig = "*" if row['p_with_ucla'] < 0.05 else ""
        print(f"{row['outcome_label'][:19]:<20} {row['predictor']:<12} {row['coef_with_ucla']:>8.3f} {row['se_with_ucla']:>8.3f} {row['p_with_ucla']:>7.4f}{sig}")

    # Hypothesis testing
    print("\n[2] Pre-specified Specificity Hypotheses")
    print("-" * 40)

    hypotheses = test_specificity_hypothesis(results)
    if len(hypotheses) > 0:
        for _, row in hypotheses.iterrows():
            status = "SUPPORTED" if row['supported'] else "NOT SUPPORTED"
            print(f"  {row['hypothesis']}")
            print(f"    b={row['coef']:.3f}, p={row['p']:.4f} -> {status}")

        hypotheses.to_csv(OUTPUT_DIR / "hypothesis_tests.csv", index=False, encoding='utf-8-sig')

    # Subscale comparisons
    print("\n[3] Between-Subscale Comparisons")
    print("-" * 40)

    comparisons = compare_subscale_effects(results)
    if len(comparisons) > 0:
        # Apply FDR correction
        comparisons['p_fdr'] = multipletests(comparisons['p'], method='fdr_bh')[1]

        for outcome in comparisons['outcome'].unique():
            print(f"\n  {outcome}:")
            outcome_comp = comparisons[comparisons['outcome'] == outcome]
            for _, row in outcome_comp.iterrows():
                sig = "*" if row['p_fdr'] < 0.05 else ""
                print(f"    {row['comparison']}: z={row['z']:.2f}, p={row['p']:.4f}, p_fdr={row['p_fdr']:.4f}{sig}")

        comparisons.to_csv(OUTPUT_DIR / "subscale_comparisons.csv", index=False, encoding='utf-8-sig')

    # Correlation matrix
    print("\n[4] DASS-EF Correlation Matrix")
    print("-" * 40)

    corr_matrix = analyze_dass_interactions(master)
    if corr_matrix is not None:
        print(corr_matrix.round(3).to_string())
        corr_matrix.to_csv(OUTPUT_DIR / "dass_ef_correlations.csv", encoding='utf-8-sig')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
DASS Subscale Specificity Analysis:
- Tests whether Depression, Anxiety, and Stress uniquely predict different EF

Key Findings:
""")
    # Find significant effects
    sig_effects = results[results['p_with_ucla'] < 0.05]
    if len(sig_effects) > 0:
        for _, row in sig_effects.iterrows():
            print(f"  - {row['predictor'].title()} → {row['outcome_label']}: b={row['coef_with_ucla']:.3f}, p={row['p_with_ucla']:.4f}")
    else:
        print("  - No significant DASS subscale effects on EF (after UCLA control)")

    print(f"""
Interpretation:
- Significant Depression effect on WCST would support cognitive slowing hypothesis
- Significant Anxiety effect on Stroop would support attentional bias hypothesis
- Non-specific effects suggest general distress → general impairment
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
