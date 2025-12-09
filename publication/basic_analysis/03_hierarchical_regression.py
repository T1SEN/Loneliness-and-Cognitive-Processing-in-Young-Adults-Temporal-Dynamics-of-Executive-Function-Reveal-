"""
Hierarchical Multiple Regression Analysis
==========================================

IRB 계획서 §7⑥ "위계적 다중 회귀 분석"
DASS-21 통제 후 UCLA 외로움의 고유 효과 검증

Model Structure (4 Steps):
    Model 0: outcome ~ age + gender                    (기본 공변량)
    Model 1: outcome ~ age + gender + DASS(3)          (정서 통제)
    Model 2: outcome ~ age + gender + DASS(3) + UCLA   (외로움 주효과)
    Model 3: outcome ~ age + gender + DASS(3) + UCLA*gender (상호작용)

Tier-1 Outcomes:
    - pe_rate: WCST Perseverative Error Rate
    - wcst_accuracy: WCST Accuracy
    - stroop_interference: Stroop Interference Effect
    - prp_bottleneck: PRP Bottleneck Effect

Output:
    results/publication/basic_analysis/hierarchical_results.csv
    results/publication/basic_analysis/model_comparison.csv
    results/publication/basic_analysis/hierarchical_summary.txt

Usage:
    python -m publication.basic_analysis.03_hierarchical_regression
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from typing import Any, Optional

from publication.basic_analysis.utils import (
    get_analysis_data,
    OUTPUT_DIR,
    TIER1_OUTCOMES,
    STANDARDIZED_PREDICTORS,
    print_section_header,
    format_pvalue,
    format_coefficient,
)


# =============================================================================
# HIERARCHICAL MODEL FORMULAS
# =============================================================================

HIERARCHICAL_FORMULAS = {
    "model0": "{outcome} ~ z_age + C(gender_male)",
    "model1": "{outcome} ~ z_age + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress",
    "model2": "{outcome} ~ z_age + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_ucla_score",
    "model3": "{outcome} ~ z_age + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_ucla_score * C(gender_male)",
}


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def find_interaction_term(param_names: pd.Index, term1: str, term2: str) -> Optional[str]:
    """Find interaction term name in model parameters."""
    for name in param_names:
        if term1 in name.lower() and term2 in name.lower() and ':' in name:
            return name
    return None


def run_hierarchical_regression(
    data: pd.DataFrame,
    outcome: str,
    label: str,
    robust: str = "HC3",
    min_n: int = 30
) -> Optional[dict[str, Any]]:
    """
    Run hierarchical regression with 4 models.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with standardized predictors
    outcome : str
        Outcome variable column name
    label : str
        Human-readable label for the outcome
    robust : str
        Robust SE type (default: HC3)
    min_n : int
        Minimum sample size required

    Returns
    -------
    dict or None
        Results dictionary or None if insufficient data
    """
    # Prepare data
    required_cols = [outcome] + STANDARDIZED_PREDICTORS + ['gender_male']
    df = data.dropna(subset=[c for c in required_cols if c in data.columns]).copy()

    if len(df) < min_n:
        print(f"  [SKIP] {label}: N={len(df)} < {min_n}")
        return None

    # Fit models
    models = {}
    for model_name, formula_template in HIERARCHICAL_FORMULAS.items():
        formula = formula_template.format(outcome=outcome)
        try:
            models[model_name] = smf.ols(formula, data=df).fit(cov_type=robust)
        except Exception as e:
            print(f"  [ERROR] {label} {model_name}: {e}")
            return None

    # Model comparisons (F-tests)
    try:
        anova_1v0 = anova_lm(models['model0'], models['model1'])
        anova_2v1 = anova_lm(models['model1'], models['model2'])
        anova_3v2 = anova_lm(models['model2'], models['model3'])
    except Exception as e:
        print(f"  [ERROR] {label} ANOVA: {e}")
        return None

    # HC3 Wald test for UCLA effect
    try:
        ucla_wald = models['model2'].wald_test('z_ucla_score = 0', use_f=True)
        p_ucla_wald = float(ucla_wald.pvalue)
    except Exception:
        p_ucla_wald = models['model2'].pvalues.get('z_ucla_score', np.nan)

    # HC3 Wald test for interaction
    int_term = find_interaction_term(models['model3'].params.index, 'ucla', 'gender')
    if int_term:
        try:
            int_wald = models['model3'].wald_test(f'{int_term} = 0', use_f=True)
            p_interaction_wald = float(int_wald.pvalue)
        except Exception:
            p_interaction_wald = models['model3'].pvalues.get(int_term, np.nan)
    else:
        p_interaction_wald = np.nan

    # Build results
    results = {
        'outcome': label,
        'outcome_column': outcome,
        'n': len(df),

        # R² values
        'model0_r2': models['model0'].rsquared,
        'model1_r2': models['model1'].rsquared,
        'model2_r2': models['model2'].rsquared,
        'model3_r2': models['model3'].rsquared,

        # Adjusted R²
        'model0_adj_r2': models['model0'].rsquared_adj,
        'model1_adj_r2': models['model1'].rsquared_adj,
        'model2_adj_r2': models['model2'].rsquared_adj,
        'model3_adj_r2': models['model3'].rsquared_adj,

        # AIC
        'model0_aic': models['model0'].aic,
        'model1_aic': models['model1'].aic,
        'model2_aic': models['model2'].aic,
        'model3_aic': models['model3'].aic,

        # ΔR² and F-tests
        'delta_r2_dass': models['model1'].rsquared - models['model0'].rsquared,
        'F_dass': anova_1v0['F'][1] if 'F' in anova_1v0.columns else np.nan,
        'p_dass': anova_1v0['Pr(>F)'][1],

        'delta_r2_ucla': models['model2'].rsquared - models['model1'].rsquared,
        'F_ucla': anova_2v1['F'][1] if 'F' in anova_2v1.columns else np.nan,
        'p_ucla_ols': anova_2v1['Pr(>F)'][1],
        'p_ucla_wald': p_ucla_wald,  # HC3 robust

        'delta_r2_interaction': models['model3'].rsquared - models['model2'].rsquared,
        'F_interaction': anova_3v2['F'][1] if 'F' in anova_3v2.columns else np.nan,
        'p_interaction_ols': anova_3v2['Pr(>F)'][1],
        'p_interaction_wald': p_interaction_wald,  # HC3 robust

        # UCLA main effect (from Model 2)
        'ucla_beta': models['model2'].params.get('z_ucla_score', np.nan),
        'ucla_se': models['model2'].bse.get('z_ucla_score', np.nan),
        'ucla_t': models['model2'].tvalues.get('z_ucla_score', np.nan),
        'ucla_p': models['model2'].pvalues.get('z_ucla_score', np.nan),
    }

    # Interaction term (from Model 3)
    if int_term and int_term in models['model3'].params:
        results['interaction_term'] = int_term
        results['interaction_beta'] = models['model3'].params[int_term]
        results['interaction_se'] = models['model3'].bse[int_term]
        results['interaction_t'] = models['model3'].tvalues[int_term]
        results['interaction_p'] = models['model3'].pvalues[int_term]
    else:
        results['interaction_term'] = None
        results['interaction_beta'] = np.nan
        results['interaction_se'] = np.nan
        results['interaction_t'] = np.nan
        results['interaction_p'] = np.nan

    # Gender-stratified UCLA effects
    for gender_val, gender_label in [(0, 'female'), (1, 'male')]:
        subset = df[df['gender_male'] == gender_val]
        if len(subset) >= 15:
            formula = f"{outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
            try:
                m = smf.ols(formula, data=subset).fit(cov_type=robust)
                results[f'{gender_label}_n'] = len(subset)
                results[f'{gender_label}_ucla_beta'] = m.params.get('z_ucla_score', np.nan)
                results[f'{gender_label}_ucla_se'] = m.bse.get('z_ucla_score', np.nan)
                results[f'{gender_label}_ucla_p'] = m.pvalues.get('z_ucla_score', np.nan)
            except Exception:
                results[f'{gender_label}_n'] = len(subset)
                results[f'{gender_label}_ucla_beta'] = np.nan
                results[f'{gender_label}_ucla_se'] = np.nan
                results[f'{gender_label}_ucla_p'] = np.nan
        else:
            results[f'{gender_label}_n'] = len(subset)
            results[f'{gender_label}_ucla_beta'] = np.nan
            results[f'{gender_label}_ucla_se'] = np.nan
            results[f'{gender_label}_ucla_p'] = np.nan

    # Diagnostics
    # VIF
    try:
        pred_cols = ['z_ucla_score', 'z_dass_depression', 'z_dass_anxiety', 'z_dass_stress', 'z_age']
        X = df[[c for c in pred_cols if c in df.columns]].dropna()
        if len(X) > 0 and X.shape[1] > 1:
            vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            results['vif_max'] = max(vif_vals)
            results['vif_mean'] = np.mean(vif_vals)
        else:
            results['vif_max'] = np.nan
            results['vif_mean'] = np.nan
    except Exception:
        results['vif_max'] = np.nan
        results['vif_mean'] = np.nan

    # Breusch-Pagan heteroscedasticity test
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(models['model3'].resid, models['model3'].model.exog)
        results['bp_stat'] = bp_stat
        results['bp_p'] = bp_p
    except Exception:
        results['bp_stat'] = np.nan
        results['bp_p'] = np.nan

    # Shapiro-Wilk normality test (sample if large)
    try:
        resid = models['model3'].resid
        if len(resid) > 5000:
            resid = np.random.default_rng(42).choice(resid, size=5000, replace=False)
        _, sw_p = stats.shapiro(resid)
        results['shapiro_p'] = sw_p
    except Exception:
        results['shapiro_p'] = np.nan

    return results


def generate_summary_report(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Generate human-readable summary report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from hierarchical regression
    output_path : str
        Path to save the report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("HIERARCHICAL REGRESSION ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Model Structure:")
    lines.append("  Model 0: age + gender")
    lines.append("  Model 1: Model 0 + DASS-21 (depression, anxiety, stress)")
    lines.append("  Model 2: Model 1 + UCLA loneliness")
    lines.append("  Model 3: Model 2 + UCLA × Gender interaction")
    lines.append("")
    lines.append("-" * 70)

    for _, row in results_df.iterrows():
        lines.append("")
        lines.append(f"Outcome: {row['outcome']} (N = {row['n']})")
        lines.append("-" * 50)

        # Model comparison
        lines.append("")
        lines.append("Model Comparison:")
        lines.append(f"  Step 1 (DASS):        ΔR² = {row['delta_r2_dass']:.4f}, p = {format_pvalue(row['p_dass'])}")
        lines.append(f"  Step 2 (UCLA):        ΔR² = {row['delta_r2_ucla']:.4f}, p = {format_pvalue(row['p_ucla_wald'])} (HC3 Wald)")
        lines.append(f"  Step 3 (Interaction): ΔR² = {row['delta_r2_interaction']:.4f}, p = {format_pvalue(row['p_interaction_wald'])} (HC3 Wald)")

        # UCLA effect
        lines.append("")
        lines.append("UCLA Main Effect (Model 2):")
        lines.append(f"  β = {row['ucla_beta']:.4f}, SE = {row['ucla_se']:.4f}, t = {row['ucla_t']:.2f}, p = {format_pvalue(row['ucla_p'])}")

        # Interaction
        if pd.notna(row.get('interaction_beta')):
            lines.append("")
            lines.append("UCLA × Gender Interaction (Model 3):")
            lines.append(f"  β = {row['interaction_beta']:.4f}, SE = {row['interaction_se']:.4f}, p = {format_pvalue(row['interaction_p'])}")

        # Gender-stratified
        lines.append("")
        lines.append("Gender-Stratified UCLA Effects:")
        if pd.notna(row.get('female_ucla_beta')):
            lines.append(f"  Female (n={row.get('female_n', 'NA')}): β = {row['female_ucla_beta']:.4f}, p = {format_pvalue(row.get('female_ucla_p'))}")
        if pd.notna(row.get('male_ucla_beta')):
            lines.append(f"  Male (n={row.get('male_n', 'NA')}):   β = {row['male_ucla_beta']:.4f}, p = {format_pvalue(row.get('male_ucla_p'))}")

        # Diagnostics
        lines.append("")
        lines.append("Diagnostics:")
        lines.append(f"  VIF max: {row.get('vif_max', np.nan):.2f}")
        lines.append(f"  Breusch-Pagan p: {format_pvalue(row.get('bp_p'))}")
        lines.append(f"  Shapiro-Wilk p: {format_pvalue(row.get('shapiro_p'))}")

        lines.append("")

    lines.append("=" * 70)
    lines.append("Note. HC3 robust standard errors used for all tests.")
    lines.append("=" * 70)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def run(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run hierarchical regression analysis for all Tier-1 outcomes.

    Parameters
    ----------
    verbose : bool
        Print results to console

    Returns
    -------
    dict
        Dictionary with results and model comparison DataFrames
    """
    if verbose:
        print_section_header("HIERARCHICAL MULTIPLE REGRESSION ANALYSIS")

    # Load data
    if verbose:
        print("\n  Loading data...")
    df = get_analysis_data(use_cache=True)

    if verbose:
        print(f"  Total participants: N = {len(df)}")

    # Run hierarchical regression for each outcome
    if verbose:
        print("\n  Running hierarchical regressions...")
        print("  " + "-" * 50)

    all_results = []
    for outcome_col, outcome_label in TIER1_OUTCOMES:
        if verbose:
            print(f"\n  Analyzing: {outcome_label}")

        result = run_hierarchical_regression(df, outcome_col, outcome_label)
        if result:
            all_results.append(result)

    if not all_results:
        print("  [WARNING] No results generated")
        return {'results': pd.DataFrame(), 'model_comparison': pd.DataFrame()}

    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "hierarchical_results.csv", index=False, encoding='utf-8-sig')

    # Create model comparison table
    comparison_cols = [
        'outcome', 'n',
        'model0_r2', 'model1_r2', 'model2_r2', 'model3_r2',
        'delta_r2_dass', 'p_dass',
        'delta_r2_ucla', 'p_ucla_wald',
        'delta_r2_interaction', 'p_interaction_wald'
    ]
    comparison_df = results_df[[c for c in comparison_cols if c in results_df.columns]]
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding='utf-8-sig')

    # Generate summary report
    generate_summary_report(results_df, OUTPUT_DIR / "hierarchical_summary.txt")

    # Print results
    if verbose:
        print("\n  " + "=" * 65)
        print("  MODEL COMPARISON SUMMARY")
        print("  " + "=" * 65)
        print(f"\n  {'Outcome':<30} {'ΔR²(DASS)':>10} {'ΔR²(UCLA)':>10} {'ΔR²(Int)':>10}")
        print("  " + "-" * 65)

        for _, row in results_df.iterrows():
            p_ucla = row['p_ucla_wald']
            p_int = row['p_interaction_wald']
            ucla_sig = "*" if p_ucla < 0.05 else ""
            int_sig = "*" if p_int < 0.05 else ""

            print(f"  {row['outcome']:<30} "
                  f"{row['delta_r2_dass']:>10.4f} "
                  f"{row['delta_r2_ucla']:>8.4f}{ucla_sig:<2} "
                  f"{row['delta_r2_interaction']:>8.4f}{int_sig:<2}")

        print("  " + "-" * 65)
        print("  Note. *p < .05 (HC3 Wald test)")

        # Significant effects
        sig_ucla = results_df[results_df['p_ucla_wald'] < 0.05]
        sig_int = results_df[results_df['p_interaction_wald'] < 0.05]

        print("\n  Significant Effects:")
        if len(sig_ucla) > 0:
            print(f"    UCLA main effects: {', '.join(sig_ucla['outcome'].tolist())}")
        else:
            print("    UCLA main effects: None")

        if len(sig_int) > 0:
            print(f"    UCLA × Gender interactions: {', '.join(sig_int['outcome'].tolist())}")
        else:
            print("    UCLA × Gender interactions: None")

        print(f"\n  Output files:")
        print(f"    - {OUTPUT_DIR / 'hierarchical_results.csv'}")
        print(f"    - {OUTPUT_DIR / 'model_comparison.csv'}")
        print(f"    - {OUTPUT_DIR / 'hierarchical_summary.txt'}")

    return {
        'results': results_df,
        'model_comparison': comparison_df,
    }


if __name__ == "__main__":
    results = run(verbose=True)
    print("\n" + "=" * 70)
    print("HIERARCHICAL REGRESSION ANALYSIS COMPLETE")
    print("=" * 70)
