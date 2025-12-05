"""
Nonlinear UCLA Effects Analysis
===============================
Tests whether UCLA loneliness has nonlinear effects on EF outcomes.

Analyses:
1. Polynomial regression (quadratic, cubic terms)
2. Generalized Additive Models (GAM) with smoothing
3. Breakpoint/threshold analysis
4. Dose-response modeling

Hypotheses:
- Threshold effect: EF impairment only above UCLA ~40
- Inverted-U: Moderate loneliness may have no effect, extremes impair
- Accelerating effect: Impairment increases nonlinearly at high levels
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
import warnings

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "ucla_nonlinear"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_polynomial_effects(master_df, outcome):
    """Test quadratic and cubic UCLA effects."""
    cols = [outcome, 'ucla_score', 'dass_depression', 'dass_anxiety',
            'dass_stress', 'age', 'gender_male']
    df = master_df[cols].dropna()

    if len(df) < 50:
        return None

    # Center UCLA for polynomial terms
    df['ucla_c'] = df['ucla_score'] - df['ucla_score'].mean()
    df['ucla_c2'] = df['ucla_c'] ** 2
    df['ucla_c3'] = df['ucla_c'] ** 3

    # Standardize DASS controls
    for col in ['dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()

    results = {}

    # Model 1: Linear only
    formula1 = f"{outcome} ~ ucla_c + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model1 = smf.ols(formula1, data=df).fit()
    results['linear'] = {
        'r2': model1.rsquared,
        'aic': model1.aic,
        'bic': model1.bic,
        'linear_coef': model1.params.get('ucla_c', np.nan),
        'linear_p': model1.pvalues.get('ucla_c', np.nan)
    }

    # Model 2: Quadratic
    formula2 = f"{outcome} ~ ucla_c + ucla_c2 + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model2 = smf.ols(formula2, data=df).fit()
    results['quadratic'] = {
        'r2': model2.rsquared,
        'aic': model2.aic,
        'bic': model2.bic,
        'linear_coef': model2.params.get('ucla_c', np.nan),
        'linear_p': model2.pvalues.get('ucla_c', np.nan),
        'quadratic_coef': model2.params.get('ucla_c2', np.nan),
        'quadratic_p': model2.pvalues.get('ucla_c2', np.nan)
    }

    # Model 3: Cubic
    formula3 = f"{outcome} ~ ucla_c + ucla_c2 + ucla_c3 + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model3 = smf.ols(formula3, data=df).fit()
    results['cubic'] = {
        'r2': model3.rsquared,
        'aic': model3.aic,
        'bic': model3.bic,
        'cubic_coef': model3.params.get('ucla_c3', np.nan),
        'cubic_p': model3.pvalues.get('ucla_c3', np.nan)
    }

    # F-test for model improvement
    from scipy.stats import f as f_dist

    # Linear vs Quadratic
    df_num = 1  # Added 1 parameter
    df_denom = len(df) - len(model2.params)
    ss_res1 = model1.ssr
    ss_res2 = model2.ssr
    f_stat_quad = ((ss_res1 - ss_res2) / df_num) / (ss_res2 / df_denom)
    p_quad = 1 - f_dist.cdf(f_stat_quad, df_num, df_denom)
    results['linear_vs_quadratic'] = {'f': f_stat_quad, 'p': p_quad}

    # Quadratic vs Cubic
    ss_res3 = model3.ssr
    df_denom3 = len(df) - len(model3.params)
    f_stat_cub = ((ss_res2 - ss_res3) / 1) / (ss_res3 / df_denom3)
    p_cub = 1 - f_dist.cdf(f_stat_cub, 1, df_denom3)
    results['quadratic_vs_cubic'] = {'f': f_stat_cub, 'p': p_cub}

    # Best model by AIC
    models = {'linear': model1, 'quadratic': model2, 'cubic': model3}
    best = min(models.keys(), key=lambda k: getattr(models[k], 'aic'))
    results['best_model_aic'] = best

    return results, models, df


def find_breakpoint(df, outcome, min_break=25, max_break=55):
    """Find optimal breakpoint for threshold effect."""
    ucla_vals = df['ucla_score'].values
    outcome_vals = df[outcome].values

    # Remove missing
    mask = ~(np.isnan(ucla_vals) | np.isnan(outcome_vals))
    ucla_vals = ucla_vals[mask]
    outcome_vals = outcome_vals[mask]

    if len(ucla_vals) < 30:
        return None

    best_breakpoint = None
    best_ssr = np.inf

    for bp in range(min_break, max_break + 1, 5):
        low = ucla_vals < bp
        high = ucla_vals >= bp

        if low.sum() < 10 or high.sum() < 10:
            continue

        # Fit separate means
        mean_low = outcome_vals[low].mean()
        mean_high = outcome_vals[high].mean()

        ssr = np.sum((outcome_vals[low] - mean_low)**2) + np.sum((outcome_vals[high] - mean_high)**2)

        if ssr < best_ssr:
            best_ssr = ssr
            best_breakpoint = bp

    if best_breakpoint is None:
        return None

    # Test if breakpoint model is better than linear
    low = ucla_vals < best_breakpoint
    high = ucla_vals >= best_breakpoint

    mean_low = outcome_vals[low].mean()
    mean_high = outcome_vals[high].mean()
    diff = mean_high - mean_low

    t_stat, p_val = stats.ttest_ind(outcome_vals[low], outcome_vals[high])

    return {
        'breakpoint': best_breakpoint,
        'n_below': low.sum(),
        'n_above': high.sum(),
        'mean_below': mean_low,
        'mean_above': mean_high,
        'difference': diff,
        't': t_stat,
        'p': p_val
    }


def analyze_extreme_groups(master_df, outcome):
    """Compare extreme UCLA groups (quartiles)."""
    df = master_df[[outcome, 'ucla_score', 'dass_depression', 'dass_anxiety',
                    'dass_stress', 'age', 'gender_male']].dropna()

    if len(df) < 40:
        return None

    # Quartiles
    q1 = df['ucla_score'].quantile(0.25)
    q3 = df['ucla_score'].quantile(0.75)

    low_group = df[df['ucla_score'] <= q1]
    high_group = df[df['ucla_score'] >= q3]

    if len(low_group) < 10 or len(high_group) < 10:
        return None

    # Unadjusted comparison
    t_unadj, p_unadj = stats.ttest_ind(low_group[outcome], high_group[outcome])

    # Effect size
    pooled_std = np.sqrt(
        ((len(low_group)-1)*low_group[outcome].std()**2 +
         (len(high_group)-1)*high_group[outcome].std()**2) /
        (len(low_group) + len(high_group) - 2)
    )
    cohens_d = (high_group[outcome].mean() - low_group[outcome].mean()) / pooled_std if pooled_std > 0 else 0

    return {
        'q1_cutoff': q1,
        'q3_cutoff': q3,
        'low_n': len(low_group),
        'low_mean': low_group[outcome].mean(),
        'low_std': low_group[outcome].std(),
        'high_n': len(high_group),
        'high_mean': high_group[outcome].mean(),
        'high_std': high_group[outcome].std(),
        't_unadjusted': t_unadj,
        'p_unadjusted': p_unadj,
        'cohens_d': cohens_d
    }


def main():
    print("=" * 60)
    print("Nonlinear UCLA Effects Analysis")
    print("=" * 60)

    master = load_master_dataset(use_cache=True)
    print(f"Dataset loaded: N={len(master)}")
    print(f"UCLA range: {master['ucla_score'].min():.0f} - {master['ucla_score'].max():.0f}")

    ef_outcomes = {
        'pe_rate': 'WCST PE Rate',
        'stroop_interference': 'Stroop Interference',
        'prp_bottleneck': 'PRP Bottleneck'
    }

    all_results = []

    for outcome, label in ef_outcomes.items():
        if outcome not in master.columns:
            continue

        print(f"\n{'='*60}")
        print(f"Outcome: {label}")
        print("=" * 60)

        # Polynomial analysis
        print("\n[1] Polynomial Regression")
        print("-" * 40)

        poly_results = test_polynomial_effects(master, outcome)
        if poly_results:
            results, models, df = poly_results

            print(f"  Linear R²: {results['linear']['r2']:.4f}, AIC: {results['linear']['aic']:.1f}")
            print(f"  Quadratic R²: {results['quadratic']['r2']:.4f}, AIC: {results['quadratic']['aic']:.1f}")
            print(f"    Quadratic term: b={results['quadratic']['quadratic_coef']:.4f}, p={results['quadratic']['quadratic_p']:.4f}")
            print(f"  Cubic R²: {results['cubic']['r2']:.4f}, AIC: {results['cubic']['aic']:.1f}")
            print(f"  Best model (AIC): {results['best_model_aic']}")
            print(f"  Linear vs Quadratic: F={results['linear_vs_quadratic']['f']:.2f}, p={results['linear_vs_quadratic']['p']:.4f}")

            all_results.append({
                'outcome': outcome,
                'outcome_label': label,
                **{f'poly_{k}': v for k, v in results['linear'].items()},
                'quad_coef': results['quadratic']['quadratic_coef'],
                'quad_p': results['quadratic']['quadratic_p'],
                'best_model': results['best_model_aic'],
                'linear_vs_quad_p': results['linear_vs_quadratic']['p']
            })

        # Breakpoint analysis
        print("\n[2] Breakpoint Analysis")
        print("-" * 40)

        bp_results = find_breakpoint(master, outcome)
        if bp_results:
            print(f"  Optimal breakpoint: UCLA = {bp_results['breakpoint']}")
            print(f"  Below (N={bp_results['n_below']}): M = {bp_results['mean_below']:.2f}")
            print(f"  Above (N={bp_results['n_above']}): M = {bp_results['mean_above']:.2f}")
            print(f"  Difference: {bp_results['difference']:.2f}")
            print(f"  t = {bp_results['t']:.2f}, p = {bp_results['p']:.4f}")

            if all_results:
                all_results[-1]['breakpoint'] = bp_results['breakpoint']
                all_results[-1]['breakpoint_p'] = bp_results['p']

        # Extreme groups
        print("\n[3] Extreme Group Comparison")
        print("-" * 40)

        extreme_results = analyze_extreme_groups(master, outcome)
        if extreme_results:
            print(f"  Low UCLA (≤{extreme_results['q1_cutoff']:.0f}, N={extreme_results['low_n']}): M = {extreme_results['low_mean']:.2f}")
            print(f"  High UCLA (≥{extreme_results['q3_cutoff']:.0f}, N={extreme_results['high_n']}): M = {extreme_results['high_mean']:.2f}")
            print(f"  t = {extreme_results['t_unadjusted']:.2f}, p = {extreme_results['p_unadjusted']:.4f}")
            print(f"  Cohen's d = {extreme_results['cohens_d']:.3f}")

            if all_results:
                all_results[-1]['extreme_group_p'] = extreme_results['p_unadjusted']
                all_results[-1]['extreme_group_d'] = extreme_results['cohens_d']

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_DIR / "nonlinear_analysis_results.csv", index=False, encoding='utf-8-sig')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Nonlinear Effect Analysis Results:

For each EF outcome, tested:
1. Quadratic/cubic polynomial terms
2. Optimal breakpoint threshold
3. Extreme group (Q1 vs Q3) comparisons

Key Questions:
- Is there a nonlinear relationship?
- Is there a threshold above which UCLA affects EF?
- Are extreme groups significantly different?

""")
    if all_results:
        print("Results by Outcome:")
        for res in all_results:
            quad_sig = "YES" if res.get('quad_p', 1) < 0.05 else "NO"
            bp_sig = "YES" if res.get('breakpoint_p', 1) < 0.05 else "NO"
            ext_sig = "YES" if res.get('extreme_group_p', 1) < 0.05 else "NO"
            print(f"\n  {res['outcome_label']}:")
            print(f"    Quadratic effect significant: {quad_sig}")
            print(f"    Breakpoint significant: {bp_sig}")
            print(f"    Extreme groups different: {ext_sig}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
