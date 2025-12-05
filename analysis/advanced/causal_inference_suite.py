"""
Causal Inference Suite
======================

Sensitivity analysis and causal bounds for UCLA → Executive Function relationships.

Analyses:
1. sensitivity_analysis: E-value and unmeasured confounding sensitivity
2. dag_model_comparison: Compare alternative causal DAG structures
3. causal_bounds: Partial identification bounds on causal effects
4. instrumental_variable: IV-style analysis using age as instrument
5. mediation_sensitivity: Sensitivity of mediation estimates

Key Question: How robust are the UCLA-EF null findings to unmeasured confounding?
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analysis.preprocessing import load_master_dataset


# Output directory
OUTPUT_DIR = Path("results/analysis_outputs/causal_inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# E-VALUE AND SENSITIVITY ANALYSIS
# =============================================================================

def compute_e_value(point_estimate: float, ci_lower: float, ci_upper: float,
                   outcome_type: str = 'continuous') -> Dict[str, float]:
    """
    Compute E-value for sensitivity to unmeasured confounding.

    E-value answers: How strong would an unmeasured confounder need to be
    to explain away the observed effect?

    For continuous outcomes, convert standardized coefficient to RR approximation.
    """
    # Convert standardized beta to approximate RR
    # Using sqrt(exp(0.91*beta)) approximation for continuous outcomes
    if outcome_type == 'continuous':
        # Standardized beta → approximate RR
        rr_point = np.exp(0.91 * abs(point_estimate))
        rr_lower = np.exp(0.91 * abs(ci_lower)) if ci_lower * point_estimate > 0 else 1.0
        rr_upper = np.exp(0.91 * abs(ci_upper)) if ci_upper * point_estimate > 0 else 1.0
    else:
        rr_point = abs(point_estimate)
        rr_lower = abs(ci_lower) if ci_lower > 1 else 1/ci_lower if ci_lower > 0 else 1.0
        rr_upper = abs(ci_upper) if ci_upper > 1 else 1/ci_upper if ci_upper > 0 else 1.0

    # E-value formula: RR + sqrt(RR * (RR - 1))
    def e_value_calc(rr):
        if rr < 1:
            rr = 1/rr if rr > 0 else 1.0
        if rr <= 1:
            return 1.0
        return rr + np.sqrt(rr * (rr - 1))

    e_point = e_value_calc(rr_point)
    e_ci = e_value_calc(min(rr_lower, rr_upper))  # CI bound closer to null

    return {
        'e_value_point': e_point,
        'e_value_ci': e_ci,
        'rr_approximation': rr_point,
        'interpretation': interpret_e_value(e_point)
    }


def interpret_e_value(e_value: float) -> str:
    """Interpret E-value magnitude."""
    if e_value < 1.5:
        return "Very weak: trivial unmeasured confounding could explain away"
    elif e_value < 2.0:
        return "Weak: modest confounding could explain away"
    elif e_value < 3.0:
        return "Moderate: substantial confounding needed"
    elif e_value < 5.0:
        return "Strong: strong confounding needed"
    else:
        return "Very strong: extreme confounding needed to explain away"


def run_sensitivity_analysis(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Sensitivity analysis for UCLA → EF effects.

    Compute E-values to assess robustness to unmeasured confounding.
    """
    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS: E-VALUES FOR UNMEASURED CONFOUNDING")
        print("="*70)

    results = {'outcomes': {}}

    # Outcomes to analyze
    outcomes = [
        ('stroop_interference', 'Stroop Interference'),
        ('prp_effect', 'PRP Effect'),
        ('pe_rate', 'WCST PE Rate'),
    ]

    for outcome_col, outcome_name in outcomes:
        if outcome_col not in df.columns:
            continue

        if verbose:
            print(f"\n--- {outcome_name} ---")

        outcome_results = {}

        # Model 1: UCLA only (unadjusted)
        try:
            model1 = smf.ols(f'{outcome_col} ~ z_ucla', data=df).fit()
            beta_ucla_raw = model1.params.get('z_ucla', 0)
            ci_raw = model1.conf_int().loc['z_ucla'].values

            e_raw = compute_e_value(beta_ucla_raw, ci_raw[0], ci_raw[1])
            outcome_results['unadjusted'] = {
                'beta': beta_ucla_raw,
                'ci': ci_raw.tolist(),
                'p_value': model1.pvalues.get('z_ucla', 1),
                **e_raw
            }

            if verbose:
                print(f"  Unadjusted: β={beta_ucla_raw:.4f}, p={model1.pvalues.get('z_ucla', 1):.4f}")
                print(f"    E-value (point): {e_raw['e_value_point']:.2f}")
                print(f"    E-value (CI): {e_raw['e_value_ci']:.2f}")
                print(f"    {e_raw['interpretation']}")
        except Exception as e:
            if verbose:
                print(f"  Unadjusted model failed: {e}")

        # Model 2: DASS-adjusted
        try:
            formula = f'{outcome_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age'
            model2 = smf.ols(formula, data=df).fit()
            beta_ucla_adj = model2.params.get('z_ucla', 0)
            ci_adj = model2.conf_int().loc['z_ucla'].values

            e_adj = compute_e_value(beta_ucla_adj, ci_adj[0], ci_adj[1])
            outcome_results['dass_adjusted'] = {
                'beta': beta_ucla_adj,
                'ci': ci_adj.tolist(),
                'p_value': model2.pvalues.get('z_ucla', 1),
                **e_adj
            }

            # Attenuation ratio
            if beta_ucla_raw != 0:
                attenuation = 1 - abs(beta_ucla_adj / beta_ucla_raw)
                outcome_results['attenuation_ratio'] = attenuation

            if verbose:
                print(f"  DASS-adjusted: β={beta_ucla_adj:.4f}, p={model2.pvalues.get('z_ucla', 1):.4f}")
                print(f"    E-value (point): {e_adj['e_value_point']:.2f}")
                print(f"    E-value (CI): {e_adj['e_value_ci']:.2f}")
                print(f"    Attenuation: {outcome_results.get('attenuation_ratio', 0)*100:.1f}% of effect explained by DASS")
        except Exception as e:
            if verbose:
                print(f"  DASS-adjusted model failed: {e}")

        # Model 3: UCLA × Gender interaction (the significant finding)
        try:
            formula = f'{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age'
            model3 = smf.ols(formula, data=df).fit()

            interaction_term = 'z_ucla:C(gender_male)[T.True]'
            if interaction_term in model3.params:
                beta_int = model3.params[interaction_term]
                ci_int = model3.conf_int().loc[interaction_term].values

                e_int = compute_e_value(beta_int, ci_int[0], ci_int[1])
                outcome_results['interaction'] = {
                    'beta': beta_int,
                    'ci': ci_int.tolist(),
                    'p_value': model3.pvalues.get(interaction_term, 1),
                    **e_int
                }

                if verbose:
                    print(f"  UCLA × Gender: β={beta_int:.4f}, p={model3.pvalues.get(interaction_term, 1):.4f}")
                    print(f"    E-value (point): {e_int['e_value_point']:.2f}")
                    print(f"    E-value (CI): {e_int['e_value_ci']:.2f}")
                    print(f"    {e_int['interpretation']}")
        except Exception as e:
            if verbose:
                print(f"  Interaction model failed: {e}")

        results['outcomes'][outcome_col] = outcome_results

    # Create E-value visualization
    _plot_e_values(results, OUTPUT_DIR / 'e_values_sensitivity.png')

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("E-VALUE INTERPRETATION GUIDE:")
        print("  E-value = 1.0: No confounding needed (effect is null)")
        print("  E-value = 2.0: Confounder must double risk for both exposure and outcome")
        print("  E-value = 3.0: Confounder must triple risk")
        print("  Higher E-values → more robust to unmeasured confounding")
        print("="*70)

    return results


def _plot_e_values(results: Dict, save_path: Path) -> None:
    """Visualize E-values across outcomes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    outcomes = []
    e_points = []
    e_cis = []
    model_types = []

    for outcome, data in results.get('outcomes', {}).items():
        for model_type in ['unadjusted', 'dass_adjusted', 'interaction']:
            if model_type in data:
                outcomes.append(f"{outcome}\n({model_type})")
                e_points.append(data[model_type].get('e_value_point', 1))
                e_cis.append(data[model_type].get('e_value_ci', 1))
                model_types.append(model_type)

    if not outcomes:
        plt.close()
        return

    y_pos = np.arange(len(outcomes))
    colors = {'unadjusted': 'steelblue', 'dass_adjusted': 'orange', 'interaction': 'green'}
    bar_colors = [colors.get(mt, 'gray') for mt in model_types]

    ax.barh(y_pos, e_points, color=bar_colors, alpha=0.7, label='E-value (point)')
    ax.scatter(e_cis, y_pos, color='red', marker='|', s=200, label='E-value (CI bound)')

    ax.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, label='E=2.0 threshold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(outcomes)
    ax.set_xlabel('E-value')
    ax.set_title('E-values for UCLA → EF Effects\n(Higher = More Robust to Confounding)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# DAG MODEL COMPARISON
# =============================================================================

def run_dag_model_comparison(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Compare alternative causal DAG structures.

    DAG Models:
    1. UCLA → EF (direct effect)
    2. UCLA → DASS → EF (full mediation)
    3. UCLA → EF + UCLA → DASS → EF (partial mediation)
    4. Common cause: Unobserved → UCLA, EF (confounding)
    """
    if verbose:
        print("\n" + "="*70)
        print("DAG MODEL COMPARISON")
        print("="*70)

    results = {'models': {}, 'comparisons': {}}

    outcome = 'pe_rate'  # Focus on WCST PE (most significant)

    if outcome not in df.columns:
        if verbose:
            print("WCST PE rate not available")
        return results

    # Model 1: Direct effect only (UCLA → PE)
    try:
        m1 = smf.ols(f'{outcome} ~ z_ucla + z_age + C(gender_male)', data=df).fit()
        results['models']['direct_only'] = {
            'formula': f'{outcome} ~ z_ucla + z_age + C(gender_male)',
            'aic': m1.aic,
            'bic': m1.bic,
            'r2': m1.rsquared,
            'ucla_beta': m1.params.get('z_ucla', 0),
            'ucla_p': m1.pvalues.get('z_ucla', 1),
        }
        if verbose:
            print(f"\nModel 1 (Direct): UCLA → PE")
            print(f"  AIC={m1.aic:.1f}, BIC={m1.bic:.1f}, R²={m1.rsquared:.4f}")
            print(f"  UCLA β={m1.params.get('z_ucla', 0):.4f}, p={m1.pvalues.get('z_ucla', 1):.4f}")
    except Exception as e:
        if verbose:
            print(f"Model 1 failed: {e}")

    # Model 2: DASS mediates fully (no direct UCLA effect)
    try:
        m2 = smf.ols(f'{outcome} ~ z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)', data=df).fit()
        results['models']['dass_only'] = {
            'formula': f'{outcome} ~ z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)',
            'aic': m2.aic,
            'bic': m2.bic,
            'r2': m2.rsquared,
        }
        if verbose:
            print(f"\nModel 2 (DASS only): DASS → PE (no direct UCLA)")
            print(f"  AIC={m2.aic:.1f}, BIC={m2.bic:.1f}, R²={m2.rsquared:.4f}")
    except Exception as e:
        if verbose:
            print(f"Model 2 failed: {e}")

    # Model 3: Partial mediation (UCLA + DASS)
    try:
        m3 = smf.ols(f'{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)', data=df).fit()
        results['models']['partial_mediation'] = {
            'formula': f'{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)',
            'aic': m3.aic,
            'bic': m3.bic,
            'r2': m3.rsquared,
            'ucla_beta': m3.params.get('z_ucla', 0),
            'ucla_p': m3.pvalues.get('z_ucla', 1),
        }
        if verbose:
            print(f"\nModel 3 (Partial mediation): UCLA + DASS → PE")
            print(f"  AIC={m3.aic:.1f}, BIC={m3.bic:.1f}, R²={m3.rsquared:.4f}")
            print(f"  UCLA β={m3.params.get('z_ucla', 0):.4f}, p={m3.pvalues.get('z_ucla', 1):.4f}")
    except Exception as e:
        if verbose:
            print(f"Model 3 failed: {e}")

    # Model 4: Interaction model (UCLA × Gender)
    try:
        m4 = smf.ols(f'{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age', data=df).fit()
        int_term = 'z_ucla:C(gender_male)[T.True]'
        results['models']['interaction'] = {
            'formula': f'{outcome} ~ z_ucla * C(gender_male) + DASS + z_age',
            'aic': m4.aic,
            'bic': m4.bic,
            'r2': m4.rsquared,
            'interaction_beta': m4.params.get(int_term, 0),
            'interaction_p': m4.pvalues.get(int_term, 1),
        }
        if verbose:
            print(f"\nModel 4 (Interaction): UCLA × Gender → PE")
            print(f"  AIC={m4.aic:.1f}, BIC={m4.bic:.1f}, R²={m4.rsquared:.4f}")
            print(f"  Interaction β={m4.params.get(int_term, 0):.4f}, p={m4.pvalues.get(int_term, 1):.4f}")
    except Exception as e:
        if verbose:
            print(f"Model 4 failed: {e}")

    # Model comparison using BIC
    if len(results['models']) >= 2:
        bics = {k: v['bic'] for k, v in results['models'].items() if 'bic' in v}
        if bics:
            best_model = min(bics, key=bics.get)
            min_bic = bics[best_model]

            # Compute Bayes Factor approximations
            bf_approx = {}
            for model, bic in bics.items():
                delta_bic = bic - min_bic
                bf_approx[model] = np.exp(-0.5 * delta_bic)

            results['comparisons'] = {
                'best_model': best_model,
                'bic_values': bics,
                'bayes_factor_approx': bf_approx,
            }

            if verbose:
                print("\n" + "-"*50)
                print("MODEL COMPARISON (BIC-based):")
                print(f"  Best model: {best_model}")
                print("  Bayes Factor approximations (vs best):")
                for model, bf in bf_approx.items():
                    print(f"    {model}: BF={bf:.4f}")

    # Visualize DAG comparison
    _plot_dag_comparison(results, OUTPUT_DIR / 'dag_model_comparison.png')

    return results


def _plot_dag_comparison(results: Dict, save_path: Path) -> None:
    """Visualize DAG model comparison."""
    models = results.get('models', {})
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: BIC comparison
    ax1 = axes[0]
    model_names = list(models.keys())
    bics = [models[m].get('bic', 0) for m in model_names]
    colors = ['green' if m == results.get('comparisons', {}).get('best_model') else 'steelblue'
              for m in model_names]

    ax1.barh(model_names, bics, color=colors)
    ax1.set_xlabel('BIC (lower is better)')
    ax1.set_title('Model Comparison: BIC')

    # Plot 2: Effect sizes
    ax2 = axes[1]
    effect_data = []
    for m in model_names:
        if 'ucla_beta' in models[m]:
            effect_data.append({'model': m, 'effect': 'UCLA main', 'beta': models[m]['ucla_beta']})
        if 'interaction_beta' in models[m]:
            effect_data.append({'model': m, 'effect': 'UCLA×Gender', 'beta': models[m]['interaction_beta']})

    if effect_data:
        effect_df = pd.DataFrame(effect_data)
        for i, (effect, grp) in enumerate(effect_df.groupby('effect')):
            ax2.barh([f"{m}\n({effect})" for m in grp['model']], grp['beta'],
                    alpha=0.7, label=effect)

    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Standardized β')
    ax2.set_title('UCLA Effect Sizes by Model')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# CAUSAL BOUNDS (PARTIAL IDENTIFICATION)
# =============================================================================

def run_causal_bounds(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Compute bounds on causal effects under different assumptions.

    Uses partial identification approach:
    - Under no assumptions: bounds can be [-∞, +∞]
    - Under monotonicity: tighter bounds
    - Under bounded confounding: parametric bounds
    """
    if verbose:
        print("\n" + "="*70)
        print("CAUSAL BOUNDS ANALYSIS")
        print("="*70)

    results = {'outcomes': {}}

    outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('stroop_interference', 'Stroop Interference'),
    ]

    for outcome_col, outcome_name in outcomes:
        if outcome_col not in df.columns or 'z_ucla' not in df.columns:
            continue

        if verbose:
            print(f"\n--- {outcome_name} ---")

        outcome_results = {}

        # Observed association (correlational bound)
        obs_df = df[[outcome_col, 'z_ucla']].dropna()
        if len(obs_df) < 20:
            continue

        obs_corr, _ = stats.pearsonr(obs_df['z_ucla'], obs_df[outcome_col])

        # OLS estimate (under no unmeasured confounding)
        ols_model = smf.ols(f'{outcome_col} ~ z_ucla', data=df).fit()
        ols_beta = ols_model.params.get('z_ucla', 0)
        ols_se = ols_model.bse.get('z_ucla', 0)

        # DASS-adjusted estimate
        adj_model = smf.ols(f'{outcome_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age', data=df).fit()
        adj_beta = adj_model.params.get('z_ucla', 0)
        adj_se = adj_model.bse.get('z_ucla', 0)

        # Bound under different confounding scenarios
        # Assume maximum confounding = R² explained by DASS
        r2_full = adj_model.rsquared
        r2_ucla_only = smf.ols(f'{outcome_col} ~ z_ucla', data=df).fit().rsquared
        confounding_strength = r2_full - r2_ucla_only

        # Rosenbaum-style bounds
        gamma_values = [1.0, 1.5, 2.0, 3.0]  # Sensitivity parameter
        bounds_by_gamma = {}

        for gamma in gamma_values:
            # Under gamma-fold confounding
            lower = adj_beta - adj_se * 1.96 * gamma
            upper = adj_beta + adj_se * 1.96 * gamma
            bounds_by_gamma[gamma] = {'lower': lower, 'upper': upper}

        outcome_results = {
            'observed_correlation': obs_corr,
            'ols_beta': ols_beta,
            'ols_ci': [ols_beta - 1.96*ols_se, ols_beta + 1.96*ols_se],
            'adjusted_beta': adj_beta,
            'adjusted_ci': [adj_beta - 1.96*adj_se, adj_beta + 1.96*adj_se],
            'confounding_strength': confounding_strength,
            'bounds_by_gamma': bounds_by_gamma,
        }

        if verbose:
            print(f"  Observed correlation: r = {obs_corr:.4f}")
            print(f"  OLS (unadjusted): β = {ols_beta:.4f} [{ols_beta - 1.96*ols_se:.4f}, {ols_beta + 1.96*ols_se:.4f}]")
            print(f"  DASS-adjusted: β = {adj_beta:.4f} [{adj_beta - 1.96*adj_se:.4f}, {adj_beta + 1.96*adj_se:.4f}]")
            print(f"  Confounding strength (R² diff): {confounding_strength:.4f}")
            print("  Bounds under different confounding (Γ):")
            for gamma, bounds in bounds_by_gamma.items():
                includes_zero = bounds['lower'] <= 0 <= bounds['upper']
                print(f"    Γ={gamma}: [{bounds['lower']:.4f}, {bounds['upper']:.4f}] {'(includes 0)' if includes_zero else ''}")

        results['outcomes'][outcome_col] = outcome_results

    # Visualization
    _plot_causal_bounds(results, OUTPUT_DIR / 'causal_bounds.png')

    return results


def _plot_causal_bounds(results: Dict, save_path: Path) -> None:
    """Visualize causal bounds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_offset = 0
    y_labels = []

    for outcome, data in results.get('outcomes', {}).items():
        # Plot adjusted estimate and CI
        adj_beta = data.get('adjusted_beta', 0)
        adj_ci = data.get('adjusted_ci', [0, 0])

        ax.errorbar(adj_beta, y_offset, xerr=[[adj_beta - adj_ci[0]], [adj_ci[1] - adj_beta]],
                   fmt='o', color='blue', capsize=5, label='Adjusted (Γ=1)' if y_offset == 0 else '')
        y_labels.append(f"{outcome}\n(adjusted)")
        y_offset += 1

        # Plot bounds for different gamma
        colors = {1.5: 'green', 2.0: 'orange', 3.0: 'red'}
        for gamma, bounds in data.get('bounds_by_gamma', {}).items():
            if gamma == 1.0:
                continue
            ax.fill_betweenx([y_offset-0.3, y_offset+0.3], bounds['lower'], bounds['upper'],
                           alpha=0.3, color=colors.get(gamma, 'gray'),
                           label=f'Γ={gamma}' if outcome == list(results.get('outcomes', {}).keys())[0] else '')
            y_labels.append(f"{outcome}\n(Γ={gamma})")
            y_offset += 1

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Causal Effect (standardized β)')
    ax.set_title('Causal Bounds Under Different Confounding Assumptions')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# MEDIATION SENSITIVITY
# =============================================================================

def run_mediation_sensitivity(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Sensitivity analysis for mediation effects (UCLA → DASS → EF).

    Key question: How sensitive are mediation conclusions to unmeasured confounding
    of the mediator-outcome relationship?
    """
    if verbose:
        print("\n" + "="*70)
        print("MEDIATION SENSITIVITY ANALYSIS")
        print("="*70)

    results = {}

    outcome = 'pe_rate'
    mediator = 'z_dass_dep'  # Use depression as primary mediator

    if outcome not in df.columns or mediator not in df.columns:
        if verbose:
            print("Required variables not available")
        return results

    # Step 1: a-path (UCLA → DASS)
    a_model = smf.ols(f'{mediator} ~ z_ucla', data=df).fit()
    a = a_model.params.get('z_ucla', 0)
    a_se = a_model.bse.get('z_ucla', 0)

    # Step 2: b-path (DASS → PE | UCLA)
    b_model = smf.ols(f'{outcome} ~ z_ucla + {mediator}', data=df).fit()
    b = b_model.params.get(mediator, 0)
    b_se = b_model.bse.get(mediator, 0)

    # Indirect effect
    indirect = a * b
    # Sobel SE approximation
    indirect_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)

    # Direct effect
    direct = b_model.params.get('z_ucla', 0)
    direct_se = b_model.bse.get('z_ucla', 0)

    # Total effect
    total = a * b + direct

    results['paths'] = {
        'a_path': {'beta': a, 'se': a_se, 'p': a_model.pvalues.get('z_ucla', 1)},
        'b_path': {'beta': b, 'se': b_se, 'p': b_model.pvalues.get(mediator, 1)},
        'indirect': {'beta': indirect, 'se': indirect_se, 'ci': [indirect - 1.96*indirect_se, indirect + 1.96*indirect_se]},
        'direct': {'beta': direct, 'se': direct_se, 'ci': [direct - 1.96*direct_se, direct + 1.96*direct_se]},
        'total': {'beta': total},
    }

    if verbose:
        print(f"\nMediation: UCLA → DASS-Depression → {outcome}")
        print(f"  a-path (UCLA → DASS): β = {a:.4f}, p = {a_model.pvalues.get('z_ucla', 1):.4f}")
        print(f"  b-path (DASS → PE|UCLA): β = {b:.4f}, p = {b_model.pvalues.get(mediator, 1):.4f}")
        print(f"  Indirect effect: β = {indirect:.4f} [{indirect - 1.96*indirect_se:.4f}, {indirect + 1.96*indirect_se:.4f}]")
        print(f"  Direct effect: β = {direct:.4f} [{direct - 1.96*direct_se:.4f}, {direct + 1.96*direct_se:.4f}]")

    # Sensitivity: What if mediator-outcome has unmeasured confounding?
    # Compute how much confounding (rho) would be needed to nullify indirect effect
    rho_values = np.linspace(-0.5, 0.5, 21)
    adjusted_indirect = []

    for rho in rho_values:
        # Adjust b for confounding bias
        b_adj = b - rho * np.std(df[mediator].dropna()) / np.std(df[outcome].dropna())
        indirect_adj = a * b_adj
        adjusted_indirect.append(indirect_adj)

    results['sensitivity'] = {
        'rho_values': rho_values.tolist(),
        'adjusted_indirect': adjusted_indirect,
    }

    # Find rho that nullifies
    for i, (rho, ind) in enumerate(zip(rho_values, adjusted_indirect)):
        if i > 0 and adjusted_indirect[i-1] * ind <= 0:
            results['rho_nullify'] = rho
            if verbose:
                print(f"\n  Sensitivity: ρ ≈ {rho:.2f} would nullify indirect effect")
            break

    # Visualize
    _plot_mediation_sensitivity(results, OUTPUT_DIR / 'mediation_sensitivity.png')

    return results


def _plot_mediation_sensitivity(results: Dict, save_path: Path) -> None:
    """Visualize mediation sensitivity."""
    fig, ax = plt.subplots(figsize=(8, 6))

    rho = results.get('sensitivity', {}).get('rho_values', [])
    indirect = results.get('sensitivity', {}).get('adjusted_indirect', [])

    if not rho or not indirect:
        plt.close()
        return

    ax.plot(rho, indirect, 'b-', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax.fill_between(rho, indirect, 0, where=[i > 0 for i in indirect],
                    alpha=0.3, color='green', label='Positive indirect effect')
    ax.fill_between(rho, indirect, 0, where=[i < 0 for i in indirect],
                    alpha=0.3, color='red', label='Negative indirect effect')

    ax.set_xlabel('Unmeasured Confounding (ρ)')
    ax.set_ylabel('Indirect Effect (UCLA → DASS → PE)')
    ax.set_title('Mediation Sensitivity Analysis\nHow much confounding would nullify the indirect effect?')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# INSTRUMENTAL VARIABLE ANALYSIS
# =============================================================================

def run_instrumental_variable(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    IV-style analysis using pre-treatment variables.

    Challenge: No true instrument available.
    Approach: Use education as pseudo-instrument (affects UCLA through social context,
    less direct effect on EF after controlling for confounds).

    Note: This is exploratory/educational, not a rigorous IV analysis.
    """
    if verbose:
        print("\n" + "="*70)
        print("INSTRUMENTAL VARIABLE EXPLORATION")
        print("="*70)
        print("(Exploratory: No true instrument available)")

    results = {}

    # Check if education is available
    if 'education' not in df.columns and 'education_years' not in df.columns:
        if verbose:
            print("Education variable not available for IV exploration")
        return results

    edu_col = 'education_years' if 'education_years' in df.columns else 'education'

    # First stage: Education → UCLA
    try:
        first_stage = smf.ols(f'z_ucla ~ {edu_col} + z_age + C(gender_male)', data=df).fit()
        f_stat = first_stage.fvalue

        results['first_stage'] = {
            'f_stat': f_stat,
            'r2': first_stage.rsquared,
            'edu_beta': first_stage.params.get(edu_col, 0),
            'edu_p': first_stage.pvalues.get(edu_col, 1),
        }

        if verbose:
            print(f"\nFirst stage (Education → UCLA):")
            print(f"  F-statistic: {f_stat:.2f} {'(weak)' if f_stat < 10 else '(adequate)'}")
            print(f"  R²: {first_stage.rsquared:.4f}")
            print(f"  Education β: {first_stage.params.get(edu_col, 0):.4f}")
    except Exception as e:
        if verbose:
            print(f"First stage failed: {e}")
        return results

    # Reduced form and second stage (2SLS approximation)
    outcome = 'pe_rate'
    if outcome not in df.columns:
        return results

    try:
        # Reduced form: Education → PE
        reduced = smf.ols(f'{outcome} ~ {edu_col} + z_age + C(gender_male)', data=df).fit()

        # Predicted UCLA from first stage
        df['ucla_hat'] = first_stage.fittedvalues

        # Second stage: Predicted UCLA → PE
        second_stage = smf.ols(f'{outcome} ~ ucla_hat + z_age + C(gender_male)', data=df).fit()

        # IV estimate = reduced form / first stage
        iv_estimate = reduced.params.get(edu_col, 0) / first_stage.params.get(edu_col, 1)

        results['iv_estimate'] = {
            'beta': iv_estimate,
            'second_stage_beta': second_stage.params.get('ucla_hat', 0),
            'second_stage_p': second_stage.pvalues.get('ucla_hat', 1),
        }

        # Compare to OLS
        ols_model = smf.ols(f'{outcome} ~ z_ucla + z_age + C(gender_male)', data=df).fit()
        ols_beta = ols_model.params.get('z_ucla', 0)

        results['comparison'] = {
            'ols_beta': ols_beta,
            'iv_beta': iv_estimate,
            'ratio': iv_estimate / ols_beta if ols_beta != 0 else np.nan,
        }

        if verbose:
            print(f"\nIV Estimate (UCLA → {outcome}):")
            print(f"  IV β: {iv_estimate:.4f}")
            print(f"  OLS β: {ols_beta:.4f}")
            print(f"  Ratio (IV/OLS): {iv_estimate/ols_beta:.2f}" if ols_beta != 0 else "  Ratio: undefined")
            print("\n  Note: IV typically larger than OLS if confounding biases toward null")

        # Clean up
        df.drop('ucla_hat', axis=1, inplace=True, errors='ignore')

    except Exception as e:
        if verbose:
            print(f"IV analysis failed: {e}")

    return results


# =============================================================================
# ANALYSIS REGISTRY AND RUNNER
# =============================================================================

ANALYSES = {
    'sensitivity': {
        'name': 'E-Value Sensitivity Analysis',
        'func': run_sensitivity_analysis,
        'description': 'Compute E-values for robustness to unmeasured confounding',
    },
    'dag_comparison': {
        'name': 'DAG Model Comparison',
        'func': run_dag_model_comparison,
        'description': 'Compare alternative causal DAG structures',
    },
    'causal_bounds': {
        'name': 'Causal Bounds',
        'func': run_causal_bounds,
        'description': 'Partial identification bounds under confounding',
    },
    'mediation_sensitivity': {
        'name': 'Mediation Sensitivity',
        'func': run_mediation_sensitivity,
        'description': 'Sensitivity of mediation estimates to confounding',
    },
    'instrumental': {
        'name': 'Instrumental Variable Exploration',
        'func': run_instrumental_variable,
        'description': 'Exploratory IV-style analysis',
    },
}


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run causal inference suite.

    Parameters
    ----------
    analysis : str, optional
        Specific analysis to run. If None, runs all.
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict
        Results from all analyses.
    """
    print("\n" + "="*70)
    print("CAUSAL INFERENCE SUITE")
    print("="*70)

    # Load data
    df = load_master_dataset()

    # Standardize predictors
    for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        z_col = f'z_{col.replace("_total", "").replace("dass_", "dass_")}'
        if col in df.columns:
            df[z_col] = (df[col] - df[col].mean()) / df[col].std()

    # Rename for consistency
    rename_map = {
        'z_dass_depression': 'z_dass_dep',
        'z_dass_anxiety': 'z_dass_anx',
        'z_dass_stress': 'z_dass_str',
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure gender_male
    if 'gender_male' not in df.columns and 'gender' in df.columns:
        df['gender_male'] = df['gender'].str.lower().isin(['male', '남성', 'm'])

    results = {'analyses': {}}

    # Run analyses
    analyses_to_run = [analysis] if analysis else ANALYSES.keys()

    for analysis_name in analyses_to_run:
        if analysis_name not in ANALYSES:
            print(f"Unknown analysis: {analysis_name}")
            continue

        analysis_info = ANALYSES[analysis_name]
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running: {analysis_info['name']}")
            print(f"{'='*70}")

        try:
            result = analysis_info['func'](df, verbose=verbose)
            results['analyses'][analysis_name] = result
        except Exception as e:
            print(f"Error in {analysis_name}: {e}")
            results['analyses'][analysis_name] = {'error': str(e)}

    # Save summary
    summary_path = OUTPUT_DIR / 'causal_inference_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CAUSAL INFERENCE SUITE SUMMARY\n")
        f.write("="*60 + "\n\n")

        for name, result in results['analyses'].items():
            f.write(f"\n{ANALYSES[name]['name']}\n")
            f.write("-"*40 + "\n")

            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                # Write key findings
                if name == 'sensitivity' and 'outcomes' in result:
                    for outcome, data in result['outcomes'].items():
                        if 'dass_adjusted' in data:
                            f.write(f"{outcome}: E-value={data['dass_adjusted'].get('e_value_point', 'N/A'):.2f}\n")

                elif name == 'dag_comparison' and 'comparisons' in result:
                    f.write(f"Best model: {result['comparisons'].get('best_model', 'N/A')}\n")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return results


if __name__ == '__main__':
    run()
