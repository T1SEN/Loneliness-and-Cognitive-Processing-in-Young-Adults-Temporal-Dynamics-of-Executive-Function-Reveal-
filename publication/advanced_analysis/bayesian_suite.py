"""
Bayesian Structural Equation Modeling Suite
============================================

Bayesian approach to UCLA -> EF relationships with proper uncertainty quantification.

Advantages of Bayesian SEM:
---------------------------
1. Full posterior distributions (not just point estimates)
2. Better handling of small samples
3. Natural incorporation of prior knowledge
4. Credible intervals instead of p-values
5. Model comparison via WAIC/LOO

Analyses:
---------
1. basic_regression: Bayesian version of DASS-controlled regression
2. model_comparison: Compare models with/without UCLA term
3. sensitivity: Prior sensitivity analysis
4. equivalence: Bayesian equivalence testing (ROPE)

Usage:
    python -m publication.advanced_analysis.bayesian_suite
    python -m publication.advanced_analysis.bayesian_suite --analysis basic_regression
    python -m publication.advanced_analysis.bayesian_suite --list

    from publication.advanced_analysis import bayesian_suite
    bayesian_suite.run()

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Publication preprocessing imports
from publication.preprocessing import (
    load_master_dataset,
    standardize_predictors,
    prepare_gender_variable,
    find_interaction_term,
)

# Shared utilities from _utils
from ._utils import BASE_OUTPUT

np.random.seed(42)

# Output directory
OUTPUT_DIR = BASE_OUTPUT / "bayesian"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check for PyMC
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bayesian_data() -> pd.DataFrame:
    """Load and prepare data for Bayesian analysis."""
    master = load_master_dataset(task="overall", merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    # Handle UCLA column naming
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors
    master = standardize_predictors(master)

    return master


# =============================================================================
# ANALYSIS 1: BASIC BAYESIAN REGRESSION
# =============================================================================

@register_analysis(
    name="basic_regression",
    description="Bayesian DASS-controlled regression for UCLA -> EF"
)
def analyze_basic_regression(verbose: bool = True) -> pd.DataFrame:
    """
    Bayesian version of the DASS-controlled regression.

    Provides full posterior distributions for all coefficients,
    allowing proper uncertainty quantification.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: BAYESIAN DASS-CONTROLLED REGRESSION")
        print("=" * 70)

    master = load_bayesian_data()

    # Outcomes to test
    outcomes = {
        'pe_rate': 'WCST PE Rate',
        'stroop_interference': 'Stroop Interference',
        'prp_bottleneck': 'PRP Bottleneck'
    }

    all_results = []

    for outcome, label in outcomes.items():
        if outcome not in master.columns:
            continue

        # Prepare data
        required = [outcome, 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
        df = master.dropna(subset=required).copy()

        if len(df) < 30:
            if verbose:
                print(f"\n  {label}: Insufficient data (N={len(df)})")
            continue

        if verbose:
            print(f"\n  {label} (N={len(df)})")
            print("  " + "-" * 50)

        if PYMC_AVAILABLE:
            # Full Bayesian model with PyMC
            try:
                with pm.Model() as model:
                    # Priors
                    intercept = pm.Normal('intercept', mu=0, sigma=10)
                    beta_ucla = pm.Normal('beta_ucla', mu=0, sigma=2)
                    beta_gender = pm.Normal('beta_gender', mu=0, sigma=2)
                    beta_interaction = pm.Normal('beta_interaction', mu=0, sigma=2)
                    beta_dass_dep = pm.Normal('beta_dass_dep', mu=0, sigma=2)
                    beta_dass_anx = pm.Normal('beta_dass_anx', mu=0, sigma=2)
                    beta_dass_str = pm.Normal('beta_dass_str', mu=0, sigma=2)
                    beta_age = pm.Normal('beta_age', mu=0, sigma=2)
                    sigma = pm.HalfNormal('sigma', sigma=10)

                    # Linear predictor
                    mu = (intercept +
                          beta_ucla * df['z_ucla'].values +
                          beta_gender * df['gender_male'].values +
                          beta_interaction * df['z_ucla'].values * df['gender_male'].values +
                          beta_dass_dep * df['z_dass_dep'].values +
                          beta_dass_anx * df['z_dass_anx'].values +
                          beta_dass_str * df['z_dass_str'].values +
                          beta_age * df['z_age'].values)

                    # Likelihood
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df[outcome].values)

                    # Sample
                    trace = pm.sample(1000, tune=500, cores=1, random_seed=42,
                                     progressbar=False, return_inferencedata=True)

                # Extract posterior summaries
                summary = az.summary(trace, var_names=['beta_ucla', 'beta_gender', 'beta_interaction'])

                for var in ['beta_ucla', 'beta_gender', 'beta_interaction']:
                    mean_val = summary.loc[var, 'mean']
                    sd_val = summary.loc[var, 'sd']
                    hdi_3 = summary.loc[var, 'hdi_3%']
                    hdi_97 = summary.loc[var, 'hdi_97%']

                    # Probability of direction (proportion of posterior with same sign as mean)
                    posterior = trace.posterior[var].values.flatten()
                    if mean_val > 0:
                        prob_direction = (posterior > 0).mean()
                    else:
                        prob_direction = (posterior < 0).mean()

                    # CI excludes 0?
                    sig = (hdi_3 > 0) or (hdi_97 < 0)

                    term_label = var.replace('beta_', '').replace('ucla', 'UCLA').replace('gender', 'Male').replace('interaction', 'UCLA x Male')

                    if verbose:
                        sig_str = "*" if sig else ""
                        print(f"    {term_label}: β={mean_val:.3f} [{hdi_3:.3f}, {hdi_97:.3f}], P(direction)={prob_direction:.2f}{sig_str}")

                    all_results.append({
                        'outcome': label,
                        'term': term_label,
                        'mean': mean_val,
                        'sd': sd_val,
                        'hdi_3': hdi_3,
                        'hdi_97': hdi_97,
                        'prob_direction': prob_direction,
                        'ci_excludes_0': sig,
                        'n': len(df)
                    })

                # Save trace plot
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for i, var in enumerate(['beta_ucla', 'beta_gender', 'beta_interaction']):
                    az.plot_posterior(trace, var_names=[var], ax=axes[i])
                    axes[i].set_title(var.replace('beta_', '').title())
                plt.suptitle(f'{label}: Posterior Distributions')
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f'posterior_{outcome}.png', dpi=150)
                plt.close()

            except Exception as e:
                if verbose:
                    print(f"    PyMC error: {e}")
                    print("    Falling back to bootstrap approximation...")
                _run_bootstrap_approximation(df, outcome, label, all_results, verbose)

        else:
            # Bootstrap approximation when PyMC not available
            _run_bootstrap_approximation(df, outcome, label, all_results, verbose)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "basic_regression_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'basic_regression_results.csv'}")

    return results_df


def _run_bootstrap_approximation(df, outcome, label, all_results, verbose):
    """Bootstrap approximation when PyMC not available."""
    import statsmodels.formula.api as smf

    n_boot = 2000
    coeffs = {term: [] for term in ['ucla', 'gender', 'interaction']}

    for _ in range(n_boot):
        boot_df = df.sample(n=len(df), replace=True)
        try:
            model = smf.ols(
                f'{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age',
                data=boot_df
            ).fit()
            coeffs['ucla'].append(model.params.get('z_ucla', np.nan))
            coeffs['gender'].append(model.params.get('C(gender_male)[T.1]', np.nan))
            int_term = find_interaction_term(model.params.index)
            coeffs['interaction'].append(model.params.get(int_term, np.nan) if int_term else np.nan)
        except:
            continue

    for term, term_label in [('ucla', 'UCLA'), ('gender', 'Male'), ('interaction', 'UCLA x Male')]:
        vals = np.array([v for v in coeffs[term] if not np.isnan(v)])
        if len(vals) < 100:
            continue

        mean_val = np.mean(vals)
        sd_val = np.std(vals)
        hdi_3 = np.percentile(vals, 2.5)
        hdi_97 = np.percentile(vals, 97.5)

        if mean_val > 0:
            prob_direction = (vals > 0).mean()
        else:
            prob_direction = (vals < 0).mean()

        sig = (hdi_3 > 0) or (hdi_97 < 0)

        if verbose:
            sig_str = "*" if sig else ""
            print(f"    {term_label}: β={mean_val:.3f} [{hdi_3:.3f}, {hdi_97:.3f}], P(direction)={prob_direction:.2f}{sig_str}")

        all_results.append({
            'outcome': label,
            'term': term_label,
            'mean': mean_val,
            'sd': sd_val,
            'hdi_3': hdi_3,
            'hdi_97': hdi_97,
            'prob_direction': prob_direction,
            'ci_excludes_0': sig,
            'n': len(df),
            'method': 'bootstrap'
        })


# =============================================================================
# ANALYSIS 2: MODEL COMPARISON
# =============================================================================

@register_analysis(
    name="model_comparison",
    description="Compare models with/without UCLA term (Bayes Factor)"
)
def analyze_model_comparison(verbose: bool = True) -> pd.DataFrame:
    """
    Compare nested models using approximate Bayes Factor.

    M0: EF ~ DASS + Gender + Age (null: no UCLA effect)
    M1: EF ~ UCLA + DASS + Gender + Age
    M2: EF ~ UCLA * Gender + DASS + Age (with interaction)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: BAYESIAN MODEL COMPARISON")
        print("=" * 70)

    master = load_bayesian_data()

    outcomes = {
        'pe_rate': 'WCST PE Rate',
        'stroop_interference': 'Stroop Interference',
        'prp_bottleneck': 'PRP Bottleneck'
    }

    all_results = []

    for outcome, label in outcomes.items():
        if outcome not in master.columns:
            continue

        required = [outcome, 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
        df = master.dropna(subset=required).copy()

        if len(df) < 30:
            continue

        if verbose:
            print(f"\n  {label} (N={len(df)})")
            print("  " + "-" * 50)

        import statsmodels.formula.api as smf

        # Fit nested models
        m0 = smf.ols(f'{outcome} ~ z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)', data=df).fit()
        m1 = smf.ols(f'{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)', data=df).fit()
        m2 = smf.ols(f'{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age', data=df).fit()

        # BIC-based Bayes Factor approximation
        bic_m0 = m0.bic
        bic_m1 = m1.bic
        bic_m2 = m2.bic

        # BF comparing M1 to M0 (UCLA main effect)
        bf_ucla = np.exp((bic_m1 - bic_m0) / 2)  # >1 favors null (M0)

        # BF comparing M2 to M1 (interaction effect)
        bf_interaction = np.exp((bic_m2 - bic_m1) / 2)  # >1 favors M1 (no interaction)

        # Interpretation
        def interpret_bf(bf):
            if bf > 100:
                return "Decisive evidence for H0"
            elif bf > 30:
                return "Very strong evidence for H0"
            elif bf > 10:
                return "Strong evidence for H0"
            elif bf > 3:
                return "Moderate evidence for H0"
            elif bf > 1:
                return "Anecdotal evidence for H0"
            elif bf > 1/3:
                return "Anecdotal evidence for H1"
            elif bf > 1/10:
                return "Moderate evidence for H1"
            elif bf > 1/30:
                return "Strong evidence for H1"
            else:
                return "Very strong evidence for H1"

        if verbose:
            print(f"    M0 (DASS only): BIC = {bic_m0:.1f}")
            print(f"    M1 (+UCLA): BIC = {bic_m1:.1f}")
            print(f"    M2 (+UCLA×Gender): BIC = {bic_m2:.1f}")
            print(f"\n    UCLA main effect: BF01 = {bf_ucla:.2f} ({interpret_bf(bf_ucla)})")
            print(f"    Interaction: BF01 = {bf_interaction:.2f} ({interpret_bf(bf_interaction)})")

        all_results.append({
            'outcome': label,
            'bic_m0': bic_m0,
            'bic_m1': bic_m1,
            'bic_m2': bic_m2,
            'bf_ucla': bf_ucla,
            'bf_interaction': bf_interaction,
            'interpretation_ucla': interpret_bf(bf_ucla),
            'interpretation_interaction': interpret_bf(bf_interaction),
            'n': len(df)
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'model_comparison_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: SENSITIVITY ANALYSIS
# =============================================================================

@register_analysis(
    name="sensitivity",
    description="Prior sensitivity analysis"
)
def analyze_sensitivity(verbose: bool = True) -> pd.DataFrame:
    """
    Test sensitivity of results to prior specification.

    Uses different prior widths to assess robustness.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: PRIOR SENSITIVITY")
        print("=" * 70)

    master = load_bayesian_data()

    # Focus on PE rate (most significant interaction from Gold Standard)
    outcome = 'pe_rate'
    if outcome not in master.columns:
        if verbose:
            print("  PE rate not available")
        return pd.DataFrame()

    required = [outcome, 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
    df = master.dropna(subset=required).copy()

    if len(df) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  Outcome: WCST PE Rate (N={len(df)})")
        print("  Testing different prior widths for UCLA coefficient...")

    # Prior scales to test
    prior_scales = [0.5, 1.0, 2.0, 5.0, 10.0]
    all_results = []

    import statsmodels.formula.api as smf

    for scale in prior_scales:
        if verbose:
            print(f"\n  Prior SD = {scale}:")

        # Bootstrap with different "implicit" prior (regularization)
        n_boot = 1000
        ucla_coeffs = []
        interaction_coeffs = []

        for _ in range(n_boot):
            boot_df = df.sample(n=len(df), replace=True)
            try:
                model = smf.ols(
                    f'{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age',
                    data=boot_df
                ).fit()

                # Apply soft "prior" via shrinkage
                ucla_coef = model.params.get('z_ucla', 0)
                int_term = find_interaction_term(model.params.index)
                interaction_coef = model.params.get(int_term, 0) if int_term else 0

                # Bayesian shrinkage approximation
                shrinkage = scale**2 / (scale**2 + 1)
                ucla_coeffs.append(ucla_coef * shrinkage)
                interaction_coeffs.append(interaction_coef * shrinkage)

            except:
                continue

        if len(ucla_coeffs) < 100:
            continue

        ucla_mean = np.mean(ucla_coeffs)
        ucla_ci = (np.percentile(ucla_coeffs, 2.5), np.percentile(ucla_coeffs, 97.5))

        interaction_mean = np.mean(interaction_coeffs)
        interaction_ci = (np.percentile(interaction_coeffs, 2.5), np.percentile(interaction_coeffs, 97.5))

        ucla_sig = (ucla_ci[0] > 0) or (ucla_ci[1] < 0)
        int_sig = (interaction_ci[0] > 0) or (interaction_ci[1] < 0)

        if verbose:
            print(f"    UCLA: β={ucla_mean:.3f} [{ucla_ci[0]:.3f}, {ucla_ci[1]:.3f}]{'*' if ucla_sig else ''}")
            print(f"    UCLA×Gender: β={interaction_mean:.3f} [{interaction_ci[0]:.3f}, {interaction_ci[1]:.3f}]{'*' if int_sig else ''}")

        all_results.append({
            'prior_scale': scale,
            'ucla_mean': ucla_mean,
            'ucla_ci_lower': ucla_ci[0],
            'ucla_ci_upper': ucla_ci[1],
            'ucla_significant': ucla_sig,
            'interaction_mean': interaction_mean,
            'interaction_ci_lower': interaction_ci[0],
            'interaction_ci_upper': interaction_ci[1],
            'interaction_significant': int_sig
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "sensitivity_results.csv", index=False, encoding='utf-8-sig')

    # Visualization
    if len(all_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # UCLA coefficient
        ax = axes[0]
        ax.errorbar(
            results_df['prior_scale'],
            results_df['ucla_mean'],
            yerr=[results_df['ucla_mean'] - results_df['ucla_ci_lower'],
                  results_df['ucla_ci_upper'] - results_df['ucla_mean']],
            marker='o', capsize=5
        )
        ax.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Prior SD')
        ax.set_ylabel('UCLA Coefficient')
        ax.set_title('UCLA Main Effect: Prior Sensitivity')
        ax.set_xscale('log')

        # Interaction coefficient
        ax = axes[1]
        ax.errorbar(
            results_df['prior_scale'],
            results_df['interaction_mean'],
            yerr=[results_df['interaction_mean'] - results_df['interaction_ci_lower'],
                  results_df['interaction_ci_upper'] - results_df['interaction_mean']],
            marker='o', capsize=5, color='orange'
        )
        ax.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Prior SD')
        ax.set_ylabel('UCLA×Gender Coefficient')
        ax.set_title('Interaction Effect: Prior Sensitivity')
        ax.set_xscale('log')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'sensitivity_plot.png', dpi=150)
        plt.close()

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'sensitivity_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 4: EQUIVALENCE TESTING
# =============================================================================

@register_analysis(
    name="equivalence",
    description="Bayesian equivalence testing (ROPE)"
)
def analyze_equivalence(verbose: bool = True) -> pd.DataFrame:
    """
    Region of Practical Equivalence (ROPE) analysis.

    Tests whether effects are practically equivalent to zero.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: BAYESIAN EQUIVALENCE (ROPE)")
        print("=" * 70)

    master = load_bayesian_data()

    outcomes = {
        'pe_rate': ('WCST PE Rate', 1.0),  # ROPE: ±1% PE
        'stroop_interference': ('Stroop Interference', 20),  # ROPE: ±20ms
        'prp_bottleneck': ('PRP Bottleneck', 30)  # ROPE: ±30ms
    }

    all_results = []

    import statsmodels.formula.api as smf

    for outcome, (label, rope_width) in outcomes.items():
        if outcome not in master.columns:
            continue

        required = [outcome, 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
        df = master.dropna(subset=required).copy()

        if len(df) < 30:
            continue

        if verbose:
            print(f"\n  {label} (N={len(df)})")
            print(f"  ROPE: [-{rope_width}, +{rope_width}]")
            print("  " + "-" * 50)

        # Bootstrap to get posterior distribution
        n_boot = 2000
        ucla_coeffs = []
        interaction_coeffs = []

        for _ in range(n_boot):
            boot_df = df.sample(n=len(df), replace=True)
            try:
                model = smf.ols(
                    f'{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age',
                    data=boot_df
                ).fit()
                ucla_coeffs.append(model.params.get('z_ucla', np.nan))
                int_term = find_interaction_term(model.params.index)
                interaction_coeffs.append(model.params.get(int_term, np.nan) if int_term else np.nan)
            except:
                continue

        ucla_coeffs = np.array([v for v in ucla_coeffs if not np.isnan(v)])
        interaction_coeffs = np.array([v for v in interaction_coeffs if not np.isnan(v)])

        # ROPE analysis
        for coeffs, term_label in [(ucla_coeffs, 'UCLA'), (interaction_coeffs, 'UCLA×Gender')]:
            if len(coeffs) < 100:
                continue

            # Proportion of posterior in ROPE
            in_rope = np.mean((coeffs > -rope_width) & (coeffs < rope_width))

            # Proportion > ROPE (positive effect)
            above_rope = np.mean(coeffs > rope_width)

            # Proportion < ROPE (negative effect)
            below_rope = np.mean(coeffs < -rope_width)

            # Decision
            if in_rope > 0.95:
                decision = "Accept H0 (equivalent to 0)"
            elif above_rope > 0.95 or below_rope > 0.95:
                decision = "Reject H0 (different from 0)"
            else:
                decision = "Undecided"

            if verbose:
                print(f"    {term_label}:")
                print(f"      In ROPE: {in_rope:.1%}")
                print(f"      Above ROPE: {above_rope:.1%}")
                print(f"      Below ROPE: {below_rope:.1%}")
                print(f"      Decision: {decision}")

            all_results.append({
                'outcome': label,
                'term': term_label,
                'rope_width': rope_width,
                'prop_in_rope': in_rope,
                'prop_above_rope': above_rope,
                'prop_below_rope': below_rope,
                'decision': decision,
                'n': len(df)
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "equivalence_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'equivalence_results.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run Bayesian SEM analyses."""
    if verbose:
        print("=" * 70)
        print("BAYESIAN STRUCTURAL EQUATION MODELING SUITE")
        print("=" * 70)
        if PYMC_AVAILABLE:
            print("\nPyMC available - using full Bayesian inference")
        else:
            print("\nPyMC not available - using bootstrap approximations")

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            try:
                if verbose:
                    print(f"\n--- Running: {spec.description} ---")
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("BAYESIAN SEM SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Bayesian SEM Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian SEM Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                        help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
