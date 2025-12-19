"""
PRP Ex-Gaussian RT Decomposition (DASS-Controlled)
==================================================

Decomposes T2 RT distributions into Ex-Gaussian parameters:
- mu: Gaussian mean (routine processing speed)
- sigma: Gaussian SD (processing variability)
- tau: Exponential component (attentional lapses, slow tail)

Tests UCLA x Gender effects on each parameter across SOA conditions (short/long).

Usage:
    python -m publication.mechanism_analysis -a prp_exgaussian_dass
    python -m publication.mechanism_analysis -a prp_exgaussian_dass --sub basic_exgaussian

    from publication.mechanism_analysis.prp_exgaussian_dass import run
    run('basic_exgaussian')

Key Question:
- Is PRP bottleneck effect (short vs long SOA) driven by mu, sigma, or tau changes?
- Does loneliness amplify specific Ex-Gaussian components under dual-task load?

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import exponnorm, pearsonr
from scipy.optimize import minimize
import statsmodels.formula.api as smf

# Publication imports
from publication.preprocessing import (
    load_master_dataset,
    load_prp_trials as load_prp_trials_shared,
    find_interaction_term,
    standardize_predictors,
    prepare_gender_variable,
    DEFAULT_RT_MIN,
    PRP_RT_MAX,
)
from ._utils import BASE_OUTPUT

np.random.seed(42)

# Output directory
OUTPUT_DIR = BASE_OUTPUT / "prp_exgaussian"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

def load_prp_data() -> pd.DataFrame:
    """Load and prepare master dataset with gender and standardized predictors."""
    master = load_master_dataset(task="prp", merge_cognitive_summary=True)
    master = prepare_gender_variable(master)
    master = standardize_predictors(master)
    return master


def load_prp_trials() -> pd.DataFrame:
    """Load PRP trial-level data using shared loader (short/long SOA only)."""
    trials, _ = load_prp_trials_shared(
        rt_min=200,  # Match legacy PRP ex-Gaussian preprocessing
        rt_max=PRP_RT_MAX,
        require_t1_correct=False,
        require_t2_correct_for_rt=False,
        enforce_short_long_only=True,  # Only short and long SOA
        drop_timeouts=True,
    )
    trials.columns = trials.columns.str.lower()

    # Normalize RT column
    if 't2_rt_ms' in trials.columns:
        trials['t2_rt'] = trials['t2_rt_ms'].fillna(trials.get('t2_rt', np.nan))
    elif 't2_rt' not in trials.columns:
        raise ValueError("No T2 RT column found in PRP trials")

    # Normalize SOA bin column
    if 'soa_bin' not in trials.columns:
        # Create SOA bin from SOA values
        soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in trials.columns else 'soa'
        if soa_col in trials.columns:
            def categorize_soa(soa):
                if soa <= 150:
                    return 'short'
                elif soa >= 1200:
                    return 'long'
                else:
                    return 'other'
            trials['soa_bin'] = trials[soa_col].apply(categorize_soa)
        else:
            raise ValueError("No SOA column found in PRP trials")

    # Filter for short/long only
    trials = trials[trials['soa_bin'].isin(['short', 'long'])].copy()

    # Additional RT filtering
    trials = trials[
        (trials['t2_rt'] > 200) &
        (trials['t2_rt'] < PRP_RT_MAX)
    ].copy()

    # Sort
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'trial']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols).reset_index(drop=True)

    return trials


# =============================================================================
# EX-GAUSSIAN FITTING
# =============================================================================

def fit_exgaussian(rts: np.ndarray, min_trials: int = 20) -> Dict:
    """
    Fit Ex-Gaussian distribution to RT data using MLE.

    Ex-Gaussian = Gaussian(mu, sigma) + Exponential(tau)
    scipy.stats.exponnorm uses K = tau/sigma parameterization

    Parameters
    ----------
    rts : array-like
        Reaction time data in milliseconds
    min_trials : int
        Minimum number of trials required for fitting

    Returns
    -------
    dict
        Dictionary with 'mu', 'sigma', 'tau', 'n' keys
    """
    rts = np.array(rts)
    rts = rts[~np.isnan(rts)]

    if len(rts) < min_trials:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}

    # Method of moments initial estimates
    m = np.mean(rts)
    s = np.std(rts)

    if s <= 0:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}

    skew = np.mean(((rts - m) / s) ** 3)

    # Initial tau estimate from skewness
    tau_init = max(10, (abs(skew) / 2) ** (1/3) * s)
    mu_init = max(100, m - tau_init)
    sigma_init = max(10, np.sqrt(max(0, s**2 - tau_init**2)))

    def neg_loglik(params):
        mu, sigma, tau = params
        if sigma <= 0 or tau <= 0:
            return 1e10
        K = tau / sigma
        try:
            loglik = np.sum(exponnorm.logpdf(rts, K, loc=mu, scale=sigma))
            if not np.isfinite(loglik):
                return 1e10
            return -loglik
        except Exception:
            return 1e10

    # Optimize with L-BFGS-B (wider bounds for PRP)
    try:
        result = minimize(
            neg_loglik,
            x0=[mu_init, sigma_init, tau_init],
            method='L-BFGS-B',
            bounds=[(100, 4000), (5, 1000), (5, 2000)]
        )

        if result.success:
            mu, sigma, tau = result.x
            return {'mu': mu, 'sigma': sigma, 'tau': tau, 'n': len(rts)}
        else:
            return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}
    except Exception:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}


def compute_exgaussian_by_soa(group: pd.DataFrame) -> pd.Series:
    """Compute Ex-Gaussian parameters for each SOA condition."""
    results = {}

    for soa_cat in ['short', 'long']:
        soa_trials = group[group['soa_bin'] == soa_cat]
        params = fit_exgaussian(soa_trials['t2_rt'].values)

        results[f'{soa_cat}_mu'] = params['mu']
        results[f'{soa_cat}_sigma'] = params['sigma']
        results[f'{soa_cat}_tau'] = params['tau']
        results[f'{soa_cat}_n'] = params['n']

    # Also fit overall (collapsed across SOA)
    params_overall = fit_exgaussian(group['t2_rt'].values)
    results['overall_mu'] = params_overall['mu']
    results['overall_sigma'] = params_overall['sigma']
    results['overall_tau'] = params_overall['tau']
    results['overall_n'] = params_overall['n']

    return pd.Series(results)


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="basic_exgaussian",
    description="Fit ex-Gaussian to PRP T2 RTs by SOA (short/long only)"
)
def analyze_basic_exgaussian(verbose: bool = True) -> pd.DataFrame:
    """
    Fit Ex-Gaussian parameters for each participant by SOA condition.

    Returns DataFrame with mu, sigma, tau for short, long, and overall conditions.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PRP EX-GAUSSIAN FITTING")
        print("=" * 70)

    trials = load_prp_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient PRP trial data")
        return pd.DataFrame()

    if verbose:
        print(f"  Total trials: {len(trials)}")
        print(f"  Participants: {trials['participant_id'].nunique()}")
        print(f"  Short SOA: {len(trials[trials['soa_bin'] == 'short'])}")
        print(f"  Long SOA: {len(trials[trials['soa_bin'] == 'long'])}")
        print("\n  Fitting ex-Gaussian to each participant x SOA...")

    # Fit per participant
    exgauss_df = trials.groupby('participant_id').apply(
        compute_exgaussian_by_soa
    ).reset_index()

    # Filter participants with valid fits
    valid_mask = (
        exgauss_df['short_mu'].notna() &
        exgauss_df['long_mu'].notna()
    )
    exgauss_df = exgauss_df[valid_mask].copy()

    if len(exgauss_df) < 20:
        if verbose:
            print(f"  Only {len(exgauss_df)} participants with valid fits")
        return pd.DataFrame()

    if verbose:
        print(f"\n  Valid participants: {len(exgauss_df)}")
        print(f"\n  Short SOA (<=150ms) - Bottleneck ACTIVE:")
        print(f"    mu: {exgauss_df['short_mu'].mean():.1f} ms (SD={exgauss_df['short_mu'].std():.1f})")
        print(f"    sigma: {exgauss_df['short_sigma'].mean():.1f} ms (SD={exgauss_df['short_sigma'].std():.1f})")
        print(f"    tau: {exgauss_df['short_tau'].mean():.1f} ms (SD={exgauss_df['short_tau'].std():.1f})")
        print(f"\n  Long SOA (>=1200ms) - Bottleneck MINIMAL:")
        print(f"    mu: {exgauss_df['long_mu'].mean():.1f} ms (SD={exgauss_df['long_mu'].std():.1f})")
        print(f"    sigma: {exgauss_df['long_sigma'].mean():.1f} ms (SD={exgauss_df['long_sigma'].std():.1f})")
        print(f"    tau: {exgauss_df['long_tau'].mean():.1f} ms (SD={exgauss_df['long_tau'].std():.1f})")

    # Save
    output_file = OUTPUT_DIR / "prp_exgaussian_parameters.csv"
    exgauss_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return exgauss_df


@register_analysis(
    name="bottleneck_effects",
    description="Compute PRP bottleneck effects on ex-Gaussian parameters"
)
def analyze_bottleneck_effects(verbose: bool = True) -> pd.DataFrame:
    """
    Compute bottleneck effects (short - long SOA) for each Ex-Gaussian parameter.

    Bottleneck effects indicate how dual-task load affects each RT component:
    - mu_bottleneck: Slowing of routine processing under load
    - sigma_bottleneck: Increased variability under load
    - tau_bottleneck: More lapses under dual-task load
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PRP BOTTLENECK EFFECTS ON EX-GAUSSIAN")
        print("=" * 70)

    # Load or compute basic parameters
    param_file = OUTPUT_DIR / "prp_exgaussian_parameters.csv"

    if not param_file.exists():
        if verbose:
            print("  Parameters not found - computing now...")
        analyze_basic_exgaussian(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  Failed to compute parameters")
        return pd.DataFrame()

    exgauss_df = pd.read_csv(param_file)

    if len(exgauss_df) < 20:
        if verbose:
            print(f"  Insufficient data (N={len(exgauss_df)})")
        return pd.DataFrame()

    # Compute bottleneck effects (short - long)
    exgauss_df['mu_bottleneck'] = exgauss_df['short_mu'] - exgauss_df['long_mu']
    exgauss_df['sigma_bottleneck'] = exgauss_df['short_sigma'] - exgauss_df['long_sigma']
    exgauss_df['tau_bottleneck'] = exgauss_df['short_tau'] - exgauss_df['long_tau']

    if verbose:
        print(f"\n  Bottleneck Effects (short - long SOA, N={len(exgauss_df)}):")
        print(f"    mu: {exgauss_df['mu_bottleneck'].mean():.1f} ms (SD={exgauss_df['mu_bottleneck'].std():.1f})")
        print(f"    sigma: {exgauss_df['sigma_bottleneck'].mean():.1f} ms (SD={exgauss_df['sigma_bottleneck'].std():.1f})")
        print(f"    tau: {exgauss_df['tau_bottleneck'].mean():.1f} ms (SD={exgauss_df['tau_bottleneck'].std():.1f})")

        print("\n  Interpretation:")
        print("    Positive mu_bottleneck -> Slowed routine processing at short SOA")
        print("    Positive sigma_bottleneck -> Increased variability at short SOA")
        print("    Positive tau_bottleneck -> More lapses at short SOA")

    # Select bottleneck columns
    bottleneck_df = exgauss_df[[
        'participant_id',
        'mu_bottleneck', 'sigma_bottleneck', 'tau_bottleneck'
    ]].copy()

    # Save
    output_file = OUTPUT_DIR / "prp_bottleneck_effects.csv"
    bottleneck_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return bottleneck_df


@register_analysis(
    name="ucla_relationship",
    description="UCLA effects on ex-Gaussian parameters (DASS-controlled + gender-stratified)"
)
def analyze_ucla_relationship(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether loneliness relates to Ex-Gaussian parameters.

    All regressions use DASS control formula per CLAUDE.md:
    param ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age

    Also computes gender-stratified correlations.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECTS ON PRP EX-GAUSSIAN (DASS-CONTROLLED)")
        print("=" * 70)

    # Load master data
    master = load_prp_data()

    # Load parameters
    param_file = OUTPUT_DIR / "prp_exgaussian_parameters.csv"

    if not param_file.exists():
        if verbose:
            print("  Parameters not found - computing now...")
        analyze_basic_exgaussian(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  Failed to compute parameters")
        return pd.DataFrame()

    params = pd.read_csv(param_file)

    # Compute bottleneck effects if not present
    if 'mu_bottleneck' not in params.columns:
        params['mu_bottleneck'] = params['short_mu'] - params['long_mu']
        params['sigma_bottleneck'] = params['short_sigma'] - params['long_sigma']
        params['tau_bottleneck'] = params['short_tau'] - params['long_tau']

    # Merge
    merged = master.merge(params, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient merged data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  N = {len(merged)}")

    all_results = []

    # Parameters to test
    test_params = [
        ('overall_mu', 'Overall mu'),
        ('overall_sigma', 'Overall sigma'),
        ('overall_tau', 'Overall tau'),
        ('short_mu', 'Short SOA mu'),
        ('short_sigma', 'Short SOA sigma'),
        ('short_tau', 'Short SOA tau'),
        ('long_mu', 'Long SOA mu'),
        ('long_sigma', 'Long SOA sigma'),
        ('long_tau', 'Long SOA tau'),
        ('mu_bottleneck', 'Bottleneck mu'),
        ('sigma_bottleneck', 'Bottleneck sigma'),
        ('tau_bottleneck', 'Bottleneck tau'),
    ]

    if verbose:
        print("\n  DASS-Controlled Regressions:")
        print("  " + "-" * 60)

    for param_col, param_name in test_params:
        if param_col not in merged.columns:
            continue

        try:
            # DASS-controlled regression
            formula = f"{param_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged.dropna(subset=[param_col])).fit(cov_type='HC3')

            result = {
                'parameter': param_name,
                'param_col': param_col,
                'n': int(model.nobs),
                'r_squared': model.rsquared,
            }

            # UCLA main effect
            if 'z_ucla' in model.params:
                result['beta_ucla'] = model.params['z_ucla']
                result['se_ucla'] = model.bse['z_ucla']
                result['p_ucla'] = model.pvalues['z_ucla']

                if verbose:
                    sig = "*" if result['p_ucla'] < 0.05 else ""
                    print(f"    {param_name}: UCLA beta={result['beta_ucla']:.3f}, p={result['p_ucla']:.4f}{sig}")

            # UCLA x Gender interaction
            int_term = find_interaction_term(model.params.index)
            if int_term:
                result['beta_interaction'] = model.params[int_term]
                result['se_interaction'] = model.bse[int_term]
                result['p_interaction'] = model.pvalues[int_term]

                if result['p_interaction'] < 0.05 and verbose:
                    print(f"      UCLA x Gender: beta={result['beta_interaction']:.3f}, p={result['p_interaction']:.4f}*")

            all_results.append(result)

        except Exception as e:
            if verbose:
                print(f"    {param_name}: Error - {e}")

    # Gender-stratified correlations
    if verbose:
        print("\n  Gender-Stratified Correlations (UCLA vs parameters):")
        print("  " + "-" * 60)

    corr_results = []
    for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
        gender_data = merged[merged['gender_male'] == gender_val]

        if len(gender_data) < 10:
            continue

        for param_col, param_name in test_params:
            if param_col not in gender_data.columns:
                continue

            valid = gender_data.dropna(subset=['z_ucla', param_col])
            if len(valid) < 10:
                continue

            r, p = pearsonr(valid['z_ucla'], valid[param_col])

            corr_results.append({
                'gender': gender_label,
                'parameter': param_name,
                'n': len(valid),
                'r': r,
                'p': p,
            })

            if p < 0.10 and verbose:
                sig = "*" if p < 0.05 else "+"
                print(f"    {gender_label} - {param_name}: r={r:.3f}, p={p:.4f}{sig}")

    # Combine results
    regression_df = pd.DataFrame(all_results)
    correlation_df = pd.DataFrame(corr_results)

    # Save
    regression_file = OUTPUT_DIR / "prp_ucla_relationship.csv"
    correlation_file = OUTPUT_DIR / "prp_gender_correlations.csv"

    regression_df.to_csv(regression_file, index=False, encoding='utf-8-sig')
    correlation_df.to_csv(correlation_file, index=False, encoding='utf-8-sig')

    # Report significant effects
    sig_main = regression_df[regression_df.get('p_ucla', 1) < 0.05] if 'p_ucla' in regression_df.columns else pd.DataFrame()
    sig_int = regression_df[regression_df.get('p_interaction', 1) < 0.05] if 'p_interaction' in regression_df.columns else pd.DataFrame()

    if verbose:
        if len(sig_main) > 0:
            print("\n  Significant UCLA Main Effects (p < 0.05):")
            for _, row in sig_main.iterrows():
                print(f"    {row['parameter']}: beta={row['beta_ucla']:.3f}, p={row['p_ucla']:.4f}")

        if len(sig_int) > 0:
            print("\n  Significant UCLA x Gender Interactions (p < 0.05):")
            for _, row in sig_int.iterrows():
                print(f"    {row['parameter']}: beta={row['beta_interaction']:.3f}, p={row['p_interaction']:.4f}")

        print(f"\n  Output: {regression_file}")
        print(f"  Output: {correlation_file}")

    return regression_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run PRP Ex-Gaussian analyses.

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
    if verbose:
        print("=" * 70)
        print("PRP EX-GAUSSIAN DECOMPOSITION")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        # Run all analyses in order
        analysis_order = [
            'basic_exgaussian',
            'bottleneck_effects',
            'ucla_relationship'
        ]

        for name in analysis_order:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")
                    import traceback
                    traceback.print_exc()

    if verbose:
        print("\n" + "=" * 70)
        print("PRP EX-GAUSSIAN ANALYSIS COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable PRP Ex-Gaussian Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PRP Ex-Gaussian Decomposition (DASS-Controlled)")
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
