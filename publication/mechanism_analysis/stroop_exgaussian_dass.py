"""
Stroop Ex-Gaussian RT Decomposition (DASS-Controlled)
=====================================================

Decomposes RT distributions into Ex-Gaussian parameters:
- mu: Gaussian mean (routine processing speed)
- sigma: Gaussian SD (processing variability)
- tau: Exponential component (attentional lapses, slow tail)

Tests UCLA x Gender effects on each parameter for congruent vs incongruent trials.

Usage:
    python -m publication.mechanism_analysis -a stroop_exgaussian_dass
    python -m publication.mechanism_analysis -a stroop_exgaussian_dass --sub basic_exgaussian

    from publication.mechanism_analysis.stroop_exgaussian_dass import run
    run('basic_exgaussian')

Rationale:
- If loneliness affects mu -> slowed routine processing
- If loneliness affects sigma -> increased processing inconsistency
- If loneliness affects tau -> more frequent attentional lapses

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
    load_stroop_trials as load_stroop_trials_shared,
    find_interaction_term,
    standardize_predictors,
    prepare_gender_variable,
    DEFAULT_RT_MIN,
    STROOP_RT_MAX,
)
from ._utils import BASE_OUTPUT

np.random.seed(42)

# Output directory
OUTPUT_DIR = BASE_OUTPUT / "stroop_exgaussian"
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

def load_stroop_data() -> pd.DataFrame:
    """Load and prepare master dataset with gender and standardized predictors."""
    master = load_master_dataset(merge_cognitive_summary=True)
    master = prepare_gender_variable(master)
    master = standardize_predictors(master)
    return master


def load_stroop_trials() -> pd.DataFrame:
    """Load Stroop trial-level data using shared loader."""
    trials, _ = load_stroop_trials_shared(
        rt_min=200,  # Legacy analysis used 200 ms lower bound
        rt_max=STROOP_RT_MAX,
        drop_timeouts=True,
        require_correct_for_rt=False,
    )
    trials.columns = trials.columns.str.lower()

    # Normalize RT column
    if 'rt_ms' in trials.columns:
        trials['rt'] = trials['rt_ms'].fillna(trials.get('rt', np.nan))
    elif 'rt' not in trials.columns:
        raise ValueError("No RT column found in Stroop trials")

    # Normalize condition column
    condition_col = None
    for cand in ['type', 'condition', 'cond']:
        if cand in trials.columns:
            condition_col = cand
            break

    if condition_col is None:
        raise ValueError("No condition column found in Stroop trials")

    trials['condition'] = trials[condition_col].str.lower().str.strip()

    # Filter for valid conditions only
    valid_conditions = ['congruent', 'incongruent']
    trials = trials[trials['condition'].isin(valid_conditions)].copy()

    # Additional RT filtering
    trials = trials[
        (trials['rt'] > 200) &
        (trials['rt'] < STROOP_RT_MAX)
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

    # Optimize with L-BFGS-B
    try:
        result = minimize(
            neg_loglik,
            x0=[mu_init, sigma_init, tau_init],
            method='L-BFGS-B',
            bounds=[(100, 2000), (5, 500), (5, 1000)]
        )

        if result.success:
            mu, sigma, tau = result.x
            return {'mu': mu, 'sigma': sigma, 'tau': tau, 'n': len(rts)}
        else:
            return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}
    except Exception:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}


def compute_exgaussian_by_condition(group: pd.DataFrame) -> pd.Series:
    """Compute Ex-Gaussian parameters for each condition."""
    results = {}

    for condition in ['congruent', 'incongruent']:
        cond_trials = group[group['condition'] == condition]
        params = fit_exgaussian(cond_trials['rt'].values)

        results[f'{condition}_mu'] = params['mu']
        results[f'{condition}_sigma'] = params['sigma']
        results[f'{condition}_tau'] = params['tau']
        results[f'{condition}_n'] = params['n']

    return pd.Series(results)


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="basic_exgaussian",
    description="Fit ex-Gaussian to Stroop RTs by condition (congruent/incongruent)"
)
def analyze_basic_exgaussian(verbose: bool = True) -> pd.DataFrame:
    """
    Fit Ex-Gaussian parameters for each participant by condition.

    Returns DataFrame with mu, sigma, tau for congruent and incongruent conditions.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STROOP EX-GAUSSIAN FITTING")
        print("=" * 70)

    trials = load_stroop_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient Stroop trial data")
        return pd.DataFrame()

    if verbose:
        print(f"  Total trials: {len(trials)}")
        print(f"  Participants: {trials['participant_id'].nunique()}")
        print(f"  Congruent: {len(trials[trials['condition'] == 'congruent'])}")
        print(f"  Incongruent: {len(trials[trials['condition'] == 'incongruent'])}")
        print("\n  Fitting ex-Gaussian to each participant x condition...")

    # Fit per participant
    exgauss_df = trials.groupby('participant_id').apply(
        compute_exgaussian_by_condition
    ).reset_index()

    # Filter participants with valid fits
    valid_mask = (
        exgauss_df['congruent_mu'].notna() &
        exgauss_df['incongruent_mu'].notna()
    )
    exgauss_df = exgauss_df[valid_mask].copy()

    if len(exgauss_df) < 20:
        if verbose:
            print(f"  Only {len(exgauss_df)} participants with valid fits")
        return pd.DataFrame()

    if verbose:
        print(f"\n  Valid participants: {len(exgauss_df)}")
        print(f"\n  Congruent condition:")
        print(f"    mu: {exgauss_df['congruent_mu'].mean():.1f} ms (SD={exgauss_df['congruent_mu'].std():.1f})")
        print(f"    sigma: {exgauss_df['congruent_sigma'].mean():.1f} ms (SD={exgauss_df['congruent_sigma'].std():.1f})")
        print(f"    tau: {exgauss_df['congruent_tau'].mean():.1f} ms (SD={exgauss_df['congruent_tau'].std():.1f})")
        print(f"\n  Incongruent condition:")
        print(f"    mu: {exgauss_df['incongruent_mu'].mean():.1f} ms (SD={exgauss_df['incongruent_mu'].std():.1f})")
        print(f"    sigma: {exgauss_df['incongruent_sigma'].mean():.1f} ms (SD={exgauss_df['incongruent_sigma'].std():.1f})")
        print(f"    tau: {exgauss_df['incongruent_tau'].mean():.1f} ms (SD={exgauss_df['incongruent_tau'].std():.1f})")

    # Save
    output_file = OUTPUT_DIR / "stroop_exgaussian_parameters.csv"
    exgauss_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return exgauss_df


@register_analysis(
    name="interference_effects",
    description="Compute Stroop interference effects on ex-Gaussian parameters"
)
def analyze_interference_effects(verbose: bool = True) -> pd.DataFrame:
    """
    Compute interference effects (incongruent - congruent) for each Ex-Gaussian parameter.

    Interference effects indicate how conflict affects each RT component:
    - mu_interference: Slowing of routine processing during conflict
    - sigma_interference: Increased variability during conflict
    - tau_interference: More lapses during conflict
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STROOP INTERFERENCE EFFECTS ON EX-GAUSSIAN")
        print("=" * 70)

    # Load or compute basic parameters
    param_file = OUTPUT_DIR / "stroop_exgaussian_parameters.csv"

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

    # Compute interference effects
    exgauss_df['mu_interference'] = exgauss_df['incongruent_mu'] - exgauss_df['congruent_mu']
    exgauss_df['sigma_interference'] = exgauss_df['incongruent_sigma'] - exgauss_df['congruent_sigma']
    exgauss_df['tau_interference'] = exgauss_df['incongruent_tau'] - exgauss_df['congruent_tau']

    if verbose:
        print(f"\n  Interference Effects (N={len(exgauss_df)}):")
        print(f"    mu: {exgauss_df['mu_interference'].mean():.1f} ms (SD={exgauss_df['mu_interference'].std():.1f})")
        print(f"    sigma: {exgauss_df['sigma_interference'].mean():.1f} ms (SD={exgauss_df['sigma_interference'].std():.1f})")
        print(f"    tau: {exgauss_df['tau_interference'].mean():.1f} ms (SD={exgauss_df['tau_interference'].std():.1f})")

    # Select interference columns
    interference_df = exgauss_df[[
        'participant_id',
        'mu_interference', 'sigma_interference', 'tau_interference'
    ]].copy()

    # Save
    output_file = OUTPUT_DIR / "stroop_interference_effects.csv"
    interference_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_file}")

    return interference_df


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
        print("UCLA EFFECTS ON STROOP EX-GAUSSIAN (DASS-CONTROLLED)")
        print("=" * 70)

    # Load master data
    master = load_stroop_data()

    # Load parameters
    param_file = OUTPUT_DIR / "stroop_exgaussian_parameters.csv"

    if not param_file.exists():
        if verbose:
            print("  Parameters not found - computing now...")
        analyze_basic_exgaussian(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  Failed to compute parameters")
        return pd.DataFrame()

    params = pd.read_csv(param_file)

    # Compute interference effects if not present
    if 'mu_interference' not in params.columns:
        params['mu_interference'] = params['incongruent_mu'] - params['congruent_mu']
        params['sigma_interference'] = params['incongruent_sigma'] - params['congruent_sigma']
        params['tau_interference'] = params['incongruent_tau'] - params['congruent_tau']

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
        ('congruent_mu', 'Congruent mu'),
        ('congruent_sigma', 'Congruent sigma'),
        ('congruent_tau', 'Congruent tau'),
        ('incongruent_mu', 'Incongruent mu'),
        ('incongruent_sigma', 'Incongruent sigma'),
        ('incongruent_tau', 'Incongruent tau'),
        ('mu_interference', 'Interference mu'),
        ('sigma_interference', 'Interference sigma'),
        ('tau_interference', 'Interference tau'),
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
    regression_file = OUTPUT_DIR / "stroop_ucla_relationship.csv"
    correlation_file = OUTPUT_DIR / "stroop_gender_correlations.csv"

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
    Run Stroop Ex-Gaussian analyses.

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
        print("STROOP EX-GAUSSIAN DECOMPOSITION")
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
            'interference_effects',
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
        print("STROOP EX-GAUSSIAN ANALYSIS COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Stroop Ex-Gaussian Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stroop Ex-Gaussian Decomposition (DASS-Controlled)")
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
