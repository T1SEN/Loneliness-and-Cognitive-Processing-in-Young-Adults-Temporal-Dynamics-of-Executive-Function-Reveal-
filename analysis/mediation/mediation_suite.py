"""
Mediation Analysis Suite
========================

Unified mediation analyses testing DASS as mediator of UCLA → EF effects.

Consolidates:
- dass_mediation_bootstrapped.py
- mechanism_mediation_analysis.py
- mediation_dass_complete.py
- mediation_gender_pathways.py
- tau_moderated_mediation.py

Pathway tested: UCLA → DASS → Executive Function

NOTE: In mediation analysis, DASS is the MEDIATOR (not covariate).
This tests whether loneliness effects operate through mood/anxiety.

Usage:
    python -m analysis.mediation.mediation_suite
    python -m analysis.mediation.mediation_suite --analysis dass_bootstrap

    from analysis.mediation import mediation_suite
    mediation_suite.run()
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from analysis.preprocessing import load_master_dataset, ANALYSIS_OUTPUT_DIR
from analysis.utils.modeling import standardize_predictors
from analysis.preprocessing import prepare_gender_variable

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "mediation_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BOOTSTRAP MEDIATION FUNCTIONS
# =============================================================================

def bootstrap_mediation(
    df: pd.DataFrame,
    x_col: str,
    m_col: str,
    y_col: str,
    covariates: Optional[List[str]] = None,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Bootstrap mediation with optional covariate control.

    Tests pathway: X ?? M ?? Y controlling for covariates.
    """
    covariates = covariates or []
    cov_str = (" + " + " + ".join(covariates)) if covariates else ""

    formula_a = f"{m_col} ~ {x_col}{cov_str}"
    formula_b = f"{y_col} ~ {x_col} + {m_col}{cov_str}"
    formula_c = f"{y_col} ~ {x_col}{cov_str}"

    a_model = smf.ols(formula_a, data=df).fit()
    bc_model = smf.ols(formula_b, data=df).fit()
    c_model = smf.ols(formula_c, data=df).fit()

    a = a_model.params.get(x_col, np.nan)
    b = bc_model.params.get(m_col, np.nan)
    c_prime = bc_model.params.get(x_col, np.nan)
    c = c_model.params.get(x_col, np.nan)

    indirect = a * b
    indirect_effects = []
    alpha = (1 - ci) / 2
    n = len(df)
    rng = np.random.default_rng(seed)
    fail_count = 0

    for i in range(n_bootstrap):
        sample = df.sample(n=n, replace=True, random_state=int(rng.integers(0, 1e9)))
        try:
            a_boot = smf.ols(formula_a, data=sample).fit().params.get(x_col, np.nan)
            b_boot = smf.ols(formula_b, data=sample).fit().params.get(m_col, np.nan)
            if np.isfinite(a_boot) and np.isfinite(b_boot):
                indirect_effects.append(a_boot * b_boot)
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            if fail_count <= 3:
                warnings.warn(f"Bootstrap iteration {i} failed: {e}")
            continue

    # Warn if high failure rate
    if fail_count > n_bootstrap * 0.1:
        warnings.warn(f"High bootstrap failure rate: {fail_count}/{n_bootstrap} ({100*fail_count/n_bootstrap:.1f}%)")

    indirect_effects = np.array(indirect_effects)
    ci_low = np.percentile(indirect_effects, alpha * 100) if len(indirect_effects) else np.nan
    ci_high = np.percentile(indirect_effects, (1 - alpha) * 100) if len(indirect_effects) else np.nan

    significant = False
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        significant = (ci_low > 0) or (ci_high < 0)

    return {
        'a': a,
        'b': b,
        'c': c,
        'c_prime': c_prime,
        'indirect': indirect,
        'indirect_ci_low': ci_low,
        'indirect_ci_high': ci_high,
        'indirect_significant': significant,
        'proportion_mediated': indirect / c if c not in (0, np.nan) else np.nan,
        'n_bootstrap': len(indirect_effects),
        'covariates_used': covariates
    }



def sobel_test(a: float, b: float, se_a: float, se_b: float) -> Tuple[float, float]:
    """
    Sobel test for indirect effect significance.

    Returns (z, p-value)
    """
    se_indirect = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    z = (a * b) / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mediation_data() -> pd.DataFrame:
    """Load and prepare data for mediation analysis."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    # Ensure ucla_total
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize
    master = standardize_predictors(master)

    return master


# =============================================================================
# ANALYSIS 1: DASS BOOTSTRAP MEDIATION
# =============================================================================

def analyze_dass_mediation(verbose: bool = True) -> pd.DataFrame:
    """
    Bootstrap mediation: UCLA → DASS subscales → EF metrics.

    Tests each DASS subscale as potential mediator.
    """
    output_dir = OUTPUT_DIR / "dass_bootstrap"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[DASS MEDIATION] Testing UCLA → DASS → EF pathways...")

    df = load_mediation_data()

    mediators = ['dass_depression', 'dass_anxiety', 'dass_stress']
    outcomes = ['pe_rate', 'stroop_interference', 'prp_bottleneck']

    # Mapping from mediator name to its z-score column
    mediator_to_z = {
        'dass_depression': 'z_dass_dep',
        'dass_anxiety': 'z_dass_anx',
        'dass_stress': 'z_dass_str'
    }

    results = []

    for mediator in mediators:
        # CRITICAL FIX: When testing a DASS subscale as mediator,
        # exclude that subscale from covariates to avoid blocking the mediation path.
        # Include OTHER DASS subscales to control for their confounding effects.
        other_dass = [z for name, z in mediator_to_z.items() if name != mediator]
        covariates = ['z_age', 'gender_male'] + other_dass

        for outcome in outcomes:
            cols = ['ucla_total', mediator, outcome]
            subset = df.dropna(subset=cols + covariates)

            if len(subset) < 50:
                if verbose:
                    print(f"  [SKIP] {mediator} -> {outcome}: N={len(subset)} < 50")
                continue

            if verbose:
                print(f"  Testing: UCLA -> {mediator} -> {outcome} (N={len(subset)})")
                print(f"    Covariates: {covariates} (excluding {mediator_to_z[mediator]})")

            med_results = bootstrap_mediation(
                subset,
                x_col='ucla_total',
                m_col=mediator,
                y_col=outcome,
                covariates=covariates,
                n_bootstrap=5000
            )

            results.append({
                'predictor': 'UCLA',
                'mediator': mediator,
                'outcome': outcome,
                'n': len(subset),
                **med_results
            })

            if verbose:
                sig = '*' if med_results['indirect_significant'] else ''
                print(f"    Indirect: {med_results['indirect']:.4f} [{med_results['indirect_ci_low']:.4f}, {med_results['indirect_ci_high']:.4f}]{sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "dass_mediation_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        n_sig = results_df['indirect_significant'].sum()
        print(f"\n  Significant mediations: {n_sig}/{len(results_df)}")
        print(f"  Saved to: {output_dir}")

    return results_df


# =============================================================================
# ANALYSIS 2: GENDER-STRATIFIED MEDIATION
# =============================================================================

def analyze_gender_stratified(verbose: bool = True) -> pd.DataFrame:
    """
    Separate mediation analyses for males and females.
    """
    output_dir = OUTPUT_DIR / "gender_stratified"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[GENDER-STRATIFIED] Testing mediation by gender...")

    df = load_mediation_data()

    results = []

    # Mapping from mediator name to its z-score column
    mediator_to_z = {
        'dass_depression': 'z_dass_dep',
        'dass_anxiety': 'z_dass_anx',
        'dass_stress': 'z_dass_str'
    }

    for gender_val, gender_name in [(0, 'female'), (1, 'male')]:
        gender_df = df[df['gender_male'] == gender_val]

        if verbose:
            print(f"\n  {gender_name.upper()} (N={len(gender_df)})")

        for mediator in ['dass_depression', 'dass_anxiety', 'dass_stress']:
            # Exclude tested mediator's z-score from covariates
            other_dass = [z for name, z in mediator_to_z.items() if name != mediator]
            covariates = ['z_age'] + other_dass

            for outcome in ['pe_rate', 'stroop_interference']:
                cols = ['ucla_total', mediator, outcome]
                subset = gender_df.dropna(subset=cols + covariates)

                if len(subset) < 30:
                    continue

                med_results = bootstrap_mediation(
                    subset,
                    x_col='ucla_total',
                    m_col=mediator,
                    y_col=outcome,
                    covariates=covariates,
                    n_bootstrap=5000
                )

                results.append({
                    'gender': gender_name,
                    'mediator': mediator,
                    'outcome': outcome,
                    'n': len(subset),
                    **med_results
                })

                if verbose and med_results['indirect_significant']:
                    print(f"    * {mediator} -> {outcome}: indirect={med_results['indirect']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "gender_stratified_mediation.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"  Saved to: {output_dir}")

    return results_df


def analyze_moderated_mediation(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether mediation effects differ by gender (moderated mediation).
    """
    output_dir = OUTPUT_DIR / "moderated_mediation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[MODERATED MEDIATION] Testing gender moderation of indirect effects...")

    df = load_mediation_data()

    # Mapping from mediator name to its z-score column
    mediator_to_z = {
        'dass_depression': 'z_dass_dep',
        'dass_anxiety': 'z_dass_anx',
        'dass_stress': 'z_dass_str'
    }

    results = []

    for mediator in ['dass_depression', 'dass_anxiety', 'dass_stress']:
        # Exclude tested mediator's z-score from covariates
        other_dass = [z for name, z in mediator_to_z.items() if name != mediator]
        covariates = ['z_age'] + other_dass

        for outcome in ['pe_rate', 'stroop_interference']:
            cols = ['ucla_total', mediator, outcome, 'gender_male']
            subset = df.dropna(subset=cols + covariates)

            if len(subset) < 60:
                continue

            male_df = subset[subset['gender_male'] == 1]
            female_df = subset[subset['gender_male'] == 0]

            if len(male_df) < 25 or len(female_df) < 25:
                continue

            male_med = bootstrap_mediation(
                male_df,
                x_col='ucla_total',
                m_col=mediator,
                y_col=outcome,
                covariates=covariates,
                n_bootstrap=5000
            )

            female_med = bootstrap_mediation(
                female_df,
                x_col='ucla_total',
                m_col=mediator,
                y_col=outcome,
                covariates=covariates,
                n_bootstrap=5000
            )

            diff_indirect = male_med['indirect'] - female_med['indirect']

            results.append({
                'mediator': mediator,
                'outcome': outcome,
                'n_male': len(male_df),
                'n_female': len(female_df),
                'indirect_male': male_med['indirect'],
                'indirect_female': female_med['indirect'],
                'indirect_diff': diff_indirect,
                'male_significant': male_med['indirect_significant'],
                'female_significant': female_med['indirect_significant']
            })

            if verbose:
                print(f"  {mediator} -> {outcome}:")
                print(f"    Male indirect: {male_med['indirect']:.4f} {'*' if male_med['indirect_significant'] else ''}")
                print(f"    Female indirect: {female_med['indirect']:.4f} {'*' if female_med['indirect_significant'] else ''}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "moderated_mediation_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"  Saved to: {output_dir}")

    return results_df


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

ANALYSES = {
    'dass_bootstrap': ('DASS bootstrap mediation', analyze_dass_mediation),
    'gender_stratified': ('Gender-stratified mediation', analyze_gender_stratified),
    'moderated_mediation': ('Moderated mediation (gender)', analyze_moderated_mediation),
}


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run mediation analyses."""
    if verbose:
        print("=" * 70)
        print("MEDIATION ANALYSIS SUITE")
        print("=" * 70)
        print("\nPathway tested: UCLA → DASS → Executive Function")
        print("NOTE: DASS is MEDIATOR (not covariate) in these analyses")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        desc, func = ANALYSES[analysis]
        results[analysis] = func(verbose=verbose)
    else:
        for name, (desc, func) in ANALYSES.items():
            try:
                results[name] = func(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mediation Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()
    run(analysis=args.analysis, verbose=not args.quiet)
