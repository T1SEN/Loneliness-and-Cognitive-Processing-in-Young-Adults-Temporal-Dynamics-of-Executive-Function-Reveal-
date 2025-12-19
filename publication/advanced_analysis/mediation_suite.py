"""
Mediation Analysis Suite
========================

Unified mediation analyses testing DASS as mediator of UCLA → EF effects.

Pathway tested: UCLA → DASS → Executive Function

NOTE: In mediation analysis, DASS is the MEDIATOR (not covariate).
This tests whether loneliness effects operate through mood/anxiety.

Usage:
    python -m publication.advanced_analysis.mediation_suite
    python -m publication.advanced_analysis.mediation_suite --analysis dass_bootstrap

    from publication.advanced_analysis import mediation_suite
    mediation_suite.run()

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
from typing import Dict, Optional
import pandas as pd

# Publication preprocessing imports
from publication.preprocessing import (
    load_master_dataset,
    standardize_predictors,
    prepare_gender_variable,
)

# Shared utilities from _utils
from ._utils import BASE_OUTPUT, bootstrap_mediation

# Output directory
OUTPUT_DIR = BASE_OUTPUT / "mediation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mediation_data() -> pd.DataFrame:
    """Load and prepare data for mediation analysis."""
    master = load_master_dataset(merge_cognitive_summary=True)

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
