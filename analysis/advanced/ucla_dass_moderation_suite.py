"""
UCLA × DASS Moderation Analysis Suite
=====================================

Tests whether DASS subscales (depression, anxiety, stress) moderate the
UCLA loneliness → EF relationship, rather than just controlling for them.

Key Research Questions:
1. Does the effect of loneliness on EF depend on depression severity?
2. Are there subgroups (e.g., lonely but NOT depressed) with different patterns?
3. Which DASS subscale shows the strongest moderation effect?

Theoretical Background:
    Current analyses treat DASS as a covariate, assuming UCLA and DASS have
    independent additive effects. However, they may interact:
    - Loneliness may only impair EF when combined with depression (synergistic)
    - Or loneliness effects may be masked by depression (suppression)
    - Identifying moderation could reveal intervention targets

Analyses:
- depression_moderation: UCLA × DASS_depression interaction on all EF outcomes
- anxiety_moderation: UCLA × DASS_anxiety interaction
- stress_moderation: UCLA × DASS_stress interaction
- johnson_neyman: At what DASS level does UCLA effect become significant?
- subgroup_analysis: Compare EF across UCLA × DASS quadrants
- summary: Integration across subscales

Usage:
    python -m analysis.advanced.ucla_dass_moderation_suite
    python -m analysis.advanced.ucla_dass_moderation_suite --analysis depression_moderation
    python -m analysis.advanced.ucla_dass_moderation_suite --list

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
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    find_interaction_term, apply_fdr_correction, safe_zscore
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "ucla_dass_moderation"
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
    source_script: str


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, source_script: str = "ucla_dass_moderation_suite.py"):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            source_script=source_script
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_moderation_data() -> pd.DataFrame:
    """Load and prepare master dataset for moderation analysis."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender
    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    return master


def get_ef_outcomes() -> List[Tuple[str, str, str]]:
    """Return list of (outcome_name, column_name, description) tuples."""
    return [
        ('wcst_pe', 'pe_rate', 'WCST Perseverative Error Rate'),
        ('wcst_accuracy', 'wcst_accuracy', 'WCST Overall Accuracy'),
        ('stroop_interference', 'stroop_effect', 'Stroop Interference Effect'),
        ('prp_bottleneck', 'prp_bottleneck', 'PRP Bottleneck'),
    ]


# =============================================================================
# CORE MODERATION FUNCTIONS
# =============================================================================

def run_moderation_analysis(
    df: pd.DataFrame,
    outcome: str,
    moderator: str,
    moderator_label: str
) -> Dict:
    """
    Run UCLA × DASS_subscale moderation analysis.

    Model: EF ~ z_ucla * z_dass_X + z_ucla * gender + other_dass + z_age
    """
    # Define other DASS controls
    all_dass = ['z_dass_dep', 'z_dass_anx', 'z_dass_str']
    other_dass = [d for d in all_dass if d != moderator]

    # Formula with interaction
    formula = f"{outcome} ~ z_ucla * {moderator} + z_ucla * C(gender_male) + {' + '.join(other_dass)} + z_age"

    # Prepare data
    model_vars = [outcome, 'z_ucla', moderator, 'gender_male', 'z_age'] + other_dass
    model_df = df.dropna(subset=[v for v in model_vars if v in df.columns])

    if len(model_df) < 50:
        return {'error': f'Insufficient data (N={len(model_df)})'}

    try:
        model = smf.ols(formula, data=model_df).fit(cov_type='HC3')

        # Find interaction term
        interaction_term = find_interaction_term(model.params.index, 'z_ucla', moderator)

        result = {
            'outcome': outcome,
            'moderator': moderator_label,
            'n': len(model_df),
            'r_squared': model.rsquared,
            'r_squared_adj': model.rsquared_adj,
            # Main effects
            'ucla_beta': model.params.get('z_ucla', np.nan),
            'ucla_se': model.bse.get('z_ucla', np.nan),
            'ucla_p': model.pvalues.get('z_ucla', np.nan),
            'moderator_beta': model.params.get(moderator, np.nan),
            'moderator_se': model.bse.get(moderator, np.nan),
            'moderator_p': model.pvalues.get(moderator, np.nan),
        }

        # Interaction effect
        if interaction_term:
            result['interaction_beta'] = model.params.get(interaction_term, np.nan)
            result['interaction_se'] = model.bse.get(interaction_term, np.nan)
            result['interaction_p'] = model.pvalues.get(interaction_term, np.nan)
        else:
            result['interaction_beta'] = np.nan
            result['interaction_se'] = np.nan
            result['interaction_p'] = np.nan

        # Gender interaction
        gender_int = find_interaction_term(model.params.index, 'z_ucla', 'C(gender_male)')
        if gender_int:
            result['gender_int_beta'] = model.params.get(gender_int, np.nan)
            result['gender_int_p'] = model.pvalues.get(gender_int, np.nan)

        return result

    except Exception as e:
        return {'error': str(e), 'outcome': outcome, 'moderator': moderator_label}


def johnson_neyman_analysis(
    df: pd.DataFrame,
    outcome: str,
    moderator: str
) -> Dict:
    """
    Johnson-Neyman analysis to find regions of significance.

    At what moderator level does the UCLA effect become significant?
    """
    formula = f"{outcome} ~ z_ucla * {moderator} + z_age"

    model_vars = [outcome, 'z_ucla', moderator, 'z_age']
    model_df = df.dropna(subset=[v for v in model_vars if v in df.columns])

    if len(model_df) < 50:
        return {'error': f'Insufficient data'}

    try:
        model = smf.ols(formula, data=model_df).fit(cov_type='HC3')

        # Get coefficients
        b_ucla = model.params.get('z_ucla', 0)
        interaction_term = find_interaction_term(model.params.index, 'z_ucla', moderator)
        b_interaction = model.params.get(interaction_term, 0) if interaction_term else 0

        # Get covariance matrix for CIs
        vcov = model.cov_params()

        # Simple slopes at different moderator levels
        moderator_values = np.linspace(
            model_df[moderator].min(),
            model_df[moderator].max(),
            50
        )

        simple_slopes = []
        significant_regions = []

        for mod_val in moderator_values:
            # Simple slope: b_ucla + b_interaction * moderator
            slope = b_ucla + b_interaction * mod_val

            # SE of simple slope (approximate)
            if interaction_term and 'z_ucla' in vcov.index and interaction_term in vcov.index:
                var_slope = (
                    vcov.loc['z_ucla', 'z_ucla'] +
                    2 * mod_val * vcov.loc['z_ucla', interaction_term] +
                    mod_val**2 * vcov.loc[interaction_term, interaction_term]
                )
                se_slope = np.sqrt(max(0, var_slope))
                t_stat = slope / se_slope if se_slope > 0 else 0
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), model.df_resid))
            else:
                se_slope = np.nan
                p_val = np.nan

            simple_slopes.append({
                'moderator_value': mod_val,
                'simple_slope': slope,
                'se': se_slope,
                'p_value': p_val,
                'significant': p_val < 0.05 if not np.isnan(p_val) else False
            })

            if p_val < 0.05:
                significant_regions.append(mod_val)

        return {
            'outcome': outcome,
            'moderator': moderator,
            'simple_slopes': simple_slopes,
            'significant_regions': significant_regions,
            'n_significant': len(significant_regions),
            'base_interaction_p': model.pvalues.get(interaction_term, np.nan) if interaction_term else np.nan
        }

    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# REGISTERED ANALYSES
# =============================================================================

@register_analysis(
    name="depression_moderation",
    description="Test UCLA × DASS_depression interaction on EF outcomes"
)
def depression_moderation_analysis():
    """Test whether depression moderates UCLA → EF relationship."""
    print("\n" + "="*70)
    print("UCLA × DASS DEPRESSION MODERATION ANALYSIS")
    print("="*70)

    df = load_moderation_data()
    outcomes = get_ef_outcomes()

    results = []
    for name, col, desc in outcomes:
        if col not in df.columns:
            print(f"  [SKIP] {name}: column {col} not found")
            continue

        result = run_moderation_analysis(df, col, 'z_dass_dep', 'Depression')
        result['outcome_name'] = name
        result['outcome_description'] = desc
        results.append(result)

        # Print result
        if 'error' not in result:
            int_p = result.get('interaction_p', np.nan)
            int_beta = result.get('interaction_beta', np.nan)
            sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "†" if int_p < 0.10 else ""
            print(f"\n  {desc}:")
            print(f"    N = {result['n']}, R² = {result['r_squared']:.3f}")
            print(f"    UCLA × Depression: β = {int_beta:.4f}, p = {int_p:.4f} {sig}")
            print(f"    UCLA main: β = {result['ucla_beta']:.4f}, p = {result['ucla_p']:.4f}")
        else:
            print(f"\n  {name}: ERROR - {result['error']}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'depression_moderation.csv', index=False, encoding='utf-8-sig')

    # Apply FDR correction
    valid_results = [r for r in results if 'error' not in r and not np.isnan(r.get('interaction_p', np.nan))]
    if len(valid_results) >= 2:
        fdr_df = pd.DataFrame({
            'outcome': [r['outcome_name'] for r in valid_results],
            'p_value': [r['interaction_p'] for r in valid_results]
        })
        fdr_df = apply_fdr_correction(fdr_df, 'p_value')
        print("\n  FDR Correction:")
        for _, row in fdr_df.iterrows():
            print(f"    {row['outcome']}: raw p = {row['p_value']:.4f}, FDR p = {row['p_fdr']:.4f}")

    return results


@register_analysis(
    name="anxiety_moderation",
    description="Test UCLA × DASS_anxiety interaction on EF outcomes"
)
def anxiety_moderation_analysis():
    """Test whether anxiety moderates UCLA → EF relationship."""
    print("\n" + "="*70)
    print("UCLA × DASS ANXIETY MODERATION ANALYSIS")
    print("="*70)

    df = load_moderation_data()
    outcomes = get_ef_outcomes()

    results = []
    for name, col, desc in outcomes:
        if col not in df.columns:
            continue

        result = run_moderation_analysis(df, col, 'z_dass_anx', 'Anxiety')
        result['outcome_name'] = name
        result['outcome_description'] = desc
        results.append(result)

        if 'error' not in result:
            int_p = result.get('interaction_p', np.nan)
            int_beta = result.get('interaction_beta', np.nan)
            sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "†" if int_p < 0.10 else ""
            print(f"\n  {desc}:")
            print(f"    N = {result['n']}, R² = {result['r_squared']:.3f}")
            print(f"    UCLA × Anxiety: β = {int_beta:.4f}, p = {int_p:.4f} {sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'anxiety_moderation.csv', index=False, encoding='utf-8-sig')

    return results


@register_analysis(
    name="stress_moderation",
    description="Test UCLA × DASS_stress interaction on EF outcomes"
)
def stress_moderation_analysis():
    """Test whether stress moderates UCLA → EF relationship."""
    print("\n" + "="*70)
    print("UCLA × DASS STRESS MODERATION ANALYSIS")
    print("="*70)

    df = load_moderation_data()
    outcomes = get_ef_outcomes()

    results = []
    for name, col, desc in outcomes:
        if col not in df.columns:
            continue

        result = run_moderation_analysis(df, col, 'z_dass_str', 'Stress')
        result['outcome_name'] = name
        result['outcome_description'] = desc
        results.append(result)

        if 'error' not in result:
            int_p = result.get('interaction_p', np.nan)
            int_beta = result.get('interaction_beta', np.nan)
            sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "†" if int_p < 0.10 else ""
            print(f"\n  {desc}:")
            print(f"    N = {result['n']}, R² = {result['r_squared']:.3f}")
            print(f"    UCLA × Stress: β = {int_beta:.4f}, p = {int_p:.4f} {sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'stress_moderation.csv', index=False, encoding='utf-8-sig')

    return results


@register_analysis(
    name="johnson_neyman",
    description="Find regions of significance for UCLA effect across DASS levels"
)
def johnson_neyman_dass():
    """Johnson-Neyman analysis for UCLA effect at different DASS levels."""
    print("\n" + "="*70)
    print("JOHNSON-NEYMAN ANALYSIS: UCLA EFFECT × DASS LEVELS")
    print("="*70)

    df = load_moderation_data()
    outcomes = get_ef_outcomes()

    all_results = []

    for dass_var, dass_label in [('z_dass_dep', 'Depression'), ('z_dass_anx', 'Anxiety'), ('z_dass_str', 'Stress')]:
        print(f"\n{'='*50}")
        print(f"  Moderator: {dass_label}")
        print(f"{'='*50}")

        for name, col, desc in outcomes:
            if col not in df.columns:
                continue

            result = johnson_neyman_analysis(df, col, dass_var)

            if 'error' not in result:
                n_sig = result['n_significant']
                base_p = result['base_interaction_p']

                print(f"\n    {desc}:")
                print(f"      Base interaction p = {base_p:.4f}")
                print(f"      Regions with significant UCLA effect: {n_sig}/50 points")

                if n_sig > 0:
                    sig_regions = result['significant_regions']
                    print(f"      Significant at {dass_label} z-scores: [{min(sig_regions):.2f}, {max(sig_regions):.2f}]")

                result['outcome'] = name
                result['moderator_label'] = dass_label
                all_results.append(result)

    # Save summary
    summary_data = []
    for r in all_results:
        if 'error' not in r:
            summary_data.append({
                'outcome': r['outcome'],
                'moderator': r['moderator_label'],
                'base_interaction_p': r['base_interaction_p'],
                'n_significant_regions': r['n_significant'],
                'significant': r['n_significant'] > 0
            })

    pd.DataFrame(summary_data).to_csv(OUTPUT_DIR / 'johnson_neyman_summary.csv', index=False, encoding='utf-8-sig')

    return all_results


@register_analysis(
    name="subgroup_analysis",
    description="Compare EF across UCLA × DASS quadrants (median split)"
)
def subgroup_quadrant_analysis():
    """Compare EF across UCLA × DASS quadrants."""
    print("\n" + "="*70)
    print("SUBGROUP QUADRANT ANALYSIS: UCLA × DASS")
    print("="*70)

    df = load_moderation_data()
    outcomes = get_ef_outcomes()

    # Create UCLA × Depression quadrants (median split)
    ucla_med = df['z_ucla'].median()
    dep_med = df['z_dass_dep'].median()

    df['ucla_high'] = df['z_ucla'] > ucla_med
    df['dep_high'] = df['z_dass_dep'] > dep_med

    # Create quadrant labels
    def get_quadrant(row):
        if row['ucla_high'] and row['dep_high']:
            return 'High UCLA + High Dep'
        elif row['ucla_high'] and not row['dep_high']:
            return 'High UCLA + Low Dep'  # Key subgroup!
        elif not row['ucla_high'] and row['dep_high']:
            return 'Low UCLA + High Dep'
        else:
            return 'Low UCLA + Low Dep'

    df['quadrant'] = df.apply(get_quadrant, axis=1)

    print("\n  Quadrant Distribution:")
    print(df['quadrant'].value_counts())

    results = []

    for name, col, desc in outcomes:
        if col not in df.columns:
            continue

        print(f"\n  {desc}:")

        # ANOVA across quadrants
        quadrant_groups = [
            df[df['quadrant'] == q][col].dropna()
            for q in ['Low UCLA + Low Dep', 'High UCLA + Low Dep', 'Low UCLA + High Dep', 'High UCLA + High Dep']
        ]

        if all(len(g) >= 5 for g in quadrant_groups):
            f_stat, p_val = stats.f_oneway(*quadrant_groups)
            print(f"    ANOVA F = {f_stat:.3f}, p = {p_val:.4f}")

            # Focus on key comparison: High UCLA + Low Dep vs others
            high_ucla_low_dep = df[df['quadrant'] == 'High UCLA + Low Dep'][col].dropna()
            high_ucla_high_dep = df[df['quadrant'] == 'High UCLA + High Dep'][col].dropna()

            if len(high_ucla_low_dep) >= 5 and len(high_ucla_high_dep) >= 5:
                t_stat, t_p = stats.ttest_ind(high_ucla_low_dep, high_ucla_high_dep)
                cohens_d = (high_ucla_low_dep.mean() - high_ucla_high_dep.mean()) / np.sqrt(
                    ((len(high_ucla_low_dep)-1)*high_ucla_low_dep.std()**2 + (len(high_ucla_high_dep)-1)*high_ucla_high_dep.std()**2) /
                    (len(high_ucla_low_dep) + len(high_ucla_high_dep) - 2)
                )

                print(f"    High UCLA + Low Dep vs High UCLA + High Dep:")
                print(f"      t = {t_stat:.3f}, p = {t_p:.4f}, Cohen's d = {cohens_d:.3f}")

                sig = "*" if t_p < 0.05 else ""
                results.append({
                    'outcome': name,
                    'comparison': 'High UCLA: Low Dep vs High Dep',
                    't_stat': t_stat,
                    'p_value': t_p,
                    'cohens_d': cohens_d,
                    'mean_low_dep': high_ucla_low_dep.mean(),
                    'mean_high_dep': high_ucla_high_dep.mean(),
                    'n_low_dep': len(high_ucla_low_dep),
                    'n_high_dep': len(high_ucla_high_dep)
                })

        # Print quadrant means
        print("    Quadrant Means:")
        for q in ['Low UCLA + Low Dep', 'High UCLA + Low Dep', 'Low UCLA + High Dep', 'High UCLA + High Dep']:
            qdata = df[df['quadrant'] == q][col].dropna()
            print(f"      {q}: M = {qdata.mean():.3f}, SD = {qdata.std():.3f}, N = {len(qdata)}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'quadrant_analysis.csv', index=False, encoding='utf-8-sig')

    return results


@register_analysis(
    name="summary",
    description="Integration summary across all moderation analyses"
)
def summary_analysis():
    """Create summary of all moderation analyses."""
    print("\n" + "="*70)
    print("SUMMARY: UCLA × DASS MODERATION")
    print("="*70)

    # Run all analyses
    dep_results = depression_moderation_analysis()
    anx_results = anxiety_moderation_analysis()
    str_results = stress_moderation_analysis()

    # Compile summary
    all_results = []
    for r in dep_results + anx_results + str_results:
        if 'error' not in r:
            all_results.append({
                'outcome': r.get('outcome_name', r.get('outcome')),
                'moderator': r.get('moderator'),
                'interaction_beta': r.get('interaction_beta'),
                'interaction_p': r.get('interaction_p'),
                'n': r.get('n')
            })

    summary_df = pd.DataFrame(all_results)

    print("\n  MODERATION SUMMARY (UCLA × DASS subscale → EF):")
    print("  " + "-"*60)

    # Find any significant interactions
    sig_results = summary_df[summary_df['interaction_p'] < 0.05]
    marginal_results = summary_df[(summary_df['interaction_p'] >= 0.05) & (summary_df['interaction_p'] < 0.10)]

    if len(sig_results) > 0:
        print("\n  SIGNIFICANT (p < 0.05):")
        for _, row in sig_results.iterrows():
            print(f"    {row['outcome']} × {row['moderator']}: β = {row['interaction_beta']:.4f}, p = {row['interaction_p']:.4f}")
    else:
        print("\n  No significant UCLA × DASS interactions found (all p > 0.05)")

    if len(marginal_results) > 0:
        print("\n  MARGINAL (0.05 < p < 0.10):")
        for _, row in marginal_results.iterrows():
            print(f"    {row['outcome']} × {row['moderator']}: β = {row['interaction_beta']:.4f}, p = {row['interaction_p']:.4f}")

    # Apply FDR across all tests
    valid_p = summary_df.dropna(subset=['interaction_p'])
    if len(valid_p) >= 2:
        fdr_df = apply_fdr_correction(valid_p, 'interaction_p')

        print("\n  FDR CORRECTION (across all tests):")
        any_fdr_sig = False
        for _, row in fdr_df.iterrows():
            if row['interaction_p'] < 0.10:
                p_fdr = row['p_fdr']
                sig_marker = "***" if p_fdr < 0.001 else "**" if p_fdr < 0.01 else "*" if p_fdr < 0.05 else "†" if p_fdr < 0.10 else ""
                print(f"    {row['outcome']}: raw p = {row['interaction_p']:.4f}, FDR p = {p_fdr:.4f} {sig_marker}")
                if p_fdr < 0.05:
                    any_fdr_sig = True

        if not any_fdr_sig:
            print("    No effects survive FDR correction")

    # Save full summary
    summary_df.to_csv(OUTPUT_DIR / 'full_moderation_summary.csv', index=False, encoding='utf-8-sig')

    print("\n  Results saved to:", OUTPUT_DIR)

    return summary_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def list_analyses():
    """List all available analyses."""
    print("\nAvailable analyses in ucla_dass_moderation_suite:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name:25s} - {spec.description}")
    print("\nUsage: python -m analysis.advanced.ucla_dass_moderation_suite --analysis <name>")
    print("       python -m analysis.advanced.ucla_dass_moderation_suite --all")


def run_analysis(name: str) -> Optional[Dict]:
    """Run a specific analysis by name."""
    if name not in ANALYSES:
        print(f"Unknown analysis: {name}")
        list_analyses()
        return None

    spec = ANALYSES[name]
    print(f"\nRunning: {spec.description}")
    return spec.function()


def run_all_analyses():
    """Run all registered analyses."""
    print("\n" + "="*70)
    print("RUNNING ALL UCLA × DASS MODERATION ANALYSES")
    print("="*70)

    results = {}
    for name, spec in ANALYSES.items():
        if name != 'summary':  # Summary will run the others
            try:
                results[name] = spec.function()
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = {'error': str(e)}

    # Run summary last
    results['summary'] = summary_analysis()

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UCLA × DASS Moderation Analysis Suite"
    )
    parser.add_argument(
        "--analysis", "-a",
        help="Run specific analysis"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available analyses"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses"
    )

    args = parser.parse_args()

    if args.list:
        list_analyses()
    elif args.analysis:
        run_analysis(args.analysis)
    elif args.all:
        run_all_analyses()
    else:
        # Default: run summary (which runs all)
        summary_analysis()


if __name__ == "__main__":
    main()
