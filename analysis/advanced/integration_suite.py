"""
Integration Analysis Suite
==========================

Integrates results from all computational modeling and mechanistic analyses.

Produces:
- Summary table of all significant effects
- Forest plot of effect sizes
- Composite mechanistic index
- Paper-ready figures and tables

Usage:
    python -m analysis.advanced.integration_suite              # Run all
    python -m analysis.advanced.integration_suite --analysis summary
    python -m analysis.advanced.integration_suite --list

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
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Project imports
from analysis.utils.data_loader_utils import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directories
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "integration"
PAPER_DIR = Path("results/paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)


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


def register_analysis(name: str, description: str, source_script: str = "integration_suite.py"):
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
# HELPER FUNCTIONS
# =============================================================================

def collect_results(base_dir: Path, pattern: str = "*.csv") -> List[pd.DataFrame]:
    """Collect all result CSV files from a directory."""
    results = []
    if base_dir.exists():
        for f in base_dir.glob(pattern):
            try:
                df = pd.read_csv(f)
                df['source_file'] = f.name
                results.append(df)
            except Exception:
                continue
    return results


def compute_cohens_d(beta: float, se: float, n: int) -> float:
    """Approximate Cohen's d from regression coefficient."""
    # Rough approximation: d ≈ 2 * t / sqrt(n)
    t = beta / se if se > 0 else 0
    d = 2 * t / np.sqrt(n) if n > 0 else 0
    return d


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="summary",
    description="Create summary table of all UCLA effects"
)
def analyze_summary(verbose: bool = True) -> pd.DataFrame:
    """
    Collect and summarize UCLA effects from all analysis suites.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("INTEGRATED SUMMARY OF UCLA EFFECTS")
        print("=" * 70)

    # Directories to search
    suite_dirs = {
        'reinforcement_learning': ANALYSIS_OUTPUT_DIR / "reinforcement_learning",
        'attention_depletion': ANALYSIS_OUTPUT_DIR / "attention_depletion",
        'error_monitoring': ANALYSIS_OUTPUT_DIR / "error_monitoring",
        'control_strategy': ANALYSIS_OUTPUT_DIR / "control_strategy",
        'ddm': ANALYSIS_OUTPUT_DIR / "ddm"
    }

    all_effects = []

    for suite_name, suite_dir in suite_dirs.items():
        if not suite_dir.exists():
            continue

        if verbose:
            print(f"\n  {suite_name.upper()}")
            print("  " + "-" * 50)

        for f in suite_dir.glob("*.csv"):
            try:
                df = pd.read_csv(f)

                # Look for UCLA-related columns
                ucla_cols = [c for c in df.columns if 'ucla' in c.lower() or 'beta' in c.lower()]

                if 'beta_ucla' in df.columns or 'p_ucla' in df.columns:
                    for _, row in df.iterrows():
                        if pd.notna(row.get('beta_ucla')) and pd.notna(row.get('p_ucla')):
                            effect = {
                                'suite': suite_name,
                                'file': f.name,
                                'task': row.get('task', ''),
                                'metric': row.get('metric', row.get('parameter', '')),
                                'beta_ucla': row.get('beta_ucla'),
                                'se_ucla': row.get('se_ucla', np.nan),
                                'p_ucla': row.get('p_ucla'),
                                'n': row.get('n', np.nan),
                                'r_squared': row.get('r_squared', np.nan)
                            }

                            # Compute effect size
                            if pd.notna(effect['se_ucla']) and pd.notna(effect['n']):
                                effect['cohens_d'] = compute_cohens_d(
                                    effect['beta_ucla'],
                                    effect['se_ucla'],
                                    effect['n']
                                )
                            else:
                                effect['cohens_d'] = np.nan

                            all_effects.append(effect)

                            if verbose:
                                sig = "*" if effect['p_ucla'] < 0.05 else ""
                                print(f"    {effect['metric']}: beta={effect['beta_ucla']:.4f}, "
                                      f"p={effect['p_ucla']:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    Error reading {f.name}: {e}")

    if len(all_effects) == 0:
        if verbose:
            print("\n  No effects found - run individual suites first")
        return pd.DataFrame()

    effects_df = pd.DataFrame(all_effects)

    # Sort by p-value
    effects_df = effects_df.sort_values('p_ucla')

    # Apply FDR correction (Benjamini-Hochberg)
    p_values = effects_df['p_ucla'].values
    if len(p_values) > 0:
        _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
        effects_df['p_fdr'] = p_fdr
        effects_df['significant_fdr'] = effects_df['p_fdr'] < 0.05
    else:
        effects_df['p_fdr'] = np.nan
        effects_df['significant_fdr'] = False

    # Add significance markers (raw)
    effects_df['significant'] = effects_df['p_ucla'] < 0.05
    effects_df['marginal'] = (effects_df['p_ucla'] >= 0.05) & (effects_df['p_ucla'] < 0.10)

    # Summary statistics
    n_total = len(effects_df)
    n_sig = effects_df['significant'].sum()
    n_sig_fdr = effects_df['significant_fdr'].sum()
    n_marginal = effects_df['marginal'].sum()

    if verbose:
        print(f"\n  SUMMARY")
        print("  " + "-" * 50)
        print(f"    Total effects tested: {n_total}")
        print(f"    Significant (p < 0.05, raw): {n_sig}")
        print(f"    Significant (FDR-corrected): {n_sig_fdr}")
        print(f"    Marginal (0.05 <= p < 0.10): {n_marginal}")

        if n_sig > 0:
            print(f"\n  SIGNIFICANT EFFECTS (raw p < 0.05):")
            for _, row in effects_df[effects_df['significant']].iterrows():
                d = f", d={row['cohens_d']:.2f}" if pd.notna(row['cohens_d']) else ""
                fdr_sig = "✓" if row.get('significant_fdr', False) else "✗"
                print(f"    {row['suite']}/{row['metric']}: beta={row['beta_ucla']:.4f}, "
                      f"p={row['p_ucla']:.4f}, p_fdr={row['p_fdr']:.4f} [{fdr_sig}]{d}")

    effects_df.to_csv(OUTPUT_DIR / "all_ucla_effects.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'all_ucla_effects.csv'}")

    return effects_df


@register_analysis(
    name="effect_sizes",
    description="Create effect size summary with confidence intervals"
)
def analyze_effect_sizes(verbose: bool = True) -> pd.DataFrame:
    """
    Compute effect sizes and confidence intervals for forest plot.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EFFECT SIZE SUMMARY")
        print("=" * 70)

    # Load summary if exists
    summary_file = OUTPUT_DIR / "all_ucla_effects.csv"
    if not summary_file.exists():
        if verbose:
            print("  Running summary analysis first...")
        analyze_summary(verbose=False)

    if not summary_file.exists():
        if verbose:
            print("  No effects available")
        return pd.DataFrame()

    effects = pd.read_csv(summary_file)

    # Compute confidence intervals
    effects['ci_lower'] = effects['beta_ucla'] - 1.96 * effects['se_ucla']
    effects['ci_upper'] = effects['beta_ucla'] + 1.96 * effects['se_ucla']

    # Filter to effects with valid CIs
    valid = effects.dropna(subset=['ci_lower', 'ci_upper'])

    if len(valid) == 0:
        if verbose:
            print("  No effects with valid confidence intervals")
        return pd.DataFrame()

    if verbose:
        print(f"  Effects with CIs: {len(valid)}")

        # Print forest plot style output
        print("\n  FOREST PLOT DATA")
        print("  " + "-" * 70)
        print(f"  {'Metric':<30} {'Beta':>10} {'95% CI':>20} {'p':>10}")
        print("  " + "-" * 70)

        for _, row in valid.iterrows():
            metric_name = f"{row['suite'][:3]}/{row['metric']}"[:30]
            ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            sig = "*" if row['p_ucla'] < 0.05 else ""
            print(f"  {metric_name:<30} {row['beta_ucla']:>10.4f} {ci:>20} {row['p_ucla']:>9.4f}{sig}")

    valid.to_csv(OUTPUT_DIR / "effect_sizes_with_ci.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'effect_sizes_with_ci.csv'}")

    return valid


@register_analysis(
    name="mechanistic_composite",
    description="Create composite mechanistic index combining all measures"
)
def analyze_mechanistic_composite(verbose: bool = True) -> pd.DataFrame:
    """
    Create a composite index from all mechanistic measures.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MECHANISTIC COMPOSITE INDEX")
        print("=" * 70)

    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
    master = standardize_predictors(master)

    # Collect participant-level metrics from each suite
    component_files = {
        'rl_basic': ANALYSIS_OUTPUT_DIR / "reinforcement_learning" / "rw_basic_parameters.csv",
        'rl_asymmetric': ANALYSIS_OUTPUT_DIR / "reinforcement_learning" / "rw_asymmetric_parameters.csv",
        'attention': ANALYSIS_OUTPUT_DIR / "attention_depletion" / "attention_components.csv",
        'dmc': ANALYSIS_OUTPUT_DIR / "control_strategy" / "dmc_by_participant.csv"
    }

    # Merge available components
    composite = master[['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male']].copy()

    for comp_name, comp_file in component_files.items():
        if comp_file.exists():
            try:
                comp_df = pd.read_csv(comp_file)
                if 'participant_id' in comp_df.columns:
                    # Keep only numeric columns
                    numeric_cols = [c for c in comp_df.columns if comp_df[c].dtype in ['float64', 'int64'] and c != 'participant_id']
                    if numeric_cols:
                        comp_df = comp_df[['participant_id'] + numeric_cols]
                        composite = composite.merge(comp_df, on='participant_id', how='left')
                        if verbose:
                            print(f"  Added {comp_name}: {len(numeric_cols)} variables")
            except Exception as e:
                if verbose:
                    print(f"  Error loading {comp_name}: {e}")

    # Drop rows with too many missing values
    numeric_cols = [c for c in composite.columns if composite[c].dtype in ['float64', 'int64']
                    and c not in ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male']]

    if len(numeric_cols) == 0:
        if verbose:
            print("  No mechanistic measures found")
        return pd.DataFrame()

    if verbose:
        print(f"\n  Total mechanistic measures: {len(numeric_cols)}")
        print(f"  Participants: {len(composite)}")

    # Z-score all mechanistic measures
    for col in numeric_cols:
        std = composite[col].std()
        if std > 0:
            composite[f'z_{col}'] = (composite[col] - composite[col].mean()) / std

    # Create simple composite (mean of z-scored measures)
    z_cols = [f'z_{c}' for c in numeric_cols if f'z_{c}' in composite.columns]

    if z_cols:
        composite['mechanistic_composite'] = composite[z_cols].mean(axis=1, skipna=True)

        if verbose:
            print(f"\n  Composite created from {len(z_cols)} measures")
            print(f"  Mean composite: {composite['mechanistic_composite'].mean():.3f}")
            print(f"  SD composite: {composite['mechanistic_composite'].std():.3f}")

        # Test UCLA effect on composite
        merged_clean = composite.dropna(subset=['mechanistic_composite', 'z_ucla'])

        if len(merged_clean) >= 30:
            try:
                formula = "mechanistic_composite ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"\n  UCLA -> Mechanistic Composite: beta={beta:.4f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"  Regression error: {e}")

    composite.to_csv(OUTPUT_DIR / "mechanistic_composite.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'mechanistic_composite.csv'}")

    return composite


@register_analysis(
    name="paper_tables",
    description="Generate paper-ready summary tables"
)
def analyze_paper_tables(verbose: bool = True) -> pd.DataFrame:
    """
    Generate formatted tables for publication.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PAPER-READY TABLES")
        print("=" * 70)

    # Load master data for descriptives
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Table 1: Descriptive statistics
    desc_vars = ['age', 'ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress',
                 'pe_rate', 'stroop_interference', 'prp_bottleneck']

    desc_stats = []
    for var in desc_vars:
        if var in master.columns:
            desc_stats.append({
                'Variable': var,
                'N': master[var].notna().sum(),
                'Mean': master[var].mean(),
                'SD': master[var].std(),
                'Min': master[var].min(),
                'Max': master[var].max()
            })

    desc_df = pd.DataFrame(desc_stats)
    desc_df.to_csv(PAPER_DIR / "table1_descriptives.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print("\n  Table 1: Descriptive Statistics")
        print("  " + "-" * 60)
        print(desc_df.to_string(index=False))

    # Table 2: UCLA effects summary (significant only)
    effects_file = OUTPUT_DIR / "all_ucla_effects.csv"
    if effects_file.exists():
        effects = pd.read_csv(effects_file)
        sig_effects = effects[effects['significant'] == True][
            ['suite', 'task', 'metric', 'beta_ucla', 'se_ucla', 'p_ucla', 'cohens_d', 'n']
        ].copy()

        sig_effects.to_csv(PAPER_DIR / "table2_significant_effects.csv", index=False, encoding='utf-8-sig')

        if verbose:
            print(f"\n  Table 2: Significant UCLA Effects (N={len(sig_effects)})")
            print("  " + "-" * 60)
            if len(sig_effects) > 0:
                print(sig_effects.to_string(index=False))
            else:
                print("  No significant effects found")

    # Table 3: Model comparison summary (if available)
    model_file = ANALYSIS_OUTPUT_DIR / "reinforcement_learning" / "model_comparison_summary.csv"
    if model_file.exists():
        model_comp = pd.read_csv(model_file)
        model_comp.to_csv(PAPER_DIR / "table3_model_comparison.csv", index=False, encoding='utf-8-sig')

        if verbose:
            print("\n  Table 3: RL Model Comparison")
            print("  " + "-" * 60)
            print(model_comp.to_string(index=False))

    if verbose:
        print(f"\n  Output directory: {PAPER_DIR}")

    return desc_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run integration analyses.
    """
    if verbose:
        print("=" * 70)
        print("INTEGRATION ANALYSIS SUITE")
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
        for name, spec in ANALYSES.items():
            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("INTEGRATION SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Paper figures: {PAPER_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Integration Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integration Analysis Suite")
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
