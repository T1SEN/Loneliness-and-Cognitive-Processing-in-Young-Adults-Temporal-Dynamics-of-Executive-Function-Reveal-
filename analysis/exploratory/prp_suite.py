"""
PRP Exploratory Analysis Suite
==============================

Unified exploratory analyses for the PRP (Psychological Refractory Period) task.

Consolidates:
- prp_bottleneck_shape.py
- prp_response_order_analysis.py
- prp_exgaussian_decomposition.py
- prp_post_error_adjustments.py
- prp_rt_variability_extended.py

Usage:
    python -m analysis.exploratory.prp_suite                    # Run all
    python -m analysis.exploratory.prp_suite --analysis bottleneck_shape
    python -m analysis.exploratory.prp_suite --analysis response_order

    from analysis.exploratory import prp_suite
    prp_suite.run('bottleneck_shape')
    prp_suite.run()  # All analyses

NOTE: These are EXPLORATORY analyses. For confirmatory results, use:
    analysis/gold_standard/pipeline.py
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# Project imports
from analysis.utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, PRP_RT_MAX
)
from analysis.utils.modeling import DASS_CONTROL_FORMULA, standardize_predictors
from analysis.utils.plotting import set_publication_style, create_scatter_with_regression

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "prp_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard SOA values
SOA_VALUES = [50, 150, 300, 600, 1200]


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


def register_analysis(name: str, description: str, source_script: str):
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

def load_prp_trials() -> pd.DataFrame:
    """Load PRP trial data."""
    df = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    # Standardize column names
    if 't1_rt_ms' in df.columns:
        df['t1_rt'] = df['t1_rt_ms']
    if 't2_rt_ms' in df.columns:
        df['t2_rt'] = df['t2_rt_ms']
    if 'soa_nominal_ms' in df.columns:
        df['soa'] = df['soa_nominal_ms']

    # Filter valid trials
    df = df[
        (df['t1_rt'] > DEFAULT_RT_MIN) &
        (df['t1_rt'] < PRP_RT_MAX) &
        (df['t2_rt'] > DEFAULT_RT_MIN) &
        (df['t2_rt'] < PRP_RT_MAX)
    ].copy()

    return df


def load_master_with_prp() -> pd.DataFrame:
    """Load master dataset with PRP metrics merged."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender
    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
    master['gender_male'] = (master['gender'] == 'male').astype(int)

    # Ensure ucla_total
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors
    master = standardize_predictors(master)

    return master


# =============================================================================
# ANALYSIS 1: BOTTLENECK SHAPE
# =============================================================================

def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


@register_analysis(
    name="bottleneck_shape",
    description="Analyze PRP effect curve shape (linear vs exponential)",
    source_script="prp_bottleneck_shape.py"
)
def analyze_bottleneck_shape(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze individual differences in PRP bottleneck curve shape.

    Fits both linear and exponential models to SOA-RT function.
    Tests whether UCLA predicts slope/asymptote parameters.
    """
    output_dir = OUTPUT_DIR / "bottleneck_shape"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[BOTTLENECK SHAPE] Analyzing PRP curve morphology...")

    # Load data
    trials = load_prp_trials()
    master = load_master_with_prp()

    # Compute RT by SOA for each participant
    rt_by_soa = trials.groupby(['participant_id', 'soa'])['t2_rt'].mean().unstack()

    # Fit linear slopes
    results = []
    for pid in rt_by_soa.index:
        row = rt_by_soa.loc[pid]
        valid_soas = [soa for soa in SOA_VALUES if soa in row.index and not pd.isna(row[soa])]

        if len(valid_soas) < 3:
            continue

        x = np.array(valid_soas)
        y = np.array([row[soa] for soa in valid_soas])

        # Linear fit
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)

        results.append({
            'participant_id': pid,
            'linear_slope': slope,
            'linear_intercept': intercept,
            'linear_r': r_val,
            'linear_r2': r_val ** 2,
            'linear_p': p_val,
            'n_soas': len(valid_soas)
        })

    curve_df = pd.DataFrame(results)

    # Merge with master
    analysis_df = curve_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N with curve fits: {len(analysis_df)}")
        print(f"  Mean slope: {analysis_df['linear_slope'].mean():.3f} ms/ms")

    # Run DASS-controlled regression on slope
    if len(analysis_df) >= 30:
        formula = "linear_slope ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'linear_slope'])).fit(cov_type='HC3')

        if verbose:
            print(f"\n  UCLA → Slope: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        # Save coefficients
        coef_df = pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'se': model.bse.values,
            'p': model.pvalues.values
        })
        coef_df.to_csv(output_dir / "slope_regression_coefficients.csv", index=False, encoding='utf-8-sig')

    # Save curve parameters
    curve_df.to_csv(output_dir / "individual_curve_parameters.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"  Saved to: {output_dir}")

    return curve_df


# =============================================================================
# ANALYSIS 2: RESPONSE ORDER
# =============================================================================

@register_analysis(
    name="response_order",
    description="Analyze T2→T1 response order reversals",
    source_script="prp_response_order_analysis.py"
)
def analyze_response_order(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze response order patterns in PRP task.

    Response order reversals (T2 before T1) may indicate:
    - Task prioritization failures
    - Impulsivity
    - Attentional control deficits
    """
    output_dir = OUTPUT_DIR / "response_order"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[RESPONSE ORDER] Analyzing T1/T2 response order patterns...")

    trials = load_prp_trials()
    master = load_master_with_prp()

    # Compute response order from RTs
    trials['response_order'] = np.where(
        trials['t1_rt'] < trials['t2_rt'], 'T1T2',
        np.where(trials['t1_rt'] > trials['t2_rt'], 'T2T1', 'simultaneous')
    )

    # Compute reversal rate per participant
    reversal_df = trials.groupby('participant_id').apply(
        lambda x: pd.Series({
            'n_trials': len(x),
            'n_t1t2': (x['response_order'] == 'T1T2').sum(),
            'n_t2t1': (x['response_order'] == 'T2T1').sum(),
            'reversal_rate': (x['response_order'] == 'T2T1').mean() * 100
        })
    ).reset_index()

    # Merge with master
    analysis_df = reversal_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean reversal rate: {analysis_df['reversal_rate'].mean():.1f}%")

    # Run DASS-controlled regression
    if len(analysis_df) >= 30:
        formula = "reversal_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'reversal_rate'])).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → Reversal: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        coef_df = pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'se': model.bse.values,
            'p': model.pvalues.values
        })
        coef_df.to_csv(output_dir / "reversal_regression_coefficients.csv", index=False, encoding='utf-8-sig')

    reversal_df.to_csv(output_dir / "response_order_metrics.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"  Saved to: {output_dir}")

    return reversal_df


# =============================================================================
# ANALYSIS 3: RT VARIABILITY
# =============================================================================

@register_analysis(
    name="rt_variability",
    description="Analyze intra-individual RT variability by SOA",
    source_script="prp_rt_variability_extended.py"
)
def analyze_rt_variability(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze intra-individual RT variability patterns.

    Computes:
    - CV (coefficient of variation) by SOA
    - IQR by SOA
    - Overall variability measures
    """
    output_dir = OUTPUT_DIR / "rt_variability"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[RT VARIABILITY] Analyzing intra-individual variability...")

    trials = load_prp_trials()
    master = load_master_with_prp()

    # Categorize SOA
    trials['soa_cat'] = pd.cut(
        trials['soa'],
        bins=[0, 150, 600, 2000],
        labels=['short', 'medium', 'long']
    )

    # Compute variability metrics per participant
    variability = trials.groupby('participant_id').apply(
        lambda x: pd.Series({
            't2_rt_mean': x['t2_rt'].mean(),
            't2_rt_sd': x['t2_rt'].std(),
            't2_rt_cv': x['t2_rt'].std() / x['t2_rt'].mean() * 100,
            't2_rt_iqr': x['t2_rt'].quantile(0.75) - x['t2_rt'].quantile(0.25),
            'n_trials': len(x)
        })
    ).reset_index()

    # Variability by SOA category
    for soa_cat in ['short', 'medium', 'long']:
        soa_trials = trials[trials['soa_cat'] == soa_cat]
        soa_var = soa_trials.groupby('participant_id').apply(
            lambda x: pd.Series({
                f'cv_{soa_cat}': x['t2_rt'].std() / x['t2_rt'].mean() * 100 if x['t2_rt'].mean() > 0 else np.nan
            })
        ).reset_index()
        variability = variability.merge(soa_var, on='participant_id', how='left')

    # Merge with master
    analysis_df = variability.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean CV: {analysis_df['t2_rt_cv'].mean():.1f}%")

    # Run DASS-controlled regression on overall CV
    if len(analysis_df) >= 30:
        formula = "t2_rt_cv ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 't2_rt_cv'])).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → CV: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        coef_df = pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'se': model.bse.values,
            'p': model.pvalues.values
        })
        coef_df.to_csv(output_dir / "variability_regression_coefficients.csv", index=False, encoding='utf-8-sig')

    variability.to_csv(output_dir / "rt_variability_metrics.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"  Saved to: {output_dir}")

    return variability


# =============================================================================
# ANALYSIS 4: POST-ERROR ADJUSTMENTS
# =============================================================================

@register_analysis(
    name="post_error",
    description="Analyze post-error RT adjustments",
    source_script="prp_post_error_adjustments.py"
)
def analyze_post_error(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze post-error slowing in PRP task.

    Examines RT changes following T1 or T2 errors.
    """
    output_dir = OUTPUT_DIR / "post_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[POST-ERROR] Analyzing post-error adjustments...")

    trials = load_prp_trials()
    master = load_master_with_prp()

    # Sort trials by participant and trial order
    trials = trials.sort_values(['participant_id', 'idx' if 'idx' in trials.columns else 'trial_index'])

    # Add previous trial info
    trials['prev_t1_correct'] = trials.groupby('participant_id')['t1_correct'].shift(1)
    trials['prev_t2_correct'] = trials.groupby('participant_id')['t2_correct'].shift(1)

    # Compute PES per participant
    pes_results = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid]

        # After T1 error
        post_t1_error = pdata[pdata['prev_t1_correct'] == 0]['t2_rt'].mean()
        post_t1_correct = pdata[pdata['prev_t1_correct'] == 1]['t2_rt'].mean()
        pes_t1 = post_t1_error - post_t1_correct if not pd.isna(post_t1_error) and not pd.isna(post_t1_correct) else np.nan

        # After T2 error
        post_t2_error = pdata[pdata['prev_t2_correct'] == 0]['t2_rt'].mean()
        post_t2_correct = pdata[pdata['prev_t2_correct'] == 1]['t2_rt'].mean()
        pes_t2 = post_t2_error - post_t2_correct if not pd.isna(post_t2_error) and not pd.isna(post_t2_correct) else np.nan

        pes_results.append({
            'participant_id': pid,
            'pes_after_t1_error': pes_t1,
            'pes_after_t2_error': pes_t2,
            'n_trials': len(pdata)
        })

    pes_df = pd.DataFrame(pes_results)

    # Merge with master
    analysis_df = pes_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean PES (T1 error): {analysis_df['pes_after_t1_error'].mean():.1f} ms")

    # Run DASS-controlled regression
    pes_col = 'pes_after_t1_error'
    analysis_clean = analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', pes_col])

    if len(analysis_clean) >= 30:
        formula = f"{pes_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_clean).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → PES: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        coef_df = pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'se': model.bse.values,
            'p': model.pvalues.values
        })
        coef_df.to_csv(output_dir / "pes_regression_coefficients.csv", index=False, encoding='utf-8-sig')

    pes_df.to_csv(output_dir / "post_error_metrics.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"  Saved to: {output_dir}")

    return pes_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(
    analysis: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run PRP exploratory analyses.

    Args:
        analysis: Specific analysis to run (default: all)
        verbose: Print progress

    Returns:
        Dict mapping analysis names to result DataFrames
    """
    if verbose:
        print("=" * 70)
        print("PRP EXPLORATORY ANALYSIS SUITE")
        print("=" * 70)
        print("\n⚠️  NOTE: These are EXPLORATORY analyses.")
        print("   For confirmatory results, use: analysis/gold_standard/pipeline.py")
        print("=" * 70)

    results = {}

    if analysis:
        # Run specific analysis
        if analysis not in ANALYSES:
            available = list(ANALYSES.keys())
            raise ValueError(f"Unknown analysis: {analysis}. Available: {available}")

        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
            print(f"  Description: {spec.description}")
            print(f"  Source: {spec.source_script}")

        results[analysis] = spec.function(verbose=verbose)
    else:
        # Run all analyses
        for name, spec in ANALYSES.items():
            if verbose:
                print(f"\n[{name.upper()}] {spec.description}")

            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR: {e}")
                results[name] = None

    if verbose:
        print("\n" + "=" * 70)
        print("PRP SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List all available analyses."""
    print("\nAvailable PRP analyses:")
    print("-" * 50)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")
        print(f"    Source: {spec.source_script}")
    print()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRP Exploratory Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                       help='Specific analysis to run (default: all)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available analyses')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output')

    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
