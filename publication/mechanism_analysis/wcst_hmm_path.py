"""
WCST HMM Path Analysis
======================

Provides SEM-style path models linking UCLA loneliness, DASS symptoms, and
HMM-derived lapse metrics (occupancy, transitions, RT differences). Keeps the
legacy DASS-controlled HMM suite untouched by isolating this analysis.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import pandas as pd

from .wcst_hmm_modeling_dass import (
    load_hmm_data,
    fit_hmm_per_participant,
)
from ._utils import BASE_OUTPUT, DEFAULT_PATH_COVARIATES, run_path_models

OUTPUT_DIR = BASE_OUTPUT / "wcst_hmm_modeling_path"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AnalysisSpec:
    name: str
    description: str
    function: Callable


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str):
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(name, description, func)
        return func
    return decorator


@register_analysis(
    name="path_models",
    description="Loneliness↔DASS path analysis for HMM lapse metrics"
)
def analyze_path_models(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 70)
        print("WCST HMM PATH MODELS")
        print("=" * 70)

    master, trials = load_hmm_data()
    hmm_df = fit_hmm_per_participant(trials)

    if len(hmm_df) < 20:
        if verbose:
            print(f"  Insufficient HMM fits (N={len(hmm_df)}).")
        return pd.DataFrame()

    merged = master.merge(hmm_df, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient merged data (N={len(merged)}).")
        return pd.DataFrame()

    dass_targets = [col for col in ['z_dass_dep', 'z_dass_anx', 'z_dass_str'] if col in merged.columns]
    if not dass_targets:
        if verbose:
            print("  No standardized DASS columns available.")
        return pd.DataFrame()

    param_targets = [
        ('lapse_occupancy', 'Lapse occupancy (%)'),
        ('trans_to_lapse', 'P(Focus→Lapse)'),
        ('trans_to_focus', 'P(Lapse→Focus)'),
        ('stay_lapse', 'P(Lapse→Lapse)'),
        ('stay_focus', 'P(Focus→Focus)'),
        ('rt_diff', 'Lapse-Focus RT difference'),
        ('lapse_rt_mean', 'Lapse RT mean'),
        ('focus_rt_mean', 'Focus RT mean'),
        ('state_changes', 'State switches'),
    ]

    records = []
    for param_col, param_label in param_targets:
        if param_col not in merged.columns:
            continue
        for dass_col in dass_targets:
            path_results = run_path_models(
                merged,
                param_col=param_col,
                param_label=param_label,
                module_tag='wcst_hmm_path',
                dass_var=dass_col,
                covariates=DEFAULT_PATH_COVARIATES,
                min_n=30,
            )
            records.extend(path_results)

    if not records:
        if verbose:
            print("  No valid HMM path models were fit.")
        return pd.DataFrame()

    results_df = pd.DataFrame(records)
    output_file = OUTPUT_DIR / "wcst_hmm_path_models.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        sig_rows = results_df[(results_df['a_p'] < 0.05) & (results_df['b_p'] < 0.05)]
        if len(sig_rows) > 0:
            print("\n  Candidate mediated HMM paths (a & b p < 0.05):")
            for _, row in sig_rows.iterrows():
                print(f"    {row['parameter']} x {row['dass_var']} ({row['model_name']}): "
                      f"a={row['a_coef']:.3f} (p={row['a_p']:.4f}), "
                      f"b={row['b_coef']:.3f} (p={row['b_p']:.4f})")

        print(f"\n  Output: {output_file}")

    return results_df


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    if verbose:
        print("=" * 70)
        print("WCST HMM PATH SUITE")
        print("=" * 70)

    if analysis and analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")

    target = ANALYSES.get(analysis or 'path_models')
    results = {}
    if target:
        results[target.name] = target.function(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("WCST HMM PATH SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    print("\nAvailable WCST HMM Path Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WCST HMM Path Analysis")
    parser.add_argument('--analysis', '-a', type=str, default=None, help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true', help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true', help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
