"""
Stroop Ex-Gaussian Path Analysis
================================

Separates path-analysis code from the DASS-controlled Stroop ex-Gaussian
decomposition to keep the original script unchanged while enabling SEM-style
exploration of UCLA↔DASS effects on Stroop parameters.
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

from .stroop_exgaussian_dass import (
    load_stroop_data,
    analyze_basic_exgaussian,
)
from ._utils import BASE_OUTPUT, DEFAULT_PATH_COVARIATES, run_path_models

STROOP_DASS_DIR = BASE_OUTPUT / "stroop_exgaussian"
OUTPUT_DIR = BASE_OUTPUT / "stroop_exgaussian_path"
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
    description="Loneliness↔DASS path analysis for Stroop ex-Gaussian parameters"
)
def analyze_path_models(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 70)
        print("STROOP EX-GAUSSIAN PATH MODELS")
        print("=" * 70)

    master = load_stroop_data()
    param_file = STROOP_DASS_DIR / "stroop_exgaussian_parameters.csv"

    if not param_file.exists():
        if verbose:
            print("  Base parameters missing - running DASS decomposition...")
        analyze_basic_exgaussian(verbose=False)

    if not param_file.exists():
        if verbose:
            print("  Unable to generate Stroop parameters.")
        return pd.DataFrame()

    params = pd.read_csv(param_file)

    if 'mu_interference' not in params.columns:
        params['mu_interference'] = params['incongruent_mu'] - params['congruent_mu']
        params['sigma_interference'] = params['incongruent_sigma'] - params['congruent_sigma']
        params['tau_interference'] = params['incongruent_tau'] - params['congruent_tau']

    merged = master.merge(params, on='participant_id', how='inner')

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

    records = []
    for param_col, param_label in param_targets:
        if param_col not in merged.columns:
            continue
        for dass_col in dass_targets:
            path_results = run_path_models(
                merged,
                param_col=param_col,
                param_label=param_label,
                module_tag='stroop_exgaussian_path',
                dass_var=dass_col,
                covariates=DEFAULT_PATH_COVARIATES,
                min_n=30,
            )
            records.extend(path_results)

    if not records:
        if verbose:
            print("  No valid path models were fit.")
        return pd.DataFrame()

    results_df = pd.DataFrame(records)
    output_file = OUTPUT_DIR / "stroop_exgaussian_path_models.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        sig_rows = results_df[(results_df['a_p'] < 0.05) & (results_df['b_p'] < 0.05)]
        if len(sig_rows) > 0:
            print("\n  Candidate mediated paths (a & b p < 0.05):")
            for _, row in sig_rows.iterrows():
                print(f"    {row['parameter']} x {row['dass_var']} ({row['model_name']}): "
                      f"a={row['a_coef']:.3f} (p={row['a_p']:.4f}), "
                      f"b={row['b_coef']:.3f} (p={row['b_p']:.4f})")

        print(f"\n  Output: {output_file}")

    return results_df


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    if verbose:
        print("=" * 70)
        print("STROOP EX-GAUSSIAN PATH SUITE")
        print("=" * 70)

    if analysis and analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")

    target = ANALYSES.get(analysis or 'path_models')
    results = {}
    if target:
        results[target.name] = target.function(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("STROOP EX-GAUSSIAN PATH SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    print("\nAvailable Stroop Ex-Gaussian Path Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stroop Ex-Gaussian Path Analysis")
    parser.add_argument('--analysis', '-a', type=str, default=None, help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true', help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true', help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
