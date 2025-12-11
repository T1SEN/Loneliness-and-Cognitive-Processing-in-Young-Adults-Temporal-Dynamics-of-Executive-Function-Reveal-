"""
WCST Reinforcement Learning Path Analysis
=========================================

Builds SEM-style pathways linking UCLA loneliness, DASS symptoms, and the
parameters recovered from WCST reinforcement learning models. This keeps the
original DASS-controlled modeling untouched while surfacing path outcomes.
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

from .wcst_rl_modeling_dass import (
    load_rl_data,
    analyze_rescorla_wagner,
    analyze_asymmetric_learning,
)
from ._utils import BASE_OUTPUT, DEFAULT_PATH_COVARIATES, run_path_models

RL_DASS_DIR = BASE_OUTPUT / "wcst_rl_modeling"
OUTPUT_DIR = BASE_OUTPUT / "wcst_rl_modeling_path"
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
    description="Loneliness↔DASS path analysis for RL parameters"
)
def analyze_path_models(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 70)
        print("WCST RL PATH MODELS")
        print("=" * 70)

    master = load_rl_data()

    param_files = {
        'basic': RL_DASS_DIR / "rw_basic_parameters.csv",
        'asymmetric': RL_DASS_DIR / "rw_asymmetric_parameters.csv",
    }

    param_map = {
        'basic': [
            ('alpha', 'Learning rate (alpha)'),
            ('beta', 'Inverse temperature (beta)'),
        ],
        'asymmetric': [
            ('alpha_pos', 'Positive learning rate'),
            ('alpha_neg', 'Negative learning rate'),
            ('alpha_asymmetry', 'Alpha asymmetry'),
            ('beta', 'Inverse temperature (beta)'),
        ],
    }

    dass_targets = ['z_dass_dep', 'z_dass_anx', 'z_dass_str']
    all_records = []

    for model_type, param_file in param_files.items():
        if not param_file.exists():
            if verbose:
                print(f"  {model_type} parameters missing - fitting source model...")
            if model_type == 'basic':
                analyze_rescorla_wagner(verbose=False)
            else:
                analyze_asymmetric_learning(verbose=False)

        if not param_file.exists():
            if verbose:
                print(f"  {model_type}: parameter file still unavailable, skipping.")
            continue

        params = pd.read_csv(param_file)
        merged = master.merge(params, on='participant_id', how='inner')

        if len(merged) < 30:
            if verbose:
                print(f"  {model_type}: insufficient merged sample (N={len(merged)}).")
            continue

        available_dass = [col for col in dass_targets if col in merged.columns]
        if not available_dass:
            if verbose:
                print("  No DASS z-scores available.")
            return pd.DataFrame()

        for param_col, param_label in param_map.get(model_type, []):
            if param_col not in merged.columns:
                continue

            for dass_col in available_dass:
                path_results = run_path_models(
                    merged,
                    param_col=param_col,
                    param_label=param_label,
                    module_tag=f"wcst_rl_path_{model_type}",
                    dass_var=dass_col,
                    covariates=DEFAULT_PATH_COVARIATES,
                    min_n=30,
                )
                for rec in path_results:
                    rec['model_type'] = model_type
                all_records.extend(path_results)

    if not all_records:
        if verbose:
            print("  No valid RL path models were fit.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_records)
    output_file = OUTPUT_DIR / "wcst_rl_path_models.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    if verbose:
        sig_rows = results_df[(results_df['a_p'] < 0.05) & (results_df['b_p'] < 0.05)]
        if len(sig_rows) > 0:
            print("\n  Candidate mediated RL paths (a & b p < 0.05):")
            for _, row in sig_rows.iterrows():
                print(f"    {row['model_type']} - {row['parameter']} x {row['dass_var']} ({row['model_name']}): "
                      f"a={row['a_coef']:.3f} (p={row['a_p']:.4f}), "
                      f"b={row['b_coef']:.3f} (p={row['b_p']:.4f})")

        print(f"\n  Output: {output_file}")

    return results_df


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    if verbose:
        print("=" * 70)
        print("WCST RL PATH SUITE")
        print("=" * 70)

    if analysis and analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")

    target = ANALYSES.get(analysis or 'path_models')
    results = {}
    if target:
        results[target.name] = target.function(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("WCST RL PATH SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    print("\nAvailable WCST RL Path Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WCST RL Path Analysis")
    parser.add_argument('--analysis', '-a', type=str, default=None, help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true', help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true', help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
