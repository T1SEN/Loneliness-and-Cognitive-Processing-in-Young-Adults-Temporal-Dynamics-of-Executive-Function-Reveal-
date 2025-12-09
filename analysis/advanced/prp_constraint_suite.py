"""
PRP Constraint Violation Analysis Suite
=======================================

Analyzes constraint violations in the PRP dual-task paradigm.
T2-pressed-while-T1-pending indicates a failure to maintain task set.

Key Research Questions:
1. Does UCLA affect constraint violation rate?
2. Are violations more common at short SOAs?
3. Does UCLA × Gender interact with violations?

Usage:
    python -m analysis.advanced.prp_constraint_suite

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
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR, find_interaction_term
)
from analysis.preprocessing.loaders import ensure_participant_id
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "prp_constraint"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class AnalysisSpec:
    name: str
    description: str
    function: Callable
    source_script: str

ANALYSES: Dict[str, AnalysisSpec] = {}

def register_analysis(name: str, description: str, source_script: str = "prp_constraint_suite.py"):
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(name=name, description=description, function=func, source_script=source_script)
        return func
    return decorator


def load_prp_trials() -> pd.DataFrame:
    """Load PRP trials with constraint violation flag."""
    path = RESULTS_DIR / '4a_prp_trials.csv'
    df = pd.read_csv(path, encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    # Check for violation flag
    violation_col = None
    for col in ['t2_pressed_while_t1_pending', 't2pressedwhilet1pending']:
        if col in df.columns:
            violation_col = col
            break

    if violation_col:
        if df[violation_col].dtype == object:
            df['constraint_violation'] = df[violation_col].astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df['constraint_violation'] = df[violation_col].fillna(False).astype(bool)
    else:
        df['constraint_violation'] = False
        print("  Warning: constraint violation column not found")

    print(f"  PRP trials loaded: {len(df)} trials, {df['participant_id'].nunique()} participants")
    print(f"  Constraint violations: {df['constraint_violation'].sum()}")

    return df


def load_master_with_standardization() -> pd.DataFrame:
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)
    return master


@register_analysis("violation_metrics", "Compute constraint violation rates per participant")
def analyze_violation_metrics(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 70)
        print("PRP CONSTRAINT VIOLATION METRICS")
        print("=" * 70)

    trials = load_prp_trials()

    results = []
    for pid, pdata in trials.groupby('participant_id'):
        n_trials = len(pdata)
        n_violations = pdata['constraint_violation'].sum()

        results.append({
            'participant_id': pid,
            'n_trials': n_trials,
            'n_violations': n_violations,
            'violation_rate': n_violations / n_trials if n_trials > 0 else 0,
        })

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  N = {len(results_df)}")
        print(f"  Mean violation rate: {results_df['violation_rate'].mean()*100:.2f}% (SD={results_df['violation_rate'].std()*100:.2f})")

    results_df.to_csv(OUTPUT_DIR / "violation_metrics.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"  Output: {OUTPUT_DIR / 'violation_metrics.csv'}")

    return results_df


@register_analysis("ucla_violations", "Test UCLA effects on constraint violations (DASS-controlled)")
def analyze_ucla_violations(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECTS ON CONSTRAINT VIOLATIONS")
        print("=" * 70)

    metrics_file = OUTPUT_DIR / "violation_metrics.csv"
    if not metrics_file.exists():
        analyze_violation_metrics(verbose=False)
    metrics = pd.read_csv(metrics_file)

    master = load_master_with_standardization()
    merged = master.merge(metrics, on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  N = {len(merged)}")

    try:
        formula = "violation_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=merged).fit(cov_type='HC3')

        results = {
            'beta_ucla': model.params.get('z_ucla', np.nan),
            'p_ucla': model.pvalues.get('z_ucla', np.nan),
            'r_squared': model.rsquared,
        }

        int_term = find_interaction_term(model.params.index)
        if int_term:
            results['beta_interaction'] = model.params[int_term]
            results['p_interaction'] = model.pvalues[int_term]

        if verbose:
            sig = "*" if results['p_ucla'] < 0.05 else ""
            print(f"  UCLA main: beta={results['beta_ucla']:.4f}, p={results['p_ucla']:.4f}{sig}")
            if 'p_interaction' in results:
                sig = "*" if results['p_interaction'] < 0.05 else ""
                print(f"  UCLA × Gender: beta={results['beta_interaction']:.4f}, p={results['p_interaction']:.4f}{sig}")

        results_df = pd.DataFrame([results])
        results_df.to_csv(OUTPUT_DIR / "ucla_violations.csv", index=False, encoding='utf-8-sig')

        return results_df

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return pd.DataFrame()


@register_analysis("summary", "Summary report")
def analyze_summary(verbose: bool = True) -> Dict:
    if verbose:
        print("\n" + "=" * 70)
        print("PRP CONSTRAINT VIOLATION SUMMARY")
        print("=" * 70)

    summary = {}

    metrics_file = OUTPUT_DIR / "violation_metrics.csv"
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)
        summary['mean_violation_rate'] = metrics['violation_rate'].mean()
        summary['n_with_violations'] = (metrics['n_violations'] > 0).sum()

    ucla_file = OUTPUT_DIR / "ucla_violations.csv"
    if ucla_file.exists():
        ucla = pd.read_csv(ucla_file)
        summary['ucla_p'] = ucla['p_ucla'].values[0] if 'p_ucla' in ucla.columns else np.nan
        summary['ucla_significant'] = summary['ucla_p'] < 0.05 if pd.notna(summary['ucla_p']) else False

    if verbose:
        if 'mean_violation_rate' in summary:
            print(f"  Mean violation rate: {summary['mean_violation_rate']*100:.2f}%")
        print(f"  UCLA significant: {summary.get('ucla_significant', 'N/A')}")

    import json
    with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict:
    if verbose:
        print("=" * 70)
        print("PRP CONSTRAINT VIOLATION SUITE")
        print("=" * 70)

    results = {}
    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}")
        results[analysis] = ANALYSES[analysis].function(verbose=verbose)
    else:
        for name in ['violation_metrics', 'ucla_violations', 'summary']:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("PRP CONSTRAINT SUITE COMPLETE")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    print("\nAvailable PRP Constraint Analyses:")
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--list', '-l', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
