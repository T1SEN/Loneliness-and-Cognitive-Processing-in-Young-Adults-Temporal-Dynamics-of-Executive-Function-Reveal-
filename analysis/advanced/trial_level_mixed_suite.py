"""
Trial-Level Mixed Effects Analysis Suite
=========================================

Tests whether UCLA loneliness affects learning curves (PROCESS) vs
baseline performance (STATE) using multilevel/hierarchical linear models.

Key Research Questions:
1. Does UCLA affect within-task performance trajectories (learning curves)?
2. Do lonely individuals show slower learning across trials?
3. Is the UCLA effect on RT/accuracy static (STATE) or dynamic (PROCESS)?

Model Structure:
    Level 1 (within-person): RT_ij = β0j + β1j*trial_number + ε_ij
    Level 2 (between-person):
        β0j = γ00 + γ01*UCLA + γ02*Gender + γ03*DASS + u0j
        β1j = γ10 + γ11*UCLA + γ12*Gender + u1j

Key Interpretation:
    - γ01 significant: UCLA affects baseline (STATE effect)
    - γ11 significant: UCLA affects learning slope (PROCESS effect)
    - If only γ01: Loneliness impairs initial performance but not learning
    - If γ11 significant: Loneliness impairs adaptive improvement over time

Analyses:
- wcst_learning: WCST accuracy learning curve × UCLA
- stroop_learning: Stroop RT learning curve × UCLA
- prp_learning: PRP bottleneck trajectory × UCLA
- summary: Integration across tasks

Usage:
    python -m analysis.advanced.trial_level_mixed_suite
    python -m analysis.advanced.trial_level_mixed_suite --analysis wcst_learning
    python -m analysis.advanced.trial_level_mixed_suite --list

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
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, PRP_RT_MAX, STROOP_RT_MAX
)
from analysis.preprocessing.trial_loaders import load_wcst_trials, load_stroop_trials, load_prp_trials
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "trial_level_mixed"
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


def register_analysis(name: str, description: str, source_script: str = "trial_level_mixed_suite.py"):
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
# DATA LOADING AND PREPARATION
# =============================================================================

def load_participant_data() -> pd.DataFrame:
    """Load and prepare participant-level data."""
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

    # Keep only needed columns
    keep_cols = ['participant_id', 'z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
    master = master[[c for c in keep_cols if c in master.columns]].copy()

    return master


def prepare_wcst_trials() -> pd.DataFrame:
    """Load and prepare WCST trial-level data."""
    trials_result = load_wcst_trials()
    # Handle tuple return (DataFrame, metadata)
    trials = trials_result[0] if isinstance(trials_result, tuple) else trials_result
    if trials.empty:
        return pd.DataFrame()

    # Filter valid trials
    trials = trials[trials['timeout'] == False].copy()

    # Handle RT column name variations
    rt_col = None
    for col in ['reactiontimems', 'reaction_time_ms', 'rt_ms', 'rt']:
        if col in trials.columns:
            rt_col = col
            break

    if rt_col is None:
        print("  [WARN] No RT column found in WCST trials")
        return pd.DataFrame()

    trials['rt_ms'] = pd.to_numeric(trials[rt_col], errors='coerce')
    trials = trials[(trials['rt_ms'] > DEFAULT_RT_MIN) & (trials['rt_ms'] < 5000)]

    # Handle participant_id
    if 'participantid' in trials.columns and 'participant_id' not in trials.columns:
        trials['participant_id'] = trials['participantid']

    # Create trial number (within participant)
    trials = trials.sort_values(['participant_id', 'trialindex' if 'trialindex' in trials.columns else 'trial_index'])
    trials['trial_num'] = trials.groupby('participant_id').cumcount() + 1

    # Normalize trial number (0-1 scale for interpretability)
    trials['trial_norm'] = trials.groupby('participant_id')['trial_num'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    # Correct indicator
    trials['correct_num'] = trials['correct'].astype(int)

    return trials


def prepare_stroop_trials() -> pd.DataFrame:
    """Load and prepare Stroop trial-level data."""
    trials_result = load_stroop_trials()
    trials = trials_result[0] if isinstance(trials_result, tuple) else trials_result
    if trials.empty:
        return pd.DataFrame()

    # Filter valid trials
    trials = trials[trials['timeout'] == False].copy()

    # Handle RT column
    rt_col = None
    for col in ['rt_ms', 'rt', 'stim_rt']:
        if col in trials.columns:
            rt_col = col
            break

    if rt_col is None:
        print("  [WARN] No RT column found in Stroop trials")
        return pd.DataFrame()

    trials['rt_ms'] = pd.to_numeric(trials[rt_col], errors='coerce')
    trials = trials[(trials['rt_ms'] > DEFAULT_RT_MIN) & (trials['rt_ms'] < STROOP_RT_MAX)]

    # Handle participant_id
    if 'participantid' in trials.columns and 'participant_id' not in trials.columns:
        trials['participant_id'] = trials['participantid']

    # Create trial number
    trials = trials.sort_values(['participant_id', 'trial_index' if 'trial_index' in trials.columns else 'trial'])
    trials['trial_num'] = trials.groupby('participant_id').cumcount() + 1

    # Normalize
    trials['trial_norm'] = trials.groupby('participant_id')['trial_num'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    # Condition indicator (incongruent = 1)
    if 'type' in trials.columns:
        trials['incongruent'] = (trials['type'].str.lower() == 'incongruent').astype(int)
    elif 'cond' in trials.columns:
        trials['incongruent'] = (trials['cond'].str.lower() == 'incongruent').astype(int)
    else:
        trials['incongruent'] = 0

    return trials


def prepare_prp_trials() -> pd.DataFrame:
    """Load and prepare PRP trial-level data."""
    trials_result = load_prp_trials()
    trials = trials_result[0] if isinstance(trials_result, tuple) else trials_result
    if trials.empty:
        return pd.DataFrame()

    # Filter valid trials
    timeout_cols = ['t1_timeout', 't2_timeout', 'timeout']
    for col in timeout_cols:
        if col in trials.columns:
            trials = trials[trials[col] == False]

    # Get T2 RT
    rt_col = None
    for col in ['t2_rt_ms', 't2_rt']:
        if col in trials.columns:
            rt_col = col
            break

    if rt_col is None:
        print("  [WARN] No T2 RT column found in PRP trials")
        return pd.DataFrame()

    trials['rt_ms'] = pd.to_numeric(trials[rt_col], errors='coerce')
    trials = trials[(trials['rt_ms'] > DEFAULT_RT_MIN) & (trials['rt_ms'] < PRP_RT_MAX)]

    # Handle participant_id
    if 'participantid' in trials.columns and 'participant_id' not in trials.columns:
        trials['participant_id'] = trials['participantid']

    # Create trial number
    trials = trials.sort_values(['participant_id', 'trial_index' if 'trial_index' in trials.columns else 'idx'])
    trials['trial_num'] = trials.groupby('participant_id').cumcount() + 1

    trials['trial_norm'] = trials.groupby('participant_id')['trial_num'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    # SOA indicator (short = 1)
    soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in trials.columns else 'soa_measured_ms' if 'soa_measured_ms' in trials.columns else 'soa'
    if soa_col in trials.columns:
        trials['soa_short'] = (trials[soa_col] <= 150).astype(int)
    else:
        trials['soa_short'] = 0

    return trials


# =============================================================================
# MIXED MODEL FUNCTIONS
# =============================================================================

def run_mixed_model(
    trials: pd.DataFrame,
    participants: pd.DataFrame,
    outcome: str,
    task_name: str,
    random_slopes: bool = True
) -> Dict:
    """
    Run mixed effects model for trial-level data.

    Model: outcome ~ trial_norm * z_ucla + trial_norm * gender + DASS + age + (1 + trial_norm | participant_id)
    """
    # Merge with participant data
    merged = trials.merge(participants, on='participant_id', how='inner')

    # Check sufficient data
    n_participants = merged['participant_id'].nunique()
    n_trials = len(merged)

    if n_participants < 30:
        return {'error': f'Insufficient participants (N={n_participants})'}

    print(f"\n    Data: {n_trials} trials from {n_participants} participants")

    # Formula
    formula = f"{outcome} ~ trial_norm * z_ucla + trial_norm * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

    try:
        # Fit mixed model
        if random_slopes:
            model = MixedLM.from_formula(
                formula,
                data=merged,
                groups=merged['participant_id'],
                re_formula="~trial_norm"  # Random intercept + slope
            )
        else:
            model = MixedLM.from_formula(
                formula,
                data=merged,
                groups=merged['participant_id']
            )

        fit = model.fit(method='lbfgs', maxiter=500)

        # Extract results
        result = {
            'task': task_name,
            'outcome': outcome,
            'n_participants': n_participants,
            'n_trials': n_trials,
            'aic': fit.aic,
            'bic': fit.bic,
            'llf': fit.llf,
        }

        # Fixed effects
        for param in fit.params.index:
            clean_name = param.replace('[', '_').replace(']', '').replace(':', '_x_')
            result[f'fe_{clean_name}_beta'] = fit.params[param]
            result[f'fe_{clean_name}_se'] = fit.bse[param]
            result[f'fe_{clean_name}_p'] = fit.pvalues[param]

        # Key effects
        # UCLA baseline effect (STATE)
        result['ucla_baseline_beta'] = fit.params.get('z_ucla', np.nan)
        result['ucla_baseline_p'] = fit.pvalues.get('z_ucla', np.nan)

        # UCLA × trial interaction (PROCESS)
        interaction_terms = [t for t in fit.params.index if 'trial_norm' in t and 'z_ucla' in t]
        if interaction_terms:
            int_term = interaction_terms[0]
            result['ucla_slope_beta'] = fit.params.get(int_term, np.nan)
            result['ucla_slope_p'] = fit.pvalues.get(int_term, np.nan)
        else:
            result['ucla_slope_beta'] = np.nan
            result['ucla_slope_p'] = np.nan

        # Random effects variance
        result['re_intercept_var'] = fit.cov_re.iloc[0, 0] if hasattr(fit, 'cov_re') and fit.cov_re is not None else np.nan
        if random_slopes and hasattr(fit, 'cov_re') and fit.cov_re is not None and fit.cov_re.shape[0] > 1:
            result['re_slope_var'] = fit.cov_re.iloc[1, 1]
        else:
            result['re_slope_var'] = np.nan

        return result

    except Exception as e:
        return {'error': str(e), 'task': task_name}


# =============================================================================
# REGISTERED ANALYSES
# =============================================================================

@register_analysis(
    name="wcst_learning",
    description="WCST accuracy learning curve × UCLA"
)
def wcst_learning_analysis():
    """Test whether UCLA affects WCST learning trajectory."""
    print("\n" + "="*70)
    print("WCST LEARNING CURVE × UCLA (Mixed Effects)")
    print("="*70)
    print("  Model: correct ~ trial_norm * z_ucla + ... + (1 + trial_norm | participant)")

    trials = prepare_wcst_trials()
    participants = load_participant_data()

    if trials.empty:
        print("  [ERROR] No WCST trial data available")
        return {'error': 'No data'}

    # Run analysis
    result = run_mixed_model(trials, participants, 'correct_num', 'WCST')

    if 'error' not in result:
        print(f"\n    === Results ===")
        print(f"    AIC: {result['aic']:.1f}, BIC: {result['bic']:.1f}")
        print(f"\n    UCLA Baseline (STATE) Effect:")
        print(f"      β = {result['ucla_baseline_beta']:.4f}, p = {result['ucla_baseline_p']:.4f}")
        print(f"\n    UCLA × Trial (PROCESS) Effect:")
        print(f"      β = {result['ucla_slope_beta']:.4f}, p = {result['ucla_slope_p']:.4f}")

        # Interpretation
        if result['ucla_slope_p'] < 0.05:
            print("\n    *** SIGNIFICANT PROCESS EFFECT ***")
            print("    UCLA affects learning rate (how performance changes over trials)")
        elif result['ucla_baseline_p'] < 0.05:
            print("\n    * Significant STATE effect only *")
            print("    UCLA affects baseline but not learning trajectory")
        else:
            print("\n    No significant UCLA effects on learning")
    else:
        print(f"  [ERROR] {result['error']}")

    # Save result
    pd.DataFrame([result]).to_csv(OUTPUT_DIR / 'wcst_learning.csv', index=False, encoding='utf-8-sig')

    return result


@register_analysis(
    name="stroop_learning",
    description="Stroop RT learning curve × UCLA (incongruent trials)"
)
def stroop_learning_analysis():
    """Test whether UCLA affects Stroop RT trajectory."""
    print("\n" + "="*70)
    print("STROOP RT TRAJECTORY × UCLA (Mixed Effects)")
    print("="*70)
    print("  Model: rt_ms ~ trial_norm * z_ucla + ... + (1 + trial_norm | participant)")

    trials = prepare_stroop_trials()
    participants = load_participant_data()

    if trials.empty:
        print("  [ERROR] No Stroop trial data available")
        return {'error': 'No data'}

    # Focus on incongruent trials for cleaner effect
    inc_trials = trials[trials['incongruent'] == 1].copy()
    print(f"    Using incongruent trials only: {len(inc_trials)} trials")

    if len(inc_trials) < 100:
        print("  [WARN] Using all trials due to insufficient incongruent trials")
        inc_trials = trials

    # Run analysis
    result = run_mixed_model(inc_trials, participants, 'rt_ms', 'Stroop')

    if 'error' not in result:
        print(f"\n    === Results ===")
        print(f"    AIC: {result['aic']:.1f}, BIC: {result['bic']:.1f}")
        print(f"\n    UCLA Baseline (STATE) Effect:")
        print(f"      β = {result['ucla_baseline_beta']:.2f} ms, p = {result['ucla_baseline_p']:.4f}")
        print(f"\n    UCLA × Trial (PROCESS) Effect:")
        print(f"      β = {result['ucla_slope_beta']:.2f} ms, p = {result['ucla_slope_p']:.4f}")

        if result['ucla_slope_p'] < 0.05:
            print("\n    *** SIGNIFICANT PROCESS EFFECT ***")
            if result['ucla_slope_beta'] > 0:
                print("    Higher UCLA → RT increases more across trials (worse fatigue)")
            else:
                print("    Higher UCLA → RT decreases less across trials (less learning)")
        elif result['ucla_baseline_p'] < 0.05:
            print("\n    * Significant STATE effect only *")
        else:
            print("\n    No significant UCLA effects")
    else:
        print(f"  [ERROR] {result['error']}")

    pd.DataFrame([result]).to_csv(OUTPUT_DIR / 'stroop_learning.csv', index=False, encoding='utf-8-sig')

    return result


@register_analysis(
    name="prp_learning",
    description="PRP T2 RT trajectory × UCLA (short SOA)"
)
def prp_learning_analysis():
    """Test whether UCLA affects PRP bottleneck trajectory."""
    print("\n" + "="*70)
    print("PRP T2 RT TRAJECTORY × UCLA (Mixed Effects)")
    print("="*70)
    print("  Model: rt_ms ~ trial_norm * z_ucla + ... + (1 + trial_norm | participant)")

    trials = prepare_prp_trials()
    participants = load_participant_data()

    if trials.empty:
        print("  [ERROR] No PRP trial data available")
        return {'error': 'No data'}

    # Focus on short SOA trials (where bottleneck is most evident)
    short_trials = trials[trials['soa_short'] == 1].copy()
    print(f"    Using short SOA trials: {len(short_trials)} trials")

    if len(short_trials) < 100:
        print("  [WARN] Using all trials due to insufficient short SOA trials")
        short_trials = trials

    # Run analysis
    result = run_mixed_model(short_trials, participants, 'rt_ms', 'PRP')

    if 'error' not in result:
        print(f"\n    === Results ===")
        print(f"    AIC: {result['aic']:.1f}, BIC: {result['bic']:.1f}")
        print(f"\n    UCLA Baseline (STATE) Effect:")
        print(f"      β = {result['ucla_baseline_beta']:.2f} ms, p = {result['ucla_baseline_p']:.4f}")
        print(f"\n    UCLA × Trial (PROCESS) Effect:")
        print(f"      β = {result['ucla_slope_beta']:.2f} ms, p = {result['ucla_slope_p']:.4f}")

        if result['ucla_slope_p'] < 0.05:
            print("\n    *** SIGNIFICANT PROCESS EFFECT ***")
        elif result['ucla_baseline_p'] < 0.05:
            print("\n    * Significant STATE effect only *")
        else:
            print("\n    No significant UCLA effects")
    else:
        print(f"  [ERROR] {result['error']}")

    pd.DataFrame([result]).to_csv(OUTPUT_DIR / 'prp_learning.csv', index=False, encoding='utf-8-sig')

    return result


@register_analysis(
    name="gender_stratified",
    description="Gender-stratified learning curve analysis"
)
def gender_stratified_analysis():
    """Run learning curve analysis separately by gender."""
    print("\n" + "="*70)
    print("GENDER-STRATIFIED LEARNING CURVES")
    print("="*70)

    trials = prepare_wcst_trials()
    participants = load_participant_data()

    if trials.empty:
        return {'error': 'No data'}

    # Merge
    merged = trials.merge(participants, on='participant_id', how='inner')

    results = []

    for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
        gender_data = merged[merged['gender_male'] == gender_val].copy()
        n_p = gender_data['participant_id'].nunique()

        print(f"\n  {gender_label} (N={n_p}):")

        if n_p < 20:
            print("    [SKIP] Insufficient participants")
            continue

        # Simple formula without gender
        formula = "correct_num ~ trial_norm * z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"

        try:
            model = MixedLM.from_formula(
                formula,
                data=gender_data,
                groups=gender_data['participant_id'],
                re_formula="~trial_norm"
            )
            fit = model.fit(method='lbfgs', maxiter=500)

            # Find interaction term
            int_terms = [t for t in fit.params.index if 'trial_norm' in t and 'z_ucla' in t]
            int_term = int_terms[0] if int_terms else None

            result = {
                'gender': gender_label,
                'n_participants': n_p,
                'n_trials': len(gender_data),
                'ucla_baseline_beta': fit.params.get('z_ucla', np.nan),
                'ucla_baseline_p': fit.pvalues.get('z_ucla', np.nan),
                'ucla_slope_beta': fit.params.get(int_term, np.nan) if int_term else np.nan,
                'ucla_slope_p': fit.pvalues.get(int_term, np.nan) if int_term else np.nan,
            }

            print(f"    UCLA Baseline: β = {result['ucla_baseline_beta']:.4f}, p = {result['ucla_baseline_p']:.4f}")
            print(f"    UCLA × Trial:  β = {result['ucla_slope_beta']:.4f}, p = {result['ucla_slope_p']:.4f}")

            sig = "*" if result['ucla_slope_p'] < 0.05 else ""
            if sig:
                print(f"    *** {gender_label}: Significant PROCESS effect ***")

            results.append(result)

        except Exception as e:
            print(f"    [ERROR] {e}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'gender_stratified_learning.csv', index=False, encoding='utf-8-sig')

    return results


@register_analysis(
    name="summary",
    description="Summary across all trial-level mixed effects analyses"
)
def summary_analysis():
    """Run all analyses and summarize."""
    print("\n" + "="*70)
    print("TRIAL-LEVEL MIXED EFFECTS: SUMMARY")
    print("="*70)

    # Run all task analyses
    wcst_result = wcst_learning_analysis()
    stroop_result = stroop_learning_analysis()
    prp_result = prp_learning_analysis()

    # Also run gender-stratified
    gender_results = gender_stratified_analysis()

    # Compile summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY: PROCESS vs STATE")
    print("="*70)

    all_results = []
    for result in [wcst_result, stroop_result, prp_result]:
        if 'error' not in result:
            all_results.append(result)

    if not all_results:
        print("  No valid results to summarize")
        return {'error': 'No results'}

    print("\n  Task-by-Task Results:")
    print("  " + "-"*60)

    process_effects = []
    state_effects = []

    for r in all_results:
        task = r['task']
        state_p = r['ucla_baseline_p']
        process_p = r['ucla_slope_p']

        state_sig = "*" if state_p < 0.05 else "†" if state_p < 0.10 else ""
        process_sig = "*" if process_p < 0.05 else "†" if process_p < 0.10 else ""

        print(f"\n  {task}:")
        print(f"    STATE (UCLA baseline):    β = {r['ucla_baseline_beta']:.4f}, p = {state_p:.4f} {state_sig}")
        print(f"    PROCESS (UCLA × trial):   β = {r['ucla_slope_beta']:.4f}, p = {process_p:.4f} {process_sig}")

        if process_p < 0.05:
            process_effects.append(task)
        if state_p < 0.05:
            state_effects.append(task)

    print("\n  " + "="*60)
    print("  CONCLUSION:")
    if process_effects:
        print(f"    PROCESS effects significant in: {', '.join(process_effects)}")
        print("    → UCLA affects learning/adaptation over time")
    if state_effects:
        print(f"    STATE effects significant in: {', '.join(state_effects)}")
        print("    → UCLA affects baseline performance")
    if not process_effects and not state_effects:
        print("    No significant UCLA effects on learning curves")
        print("    → Loneliness does not affect within-task trajectories")

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_DIR / 'summary.csv', index=False, encoding='utf-8-sig')

    print(f"\n  Results saved to: {OUTPUT_DIR}")

    return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def list_analyses():
    """List all available analyses."""
    print("\nAvailable analyses in trial_level_mixed_suite:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name:25s} - {spec.description}")
    print("\nUsage: python -m analysis.advanced.trial_level_mixed_suite --analysis <name>")


def run_analysis(name: str) -> Optional[Dict]:
    """Run a specific analysis by name."""
    if name not in ANALYSES:
        print(f"Unknown analysis: {name}")
        list_analyses()
        return None

    spec = ANALYSES[name]
    print(f"\nRunning: {spec.description}")
    return spec.function()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trial-Level Mixed Effects Analysis Suite"
    )
    parser.add_argument("--analysis", "-a", help="Run specific analysis")
    parser.add_argument("--list", "-l", action="store_true", help="List available analyses")
    parser.add_argument("--all", action="store_true", help="Run all analyses")

    args = parser.parse_args()

    if args.list:
        list_analyses()
    elif args.analysis:
        run_analysis(args.analysis)
    elif args.all:
        summary_analysis()
    else:
        summary_analysis()


if __name__ == "__main__":
    main()
