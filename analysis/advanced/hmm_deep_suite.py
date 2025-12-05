"""
HMM Deep Analysis Suite
=======================

심층 Hidden Markov Model 분석으로 주의 상태 역동성과 UCLA 외로움의 관계를 탐색.

Analyses:
1. model_comparison: 2-state vs 3-state HMM 비교 (BIC/AIC)
2. transition_analysis: 상태 전환 확률과 UCLA 관계 (DASS 통제)
3. state_duration: Lapse 상태 지속 시간 및 회복 패턴
4. recovery_dynamics: Lapse에서 Focus로의 회복 궤적

Usage:
    python -m analysis.advanced.hmm_deep_suite                    # Run all
    python -m analysis.advanced.hmm_deep_suite --analysis model_comparison
    python -m analysis.advanced.hmm_deep_suite --list

    from analysis.advanced import hmm_deep_suite
    hmm_deep_suite.run('model_comparison')
    hmm_deep_suite.run()  # All analyses

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
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "hmm_deep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable
    priority: int  # 1=high, 2=medium, 3=low


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, priority: int = 1):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            priority=priority
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_hmm_data() -> pd.DataFrame:
    """Load and prepare master dataset."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    return master


def load_wcst_trials() -> pd.DataFrame:
    """Load WCST trial data (best for HMM - most trials)."""
    path = RESULTS_DIR / '4b_wcst_trials.csv'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8')
    df.columns = df.columns.str.lower()

    # Handle duplicate participant_id columns
    if 'participantid' in df.columns and 'participant_id' in df.columns:
        df = df.drop(columns=['participantid'])
    elif 'participantid' in df.columns:
        df = df.rename(columns={'participantid': 'participant_id'})

    return df


# =============================================================================
# HMM FITTING UTILITIES
# =============================================================================

def fit_hmm(rt_sequence: np.ndarray, n_states: int = 2,
            n_iter: int = 100, random_state: int = 42) -> Tuple:
    """
    Fit Gaussian HMM to RT sequence.

    Returns:
        model, state_sequence, log_likelihood, bic, aic, converged
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        raise ImportError("hmmlearn package required. Install with: pip install hmmlearn")

    if len(rt_sequence) < 20:
        return None, None, None, None, None, False

    # Reshape for hmmlearn (needs column vector)
    X = rt_sequence.reshape(-1, 1)
    n_samples = len(X)

    # Initialize Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        init_params="stmc"
    )

    try:
        model.fit(X)
        state_sequence = model.predict(X)
        log_likelihood = model.score(X)

        # Compute BIC and AIC
        n_params = n_states**2 + 2*n_states - 1  # transmat + means + covars
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        aic = -2 * log_likelihood + 2 * n_params

        return model, state_sequence, log_likelihood, bic, aic, model.monitor_.converged

    except Exception as e:
        return None, None, None, None, None, False


def extract_state_info(model, state_sequence: np.ndarray) -> dict:
    """Extract interpretable state information from fitted HMM."""
    n_states = model.n_components
    means = model.means_.flatten()
    stds = np.sqrt(np.array([model.covars_[i].flatten()[0] for i in range(n_states)]))

    # Sort states by mean RT (Focus < Normal < Lapse/Hyperfocus)
    state_order = np.argsort(means)

    state_info = {
        'n_states': n_states,
        'state_order': state_order,
        'transmat': model.transmat_,
    }

    # State-specific metrics
    for i, idx in enumerate(state_order):
        state_name = ['fast', 'medium', 'slow'][i] if n_states == 3 else ['focus', 'lapse'][i]
        state_info[f'{state_name}_mean_rt'] = means[idx]
        state_info[f'{state_name}_sd_rt'] = stds[idx]
        state_info[f'{state_name}_occupancy'] = (state_sequence == idx).mean()

    return state_info


def extract_lapse_episodes(state_sequence: np.ndarray, lapse_state: int) -> List[dict]:
    """Extract lapse episode durations and recovery patterns."""
    episodes = []
    in_lapse = False
    lapse_start = 0

    for i, state in enumerate(state_sequence):
        if state == lapse_state and not in_lapse:
            in_lapse = True
            lapse_start = i
        elif state != lapse_state and in_lapse:
            episodes.append({
                'start': lapse_start,
                'duration': i - lapse_start,
                'recovery_trial': i,
                'trials_from_end': len(state_sequence) - i
            })
            in_lapse = False

    # Handle case where sequence ends in lapse
    if in_lapse:
        episodes.append({
            'start': lapse_start,
            'duration': len(state_sequence) - lapse_start,
            'recovery_trial': None,  # No recovery
            'trials_from_end': 0
        })

    return episodes


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="model_comparison",
    description="2-state vs 3-state HMM model comparison using BIC/AIC",
    priority=1
)
def analyze_model_comparison(verbose: bool = True) -> pd.DataFrame:
    """
    Compare 2-state and 3-state HMM models.

    2-state: Focus vs Lapse (기존)
    3-state: Focus vs Normal vs Lapse (확장)
             또는 Hyperfocus vs Focus vs Lapse
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HMM MODEL COMPARISON: 2-state vs 3-state")
        print("=" * 70)

    trials = load_wcst_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient trial data")
        return pd.DataFrame()

    # Sort trials
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'timestamp', 'trialnum']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols)

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    if rt_col not in trials.columns:
        if verbose:
            print("  No RT column found")
        return pd.DataFrame()

    trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

    if verbose:
        print(f"  Fitting 2-state and 3-state HMMs per participant...")
        print(f"  Total trials: {len(trials)}, Participants: {trials['participant_id'].nunique()}")

    comparison_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 50:  # Need enough trials for 3-state
            continue

        rts = pdata[rt_col].values
        rts = rts[~np.isnan(rts)]

        if len(rts) < 50:
            continue

        # Fit 2-state
        model_2, seq_2, ll_2, bic_2, aic_2, conv_2 = fit_hmm(rts, n_states=2)

        # Fit 3-state
        model_3, seq_3, ll_3, bic_3, aic_3, conv_3 = fit_hmm(rts, n_states=3)

        if model_2 is None or model_3 is None:
            continue

        # Model comparison
        delta_bic = bic_3 - bic_2  # Negative = 3-state preferred
        delta_aic = aic_3 - aic_2

        # Likelihood ratio test
        lr_stat = 2 * (ll_3 - ll_2)
        df = 6  # Additional parameters in 3-state (approx)
        lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df)

        comparison_results.append({
            'participant_id': pid,
            'n_trials': len(rts),
            # 2-state metrics
            'bic_2state': bic_2,
            'aic_2state': aic_2,
            'll_2state': ll_2,
            'converged_2state': conv_2,
            # 3-state metrics
            'bic_3state': bic_3,
            'aic_3state': aic_3,
            'll_3state': ll_3,
            'converged_3state': conv_3,
            # Comparison
            'delta_bic': delta_bic,
            'delta_aic': delta_aic,
            'lr_statistic': lr_stat,
            'lr_pvalue': lr_pvalue,
            'preferred_bic': '3-state' if delta_bic < 0 else '2-state',
            'preferred_aic': '3-state' if delta_aic < 0 else '2-state',
            'preferred_lr': '3-state' if lr_pvalue < 0.05 else '2-state'
        })

    results_df = pd.DataFrame(comparison_results)

    if len(results_df) == 0:
        if verbose:
            print("  No successful fits")
        return pd.DataFrame()

    # Summary
    n_prefer_3_bic = (results_df['preferred_bic'] == '3-state').sum()
    n_prefer_3_aic = (results_df['preferred_aic'] == '3-state').sum()
    n_prefer_3_lr = (results_df['preferred_lr'] == '3-state').sum()
    n_total = len(results_df)

    if verbose:
        print(f"\n  Successfully fitted: {n_total} participants")
        print(f"\n  Model Preference:")
        print(f"    BIC: {n_prefer_3_bic}/{n_total} prefer 3-state ({100*n_prefer_3_bic/n_total:.1f}%)")
        print(f"    AIC: {n_prefer_3_aic}/{n_total} prefer 3-state ({100*n_prefer_3_aic/n_total:.1f}%)")
        print(f"    LR Test: {n_prefer_3_lr}/{n_total} prefer 3-state ({100*n_prefer_3_lr/n_total:.1f}%)")
        print(f"\n  Mean ΔBIC: {results_df['delta_bic'].mean():.2f} (SD={results_df['delta_bic'].std():.2f})")

    # Save results
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding='utf-8-sig')

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('HMM Model Comparison: 2-state vs 3-state', fontsize=14, fontweight='bold')

    # ΔBIC distribution
    ax1 = axes[0]
    ax1.hist(results_df['delta_bic'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('ΔBIC (3-state - 2-state)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'BIC Comparison\n(< 0 favors 3-state)')

    # ΔAIC distribution
    ax2 = axes[1]
    ax2.hist(results_df['delta_aic'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('ΔAIC (3-state - 2-state)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'AIC Comparison\n(< 0 favors 3-state)')

    # Preference summary
    ax3 = axes[2]
    labels = ['BIC', 'AIC', 'LR Test']
    prefer_3 = [n_prefer_3_bic, n_prefer_3_aic, n_prefer_3_lr]
    prefer_2 = [n_total - n_prefer_3_bic, n_total - n_prefer_3_aic, n_total - n_prefer_3_lr]

    x = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width/2, prefer_2, width, label='2-state', color='#3498db')
    ax3.bar(x + width/2, prefer_3, width, label='3-state', color='#e74c3c')
    ax3.set_ylabel('Number of Participants')
    ax3.set_title('Model Preference by Criterion')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'model_comparison.csv'}")
        print(f"  Figure: {FIGURES_DIR / 'model_comparison.png'}")

    return results_df


@register_analysis(
    name="transition_analysis",
    description="Transition probability analysis with UCLA/DASS control",
    priority=1
)
def analyze_transitions(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze state transition probabilities and their relationship with UCLA.

    Key transitions:
    - P(Focus → Lapse): 주의 이탈 경향
    - P(Lapse → Focus): 주의 회복 능력
    - P(Lapse → Lapse): Lapse 지속 경향
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HMM TRANSITION PROBABILITY ANALYSIS")
        print("=" * 70)

    try:
        from hmmlearn import hmm as hmm_module
    except ImportError:
        if verbose:
            print("  hmmlearn package not available")
        return pd.DataFrame()

    master = load_hmm_data()
    trials = load_wcst_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient trial data")
        return pd.DataFrame()

    # Sort trials
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'timestamp']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols)

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

    if verbose:
        print(f"  Extracting transition probabilities per participant...")

    trans_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 50:
            continue

        rts = pdata[rt_col].values
        rts = rts[~np.isnan(rts)]

        if len(rts) < 50:
            continue

        # Fit 2-state HMM
        model, state_seq, ll, bic, aic, converged = fit_hmm(rts, n_states=2)

        if model is None or not converged:
            continue

        # Identify states
        means = model.means_.flatten()
        focus_state = np.argmin(means)
        lapse_state = np.argmax(means)

        transmat = model.transmat_

        trans_results.append({
            'participant_id': pid,
            'n_trials': len(rts),
            'converged': converged,
            # State characteristics
            'focus_mean_rt': means[focus_state],
            'lapse_mean_rt': means[lapse_state],
            'focus_occupancy': (state_seq == focus_state).mean(),
            'lapse_occupancy': (state_seq == lapse_state).mean(),
            # Transition probabilities
            'p_focus_to_focus': transmat[focus_state, focus_state],
            'p_focus_to_lapse': transmat[focus_state, lapse_state],
            'p_lapse_to_focus': transmat[lapse_state, focus_state],
            'p_lapse_to_lapse': transmat[lapse_state, lapse_state],
        })

    trans_df = pd.DataFrame(trans_results)

    if len(trans_df) < 20:
        if verbose:
            print(f"  Only {len(trans_df)} participants fitted")
        return pd.DataFrame()

    # Merge with master data
    merged = master.merge(trans_df, on='participant_id', how='inner')

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"\n  Transition Probability Descriptives:")
        print(f"    P(Focus → Lapse): {merged['p_focus_to_lapse'].mean():.3f} ± {merged['p_focus_to_lapse'].std():.3f}")
        print(f"    P(Lapse → Focus): {merged['p_lapse_to_focus'].mean():.3f} ± {merged['p_lapse_to_focus'].std():.3f}")
        print(f"    P(Lapse → Lapse): {merged['p_lapse_to_lapse'].mean():.3f} ± {merged['p_lapse_to_lapse'].std():.3f}")

    # Logit transform probabilities for regression
    epsilon = 1e-4
    for col in ['p_focus_to_lapse', 'p_lapse_to_focus', 'p_lapse_to_lapse']:
        prob = np.clip(merged[col], epsilon, 1 - epsilon)
        merged[f'{col}_logit'] = np.log(prob / (1 - prob))

    # Regression analysis with DASS control
    regression_results = []

    outcomes = [
        ('p_lapse_to_focus', 'P(Lapse → Focus) - Recovery'),
        ('p_lapse_to_lapse', 'P(Lapse → Lapse) - Persistence'),
        ('p_focus_to_lapse', 'P(Focus → Lapse) - Vulnerability'),
        ('lapse_occupancy', 'Lapse Occupancy')
    ]

    if verbose:
        print(f"\n  UCLA Effects on Transitions (DASS-controlled):")

    for outcome, label in outcomes:
        dv = f"{outcome}_logit" if outcome.startswith('p_') else outcome

        if dv not in merged.columns:
            continue

        try:
            formula = f"{dv} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            # Extract coefficients
            result = {
                'outcome': outcome,
                'outcome_label': label,
                'n': len(merged),
            }

            for param in ['z_ucla', 'z_ucla:C(gender_male)[T.1]', 'C(gender_male)[T.1]']:
                if param in model.params:
                    short_name = param.replace('z_', '').replace('C(gender_male)[T.1]', 'gender').replace(':gender', '_x_gender')
                    result[f'beta_{short_name}'] = model.params[param]
                    result[f'se_{short_name}'] = model.bse[param]
                    result[f'p_{short_name}'] = model.pvalues[param]

            result['r_squared'] = model.rsquared
            result['adj_r_squared'] = model.rsquared_adj

            regression_results.append(result)

            # Print key results
            if verbose and 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {label}: β={beta:.4f}, p={p:.4f} {sig}")

            # Save full model summary
            with open(OUTPUT_DIR / f"regression_{outcome}.txt", 'w', encoding='utf-8') as f:
                f.write(str(model.summary()))

        except Exception as e:
            if verbose:
                print(f"    {label}: ERROR - {e}")

    reg_df = pd.DataFrame(regression_results)

    # Save results
    trans_df.to_csv(OUTPUT_DIR / "transition_matrix.csv", index=False, encoding='utf-8-sig')
    merged.to_csv(OUTPUT_DIR / "transition_merged.csv", index=False, encoding='utf-8-sig')
    reg_df.to_csv(OUTPUT_DIR / "transition_regression.csv", index=False, encoding='utf-8-sig')

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Transition Probabilities by UCLA Loneliness', fontsize=14, fontweight='bold')

    # Add UCLA tertiles
    merged['ucla_tertile'] = pd.qcut(merged['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

    for idx, (outcome, label) in enumerate(outcomes[:4]):
        ax = axes[idx // 2, idx % 2]

        if outcome in merged.columns:
            # Box plot by UCLA tertile
            sns.boxplot(data=merged, x='ucla_tertile', y=outcome, palette='RdYlGn_r', ax=ax)
            ax.set_xlabel('UCLA Tertile')
            ax.set_ylabel(outcome.replace('_', ' ').title())
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "transition_by_ucla.png", dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'transition_regression.csv'}")
        print(f"  Figure: {FIGURES_DIR / 'transition_by_ucla.png'}")

    return reg_df


@register_analysis(
    name="state_duration",
    description="Lapse state duration and recovery patterns",
    priority=1
)
def analyze_state_duration(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze lapse episode duration and recovery patterns.

    Metrics:
    - Mean/Median lapse duration (trials)
    - Number of lapse episodes
    - Recovery success rate
    - First-try recovery rate
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HMM STATE DURATION ANALYSIS")
        print("=" * 70)

    master = load_hmm_data()
    trials = load_wcst_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient trial data")
        return pd.DataFrame()

    # Sort trials
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'timestamp']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols)

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

    if verbose:
        print(f"  Extracting lapse episode durations...")

    duration_results = []
    all_episodes = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 50:
            continue

        rts = pdata[rt_col].values
        rts = rts[~np.isnan(rts)]

        if len(rts) < 50:
            continue

        # Fit HMM
        model, state_seq, ll, bic, aic, converged = fit_hmm(rts, n_states=2)

        if model is None or not converged:
            continue

        # Identify lapse state
        means = model.means_.flatten()
        lapse_state = np.argmax(means)

        # Extract lapse episodes
        episodes = extract_lapse_episodes(state_seq, lapse_state)

        if len(episodes) == 0:
            continue

        # Episode metrics
        durations = [ep['duration'] for ep in episodes]
        recovered_episodes = [ep for ep in episodes if ep['recovery_trial'] is not None]

        result = {
            'participant_id': pid,
            'n_trials': len(rts),
            'n_lapse_episodes': len(episodes),
            'lapse_episodes_per_100_trials': len(episodes) / len(rts) * 100,
            'mean_lapse_duration': np.mean(durations),
            'median_lapse_duration': np.median(durations),
            'max_lapse_duration': np.max(durations),
            'sd_lapse_duration': np.std(durations) if len(durations) > 1 else 0,
            'n_recovered_episodes': len(recovered_episodes),
            'recovery_rate': len(recovered_episodes) / len(episodes) if len(episodes) > 0 else 0,
            'single_trial_lapses': sum(1 for d in durations if d == 1),
            'extended_lapses_3plus': sum(1 for d in durations if d >= 3),
        }

        # First-try recovery (lapse duration = 1)
        result['first_try_recovery_rate'] = result['single_trial_lapses'] / len(episodes) if len(episodes) > 0 else 0

        duration_results.append(result)

        # Store individual episodes for later analysis
        for ep in episodes:
            all_episodes.append({
                'participant_id': pid,
                **ep
            })

    duration_df = pd.DataFrame(duration_results)
    episodes_df = pd.DataFrame(all_episodes)

    if len(duration_df) < 20:
        if verbose:
            print(f"  Only {len(duration_df)} participants")
        return pd.DataFrame()

    # Merge with master
    merged = master.merge(duration_df, on='participant_id', how='inner')

    if verbose:
        print(f"  N = {len(merged)}")
        print(f"\n  Lapse Episode Descriptives:")
        print(f"    Mean episodes per participant: {merged['n_lapse_episodes'].mean():.1f} ± {merged['n_lapse_episodes'].std():.1f}")
        print(f"    Mean lapse duration: {merged['mean_lapse_duration'].mean():.2f} ± {merged['mean_lapse_duration'].std():.2f} trials")
        print(f"    First-try recovery rate: {merged['first_try_recovery_rate'].mean()*100:.1f}%")

    # Regression analysis
    regression_results = []

    outcomes = [
        ('mean_lapse_duration', 'Mean Lapse Duration'),
        ('n_lapse_episodes', 'Number of Lapse Episodes'),
        ('first_try_recovery_rate', 'First-Try Recovery Rate'),
        ('extended_lapses_3plus', 'Extended Lapses (3+ trials)')
    ]

    if verbose:
        print(f"\n  UCLA Effects on Lapse Duration (DASS-controlled):")

    for outcome, label in outcomes:
        if outcome not in merged.columns:
            continue

        try:
            formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            result = {
                'outcome': outcome,
                'outcome_label': label,
                'n': len(merged),
            }

            for param in ['z_ucla', 'z_ucla:C(gender_male)[T.1]']:
                if param in model.params:
                    short_name = param.replace('z_', '').replace(':C(gender_male)[T.1]', '_x_gender')
                    result[f'beta_{short_name}'] = model.params[param]
                    result[f'p_{short_name}'] = model.pvalues[param]

            result['r_squared'] = model.rsquared
            regression_results.append(result)

            if verbose and 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {label}: β={beta:.4f}, p={p:.4f} {sig}")

        except Exception as e:
            if verbose:
                print(f"    {label}: ERROR - {e}")

    reg_df = pd.DataFrame(regression_results)

    # Save results
    duration_df.to_csv(OUTPUT_DIR / "state_duration.csv", index=False, encoding='utf-8-sig')
    episodes_df.to_csv(OUTPUT_DIR / "lapse_episodes.csv", index=False, encoding='utf-8-sig')
    merged.to_csv(OUTPUT_DIR / "duration_merged.csv", index=False, encoding='utf-8-sig')
    reg_df.to_csv(OUTPUT_DIR / "duration_regression.csv", index=False, encoding='utf-8-sig')

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Lapse Episode Metrics by UCLA Loneliness', fontsize=14, fontweight='bold')

    merged['ucla_tertile'] = pd.qcut(merged['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

    # Mean duration by UCLA
    ax1 = axes[0, 0]
    sns.boxplot(data=merged, x='ucla_tertile', y='mean_lapse_duration', palette='RdYlGn_r', ax=ax1)
    ax1.set_xlabel('UCLA Tertile')
    ax1.set_ylabel('Mean Lapse Duration (trials)')
    ax1.set_title('Lapse Duration by UCLA')
    ax1.grid(True, alpha=0.3)

    # Episode count by UCLA
    ax2 = axes[0, 1]
    sns.boxplot(data=merged, x='ucla_tertile', y='n_lapse_episodes', palette='RdYlGn_r', ax=ax2)
    ax2.set_xlabel('UCLA Tertile')
    ax2.set_ylabel('Number of Lapse Episodes')
    ax2.set_title('Episode Count by UCLA')
    ax2.grid(True, alpha=0.3)

    # First-try recovery by UCLA
    ax3 = axes[1, 0]
    sns.boxplot(data=merged, x='ucla_tertile', y='first_try_recovery_rate', palette='RdYlGn', ax=ax3)
    ax3.set_xlabel('UCLA Tertile')
    ax3.set_ylabel('First-Try Recovery Rate')
    ax3.set_title('Recovery Efficiency by UCLA')
    ax3.grid(True, alpha=0.3)

    # Duration distribution histogram
    ax4 = axes[1, 1]
    for tertile, color in zip(['Low', 'Medium', 'High'], ['green', 'orange', 'red']):
        subset = merged[merged['ucla_tertile'] == tertile]
        ax4.hist(subset['mean_lapse_duration'], bins=15, alpha=0.5, label=f'{tertile} UCLA', color=color)
    ax4.set_xlabel('Mean Lapse Duration (trials)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Duration Distribution by UCLA')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "state_duration.png", dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'state_duration.csv'}")
        print(f"  Figure: {FIGURES_DIR / 'state_duration.png'}")

    return reg_df


@register_analysis(
    name="recovery_dynamics",
    description="Post-lapse recovery trajectory analysis",
    priority=2
)
def analyze_recovery_dynamics(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze recovery trajectory after lapse episodes.

    Questions:
    - How quickly do participants return to focused state?
    - Does UCLA moderate recovery speed?
    - Is there a gender × UCLA interaction in recovery?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HMM RECOVERY DYNAMICS ANALYSIS")
        print("=" * 70)

    master = load_hmm_data()
    trials = load_wcst_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient trial data")
        return pd.DataFrame()

    # Sort trials
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'timestamp']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols)

    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    correct_col = 'correct' if 'correct' in trials.columns else None

    trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

    if verbose:
        print(f"  Analyzing recovery trajectories...")

    recovery_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 50:
            continue

        pdata = pdata.reset_index(drop=True)
        rts = pdata[rt_col].values
        rts_clean = rts[~np.isnan(rts)]

        if len(rts_clean) < 50:
            continue

        # Fit HMM
        model, state_seq, ll, bic, aic, converged = fit_hmm(rts_clean, n_states=2)

        if model is None or not converged:
            continue

        # Identify states
        means = model.means_.flatten()
        lapse_state = np.argmax(means)
        focus_state = np.argmin(means)

        # Find lapse-to-focus transitions
        transitions = []
        for i in range(1, len(state_seq)):
            if state_seq[i-1] == lapse_state and state_seq[i] == focus_state:
                transitions.append(i)

        if len(transitions) < 3:
            continue

        # Analyze RT pattern around transitions (lag -2 to +5)
        lags = list(range(-2, 6))
        lag_rts = {lag: [] for lag in lags}

        for t in transitions:
            for lag in lags:
                idx = t + lag
                if 0 <= idx < len(rts_clean):
                    lag_rts[lag].append(rts_clean[idx])

        result = {
            'participant_id': pid,
            'n_trials': len(rts_clean),
            'n_transitions': len(transitions),
        }

        # Mean RT at each lag
        for lag in lags:
            if len(lag_rts[lag]) > 0:
                result[f'rt_lag{lag:+d}'] = np.mean(lag_rts[lag])

        # Recovery slope (lag 0 to lag +3)
        if 'rt_lag+0' in result and 'rt_lag+3' in result:
            result['recovery_slope'] = (result['rt_lag+3'] - result['rt_lag+0']) / 3

        recovery_results.append(result)

    recovery_df = pd.DataFrame(recovery_results)

    if len(recovery_df) < 20:
        if verbose:
            print(f"  Only {len(recovery_df)} participants with sufficient transitions")
        return pd.DataFrame()

    # Merge with master
    merged = master.merge(recovery_df, on='participant_id', how='inner')

    if verbose:
        print(f"  N = {len(merged)}")
        if 'recovery_slope' in merged.columns:
            print(f"\n  Recovery Slope: {merged['recovery_slope'].mean():.2f} ± {merged['recovery_slope'].std():.2f} ms/trial")

    # Regression: Does UCLA predict recovery slope?
    regression_results = []

    if 'recovery_slope' in merged.columns:
        try:
            formula = "recovery_slope ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            result = {
                'outcome': 'recovery_slope',
                'n': len(merged),
            }

            for param in ['z_ucla', 'z_ucla:C(gender_male)[T.1]']:
                if param in model.params:
                    short_name = param.replace('z_', '').replace(':C(gender_male)[T.1]', '_x_gender')
                    result[f'beta_{short_name}'] = model.params[param]
                    result[f'p_{short_name}'] = model.pvalues[param]

            result['r_squared'] = model.rsquared
            regression_results.append(result)

            if verbose and 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"\n  UCLA → Recovery Slope: β={beta:.4f}, p={p:.4f} {sig}")

            # Save model summary
            with open(OUTPUT_DIR / "recovery_regression.txt", 'w', encoding='utf-8') as f:
                f.write(str(model.summary()))

        except Exception as e:
            if verbose:
                print(f"  Regression error: {e}")

    reg_df = pd.DataFrame(regression_results)

    # Save results
    recovery_df.to_csv(OUTPUT_DIR / "recovery_dynamics.csv", index=False, encoding='utf-8-sig')
    merged.to_csv(OUTPUT_DIR / "recovery_merged.csv", index=False, encoding='utf-8-sig')
    if len(reg_df) > 0:
        reg_df.to_csv(OUTPUT_DIR / "recovery_regression.csv", index=False, encoding='utf-8-sig')

    # Visualization: Recovery trajectory by UCLA
    merged['ucla_tertile'] = pd.qcut(merged['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

    lag_cols = [c for c in merged.columns if c.startswith('rt_lag')]

    if len(lag_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

        for tertile in ['Low', 'Medium', 'High']:
            subset = merged[merged['ucla_tertile'] == tertile]

            lags = sorted([int(c.replace('rt_lag', '').replace('+', '')) for c in lag_cols])
            means = [subset[f'rt_lag{l:+d}'].mean() for l in lags if f'rt_lag{l:+d}' in subset.columns]
            sems = [subset[f'rt_lag{l:+d}'].sem() for l in lags if f'rt_lag{l:+d}' in subset.columns]
            valid_lags = [l for l in lags if f'rt_lag{l:+d}' in subset.columns]

            ax.errorbar(valid_lags, means, yerr=sems, label=f'{tertile} UCLA',
                       color=colors[tertile], marker='o', capsize=3, linewidth=2)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Lapse→Focus transition')
        ax.set_xlabel('Lag (trials relative to transition)', fontsize=12)
        ax.set_ylabel('Mean RT (ms)', fontsize=12)
        ax.set_title('Recovery Trajectory After Lapse-to-Focus Transition', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "recovery_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close()

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'recovery_dynamics.csv'}")
        print(f"  Figure: {FIGURES_DIR / 'recovery_trajectory.png'}")

    return reg_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run HMM deep analyses.
    """
    if verbose:
        print("=" * 70)
        print("HMM DEEP ANALYSIS SUITE")
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
        # Run by priority
        for priority in [1, 2, 3]:
            priority_analyses = {k: v for k, v in ANALYSES.items() if v.priority == priority}
            if priority_analyses:
                for name, spec in priority_analyses.items():
                    try:
                        results[name] = spec.function(verbose=verbose)
                    except Exception as e:
                        print(f"  ERROR in {name}: {e}")

    # Generate summary report
    if verbose:
        _generate_summary_report(results)

    if verbose:
        print("\n" + "=" * 70)
        print("HMM DEEP SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def _generate_summary_report(results: Dict[str, pd.DataFrame]):
    """Generate summary report."""
    lines = [
        "=" * 80,
        "HMM DEEP ANALYSIS SUITE - SUMMARY REPORT",
        "=" * 80,
        "",
        "KEY FINDINGS:",
        ""
    ]

    # Check for significant effects
    for name, df in results.items():
        if df is None or len(df) == 0:
            continue

        if 'p_ucla' in df.columns:
            sig_effects = df[df['p_ucla'] < 0.05]
            if len(sig_effects) > 0:
                lines.append(f"\n{name}:")
                for _, row in sig_effects.iterrows():
                    label = row.get('outcome_label', row.get('outcome', 'Unknown'))
                    beta = row.get('beta_ucla', row.get('beta_ucla_logit', 'N/A'))
                    p = row['p_ucla']
                    lines.append(f"  - {label}: β={beta:.4f}, p={p:.4f} *")

    lines.extend([
        "",
        "=" * 80,
        "INTERPRETATION:",
        "",
        "- If UCLA → HIGHER P(Lapse→Lapse): 외로운 사람들은 Lapse 상태에 더 오래 머무름",
        "- If UCLA → LOWER P(Lapse→Focus): 외로운 사람들은 주의 회복이 더 어려움",
        "- If UCLA → LONGER mean duration: 외로운 사람들의 Lapse 에피소드가 더 길음",
        "- If UCLA → HIGHER n_episodes: 외로운 사람들이 더 자주 주의 이탈",
        "",
        "=" * 80
    ])

    summary_text = "\n".join(lines)

    with open(OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)


def list_analyses():
    """List available analyses."""
    print("\nAvailable HMM Deep Analyses:")
    print("-" * 60)
    for priority in [1, 2, 3]:
        priority_analyses = {k: v for k, v in ANALYSES.items() if v.priority == priority}
        if priority_analyses:
            label = {1: 'High', 2: 'Medium', 3: 'Low'}[priority]
            print(f"\n  Priority {priority} ({label}):")
            for name, spec in priority_analyses.items():
                print(f"    {name}")
                print(f"      {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Deep Analysis Suite")
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
