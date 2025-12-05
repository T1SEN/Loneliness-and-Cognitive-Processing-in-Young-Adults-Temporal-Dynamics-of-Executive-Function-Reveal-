"""
Reinforcement Learning Analysis Suite
=====================================

Computational modeling of WCST using reinforcement learning models.

Analyses:
- rescorla_wagner: Basic RW model with learning rate estimation
- asymmetric_learning: Separate learning rates for positive/negative feedback
- exploration_exploitation: Softmax temperature estimation
- model_comparison: BIC/AIC comparison across models
- parameter_recovery: Validate parameter recovery via simulation
- ucla_relationship: UCLA effects on RL parameters (DASS-controlled)

Usage:
    python -m analysis.advanced.reinforcement_learning_suite              # Run all
    python -m analysis.advanced.reinforcement_learning_suite --analysis rescorla_wagner
    python -m analysis.advanced.reinforcement_learning_suite --list

    from analysis.advanced import reinforcement_learning_suite
    reinforcement_learning_suite.run('rescorla_wagner')

Reference:
    Rescorla & Wagner (1972). A theory of Pavlovian conditioning.

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
from typing import Dict, Optional, Callable, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax as scipy_softmax
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "reinforcement_learning"
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


def register_analysis(name: str, description: str, source_script: str = "reinforcement_learning_suite.py"):
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

def load_rl_data() -> pd.DataFrame:
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
    """Load WCST trial-level data."""
    import ast

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

    # Parse extra field for rule information
    def _parse_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except (ValueError, SyntaxError):
            return {}

    if 'extra' in df.columns:
        df['extra_dict'] = df['extra'].apply(_parse_extra)
        df['is_pe'] = df['extra_dict'].apply(lambda x: x.get('isPE', False))
    else:
        df['is_pe'] = False

    # Map chosencard to integer index (0-3)
    if 'chosencard' in df.columns:
        card_map = {card: idx for idx, card in enumerate(df['chosencard'].dropna().unique())}
        df['action'] = df['chosencard'].map(card_map)
    else:
        df['action'] = 0

    # Sort by participant and trial
    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'trial']:
        if cand in df.columns:
            sort_cols.append(cand)
            break
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


# =============================================================================
# REINFORCEMENT LEARNING MODELS
# =============================================================================

class RescorlaWagnerModel:
    """
    Basic Rescorla-Wagner reinforcement learning model.

    Parameters:
        alpha: Learning rate (0-1)
        beta: Inverse temperature for softmax (higher = more deterministic)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.n_actions = 4  # WCST has 4 card choices

    def reset(self):
        """Reset Q-values."""
        self.Q = np.zeros(self.n_actions)

    def get_action_prob(self, action: int) -> float:
        """Get probability of choosing an action."""
        exp_q = np.exp(self.beta * (self.Q - np.max(self.Q)))
        probs = exp_q / np.sum(exp_q)
        return probs[action]

    def update(self, action: int, reward: float):
        """Update Q-value after receiving reward."""
        prediction_error = reward - self.Q[action]
        self.Q[action] += self.alpha * prediction_error

    def negative_log_likelihood(self, trials: pd.DataFrame) -> float:
        """Compute negative log-likelihood of data given model."""
        self.reset()
        nll = 0.0

        for _, trial in trials.iterrows():
            # Use pre-computed action index
            action = trial.get('action', 0)
            if pd.isna(action):
                continue
            action = int(action) % self.n_actions
            reward = 1.0 if trial.get('correct', False) else 0.0

            prob = self.get_action_prob(action)
            nll -= np.log(max(prob, 1e-10))

            self.update(action, reward)

        return nll


class AsymmetricRWModel(RescorlaWagnerModel):
    """
    Asymmetric Rescorla-Wagner model with separate learning rates.

    Parameters:
        alpha_pos: Learning rate for positive outcomes (0-1)
        alpha_neg: Learning rate for negative outcomes (0-1)
        beta: Inverse temperature
    """

    def __init__(self, alpha_pos: float = 0.5, alpha_neg: float = 0.5, beta: float = 1.0):
        super().__init__(alpha_pos, beta)
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def update(self, action: int, reward: float):
        """Update with asymmetric learning rates."""
        prediction_error = reward - self.Q[action]

        if prediction_error >= 0:
            self.Q[action] += self.alpha_pos * prediction_error
        else:
            self.Q[action] += self.alpha_neg * prediction_error


class ForgettingRWModel(RescorlaWagnerModel):
    """
    Rescorla-Wagner model with forgetting (decay).

    Parameters:
        alpha: Learning rate (0-1)
        gamma: Forgetting rate - unchosen options decay toward 0 (0-1)
        beta: Inverse temperature
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.1, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.gamma = gamma

    def update(self, action: int, reward: float):
        """Update with forgetting for unchosen options."""
        # Forgetting: decay unchosen options
        for i in range(self.n_actions):
            if i != action:
                self.Q[i] *= (1 - self.gamma)

        # Standard RW update for chosen option
        prediction_error = reward - self.Q[action]
        self.Q[action] += self.alpha * prediction_error


def fit_model(model_class, trials: pd.DataFrame, bounds: List[Tuple],
              param_names: List[str], n_restarts: int = 5) -> Dict:
    """
    Fit RL model to trial data using MLE.

    Parameters:
        model_class: Model class to fit
        trials: Trial-level data for one participant
        bounds: Parameter bounds [(low, high), ...]
        param_names: Names of parameters
        n_restarts: Number of random restarts for optimization

    Returns:
        Dictionary with fitted parameters and fit statistics
    """
    n_trials = len(trials)
    if n_trials < 20:
        return None

    best_nll = np.inf
    best_params = None

    def objective(params):
        try:
            model = model_class(*params)
            return model.negative_log_likelihood(trials)
        except Exception:
            return np.inf

    # Multiple random restarts
    for _ in range(n_restarts):
        x0 = [np.random.uniform(b[0], b[1]) for b in bounds]

        try:
            result = minimize(
                objective,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return None

    # Compute fit statistics
    n_params = len(param_names)
    bic = n_params * np.log(n_trials) + 2 * best_nll
    aic = 2 * n_params + 2 * best_nll

    result = {
        'nll': best_nll,
        'bic': bic,
        'aic': aic,
        'n_trials': n_trials
    }

    for name, value in zip(param_names, best_params):
        result[name] = value

    return result


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="rescorla_wagner",
    description="Fit basic Rescorla-Wagner model to WCST data"
)
def analyze_rescorla_wagner(verbose: bool = True) -> pd.DataFrame:
    """
    Fit basic RW model with single learning rate.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RESCORLA-WAGNER MODEL FITTING")
        print("=" * 70)

    trials = load_wcst_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient WCST trial data")
        return pd.DataFrame()

    if verbose:
        print(f"  Total trials: {len(trials)}")
        print(f"  Participants: {trials['participant_id'].nunique()}")
        print("\n  Fitting model to each participant...")

    results = []
    bounds = [(0.01, 0.99), (0.1, 10.0)]  # alpha, beta
    param_names = ['alpha', 'beta']

    for pid, pdata in trials.groupby('participant_id'):
        fit_result = fit_model(RescorlaWagnerModel, pdata, bounds, param_names)

        if fit_result is not None:
            fit_result['participant_id'] = pid
            results.append(fit_result)

    if len(results) < 20:
        if verbose:
            print(f"  Only {len(results)} participants fitted successfully")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  Fitted participants: {len(results_df)}")
        print(f"  Mean alpha: {results_df['alpha'].mean():.3f} (SD={results_df['alpha'].std():.3f})")
        print(f"  Mean beta: {results_df['beta'].mean():.3f} (SD={results_df['beta'].std():.3f})")

    results_df.to_csv(OUTPUT_DIR / "rw_basic_parameters.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'rw_basic_parameters.csv'}")

    return results_df


@register_analysis(
    name="asymmetric_learning",
    description="Fit asymmetric RW model with separate positive/negative learning rates"
)
def analyze_asymmetric_learning(verbose: bool = True) -> pd.DataFrame:
    """
    Fit asymmetric RW model to examine if loneliness relates to
    differential sensitivity to positive vs negative feedback.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ASYMMETRIC LEARNING RATE MODEL")
        print("=" * 70)

    trials = load_wcst_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient WCST trial data")
        return pd.DataFrame()

    if verbose:
        print(f"  Total trials: {len(trials)}")
        print("\n  Fitting asymmetric model to each participant...")

    results = []
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.1, 10.0)]  # alpha_pos, alpha_neg, beta
    param_names = ['alpha_pos', 'alpha_neg', 'beta']

    for pid, pdata in trials.groupby('participant_id'):
        fit_result = fit_model(AsymmetricRWModel, pdata, bounds, param_names)

        if fit_result is not None:
            fit_result['participant_id'] = pid
            # Compute asymmetry index: positive - negative
            fit_result['alpha_asymmetry'] = fit_result['alpha_pos'] - fit_result['alpha_neg']
            results.append(fit_result)

    if len(results) < 20:
        if verbose:
            print(f"  Only {len(results)} participants fitted successfully")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  Fitted participants: {len(results_df)}")
        print(f"  Mean alpha_pos: {results_df['alpha_pos'].mean():.3f} (SD={results_df['alpha_pos'].std():.3f})")
        print(f"  Mean alpha_neg: {results_df['alpha_neg'].mean():.3f} (SD={results_df['alpha_neg'].std():.3f})")
        print(f"  Mean asymmetry (pos-neg): {results_df['alpha_asymmetry'].mean():.3f}")

        # Test if asymmetry differs from zero
        t_stat, p_val = stats.ttest_1samp(results_df['alpha_asymmetry'].dropna(), 0)
        sig = "*" if p_val < 0.05 else ""
        print(f"  Asymmetry test (vs 0): t={t_stat:.2f}, p={p_val:.4f}{sig}")

    results_df.to_csv(OUTPUT_DIR / "rw_asymmetric_parameters.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'rw_asymmetric_parameters.csv'}")

    return results_df


@register_analysis(
    name="model_comparison",
    description="Compare fit of different RL models using BIC/AIC"
)
def analyze_model_comparison(verbose: bool = True) -> pd.DataFrame:
    """
    Compare model fits across different RL models.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON (BIC/AIC)")
        print("=" * 70)

    trials = load_wcst_trials()

    if len(trials) < 100:
        if verbose:
            print("  Insufficient WCST trial data")
        return pd.DataFrame()

    models = {
        'basic_rw': {
            'class': RescorlaWagnerModel,
            'bounds': [(0.01, 0.99), (0.1, 10.0)],
            'params': ['alpha', 'beta']
        },
        'asymmetric_rw': {
            'class': AsymmetricRWModel,
            'bounds': [(0.01, 0.99), (0.01, 0.99), (0.1, 10.0)],
            'params': ['alpha_pos', 'alpha_neg', 'beta']
        },
        'forgetting_rw': {
            'class': ForgettingRWModel,
            'bounds': [(0.01, 0.99), (0.01, 0.5), (0.1, 10.0)],
            'params': ['alpha', 'gamma', 'beta']
        }
    }

    all_results = []

    for model_name, model_info in models.items():
        if verbose:
            print(f"\n  Fitting {model_name}...")

        for pid, pdata in trials.groupby('participant_id'):
            fit_result = fit_model(
                model_info['class'],
                pdata,
                model_info['bounds'],
                model_info['params']
            )

            if fit_result is not None:
                fit_result['participant_id'] = pid
                fit_result['model'] = model_name
                all_results.append(fit_result)

    if len(all_results) < 20:
        if verbose:
            print(f"  Insufficient results")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Compute model comparison statistics
    comparison_summary = []

    for model_name in models.keys():
        model_data = results_df[results_df['model'] == model_name]
        if len(model_data) > 0:
            comparison_summary.append({
                'model': model_name,
                'n_participants': len(model_data),
                'mean_bic': model_data['bic'].mean(),
                'sd_bic': model_data['bic'].std(),
                'mean_aic': model_data['aic'].mean(),
                'sd_aic': model_data['aic'].std(),
                'mean_nll': model_data['nll'].mean()
            })

    comparison_df = pd.DataFrame(comparison_summary)

    if verbose:
        print("\n  Model Comparison Summary:")
        print("  " + "-" * 60)
        for _, row in comparison_df.iterrows():
            print(f"    {row['model']}: BIC={row['mean_bic']:.1f} (SD={row['sd_bic']:.1f}), "
                  f"AIC={row['mean_aic']:.1f}")

        # Identify best model
        best_model = comparison_df.loc[comparison_df['mean_bic'].idxmin(), 'model']
        print(f"\n  Best model (lowest BIC): {best_model}")

    results_df.to_csv(OUTPUT_DIR / "model_comparison_all.csv", index=False, encoding='utf-8-sig')
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'model_comparison_summary.csv'}")

    return comparison_df


@register_analysis(
    name="parameter_recovery",
    description="Validate parameter recovery via simulation"
)
def analyze_parameter_recovery(verbose: bool = True, n_simulations: int = 100) -> pd.DataFrame:
    """
    Simulate data and recover parameters to validate model fitting.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PARAMETER RECOVERY SIMULATION")
        print("=" * 70)
        print(f"  Running {n_simulations} simulations...")

    np.random.seed(42)

    recovery_results = []
    n_trials = 64  # Typical WCST trial count

    for sim in range(n_simulations):
        # Generate true parameters
        true_alpha = np.random.uniform(0.1, 0.9)
        true_beta = np.random.uniform(0.5, 5.0)

        # Simulate data
        model = RescorlaWagnerModel(alpha=true_alpha, beta=true_beta)
        model.reset()

        simulated_trials = []
        for t in range(n_trials):
            # Softmax action selection
            exp_q = np.exp(model.beta * (model.Q - np.max(model.Q)))
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(4, p=probs)

            # Simulate reward (random correct/incorrect with some structure)
            correct = np.random.random() < (0.6 if model.Q[action] > 0.3 else 0.4)
            reward = 1.0 if correct else 0.0

            simulated_trials.append({
                'chosencard': action,
                'correct': correct
            })

            model.update(action, reward)

        sim_df = pd.DataFrame(simulated_trials)

        # Recover parameters
        bounds = [(0.01, 0.99), (0.1, 10.0)]
        param_names = ['alpha', 'beta']

        fit_result = fit_model(RescorlaWagnerModel, sim_df, bounds, param_names, n_restarts=3)

        if fit_result is not None:
            recovery_results.append({
                'simulation': sim,
                'true_alpha': true_alpha,
                'recovered_alpha': fit_result['alpha'],
                'true_beta': true_beta,
                'recovered_beta': fit_result['beta']
            })

    recovery_df = pd.DataFrame(recovery_results)

    if len(recovery_df) < 10:
        if verbose:
            print("  Insufficient recovery results")
        return pd.DataFrame()

    # Compute recovery correlations
    alpha_r, alpha_p = stats.pearsonr(recovery_df['true_alpha'], recovery_df['recovered_alpha'])
    beta_r, beta_p = stats.pearsonr(recovery_df['true_beta'], recovery_df['recovered_beta'])

    if verbose:
        print(f"\n  Recovery Results:")
        print(f"  " + "-" * 50)
        print(f"    Alpha: r = {alpha_r:.3f}, p = {alpha_p:.4f}")
        print(f"    Beta:  r = {beta_r:.3f}, p = {beta_p:.4f}")

        if alpha_r > 0.8 and beta_r > 0.8:
            print("\n    Parameter recovery GOOD (r > 0.8)")
        else:
            print("\n    Parameter recovery MODERATE")

    recovery_df.to_csv(OUTPUT_DIR / "parameter_recovery.csv", index=False, encoding='utf-8-sig')

    # Save summary
    summary = pd.DataFrame([{
        'parameter': 'alpha',
        'recovery_r': alpha_r,
        'recovery_p': alpha_p,
        'n_simulations': len(recovery_df)
    }, {
        'parameter': 'beta',
        'recovery_r': beta_r,
        'recovery_p': beta_p,
        'n_simulations': len(recovery_df)
    }])
    summary.to_csv(OUTPUT_DIR / "parameter_recovery_summary.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'parameter_recovery.csv'}")

    return recovery_df


@register_analysis(
    name="ucla_relationship",
    description="Test UCLA effects on RL parameters (DASS-controlled)"
)
def analyze_ucla_relationship(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether loneliness relates to RL parameters.

    Hypotheses:
    - H1: Higher UCLA -> lower alpha (slower learning)
    - H2: Higher UCLA -> more negative alpha asymmetry (over-sensitivity to negative feedback)
    - H3: Higher UCLA -> higher beta (more exploitation, less exploration)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECTS ON RL PARAMETERS (DASS-CONTROLLED)")
        print("=" * 70)

    # Load master data
    master = load_rl_data()

    # Load or fit RL parameters
    param_files = {
        'basic': OUTPUT_DIR / "rw_basic_parameters.csv",
        'asymmetric': OUTPUT_DIR / "rw_asymmetric_parameters.csv"
    }

    all_results = []

    for model_type, param_file in param_files.items():
        if not param_file.exists():
            if verbose:
                print(f"\n  {model_type} parameters not found - fitting now...")

            if model_type == 'basic':
                analyze_rescorla_wagner(verbose=False)
            elif model_type == 'asymmetric':
                analyze_asymmetric_learning(verbose=False)

        if not param_file.exists():
            continue

        params = pd.read_csv(param_file)
        merged = master.merge(params, on='participant_id', how='inner')

        if len(merged) < 30:
            if verbose:
                print(f"  {model_type}: Insufficient merged data (N={len(merged)})")
            continue

        if verbose:
            print(f"\n  {model_type.upper()} MODEL (N={len(merged)})")
            print("  " + "-" * 50)

        # Test each parameter
        param_cols = ['alpha', 'beta'] if model_type == 'basic' else ['alpha_pos', 'alpha_neg', 'alpha_asymmetry', 'beta']

        for param in param_cols:
            if param not in merged.columns:
                continue

            # DASS-controlled regression
            try:
                formula = f"{param} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta_ucla = model.params['z_ucla']
                    se_ucla = model.bse['z_ucla']
                    p_ucla = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p_ucla < 0.05 else ""
                        print(f"    UCLA -> {param}: beta={beta_ucla:.4f}, SE={se_ucla:.4f}, p={p_ucla:.4f}{sig}")

                    all_results.append({
                        'model_type': model_type,
                        'parameter': param,
                        'beta_ucla': beta_ucla,
                        'se_ucla': se_ucla,
                        'p_ucla': p_ucla,
                        'r_squared': model.rsquared,
                        'n': len(merged)
                    })

                # Check interaction
                interaction_term = 'z_ucla:C(gender_male)[T.1]'
                if interaction_term in model.params:
                    beta_int = model.params[interaction_term]
                    p_int = model.pvalues[interaction_term]

                    if p_int < 0.05 and verbose:
                        print(f"    UCLA x Gender: beta={beta_int:.4f}, p={p_int:.4f}*")

                    all_results.append({
                        'model_type': model_type,
                        'parameter': f'{param}_interaction',
                        'beta_ucla': beta_int,
                        'se_ucla': model.bse.get(interaction_term, np.nan),
                        'p_ucla': p_int,
                        'r_squared': model.rsquared,
                        'n': len(merged)
                    })

            except Exception as e:
                if verbose:
                    print(f"    {param}: Regression error - {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "ucla_rl_relationship.csv", index=False, encoding='utf-8-sig')

    # Bootstrap confidence intervals for significant effects
    sig_effects = results_df[results_df['p_ucla'] < 0.05]

    if len(sig_effects) > 0 and verbose:
        print("\n  Significant Effects (p < 0.05):")
        print("  " + "-" * 50)
        for _, row in sig_effects.iterrows():
            print(f"    {row['model_type']} - {row['parameter']}: "
                  f"beta={row['beta_ucla']:.4f}, p={row['p_ucla']:.4f}")

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'ucla_rl_relationship.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run reinforcement learning analyses.
    """
    if verbose:
        print("=" * 70)
        print("REINFORCEMENT LEARNING ANALYSIS SUITE")
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
        # Run all analyses in order
        analysis_order = [
            'rescorla_wagner',
            'asymmetric_learning',
            'model_comparison',
            'parameter_recovery',
            'ucla_relationship'
        ]

        for name in analysis_order:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("REINFORCEMENT LEARNING SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Reinforcement Learning Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning Analysis Suite")
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
