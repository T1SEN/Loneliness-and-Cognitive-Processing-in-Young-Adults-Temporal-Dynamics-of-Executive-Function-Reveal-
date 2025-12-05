"""
Validation Analysis Suite
=========================

Unified methodological validation analyses for UCLA × Executive Function research.

Consolidates all 7 analyses:
- validation_1_cross_validation.py
- validation_2_robust_quantile.py
- validation_3_type_ms_simulation.py
- replication_verification_corrected.py
- split_half_internal_replication.py
- specification_curve_analysis.py
- statistical_robustness.py

Usage:
    python -m analysis.validation.validation_suite                    # Run all
    python -m analysis.validation.validation_suite --analysis cross_validation
    python -m analysis.validation.validation_suite --list

    from analysis.validation import validation_suite
    validation_suite.run('cross_validation')
    validation_suite.run()  # All analyses

CRITICAL: All models control for DASS-21 subscales (depression, anxiety, stress) + age.

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
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import QuantileRegressor
from itertools import product

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)
from analysis.preprocessing import (
    safe_zscore,
    prepare_gender_variable,
    find_interaction_term
)

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "validation_suite"
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

def load_validation_data() -> pd.DataFrame:
    """Load and prepare master dataset for validation analyses."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    # Ensure ucla_total exists
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors using NaN-safe z-score (ddof=1)
    required_cols = ['age', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']

    for col in required_cols:
        if col not in master.columns:
            raise ValueError(f"Missing required column: {col}")

    master['z_age'] = safe_zscore(master['age'])
    master['z_ucla'] = safe_zscore(master['ucla_total'])
    master['z_dass_dep'] = safe_zscore(master['dass_depression'])
    master['z_dass_anx'] = safe_zscore(master['dass_anxiety'])
    master['z_dass_str'] = safe_zscore(master['dass_stress'])

    return master


def get_outcomes(df: pd.DataFrame) -> list:
    """Get available outcome variables."""
    outcomes = []

    if 'pe_rate' in df.columns:
        outcomes.append(('pe_rate', 'WCST Perseverative Error Rate'))
    if 'prp_bottleneck' in df.columns:
        outcomes.append(('prp_bottleneck', 'PRP Bottleneck Effect'))
    if 'stroop_interference' in df.columns:
        outcomes.append(('stroop_interference', 'Stroop Interference'))

    return outcomes


# =============================================================================
# ANALYSIS 1: K-FOLD CROSS-VALIDATION
# =============================================================================

@register_analysis(
    name="cross_validation",
    description="k-fold CV comparing Base vs Full model (with UCLA x Gender)",
    source_script="validation_1_cross_validation.py"
)
def analyze_cross_validation(verbose: bool = True, n_folds: int = 5) -> pd.DataFrame:
    """
    Compare out-of-sample predictive performance between:
    - Base model: DASS + Age + Gender (no UCLA)
    - Full model: DASS + Age + Gender + UCLA + UCLA×Gender
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION ANALYSIS")
        print("=" * 70)

    master = load_validation_data()
    outcomes = get_outcomes(master)

    if verbose:
        print(f"  N = {len(master)}, Folds = {n_folds}")
        print(f"  Outcomes: {len(outcomes)}")

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        # Prepare data
        analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                         'z_ucla', 'gender_male', outcome_col]
        df_clean = master[analysis_cols].dropna()

        if len(df_clean) < 20:
            if verbose:
                print(f"    WARNING: Only {len(df_clean)} cases. Skipping.")
            continue

        # K-fold CV
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_clean)):
            df_train = df_clean.iloc[train_idx]
            df_test = df_clean.iloc[test_idx]

            # Base model (no UCLA)
            formula_base = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str"
            model_base = smf.ols(formula_base, data=df_train).fit()
            pred_base = model_base.predict(df_test)

            # Full model (with UCLA × Gender)
            formula_full = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
            model_full = smf.ols(formula_full, data=df_train).fit()
            pred_full = model_full.predict(df_test)

            y_test = df_test[outcome_col].values

            fold_results.append({
                'fold': fold_idx + 1,
                'base_rmse': np.sqrt(mean_squared_error(y_test, pred_base)),
                'base_r2': r2_score(y_test, pred_base),
                'full_rmse': np.sqrt(mean_squared_error(y_test, pred_full)),
                'full_r2': r2_score(y_test, pred_full),
            })

        fold_df = pd.DataFrame(fold_results)
        fold_df['delta_rmse'] = fold_df['full_rmse'] - fold_df['base_rmse']
        fold_df['delta_r2'] = fold_df['full_r2'] - fold_df['base_r2']

        # Statistical test
        t_rmse, p_rmse = stats.ttest_1samp(fold_df['delta_rmse'], 0)

        summary = {
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': len(df_clean),
            'base_rmse_mean': fold_df['base_rmse'].mean(),
            'base_rmse_sd': fold_df['base_rmse'].std(),
            'full_rmse_mean': fold_df['full_rmse'].mean(),
            'full_rmse_sd': fold_df['full_rmse'].std(),
            'delta_rmse_mean': fold_df['delta_rmse'].mean(),
            'delta_rmse_p': p_rmse,
            'full_better': fold_df['delta_rmse'].mean() < 0 and p_rmse < 0.05
        }
        all_results.append(summary)

        if verbose:
            print(f"    N = {len(df_clean)}")
            print(f"    Base RMSE: {summary['base_rmse_mean']:.4f} +/- {summary['base_rmse_sd']:.4f}")
            print(f"    Full RMSE: {summary['full_rmse_mean']:.4f} +/- {summary['full_rmse_sd']:.4f}")
            print(f"    Delta: {summary['delta_rmse_mean']:.4f} (p = {p_rmse:.4f})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'cross_validation_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 2: ROBUST & QUANTILE REGRESSION
# =============================================================================

@register_analysis(
    name="robust_regression",
    description="Huber robust regression vs OLS for outlier sensitivity",
    source_script="validation_2_robust_quantile.py"
)
def analyze_robust_regression(verbose: bool = True) -> pd.DataFrame:
    """
    Test robustness of UCLA × Gender interaction to outliers using Huber M-estimator.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ROBUST REGRESSION (HUBER M-ESTIMATOR)")
        print("=" * 70)

    master = load_validation_data()
    outcomes = get_outcomes(master)

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                         'z_ucla', 'gender_male', outcome_col]
        df_clean = master[analysis_cols].dropna()

        if len(df_clean) < 20:
            continue

        # OLS regression
        formula = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
        ols_model = smf.ols(formula, data=df_clean).fit()

        # Dynamic interaction term detection
        interaction_term = find_interaction_term(ols_model.params.index, 'ucla', 'gender')
        if interaction_term is None:
            if verbose:
                print(f"    WARNING: Interaction term not found in model. Skipping.")
            continue

        ols_beta = ols_model.params[interaction_term]
        ols_p = ols_model.pvalues[interaction_term]

        # Huber robust regression
        X_design = pd.DataFrame({
            'const': 1,
            'z_age': df_clean['z_age'],
            'gender_male': df_clean['gender_male'],
            'z_dass_dep': df_clean['z_dass_dep'],
            'z_dass_anx': df_clean['z_dass_anx'],
            'z_dass_str': df_clean['z_dass_str'],
            'z_ucla': df_clean['z_ucla'],
            'ucla_x_gender': df_clean['z_ucla'] * df_clean['gender_male']
        })
        y = df_clean[outcome_col].values

        huber_model = RLM(y, X_design, M=HuberT()).fit()
        huber_beta = huber_model.params['ucla_x_gender']
        huber_p = huber_model.pvalues['ucla_x_gender']

        beta_diff_pct = ((huber_beta - ols_beta) / ols_beta * 100) if ols_beta != 0 else np.nan

        result = {
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': len(df_clean),
            'ols_beta': ols_beta,
            'ols_p': ols_p,
            'huber_beta': huber_beta,
            'huber_p': huber_p,
            'beta_diff_pct': beta_diff_pct,
            'robust_to_outliers': abs(beta_diff_pct) < 25 if not np.isnan(beta_diff_pct) else None
        }
        all_results.append(result)

        if verbose:
            print(f"    N = {len(df_clean)}")
            print(f"    OLS:   beta = {ols_beta:.4f}, p = {ols_p:.4f}")
            print(f"    Huber: beta = {huber_beta:.4f}, p = {huber_p:.4f}")
            print(f"    Change: {beta_diff_pct:.1f}%")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "robust_regression_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'robust_regression_results.csv'}")

    return results_df


@register_analysis(
    name="quantile_regression",
    description="Quantile regression at tau=0.25, 0.50, 0.75, 0.90",
    source_script="validation_2_robust_quantile.py"
)
def analyze_quantile_regression(verbose: bool = True) -> pd.DataFrame:
    """
    Test if UCLA × Gender effect varies across the outcome distribution.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("QUANTILE REGRESSION")
        print("=" * 70)

    master = load_validation_data()
    outcomes = get_outcomes(master)
    quantiles = [0.25, 0.50, 0.75, 0.90]

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                         'z_ucla', 'gender_male', outcome_col]
        df_clean = master[analysis_cols].dropna()

        if len(df_clean) < 20:
            continue

        X = pd.DataFrame({
            'z_age': df_clean['z_age'],
            'gender_male': df_clean['gender_male'],
            'z_dass_dep': df_clean['z_dass_dep'],
            'z_dass_anx': df_clean['z_dass_anx'],
            'z_dass_str': df_clean['z_dass_str'],
            'z_ucla': df_clean['z_ucla'],
            'ucla_x_gender': df_clean['z_ucla'] * df_clean['gender_male']
        })
        y = df_clean[outcome_col].values

        if verbose:
            print(f"    N = {len(df_clean)}")
            print(f"    {'tau':>6}  {'beta':>10}")
            print("    " + "-" * 20)

        # Get feature names for coefficient mapping
        feature_names = X.columns.tolist()

        for quantile in quantiles:
            qr = QuantileRegressor(quantile=quantile, solver='highs', alpha=0)
            qr.fit(X, y)

            # Use name-based lookup instead of position-based
            coef_dict = dict(zip(feature_names, qr.coef_))
            coef_interaction = coef_dict.get('ucla_x_gender', np.nan)

            all_results.append({
                'outcome': outcome_col,
                'outcome_label': outcome_label,
                'n': len(df_clean),
                'quantile': quantile,
                'beta_interaction': coef_interaction
            })

            if verbose:
                print(f"    {quantile:>6.2f}  {coef_interaction:>10.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "quantile_regression_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'quantile_regression_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: TYPE M/S ERROR SIMULATION
# =============================================================================

@register_analysis(
    name="type_ms_simulation",
    description="Simulation-based Type M (magnitude) and Type S (sign) error analysis",
    source_script="validation_3_type_ms_simulation.py"
)
def analyze_type_ms_simulation(
    verbose: bool = True,
    n_simulations: int = 500,
    true_effect_multipliers: List[float] = None
) -> pd.DataFrame:
    """
    Quantify the risk of effect size overestimation (Type M) and sign error (Type S).

    Parameters
    ----------
    verbose : bool
        Print progress messages.
    n_simulations : int
        Number of simulation iterations per multiplier.
    true_effect_multipliers : list of float, optional
        Multipliers for true effect size sensitivity analysis.
        Default: [0.5, 0.7, 0.9] to test conservative to optimistic assumptions.
    """
    if true_effect_multipliers is None:
        true_effect_multipliers = [0.5, 0.7, 0.9]  # Sensitivity grid

    if verbose:
        print("\n" + "=" * 70)
        print("TYPE M/S ERROR SIMULATION")
        print("=" * 70)
        print(f"  Simulations: {n_simulations}")
        print(f"  Effect multipliers: {true_effect_multipliers}")

    master = load_validation_data()
    outcomes = get_outcomes(master)

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                         'z_ucla', 'gender_male', outcome_col]
        df_clean = master[analysis_cols].dropna()

        if len(df_clean) < 20:
            continue

        n = len(df_clean)

        # Fit observed model
        formula = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
        obs_model = smf.ols(formula, data=df_clean).fit()

        # Dynamic interaction term detection
        interaction_term = find_interaction_term(obs_model.params.index, 'ucla', 'gender')
        if interaction_term is None:
            if verbose:
                print(f"    WARNING: Interaction term not found. Skipping.")
            continue

        observed_beta = obs_model.params[interaction_term]
        residual_std = np.sqrt(obs_model.mse_resid)

        if verbose:
            print(f"    N = {n}")
            print(f"    Observed beta = {observed_beta:.4f}")

        # Get true coefficients (dynamically find gender term)
        true_intercept = obs_model.params['Intercept']
        true_age = obs_model.params['z_age']
        gender_term = find_interaction_term(obs_model.params.index, 'gender', 'male')
        if gender_term is None:
            # If gender is numeric (e.g., gender_male), statsmodels names it without C()
            if 'gender_male' in obs_model.params.index:
                gender_term = 'gender_male'
            else:
                gender_term = 'C(gender_male)[T.1]'  # fallback
        true_gender = obs_model.params.get(gender_term, 0)
        true_dass_dep = obs_model.params['z_dass_dep']
        true_dass_anx = obs_model.params['z_dass_anx']
        true_dass_str = obs_model.params['z_dass_str']
        true_ucla = obs_model.params['z_ucla']

        X = df_clean[['z_age', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_ucla']].values

        # Run sensitivity analysis across multipliers
        for multiplier in true_effect_multipliers:
            true_beta_interaction = observed_beta * multiplier

            if verbose:
                print(f"\n    Multiplier = {multiplier} (true beta = {true_beta_interaction:.4f})")

            # Run simulations
            sim_results = []
            fail_count = 0

            for sim_i in range(n_simulations):
                y_sim = (true_intercept +
                         true_age * X[:, 0] +
                         true_gender * X[:, 1] +
                         true_dass_dep * X[:, 2] +
                         true_dass_anx * X[:, 3] +
                         true_dass_str * X[:, 4] +
                         true_ucla * X[:, 5] +
                         true_beta_interaction * X[:, 5] * X[:, 1] +
                         np.random.normal(0, residual_std, n))

                sim_df = pd.DataFrame({
                    'y': y_sim,
                    'z_age': X[:, 0],
                    'gender_male': X[:, 1],
                    'z_dass_dep': X[:, 2],
                    'z_dass_anx': X[:, 3],
                    'z_dass_str': X[:, 4],
                    'z_ucla': X[:, 5],
                    'ucla_x_gender': X[:, 5] * X[:, 1]
                })

                try:
                    sim_model = smf.ols("y ~ z_age + gender_male + z_dass_dep + z_dass_anx + z_dass_str + z_ucla + ucla_x_gender",
                                       data=sim_df).fit()
                    beta_hat = sim_model.params['ucla_x_gender']
                    p_hat = sim_model.pvalues['ucla_x_gender']

                    sim_results.append({
                        'beta_hat': beta_hat,
                        'p_value': p_hat,
                        'significant': p_hat < 0.05,
                        'sign_correct': np.sign(beta_hat) == np.sign(true_beta_interaction)
                    })
                except Exception as e:
                    fail_count += 1
                    if verbose and fail_count <= 3:
                        warnings.warn(f"Simulation {sim_i} failed: {e}")
                    continue

            # Warn if high failure rate
            if fail_count > n_simulations * 0.1:
                warnings.warn(f"High simulation failure rate: {fail_count}/{n_simulations} ({fail_count/n_simulations*100:.1f}%)")

            if len(sim_results) == 0:
                continue

            sim_df_results = pd.DataFrame(sim_results)
            sig_results = sim_df_results[sim_df_results['significant']]

            power = len(sig_results) / len(sim_df_results)
            type_m_error = sig_results['beta_hat'].abs().mean() / abs(true_beta_interaction) if len(sig_results) > 0 else np.nan
            type_s_error = (~sig_results['sign_correct']).mean() if len(sig_results) > 0 else np.nan

            result = {
                'outcome': outcome_col,
                'outcome_label': outcome_label,
                'n': n,
                'effect_multiplier': multiplier,
                'n_simulations': len(sim_df_results),
                'n_failed': fail_count,
                'observed_beta': observed_beta,
                'true_beta': true_beta_interaction,
                'power': power,
                'type_m_error': type_m_error,
                'type_s_error': type_s_error
            }
            all_results.append(result)

            if verbose:
                print(f"      Power: {power:.3f}")
                print(f"      Type M error: {type_m_error:.3f}" if not np.isnan(type_m_error) else "      Type M error: N/A")
                print(f"      Type S error: {type_s_error:.3f}" if not np.isnan(type_s_error) else "      Type S error: N/A")
                if fail_count > 0:
                    print(f"      (Failed: {fail_count}/{n_simulations})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "type_ms_simulation_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'type_ms_simulation_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 4: SPLIT-HALF INTERNAL REPLICATION
# =============================================================================

@register_analysis(
    name="split_half_replication",
    description="Split-half internal replication to test effect stability",
    source_script="split_half_internal_replication.py"
)
def analyze_split_half_replication(verbose: bool = True, n_splits: int = 100) -> pd.DataFrame:
    """
    Test effect stability via random split-half replication.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SPLIT-HALF INTERNAL REPLICATION")
        print("=" * 70)
        print(f"  Splits: {n_splits}")

    master = load_validation_data()
    outcomes = get_outcomes(master)

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                         'z_ucla', 'gender_male', outcome_col]
        df_clean = master[analysis_cols].dropna()

        if len(df_clean) < 40:
            if verbose:
                print(f"    WARNING: Only {len(df_clean)} cases. Need >= 40 for split-half.")
            continue

        n = len(df_clean)
        half = n // 2

        split_results = []
        fail_count = 0

        for split_idx in range(n_splits):
            # Random split
            indices = np.random.permutation(n)
            half1_idx = indices[:half]
            half2_idx = indices[half:]

            df_half1 = df_clean.iloc[half1_idx]
            df_half2 = df_clean.iloc[half2_idx]

            formula = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"

            try:
                model1 = smf.ols(formula, data=df_half1).fit()
                model2 = smf.ols(formula, data=df_half2).fit()

                # Dynamic interaction term detection
                int_term1 = find_interaction_term(model1.params.index, 'ucla', 'gender')
                int_term2 = find_interaction_term(model2.params.index, 'ucla', 'gender')

                if int_term1 is not None and int_term2 is not None:
                    beta1 = model1.params[int_term1]
                    beta2 = model2.params[int_term2]
                    p1 = model1.pvalues[int_term1]
                    p2 = model2.pvalues[int_term2]

                    split_results.append({
                        'split': split_idx,
                        'beta_half1': beta1,
                        'beta_half2': beta2,
                        'p_half1': p1,
                        'p_half2': p2,
                        'same_sign': np.sign(beta1) == np.sign(beta2),
                        'both_sig': (p1 < 0.05) and (p2 < 0.05)
                    })
            except Exception as e:
                fail_count += 1
                if verbose and fail_count <= 3:
                    warnings.warn(f"Split {split_idx} failed: {e}")
                continue

        # Warn if high failure rate
        if fail_count > n_splits * 0.1:
            warnings.warn(f"High split failure rate: {fail_count}/{n_splits} ({fail_count/n_splits*100:.1f}%)")

        if len(split_results) == 0:
            continue

        split_df = pd.DataFrame(split_results)

        # Calculate correlation between halves
        r_betas = np.corrcoef(split_df['beta_half1'], split_df['beta_half2'])[0, 1]
        same_sign_rate = split_df['same_sign'].mean()
        both_sig_rate = split_df['both_sig'].mean()

        result = {
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': n,
            'n_splits': len(split_df),
            'beta_correlation': r_betas,
            'same_sign_rate': same_sign_rate,
            'both_significant_rate': both_sig_rate
        }
        all_results.append(result)

        if verbose:
            print(f"    N = {n}")
            print(f"    Beta correlation: r = {r_betas:.3f}")
            print(f"    Same sign rate: {same_sign_rate:.1%}")
            print(f"    Both significant rate: {both_sig_rate:.1%}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "split_half_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'split_half_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 5: SPECIFICATION CURVE
# =============================================================================

@register_analysis(
    name="specification_curve",
    description="Multiverse analysis testing robustness across analytical choices",
    source_script="specification_curve_analysis.py"
)
def analyze_specification_curve(verbose: bool = True) -> pd.DataFrame:
    """
    Test robustness of UCLA × Gender → WCST PE across specifications.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SPECIFICATION CURVE ANALYSIS")
        print("=" * 70)

    master = load_validation_data()

    if 'pe_rate' not in master.columns:
        if verbose:
            print("  No pe_rate column")
        return pd.DataFrame()

    # Specification choices
    specs = {
        'outlier': ['none', 'winsorize'],
        'covariates': ['none', 'dass', 'dass_age'],
        'standardize': ['raw', 'zscore']
    }

    all_results = []

    df_base = master[['participant_id', 'ucla_total', 'gender_male', 'pe_rate',
                      'dass_depression', 'dass_anxiety', 'dass_stress', 'age']].dropna()

    if len(df_base) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(df_base)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df_base)}")

    spec_count = 0

    for outlier, covariates, standardize in product(specs['outlier'], specs['covariates'], specs['standardize']):
        df = df_base.copy()

        # Outlier handling
        if outlier == 'winsorize':
            for col in ['pe_rate', 'ucla_total']:
                lower = df[col].quantile(0.025)
                upper = df[col].quantile(0.975)
                df[col] = df[col].clip(lower, upper)

        # Standardization
        if standardize == 'zscore':
            for col in ['ucla_total', 'pe_rate', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
                df[col] = (df[col] - df[col].mean()) / df[col].std()

        # Build formula
        if covariates == 'none':
            formula = "pe_rate ~ ucla_total * C(gender_male)"
        elif covariates == 'dass':
            formula = "pe_rate ~ ucla_total * C(gender_male) + dass_depression + dass_anxiety + dass_stress"
        else:  # dass_age
            formula = "pe_rate ~ ucla_total * C(gender_male) + dass_depression + dass_anxiety + dass_stress + age"

        try:
            model = smf.ols(formula, data=df).fit()

            interaction_term = 'ucla_total:C(gender_male)[T.1]'
            if interaction_term in model.params.index:
                beta = model.params[interaction_term]
                se = model.bse[interaction_term]
                p = model.pvalues[interaction_term]

                all_results.append({
                    'outlier': outlier,
                    'covariates': covariates,
                    'standardize': standardize,
                    'beta': beta,
                    'se': se,
                    'p': p,
                    'significant': p < 0.05
                })

                spec_count += 1

        except Exception:
            continue

    if len(all_results) == 0:
        if verbose:
            print("  No valid specifications")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Summary
    median_beta = results_df['beta'].median()
    median_p = results_df['p'].median()
    pct_significant = results_df['significant'].mean() * 100

    if verbose:
        print(f"\n  Specifications tested: {spec_count}")
        print(f"  Median beta: {median_beta:.4f}")
        print(f"  Median p-value: {median_p:.4f}")
        print(f"  % Significant: {pct_significant:.1f}%")

    results_df.to_csv(OUTPUT_DIR / "specification_curve_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'specification_curve_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 6: STATISTICAL ROBUSTNESS
# =============================================================================

@register_analysis(
    name="robustness",
    description="Leave-one-out sensitivity and effect stability analysis",
    source_script="statistical_robustness.py"
)
def analyze_robustness(verbose: bool = True) -> pd.DataFrame:
    """
    Test robustness via leave-one-out and bootstrap sensitivity.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STATISTICAL ROBUSTNESS")
        print("=" * 70)

    master = load_validation_data()
    outcomes = get_outcomes(master)

    all_results = []

    for outcome_col, outcome_label in outcomes:
        if verbose:
            print(f"\n  {outcome_label}")
            print("  " + "-" * 50)

        analysis_cols = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                        'z_ucla', 'gender_male', outcome_col]
        df_clean = master[analysis_cols].dropna()

        if len(df_clean) < 30:
            if verbose:
                print(f"    Insufficient data (N={len(df_clean)})")
            continue

        n = len(df_clean)

        # Fit full model
        formula = f"{outcome_col} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)"
        full_model = smf.ols(formula, data=df_clean).fit()

        interaction_term = find_interaction_term(full_model.params.index, 'ucla', 'gender')
        if interaction_term is None:
            if verbose:
                print("    WARNING: Interaction term not found. Skipping.")
            continue

        full_beta = full_model.params[interaction_term]
        full_p = full_model.pvalues[interaction_term]

        if verbose:
            sig = "*" if full_p < 0.05 else ""
            print(f"    N = {n}")
            print(f"    Full model: β = {full_beta:.4f}, p = {full_p:.4f}{sig}")

        # Leave-one-out analysis
        loo_betas = []
        loo_fail = 0

        for i in range(n):
            df_loo = df_clean.drop(df_clean.index[i])
            try:
                loo_model = smf.ols(formula, data=df_loo).fit()
                int_term_loo = find_interaction_term(loo_model.params.index, 'ucla', 'gender')
                if int_term_loo is not None:
                    loo_betas.append(loo_model.params[int_term_loo])
            except Exception as e:
                loo_fail += 1
                if verbose and loo_fail <= 3:
                    warnings.warn(f"LOO fit failed at i={i}: {e}")
                continue

        if len(loo_betas) > 0:
            loo_min = min(loo_betas)
            loo_max = max(loo_betas)
            loo_range = loo_max - loo_min
            loo_sign_stable = all(np.sign(b) == np.sign(full_beta) for b in loo_betas)

            if verbose:
                print(f"    LOO range: [{loo_min:.4f}, {loo_max:.4f}]")
                print(f"    LOO sign stable: {loo_sign_stable}")

            all_results.append({
                'outcome': outcome_col,
                'outcome_label': outcome_label,
                'n': n,
                'full_beta': full_beta,
                'full_p': full_p,
                'loo_min': loo_min,
                'loo_max': loo_max,
                'loo_range': loo_range,
                'loo_sign_stable': loo_sign_stable
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "robustness_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'robustness_results.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run validation analyses.

    Parameters
    ----------
    analysis : str, optional
        Specific analysis to run. If None, runs all.
    verbose : bool
        Print progress messages.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Results from each analysis.
    """
    if verbose:
        print("=" * 70)
        print("VALIDATION ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
            print(f"  Description: {spec.description}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Running: {spec.name}")
                print(f"  Description: {spec.description}")
            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Validation Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")
        print(f"    Source: {spec.source_script}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation Analysis Suite")
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
