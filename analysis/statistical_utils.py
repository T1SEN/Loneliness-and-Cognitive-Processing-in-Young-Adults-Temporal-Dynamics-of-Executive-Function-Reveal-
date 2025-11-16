# -*- coding: utf-8 -*-
"""
Statistical Utility Functions
Centralized functions for statistical rigor across all analysis scripts
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Dict, Any
import warnings


def apply_multiple_comparison_correction(
    p_values: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple comparison correction to p-values

    Parameters:
    -----------
    p_values : array-like
        Array of p-values to correct
    method : str
        Correction method: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg), 'fdr_by'
    alpha : float
        Family-wise error rate

    Returns:
    --------
    reject : np.ndarray
        Boolean array indicating which hypotheses are rejected
    p_adjusted : np.ndarray
        Adjusted p-values
    """
    from statsmodels.stats.multitest import multipletests

    p_values = np.asarray(p_values)
    if len(p_values) == 0:
        return np.array([], dtype=bool), np.array([])

    reject, p_adjusted, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method=method
    )

    return reject, p_adjusted


def test_ols_assumptions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test statistical assumptions for OLS regression

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values from model
    X : array-like
        Feature matrix (n_samples, n_features)
    feature_names : list, optional
        Names of features for VIF reporting
    verbose : bool
        Print warnings if assumptions violated

    Returns:
    --------
    results : dict
        Dictionary with test statistics and p-values
    """
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    residuals = y_true - y_pred
    n = len(residuals)

    results = {}

    # 1. Test normality of residuals (Shapiro-Wilk)
    if n >= 3:
        try:
            stat_shapiro, p_shapiro = stats.shapiro(residuals)
            results['shapiro_stat'] = stat_shapiro
            results['shapiro_p'] = p_shapiro
            results['normality_ok'] = p_shapiro >= 0.05

            if verbose and p_shapiro < 0.05:
                print(f"⚠️  WARNING: Residuals not normally distributed (Shapiro-Wilk p={p_shapiro:.4f})")
                print("   Consider: robust regression, transformation, or non-parametric methods")
        except Exception as e:
            results['shapiro_error'] = str(e)

    # 2. Test homoscedasticity (Breusch-Pagan)
    try:
        _, p_bp, _, _ = het_breuschpagan(residuals, X)
        results['breusch_pagan_p'] = p_bp
        results['homoscedasticity_ok'] = p_bp >= 0.05

        if verbose and p_bp < 0.05:
            print(f"⚠️  WARNING: Heteroscedasticity detected (Breusch-Pagan p={p_bp:.4f})")
            print("   Consider: robust standard errors (HC3), WLS, or log transformation")
    except Exception as e:
        results['breusch_pagan_error'] = str(e)

    # 3. Test independence (Durbin-Watson)
    try:
        dw_stat = durbin_watson(residuals)
        results['durbin_watson'] = dw_stat
        results['independence_ok'] = 1.5 <= dw_stat <= 2.5

        if verbose and not (1.5 <= dw_stat <= 2.5):
            print(f"⚠️  WARNING: Autocorrelation detected (Durbin-Watson={dw_stat:.3f})")
            print("   Consider: time series methods or clustered standard errors")
    except Exception as e:
        results['durbin_watson_error'] = str(e)

    # 4. Check multicollinearity (VIF)
    if X.shape[1] > 1:
        try:
            vif_data = []
            for i in range(X.shape[1]):
                vif_val = variance_inflation_factor(X, i)
                vif_data.append(vif_val)
                if feature_names and verbose and vif_val > 10:
                    print(f"⚠️  WARNING: Multicollinearity in '{feature_names[i]}' (VIF={vif_val:.2f})")

            results['vif_values'] = vif_data
            results['max_vif'] = max(vif_data) if vif_data else np.nan
            results['multicollinearity_ok'] = all(v < 10 for v in vif_data)
        except Exception as e:
            results['vif_error'] = str(e)

    return results


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two independent groups

    Parameters:
    -----------
    group1, group2 : array-like
        Data for the two groups

    Returns:
    --------
    d : float
        Cohen's d effect size
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return d


def eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    """
    Calculate eta-squared (η²) effect size for ANOVA

    Parameters:
    -----------
    f_stat : float
        F-statistic from ANOVA
    df_between : int
        Degrees of freedom between groups
    df_within : int
        Degrees of freedom within groups

    Returns:
    --------
    eta_sq : float
        Eta-squared effect size
    """
    eta_sq = (df_between * f_stat) / (df_between * f_stat + df_within)
    return eta_sq


def check_bayesian_convergence(
    idata,
    var_names: Optional[list] = None,
    rhat_threshold: float = 1.01,
    ess_bulk_threshold: int = 400,
    ess_tail_threshold: int = 400,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Check convergence diagnostics for Bayesian MCMC samples

    Parameters:
    -----------
    idata : arviz.InferenceData
        Inference data from PyMC/NumPyro sampling
    var_names : list, optional
        Variable names to check (None = all variables)
    rhat_threshold : float
        Threshold for R-hat (should be < 1.01)
    ess_bulk_threshold : int
        Minimum effective sample size for bulk
    ess_tail_threshold : int
        Minimum effective sample size for tails
    verbose : bool
        Print warnings

    Returns:
    --------
    results : dict
        Convergence diagnostics
    """
    import arviz as az

    summary = az.summary(idata, var_names=var_names)

    results = {
        'summary': summary,
        'converged': True,
        'warnings': []
    }

    # Check R-hat
    if 'r_hat' in summary.columns:
        max_rhat = summary['r_hat'].max()
        results['max_rhat'] = max_rhat

        if max_rhat > rhat_threshold:
            results['converged'] = False
            msg = f"⚠️  WARNING: R-hat > {rhat_threshold} (max={max_rhat:.4f}) - chains have not converged!"
            results['warnings'].append(msg)
            if verbose:
                print(msg)
                print("   Recommendation: Increase draws/tune or check model specification")

    # Check ESS bulk
    if 'ess_bulk' in summary.columns:
        min_ess_bulk = summary['ess_bulk'].min()
        results['min_ess_bulk'] = min_ess_bulk

        if min_ess_bulk < ess_bulk_threshold:
            results['converged'] = False
            msg = f"⚠️  WARNING: ESS bulk < {ess_bulk_threshold} (min={min_ess_bulk:.0f}) - increase draws!"
            results['warnings'].append(msg)
            if verbose:
                print(msg)

    # Check ESS tail
    if 'ess_tail' in summary.columns:
        min_ess_tail = summary['ess_tail'].min()
        results['min_ess_tail'] = min_ess_tail

        if min_ess_tail < ess_tail_threshold:
            msg = f"⚠️  WARNING: ESS tail < {ess_tail_threshold} (min={min_ess_tail:.0f})"
            results['warnings'].append(msg)
            if verbose:
                print(msg)

    return results


def r_squared_change(
    r2_full: float,
    r2_reduced: float,
    n: int,
    p_full: int,
    p_reduced: int
) -> Tuple[float, float, float]:
    """
    Test significance of R² change in hierarchical regression

    Parameters:
    -----------
    r2_full : float
        R² of full model
    r2_reduced : float
        R² of reduced model
    n : int
        Sample size
    p_full : int
        Number of predictors in full model
    p_reduced : int
        Number of predictors in reduced model

    Returns:
    --------
    delta_r2 : float
        Change in R²
    f_stat : float
        F-statistic for the change
    p_value : float
        p-value for F-test
    """
    delta_r2 = r2_full - r2_reduced
    df1 = p_full - p_reduced
    df2 = n - p_full - 1

    # F-statistic for R² change
    f_stat = (delta_r2 / df1) / ((1 - r2_full) / df2)

    # p-value
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    return delta_r2, f_stat, p_value


def standardize_dataframe(
    df: pd.DataFrame,
    columns: list,
    suffix: str = '_z'
) -> pd.DataFrame:
    """
    Standardize (z-score) specified columns in a dataframe

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Column names to standardize
    suffix : str
        Suffix to add to standardized column names

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added standardized columns
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[f'{col}{suffix}'] = (df[col] - mean_val) / std_val
            else:
                warnings.warn(f"Column '{col}' has zero variance - cannot standardize")
                df[f'{col}{suffix}'] = 0

    return df


def simple_slope_test(
    beta_main: float,
    beta_interaction: float,
    se_main: float,
    se_interaction: float,
    cov_main_interaction: float,
    moderator_value: float
) -> Tuple[float, float, float, float]:
    """
    Test simple slope at a specific moderator value

    Parameters:
    -----------
    beta_main : float
        Main effect coefficient
    beta_interaction : float
        Interaction coefficient
    se_main : float
        Standard error of main effect
    se_interaction : float
        Standard error of interaction
    cov_main_interaction : float
        Covariance between main and interaction terms
    moderator_value : float
        Value of moderator to test (e.g., 0 for female, 1 for male)

    Returns:
    --------
    simple_slope : float
        Slope at specified moderator value
    se_simple : float
        Standard error of simple slope (using delta method)
    t_stat : float
        t-statistic
    p_value : float
        Two-tailed p-value
    """
    # Simple slope
    simple_slope = beta_main + beta_interaction * moderator_value

    # Variance using delta method
    var_simple = (
        se_main**2 +
        (moderator_value**2) * se_interaction**2 +
        2 * moderator_value * cov_main_interaction
    )
    se_simple = np.sqrt(var_simple)

    # t-test
    t_stat = simple_slope / se_simple
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=100))  # Conservative df

    return simple_slope, se_simple, t_stat, p_value


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 10000,
    statistic: str = 'mean_diff',
    random_state: Optional[int] = 42
) -> Tuple[float, float]:
    """
    Non-parametric permutation test for group differences

    Parameters:
    -----------
    group1, group2 : array-like
        Data for the two groups
    n_permutations : int
        Number of permutations
    statistic : str
        Test statistic: 'mean_diff', 't_stat', 'median_diff'
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    observed_stat : float
        Observed test statistic
    p_value : float
        Two-tailed permutation p-value
    """
    rng = np.random.default_rng(random_state)

    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # Compute observed statistic
    if statistic == 'mean_diff':
        observed_stat = np.mean(group1) - np.mean(group2)
    elif statistic == 't_stat':
        observed_stat, _ = stats.ttest_ind(group1, group2)
    elif statistic == 'median_diff':
        observed_stat = np.median(group1) - np.median(group2)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Permutation distribution
    perm_stats = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_group1 = shuffled[:n1]
        perm_group2 = shuffled[n1:]

        if statistic == 'mean_diff':
            perm_stat = np.mean(perm_group1) - np.mean(perm_group2)
        elif statistic == 't_stat':
            perm_stat, _ = stats.ttest_ind(perm_group1, perm_group2)
        elif statistic == 'median_diff':
            perm_stat = np.median(perm_group1) - np.median(perm_group2)

        perm_stats.append(perm_stat)

    perm_stats = np.array(perm_stats)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

    return observed_stat, p_value


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval

    Parameters:
    -----------
    data : array-like
        Input data
    statistic_func : callable
        Function that takes data and returns statistic (e.g., np.mean)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed

    Returns:
    --------
    point_estimate : float
        Point estimate on original data
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)

    # Point estimate
    point_estimate = statistic_func(data)

    # Bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper
