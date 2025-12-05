"""
Ex-Gaussian Distribution Utilities
===================================

Provides functions for fitting Ex-Gaussian distributions to RT data.

The Ex-Gaussian distribution decomposes RT into 3 parameters:
- μ (mu): Normal component mean (routine processing speed)
- σ (sigma): Normal component SD (processing variability)
- τ (tau): Exponential tail (attentional lapses/slow responses)

Usage:
    from analysis.utils.exgaussian import fit_exgaussian, fit_exgaussian_by_condition

    # Fit single RT distribution
    mu, sigma, tau = fit_exgaussian(rt_array)

    # Fit by condition (e.g., congruent/incongruent)
    params_df = fit_exgaussian_by_condition(trials_df, 'rt_ms', 'condition')
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Random state for reproducibility
RNG = np.random.default_rng(42)


# =============================================================================
# EX-GAUSSIAN PDF/CDF
# =============================================================================

def exgaussian_pdf(
    x: Union[float, np.ndarray],
    mu: float,
    sigma: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Probability density function of Ex-Gaussian distribution.

    Parameters
    ----------
    x : float or array-like
        RT values
    mu : float
        Normal component mean
    sigma : float
        Normal component standard deviation
    tau : float
        Exponential component mean (1/lambda)

    Returns
    -------
    float or array-like
        PDF values
    """
    x = np.asarray(x)

    if sigma <= 0 or tau <= 0:
        return np.zeros_like(x, dtype=float)

    lambda_val = 1 / tau

    term1 = lambda_val / 2
    term2 = np.exp(lambda_val / 2 * (2 * mu + lambda_val * sigma**2 - 2 * x))
    term3 = 1 - stats.norm.cdf((mu + lambda_val * sigma**2 - x) / sigma)

    pdf = term1 * term2 * term3
    return pdf


def exgaussian_cdf(
    x: Union[float, np.ndarray],
    mu: float,
    sigma: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Cumulative distribution function of Ex-Gaussian distribution.

    Parameters
    ----------
    x : float or array-like
        RT values
    mu : float
        Normal component mean
    sigma : float
        Normal component standard deviation
    tau : float
        Exponential component mean (1/lambda)

    Returns
    -------
    float or array-like
        CDF values
    """
    x = np.asarray(x)

    if sigma <= 0 or tau <= 0:
        return np.zeros_like(x, dtype=float)

    lambda_val = 1 / tau

    # Standard normal CDF
    z1 = (x - mu) / sigma
    z2 = (mu + lambda_val * sigma**2 - x) / sigma

    term1 = stats.norm.cdf(z1)
    term2 = np.exp(lambda_val / 2 * (2 * mu + lambda_val * sigma**2 - 2 * x))
    term3 = stats.norm.cdf(z2)

    cdf = term1 - term2 * term3
    return np.clip(cdf, 0, 1)


def exgaussian_mean(mu: float, sigma: float, tau: float) -> float:
    """Expected value of Ex-Gaussian: E[X] = μ + τ"""
    return mu + tau


def exgaussian_var(mu: float, sigma: float, tau: float) -> float:
    """Variance of Ex-Gaussian: Var[X] = σ² + τ²"""
    return sigma**2 + tau**2


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def fit_exgaussian(
    rt_data: Union[List, np.ndarray, pd.Series],
    method: str = "MLE",
    min_trials: int = 20,
    n_starts: int = 5,
) -> Tuple[float, float, float]:
    """
    Fit Ex-Gaussian distribution to RT data using maximum likelihood.

    Parameters
    ----------
    rt_data : array-like
        Reaction time data (positive values)
    method : str
        Fitting method ('MLE' only currently supported)
    min_trials : int
        Minimum number of valid trials required
    n_starts : int
        Number of random initializations to try

    Returns
    -------
    tuple
        (mu, sigma, tau) or (nan, nan, nan) if fitting fails
    """
    rt_data = np.asarray(rt_data, dtype=float)
    rt_data = rt_data[~np.isnan(rt_data)]
    rt_data = rt_data[rt_data > 0]

    if len(rt_data) < min_trials:
        return np.nan, np.nan, np.nan

    mean_rt = np.mean(rt_data)
    std_rt = np.std(rt_data)

    def neg_log_likelihood(params):
        mu, sigma, tau = params

        if sigma <= 0 or tau <= 0 or mu <= 0:
            return 1e10

        pdf_vals = exgaussian_pdf(rt_data, mu, sigma, tau)
        pdf_vals = np.maximum(pdf_vals, 1e-10)

        return -np.sum(np.log(pdf_vals))

    best_params = None
    best_nll = np.inf

    for _ in range(n_starts):
        init_mu = mean_rt * RNG.uniform(0.7, 1.0)
        init_sigma = std_rt * RNG.uniform(0.3, 0.7)
        init_tau = std_rt * RNG.uniform(0.1, 0.5)

        try:
            result = minimize(
                neg_log_likelihood,
                [init_mu, init_sigma, init_tau],
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return np.nan, np.nan, np.nan

    mu, sigma, tau = best_params

    # Sanity check
    if sigma <= 0 or tau <= 0 or mu <= 0:
        return np.nan, np.nan, np.nan

    return float(mu), float(sigma), float(tau)


def fit_exgaussian_dict(
    rt_data: Union[List, np.ndarray, pd.Series],
    **kwargs,
) -> Dict[str, float]:
    """
    Fit Ex-Gaussian and return as dictionary.

    Returns
    -------
    dict
        {'mu': float, 'sigma': float, 'tau': float}
    """
    mu, sigma, tau = fit_exgaussian(rt_data, **kwargs)
    return {"mu": mu, "sigma": sigma, "tau": tau}


def fit_exgaussian_by_participant(
    trials_df: pd.DataFrame,
    rt_col: str,
    participant_col: str = "participant_id",
    **fit_kwargs,
) -> pd.DataFrame:
    """
    Fit Ex-Gaussian for each participant.

    Parameters
    ----------
    trials_df : pd.DataFrame
        Trial-level data with RT column
    rt_col : str
        Name of RT column
    participant_col : str
        Name of participant ID column

    Returns
    -------
    pd.DataFrame
        One row per participant with columns: participant_id, mu, sigma, tau, n_trials
    """
    records = []
    for pid, group in trials_df.groupby(participant_col):
        rt_vals = group[rt_col].dropna().values
        mu, sigma, tau = fit_exgaussian(rt_vals, **fit_kwargs)
        records.append({
            participant_col: pid,
            "exg_mu": mu,
            "exg_sigma": sigma,
            "exg_tau": tau,
            "exg_n_trials": len(rt_vals),
        })
    return pd.DataFrame(records)


def fit_exgaussian_by_condition(
    trials_df: pd.DataFrame,
    rt_col: str,
    condition_col: str,
    participant_col: str = "participant_id",
    **fit_kwargs,
) -> pd.DataFrame:
    """
    Fit Ex-Gaussian for each participant × condition combination.

    Parameters
    ----------
    trials_df : pd.DataFrame
        Trial-level data
    rt_col : str
        Name of RT column
    condition_col : str
        Name of condition column
    participant_col : str
        Name of participant ID column

    Returns
    -------
    pd.DataFrame
        One row per participant×condition with columns:
        participant_id, condition, mu, sigma, tau, n_trials
    """
    records = []
    for (pid, cond), group in trials_df.groupby([participant_col, condition_col]):
        rt_vals = group[rt_col].dropna().values
        mu, sigma, tau = fit_exgaussian(rt_vals, **fit_kwargs)
        records.append({
            participant_col: pid,
            condition_col: cond,
            "exg_mu": mu,
            "exg_sigma": sigma,
            "exg_tau": tau,
            "exg_n_trials": len(rt_vals),
        })
    return pd.DataFrame(records)


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def compute_exgaussian_metrics(
    trials_df: pd.DataFrame,
    rt_col: str,
    participant_col: str = "participant_id",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Compute standard Ex-Gaussian metrics for each participant.

    Returns participant-level dataframe with:
    - mu, sigma, tau parameters
    - tau_proportion: τ / (μ + τ) - proportion of mean due to exponential
    - lapse_index: τ / σ - relative tail heaviness
    """
    params = fit_exgaussian_by_participant(
        trials_df, rt_col, participant_col
    )

    # Derived metrics
    params[f"{prefix}tau_proportion"] = params["exg_tau"] / (params["exg_mu"] + params["exg_tau"])
    params[f"{prefix}lapse_index"] = params["exg_tau"] / params["exg_sigma"]

    # Rename with prefix
    if prefix:
        rename_map = {
            "exg_mu": f"{prefix}mu",
            "exg_sigma": f"{prefix}sigma",
            "exg_tau": f"{prefix}tau",
            "exg_n_trials": f"{prefix}n_trials",
        }
        params = params.rename(columns=rename_map)

    return params


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo and validation of Ex-Gaussian fitting."""
    print("Ex-Gaussian Fitting Demo")
    print("=" * 50)

    # Generate synthetic Ex-Gaussian data
    np.random.seed(42)
    true_mu, true_sigma, true_tau = 400, 50, 100

    n_samples = 500
    normal_part = np.random.normal(true_mu, true_sigma, n_samples)
    exp_part = np.random.exponential(true_tau, n_samples)
    synthetic_rt = normal_part + exp_part

    # Fit
    fitted_mu, fitted_sigma, fitted_tau = fit_exgaussian(synthetic_rt)

    print(f"\nTrue parameters:   μ={true_mu}, σ={true_sigma}, τ={true_tau}")
    print(f"Fitted parameters: μ={fitted_mu:.1f}, σ={fitted_sigma:.1f}, τ={fitted_tau:.1f}")
    print(f"\nRecovery error:")
    print(f"  μ: {abs(fitted_mu - true_mu):.1f} ms ({abs(fitted_mu - true_mu)/true_mu*100:.1f}%)")
    print(f"  σ: {abs(fitted_sigma - true_sigma):.1f} ms ({abs(fitted_sigma - true_sigma)/true_sigma*100:.1f}%)")
    print(f"  τ: {abs(fitted_tau - true_tau):.1f} ms ({abs(fitted_tau - true_tau)/true_tau*100:.1f}%)")


if __name__ == "__main__":
    main()
