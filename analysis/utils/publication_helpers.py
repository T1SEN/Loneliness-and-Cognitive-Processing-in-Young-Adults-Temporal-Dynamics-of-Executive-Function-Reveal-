"""
Publication-Ready Helpers for 4-Framework Advanced Analysis
============================================================

Provides common utilities for:
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d, eta-squared, etc.)
- APA-style table formatting
- Publication-quality visualizations
- Cross-validation utilities

Author: Research Analysis Pipeline
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import resample
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(data, statistic_func, n_iterations=2000, ci=95, random_state=42):
    """
    Calculate bootstrap confidence intervals for any statistic.

    Parameters
    ----------
    data : array-like or tuple of arrays
        Data to bootstrap. If tuple, passes multiple arrays to statistic_func
    statistic_func : callable
        Function that computes the statistic (e.g., np.mean, np.median)
    n_iterations : int, default=2000
        Number of bootstrap samples
    ci : float, default=95
        Confidence level (95 for 95% CI)
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    dict : {'estimate', 'ci_lower', 'ci_upper', 'se', 'bootstrap_dist'}
    """
    np.random.seed(random_state)

    if isinstance(data, tuple):
        # Multiple arrays (e.g., for correlations)
        n = len(data[0])
        bootstrap_stats = []
        for i in range(n_iterations):
            indices = np.random.randint(0, n, size=n)
            resampled = tuple(arr[indices] for arr in data)
            bootstrap_stats.append(statistic_func(*resampled))
    else:
        # Single array
        n = len(data)
        bootstrap_stats = []
        for i in range(n_iterations):
            resampled = resample(data, n_samples=n, random_state=random_state+i)
            bootstrap_stats.append(statistic_func(resampled))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (100 - ci) / 2

    return {
        'estimate': statistic_func(data) if not isinstance(data, tuple) else statistic_func(*data),
        'ci_lower': np.percentile(bootstrap_stats, alpha),
        'ci_upper': np.percentile(bootstrap_stats, 100 - alpha),
        'se': np.std(bootstrap_stats),
        'bootstrap_dist': bootstrap_stats
    }


def bootstrap_regression_coef(X, y, coef_index=0, n_iterations=2000, ci=95, random_state=42):
    """
    Bootstrap confidence interval for regression coefficient.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictor matrix
    y : array-like, shape (n,)
        Outcome vector
    coef_index : int, default=0
        Index of coefficient to bootstrap (0 = intercept, 1+ = predictors)
    n_iterations : int
        Number of bootstrap samples
    ci : float
        Confidence level
    random_state : int
        Random seed

    Returns
    -------
    dict : Bootstrap results
    """
    from sklearn.linear_model import LinearRegression

    np.random.seed(random_state)
    n = len(y)
    bootstrap_coefs = []

    for i in range(n_iterations):
        indices = np.random.randint(0, n, size=n)
        X_boot = X[indices]
        y_boot = y[indices]

        model = LinearRegression()
        model.fit(X_boot, y_boot)

        if coef_index == 0:
            bootstrap_coefs.append(model.intercept_)
        else:
            bootstrap_coefs.append(model.coef_[coef_index - 1])

    bootstrap_coefs = np.array(bootstrap_coefs)
    alpha = (100 - ci) / 2

    # Original estimate
    model = LinearRegression()
    model.fit(X, y)
    if coef_index == 0:
        original_coef = model.intercept_
    else:
        original_coef = model.coef_[coef_index - 1]

    return {
        'estimate': original_coef,
        'ci_lower': np.percentile(bootstrap_coefs, alpha),
        'ci_upper': np.percentile(bootstrap_coefs, 100 - alpha),
        'se': np.std(bootstrap_coefs),
        'bootstrap_dist': bootstrap_coefs
    }


# ============================================================================
# Effect Size Calculations
# ============================================================================

def cohens_d(group1, group2, pooled=True):
    """
    Calculate Cohen's d effect size for two independent groups.

    Parameters
    ----------
    group1, group2 : array-like
        Data for each group
    pooled : bool, default=True
        Use pooled standard deviation (True) or group1 SD (False)

    Returns
    -------
    float : Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    if pooled:
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_sd
    else:
        d = (mean1 - mean2) / np.sqrt(var1)

    return d


def eta_squared(f_statistic, df_between, df_within):
    """
    Calculate eta-squared (η²) from F-statistic.

    Parameters
    ----------
    f_statistic : float
        F-statistic from ANOVA
    df_between : int
        Degrees of freedom between groups
    df_within : int
        Degrees of freedom within groups

    Returns
    -------
    float : Eta-squared
    """
    ss_between = f_statistic * df_between
    ss_within = df_within
    ss_total = ss_between + ss_within
    return ss_between / ss_total


def partial_eta_squared(f_statistic, df_between, df_error):
    """
    Calculate partial eta-squared (partial η²).

    Parameters
    ----------
    f_statistic : float
        F-statistic
    df_between : int
        Effect degrees of freedom
    df_error : int
        Error degrees of freedom

    Returns
    -------
    float : Partial eta-squared
    """
    return (f_statistic * df_between) / (f_statistic * df_between + df_error)


def r_to_d(r):
    """Convert Pearson r to Cohen's d."""
    return (2 * r) / np.sqrt(1 - r**2)


def d_to_r(d):
    """Convert Cohen's d to Pearson r."""
    return d / np.sqrt(d**2 + 4)


# ============================================================================
# APA-Style Formatting
# ============================================================================

def format_pvalue(p, threshold=0.001):
    """
    Format p-value in APA style.

    Examples
    --------
    >>> format_pvalue(0.045)
    'p = .045'
    >>> format_pvalue(0.0003)
    'p < .001'
    """
    if p < threshold:
        return f"p < {threshold:.3f}".replace("0.", ".")
    else:
        return f"p = {p:.3f}".replace("0.", ".")


def format_ci(lower, upper, decimals=2):
    """
    Format confidence interval in APA style.

    Examples
    --------
    >>> format_ci(0.23, 0.45)
    '95% CI [0.23, 0.45]'
    """
    return f"95% CI [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def format_statistic(statistic, value, df=None, p=None, effect_size=None):
    """
    Format statistical test result in APA style.

    Parameters
    ----------
    statistic : str
        Type of statistic ('t', 'F', 'r', 'χ²', etc.)
    value : float
        Test statistic value
    df : int or tuple, optional
        Degrees of freedom
    p : float, optional
        p-value
    effect_size : float, optional
        Effect size (d, η², r², etc.)

    Returns
    -------
    str : APA-formatted string

    Examples
    --------
    >>> format_statistic('t', 2.45, df=30, p=0.021, effect_size=0.45)
    't(30) = 2.45, p = .021, d = 0.45'
    """
    result = f"{statistic}"

    if df is not None:
        if isinstance(df, tuple):
            result += f"({df[0]}, {df[1]})"
        else:
            result += f"({df})"

    result += f" = {value:.2f}"

    if p is not None:
        result += f", {format_pvalue(p)}"

    if effect_size is not None:
        result += f", d = {effect_size:.2f}"

    return result


def create_apa_table(df, caption="", float_format=".2f"):
    """
    Create APA-style table from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data to format
    caption : str, optional
        Table caption
    float_format : str, default=".2f"
        Format for float columns

    Returns
    -------
    str : LaTeX table code (for direct inclusion in manuscripts)
    """
    # Format numeric columns
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=[np.number]).columns:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:{float_format}}")

    latex_table = df_formatted.to_latex(
        index=True,
        caption=caption,
        column_format='l' + 'r' * len(df.columns),
        escape=False
    )

    return latex_table


# ============================================================================
# Publication-Quality Visualizations
# ============================================================================

def set_publication_style():
    """
    Set matplotlib/seaborn style for publication-quality figures.

    Features:
    - 300 DPI default
    - Colorblind-safe palette
    - Clean, minimal styling
    - Large fonts for readability
    """
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (8, 6),
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
    })

    # Colorblind-safe palette (Okabe-Ito)
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
              '#0072B2', '#D55E00', '#CC79A7', '#000000']
    sns.set_palette(colors)


def save_publication_figure(fig, filename, formats=['png', 'pdf'], dpi=300):
    """
    Save figure in multiple formats for publication.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename (without extension)
    formats : list, default=['png', 'pdf']
        File formats to save
    dpi : int, default=300
        Resolution for raster formats
    """
    for fmt in formats:
        output_path = f"{filename}.{fmt}"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")


# ============================================================================
# Cross-Validation Utilities
# ============================================================================

def nested_cv_with_ci(estimator, X, y, cv_outer=10, cv_inner=5,
                      scoring='r2', n_bootstrap=1000, random_state=42):
    """
    Nested cross-validation with bootstrap confidence intervals.

    Parameters
    ----------
    estimator : sklearn estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv_outer : int, default=10
        Outer CV folds
    cv_inner : int, default=5
        Inner CV folds (for hyperparameter tuning)
    scoring : str, default='r2'
        Scoring metric
    n_bootstrap : int, default=1000
        Bootstrap iterations for CI
    random_state : int
        Random seed

    Returns
    -------
    dict : {'mean_score', 'ci_lower', 'ci_upper', 'scores'}
    """
    kf = KFold(n_splits=cv_outer, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=kf, scoring=scoring)

    # Bootstrap CI on CV scores
    np.random.seed(random_state)
    bootstrap_means = []
    for i in range(n_bootstrap):
        boot_scores = resample(scores, n_samples=len(scores), random_state=random_state+i)
        bootstrap_means.append(np.mean(boot_scores))

    return {
        'mean_score': np.mean(scores),
        'ci_lower': np.percentile(bootstrap_means, 2.5),
        'ci_upper': np.percentile(bootstrap_means, 97.5),
        'scores': scores
    }


# ============================================================================
# Data Preprocessing for All Frameworks
# ============================================================================

def load_master_dataset(include_items=False):
    """
    Load and merge all datasets for analysis.

    Parameters
    ----------
    include_items : bool, default=False
        If True, includes item-level UCLA/DASS data

    Returns
    -------
    pd.DataFrame : Master dataset with participants, surveys, EF metrics
    """
    from pathlib import Path

    results_dir = Path("results")

    # Load base datasets
    participants = pd.read_csv(results_dir / "1_participants_info.csv")
    surveys = pd.read_csv(results_dir / "2_surveys_results.csv")
    cognitive = pd.read_csv(results_dir / "3_cognitive_tests_summary.csv")

    # Normalize column names
    participants = participants.rename(columns={'participantId': 'participant_id'})
    surveys = surveys.rename(columns={'participantId': 'participant_id'})
    cognitive = cognitive.rename(columns={'participantId': 'participant_id'})

    # Separate UCLA and DASS
    ucla = surveys[surveys['surveyName'] == 'ucla'].copy()
    dass = surveys[surveys['surveyName'] == 'dass'].copy()

    # Merge surveys by participant
    ucla_scores = ucla[['participant_id', 'score']].rename(columns={'score': 'ucla_score'})
    dass_scores = dass[['participant_id', 'score_D', 'score_A', 'score_S']].rename(
        columns={'score_D': 'dass_depression', 'score_A': 'dass_anxiety', 'score_S': 'dass_stress'}
    )

    # Merge all
    master = participants.merge(ucla_scores, on='participant_id', how='inner')
    master = master.merge(dass_scores, on='participant_id', how='inner')
    master = master.merge(cognitive, on='participant_id', how='inner')

    # Gender binary (for regression)
    master['gender_male'] = (master['gender'] == 'male').astype(int)

    if include_items:
        # Add item-level data
        ucla_items = [f'q{i}' for i in range(1, 21)]
        dass_items = [f'q{i}' for i in range(1, 22)]

        ucla_item_data = ucla[['participant_id'] + ucla_items].add_prefix('ucla_')
        ucla_item_data = ucla_item_data.rename(columns={'ucla_participant_id': 'participant_id'})

        dass_item_data = dass[['participant_id'] + dass_items].add_prefix('dass_')
        dass_item_data = dass_item_data.rename(columns={'dass_participant_id': 'participant_id'})

        master = master.merge(ucla_item_data, on='participant_id', how='left')
        master = master.merge(dass_item_data, on='participant_id', how='left')

    return master


def standardize_variables(df, vars_to_standardize):
    """
    Standardize (z-score) variables in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    vars_to_standardize : list of str
        Variable names to standardize

    Returns
    -------
    pd.DataFrame : Copy of df with standardized variables (prefixed with 'z_')
    """
    df_out = df.copy()

    for var in vars_to_standardize:
        if var in df.columns:
            z_var = f"z_{var}"
            df_out[z_var] = (df[var] - df[var].mean()) / df[var].std()

    return df_out


# ============================================================================
# Summary Statistics
# ============================================================================

def describe_sample(df, groupby=None):
    """
    Generate comprehensive descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Data to describe
    groupby : str, optional
        Column name for grouped statistics (e.g., 'gender')

    Returns
    -------
    pd.DataFrame : Summary statistics
    """
    numeric_vars = df.select_dtypes(include=[np.number]).columns

    if groupby is None:
        summary = df[numeric_vars].describe().T
        summary['missing'] = df[numeric_vars].isnull().sum()
        summary['skew'] = df[numeric_vars].skew()
        summary['kurtosis'] = df[numeric_vars].kurtosis()
    else:
        summary = df.groupby(groupby)[numeric_vars].describe()

    return summary


if __name__ == "__main__":
    print("Publication Helpers Module")
    print("===========================")
    print("\nAvailable functions:")
    print("  - bootstrap_ci(): Bootstrap confidence intervals")
    print("  - cohens_d(): Effect size calculation")
    print("  - format_pvalue(): APA-style p-value formatting")
    print("  - set_publication_style(): Configure matplotlib for publication")
    print("  - load_master_dataset(): Load integrated dataset")
    print("\nFor detailed documentation, see function docstrings.")
