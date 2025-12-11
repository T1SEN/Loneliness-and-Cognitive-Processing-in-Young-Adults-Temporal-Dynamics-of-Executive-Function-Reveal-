"""
Descriptive Statistics Analysis
===============================

Generates Table-1 style descriptive summaries (N, Mean, SD, Min, Max) for the
core demographic, survey, and executive-function measures.

Variables:
    - Demographics: age
    - Surveys: UCLA loneliness, DASS-21 subscales
    - Executive Function: WCST (perseverative error rate), Stroop interference, PRP bottleneck

Output:
    results/publication/basic_analysis/table1_descriptives.csv
    results/publication/basic_analysis/table1_descriptives_by_gender.csv

Usage:
    python -m publication.basic_analysis.descriptive_statistics
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats

from publication.basic_analysis.utils import (
    get_analysis_data,
    OUTPUT_DIR,
    DESCRIPTIVE_VARS,
    print_section_header,
    format_coefficient,
)


def compute_descriptive_stats(
    df: pd.DataFrame,
    variables: list[tuple[str, str]],
    group_label: str = "Total"
) -> pd.DataFrame:
    """
    Compute descriptive statistics for specified variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of (column_name, display_label) tuples
        Variables to analyze
    group_label : str
        Label for this group (e.g., "Total", "Male", "Female")

    Returns
    -------
    pd.DataFrame
        Descriptive statistics table
    """
    results = []

    for col, label in variables:
        if col not in df.columns:
            print(f"  [WARNING] Variable '{col}' not found in dataset")
            continue

        series = df[col].dropna()

        results.append({
            'Group': group_label,
            'Variable': label,
            'Column': col,
            'N': len(series),
            'Mean': series.mean(),
            'SD': series.std(),
            'Min': series.min(),
            'Max': series.max(),
            'Median': series.median(),
            'Skewness': stats.skew(series) if len(series) > 2 else np.nan,
            'Kurtosis': stats.kurtosis(series) if len(series) > 2 else np.nan,
        })

    return pd.DataFrame(results)


def compute_gender_comparison(
    df: pd.DataFrame,
    variables: list[tuple[str, str]]
) -> pd.DataFrame:
    """
    Compute descriptive statistics by gender with t-test comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'gender_male' column
    variables : list of (column_name, display_label) tuples
        Variables to analyze

    Returns
    -------
    pd.DataFrame
        Descriptive statistics by gender with comparison tests
    """
    results = []

    for col, label in variables:
        if col not in df.columns:
            continue

        male_data = df[df['gender_male'] == 1][col].dropna()
        female_data = df[df['gender_male'] == 0][col].dropna()

        # Skip if insufficient data
        if len(male_data) < 5 or len(female_data) < 5:
            continue

        # Independent samples t-test (Welch's, unequal variance)
        t_stat, p_value = stats.ttest_ind(male_data, female_data, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(male_data) - 1) * male_data.std()**2 +
             (len(female_data) - 1) * female_data.std()**2) /
            (len(male_data) + len(female_data) - 2)
        )
        cohens_d = (male_data.mean() - female_data.mean()) / pooled_std if pooled_std > 0 else np.nan

        results.append({
            'Variable': label,
            'Column': col,
            'Male_N': len(male_data),
            'Male_Mean': male_data.mean(),
            'Male_SD': male_data.std(),
            'Female_N': len(female_data),
            'Female_Mean': female_data.mean(),
            'Female_SD': female_data.std(),
            't': t_stat,
            'p': p_value,
            'Cohen_d': cohens_d,
            'Significant': p_value < 0.05,
        })

    return pd.DataFrame(results)


def compute_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute categorical variable statistics (gender frequency/percentage).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'gender_male' column

    Returns
    -------
    pd.DataFrame
        Categorical statistics with N and Percent
    """
    results = []

    if 'gender_male' in df.columns:
        n_male = (df['gender_male'] == 1).sum()
        n_female = (df['gender_male'] == 0).sum()
        n_missing = df['gender_male'].isna().sum()
        n_total = n_male + n_female

        if n_total > 0:
            results.append({
                'Variable': 'Gender',
                'Category': 'Male',
                'N': n_male,
                'Percent': n_male / n_total * 100
            })
            results.append({
                'Variable': 'Gender',
                'Category': 'Female',
                'N': n_female,
                'Percent': n_female / n_total * 100
            })
            if n_missing > 0:
                results.append({
                    'Variable': 'Gender',
                    'Category': 'Missing',
                    'N': n_missing,
                    'Percent': np.nan
                })

    return pd.DataFrame(results)


def print_apa_table(desc_df: pd.DataFrame, cat_df: pd.DataFrame = None) -> None:
    """Print descriptive statistics in APA-style format."""
    print("\n  Table 1. Descriptive Statistics")
    print("  " + "-" * 65)

    # Categorical variables (Gender)
    if cat_df is not None and len(cat_df) > 0:
        print(f"  {'Variable':<35} {'N':>6} {'%':>10}")
        print("  " + "-" * 65)
        for _, row in cat_df.iterrows():
            if pd.notna(row['Percent']):
                print(f"  {row['Category']:<35} {row['N']:>6} {row['Percent']:>10.1f}")
            else:
                print(f"  {row['Category']:<35} {row['N']:>6} {'--':>10}")
        print("  " + "-" * 65)

    # Continuous variables
    print(f"  {'Variable':<35} {'N':>6} {'M':>10} {'SD':>10}")
    print("  " + "-" * 65)

    for _, row in desc_df.iterrows():
        print(f"  {row['Variable']:<35} {row['N']:>6} {row['Mean']:>10.2f} {row['SD']:>10.2f}")

    print("  " + "-" * 65)
    print("  Note. M = Mean; SD = Standard Deviation")


def run(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run descriptive statistics analysis.

    Parameters
    ----------
    verbose : bool
        Print results to console

    Returns
    -------
    dict
        Dictionary with 'total' and 'by_gender' DataFrames
    """
    if verbose:
        print_section_header("DESCRIPTIVE STATISTICS ANALYSIS")

    # Load data
    if verbose:
        print("\n  Loading data...")
    df = get_analysis_data(use_cache=True)

    if verbose:
        print(f"  Total participants: N = {len(df)}")
        if 'gender_male' in df.columns:
            n_male = (df['gender_male'] == 1).sum()
            n_female = (df['gender_male'] == 0).sum()
            print(f"  Male: n = {n_male}, Female: n = {n_female}")

    # Compute overall descriptive statistics
    if verbose:
        print("\n  Computing descriptive statistics...")

    desc_total = compute_descriptive_stats(df, DESCRIPTIVE_VARS, group_label="Total")

    # Compute categorical statistics (gender frequency/percentage)
    cat_stats = compute_categorical_stats(df)

    # Compute by gender
    if 'gender_male' in df.columns:
        desc_male = compute_descriptive_stats(
            df[df['gender_male'] == 1], DESCRIPTIVE_VARS, group_label="Male"
        )
        desc_female = compute_descriptive_stats(
            df[df['gender_male'] == 0], DESCRIPTIVE_VARS, group_label="Female"
        )
        desc_by_gender = pd.concat([desc_total, desc_male, desc_female], ignore_index=True)

        # Gender comparison with t-tests
        gender_comparison = compute_gender_comparison(df, DESCRIPTIVE_VARS)
    else:
        desc_by_gender = desc_total
        gender_comparison = pd.DataFrame()

    # Save results
    output_file = OUTPUT_DIR / "table1_descriptives.csv"
    desc_total.to_csv(output_file, index=False, encoding='utf-8-sig')

    output_file_gender = OUTPUT_DIR / "table1_descriptives_by_gender.csv"
    desc_by_gender.to_csv(output_file_gender, index=False, encoding='utf-8-sig')

    # Save categorical statistics
    if len(cat_stats) > 0:
        output_file_categorical = OUTPUT_DIR / "table1_categorical.csv"
        cat_stats.to_csv(output_file_categorical, index=False, encoding='utf-8-sig')

    if len(gender_comparison) > 0:
        output_file_comparison = OUTPUT_DIR / "table1_gender_comparison.csv"
        gender_comparison.to_csv(output_file_comparison, index=False, encoding='utf-8-sig')

    if verbose:
        print_apa_table(desc_total, cat_stats)

        if len(gender_comparison) > 0:
            print("\n  Gender Comparison (t-tests)")
            print("  " + "-" * 65)
            sig_vars = gender_comparison[gender_comparison['Significant']]
            if len(sig_vars) > 0:
                for _, row in sig_vars.iterrows():
                    print(f"  {row['Variable']}: t = {row['t']:.2f}, p = {row['p']:.3f}, d = {row['Cohen_d']:.2f}")
            else:
                print("  No significant gender differences found (p < .05)")

        print(f"\n  Output files:")
        print(f"    - {output_file}")
        print(f"    - {output_file_gender}")
        if len(cat_stats) > 0:
            print(f"    - {OUTPUT_DIR / 'table1_categorical.csv'}")
        if len(gender_comparison) > 0:
            print(f"    - {OUTPUT_DIR / 'table1_gender_comparison.csv'}")

    return {
        'total': desc_total,
        'categorical': cat_stats,
        'by_gender': desc_by_gender,
        'gender_comparison': gender_comparison,
    }


if __name__ == "__main__":
    results = run(verbose=True)
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS COMPLETE")
    print("=" * 70)
