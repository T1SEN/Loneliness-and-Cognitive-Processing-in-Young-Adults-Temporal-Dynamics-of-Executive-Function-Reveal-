"""
Correlation Analysis
====================

Computes the Pearson correlation matrix (with p-values and 95% CIs) across UCLA
loneliness, DASS-21 subscales, and executive-function outcomes.

Variables:
    - UCLA Loneliness
    - DASS-21 subscales (Depression, Anxiety, Stress)
    - Executive Function: WCST (perseverative error rate), Stroop interference, PRP bottleneck

Output:
    results/publication/basic_analysis/correlation_matrix.csv
    results/publication/basic_analysis/correlation_pvalues.csv
    results/publication/basic_analysis/correlation_heatmap.png

Usage:
    python -m publication.basic_analysis.correlation_analysis --task overall
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import matplotlib
matplotlib.use("Agg")  # Headless backend

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from publication.basic_analysis.utils import (
    get_analysis_data,
    filter_vars,
    get_output_dir,
    CORRELATION_VARS,
    print_section_header,
    format_pvalue,
)
from publication.preprocessing.constants import VALID_TASKS


def compute_correlation_matrix(
    df: pd.DataFrame,
    variables: list[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlation matrix with p-values and 95% CIs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of (column_name, display_label) tuples
        Variables to correlate

    Returns
    -------
    tuple of (correlation_matrix, pvalue_matrix, ci_matrix)
    """
    cols = [c for c, _ in variables if c in df.columns]
    labels = [l for c, l in variables if c in df.columns]

    n_vars = len(cols)
    r_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))
    ci_lower = np.zeros((n_vars, n_vars))
    ci_upper = np.zeros((n_vars, n_vars))

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i == j:
                r_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                ci_lower[i, j] = 1.0
                ci_upper[i, j] = 1.0
            else:
                # Get pairwise complete cases
                mask = df[[col_i, col_j]].notna().all(axis=1)
                x = df.loc[mask, col_i]
                y = df.loc[mask, col_j]

                if len(x) < 3:
                    r_matrix[i, j] = np.nan
                    p_matrix[i, j] = np.nan
                    ci_lower[i, j] = np.nan
                    ci_upper[i, j] = np.nan
                    continue

                r, p = stats.pearsonr(x, y)
                r_matrix[i, j] = r
                p_matrix[i, j] = p

                # Fisher's z transformation for 95% CI
                z = np.arctanh(r)
                se_z = 1 / np.sqrt(len(x) - 3)
                ci_lower[i, j] = np.tanh(z - 1.96 * se_z)
                ci_upper[i, j] = np.tanh(z + 1.96 * se_z)

    # Create DataFrames
    r_df = pd.DataFrame(r_matrix, index=labels, columns=labels)
    p_df = pd.DataFrame(p_matrix, index=labels, columns=labels)

    # CI as formatted strings
    ci_strings = np.empty((n_vars, n_vars), dtype=object)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                ci_strings[i, j] = "[1.00, 1.00]"
            elif np.isnan(ci_lower[i, j]):
                ci_strings[i, j] = "NA"
            else:
                ci_strings[i, j] = f"[{ci_lower[i, j]:.3f}, {ci_upper[i, j]:.3f}]"

    ci_df = pd.DataFrame(ci_strings, index=labels, columns=labels)

    return r_df, p_df, ci_df


def create_correlation_heatmap(
    r_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    output_path: str,
    title: str = "Correlation Matrix"
) -> None:
    """
    Create publication-ready correlation heatmap.

    Parameters
    ----------
    r_matrix : pd.DataFrame
        Correlation coefficients
    p_matrix : pd.DataFrame
        P-values
    output_path : str
        Path to save the figure
    title : str
        Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(r_matrix, dtype=bool), k=1)

    # Create annotation matrix with significance markers
    annot_matrix = r_matrix.copy().astype(str)
    for i in range(len(r_matrix)):
        for j in range(len(r_matrix)):
            r_val = r_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]

            if i == j:
                annot_matrix.iloc[i, j] = ""
            elif pd.isna(r_val):
                annot_matrix.iloc[i, j] = "NA"
            else:
                sig_marker = ""
                if p_val < 0.001:
                    sig_marker = "***"
                elif p_val < 0.01:
                    sig_marker = "**"
                elif p_val < 0.05:
                    sig_marker = "*"
                annot_matrix.iloc[i, j] = f"{r_val:.2f}{sig_marker}"

    # Plot heatmap
    sns.heatmap(
        r_matrix,
        mask=mask,
        annot=annot_matrix,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add note
    plt.figtext(
        0.5, 0.02,
        "Note. *p < .05, **p < .01, ***p < .001",
        ha='center',
        fontsize=10,
        style='italic'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_ucla_ef_correlations(
    df: pd.DataFrame,
    variables: list[tuple[str, str]]
) -> pd.DataFrame:
    """
    Compute detailed correlations between UCLA and EF variables with CIs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of (column_name, display_label) tuples
        Variables to correlate with UCLA

    Returns
    -------
    pd.DataFrame
        Detailed correlation results
    """
    results = []

    ucla_col = 'ucla_score'
    if ucla_col not in df.columns:
        return pd.DataFrame()

    for col, label in variables:
        if col not in df.columns or col == ucla_col:
            continue

        mask = df[[ucla_col, col]].notna().all(axis=1)
        x = df.loc[mask, ucla_col]
        y = df.loc[mask, col]

        if len(x) < 3:
            continue

        r, p = stats.pearsonr(x, y)

        # Fisher's z for CI
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(len(x) - 3)
        ci_lower = np.tanh(z - 1.96 * se_z)
        ci_upper = np.tanh(z + 1.96 * se_z)

        results.append({
            'Variable': label,
            'Column': col,
            'N': len(x),
            'r': r,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'p': p,
            'Significant': p < 0.05,
        })

    return pd.DataFrame(results)


def run(task: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run correlation analysis.

    Parameters
    ----------
    verbose : bool
        Print results to console

    Returns
    -------
    dict
        Dictionary with correlation matrices and detailed results
    """
    if verbose:
        print_section_header("CORRELATION ANALYSIS")

    # Load data
    if verbose:
        print("\n  Loading data...")
    df = get_analysis_data(task)
    variables = filter_vars(df, CORRELATION_VARS)
    output_dir = get_output_dir(task)

    if verbose:
        print(f"  Total participants: N = {len(df)}")

    # Compute correlation matrix
    if verbose:
        print("\n  Computing correlation matrix...")

    r_matrix, p_matrix, ci_matrix = compute_correlation_matrix(df, variables)

    # Save correlation matrices
    r_matrix.to_csv(output_dir / "correlation_matrix.csv", encoding='utf-8-sig')
    p_matrix.to_csv(output_dir / "correlation_pvalues.csv", encoding='utf-8-sig')
    ci_matrix.to_csv(output_dir / "correlation_ci.csv", encoding='utf-8-sig')

    # Create heatmap
    if verbose:
        print("  Creating correlation heatmap...")
    create_correlation_heatmap(
        r_matrix, p_matrix,
        output_dir / "correlation_heatmap.png",
        title="Correlation Matrix: UCLA Loneliness, DASS-21, and Executive Function"
    )

    # Detailed UCLA-EF correlations
    ucla_ef_corr = compute_ucla_ef_correlations(df, variables)
    if len(ucla_ef_corr) > 0:
        ucla_ef_corr.to_csv(output_dir / "ucla_correlations_detailed.csv", index=False, encoding='utf-8-sig')

    # Print results
    if verbose:
        print("\n  Correlation Matrix (Pearson r)")
        print("  " + "-" * 65)

        # Print lower triangle only
        labels = r_matrix.columns.tolist()
        print(f"  {'':>12}", end="")
        for label in labels:
            print(f" {label:>10}", end="")
        print()

        for i, row_label in enumerate(labels):
            print(f"  {row_label:>12}", end="")
            for j in range(len(labels)):
                if j <= i:
                    r_val = r_matrix.iloc[i, j]
                    p_val = p_matrix.iloc[i, j]
                    if i == j:
                        print(f" {'--':>10}", end="")
                    elif pd.isna(r_val):
                        print(f" {'NA':>10}", end="")
                    else:
                        sig = "*" if p_val < 0.05 else ""
                        sig = "**" if p_val < 0.01 else sig
                        sig = "***" if p_val < 0.001 else sig
                        print(f" {r_val:>7.3f}{sig:<2}", end="")
                else:
                    print(f" {'':>10}", end="")
            print()

        print("  " + "-" * 65)
        print("  Note. *p < .05, **p < .01, ***p < .001")

        # UCLA-specific correlations
        if len(ucla_ef_corr) > 0:
            print("\n  UCLA Loneliness Correlations with 95% CI")
            print("  " + "-" * 65)
            for _, row in ucla_ef_corr.iterrows():
                sig = "*" if row['Significant'] else ""
                print(f"  {row['Variable']:<25}: r = {row['r']:.3f}{sig} "
                      f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}], p = {format_pvalue(row['p'])}")

        print("\n  Output files:")
        print(f"    - {output_dir / 'correlation_matrix.csv'}")
        print(f"    - {output_dir / 'correlation_pvalues.csv'}")
        print(f"    - {output_dir / 'correlation_heatmap.png'}")
        if len(ucla_ef_corr) > 0:
            print(f"    - {output_dir / 'ucla_correlations_detailed.csv'}")

    return {
        'r_matrix': r_matrix,
        'p_matrix': p_matrix,
        'ci_matrix': ci_matrix,
        'ucla_ef_correlations': ucla_ef_corr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlation analysis")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(VALID_TASKS),
        help="Dataset task to analyze (overall, stroop, prp, wcst).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run(task=args.task, verbose=True)
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS COMPLETE")
    print("=" * 70)
