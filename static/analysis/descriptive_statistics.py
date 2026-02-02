"""
Descriptive Statistics Analysis
===============================

Generates Table-1 style descriptive summaries (N, Mean, SD, Min, Max) for the
core demographic, survey, and executive-function measures.

Variables:
    - Demographics: age
    - Surveys: UCLA loneliness, DASS-21 subscales
    - Executive Function: Stroop interference + interference slope
    - Executive Function: WCST PE rate + phase RTs (exploration, confirmation, exploitation)

Output:
    results/static/analysis/table1_descriptives.csv
    results/static/analysis/table1_descriptives_by_gender.csv

Usage:
    python -m static.analysis.descriptive_statistics --task overall
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

from static.analysis.utils import (
    get_analysis_data,
    filter_vars,
    get_output_dir,
    DESCRIPTIVE_VARS,
    print_section_header,
    format_coefficient,
    run_ucla_regression,
)
from static.preprocessing.constants import (
    VALID_TASKS,
    STROOP_RT_MIN,
    STROOP_RT_MAX,
    get_results_dir,
)
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.wcst.qc import clean_wcst_trials


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes"})


def _load_qc_ids(task: str) -> set[str]:
    task_dir = get_results_dir(task)
    ids_path = task_dir / "filtered_participant_ids.csv"
    if not ids_path.exists():
        return set()
    ids_df = pd.read_csv(ids_path, encoding="utf-8-sig")
    ids_df = ensure_participant_id(ids_df)
    if "participant_id" not in ids_df.columns:
        return set()
    return set(ids_df["participant_id"].dropna().astype(str))


def _prepare_stroop_trials(task: str) -> pd.DataFrame:
    data_dir = get_results_dir(task)
    trials = _read_csv(data_dir / "4a_stroop_trials.csv")
    if trials.empty:
        return trials
    trials = ensure_participant_id(trials)
    qc_ids = _load_qc_ids(task)
    if qc_ids:
        trials = trials[trials["participant_id"].isin(qc_ids)]
    trials["is_rt_valid"] = _coerce_bool(trials["is_rt_valid"])
    trials["timeout"] = _coerce_bool(trials["timeout"])
    trials["correct"] = _coerce_bool(trials["correct"])
    trials["cond"] = trials["cond"].astype(str).str.lower()
    trials["rt_ms"] = pd.to_numeric(trials["rt_ms"], errors="coerce")
    if "trial_index" in trials.columns:
        trials["trial_order"] = pd.to_numeric(trials["trial_index"], errors="coerce")
    else:
        trials["trial_order"] = pd.to_numeric(trials["trial_order"], errors="coerce")
    valid = (
        trials["cond"].isin({"congruent", "incongruent"})
        & trials["correct"]
        & (~trials["timeout"])
        & trials["is_rt_valid"]
        & trials["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
    )
    return trials[valid].dropna(subset=["participant_id", "trial_order", "rt_ms"])


def _load_stroop_trials_raw(task: str) -> pd.DataFrame:
    data_dir = get_results_dir(task)
    trials = _read_csv(data_dir / "4a_stroop_trials.csv")
    if trials.empty:
        return trials
    trials = ensure_participant_id(trials)
    qc_ids = _load_qc_ids(task)
    if qc_ids:
        trials = trials[trials["participant_id"].isin(qc_ids)]
    trials["cond"] = trials["cond"].astype(str).str.lower()
    trials["rt_ms"] = pd.to_numeric(trials["rt_ms"], errors="coerce")
    if "trial_index" in trials.columns:
        trials["trial_order"] = pd.to_numeric(trials["trial_index"], errors="coerce")
    else:
        trials["trial_order"] = pd.to_numeric(trials["trial_order"], errors="coerce")
    if "timeout" in trials.columns:
        trials["timeout"] = _coerce_bool(trials["timeout"])
    else:
        trials["timeout"] = False
    if "correct" in trials.columns:
        trials["correct"] = _coerce_bool(trials["correct"])
    else:
        trials["correct"] = False
    if "is_rt_valid" in trials.columns:
        trials["is_rt_valid"] = _coerce_bool(trials["is_rt_valid"])
    else:
        trials["is_rt_valid"] = trials["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
    return trials


def _prepare_wcst_trials(task: str) -> pd.DataFrame:
    data_dir = get_results_dir(task)
    wcst_raw = _read_csv(data_dir / "4b_wcst_trials.csv")
    if wcst_raw.empty:
        return wcst_raw
    wcst = clean_wcst_trials(wcst_raw)
    qc_ids = _load_qc_ids(task)
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)].copy()
    return wcst


def _compute_stroop_interference_ttest(task: str) -> pd.DataFrame:
    trials = _prepare_stroop_trials(task)
    if trials.empty:
        return pd.DataFrame()
    means = trials.groupby(["participant_id", "cond"])["rt_ms"].mean().unstack()
    if "incongruent" not in means.columns or "congruent" not in means.columns:
        return pd.DataFrame()
    interference = means["incongruent"] - means["congruent"]
    interference = interference.dropna()
    n = int(len(interference))
    t_stat, p_val = stats.ttest_1samp(interference, 0.0)
    return pd.DataFrame(
        [
            {
                "n": n,
                "mean": float(interference.mean()),
                "sd": float(interference.std(ddof=1)),
                "t": float(t_stat),
                "p": float(p_val),
            }
        ]
    )


def _compute_wcst_normative_stats(task: str) -> pd.DataFrame:
    wcst = _prepare_wcst_trials(task)
    if wcst.empty:
        return pd.DataFrame()
    wcst = wcst.sort_values(["participant_id", "trial_order"])
    cat_counts = (
        wcst.groupby("participant_id")["rule"]
        .apply(lambda s: s.ne(s.shift()).sum())
        .rename("n_categories")
    )
    cat_counts = cat_counts.clip(upper=6)
    n = int(cat_counts.shape[0])
    complete6 = int((cat_counts >= 6).sum())
    pct_complete6 = float(complete6 / n * 100) if n else np.nan
    return pd.DataFrame(
        [
            {
                "n": n,
                "mean_categories": float(cat_counts.mean()),
                "sd_categories": float(cat_counts.std(ddof=1)),
                "n_complete6": complete6,
                "pct_complete6": pct_complete6,
            }
        ]
    )


def _assign_segments(df: pd.DataFrame, n_segments: int) -> pd.DataFrame:
    if df.empty:
        return df.assign(segment=np.nan)

    def _assign(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trial_order").copy()
        group["segment"] = np.nan
        valid = group["trial_order"].notna()
        if not valid.any():
            return group
        sub = group[valid].copy()
        n_trials = len(sub)
        if n_trials == 0:
            return group
        positions = np.arange(1, n_trials + 1)
        edges = np.linspace(0, n_trials, n_segments + 1)
        sub["segment"] = pd.cut(
            positions,
            bins=edges,
            labels=list(range(1, n_segments + 1)),
            include_lowest=True,
        ).astype(float)
        group.loc[sub.index, "segment"] = sub["segment"]
        return group

    return df.groupby("participant_id", group_keys=False).apply(_assign)


def _compute_condition_balance(task: str, n_segments: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = _load_stroop_trials_raw(task)
    if trials.empty:
        return pd.DataFrame(), pd.DataFrame()

    trials = trials[trials["cond"].isin({"congruent", "incongruent", "neutral"})]
    trials = trials.dropna(subset=["trial_order"])
    if trials.empty:
        return pd.DataFrame(), pd.DataFrame()

    trials = _assign_segments(trials, n_segments)
    trials = trials.dropna(subset=["segment"])
    if trials.empty:
        return pd.DataFrame(), pd.DataFrame()

    trials["segment"] = trials["segment"].astype(int)
    counts = trials.groupby(["segment", "cond"]).size().reset_index(name="count")
    totals = counts.groupby("segment")["count"].transform("sum")
    counts["total_segment"] = totals
    counts["proportion"] = counts["count"] / counts["total_segment"]

    pivot = (
        counts.pivot(index="segment", columns="cond", values="proportion")
        .reindex(columns=["congruent", "incongruent", "neutral"])
        .reset_index()
    )
    return counts, pivot


def _compute_interference_slope(trials: pd.DataFrame, n_segments: int) -> pd.Series:
    if trials.empty:
        return pd.Series(dtype=float)
    trials = trials[trials["cond"].isin({"congruent", "incongruent"})].copy()
    valid = (
        trials["correct"]
        & (~trials["timeout"])
        & trials["is_rt_valid"]
        & trials["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
    )
    trials = trials[valid].copy()
    if trials.empty:
        return pd.Series(dtype=float)

    trials = _assign_segments(trials, n_segments)
    trials = trials.dropna(subset=["segment"])
    if trials.empty:
        return pd.Series(dtype=float)

    seg_means = (
        trials.groupby(["participant_id", "segment", "cond"])["rt_ms"].mean().unstack()
    )
    if "incongruent" not in seg_means.columns or "congruent" not in seg_means.columns:
        return pd.Series(dtype=float)

    seg_means["interference"] = seg_means["incongruent"] - seg_means["congruent"]
    seg_means = seg_means.reset_index()[["participant_id", "segment", "interference"]]

    def _slope(group: pd.DataFrame) -> float:
        group = group.dropna(subset=["interference"])
        if len(group) < 2:
            return np.nan
        x = group["segment"].astype(float).to_numpy()
        y = group["interference"].to_numpy()
        return float(np.polyfit(x, y, 1)[0])

    return seg_means.groupby("participant_id").apply(_slope)


def run_condition_balance(
    task: str = "overall",
    segment_sizes: tuple[int, ...] = (4, 2, 3, 6),
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    output_dir = get_output_dir(task, bucket="supplementary")
    results: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for n_segments in segment_sizes:
        long_df, pivot_df = _compute_condition_balance(task, n_segments)
        if long_df.empty or pivot_df.empty:
            continue
        if n_segments == 4:
            base = "stroop_condition_balance_by_segment"
        else:
            base = f"stroop_condition_balance_by_segment_{n_segments}"
        long_df.to_csv(output_dir / f"{base}.csv", index=False, encoding="utf-8-sig")
        pivot_df.to_csv(output_dir / f"{base}_pivot.csv", index=False, encoding="utf-8-sig")
        results[n_segments] = (long_df, pivot_df)
    return results


def run_interference_slope_segment_sensitivity(
    task: str = "overall",
    segment_sizes: tuple[int, ...] = (2, 3, 6),
) -> pd.DataFrame:
    output_dir = get_output_dir(task, bucket="supplementary")
    master = get_analysis_data(task)
    trials = _load_stroop_trials_raw(task)
    rows = []
    for n_segments in segment_sizes:
        slopes = _compute_interference_slope(trials, n_segments)
        if slopes.empty:
            continue
        slope_df = slopes.rename("stroop_interference_slope").reset_index()
        merged = master.merge(slope_df, on="participant_id", how="inner")
        res = run_ucla_regression(merged, "stroop_interference_slope", cov_type="nonrobust")
        if res is None:
            continue
        rows.append({"segments": n_segments, **res})

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(
            output_dir / "stroop_interference_slope_segment_sensitivity_2_3_6.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return out


def _safe_run(step: str, func, *args, **kwargs) -> pd.DataFrame:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] {step} failed: {exc}")
        return pd.DataFrame()


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


def run(
    task: str,
    verbose: bool = True,
    run_supplementary: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run descriptive statistics analysis.

    Parameters
    ----------
    verbose : bool
        Print results to console

    Returns
    -------
    dict
        Dictionary with descriptive and supplementary outputs
    """
    if verbose:
        print_section_header("DESCRIPTIVE STATISTICS ANALYSIS")

    # Load data
    if verbose:
        print("\n  Loading data...")
    df = get_analysis_data(task)
    variables = filter_vars(df, DESCRIPTIVE_VARS)
    output_dir = get_output_dir(task)

    if verbose:
        print(f"  Total participants: N = {len(df)}")
        if 'gender_male' in df.columns:
            n_male = (df['gender_male'] == 1).sum()
            n_female = (df['gender_male'] == 0).sum()
            print(f"  Male: n = {n_male}, Female: n = {n_female}")

    # Compute overall descriptive statistics
    if verbose:
        print("\n  Computing descriptive statistics...")

    desc_total = compute_descriptive_stats(df, variables, group_label="Total")

    # Compute categorical statistics (gender frequency/percentage)
    cat_stats = compute_categorical_stats(df)

    # Compute by gender
    if 'gender_male' in df.columns:
        desc_male = compute_descriptive_stats(
            df[df['gender_male'] == 1], variables, group_label="Male"
        )
        desc_female = compute_descriptive_stats(
            df[df['gender_male'] == 0], variables, group_label="Female"
        )
        desc_by_gender = pd.concat([desc_total, desc_male, desc_female], ignore_index=True)

        # Gender comparison with t-tests
        gender_comparison = compute_gender_comparison(df, variables)
    else:
        desc_by_gender = desc_total
        gender_comparison = pd.DataFrame()

    # Save results
    output_file = output_dir / "table1_descriptives.csv"
    desc_total.to_csv(output_file, index=False, encoding='utf-8-sig')

    output_file_gender = output_dir / "table1_descriptives_by_gender.csv"
    desc_by_gender.to_csv(output_file_gender, index=False, encoding='utf-8-sig')

    # Save categorical statistics
    if len(cat_stats) > 0:
        output_file_categorical = output_dir / "table1_categorical.csv"
        cat_stats.to_csv(output_file_categorical, index=False, encoding='utf-8-sig')

    if len(gender_comparison) > 0:
        output_file_comparison = output_dir / "table1_gender_comparison.csv"
        gender_comparison.to_csv(output_file_comparison, index=False, encoding='utf-8-sig')

    stroop_ttest = pd.DataFrame()
    wcst_norms = pd.DataFrame()
    if run_supplementary:
        supp_output_dir = get_output_dir(task, bucket="supplementary")
        stroop_ttest = _safe_run(
            "stroop_interference_ttest",
            _compute_stroop_interference_ttest,
            task,
        )
        if not stroop_ttest.empty:
            stroop_ttest.to_csv(
                supp_output_dir / "stroop_interference_ttest.csv",
                index=False,
                encoding="utf-8-sig",
            )

        wcst_norms = _safe_run(
            "wcst_normative_stats",
            _compute_wcst_normative_stats,
            task,
        )
        if not wcst_norms.empty:
            wcst_norms.to_csv(
                supp_output_dir / "wcst_normative_stats.csv",
                index=False,
                encoding="utf-8-sig",
            )

        _safe_run(
            "stroop_condition_balance_by_segment",
            run_condition_balance,
            task,
            (4, 2, 3, 6),
        )
        _safe_run(
            "stroop_interference_slope_segment_sensitivity_2_3_6",
            run_interference_slope_segment_sensitivity,
            task,
            (2, 3, 6),
        )

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

        print("\n  Output files:")
        print(f"    - {output_file}")
        print(f"    - {output_file_gender}")
        if len(cat_stats) > 0:
            print(f"    - {output_dir / 'table1_categorical.csv'}")
        if len(gender_comparison) > 0:
            print(f"    - {output_dir / 'table1_gender_comparison.csv'}")

    return {
        'total': desc_total,
        'categorical': cat_stats,
        'by_gender': desc_by_gender,
        'gender_comparison': gender_comparison,
        'stroop_ttest': stroop_ttest,
        'wcst_norms': wcst_norms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descriptive statistics analysis")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(VALID_TASKS),
        help="Dataset task to analyze (overall, stroop, wcst).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run(task=args.task, verbose=True)
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS COMPLETE")
    print("=" * 70)
