"""
Quantile Regression Analysis
============================

Runs quantile regression for all Tier-1 outcomes using standardized predictors.

Usage:
    python -m publication.analysis.quantile_regression --task stroop
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from publication.analysis.utils import (
    get_analysis_data,
    filter_vars,
    get_output_dir,
    get_tier1_outcomes,
    print_section_header,
)
from publication.preprocessing.constants import VALID_TASKS


DEFAULT_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
PREDICTOR_COLS = [
    "z_age",
    "gender_male",
    "z_dass_depression",
    "z_dass_anxiety",
    "z_dass_stress",
    "z_ucla_score",
]


def _parse_quantiles(values: List[float] | None) -> List[float]:
    if values is None:
        return DEFAULT_QUANTILES
    quantiles = [float(v) for v in values]
    for q in quantiles:
        if q <= 0 or q >= 1:
            raise ValueError(f"Quantiles must be in (0, 1). Got: {q}")
    return quantiles


def run(
    task: str,
    quantiles: List[float] | None = None,
    min_n: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print_section_header("QUANTILE REGRESSION ANALYSIS")

    df = get_analysis_data(task)
    outcomes = filter_vars(df, get_tier1_outcomes(task))
    output_dir = get_output_dir(task)
    quantiles = _parse_quantiles(quantiles)

    if verbose:
        print(f"\n  Total participants: N = {len(df)}")
        print(f"  Quantiles: {', '.join(f'{q:.2f}' for q in quantiles)}")

    results = []

    for outcome_col, outcome_label in outcomes:
        if verbose:
            print(f"\n  Analyzing: {outcome_label}")

        data = df[PREDICTOR_COLS + [outcome_col]].dropna()
        if len(data) < min_n:
            if verbose:
                print(f"  [SKIP] {outcome_label}: N={len(data)} < {min_n}")
            continue

        X = data[PREDICTOR_COLS].copy()
        X["ucla_x_gender"] = X["z_ucla_score"] * X["gender_male"]
        y = data[outcome_col].values
        feature_names = X.columns.tolist()

        for q in quantiles:
            model = QuantileRegressor(quantile=q, solver="highs", alpha=0)
            try:
                model.fit(X, y)
            except Exception as exc:
                if verbose:
                    print(f"  [ERROR] {outcome_label} q={q:.2f}: {exc}")
                continue

            coef_dict = dict(zip(feature_names, model.coef_))
            results.append({
                "outcome": outcome_label,
                "outcome_column": outcome_col,
                "n": len(data),
                "quantile": q,
                "intercept": float(model.intercept_),
                **{k: float(v) for k, v in coef_dict.items()},
            })

    results_df = pd.DataFrame(results)
    output_path = output_dir / "quantile_regression_results.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"\n  Output: {output_path}")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantile regression analysis")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(VALID_TASKS),
        help="Dataset task to analyze (overall, stroop, prp, wcst).",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=None,
        help="Quantiles to estimate (e.g., 0.1 0.5 0.9).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="Minimum sample size required for each outcome.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        task=args.task,
        quantiles=args.quantiles,
        min_n=args.min_n,
        verbose=not args.quiet,
    )
