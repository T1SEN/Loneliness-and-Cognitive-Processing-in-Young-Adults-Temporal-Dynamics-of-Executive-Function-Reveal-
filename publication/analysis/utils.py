"""
Common Utilities for Basic Analysis Scripts
============================================

Shared functions and constants for publication-ready basic analyses.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Import from publication preprocessing
from publication.preprocessing import load_master_dataset

# =============================================================================
# PATHS (constants.py에서 중앙 관리)
# =============================================================================

from publication.preprocessing.constants import (
    ANALYSIS_OUTPUT_DIR,
    BASE_DIR,
    VALID_TASKS,
    get_results_dir,
)
from publication.preprocessing.core import ensure_participant_id

# =============================================================================
# VARIABLE DEFINITIONS
# =============================================================================

# Variables for descriptive statistics (Table 1)
DESCRIPTIVE_VARS = [
    ("age", "Age (years)"),
    ("ucla_score", "UCLA Loneliness"),
    ("dass_depression", "DASS-21 Depression"),
    ("dass_anxiety", "DASS-21 Anxiety"),
    ("dass_stress", "DASS-21 Stress"),
    ("stroop_interference", "Stroop Interference RT"),
    ("stroop_interference_slope", "Stroop Interference RT Slope"),
    ("wcst_perseverative_error_rate", "WCST Perseverative Error Rate (%)"),
    ("wcst_exploration_rt", "WCST Exploration RT (ms)"),
    ("wcst_confirmation_rt", "WCST Confirmation RT (ms)"),
    ("wcst_exploitation_rt", "WCST Exploitation RT (ms)"),
]

# Variables for correlation matrix
CORRELATION_VARS = [
    ("ucla_score", "UCLA"),
    ("dass_depression", "DASS-Dep"),
    ("dass_anxiety", "DASS-Anx"),
    ("dass_stress", "DASS-Str"),
    ("stroop_interference", "Stroop Interference RT"),
    ("wcst_perseverative_error_rate", "WCST PE Rate"),
    ("stroop_interference_slope", "Stroop Interference RT Slope"),
    ("wcst_confirmation_rt", "WCST Confirmation RT"),
    ("wcst_exploitation_rt", "WCST Exploitation RT"),
    ("wcst_confirmation_minus_exploitation_rt", "WCST Confirm - Exploit RT"),
]

# Primary outcomes for manuscript analyses
PRIMARY_OUTCOMES = [
    ("stroop_interference", "Stroop Interference RT"),
    ("wcst_perseverative_error_rate", "WCST Perseverative Error Rate (%)"),
    ("stroop_interference_slope", "Stroop Interference RT Slope"),
    ("wcst_confirmation_rt", "WCST Confirmation RT (ms)"),
    ("wcst_exploitation_rt", "WCST Exploitation RT (ms)"),
    ("wcst_confirmation_minus_exploitation_rt", "WCST Confirmation - Exploitation RT (ms)"),
]

OUTCOMES_BY_TASK = {
    "overall": PRIMARY_OUTCOMES,
    "stroop": [
        ("stroop_interference", "Stroop Interference RT"),
        ("stroop_interference_slope", "Stroop Interference RT Slope"),
    ],
    "wcst": [
        ("wcst_perseverative_error_rate", "WCST Perseverative Error Rate (%)"),
        ("wcst_confirmation_rt", "WCST Confirmation RT (ms)"),
        ("wcst_exploitation_rt", "WCST Exploitation RT (ms)"),
        ("wcst_confirmation_minus_exploitation_rt", "WCST Confirmation - Exploitation RT (ms)"),
    ],
}

# Standardized predictor columns (already computed in master dataset)
STANDARDIZED_PREDICTORS = [
    'z_ucla_score',
    'z_dass_depression',
    'z_dass_anxiety',
    'z_dass_stress',
    'z_age',
]

# =============================================================================
# DATA LOADING
# =============================================================================

def _load_qc_ids(task: str) -> set[str]:
    ids_path = get_results_dir(task) / "filtered_participant_ids.csv"
    if not ids_path.exists():
        return set()
    ids_df = pd.read_csv(ids_path, encoding="utf-8-sig")
    if ids_df.empty:
        return set()
    ids_df = ensure_participant_id(ids_df)
    if "participant_id" not in ids_df.columns:
        return set()
    return set(ids_df["participant_id"].dropna().astype(str))


def _apply_qc_filter(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if "participant_id" not in df.columns:
        return df
    qc_ids = _load_qc_ids(task)
    if not qc_ids:
        return df
    before = len(df)
    filtered = df[df["participant_id"].isin(qc_ids)].copy()
    after = len(filtered)
    if before != after:
        print(f"  QC filter ({task}): {before} -> {after} rows")
    return filtered


def get_analysis_data(task: str, apply_qc: bool = True) -> pd.DataFrame:
    """
    Load master dataset with core outcomes pre-computed.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {sorted(VALID_TASKS)}")
    df = load_master_dataset(task=task)
    if apply_qc:
        df = _apply_qc_filter(df, task)
    return df


def get_primary_outcomes(task: str) -> list[tuple[str, str]]:
    """Return task-specific primary outcome list."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {sorted(VALID_TASKS)}")
    return OUTCOMES_BY_TASK[task]


def filter_vars(
    df: pd.DataFrame,
    var_list: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Filter variable list to only include columns present in the dataframe."""
    return [(col, label) for col, label in var_list if col in df.columns]


def get_output_dir(task: str) -> Path:
    """Return task-specific output directory for basic analysis."""
    base_dir = ANALYSIS_OUTPUT_DIR / "analysis"
    output_dir = base_dir / task
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_figures_dir() -> Path:
    """Return the publication figures directory."""
    figures_dir = BASE_DIR / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for hierarchical regression.

    Ensures all required standardized predictors exist and
    drops rows with missing values in key variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for regression
    """
    required_cols = STANDARDIZED_PREDICTORS + ['gender_male']

    # Check for missing columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing predictors
    return df.dropna(subset=required_cols)


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for publication."""
    if pd.isna(p):
        return "NA"
    if p < threshold:
        return f"< {threshold}"
    return f"{p:.3f}"


def format_coefficient(value: float, decimals: int = 3) -> str:
    """Format coefficient for publication."""
    if pd.isna(value):
        return "NA"
    return f"{value:.{decimals}f}"


def print_section_header(title: str, width: int = 70) -> None:
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def run_ucla_regression(
    df: pd.DataFrame,
    outcome: str,
    cov_type: str | None = "OLS",
    min_n: int = 30,
) -> dict | None:
    required = [
        outcome,
        "z_ucla_score",
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
    ]
    cols = [c for c in required if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) < min_n:
        return None

    formula = (
        f"{outcome} ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    )
    cov_label = "OLS"
    cov_clean = "" if cov_type is None else str(cov_type).strip().lower()
    try:
        if cov_clean in {"", "ols", "nonrobust", "none"}:
            model = smf.ols(formula, data=sub).fit()
            cov_label = "OLS"
        else:
            model = smf.ols(formula, data=sub).fit(cov_type=cov_type)
            cov_label = str(cov_type)
    except Exception:
        return None

    return {
        "outcome_column": outcome,
        "n": int(len(sub)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "cov_type": cov_label,
    }
