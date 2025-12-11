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
import pandas as pd

# Import from publication preprocessing
from publication.preprocessing import load_master_dataset

# =============================================================================
# PATHS (constants.py에서 중앙 관리)
# =============================================================================

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "basic_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VARIABLE DEFINITIONS
# =============================================================================

# Variables for descriptive statistics (Table 1)
DESCRIPTIVE_VARS = [
    ('age', 'Age (years)'),
    ('ucla_score', 'UCLA Loneliness'),
    ('dass_depression', 'DASS-21 Depression'),
    ('dass_anxiety', 'DASS-21 Anxiety'),
    ('dass_stress', 'DASS-21 Stress'),
    ('pe_rate', 'WCST Perseverative Error Rate'),
    ('stroop_interference', 'Stroop Interference Effect'),
    ('prp_bottleneck', 'PRP Delay Effect'),  # RT2(short SOA) - RT2(long SOA)
]

# Variables for correlation matrix
CORRELATION_VARS = [
    ('ucla_score', 'UCLA'),
    ('dass_depression', 'DASS-Dep'),
    ('dass_anxiety', 'DASS-Anx'),
    ('dass_stress', 'DASS-Str'),
    ('pe_rate', 'WCST PE'),
    ('stroop_interference', 'Stroop Int'),
    ('prp_bottleneck', 'PRP Delay'),
]

# Tier 1 outcomes for hierarchical regression
TIER1_OUTCOMES = [
    # Core EF outcomes
    ('pe_rate', 'WCST Perseverative Error Rate'),
    ('stroop_interference', 'Stroop Interference Effect'),
    ('prp_bottleneck', 'PRP Delay Effect'),

    # WCST summary metrics
    ('wcst_accuracy', 'WCST Accuracy (%)'),
    ('wcst_mean_rt', 'WCST Mean Reaction Time (ms)'),
    ('wcst_sd_rt', 'WCST Reaction Time SD'),
    ('pe_count', 'WCST Perseverative Error Count'),
    ('perseverativeResponses', 'WCST Perseverative Responses (count)'),
    ('perseverativeErrorCount', 'WCST Perseverative Errors (count)'),
    ('perseverativeResponsesPercent', 'WCST Perseverative Responses (%)'),
    # WCST trial-derived features
    ('wcst_pes', 'WCST Post-Error Slowing (ms)'),
    ('wcst_post_switch_error_rate', 'WCST Post-Switch Error Rate'),
    ('wcst_cv_rt', 'WCST Reaction Time Coefficient of Variation'),
    ('wcst_trials', 'WCST Valid Trial Count'),

    # PRP summary metrics (SOA-specific RT/variability)
    ('t2_rt_mean_short', 'PRP T2 Mean RT (Short SOA)'),
    ('t2_rt_mean_long', 'PRP T2 Mean RT (Long SOA)'),
    ('t2_rt_sd_short', 'PRP T2 RT SD (Short SOA)'),
    ('t2_rt_sd_long', 'PRP T2 RT SD (Long SOA)'),

    # PRP trial-derived features
    ('prp_t2_cv_all', 'PRP T2 Coefficient of Variation (All)'),
    ('prp_t2_cv_short', 'PRP T2 CV (Short SOA)'),
    ('prp_t2_cv_long', 'PRP T2 CV (Long SOA)'),
    ('prp_cascade_rate', 'PRP Error Cascade Rate'),
    ('prp_cascade_inflation', 'PRP Cascade Inflation'),
    ('prp_pes', 'PRP Post-Error Slowing (ms)'),
    ('prp_t2_trials', 'PRP Valid T2 Trial Count'),

    # Stroop summary metrics
    ('rt_mean_incongruent', 'Stroop Mean RT (Incongruent)'),
    ('rt_mean_congruent', 'Stroop Mean RT (Congruent)'),
    ('rt_mean_neutral', 'Stroop Mean RT (Neutral)'),
    ('accuracy_incongruent', 'Stroop Accuracy (Incongruent)'),
    ('accuracy_congruent', 'Stroop Accuracy (Congruent)'),
    ('accuracy_neutral', 'Stroop Accuracy (Neutral)'),
    ('stroop_effect', 'Stroop Effect (RT Difference)'),

    # Stroop trial-derived features
    ('stroop_post_error_slowing', 'Stroop Post-Error Slowing (ms)'),
    ('stroop_post_error_rt', 'Stroop Post-Error RT (ms)'),
    ('stroop_post_correct_rt', 'Stroop Post-Correct RT (ms)'),
    ('stroop_incong_slope', 'Stroop Incongruent RT Slope'),
    ('stroop_cv_all', 'Stroop RT Coefficient of Variation (All)'),
    ('stroop_cv_incong', 'Stroop RT CV (Incongruent)'),
    ('stroop_cv_cong', 'Stroop RT CV (Congruent)'),
    ('stroop_trials', 'Stroop Valid Trial Count'),
]

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

def get_analysis_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load master dataset with Tier-1 metrics pre-computed.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached parquet file (default: True)

    Returns
    -------
    pd.DataFrame
        Master dataset with all required variables
    """
    return load_master_dataset(use_cache=use_cache)


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
