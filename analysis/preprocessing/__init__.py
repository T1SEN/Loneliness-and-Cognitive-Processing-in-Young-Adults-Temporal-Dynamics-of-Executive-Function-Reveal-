"""
Preprocessing Module
====================

Centralized data loading, cleaning, and feature extraction.

Usage:
    from analysis.preprocessing import (
        load_master_dataset,
        load_prp_trials,
        standardize_predictors,
        prepare_gender_variable,
    )
"""

# Constants
from analysis.preprocessing.constants import (
    RESULTS_DIR,
    ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN,
    DEFAULT_RT_MAX,
    PRP_RT_MAX,
    STROOP_RT_MAX,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
    MASTER_CACHE_PATH,
    STANDARDIZE_COLS,
    PARTICIPANT_ID_ALIASES,
)

# Core loaders (from loaders.py)
from analysis.preprocessing.loaders import (
    load_master_dataset,
    load_participants,
    load_ucla_scores,
    load_dass_scores,
    load_survey_items,
    load_wcst_summary,
    load_prp_summary,
    load_stroop_summary,
    load_exgaussian_params,
    ensure_participant_id,
    normalize_gender_value,
    normalize_gender_series,
)

# Trial-level loaders (from trial_loaders.py)
from analysis.preprocessing.trial_loaders import (
    load_prp_trials,
    load_stroop_trials,
    load_wcst_trials,
)

# Feature derivation (from features.py)
from analysis.preprocessing.features import (
    derive_all_features,
    derive_prp_features,
    derive_stroop_features,
    derive_wcst_features,
    coefficient_of_variation,
    compute_post_error_slowing,
)

# Standardization (from standardization.py)
from analysis.preprocessing.standardization import (
    safe_zscore,
    standardize_predictors,
    prepare_gender_variable,
    apply_fdr_correction,
    find_interaction_term,
    PREDICTOR_COLUMN_MAPPING,
    DEFAULT_PREDICTOR_COLUMNS,
)

__all__ = [
    # Constants
    'RESULTS_DIR',
    'ANALYSIS_OUTPUT_DIR',
    'DEFAULT_RT_MIN',
    'DEFAULT_RT_MAX',
    'PRP_RT_MAX',
    'STROOP_RT_MAX',
    'DEFAULT_SOA_SHORT',
    'DEFAULT_SOA_LONG',
    'MASTER_CACHE_PATH',
    'STANDARDIZE_COLS',
    'PARTICIPANT_ID_ALIASES',
    'PREDICTOR_COLUMN_MAPPING',
    'DEFAULT_PREDICTOR_COLUMNS',
    # Loaders
    'load_master_dataset',
    'load_participants',
    'load_ucla_scores',
    'load_dass_scores',
    'load_survey_items',
    'load_wcst_summary',
    'load_prp_summary',
    'load_stroop_summary',
    'load_exgaussian_params',
    'ensure_participant_id',
    'normalize_gender_value',
    'normalize_gender_series',
    # Trial loaders
    'load_prp_trials',
    'load_stroop_trials',
    'load_wcst_trials',
    # Features
    'derive_all_features',
    'derive_prp_features',
    'derive_stroop_features',
    'derive_wcst_features',
    'coefficient_of_variation',
    'compute_post_error_slowing',
    # Standardization
    'safe_zscore',
    'standardize_predictors',
    'prepare_gender_variable',
    'apply_fdr_correction',
    'find_interaction_term',
]
