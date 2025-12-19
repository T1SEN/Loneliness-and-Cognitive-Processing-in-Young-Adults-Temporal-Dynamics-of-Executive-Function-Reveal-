"""
Publication Preprocessing Module
================================

Independent data loading, cleaning, and feature extraction for publication.

Task-specific datasets:
    from publication.preprocessing import load_master_dataset
    df_stroop = load_master_dataset(task='stroop')
    df_prp = load_master_dataset(task='prp')
    df_wcst = load_master_dataset(task='wcst')

CLI:
    python -m publication.preprocessing --build stroop
    python -m publication.preprocessing --build all
    python -m publication.preprocessing --list
"""

# Constants
from .constants import (
    ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN,
    DEFAULT_RT_MAX,
    PRP_RT_MAX,
    STROOP_RT_MAX,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
    STANDARDIZE_COLS,
    PARTICIPANT_ID_ALIASES,
    # New exports
    RAW_DIR,
    DATA_DIR,
    VALID_TASKS,
    get_results_dir,
    get_cache_path,
)

# Core loaders (from loaders.py)
from .loaders import (
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
from .trial_loaders import (
    load_prp_trials,
    load_stroop_trials,
    load_wcst_trials,
)

# Feature derivation (from features.py)
from .features import (
    derive_all_features,
    derive_prp_features,
    derive_stroop_features,
    derive_wcst_features,
    coefficient_of_variation,
    compute_post_error_slowing,
)

# Standardization (from standardization.py)
from .standardization import (
    safe_zscore,
    standardize_predictors,
    prepare_gender_variable,
    apply_fdr_correction,
    find_interaction_term,
    PREDICTOR_COLUMN_MAPPING,
    DEFAULT_PREDICTOR_COLUMNS,
)

# Filters (from filters.py)
from .filters import (
    SurveyQCCriteria,
    StroopQCCriteria,
    PRPQCCriteria,
    WCSTQCCriteria,
    get_survey_valid_participants,
    get_stroop_valid_participants,
    get_prp_valid_participants,
    get_wcst_valid_participants,
    get_task_valid_participants,
)

# Dataset builder (from dataset_builder.py)
from .dataset_builder import (
    build_task_dataset,
    build_all_datasets,
    get_dataset_info,
    print_dataset_summary,
)

__all__ = [
    # Constants
    'ANALYSIS_OUTPUT_DIR',
    'DEFAULT_RT_MIN',
    'DEFAULT_RT_MAX',
    'PRP_RT_MAX',
    'STROOP_RT_MAX',
    'DEFAULT_SOA_SHORT',
    'DEFAULT_SOA_LONG',
    'STANDARDIZE_COLS',
    'PARTICIPANT_ID_ALIASES',
    'PREDICTOR_COLUMN_MAPPING',
    'DEFAULT_PREDICTOR_COLUMNS',
    # New constants
    'RAW_DIR',
    'DATA_DIR',
    'VALID_TASKS',
    'get_results_dir',
    'get_cache_path',
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
    # Filters
    'SurveyQCCriteria',
    'StroopQCCriteria',
    'PRPQCCriteria',
    'WCSTQCCriteria',
    'get_survey_valid_participants',
    'get_stroop_valid_participants',
    'get_prp_valid_participants',
    'get_wcst_valid_participants',
    'get_task_valid_participants',
    # Dataset builder
    'build_task_dataset',
    'build_all_datasets',
    'get_dataset_info',
    'print_dataset_summary',
]
