"""
Publication Preprocessing Module
================================

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

from .constants import (
    ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN,
    DEFAULT_RT_MAX,
    PRP_RT_MAX,
    STROOP_RT_MAX,
    STROOP_RT_MIN,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
    STANDARDIZE_COLS,
    PARTICIPANT_ID_ALIASES,
    RAW_DIR,
    DATA_DIR,
    VALID_TASKS,
    get_results_dir,
)
from .core import (
    ensure_participant_id,
    normalize_gender_value,
    normalize_gender_series,
    coefficient_of_variation,
    compute_post_error_slowing,
)
from .surveys import (
    SurveyQCCriteria,
    get_survey_valid_participants,
    load_participants,
    load_ucla_scores,
    load_dass_scores,
    load_survey_items,
)
from .prp import (
    load_prp_trials,
    load_prp_summary,
    PRPQCCriteria,
    get_prp_valid_participants,
    derive_prp_features,
    build_prp_dataset,
    get_prp_complete_participants,
)
from .stroop import (
    load_stroop_trials,
    load_stroop_summary,
    StroopQCCriteria,
    compute_stroop_qc_stats,
    get_stroop_valid_participants,
    derive_stroop_features,
    build_stroop_dataset,
    get_stroop_complete_participants,
)
from .wcst import (
    load_wcst_trials,
    load_wcst_summary,
    WCSTQCCriteria,
    get_wcst_valid_participants,
    derive_wcst_features,
    build_wcst_dataset,
    get_wcst_complete_participants,
)
from .datasets import (
    load_master_dataset,
    build_all_datasets,
    get_dataset_info,
    print_dataset_summary,
)

__all__ = [
    "ANALYSIS_OUTPUT_DIR",
    "DEFAULT_RT_MIN",
    "DEFAULT_RT_MAX",
    "PRP_RT_MAX",
    "STROOP_RT_MAX",
    "STROOP_RT_MIN",
    "DEFAULT_SOA_SHORT",
    "DEFAULT_SOA_LONG",
    "STANDARDIZE_COLS",
    "PARTICIPANT_ID_ALIASES",
    "RAW_DIR",
    "DATA_DIR",
    "VALID_TASKS",
    "get_results_dir",
    "ensure_participant_id",
    "normalize_gender_value",
    "normalize_gender_series",
    "coefficient_of_variation",
    "compute_post_error_slowing",
    "SurveyQCCriteria",
    "get_survey_valid_participants",
    "load_participants",
    "load_ucla_scores",
    "load_dass_scores",
    "load_survey_items",
    "load_prp_trials",
    "load_prp_summary",
    "PRPQCCriteria",
    "get_prp_valid_participants",
    "derive_prp_features",
    "build_prp_dataset",
    "get_prp_complete_participants",
    "load_stroop_trials",
    "load_stroop_summary",
    "StroopQCCriteria",
    "compute_stroop_qc_stats",
    "get_stroop_valid_participants",
    "derive_stroop_features",
    "build_stroop_dataset",
    "get_stroop_complete_participants",
    "load_wcst_trials",
    "load_wcst_summary",
    "WCSTQCCriteria",
    "get_wcst_valid_participants",
    "derive_wcst_features",
    "build_wcst_dataset",
    "get_wcst_complete_participants",
    "load_master_dataset",
    "build_all_datasets",
    "get_dataset_info",
    "print_dataset_summary",
]
