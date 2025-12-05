# analysis.utils package
"""
Shared Utility Modules
======================

Data Loading:
- data_loader_utils: Master dataset loading and caching
- trial_data_loader: Trial-level data loaders (PRP, Stroop, WCST)

Modeling:
- modeling: Regression utilities, standardization, DASS control verification

Analysis Utilities:
- trial_features: Derive participant-level features from trial data
- exgaussian: Ex-Gaussian RT distribution fitting
- post_error: Post-error slowing computation

Visualization:
- plotting: Matplotlib helpers
- publication_helpers: APA formatting, effect sizes, bootstrap CI
"""

from analysis.utils.data_loader_utils import (
    load_master_dataset,
    load_participants,
    load_ucla_scores,
    load_dass_scores,
    ensure_participant_id,
    normalize_gender_series,
    RESULTS_DIR,
    ANALYSIS_OUTPUT_DIR,
)

from analysis.utils.trial_data_loader import (
    load_prp_trials,
    load_stroop_trials,
    load_wcst_trials,
)

from analysis.utils.modeling import (
    standardize_predictors,
    DASS_CONTROL_FORMULA,
    fit_dass_controlled_model,
    verify_dass_control,
)

from analysis.utils.trial_features import (
    derive_all_features,
    derive_prp_features,
    derive_stroop_features,
    derive_wcst_features,
)

from analysis.utils.exgaussian import (
    fit_exgaussian,
    fit_exgaussian_by_participant,
    fit_exgaussian_by_condition,
    exgaussian_pdf,
)

from analysis.utils.post_error import (
    compute_pes,
    compute_post_error_accuracy,
    compute_all_task_pes,
)

__all__ = [
    # Data loading
    "load_master_dataset",
    "load_participants",
    "load_ucla_scores",
    "load_dass_scores",
    "load_prp_trials",
    "load_stroop_trials",
    "load_wcst_trials",
    "ensure_participant_id",
    "normalize_gender_series",
    # Modeling
    "standardize_predictors",
    "DASS_CONTROL_FORMULA",
    "fit_dass_controlled_model",
    "verify_dass_control",
    # Trial features
    "derive_all_features",
    "derive_prp_features",
    "derive_stroop_features",
    "derive_wcst_features",
    # Ex-Gaussian
    "fit_exgaussian",
    "fit_exgaussian_by_participant",
    "fit_exgaussian_by_condition",
    "exgaussian_pdf",
    # Post-error
    "compute_pes",
    "compute_post_error_accuracy",
    "compute_all_task_pes",
    # Paths
    "RESULTS_DIR",
    "ANALYSIS_OUTPUT_DIR",
]
