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
    ('age', 'Age (years)'),
    ('ucla_score', 'UCLA Loneliness'),
    ('dass_depression', 'DASS-21 Depression'),
    ('dass_anxiety', 'DASS-21 Anxiety'),
    ('dass_stress', 'DASS-21 Stress'),
    ('pe_rate', 'WCST Perseverative Error Rate'),
    ('stroop_interference', 'Stroop Interference Effect'),
]

# Variables for correlation matrix
CORRELATION_VARS = [
    ('ucla_score', 'UCLA'),
    ('dass_depression', 'DASS-Dep'),
    ('dass_anxiety', 'DASS-Anx'),
    ('dass_stress', 'DASS-Str'),
    ('pe_rate', 'WCST PE'),
    ('stroop_interference', 'Stroop Int'),
]

# Shared constants for derived outcome lists
STROOP_CONDITIONS = ["congruent", "incongruent", "neutral"]
STROOP_VINCENTILES = [10, 30, 50, 70, 90]
WCST_SWITCH_K = [1, 2, 3, 4, 5]
WCST_CATEGORY_MAX = 6

# Tier 1 outcomes for hierarchical regression (overall / full list)
TIER1_OUTCOMES = [
    # Core EF outcomes
    ('pe_rate', 'WCST Perseverative Error Rate'),
    ('stroop_interference', 'Stroop Interference Effect'),

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
    ('wcst_mad_rt', 'WCST RT MAD'),
    ('wcst_iqr_rt', 'WCST RT IQR'),
    ('wcst_mad_rt_correct', 'WCST RT MAD (correct)'),
    ('wcst_iqr_rt_correct', 'WCST RT IQR (correct)'),
    ('wcst_trials', 'WCST Valid Trial Count'),

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
    ('stroop_mad_all', 'Stroop RT MAD (All)'),
    ('stroop_mad_incong', 'Stroop RT MAD (Incongruent)'),
    ('stroop_mad_cong', 'Stroop RT MAD (Congruent)'),
    ('stroop_iqr_all', 'Stroop RT IQR (All)'),
    ('stroop_iqr_incong', 'Stroop RT IQR (Incongruent)'),
    ('stroop_iqr_cong', 'Stroop RT IQR (Congruent)'),
    ('stroop_mad_all_correct', 'Stroop RT MAD (All, correct)'),
    ('stroop_mad_incong_correct', 'Stroop RT MAD (Incongruent, correct)'),
    ('stroop_mad_cong_correct', 'Stroop RT MAD (Congruent, correct)'),
    ('stroop_iqr_all_correct', 'Stroop RT IQR (All, correct)'),
    ('stroop_iqr_incong_correct', 'Stroop RT IQR (Incongruent, correct)'),
    ('stroop_iqr_cong_correct', 'Stroop RT IQR (Congruent, correct)'),
    ('stroop_trials', 'Stroop Valid Trial Count'),

    # Stroop Ex-Gaussian parameters
    ('stroop_exg_congruent_mu', 'Stroop Ex-Gaussian mu (Congruent)'),
    ('stroop_exg_congruent_sigma', 'Stroop Ex-Gaussian sigma (Congruent)'),
    ('stroop_exg_congruent_tau', 'Stroop Ex-Gaussian tau (Congruent)'),
    ('stroop_exg_incongruent_mu', 'Stroop Ex-Gaussian mu (Incongruent)'),
    ('stroop_exg_incongruent_sigma', 'Stroop Ex-Gaussian sigma (Incongruent)'),
    ('stroop_exg_incongruent_tau', 'Stroop Ex-Gaussian tau (Incongruent)'),
    ('stroop_exg_neutral_mu', 'Stroop Ex-Gaussian mu (Neutral)'),
    ('stroop_exg_neutral_sigma', 'Stroop Ex-Gaussian sigma (Neutral)'),
    ('stroop_exg_neutral_tau', 'Stroop Ex-Gaussian tau (Neutral)'),
    ('stroop_exg_mu_interference', 'Stroop Ex-Gaussian mu (Interference)'),
    ('stroop_exg_sigma_interference', 'Stroop Ex-Gaussian sigma (Interference)'),
    ('stroop_exg_tau_interference', 'Stroop Ex-Gaussian tau (Interference)'),

    # WCST HMM parameters
    ('wcst_hmm_lapse_occupancy', 'WCST HMM Lapse Occupancy (%)'),
    ('wcst_hmm_trans_to_lapse', 'WCST HMM P(Focus->Lapse)'),
    ('wcst_hmm_trans_to_focus', 'WCST HMM P(Lapse->Focus)'),
    ('wcst_hmm_stay_lapse', 'WCST HMM P(Lapse->Lapse)'),
    ('wcst_hmm_stay_focus', 'WCST HMM P(Focus->Focus)'),
    ('wcst_hmm_lapse_rt_mean', 'WCST HMM Lapse RT Mean'),
    ('wcst_hmm_focus_rt_mean', 'WCST HMM Focus RT Mean'),
    ('wcst_hmm_rt_diff', 'WCST HMM RT Difference'),
    ('wcst_hmm_state_changes', 'WCST HMM State Changes'),

    # WCST RL parameters
    ('wcst_rl_alpha', 'WCST RL alpha'),
    ('wcst_rl_beta', 'WCST RL beta'),
    ('wcst_rl_alpha_pos', 'WCST RL alpha (pos)'),
    ('wcst_rl_alpha_neg', 'WCST RL alpha (neg)'),
    ('wcst_rl_alpha_asymmetry', 'WCST RL alpha asymmetry'),
    ('wcst_rl_beta_asym', 'WCST RL beta (asym)'),

    # WCST WSLS parameters
    ('wcst_wsls_p_stay_win', 'WCST WSLS P(stay|win)'),
    ('wcst_wsls_p_shift_lose', 'WCST WSLS P(shift|lose)'),

    # WCST Bayesian rule learner parameters
    ('wcst_brl_hazard', 'WCST Bayesian RL hazard'),
    ('wcst_brl_noise', 'WCST Bayesian RL noise'),
    ('wcst_brl_beta', 'WCST Bayesian RL beta'),

]

# -----------------------------------------------------------------------------
# Additional outcomes for expanded metrics
# -----------------------------------------------------------------------------
STROOP_EXTRA_OUTCOMES = [
    ('stroop_slow_prob_baseline', 'Stroop Slow-State Probability (Baseline)'),
    ('stroop_slow_prob_post_error', 'Stroop Slow-State Prob (Post-Error)'),
    ('stroop_slow_prob_post_error_delta', 'Stroop Slow-State Prob Delta (Post-Error)'),
    ('stroop_slow_prob_stable', 'Stroop Slow-State Prob (Stable)'),
    ('stroop_slow_prob_stable_delta', 'Stroop Slow-State Prob Delta (Stable)'),
    ('stroop_rt_interference', 'Stroop RT Interference'),
    ('stroop_acc_interference', 'Stroop Accuracy Interference'),
    ('stroop_interference_slope', 'Stroop Interference Slope'),
    ('stroop_rt_sd_incong', 'Stroop RT SD (Incongruent)'),
    ('stroop_rt_facilitation', 'Stroop RT Facilitation'),
    ('stroop_acc_facilitation', 'Stroop Accuracy Facilitation'),
    ('stroop_rt_incong_minus_neutral', 'Stroop RT Incongruent minus Neutral'),
    ('stroop_rt_neutral_minus_cong', 'Stroop RT Neutral minus Congruent'),
    ('stroop_rt_interference_correct', 'Stroop RT Interference (Correct-only)'),
    ('stroop_rt_facilitation_correct', 'Stroop RT Facilitation (Correct-only)'),
    ('stroop_rt_incong_minus_neutral_correct', 'Stroop RT Incongruent minus Neutral (Correct-only)'),
    ('stroop_rt_neutral_minus_cong_correct', 'Stroop RT Neutral minus Congruent (Correct-only)'),
    ('stroop_cse_rt', 'Stroop CSE RT (Gratton)'),
    ('stroop_cse_acc', 'Stroop CSE Accuracy (Gratton)'),
    ('stroop_post_conflict_slowing', 'Stroop Post-Conflict Slowing'),
    ('stroop_post_conflict_accuracy', 'Stroop Post-Conflict Accuracy'),
    ('stroop_post_error_accuracy', 'Stroop Post-Error Accuracy'),
    ('stroop_post_error_interference_rt', 'Stroop Post-Error Interference (RT)'),
    ('stroop_post_error_interference_acc', 'Stroop Post-Error Interference (Accuracy)'),
    ('stroop_error_run_length_mean', 'Stroop Error Run Length Mean'),
    ('stroop_error_run_length_max', 'Stroop Error Run Length Max'),
    ('stroop_error_recovery_rt', 'Stroop Error Recovery RT Slope'),
    ('stroop_dfa_alpha_correct', 'Stroop DFA Alpha (correct)'),
    ('stroop_lag1_correct', 'Stroop Lag-1 Autocorr (correct)'),
    ('stroop_slow_run_mean_correct', 'Stroop Slow Run Mean (correct)'),
    ('stroop_slow_run_max_correct', 'Stroop Slow Run Max (correct)'),
    ('stroop_fast_run_mean_correct', 'Stroop Fast Run Mean (correct)'),
    ('stroop_fast_run_max_correct', 'Stroop Fast Run Max (correct)'),
    ('stroop_dfa_alpha_incong_correct', 'Stroop DFA Alpha (incongruent, correct)'),
    ('stroop_lag1_incong_correct', 'Stroop Lag-1 Autocorr (incongruent, correct)'),
    ('stroop_slow_run_mean_incong_correct', 'Stroop Slow Run Mean (incongruent, correct)'),
    ('stroop_slow_run_max_incong_correct', 'Stroop Slow Run Max (incongruent, correct)'),
    ('stroop_fast_run_mean_incong_correct', 'Stroop Fast Run Mean (incongruent, correct)'),
    ('stroop_fast_run_max_incong_correct', 'Stroop Fast Run Max (incongruent, correct)'),
    ('stroop_exg_correct_mu_interference', 'Stroop Ex-Gaussian mu (Interference, correct)'),
    ('stroop_exg_correct_sigma_interference', 'Stroop Ex-Gaussian sigma (Interference, correct)'),
    ('stroop_exg_correct_tau_interference', 'Stroop Ex-Gaussian tau (Interference, correct)'),
    ('stroop_delta_plot_slope_correct', 'Stroop Delta Plot Slope (correct)'),
]

STROOP_EZ_PARAMS = ["v", "a", "t0", "sv", "sz", "st0"]
STROOP_EXTRA_OUTCOMES += [
    (f"stroop_ez_{cond}_{param}", f"Stroop EZ-DDM {param} ({cond})")
    for cond in STROOP_CONDITIONS
    for param in STROOP_EZ_PARAMS
]
STROOP_EXTRA_OUTCOMES += [
    ('stroop_ez_v_interference', 'Stroop EZ-DDM v Interference'),
    ('stroop_ez_a_interference', 'Stroop EZ-DDM a Interference'),
    ('stroop_ez_t0_interference', 'Stroop EZ-DDM t0 Interference'),
]

STROOP_EXTRA_OUTCOMES += [
    (f"stroop_exg_correct_{cond}_{param}", f"Stroop Ex-Gaussian {param} ({cond}, correct)")
    for cond in STROOP_CONDITIONS
    for param in ("mu", "sigma", "tau")
]
STROOP_EXTRA_OUTCOMES += [
    (f"stroop_lognorm_correct_{cond}_{param}", f"Stroop Lognormal {param} ({cond}, correct)")
    for cond in STROOP_CONDITIONS
    for param in ("mu", "sigma", "shift")
]
STROOP_EXTRA_OUTCOMES += [
    (f"stroop_mix_slow_prop_correct_{cond}", f"Stroop Mixture Slow Prop ({cond}, correct)")
    for cond in STROOP_CONDITIONS
]
STROOP_EXTRA_OUTCOMES += [
    (f"stroop_vincentile_interference_p{p}_correct", f"Stroop Vincentile Interference p{p} (correct)")
    for p in STROOP_VINCENTILES
]

STROOP_EXTRA_OUTCOMES += [
    ('stroop_rt_fatigue_slope', 'Stroop RT Fatigue Slope (Q4-Q1)'),
    ('stroop_cv_fatigue_slope', 'Stroop CV Fatigue Slope (Q4-Q1)'),
    ('stroop_cv_fatigue_slope_rolling', 'Stroop CV Fatigue Slope (Rolling)'),
    ('stroop_acc_fatigue_slope', 'Stroop Accuracy Fatigue Slope (Q4-Q1)'),
    ('stroop_rt_sd_block_slope', 'Stroop RT SD Block Slope'),
    ('stroop_rt_p90_block_slope', 'Stroop RT p90 Block Slope'),
    ('stroop_residual_sd_block_slope', 'Stroop Residual SD Block Slope'),
    ('stroop_tau_q1', 'Stroop Tau Q1'),
    ('stroop_tau_q2', 'Stroop Tau Q2'),
    ('stroop_tau_q3', 'Stroop Tau Q3'),
    ('stroop_tau_q4', 'Stroop Tau Q4'),
    ('stroop_tau_slope', 'Stroop Tau Slope (Q4-Q1)'),
    ('stroop_error_cascade_count', 'Stroop Error Cascade Count'),
    ('stroop_error_cascade_rate', 'Stroop Error Cascade Rate'),
    ('stroop_error_cascade_mean_len', 'Stroop Error Cascade Mean Length'),
    ('stroop_error_cascade_max_len', 'Stroop Error Cascade Max Length'),
    ('stroop_error_cascade_trials', 'Stroop Error Cascade Trials'),
    ('stroop_error_cascade_prop', 'Stroop Error Cascade Proportion'),
    ('stroop_recovery_rt_slope', 'Stroop Recovery RT Slope'),
    ('stroop_recovery_acc_slope', 'Stroop Recovery Accuracy Slope'),
    ('stroop_momentum_slope', 'Stroop Momentum Slope'),
    ('stroop_momentum_mean_streak', 'Stroop Momentum Mean Streak'),
    ('stroop_momentum_max_streak', 'Stroop Momentum Max Streak'),
    ('stroop_volatility_rmssd', 'Stroop Volatility RMSSD'),
    ('stroop_volatility_adj', 'Stroop Volatility (Detrended SD)'),
    ('stroop_intercept', 'Stroop IIV Intercept'),
    ('stroop_slope', 'Stroop IIV Slope'),
    ('stroop_residual_sd', 'Stroop IIV Residual SD'),
    ('stroop_raw_cv', 'Stroop IIV Raw CV'),
    ('stroop_iiv_r_squared', 'Stroop IIV R-squared'),
    ('stroop_post_error_cv_reduction', 'Stroop Post-Error CV Reduction'),
    ('stroop_post_error_acc_diff', 'Stroop Post-Error Accuracy Diff'),
    ('stroop_post_error_recovery_rate', 'Stroop Post-Error Recovery Rate'),
    ('stroop_pes_adaptive', 'Stroop Adaptive PES Flag'),
    ('stroop_pes_maladaptive', 'Stroop Maladaptive PES Flag'),
    ('stroop_error_awareness_index', 'Stroop Error Awareness Index'),
]
STROOP_EXTRA_OUTCOMES += [
    (f"stroop_recovery_rt_lag{k}", f"Stroop Recovery RT Lag{k}")
    for k in range(1, 6)
]
STROOP_EXTRA_OUTCOMES += [
    (f"stroop_recovery_acc_lag{k}", f"Stroop Recovery Accuracy Lag{k}")
    for k in range(1, 6)
]
STROOP_EXTRA_OUTCOMES += [
    ('stroop_mean_rt_all', 'Stroop Mean RT (All)'),
    ('stroop_accuracy_all', 'Stroop Accuracy (All)'),
    ('stroop_error_rate_all', 'Stroop Error Rate (All)'),
    ('stroop_ies', 'Stroop Inverse Efficiency Score'),
    ('stroop_pre_error_slope_mean', 'Stroop Pre-Error RT Slope (Mean)'),
    ('stroop_pre_error_slope_std', 'Stroop Pre-Error RT Slope (SD)'),
    ('stroop_pre_error_n', 'Stroop Pre-Error Events (N)'),
]

WCST_EXTRA_OUTCOMES = [
    ('wcst_total_errors', 'WCST Total Errors'),
    ('wcst_error_rate', 'WCST Error Rate'),
    ('wcst_perseverative_errors', 'WCST Perseverative Errors (trial-derived)'),
    ('wcst_perseverative_error_rate', 'WCST Perseverative Error Rate (trial-derived)'),
    ('wcst_nonperseverative_errors', 'WCST Non-Perseverative Errors'),
    ('wcst_perseverative_responses', 'WCST Perseverative Responses (trial-derived)'),
    ('wcst_perseverative_response_percent', 'WCST Perseverative Response Percent (trial-derived)'),
    ('wcst_error_pr_ratio', 'WCST PR Ratio among Errors'),
    ('wcst_error_npe_ratio', 'WCST NPE Ratio among Errors'),
    ('wcst_post_error_rt', 'WCST Post-Error RT (ms)'),
    ('wcst_post_correct_rt', 'WCST Post-Correct RT (ms)'),
    ('wcst_post_pe_rt', 'WCST Post-PE RT (ms)'),
    ('wcst_post_npe_rt', 'WCST Post-NPE RT (ms)'),
    ('wcst_post_pe_slowing', 'WCST Post-PE Slowing (ms)'),
    ('wcst_pe_rt_mean', 'WCST PE RT Mean (ms)'),
    ('wcst_npe_rt_mean', 'WCST Non-PE RT Mean (ms)'),
    ('wcst_pe_minus_npe_rt_mean', 'WCST PE-Non-PE RT (Mean)'),
    ('wcst_pe_run_length_mean', 'WCST PE Run Length Mean'),
    ('wcst_pe_run_length_max', 'WCST PE Run Length Max'),
    ('wcst_pr_run_length_mean', 'WCST PR Run Length Mean'),
    ('wcst_pr_run_length_max', 'WCST PR Run Length Max'),
    ('wcst_pe_cluster_rate', 'WCST PE Cluster Rate'),
    ('wcst_trials_to_first_category', 'WCST Trials to First Category'),
    ('wcst_categories_completed', 'WCST Categories Completed'),
    ('wcst_trials_per_category_mean', 'WCST Trials per Category Mean'),
    ('wcst_trials_per_category_sd', 'WCST Trials per Category SD'),
    ('wcst_delta_trials_first3_last3', 'WCST Delta Trials (First3-Last3)'),
    ('wcst_learning_slope_trials', 'WCST Learning Slope (Trials)'),
      ('wcst_rt_slope_overall', 'WCST RT Slope (Overall)'),
      ('wcst_rt_slope_correct', 'WCST RT Slope (Overall, correct)'),
      ('wcst_category_total_rt_slope', 'WCST Category Total RT Slope'),
      ('wcst_rt_slope_within_category', 'WCST RT Slope within Category'),
      ('wcst_mean_rt_within_category', 'WCST Mean RT within Category'),
      ('wcst_acc_slope_within_category', 'WCST Accuracy Slope within Category'),
      ('wcst_category_accuracy_slope', 'WCST Category Accuracy Slope'),
      ('wcst_rt_jump_at_switch', 'WCST RT Jump at Switch'),
      ('wcst_switch_rt', 'WCST Switch RT (ms)'),
      ('wcst_repeat_rt', 'WCST Repeat RT (ms)'),
      ('wcst_switch_cost_rt', 'WCST Switch Cost RT (ms)'),
    ('wcst_post_switch_recovery_slope', 'WCST Post-Switch Recovery RT Slope'),
    ('wcst_dfa_alpha_correct', 'WCST DFA Alpha (correct)'),
    ('wcst_lag1_correct', 'WCST Lag-1 Autocorr (correct)'),
    ('wcst_slow_run_mean_correct', 'WCST Slow Run Mean (correct)'),
    ('wcst_slow_run_max_correct', 'WCST Slow Run Max (correct)'),
    ('wcst_fast_run_mean_correct', 'WCST Fast Run Mean (correct)'),
    ('wcst_fast_run_max_correct', 'WCST Fast Run Max (correct)'),
    ('wcst_failure_to_maintain_set', 'WCST Failure to Maintain Set'),
    ('wcst_trials_to_rule_reacquisition', 'WCST Trials to Rule Reacquisition'),
    ('wcst_clr_count', 'WCST CLR Count'),
    ('wcst_clr_percent', 'WCST CLR Percent'),
    ('wcst_learning_to_learn', 'WCST Learning-to-Learn'),
    ('wcst_learning_to_learn_heaton_clr_delta', 'WCST Learning-to-Learn Heaton CLR Delta'),
    ('wcst_learning_efficiency_delta_trials', 'WCST Learning Efficiency Delta Trials'),
    ('wcst_trials_to_first_conceptual_resp', 'WCST Trials to First Conceptual Response'),
    ('wcst_trials_to_first_conceptual_resp0', 'WCST Trials to First Conceptual Response (0)'),
    ('wcst_has_first_clr', 'WCST Has First CLR'),
    ('wcst_delta_clr_percent_first3_last3', 'WCST Delta CLR Percent (First3-Last3)'),
    ('wcst_learning_slope_clr_percent', 'WCST Learning Slope CLR Percent'),
    ('wcst_hmm_lapse_dwell_mean', 'WCST HMM Lapse Dwell Mean'),
    ('wcst_hmm_focus_dwell_mean', 'WCST HMM Focus Dwell Mean'),
    ('wcst_hmm_long_lapse_episode_rate', 'WCST HMM Long Lapse Episode Rate'),
    ('wcst_hmm_lapse_rt_sd', 'WCST HMM Lapse RT SD'),
    ('wcst_hmm_focus_rt_sd', 'WCST HMM Focus RT SD'),
    ('wcst_hmm_state_entropy', 'WCST HMM State Entropy'),
    ('wcst_slow_prob_baseline', 'WCST Slow-State Probability (Baseline)'),
    ('wcst_slow_prob_post_error', 'WCST Slow-State Prob (Post-Error)'),
    ('wcst_slow_prob_post_error_delta', 'WCST Slow-State Prob Delta (Post-Error)'),
    ('wcst_slow_prob_stable', 'WCST Slow-State Prob (Stable)'),
    ('wcst_slow_prob_stable_delta', 'WCST Slow-State Prob Delta (Stable)'),
    ('wcst_slow_prob_shift_k0', 'WCST Slow-State Prob (Shift k0)'),
    ('wcst_slow_prob_shift_k1', 'WCST Slow-State Prob (Shift k1)'),
    ('wcst_slow_prob_shift_k2', 'WCST Slow-State Prob (Shift k2)'),
    ('wcst_slow_prob_shift_k3', 'WCST Slow-State Prob (Shift k3)'),
    ('wcst_slow_prob_shift_k4', 'WCST Slow-State Prob (Shift k4)'),
    ('wcst_slow_prob_shift_k5', 'WCST Slow-State Prob (Shift k5)'),
    ('wcst_slow_prob_shift_k0_delta', 'WCST Slow-State Prob Delta (Shift k0)'),
    ('wcst_slow_prob_shift_k1_delta', 'WCST Slow-State Prob Delta (Shift k1)'),
    ('wcst_slow_prob_shift_k2_delta', 'WCST Slow-State Prob Delta (Shift k2)'),
    ('wcst_slow_prob_shift_k3_delta', 'WCST Slow-State Prob Delta (Shift k3)'),
    ('wcst_slow_prob_shift_k4_delta', 'WCST Slow-State Prob Delta (Shift k4)'),
    ('wcst_slow_prob_shift_k5_delta', 'WCST Slow-State Prob Delta (Shift k5)'),
    ('wcst_hmm2d_lapse_occupancy', 'WCST HMM2D Lapse Occupancy'),
    ('wcst_hmm2d_trans_to_lapse', 'WCST HMM2D P(Focus->Lapse)'),
    ('wcst_hmm2d_trans_to_focus', 'WCST HMM2D P(Lapse->Focus)'),
    ('wcst_hmm2d_stay_lapse', 'WCST HMM2D P(Lapse->Lapse)'),
    ('wcst_hmm2d_stay_focus', 'WCST HMM2D P(Focus->Focus)'),
    ('wcst_hmm2d_lapse_dwell_mean', 'WCST HMM2D Lapse Dwell Mean'),
    ('wcst_hmm2d_focus_dwell_mean', 'WCST HMM2D Focus Dwell Mean'),
    ('wcst_hmm2d_state_entropy', 'WCST HMM2D State Entropy'),
    ('wcst_rl_negloglik', 'WCST RL Neg LogLik'),
    ('wcst_rl_aic', 'WCST RL AIC'),
    ('wcst_rl_bic', 'WCST RL BIC'),
    ('wcst_rl_asym_negloglik', 'WCST RL Asym Neg LogLik'),
    ('wcst_rl_asym_aic', 'WCST RL Asym AIC'),
    ('wcst_rl_asym_bic', 'WCST RL Asym BIC'),
    ('wcst_rl_sticky_alpha', 'WCST RL Sticky alpha'),
    ('wcst_rl_sticky_beta', 'WCST RL Sticky beta'),
    ('wcst_rl_stickiness', 'WCST RL Stickiness'),
    ('wcst_rl_lapse', 'WCST RL Lapse'),
    ('wcst_rl_sticky_negloglik', 'WCST RL Sticky Neg LogLik'),
    ('wcst_rl_sticky_aic', 'WCST RL Sticky AIC'),
    ('wcst_rl_sticky_bic', 'WCST RL Sticky BIC'),
    ('wcst_rl_alpha0', 'WCST RL alpha0'),
    ('wcst_rl_alpha_decay', 'WCST RL alpha Decay'),
    ('wcst_rl_beta_decay', 'WCST RL beta (Decay)'),
    ('wcst_rl_decay_negloglik', 'WCST RL Decay Neg LogLik'),
    ('wcst_rl_decay_aic', 'WCST RL Decay AIC'),
    ('wcst_rl_decay_bic', 'WCST RL Decay BIC'),
    ('wcst_wsls_p_shift_win', 'WCST WSLS P(shift|win)'),
    ('wcst_wsls_p_stay_lose', 'WCST WSLS P(stay|lose)'),
    ('wcst_wsls_p_stay_win_switch', 'WCST WSLS P(stay|win, switch)'),
    ('wcst_wsls_p_shift_lose_switch', 'WCST WSLS P(shift|lose, switch)'),
    ('wcst_wsls_p_stay_win_stable', 'WCST WSLS P(stay|win, stable)'),
    ('wcst_wsls_p_shift_lose_stable', 'WCST WSLS P(shift|lose, stable)'),
    ('wcst_wsls_p_stay_win_colour', 'WCST WSLS P(stay|win, colour)'),
    ('wcst_wsls_p_shift_lose_colour', 'WCST WSLS P(shift|lose, colour)'),
    ('wcst_wsls_p_stay_win_shape', 'WCST WSLS P(stay|win, shape)'),
    ('wcst_wsls_p_shift_lose_shape', 'WCST WSLS P(shift|lose, shape)'),
    ('wcst_wsls_p_stay_win_number', 'WCST WSLS P(stay|win, number)'),
    ('wcst_wsls_p_shift_lose_number', 'WCST WSLS P(shift|lose, number)'),
    ('wcst_brl_posterior_entropy_mean', 'WCST BRL Posterior Entropy Mean'),
    ('wcst_brl_entropy_drop_mean', 'WCST BRL Entropy Drop Mean'),
    ('wcst_brl_p_rule_max_mean', 'WCST BRL P(rule max) Mean'),
    ('wcst_brl_change_point_prob_mean', 'WCST BRL Change Point Prob Mean'),
    ('wcst_brl_surprise_mean', 'WCST BRL Surprise Mean'),
]

WCST_EXTRA_OUTCOMES += [
    ('wcst_rt_fatigue_slope', 'WCST RT Fatigue Slope (Q4-Q1)'),
    ('wcst_cv_fatigue_slope', 'WCST CV Fatigue Slope (Q4-Q1)'),
    ('wcst_cv_fatigue_slope_rolling', 'WCST CV Fatigue Slope (Rolling)'),
    ('wcst_acc_fatigue_slope', 'WCST Accuracy Fatigue Slope (Q4-Q1)'),
      ('wcst_rt_sd_block_slope', 'WCST RT SD Block Slope'),
      ('wcst_rt_p90_block_slope', 'WCST RT p90 Block Slope'),
      ('wcst_residual_sd_block_slope', 'WCST Residual SD Block Slope'),
      ('wcst_residual_abs_slope_correct', 'WCST Residual Abs Slope (Correct)'),
      ('wcst_tau_q1', 'WCST Tau Q1'),
    ('wcst_tau_q2', 'WCST Tau Q2'),
    ('wcst_tau_q3', 'WCST Tau Q3'),
    ('wcst_tau_q4', 'WCST Tau Q4'),
    ('wcst_tau_slope', 'WCST Tau Slope (Q4-Q1)'),
    ('wcst_error_cascade_count', 'WCST Error Cascade Count'),
    ('wcst_error_cascade_rate', 'WCST Error Cascade Rate'),
    ('wcst_error_cascade_mean_len', 'WCST Error Cascade Mean Length'),
    ('wcst_error_cascade_max_len', 'WCST Error Cascade Max Length'),
    ('wcst_error_cascade_trials', 'WCST Error Cascade Trials'),
    ('wcst_error_cascade_prop', 'WCST Error Cascade Proportion'),
    ('wcst_recovery_rt_slope', 'WCST Recovery RT Slope'),
    ('wcst_recovery_acc_slope', 'WCST Recovery Accuracy Slope'),
    ('wcst_momentum_slope', 'WCST Momentum Slope'),
    ('wcst_momentum_mean_streak', 'WCST Momentum Mean Streak'),
    ('wcst_momentum_max_streak', 'WCST Momentum Max Streak'),
    ('wcst_volatility_rmssd', 'WCST Volatility RMSSD'),
    ('wcst_volatility_adj', 'WCST Volatility (Detrended SD)'),
    ('wcst_intercept', 'WCST IIV Intercept'),
    ('wcst_slope', 'WCST IIV Slope'),
    ('wcst_residual_sd', 'WCST IIV Residual SD'),
    ('wcst_raw_cv', 'WCST IIV Raw CV'),
    ('wcst_iiv_r_squared', 'WCST IIV R-squared'),
    ('wcst_post_error_cv_reduction', 'WCST Post-Error CV Reduction'),
    ('wcst_post_error_accuracy', 'WCST Post-Error Accuracy'),
    ('wcst_post_error_acc_diff', 'WCST Post-Error Accuracy Diff'),
    ('wcst_post_error_recovery_rate', 'WCST Post-Error Recovery Rate'),
    ('wcst_pes_adaptive', 'WCST Adaptive PES Flag'),
    ('wcst_pes_maladaptive', 'WCST Maladaptive PES Flag'),
    ('wcst_error_awareness_index', 'WCST Error Awareness Index'),
]
WCST_EXTRA_OUTCOMES += [
    (f"wcst_recovery_rt_lag{k}", f"WCST Recovery RT Lag{k}")
    for k in range(1, 6)
]
WCST_EXTRA_OUTCOMES += [
    (f"wcst_recovery_acc_lag{k}", f"WCST Recovery Accuracy Lag{k}")
    for k in range(1, 6)
]

WCST_EXTRA_OUTCOMES += [
    (f"wcst_trials_per_category_{idx}", f"WCST Trials per Category {idx}")
    for idx in range(1, WCST_CATEGORY_MAX + 1)
]
WCST_EXTRA_OUTCOMES += [
    (f"wcst_switch_cost_error_k{k}", f"WCST Switch Cost Error k{k}")
    for k in WCST_SWITCH_K
]
WCST_EXTRA_OUTCOMES += [
    (f"wcst_switch_cost_rt_k{k}", f"WCST Switch Cost RT k{k}")
    for k in WCST_SWITCH_K
]
WCST_EXTRA_OUTCOMES += [
    ('wcst_delta_plot_slope_correct', 'WCST Delta Plot Slope (correct)'),
]
WCST_EXTRA_OUTCOMES += [
    (f"wcst_rt_vincentile_p{p}_correct", f"WCST RT Vincentile p{p} (correct)")
    for p in STROOP_VINCENTILES
]
WCST_EXTRA_OUTCOMES += [
    ('wcst_mean_rt_all', 'WCST Mean RT (All)'),
    ('wcst_accuracy_all', 'WCST Accuracy (All)'),
    ('wcst_error_rate_all', 'WCST Error Rate (All)'),
    ('wcst_ies', 'WCST Inverse Efficiency Score'),
    ('wcst_pre_error_slope_mean', 'WCST Pre-Error RT Slope (Mean)'),
    ('wcst_pre_error_slope_std', 'WCST Pre-Error RT Slope (SD)'),
    ('wcst_pre_error_n', 'WCST Pre-Error Events (N)'),
    ('wcst_block_pe_slope', 'WCST Block PE Slope'),
    ('wcst_block_pe_intercept', 'WCST Block PE Intercept'),
    ('wcst_block_pe_r2', 'WCST Block PE R-squared'),
    ('wcst_block_pe_initial', 'WCST Block PE Initial'),
      ('wcst_block_pe_final', 'WCST Block PE Final'),
      ('wcst_block_pe_change', 'WCST Block PE Change'),
      ('wcst_block_pe_blocks', 'WCST Block PE Blocks'),
      ('wcst_pe_rate_slope_half', 'WCST PE Rate Slope (Half)'),
      ('wcst_pe_rate_slope_quartile', 'WCST PE Rate Slope (Quartile)'),
      ('wcst_category_pe_slope', 'WCST Category PE Slope'),
      ('wcst_category_pe_delta_first3_last3', 'WCST Category PE Delta (First3-Last3)'),
      ('wcst_post_shift_pe_count_slope', 'WCST Post-Shift PE Count Slope'),
      ('wcst_post_shift_pe_rate_slope_k3', 'WCST Post-Shift PE Rate Slope (K3)'),
      ('wcst_post_shift_pe_rate_slope_k5', 'WCST Post-Shift PE Rate Slope (K5)'),
  ]

TIER1_OUTCOMES += STROOP_EXTRA_OUTCOMES + WCST_EXTRA_OUTCOMES
TIER1_OUTCOMES += [
    ('sustained_attention_index', 'Sustained Attention Index'),
    ('sustained_attention_cv_slope', 'Sustained Attention CV Slope'),
    ('sustained_attention_tau_slope', 'Sustained Attention Tau Slope'),
    ('sustained_attention_acc_slope', 'Sustained Attention Accuracy Slope'),
    ('sustained_attention_rt_slope', 'Sustained Attention RT Slope'),
]
TIER1_OUTCOMES += [
    ('cross_task_cv', 'Cross-Task CV'),
    ('cross_task_range', 'Cross-Task Range'),
    ('cross_task_mean', 'Cross-Task Mean'),
    ('cross_task_sd', 'Cross-Task SD'),
    ('cross_task_n_tasks', 'Cross-Task N Tasks'),
]

# Task-specific Tier 1 outcomes
TIER1_OUTCOMES_BY_TASK = {
    "overall": TIER1_OUTCOMES,
    "wcst": [
        ('pe_rate', 'WCST Perseverative Error Rate'),
        ('wcst_accuracy', 'WCST Accuracy (%)'),
        ('wcst_mean_rt', 'WCST Mean Reaction Time (ms)'),
        ('wcst_sd_rt', 'WCST Reaction Time SD'),
        ('pe_count', 'WCST Perseverative Error Count'),
        ('perseverativeResponses', 'WCST Perseverative Responses (count)'),
        ('perseverativeErrorCount', 'WCST Perseverative Errors (count)'),
        ('perseverativeResponsesPercent', 'WCST Perseverative Responses (%)'),
        ('wcst_pes', 'WCST Post-Error Slowing (ms)'),
        ('wcst_post_switch_error_rate', 'WCST Post-Switch Error Rate'),
        ('wcst_cv_rt', 'WCST Reaction Time Coefficient of Variation'),
        ('wcst_trials', 'WCST Valid Trial Count'),
        ('wcst_pre_switch_rt_mean_w1', 'WCST Pre-Switch RT Mean (w1)'),
        ('wcst_pre_switch_rt_mean_w2', 'WCST Pre-Switch RT Mean (w2)'),
        ('wcst_pre_switch_rt_mean_w3', 'WCST Pre-Switch RT Mean (w3)'),
        ('wcst_pre_switch_rt_mean_w4', 'WCST Pre-Switch RT Mean (w4)'),
        ('wcst_pre_switch_rt_mean_w5', 'WCST Pre-Switch RT Mean (w5)'),
        ('wcst_pre_switch_rt_delta_w1', 'WCST Pre-Switch RT Delta (w1)'),
        ('wcst_pre_switch_rt_delta_w2', 'WCST Pre-Switch RT Delta (w2)'),
        ('wcst_pre_switch_rt_delta_w3', 'WCST Pre-Switch RT Delta (w3)'),
        ('wcst_pre_switch_rt_delta_w4', 'WCST Pre-Switch RT Delta (w4)'),
        ('wcst_pre_switch_rt_delta_w5', 'WCST Pre-Switch RT Delta (w5)'),
        ('wcst_pre_switch_rt_mean_w1_correct', 'WCST Pre-Switch RT Mean (w1, correct)'),
        ('wcst_pre_switch_rt_mean_w2_correct', 'WCST Pre-Switch RT Mean (w2, correct)'),
        ('wcst_pre_switch_rt_mean_w3_correct', 'WCST Pre-Switch RT Mean (w3, correct)'),
        ('wcst_pre_switch_rt_mean_w4_correct', 'WCST Pre-Switch RT Mean (w4, correct)'),
        ('wcst_pre_switch_rt_mean_w5_correct', 'WCST Pre-Switch RT Mean (w5, correct)'),
        ('wcst_pre_switch_rt_delta_w1_correct', 'WCST Pre-Switch RT Delta (w1, correct)'),
        ('wcst_pre_switch_rt_delta_w2_correct', 'WCST Pre-Switch RT Delta (w2, correct)'),
        ('wcst_pre_switch_rt_delta_w3_correct', 'WCST Pre-Switch RT Delta (w3, correct)'),
        ('wcst_pre_switch_rt_delta_w4_correct', 'WCST Pre-Switch RT Delta (w4, correct)'),
        ('wcst_pre_switch_rt_delta_w5_correct', 'WCST Pre-Switch RT Delta (w5, correct)'),
        ('wcst_switch_cost_rt_mean_k1_k5', 'WCST Switch Cost RT Mean (k1-k5)'),
        ('wcst_switch_cost_rt_slope_k1_k5', 'WCST Switch Cost RT Slope (k1-k5)'),
        ('wcst_switch_cost_error_mean_k1_k5', 'WCST Switch Cost Error Mean (k1-k5)'),
        ('wcst_switch_cost_error_slope_k1_k5', 'WCST Switch Cost Error Slope (k1-k5)'),
        ('wcst_switch_cost_rt_k1_sd', 'WCST Switch Cost RT SD (k1)'),
        ('wcst_switch_cost_error_k1_sd', 'WCST Switch Cost Error SD (k1)'),
        ('wcst_switch_cost_rt_early', 'WCST Switch Cost RT Early'),
        ('wcst_switch_cost_rt_late', 'WCST Switch Cost RT Late'),
        ('wcst_switch_cost_rt_early_late_delta', 'WCST Switch Cost RT Early-Late'),
        ('wcst_switch_cost_error_early', 'WCST Switch Cost Error Early'),
        ('wcst_switch_cost_error_late', 'WCST Switch Cost Error Late'),
        ('wcst_switch_cost_error_early_late_delta', 'WCST Switch Cost Error Early-Late'),
        ('wcst_switch_cost_rt_k1_to_colour', 'WCST Switch Cost RT k1 (to colour)'),
        ('wcst_switch_cost_rt_k1_to_shape', 'WCST Switch Cost RT k1 (to shape)'),
        ('wcst_switch_cost_rt_k1_to_number', 'WCST Switch Cost RT k1 (to number)'),
        ('wcst_switch_cost_error_k1_to_colour', 'WCST Switch Cost Error k1 (to colour)'),
        ('wcst_switch_cost_error_k1_to_shape', 'WCST Switch Cost Error k1 (to shape)'),
        ('wcst_switch_cost_error_k1_to_number', 'WCST Switch Cost Error k1 (to number)'),
        ('wcst_exploration_rt_mean', 'WCST Exploration RT Mean'),
        ('wcst_exploration_rt_sd', 'WCST Exploration RT SD'),
        ('wcst_exploitation_rt_mean', 'WCST Exploitation RT Mean'),
        ('wcst_exploitation_rt_sd', 'WCST Exploitation RT SD'),
        ('wcst_exploration_penalty', 'WCST Exploration RT Penalty'),
        ('wcst_switch_first_trial_rt', 'WCST Switch First Trial RT'),
        ('wcst_switch_first_trial_correct', 'WCST Switch First Trial Accuracy'),
        ('wcst_switch_first_error_latency', 'WCST Switch First Error Latency'),
        ('wcst_switch_first_error_rt', 'WCST Switch First Error RT'),
        ('wcst_switch_signal_rt', 'WCST Switch Signal RT'),
        ('wcst_switch_signal_delta', 'WCST Switch Signal Delta'),
        ('wcst_lucky_streak_len', 'WCST Switch Lucky Streak Length'),
        ('wcst_exploration_rt_slope', 'WCST Exploration RT Slope'),
        ('wcst_exploration_error_rate', 'WCST Exploration Error Rate'),
        ('wcst_confirmation_rt_mean', 'WCST Confirmation RT Mean'),
        ('wcst_post_reacq_rt_cv', 'WCST Post-Reacq RT CV'),
        ('wcst_shift_pe_rt_mean', 'WCST Shift PE RT Mean'),
        ('wcst_shift_pe_count_mean', 'WCST Shift PE Count Mean'),
        ('wcst_shift_pe_rate_mean', 'WCST Shift PE Rate Mean'),
        ('wcst_post_shift_pe_rt_mean', 'WCST Post-Shift PE RT Mean'),
        ('wcst_post_shift_pe_slowing', 'WCST Post-Shift PE Slowing (ms)'),
        ('wcst_shift_pe_rt_mean_reacq', 'WCST Shift PE RT Mean (to reacq)'),
        ('wcst_shift_pe_count_mean_reacq', 'WCST Shift PE Count Mean (to reacq)'),
        ('wcst_shift_pe_rate_mean_reacq', 'WCST Shift PE Rate Mean (to reacq)'),
        ('wcst_post_shift_pe_rt_mean_reacq', 'WCST Post-Shift PE RT Mean (to reacq)'),
        ('wcst_post_shift_pe_slowing_reacq', 'WCST Post-Shift PE Slowing (to reacq)'),
        ('wcst_stable_pe_rt_mean', 'WCST Stable-Context PE RT Mean'),
        ('wcst_stable_pe_count_mean', 'WCST Stable-Context PE Count Mean'),
        ('wcst_stable_pe_rate_mean', 'WCST Stable-Context PE Rate Mean'),
        ('wcst_pe_context_rt_delta', 'WCST PE Context RT Delta (Stable-Shift)'),
        ('wcst_pe_context_rate_delta', 'WCST PE Context Rate Delta (Stable-Shift)'),
        ('wcst_pe_context_rt_delta_reacq', 'WCST PE Context RT Delta (Stable-Shift, reacq)'),
        ('wcst_pe_context_rate_delta_reacq', 'WCST PE Context Rate Delta (Stable-Shift, reacq)'),
        ('wcst_shift_error_rt_mean', 'WCST Shift-Context Error RT Mean'),
        ('wcst_post_shift_error_rt_mean', 'WCST Post-Shift Error RT Mean'),
        ('wcst_shift_post_error_slowing', 'WCST Shift-Context Post-Error Slowing'),
        ('wcst_stable_error_rt_mean', 'WCST Stable-Context Error RT Mean'),
        ('wcst_post_stable_error_rt_mean', 'WCST Post-Stable Error RT Mean'),
        ('wcst_stable_post_error_slowing', 'WCST Stable-Context Post-Error Slowing'),
        ('wcst_error_context_post_error_rt_delta', 'WCST Error Context Post-Error RT Delta (Stable-Shift)'),
        ('wcst_error_context_pes_delta', 'WCST Error Context PES Delta (Stable-Shift)'),
        ('wcst_nonshift_error_rt_mean', 'WCST Non-Shift Error RT Mean'),
        ('wcst_shift_error_count_total', 'WCST Shift-Context Error Count (Total)'),
        ('wcst_shift_error_rate', 'WCST Shift-Context Error Rate'),
        ('wcst_nonshift_error_count_total', 'WCST Non-Shift Error Count (Total)'),
        ('wcst_nonshift_error_rate', 'WCST Non-Shift Error Rate'),
        ('wcst_hmm_lapse_occupancy', 'WCST HMM Lapse Occupancy (%)'),
        ('wcst_hmm_trans_to_lapse', 'WCST HMM P(Focus->Lapse)'),
        ('wcst_hmm_trans_to_focus', 'WCST HMM P(Lapse->Focus)'),
        ('wcst_hmm_stay_lapse', 'WCST HMM P(Lapse->Lapse)'),
        ('wcst_hmm_stay_focus', 'WCST HMM P(Focus->Focus)'),
        ('wcst_hmm_lapse_rt_mean', 'WCST HMM Lapse RT Mean'),
        ('wcst_hmm_focus_rt_mean', 'WCST HMM Focus RT Mean'),
        ('wcst_hmm_rt_diff', 'WCST HMM RT Difference'),
        ('wcst_hmm_state_changes', 'WCST HMM State Changes'),
        ('wcst_rl_alpha', 'WCST RL alpha'),
        ('wcst_rl_beta', 'WCST RL beta'),
        ('wcst_rl_alpha_pos', 'WCST RL alpha (pos)'),
        ('wcst_rl_alpha_neg', 'WCST RL alpha (neg)'),
        ('wcst_rl_alpha_asymmetry', 'WCST RL alpha asymmetry'),
        ('wcst_rl_beta_asym', 'WCST RL beta (asym)'),
        ('wcst_wsls_p_stay_win', 'WCST WSLS P(stay|win)'),
        ('wcst_wsls_p_shift_lose', 'WCST WSLS P(shift|lose)'),
        ('wcst_brl_hazard', 'WCST Bayesian RL hazard'),
        ('wcst_brl_noise', 'WCST Bayesian RL noise'),
        ('wcst_brl_beta', 'WCST Bayesian RL beta'),
    ] + WCST_EXTRA_OUTCOMES,
    "stroop": [
        ('stroop_interference', 'Stroop Interference Effect'),
        ('rt_mean_incongruent', 'Stroop Mean RT (Incongruent)'),
        ('rt_mean_congruent', 'Stroop Mean RT (Congruent)'),
        ('rt_mean_neutral', 'Stroop Mean RT (Neutral)'),
        ('accuracy_incongruent', 'Stroop Accuracy (Incongruent)'),
        ('accuracy_congruent', 'Stroop Accuracy (Congruent)'),
        ('accuracy_neutral', 'Stroop Accuracy (Neutral)'),
        ('stroop_effect', 'Stroop Effect (RT Difference)'),
        ('stroop_post_error_slowing', 'Stroop Post-Error Slowing (ms)'),
        ('stroop_post_error_rt', 'Stroop Post-Error RT (ms)'),
        ('stroop_post_correct_rt', 'Stroop Post-Correct RT (ms)'),
        ('stroop_incong_slope', 'Stroop Incongruent RT Slope'),
        ('stroop_cv_all', 'Stroop RT Coefficient of Variation (All)'),
        ('stroop_cv_incong', 'Stroop RT CV (Incongruent)'),
        ('stroop_cv_cong', 'Stroop RT CV (Congruent)'),
        ('stroop_trials', 'Stroop Valid Trial Count'),
        ('stroop_exg_congruent_mu', 'Stroop Ex-Gaussian mu (Congruent)'),
        ('stroop_exg_congruent_sigma', 'Stroop Ex-Gaussian sigma (Congruent)'),
        ('stroop_exg_congruent_tau', 'Stroop Ex-Gaussian tau (Congruent)'),
        ('stroop_exg_incongruent_mu', 'Stroop Ex-Gaussian mu (Incongruent)'),
        ('stroop_exg_incongruent_sigma', 'Stroop Ex-Gaussian sigma (Incongruent)'),
        ('stroop_exg_incongruent_tau', 'Stroop Ex-Gaussian tau (Incongruent)'),
        ('stroop_exg_neutral_mu', 'Stroop Ex-Gaussian mu (Neutral)'),
        ('stroop_exg_neutral_sigma', 'Stroop Ex-Gaussian sigma (Neutral)'),
        ('stroop_exg_neutral_tau', 'Stroop Ex-Gaussian tau (Neutral)'),
        ('stroop_exg_mu_interference', 'Stroop Ex-Gaussian mu (Interference)'),
        ('stroop_exg_sigma_interference', 'Stroop Ex-Gaussian sigma (Interference)'),
        ('stroop_exg_tau_interference', 'Stroop Ex-Gaussian tau (Interference)'),
    ] + STROOP_EXTRA_OUTCOMES,
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

def _merge_extra_features(df: pd.DataFrame, extra_path: Path) -> pd.DataFrame:
    if not extra_path.exists():
        return df
    extra_df = pd.read_csv(extra_path, encoding="utf-8-sig")
    if extra_df.empty:
        return df
    extra_df = ensure_participant_id(extra_df)
    overlap = [c for c in extra_df.columns if c != "participant_id" and c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)
    return df.merge(extra_df, on="participant_id", how="left")


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
    Load master dataset with Tier-1 metrics pre-computed.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {sorted(VALID_TASKS)}")
    df = load_master_dataset(task=task)
    if task == "wcst":
        df = _merge_extra_features(
            df,
            get_results_dir("wcst") / "5_wcst_dynamic_recovery_features.csv",
        )
        df = _merge_extra_features(
            df,
            get_results_dir("wcst") / "5_wcst_pre_switch_features.csv",
        )
        df = _merge_extra_features(
            df,
            get_results_dir("wcst") / "5_wcst_pre_switch_features_correct.csv",
        )
        df = _merge_extra_features(
            df,
            get_results_dir("wcst") / "5_wcst_switching_features.csv",
        )
    if apply_qc:
        df = _apply_qc_filter(df, task)
    return df


def get_tier1_outcomes(task: str) -> list[tuple[str, str]]:
    """Return task-specific Tier 1 outcome list."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {sorted(VALID_TASKS)}")
    return TIER1_OUTCOMES_BY_TASK[task]


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
    cov_type: str = "HC3",
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
    try:
        model = smf.ols(formula, data=sub).fit(cov_type=cov_type)
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
        "cov_type": cov_type,
    }
