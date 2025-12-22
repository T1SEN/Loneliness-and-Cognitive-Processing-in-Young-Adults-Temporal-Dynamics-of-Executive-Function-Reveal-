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

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR, VALID_TASKS

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

# Shared constants for derived outcome lists
STROOP_CONDITIONS = ["congruent", "incongruent", "neutral"]
STROOP_VINCENTILES = [10, 30, 50, 70, 90]
PRP_SOA_LEVELS = [50, 150, 300, 600, 1200]
WCST_SWITCH_K = [1, 2, 3, 4, 5]
WCST_CATEGORY_MAX = 6

# Tier 1 outcomes for hierarchical regression (overall / full list)
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

    # PRP Ex-Gaussian parameters
    ('prp_exg_short_mu', 'PRP Ex-Gaussian mu (Short SOA)'),
    ('prp_exg_short_sigma', 'PRP Ex-Gaussian sigma (Short SOA)'),
    ('prp_exg_short_tau', 'PRP Ex-Gaussian tau (Short SOA)'),
    ('prp_exg_long_mu', 'PRP Ex-Gaussian mu (Long SOA)'),
    ('prp_exg_long_sigma', 'PRP Ex-Gaussian sigma (Long SOA)'),
    ('prp_exg_long_tau', 'PRP Ex-Gaussian tau (Long SOA)'),
    ('prp_exg_overall_mu', 'PRP Ex-Gaussian mu (Overall)'),
    ('prp_exg_overall_sigma', 'PRP Ex-Gaussian sigma (Overall)'),
    ('prp_exg_overall_tau', 'PRP Ex-Gaussian tau (Overall)'),
    ('prp_exg_mu_bottleneck', 'PRP Ex-Gaussian mu (Bottleneck)'),
    ('prp_exg_sigma_bottleneck', 'PRP Ex-Gaussian sigma (Bottleneck)'),
    ('prp_exg_tau_bottleneck', 'PRP Ex-Gaussian tau (Bottleneck)'),

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

    # PRP Central Bottleneck model parameters
    ('prp_cb_base', 'PRP CB Base RT (ms)'),
    ('prp_cb_bottleneck', 'PRP CB Bottleneck Duration (ms)'),
    ('prp_cb_r_squared', 'PRP CB Model R-squared'),
    ('prp_cb_slope', 'PRP CB Slope (short SOA)'),

    # Stroop LBA model parameters (interference indices)
    ('stroop_lba_v_correct_interference', 'Stroop LBA v_correct Interference'),
    ('stroop_lba_b_interference', 'Stroop LBA b (Threshold) Interference'),
    ('stroop_lba_t0_interference', 'Stroop LBA t0 Interference'),
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

STROOP_LBA_PARAMS = ["v_correct", "v_incorrect", "A", "b", "t0", "negloglik", "aic", "bic"]
STROOP_EXTRA_OUTCOMES += [
    (f"stroop_lba_{cond}_{param}", f"Stroop LBA {param} ({cond})")
    for cond in STROOP_CONDITIONS
    for param in STROOP_LBA_PARAMS
]

PRP_EXTRA_OUTCOMES = [
    ('prp_slow_prob_baseline', 'PRP Slow-State Probability (Baseline)'),
    ('prp_slow_prob_post_error', 'PRP Slow-State Prob (Post-Error)'),
    ('prp_slow_prob_post_error_delta', 'PRP Slow-State Prob Delta (Post-Error)'),
    ('prp_slow_prob_stable', 'PRP Slow-State Prob (Stable)'),
    ('prp_slow_prob_stable_delta', 'PRP Slow-State Prob Delta (Stable)'),
    ('prp_t1_cost', 'PRP T1 Cost (Short - Long)'),
    ('prp_rt1_rt2_coupling', 'PRP RT1-RT2 Coupling'),
    ('prp_t2_rt_t1_error', 'PRP T2 RT (T1 Error)'),
    ('prp_t2_rt_t1_correct', 'PRP T2 RT (T1 Correct)'),
    ('prp_t2_interference_t1_error', 'PRP T2 Interference (T1 Error - Correct)'),
    ('prp_order_violation_rate', 'PRP Order Violation Rate'),
    ('prp_t1_only_rate', 'PRP T1-only Rate'),
    ('prp_t2_only_rate', 'PRP T2-only Rate'),
    ('prp_t2_while_t1_pending_rate', 'PRP T2 While T1 Pending Rate'),
    ('prp_iri_mean', 'PRP IRI Mean'),
    ('prp_iri_median', 'PRP IRI Median'),
    ('prp_iri_p10', 'PRP IRI p10'),
    ('prp_iri_p25', 'PRP IRI p25'),
    ('prp_t1_error_run_mean', 'PRP T1 Error Run Mean'),
    ('prp_t1_error_run_max', 'PRP T1 Error Run Max'),
    ('prp_t2_error_run_mean', 'PRP T2 Error Run Mean'),
    ('prp_t2_error_run_max', 'PRP T2 Error Run Max'),
    ('prp_cascade_run_mean', 'PRP Cascade Run Mean'),
    ('prp_cascade_run_max', 'PRP Cascade Run Max'),
    ('prp_exg_correct_mu_bottleneck', 'PRP Ex-Gaussian mu (Bottleneck, correct)'),
    ('prp_exg_correct_sigma_bottleneck', 'PRP Ex-Gaussian sigma (Bottleneck, correct)'),
    ('prp_exg_correct_tau_bottleneck', 'PRP Ex-Gaussian tau (Bottleneck, correct)'),
    ('prp_exg_tau_ratio_short_long', 'PRP Ex-Gaussian tau Ratio (Short/Long)'),
    ('prp_exg_correct_tau_ratio_short_long', 'PRP Ex-Gaussian tau Ratio (Short/Long, correct)'),
    ('prp_bottleneck_auc', 'PRP Bottleneck AUC'),
    ('prp_bottleneck_slope_short', 'PRP Bottleneck Slope (Short Range)'),
    ('prp_bottleneck_slope_long', 'PRP Bottleneck Slope (Long Range)'),
    ('prp_rt2_asymptote_long_soa', 'PRP RT2 Asymptote (Long SOA)'),
    ('prp_bottleneck_curvature', 'PRP Bottleneck Curvature'),
    ('prp_cb_aic', 'PRP CB AIC'),
    ('prp_cb_bic', 'PRP CB BIC'),
    ('prp_cs_base', 'PRP CS Base RT'),
    ('prp_cs_amplitude', 'PRP CS Amplitude'),
    ('prp_cs_tau', 'PRP CS Tau'),
    ('prp_cs_r_squared', 'PRP CS R-squared'),
    ('prp_cs_rmse', 'PRP CS RMSE'),
    ('prp_cs_aic', 'PRP CS AIC'),
    ('prp_cs_bic', 'PRP CS BIC'),
    ('prp_mix_cb_weight', 'PRP Mix CB Weight'),
    ('prp_mix_r_squared', 'PRP Mix R-squared'),
    ('prp_mix_rmse', 'PRP Mix RMSE'),
    ('prp_mix_aic', 'PRP Mix AIC'),
    ('prp_mix_bic', 'PRP Mix BIC'),
]

PRP_EXTRA_OUTCOMES += [
    (f"prp_t2_rt_mean_soa_{soa}", f"PRP T2 Mean RT (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_t2_rt_median_soa_{soa}", f"PRP T2 Median RT (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_t2_rt_sd_soa_{soa}", f"PRP T2 RT SD (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_t1_rt_mean_soa_{soa}", f"PRP T1 Mean RT (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_t1_rt_median_soa_{soa}", f"PRP T1 Median RT (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_t1_rt_sd_soa_{soa}", f"PRP T1 RT SD (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_t1_acc_soa_{soa}", f"PRP T1 Accuracy (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_rt1_rt2_coupling_soa_{soa}", f"PRP RT1-RT2 Coupling (SOA {soa})")
    for soa in PRP_SOA_LEVELS
]

PRP_EXTRA_OUTCOMES += [
    (f"prp_exg_correct_{soa}_{param}", f"PRP Ex-Gaussian {param} ({soa}, correct)")
    for soa in ("short", "long", "overall")
    for param in ("mu", "sigma", "tau")
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_exg_soa_{soa}_{param}", f"PRP Ex-Gaussian {param} (SOA {soa})")
    for soa in PRP_SOA_LEVELS
    for param in ("mu", "sigma", "tau")
]
PRP_EXTRA_OUTCOMES += [
    (f"prp_exg_correct_soa_{soa}_{param}", f"PRP Ex-Gaussian {param} (SOA {soa}, correct)")
    for soa in PRP_SOA_LEVELS
    for param in ("mu", "sigma", "tau")
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
    ('wcst_rt_slope_within_category', 'WCST RT Slope within Category'),
    ('wcst_acc_slope_within_category', 'WCST Accuracy Slope within Category'),
    ('wcst_rt_jump_at_switch', 'WCST RT Jump at Switch'),
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

TIER1_OUTCOMES += STROOP_EXTRA_OUTCOMES + PRP_EXTRA_OUTCOMES + WCST_EXTRA_OUTCOMES

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
    "prp": [
        ('prp_bottleneck', 'PRP Delay Effect'),
        ('t2_rt_mean_short', 'PRP T2 Mean RT (Short SOA)'),
        ('t2_rt_mean_long', 'PRP T2 Mean RT (Long SOA)'),
        ('t2_rt_sd_short', 'PRP T2 RT SD (Short SOA)'),
        ('t2_rt_sd_long', 'PRP T2 RT SD (Long SOA)'),
        ('prp_t2_cv_all', 'PRP T2 Coefficient of Variation (All)'),
        ('prp_t2_cv_short', 'PRP T2 CV (Short SOA)'),
        ('prp_t2_cv_long', 'PRP T2 CV (Long SOA)'),
        ('prp_cascade_rate', 'PRP Error Cascade Rate'),
        ('prp_cascade_inflation', 'PRP Cascade Inflation'),
        ('prp_pes', 'PRP Post-Error Slowing (ms)'),
        ('prp_t2_trials', 'PRP Valid T2 Trial Count'),
        ('prp_exg_short_mu', 'PRP Ex-Gaussian mu (Short SOA)'),
        ('prp_exg_short_sigma', 'PRP Ex-Gaussian sigma (Short SOA)'),
        ('prp_exg_short_tau', 'PRP Ex-Gaussian tau (Short SOA)'),
        ('prp_exg_long_mu', 'PRP Ex-Gaussian mu (Long SOA)'),
        ('prp_exg_long_sigma', 'PRP Ex-Gaussian sigma (Long SOA)'),
        ('prp_exg_long_tau', 'PRP Ex-Gaussian tau (Long SOA)'),
        ('prp_exg_overall_mu', 'PRP Ex-Gaussian mu (Overall)'),
        ('prp_exg_overall_sigma', 'PRP Ex-Gaussian sigma (Overall)'),
        ('prp_exg_overall_tau', 'PRP Ex-Gaussian tau (Overall)'),
        ('prp_exg_mu_bottleneck', 'PRP Ex-Gaussian mu (Bottleneck)'),
        ('prp_exg_sigma_bottleneck', 'PRP Ex-Gaussian sigma (Bottleneck)'),
        ('prp_exg_tau_bottleneck', 'PRP Ex-Gaussian tau (Bottleneck)'),
        ('prp_cb_base', 'PRP CB Base RT (ms)'),
        ('prp_cb_bottleneck', 'PRP CB Bottleneck Duration (ms)'),
        ('prp_cb_r_squared', 'PRP CB Model R-squared'),
        ('prp_cb_slope', 'PRP CB Slope (short SOA)'),
    ] + PRP_EXTRA_OUTCOMES,
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
        ('stroop_lba_v_correct_interference', 'Stroop LBA v_correct Interference'),
        ('stroop_lba_b_interference', 'Stroop LBA b (Threshold) Interference'),
        ('stroop_lba_t0_interference', 'Stroop LBA t0 Interference'),
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

def get_analysis_data(task: str) -> pd.DataFrame:
    """
    Load master dataset with Tier-1 metrics pre-computed.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {sorted(VALID_TASKS)}")
    return load_master_dataset(task=task)


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
    base_dir = ANALYSIS_OUTPUT_DIR / "basic_analysis"
    output_dir = base_dir / task
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
