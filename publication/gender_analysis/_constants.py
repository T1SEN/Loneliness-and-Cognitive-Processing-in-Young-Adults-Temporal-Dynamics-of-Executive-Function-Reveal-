"""
Gender Analysis Constants
=========================

Shared constants for gender-specific analyses.
"""

# Minimum sample sizes
MIN_SAMPLE_STRATIFIED = 15  # Minimum N per gender for stratified analysis
MIN_SAMPLE_INTERACTION = 30  # Minimum total N for interaction testing

# EF outcome variables for gender analyses
EF_OUTCOMES = [
    # WCST
    'pe_rate', 'npe_rate', 'total_error_rate', 'categories_completed',
    'trials_to_first_category', 'conceptual_level_responses',
    # Stroop
    'stroop_interference', 'stroop_effect', 'stroop_acc', 'stroop_rt_mean',
    # PRP
    'prp_bottleneck', 'prp_short_rt', 'prp_long_rt', 'prp_pep_effect',
    # Derived
    'wcst_pe', 'stroop_congruent_rt', 'stroop_incongruent_rt',
]

# Primary outcomes for focused analysis
PRIMARY_OUTCOMES = ['pe_rate', 'stroop_interference', 'prp_bottleneck']

# DDM parameters
DDM_PARAMS = ['v', 'a', 't', 'efficiency']

# Network analysis variables
NETWORK_VARS = [
    'ucla_total', 'pe_rate', 'stroop_interference',
    'prp_bottleneck', 'dass_dep', 'dass_anx', 'dass_str'
]

# DASS control formula components
DASS_COVARIATES = ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']

# Standard formula for gender-stratified analysis (within gender)
STRATIFIED_FORMULA = "{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"

# Standard formula for interaction testing
INTERACTION_FORMULA = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

__all__ = [
    'MIN_SAMPLE_STRATIFIED',
    'MIN_SAMPLE_INTERACTION',
    'EF_OUTCOMES',
    'PRIMARY_OUTCOMES',
    'DDM_PARAMS',
    'NETWORK_VARS',
    'DASS_COVARIATES',
    'STRATIFIED_FORMULA',
    'INTERACTION_FORMULA',
]
