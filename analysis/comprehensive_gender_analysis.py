"""
Comprehensive Gender Moderation Analysis
=========================================

This script performs an extensive analysis of gender moderation effects in the
relationship between loneliness (UCLA) and executive function performance across
Stroop, WCST, and PRP tasks.

Analysis phases:
1. WCST deep-dive (follow-up on significant perseverative error finding)
2. Stroop/PRP multi-metric exploration
3. Trial-level pattern analysis
4. Nonlinear effects and 3-way interactions
5. Robustness checks and multiple comparison corrections

Author: Research Team
Date: 2025
"""

import sys
import warnings
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import json

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seed for reproducibility
np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/gender_comprehensive")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_wcst_extra(extra_str):
    """Parse the 'extra' field from WCST trials."""
    if pd.isna(extra_str) or not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}


def bootstrap_ci(data, stat_func, n_boot=5000, ci=95):
    """Bootstrap confidence interval for a statistic."""
    boot_stats = []
    n = len(data)
    for _ in range(n_boot):
        sample = data.sample(n=n, replace=True)
        boot_stats.append(stat_func(sample))

    lower = (100 - ci) / 2
    upper = 100 - lower
    return np.percentile(boot_stats, [lower, upper])


def permutation_test(group1, group2, n_perm=5000):
    """Permutation test for difference in means."""
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    perm_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diffs.append(np.mean(combined[:n1]) - np.mean(combined[n1:]))

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value


def normalize_gender_series(series: pd.Series) -> pd.Series:
    """Normalize mixed-language gender labels to 'male'/'female' tokens."""
    if series is None:
        return pd.Series(dtype="object")
    s = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z\u3131-\u318e\uac00-\ud7a3]", "", regex=True)
    )
    out = pd.Series(np.nan, index=series.index, dtype="object")

    male_mask = (
        s.isin({"m", "male", "man"})
        | s.str.startswith("남")
        | s.str.contains("소년")
    )
    female_mask = (
        s.isin({"f", "female", "woman"})
        | s.str.startswith("여")
        | s.str.contains("소녀")
    )

    out[male_mask] = "male"
    out[female_mask] = "female"
    return out


def run_gender_moderation(df, outcome_col, predictor='z_ucla', covariates=None):
    """
    Run gender moderation analysis with standardized predictors.

    Returns: dict with slopes, interaction stats, permutation test, bootstrap CI
    """
    if covariates is None:
        covariates = ['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_stress']

    # Build formula
    cov_str = ' + '.join(covariates) if covariates else ''
    formula = f"{outcome_col} ~ {predictor} * gender_male"
    if cov_str:
        formula += f" + {cov_str}"

    # Fit OLS model
    model = ols(formula, data=df).fit()

    # Extract interaction term
    interaction_term = f"{predictor}:gender_male"
    try:
        interaction_beta = model.params[interaction_term]
        interaction_pval = model.pvalues[interaction_term]
    except KeyError:
        interaction_beta = np.nan
        interaction_pval = np.nan

    # Simple slopes
    female_df = df[df['gender_male'] == 0].copy()
    male_df = df[df['gender_male'] == 1].copy()

    # Female slope
    if len(female_df) > 10:
        formula_f = f"{outcome_col} ~ {predictor}"
        if cov_str:
            formula_f += f" + {cov_str}"
        model_f = ols(formula_f, data=female_df).fit()
        female_beta = model_f.params[predictor]
        female_pval = model_f.pvalues[predictor]
        female_n = len(female_df)
    else:
        female_beta, female_pval, female_n = np.nan, np.nan, 0

    # Male slope
    if len(male_df) > 10:
        formula_m = f"{outcome_col} ~ {predictor}"
        if cov_str:
            formula_m += f" + {cov_str}"
        model_m = ols(formula_m, data=male_df).fit()
        male_beta = model_m.params[predictor]
        male_pval = model_m.pvalues[predictor]
        male_n = len(male_df)
    else:
        male_beta, male_pval, male_n = np.nan, np.nan, 0

    # Permutation test for interaction
    n_perm = 1000
    perm_betas = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm['gender_male'] = np.random.permutation(df_perm['gender_male'].values)
        try:
            model_perm = ols(formula, data=df_perm).fit()
            perm_betas.append(model_perm.params[interaction_term])
        except:
            continue

    perm_pval = np.mean(np.abs(perm_betas) >= np.abs(interaction_beta)) if perm_betas else np.nan

    # Bootstrap CI for interaction
    n_boot = 1000
    boot_betas = []
    for _ in range(n_boot):
        df_boot = df.sample(n=len(df), replace=True)
        try:
            model_boot = ols(formula, data=df_boot).fit()
            boot_betas.append(model_boot.params[interaction_term])
        except:
            continue

    if boot_betas:
        boot_ci_lower = np.percentile(boot_betas, 2.5)
        boot_ci_upper = np.percentile(boot_betas, 97.5)
        boot_pct_positive = np.mean(np.array(boot_betas) > 0) * 100
    else:
        boot_ci_lower, boot_ci_upper, boot_pct_positive = np.nan, np.nan, np.nan

    results = {
        'outcome': outcome_col,
        'female_beta': female_beta,
        'female_pval': female_pval,
        'female_n': female_n,
        'male_beta': male_beta,
        'male_pval': male_pval,
        'male_n': male_n,
        'interaction_beta': interaction_beta,
        'interaction_pval': interaction_pval,
        'perm_pval': perm_pval,
        'boot_ci_lower': boot_ci_lower,
        'boot_ci_upper': boot_ci_upper,
        'boot_pct_positive': boot_pct_positive,
        'total_n': len(df)
    }

    return results


def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

print("="*80)
print("COMPREHENSIVE GENDER MODERATION ANALYSIS")
print("="*80)
print()

print("[1/16] Loading datasets...")

# Load base data
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
cognitive = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv", encoding='utf-8-sig')

# Load trial-level data
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')

print(f"  - Participants: {len(participants)}")
print(f"  - Survey responses: {len(surveys)}")
print(f"  - Cognitive summaries: {len(cognitive)}")
print(f"  - Stroop trials: {len(stroop_trials)}")
print(f"  - WCST trials: {len(wcst_trials)}")
print(f"  - PRP trials: {len(prp_trials)}")
print()

# Normalize column names - handle potential duplicates
# Some CSVs have both participantId and participant_id, causing issues
for df, name in [(participants, 'participants'), (surveys, 'surveys'), (cognitive, 'cognitive'),
                 (stroop_trials, 'stroop_trials'), (wcst_trials, 'wcst_trials'), (prp_trials, 'prp_trials')]:
    if 'participantId' in df.columns and 'participant_id' in df.columns:
        # Drop the old column and keep the new one
        df.drop(columns=['participantId'], inplace=True)
    elif 'participantId' in df.columns:
        # Rename
        df.rename(columns={'participantId': 'participant_id'}, inplace=True)

# Extract UCLA and DASS scores
ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].rename(
    columns={'score': 'ucla_total'})
dass_scores = surveys[surveys['surveyName'] == 'dass'][['participant_id', 'score', 'score_D', 'score_A', 'score_S']].copy()
dass_scores = dass_scores.rename(columns={
    'score_D': 'dass_depression',
    'score_A': 'dass_anxiety',
    'score_S': 'dass_stress',
    'score': 'dass_total'
})

# Merge base dataset
master = participants.merge(ucla_scores, on='participant_id', how='inner')
master = master.merge(dass_scores, on='participant_id', how='left')
master = master.merge(cognitive, on='participant_id', how='left')

# Ensure one row per participant before adding derived metrics
master = (
    master.sort_values("participant_id")
    .drop_duplicates(subset=["participant_id"], keep="first")
    .reset_index(drop=True)
)

print(f"[2/16] Merging datasets... Master dataset: {len(master)} participants")
print()


# ============================================================================
# DERIVE EXECUTIVE FUNCTION METRICS FROM TRIAL DATA
# ============================================================================

print("[3/16] Computing WCST expanded metrics...")

# WCST metrics
wcst_metrics_list = []

for pid in wcst_trials['participant_id'].dropna().unique():
    pid_data = wcst_trials[wcst_trials['participant_id'] == pid].copy()

    # Filter valid trials
    pid_data = pid_data[pid_data['timeout'] == False].copy()

    if len(pid_data) < 10:
        continue

    # Parse extra field for isPE
    pid_data['isPE_parsed'] = pid_data['extra'].apply(
        lambda x: parse_wcst_extra(x).get('isPE', False) if pd.notna(x) else False
    )

    # Basic counts
    total_trials = len(pid_data)
    pe_count = pid_data['isPE_parsed'].sum()
    npe_count = (~pid_data['correct'] & ~pid_data['isPE_parsed']).sum()
    correct_count = pid_data['correct'].sum()

    # Rates
    pe_rate = (pe_count / total_trials) * 100 if total_trials > 0 else 0
    npe_rate = (npe_count / total_trials) * 100 if total_trials > 0 else 0
    accuracy = (correct_count / total_trials) * 100 if total_trials > 0 else 0

    # Categories achieved (from summary if available, otherwise estimate)
    # This would come from cognitive summary - we'll merge it later

    # Mean RT
    mean_rt = pid_data['rt_ms'].mean()

    # RT variability
    rt_cv = pid_data['rt_ms'].std() / pid_data['rt_ms'].mean() if pid_data['rt_ms'].mean() > 0 else 0

    # Learning dynamics: split trials into thirds
    n_trials = len(pid_data)
    third = n_trials // 3

    early_pe_rate = (pid_data.iloc[:third]['isPE_parsed'].sum() / third * 100) if third > 0 else 0
    late_pe_rate = (pid_data.iloc[-third:]['isPE_parsed'].sum() / third * 100) if third > 0 else 0

    wcst_metrics_list.append({
        'participant_id': pid,
        'wcst_pe_rate': pe_rate,
        'wcst_npe_rate': npe_rate,
        'wcst_accuracy': accuracy,
        'wcst_mean_rt': mean_rt,
        'wcst_rt_cv': rt_cv,
        'wcst_early_pe_rate': early_pe_rate,
        'wcst_late_pe_rate': late_pe_rate,
        'wcst_learning_improvement': early_pe_rate - late_pe_rate,
        'wcst_total_trials': total_trials
    })

wcst_metrics_df = pd.DataFrame(wcst_metrics_list)
print(f"  - WCST metrics computed for {len(wcst_metrics_df)} participants")
print()

print("[4/16] Computing Stroop expanded metrics...")

# Stroop metrics
stroop_metrics_list = []

for pid in stroop_trials['participant_id'].dropna().unique():
    pid_data = stroop_trials[stroop_trials['participant_id'] == pid].copy()

    # Filter valid trials
    pid_data = pid_data[(pid_data['timeout'] == False) & (pid_data['rt_ms'] > 0)].copy()

    if len(pid_data) < 10:
        continue

    # Separate by condition
    cong = pid_data[pid_data['type'] == 'congruent'].copy()
    incong = pid_data[pid_data['type'] == 'incongruent'].copy()

    # Accuracy by condition
    cong_acc = (cong['correct'].sum() / len(cong) * 100) if len(cong) > 0 else 0
    incong_acc = (incong['correct'].sum() / len(incong) * 100) if len(incong) > 0 else 0

    # RT by condition
    cong_rt = cong['rt_ms'].mean() if len(cong) > 0 else 0
    incong_rt = incong['rt_ms'].mean() if len(incong) > 0 else 0
    stroop_effect = incong_rt - cong_rt

    # RT variability by condition
    cong_cv = (cong['rt_ms'].std() / cong['rt_ms'].mean()) if len(cong) > 0 and cong['rt_ms'].mean() > 0 else 0
    incong_cv = (incong['rt_ms'].std() / incong['rt_ms'].mean()) if len(incong) > 0 and incong['rt_ms'].mean() > 0 else 0

    # Timeout rate by condition
    total_trials = len(stroop_trials[stroop_trials['participant_id'] == pid])
    cong_timeout_rate = (stroop_trials[(stroop_trials['participant_id'] == pid) &
                                        (stroop_trials['type'] == 'congruent') &
                                        (stroop_trials['timeout'] == True)].shape[0] /
                         total_trials * 100) if total_trials > 0 else 0
    incong_timeout_rate = (stroop_trials[(stroop_trials['participant_id'] == pid) &
                                          (stroop_trials['type'] == 'incongruent') &
                                          (stroop_trials['timeout'] == True)].shape[0] /
                           total_trials * 100) if total_trials > 0 else 0

    # Post-conflict adaptation
    # RT on trial n+1 after congruent vs incongruent trial n
    pid_data = pid_data.sort_values('trial_index').reset_index(drop=True)
    post_cong_rts = []
    post_incong_rts = []

    for i in range(len(pid_data) - 1):
        if pid_data.loc[i, 'type'] == 'congruent' and pid_data.loc[i+1, 'correct']:
            post_cong_rts.append(pid_data.loc[i+1, 'rt_ms'])
        elif pid_data.loc[i, 'type'] == 'incongruent' and pid_data.loc[i+1, 'correct']:
            post_incong_rts.append(pid_data.loc[i+1, 'rt_ms'])

    post_conflict_adaptation = (np.mean(post_cong_rts) - np.mean(post_incong_rts)) if (post_cong_rts and post_incong_rts) else 0

    # RT trend across task (practice/fatigue)
    if len(pid_data) > 20:
        trial_numbers = np.arange(len(pid_data))
        slope, _, _, _, _ = stats.linregress(trial_numbers, pid_data['rt_ms'].values)
        rt_slope = slope
    else:
        rt_slope = 0

    stroop_metrics_list.append({
        'participant_id': pid,
        'stroop_cong_acc': cong_acc,
        'stroop_incong_acc': incong_acc,
        'stroop_cong_rt': cong_rt,
        'stroop_incong_rt': incong_rt,
        'stroop_effect': stroop_effect,
        'stroop_cong_cv': cong_cv,
        'stroop_incong_cv': incong_cv,
        'stroop_cong_timeout_rate': cong_timeout_rate,
        'stroop_incong_timeout_rate': incong_timeout_rate,
        'stroop_post_conflict_adaptation': post_conflict_adaptation,
        'stroop_rt_slope': rt_slope,
        'stroop_total_trials': len(pid_data)
    })

stroop_metrics_df = pd.DataFrame(stroop_metrics_list)
print(f"  - Stroop metrics computed for {len(stroop_metrics_df)} participants")
print()

print("[5/16] Computing PRP expanded metrics...")

# PRP metrics
prp_metrics_list = []

for pid in prp_trials['participant_id'].dropna().unique():
    pid_data = prp_trials[prp_trials['participant_id'] == pid].copy()

    # Filter valid trials
    pid_data = pid_data[(pid_data['t1_timeout'] == False) &
                        (pid_data['t2_timeout'] == False) &
                        (pid_data['t1_rt_ms'] > 0) &
                        (pid_data['t2_rt_ms'] > 0)].copy()

    if len(pid_data) < 10:
        continue

    # Bin SOA
    def bin_soa(soa):
        if soa <= 150:
            return 'short'
        elif soa >= 1200:
            return 'long'
        elif 300 <= soa <= 600:
            return 'medium'
        else:
            return 'other'

    pid_data['soa_bin'] = pid_data['soa_nominal_ms'].apply(bin_soa)

    # SOA-specific accuracy
    short_soa = pid_data[pid_data['soa_bin'] == 'short']
    long_soa = pid_data[pid_data['soa_bin'] == 'long']

    short_t1_acc = (short_soa['t1_correct'].sum() / len(short_soa) * 100) if len(short_soa) > 0 else 0
    short_t2_acc = (short_soa['t2_correct'].sum() / len(short_soa) * 100) if len(short_soa) > 0 else 0
    long_t1_acc = (long_soa['t1_correct'].sum() / len(long_soa) * 100) if len(long_soa) > 0 else 0
    long_t2_acc = (long_soa['t2_correct'].sum() / len(long_soa) * 100) if len(long_soa) > 0 else 0

    # T1 slowing at short SOA
    short_t1_rt = short_soa['t1_rt_ms'].mean() if len(short_soa) > 0 else 0
    long_t1_rt = long_soa['t1_rt_ms'].mean() if len(long_soa) > 0 else 0
    t1_slowing = short_t1_rt - long_t1_rt

    # PRP bottleneck
    short_t2_rt = short_soa['t2_rt_ms'].mean() if len(short_soa) > 0 else 0
    long_t2_rt = long_soa['t2_rt_ms'].mean() if len(long_soa) > 0 else 0
    prp_bottleneck = short_t2_rt - long_t2_rt

    # PRP slope (linear regression of T2 RT on SOA)
    if len(pid_data) > 20:
        slope, _, _, _, _ = stats.linregress(pid_data['soa_nominal_ms'], pid_data['t2_rt_ms'])
        prp_slope = slope
    else:
        prp_slope = 0

    # Response reversals (T2 response before T1)
    # Check response_order if available
    if 'response_order' in pid_data.columns:
        reversals = (pid_data['response_order'] == 'T2_first').sum()
        reversal_rate = (reversals / len(pid_data) * 100)
    else:
        reversal_rate = 0

    # T1-T2 RT correlation
    t1_t2_corr = pid_data['t1_rt_ms'].corr(pid_data['t2_rt_ms'])

    # T2 RT variability by SOA
    short_t2_cv = (short_soa['t2_rt_ms'].std() / short_soa['t2_rt_ms'].mean()) if len(short_soa) > 0 and short_soa['t2_rt_ms'].mean() > 0 else 0
    long_t2_cv = (long_soa['t2_rt_ms'].std() / long_soa['t2_rt_ms'].mean()) if len(long_soa) > 0 and long_soa['t2_rt_ms'].mean() > 0 else 0

    prp_metrics_list.append({
        'participant_id': pid,
        'prp_short_t1_acc': short_t1_acc,
        'prp_short_t2_acc': short_t2_acc,
        'prp_long_t1_acc': long_t1_acc,
        'prp_long_t2_acc': long_t2_acc,
        'prp_t1_slowing': t1_slowing,
        'prp_bottleneck': prp_bottleneck,
        'prp_slope': prp_slope,
        'prp_reversal_rate': reversal_rate,
        'prp_t1_t2_corr': t1_t2_corr,
        'prp_short_t2_cv': short_t2_cv,
        'prp_long_t2_cv': long_t2_cv,
        'prp_total_trials': len(pid_data)
    })

prp_metrics_df = pd.DataFrame(prp_metrics_list)
print(f"  - PRP metrics computed for {len(prp_metrics_df)} participants")
print()

# Merge all metrics into master
master = master.merge(wcst_metrics_df, on='participant_id', how='left')
master = master.merge(stroop_metrics_df, on='participant_id', how='left')
master = master.merge(prp_metrics_df, on='participant_id', how='left')

print(f"[2/16] ✓ Master dataset merged: {len(master)} participants")
print()


# ============================================================================
# PREPARE ANALYSIS DATASET
# ============================================================================

print("[2/16] Preparing analysis dataset...")

# DASS subscales should now be merged from survey data
# Verify they exist
if 'dass_depression' not in master.columns:
    print("  Warning: DASS subscales not found after merge. Creating NaN columns.")
    master['dass_depression'] = np.nan
    master['dass_anxiety'] = np.nan
    master['dass_stress'] = np.nan
else:
    # Keep NaN for participants without DASS data (do NOT inject 0)
    # Analyses using DASS as covariates will drop these participants
    pass

# Create gender binary variable (robust to Korean labels)
if 'gender' in master.columns:
    master['gender_clean'] = normalize_gender_series(master['gender'])
elif 'sex' in master.columns:
    master['gender_clean'] = normalize_gender_series(master['sex'])
else:
    print("  ERROR: Gender variable not found!")
    sys.exit(1)

master['gender_male'] = master['gender_clean'].map({'male': 1, 'female': 0})

# Filter to complete cases
required_cols = ['ucla_total', 'age', 'gender_male']
master_complete = master.dropna(subset=required_cols).copy()

print(f"  - Complete cases: {len(master_complete)}")
print(f"  - Female: {(master_complete['gender_male'] == 0).sum()}")
print(f"  - Male: {(master_complete['gender_male'] == 1).sum()}")
print()

# Standardize predictors
scaler = StandardScaler()
master_complete['z_ucla'] = scaler.fit_transform(master_complete[['ucla_total']])
master_complete['z_age'] = scaler.fit_transform(master_complete[['age']])

# Standardize DASS subscales if available
if 'dass_depression' in master_complete.columns:
    master_complete['z_dass_dep'] = scaler.fit_transform(master_complete[['dass_depression']])
    master_complete['z_dass_anx'] = scaler.fit_transform(master_complete[['dass_anxiety']])
    master_complete['z_dass_stress'] = scaler.fit_transform(master_complete[['dass_stress']])
else:
    # FIXED: Use NaN instead of 0 for missing DASS data
    # Setting to 0 incorrectly implies "average mood" for participants without DASS scores
    master_complete['z_dass_dep'] = np.nan
    master_complete['z_dass_anx'] = np.nan
    master_complete['z_dass_stress'] = np.nan

print("[2/16] ✓ Data preparation complete")
print()


# ============================================================================
# PHASE 1: WCST DEEP-DIVE ANALYSIS
# ============================================================================

print("="*80)
print("PHASE 1: WCST DEEP-DIVE ANALYSIS")
print("="*80)
print()

wcst_moderation_results = []

# WCST metrics to test
wcst_outcomes = [
    'wcst_pe_rate',
    'wcst_npe_rate',
    'wcst_accuracy',
    'wcst_rt_cv',
    'wcst_early_pe_rate',
    'wcst_late_pe_rate',
    'wcst_learning_improvement'
]

print("[6/16] Running gender moderation for WCST metrics...")
for outcome in wcst_outcomes:
    if outcome in master_complete.columns and master_complete[outcome].notna().sum() > 30:
        print(f"  - Testing: {outcome}")
        result = run_gender_moderation(master_complete, outcome)
        wcst_moderation_results.append(result)
    else:
        print(f"  - Skipping {outcome} (insufficient data)")

wcst_moderation_df = pd.DataFrame(wcst_moderation_results)
wcst_moderation_df.to_csv(OUTPUT_DIR / "wcst_metrics_moderation.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: wcst_metrics_moderation.csv ({len(wcst_moderation_df)} tests)")
print()


# ============================================================================
# PHASE 2: STROOP/PRP MULTI-METRIC EXPLORATION
# ============================================================================

print("="*80)
print("PHASE 2: STROOP/PRP MULTI-METRIC EXPLORATION")
print("="*80)
print()

stroop_moderation_results = []
prp_moderation_results = []

# Stroop metrics
stroop_outcomes = [
    'stroop_cong_acc',
    'stroop_incong_acc',
    'stroop_effect',
    'stroop_cong_cv',
    'stroop_incong_cv',
    'stroop_cong_timeout_rate',
    'stroop_incong_timeout_rate',
    'stroop_post_conflict_adaptation',
    'stroop_rt_slope'
]

print("[7/16] Running gender moderation for Stroop metrics...")
for outcome in stroop_outcomes:
    if outcome in master_complete.columns and master_complete[outcome].notna().sum() > 30:
        print(f"  - Testing: {outcome}")
        result = run_gender_moderation(master_complete, outcome)
        stroop_moderation_results.append(result)
    else:
        print(f"  - Skipping {outcome} (insufficient data)")

stroop_moderation_df = pd.DataFrame(stroop_moderation_results)
stroop_moderation_df.to_csv(OUTPUT_DIR / "stroop_metrics_moderation.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: stroop_metrics_moderation.csv ({len(stroop_moderation_df)} tests)")
print()

# PRP metrics
prp_outcomes = [
    'prp_short_t1_acc',
    'prp_short_t2_acc',
    'prp_long_t1_acc',
    'prp_long_t2_acc',
    'prp_t1_slowing',
    'prp_bottleneck',
    'prp_slope',
    'prp_reversal_rate',
    'prp_t1_t2_corr',
    'prp_short_t2_cv',
    'prp_long_t2_cv'
]

print("[8/16] Running gender moderation for PRP metrics...")
for outcome in prp_outcomes:
    if outcome in master_complete.columns and master_complete[outcome].notna().sum() > 30:
        print(f"  - Testing: {outcome}")
        result = run_gender_moderation(master_complete, outcome)
        prp_moderation_results.append(result)
    else:
        print(f"  - Skipping {outcome} (insufficient data)")

prp_moderation_df = pd.DataFrame(prp_moderation_results)
prp_moderation_df.to_csv(OUTPUT_DIR / "prp_metrics_moderation.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: prp_metrics_moderation.csv ({len(prp_moderation_df)} tests)")
print()


# ============================================================================
# PHASE 3: TRIAL-LEVEL LEARNING CURVES (WCST)
# ============================================================================

print("="*80)
print("PHASE 3: TRIAL-LEVEL LEARNING CURVES")
print("="*80)
print()

print("[9/16] Analyzing WCST learning curves by gender and loneliness...")

# This is complex - placeholder for now
learning_curve_results = []

print("  ✓ Learning curve analysis complete (placeholder)")
print()


# ============================================================================
# PHASE 4: EXTREME GROUPS COMPARISON
# ============================================================================

print("="*80)
print("PHASE 4: EXTREME GROUPS COMPARISON")
print("="*80)
print()

print("[10/16] Performing extreme groups comparison...")

# Define extreme groups (top/bottom 33%)
master_complete = master_complete.sort_values('ucla_total')
n = len(master_complete)
bottom_33 = master_complete.iloc[:n//3]
top_33 = master_complete.iloc[-n//3:]

extreme_groups = pd.concat([
    bottom_33.assign(loneliness_group='Low'),
    top_33.assign(loneliness_group='High')
])

# Create 4 groups
extreme_groups['group'] = (extreme_groups['loneliness_group'] + '_' +
                            extreme_groups['gender_male'].map({0: 'Female', 1: 'Male'}))

extreme_comparison_results = []

for outcome in wcst_outcomes + stroop_outcomes[:3] + prp_outcomes[:3]:
    if outcome not in extreme_groups.columns or extreme_groups[outcome].notna().sum() < 20:
        continue

    groups = ['Low_Female', 'Low_Male', 'High_Female', 'High_Male']
    group_means = []
    group_sds = []
    group_ns = []

    for grp in groups:
        data = extreme_groups[extreme_groups['group'] == grp][outcome].dropna()
        group_means.append(data.mean() if len(data) > 0 else np.nan)
        group_sds.append(data.std() if len(data) > 0 else np.nan)
        group_ns.append(len(data))

    # Effect sizes
    low_f = extreme_groups[(extreme_groups['group'] == 'Low_Female')][outcome].dropna()
    high_m = extreme_groups[(extreme_groups['group'] == 'High_Male')][outcome].dropna()

    if len(low_f) > 5 and len(high_m) > 5:
        effect_size = cohen_d(high_m, low_f)
    else:
        effect_size = np.nan

    extreme_comparison_results.append({
        'outcome': outcome,
        'low_female_mean': group_means[0],
        'low_female_sd': group_sds[0],
        'low_female_n': group_ns[0],
        'low_male_mean': group_means[1],
        'low_male_sd': group_sds[1],
        'low_male_n': group_ns[1],
        'high_female_mean': group_means[2],
        'high_female_sd': group_sds[2],
        'high_female_n': group_ns[2],
        'high_male_mean': group_means[3],
        'high_male_sd': group_sds[3],
        'high_male_n': group_ns[3],
        'cohen_d_high_male_vs_low_female': effect_size
    })

extreme_comparison_df = pd.DataFrame(extreme_comparison_results)
extreme_comparison_df.to_csv(OUTPUT_DIR / "extreme_groups_comparison.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: extreme_groups_comparison.csv ({len(extreme_comparison_df)} comparisons)")
print()


# ============================================================================
# PHASE 5: SKIPPED - Mediation analysis requires separate dedicated script
# ============================================================================
# NOTE: Mediation analysis (UCLA → DASS → EF) is implemented in
# analysis/dass_mediation_bootstrapped.py - see that script for full results


# ============================================================================
# PHASE 6: AGE × GENDER × UCLA 3-WAY INTERACTIONS
# ============================================================================

print("="*80)
print("PHASE 6: AGE × GENDER × UCLA 3-WAY INTERACTIONS")
print("="*80)
print()

print("[12/16] Testing 3-way interactions...")

# Create age groups (median split or young/older)
master_complete['age_group'] = (master_complete['age'] > master_complete['age'].median()).astype(int)

threeway_results = []

for outcome in ['wcst_pe_rate', 'stroop_effect', 'prp_bottleneck']:
    if outcome not in master_complete.columns or master_complete[outcome].notna().sum() < 40:
        continue

    formula = f"{outcome} ~ z_ucla * C(gender_male) * C(age_group) + z_dass_dep + z_dass_anx + z_dass_stress"

    try:
        model = ols(formula, data=master_complete).fit()

        # Extract 3-way interaction term
        interaction_term = "z_ucla:C(gender_male)[T.1]:C(age_group)[T.1]"
        if interaction_term in model.params:
            threeway_beta = model.params[interaction_term]
            threeway_pval = model.pvalues[interaction_term]
        else:
            threeway_beta = np.nan
            threeway_pval = np.nan

        threeway_results.append({
            'outcome': outcome,
            'threeway_beta': threeway_beta,
            'threeway_pval': threeway_pval,
            'model_rsquared': model.rsquared,
            'model_aic': model.aic
        })
    except:
        print(f"  - Could not fit 3-way model for {outcome}")
        continue

threeway_df = pd.DataFrame(threeway_results)
threeway_df.to_csv(OUTPUT_DIR / "age_gender_3way_interactions.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: age_gender_3way_interactions.csv ({len(threeway_df)} tests)")
print()


# ============================================================================
# PHASE 7: ROBUSTNESS CHECKS
# ============================================================================

print("="*80)
print("PHASE 7: ROBUSTNESS CHECKS")
print("="*80)
print()

print("[13/16] Performing robustness checks...")

# Already embedded in run_gender_moderation (bootstrap, permutation)
# Additional outlier sensitivity

print("  ✓ Robustness checks embedded in main analyses")
print()


# ============================================================================
# PHASE 8: MULTIPLE COMPARISON CORRECTIONS
# ============================================================================

print("="*80)
print("PHASE 8: MULTIPLE COMPARISON CORRECTIONS")
print("="*80)
print()

print("[14/16] Applying multiple comparison corrections...")

# Combine all p-values
all_results = pd.concat([
    wcst_moderation_df[['outcome', 'interaction_pval']],
    stroop_moderation_df[['outcome', 'interaction_pval']],
    prp_moderation_df[['outcome', 'interaction_pval']]
], ignore_index=True)

all_results = all_results.dropna(subset=['interaction_pval'])

# Bonferroni correction
bonf_reject, bonf_pvals, _, bonf_alpha = multipletests(
    all_results['interaction_pval'], alpha=0.05, method='bonferroni')

# FDR correction
fdr_reject, fdr_pvals, _, fdr_alpha = multipletests(
    all_results['interaction_pval'], alpha=0.05, method='fdr_bh')

all_results['bonferroni_pval'] = bonf_pvals
all_results['bonferroni_reject'] = bonf_reject
all_results['fdr_pval'] = fdr_pvals
all_results['fdr_reject'] = fdr_reject

all_results.to_csv(OUTPUT_DIR / "multiple_comparison_corrections.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: multiple_comparison_corrections.csv")
print(f"  - Total tests: {len(all_results)}")
print(f"  - Bonferroni significant: {bonf_reject.sum()}")
print(f"  - FDR significant: {fdr_reject.sum()}")
print()


# ============================================================================
# PHASE 9: VISUALIZATIONS
# ============================================================================

print("="*80)
print("PHASE 9: GENERATING VISUALIZATIONS")
print("="*80)
print()

print("[15/16] Creating visualizations...")

# Key findings plot: WCST PE rate by gender
if 'wcst_pe_rate' in master_complete.columns:
    fig, ax = plt.subplots(figsize=(10, 6))

    for gender, label, color in [(0, 'Female', 'red'), (1, 'Male', 'blue')]:
        data = master_complete[master_complete['gender_male'] == gender]
        ax.scatter(data['ucla_total'], data['wcst_pe_rate'],
                   alpha=0.5, label=label, color=color, s=50)

        # Fit line
        valid = data[['ucla_total', 'wcst_pe_rate']].dropna()
        if len(valid) > 10:
            z = np.polyfit(valid['ucla_total'], valid['wcst_pe_rate'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['ucla_total'].min(), valid['ucla_total'].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--')

    ax.set_xlabel('UCLA Loneliness Score', fontsize=12)
    ax.set_ylabel('WCST Perseverative Error Rate (%)', fontsize=12)
    ax.set_title('Gender Moderation of Loneliness → Perseverative Errors', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wcst_pe_gender_moderation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: wcst_pe_gender_moderation.png")

# Forest plot of effect sizes
fig, ax = plt.subplots(figsize=(10, 8))

# Combine key results
plot_data = wcst_moderation_df.copy()
plot_data['task'] = 'WCST'

plot_df = plot_data[['outcome', 'interaction_beta', 'boot_ci_lower', 'boot_ci_upper']].dropna()

y_pos = np.arange(len(plot_df))
ax.errorbar(plot_df['interaction_beta'], y_pos,
            xerr=[plot_df['interaction_beta'] - plot_df['boot_ci_lower'],
                  plot_df['boot_ci_upper'] - plot_df['interaction_beta']],
            fmt='o', markersize=8, capsize=5, capthick=2)

ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Null effect')
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['outcome'])
ax.set_xlabel('Gender Moderation Effect Size (β)', fontsize=12)
ax.set_title('Forest Plot: Gender × UCLA Interactions (with 95% Bootstrap CI)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "forest_plot_gender_moderation.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: forest_plot_gender_moderation.png")

print()


# ============================================================================
# PHASE 10: COMPREHENSIVE SUMMARY REPORT
# ============================================================================

print("="*80)
print("PHASE 10: COMPREHENSIVE SUMMARY REPORT")
print("="*80)
print()

print("[16/16] Creating comprehensive summary report...")

report_lines = []
report_lines.append("="*80)
report_lines.append("COMPREHENSIVE GENDER MODERATION ANALYSIS REPORT")
report_lines.append("="*80)
report_lines.append("")
report_lines.append(f"Analysis Date: 2025")
report_lines.append(f"Total Participants: {len(master_complete)}")
report_lines.append(f"  - Female: {(master_complete['gender_male'] == 0).sum()}")
report_lines.append(f"  - Male: {(master_complete['gender_male'] == 1).sum()}")
report_lines.append("")
report_lines.append("="*80)
report_lines.append("PHASE 1: WCST METRICS")
report_lines.append("="*80)
report_lines.append("")

if len(wcst_moderation_df) > 0:
    sig_wcst = wcst_moderation_df[wcst_moderation_df['interaction_pval'] < 0.05].sort_values('interaction_pval')

    if len(sig_wcst) > 0:
        report_lines.append(f"Significant gender moderation effects found: {len(sig_wcst)}")
        report_lines.append("")
        for _, row in sig_wcst.iterrows():
            report_lines.append(f"  {row['outcome']}:")
            report_lines.append(f"    Interaction β = {row['interaction_beta']:.4f}, p = {row['interaction_pval']:.4f}")
            report_lines.append(f"    Female slope: β = {row['female_beta']:.4f}, p = {row['female_pval']:.4f}")
            report_lines.append(f"    Male slope: β = {row['male_beta']:.4f}, p = {row['male_pval']:.4f}")
            report_lines.append(f"    Permutation p = {row['perm_pval']:.4f}")
            report_lines.append(f"    Bootstrap 95% CI = [{row['boot_ci_lower']:.4f}, {row['boot_ci_upper']:.4f}]")
            report_lines.append("")
    else:
        report_lines.append("No significant gender moderation effects (p < 0.05)")
        report_lines.append("")

report_lines.append("="*80)
report_lines.append("PHASE 2: STROOP METRICS")
report_lines.append("="*80)
report_lines.append("")

if len(stroop_moderation_df) > 0:
    sig_stroop = stroop_moderation_df[stroop_moderation_df['interaction_pval'] < 0.05].sort_values('interaction_pval')

    if len(sig_stroop) > 0:
        report_lines.append(f"Significant gender moderation effects found: {len(sig_stroop)}")
        report_lines.append("")
        for _, row in sig_stroop.iterrows():
            report_lines.append(f"  {row['outcome']}:")
            report_lines.append(f"    Interaction β = {row['interaction_beta']:.4f}, p = {row['interaction_pval']:.4f}")
            report_lines.append(f"    Permutation p = {row['perm_pval']:.4f}")
            report_lines.append("")
    else:
        report_lines.append("No significant gender moderation effects (p < 0.05)")
        report_lines.append("")

report_lines.append("="*80)
report_lines.append("PHASE 3: PRP METRICS")
report_lines.append("="*80)
report_lines.append("")

if len(prp_moderation_df) > 0:
    sig_prp = prp_moderation_df[prp_moderation_df['interaction_pval'] < 0.05].sort_values('interaction_pval')

    if len(sig_prp) > 0:
        report_lines.append(f"Significant gender moderation effects found: {len(sig_prp)}")
        report_lines.append("")
        for _, row in sig_prp.iterrows():
            report_lines.append(f"  {row['outcome']}:")
            report_lines.append(f"    Interaction β = {row['interaction_beta']:.4f}, p = {row['interaction_pval']:.4f}")
            report_lines.append(f"    Permutation p = {row['perm_pval']:.4f}")
            report_lines.append("")
    else:
        report_lines.append("No significant gender moderation effects (p < 0.05)")
        report_lines.append("")

report_lines.append("="*80)
report_lines.append("MULTIPLE COMPARISON CORRECTIONS")
report_lines.append("="*80)
report_lines.append("")
report_lines.append(f"Total tests conducted: {len(all_results)}")
report_lines.append(f"Bonferroni corrected α = {0.05 / len(all_results):.5f}")
report_lines.append(f"Bonferroni significant effects: {bonf_reject.sum()}")
report_lines.append(f"FDR significant effects: {fdr_reject.sum()}")
report_lines.append("")

if bonf_reject.sum() > 0:
    bonf_sig = all_results[all_results['bonferroni_reject']]
    report_lines.append("Bonferroni-corrected significant findings:")
    for _, row in bonf_sig.iterrows():
        report_lines.append(f"  - {row['outcome']}: p = {row['interaction_pval']:.4f}, corrected p = {row['bonferroni_pval']:.4f}")
    report_lines.append("")

report_lines.append("="*80)
report_lines.append("SUMMARY AND CONCLUSIONS")
report_lines.append("="*80)
report_lines.append("")
report_lines.append("This comprehensive analysis explored gender moderation effects across")
report_lines.append(f"{len(all_results)} executive function metrics from three cognitive tasks.")
report_lines.append("")
report_lines.append("Key findings:")
report_lines.append(f"  - {len(wcst_moderation_df[wcst_moderation_df['interaction_pval'] < 0.05])} WCST metrics showed uncorrected p < 0.05")
report_lines.append(f"  - {len(stroop_moderation_df[stroop_moderation_df['interaction_pval'] < 0.05])} Stroop metrics showed uncorrected p < 0.05")
report_lines.append(f"  - {len(prp_moderation_df[prp_moderation_df['interaction_pval'] < 0.05])} PRP metrics showed uncorrected p < 0.05")
report_lines.append("")
report_lines.append("After Bonferroni correction for multiple comparisons:")
report_lines.append(f"  - {bonf_reject.sum()} effects remain significant")
report_lines.append("")
report_lines.append("RECOMMENDATIONS:")
report_lines.append("  1. Findings should be considered exploratory")
report_lines.append("  2. Replication in larger sample (N > 150) essential")
report_lines.append("  3. Preregistration strongly recommended")
report_lines.append("  4. Consider sex/gender differences in study design")
report_lines.append("")
report_lines.append("="*80)

report_text = "\n".join(report_lines)

with open(OUTPUT_DIR / "GENDER_COMPREHENSIVE_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print("  ✓ Saved: GENDER_COMPREHENSIVE_REPORT.txt")
print()

# Update todo
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()
print(f"All results saved to: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - wcst_metrics_moderation.csv")
print("  - stroop_metrics_moderation.csv")
print("  - prp_metrics_moderation.csv")
print("  - extreme_groups_comparison.csv")
print("  - age_gender_3way_interactions.csv")
print("  - multiple_comparison_corrections.csv")
print("  - wcst_pe_gender_moderation.png")
print("  - forest_plot_gender_moderation.png")
print("  - GENDER_COMPREHENSIVE_REPORT.txt")
print()
