#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ultra-deep analysis of weak signals in loneliness-EF relationship
외로움-EF 관계에서 미약한 신호를 정밀하게 탐지
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("ULTRA-DEEP SIGNAL ANALYSIS: Loneliness-EF Weak Signal Detection")
print("외로움-EF 관계 미약 신호 정밀 탐지")
print("=" * 80)

# =============================================================================
# 1. 데이터 로딩 및 전처리
# =============================================================================

# Load base data
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
cognitive = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv")

# Normalize column names
participants = participants.rename(columns={'participantId': 'participant_id'})
surveys = surveys.rename(columns={'participantId': 'participant_id'})
cognitive = cognitive.rename(columns={'participantId': 'participant_id'})

# Extract UCLA
ucla_df = surveys[surveys['surveyName'] == 'UCLA Loneliness Scale'].copy()
ucla_df = ucla_df.groupby('participant_id')['score'].first().reset_index()
ucla_df.columns = ['participant_id', 'ucla_total']

# Extract DASS-21
dass_df = surveys[surveys['surveyName'].str.contains('DASS', na=False)].copy()
dass_pivot = dass_df.pivot_table(
    index='participant_id',
    columns='surveyName',
    values='score',
    aggfunc='first'
).reset_index()

dass_pivot.columns.name = None
if 'DASS-21 불안' in dass_pivot.columns:
    dass_pivot = dass_pivot.rename(columns={
        'DASS-21 우울': 'dass_depression',
        'DASS-21 불안': 'dass_anxiety',
        'DASS-21 스트레스': 'dass_stress'
    })

# Merge
master = participants[['participant_id', 'age', 'gender']].copy()
master = master.merge(ucla_df, on='participant_id', how='inner')
master = master.merge(dass_pivot, on='participant_id', how='left')
master = master.merge(cognitive, on='participant_id', how='inner')

# Check if DASS columns exist, if not create them from surveys
if 'dass_depression' not in master.columns:
    # Extract DASS scores differently
    dass_depression = surveys[surveys['surveyName'].str.contains('우울', na=False)].copy()
    dass_anxiety = surveys[surveys['surveyName'].str.contains('불안', na=False)].copy()
    dass_stress = surveys[surveys['surveyName'].str.contains('스트레스', na=False)].copy()

    if len(dass_depression) > 0:
        dass_dep_df = dass_depression.groupby('participant_id')['score'].first().reset_index()
        dass_dep_df.columns = ['participant_id', 'dass_depression']
        master = master.merge(dass_dep_df, on='participant_id', how='left')

    if len(dass_anxiety) > 0:
        dass_anx_df = dass_anxiety.groupby('participant_id')['score'].first().reset_index()
        dass_anx_df.columns = ['participant_id', 'dass_anxiety']
        master = master.merge(dass_anx_df, on='participant_id', how='left')

    if len(dass_stress) > 0:
        dass_str_df = dass_stress.groupby('participant_id')['score'].first().reset_index()
        dass_str_df.columns = ['participant_id', 'dass_stress']
        master = master.merge(dass_str_df, on='participant_id', how='left')

# Convert gender to binary
master['gender_male'] = (master['gender'] == '남성').astype(int)

# Get Stroop data from cognitive summary
stroop_data = cognitive[cognitive['testName'] == 'stroop'][
    ['participant_id', 'mrt_incong', 'mrt_cong', 'stroop_effect']
].copy()

# Compute Stroop interference
stroop_data['stroop_interference'] = stroop_data['mrt_incong'] - stroop_data['mrt_cong']
# Use stroop_effect if available, otherwise use computed interference
stroop_data['stroop_interference'] = stroop_data['stroop_effect'].fillna(
    stroop_data['stroop_interference']
)

# Merge Stroop data
master = master.merge(
    stroop_data[['participant_id', 'stroop_interference']],
    on='participant_id',
    how='left'
)

# WCST perseverative errors from trial data
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
# Handle duplicate columns
if 'participantId' in wcst_trials.columns and 'participant_id' in wcst_trials.columns:
    wcst_trials = wcst_trials.drop(columns=['participantId'])
elif 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

import ast
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_dict'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials['isPE'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

wcst_pe = wcst_trials.groupby('participant_id').agg({
    'isPE': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else np.nan,
    'trialIndex': 'count'
}).reset_index()
wcst_pe.columns = ['participant_id', 'perseverative_error_rate', 'wcst_n_trials']

master = master.merge(wcst_pe[['participant_id', 'perseverative_error_rate']],
                     on='participant_id', how='left')

# PRP bottleneck
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")
# Handle duplicate columns
if 'participantId' in prp_trials.columns and 'participant_id' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participantId'])
elif 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.rename(columns={'participantId': 'participant_id'})

prp_trials = prp_trials[
    (prp_trials['t1_timeout'] == False) &
    (prp_trials['t2_timeout'] == False) &
    (prp_trials['t2_rt_ms'] > 0)
].copy()

# Use soa_nominal_ms if available, otherwise soa
soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in prp_trials.columns else 'soa'
prp_trials['soa_bin'] = pd.cut(
    prp_trials[soa_col],
    bins=[0, 150, 600, 1200, 3000],
    labels=['short', 'medium', 'long', 'very_long']
)

prp_soa = prp_trials.groupby(['participant_id', 'soa_bin'])['t2_rt_ms'].mean().unstack()
if 'short' in prp_soa.columns and 'long' in prp_soa.columns:
    prp_soa['prp_bottleneck'] = prp_soa['short'] - prp_soa['long']
    master = master.merge(prp_soa[['prp_bottleneck']], on='participant_id', how='left')

# Remove missing
analysis_cols = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
                'stroop_interference', 'perseverative_error_rate', 'prp_bottleneck',
                'age', 'gender_male']
master_clean = master.dropna(subset=analysis_cols).copy()

print(f"\n최종 분석 샘플: N = {len(master_clean)}")
print(f"UCLA 범위: {master_clean['ucla_total'].min():.1f} - {master_clean['ucla_total'].max():.1f}")
print(f"평균 UCLA: {master_clean['ucla_total'].mean():.2f} ± {master_clean['ucla_total'].std():.2f}")

# =============================================================================
# 2. DATA QUALITY CHECKS - 데이터 품질 검사
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 1: DATA QUALITY CHECKS")
print("=" * 80)

quality_issues = []

# Check for negative/implausible Stroop interference
neg_stroop = master_clean[master_clean['stroop_interference'] < 0]
print(f"\n음수 Stroop interference: {len(neg_stroop)} / {len(master_clean)} ({len(neg_stroop)/len(master_clean)*100:.1f}%)")
if len(neg_stroop) > 0:
    print(f"  범위: {neg_stroop['stroop_interference'].min():.1f} ~ {neg_stroop['stroop_interference'].max():.1f} ms")
    quality_issues.append({
        'issue': 'Negative Stroop interference',
        'n_cases': len(neg_stroop),
        'percentage': len(neg_stroop)/len(master_clean)*100,
        'min_value': neg_stroop['stroop_interference'].min()
    })

# Check for extreme outliers (>3 SD)
for col in ['stroop_interference', 'perseverative_error_rate', 'prp_bottleneck']:
    mean_val = master_clean[col].mean()
    std_val = master_clean[col].std()
    outliers = master_clean[(master_clean[col] < mean_val - 3*std_val) |
                            (master_clean[col] > mean_val + 3*std_val)]
    if len(outliers) > 0:
        print(f"\n{col} 극단값 (>3 SD): {len(outliers)} cases")
        print(f"  값: {outliers[col].tolist()}")
        quality_issues.append({
            'issue': f'{col} extreme outliers',
            'n_cases': len(outliers),
            'values': outliers[col].tolist()
        })

# Check reliability (split-half for trial-level tasks)
print("\n\n내적 신뢰도 추정 (split-half reliability):")

# Stroop split-half
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
# Handle duplicate columns
if 'participantId' in stroop_trials.columns and 'participant_id' in stroop_trials.columns:
    stroop_trials = stroop_trials.drop(columns=['participantId'])
elif 'participantId' in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})
stroop_trials = stroop_trials[
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 0)
].copy()

stroop_split = []
for pid in master_clean['participant_id'].unique():
    p_trials = stroop_trials[stroop_trials['participant_id'] == pid].copy()
    if len(p_trials) < 20:
        continue

    p_trials = p_trials.sort_values('trialIndex').reset_index(drop=True)
    p_trials['half'] = ['first' if i % 2 == 0 else 'second'
                        for i in range(len(p_trials))]

    for half in ['first', 'second']:
        half_data = p_trials[p_trials['half'] == half]
        cong_rt = half_data[half_data['condition'] == 'congruent']['rt_ms'].mean()
        incong_rt = half_data[half_data['condition'] == 'incongruent']['rt_ms'].mean()
        if pd.notna(cong_rt) and pd.notna(incong_rt):
            stroop_split.append({
                'participant_id': pid,
                'half': half,
                'interference': incong_rt - cong_rt
            })

if len(stroop_split) > 0:
    stroop_split_df = pd.DataFrame(stroop_split)
    stroop_wide = stroop_split_df.pivot(index='participant_id',
                                        columns='half',
                                        values='interference')
    if 'first' in stroop_wide.columns and 'second' in stroop_wide.columns:
        stroop_wide_clean = stroop_wide.dropna()
        if len(stroop_wide_clean) > 5:
            r_split, p_split = pearsonr(stroop_wide_clean['first'],
                                       stroop_wide_clean['second'])
            # Spearman-Brown correction
            reliability = 2 * r_split / (1 + r_split)
            print(f"  Stroop interference: r_split = {r_split:.3f}, "
                  f"Reliability (SB corrected) = {reliability:.3f}")

# =============================================================================
# 3. STRATIFIED ANALYSES - 층화 분석
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 2: STRATIFIED ANALYSES (Gender, Age)")
print("=" * 80)

# Gender-stratified correlations
print("\n성별별 층화 상관분석:")
print("-" * 80)

gender_results = []

for gender_val, gender_label in [(0, '여성'), (1, '남성')]:
    subset = master_clean[master_clean['gender_male'] == gender_val]
    print(f"\n{gender_label} (N={len(subset)}):")

    for ef_var, ef_label in [
        ('stroop_interference', 'Stroop 간섭'),
        ('perseverative_error_rate', 'WCST 보속오류율'),
        ('prp_bottleneck', 'PRP 병목효과')
    ]:
        if ef_var in subset.columns:
            # Pearson correlation
            r, p = pearsonr(subset['ucla_total'], subset[ef_var])
            # Spearman correlation
            rho, p_spear = spearmanr(subset['ucla_total'], subset[ef_var])

            print(f"  {ef_label}: r={r:.3f}, p={p:.3f} | rho={rho:.3f}, p={p_spear:.3f}")

            gender_results.append({
                'gender': gender_label,
                'n': len(subset),
                'outcome': ef_label,
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spear
            })

gender_df = pd.DataFrame(gender_results)
gender_df.to_csv(OUTPUT_DIR / "ultradeep_gender_stratified.csv",
                 index=False, encoding='utf-8-sig')

# Age-stratified (median split)
print("\n\n연령별 층화 분석 (중앙값 분할):")
print("-" * 80)

age_median = master_clean['age'].median()
print(f"연령 중앙값: {age_median}")

age_results = []

for age_group, age_label in [('young', f'≤{age_median}세'), ('old', f'>{age_median}세')]:
    if age_group == 'young':
        subset = master_clean[master_clean['age'] <= age_median]
    else:
        subset = master_clean[master_clean['age'] > age_median]

    print(f"\n{age_label} (N={len(subset)}, 연령범위: {subset['age'].min()}-{subset['age'].max()}):")

    for ef_var, ef_label in [
        ('stroop_interference', 'Stroop 간섭'),
        ('perseverative_error_rate', 'WCST 보속오류율'),
        ('prp_bottleneck', 'PRP 병목효과')
    ]:
        r, p = pearsonr(subset['ucla_total'], subset[ef_var])
        rho, p_spear = spearmanr(subset['ucla_total'], subset[ef_var])

        print(f"  {ef_label}: r={r:.3f}, p={p:.3f} | rho={rho:.3f}, p={p_spear:.3f}")

        age_results.append({
            'age_group': age_label,
            'n': len(subset),
            'age_range': f"{subset['age'].min()}-{subset['age'].max()}",
            'outcome': ef_label,
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spear
        })

age_df = pd.DataFrame(age_results)
age_df.to_csv(OUTPUT_DIR / "ultradeep_age_stratified.csv",
              index=False, encoding='utf-8-sig')

# =============================================================================
# 4. NON-LINEAR RELATIONSHIP TESTS - 비선형 관계 검정
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 3: NON-LINEAR RELATIONSHIP TESTS")
print("=" * 80)

nonlinear_results = []

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop 간섭'),
    ('perseverative_error_rate', 'WCST 보속오류율'),
    ('prp_bottleneck', 'PRP 병목효과')
]:
    print(f"\n{ef_label}:")

    # Create quadratic term
    master_clean['ucla_squared'] = master_clean['ucla_total'] ** 2
    master_clean['ucla_centered'] = master_clean['ucla_total'] - master_clean['ucla_total'].mean()
    master_clean['ucla_centered_sq'] = master_clean['ucla_centered'] ** 2

    # Linear model
    formula_linear = f"{ef_var} ~ ucla_centered + age + gender_male + dass_depression + dass_anxiety + dass_stress"
    model_linear = smf.ols(formula_linear, data=master_clean).fit()

    # Quadratic model
    formula_quad = f"{ef_var} ~ ucla_centered + ucla_centered_sq + age + gender_male + dass_depression + dass_anxiety + dass_stress"
    model_quad = smf.ols(formula_quad, data=master_clean).fit()

    # Compare models
    lr_stat = 2 * (model_quad.llf - model_linear.llf)
    lr_p = stats.chi2.sf(lr_stat, 1)

    quad_coef = model_quad.params.get('ucla_centered_sq', np.nan)
    quad_p = model_quad.pvalues.get('ucla_centered_sq', np.nan)

    print(f"  선형 모델 R²: {model_linear.rsquared:.4f}")
    print(f"  2차 모델 R²: {model_quad.rsquared:.4f}")
    print(f"  2차 항 계수: {quad_coef:.6f}, p={quad_p:.4f}")
    print(f"  LR test: χ²={lr_stat:.4f}, p={lr_p:.4f}")

    # Check if quadratic improves fit
    if lr_p < 0.10:
        print(f"  *** 2차 관계 가능성 (p < 0.10) ***")

    nonlinear_results.append({
        'outcome': ef_label,
        'linear_r2': model_linear.rsquared,
        'quadratic_r2': model_quad.rsquared,
        'delta_r2': model_quad.rsquared - model_linear.rsquared,
        'quad_coef': quad_coef,
        'quad_p': quad_p,
        'lr_stat': lr_stat,
        'lr_p': lr_p
    })

nonlinear_df = pd.DataFrame(nonlinear_results)
nonlinear_df.to_csv(OUTPUT_DIR / "ultradeep_nonlinear_tests.csv",
                    index=False, encoding='utf-8-sig')

# =============================================================================
# 5. ROBUST REGRESSION - 로버스트 회귀
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 4: ROBUST REGRESSION (outlier-resistant)")
print("=" * 80)

robust_results = []

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop 간섭'),
    ('perseverative_error_rate', 'WCST 보속오류율'),
    ('prp_bottleneck', 'PRP 병목효과')
]:
    print(f"\n{ef_label}:")

    # OLS regression
    formula = f"{ef_var} ~ ucla_total + age + gender_male + dass_depression + dass_anxiety + dass_stress"
    ols_model = smf.ols(formula, data=master_clean).fit()

    # Robust regression (M-estimator)
    robust_model = smf.rlm(formula, data=master_clean, M=sm.robust.norms.HuberT()).fit()

    ols_coef = ols_model.params['ucla_total']
    ols_p = ols_model.pvalues['ucla_total']

    robust_coef = robust_model.params['ucla_total']
    robust_p = robust_model.pvalues['ucla_total']

    print(f"  OLS:    β={ols_coef:.4f}, p={ols_p:.4f}")
    print(f"  Robust: β={robust_coef:.4f}, p={robust_p:.4f}")
    print(f"  차이:   Δβ={robust_coef - ols_coef:.4f}")

    if abs(robust_coef - ols_coef) / abs(ols_coef) > 0.2:
        print(f"  *** 계수가 20% 이상 변화 - outlier 영향 가능성 ***")

    robust_results.append({
        'outcome': ef_label,
        'ols_coef': ols_coef,
        'ols_p': ols_p,
        'robust_coef': robust_coef,
        'robust_p': robust_p,
        'delta_coef': robust_coef - ols_coef,
        'percent_change': (robust_coef - ols_coef) / abs(ols_coef) * 100 if ols_coef != 0 else np.nan
    })

robust_df = pd.DataFrame(robust_results)
robust_df.to_csv(OUTPUT_DIR / "ultradeep_robust_regression.csv",
                 index=False, encoding='utf-8-sig')

# =============================================================================
# 6. PARTIAL CORRELATION - 편상관
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 5: PARTIAL CORRELATIONS (controlling different covariate sets)")
print("=" * 80)

def partial_corr(df, x, y, covariates):
    """Compute partial correlation between x and y controlling for covariates"""
    # Residualize x
    X_cov = df[covariates].values
    X_cov = sm.add_constant(X_cov)
    model_x = sm.OLS(df[x], X_cov).fit()
    resid_x = model_x.resid

    # Residualize y
    model_y = sm.OLS(df[y], X_cov).fit()
    resid_y = model_y.resid

    # Correlation of residuals
    r, p = pearsonr(resid_x, resid_y)
    return r, p

partial_results = []

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop 간섭'),
    ('perseverative_error_rate', 'WCST 보속오류율'),
    ('prp_bottleneck', 'PRP 병목효과')
]:
    print(f"\n{ef_label}:")

    # Zero-order correlation
    r0, p0 = pearsonr(master_clean['ucla_total'], master_clean[ef_var])
    print(f"  Zero-order: r={r0:.3f}, p={p0:.3f}")

    # Partial out age + gender only
    r1, p1 = partial_corr(master_clean, 'ucla_total', ef_var, ['age', 'gender_male'])
    print(f"  Controlling age+gender: r={r1:.3f}, p={p1:.3f}")

    # Partial out DASS only
    r2, p2 = partial_corr(master_clean, 'ucla_total', ef_var,
                         ['dass_depression', 'dass_anxiety', 'dass_stress'])
    print(f"  Controlling DASS: r={r2:.3f}, p={p2:.3f}")

    # Partial out all
    r3, p3 = partial_corr(master_clean, 'ucla_total', ef_var,
                         ['age', 'gender_male', 'dass_depression', 'dass_anxiety', 'dass_stress'])
    print(f"  Controlling all: r={r3:.3f}, p={p3:.3f}")

    partial_results.append({
        'outcome': ef_label,
        'zero_order_r': r0,
        'zero_order_p': p0,
        'partial_age_gender_r': r1,
        'partial_age_gender_p': p1,
        'partial_dass_r': r2,
        'partial_dass_p': p2,
        'partial_all_r': r3,
        'partial_all_p': p3
    })

partial_df = pd.DataFrame(partial_results)
partial_df.to_csv(OUTPUT_DIR / "ultradeep_partial_correlations.csv",
                  index=False, encoding='utf-8-sig')

# =============================================================================
# 7. SENSITIVITY TO EXTREME GROUPS - 극단 그룹 민감도
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 6: SENSITIVITY TO EXTREME UCLA GROUPS")
print("=" * 80)

# Compare effect sizes across different cutoff strategies
extreme_sensitivity = []

for cutoff_pct in [10, 15, 20, 25, 33]:
    low_cutoff = np.percentile(master_clean['ucla_total'], cutoff_pct)
    high_cutoff = np.percentile(master_clean['ucla_total'], 100 - cutoff_pct)

    low_group = master_clean[master_clean['ucla_total'] <= low_cutoff]
    high_group = master_clean[master_clean['ucla_total'] >= high_cutoff]

    print(f"\n극단 {cutoff_pct}% vs {cutoff_pct}%:")
    print(f"  저외로움: N={len(low_group)}, UCLA={low_group['ucla_total'].mean():.1f}±{low_group['ucla_total'].std():.1f}")
    print(f"  고외로움: N={len(high_group)}, UCLA={high_group['ucla_total'].mean():.1f}±{high_group['ucla_total'].std():.1f}")

    for ef_var, ef_label in [
        ('stroop_interference', 'Stroop'),
        ('perseverative_error_rate', 'WCST'),
        ('prp_bottleneck', 'PRP')
    ]:
        low_mean = low_group[ef_var].mean()
        high_mean = high_group[ef_var].mean()

        # t-test
        t_stat, t_p = stats.ttest_ind(high_group[ef_var], low_group[ef_var])

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(low_group) - 1) * low_group[ef_var].std()**2 +
             (len(high_group) - 1) * high_group[ef_var].std()**2) /
            (len(low_group) + len(high_group) - 2)
        )
        cohens_d = (high_mean - low_mean) / pooled_std

        print(f"  {ef_label}: Δ={high_mean - low_mean:.2f}, d={cohens_d:.3f}, p={t_p:.3f}")

        extreme_sensitivity.append({
            'cutoff_percentile': cutoff_pct,
            'low_n': len(low_group),
            'high_n': len(high_group),
            'outcome': ef_label,
            'low_mean': low_mean,
            'high_mean': high_mean,
            'diff': high_mean - low_mean,
            'cohens_d': cohens_d,
            't_stat': t_stat,
            'p_value': t_p
        })

extreme_sens_df = pd.DataFrame(extreme_sensitivity)
extreme_sens_df.to_csv(OUTPUT_DIR / "ultradeep_extreme_group_sensitivity.csv",
                       index=False, encoding='utf-8-sig')

# =============================================================================
# 8. INTERACTION DETAILS - 상호작용 상세 분석
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 7: DETAILED INTERACTION ANALYSES")
print("=" * 80)

interaction_results = []

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop 간섭'),
    ('perseverative_error_rate', 'WCST 보속오류율'),
    ('prp_bottleneck', 'PRP 병목효과')
]:
    print(f"\n{ef_label}:")

    # UCLA × Gender
    formula_int = f"{ef_var} ~ ucla_total * C(gender_male) + age + dass_depression + dass_anxiety + dass_stress"
    model_int = smf.ols(formula_int, data=master_clean).fit()

    int_coef = model_int.params.get('ucla_total:C(gender_male)[T.1]', np.nan)
    int_p = model_int.pvalues.get('ucla_total:C(gender_male)[T.1]', np.nan)

    print(f"  UCLA × Gender: β={int_coef:.4f}, p={int_p:.4f}")

    # UCLA × Age (continuous)
    master_clean['age_centered'] = master_clean['age'] - master_clean['age'].mean()
    master_clean['ucla_centered'] = master_clean['ucla_total'] - master_clean['ucla_total'].mean()

    formula_age_int = f"{ef_var} ~ ucla_centered * age_centered + gender_male + dass_depression + dass_anxiety + dass_stress"
    model_age_int = smf.ols(formula_age_int, data=master_clean).fit()

    age_int_coef = model_age_int.params.get('ucla_centered:age_centered', np.nan)
    age_int_p = model_age_int.pvalues.get('ucla_centered:age_centered', np.nan)

    print(f"  UCLA × Age: β={age_int_coef:.4f}, p={age_int_p:.4f}")

    # UCLA × DASS (check if depression moderates)
    master_clean['dass_dep_centered'] = master_clean['dass_depression'] - master_clean['dass_depression'].mean()

    formula_dass_int = f"{ef_var} ~ ucla_centered * dass_dep_centered + age + gender_male + dass_anxiety + dass_stress"
    model_dass_int = smf.ols(formula_dass_int, data=master_clean).fit()

    dass_int_coef = model_dass_int.params.get('ucla_centered:dass_dep_centered', np.nan)
    dass_int_p = model_dass_int.pvalues.get('ucla_centered:dass_dep_centered', np.nan)

    print(f"  UCLA × DASS_dep: β={dass_int_coef:.4f}, p={dass_int_p:.4f}")

    interaction_results.append({
        'outcome': ef_label,
        'gender_int_coef': int_coef,
        'gender_int_p': int_p,
        'age_int_coef': age_int_coef,
        'age_int_p': age_int_p,
        'dass_int_coef': dass_int_coef,
        'dass_int_p': dass_int_p
    })

interaction_df = pd.DataFrame(interaction_results)
interaction_df.to_csv(OUTPUT_DIR / "ultradeep_interactions.csv",
                      index=False, encoding='utf-8-sig')

# =============================================================================
# 9. BOOTSTRAP CONFIDENCE INTERVALS - 부트스트랩 신뢰구간
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 8: BOOTSTRAP CONFIDENCE INTERVALS FOR CORRELATIONS")
print("=" * 80)

from sklearn.utils import resample

def bootstrap_correlation(x, y, n_iterations=5000, alpha=0.05):
    """Bootstrap CI for Pearson correlation"""
    correlations = []
    n = len(x)

    for _ in range(n_iterations):
        indices = resample(range(n), replace=True, n_samples=n)
        r, _ = pearsonr(x.iloc[indices], y.iloc[indices])
        correlations.append(r)

    correlations = np.array(correlations)
    ci_lower = np.percentile(correlations, alpha/2 * 100)
    ci_upper = np.percentile(correlations, (1 - alpha/2) * 100)

    return ci_lower, ci_upper, correlations

bootstrap_results = []

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop 간섭'),
    ('perseverative_error_rate', 'WCST 보속오류율'),
    ('prp_bottleneck', 'PRP 병목효과')
]:
    print(f"\n{ef_label}:")

    r_obs, p_obs = pearsonr(master_clean['ucla_total'], master_clean[ef_var])
    ci_lower, ci_upper, boot_dist = bootstrap_correlation(
        master_clean['ucla_total'],
        master_clean[ef_var],
        n_iterations=5000
    )

    # Proportion of bootstrap samples with r > 0
    prop_positive = np.mean(boot_dist > 0)

    print(f"  관찰 r: {r_obs:.3f}, p={p_obs:.3f}")
    print(f"  Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  양의 상관 비율: {prop_positive:.1%}")

    # Check if CI excludes zero
    if ci_lower > 0 or ci_upper < 0:
        print(f"  *** 95% CI가 0을 제외 - 유의한 효과 ***")

    bootstrap_results.append({
        'outcome': ef_label,
        'r_observed': r_obs,
        'p_value': p_obs,
        'boot_ci_lower': ci_lower,
        'boot_ci_upper': ci_upper,
        'prop_positive': prop_positive,
        'excludes_zero': ci_lower > 0 or ci_upper < 0
    })

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv(OUTPUT_DIR / "ultradeep_bootstrap_ci.csv",
                    index=False, encoding='utf-8-sig')

# =============================================================================
# 10. SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 80)
print("ULTRA-DEEP ANALYSIS SUMMARY")
print("=" * 80)

print("\n주요 발견 요약:")

# Find strongest signals
print("\n1. 가장 강한 신호 (성별별):")
gender_df_sorted = gender_df.sort_values('pearson_p')
for idx in range(min(3, len(gender_df_sorted))):
    row = gender_df_sorted.iloc[idx]
    print(f"   {row['gender']} - {row['outcome']}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.3f}")

print("\n2. 비선형 관계 가능성:")
nonlinear_sig = nonlinear_df[nonlinear_df['lr_p'] < 0.10]
if len(nonlinear_sig) > 0:
    for _, row in nonlinear_sig.iterrows():
        print(f"   {row['outcome']}: 2차 항 p={row['quad_p']:.3f}, LR p={row['lr_p']:.3f}")
else:
    print("   비선형 관계 증거 없음")

print("\n3. 로버스트 회귀에서 큰 변화:")
robust_large = robust_df[abs(robust_df['percent_change']) > 20]
if len(robust_large) > 0:
    for _, row in robust_large.iterrows():
        print(f"   {row['outcome']}: 계수 변화 {row['percent_change']:.1f}% (outlier 영향 가능)")
else:
    print("   Outlier의 큰 영향 없음")

print("\n4. 부트스트랩 CI가 0을 제외하는 경우:")
boot_sig = bootstrap_df[bootstrap_df['excludes_zero']]
if len(boot_sig) > 0:
    for _, row in boot_sig.iterrows():
        print(f"   {row['outcome']}: r={row['r_observed']:.3f}, 95% CI=[{row['boot_ci_lower']:.3f}, {row['boot_ci_upper']:.3f}]")
else:
    print("   없음 - 모든 효과가 0과 양립 가능")

print("\n5. 상호작용 효과 (p < 0.10):")
for _, row in interaction_df.iterrows():
    if row['gender_int_p'] < 0.10:
        print(f"   {row['outcome']} × Gender: β={row['gender_int_coef']:.4f}, p={row['gender_int_p']:.3f}")
    if row['age_int_p'] < 0.10:
        print(f"   {row['outcome']} × Age: β={row['age_int_coef']:.4f}, p={row['age_int_p']:.3f}")
    if row['dass_int_p'] < 0.10:
        print(f"   {row['outcome']} × DASS: β={row['dass_int_coef']:.4f}, p={row['dass_int_p']:.3f}")

print("\n\n분석 완료!")
print(f"결과 파일 저장 위치: {OUTPUT_DIR}")
