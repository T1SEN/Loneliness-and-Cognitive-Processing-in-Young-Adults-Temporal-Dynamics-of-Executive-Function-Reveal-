#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
성별 조절효과 확증 분석
Gender Moderation Confirmatory Analysis

Methods:
- Simple slopes analysis
- Johnson-Neyman intervals
- Permutation test
- Bootstrap confidence intervals
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("성별 조절효과 확증 분석")
print("Gender Moderation Confirmatory Analysis")
print("=" * 80)

# =============================================================================
# 1. 데이터 로딩
# =============================================================================

master = pd.read_csv(OUTPUT_DIR / "master_dataset.csv")
master = master.rename(columns={'pe_rate': 'perseverative_error_rate'})

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

master['gender_male'] = (master['gender'] == '남성').astype(int)
master_clean = master.dropna().copy()

# Standardize predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
master_clean['z_ucla'] = scaler.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_stress'] = scaler.fit_transform(master_clean[['dass_stress']])
master_clean['z_age'] = scaler.fit_transform(master_clean[['age']])

print(f"\n샘플 크기: N = {len(master_clean)}")
print(f"남성: {master_clean['gender_male'].sum()}명 ({master_clean['gender_male'].sum()/len(master_clean)*100:.1f}%)")
print(f"여성: {(1-master_clean['gender_male']).sum()}명 ({(1-master_clean['gender_male']).sum()/len(master_clean)*100:.1f}%)")

# =============================================================================
# 2. SIMPLE SLOPES ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("1. SIMPLE SLOPES ANALYSIS")
print("=" * 80)

results_simple_slopes = []

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop Interference'),
    ('perseverative_error_rate', 'WCST Perseverative Error Rate'),
    ('prp_bottleneck', 'PRP Bottleneck Effect')
]:
    print(f"\n{ef_label}:")
    print("-" * 80)

    # Interaction model
    formula = f"{ef_var} ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
    model = smf.ols(formula, data=master_clean).fit(cov_type='HC3')  # Robust SE

    # Extract coefficients
    beta_main = model.params.get('z_ucla', 0)
    beta_interaction = model.params.get('z_ucla:C(gender_male)[T.1]', 0)

    # Simple slopes
    beta_female = beta_main  # Reference category
    beta_male = beta_main + beta_interaction

    # Standard errors
    se_female = model.bse.get('z_ucla', 0)
    # For male slope, need delta method - approximate with interaction SE
    se_male = np.sqrt(model.bse.get('z_ucla', 0)**2 +
                      model.bse.get('z_ucla:C(gender_male)[T.1]', 0)**2 +
                      2 * model.cov_params().loc['z_ucla', 'z_ucla:C(gender_male)[T.1]'])

    # p-values
    t_female = beta_female / se_female if se_female > 0 else 0
    p_female = 2 * (1 - stats.t.cdf(abs(t_female), df=model.df_resid))

    t_male = beta_male / se_male if se_male > 0 else 0
    p_male = 2 * (1 - stats.t.cdf(abs(t_male), df=model.df_resid))

    print(f"\n여성 (Female):")
    print(f"  β = {beta_female:.4f}, SE = {se_female:.4f}, p = {p_female:.4f}")

    print(f"\n남성 (Male):")
    print(f"  β = {beta_male:.4f}, SE = {se_male:.4f}, p = {p_male:.4f}")

    print(f"\n상호작용 (Interaction):")
    p_interaction = model.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)
    print(f"  β = {beta_interaction:.4f}, p = {p_interaction:.4f}")

    if p_interaction < 0.05:
        print(f"  *** 유의한 상호작용 (p < 0.05) ***")
    elif p_interaction < 0.10:
        print(f"  ** Marginal 상호작용 (p < 0.10) **")

    results_simple_slopes.append({
        'outcome': ef_label,
        'beta_female': beta_female,
        'se_female': se_female,
        'p_female': p_female,
        'beta_male': beta_male,
        'se_male': se_male,
        'p_male': p_male,
        'beta_interaction': beta_interaction,
        'p_interaction': p_interaction
    })

df_simple_slopes = pd.DataFrame(results_simple_slopes)
df_simple_slopes.to_csv(OUTPUT_DIR / "gender_simple_slopes.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 3. STRATIFIED CORRELATIONS
# =============================================================================

print("\n" + "=" * 80)
print("2. STRATIFIED CORRELATIONS (by Gender)")
print("=" * 80)

strat_corr_results = []

for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
    subset = master_clean[master_clean['gender_male'] == gender_val]
    print(f"\n{gender_label} (N={len(subset)}):")

    for ef_var, ef_label in [
        ('stroop_interference', 'Stroop'),
        ('perseverative_error_rate', 'WCST'),
        ('prp_bottleneck', 'PRP')
    ]:
        r, p = pearsonr(subset['ucla_total'], subset[ef_var])
        print(f"  {ef_label}: r = {r:.3f}, p = {p:.3f}")

        strat_corr_results.append({
            'gender': gender_label,
            'n': len(subset),
            'outcome': ef_label,
            'r': r,
            'p': p
        })

df_strat_corr = pd.DataFrame(strat_corr_results)
df_strat_corr.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv",
                     index=False, encoding='utf-8-sig')

# =============================================================================
# 4. PERMUTATION TEST
# =============================================================================

print("\n" + "=" * 80)
print("3. PERMUTATION TEST FOR INTERACTION")
print("=" * 80)

permutation_results = []

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop'),
    ('perseverative_error_rate', 'WCST'),
    ('prp_bottleneck', 'PRP')
]:
    print(f"\n{ef_label}:")

    # Observed interaction
    formula = f"{ef_var} ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
    model_obs = smf.ols(formula, data=master_clean).fit()
    observed_int = model_obs.params.get('z_ucla:C(gender_male)[T.1]', 0)

    # Permutation null distribution
    null_dist = []
    n_perm = 1000

    print(f"  Running {n_perm} permutations...")

    for i in range(n_perm):
        shuffled = master_clean.copy()
        shuffled['gender_male'] = np.random.permutation(shuffled['gender_male'].values)

        try:
            model_perm = smf.ols(formula, data=shuffled).fit()
            null_int = model_perm.params.get('z_ucla:C(gender_male)[T.1]', 0)
            null_dist.append(null_int)
        except:
            null_dist.append(0)

    null_dist = np.array(null_dist)

    # Two-tailed p-value
    p_perm = np.mean(np.abs(null_dist) >= np.abs(observed_int))

    print(f"  Observed interaction: {observed_int:.4f}")
    print(f"  Null distribution: M = {np.mean(null_dist):.4f}, SD = {np.std(null_dist):.4f}")
    print(f"  Permutation p-value: {p_perm:.4f}")

    if p_perm < 0.05:
        print(f"  *** 유의함 (permutation p < 0.05) ***")
    elif p_perm < 0.10:
        print(f"  ** Marginal (permutation p < 0.10) **")

    permutation_results.append({
        'outcome': ef_label,
        'observed_interaction': observed_int,
        'null_mean': np.mean(null_dist),
        'null_sd': np.std(null_dist),
        'p_permutation': p_perm,
        'n_permutations': n_perm
    })

df_permutation = pd.DataFrame(permutation_results)
df_permutation.to_csv(OUTPUT_DIR / "gender_permutation_test.csv",
                      index=False, encoding='utf-8-sig')

# =============================================================================
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

print("\n" + "=" * 80)
print("4. BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 80)

bootstrap_results = []

def bootstrap_interaction(data, formula, n_boot=5000, alpha=0.05):
    """Bootstrap CI for interaction term"""
    interactions = []

    for _ in range(n_boot):
        sample = data.sample(n=len(data), replace=True)
        try:
            m = smf.ols(formula, data=sample).fit()
            int_coef = m.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
            interactions.append(int_coef)
        except:
            interactions.append(np.nan)

    interactions = np.array([x for x in interactions if not np.isnan(x)])

    ci_lower = np.percentile(interactions, alpha/2 * 100)
    ci_upper = np.percentile(interactions, (1 - alpha/2) * 100)

    return ci_lower, ci_upper, interactions

for ef_var, ef_label in [
    ('stroop_interference', 'Stroop'),
    ('perseverative_error_rate', 'WCST'),
    ('prp_bottleneck', 'PRP')
]:
    print(f"\n{ef_label}:")

    formula = f"{ef_var} ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"

    # Observed
    model_obs = smf.ols(formula, data=master_clean).fit()
    observed = model_obs.params.get('z_ucla:C(gender_male)[T.1]', 0)

    # Bootstrap
    print(f"  Running 5000 bootstrap samples...")
    ci_lower, ci_upper, boot_dist = bootstrap_interaction(master_clean, formula, n_boot=5000)

    # Proportion of bootstrap samples where interaction > 0
    prop_positive = np.mean(boot_dist > 0)

    print(f"  Observed: {observed:.4f}")
    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Proportion positive: {prop_positive:.1%}")

    # Check if CI excludes zero
    excludes_zero = (ci_lower > 0) or (ci_upper < 0)

    if excludes_zero:
        print(f"  *** 95% CI가 0을 제외 - 유의한 효과 ***")

    bootstrap_results.append({
        'outcome': ef_label,
        'observed': observed,
        'boot_ci_lower': ci_lower,
        'boot_ci_upper': ci_upper,
        'prop_positive': prop_positive,
        'excludes_zero': excludes_zero
    })

df_bootstrap = pd.DataFrame(bootstrap_results)
df_bootstrap.to_csv(OUTPUT_DIR / "gender_bootstrap_ci.csv",
                    index=False, encoding='utf-8-sig')

# =============================================================================
# 6. JOHNSON-NEYMAN INTERVALS
# =============================================================================

print("\n" + "=" * 80)
print("5. JOHNSON-NEYMAN TECHNIQUE")
print("=" * 80)
print("(성별은 이분 변수이므로 J-N 적용 불가, 대신 연령으로 시연)")

# For continuous moderator (Age)
jn_results = []

for ef_var, ef_label in [
    ('perseverative_error_rate', 'WCST'),  # Focus on WCST where interaction is significant
]:
    print(f"\n{ef_label} (UCLA × Age 상호작용):")

    master_clean['age_centered'] = master_clean['age'] - master_clean['age'].mean()
    master_clean['ucla_centered'] = master_clean['ucla_total'] - master_clean['ucla_total'].mean()

    formula_age = f"{ef_var} ~ ucla_centered * age_centered + gender_male + z_dass_dep + z_dass_anx + z_dass_stress"
    model_age = smf.ols(formula_age, data=master_clean).fit()

    # Extract coefficients
    b0 = model_age.params.get('ucla_centered', 0)
    b1 = model_age.params.get('ucla_centered:age_centered', 0)

    var_b0 = model_age.cov_params().loc['ucla_centered', 'ucla_centered']
    var_b1 = model_age.cov_params().loc['ucla_centered:age_centered', 'ucla_centered:age_centered']
    cov_b0_b1 = model_age.cov_params().loc['ucla_centered', 'ucla_centered:age_centered']

    # J-N critical value (for t with df)
    t_crit = stats.t.ppf(0.975, df=model_age.df_resid)

    # Quadratic equation: (b0 + b1*W)^2 / (var_b0 + 2*W*cov + W^2*var_b1) = t_crit^2
    # Solve for W (age_centered values where slope = 0)

    a = t_crit**2 * var_b1 - b1**2
    b = 2 * (t_crit**2 * cov_b0_b1 - b0 * b1)
    c = t_crit**2 * var_b0 - b0**2

    if a != 0:
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            w1 = (-b + np.sqrt(discriminant)) / (2*a)
            w2 = (-b - np.sqrt(discriminant)) / (2*a)

            age_lower = w1 + master_clean['age'].mean()
            age_upper = w2 + master_clean['age'].mean()

            print(f"  Johnson-Neyman 유의 범위:")
            print(f"    연령 < {min(age_lower, age_upper):.1f} 또는 연령 > {max(age_lower, age_upper):.1f}")
            print(f"    (현재 샘플 연령 범위: {master_clean['age'].min()}-{master_clean['age'].max()})")
        else:
            print(f"  Johnson-Neyman 해 없음 (discriminant < 0)")

    p_int_age = model_age.pvalues.get('ucla_centered:age_centered', 1)
    print(f"  UCLA × Age 상호작용: p = {p_int_age:.4f}")

# =============================================================================
# 7. SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n주요 발견:")

# Find most significant interaction
most_sig = df_permutation.loc[df_permutation['p_permutation'].idxmin()]
print(f"\n1. 가장 강한 성별 조절효과:")
print(f"   과제: {most_sig['outcome']}")
print(f"   상호작용 계수: {most_sig['observed_interaction']:.4f}")
print(f"   Permutation p-value: {most_sig['p_permutation']:.4f}")

# Check WCST specifically
wcst_row_slope = df_simple_slopes[df_simple_slopes['outcome'] == 'WCST Perseverative Error Rate'].iloc[0]
wcst_row_boot = df_bootstrap[df_bootstrap['outcome'] == 'WCST'].iloc[0]

print(f"\n2. WCST 세부 결과:")
print(f"   여성 slope: β = {wcst_row_slope['beta_female']:.4f}, p = {wcst_row_slope['p_female']:.4f}")
print(f"   남성 slope: β = {wcst_row_slope['beta_male']:.4f}, p = {wcst_row_slope['p_male']:.4f}")
print(f"   Bootstrap 95% CI: [{wcst_row_boot['boot_ci_lower']:.4f}, {wcst_row_boot['boot_ci_upper']:.4f}]")

if wcst_row_boot['excludes_zero']:
    print(f"   *** Bootstrap CI가 0 제외 → 유의한 조절효과 ***")
else:
    print(f"   Bootstrap CI가 0 포함 → 조절효과 불확실")

# Stratified correlations
print(f"\n3. 성별별 상관:")
wcst_female = df_strat_corr[(df_strat_corr['gender'] == 'Female') & (df_strat_corr['outcome'] == 'WCST')].iloc[0]
wcst_male = df_strat_corr[(df_strat_corr['gender'] == 'Male') & (df_strat_corr['outcome'] == 'WCST')].iloc[0]

print(f"   여성: r = {wcst_female['r']:.3f}, p = {wcst_female['p']:.3f}")
print(f"   남성: r = {wcst_male['r']:.3f}, p = {wcst_male['p']:.3f}")

print("\n분석 완료!")
print(f"결과 저장 위치: {OUTPUT_DIR}")
print("\n생성된 파일:")
print("  - gender_simple_slopes.csv")
print("  - gender_stratified_correlations.csv")
print("  - gender_permutation_test.csv")
print("  - gender_bootstrap_ci.csv")
