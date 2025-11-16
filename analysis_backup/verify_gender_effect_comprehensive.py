#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steps 3-6 통합: 영향력 진단, 통계 검증, 다중검정, 검증력
Comprehensive Verification: Influence, Robustness, Multiple Testing, Power
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
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
VERIFICATION_DIR = OUTPUT_DIR / "gender_verification"

print("=" * 80)
print("Steps 3-6: 종합 검증")
print("Comprehensive Verification")
print("=" * 80)

# Load data
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

# Standardize
scaler = StandardScaler()
master_clean['z_ucla'] = scaler.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_stress'] = scaler.fit_transform(master_clean[['dass_stress']])
master_clean['z_age'] = scaler.fit_transform(master_clean[['age']])

# =============================================================================
# STEP 3: 영향력 있는 관측치 진단
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: 영향력 있는 관측치 진단")
print("=" * 80)

# Fit the model
formula = "perseverative_error_rate ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
model = smf.ols(formula, data=master_clean).fit()

# Influence measures
influence = OLSInfluence(model)

# Cook's Distance
cooks_d = influence.cooks_distance[0]
threshold_cooks = 4 / len(master_clean)

# DFBETAS for interaction term
dfbetas = influence.dfbetas
# Interaction term is at index 3
dfbetas_interaction = dfbetas[:, 3]
threshold_dfbetas = 2 / np.sqrt(len(master_clean))

print(f"\nCook's Distance threshold: {threshold_cooks:.4f}")
print(f"DFBETAS threshold: {threshold_dfbetas:.4f}")

# Influential cases
influential_cooks = cooks_d > threshold_cooks
influential_dfbetas = np.abs(dfbetas_interaction) > threshold_dfbetas

print(f"\n영향력 큰 관측치:")
print(f"  Cook's D > {threshold_cooks:.4f}: {influential_cooks.sum()}개")
print(f"  |DFBETAS| > {threshold_dfbetas:.4f}: {influential_dfbetas.sum()}개")

if influential_cooks.sum() > 0:
    print(f"\nCook's D가 큰 케이스:")
    master_clean['cooks_d'] = cooks_d
    influential_cases = master_clean[influential_cooks][[
        'participant_id', 'gender', 'ucla_total', 'perseverative_error_rate', 'cooks_d'
    ]]
    print(influential_cases.to_string(index=False))

# Re-run without influential cases
print(f"\n=== 영향력 큰 관측치 제거 후 재분석 ===")

master_no_influence = master_clean[~(influential_cooks | influential_dfbetas)]
print(f"제거된 관측치: {len(master_clean) - len(master_no_influence)}개")
print(f"남은 샘플: N={len(master_no_influence)}")

model_no_influence = smf.ols(formula, data=master_no_influence).fit()

int_coef_no_inf = model_no_influence.params.get('z_ucla:C(gender_male)[T.1]', 0)
int_p_no_inf = model_no_influence.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

print(f"  원래: β = {model.params['z_ucla:C(gender_male)[T.1]']:.4f}, p = {model.pvalues['z_ucla:C(gender_male)[T.1]']:.4f}")
print(f"  제거 후: β = {int_coef_no_inf:.4f}, p = {int_p_no_inf:.4f}")

coef_change = abs(int_coef_no_inf - model.params['z_ucla:C(gender_male)[T.1]'])
p_change = abs(int_p_no_inf - model.pvalues['z_ucla:C(gender_male)[T.1]'])

print(f"  계수 변화: {coef_change:.4f}")
print(f"  p-value 변화: {p_change:.4f}")

if int_p_no_inf < 0.05:
    print("  ✅ 영향력 관측치 제거 후에도 유의함")
else:
    print("  ⚠️ 영향력 관측치 제거 시 비유의")

# =============================================================================
# STEP 3.2: 이상치 제거 민감도
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3.2: 이상치 제거 민감도 분석")
print("=" * 80)

outlier_results = []

# Scenario 1: ±2 SD
master_clean['z_wcst'] = (master_clean['perseverative_error_rate'] - master_clean['perseverative_error_rate'].mean()) / master_clean['perseverative_error_rate'].std()
master_2sd = master_clean[np.abs(master_clean['z_wcst']) <= 2]

model_2sd = smf.ols(formula, data=master_2sd).fit()
int_2sd = model_2sd.params.get('z_ucla:C(gender_male)[T.1]', 0)
p_2sd = model_2sd.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

print(f"\n±2 SD (N={len(master_2sd)}, 제거={len(master_clean)-len(master_2sd)}):")
print(f"  β = {int_2sd:.4f}, p = {p_2sd:.4f}")

outlier_results.append({
    'scenario': '±2 SD',
    'n': len(master_2sd),
    'removed': len(master_clean) - len(master_2sd),
    'beta': int_2sd,
    'p_value': p_2sd
})

# Scenario 2: ±3 SD
master_3sd = master_clean[np.abs(master_clean['z_wcst']) <= 3]

model_3sd = smf.ols(formula, data=master_3sd).fit()
int_3sd = model_3sd.params.get('z_ucla:C(gender_male)[T.1]', 0)
p_3sd = model_3sd.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

print(f"\n±3 SD (N={len(master_3sd)}, 제거={len(master_clean)-len(master_3sd)}):")
print(f"  β = {int_3sd:.4f}, p = {p_3sd:.4f}")

outlier_results.append({
    'scenario': '±3 SD',
    'n': len(master_3sd),
    'removed': len(master_clean) - len(master_3sd),
    'beta': int_3sd,
    'p_value': p_3sd
})

# Scenario 3: Robust regression
print(f"\nRobust regression (M-estimator):")
model_robust = smf.rlm(formula, data=master_clean, M=sm.robust.norms.HuberT()).fit()
int_robust = model_robust.params.get('z_ucla:C(gender_male)[T.1]', 0)
p_robust = model_robust.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

print(f"  β = {int_robust:.4f}, p = {p_robust:.4f}")

outlier_results.append({
    'scenario': 'Robust',
    'n': len(master_clean),
    'removed': 0,
    'beta': int_robust,
    'p_value': p_robust
})

# Save
outlier_df = pd.DataFrame(outlier_results)
outlier_df.to_csv(VERIFICATION_DIR / "step3_outlier_sensitivity.csv",
                  index=False, encoding='utf-8-sig')

# =============================================================================
# STEP 4: 통계적 검증 강화
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: 통계적 검증 강화")
print("=" * 80)

# 4.1 Permutation test (multiple seeds)
print("\n4.1 Permutation Test (다중 시드)")
print("-" * 80)

observed = model.params['z_ucla:C(gender_male)[T.1]']

perm_results = []

for seed in [42, 123, 456, 789, 999]:
    np.random.seed(seed)
    null_dist = []

    for _ in range(1000):
        shuffled = master_clean.copy()
        shuffled['gender_male'] = np.random.permutation(shuffled['gender_male'].values)

        try:
            m = smf.ols(formula, data=shuffled).fit()
            null_dist.append(m.params.get('z_ucla:C(gender_male)[T.1]', 0))
        except:
            null_dist.append(0)

    null_dist = np.array(null_dist)
    p_perm = np.mean(np.abs(null_dist) >= np.abs(observed))

    perm_results.append(p_perm)
    print(f"  Seed {seed}: p = {p_perm:.4f}")

print(f"\n평균 p-value: {np.mean(perm_results):.4f}")
print(f"범위: [{min(perm_results):.4f}, {max(perm_results):.4f}]")

if max(perm_results) < 0.05:
    print("✅ 모든 시드에서 p < 0.05")
else:
    print("⚠️ 일부 시드에서 p > 0.05")

# 4.2 Bootstrap CI (increased iterations)
print("\n4.2 Bootstrap CI (10000 iterations)")
print("-" * 80)

boot_interactions = []

for _ in range(10000):
    sample = master_clean.sample(n=len(master_clean), replace=True)
    try:
        m = smf.ols(formula, data=sample).fit()
        boot_interactions.append(m.params.get('z_ucla:C(gender_male)[T.1]', np.nan))
    except:
        boot_interactions.append(np.nan)

boot_interactions = np.array([x for x in boot_interactions if not np.isnan(x)])

ci_lower = np.percentile(boot_interactions, 2.5)
ci_upper = np.percentile(boot_interactions, 97.5)

print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  CI 제외 0: {ci_lower > 0 or ci_upper < 0}")

# =============================================================================
# STEP 5: 다중검정 보정
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: 다중검정 보정")
print("=" * 80)

# We tested 3 EF tasks for gender interaction
p_values = []
outcomes = []

# WCST (our finding)
p_wcst = model.pvalues['z_ucla:C(gender_male)[T.1]']
p_values.append(p_wcst)
outcomes.append('WCST')

# Stroop
formula_stroop = "stroop_interference ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
model_stroop = smf.ols(formula_stroop, data=master_clean).fit()
p_stroop = model_stroop.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)
p_values.append(p_stroop)
outcomes.append('Stroop')

# PRP
formula_prp = "prp_bottleneck ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
model_prp = smf.ols(formula_prp, data=master_clean).fit()
p_prp = model_prp.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)
p_values.append(p_prp)
outcomes.append('PRP')

# Bonferroni
alpha_bonf = 0.05 / 3
print(f"\nBonferroni correction: α = {alpha_bonf:.4f}")

for outcome, p in zip(outcomes, p_values):
    sig = "***" if p < alpha_bonf else ""
    print(f"  {outcome}: p = {p:.4f} {sig}")

# FDR (Benjamini-Hochberg)
from statsmodels.stats.multitest import multipletests

reject, q_values, _, _ = multipletests(p_values, method='fdr_bh')

print(f"\nFDR (Benjamini-Hochberg) q-values:")
for outcome, q, rej in zip(outcomes, q_values, reject):
    sig = "***" if rej else ""
    print(f"  {outcome}: q = {q:.4f} {sig}")

# =============================================================================
# STEP 6: 검증력 분석
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: 검증력 분석")
print("=" * 80)

# Post-hoc power
from statsmodels.stats.power import FTestAnovaPower

power_analysis = FTestAnovaPower()

# Effect size (f^2) for interaction
r2_full = model.rsquared
formula_no_int = "perseverative_error_rate ~ z_ucla + C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
model_no_int = smf.ols(formula_no_int, data=master_clean).fit()
r2_no_int = model_no_int.rsquared

f2 = (r2_full - r2_no_int) / (1 - r2_full)

print(f"\nEffect size f² = {f2:.4f}")

# Convert to Cohen's d approximation
cohens_d = 2 * np.sqrt(f2)
print(f"Cohen's d ≈ {cohens_d:.4f}")

# Power for current N
achieved_power = power_analysis.solve_power(
    effect_size=f2,
    nobs=len(master_clean),
    alpha=0.05
)

print(f"\n현재 N={len(master_clean)}에서 검증력: {achieved_power:.3f}")

# Required N for power=0.80
n_required = power_analysis.solve_power(
    effect_size=f2,
    power=0.80,
    alpha=0.05
)

print(f"검증력 0.80을 위한 필요 N: {int(n_required)}")

# Correlation-based power for males only
female_r = pearsonr(
    master_clean[master_clean['gender_male']==0]['ucla_total'],
    master_clean[master_clean['gender_male']==0]['perseverative_error_rate']
)[0]

male_r = pearsonr(
    master_clean[master_clean['gender_male']==1]['ucla_total'],
    master_clean[master_clean['gender_male']==1]['perseverative_error_rate']
)[0]

print(f"\n성별별 상관:")
print(f"  여성 (N=45): r = {female_r:.3f}")
print(f"  남성 (N=27): r = {male_r:.3f}")

# Fisher's z-based power for males
from scipy.stats import norm

z_r = 0.5 * np.log((1 + male_r) / (1 - male_r))
n_male = (master_clean['gender_male'] == 1).sum()

z_alpha = norm.ppf(0.975)
delta = z_r * np.sqrt(n_male - 3)

power_male = 1 - norm.cdf(z_alpha - delta) + norm.cdf(-z_alpha - delta)

print(f"\n남성에서 r={male_r:.3f} 탐지 검증력 (N={n_male}): {power_male:.3f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("Steps 3-6 종합 요약")
print("=" * 80)

print("\n✅ 주요 발견:")
print(f"  1. 영향력 관측치: {influential_cooks.sum()}개 (Cook's D)")
print(f"  2. 제거 후 유의성: p = {int_p_no_inf:.4f} {'(유의)' if int_p_no_inf < 0.05 else '(비유의)'}")
print(f"  3. Permutation 평균 p: {np.mean(perm_results):.4f}")
print(f"  4. Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  5. Bonferroni 보정: p={p_wcst:.4f} < {alpha_bonf:.4f}? {'Yes' if p_wcst < alpha_bonf else 'No'}")
print(f"  6. 검증력: {achieved_power:.3f} (현재 N={len(master_clean)})")

print("\n생성된 파일:")
print(f"  - {VERIFICATION_DIR / 'step3_outlier_sensitivity.csv'}")

print("\n" + "=" * 80)
print("Steps 3-6 완료")
print("=" * 80)
