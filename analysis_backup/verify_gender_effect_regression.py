#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: 회귀분석 재현 및 검증
Regression Analysis Re-run and Verification
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
VERIFICATION_DIR = OUTPUT_DIR / "gender_verification"

print("=" * 80)
print("Step 2: 회귀분석 재현 및 검증")
print("Regression Analysis Re-run and Verification")
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

print(f"\n분석 샘플: N = {len(master_clean)}")

# =============================================================================
# 2.1 기본 상호작용 모형 재현
# =============================================================================

print("\n" + "=" * 80)
print("2.1 기본 상호작용 모형 재현")
print("=" * 80)

# Full model (원래 분석과 동일)
formula_full = "perseverative_error_rate ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
model_full = smf.ols(formula_full, data=master_clean).fit(cov_type='HC3')

print("\n전체 모형 결과:")
print(model_full.summary())

# Extract key coefficients
beta_main = model_full.params.get('z_ucla', 0)
beta_interaction = model_full.params.get('z_ucla:C(gender_male)[T.1]', 0)
p_interaction = model_full.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

# Simple slopes
beta_female = beta_main
beta_male = beta_main + beta_interaction

# Standard errors for simple slopes
se_female = model_full.bse.get('z_ucla', 0)
se_male = np.sqrt(
    model_full.bse.get('z_ucla', 0)**2 +
    model_full.bse.get('z_ucla:C(gender_male)[T.1]', 0)**2 +
    2 * model_full.cov_params().loc['z_ucla', 'z_ucla:C(gender_male)[T.1]']
)

# p-values for simple slopes
t_female = beta_female / se_female
p_female = 2 * (1 - stats.t.cdf(abs(t_female), df=model_full.df_resid))

t_male = beta_male / se_male
p_male = 2 * (1 - stats.t.cdf(abs(t_male), df=model_full.df_resid))

print(f"\n=== 상호작용 계수 ===")
print(f"β_interaction = {beta_interaction:.4f}, p = {p_interaction:.4f}")
print(f"\n보고된 값: β = 2.5944, p = 0.0042")
print(f"차이: {abs(beta_interaction - 2.5944):.6f}")

print(f"\n=== Simple Slopes ===")
print(f"여성: β = {beta_female:.4f}, SE = {se_female:.4f}, p = {p_female:.4f}")
print(f"남성: β = {beta_male:.4f}, SE = {se_male:.4f}, p = {p_male:.4f}")

print(f"\n보고된 값:")
print(f"  여성: β = -0.3041, p = 0.7241")
print(f"  남성: β = 2.2902, p = 0.0675")

# =============================================================================
# 2.2 모형 사양 민감도 분석
# =============================================================================

print("\n" + "=" * 80)
print("2.2 모형 사양 민감도 분석")
print("=" * 80)

model_specs = []

# Model 1: Minimal (no covariates)
print("\n[모형 1] 공변량 없음")
formula1 = "perseverative_error_rate ~ z_ucla * C(gender_male)"
model1 = smf.ols(formula1, data=master_clean).fit()

int_coef1 = model1.params.get('z_ucla:C(gender_male)[T.1]', 0)
int_p1 = model1.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

print(f"  상호작용: β = {int_coef1:.4f}, p = {int_p1:.4f}")

model_specs.append({
    'model': '공변량 없음',
    'formula': formula1,
    'interaction_beta': int_coef1,
    'interaction_p': int_p1,
    'r2': model1.rsquared
})

# Model 2: Age only
print("\n[모형 2] 연령만")
formula2 = "perseverative_error_rate ~ z_ucla * C(gender_male) + z_age"
model2 = smf.ols(formula2, data=master_clean).fit()

int_coef2 = model2.params.get('z_ucla:C(gender_male)[T.1]', 0)
int_p2 = model2.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

print(f"  상호작용: β = {int_coef2:.4f}, p = {int_p2:.4f}")

model_specs.append({
    'model': '연령만',
    'formula': formula2,
    'interaction_beta': int_coef2,
    'interaction_p': int_p2,
    'r2': model2.rsquared
})

# Model 3: DASS only
print("\n[모형 3] DASS만")
formula3 = "perseverative_error_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_stress"
model3 = smf.ols(formula3, data=master_clean).fit()

int_coef3 = model3.params.get('z_ucla:C(gender_male)[T.1]', 0)
int_p3 = model3.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

print(f"  상호작용: β = {int_coef3:.4f}, p = {int_p3:.4f}")

model_specs.append({
    'model': 'DASS만',
    'formula': formula3,
    'interaction_beta': int_coef3,
    'interaction_p': int_p3,
    'r2': model3.rsquared
})

# Model 4: Full (same as above)
print("\n[모형 4] 전체 (원래 모형)")
print(f"  상호작용: β = {beta_interaction:.4f}, p = {p_interaction:.4f}")

model_specs.append({
    'model': '전체',
    'formula': formula_full,
    'interaction_beta': beta_interaction,
    'interaction_p': p_interaction,
    'r2': model_full.rsquared
})

# Save model comparisons
model_specs_df = pd.DataFrame(model_specs)
model_specs_df.to_csv(VERIFICATION_DIR / "step2_model_specifications.csv",
                     index=False, encoding='utf-8-sig')

print("\n=== 모형 간 비교 ===")
print(model_specs_df[['model', 'interaction_beta', 'interaction_p', 'r2']].to_string(index=False))

# Check consistency
all_p_significant = all(model_specs_df['interaction_p'] < 0.05)

if all_p_significant:
    print("\n✅ 모든 모형에서 상호작용 유의 (p < 0.05)")
else:
    print("\n⚠️ 일부 모형에서 상호작용 비유의")

# =============================================================================
# 2.3 표준화 vs 비표준화
# =============================================================================

print("\n" + "=" * 80)
print("2.3 표준화 vs 비표준화 비교")
print("=" * 80)

# Unstandardized (raw scores)
print("\n[비표준화] 원점수 사용")
formula_raw = "perseverative_error_rate ~ ucla_total * C(gender_male) + age + dass_depression + dass_anxiety + dass_stress"
model_raw = smf.ols(formula_raw, data=master_clean).fit()

int_coef_raw = model_raw.params.get('ucla_total:C(gender_male)[T.1]', 0)
int_p_raw = model_raw.pvalues.get('ucla_total:C(gender_male)[T.1]', 1)

print(f"  상호작용: β = {int_coef_raw:.4f}, p = {int_p_raw:.4f}")

# Standardized (current)
print("\n[표준화] Z-score 사용")
print(f"  상호작용: β = {beta_interaction:.4f}, p = {p_interaction:.4f}")

# Compare p-values
p_diff = abs(int_p_raw - p_interaction)
print(f"\np-value 차이: {p_diff:.6f}")

if p_diff < 0.001:
    print("✅ 표준화 여부와 무관하게 일관된 결과")
else:
    print("⚠️ 표준화에 따라 결과 변동")

# =============================================================================
# 2.4 성별별 개별 회귀 (확인용)
# =============================================================================

print("\n" + "=" * 80)
print("2.4 성별별 개별 회귀")
print("=" * 80)

for gender_val, gender_label in [(0, '여성'), (1, '남성')]:
    subset = master_clean[master_clean['gender_male'] == gender_val]

    print(f"\n{gender_label} (N={len(subset)}):")

    # Regression with covariates
    formula_gender = "perseverative_error_rate ~ z_ucla + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
    model_gender = smf.ols(formula_gender, data=subset).fit()

    beta_ucla = model_gender.params.get('z_ucla', 0)
    se_ucla = model_gender.bse.get('z_ucla', 0)
    p_ucla = model_gender.pvalues.get('z_ucla', 1)

    print(f"  β_UCLA = {beta_ucla:.4f}, SE = {se_ucla:.4f}, p = {p_ucla:.4f}")

    # Compare with simple slopes from interaction model
    if gender_val == 0:
        expected_beta = beta_female
        print(f"  Simple slope에서: β = {expected_beta:.4f}")
    else:
        expected_beta = beta_male
        print(f"  Simple slope에서: β = {expected_beta:.4f}")

    diff = abs(beta_ucla - expected_beta)
    print(f"  차이: {diff:.4f} {'✓' if diff < 0.5 else '⚠️'}")

# =============================================================================
# 2.5 Summary
# =============================================================================

print("\n" + "=" * 80)
print("Step 2 요약")
print("=" * 80)

print("\n✅ 검증 결과:")
print(f"  1. 상호작용 계수 재현: β = {beta_interaction:.4f}, p = {p_interaction:.4f}")
print(f"  2. 모형 사양 민감도: {len([x for x in model_specs_df['interaction_p'] if x < 0.05])}/{len(model_specs_df)} 모형에서 유의")
print(f"  3. 표준화 무관성: p-value 차이 {p_diff:.6f}")

print("\n생성된 파일:")
print(f"  - {VERIFICATION_DIR / 'step2_model_specifications.csv'}")

print("\n" + "=" * 80)
print("Step 2 완료")
print("=" * 80)
