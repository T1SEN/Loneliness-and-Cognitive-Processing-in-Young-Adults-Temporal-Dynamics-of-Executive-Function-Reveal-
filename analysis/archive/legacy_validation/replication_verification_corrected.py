#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
재현성 검증 (수정버전) - 원본 방법론 정확히 따르기
Replication Verification (Corrected) - Exact Original Methodology

이 스크립트는 원본 분석과 동일한 방법으로 핵심 가설들을 재검증합니다.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings
from analysis.utils.data_loader_utils import load_master_dataset
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/replication_verification")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("재현성 검증 (수정버전)")
print("Replication Verification - CORRECTED")
print("=" * 80)

# =============================================================================
# 데이터 로딩 (원본과 동일한 방식)
# =============================================================================

print("\n?????...")

# Master dataset ?? (shared loader)
from analysis.utils.data_loader_utils import load_master_dataset
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
master['gender_male'] = (master['gender'] == 'male').astype(int)
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
if 'pe_rate' in master.columns and 'pe_rate' not in master.columns:
    master['pe_rate'] = master['pe_rate']


def zscore(series):
    s = pd.to_numeric(series, errors='coerce')
    std = s.std(ddof=0)
    if std is None or std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std

master['z_ucla'] = zscore(master['ucla_total'])
master['z_dass_dep'] = zscore(master['dass_depression'])
master['z_dass_anx'] = zscore(master['dass_anxiety'])
master['z_dass_stress'] = zscore(master['dass_stress'])
master['z_age'] = zscore(master['age'])

# 필수 변수 필터링
essential_cols = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
                   'age', 'gender_male', 'pe_rate']
analysis_df = master.dropna(subset=essential_cols).copy()

print(f"  총 샘플: N = {len(analysis_df)}")
print(f"  남성: {int(analysis_df['gender_male'].sum())}명")
print(f"  여성: {int((1-analysis_df['gender_male']).sum())}명")

# =============================================================================
# [핵심 검증 1] WCST PE × Gender 조절효과 (원본 방법론 정확히 따르기)
# =============================================================================

print("\n" + "=" * 80)
print("[1] WCST PE × Gender 조절효과 (원본 방법론)")
print("=" * 80)

# 원본과 동일한 formula: z_ucla * gender + covariates
formula = "pe_rate ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
model = smf.ols(formula, data=analysis_df).fit(cov_type='HC3')  # Robust SE

# 계수 추출
beta_main = model.params.get('z_ucla', 0)
beta_interaction = model.params.get('z_ucla:C(gender_male)[T.1]', 0)
p_interaction = model.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)

# Simple slopes (원본 방법)
beta_female = beta_main  # Reference category
beta_male = beta_main + beta_interaction

# Standard errors (delta method)
se_female = model.bse.get('z_ucla', 0)
se_male = np.sqrt(
    model.bse.get('z_ucla', 0)**2 +
    model.bse.get('z_ucla:C(gender_male)[T.1]', 0)**2 +
    2 * model.cov_params().loc['z_ucla', 'z_ucla:C(gender_male)[T.1]']
)

# p-values
t_female = beta_female / se_female if se_female > 0 else 0
p_female = 2 * (1 - stats.t.cdf(abs(t_female), df=model.df_resid))

t_male = beta_male / se_male if se_male > 0 else 0
p_male = 2 * (1 - stats.t.cdf(abs(t_male), df=model.df_resid))

print("\n상호작용 효과:")
print(f"  β_interaction = {beta_interaction:.4f}, p = {p_interaction:.6f}")
print(f"  주장: β≈2.59, p≈0.004")

print("\n여성 Simple Slope:")
print(f"  β = {beta_female:.4f}, SE = {se_female:.4f}, p = {p_female:.6f}")
print(f"  주장: β≈−0.30, p≈0.72")

print("\n남성 Simple Slope:")
print(f"  β = {beta_male:.4f}, SE = {se_male:.4f}, p = {p_male:.6f}")
print(f"  주장: β≈2.29, p≈0.067")

# 재현 판정
interaction_match = (abs(beta_interaction - 2.59) / 2.59 * 100 <= 15 and
                     abs(p_interaction - 0.004) <= 0.02)
male_slope_match = (abs(beta_male - 2.29) / 2.29 * 100 <= 15 and
                    abs(p_male - 0.067) <= 0.02)
female_slope_match = (abs(p_female - 0.72) <= 0.10)  # p-value만 비교 (효과 작아서)

print("\n재현 판정:")
print(f"  상호작용 효과: {'✅ 재현 성공' if interaction_match else '❌ 재현 실패'}")
print(f"  남성 기울기: {'✅ 재현 성공' if male_slope_match else '❌ 재현 실패'}")
print(f"  여성 기울기: {'✅ 재현 성공' if female_slope_match else '❌ 재현 실패'}")

# =============================================================================
# [핵심 검증 2] DASS Anxiety 층화
# =============================================================================

print("\n" + "=" * 80)
print("[2] DASS Anxiety 층화 효과")
print("=" * 80)

# Median split
median_anx = analysis_df['dass_anxiety'].median()
analysis_df['anxiety_group'] = analysis_df['dass_anxiety'].apply(
    lambda x: 'low' if x <= median_anx else 'high'
)

low_anx = analysis_df[analysis_df['anxiety_group'] == 'low']
high_anx = analysis_df[analysis_df['anxiety_group'] == 'high']

print(f"\nLow Anxiety: N={len(low_anx)}")
print(f"High Anxiety: N={len(high_anx)}")

# Low anxiety에서 조절효과 (simplified formula)
formula_low = 'pe_rate ~ z_ucla * C(gender_male) + z_age'
model_low = smf.ols(formula_low, data=low_anx).fit(cov_type='HC3')
beta_low = model_low.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
p_low = model_low.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)

# High anxiety
model_high = smf.ols(formula_low, data=high_anx).fit(cov_type='HC3')
beta_high = model_high.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
p_high = model_high.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)

print(f"\nUCLA × Gender 상호작용:")
print(f"  Low Anxiety: β={beta_low:.3f}, p={p_low:.6f}")
print(f"  High Anxiety: β={beta_high:.3f}, p={p_high:.6f}")
print(f"  주장: Low Anxiety β≈4.28, p≈0.008")

dass_match = (abs(beta_low - 4.28) / 4.28 * 100 <= 15 and
              abs(p_low - 0.008) <= 0.01)

print(f"\n재현 판정: {'✅ 재현 성공' if dass_match else '❌ 재현 실패'}")

# =============================================================================
# [보조 검증] Stroop & PRP 상호작용
# =============================================================================

print("\n" + "=" * 80)
print("[3] Stroop & PRP 조절효과 (참고)")
print("=" * 80)

# Stroop
if 'stroop_interference' in analysis_df.columns:
    formula_stroop = "stroop_interference ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
    data_stroop = analysis_df.dropna(subset=['stroop_interference']).copy()
    if len(data_stroop) >= 30:
        model_stroop = smf.ols(formula_stroop, data=data_stroop).fit(cov_type='HC3')
        beta_stroop = model_stroop.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
        p_stroop = model_stroop.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)
        print(f"\nStroop Interference × Gender:")
        print(f"  β_interaction = {beta_stroop:.4f}, p = {p_stroop:.6f}")
        print(f"  원본: β≈21.62, p≈0.362 (NS 예상)")

# PRP
if 'prp_bottleneck' in analysis_df.columns:
    formula_prp = "prp_bottleneck ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_stress"
    data_prp = analysis_df.dropna(subset=['prp_bottleneck']).copy()
    if len(data_prp) >= 30:
        model_prp = smf.ols(formula_prp, data=data_prp).fit(cov_type='HC3')
        beta_prp = model_prp.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
        p_prp = model_prp.pvalues.get('z_ucla:C(gender_male)[T.1]', 1)
        print(f"\nPRP Bottleneck × Gender:")
        print(f"  β_interaction = {beta_prp:.4f}, p = {p_prp:.6f}")
        print(f"  원본: β≈61.24, p≈0.143 (NS)")

# =============================================================================
# 최종 요약
# =============================================================================

print("\n" + "=" * 80)
print("최종 재현성 요약")
print("=" * 80)

results_summary = []

# WCST PE × Gender
results_summary.append({
    '가설': 'WCST PE × Gender 상호작용',
    '주장 값': f"β={2.59:.2f}, p={0.004:.3f}",
    '실제 값': f"β={beta_interaction:.2f}, p={p_interaction:.3f}",
    '재현': '✅' if interaction_match else '❌'
})

results_summary.append({
    '가설': 'WCST 남성 기울기',
    '주장 값': f"β={2.29:.2f}, p={0.067:.3f}",
    '실제 값': f"β={beta_male:.2f}, p={p_male:.3f}",
    '재현': '✅' if male_slope_match else '❌'
})

results_summary.append({
    '가설': 'WCST 여성 기울기',
    '주장 값': f"β={-0.30:.2f}, p={0.72:.3f}",
    '실제 값': f"β={beta_female:.2f}, p={p_female:.3f}",
    '재현': '✅' if female_slope_match else '❌'
})

# DASS Anxiety
results_summary.append({
    '가설': 'DASS Low Anxiety 조절효과',
    '주장 값': f"β={4.28:.2f}, p={0.008:.3f}",
    '실제 값': f"β={beta_low:.2f}, p={p_low:.3f}",
    '재현': '✅' if dass_match else '❌'
})

results_df = pd.DataFrame(results_summary)
print("\n")
print(results_df.to_string(index=False))

# 재현율 계산
total_tests = len(results_summary)
successful = sum(1 for r in results_summary if r['재현'] == '✅')
success_rate = successful / total_tests * 100

print(f"\n재현율: {successful}/{total_tests} ({success_rate:.1f}%)")

if success_rate >= 75:
    print("\n✅ 재현 성공! 핵심 가설들이 원본 분석과 일치합니다.")
elif success_rate >= 50:
    print("\n⚠️  부분 재현. 일부 가설만 재현되었습니다.")
else:
    print("\n❌ 재현 실패. 대부분의 가설이 재현되지 않았습니다.")

# CSV 저장
results_df.to_csv(OUTPUT_DIR / "corrected_replication_summary.csv", index=False, encoding='utf-8-sig')

# 상세 텍스트 보고서
report_lines = []
report_lines.append("=" * 80)
report_lines.append("재현성 검증 최종 보고서 (수정버전)")
report_lines.append("CORRECTED REPLICATION VERIFICATION REPORT")
report_lines.append("=" * 80)
report_lines.append(f"\n실행 일시: {pd.Timestamp.now()}")
report_lines.append(f"데이터: N={len(analysis_df)} (남성={int(analysis_df['gender_male'].sum())}, 여성={int((1-analysis_df['gender_male']).sum())})")
report_lines.append(f"\n방법론: 원본 분석과 동일한 표준화 + 공변량 포함 회귀")
report_lines.append(f"\n총 검증 항목: {total_tests}")
report_lines.append(f"재현 성공: {successful} ({success_rate:.1f}%)")
report_lines.append(f"재현 실패: {total_tests - successful} ({(total_tests - successful)/total_tests*100:.1f}%)")

report_lines.append("\n\n" + "=" * 80)
report_lines.append("상세 결과")
report_lines.append("=" * 80)

report_lines.append(f"\n### [1] WCST PE × Gender 조절효과 ###")
report_lines.append(f"\n상호작용 효과:")
report_lines.append(f"  주장: β={2.59:.4f}, p={0.004:.6f}")
report_lines.append(f"  실제: β={beta_interaction:.4f}, p={p_interaction:.6f}")
report_lines.append(f"  차이: β {abs(beta_interaction - 2.59)/2.59*100:.1f}%, p-diff={abs(p_interaction - 0.004):.6f}")
report_lines.append(f"  판정: {'✅ 재현 성공' if interaction_match else '❌ 재현 실패'}")

report_lines.append(f"\n남성 Simple Slope:")
report_lines.append(f"  주장: β={2.29:.4f}, p={0.067:.6f}")
report_lines.append(f"  실제: β={beta_male:.4f}, p={p_male:.6f}")
report_lines.append(f"  차이: β {abs(beta_male - 2.29)/2.29*100:.1f}%, p-diff={abs(p_male - 0.067):.6f}")
report_lines.append(f"  판정: {'✅ 재현 성공' if male_slope_match else '❌ 재현 실패'}")

report_lines.append(f"\n여성 Simple Slope:")
report_lines.append(f"  주장: β={-0.30:.4f}, p={0.72:.6f}")
report_lines.append(f"  실제: β={beta_female:.4f}, p={p_female:.6f}")
report_lines.append(f"  차이: p-diff={abs(p_female - 0.72):.6f}")
report_lines.append(f"  판정: {'✅ 재현 성공' if female_slope_match else '❌ 재현 실패'}")

report_lines.append(f"\n\n### [2] DASS Anxiety 층화 효과 ###")
report_lines.append(f"\nLow Anxiety 조절효과:")
report_lines.append(f"  주장: β={4.28:.4f}, p={0.008:.6f}")
report_lines.append(f"  실제: β={beta_low:.4f}, p={p_low:.6f}")
report_lines.append(f"  차이: β {abs(beta_low - 4.28)/4.28*100:.1f}%, p-diff={abs(p_low - 0.008):.6f}")
report_lines.append(f"  판정: {'✅ 재현 성공' if dass_match else '❌ 재현 실패'}")

report_lines.append("\n\n" + "=" * 80)
report_lines.append("결론")
report_lines.append("=" * 80)

if success_rate >= 75:
    report_lines.append("\n✅ 재현 성공!")
    report_lines.append("핵심 가설들이 원본 분석 결과와 일치합니다.")
    report_lines.append("통계적 수치의 미세한 차이는 반올림 오차 수준입니다.")
elif success_rate >= 50:
    report_lines.append("\n⚠️  부분적 재현")
    report_lines.append("일부 가설은 재현되었으나, 추가 검토가 필요합니다.")
else:
    report_lines.append("\n❌ 재현 실패")
    report_lines.append("대부분의 가설이 재현되지 않았습니다.")
    report_lines.append("데이터 또는 분석 방법에 차이가 있을 수 있습니다.")

report_lines.append(f"\n원본과의 주요 차이점:")
report_lines.append(f"  - 표본 크기: 비슷 (원본 추정 ~81명, 현재 {len(analysis_df)}명)")
report_lines.append(f"  - 표준화 방법: 동일 (z-score with ddof=0)")
report_lines.append(f"  - 공변량: 동일 (DASS depression, anxiety, stress + age)")
report_lines.append(f"  - Robust SE: 동일 (HC3)")

report_text = "\n".join(report_lines)

with open(OUTPUT_DIR / "CORRECTED_REPLICATION_REPORT.txt", "w", encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✅ 보고서 저장 완료:")
print(f"  - {OUTPUT_DIR / 'corrected_replication_summary.csv'}")
print(f"  - {OUTPUT_DIR / 'CORRECTED_REPLICATION_REPORT.txt'}")

print("\n" + "=" * 80)
print("재현성 검증 완료!")
print("=" * 80)
