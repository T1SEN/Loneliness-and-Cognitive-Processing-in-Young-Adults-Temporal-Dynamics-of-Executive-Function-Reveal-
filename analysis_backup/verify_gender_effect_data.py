#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 원시 데이터 검증
Raw Data Verification for Gender Moderation Effect
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

VERIFICATION_DIR = OUTPUT_DIR / "gender_verification"
VERIFICATION_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Step 1: 원시 데이터 검증")
print("Raw Data Verification")
print("=" * 80)

# =============================================================================
# 1. 데이터 로딩 및 기본 확인
# =============================================================================

print("\n" + "=" * 80)
print("1.1 데이터 로딩 및 구조 확인")
print("=" * 80)

master = pd.read_csv(OUTPUT_DIR / "master_dataset.csv")
master = master.rename(columns={'pe_rate': 'perseverative_error_rate'})

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

print(f"\n전체 참가자 수: {len(master)}")
print(f"변수 개수: {len(master.columns)}")

# 성별 코딩 확인
print("\n성별 분포:")
print(master['gender'].value_counts())

# Binary coding
master['gender_male'] = (master['gender'] == '남성').astype(int)

print("\n성별 이진 코딩:")
print(f"  0 (여성): {(master['gender_male'] == 0).sum()}명")
print(f"  1 (남성): {(master['gender_male'] == 1).sum()}명")

# =============================================================================
# 2. 핵심 변수 기술통계
# =============================================================================

print("\n" + "=" * 80)
print("1.2 핵심 변수 기술통계 (전체)")
print("=" * 80)

key_vars = ['ucla_total', 'perseverative_error_rate', 'dass_depression',
            'dass_anxiety', 'dass_stress', 'age']

desc_stats = master[key_vars].describe()
print("\n", desc_stats.to_string())

# 결측치 확인
print("\n결측치 개수:")
print(master[key_vars + ['gender']].isnull().sum())

# Complete cases
master_complete = master[key_vars + ['gender', 'gender_male', 'participant_id']].dropna()
print(f"\n완전 데이터: {len(master_complete)}명 (결측 제거 후)")

# =============================================================================
# 3. 성별별 기술통계
# =============================================================================

print("\n" + "=" * 80)
print("1.3 성별별 기술통계")
print("=" * 80)

verification_results = []

for gender_val, gender_label in [(0, '여성 (Female)'), (1, '남성 (Male)')]:
    subset = master_complete[master_complete['gender_male'] == gender_val]

    print(f"\n{gender_label}: N = {len(subset)}")
    print("-" * 80)

    for var in ['ucla_total', 'perseverative_error_rate']:
        mean = subset[var].mean()
        std = subset[var].std()
        min_val = subset[var].min()
        max_val = subset[var].max()

        print(f"\n{var}:")
        print(f"  평균: {mean:.2f}")
        print(f"  표준편차: {std:.2f}")
        print(f"  범위: [{min_val:.2f}, {max_val:.2f}]")

        # Z-scores for outlier detection
        z_scores = np.abs((subset[var] - mean) / std)
        outliers_2sd = (z_scores > 2).sum()
        outliers_3sd = (z_scores > 3).sum()

        print(f"  이상치 (>2 SD): {outliers_2sd}개")
        print(f"  극단값 (>3 SD): {outliers_3sd}개")

        if outliers_3sd > 0:
            outlier_ids = subset[z_scores > 3]['participant_id'].tolist()
            outlier_values = subset[z_scores > 3][var].tolist()
            print(f"  ⚠️  극단값 참가자 ID: {outlier_ids}")
            print(f"  ⚠️  값: {[f'{v:.2f}' for v in outlier_values]}")

# =============================================================================
# 4. 수동 상관계수 계산
# =============================================================================

print("\n" + "=" * 80)
print("1.4 수동 상관계수 계산 (검증)")
print("=" * 80)

for gender_val, gender_label in [(0, '여성'), (1, '남성')]:
    subset = master_complete[master_complete['gender_male'] == gender_val]

    # Pearson correlation
    r, p = pearsonr(subset['ucla_total'], subset['perseverative_error_rate'])

    # Spearman correlation
    rho, p_spear = spearmanr(subset['ucla_total'], subset['perseverative_error_rate'])

    print(f"\n{gender_label} (N={len(subset)}):")
    print(f"  Pearson: r = {r:.4f}, p = {p:.4f}")
    print(f"  Spearman: rho = {rho:.4f}, p = {p_spear:.4f}")

    # Compare with reported values
    if gender_val == 0:
        reported_r = -0.249
        diff = abs(r - reported_r)
        print(f"  보고된 값: r = {reported_r:.3f}")
        print(f"  차이: {diff:.6f} {'✓ 일치' if diff < 0.001 else '⚠️ 불일치'}")
    else:
        reported_r = 0.241
        diff = abs(r - reported_r)
        print(f"  보고된 값: r = {reported_r:.3f}")
        print(f"  차이: {diff:.6f} {'✓ 일치' if diff < 0.001 else '⚠️ 불일치'}")

    verification_results.append({
        'gender': gender_label,
        'n': len(subset),
        'pearson_r': r,
        'pearson_p': p,
        'spearman_rho': rho,
        'spearman_p': p_spear,
        'ucla_mean': subset['ucla_total'].mean(),
        'ucla_sd': subset['ucla_total'].std(),
        'wcst_mean': subset['perseverative_error_rate'].mean(),
        'wcst_sd': subset['perseverative_error_rate'].std()
    })

# Save verification
verification_df = pd.DataFrame(verification_results)
verification_df.to_csv(VERIFICATION_DIR / "step1_data_verification.csv",
                      index=False, encoding='utf-8-sig')

# =============================================================================
# 5. 극단값 상세 분석
# =============================================================================

print("\n" + "=" * 80)
print("1.5 극단값 상세 분석")
print("=" * 80)

# Find extreme values across all participants
master_complete['z_ucla'] = (master_complete['ucla_total'] - master_complete['ucla_total'].mean()) / master_complete['ucla_total'].std()
master_complete['z_wcst'] = (master_complete['perseverative_error_rate'] - master_complete['perseverative_error_rate'].mean()) / master_complete['perseverative_error_rate'].std()

extreme_cases = master_complete[
    (np.abs(master_complete['z_ucla']) > 2.5) |
    (np.abs(master_complete['z_wcst']) > 2.5)
].copy()

if len(extreme_cases) > 0:
    print(f"\n극단값 케이스: {len(extreme_cases)}명")
    print("\nID\t성별\tUCLA\tWCST\tz_UCLA\tz_WCST")
    print("-" * 80)

    for _, row in extreme_cases.iterrows():
        gender_str = '남성' if row['gender_male'] == 1 else '여성'
        print(f"{row['participant_id'][:8]}\t{gender_str}\t{row['ucla_total']:.1f}\t{row['perseverative_error_rate']:.1f}\t{row['z_ucla']:.2f}\t{row['z_wcst']:.2f}")

    # Save extreme cases
    extreme_cases.to_csv(VERIFICATION_DIR / "step1_extreme_cases.csv",
                        index=False, encoding='utf-8-sig')
else:
    print("\n극단값 케이스 없음 (모두 ±2.5 SD 이내)")

# =============================================================================
# 6. 데이터 품질 체크리스트
# =============================================================================

print("\n" + "=" * 80)
print("1.6 데이터 품질 체크리스트")
print("=" * 80)

checklist = []

# Check 1: Sample sizes
check1 = len(master_complete[master_complete['gender_male'] == 0]) == 45
check1_male = len(master_complete[master_complete['gender_male'] == 1]) == 27

print(f"\n✓ 1. 참가자 수 확인:")
print(f"    여성: {len(master_complete[master_complete['gender_male'] == 0])}명 (보고: 45명) {'✓' if check1 else '✗'}")
print(f"    남성: {len(master_complete[master_complete['gender_male'] == 1])}명 (보고: 27명) {'✓' if check1_male else '✗'}")

# Check 2: UCLA range
ucla_min = master_complete['ucla_total'].min()
ucla_max = master_complete['ucla_total'].max()
check2 = (20 <= ucla_min <= 30) and (60 <= ucla_max <= 80)

print(f"\n✓ 2. UCLA 점수 범위:")
print(f"    범위: [{ucla_min:.0f}, {ucla_max:.0f}] (예상: 20-80) {'✓' if check2 else '⚠️'}")

# Check 3: WCST range
wcst_min = master_complete['perseverative_error_rate'].min()
wcst_max = master_complete['perseverative_error_rate'].max()
check3 = (0 <= wcst_min <= 10) and (10 <= wcst_max <= 40)

print(f"\n✓ 3. WCST 보속오류율 범위:")
print(f"    범위: [{wcst_min:.1f}%, {wcst_max:.1f}%] (예상: 0-40%) {'✓' if check3 else '⚠️'}")

# Check 4: Missing data
check4 = master_complete[key_vars].isnull().sum().sum() == 0

print(f"\n✓ 4. 결측치:")
print(f"    완전 데이터: {len(master_complete)}명 {'✓' if check4 else '✗'}")

# Check 5: Correlation direction
female_r = verification_df[verification_df['gender'] == '여성']['pearson_r'].values[0]
male_r = verification_df[verification_df['gender'] == '남성']['pearson_r'].values[0]
check5 = (female_r < 0) and (male_r > 0)

print(f"\n✓ 5. 상관 방향:")
print(f"    여성: r = {female_r:.3f} (음수) {'✓' if female_r < 0 else '✗'}")
print(f"    남성: r = {male_r:.3f} (양수) {'✓' if male_r > 0 else '✗'}")

# =============================================================================
# 7. Summary
# =============================================================================

print("\n" + "=" * 80)
print("Step 1 요약")
print("=" * 80)

all_checks = [check1, check1_male, check2, check3, check4, check5]
passed = sum(all_checks)
total = len(all_checks)

print(f"\n데이터 품질 체크: {passed}/{total} 통과")

if passed == total:
    print("\n✅ 모든 데이터 품질 검사 통과")
    print("   → 원시 데이터에 문제 없음")
else:
    print(f"\n⚠️ {total - passed}개 항목에서 주의 필요")

print("\n생성된 파일:")
print(f"  - {VERIFICATION_DIR / 'step1_data_verification.csv'}")
if len(extreme_cases) > 0:
    print(f"  - {VERIFICATION_DIR / 'step1_extreme_cases.csv'}")

print("\n" + "=" * 80)
print("Step 1 완료")
print("=" * 80)
