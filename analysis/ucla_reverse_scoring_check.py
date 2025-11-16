#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCLA Loneliness Scale 역점수 확인 및 수정

UCLA-20의 역점수 문항: 1, 5, 6, 9, 10, 15, 16, 19, 20
(Russell, 1996; https://fetzer.org/sites/default/files/images/stories/pdf/selfmeasures/Self_Measures_for_Loneliness_and_Interpersonal_Problems_UCLA_LONELINESS.pdf)

목표:
1. 현재 score가 제대로 역점수 적용되었는지 검증
2. 문항별 점수 분포 확인 (역점수 오류 감지)
3. 필요시 올바른 총점 재계산
4. Factor analysis용 개별 문항 점수 저장
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# UTF-8 설정
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# 경로 설정
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/ucla_scoring_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("UCLA LONELINESS SCALE 역점수 검증 및 수정")
print("="*80)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1] 데이터 로딩...")
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')

# UCLA만 필터링
ucla = surveys[surveys['surveyName'] == 'ucla'].copy()
print(f"   총 UCLA 응답: {len(ucla)}개")

# ============================================================================
# 2. UCLA 역점수 문항 정의
# ============================================================================
# UCLA-20 역점수 문항 (1-indexed, 따라서 q1, q5, q6, q9, q10, q15, q16, q19, q20)
REVERSE_ITEMS = [1, 5, 6, 9, 10, 15, 16, 19, 20]
FORWARD_ITEMS = [2, 3, 4, 7, 8, 11, 12, 13, 14, 17, 18]

print(f"\n[2] UCLA-20 척도 구조")
print(f"   역점수 문항 (N=9): {REVERSE_ITEMS}")
print(f"   정방향 문항 (N=11): {FORWARD_ITEMS}")

# ============================================================================
# 3. 현재 점수 검증
# ============================================================================
print(f"\n[3] 현재 저장된 점수 검증...")

# 문항별 컬럼 추출
item_cols = [f'q{i}' for i in range(1, 21)]
ucla_items = ucla[['participantId'] + item_cols + ['score']].copy()

# 결측치 제거
ucla_items = ucla_items.dropna(subset=item_cols + ['score'])
print(f"   유효 응답: {len(ucla_items)}개")

# 현재 총점 분포
current_score = ucla_items['score']
print(f"\n   현재 score 분포:")
print(f"   - 평균: {current_score.mean():.2f}")
print(f"   - 표준편차: {current_score.std():.2f}")
print(f"   - 범위: {current_score.min():.0f} ~ {current_score.max():.0f}")
print(f"   - 중앙값: {current_score.median():.0f}")

# ============================================================================
# 4. 역점수 적용 여부 확인 (3가지 시나리오 검증)
# ============================================================================
print(f"\n[4] 역점수 적용 여부 확인...")

# Scenario A: 역점수 이미 적용됨 (현재 score 그대로)
# Scenario B: 역점수 미적용 (raw sum)
# Scenario C: 역점수 반대로 적용됨 (정방향을 역점수 처리)

# Scenario A: 현재 score (역점수 이미 적용 가정)
scenario_a = current_score.copy()

# Scenario B: 역점수 미적용 (raw sum)
scenario_b = ucla_items[item_cols].sum(axis=1)

# Scenario C: 올바른 역점수 적용
# 역점수 문항: 5 - raw_score
# 정방향 문항: 그대로
scenario_c = pd.Series(0, index=ucla_items.index)

for i in range(1, 21):
    col = f'q{i}'
    if i in REVERSE_ITEMS:
        # 역점수: 5 - raw
        scenario_c += (5 - ucla_items[col])
    else:
        # 정방향: 그대로
        scenario_c += ucla_items[col]

# 각 시나리오 비교
print(f"\n   시나리오별 총점 비교:")
print(f"   A (현재 score 그대로):        평균={scenario_a.mean():.2f}, SD={scenario_a.std():.2f}")
print(f"   B (역점수 미적용, raw sum):   평균={scenario_b.mean():.2f}, SD={scenario_b.std():.2f}")
print(f"   C (올바른 역점수 적용):       평균={scenario_c.mean():.2f}, SD={scenario_c.std():.2f}")

# 시나리오 간 상관
corr_ab = pearsonr(scenario_a, scenario_b)[0]
corr_ac = pearsonr(scenario_a, scenario_c)[0]
corr_bc = pearsonr(scenario_b, scenario_c)[0]

print(f"\n   시나리오 간 상관:")
print(f"   A vs B: r = {corr_ab:.4f}")
print(f"   A vs C: r = {corr_ac:.4f}")
print(f"   B vs C: r = {corr_bc:.4f}")

# 어떤 시나리오가 현재 score와 일치하는지
diff_ab = (scenario_a - scenario_b).abs().mean()
diff_ac = (scenario_a - scenario_c).abs().mean()

print(f"\n   현재 score와의 평균 차이:")
print(f"   A vs B: {diff_ab:.2f}")
print(f"   A vs C: {diff_ac:.2f}")

if diff_ac < 0.01:
    print(f"\n   ✓ 결론: 현재 score는 **이미 올바른 역점수가 적용됨**")
    correct_score = scenario_a
    needs_correction = False
elif diff_ab < 0.01:
    print(f"\n   ✗ 경고: 현재 score는 **역점수가 미적용됨** (raw sum)")
    correct_score = scenario_c
    needs_correction = True
else:
    print(f"\n   ✗ 경고: 현재 score의 계산 방식이 불명확함 (추가 조사 필요)")
    correct_score = scenario_c
    needs_correction = True

# ============================================================================
# 5. 문항별 분포 확인 (역점수 오류 감지)
# ============================================================================
print(f"\n[5] 문항별 분포 확인 (역점수 오류 감지)...")

item_stats = []
for i in range(1, 21):
    col = f'q{i}'
    item_type = "Reverse" if i in REVERSE_ITEMS else "Forward"
    mean_val = ucla_items[col].mean()
    std_val = ucla_items[col].std()
    min_val = ucla_items[col].min()
    max_val = ucla_items[col].max()

    item_stats.append({
        'item': i,
        'type': item_type,
        'mean': mean_val,
        'sd': std_val,
        'min': min_val,
        'max': max_val
    })

df_item_stats = pd.DataFrame(item_stats)

# 역점수 vs 정방향 평균 비교
reverse_mean = df_item_stats[df_item_stats['type'] == 'Reverse']['mean'].mean()
forward_mean = df_item_stats[df_item_stats['type'] == 'Forward']['mean'].mean()

print(f"\n   문항 유형별 평균:")
print(f"   - 역점수 문항 평균: {reverse_mean:.2f}")
print(f"   - 정방향 문항 평균: {forward_mean:.2f}")
print(f"   - 차이: {abs(reverse_mean - forward_mean):.2f}")

# 이론적으로 역점수와 정방향은 비슷한 평균을 가져야 함 (external loneliness 측정)
# 만약 역점수 평균 >> 정방향 평균이면, 역점수가 안 되어 있을 가능성
if reverse_mean > forward_mean + 0.5:
    print(f"   ⚠ 경고: 역점수 문항 평균이 정방향보다 높음 (역점수 미적용 가능성)")
elif forward_mean > reverse_mean + 0.5:
    print(f"   ⚠ 경고: 정방향 문항 평균이 역점수보다 높음 (역점수 과다 적용 가능성)")
else:
    print(f"   ✓ 정상: 두 유형의 평균이 유사함 (역점수 적절히 적용됨)")

# ============================================================================
# 6. 올바른 UCLA 총점 저장
# ============================================================================
print(f"\n[6] 올바른 UCLA 총점 저장...")

# 올바른 점수로 데이터프레임 구성
ucla_corrected = ucla_items[['participantId']].copy()
ucla_corrected['ucla_total_original'] = scenario_a
ucla_corrected['ucla_total_corrected'] = correct_score
ucla_corrected['needs_correction'] = needs_correction

if needs_correction:
    print(f"   역점수 수정 적용됨")
else:
    print(f"   기존 점수가 정확하므로 수정 불필요")

# 저장
output_file = OUTPUT_DIR / "ucla_score_corrected.csv"
ucla_corrected.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   저장: {output_file}")

# ============================================================================
# 7. Factor Analysis용 역점수 적용 문항 저장
# ============================================================================
print(f"\n[7] Factor Analysis용 역점수 적용 문항 저장...")

# 모든 문항을 역점수 적용 후 저장
ucla_items_reversed = ucla_items[['participantId']].copy()

for i in range(1, 21):
    col = f'q{i}'
    if i in REVERSE_ITEMS:
        # 역점수 적용
        ucla_items_reversed[col] = 5 - ucla_items[col]
    else:
        # 정방향 그대로
        ucla_items_reversed[col] = ucla_items[col]

# UCLA 총점 추가
ucla_items_reversed['ucla_total'] = correct_score

# 저장
output_file2 = OUTPUT_DIR / "ucla_items_reversed.csv"
ucla_items_reversed.to_csv(output_file2, index=False, encoding='utf-8-sig')
print(f"   저장: {output_file2}")

# ============================================================================
# 8. 기존 분석과의 비교 (WCST PE와의 상관)
# ============================================================================
print(f"\n[8] 기존 분석과의 비교 (WCST PE와의 상관)...")

try:
    # WCST 데이터 로드
    cognitive = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv", encoding='utf-8-sig')

    # WCST만 필터링
    wcst = cognitive[cognitive['testName'] == 'wcst'].copy()

    # PE rate 계산
    wcst['pe_rate'] = (wcst['perseverativeErrorCount'] / wcst['totalTrialCount']) * 100

    # Merge
    merged = ucla_corrected.merge(participants[['participantId', 'gender']], on='participantId', how='left')
    merged = merged.merge(wcst[['participantId', 'pe_rate']], on='participantId', how='left')
    merged = merged.dropna(subset=['pe_rate', 'gender'])

    # 성별 분리
    males = merged[merged['gender'] == '남성']
    females = merged[merged['gender'] == '여성']

    # 상관 계산
    if len(males) > 10:
        r_male_orig, p_male_orig = pearsonr(males['ucla_total_original'], males['pe_rate'])
        r_male_corr, p_male_corr = pearsonr(males['ucla_total_corrected'], males['pe_rate'])

        print(f"\n   남성 (N={len(males)}):")
        print(f"   - Original UCLA vs PE: r = {r_male_orig:+.3f}, p = {p_male_orig:.4f}")
        print(f"   - Corrected UCLA vs PE: r = {r_male_corr:+.3f}, p = {p_male_corr:.4f}")

        if needs_correction:
            print(f"   → 상관 방향 변화: {r_male_orig:+.3f} → {r_male_corr:+.3f}")

    if len(females) > 10:
        r_fem_orig, p_fem_orig = pearsonr(females['ucla_total_original'], females['pe_rate'])
        r_fem_corr, p_fem_corr = pearsonr(females['ucla_total_corrected'], females['pe_rate'])

        print(f"\n   여성 (N={len(females)}):")
        print(f"   - Original UCLA vs PE: r = {r_fem_orig:+.3f}, p = {p_fem_orig:.4f}")
        print(f"   - Corrected UCLA vs PE: r = {r_fem_corr:+.3f}, p = {p_fem_corr:.4f}")

        if needs_correction:
            print(f"   → 상관 방향 변화: {r_fem_orig:+.3f} → {r_fem_corr:+.3f}")

except Exception as e:
    print(f"   ⚠ WCST 데이터 비교 실패: {e}")

# ============================================================================
# 9. 문항별 통계 저장
# ============================================================================
output_file3 = OUTPUT_DIR / "ucla_item_statistics.csv"
df_item_stats.to_csv(output_file3, index=False, encoding='utf-8-sig')
print(f"\n[9] 문항별 통계 저장: {output_file3}")

# ============================================================================
# 10. 요약 보고서
# ============================================================================
report_file = OUTPUT_DIR / "UCLA_SCORING_CHECK_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("UCLA LONELINESS SCALE 역점수 검증 보고서\n")
    f.write("="*80 + "\n\n")

    f.write(f"분석 일자: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"유효 응답 수: {len(ucla_items)}\n\n")

    f.write("1. 역점수 문항 정의\n")
    f.write(f"   - 역점수 문항 (N=9): {REVERSE_ITEMS}\n")
    f.write(f"   - 정방향 문항 (N=11): {FORWARD_ITEMS}\n\n")

    f.write("2. 점수 검증 결과\n")
    f.write(f"   - 현재 score 평균: {scenario_a.mean():.2f} (SD={scenario_a.std():.2f})\n")
    f.write(f"   - 올바른 점수 평균: {correct_score.mean():.2f} (SD={correct_score.std():.2f})\n")
    f.write(f"   - 평균 차이: {diff_ac:.2f}\n\n")

    if needs_correction:
        f.write("   ✗ 결론: 역점수가 제대로 적용되지 않았음 → 수정 필요\n\n")
    else:
        f.write("   ✓ 결론: 역점수가 이미 올바르게 적용되어 있음\n\n")

    f.write("3. 문항별 분포\n")
    f.write(f"   - 역점수 문항 평균: {reverse_mean:.2f}\n")
    f.write(f"   - 정방향 문항 평균: {forward_mean:.2f}\n")
    f.write(f"   - 차이: {abs(reverse_mean - forward_mean):.2f}\n\n")

    f.write("4. 산출물\n")
    f.write(f"   - ucla_score_corrected.csv: 수정된 UCLA 총점\n")
    f.write(f"   - ucla_items_reversed.csv: Factor analysis용 역점수 적용 문항\n")
    f.write(f"   - ucla_item_statistics.csv: 문항별 기술 통계\n\n")

    f.write("="*80 + "\n")

print(f"\n[10] 요약 보고서 저장: {report_file}")

print("\n" + "="*80)
print("UCLA 역점수 검증 완료!")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print("="*80)
