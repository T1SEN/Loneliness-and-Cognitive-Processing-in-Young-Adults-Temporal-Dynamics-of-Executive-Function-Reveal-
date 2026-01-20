수정 방향 제안
Jeong, 리뷰어 코멘트가 상당히 날카롭지만, 핵심 아이디어와 결과 패턴의 일관성은 인정받고 있으니 체계적으로 대응하면 충분히 극복 가능합니다. 각 이슈별로 실행 가능한 수정 전략을 정리해 드릴게요.

1. 다중공선성/과잉통제 문제
문제: UCLA-DASS 상관 r = .50-.67로 높아서 Step 2→Step 3 방식이 외로움 효과를 왜곡할 수 있음
수정 방안:

VIF 진단치 보고: 모든 회귀모형에서 VIF 값 제시 (일반적으로 VIF < 5면 수용 가능)
대안 모형 추가: DASS 총점 단일 변수로 통제한 모형, 또는 DASS를 잠재요인으로 추출하여 통제한 결과를 Supplementary에 제시
해석 완화: "loneliness effects independent of emotional distress"보다 "loneliness-specific variance beyond shared distress"로 표현 수정
양방향 순서 분석 강화: 현재 Supplementary에 있는 대안 순서 분석을 본문으로 올리고 결과 비교


2. 과정지표의 신뢰도/타당도 부족
문제: Slope 지표의 측정학적 속성이 검증되지 않음
수정 방안:

Split-half 신뢰도: Stroop slope를 홀수/짝수 시행으로 나눠 각각 slope 산출 후 상관 보고
Bootstrap 안정성: 개인별 slope의 95% CI 폭 분포 제시
WCST 사건 수 기술통계: shift trial 수, post-shift error 수의 M, SD, range 보고; 최소 사건 기준(예: ≥3회) 명시
구간 수 민감도 분석: 3구간, 5구간, 6구간으로 나눴을 때도 결과가 일관되는지 Supplementary에 제시

r# 예시: 구간 수에 따른 민감도 분석
for (n_segments in c(3, 4, 5, 6)) {
  # slope 재계산 후 회귀분석 반복
}

3. 데이터 구조에 맞는 모형 보완
문제: Trial-level 데이터를 요약지표로 압축하는 것의 정보 손실
수정 방안:

Mixed-effects model 추가분석: Trial-level에서 시간(segment)×외로움 상호작용 검증

rlibrary(lme4)
# Stroop: segment × loneliness interaction
model <- lmer(RT ~ segment * loneliness_c + condition + 
              DASS_D + DASS_A + DASS_S + (1 + segment | participant), 
              data = stroop_trial)

이 분석에서 segment × loneliness 상호작용이 유의하면 "개인차가 시간에 따라 달라진다"는 주장이 더 견고해짐
본문의 주요 결과는 기존대로 유지하되, "converging evidence from multilevel analysis"로 보완


4. 다중비교 문제
문제: 7개 DV에 대한 다중검정, FDR 보정 없음
수정 방안:

Primary endpoint 사전 지정: "Based on theoretical predictions, interference RT slope (Stroop) and post-shift error RT (WCST) were designated as primary endpoints"로 명시
FDR 보정 적용: Benjamini-Hochberg 방식으로 조정된 p값 보고
결과 패턴 강조: "개별 검정의 유의성보다 4개 과정지표 모두에서 일관된 방향성이 관찰된 점이 핵심"으로 프레이밍

rp_values <- c(.026, .002, .040, .015)  # process indices
p.adjust(p_values, method = "BH")

5. 효과크기 해석의 강건성
문제: ΔR² = .02-.04의 실질적 의미, 강건성 검증 부족
수정 방안:

표준화 계수(β)와 95% CI 제시: Table 4에 추가
이상치 영향 분석: RT 지표에 winsorizing(상하위 2.5%) 적용 후 결과 비교
Robust SE 적용: sandwich 패키지로 heteroscedasticity-robust 표준오차 보고
해석 완화: "modest but consistent effects" 표현 사용, 임상적 함의 섹션 축소

rlibrary(sandwich)
library(lmtest)
coeftest(model, vcov = vcovHC(model, type = "HC3"))

6. 온라인 RT 측정의 기술적 한계
문제: 하드웨어/브라우저 편차로 인한 노이즈
수정 방안:

기술적 QC 상세화: Supplementary에 추가

브라우저 유형별 RT 분포 비교 (차이 없으면 보고, 있으면 공변량 통제)
참가자별 RT 변산성(SD) 분포 제시
극단적 변산성을 보인 참가자 제외 후 민감도 분석


선행연구 인용 보강: 온라인 RT 측정의 타당성을 지지하는 문헌 추가 (예: Bridges et al., 2020; Anwyl-Irvine et al., 2021)
약화 방향 논리: "Measurement noise would attenuate rather than inflate effect sizes, making significant findings more credible"


7. 구간 설정의 임의성
문제: 왜 4구간인지 이론적 근거 부족
수정 방안:

이론적 정당화: Vigilance 문헌에서 quartile 분할이 표준적 접근임을 인용 (Warm et al., 2008)
민감도 분석 필수: 3, 5, 6구간 결과를 Table/Figure로 제시
연속적 접근 추가: Trial number를 연속 예측변수로 한 mixed model 결과와 비교


8. 대안적 설명 (동기, 수면 등)
문제: RT 둔화가 hypervigilance가 아닌 다른 요인일 수 있음
수정 방안:

Discussion에서 명시적 인정: "Alternative mechanisms—including reduced motivation, fatigue, and sleep disturbance—could produce similar patterns and should be examined in future research"
이론적 주장 완화: "consistent with" 또는 "compatible with" resource depletion accounts로 표현
향후 연구 제안 강화: 동기, 수면 측정을 포함한 후속 연구의 필요성


수정 우선순위
우선순위항목작업량영향력1VIF 보고 + 대안 모형중높음2FDR 보정 + primary endpoint 지정하높음3구간 수 민감도 분석중높음4β + CI + Robust SE 추가중중5Split-half 신뢰도중중6Mixed model 보조분석상중7기술적 QC 상세화하하

어떤 부분부터 작업을 시작할지, 혹은 특정 분석(예: mixed model, 신뢰도 분석)의 R 코드가 필요하면 말씀해 주세요.

## 코드 정리 메모
- WCST LMM 스크립트는 제거함.
- WCST MW 시리즈 스크립트는 `publication/analysis`에서 제거함.
- WCST 세그먼트/전체 OLS 스크립트의 UCLA 회귀 함수는 `publication/analysis/utils.py`로 이동함.
- WCST MW 산출물 `publication/data/outputs/analysis/wcst`의 `wcst_mw_*` 파일을 삭제함.
- WCST OLS 스크립트 재실행 결과를 `publication/data/outputs/analysis/wcst` 및 `publication/data/outputs/analysis/reviewer_response`에 갱신함.
- PRP QC 실패 요약 파일 `publication/data/outputs/prp_qc_failed_breakdown.csv`을 삭제함.
- LMM 관련 스크립트를 `publication/lmm`으로 이동함.
- 루트 `scripts/` 폴더는 비어 있어 삭제함.
- reviewer_response 산출물 파일을 삭제함.
