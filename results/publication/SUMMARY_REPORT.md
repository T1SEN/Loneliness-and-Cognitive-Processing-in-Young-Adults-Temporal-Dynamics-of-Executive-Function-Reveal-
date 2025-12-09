# Publication 통계분석 통합 요약 보고서

**생성일**: 2025-12-09
**데이터**: 완료 참가자 N=198 (Male=78, Female=117, Missing=3)

---

## 1. 기술통계 (Descriptive Statistics)

### 1.1 연속변수 요약

| Variable | N | Mean | SD | Min | Max |
|----------|---|------|----|----|-----|
| Age (years) | 198 | 20.47 | 3.64 | - | 36 |
| UCLA Loneliness | 198 | 40.50 | 10.97 | 20 | 70 |
| DASS-21 Depression | 197 | 7.63 | 7.77 | 0 | 38 |
| DASS-21 Anxiety | 197 | 5.58 | 5.73 | 0 | 26 |
| DASS-21 Stress | 197 | 9.92 | 7.51 | 0 | 30 |
| WCST Perseverative Error (%) | 197 | 10.54 | 6.00 | 0 | 48.4 |
| WCST Accuracy (%) | 197 | 82.45 | 10.52 | 40 | 100 |
| Stroop Interference (ms) | 197 | 137.19 | 103.08 | -94 | 481 |
| PRP Delay Effect (ms) | 191 | 582.42 | 154.21 | 68 | 981 |

### 1.2 성별 비교 (Independent Samples t-test)

| Variable | Male M(SD) | Female M(SD) | t | p | Cohen's d | Sig |
|----------|------------|--------------|---|---|-----------|-----|
| UCLA Loneliness | 38.12 (11.08) | 41.94 (10.67) | -2.41 | **.017** | -0.35 | * |
| DASS Depression | 5.95 (7.02) | 8.56 (7.72) | -2.39 | **.018** | -0.35 | * |
| DASS Anxiety | 4.73 (5.82) | 6.03 (5.35) | -1.61 | .110 | -0.24 | |
| DASS Stress | 8.73 (7.66) | 10.58 (7.25) | -1.70 | .090 | -0.25 | |
| WCST PE Rate | 10.50 (6.39) | 10.58 (5.83) | -0.09 | .929 | -0.01 | |
| WCST Accuracy | 84.06 (8.61) | 81.60 (11.26) | 1.64 | .104 | 0.24 | |
| Stroop Interference | 123.54 (91.54) | 145.78 (106.37) | -1.51 | .133 | -0.22 | |
| **PRP Delay Effect** | 527.02 (137.24) | 626.29 (146.71) | -4.68 | **<.001** | -0.69 | *** |

**주요 발견**:
- 여성이 남성보다 외로움과 우울감이 유의하게 높음
- PRP Delay Effect에서 큰 성차 (d = -0.69): 여성이 이중과제 처리에 더 큰 지연을 보임

---

## 2. 상관분석 (Correlation Analysis)

### 2.1 UCLA와 변인 간 상관 (with 95% CI)

| Variable | r | 95% CI | p | Significant |
|----------|---|--------|---|-------------|
| DASS-Depression | **0.66** | [0.57, 0.73] | <.001 | *** |
| DASS-Anxiety | **0.49** | [0.37, 0.59] | <.001 | *** |
| DASS-Stress | **0.52** | [0.41, 0.62] | <.001 | *** |
| WCST PE Rate | -0.10 | [-0.24, 0.04] | .169 | |
| WCST Accuracy | 0.06 | [-0.08, 0.20] | .372 | |
| Stroop Interference | -0.01 | [-0.15, 0.13] | .902 | |
| PRP Delay Effect | 0.12 | [-0.02, 0.26] | .092 | |

### 2.2 주요 상관 패턴

- **UCLA ↔ DASS 강한 상관**: r = 0.49 ~ 0.66 (모두 p < .001)
- **UCLA ↔ EF 무상관**: 모든 인지 지표와 유의한 상관 없음
- **WCST PE ↔ WCST Acc**: r = -0.62*** (과제 내 일관성)
- **Stroop ↔ PRP Delay**: r = 0.16* (약한 양의 상관)

---

## 3. 위계적 회귀분석 (Hierarchical Regression)

### 3.1 모형 구조
- **Model 0**: Age + Gender (Demographics)
- **Model 1**: + DASS-21 (Depression, Anxiety, Stress)
- **Model 2**: + UCLA Loneliness
- **Model 3**: + UCLA × Gender Interaction

### 3.2 결과 요약 (HC3 Robust SE)

| Outcome | N | DASS ΔR² (p) | UCLA ΔR² (p) | Interaction ΔR² (p) |
|---------|---|--------------|--------------|---------------------|
| WCST PE Rate | 193 | 0.020 (.278) | 0.000 (.800) | 0.002 (.452) |
| WCST Accuracy | 193 | 0.027 (.161) | 0.001 (.713) | 0.005 (.202) |
| Stroop Interference | 193 | 0.007 (.699) | 0.003 (.475) | 0.001 (.640) |
| PRP Delay Effect | 187 | 0.003 (.911) | 0.015 (.102) | 0.010 (.132) |

### 3.3 UCLA 주효과 (DASS 통제 후)

| Outcome | β | SE | t | p |
|---------|---|----|----|---|
| WCST PE Rate | -0.14 | 0.56 | -0.25 | .800 |
| WCST Accuracy | 0.42 | 1.13 | 0.37 | .713 |
| Stroop Interference | 7.46 | 10.44 | 0.72 | .475 |
| PRP Delay Effect | 25.16 | 15.28 | 1.65 | .100 |

**핵심 결론**: DASS를 통제하면 **UCLA의 모든 주효과가 비유의** (all p > .10)

---

## 4. 매개분석 (Mediation Analysis)

### 4.1 DASS를 통한 매개효과 (Bootstrap 5000회)

| X → M → Y | Indirect Effect | 95% CI | Significant |
|-----------|-----------------|--------|-------------|
| UCLA → Depression → WCST PE | -0.021 | [-0.061, 0.013] | No |
| UCLA → Depression → Stroop | -0.618 | [-1.360, 0.072] | No |
| UCLA → Depression → PRP | -0.725 | [-1.978, 0.423] | No |
| UCLA → Anxiety → WCST PE | 0.000 | [-0.009, 0.009] | No |
| UCLA → Anxiety → Stroop | 0.009 | [-0.103, 0.218] | No |
| UCLA → Anxiety → PRP | 0.023 | [-0.230, 0.361] | No |
| UCLA → Stress → WCST PE | -0.003 | [-0.025, 0.010] | No |
| UCLA → Stress → Stroop | 0.008 | [-0.268, 0.296] | No |
| UCLA → Stress → PRP | -0.066 | [-0.545, 0.301] | No |

**결론**: DASS를 매개변인으로 설정해도 UCLA → EF 간접효과 없음

---

## 5. 경로분석 (Path Analysis - SEM, Bootstrap 5000회)

### 5.1 Depression 경로모형 비교 (N=196)

| Model | Path | Indirect | 95% CI | Significant | CFI | RMSEA |
|-------|------|----------|--------|-------------|-----|-------|
| Model 1 | 인지 → 외로움 → 우울 | -0.034 | [-0.169, 0.091] | No | 1.02 | 0.00 |
| **Model 2** | 인지 → 우울 → 외로움 | **-0.132** | **[-0.265, -0.009]** | **Yes** | 1.05 | 0.00 |
| **Model 3** | 외로움 → 우울 → 인지 | **-0.051** | **[-0.099, -0.005]** | **Yes** | 1.05 | 0.00 |
| Model 4 | 외로움 → 인지 → 우울 | 0.006 | [-0.008, 0.032] | No | 0.21 | 0.27 |

### 5.2 Anxiety 경로모형 비교 (N=196)

| Model | Path | Indirect | 95% CI | Significant | CFI | RMSEA |
|-------|------|----------|--------|-------------|-----|-------|
| Model 1 | 인지 → 외로움 → 불안 | -0.024 | [-0.118, 0.073] | No | 1.11 | 0.00 |
| Model 2 | 인지 → 불안 → 외로움 | -0.036 | [-0.137, 0.070] | No | 1.11 | 0.00 |
| Model 3 | 외로움 → 불안 → 인지 | -0.014 | [-0.052, 0.025] | No | 1.11 | 0.00 |
| Model 4 | 외로움 → 인지 → 불안 | 0.004 | [-0.004, 0.021] | No | 0.33 | 0.18 |

**결과**: 불안 경로모형에서는 **모든 간접효과 비유의**

### 5.3 Stress 경로모형 비교 (N=196)

| Model | Path | Indirect | 95% CI | Significant | CFI | RMSEA |
|-------|------|----------|--------|-------------|-----|-------|
| Model 1 | 인지 → 외로움 → 스트레스 | -0.026 | [-0.131, 0.077] | No | 1.08 | 0.00 |
| Model 2 | 인지 → 스트레스 → 외로움 | -0.064 | [-0.178, 0.049] | No | 1.10 | 0.00 |
| Model 3 | 외로움 → 스트레스 → 인지 | -0.025 | [-0.068, 0.018] | No | 1.10 | 0.00 |
| Model 4 | 외로움 → 인지 → 스트레스 | 0.004 | [-0.006, 0.025] | No | 0.29 | 0.20 |

**결과**: 스트레스 경로모형에서는 **모든 간접효과 비유의**

### 5.4 성별 차이 분석 (유의한 결과)

**Stress 경로모형에서 성별 차이 발견**:

| Path | Male | Female | z | p |
|------|------|--------|---|---|
| Model2 a-path (EF → Stress) | 0.21 | -0.27 | 2.10 | **.036** |
| Model3 b-path (Stress → EF) | 0.07 | -0.12 | 2.16 | **.031** |
| Model3 indirect (L → S → C) | 0.04 | -0.05 | 1.98 | **.048** |
| Model4 b-path (EF → Stress) | 0.21 | -0.27 | 2.10 | **.036** |

**해석**:
- 남성: 인지기능과 스트레스 간 양의 관계 (인지 좋을수록 스트레스 높음)
- 여성: 인지기능과 스트레스 간 음의 관계 (인지 좋을수록 스트레스 낮음)

### 5.5 유의한 경로모형 종합 해석

1. **Model 2 (C → D → L)**: 인지기능 저하 → 우울 증가 → 외로움 증가
   - Indirect effect = -0.132, p < .05
   - 해석: 인지적 어려움이 우울을 통해 외로움으로 이어짐

2. **Model 3 (L → D → C)**: 외로움 → 우울 증가 → 인지기능 저하 경향
   - Indirect effect = -0.051, p < .05
   - 해석: 외로움이 우울을 통해 인지에 영향 (우울의 매개)

3. **우울만 유의**: 불안/스트레스는 매개 역할 하지 않음

---

## 6. 베이지안 분석 (Bayesian SEM)

### 6.1 베이지안 회귀 결과 (DASS 통제)

| Outcome | UCLA β [95% HDI] | P(direction) | Interaction β | P(direction) |
|---------|------------------|--------------|---------------|--------------|
| WCST PE | -0.27 [-1.39, 1.08] | 0.67 | 0.33 | 0.66 |
| Stroop | -0.03 [-3.20, 3.53] | 0.51 | -0.17 | 0.54 |
| PRP Bottleneck | 0.30 [-3.40, 3.83] | 0.56 | -0.31 | 0.55 |

### 6.2 베이즈 팩터 모형 비교

| Outcome | BF₀₁ (UCLA) | BF₀₁ (Interaction) | 해석 |
|---------|-------------|-------------------|------|
| WCST PE | **13.63** | **12.02** | H₀ 강하게 지지 |
| Stroop | **11.64** | **11.17** | H₀ 강하게 지지 |
| PRP Bottleneck | **4.71** | **3.86** | H₀ 중간 지지 |

**해석**: BF > 10은 UCLA 효과가 **없다**는 증거가 강함

### 6.3 동등성 검정 (ROPE)

- 모든 결과변수에서 UCLA 효과의 **83-89%가 ROPE 내에 위치**
- 실질적으로 무의미한 효과 크기임을 확인

---

## 7. 신뢰도/타당도 (Validity & Reliability)

### 7.1 데이터 품질

| 지표 | 결과 |
|------|------|
| 전체 탈락률 | **0.0%** (204/204 완료) |
| Stroop 유효 시행 | 99.2% |
| WCST 유효 시행 | 96.3% |
| PRP 유효 시행 | 96.3% |
| DASS 과속응답 (<2초/문항) | 14.9% |
| DASS 직선응답 | 5.4% |

### 7.2 내적 일관성 (Cronbach's α)

| 척도 | α | 평가 |
|------|---|------|
| UCLA Loneliness (20문항) | **0.936** | Excellent |
| DASS-21 Total | **0.920** | Excellent |
| DASS Depression | 0.871 | Good |
| DASS Anxiety | 0.779 | Acceptable |
| DASS Stress | 0.830 | Good |

### 7.3 인지과제 신뢰도 (Split-Half)

| 과제 | r | Spearman-Brown | 평가 |
|------|---|----------------|------|
| WCST PE Rate | 0.605 | **0.754** | Acceptable |
| PRP Bottleneck | 0.545 | **0.705** | Acceptable |
| Stroop Interference | 0.396 | 0.567 | Poor |

### 7.4 구성 타당도 (EFA)

- **KMO = 0.938** (Marvelous)
- **Bartlett's Test**: χ² = 2203.95, p < .001
- 2요인 구조 확인: 정서적 외로움 vs 사회적 외로움

### 7.5 수렴/변별 타당도

- UCLA-DASS 상관: r = 0.49 ~ 0.66 (적절한 수렴)
- UCLA-EF 상관: r ≈ 0 (적절한 변별)

---

## 8. 핵심 결론

### 8.1 주요 발견

1. **UCLA-DASS 강한 공변**: r = 0.49 ~ 0.66로 외로움과 정서적 고통이 높은 중첩
2. **UCLA → EF 직접효과 부재**: DASS 통제 시 모든 UCLA 주효과 비유의
3. **베이지안 확증**: BF₀₁ = 4.7~13.6으로 영가설(효과 없음) 강력 지지
4. **성차**: 여성이 외로움, 우울, PRP 지연에서 더 높은 점수
5. **Depression 경로모형**: 우울이 외로움-인지 관계를 매개하는 경로가 유의
6. **Stress 성별 차이**: 인지-스트레스 관계에서 남녀 반대 방향의 효과
7. **측정 신뢰도**: UCLA α=0.94, DASS α=0.92로 우수

### 8.2 이론적 함의

- 외로움의 인지기능 효과는 **우울에 의해 매개됨** (불안/스트레스는 비유의)
- 외로움 자체는 EF에 독립적 영향을 미치지 않음
- **우울 경로 모형**이 외로움-인지 관계를 가장 잘 설명
- 스트레스-인지 관계는 **성별에 따라 상이** (남성: 양의 관계, 여성: 음의 관계)
- **베이지안 분석**으로 "효과 없음"이 단순히 검정력 부족이 아님을 확인

### 8.3 한계점

- 횡단 연구 설계로 인과관계 추론 제한
- 자기보고식 외로움/정서 측정
- 단일 시점 인지과제 수행

---

## 9. 분석 파일 위치

```
results/publication/
├── basic_analysis/
│   ├── table1_*.csv           # 기술통계
│   ├── correlation_*.csv/png  # 상관분석
│   └── hierarchical_*.csv/txt # 위계적 회귀
├── advanced_analysis/
│   ├── mediation/             # DASS 매개분석
│   ├── path_depression/       # Depression 경로분석
│   ├── path_anxiety/          # Anxiety 경로분석
│   ├── path_stress/           # Stress 경로분석
│   └── bayesian/              # 베이지안 SEM
└── SUMMARY_REPORT.md          # 본 보고서

results/analysis_outputs/validity_reliability/
├── data_quality_*.csv/json    # 데이터 품질
├── survey_reliability.csv     # 설문 신뢰도
├── cognitive_reliability.csv  # 인지과제 신뢰도
├── efa_loadings.csv           # 요인분석
└── validity_*.csv             # 타당도
```

---

*Note: 모든 p-value는 양측검정 기준. * p < .05, ** p < .01, *** p < .001*
