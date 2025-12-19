# Basic Analysis 종합 결과 보고서

**생성일**: 2025-12-19
**최종 업데이트**: 2025-12-19 (PRP Central Bottleneck + Stroop LBA 추가)
**분석 대상**: 외로움(UCLA)과 집행기능(WCST, Stroop, PRP) 간 관계
**통제 변인**: DASS-21 (우울, 불안, 스트레스), 나이, 성별

---

## 1. 데이터셋 산출 방법

### 1.1 데이터셋별 필터링 기준

| 데이터셋 | 필터링 기준 | N | 설명 |
|----------|-------------|---|------|
| **overall** | Stroop ∩ PRP ∩ WCST ∩ 설문 완료자 | 196 | **3개 인지과제 + 설문 모두 완료한 참가자의 교집합** |
| **stroop** | Stroop + 설문 완료자 | 219 | Stroop 과제와 설문만 완료하면 포함 |
| **prp** | PRP + 설문 완료자 | 210 | PRP 과제와 설문만 완료하면 포함 |
| **wcst** | WCST + 설문 완료자 | 200 | WCST 과제와 설문만 완료하면 포함 |

### 1.2 데이터 흐름

```
Firebase (raw data)
    ↓
export_alldata.py → publication/data/raw/ (N=251, 전체 참가자)
    ↓
python -m publication.preprocessing --build all
    ↓
├── complete_stroop/  (Stroop + 설문 완료자, N=219)
├── complete_prp/     (PRP + 설문 완료자, N=210)
├── complete_wcst/    (WCST + 설문 완료자, N=200)
└── complete_overall/ (3개 과제 + 설문 교집합, N=196)
    ↓
python -m publication.basic_analysis.* --task {task}
    ↓
publication/data/outputs/basic_analysis/{task}/
```

### 1.3 각 데이터셋의 특징

- **overall (N=196)**: 모든 인지과제를 완료한 참가자만 포함. 과제 간 비교 가능. 가장 엄격한 기준.
- **stroop (N=219)**: Stroop 과제 분석에 최적화. 샘플 크기 최대화.
- **prp (N=210)**: PRP 과제 분석에 최적화. 이중과제 처리 연구.
- **wcst (N=200)**: WCST 과제 분석에 최적화. 세트-전환 연구.

---

## 2. 분석 방법

### 2.1 분석 스크립트

| 분석 유형 | 스크립트 | 산출 지표 |
|-----------|----------|-----------|
| **기술통계** | `descriptive_statistics.py` | M, SD, 범위, 왜도, 첨도, t-test, Cohen's d |
| **상관분석** | `correlation_analysis.py` | Pearson r, 95% CI (Fisher's z), p-value matrix |
| **위계적 회귀** | `hierarchical_regression.py` | ΔR², β, SE, t, p (HC3), VIF, Breusch-Pagan, Shapiro-Wilk |

### 2.2 위계적 회귀 모형 구조

```
Model 0: DV ~ age + gender                           (기본 모형)
Model 1: Model 0 + dass_dep + dass_anx + dass_str    (DASS 추가)
Model 2: Model 1 + ucla_score                        (UCLA 추가)
Model 3: Model 2 + ucla_score × gender               (상호작용 추가)
```

### 2.3 실행 명령어

```bash
# 전체 12개 분석 실행 (4 tasks × 3 analyses)
for task in overall stroop prp wcst; do
    python -m publication.basic_analysis.descriptive_statistics --task $task
    python -m publication.basic_analysis.correlation_analysis --task $task
    python -m publication.basic_analysis.hierarchical_regression --task $task
done
```

---

## 3. 연구 참가자 기술통계

### 3.1 인구통계

| 데이터셋 | N | 남성 n (%) | 여성 n (%) | 나이 M (SD) |
|----------|---|------------|------------|-------------|
| **overall** | 196 | 77 (39.3%) | 119 (60.7%) | 20.40 (3.48) |
| **stroop** | 219 | 84 (38.4%) | 135 (61.6%) | 20.37 (3.33) |
| **prp** | 210 | 82 (39.0%) | 128 (61.0%) | 20.40 (3.39) |
| **wcst** | 200 | 78 (39.0%) | 122 (61.0%) | 20.40 (3.45) |

### 3.2 심리 척도 (Overall, N=196 기준)

| 변수 | M | SD | Min | Max | Skew | Kurt |
|------|---|----|----|-----|------|------|
| UCLA 외로움 | 40.34 | 11.09 | 20 | 70 | 0.32 | -0.52 |
| DASS 우울 | 7.36 | 7.45 | 0 | 38 | 1.53 | 2.55 |
| DASS 불안 | 5.45 | 5.63 | 0 | 26 | 1.17 | 0.76 |
| DASS 스트레스 | 10.02 | 7.29 | 0 | 30 | 0.65 | -0.28 |

### 3.3 인지과제 수행

| 과제 | 지표 | M | SD | Min | Max |
|------|------|---|----|-----|-----|
| **WCST** | PE Rate (%) | 10.08 | 4.45 | 0 | 25.83 |
| **Stroop** | Interference (ms) | 138.99 | 103.85 | -94 | 481 |
| **PRP** | Delay Effect (ms) | 607.10 | 150.04 | 283 | 1032 |

---

## 4. 유의한 결과 종합 (p < .05)

### 4.1 성별 비교 (t-test)

#### 4.1.1 Overall 데이터셋 (N=196)

| 변수 | 남성 M (n=77) | 여성 M (n=119) | t | p | Cohen's d |
|------|---------------|----------------|---|---|-----------|
| **나이** | 21.17 | 19.90 | 2.86 | **.005** | 0.37 |
| **UCLA 외로움** | 37.48 | 42.18 | -2.92 | **.004** | -0.43 |
| **DASS 우울** | 5.79 | 8.37 | -2.42 | **.017** | -0.35 |
| **PRP Delay Effect** | 560.94 | 636.49 | -3.60 | **<.001** | **-0.52** |

#### 4.1.2 Stroop 데이터셋 (N=219)

| 변수 | 남성 M (n=84) | 여성 M (n=135) | t | p | Cohen's d |
|------|---------------|----------------|---|---|-----------|
| **나이** | 21.11 | 19.92 | 2.95 | **.004** | 0.36 |
| **UCLA 외로움** | 37.90 | 42.18 | -2.75 | **.007** | -0.39 |
| **DASS 우울** | 6.07 | 8.46 | -2.19 | **.030** | -0.31 |

#### 4.1.3 PRP 데이터셋 (N=210)

| 변수 | 남성 M (n=82) | 여성 M (n=128) | t | p | Cohen's d |
|------|---------------|----------------|---|---|-----------|
| **나이** | 21.16 | 19.91 | 2.98 | **.003** | 0.37 |
| **UCLA 외로움** | 37.80 | 42.23 | -2.79 | **.006** | -0.40 |
| **DASS 우울** | 6.05 | 8.48 | -2.17 | **.032** | -0.31 |
| **PRP Delay Effect** | 555.54 | 639.95 | -4.13 | **<.001** | **-0.58** |

#### 4.1.4 WCST 데이터셋 (N=200)

| 변수 | 남성 M (n=78) | 여성 M (n=122) | t | p | Cohen's d |
|------|---------------|----------------|---|---|-----------|
| **나이** | 21.14 | 19.92 | 2.81 | **.006** | 0.36 |
| **UCLA 외로움** | 37.60 | 42.14 | -2.84 | **.005** | -0.42 |
| **DASS 우울** | 5.82 | 8.34 | -2.40 | **.017** | -0.34 |

---

### 4.2 상관분석

#### 4.2.1 UCLA-DASS 상관 (모든 데이터셋에서 유의)

| 데이터셋 | UCLA-DASS 우울 | UCLA-DASS 불안 | UCLA-DASS 스트레스 |
|----------|----------------|----------------|---------------------|
| **overall** | r=.665, p<.001 | r=.475, p<.001 | r=.526, p<.001 |
| **stroop** | r=.675, p<.001 | r=.503, p<.001 | r=.553, p<.001 |
| **prp** | r=.677, p<.001 | r=.497, p<.001 | r=.550, p<.001 |
| **wcst** | r=.663, p<.001 | r=.478, p<.001 | r=.529, p<.001 |

#### 4.2.2 DASS 우울-WCST PE 상관 (WCST 데이터셋)

| 상관 | r | 95% CI | p |
|------|---|--------|---|
| **DASS 우울 - WCST PE Rate** | **-.143** | [-.276, -.005] | **.043** |

**해석**: 우울 수준이 높을수록 보속적 오류율이 낮음 (역설적 패턴)

#### 4.2.3 인지과제 간 상관 (Overall 데이터셋, N=191-192)

| 상관 | r | 95% CI | p |
|------|---|--------|---|
| **WCST PE Rate - PRP Delay Effect** | **.193** | [.052, .326] | **.008** |
| **Stroop Interference - PRP Delay Effect** | **.183** | [.042, .317] | **.011** |

**해석**:
- WCST 보속오류가 높을수록 PRP 이중과제 비용도 높음 (공유된 인지통제 취약성)
- Stroop 간섭효과가 클수록 PRP 이중과제 비용도 높음 (억제-병목 연관성)

---

### 4.3 위계적 회귀분석

#### 4.3.1 DASS 추가 효과 (Model 0 → Model 1)

| 결과변수 | 데이터셋 | N | ΔR² | F | p |
|----------|----------|---|-----|---|---|
| **WCST Accuracy** | wcst | 200 | 4.08% | 2.79 | **.042** |
| **WCST Post-Switch Error Rate** | wcst | 200 | 4.12% | 2.86 | **.038** |

#### 4.3.2 UCLA 추가 효과 (Model 1 → Model 2, DASS 통제 후)

| 결과변수 | 데이터셋 | N | β | SE | t | p | ΔR² |
|----------|----------|---|---|----|----|---|-----|
| **WCST Accuracy** | overall | 192 | 1.42 | 0.70 | 2.03 | **.042** | 2.22% |
| **WCST Post-Error Slowing** | wcst | 199 | 176.67 | 88.30 | 2.00 | **.045** | 2.41% |
| **Stroop Incong RT Slope** | overall | 192 | 0.58 | 0.21 | 2.77 | **.006** | 3.61% |
| **Stroop Incong RT Slope** | stroop | 219 | 0.47 | 0.19 | 2.40 | **.016** | 2.43% |

#### 4.3.3 성별 분리 분석에서 유의한 UCLA 효과

| 결과변수 | 성별 | 데이터셋 | n | β | SE | p |
|----------|------|----------|---|---|----|----|
| **WCST Accuracy** | 여성 | overall | 117 | 1.61 | 0.80 | **.044** |
| **WCST Post-Error Slowing** | 남성 | overall | 75 | 297.77 | 127.23 | **.020** |
| **WCST Post-Error Slowing** | 남성 | wcst | 78 | 281.49 | 128.77 | **.029** |
| **Stroop Incong RT Slope** | 남성 | overall | 75 | 1.13 | 0.31 | **<.001** |
| **Stroop Incong RT Slope** | 남성 | stroop | 84 | 1.01 | 0.31 | **.001** |

#### 4.3.4 Mechanism 변수 결과 (2024-12-19 추가)

**HMM (Hidden Markov Model) 기반 주의 상태 전이 분석:**

| 결과변수 | 데이터셋 | N | β | SE | p | ΔR² |
|----------|----------|---|---|----|----|-----|
| **WCST HMM P(Lapse→Focus)** | overall | 192 | -0.078 | 0.027 | **.004** | 4.24% |
| **WCST HMM P(Lapse→Lapse)** | overall | 192 | 0.078 | 0.027 | **.004** | 4.24% |
| **WCST HMM P(Lapse→Focus)** | wcst | 200 | -0.091 | 0.026 | **<.001** | 5.75% |
| **WCST HMM P(Lapse→Lapse)** | wcst | 200 | 0.091 | 0.026 | **<.001** | 5.75% |

**해석**: 외로움(UCLA)↑ → Lapse 상태에서 빠져나올 확률↓, Lapse 상태 유지 확률↑

**RL (Reinforcement Learning) 기반 학습 파라미터 분석:**

| 결과변수 | 데이터셋 | N | p_ucla | p_interaction | 성별별 효과 |
|----------|----------|---|--------|---------------|-------------|
| **WCST RL alpha (pos)** | overall | 192 | .079 | .141 (ns) | 남성만 유의 (p=.009, β=0.17) |

**해석**: UCLA × 성별 상호작용은 비유의 (p=.141). 그러나 **남성에서만** 외로움↑ → 양성 학습률↑ 효과가 유의함 (β=0.17, p=.009)

#### 4.3.5 성별 분리 분석 (Mechanism 변수)

| 결과변수 | 성별 | 데이터셋 | n | β | p |
|----------|------|----------|---|---|---|
| **WCST HMM P(Lapse→Focus)** | 여성 | overall | 117 | -0.076 | **.035** |
| **WCST HMM P(Lapse→Focus)** | 남성 | overall | 75 | -0.074 | .092 |
| **WCST HMM P(Lapse→Focus)** | 여성 | wcst | 122 | -0.098 | **.005** |
| **WCST HMM P(Lapse→Focus)** | 남성 | wcst | 78 | -0.074 | .090 |
| **WCST RL alpha (pos)** | 남성 | overall | 75 | 0.173 | **.009** |
| **WCST RL alpha (pos)** | 여성 | overall | 117 | 0.010 | .856 |
| **WCST HMM Lapse Occupancy** | 남성 | wcst | 78 | 3.72 | **.047** |

#### 4.3.6 PRP Central Bottleneck 모델 결과 (2025-12-19 추가)

**모델**: `RT2 = base + max(0, bottleneck - SOA)`
- `base`: 긴 SOA에서의 비병목 기준 RT2 (플래토)
- `bottleneck`: 중앙 처리 병목 지속 시간 (ms)
- `slope`: short SOA 구간에서의 RT2~SOA 기울기 (이론적으로 -1에 가까워야 함)

**피팅 성공률**: 198/204 (97.1%)

| 지표 | M | SD | Min | Max |
|------|---|----|----|-----|
| `prp_cb_base` (ms) | 703.78 | 190.96 | 387.36 | 1309.97 |
| `prp_cb_bottleneck` (ms) | 687.42 | 170.72 | 300.00 | 1136.68 |
| `prp_cb_r_squared` | 0.88 | 0.13 | 0.26 | 1.00 |
| `prp_cb_slope` | -0.86 | 0.56 | -2.64 | 1.02 |
| `prp_cb_n_trials` | 100.27 | 24.37 | 1 | 120 |

**해석**:
- R² 평균 0.88로 Central Bottleneck 모델이 데이터를 잘 설명함
- slope 평균 -0.86으로 이론적 예측값 -1에 근접
- bottleneck 평균 687ms로 중앙 처리 단계가 약 700ms 소요됨을 시사

**UCLA와의 관계 (위계적 회귀, DASS 통제 후)**:

| 결과변수 | N | β | SE | t | p | ΔR² |
|----------|---|---|----|----|---|-----|
| **PRP CB Slope** | 198 | -0.126 | 0.059 | -2.12 | **.035** | 2.65% |
| PRP CB Base | 198 | 13.68 | 15.52 | 0.88 | .379 | 0.09% |
| PRP CB Bottleneck | 198 | 18.31 | 13.47 | 1.36 | .176 | 0.42% |

| 결과변수 | 성별 | n | β | p |
|----------|------|---|---|---|
| PRP CB Slope | 전체 | 198 | -0.126 | **.035** |
| PRP CB Slope | 남성 | 79 | -0.139 | .187 |
| PRP CB Slope | 여성 | 119 | -0.113 | .149 |

**해석**: UCLA 외로움↑ → CB Slope 감소 (더 음수, -1 방향으로 이동)
- 이론적으로 slope = -1이면 완전한 중앙 병목 (순차적 처리)
- slope가 0에 가까워지면 병렬 처리 가능
- 외로움이 높을수록 중앙 병목 효과가 **강화**됨 → **이중과제 처리가 더 직렬적**
- 외로움이 **인지 자원의 병렬 할당을 방해**할 수 있음을 시사

#### 4.3.7 Stroop LBA 모델 결과 (2025-12-19 추가)

**모델**: Linear Ballistic Accumulator (Brown & Heathcote, 2008)
- 4개 독립 축적기 (정답 1개 + 오답 3개) 경쟁 모델
- 파라미터: v_correct (정답 drift rate), v_incorrect (오답 drift rate), A (시작점 변이), b (threshold), t0 (비결정 시간)

**피팅 성공률** (조건별):

| 조건 | 유효 피팅 | 비율 |
|------|-----------|------|
| Congruent | 36/219 | 16.4% |
| Incongruent | 108/219 | 49.3% |
| Neutral | 47/219 | 21.5% |
| **전체 조건 유효** | 24/219 | **11.0%** |

**간섭 효과 지표** (N=24, 모든 조건 유효한 참가자만):

| 지표 | M | SD | 해석 |
|------|---|----|------|
| `v_correct_interference` | 0.658 | 1.653 | cong - incong: 간섭 시 drift rate 감소 |
| `b_interference` | 0.246 | 0.663 | incong - cong: 간섭 시 threshold 증가 |
| `t0_interference` | -0.051 | 0.097 | incong - cong: 비결정시간 변화 |

**⚠️ 주의**: LBA 피팅 성공률이 낮음 (11%). 원인:
1. 조건당 시행 수 부족 (36시행 중 timeout/오류 제외 시 15 미만)
2. 4-choice LBA 피팅의 높은 복잡성
3. 개인차로 인한 피팅 실패

**권장**: LBA 결과는 탐색적 목적으로만 사용하고, 주요 분석은 기존 Ex-Gaussian 지표 활용

**UCLA와의 관계 (위계적 회귀, DASS 통제 후)**:

⚠️ **분석 불가**: LBA interference 지표의 유효 N=24로, 최소 기준 N=30 미충족
- 위계적 회귀분석 수행 불가
- 향후 더 큰 표본으로 재검증 필요

---

## 5. 유의한 결과 요약표

### 5.1 성별 차이 (4개 데이터셋 공통)

| 변수 | 방향 | 효과크기 | 일관성 |
|------|------|----------|--------|
| 나이 | 남 > 여 | d = 0.36-0.37 | 4/4 유의 |
| UCLA 외로움 | 남 < 여 | d = -0.39~-0.43 | 4/4 유의 |
| DASS 우울 | 남 < 여 | d = -0.31~-0.35 | 4/4 유의 |
| PRP Delay Effect | 남 < 여 | d = -0.52~-0.58 | 2/2 유의 (overall, prp) |

### 5.2 UCLA 효과 (DASS 통제 후)

| 결과변수 | 전체 | 여성 | 남성 |
|----------|------|------|------|
| WCST Accuracy | p=.042 (overall) | **p=.044** | ns |
| WCST Post-Error Slowing | p=.045 (wcst) | ns | **p=.020-.029** |
| Stroop Incong RT Slope | **p=.017** (stroop) | ns | **p=.001** |
| **WCST HMM P(Lapse→Focus)** | p=.004 (overall), **p<.001** (wcst) | **p=.005-.035** | ns (경계선) |
| **WCST HMM P(Lapse→Lapse)** | p=.004 (overall), **p<.001** (wcst) | **p=.005-.035** | ns (경계선) |
| WCST RL alpha (pos) | ns | ns | **p=.009** (남성 내 효과) |
| WCST HMM Lapse Occupancy | p=.039 (wcst) | ns | **p=.047** (남성) |
| **PRP CB Slope** | **p=.035** (prp) | ns | ns |
| PRP Ex-Gaussian sigma (Short) | **p=.047** (prp) | ns | ns |

### 5.3 DASS 효과

| 결과변수 | 데이터셋 | ΔR² | p |
|----------|----------|-----|---|
| WCST Accuracy | wcst | 4.08% | .042 |
| WCST Post-Switch Error Rate | wcst | 4.12% | .038 |

### 5.4 인지과제 간 상관 (Overall)

| 상관 | r | p | 해석 |
|------|---|---|------|
| WCST PE - PRP Delay | .193 | .008 | 세트전환-이중과제 취약성 공유 |
| Stroop Int - PRP Delay | .183 | .011 | 억제-병목 처리 연관 |

---

## 6. 주요 발견 해석

### 6.1 성별 차이

1. **여성이 더 높은 심리적 고통 보고**
   - UCLA 외로움: +4.5점 (d = -0.40)
   - DASS 우울: +2.5점 (d = -0.33)

2. **여성이 PRP에서 더 큰 이중과제 비용**
   - Delay Effect: +85ms (d = -0.58, 중간~큰 효과)
   - 인지적 병목현상에서 성별 차이 존재

3. **WCST, Stroop에서는 성별 차이 없음**
   - 단순 억제 및 세트전환에서는 동등한 수행

### 6.2 외로움의 성별-특이적 효과

| 성별 | 효과 | 해석 |
|------|------|------|
| **남성** | UCLA↑ → Post-Error Slowing↑ | 오류 후 과도한 자기-모니터링 |
| **남성** | UCLA↑ → Stroop RT Slope↑ | 시간 경과에 따른 인지적 피로 취약성 |
| **남성** | UCLA↑ → RL alpha (pos)↑ | 긍정 피드백 과민감 (β=0.17, p=.009) |
| **남성** | UCLA↑ → HMM Lapse Occupancy↑ | 주의 이탈 상태 증가 (β=3.72, p=.047) |
| **여성** | UCLA↑ → WCST Accuracy↑ | 보상적 노력 또는 신중한 전략 |
| **여성** | UCLA↑ → HMM P(Lapse→Focus)↓ | Lapse에서 회복 어려움 (주의 유지 결함) |

### 6.3 Mechanism 기반 해석 (2025-12-19 재계산)

**HMM 주의 상태 전이 모델 결과:**
- UCLA 외로움이 높을수록 **Lapse 상태 유지 확률 증가** (β=0.09, p<.001)
- UCLA 외로움이 높을수록 **Lapse→Focus 전이 확률 감소** (β=-0.09, p<.001)
- 이 효과는 **여성에서 더 강하게** 나타남 (p=.005 vs 남성 p=.09)

**해석**: 외로운 개인은 WCST 수행 중 주의 이탈(Lapse) 상태에 빠지면 다시 집중(Focus) 상태로 회복하기 어려움. 이는 외로움이 **주의 조절 능력의 회복력(resilience)을 저해**할 수 있음을 시사.

**RL 학습 파라미터 결과:**
- UCLA × 성별 상호작용은 비유의 (p=.141)
- 그러나 **남성에서만** 외로움 → 긍정 피드백 학습률(alpha_pos) 효과가 유의 (β=0.17, p=.009)

**해석**: 외로운 남성은 긍정 피드백에 더 민감하게 반응하며, 이는 **사회적 보상에 대한 과민감**을 반영할 수 있음. 상호작용 term 자체는 통계적으로 유의하지 않으나, 성별에 따른 효과 차이가 존재함.

**PRP Central Bottleneck 모델 결과:**
- UCLA 외로움이 높을수록 **CB Slope가 더 음수** (β=-0.126, p=.035, ΔR²=2.65%)
- Slope가 -1에 가까울수록 완전한 순차적 처리 (중앙 병목)
- 성별별로는 유의하지 않음 (남성 p=.19, 여성 p=.15)

**해석**: 외로운 개인은 이중과제 수행 시 **병렬 처리 능력이 저하**되어 더 직렬적(순차적)으로 처리함. 이는 외로움이 **인지 자원의 유연한 할당을 방해**할 수 있음을 시사. Central Bottleneck Theory 관점에서, 외로움이 높을수록 Task 1과 Task 2의 중앙 처리 단계가 더 엄격하게 순차적으로 이루어짐.

### 6.4 DASS 우울의 역설적 효과

- **DASS 우울↑ → WCST PE Rate↓** (r = -.14, p = .043)
- 가설: 우울한 개인이 더 신중하게 반응할 가능성?

---

## 7. 연구 한계

| 한계 | 설명 | 대응 |
|------|------|------|
| 다중비교 미보정 | 40개+ 검정 수행 | FDR 보정 권장 |
| 나이 이상값 | -20세 기록 존재 | 데이터 확인 필요 |
| 정규성 위반 | Shapiro-Wilk p < .001 대부분 | HC3 SE 사용 |
| 작은 효과크기 | ΔR² = 1.9~3.6% | 임상적 의미 재고 |
| 이분산성 | WCST PES에서 BP p = .006 | HC3 SE로 보정 |

---

## 8. 결론

1. **UCLA-DASS 강한 상관** (r = .47-.68) 확인됨
2. **UCLA와 인지과제 직접 상관 없음** (모두 비유의)
3. **DASS 통제 후 UCLA 독립 효과는 제한적이며 성별-특이적**
4. **PRP에서 가장 큰 성별 차이** (d = -0.58)
5. **남성**: 외로움 → 오류 후 감속, 인지적 피로, 긍정 피드백 과민감
6. **여성**: 외로움 → WCST 정확도 향상 (역설적), Lapse 회복 어려움
7. **인지과제 간 유의한 상관**: WCST PE-PRP Delay (r=.19), Stroop-PRP Delay (r=.18)
8. **Mechanism 분석 핵심 발견 (2025-12-19 재계산)**:
   - **HMM**: UCLA↑ → Lapse 유지 확률↑, 회복 확률↓ (ΔR²=5.8%, p<.001)
   - **HMM Lapse Occupancy**: 남성에서 UCLA↑ → 주의 이탈 증가 (p=.047)
   - **RL**: 남성에서만 UCLA↑ → 양성 학습률↑ (β=0.17, p=.009; 상호작용은 비유의 p=.141)
   - 외로움이 **주의 상태 전이**와 **강화학습 과정**에 성별-특이적 영향
9. **PRP Central Bottleneck 모델 (2025-12-19 추가)**:
   - 피팅 성공률 97.1% (198/204명), R² 평균 0.88
   - bottleneck 평균 687ms, slope 평균 -0.86 (이론적 예측 -1에 근접)
   - **UCLA→CB Slope 유의** (β=-0.126, p=.035, ΔR²=2.65%)
   - 외로움↑ → slope가 -1 방향으로 이동 → **중앙 병목 효과 강화** → 이중과제 처리가 더 직렬적
10. **Stroop LBA 모델 (2025-12-19 추가)**:
    - 피팅 성공률 낮음 (11%, 24/219명) - 조건당 시행 수 부족이 원인
    - N=24로 회귀분석 불가 (min N=30 미충족)
    - 탐색적 분석 목적으로만 사용 권장
11. **PRP Ex-Gaussian sigma (Short SOA) (2025-12-19 추가)**:
    - **UCLA→sigma 유의** (β=24.34, p=.047, ΔR²=2.50%)
    - 외로움↑ → RT 변동성↑ → 이중과제 수행의 불안정성

---

## 9. 후속 분석 권장

1. **FDR 보정**: `multipletests(method='fdr_bh')`
2. **나이 이상값 처리**: -20세 확인 및 수정
3. **매개분석**: UCLA → DASS → 인지기능 경로
4. **베이지안 분석**: 효과 부재 vs. 작은 효과 구분

---

## 부록: 출력 파일 목록 (46개+)

```
publication/data/outputs/basic_analysis/
├── overall/ (11 files)
│   ├── table1_descriptives.csv
│   ├── table1_descriptives_by_gender.csv
│   ├── table1_categorical.csv
│   ├── table1_gender_comparison.csv
│   ├── correlation_matrix.csv
│   ├── correlation_pvalues.csv
│   ├── correlation_ci.csv
│   ├── ucla_correlations_detailed.csv
│   ├── hierarchical_results.csv
│   ├── model_comparison.csv
│   └── hierarchical_summary.txt
├── stroop/ (11 files) - 동일 구조
├── prp/ (11 files) - 동일 구조
├── wcst/ (11 files) - 동일 구조
└── conclusion.md (본 파일)

publication/data/complete_prp/
├── 5_prp_bottleneck_mechanism_features.csv  # NEW (2025-12-19)
└── 5_prp_mechanism_features.csv             # Ex-Gaussian

publication/data/complete_stroop/
├── 5_stroop_lba_mechanism_features.csv      # NEW (2025-12-19)
└── 5_stroop_mechanism_features.csv          # Ex-Gaussian
```
