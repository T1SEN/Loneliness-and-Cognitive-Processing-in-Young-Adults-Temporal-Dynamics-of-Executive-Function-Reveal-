# Analysis Results - Complete Only Dataset

**데이터셋:** `results/complete_only/` (모든 과제를 완료한 참가자만 포함)
**분석 날짜:** 2025-12-05
**총 참가자:** N=171 (Male=65, Female=106, Unknown=0)

---

## Gold Standard (Confirmatory - DASS-controlled)

**p-value 산출 방식**: HC3 Wald test (이질분산에 robust)

### Raw p-values (미보정)

| 분석명 | 결과변수 | N | UCLA β | UCLA p | UCLA×Gender β | Interaction p | ΔR² |
|--------|----------|---|--------|--------|---------------|---------------|-----|
| wcst_pe | pe_rate | 170 | -0.199 | 0.729 | 0.646 | 0.341 | 0.36% |
| wcst_accuracy | wcst_accuracy | 170 | 0.811 | 0.486 | -1.015 | 0.396 | 0.25% |
| stroop_interference | stroop_interference | 170 | 8.922 | 0.428 | 11.208 | 0.497 | 0.25% |
| **prp_bottleneck** | prp_bottleneck | 166 | 18.371 | 0.272 | **39.533** | **0.070†** | 1.53% |
| prp_soa_slope | prp_soa_slope | 157 | -0.009 | 0.563 | -0.034 | 0.108 | 1.34% |
| prp_cv_short | prp_cv_short | 157 | 0.003 | 0.747 | -0.007 | 0.570 | 0.22% |
| prp_tau | prp_tau | 157 | 5.233 | 0.570 | 10.933 | 0.354 | 0.43% |
| prp_mu | prp_mu | 157 | -0.276 | 0.991 | 13.136 | 0.763 | 0.07% |
| prp_sigma | prp_sigma | 157 | 9.457 | 0.465 | 12.227 | 0.464 | 0.35% |
| **meta_control** | meta_control_score | 166 | 0.108 | 0.359 | **0.288** | **0.068†** | 1.53% |

**†** = marginal (p < 0.10)

### Marginal Effects (0.05 < p < 0.10)

| 분석 | 효과 | β | SE | raw p | 해석 |
|------|------|---|----|----|------|
| PRP Bottleneck | UCLA × Gender | 39.53 | 21.68 | 0.070 | 남성에서 외로움이 높을수록 PRP 병목 증가 |
| Meta-Control | UCLA × Gender | 0.29 | 0.16 | 0.068 | 남성에서 외로움이 높을수록 메타-통제 저하 |

### Gender-Stratified Simple Slopes

| 결과변수 | Female β | Female p | Male β | Male p |
|----------|----------|----------|--------|--------|
| PRP Bottleneck | 4.39 | 0.838 | 25.38 | 0.297 |
| Meta-Control | 0.01 | 0.938 | 0.19 | 0.345 |

**해석:** UCLA × Gender 상호작용이 marginal하나, 성별 분리 시 남성에서도 개별적으로 유의하지 않음

### FDR 보정 후

| 분석 | Interaction raw p | FDR-corrected p |
|------|-------------------|-----------------|
| PRP Bottleneck | 0.070 | 0.351 |
| Meta-Control | 0.068 | 0.351 |

**결론:**
- Raw p-value 기준: PRP Bottleneck과 Meta-Control에서 UCLA × Gender **marginal** (p < 0.10)
- FDR 보정 후: 모든 효과 **비유의** (all p > 0.35)
- 다중비교 10개 중 2개가 p < 0.10이면 우연 기대치(1개) 수준

---

## Exploratory Findings (p < 0.05)

### Cross-Task Suite

| 날짜 | 분석명 | 결과변수 | 효과 | β/r | p-value | N |
|------|--------|----------|------|-----|---------|---|
| 2025-12-05 | cross_task_correlations | Stroop × PRP | Partial correlation | r=0.175 | p=0.0238 | 166 |
| 2025-12-05 | age_gam | pe_rate | UCLA × Age | β=0.756 | p=0.0284 | 170 |
| 2025-12-05 | threeway_interaction | pe_rate | UCLA × Gender × Age | β=-0.898 | p=0.0210 | 170 |
| 2025-12-05 | threeway_interaction (Younger) | pe_rate | UCLA × Gender | β=2.889 | p=0.0220 | 71 |

### WCST Suite

| 날짜 | 분석명 | 결과변수 | 효과 | β/r | p-value | N |
|------|--------|----------|------|-----|---------|---|
| 2025-12-05 | switching_dynamics | post_switch_error | UCLA (Female only) | r=-0.213 | p=0.0289 | 106 |

### PRP Suite

| 날짜 | 분석명 | 결과변수 | 효과 | β/r | p-value | N |
|------|--------|----------|------|-----|---------|---|
| 2025-12-05 | exgaussian | mu (Male, short SOA) | UCLA correlation | r=0.292 | p=0.0194 | 64 |
| 2025-12-05 | exgaussian | sigma (Male, short SOA) | UCLA correlation | r=0.317 | p=0.0108 | 64 |

### Fatigue Suite

| 날짜 | 분석명 | 결과변수 | 효과 | β/r | p-value | N |
|------|--------|----------|------|-----|---------|---|
| 2025-12-05 | fatigue_moderation | STROOP Early (Q1) Accuracy | UCLA main | β=-0.005 | p=0.0390 | 171 |

### Synthesis Suite

| 날짜 | 분석명 | 결과변수 | 효과 | Statistic | p-value | N |
|------|--------|----------|------|-----------|---------|---|
| 2025-12-05 | variance_tests | wcst_accuracy | Gender variance diff | Levene | p=0.0214 | 170 |

### Mechanistic Suite

| 날짜 | 분석명 | 결과변수 | 효과 | β/r | p-value | N |
|------|--------|----------|------|-----|---------|---|
| 2025-12-05 | speed_accuracy | WCST RT-Accuracy | Low UCLA group | r=-0.231 | p=0.0297 | 85 |
| 2025-12-05 | cross_task_coupling | Stroop × PRP | High UCLA group | r=0.254 | p<0.05 | 83 |
| 2025-12-05 | cross_task_coupling | PE × Stroop | Low UCLA group | r=0.215 | p<0.05 | 84 |
| 2025-12-05 | cross_task_coupling | PE × PRP | Low UCLA group | r=0.253 | p<0.05 | 84 |

### DDM Suite

| 날짜 | 분석명 | 결과변수 | 효과 | β | p-value | N |
|------|--------|----------|------|---|---------|---|
| 2025-12-05 | EZ-DDM | Drift Rate (v) | UCLA main (DASS-controlled) | β=-0.0141 | p=0.0128 | 102 |
| 2025-12-05 | EZ-DDM | Drift Rate (v) | UCLA × Gender | β=0.0140 | p=0.0623 | 102 |

**해석:** DASS 통제 후에도 Drift rate에서 UCLA 주효과 유의 - 외로움이 높을수록 정보 축적 속도 저하

### Developmental Window Suite

| 날짜 | 분석명 | 결과변수 | 효과 | β | p-value | N |
|------|--------|----------|------|---|---------|---|
| 2025-12-05 | Age-Stratified (22-25) | PRP Bottleneck | UCLA × Gender | β=94.409 | p=0.0132 | 48 |
| 2025-12-05 | Three-way Interaction | WCST PE Rate | UCLA × Age | β=2.774 | p=0.0259 | 170 |
| 2025-12-05 | Three-way Interaction | WCST PE Rate | UCLA × Gender × Age | β=-3.287 | p=0.0210 | 170 |
| 2025-12-05 | Three-way Interaction | PRP Bottleneck | UCLA × Gender | β=46.740 | p=0.0421 | 166 |
| 2025-12-05 | Simple Slopes (18-21 Female) | WCST PE Rate | UCLA main | β=-1.367 | p=0.0491 | 84 |
| 2025-12-05 | Johnson-Neyman | WCST PE Rate | UCLA×Gender (ages 18-36) | - | p<0.05 | 170 |
| 2025-12-05 | Johnson-Neyman | PRP Bottleneck | UCLA×Gender (ages 21-22) | - | p<0.05 | 166 |

---

## UCLA-DASS Correlations (All p < 0.0001)

| 성별 | N | 하위척도 | r |
|------|---|----------|---|
| Male | 65 | Depression | 0.688 |
| Male | 65 | Anxiety | 0.453 |
| Male | 65 | Stress | 0.596 |
| Female | 106 | Depression | 0.641 |
| Female | 106 | Anxiety | 0.471 |
| Female | 106 | Stress | 0.441 |

---

## DASS Specificity Analysis

### Dominance Analysis - Primary Confound

| 결과변수 | Primary Confound | UCLA β 감소 |
|----------|------------------|-------------|
| WCST PE Rate | Depression | -51.2% |
| WCST Accuracy | Stress | 12.1% |
| Stroop Interference | Depression | -197.9% |
| PRP Bottleneck | Depression | -137.5% |

**결론:** 모든 EF 결과변수에서 **Depression**이 UCLA-EF 관계의 주요 혼란 변수

### Sequential DASS Control

| 결과변수 | UCLA only | + Depression | + All DASS |
|----------|-----------|--------------|------------|
| WCST PE Rate | β=-0.906, p=0.051 | β=-0.442, p=0.514 | β=-0.442, p=0.521 |
| WCST Accuracy | β=1.270, p=0.137 | β=1.229, p=0.304 | β=1.192, p=0.341 |
| Stroop | β=-5.466, p=0.593 | β=5.349, p=0.680 | β=4.664, p=0.721 |
| PRP | β=-9.181, p=0.521 | β=3.447, p=0.852 | β=2.978, p=0.874 |

---

## Bayesian Equivalence Testing

| 결과변수 | N | BF01 | 해석 |
|----------|---|------|------|
| WCST PE Rate | 170 | 115.70 | Decisive evidence for H0 |
| WCST Accuracy | 170 | 98.35 | Strong evidence for H0 |
| Stroop Interference | 170 | 97.15 | Strong evidence for H0 |
| PRP Bottleneck | 166 | 20.17 | Strong evidence for H0 |

**결론:** DASS 통제 후 UCLA 효과가 실질적으로 **0과 동등함** (all BF01 > 20)

---

## Pure UCLA Effect Analysis

**UCLA-DASS 관계:**
- DASS가 UCLA 분산의 **44.9%** 설명 (R² = 0.452)
- UCLA_pure (잔차) = DASS 독립적인 "순수 외로움"

| 결과변수 | UCLA (통제 없음) | UCLA (DASS 통제) | UCLA_pure (잔차) |
|----------|-----------------|------------------|------------------|
| WCST Accuracy | β=1.27, p=0.14 | β=1.19, p=0.34 | β=1.03, p=0.35 |
| Stroop Interference | β=-5.46, p=0.59 | β=4.66, p=0.72 | β=3.70, p=0.72 |
| PRP Bottleneck | β=-9.18, p=0.52 | β=2.98, p=0.87 | β=7.82, p=0.63 |

**분산 분해 (평균):**
- UCLA 고유: **37.2%**
- DASS 고유: **71.3%**
- 공유: **13.2%**

---

## DDM Parameter Correlations (N=102)

| Parameter | EF Metric | r | p-value |
|-----------|-----------|---|---------|
| Drift Rate (v) | Stroop Interference | -0.494 | <0.0001 |
| Drift Rate (v) | PE Rate | -0.208 | 0.036 |
| Drift Rate (v) | PRP Bottleneck | -0.223 | 0.027 |
| Boundary (a) | Stroop Interference | 0.359 | 0.0002 |
| Boundary (a) | PRP Bottleneck | 0.222 | 0.028 |
| Non-decision (t) | Stroop Interference | 0.294 | 0.003 |
| Non-decision (t) | PRP Bottleneck | 0.255 | 0.011 |

---

## Validation Summary

### Cross-Validation (5-fold)
- Base vs Full (with UCLA×Gender) model: No significant improvement
- All Delta RMSE p > 0.19

### Robust Regression (Huber)
- Estimates consistent with OLS (changes < 31%)

### Split-Half Reliability
- Both-significant rate: 0% across all outcomes
- Beta correlations negative (effects unstable across splits)

### Type M/S Error Simulation
- Power: 6.6-22.4%
- Type M error: 2.2-5.0x
- Type S error: 0.6-7.1%

---

## Advanced Analyses Integration (2025-12-05)

### 전체 효과 요약 (56개 검정)

| Suite | 유의 (p<0.05) | Marginal (p<0.10) | 총 검정 |
|-------|--------------|-------------------|---------|
| DDM | 1 | 1 | 6 |
| Reinforcement Learning | 1 | 1 | 12 |
| Attention Depletion | 1 | 1 | 10 |
| Error Monitoring | 0 | 2 | 22 |
| Control Strategy | 1 | 0 | 6 |
| **합계** | **4** | **5** | **56** |

### 유의한 효과 (raw p < 0.05)

| Suite | Metric | β | p (raw) | p (FDR) | Cohen's d |
|-------|--------|---|---------|---------|-----------|
| control_strategy | interference_change | 46.26 | 0.0095 | 0.359 | 0.40 |
| ddm | drift rate (v) | -0.0141 | 0.0128 | 0.359 | -0.49 |
| reinforcement_learning | alpha_pos × Gender | 0.155 | 0.0350 | 0.653 | 0.33 |
| attention_depletion | cv_fatigue_slope (WCST) | -0.023 | 0.0500 | 0.698 | -0.30 |

**FDR 보정 후 모든 효과 비유의** (all p_fdr > 0.35)

### HMM Deep Analysis

| 분석 | 결과변수 | UCLA β | p-value | 해석 |
|------|----------|--------|---------|------|
| Model Comparison | 2-state preference | 97.2% | - | BIC 기준 2-state 모델 선호 |
| Transition | P(Lapse→Focus) | 0.105 | 0.799 | 비유의 |
| State Duration | Mean Lapse Duration | 0.133 | 0.653 | 비유의 |
| Recovery | Recovery Slope | 6.993 | 0.353 | 비유의 |

### HMM Mechanism (Gender-Stratified)

| 성별 | 결과변수 | UCLA β | p-value |
|------|----------|--------|---------|
| **Male** | Lapse Occupancy | 5.905 | **0.0158*** |
| **Male** | P(Lapse→Focus) | -0.089 | 0.0775† |
| Female | Lapse Occupancy | 0.930 | 0.600 |
| Female | P(Lapse→Focus) | -0.003 | 0.939 |

**해석:** 남성에서만 외로움이 lapse 점유율과 양의 상관 (r=0.322, p=0.009)

### Causal Inference

| 분석 | 결과 |
|------|------|
| E-value (WCST PE) | 1.62 (weak robustness) |
| DAG Model Comparison | Direct model (UCLA→PE) 선호 |
| Causal Bounds (Γ=1.5) | All include 0 |
| IV Estimate | F=1.44 (weak instrument) |

### Bayesian SEM

| 결과변수 | P(UCLA effect direction) | BF01 (H0) | ROPE % |
|----------|-------------------------|-----------|--------|
| WCST PE | 0.74 (negative) | 12.14 | 77.7% |
| Stroop | 0.51 (positive) | 9.31 | 84.7% |
| PRP | 0.54 (positive) | 6.33 | 88.6% |

**해석:** Bayesian 분석에서도 UCLA 효과 방향 불확실, H0에 대한 moderate-strong evidence

### Temporal Dynamics

| 분석 | Metric | UCLA β | p-value |
|------|--------|--------|---------|
| Autocorrelation | ACF lag-1 (Stroop) | 0.013 | 0.361 |
| DFA | Alpha (WCST) | 0.016 | 0.419 |
| Variability Decomposition | Fast/Total ratio | -0.004 | 0.789 |

**결론:** 시계열 특성에서 UCLA 효과 없음

---

## Summary

### 핵심 발견
1. **DASS 통제 시 UCLA 주효과 소멸**: Gold Standard 분석에서 모든 UCLA 효과 비유의 (FDR p > 0.45)
2. **Bayesian 확증**: BF01 > 20으로 UCLA 효과 없음에 대한 강력한 증거
3. **UCLA-DASS 혼란**: Depression이 모든 EF에서 주요 혼란 변수 (β 감소 50-200%)

### 탐색적 발견 (다중비교 미보정)
1. **DDM Drift Rate**: UCLA 주효과 유의 (β=-0.014, p=0.013) - DASS 통제 후에도
2. **Interference Change**: UCLA 주효과 유의 (β=46.26, p=0.0095) - 가장 강한 효과
3. **HMM Lapse (Male)**: 남성에서만 UCLA → Lapse 양의 효과 (β=5.91, p=0.016)
4. **연령 × 성별 상호작용**: 젊은 여성에서 UCLA → PE 음의 효과

### Mediation Analysis (완료)
- **UCLA → DASS → EF 간접효과**: 모두 ~0 (9개 경로 검정)
- **유의한 매개효과**: 0/9
- **해석**: DASS는 UCLA-EF 관계를 매개하지 않음 (직접 효과 자체가 없으므로)

### Reinforcement Learning (완료)
- **Basic RW**: UCLA 효과 없음 (alpha p=0.46, beta p=0.31)
- **Asymmetric RW**: **UCLA × Gender → alpha_pos** β=0.168, **p=0.017***
- **학습률 비대칭**: alpha_neg > alpha_pos (t=-2.24, p=0.027)

### Advanced Analyses 통합 결과
- **총 ~70개 검정** (Gold Standard 10 + Advanced 56 + Mediation 9)
- **유의 (raw p<0.05)**: ~10개 (~14%) - 우연 기대치(5%) 초과
- **FDR 보정 후 유의**: 0개

### 방법론적 한계
- 검정력 부족 (power 6-22%)
- 분할 신뢰도 낮음 (효과 불안정)
- 다중 비교 문제 (FDR 보정 시 모두 비유의)
- Ex-Gaussian 추정 방법 근사치 사용

### 결론
**"순수 외로움" (DASS 독립적) 효과는 Executive Function에 대해 유의하지 않음.**

UCLA가 EF에 미치는 영향은 대부분 DASS(우울, 불안, 스트레스)와 공유되는 분산이다.
DDM drift rate, interference change, HMM lapse (male only) 등에서 유의한 효과가 관찰되었으나,
다중비교 보정(FDR) 후 모두 비유의해지며, 검정력 부족으로 효과 크기 해석에 주의가 필요하다.

**핵심 결론:** 66개 검정 중 FDR 보정 후 유의한 효과 **0개**.

---

## Machine Learning Analysis (2025-12-05)

### 목적
순수 인지기능(NCFT) 피처만으로 UCLA 외로움을 예측할 수 있는지 탐색

### Nested CV - EF Only Features

#### Classification (High vs Low UCLA, 상/하위 25%)

| Fold | Model | Accuracy | AUC | F1 |
|------|-------|----------|-----|-----|
| 1 | GBRT | 0.629 | 0.624 | 0.316 |
| 2 | GBRT | 0.676 | 0.742 | 0.154 |
| 3 | GBRT | 0.676 | 0.551 | 0.353 |
| 4 | GBRT | 0.676 | 0.524 | 0.476 |
| 5 | GBRT | 0.647 | 0.533 | 0.333 |
| **Mean** | - | **0.661** | **0.595** | **0.326** |

**해석:** AUC 0.595는 chance(0.50)보다 약간 높지만 유의한 예측력 없음

#### Regression (UCLA 연속 점수)

| Fold | Model | R² | RMSE | MAE |
|------|-------|----|------|-----|
| 1 | RF | -0.122 | 12.78 | 10.43 |
| 2 | RF | -0.309 | 11.57 | 9.72 |
| 3 | RF | -0.228 | 12.36 | 10.32 |
| 4 | Ridge | -0.095 | 11.19 | 9.03 |
| 5 | RF | -0.169 | 12.09 | 10.31 |
| **Mean** | - | **-0.185** | **12.00** | **9.96** |

**해석:** R² 음수 = EF만으로는 UCLA 점수 예측 불가

### Gender-Stratified Classification (EF + DASS features)

| Group | Model | AUC | N |
|-------|-------|-----|---|
| Overall | Random Forest | **0.900** | 21 |
| Overall | Logistic Regression | **0.867** | 21 |
| Female | Logistic Regression | 0.611 | 14 |
| Male | - | N/A (N=7) | - |

**주의:** N=21로 작은 샘플, DASS 피처 포함 시 AUC 크게 상승

### SHAP Feature Importance (XGBoost)

#### Classification Task
| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | **prp_bottleneck** | 0.711 |
| 2 | **stroop_effect** | 0.684 |
| 3 | wcst_conceptual_pct | 0.500 |
| 4 | mrt_incong | 0.280 |
| 5 | wcst_persev_errors | 0.275 |

#### Regression Task
| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | **prp_bottleneck** | 2.488 |
| 2 | wcst_conceptual_pct | 1.790 |
| 3 | **stroop_effect** | 1.438 |
| 4 | prp_mrt_t1 | 1.234 |
| 5 | mrt_cong | 1.160 |

**해석:**
- PRP bottleneck과 Stroop effect가 가장 중요한 피처
- 하지만 전체 예측력(AUC, R²)이 낮아 실용적 의미 제한적

### ML 분석 결론

1. **EF만으로는 UCLA 예측 불가**
   - Classification AUC ~0.60 (chance 수준)
   - Regression R² < 0 (예측 실패)

2. **DASS 포함 시 예측력 급증**
   - AUC 0.90 (N=21, small sample)
   - DASS가 UCLA와 강한 상관 (r~0.5-0.7) 때문

3. **SHAP 기반 중요 피처**
   - PRP bottleneck (dual-task coordination)
   - Stroop effect (interference control)
   - WCST conceptual % (set-shifting)

4. **한계**
   - Small sample size (N=171, extreme groups N=21)
   - EF 피처의 낮은 신뢰도
   - DASS 없이는 예측력 없음

**결론:** 기계학습에서도 **순수 인지기능(EF)만으로는 외로움을 예측할 수 없음**. DASS(정서적 디스트레스)가 주요 예측 변수이며, EF는 부가적 정보만 제공.
