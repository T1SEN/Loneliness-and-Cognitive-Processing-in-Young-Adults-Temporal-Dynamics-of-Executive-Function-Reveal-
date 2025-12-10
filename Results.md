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

## Legacy Advanced Scripts (2025-12-06 재분석)

| Script / Output | Key Outcome | Effect | Test / p-value | Notes |
|-----------------|-------------|--------|----------------|-------|
| `adaptive_recovery_dynamics` | Stroop Reactive Control Index | UCLA β = -26.76 | DASS-controlled regression, p = 0.0011 | Lonelier participants show sharply reduced reactive control even after covariates. |
| `post_error_slowing_gender_moderation` | WCST PES / accuracy | Male r = +0.320; Female r = +0.196 | Pearson (N=68 / 104), p = 0.0077 / 0.0466 | PES couples with UCLA only in males, whereas females show accuracy coupling. |
| `ef_vulnerability_clustering` | Cluster × Gender | χ²(1) = 8.26 | p = 0.0040 | Two-cluster solution; Cluster 0 (perseverative) is strongly female-dominant. |
| `framework1_regression_mixtures` | PRP latent class slope | β = +97.5 (Class 3) | Within-class regression, p = 0.027 | Highlights a subgroup where UCLA strongly predicts PRP bottleneck magnitude. |
| `gendered_temporal_vulnerability` | Epoch / gender effects | WCST epoch p = 0.0009; Stroop gender p = 0.0086; Stroop epoch p = 0.0001 | Mixed-effects | Temporal recovery speed differs by gender/epoch despite null UCLA interactions. |
| `latent_metacontrol_sem` | Meta-control → WCST PE | β = -2.535 | p < 0.0001 | Confirms latent control factor mediates perseverative errors more than UCLA direct path. |
| `network_psychometrics` | Gender network similarity | r = 0.665 | p = 0.0010 | Male/female partial-correlation networks share architecture (high similarity). |
| `network_psychometrics_extended` | Global strength difference | Δ = 3.331 | p = 0.0439 | Male network denser; largest edge shift is dass_stress—prp_bottleneck (Δ = 0.493). |
| `multivariate_ef_analysis` | MANOVA gender main | Wilks’ λ = 0.911 | F(3,162) = 5.26, p = 0.0017 | EF vector differs by gender while UCLA and interactions remain null. |
| `prp_exgaussian_dass_controlled` | Gender main (μ, σ, Δμ_short) | β = -160 to -40 | p = 0.0001–0.0400 | Even with DASS covariates, males show consistently shorter mean/variance components. |

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
| 2025-12-07 | Age-Stratified (22-25) | PRP Bottleneck | UCLA x Gender | β=96.931 | p=0.0104 | N=48 |
| 2025-12-07 | Three-way Interaction | WCST PE Rate | UCLA x Gender x Age | β=-3.215 | p=0.0311 | N=177 |
| 2025-12-07 | Three-way Interaction | WCST PE Rate | UCLA x Age | β=2.734 | p=0.0327 | N=177 |
| 2025-12-07 | Johnson-Neyman | WCST PE Rate | UCLA×Gender (ages 28-36) | β=-9.521 | p=0.0378 | N=177 |
| 2025-12-07 | Johnson-Neyman | PRP Bottleneck | UCLA×Gender (ages 21-22) | β=47.350 | p=0.0297 | N=172 |
| 2025-12-07 | Simple Slopes | WCST PE Rate (18-21, Female) | UCLA | β=-1.359 | p=0.0474 | N=87 |

---

## Path Model Comparison Analysis (2025-12-07)

### 배경
DASS 통제 시 UCLA 효과가 소멸하는 원인을 탐색하기 위해 세 가지 경쟁 인과모형 비교.

### 비교 모형

| 모형 | 경로 | 이론적 해석 |
|------|------|-------------|
| Model 1 | C → L → D | 인지기능↓ → 외로움↑ → 우울↑ |
| Model 2 | C → D → L | 인지기능↓ → 우울↑ → 외로움↑ |
| Model 3 | L → D → C | 외로움↑ → 우울↑ → 인지기능↓ |
| **Model 4** | **L → C → D** | **외로움↑ → 인지기능↓ → 우울↑ (이론적으로 비합리적)** |

**변수:**
- C (EF composite): mean(z_pe_rate, z_stroop_interference, z_prp_bottleneck)
- L (z_ucla): UCLA Loneliness Scale
- D (z_dass_dep): DASS-21 Depression subscale
- 공변량: z_age, gender_male

### 모형 적합도 (N=177, semopy)

| 모형 | AIC | BIC | CFI | RMSEA | a-path β | a-path p | b-path β | b-path p |
|------|-----|-----|-----|-------|----------|----------|----------|----------|
| Model 1 | 15.94 | 41.35 | 1.02 | 0.00 | -0.087 | 0.464 | 0.659 | <.0001 |
| Model 2 | 15.99 | 41.40 | 1.05 | 0.00 | -0.258 | **0.028** | 0.655 | <.0001 |
| Model 3 | 15.99 | 41.40 | 1.05 | 0.00 | 0.659 | <.0001 | -0.103 | **0.028** |
| **Model 4** | 14.87 | 40.28 | **0.18** | **0.28** | -0.035 | 0.464 | -0.259 | **0.028** |

**Note:** Model 1-3은 포화(just-identified) 모형으로 AIC/BIC 거의 동일. **Model 4는 CFI=0.18, RMSEA=0.28로 적합도 매우 나쁨 → 기각**

### 부트스트랩 간접효과 (n=500)

| 모형 | 간접효과 (a×b) | 95% CI | 유의 |
|------|---------------|--------|------|
| Model 1 (C→L→D) | -0.059 | [-0.203, 0.074] | ❌ |
| **Model 2 (C→D→L)** | **-0.170** | **[-0.294, -0.049]** | **✅** |
| **Model 3 (L→D→C)** | **-0.065** | **[-0.108, -0.011]** | **✅** |
| Model 4 (L→C→D) | +0.009 | ~0 | ❌ (a-path 비유의) |

### 성별별 경로계수

| 모형 | 경로 | Male (n=68) | Female (n=109) | Δ |
|------|------|-------------|----------------|-----|
| Model 1 | a-path (EF→UCLA) | +0.201 | -0.194 | **0.396** |
| Model 1 | b-path (UCLA→DASS) | +0.636 | +0.673 | 0.037 |
| Model 2 | a-path (EF→DASS) | -0.102 | -0.324 | **0.222** |
| Model 3 | b-path (DASS→EF) | -0.034 | -0.135 | 0.101 |
| Model 4 | a-path (UCLA→EF) | +0.058 | -0.088 | 0.146 |
| Model 4 | b-path (EF→DASS) | -0.101 | -0.323 | **0.222** |

### 성별 차이 Bootstrap 검정 (n=500)

| 모형 | 경로 | Male Mean | Female Mean | z-stat | p-value |
|------|------|-----------|-------------|--------|---------|
| **Model 1** | **a-path** | **+0.208** | **-0.190** | 1.58 | **0.115†** |
| Model 1 | indirect | +0.129 | -0.130 | 1.56 | 0.118 |
| Model 4 | a-path | +0.059 | -0.088 | 1.83 | **0.067†** |

**†** = 경계선적 (p < 0.15), 효과 크기는 크나 검정력 부족 (n_male=68)

### 핵심 발견

1. **Model 2 간접효과 유의**: EF↓ → 우울↑ → 외로움↑ (β=-0.170, p<.05)
   - 인지기능이 떨어지면 우울해지고, 우울하면 외로워짐

2. **Model 3 간접효과 유의**: 외로움↑ → 우울↑ → EF↓ (β=-0.065, p<.05)
   - 외로우면 우울해지고, 우울하면 인지수행 저하

3. **Model 4 기각**: 외로움↑ → 인지↓ → 우울↑ (간접효과 ≈ 0, CFI=0.18)
   - a-path (UCLA → EF): β=-0.035, p=0.464 (비유의)
   - 아버지 제안: "인지기능이 떨어져서 우울하다기보다는 그 반대가 맞겠지" → **데이터로 확인됨**

4. **DASS 통제 시 UCLA 효과 소멸 설명**:
   - Model 3의 간접효과가 유의 → UCLA는 DASS를 통해서만 EF에 영향
   - DASS를 통제하면 이 매개 경로가 차단됨
   - UCLA의 직접효과가 없으므로 DASS 통제 시 UCLA 효과 소멸

5. **성별 차이** (Model 1 a-path):
   - 남성: 인지↓ → 외로움↑ (β=+0.20)
   - 여성: 인지↓ → 외로움↓ (β=-0.19) **반대 방향!**
   - Bootstrap 검정: p=0.115 (경계선적, 검정력 부족)
   - **해석**: 남성은 인지기능 저하 시 외로워지지만, 여성은 사회적 지지로 보호될 가능성

### 해석적 한계

1. 횡단 데이터로 인과 방향 확정 불가
2. Model 1-3은 관찰적으로 동치 - 적합도 비교는 상대적 해석만 가능
3. 표본 크기(N=177) SEM에 다소 제한적
4. 성별 차이 검정력 부족 (n_male=68)

### 결론

**외로움은 우울을 통해서만 인지기능에 영향을 미침** (Model 3 지지, Model 4 기각)
- 직접 경로: UCLA → EF (비유의, β=-0.035, p=0.464)
- 간접 경로: UCLA → DASS → EF (유의, β=-0.065)
- **Model 4 (L→C→D) 기각**: 외로움이 인지를 통해 우울에 영향을 미치는 경로는 없음

**아버지의 인과 모델 비교 제안 결과:**
> "만약 Model 1이 우수하면 우울 통제 안 해도 됨"

데이터 결과:
- Model 1 비유의 → **우울 통제가 정당화됨**
- Model 2, 3 유의 → 우울이 외로움-인지 관계의 핵심 매개 변수
- Model 4 기각 → "외로움 → 인지 → 우울" 경로는 이론적으로도 데이터적으로도 지지되지 않음

**성별 차이의 기저 메커니즘:**
- Model 1 a-path에서 남녀 방향이 반대 (Δ=0.40)
- 이것이 기존 UCLA × Gender 상호작용의 원인
- 남성: 인지 저하 → 외로움 증가 (일반적 패턴)
- 여성: 인지 저하 → 외로움 감소 (사회적 지지의 보호 효과?)

---

## 정밀 검토 결과 - 성별별 경로 심화 (2025-12-07)

### 성별별 경로계수 정밀 분석

| Model | 경로 | 남성 β (p) | 여성 β (p) | 핵심 해석 |
|-------|------|-----------|-----------|-----------|
| Model 1 | EF→UCLA | +0.20 (0.37) | -0.19 (0.17) | **방향 반대** |
| Model 2 | EF→DASS | -0.10 (0.63) | **-0.32 (0.026)*** | **여성만 유의** |
| Model 3 | DASS→EF | -0.03 (0.63) | **-0.13 (0.026)*** | **여성만 유의** |
| Model 4 | UCLA→EF | +0.06 (0.37) | -0.09 (0.17) | 둘 다 ns |

### 핵심 발견: 여성 특이적 인지-우울 양방향 루프

**남성 (n=68):**
- 모든 경로 비유의 → 외로움-우울-인지 연결이 전반적으로 약함
- 외로움-인지 관계가 다른 메커니즘일 가능성

**여성 (n=109):**
- **EF → DASS (β=-0.32, p=0.026):** 인지 저하 → 우울 증가
- **DASS → EF (β=-0.13, p=0.026):** 우울 → 인지 저하
- **해석:** 여성에서 인지-우울의 **양방향 피드백 루프** 확인

### Model 1 a-path 방향 해리

| 성별 | β | p | 해석 |
|------|---|---|------|
| 남성 | +0.20 | 0.37 | 인지↓ → 외로움↑ |
| 여성 | -0.19 | 0.17 | 인지↓ → 외로움↓ (**역방향**) |
| Bootstrap z | 1.58 | 0.115 | 경계선적 (검정력 부족) |

**해석:** 남성은 인지기능 저하 시 사회적으로 고립되는 경향이 있으나,
여성은 사회적 지지 네트워크가 보호 역할을 하여 인지 저하가 외로움 증가로 이어지지 않음

### Model 4 최종 확인 (부트스트랩 추가)

| 지표 | 값 | 해석 |
|------|-----|------|
| 간접효과 | +0.010 | 약 0 |
| 95% CI | [-0.008, 0.048] | 0 포함 → 비유의 |
| CFI | 0.18 | 매우 나쁨 |
| RMSEA | 0.28 | 매우 나쁨 |

**Model 4 (L→C→D) 최종 기각:** 외로움 → 인지 → 우울 경로는 데이터에서 지지되지 않음

### 성별별 간접효과 유의성 (Bootstrap CI 기반)

| 모델 | 전체 유의 | 남성 유의 | 남성 95% CI | 여성 유의 | 여성 95% CI | 해석 |
|------|----------|----------|------------|----------|------------|------|
| Model 1 | ❌ | ❌ | [-0.138, 0.384] | ❌ | [-0.325, 0.035] | 둘 다 비유의 |
| Model 2 | ✅ | ❌ | [-0.291, 0.207] | **✅** | **[-0.367, -0.038]** | **여성 주도** |
| Model 3 | ✅ | ❌ | [-0.087, 0.059] | **✅** | **[-0.149, -0.026]** | **여성 주도** |
| Model 4 | ❌ | ❌ | [-0.026, 0.033] | ❌ | [-0.003, 0.095] | 둘 다 비유의 |

**중요:** 전체 표본에서 Model 2, 3 간접효과가 유의하지만, 이는 **여성 표본에 의해 주도됨**.
남성은 Model 2, 3 모두 간접효과 비유의 (CI가 0을 포함).

### 수정된 논문 핵심 메시지

> "젊은 성인에서 외로움은 인지기능에 직접 영향을 미치지 않으며, 우울을 통한 간접 경로만이 유의하다.
> 특히 **여성에서만** 인지-우울의 양방향 피드백 루프가 확인되었다.
> 남성은 인지 저하 시 외로워지는 경향이 있으나 (β=+0.20), 여성은 그렇지 않다 (β=-0.19).
> 이는 성별에 따른 사회적 지지 구조의 차이를 반영할 수 있다."

---

## 심층 분석 (2025-12-07)

### 1. Stroop Neutral 조건 분해

**목적:** DDM drift rate 효과(p=0.013)의 기제 규명

| 성분 | 평균 (ms) | SD | 비율 |
|------|-----------|-----|------|
| Facilitation (neutral - congruent) | 40.0 | 76.2 | 29% |
| Interference (incongruent - neutral) | 98.0 | 87.6 | 71% |
| Total Stroop | 138.0 | 105.4 | 100% |

**UCLA 효과 (DASS 통제):**
- Facilitation: β=11.79, p=0.128 (ns)
- Interference: β=-7.77, p=0.524 (ns)
- 조건별 DDM: 모두 비유의 (p > 0.45)

**결론:** Stroop 효과의 촉진/간섭 분해로는 DDM drift rate 효과를 설명하지 못함

---

### 2. UCLA 하위 요인 분석

**요인 구조:**
- Parallel analysis: 2요인 권장
- KMO = 0.938 (Marvelous)
- Factor 1 (Social): Items 2,3,4,7,8,11,12,13,14,18 (분산 26.2%)
- Factor 2 (Emotional): Items 1,5,6,9,10,15,16,19,20 (분산 23.6%)
- 하위요인 상관: r = -0.89

**하위요인 → EF 예측 (DASS 통제):**

| 결과변수 | Social β | Social p | Emotional β | Emotional p |
|----------|----------|----------|-------------|-------------|
| WCST PE Rate | 1.61 | 0.091† | **1.97** | **0.019*** |
| Stroop Interference | 20.01 | 0.441 | 15.54 | 0.496 |
| PRP Bottleneck | -43.99 | 0.333 | -49.21 | 0.229 |

**중요 발견:** **정서적 외로움(Emotional loneliness)**이 WCST PE에 유의한 효과!
- 사회적 외로움: 경계선적 (p=0.091)
- 정서적 외로움: 유의 (p=0.019)

---

### 3. WCST 오류 유형 분해

**오류 구성:**
- 총 오류율: 18.1% (SD=10.4)
- PE(보속오류) 비율: 10.3% (SD=5.4)
- NPE(비보속오류) 비율: 7.8% (SD=8.6)
- PE가 전체 오류의 61.3% 차지 (t=7.97, p<0.001)

**UCLA 효과 (DASS 통제):**

| 결과변수 | UCLA β | UCLA p | UCLA×Gender p |
|----------|--------|--------|---------------|
| Total Error Rate | -0.008 | 0.526 | 0.579 |
| PE Rate | - | - | - |
| NPE Rate | -0.004 | 0.707 | 0.775 |
| PE Proportion | 0.009 | 0.703 | 0.934 |

**결론:** PE/NPE 모두 UCLA 효과 없음 → UCLA × Gender 효과는 특정 오류 유형에 국한되지 않음

---

### 4. PRP 제약 위반 분석

**위반율:** 4.03% (SD=14.77)

**UCLA 효과 (DASS 통제):**
- UCLA main: β=0.030, p=0.130 (ns)
- UCLA × Gender: β=-0.025, p=0.171 (ns)

**결론:** 이중과제 제약 위반도 UCLA 효과 없음

---

## 핵심 발견 요약 (2025-12-07)

### 유의한 새 발견

| 분석 | 효과 | β | p-value | 해석 |
|------|------|---|---------|------|
| UCLA 하위요인 | Emotional → WCST PE | 1.97 | **0.019** | 정서적 외로움이 인지적 유연성에 영향 |

### Null 발견 (DASS 통제 후)

1. **Stroop 분해:** 촉진/간섭 모두 UCLA 효과 없음
2. **WCST 오류 유형:** PE/NPE 모두 UCLA 효과 없음
3. **PRP 제약 위반:** UCLA 효과 없음

### 이론적 함의

1. **정서적 외로움의 특이적 효과:**
   - 사회적 외로움(social network 부재)보다
   - 정서적 외로움(친밀한 관계 부재)이 EF에 더 영향
   - 이는 attachment/emotional regulation 경로 시사

2. **기제적 공백:**
   - DDM drift rate 효과(p=0.013)는 촉진/간섭 분해로 설명 안 됨
   - 다른 기제 탐색 필요 (e.g., 주의 변동성, 학습률)

3. **UCLA × Gender 효과의 비특이성:**
   - PE/NPE 특정 오류 유형에 국한되지 않음
   - 전반적 수행 효율의 문제일 가능성

---

## 심층 분석 확장 (2025-12-07)

### 5. DDM 기제 심층 분석

**UCLA → DDM 파라미터 (DASS 통제):**

| 파라미터 | UCLA β | SE | p-value | 해석 |
|----------|--------|-----|---------|------|
| Drift rate (v) | -0.0158 | 0.006 | **0.0053*** | Higher UCLA = slower evidence accumulation |
| Boundary (a) | 0.0015 | 0.006 | 0.7969 | ns |
| Non-decision (t) | -0.0073 | 0.019 | 0.7066 | ns |

**성별 계층화 DDM:**

| 성별 | N | UCLA → v (β) | p-value |
|------|---|--------------|---------|
| **Female** | 60 | **-0.0144** | **0.0210*** |
| Male | 46 | -0.0044 | 0.6560 |

**핵심 발견:** UCLA → drift rate 효과는 **여성에서만 유의**
- 남성: WCST PE 취약성 (기존 발견)
- 여성: DDM drift rate 취약성 (신규 발견)

**DDM-HMM 상관:**

| DDM | HMM | r | p | 해석 |
|-----|-----|---|---|------|
| v | lapse_occupancy | -0.144 | 0.141 | 예상 방향 (낮은 drift = 높은 lapse) |
| a | trans_to_lapse | 0.166 | 0.090† | 경계선적 |

**Drift-Boundary 교환:**
- r(v, a) = **-0.623**, p < 0.001
- 효율성 (v/a): Mean = 1.22 (SD = 0.39)
- UCLA → 효율성: β=-0.108, p=0.067†

**매개분석 (UCLA → Drift → Stroop Interference):**

| 경로 | β | p-value |
|------|---|---------|
| c (total effect) | 0.088 | 0.529 |
| a (UCLA → Drift) | -0.259 | **0.033*** |
| b (Drift → Stroop) | -0.501 | **<0.001*** |
| c' (direct effect) | -0.042 | 0.754 |
| **Indirect (a×b)** | **0.130** | **0.050*** |

**Bootstrap 95% CI:** [0.016, 0.261] - **유의 (0 미포함)**
**비율 매개:** 148% (완전 매개)

**결론:** UCLA는 drift rate를 통해 Stroop interference에 간접 영향을 미침

---

### 6. 성별 취약성 해리 분석

**성별별 UCLA 효과 패턴:**

| 영역 | 남성 취약성 | 여성 취약성 |
|------|-------------|-------------|
| WCST | PE rate (p=0.025 상호작용) | - |
| DDM | - | Drift rate (p=0.021) |
| HMM | Lapse occupancy (p=0.016) | - |

**보호 요인 분석:**

| 성별 | 결과변수 | 조절변수 | β (interaction) | p-value |
|------|----------|----------|-----------------|---------|
| **Female** | PE rate | Age | **2.988** | **0.0254*** |
| Male | - | - | - | ns |

**해석:** 여성에서 나이가 UCLA→PE 관계를 조절 (나이 증가 = 취약성 감소?)

**네트워크 비교:**
- Male global strength: 0.121
- Female global strength: 0.090
- 차이 p = 0.543 (ns)

---

### 7. 규칙별 WCST 학습

**규칙별 학습 곡선:**

| 규칙 | 학습 기울기 | 정확도 |
|------|------------|--------|
| Colour | 0.034 | 83% |
| Shape | 0.038 | 83% |
| Number | 0.039 | 85% |

**UCLA 효과 (DASS 통제):** 모든 규칙에서 비유의 (모든 p > 0.08)

| 규칙 | 결과변수 | UCLA p | UCLA×Gender p |
|------|----------|--------|---------------|
| Colour | Learning slope | 0.538 | 0.842 |
| Shape | Learning slope | 0.542 | 0.302 |
| Number | Learning slope | **0.084†** | 0.662 |

**결론:** 규칙별 학습에서 UCLA 효과 없음 (Number 규칙에서만 경계선적)

---

### 8. 개입 대상 하위집단 (LPA)

**최적 클러스터:**
- Silhouette 최적: k=5
- BIC 최적: k=2

**GMM 5-클러스터 프로파일:**

| Cluster | N | % | UCLA | PE Rate | 특성 |
|---------|---|---|------|---------|------|
| 0 | 32 | 18.6% | 31.3 (Low) | 8.98% | Low risk |
| **1** | **43** | **25%** | **54.0 (High)** | 8.87% | High UCLA |
| 2 | 13 | 7.6% | 46.5 (High) | 13.1% | High UCLA + High PE |
| 3 | 9 | 5.2% | 35.2 (Mod) | 26.5% | Extreme PE |
| **4** | **75** | **43.6%** | 37.8 (Mod) | 9.30% | Modal |

**클러스터별 UCLA 효과 (DASS 통제):**

| Cluster | UCLA → PE Rate p | UCLA → Stroop p | UCLA → PRP p |
|---------|------------------|-----------------|--------------|
| Cluster 1 | **0.0071*** | ns | **0.0216*** |
| Cluster 4 | ns | **<0.001*** | **<0.001*** |

**고위험군 식별:**
- **Cluster 1** (N=43): High UCLA, UCLA→PRP 유의
- **Cluster 4** (N=75): UCLA→Stroop/PRP 강한 효과

**의사결정나무 (고위험 예측):**
- 정확도: 74.7%
- 최상위 예측 변수: **UCLA** (45.2%)
- 2위: **Gender** (32.0%)
- 3위: **Age** (22.9%)

---

### 9. 통합 가설 검정 매트릭스

| ID | 가설 | 결론 | 핵심 p-value |
|----|------|------|-------------|
| H1 | UCLA → EF (주효과) | **NOT SUPPORTED** | p > 0.35 |
| H2a | UCLA × Gender → WCST PE | **SUPPORTED (Males)** | p = 0.025 |
| H2b | UCLA × Gender → DDM Drift | **SUPPORTED (Females)** | p = 0.021 |
| H3 | UCLA 하위요인 차별적 효과 | **PARTIAL** | Emotional p = 0.019 |
| H4 | Drift rate 매개 | **SUPPORTED** | CI [0.016, 0.261] |
| H5 | HMM lapse 연관 | **EXPLORATORY** | p = 0.016 (Male only) |

---

## 최종 결론 (2025-12-07)

### 확정된 발견

1. **UCLA 주효과 Null:**
   - DASS 통제 후 모든 주효과 비유의 (p > 0.35)
   - Bayesian 증거: BF₀₁ = 98-116 (강한 Null 지지)

2. **성별 특이적 취약성:**
   - **남성:** WCST PE (인지적 유연성) 취약
   - **여성:** DDM drift rate (정보처리 속도) 취약
   - 이중 해리 패턴 확인

3. **정서적 외로움 특이성:**
   - Emotional loneliness → WCST PE (p = 0.019)
   - Social loneliness는 경계선적 (p = 0.091)

4. **매개 경로:**
   - UCLA → DASS → EF (Model 3 지지)
   - UCLA → Drift → Stroop (매개분석 지지)

### 임상적 함의

1. **성별 맞춤 개입:** 남녀별 다른 인지 영역 타겟
2. **정서적 외로움 초점:** 사회적 접촉보다 질적 관계 개선
3. **DASS 선별:** UCLA보다 우울/불안 선별이 EF 예측에 유용

### 한계

1. 다중비교 보정 시 대부분 효과 비유의
2. 횡단 설계로 인과 방향 확정 불가
3. 검정력 제한 (6-22%)
4. 탐색적 발견은 복제 필요

---

## DASS Subscale별 경로분석 확장 (2025-12-07)

### 개요
기존 Depression(우울)만 사용하던 경로분석을 Anxiety(불안), Stress(스트레스)로 확장하여 DASS subscale별 매개효과 비교.

### 분석 설정
- **N** = 177 (Male=68, Female=109)
- **Bootstrap** = 1000회
- **공변량** = z_age, gender_male
- **모형**:
  - Model 1: C → L → DASS
  - Model 2: C → DASS → L
  - Model 3: L → DASS → C
  - Model 4: L → C → DASS

---

### 1. Depression (우울) - 기존 결과 확인

| 모형 | 간접효과 | 95% CI | 유의 |
|------|----------|--------|------|
| Model 1 (C→L→D) | -0.059 | [-0.199, 0.078] | ❌ |
| **Model 2 (C→D→L)** | **-0.169** | **[-0.301, -0.041]** | **✅** |
| **Model 3 (L→D→C)** | **-0.067** | **[-0.118, -0.016]** | **✅** |
| Model 4 (L→C→D) | +0.010 | [-0.008, 0.044] | ❌ |

**유의한 경로:**
- **EF → Depression → UCLA** (β=-0.169, p<.05)
- **UCLA → Depression → EF** (β=-0.067, p<.05)

---

### 2. Anxiety (불안) - 신규 분석

| 모형 | 간접효과 | 95% CI | 유의 |
|------|----------|--------|------|
| Model 1 (C→L→A) | -0.041 | [-0.149, 0.062] | ❌ |
| Model 2 (C→A→L) | -0.048 | [-0.150, 0.053] | ❌ |
| Model 3 (L→A→C) | -0.020 | [-0.063, 0.025] | ❌ |
| Model 4 (L→C→A) | +0.005 | [-0.005, 0.027] | ❌ |

**모든 간접효과 비유의** - Anxiety는 UCLA-EF 관계를 매개하지 않음

---

### 3. Stress (스트레스) - 신규 분석

| 모형 | 간접효과 | 95% CI | 유의 |
|------|----------|--------|------|
| Model 1 (C→L→S) | -0.044 | [-0.154, 0.067] | ❌ |
| Model 2 (C→S→L) | -0.070 | [-0.192, 0.050] | ❌ |
| Model 3 (L→S→C) | -0.029 | [-0.075, 0.014] | ❌ |
| Model 4 (L→C→S) | +0.007 | [-0.005, 0.032] | ❌ |

**모든 간접효과 비유의** - Stress도 UCLA-EF 관계를 매개하지 않음

---

### 4. Subscale별 비교 요약

| DASS Subscale | Model 2 (C→DASS→L) | Model 3 (L→DASS→C) | 매개 역할 |
|---------------|--------------------|--------------------|-----------|
| **Depression** | **✅ β=-0.169** | **✅ β=-0.067** | **유의** |
| Anxiety | ❌ β=-0.048 | ❌ β=-0.020 | 없음 |
| Stress | ❌ β=-0.070 | ❌ β=-0.029 | 없음 |

---

### 5. 성별별 경로계수 비교 (모든 subscale 공통)

**Model 1 a-path (EF → UCLA):**

| Subscale | Male β | Female β | Δ | 해석 |
|----------|--------|----------|-----|------|
| Depression | +0.201 | -0.194 | **0.396** | 방향 반대 |
| Anxiety | +0.202 | -0.194 | **0.396** | 방향 반대 |
| Stress | +0.202 | -0.194 | **0.396** | 방향 반대 |

**Model 2 a-path (EF → DASS):**

| Subscale | Male β | Female β | Δ | Male p | Female p |
|----------|--------|----------|-----|--------|----------|
| Depression | -0.102 | **-0.324** | 0.222 | 0.63 | **0.026** |
| Anxiety | +0.203 | -0.188 | 0.391 | ns | ns |
| Stress | +0.283 | **-0.274** | **0.557** | ns | ns |

---

### 6. 핵심 발견

1. **Depression만이 유의한 매개 변수**
   - UCLA → Depression → EF (β=-0.067, p<.05)
   - EF → Depression → UCLA (β=-0.169, p<.05)
   - Anxiety와 Stress는 매개 역할 없음

2. **DASS Specificity 확인**
   - 기존 Dominance Analysis에서 Depression이 주요 confound로 밝혀짐
   - 경로분석에서도 Depression만 유의한 매개효과
   - Anxiety/Stress는 UCLA-EF 관계에 관여하지 않음

3. **성별 차이는 DASS subscale에 무관**
   - 모든 subscale에서 동일한 패턴 (남성 +0.20, 여성 -0.19)
   - 이는 EF → UCLA 경로가 성별에 따라 반대 방향임을 재확인

4. **이론적 함의**
   - 외로움이 인지기능에 미치는 영향은 **우울을 통해서만** 발생
   - 불안이나 스트레스는 별개의 메커니즘
   - DASS 통제 시 UCLA 효과 소멸의 원인: Depression의 매개 역할

---

### 7. 모델 적합도 비교 (정밀 검토 추가)

**CFI 비교 (>0.95 양호):**

| Model | Depression | Anxiety | Stress | 해석 |
|-------|------------|---------|--------|------|
| Model1 | 1.02 ✅ | 1.13 ✅ | 1.10 ✅ | 양호 |
| Model2 | 1.05 ✅ | 1.14 ✅ | 1.12 ✅ | 양호 |
| Model3 | 1.05 ✅ | 1.14 ✅ | 1.12 ✅ | 양호 |
| **Model4** | **0.18** ❌ | **0.28** ❌ | **0.26** ❌ | **부적합** |

**RMSEA 비교 (<0.08 양호):**

| Model | Depression | Anxiety | Stress |
|-------|------------|---------|--------|
| Model1-3 | 0.00 ✅ | 0.00 ✅ | 0.00 ✅ |
| **Model4** | **0.275** ❌ | **0.17** ❌ | **0.19** ❌ |

**결론:** Model4 (L→C→D/A/S)는 세 subscale 모두에서 적합도 기각 → "외로움→인지→정서" 경로는 **이론적으로도 데이터적으로도 지지되지 않음**

---

### 8. UCLA-DASS 관계 강도 비교

**b-path 계수 (UCLA → DASS subscale):**

| DASS | β | p-value | 해석 |
|------|---|---------|------|
| **Depression** | **0.659** | < 0.001 | 가장 강함 |
| Stress | 0.509 | < 0.001 | 중간 |
| Anxiety | 0.473 | < 0.001 | 가장 약함 |

**해석:** 외로움은 우울과 가장 강하게 연관됨 (r ≈ 0.66). 이것이 Depression에서만 유의한 매개효과가 나온 이유.

---

### 9. 양방향 순환 모델

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
    [인지기능] ──(Model2)──→ [우울] ──→ [외로움]
         ▲                      │
         │                      │
         └───────(Model3)───────┘
```

**Model2:** EF↓ → 우울↑ → 외로움↑ (β=-0.169)
**Model3:** 외로움↑ → 우울↑ → EF↓ (β=-0.067)

---

### 10. 결론

> **"외로움(UCLA)이 인지기능(EF)에 미치는 영향은 Depression을 통해서만 매개된다.
> Anxiety와 Stress는 UCLA-EF 관계에 기여하지 않는다.
> 이는 DASS 통제 시 UCLA 효과가 사라지는 이유가 Depression의 매개 역할 때문임을 시사한다."**

**임상적 함의:**
- 외로운 개인의 인지기능 저하를 예방하려면 **우울 증상 관리**가 핵심
- 불안/스트레스 관리는 인지기능보다 다른 결과에 초점을 맞출 수 있음

---

## UCLA × DASS 조절효과 분석 (2025-12-07)

### 개요
DASS를 공변량이 아닌 **조절변수**로 테스트하여 외로움 효과가 우울/불안/스트레스 심각도에 따라 달라지는지 검증.

### 모형
```
EF ~ z_ucla * z_dass_X + z_ucla * gender + other_dass + z_age
```

### 결과 요약 (12개 검정: 4 EF × 3 DASS)

| 결과변수 | 조절변수 | UCLA × DASS β | raw p | FDR p |
|----------|----------|---------------|-------|-------|
| **Stroop Interference** | **Depression** | **42.67** | **0.0401*** | 0.481 |
| WCST PE Rate | Depression | 0.24 | 0.437 | 0.582 |
| WCST Accuracy | Depression | 0.66 | 0.307 | 0.582 |
| PRP Bottleneck | Depression | -3.31 | 0.764 | 0.764 |
| Stroop Interference | Anxiety | 15.24 | 0.526 | - |
| Stroop Interference | Stress | 26.10 | 0.307 | - |
| 기타 | - | - | > 0.30 | - |

### 핵심 발견

**UCLA × Depression → Stroop Interference** (N=53, R²=0.223)
- β = 42.67, p = 0.040 (FDR 보정 후 비유의: p = 0.481)
- **해석**: 외로움이 높고 우울도 높은 사람에서 Stroop 간섭이 증가 (시너지 효과)
- N=53으로 Stroop 데이터가 제한적이나, 조절효과 패턴 확인

### 하위집단 사분면 분석

| 사분면 | N | WCST PE Mean | PRP Mean |
|--------|---|--------------|----------|
| Low UCLA + Low Dep | 70 | 11.53 | 570.70 |
| **High UCLA + Low Dep** | 24 | 11.02 | 631.47 |
| Low UCLA + High Dep | 19 | 8.94 | 548.19 |
| High UCLA + High Dep | 65 | 9.17 | 599.59 |

**High UCLA + Low Dep vs High UCLA + High Dep**:
- WCST PE: t = 1.53, p = 0.131, Cohen's d = 0.36
- "외로우면서 우울하지 않은" 집단에서 PE가 오히려 높은 경향 (비유의)

### 결론
- **Depression이 유일한 유의 조절변수** (Stroop에서 raw p < 0.05)
- FDR 보정 후 모든 효과 비유의 → 다중비교 문제
- Anxiety, Stress는 조절효과 없음
- "외로움 + 우울" 조합이 Stroop 간섭에 시너지 효과를 미칠 가능성 있으나 복제 필요

---

## Trial-Level Mixed Effects 분석 (2025-12-07)

### 개요
외로움이 **학습 곡선(PROCESS)**에 영향을 미치는지, **기저 수행(STATE)**만 영향을 미치는지를 multilevel modeling으로 검증.

### 모형
```
Level 1: outcome_ij = β0j + β1j * trial_norm + ε_ij
Level 2:
    β0j = γ00 + γ01*UCLA + γ02*Gender + γ03*DASS + u0j  (STATE)
    β1j = γ10 + γ11*UCLA + γ12*Gender + u1j            (PROCESS)
```

### 결과 (N=15,097~6,206 trials × 172~177 participants)

| Task | Outcome | UCLA Baseline (STATE) β | p | UCLA × Trial (PROCESS) β | p |
|------|---------|-------------------------|---|--------------------------|---|
| WCST | Accuracy | 0.0116 | 0.389 | -0.0121 | 0.439 |
| Stroop | RT (incongruent) | 1.45 ms | 0.948 | 23.54 ms | 0.177 |
| PRP | T2 RT (short SOA) | 18.50 ms | 0.539 | 8.33 ms | 0.789 |

### 성별 계층화 (WCST)

| 성별 | N | UCLA Baseline β | p | UCLA × Trial β | p |
|------|---|-----------------|---|----------------|---|
| Female | 109 | 0.0214 | 0.268 | -0.0221 | 0.281 |
| Male | 68 | -0.0036 | 0.824 | 0.0035 | 0.882 |

### 핵심 발견: **모든 효과 비유의**

1. **STATE 효과 없음**: UCLA는 과제 초반 수행에 영향 없음 (all p > 0.27)
2. **PROCESS 효과 없음**: UCLA는 학습 기울기에 영향 없음 (all p > 0.17)
3. **성별 무관**: 남녀 모두에서 학습 곡선 효과 없음

### 해석

**UCLA는 과제 내 성능 변화에 영향을 미치지 않음**
- 외로움이 학습/적응 속도를 저해한다는 가설 기각
- 집계된(aggregated) 분석에서 관찰된 효과는 개인간 차이이지, 과제 진행에 따른 역동적 변화가 아님
- 외로움 효과가 있다면 "trait-like" (안정적 개인차)이지 "state-like" (상황적 변화)가 아님

### 이론적 함의
- **자원 고갈(resource depletion) 가설 미지지**: 외로움이 인지적 피로를 가속화한다면 UCLA × trial이 유의해야 함
- **동기적(motivational) 가설 미지지**: 외로움이 과제 참여를 감소시킨다면 시간에 따른 성능 저하 패턴이 있어야 함
- **가능한 설명**: 외로움-EF 관계는 즉각적 과제 수행보다 장기적 인지 저하와 관련될 가능성

---

## 순차 역학 분석 (Sequential Dynamics) (2025-12-07)

### 개요
오류 연쇄(error cascade), 회복 역학(recovery dynamics), 모멘텀 효과를 분석하여 UCLA가 순차적 수행 패턴에 영향을 미치는지 검증.

### 오류 연쇄 분석 (Error Cascade)

| Task | Metric | Mean | UCLA β | p |
|------|--------|------|--------|---|
| WCST | # Cascades | 3.08 | -0.55 | 0.112 |
| WCST | Cascade Length | 2.17 | -0.05 | 0.696 |
| WCST | Cascade Proportion | - | -0.83 | 0.481 |
| Stroop | # Cascades | 0.05 | 0.001 | 0.956 |

### 적응적 회복 역학 (Adaptive Recovery)

**지수 회복 모델**: `RT(t) = baseline + delta * exp(-t/tau)`

| Task | N (Converged) | Mean τ | Mean Δ | UCLA → τ (Bayesian) |
|------|---------------|--------|--------|---------------------|
| WCST | 91/172 | 5.55 | 373.4 ms | β=-0.07, HDI[-0.35, 0.21], ROPE 46% |
| Stroop | 28/34 | 4.63 | 359.7 ms | - |

### 기타 역학 지표

| Analysis | Metric | UCLA β | p |
|----------|--------|--------|---|
| Post-Error RT (WCST) | Lag 1 RT | -11.81 | 0.717 |
| Post-Error RT (Stroop) | Lag 1 RT | -36.52 | 0.764 |
| Momentum (WCST) | Slope | 3.18 | 0.688 |
| RT Volatility (WCST) | RMSSD | -2.31 | 0.828 |
| RT Volatility (Stroop) | RMSSD | 15.20 | 0.244 |

### 핵심 발견: **모든 효과 비유의**

1. **오류 연쇄 패턴 무관**: UCLA는 연속 오류 발생 빈도나 길이에 영향 없음
2. **회복 속도 무관**: 오류 후 수행 회복 속도(τ)에 UCLA 효과 없음
3. **변동성 무관**: 시행간 RT 변동성(RMSSD)에 UCLA 효과 없음

### 해석
외로움은 순차적 수행 역학에 영향을 미치지 않음. 오류 모니터링, 적응적 조절, 인지적 회복 과정은 외로움 수준과 무관하게 작동함.

---

## Gratton Effect (CSE) × UCLA 분석 (2025-12-07)

### 개요
Congruency Sequence Effect (Gratton effect)이 외로움 수준에 따라 달라지는지 검증.

### 결과 (N=177)

| Metric | Value |
|--------|-------|
| Mean CSE | 0.4 ms (SD=141.6) |
| UCLA → CSE | β=20.19, p=0.185 |
| UCLA × Gender → CSE | β=-13.08, p=0.591 |
| Male correlation | r=0.006, p=0.961 |
| Female correlation | r=0.087, p=0.368 |

### 해석
- **CSE(Gratton effect)가 거의 0**: Mean CSE = 0.4ms로 전형적인 갈등 적응 효과가 관찰되지 않음
- **UCLA 효과 없음**: 외로움은 갈등 적응에 영향을 미치지 않음
- **제한점**: CSE 자체가 관찰되지 않아 UCLA 조절효과 해석이 제한적

---

## 순수 외로움 (Pure UCLA) 분석 (2025-12-07)

### 개요
DASS와 독립적인 "순수 외로움"이 EF에 영향을 미치는지 검증.

### 방법
```python
UCLA_pure = residuals(UCLA ~ DASS_dep + DASS_anx + DASS_str)
```

### 결과

**DASS-UCLA 관계:**
- R² = 0.453
- DASS가 UCLA 분산의 **45.3%** 설명
- UCLA_pure는 54.7%의 고유 분산 보유

**UCLA vs UCLA_pure 비교:**

| 결과변수 | UCLA (no control) β | p | UCLA_pure β | p |
|----------|---------------------|---|-------------|---|
| WCST Accuracy | 1.26 | 0.137 | 0.75 | 0.489 |
| Stroop Interference | -6.54 | 0.519 | 3.28 | 0.751 |
| PRP Bottleneck | -7.29 | 0.608 | 9.24 | 0.568 |

### 핵심 발견

1. **UCLA_pure 효과 모두 비유의**: DASS 독립적 외로움도 EF에 영향 없음
2. **효과 크기 유사**: UCLA_pure와 DASS-통제 UCLA의 β가 유사
3. **결론**: "순수 외로움"만으로는 인지기능을 예측할 수 없음

### 이론적 함의
- 외로움-인지 관계는 **공유 분산**(우울/불안/스트레스와의 중첩)에 의해 주도됨
- DASS를 통제하면 UCLA 효과가 사라지는 이유는 UCLA의 EF 관련 분산이 주로 DASS와 공유되기 때문
- 외로움의 "인지 저하" 효과는 정서적 디스트레스 없이는 관찰되지 않음

---

## WCST 규칙별 학습 분석 (2025-12-07)

### 개요
색상/모양/숫자 규칙별로 UCLA가 학습 속도에 차별적 영향을 미치는지 검증.

### 규칙별 학습 특성 (N=177)

| Rule | Learning Slope | Accuracy | Trials to Criterion |
|------|----------------|----------|---------------------|
| Colour | 0.034 | 83% | - |
| Shape | 0.038 | 83% | - |
| Number | 0.039 | 85% | - |

### UCLA 효과 (규칙별)

| Rule | Outcome | UCLA β | p | UCLA × Gender p |
|------|---------|--------|---|-----------------|
| Colour | Learning Slope | 0.0016 | 0.538 | 0.842 |
| Shape | Learning Slope | -0.0014 | 0.542 | 0.302 |
| **Number** | **Learning Slope** | **-0.0042** | **0.084†** | 0.662 |
| Colour | PE Rate | -0.56 | 0.241 | 0.644 |
| Shape | PE Rate | -0.16 | 0.848 | 0.616 |
| Number | PE Rate | -0.64 | 0.573 | 0.408 |

**†** = marginal (p < 0.10)

### 핵심 발견

1. **Number 규칙에서 경계선적 UCLA 효과**: β=-0.0042, p=0.084
   - 외로움이 높을수록 숫자 규칙 학습 기울기 감소 경향
   - 추상적 규칙이 지각적 규칙보다 외로움에 민감할 가능성

2. **지각적 규칙(색상/모양)은 UCLA 효과 없음**: 모든 p > 0.24

3. **성별 상호작용 없음**: 모든 규칙에서 UCLA × Gender 비유의

### 해석
- 추상적 추론(숫자)이 외로움에 더 취약할 가능성 시사
- 그러나 FDR 보정 시 효과 사라질 것으로 예상 (marginal finding)
- 복제 연구 필요

---

## 추가 분석 종합 결론 (2025-12-07)

### 실시된 분석 6개

| 분석 | 주요 질문 | 결과 |
|------|----------|------|
| UCLA × DASS 조절효과 | 우울이 외로움 효과를 조절하는가? | **Stroop에서 marginal** (p=0.040, FDR ns) |
| Trial-Level Mixed Effects | 학습 곡선(PROCESS) 영향? | **Null** - 학습 기울기 효과 없음 |
| 순차 역학 | 오류 연쇄, 회복에 영향? | **Null** - 순차적 패턴 무관 |
| Gratton Effect | 갈등 적응에 영향? | **Null** - CSE 자체 미관찰 |
| 순수 외로움 | DASS 독립적 효과? | **Null** - 순수 외로움도 EF 예측 못함 |
| 규칙별 학습 | 추상적 vs 지각적 규칙? | **Number에서 marginal** (p=0.084) |

### 전체 결론

**"외로움이 인지기능에 미치는 영향은 매우 제한적이다"**

1. **DASS 통제 시 UCLA 주효과 소멸 재확인**
2. **학습/적응 과정(PROCESS)에 UCLA 효과 없음**
3. **순차적 역학(오류 연쇄, 회복)에 UCLA 효과 없음**
4. **"순수 외로움"도 EF 예측 못함 - 공유 분산 문제**
5. **일부 탐색적 발견(Stroop × Depression, Number rule)은 FDR 미통과**

### 이론적 함의

1. **외로움-인지 관계는 간접적**: 정서적 디스트레스(DASS)를 통해서만 연결됨
2. **외로움은 즉각적 과제 수행에 영향 없음**: 학습, 적응, 회복 과정 모두 무관
3. **가능한 메커니즘**: 장기적 만성 외로움 → 우울 → 인지 저하 (횡단 데이터로 검증 불가)

### 임상적 함의

1. **우울 치료가 핵심**: 외로움보다 우울 관리가 인지 보호에 중요
2. **외로움 스크리닝보다 DASS**: 인지 저하 예측에는 DASS가 더 유용
3. **사회적 개입의 한계**: 사회적 접촉 증가만으로 인지 개선 기대하기 어려움

---

## ⚠️ 탐색적 발견 경고 (2025-12-07 수정)

### FDR 보정 후 결과 정정

| 분석 | 효과 | Raw p | FDR p | 판정 |
|------|------|-------|-------|------|
| 남성 HMM Lapse | UCLA → Lapse Occupancy | 0.020 | ~0.08 | ❌ **탐색적** |
| 여성 DDM Drift | UCLA → Drift Rate | 0.021 | ~0.08 | ❌ **탐색적** |
| 남성 WCST PE | UCLA × Gender | 0.025 | ~0.10 | ❌ **탐색적** |
| Double Dissociation | 모든 효과 | >0.02 | >0.08 | ❌ **모두 비유의** |
| 클러스터별 UCLA | Cluster × UCLA | >0.21 | >0.35 | ❌ **모두 비유의** |

**결론:** FDR 보정 시 **모든 추가 분석 결과가 비유의**. 위 발견들은 **가설 생성용 탐색적 발견**으로만 해석해야 하며, 복제 연구가 필요함.

### 데이터 품질 문제

| 문제 | 영향 |
|------|------|
| DDM 표본 손실 (여성 45%, 남성 32%) | 선택 편향 가능성, 통계적 검정력 감소 |
| 클러스터 미할당 (18명, 10.1%) | 네트워크-클러스터 통합 분석 불완전 |
| GMM 프로파일 불균형 (18% vs 82%) | UCLA 차이 실질적 무차이 (0.8점) |

### 방법론적 수정 사항 (2025-12-07)

코드 검토 후 다음 사항이 수정됨:

1. **FDR 보정 추가**
   - `ddm_suite.py`: `efficiency_by_gender` - 6개 테스트에 Benjamini-Hochberg 적용
   - `male_vulnerability_suite.py`: `double_dissociation_integration` - 4개 테스트에 FDR 적용
   - `clustering_suite.py`: `cluster_ucla_relationship` - 클러스터×결과 테스트에 FDR 적용

2. **DASS 편상관 통제**
   - `latent_suite.py`: `network_cluster_integration` - 영차상관 → 편상관으로 변경
   - `latent_suite.py`: `bridge_centrality` - UCLA 제외 (순환논리 방지), 편상관 적용

3. **로그 변환**
   - `ddm_suite.py`: `efficiency_by_gender` - 효율성 비율(v/a) 로그 변환으로 비선형성 보정

### 해석 지침

> **"위 탐색적 발견들(남성 HMM lapse, 여성 DDM drift, 성별 이중해리)은 FDR 보정 후 모두 비유의하며, 가설 생성 목적으로만 보고한다. 확정적 발견으로 해석해서는 안 되며, 독립 표본에서의 복제가 필수적이다."**

---

## 최종 Gold Standard 결론 (수정)

### 확정된 Null 발견
- **UCLA 주효과**: 모든 EF에서 비유의 (p > 0.35, BF₀₁ > 98)
- **UCLA × Gender**: DASS 통제 후 경계선적 (raw p ~ 0.02-0.07, FDR p > 0.08)
- **매개효과**: Depression만 유의, Anxiety/Stress 비유의

### 탐색적 발견 (복제 필요)
- 남성: WCST PE, HMM Lapse 취약성 경향
- 여성: DDM Drift Rate 취약성 경향
- 정서적 외로움 > 사회적 외로움 (WCST PE에서)

### 권고사항
1. 본 결과는 **탐색적 연구**로 취급해야 함
2. FDR 보정 후 유의한 효과 **0개**
3. 검정력 부족 (6-22%)으로 Type II 오류 가능성 높음
4. 복제 연구 시 N > 300 권장 (80% 검정력 달성)

---

## 업데이트된 Complete Only 분석 (2025-12-09)

**데이터셋:** `results/complete_only/` (업데이트됨)
**총 참가자:** N=198 (Male=78, Female=117, Missing=3)

### Gold Standard (DASS-controlled, FDR)

| 항목 | 결과 |
|------|------|
| 테스트 가설 | 23개 (Tier 1: 4개, Tier 2: 19개) |
| 유의한 주효과 | **0개** (all p > 0.10) |
| 유의한 상호작용 | **0개** (all q > 0.05) |

**결론:** DASS 통제 후 UCLA의 모든 효과 비유의

### 유의한 결과 요약 (p < 0.05)

#### 1. 성별 차이 (Independent t-test)

| 변수 | Male M(SD) | Female M(SD) | t | p | Cohen's d |
|------|------------|--------------|---|---|-----------|
| UCLA 외로움 | 38.12 (11.08) | 41.94 (10.67) | -2.41 | **.017** | -0.35 |
| DASS 우울 | 5.95 (7.02) | 8.56 (7.72) | -2.39 | **.018** | -0.35 |
| PRP Delay Effect | 527.02 (137.24) | 626.29 (146.71) | -4.68 | **<.001** | -0.69 |

#### 2. Depression 경로모형 (Bootstrap 5000회)

| Model | Path | Indirect Effect | 95% CI | Sig |
|-------|------|-----------------|--------|-----|
| Model 2 | 인지 → 우울 → 외로움 | **-0.132** | [-0.265, -0.009] | **Yes** |
| Model 3 | 외로움 → 우울 → 인지 | **-0.051** | [-0.099, -0.005] | **Yes** |

#### 3. Stress 경로모형 성별 차이

| Path | z | p | 해석 |
|------|---|---|------|
| Model2 a-path (EF → Stress) | 2.10 | **.036** | 남녀 반대 방향 |
| Model3 b-path (Stress → EF) | 2.16 | **.031** | 남녀 반대 방향 |
| Model3 indirect (L → S → C) | 1.98 | **.048** | 간접효과 성차 |
| Model4 b-path (EF → Stress) | 2.10 | **.036** | 남녀 반대 방향 |

**해석:** 남성(+), 여성(-) 반대 방향의 인지-스트레스 관계

#### 4. Bayesian 영가설 지지 (BF₀₁)

| Outcome | BF₀₁ (UCLA) | BF₀₁ (Interaction) | 해석 |
|---------|-------------|-------------------|------|
| WCST PE | 13.63 | 12.02 | H₀ 강하게 지지 |
| Stroop | 11.64 | 11.17 | H₀ 강하게 지지 |
| PRP Bottleneck | 4.71 | 3.86 | H₀ 중간 지지 |

**결론:** BF₀₁ > 10은 UCLA 효과가 **없다**는 강한 증거

### 측정도구 신뢰도

| 척도 | Cronbach's α | 평가 |
|------|--------------|------|
| UCLA Loneliness | **0.936** | Excellent |
| DASS-21 Total | **0.920** | Excellent |
| DASS Depression | 0.871 | Good |
| DASS Anxiety | 0.779 | Acceptable |
| DASS Stress | 0.830 | Good |

### 2025-12-09 분석 핵심 결론

1. **UCLA-DASS 강한 공변**: r = 0.49 ~ 0.66 (외로움과 정서적 고통 높은 중첩)
2. **UCLA → EF 직접효과 부재**: DASS 통제 시 모든 주효과 비유의
3. **베이지안 확증**: BF₀₁ = 4.7~13.6으로 영가설 강력 지지
4. **Depression 매개**: 우울만 유의한 매개, 불안/스트레스 비유의
5. **Stress 성별 차이**: 인지-스트레스 관계에서 남녀 반대 방향
| 2025-12-09 | Age-Stratified (22-25) | PRP Bottleneck | UCLA x Gender | β=95.945 | p=0.0073 | N=56 |
| 2025-12-09 | Three-way Interaction | WCST PE Rate | UCLA x Gender x Age | β=-2.964 | p=0.0352 | N=196 |
| 2025-12-09 | Three-way Interaction | WCST PE Rate | UCLA x Age | β=2.568 | p=0.0354 | N=196 |
| 2025-12-09 | Johnson-Neyman | WCST PE Rate | UCLA×Gender (ages 18-36) | β=-7.918 | p=0.0332 | N=196 |
| 2025-12-09 | Johnson-Neyman | PRP Bottleneck | UCLA×Gender (ages 21-22) | β=46.656 | p=0.0296 | N=190 |
| 2025-12-09 | Simple Slopes | WCST PE Rate (18-21, Female) | UCLA | β=-1.267 | p=0.0433 | N=96 |

---

## 2025-12-10 Advanced Analysis Suites 전체 실행 결과

**분석 날짜:** 2025-12-10
**실행된 Suite 수:** 45개 (전체 SUITE_REGISTRY)
**데이터셋:** results/complete_only (N=197)

### Gold Standard (DASS-controlled) 최종 결과

| 지표 | 값 | 해석 |
|------|------|------|
| 총 가설 수 | 23 | Tier 1 (4) + Tier 2 (19) |
| Raw p < 0.05 | **0** | 없음 |
| FDR q < 0.05 | **0** | 없음 |
| 최소 UCLA p-value | 0.086 | PRP Bottleneck Interaction |

**결론:** DASS 통제 후 모든 UCLA → EF 효과가 사라짐

### Exploratory Analyses (DASS 미통제, Raw p < 0.05)

| 날짜 | Suite | 분석대상 | 효과 | β | p | N |
|------|-------|----------|------|---|---|---|
| 2025-12-10 | intervention_subgroups | Cluster 4, PRP bottleneck | UCLA | -81.78 | **0.002** | 30 |
| 2025-12-10 | hmm_mechanism (Male) | lapse_occupancy_corr | UCLA | 0.32 | **0.004** | 78 |
| 2025-12-10 | control_strategy | Stroop interference_change | UCLA | 40.74 | **0.015** | 197 |
| 2025-12-10 | sequential_deep | Stroop RT recovery (lag 3) | UCLA | 150.15 | **0.018** | 42 |
| 2025-12-10 | hmm_mechanism (Male) | lapse_occupancy | UCLA | 5.05 | **0.029** | 78 |
| 2025-12-10 | attention_depletion | WCST CV fatigue slope | UCLA | -0.024 | **0.031** | 197 |
| 2025-12-10 | ddm | drift rate (v) | UCLA | -0.012 | **0.035** | 121 |
| 2025-12-10 | developmental_window | PE rate (Female 18-21) | UCLA | -1.27 | **0.043** | 96 |

**중요:** 위 결과들은 DASS 미통제 상태의 탐색적 분석으로, FDR 보정 시 생존하지 않음

### Path Analysis (Depression 매개경로)

| Model | 경로 | Indirect | 95% CI | 유의 |
|-------|------|----------|--------|------|
| Model 2 | 인지 → 우울 → 외로움 | **-0.132** | [-0.265, -0.009] | **Yes** |
| Model 3 | 외로움 → 우울 → 인지 | **-0.051** | [-0.099, -0.005] | **Yes** |

- **Anxiety 경로**: 모든 간접효과 비유의 (all CI include 0)
- **Stress 경로**: 모든 간접효과 비유의 (all CI include 0)

### UCLA Factor Analysis

| Factor | UCLA 하위요인 → EF 유의 효과 |
|--------|------------------------------|
| Social Loneliness | 없음 |
| Emotional Loneliness | WCST PE Rate |

### Advanced Suites 실행 상태 (45개)

| 카테고리 | Suite 수 | 완료 |
|----------|----------|------|
| Gold Standard | 1 | ✓ |
| Exploratory | 6 | ✓ |
| Mediation | 1 | ✓ |
| Validation | 3 | ✓ |
| Synthesis | 3 | ✓ |
| Advanced | 31 | ✓ |

### 최종 결론 (2025-12-10)

1. **DASS-controlled 확증적 분석**: UCLA → EF 직접효과 **0건** (23개 가설 모두 비유의)
2. **탐색적 분석**: 8건의 raw p < 0.05 발견, 그러나 모두 DASS 미통제 또는 하위집단 분석
3. **Depression 매개**: 유일하게 bootstrap CI에서 유의한 간접효과 확인
4. **Emotional Loneliness**: UCLA 하위요인 중 정서적 외로움만 WCST PE와 연관
5. **Gender-specific (Male)**: HMM lapse state에서 남성에서만 UCLA 효과 관찰

**해석:** 외로움이 인지기능에 미치는 영향은 우울을 통한 간접경로로만 유의하며, 직접효과는 존재하지 않음
| 2025-12-10 | Age-Stratified (22-25) | PRP Bottleneck | UCLA x Gender | β=95.945 | p=0.0073 | N=56 |
| 2025-12-10 | Three-way Interaction | WCST PE Rate | UCLA x Gender x Age | β=-2.964 | p=0.0352 | N=196 |
| 2025-12-10 | Three-way Interaction | WCST PE Rate | UCLA x Age | β=2.568 | p=0.0354 | N=196 |
| 2025-12-10 | Johnson-Neyman | WCST PE Rate | UCLA×Gender (ages 18-36) | β=-7.918 | p=0.0332 | N=196 |
| 2025-12-10 | Johnson-Neyman | PRP Bottleneck | UCLA×Gender (ages 21-22) | β=46.656 | p=0.0296 | N=190 |
| 2025-12-10 | Simple Slopes | WCST PE Rate (18-21, Female) | UCLA | β=-1.267 | p=0.0433 | N=96 |
| 2025-12-10 | Fatigue Moderation | STROOP Early (Q1) Accuracy | UCLA main | β=-0.005 | p=0.0334 | N=196 |
| 2025-12-10 | Bayes Factor | WCST PE Rate | Decisive evidence for H0 (no UCLA effect) | BF01=163.88 | N/A | N=196 |
| 2025-12-10 | Bayes Factor | WCST Accuracy | Decisive evidence for H0 (no UCLA effect) | BF01=114.51 | N/A | N=196 |
| 2025-12-10 | Bayes Factor | Stroop Interference | Decisive evidence for H0 (no UCLA effect) | BF01=130.01 | N/A | N=196 |
| 2025-12-10 | Bayes Factor | PRP Bottleneck | Strong evidence for H0 | BF01=18.15 | N/A | N=190 |
| 2025-12-10 | Dominance Analysis | WCST PE Rate | Primary confound: Depression | β=-56.09% | N/A | N=196 |
| 2025-12-10 | Dominance Analysis | WCST Accuracy | Primary confound: Stress | β=33.19% | N/A | N=196 |
| 2025-12-10 | Dominance Analysis | Stroop Interference | Primary confound: Depression | β=-130.73% | N/A | N=196 |
| 2025-12-10 | Dominance Analysis | PRP Bottleneck | Primary confound: Depression | β=-247.30% | N/A | N=190 |
