# UCLA 외로움-집행기능 연구: 유의한 결과 종합 보고서

**작성일**: 2025-12-03
**총 분석 스크립트**: 95+개
**총 출력 파일**: 300+개
**표본 크기**: N = 150-169

---

## 1. 핵심 결론 (Executive Summary)

### DASS-21 통제 후 UCLA 외로움 주 효과

| 종속변수 | N | UCLA β | p-value | R² | 결론 |
|---------|---|--------|---------|-----|------|
| WCST PE Rate | 150 | -0.03 | 0.959 | 0.034 | NS |
| WCST Accuracy | 150 | +0.54 | 0.692 | 0.037 | NS |
| Stroop Interference | 158 | +10.85 | 0.352 | 0.030 | NS |
| PRP Bottleneck | 154 | +14.14 | 0.437 | 0.087 | NS |

> **결론**: DASS-21(우울/불안/스트레스) 통제 후, UCLA 외로움의 독립적 효과는 관찰되지 않음.
> 이전 연구에서 보고된 "외로움-집행기능" 연관성은 정서적 고통이 혼입변수로 작용했을 가능성 높음.

---

## 2. 유의한 결과 (p < 0.05)

### A. 분산/변동성 효과 (Paper 1: Distributional Variance)

| 측정치 | UCLA β | SE | p-value | 해석 |
|--------|--------|-----|---------|------|
| **WCST RMSSD (log σ²)** | 0.656 | 0.232 | **0.005** | RT 변동성 증가 |
| **PRP σ long SOA (log σ²)** | 0.716 | 0.258 | **0.006** | 반응시간 표준편차 증가 |
| **WCST τ (log σ²)** | 0.737 | 0.253 | **0.004** | Ex-Gaussian 꼬리 길어짐 |

> **해석**: 외로움은 평균 수행이 아닌 **수행 변동성/분포 꼬리**에 영향을 미침

**참조 파일**: `paper1_distributional/paper1_location_scale_results.csv`

---

### B. 주의 이탈 모형 (Attentional Lapse Mixture Model)

| 측정치 | UCLA β | SE | p-value | 95% CI |
|--------|--------|-----|---------|--------|
| **Stroop Lapse Probability** | 0.0397 | 0.017 | **0.0198** | [0.006, 0.073] |

> **해석**: 외로움이 높을수록 Stroop 과제에서 **이산적 주의 이탈 상태 확률 증가**

**추가 발견 (DASS 효과)**:
- Stroop Lapse Prob × Depression: β = -0.043, p = **0.0115**
- WCST Lapse Prob × Anxiety: β = 0.028, p = **0.0462**

**참조 파일**: `attentional_lapse/LAPSE_MIXTURE_REPORT.txt`

---

### C. 시간적 취약성 (Gendered Temporal Vulnerability)

**3-way Interaction: UCLA × Gender × Epoch**

| 과제 | β | p-value | 해석 |
|------|---|---------|------|
| **WCST (epoch_num)** | 0.0108 | **0.0065** | 남성에서 시간 경과에 따른 취약성 증가 |
| **Stroop (epoch_num)** | 13.582 | **0.0007** | 남성에서 시간 경과에 따른 취약성 증가 |

> **해석**: 남성에서 과제 후반부로 갈수록 외로움 효과가 증가 (자원 고갈 모델)

**참조 파일**: `temporal_vulnerability/TEMPORAL_VULNERABILITY_REPORT.txt`

---

### D. 성별 주 효과 (Gender Main Effects)

#### IIV 분해 분석

| 측정치 | Gender (Male) β | p-value | 95% CI |
|--------|-----------------|---------|--------|
| **PRP Baseline RT** | -143.73 | **0.0012** | [-229.7, -57.8] |
| **PRP Residual Noise** | -45.55 | **0.0028** | [-75.2, -15.9] |
| **WCST Baseline RT** | -191.09 | **0.0026** | [-314.1, -68.1] |
| **Stroop Engaged Mean RT** | -55.36 | **0.0387** | [-107.8, -2.9] |

#### Attentional Lapse 분석

| 측정치 | Gender (Male) β | p-value |
|--------|-----------------|---------|
| WCST Lapse Probability | -0.043 | **0.0249** |
| WCST State Separation | 223.91 | **0.0190** |
| WCST Engaged Mean RT | -99.20 | **0.0175** |
| PRP State Separation | -75.55 | **0.0035** |
| PRP Engaged Mean RT | -70.41 | **0.0183** |
| PRP Lapse Mean RT | -145.95 | **0.0028** |

#### Error Burst 분석

| 측정치 | Gender (Male) β | p-value |
|--------|-----------------|---------|
| WCST Clustering Coefficient | -0.083 | **0.0161** |

> **해석**: 남성이 전반적으로 **빠른 RT, 낮은 노이즈, 낮은 오류 클러스터링** 보임

---

### E. 클러스터링 분석

| 검정 | χ² / F | p-value | 결과 |
|------|--------|---------|------|
| **Gender × Cluster** | 4.249 | **0.0393** | 유의한 성별 분리 |
| UCLA across clusters | 2.052 | 0.1541 | NS |

**클러스터 특성**:
- Cluster 0 (Perseverative, Female-Dominant): N=57, 73.7% 여성, UCLA M=39.75
- Cluster 1 (Perseverative): N=92, 55.4% 여성, UCLA M=42.47

**참조 파일**: `ef_vulnerability_clustering/CLUSTERING_REPORT.txt`

---

### F. DASS 하위척도 효과

| 측정치 | DASS 변수 | β | p-value |
|--------|-----------|---|---------|
| **PRP Learning Rate** | Anxiety | 102.04 | **0.0022** |
| PRP Baseline RT | Anxiety | -81.27 | **0.0100** |
| PRP Baseline RT | Stress | 77.40 | **0.0152** |
| Stroop Lapse Prob | Depression | -0.043 | **0.0115** |
| WCST Lapse Prob | Anxiety | 0.028 | **0.0462** |
| WCST Lapse Prob | Stress | -0.029 | **0.0409** |
| WCST Lapse Mean RT | Anxiety | -161.87 | **0.0437** |
| Stroop Engaged Mean RT | Anxiety | -42.26 | **0.0270** |

**참조 파일**: `iiv_decomposition/IIV_DECOMPOSITION_REPORT.txt`

---

### G. 기타 유의한 효과

#### PRP 남성 층화 분석

| 변수 | β | p-value |
|------|---|---------|
| DASS Stress | 78.16 | **0.0063** |
| Age | 89.15 | **0.0032** |

**참조 파일**: `synthesis_analysis/gender_stratified_coefficients.csv`

---

### H. MANOVA 다변량 분석

| 효과 | Wilks' λ | F | df | p-value | 해석 |
|------|----------|---|-----|---------|------|
| **Gender Main Effect** | 0.921 | 3.934 | 3, 138 | **0.0099** | 성별에 따른 다변량 EF 차이 |
| UCLA Main | 0.996 | 0.178 | 3, 138 | 0.912 | NS |
| UCLA × Gender | 0.972 | 1.339 | 3, 138 | 0.264 | NS |

> **해석**: 3개 EF 과제(WCST, PRP, Stroop)를 동시 분석 시, **성별 주효과만 유의**

**참조 파일**: `multivariate_ef_analysis/MULTIVARIATE_EF_REPORT.txt`

---

### I. PRP Ex-Gaussian UCLA × Gender 상호작용

| 파라미터 | UCLA × Gender β | p-value | 해석 |
|----------|-----------------|---------|------|
| **τ (Exponential tail)** | 50.68 | **0.0432** | 남성에서 주의 이탈 증가 |

> **해석**: 남성에서 외로움이 높을수록 **PRP 과제의 주의 이탈(τ) 증가**

**참조 파일**: `prp_exgaussian_dass_controlled/EXGAUSSIAN_MECHANISTIC_SUMMARY.txt`

---

### J. 적응적 회복 역학 (Adaptive Recovery Dynamics)

| 측정치 | UCLA β | p-value | 95% CI | 해석 |
|--------|--------|---------|--------|------|
| **Stroop Conflict Adaptation (CSE)** | -36.48 | **0.0364** | [-70.6, -2.3] | 갈등 적응 감소 |
| **Stroop Reactive Control Index** | -26.34 | **0.0020** | [-42.8, -9.8] | 반응적 통제 감소 |

> **해석**: 외로움이 높을수록 **Stroop에서 시행-간 통제 조절 능력 저하**

**참조 파일**: `adaptive_dynamics/RECOVERY_DYNAMICS_REPORT.txt`

---

### K. Meta-Control 경로 분석

| 경로 | β | p-value | 해석 |
|------|---|---------|------|
| **b: Meta-Control → PE** | -2.501 | **2.6e-13** | 메타-통제가 PE 강력 예측 |
| a: UCLA → Meta-Control | -0.118 | 0.346 | NS |

> **해석**: Meta-Control 잠재요인이 WCST PE를 강력히 예측 (간접효과 경로는 미유의)

**참조 파일**: `advanced_comprehensive/ADVANCED_ANALYSES_SUMMARY.txt`

---

### L. 교차-과제 상관

| 상관 | r | p-value | 해석 |
|------|---|---------|------|
| **PRP × Stroop** | 0.186 | **0.0235** | 유의한 교차-과제 상관 |
| WCST × PRP | 0.109 | 0.186 | NS |
| WCST × Stroop | 0.107 | 0.195 | NS |

**참조 파일**: `advanced_analyses/meta_control/META_CONTROL_SUMMARY.txt`

---

### M. 여성 τ 매개 경로 (Tau Mediation)

| 경로 | β | p-value | 해석 |
|------|---|---------|------|
| **a (UCLA → τ) Female** | -3.735 | **0.0253** | 여성에서 UCLA가 τ 예측 |
| a (UCLA → τ) Male | -1.491 | 0.347 | NS |

> **해석**: 여성에서만 외로움이 τ(주의 이탈) 감소와 연관

**참조 파일**: `advanced_analyses/tau_mediation/MODERATED_MEDIATION_REPORT.txt`

---

### N. Trial-Level MVPA 분류기 성능

| 모델 | AUC | SD | 해석 |
|------|-----|-----|------|
| **전체 (PE 예측)** | 0.786 | 0.014 | Above-chance 예측 |
| 남성 | 0.793 | 0.024 | |
| 여성 | 0.775 | 0.022 | |

**Top 5 중요 특성**:
1. pe_last_10 (0.219)
2. trial_num_norm (0.187)
3. correct_streak (0.169)
4. rt_sd_last_5 (0.104)
5. current_rt (0.097)

**참조 파일**: `trial_mvpa/MVPA_VULNERABILITY_REPORT.txt`

---

## 3. 경계선 수준 결과 (0.05 < p < 0.10)

| 분석 | β | p-value | 해석 |
|------|---|---------|------|
| 남성 Stroop Interference (UCLA) | 33.13 | 0.092 | 남성에서 더 큰 간섭 효과 경향 |
| PRP Bottleneck × Gender Interaction | 38.14 | 0.100 | 남성에서 더 큰 병목 효과 경향 |
| PRP Bottleneck Quadratic (Full) | - | 0.059 | 비선형 효과 경향 |
| Stroop Kurtosis (log σ²) UCLA | -0.250 | 0.067 | 첨도 감소 경향 |
| UCLA × Age (z_ucla:age_mc) | 0.604 | 0.083 | 나이에 따른 UCLA 효과 변화 경향 |

---

## 4. 귀무가설 지지 (베이지안 분석)

| 종속변수 | BF₁₀ | 해석 |
|----------|------|------|
| WCST PE Rate | 0.23 | **Moderate evidence for null** (효과 없음 지지) |
| WCST Accuracy | 0.46 | Inconclusive |

> **해석**: BF < 0.33은 귀무가설(효과 없음)에 대한 중간 수준 증거

**참조 파일**: `deep_dive_analysis/statistical_robustness/bayesian_inference.csv`

---

## 5. 통계적 검정력 분석

| 지표 | 값 |
|------|-----|
| 전체 표본 | N = 169 |
| 남성 | N = 60 |
| 여성 | N = 106 |
| 평균 검정력 | 0.142 (14.2%) |
| 적정 검정력 분석 수 | 0/20 (0%) |
| 탐지 가능 최소 효과 (전체) | r ≥ 0.214 |
| 탐지 가능 최소 효과 (남성) | r ≥ 0.355 |

**향후 연구 권장 표본 크기**:
- r = 0.10 탐지: N = 783 (80% power), N = 1047 (90% power)
- r = 0.20 탐지: N = 194 (80% power), N = 259 (90% power)
- r = 0.30 탐지: N = 85 (80% power), N = 113 (90% power)

**참조 파일**: `extended_analyses/power_analysis/power_analysis_report.txt`

---

## 6. 다중비교 보정

| 종속변수 | 보정 전 유의 | 보정 후 유의 | 손실 |
|----------|-------------|-------------|------|
| WCST PE Rate | 0 | 0 | 0 |
| Stroop Interference | 1 | 0 | 1 |
| PRP Bottleneck | 0 | 0 | 0 |

**참조 파일**: `bonferroni_correction/bonferroni_summary.csv`

---

## 7. 네트워크 분석 요약

### 중심성 순위 (Centrality)
1. **Strength**: dass_depression (1.189) > dass_anxiety (1.103) > dass_stress (1.012)
2. **Betweenness**: dass_depression (0.571) > dass_anxiety (0.476) > stroop_interference (0.381)

### Bridge Centrality (교량 중심성)
1. wcst_accuracy (0.734)
2. ucla_score (0.728)
3. dass_depression (0.625)

### 성별 네트워크 비교
- Global strength difference: 3.038, p = 0.162 (NS)
- Max edge difference: 0.558, p = 0.116 (NS)

**참조 파일**: `extended_analyses/network_analysis/network_analysis_report.txt`

---

## 8. 주요 해석 및 결론

### 확정된 결론

1. **DASS 통제 후 UCLA 주 효과 없음**
   - 모든 집행기능 지표에서 UCLA 외로움의 독립적 효과 미유의
   - 외로움-EF 연관성은 정서적 고통(우울/불안/스트레스)의 혼입 효과

2. **분산/변동성 효과 존재**
   - 평균 수행이 아닌 RT 변동성, 분포 꼬리(τ)에서 UCLA 효과
   - WCST RMSSD, PRP σ, WCST τ 모두 유의

3. **성별 차이 명확**
   - 남성이 특정 메커니즘(baseline RT, noise, clustering)에서 차이
   - 시간적 취약성에서 남성 특이적 패턴

### 탐색적 발견

1. **Stroop 주의 이탈 확률** - UCLA 유의 효과 (p = 0.02)
2. **시간적 취약성** - 남성에서 자원 고갈 패턴 (3-way interaction 유의)
3. **클러스터링 성별 분리** - 여성-우세 취약성 프로필 확인
4. **Stroop 적응적 통제** - UCLA가 CSE, Reactive Control 예측 (p < 0.04)
5. **PRP τ × Gender 상호작용** - 남성에서 주의 이탈 증가 (p = 0.04)
6. **MANOVA 성별 주효과** - 3개 EF 다변량에서 유의 (p = 0.01)
7. **Trial-Level MVPA** - PE 예측 AUC = 0.79 (above-chance)

### 제한점

1. 표본 크기: N = 150-169 (중소규모)
2. 통계적 검정력 부족: 평균 14.2%
3. Stroop interference 신뢰도 중간 수준 (r = 0.58)
4. 횡단 연구 설계 (인과관계 추론 불가)

### 향후 연구 제언

1. 더 큰 표본 확보 (N > 300)
2. 종단 연구 설계 적용
3. 분산/변동성 지표 중심 가설 검증
4. 남성 특이적 시간적 취약성 메커니즘 탐구

---

## 9. 핵심 참조 파일

| 분석 유형 | 파일 경로 |
|-----------|----------|
| Gold Standard 확인 분석 | `master_dass_controlled/FINAL_DASS_CONTROL_REPORT.txt` |
| 분산/변동성 효과 | `paper1_distributional/paper1_location_scale_results.csv` |
| 주의 이탈 모형 | `attentional_lapse/LAPSE_MIXTURE_REPORT.txt` |
| 시간적 취약성 | `temporal_vulnerability/TEMPORAL_VULNERABILITY_REPORT.txt` |
| IIV 분해 | `iiv_decomposition/IIV_DECOMPOSITION_REPORT.txt` |
| 클러스터링 | `ef_vulnerability_clustering/CLUSTERING_REPORT.txt` |
| 성별 층화 | `synthesis_analysis/gender_stratified_coefficients.csv` |
| 베이지안 | `deep_dive_analysis/statistical_robustness/bayesian_inference.csv` |
| 검정력 분석 | `extended_analyses/power_analysis/power_analysis_report.txt` |
| 네트워크 분석 | `extended_analyses/network_analysis/network_analysis_report.txt` |
| MANOVA 다변량 | `multivariate_ef_analysis/MULTIVARIATE_EF_REPORT.txt` |
| PRP Ex-Gaussian | `prp_exgaussian_dass_controlled/EXGAUSSIAN_MECHANISTIC_SUMMARY.txt` |
| 적응적 회복 역학 | `adaptive_dynamics/RECOVERY_DYNAMICS_REPORT.txt` |
| Meta-Control 경로 | `advanced_comprehensive/ADVANCED_ANALYSES_SUMMARY.txt` |
| τ 매개분석 | `advanced_analyses/tau_mediation/MODERATED_MEDIATION_REPORT.txt` |
| Trial-Level MVPA | `trial_mvpa/MVPA_VULNERABILITY_REPORT.txt` |
| 측정 불변성 | `extended_analyses/measurement_invariance/invariance_report.txt` |
| 비선형 임계값 | `extended_analyses/nonlinear_threshold/nonlinear_analysis_report.txt` |

---

**보고서 생성일**: 2025-12-03
**최종 업데이트**: 2025-12-03 (추가 분석 결과 반영)
**생성 위치**: `results/analysis_outputs/SIGNIFICANT_FINDINGS_SUMMARY.md`
**검토한 보고서 수**: 35+ txt/csv 파일
