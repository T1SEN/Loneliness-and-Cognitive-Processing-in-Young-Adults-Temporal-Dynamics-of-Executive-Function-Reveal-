# UCLA 외로움-집행기능 연구: 완전한 유의한 결과 목록

**작성일**: 2025-12-03
**검토 파일 수**: 216 CSV + 45 TXT = 261개 파일 (전수 검토 완료)
**기준**: p < 0.05

---

## I. UCLA-DASS 상관 (매우 강력한 효과)

### 여성 (N=107)
| 변수 | r | p-value |
|------|---|---------|
| **UCLA-DASS Depression** | 0.638 | **1.4e-13** |
| **UCLA-DASS Anxiety** | 0.469 | **3.5e-07** |
| **UCLA-DASS Stress** | 0.446 | **1.4e-06** |
| **UCLA-DASS Total** | 0.605 | **5.0e-12** |

### 남성 (N=59)
| 변수 | r | p-value |
|------|---|---------|
| **UCLA-DASS Depression** | 0.734 | **3.7e-11** |
| **UCLA-DASS Anxiety** | 0.504 | **4.8e-05** |
| **UCLA-DASS Stress** | 0.633 | **7.6e-08** |
| **UCLA-DASS Total** | 0.710 | **2.9e-10** |

---

## II. 성별 주효과 (Gender Main Effects)

### A. MANOVA 다변량 분석
| 효과 | Wilks' λ | F | df | p-value |
|------|----------|---|-----|---------|
| **Gender Main Effect** | 0.921 | 3.934 | 3, 138 | **0.0099** |

### B. PRP 과제
| 측정치 | Gender β | p-value | 출처 |
|--------|----------|---------|------|
| **PRP Bottleneck** | -94.05 | **0.00051** | prp_comprehensive |
| **Mean T2 RT** | -127.40 | **0.0011** | prp_comprehensive |
| **SOA Slope** | 47.03 | **0.00051** | prp_comprehensive |
| **T2 RT IQR** | -96.33 | **0.00033** | prp_comprehensive |
| **PRP Baseline RT** | -143.73 | **0.0012** | iiv_decomposition |
| **PRP Residual Noise** | -45.55 | **0.0028** | iiv_decomposition |
| **PRP State Separation** | -75.55 | **0.0035** | attentional_lapse |
| **PRP Engaged Mean RT** | -70.41 | **0.018** | attentional_lapse |
| **PRP Lapse Mean RT** | -145.95 | **0.0028** | attentional_lapse |
| **μ (Gaussian mean)** | -142.55 | **0.0014** | prp_exgaussian |
| **σ (Gaussian SD)** | -88.36 | **0.0002** | prp_exgaussian |
| **Δμ bottleneck** | -147.71 | **0.0006** | prp_exgaussian |
| **μ at short SOA** | -151.16 | **0.0029** | prp_exgaussian |

### C. WCST 과제
| 측정치 | Gender β | p-value | 출처 |
|--------|----------|---------|------|
| **WCST Baseline RT** | -191.09 | **0.0026** | iiv_decomposition |
| **WCST Lapse Probability** | -0.043 | **0.025** | attentional_lapse |
| **WCST State Separation** | 223.91 | **0.019** | attentional_lapse |
| **WCST Engaged Mean RT** | -99.20 | **0.017** | attentional_lapse |
| **WCST Clustering Coef** | -0.083 | **0.016** | error_burst |

### D. Stroop 과제
| 측정치 | Gender β | p-value | 출처 |
|--------|----------|---------|------|
| **Stroop Early** | -62.16 | **0.046** | temporal_vulnerability |
| **Stroop Engaged Mean RT** | -55.36 | **0.039** | attentional_lapse |

### E. Cross-Task
| 측정치 | Gender β | p-value | 출처 |
|--------|----------|---------|------|
| **Cross-Task Range** | -80.57 | **0.0039** | cross_task_consistency |
| **Cross-Task SD** | -37.20 | **0.0026** | cross_task_consistency |

---

## III. UCLA 주효과 (UCLA Main Effects)

### A. Stroop 과제
| 측정치 | UCLA β | p-value | 출처 |
|--------|--------|---------|------|
| **Stroop Lapse Probability** | 0.0397 | **0.0198** | attentional_lapse |
| **Stroop CSE (Conflict Adaptation)** | -36.48 | **0.036** | adaptive_dynamics |
| **Stroop Reactive Control Index** | -26.34 | **0.0020** | adaptive_dynamics |

### B. PRP 과제
| 측정치 | UCLA β | p-value | 출처 |
|--------|--------|---------|------|
| **σ at short SOA** | 36.96 | **0.017** | prp_exgaussian |

---

## IV. UCLA × Gender 상호작용

### A. Ex-Gaussian Parameters
| 측정치 | Interaction β | p-value | 출처 |
|--------|---------------|---------|------|
| **PRP τ (Exponential tail)** | 50.68 | **0.043** | prp_exgaussian |

### B. Clustering
| 검정 | χ² | p-value | 출처 |
|------|-----|---------|------|
| **Gender × Cluster** | 4.249 | **0.039** | ef_vulnerability_clustering |

---

## V. DASS 하위척도 효과

### A. IIV 분해 분석
| 측정치 | DASS 변수 | β | p-value |
|--------|-----------|---|---------|
| **PRP Learning Rate** | Anxiety | 102.04 | **0.0022** |
| **PRP Baseline RT** | Anxiety | -81.27 | **0.010** |
| **PRP Baseline RT** | Stress | 77.40 | **0.015** |

### B. Attentional Lapse
| 측정치 | DASS 변수 | β | p-value |
|--------|-----------|---|---------|
| **Stroop Lapse Prob** | Depression | -0.043 | **0.0115** |
| **Stroop Engaged Mean RT** | Anxiety | -42.26 | **0.027** |
| **WCST Lapse Prob** | Anxiety | 0.028 | **0.046** |
| **WCST Lapse Prob** | Stress | -0.029 | **0.041** |
| **WCST Lapse Mean RT** | Anxiety | -161.87 | **0.044** |

### C. Temporal Vulnerability
| 측정치 | 변수 | β | p-value |
|--------|------|---|---------|
| **Stroop Slope** | Depression | 20.05 | **0.050** |

---

## VI. 시간적 효과 (Temporal/Epoch Effects)

### 3-way Interaction: UCLA × Gender × Epoch
| 과제 | β | p-value | 해석 |
|------|---|---------|------|
| **WCST epoch_num** | 0.0108 | **0.0065** | 남성 시간적 취약성 |
| **Stroop epoch_num** | 13.58 | **0.0007** | 남성 시간적 취약성 |

### Epoch 주효과
| 측정치 | 변수 | β | p-value |
|--------|------|---|---------|
| **WCST Middle** | Age | -0.014 | **0.017** |

---

## VII. Meta-Control 경로 분석

| 경로 | β | p-value |
|------|---|---------|
| **b: Meta-Control → PE** | -2.501 | **2.6e-13** |
| **Female a-path (UCLA → τ)** | -3.735 | **0.025** |

---

## VIII. PRP Ex-Gaussian UCLA 상관 (성별 층화)

### 여성 (Short SOA)
| 파라미터 | r | p-value |
|----------|---|---------|
| **σ** | 0.244 | **0.017** |
| **τ** | -0.243 | **0.018** |

### 남성 (Short SOA)
| 파라미터 | r | p-value |
|----------|---|---------|
| **σ** | 0.261 | **0.048** |

### PRP Bottleneck Ex-Gaussian 상관 (추가)
| 성별 | 파라미터 | r | p-value |
|------|----------|---|---------|
| **남성** | **μ_bottleneck** | 0.294 | **0.025** |
| **남성** | **σ_bottleneck** | 0.329 | **0.012** |
| **여성** | **τ_bottleneck** | -0.218 | **0.035** |

---

## IX. 남성 층화 분석 (Male-Stratified)

| 측정치 | 변수 | β / r | p-value |
|--------|------|-------|---------|
| **T2 RT CV** | UCLA | 0.026 | **0.018** |
| **PRP Bottleneck** | DASS Stress | 78.16 | **0.0063** |
| **PRP Bottleneck** | Age | 89.15 | **0.0032** |

---

## X. 분산/변동성 효과 (Paper 1)

| 측정치 | UCLA β | p-value |
|--------|--------|---------|
| **WCST RMSSD (log σ²)** | 0.656 | **0.005** |
| **PRP σ long SOA (log σ²)** | 0.716 | **0.006** |
| **WCST τ (log σ²)** | 0.737 | **0.004** |

---

## XI. 신뢰도 지표

| 측정치 | Split-half r | Spearman-Brown |
|--------|-------------|----------------|
| WCST PE Rate | 0.658 | 0.794 |
| Stroop Interference | 0.410 | 0.582 |

---

## XII. 분산 동질성 검정

| 측정치 | F-max | p-value |
|--------|-------|---------|
| **WCST PE Rate** | 2.375 | **0.00068** |

---

## XIII. Trial-Level MVPA 성능

| 모델 | AUC | 해석 |
|------|-----|------|
| **전체 (PE 예측)** | 0.786 | Above-chance |
| 남성 | 0.793 | |
| 여성 | 0.775 | |

---

## XIV. Post-Error Slowing (PES) GLMM

| 파라미터 | β | p-value | 출처 |
|----------|---|---------|------|
| **Gender** | -119.27 | **0.0061** | pes_glmm |
| **t2_error_prev (PES)** | 309.14 | **3.4e-07** | pes_glmm |
| **DASS Stress** | 72.38 | **0.022** | pes_glmm |
| **SOA** | -222.12 | **<1e-100** | pes_glmm |
| **Group Variance** | 0.352 | **4.8e-13** | pes_glmm |

---

## XV. Cascade GLMM (Error Propagation)

| 파라미터 | β | p-value | 출처 |
|----------|---|---------|------|
| **t2_error_prev** | 0.020 | **0.012** | cascade_glmm |

---

## XVI. Cross-Task 상관

| 과제 1 | 과제 2 | r | p-value |
|--------|--------|---|---------|
| **Stroop Interference** | **PRP Bottleneck** | 0.166 | **0.039** |

---

## XVII. RT 변동성 효과 (Paper 4)

| 과제 | 측정치 | UCLA β | p-value |
|------|--------|--------|---------|
| **WCST** | **SD_RT** | 78.79 | **0.047** |

---

## XVIII. Mixture Model 클러스터 차이 (Framework 1)

### A. 클러스터 간 차이 검정
| 변수 | F | p-value | 출처 |
|------|---|---------|------|
| **UCLA Score** | 9.17 | **1.3e-05** | class_differences_tests |
| **DASS Depression** | 26.08 | **1.2e-13** | class_differences_tests |
| **DASS Anxiety** | 12.16 | **3.5e-07** | class_differences_tests |
| **DASS Stress** | 17.06 | **1.3e-09** | class_differences_tests |
| **Age** | 3.58 | **0.015** | class_differences_tests |
| **Gender** | 149.88 | **2.8e-32** | class_differences_tests |

### B. 클래스별 UCLA 효과
| Class | DV | UCLA β | p-value | N |
|-------|-----|--------|---------|---|
| **Class 2** | **Stroop Interference** | 72.33 | **0.0056** | 30 |

---

## XIX. Threshold Effects (비선형 임계치)

| DV | Threshold | Group | r | p-value |
|----|-----------|-------|---|---------|
| **PRP Bottleneck** | 47 (67th %ile) | Above | 0.196 | **0.042** |
| **PRP Bottleneck** | 49 (75th %ile) | Above | 0.188 | **0.037** |

---

## 요약 통계

| 범주 | 유의한 결과 수 |
|------|---------------|
| UCLA-DASS 상관 | 8 |
| 성별 주효과 | 25+ |
| UCLA 주효과 | 5 |
| UCLA × Gender 상호작용 | 2 |
| DASS 하위척도 효과 | 9 |
| 시간적 효과 | 3 |
| Meta-Control 경로 | 2 |
| PRP Ex-Gaussian 상관 | 6 |
| 남성 층화 | 3 |
| 분산/변동성 | 4 |
| Post-Error Slowing | 5 |
| Cascade GLMM | 1 |
| Cross-Task 상관 | 1 |
| Mixture Model 클러스터 | 7 |
| Threshold Effects | 2 |
| **총계** | **82+ 유의한 결과** |

---

## 핵심 결론

1. **UCLA 주효과**: DASS 통제 후 대부분 사라짐 → 정서적 고통 혼입
2. **성별 주효과**: 매우 강력함 (25+ 유의한 결과)
3. **분산/변동성**: UCLA 효과가 평균이 아닌 변동성에서 나타남
4. **시간적 취약성**: 남성에서 과제 후반부 취약성 (3-way interaction)
5. **적응적 통제**: UCLA가 Stroop CSE, Reactive Control 예측
6. **Post-Error Slowing**: 강력한 PES 효과 (p<1e-06), 성별 및 스트레스 조절
7. **Ex-Gaussian 분해**: 남성에서 bottleneck μ/σ-UCLA 상관 유의

---

## 검토 완료 확인

**총 261개 파일** (`results/analysis_outputs/` 하위)을 체계적으로 전수 검토 완료:
- frequentist_results.csv, bayesian_results.csv (각 분석 폴더)
- *_correlations.csv, *_coefficients.csv
- *_summary.csv, *_results.csv
- class_differences_tests.csv, threshold_effect_tests.csv
- 모든 report 텍스트 파일

**미유의 결과 (p > 0.05)로 확인된 분석**:
- Specification curve analysis (48개 specification 모두 non-significant)
- TOST equivalence tests (non-equivalent)
- Anxiety stratified analysis (no sig. interactions)
- Age group stratified (no sig. interactions)
- Network comparison test (p > 0.05)
- Quadratic nonlinearity tests (all ns)
- Fatigue analysis (all ns)
- UCLA autocorrelation effects (all ns)
- Perseveration momentum correlations (all ns)
- Stroop Ex-Gaussian interference correlations (all ns)

---

**파일 생성**: 2025-12-03
**최종 업데이트**: 전수 검토 완료
**경로**: `results/analysis_outputs/ALL_SIGNIFICANT_FINDINGS_COMPLETE.md`
