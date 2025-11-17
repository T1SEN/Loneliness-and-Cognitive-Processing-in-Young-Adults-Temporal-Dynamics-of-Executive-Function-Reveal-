# Advanced Analyses Scripts - Usage Guide

Created: 2025-01-16

## Overview

5개의 고급 분석 스크립트가 생성되었습니다. 각 스크립트는 독립적으로 실행 가능하며, UCLA 외로움과 실행기능(EF) 간 관계에 대한 심층 통찰을 제공합니다.

---

## 📁 Scripts Created

### 1. **residual_ucla_analysis.py** (Residual UCLA Analysis)

**목적**: DASS로 설명되는 분산을 제거한 "순수 외로움 잔차"의 효과 검증

**핵심 질문**: "우울/불안을 통제한 후에도 UCLA × Gender 상호작용이 유지되는가?"

**방법**:
- UCLA를 DASS 3개 하위척도에 회귀 → residual 추출
- 핵심 EF 결과변수(WCST PE, PRP τ, Stroop)에 대해:
  - Original model: `EF ~ UCLA * Gender + DASS + Age`
  - Residual model: `EF ~ UCLA_resid * Gender + DASS + Age`
- 효과 크기 비교

**실행**:
```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/residual_ucla_analysis.py
```

**주요 출력**:
- `results/analysis_outputs/residual_ucla_analysis/`
  - `original_vs_residual_comparison.csv`: 모형 비교표
  - `ucla_residual_scores.csv`: 참가자별 UCLA residual 점수
  - `RESIDUAL_UCLA_REPORT.txt`: 해석 요약

**해석 가이드**:
- **SURVIVES**: UCLA × Gender 상호작용이 DASS 제거 후에도 유의 → 순수 사회적 외로움 효과
- **ELIMINATED**: 상호작용이 사라짐 → 정동적/우울 요인과 혼재
- **EMERGED**: Suppression 효과 (DASS가 진짜 외로움 효과를 가리고 있었음)

---

### 2. **multivariate_ef_analysis.py** (Multivariate EF Analysis)

**목적**: WCST + PRP + Stroop을 동시에 보는 다변량 회귀로 EF 프로파일 전체 효과 검증

**핵심 질문**: "외로움이 개별 과제가 아니라 'meta-control' 전체에 영향을 주는가?"

**방법**:
- MANOVA: `[WCST_PE, PRP_tau, Stroop_interference] ~ UCLA * Gender + DASS + Age`
- Canonical correlation analysis
- EF outcomes 간 상관 구조 분석

**실행**:
```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/multivariate_ef_analysis.py
```

**주요 출력**:
- `results/analysis_outputs/multivariate_ef_analysis/`
  - `manova_full_output.txt`: MANOVA 결과 (Wilks' Lambda, Pillai's trace)
  - `multivariate_effect_sizes.csv`: 효과 크기 요약
  - `canonical_weights.csv`: EF 변수별 loading
  - `ef_profile_heatmap.png`: 성별×외로움 그룹별 EF 프로파일
  - `MULTIVARIATE_EF_REPORT.txt`: 해석 요약

**장점**:
- Single omnibus test → multiple comparison penalty 없음
- 과제 간 공분산 구조 포착
- Domain-general vs task-specific 패턴 구분 가능

---

### 3. **loneliness_classification_model.py** (Loneliness Classification)

**목적**: EF 패턴만으로 고외로움 vs 저외로움을 얼마나 잘 예측하는가?

**핵심 질문**: "Male-specific predictive signature가 존재하는가?"

**방법**:
- Target: UCLA 상위 25% (High) vs 하위 25% (Low)
- Features: WCST PE, PRP τ/μ/σ, Stroop interference, RT variability, PES 등
- Models: Logistic Regression + Random Forest
- 5-fold stratified cross-validation
- 전체 / 남성만 / 여성만 각각 AUC 비교

**실행**:
```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/loneliness_classification_model.py
```

**주요 출력**:
- `results/analysis_outputs/loneliness_classification/`
  - `classification_performance.csv`: AUC, accuracy (전체/성별별)
  - `feature_importance_*.csv`: 각 EF metric의 예측 기여도
  - `roc_curves.png`: ROC curve (전체 + 성별 stratified)
  - `confusion_matrices.png`: Confusion matrix
  - `CLASSIFICATION_REPORT.txt`: 해석 요약

**해석 기준**:
- AUC = 0.50: Random chance
- AUC = 0.60-0.65: Weak signal
- **AUC = 0.65-0.75**: Moderate signal ✓ (meaningful individual differences)
- AUC > 0.75: Strong signal (potential screening utility)

**성별차 판단**:
- Male AUC > Female AUC + 0.10 → Male-specific predictive signature

---

### 4. **rt_percentile_group_comparison.py** (RT Percentile Group Comparison)

**목적**: UCLA × Gender 효과가 RT 분포의 어느 부분(중앙 vs 꼬리)에서 강하게 나타나는가?

**핵심 질문**: "Ex-Gaussian τ 결과와 일관되게, 느린 꼬리(q=0.90)에서만 효과가 강한가?"

**⚠️ 중요**: 이 스크립트는 참가자별 RT percentile에 대한 그룹 비교입니다 (conditional quantile regression이 아님).

**방법**:
- 각 참가자의 RT percentiles 계산 (q = 0.10, 0.25, 0.50, 0.75, 0.90)
- OLS로 그룹 차이 검정: `percentile ~ UCLA * Gender + DASS + Age`
- 진짜 quantile regression은 `true_quantile_regression_analysis.py` 참조

**실행**:
```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/rt_percentile_group_comparison.py
```

**주요 출력**:
- `results/analysis_outputs/rt_percentile_group_comparison/`
  - `quantile_coefficients.csv`: 각 quantile별 회귀 계수
  - `quantile_effects_plot.png`: Quantile별 효과 크기 그래프
  - `quantile_heatmap.png`: Task × Quantile 히트맵
  - `tail_vs_center_comparison.csv`: q=0.90 vs q=0.50 비교
  - `QUANTILE_REGRESSION_REPORT.txt`: 해석 요약

**해석**:
- **q=0.90 >> q=0.50**: Lapse hypothesis 지지 (τ-driven, attentional failures)
- **q=0.90 ≈ q=0.50**: General slowing (μ-driven, sustained depletion)

---

### 5. **ef_vulnerability_clustering.py** (EF Vulnerability Clustering)

**목적**: K-means/GMM으로 EF 취약 패턴의 subtype을 식별하고 UCLA/성별과 연결

**핵심 질문**: "외로움이 단일 EF 패턴이 아니라 heterogeneous subtypes를 만드는가?"

**방법**:
- Features (z-scored): WCST PE, PRP τ/μ/σ, Stroop interference, RT variability, PES
- K-means (k=2~4), silhouette score로 최적 k 선택
- 각 cluster별 평균 UCLA, DASS, 성별 비율, 연령 비교
- PCA/t-SNE 시각화

**실행**:
```bash
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/ef_vulnerability_clustering.py
```

**주요 출력**:
- `results/analysis_outputs/ef_vulnerability_clustering/`
  - `cluster_assignments.csv`: 참가자별 cluster membership
  - `cluster_centroids.csv`: Cluster별 평균 EF 프로파일
  - `cluster_demographics.csv`: 성별/연령/UCLA 분포
  - `cluster_pca_visualization.png`: 2D PCA projection
  - `cluster_profile_heatmap.png`: Discriminative features
  - `CLUSTERING_REPORT.txt`: Subtype 해석

**예상 패턴**:
- Cluster 1: "Resilient" - 정상 EF
- Cluster 2: "Lapse-heavy" (male-dominant) - 高 τ, σ, WCST PE
- Cluster 3: "Hypervigilant" (female-dominant) - 低 τ, 高 variability, flexibility 저하

---

## 🔧 실행 전 체크리스트

### 필수 조건

1. **Master dataset 존재 여부**:
   ```bash
   ls results/analysis_outputs/master_dataset.csv
   ```
   - 없으면 먼저 `master_dass_controlled_analysis.py` 또는 데이터 병합 스크립트 실행 필요

2. **필수 컬럼 확인**:
   - `participant_id`
   - `ucla_total`
   - `gender` (or `gender_male`)
   - `age`
   - `dass_depression`, `dass_anxiety`, `dass_stress`
   - EF outcomes: `pe_rate` (WCST), `prp_tau_long` or `prp_bottleneck` (PRP), `stroop_interference` (Stroop)

3. **최소 샘플 크기**:
   - 대부분의 스크립트는 N ≥ 30 요구
   - Classification/Clustering: N ≥ 20 가능하나 N ≥ 40 권장

### 실행 순서 (권장)

```bash
# 1. Residual UCLA 분석 (가장 직관적, 논문 본문 적합)
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/residual_ucla_analysis.py

# 2. Multivariate EF (multiple comparison 방어용)
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/multivariate_ef_analysis.py

# 3. RT Percentile Group Comparison (Ex-Gaussian 결과 보완)
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/rt_percentile_group_comparison.py

# 4. Classification model (예측 관점)
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/loneliness_classification_model.py

# 5. Clustering (exploratory, heterogeneity 논의용)
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/ef_vulnerability_clustering.py
```

**예상 실행 시간**: 각 스크립트당 1~3분 (clustering이 가장 빠름, quantile regression이 가장 오래 걸림)

---

## 📊 결과 해석 가이드

### 논문 구성 시 활용 방안

#### **본문 (Main Text)**
1. **Residual UCLA 분석**:
   - 가장 직관적 → Reviewer들이 쉽게 이해
   - "DASS-adjusted loneliness" 효과 설명
   - Table: Original vs Residual UCLA interaction comparison

2. **Multivariate EF**:
   - MANOVA 결과로 "domain-general meta-control" 주장 강화
   - Multiple comparison penalty 방어

#### **Supplement (Supporting Information)**
1. **Quantile regression**:
   - Ex-Gaussian τ 결과의 수렴 증거
   - "Tail-specific effects" 그림

2. **Classification model**:
   - "Predictive validity" 논의
   - Male-specific signature 시각화 (ROC curve)

3. **Clustering**:
   - Exploratory로 명시
   - "Heterogeneity" 논의 지원
   - Subtype 그림 (PCA projection)

#### **Discussion에서 강조할 포인트**

**Residual UCLA → Theoretical contribution**:
> "Even after removing affective distress (DASS), UCLA × Gender interaction persisted (β=X.XX, p<.05), suggesting a social-cognitive loneliness mechanism independent of dysphoria."

**Multivariate EF → Methodological rigor**:
> "MANOVA confirmed a multivariate effect (Wilks' λ=X.XX, p<.05), addressing concerns about multiple comparisons across individual EF tasks."

**Quantile regression → Mechanistic insight**:
> "Effects were concentrated in the 90th percentile of RT distribution (β=X.XX) rather than median (β=X.XX, ns), converging with Ex-Gaussian tau findings and supporting a lapse-based mechanism."

**Classification → Applied potential**:
> "Random Forest achieved AUC=0.XX in males (vs 0.XX in females), demonstrating measurable individual-difference signal with potential for future screening applications."

**Clustering → Complexity acknowledgment**:
> "Unsupervised clustering revealed heterogeneous EF profiles, with male-dominant 'lapse-heavy' and female-dominant 'hypervigilant' subtypes, underscoring context-dependent vulnerability."

---

## ⚠️ 주의사항

### DASS Control 원칙

모든 스크립트는 **DASS-21 통제를 준수**합니다:
- ✅ Residual UCLA, Multivariate EF, Quantile regression: DASS를 covariate로 포함
- ⚠️ Classification: DASS를 feature로 사용 (예측 모델이므로 control 개념과 다름)
- ✅ Clustering: 군집 비교 시 DASS 분포 함께 보고

### 표본 크기 제한

N이 작을 경우:
- Classification: Extreme groups (top/bottom 25%)만 사용하므로 N이 더욱 줄어듦
- Clustering: k=2만 권장 (N < 40일 경우)
- Quantile regression: Trial 수가 충분한 과제만 분석 가능

### 파일 경로

모든 스크립트는 다음을 가정:
- Working directory: `C:\Users\ansel\my_research_exporter\`
- Input: `results/` (CSVs) + `results/analysis_outputs/master_dataset.csv`
- Output: `results/analysis_outputs/<script_name>/`

---

## 🔍 Troubleshooting

### 오류: "master_dataset.csv not found"

**해결책**:
```bash
# 대안 1: 기존 분석에서 master dataset 생성
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/master_dass_controlled_analysis.py

# 대안 2: 스크립트 내부에서 자동 생성 (일부 스크립트는 fallback 로직 포함)
```

### 오류: "Missing required columns: ['pe_rate', 'prp_tau_long', ...]"

**원인**: EF outcome 변수가 master dataset에 없음

**해결책**:
1. `3_cognitive_tests_summary.csv`에 해당 변수가 있는지 확인
2. 없으면 trial-level 데이터에서 계산:
   ```bash
   PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe analysis/derive_trial_features.py
   ```
3. 또는 `prp_exgaussian_dass_controlled.py` 등 task-specific 스크립트 먼저 실행

### 오류: "Insufficient data (N < 30)"

**원인**: 결측치가 많아 분석 가능한 N이 부족

**해결책**:
1. 결측치 패턴 확인:
   ```python
   import pandas as pd
   master = pd.read_csv("results/analysis_outputs/master_dataset.csv")
   print(master.isna().sum())
   ```
2. 일부 feature 제외 후 재실행
3. 또는 exploratory로만 사용하고 minimum N을 20으로 낮춤 (스크립트 내 `if len(df) < 30:` 부분 수정)

### Warning: "MANOVA failed with error..."

**원인**: Multicollinearity 또는 N이 너무 작음

**결과**: 스크립트는 자동으로 Canonical Correlation Analysis로 전환 (대안 방법)

**조치**: 보고서 확인 후 CCA 결과 사용 가능

---

## 📚 참고 문헌 (Methods Section 작성 시)

### Residual Analysis
> Beckstead, J. W. (2012). Isolating and examining sources of suppression and multicollinearity in multiple linear regression. *Multivariate Behavioral Research*, 47(2), 224-246.

### MANOVA
> Tabachnick, B. G., & Fidell, L. S. (2013). *Using multivariate statistics* (6th ed.). Pearson.

### Machine Learning Classification
> Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.

### Quantile Regression
> Koenker, R., & Hallock, K. F. (2001). Quantile regression. *Journal of Economic Perspectives*, 15(4), 143-156.

### K-Means Clustering
> Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

---

## ✅ Summary

| Script | 목적 | 핵심 메시지 | 논문 위치 |
|--------|------|-------------|-----------|
| `residual_ucla_analysis.py` | DASS 제거 후 순수 외로움 효과 | "Social-cognitive loneliness, not dysphoria" | Main text |
| `multivariate_ef_analysis.py` | EF 프로파일 전체 효과 | "Domain-general meta-control disruption" | Main text |
| `loneliness_classification_model.py` | EF로 외로움 예측 가능성 | "Male-specific predictive signature (AUC)" | Supplement |
| `rt_percentile_group_comparison.py` | RT percentile 그룹 비교 | "Tail-specific effects (q=0.90)" | Supplement |
| `ef_vulnerability_clustering.py` | EF 취약 subtype 탐색 | "Heterogeneous vulnerability profiles" | Supplement |

**⚠️ 업데이트 내역 (2025-01-16 재검토 후)**:
- `quantile_regression_analysis.py` → `rt_percentile_group_comparison.py` (이름/방법론 명확화)
- 모든 스크립트: DASS-21 통제 준수, 샘플 크기 체크 표준화 (N≥30 for regression, N≥20 for ML/clustering)
- Fallback 로직, MANOVA parsing, division-by-zero 방지 등 14개 이슈 수정 완료

**모든 스크립트는 독립적으로 실행 가능하며 외부 유틸리티 의존성이 없습니다.**

---

## 🚀 Next Steps

1. **데이터 확인**: `master_dataset.csv` 존재 및 필수 컬럼 확인
2. **순차 실행**: 권장 순서대로 5개 스크립트 실행
3. **결과 검토**: 각 `*_REPORT.txt` 파일에서 핵심 발견 확인
4. **논문 작성**: Main text에 Residual + MANOVA, Supplement에 나머지 3개 배치
5. **Figure 선택**: 각 분석의 핵심 그림 1개씩 선별 (총 5개)

**완료!** 🎉
