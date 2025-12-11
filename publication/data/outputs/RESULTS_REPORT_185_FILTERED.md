# 185명 품질 필터링 데이터 분석 결과 보고서

**분석 일시:** 2025-12-11
**데이터:** `publication/data/complete_filtered/` (N=185)
**필터링 기준:** 음수 나이, 로그 오류, PRP/WCST/Stroop 품질 실패자 15명 제외

---

## 1. 기술통계 (Descriptive Statistics)

### 1.1 전체 표본 특성 (N=185)

| 변수 | N | M | SD | Min | Max | Skew | Kurt |
|------|---|---|-----|-----|-----|------|------|
| Age (years) | 185 | 20.60 | 1.94 | 18 | 29 | 1.28 | 2.40 |
| UCLA Loneliness | 185 | 40.03 | 10.96 | 20 | 70 | 0.39 | -0.46 |
| DASS-Depression | 185 | 7.32 | 7.56 | 0 | 38 | 1.57 | 2.62 |
| DASS-Anxiety | 185 | 5.42 | 5.55 | 0 | 26 | 1.20 | 0.86 |
| DASS-Stress | 185 | 9.97 | 7.34 | 0 | 30 | 0.67 | -0.23 |
| WCST PE Rate (%) | 185 | 9.99 | 4.37 | 0 | 25.8 | 0.71 | 1.66 |
| Stroop Interference (ms) | 185 | 135.71 | 100.69 | -94.2 | 408.8 | 0.59 | 0.08 |
| PRP Delay (ms) | 185 | 587.07 | 147.71 | 269.7 | 981.3 | 0.21 | -0.31 |

### 1.2 성별 분포

| Gender | N | % |
|--------|---|---|
| Male | 76 | 41.1% |
| Female | 109 | 58.9% |

### 1.3 성별 비교 (Independent t-tests)

| 변수 | Male M(SD) | Female M(SD) | t | p | Cohen's d | Sig |
|------|------------|--------------|---|---|-----------|-----|
| Age | 21.14 (2.14) | 20.22 (1.70) | 3.14 | **.002** | 0.49 | **Yes** |
| UCLA | 37.71 (11.13) | 41.65 (10.60) | -2.42 | **.017** | -0.36 | **Yes** |
| DASS-Dep | 5.87 (7.06) | 8.33 (7.76) | -2.24 | **.026** | -0.33 | **Yes** |
| DASS-Anx | 4.58 (5.80) | 6.00 (5.31) | -1.70 | .092 | -0.26 | No |
| DASS-Str | 8.84 (7.53) | 10.75 (7.12) | -1.73 | .085 | -0.26 | No |
| WCST PE | 9.39 (3.86) | 10.40 (4.67) | -1.61 | .109 | -0.23 | No |
| Stroop Int | 127.65 (97.91) | 141.33 (102.65) | -0.92 | .361 | -0.14 | No |
| PRP Delay | 531.62 (134.32) | 625.73 (144.81) | -4.54 | **<.001** | -0.67 | **Yes** |

**주요 발견:** 여성이 남성보다 UCLA 외로움, DASS 우울, PRP Delay가 유의하게 높음

---

## 2. 상관분석 (Correlation Analysis)

### 2.1 Pearson 상관행렬

|  | UCLA | DASS-Dep | DASS-Anx | DASS-Str | WCST PE | Stroop | PRP |
|--|------|----------|----------|----------|---------|--------|-----|
| UCLA | - | | | | | | |
| DASS-Dep | **.674*** | - | | | | | |
| DASS-Anx | **.500*** | **.597*** | - | | | | |
| DASS-Str | **.532*** | **.623*** | **.717*** | - | | | |
| WCST PE | -.097 | -.115 | -.071 | -.047 | - | | |
| Stroop | .012 | -.082 | -.014 | -.029 | .091 | - | |
| PRP | .143† | .020 | .078 | .043 | **.162*** | .135 | - |

*Note.* †p < .10, *p < .05, **p < .01, ***p < .001

### 2.2 UCLA와 주요 변수 상관 (95% CI)

| 변수 | r | 95% CI | p | Sig |
|------|---|--------|---|-----|
| DASS-Depression | .674 | [.587, .746] | <.001 | **Yes** |
| DASS-Anxiety | .500 | [.384, .601] | <.001 | **Yes** |
| DASS-Stress | .532 | [.420, .628] | <.001 | **Yes** |
| WCST PE | -.097 | [-.238, .048] | .189 | No |
| Stroop Interference | .012 | [-.132, .156] | .867 | No |
| PRP Delay | .143 | [-.001, .282] | .052 | **Marginal** |

**주요 발견:** UCLA-DASS 간 강한 상관 (r = .50-.67), UCLA-EF 직접 상관 없음

---

## 3. 위계적 회귀분석 (Hierarchical Regression)

### 3.1 모형 구조
- **Model 0:** Age + Gender (통제변수)
- **Model 1:** Model 0 + DASS-21 (Depression, Anxiety, Stress)
- **Model 2:** Model 1 + UCLA Loneliness
- **Model 3:** Model 2 + UCLA × Gender 상호작용

### 3.2 모형 비교 (ΔR² 검정)

| Outcome | ΔR² (DASS) | p | ΔR² (UCLA) | p | ΔR² (Int) | p |
|---------|------------|---|------------|---|-----------|---|
| WCST PE | .020 | .304 | .002 | .548 | .000 | .908 |
| Stroop | .011 | .562 | .008 | .258 | .001 | .764 |
| **PRP Delay** | .007 | .717 | **.024** | **.036** | .010 | .130 |

### 3.3 UCLA 주효과 (DASS 통제 후)

| Outcome | β | SE | t | p | Sig |
|---------|---|-----|---|---|-----|
| WCST PE | -0.27 | 0.44 | -0.60 | .547 | No |
| Stroop | 12.87 | 11.34 | 1.14 | .256 | No |
| **PRP Delay** | **31.77** | **15.03** | **2.11** | **.035** | **Yes** |

### 3.4 UCLA × Gender 상호작용

| Outcome | β | SE | t | p | Sig |
|---------|---|-----|---|---|-----|
| WCST PE | 0.08 | 0.65 | 0.12 | .908 | No |
| Stroop | 4.63 | 15.40 | 0.30 | .764 | No |
| PRP Delay | 30.46 | 20.02 | 1.52 | .128 | No |

### 3.5 성별 층화 UCLA 효과

| Outcome | Female β (p) | Male β (p) |
|---------|--------------|------------|
| WCST PE | -0.05 (.929) | -0.65 (.413) |
| Stroop | 11.77 (.438) | 13.08 (.493) |
| PRP Delay | 22.45 (.251) | 30.73 (.174) |

### 3.6 모형 진단

| Outcome | VIF max | Breusch-Pagan p | Shapiro-Wilk p |
|---------|---------|-----------------|----------------|
| WCST PE | 2.38 | .069 | <.001 |
| Stroop | 2.38 | .420 | .003 |
| PRP Delay | 2.38 | .518 | .217 |

**주요 발견:** DASS 통제 후 **PRP Delay에서만 UCLA main effect 유의** (β=31.77, p=.035)

---

## 4. 경로분석 (Path Analysis)

### 4.1 매개효과 검정: UCLA → DASS → EF

#### 4.1.1 Depression 매개 (전체 표본)

| 경로 | β | Sobel z | p | Bootstrap CI | Sig |
|------|---|---------|---|--------------|-----|
| UCLA → DASS-Dep | .678*** | - | <.001 | - | Yes |
| DASS-Dep → EF | -.137 | - | - | - | - |
| **Indirect Effect** | **-.093** | **-2.11** | **.035** | **[-.177, -.013]** | **Yes** |

#### 4.1.2 Depression 매개 (성별 층화)

| Gender | a path (UCLA→DASS) | b path (DASS→EF) | Indirect | 95% CI | Sig |
|--------|-------------------|------------------|----------|--------|-----|
| **Female** | .684*** | -.191* | **-.129** | **[-.245, -.023]** | **Yes** |
| Male | .650*** | -.080 | -.053 | [-.166, .069] | No |

#### 4.1.3 Anxiety 매개

| Gender | Indirect | 95% CI | Sobel p | Sig |
|--------|----------|--------|---------|-----|
| Female | -.034 | [-.111, .028] | .300 | No |
| Male | .012 | [-.071, .108] | .774 | No |
| **Total** | **-.009** | **[-.061, .040]** | **.758** | **No** |

#### 4.1.4 Stress 매개

| Gender | Indirect | 95% CI | Sobel p | Sig |
|--------|----------|--------|---------|-----|
| Female | -.055 | [-.136, .008] | .107 | No |
| Male | .058 | [-.044, .209] | .402 | No |
| **Total** | **-.016** | **[-.079, .046]** | **.592** | **No** |

### 4.2 대안 모형 검정

#### 4.2.1 EF → DASS → UCLA (역방향 매개)

| DASS | Gender | Indirect | 95% CI | Sig |
|------|--------|----------|--------|-----|
| Depression | Female | **-.232** | **[-.423, -.062]** | **Yes** |
| Depression | Male | -.003 | [-.200, .219] | No |
| Anxiety | Female | -.087 | [-.215, .043] | No |
| Anxiety | Male | .067 | [-.139, .300] | No |
| Stress | Female | -.127 | [-.271, .002] | Marginal |
| Stress | Male | .155 | [-.046, .429] | No |

### 4.3 성별 차이 유의성 검정

| DASS | Model | Diff p | 해석 |
|------|-------|--------|------|
| Depression | UCLA→DASS→EF | .671 | NS |
| Depression | DASS→UCLA→EF | **.046** | **Male > Female** |
| Depression | EF→DASS→UCLA | **.011** | **Female > Male** |
| Anxiety | EF→DASS→UCLA | **.040** | **Female > Male** |
| Stress | EF→DASS→UCLA | **<.001** | **Female > Male** |

### 4.4 모형 적합도 비교

| DASS | Best Model | AIC | BIC | Total R² |
|------|------------|-----|-----|----------|
| Depression | DASS→UCLA→EF | 780.66 | 803.20 | .248 |
| Anxiety | DASS→UCLA→EF | 844.44 | 866.98 | .134 |
| Stress | DASS→UCLA→EF | 836.01 | 858.55 | .151 |

---

## 5. 핵심 결론

### 5.1 주요 발견 요약

1. **UCLA-DASS 강한 연관:** r = .50-.67 (모두 p < .001)

2. **UCLA-EF 직접 효과:** DASS 통제 후 **PRP Delay에서만 유의** (β=31.77, p=.035)
   - WCST, Stroop에서는 효과 없음

3. **매개효과:** UCLA → **Depression** → EF 경로만 유의
   - 전체: Sobel z = -2.11, p = .035
   - **여성에서만 유의:** β = -.129, 95% CI [-.245, -.023]
   - 남성: β = -.053, 95% CI [-.166, .069] (NS)

4. **역방향 매개 (EF→DASS→UCLA):** 여성에서만 유의
   - Depression: β = -.232, p < .05
   - 남성과 유의한 차이 (p = .011)

5. **성별 차이:**
   - 여성이 UCLA, DASS-Depression, PRP Delay 더 높음
   - 매개효과는 여성에서만 관찰됨

### 5.2 이론적 함의

- 외로움(UCLA)은 우울(DASS-Depression)을 통해 집행기능(EF)에 간접적으로 영향
- 이 매개효과는 **여성에게 특이적** (gender-specific)
- 남성에서는 외로움-EF 연관이 관찰되지 않음

### 5.3 제한점

- 횡단연구 설계로 인과관계 추론 제한
- WCST, Stroop 잔차 비정규성 (Shapiro-Wilk p < .05)
- 다중검정 보정 미적용

---

## 6. 출력 파일 목록

### 6.1 Basic Analysis (12 files)
```
publication/data/outputs/basic_analysis/
├── table1_descriptives.csv
├── table1_descriptives_by_gender.csv
├── table1_categorical.csv
├── table1_gender_comparison.csv
├── correlation_matrix.csv
├── correlation_pvalues.csv
├── correlation_ci.csv
├── correlation_heatmap.png
├── ucla_correlations_detailed.csv
├── hierarchical_results.csv
├── model_comparison.csv
└── hierarchical_summary.txt
```

### 6.2 Path Analysis (45 files)
```
publication/data/outputs/path_analysis/
├── depression/
│   ├── loneliness_to_dass_to_ef/ (5 files)
│   ├── loneliness_to_ef_to_dass/ (5 files)
│   └── dass_to_loneliness_to_ef/ (5 files)
├── anxiety/
│   ├── loneliness_to_dass_to_ef/ (5 files)
│   ├── loneliness_to_ef_to_dass/ (5 files)
│   └── dass_to_loneliness_to_ef/ (5 files)
├── stress/
│   ├── loneliness_to_dass_to_ef/ (5 files)
│   ├── loneliness_to_ef_to_dass/ (5 files)
│   └── dass_to_loneliness_to_ef/ (5 files)
└── comprehensive/
    ├── comprehensive_results.csv
    ├── summary_indirect_effects.csv
    ├── gender_stratified_summary.csv
    └── model_fit_comparison.csv
```

---

*Report generated: 2025-12-11*
