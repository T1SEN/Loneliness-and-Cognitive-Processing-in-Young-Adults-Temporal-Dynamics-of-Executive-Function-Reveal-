# Analysis Results - Quality-Controlled Dataset (N=185)

**데이터셋:** `publication/data/complete_filtered/` (품질 통제 후 185명)
**분석 날짜:** 2025-12-12
**총 참가자:** N=185 (Male=76, Female=109)

**품질 통제 기준:**
- 음수 나이 제외 (1명)
- 비현실적 지속시간 제외 (3명)
- PRP 정확도 <80% 제외 (3명)
- WCST 총오류≥60 또는 보속반응≥60 제외 (7명)
- Stroop 정확도 <93% 또는 평균 RT>1.5s 제외 (2명)
- **총 15명 제외** (중복 포함)

---

## 1. Basic Analysis Results

### 1.1 Descriptive Statistics (Table 1)

| Variable | N | M | SD |
|----------|---|---|---|
| Age (years) | 185 | 20.60 | 1.94 |
| UCLA Loneliness | 185 | 40.03 | 10.96 |
| DASS-21 Depression | 185 | 7.32 | 7.56 |
| DASS-21 Anxiety | 185 | 5.42 | 5.55 |
| DASS-21 Stress | 185 | 9.97 | 7.34 |
| WCST PE Rate (%) | 185 | 9.99 | 4.37 |
| Stroop Interference (ms) | 185 | 135.71 | 100.69 |
| PRP Delay Effect (ms) | 185 | 587.07 | 147.71 |

| Gender | N | % |
|--------|---|---|
| Male | 76 | 41.1 |
| Female | 109 | 58.9 |

### 1.2 Gender Comparison (Welch's t-test)

| Variable | t | p | Cohen's d | 해석 |
|----------|---|---|-----------|------|
| **Age** | 3.14 | **0.002** | 0.49 | 남성이 더 나이가 많음 |
| **UCLA Loneliness** | -2.42 | **0.017** | -0.36 | 여성이 더 외로움 |
| **DASS-21 Depression** | -2.24 | **0.026** | -0.33 | 여성이 더 우울 |
| DASS-21 Anxiety | - | ns | - | |
| DASS-21 Stress | - | ns | - | |
| WCST PE Rate | - | ns | - | |
| Stroop Interference | - | ns | - | |
| **PRP Delay Effect** | -4.54 | **<0.001** | -0.67 | 여성이 더 큰 PRP 병목 |

### 1.3 Correlation Matrix (Pearson r)

|  | UCLA | DASS-Dep | DASS-Anx | DASS-Str | WCST PE | Stroop Int | PRP Delay |
|--|------|----------|----------|----------|---------|------------|-----------|
| UCLA | -- | | | | | | |
| DASS-Dep | 0.674*** | -- | | | | | |
| DASS-Anx | 0.500*** | 0.597*** | -- | | | | |
| DASS-Str | 0.532*** | 0.623*** | 0.717*** | -- | | | |
| WCST PE | -0.097 | -0.115 | -0.071 | -0.047 | -- | | |
| Stroop Int | 0.012 | -0.082 | -0.014 | -0.029 | 0.091 | -- | |
| PRP Delay | 0.143+ | 0.020 | 0.078 | 0.043 | 0.162* | 0.135 | -- |

***p < .001, **p < .01, *p < .05, +p < .10

**UCLA-EF Correlations with 95% CI:**
| Variable | r | 95% CI | p |
|----------|---|--------|---|
| DASS-Depression | 0.674 | [0.586, 0.746] | < 0.001 |
| DASS-Anxiety | 0.500 | [0.384, 0.601] | < 0.001 |
| DASS-Stress | 0.532 | [0.420, 0.628] | < 0.001 |
| WCST PE Rate | -0.097 | [-0.238, 0.048] | 0.189 |
| Stroop Interference | 0.012 | [-0.132, 0.156] | 0.867 |
| PRP Delay Effect | 0.143 | [-0.001, 0.282] | 0.052+ |

### 1.4 Hierarchical Regression (DASS-Controlled)

**Model Structure:**
- Model 0: EF ~ age + gender
- Model 1: EF ~ age + gender + DASS (depression, anxiety, stress)
- Model 2: EF ~ Model 1 + UCLA
- Model 3: EF ~ Model 2 + UCLA × Gender

**Significant UCLA Main Effects (ΔR² for UCLA step, p < .05):**

| Outcome | ΔR² (UCLA) | p | 해석 |
|---------|------------|---|------|
| **PRP Delay Effect** | 2.38% | < .05 | DASS 통제 후에도 UCLA 유의 |
| **WCST Post-Error Slowing** | 2.53% | < .05 | DASS 통제 후에도 UCLA 유의 |
| **Stroop Incongruent RT Slope** | 3.74% | < .05 | DASS 통제 후에도 UCLA 유의 |

**UCLA × Gender Interactions:** None (all p > .05)

---

## 2. Mechanism Analysis (DASS-Controlled)

### 2.1 PRP Ex-Gaussian Decomposition (N=175)

**Parameters by SOA:**

| SOA | mu (ms) | sigma (ms) | tau (ms) |
|-----|---------|------------|----------|
| Short (≤150ms) | 917.8 (291.2) | 186.2 (118.0) | 347.9 (163.8) |
| Long (≥1200ms) | 438.3 (129.0) | 61.8 (64.8) | 247.6 (147.4) |
| **Bottleneck Effect** | 479.5 (244.7) | 124.4 (116.4) | 100.3 (192.9) |

**DASS-Controlled Regressions (all ns):**

| Parameter | UCLA β | p |
|-----------|--------|---|
| Overall mu | 12.561 | 0.737 |
| Overall sigma | 8.924 | 0.671 |
| Overall tau | 1.580 | 0.954 |
| Short SOA mu | 48.181 | 0.212 |
| Short SOA sigma | 25.869 | 0.113 |
| Short SOA tau | -27.982 | 0.307 |
| Bottleneck mu | 41.940 | 0.166 |
| Bottleneck sigma | 18.479 | 0.206 |
| Bottleneck tau | -20.419 | 0.429 |

**Gender-Stratified Correlations (Male Only):**

| Parameter | r | p | 해석 |
|-----------|---|---|------|
| **Overall sigma** | 0.310 | **0.008** | 남성: UCLA↑ → RT 변동성↑ |
| **Short SOA mu** | 0.259 | **0.028** | 남성: UCLA↑ → 루틴처리 느림 |
| **Short SOA sigma** | 0.278 | **0.018** | 남성: UCLA↑ → RT 변동성↑ |
| Short SOA tau | -0.208 | 0.080+ | |
| **Bottleneck mu** | 0.299 | **0.011** | 남성: UCLA↑ → 병목 증가 |
| **Bottleneck sigma** | 0.267 | **0.023** | 남성: UCLA↑ → 병목 변동성↑ |

### 2.2 Stroop Ex-Gaussian Decomposition (N=174)

**Parameters by Condition:**

| Condition | mu (ms) | sigma (ms) | tau (ms) |
|-----------|---------|------------|----------|
| Congruent | 662.7 (155.0) | 117.2 (68.4) | 174.9 (102.7) |
| Incongruent | 732.1 (210.8) | 152.8 (92.5) | 239.3 (130.0) |
| **Interference** | 69.4 (143.7) | 35.6 (86.4) | 64.5 (135.1) |

**DASS-Controlled Regressions (all ns):**

| Parameter | UCLA β | p |
|-----------|--------|---|
| Congruent mu | -16.173 | 0.416 |
| Incongruent mu | -15.571 | 0.573 |
| Interference mu | 0.602 | 0.975 |
| Congruent tau | 11.087 | 0.356 |
| Incongruent tau | 20.629 | 0.257 |
| Interference tau | 9.542 | 0.622 |

**Gender-Stratified Correlations:** None significant

### 2.3 WCST Reinforcement Learning (N=175)

**Model Parameters:**

| Model | Parameter | M | SD |
|-------|-----------|---|---|
| Basic RW | alpha | 0.151 | 0.295 |
| Basic RW | beta | 0.467 | 1.135 |
| Asymmetric | alpha_pos | 0.400 | 0.419 |
| Asymmetric | alpha_neg | 0.475 | 0.431 |

**Model Comparison:**

| Model | BIC | AIC |
|-------|-----|-----|
| **Basic RW (best)** | 254.1 | 249.2 |
| Asymmetric RW | 258.0 | 250.6 |
| Forgetting RW | 258.3 | 250.9 |

**DASS-Controlled Regressions (all ns):**

| Parameter | UCLA β | SE | p |
|-----------|--------|----|----|
| alpha | 0.050 | 0.048 | 0.297 |
| beta | -0.070 | 0.134 | 0.604 |
| alpha_pos | 0.073 | 0.059 | 0.215 |
| alpha_neg | -0.004 | 0.061 | 0.954 |
| alpha_asymmetry | 0.077 | 0.086 | 0.370 |

### 2.4 WCST HMM Modeling (N=175)

**HMM State Characteristics:**

| Metric | M | SD |
|--------|---|---|
| Lapse Occupancy | 21.0% | - |
| P(Lapse→Focus) | 0.604 | - |
| Focus RT | 1197 ms | - |
| Lapse RT | 2545 ms | - |
| RT Difference | 1348 ms | - |
| Mean Lapse Duration | 2.91 trials | - |
| Mean Lapse Episodes | 8.3 | - |

**Gender-Stratified HMM Results:**

| Gender | N | Lapse Occupancy | UCLA→Lapse β | p | UCLA-Lapse r | p |
|--------|---|-----------------|--------------|---|--------------|---|
| Female | 103 | 21.2% | 0.756 | 0.699 | -0.046 | 0.646 |
| **Male** | 72 | 20.6% | **5.355** | **0.024** | **0.335** | **0.004** |

**해석:** 남성에서만 UCLA가 높을수록 주의 이탈(Lapse) 상태에 더 많이 머무름

**Mediation Analysis (UCLA → DASS → Lapse → PE):**

| Path | β | p |
|------|---|---|
| a (UCLA → DASS) | 5.332 | < 0.001 |
| b (DASS → Lapse) | -0.174 | 0.339 |
| c (Lapse → PE) | -0.005 | 0.835 |
| **Indirect Effect** | 0.0048 | ns (CI: [-0.051, 0.079]) |

---

## 3. Summary of Significant Findings (p < .05)

### 3.1 Basic Analysis

| 날짜 | 분석 | 결과변수 | 효과 | Statistic | p |
|------|------|----------|------|-----------|---|
| 2025-12-12 | Gender Comparison | Age | t-test | t=3.14 | 0.002 |
| 2025-12-12 | Gender Comparison | UCLA | t-test | t=-2.42 | 0.017 |
| 2025-12-12 | Gender Comparison | DASS-Dep | t-test | t=-2.24 | 0.026 |
| 2025-12-12 | Gender Comparison | PRP Delay | t-test | t=-4.54 | <0.001 |
| 2025-12-12 | Hierarchical Reg | PRP Delay Effect | UCLA main | ΔR²=2.38% | <0.05 |
| 2025-12-12 | Hierarchical Reg | WCST PES | UCLA main | ΔR²=2.53% | <0.05 |
| 2025-12-12 | Hierarchical Reg | Stroop Incong Slope | UCLA main | ΔR²=3.74% | <0.05 |

### 3.2 Mechanism Analysis

| 날짜 | 분석 | 결과변수 | 효과 | r/β | p | N |
|------|------|----------|------|-----|---|---|
| 2025-12-12 | PRP Ex-Gaussian (Male) | Overall sigma | UCLA corr | r=0.310 | 0.008 | 72 |
| 2025-12-12 | PRP Ex-Gaussian (Male) | Short SOA mu | UCLA corr | r=0.259 | 0.028 | 72 |
| 2025-12-12 | PRP Ex-Gaussian (Male) | Short SOA sigma | UCLA corr | r=0.278 | 0.018 | 72 |
| 2025-12-12 | PRP Ex-Gaussian (Male) | Bottleneck mu | UCLA corr | r=0.299 | 0.011 | 72 |
| 2025-12-12 | PRP Ex-Gaussian (Male) | Bottleneck sigma | UCLA corr | r=0.267 | 0.023 | 72 |
| 2025-12-12 | WCST HMM (Male) | Lapse Occupancy | UCLA→Lapse | β=5.355 | 0.024 | 72 |
| 2025-12-12 | WCST HMM (Male) | Lapse Occupancy | UCLA corr | r=0.335 | 0.004 | 72 |

---

## 4. Key Conclusions

1. **DASS 통제 후 UCLA 주효과**:
   - Hierarchical regression에서 3개 변수 유의 (PRP Delay, WCST PES, Stroop Slope)
   - UCLA × Gender 상호작용은 없음

2. **남성 특이적 취약성 (Gender-Stratified)**:
   - **PRP**: 남성에서만 UCLA가 ex-Gaussian 파라미터(mu, sigma)와 유의한 상관
   - **WCST HMM**: 남성에서만 UCLA가 Lapse 상태와 유의한 관계 (r=0.335, p=0.004)

3. **UCLA-DASS 관계**:
   - 높은 상관 (r=0.50~0.67)
   - 대부분의 UCLA 효과는 DASS 통제 후 소실

4. **Mechanism 분석 결과**:
   - DASS-Controlled 회귀에서는 UCLA 주효과 비유의 (Ex-Gaussian, RL 모두)
   - 성별 층화 분석에서만 남성 특이적 효과 발견

---

## 5. Reliability & Validity Analysis

### 5.1 Survey Internal Consistency (Cronbach's Alpha)

| Scale | Items | N | α | Interpretation |
|-------|-------|---|---|----------------|
| **UCLA Loneliness** | 20 | 185 | **0.936** | Excellent |
| **DASS-21 Total** | 21 | 185 | **0.914** | Excellent |
| DASS-21 Depression | 7 | 185 | 0.864 | Good |
| DASS-21 Anxiety | 7 | 185 | 0.763 | Acceptable |
| DASS-21 Stress | 7 | 185 | 0.821 | Good |

### 5.2 Cognitive Task Reliability (Split-Half)

| Task | Method | N | r | Spearman-Brown | Interpretation |
|------|--------|---|---|----------------|----------------|
| **WCST PE Rate** | First/Second Half | 206 | 0.621 | **0.766** | Acceptable |
| **PRP Bottleneck** | Odd/Even | 207 | 0.552 | **0.711** | Acceptable |
| Stroop Interference | Odd/Even | 185 | 0.367 | 0.537 | Poor |

### 5.3 Construct Validity (UCLA EFA)

**Factorability:**
- KMO = **0.933** (Marvelous)
- Bartlett's Test: χ² = 2043.35, p < 0.001

**2-Factor Solution:**
- Factor 1: Social isolation items (q2, q4, q7, q8, q11-14, q17-18)
- Factor 2: Emotional connectedness items (q1, q5, q6, q9, q10, q15, q16, q19, q20)
- Communalities: 0.18 - 0.69

### 5.4 Convergent Validity (UCLA-DASS)

| Correlation | r | p |
|-------------|---|---|
| UCLA - DASS Depression | 0.674 | < 0.001 |
| UCLA - DASS Anxiety | 0.500 | < 0.001 |
| UCLA - DASS Stress | 0.532 | < 0.001 |

**해석:** UCLA와 DASS는 강한 상관을 보여 수렴 타당도 지지

### 5.5 Criterion Validity (UCLA → EF)

| Outcome | UCLA β | p | ΔR² |
|---------|--------|---|-----|
| **PRP Bottleneck** | 36.322 | **0.016** | 3.13% |
| Stroop Interference | 13.866 | 0.179 | 0.98% |
| WCST PE Rate | -0.214 | 0.636 | 0.12% |

**해석:** PRP Bottleneck에서만 UCLA가 유의한 예측력

### 5.6 Data Quality

**Survey Response Quality:**

| Survey | M Duration | M/item | Too Fast (<2s/item) | Straight-line |
|--------|------------|--------|---------------------|---------------|
| UCLA | 154.7s | 7.73s | 0 (0.0%) | 0 (0.0%) |
| DASS | 80.8s | 3.85s | 25 (13.5%) | 7 (3.8%) |

**Cognitive Task Quality:**

| Task | Total Trials | Valid (%) | Timeouts | Anticipations | M Accuracy |
|------|--------------|-----------|----------|---------------|------------|
| Stroop | 19,980 | 99.8% | 34 (0.2%) | 0 | 0.987 |
| WCST | 15,876 | 97.3% | 0 | 1 | 0.836 |
| PRP | 22,200 | 99.1% | 185 (0.8%) | 14 (0.1%) | 0.967 |

**Sample:**
- Attrition: 0% (185명 전원 완료)
- Below chance performance: 0명

---

## 6. Output File Locations

| Analysis | Directory |
|----------|-----------|
| Basic Analysis | `publication/data/outputs/basic_analysis/` |
| PRP Ex-Gaussian | `publication/data/outputs/mechanism_analysis/prp_exgaussian/` |
| Stroop Ex-Gaussian | `publication/data/outputs/mechanism_analysis/stroop_exgaussian/` |
| WCST RL Modeling | `publication/data/outputs/mechanism_analysis/wcst_rl_modeling/` |
| WCST HMM Modeling | `publication/data/outputs/mechanism_analysis/wcst_hmm_modeling/` |
| Validity & Reliability | `publication/data/outputs/validity_reliability/` |
