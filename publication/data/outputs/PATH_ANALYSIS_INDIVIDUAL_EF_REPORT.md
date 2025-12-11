# 개별 EF 과제 경로분석 종합 보고서

**분석 일시:** 2025-12-11
**데이터:** `publication/data/complete_filtered/` (N=185, 품질 필터링 완료)
**분석 방법:** 성별 층화 매개분석 (Bootstrap 5000회)

---

## 분석 개요

### 배경
- 기존 분석: 3개 EF 지표를 composite로 통합하여 분석
- 문제점: EF 과제 간 상관이 거의 없음 (r = .09~.16)
- 해결: 각 EF 과제를 **개별 종속변수**로 분석

### 분석 구조
- **DASS 유형:** Depression, Anxiety, Stress (3개)
- **EF 과제:** WCST PE, Stroop Interference, PRP Bottleneck (3개)
- **경로 모델:** 3가지
  1. UCLA → DASS → EF (외로움 → 정서 → 집행기능)
  2. UCLA → EF → DASS (외로움 → 집행기능 → 정서)
  3. DASS → UCLA → EF (정서 → 외로움 → 집행기능)
- **총 모델 수:** 3 DASS × 3 EF × 3 경로 × 2 성별 = 54개

---

## 유의한 결과 요약

### ✅ 유의한 매개효과 (p < .05 또는 Bootstrap CI ≠ 0)

| # | DASS | EF Outcome | 경로 | 성별 | Sobel z | p | Bootstrap β | 95% CI |
|---|------|------------|------|------|---------|---|-------------|--------|
| 1 | **Depression** | **WCST PE** | UCLA→DASS→EF | **Female** | **-2.21** | **.027** | **-0.64** | **[-1.23, -0.13]** |
| 2 | **Stress** | **PRP** | UCLA→DASS→EF | **Female** | -1.87 | .062 | **-16.47** | **[-35.56, -1.19]** |
| 3 | **Depression** | **PRP** | DASS→UCLA→EF | **Male** | **2.01** | **.045** | **33.99** | **[1.08, 69.20]** |

---

## 상세 결과: Depression 매개

### 1. WCST Perseverative Error Rate

#### Model 1: UCLA → DASS-Dep → WCST PE
| Gender | a path (UCLA→DASS) | b path (DASS→PE) | Indirect β | 95% CI | Sobel z | p |
|--------|-------------------|------------------|------------|--------|---------|---|
| **Female** | .684*** | **-.930*** | **-0.64** | **[-1.23, -0.13]** | **-2.21** | **.027** |
| Male | .650*** | .548 | 0.37 | [-0.37, 1.25] | 0.86 | .387 |

**해석:** 여성에서 외로움이 높을수록 → 우울 증가 → WCST 보속오류 감소 (역설적)

#### Model 2: UCLA → WCST PE → DASS-Dep
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | 0.01 | [-0.01, 0.04] | 1.17 | .242 | No |
| Male | -0.01 | [-0.04, 0.01] | -0.61 | .541 | No |

#### Model 3: DASS-Dep → UCLA → WCST PE
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | 0.03 | [-0.63, 0.73] | 0.07 | .945 | No |
| Male | -0.65 | [-1.78, 0.31] | -1.17 | .241 | No |

---

### 2. Stroop Interference Effect

#### Model 1: UCLA → DASS-Dep → Stroop
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | -10.87 | [-27.78, 4.85] | -1.38 | .169 | No |
| Male | -15.29 | [-36.44, 5.58] | -1.38 | .169 | No |

#### Model 2: UCLA → Stroop → DASS-Dep
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | 0.00 | [-0.03, 0.02] | 0.13 | .898 | No |
| Male | -0.01 | [-0.05, 0.02] | -0.35 | .729 | No |

#### Model 3: DASS-Dep → UCLA → Stroop
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | 5.78 | [-10.41, 22.91] | 0.68 | .499 | No |
| Male | 14.87 | [-10.46, 40.63] | 1.07 | .285 | No |

---

### 3. PRP Bottleneck Effect

#### Model 1: UCLA → DASS-Dep → PRP
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | -20.15 | [-49.03, 7.48] | -1.41 | .159 | No |
| Male | -12.20 | [-37.76, 11.36] | -1.01 | .313 | No |

#### Model 2: UCLA → PRP → DASS-Dep
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | 0.00 | [-0.03, 0.03] | -0.14 | .891 | No |
| Male | -0.02 | [-0.07, 0.01] | -0.96 | .336 | No |

#### Model 3: DASS-Dep → UCLA → PRP ⭐
| Gender | Indirect β | 95% CI | Sobel z | p | Sig |
|--------|------------|--------|---------|---|-----|
| Female | 13.55 | [-8.88, 37.36] | 1.11 | .267 | No |
| **Male** | **33.99** | **[1.08, 69.20]** | **2.01** | **.045** | **Yes** |

**해석:** 남성에서 우울이 높을수록 → 외로움 증가 → PRP 지연 증가

---

## 상세 결과: Anxiety 매개

### 1. WCST PE
| Model | Female β [CI] | Male β [CI] | Sig |
|-------|---------------|-------------|-----|
| UCLA→DASS→EF | -0.17 [-0.69, 0.24] | 0.06 [-0.40, 0.59] | No |
| UCLA→EF→DASS | 0.01 [-0.01, 0.04] | -0.01 [-0.05, 0.03] | No |
| DASS→UCLA→EF | -0.21 [-0.69, 0.24] | -0.25 [-0.82, 0.25] | No |

### 2. Stroop
| Model | Female β [CI] | Male β [CI] | Sig |
|-------|---------------|-------------|-----|
| UCLA→DASS→EF | -4.47 [-16.17, 5.47] | 1.49 [-11.77, 16.55] | No |
| UCLA→EF→DASS | 0.00 [-0.01, 0.03] | 0.00 [-0.03, 0.04] | No |
| DASS→UCLA→EF | 1.63 [-9.75, 14.20] | 1.26 [-11.58, 13.61] | No |

### 3. PRP
| Model | Female β [CI] | Male β [CI] | Sig |
|-------|---------------|-------------|-----|
| UCLA→DASS→EF | -3.30 [-20.61, 12.22] | 1.23 [-14.44, 16.18] | No |
| UCLA→EF→DASS | 0.00 [-0.04, 0.03] | -0.01 [-0.05, 0.02] | No |
| DASS→UCLA→EF | 2.73 [-11.87, 18.78] | 14.72 [-0.36, 34.76] | No (marginal) |

---

## 상세 결과: Stress 매개

### 1. WCST PE
| Model | Female β [CI] | Male β [CI] | Sig |
|-------|---------------|-------------|-----|
| UCLA→DASS→EF | 0.05 [-0.36, 0.44] | -0.10 [-0.89, 0.73] | No |
| UCLA→EF→DASS | 0.00 [-0.03, 0.02] | 0.00 [-0.04, 0.04] | No |
| DASS→UCLA→EF | -0.29 [-0.76, 0.13] | -0.23 [-1.09, 0.56] | No |

### 2. Stroop
| Model | Female β [CI] | Male β [CI] | Sig |
|-------|---------------|-------------|-----|
| UCLA→DASS→EF | -6.90 [-18.46, 3.36] | 6.76 [-9.93, 26.73] | No |
| UCLA→EF→DASS | 0.00 [-0.02, 0.04] | 0.00 [-0.03, 0.03] | No |
| DASS→UCLA→EF | 2.58 [-7.62, 13.47] | -1.72 [-22.34, 15.70] | No |

### 3. PRP ⭐
| Model | Female β [CI] | Male β [CI] | Sig |
|-------|---------------|-------------|-----|
| **UCLA→DASS→EF** | **-16.47** | **[-35.56, -1.19]** | **Yes (Female)** |
| | 20.29 | [-1.83, 48.22] | No |
| UCLA→EF→DASS | 0.00 [-0.04, 0.03] | 0.02 [-0.02, 0.08] | No |
| DASS→UCLA→EF | 8.54 [-4.02, 23.76] | 7.12 [-19.57, 31.20] | No |

**해석:** 여성에서 외로움이 높을수록 → 스트레스 증가 → PRP 지연 감소 (역설적)

---

## Sobel Test 전체 결과표

### Depression

| EF | Model | Female z (p) | Male z (p) |
|----|-------|--------------|------------|
| WCST PE | UCLA→DASS→EF | **-2.21 (.027)** | 0.86 (.387) |
| WCST PE | UCLA→EF→DASS | 1.17 (.242) | -0.61 (.541) |
| WCST PE | DASS→UCLA→EF | 0.07 (.945) | -1.17 (.241) |
| Stroop | UCLA→DASS→EF | -1.38 (.169) | -1.38 (.169) |
| Stroop | UCLA→EF→DASS | 0.13 (.898) | -0.35 (.729) |
| Stroop | DASS→UCLA→EF | 0.68 (.499) | 1.07 (.285) |
| PRP | UCLA→DASS→EF | -1.41 (.159) | -1.01 (.313) |
| PRP | UCLA→EF→DASS | -0.14 (.891) | -0.96 (.336) |
| PRP | DASS→UCLA→EF | 1.11 (.267) | **2.01 (.045)** |

### Anxiety

| EF | Model | Female z (p) | Male z (p) |
|----|-------|--------------|------------|
| WCST PE | UCLA→DASS→EF | -0.81 (.419) | 0.20 (.843) |
| WCST PE | UCLA→EF→DASS | 0.69 (.491) | -0.18 (.855) |
| WCST PE | DASS→UCLA→EF | -0.89 (.371) | -0.86 (.389) |
| Stroop | UCLA→DASS→EF | -0.84 (.403) | 0.22 (.823) |
| Stroop | UCLA→EF→DASS | 0.13 (.900) | -0.04 (.967) |
| Stroop | DASS→UCLA→EF | 0.26 (.793) | 0.17 (.864) |
| PRP | UCLA→DASS→EF | -0.38 (.701) | 0.23 (.816) |
| PRP | UCLA→EF→DASS | -0.06 (.949) | -0.19 (.852) |
| PRP | DASS→UCLA→EF | 0.33 (.742) | 1.69 (.092) |

### Stress

| EF | Model | Female z (p) | Male z (p) |
|----|-------|--------------|------------|
| WCST PE | UCLA→DASS→EF | 0.23 (.818) | -0.30 (.768) |
| WCST PE | UCLA→EF→DASS | -0.22 (.823) | 0.26 (.791) |
| WCST PE | DASS→UCLA→EF | -1.26 (.208) | -0.50 (.619) |
| Stroop | UCLA→DASS→EF | -1.25 (.211) | 0.67 (.502) |
| Stroop | UCLA→EF→DASS | 0.13 (.899) | 0.22 (.828) |
| Stroop | DASS→UCLA→EF | 0.47 (.637) | -0.14 (.887) |
| PRP | UCLA→DASS→EF | **-1.87 (.062)** | 1.56 (.119) |
| PRP | UCLA→EF→DASS | -0.14 (.891) | 0.98 (.325) |
| PRP | DASS→UCLA→EF | 1.17 (.242) | 0.63 (.529) |

---

## 핵심 결론

### 1. 과제 특이적 효과
- **WCST PE:** Depression 매개 (여성만)
- **PRP:** Stress 매개 (여성), Depression 역방향 (남성)
- **Stroop:** 모든 경로에서 비유의 → 외로움/DASS와 독립적

### 2. 성별 특이적 패턴
| 성별 | 유의한 경로 | 방향 |
|------|------------|------|
| **여성** | UCLA → DASS → EF | 순방향 (외로움이 원인) |
| **남성** | DASS → UCLA → EF | 역방향 (우울이 원인) |

### 3. 역설적 결과에 대한 고려
- DASS-Dep → WCST PE: β = -0.93 (여성) → 우울 높을수록 PE 감소?
- DASS-Str → PRP: β = -36.52 (여성) → 스트레스 높을수록 PRP 지연 감소?
- **해석 주의:** 횡단연구 한계, 억제 효과, 또는 통계적 변동 가능성

### 4. EF Composite 분석과의 비교
| 분석 | 유의한 결과 |
|------|------------|
| Composite (이전) | UCLA → Dep → EF (여성), EF → Dep → UCLA (여성) |
| **Individual (현재)** | **3개 특이적 효과 발견** |

---

## 출력 파일 구조

```
publication/data/outputs/path_analysis/
├── depression/
│   ├── pe_rate/
│   │   ├── loneliness_to_dass_to_ef/ (5 files)
│   │   ├── loneliness_to_ef_to_dass/ (5 files)
│   │   └── dass_to_loneliness_to_ef/ (5 files)
│   ├── stroop_interference/ (same structure)
│   └── prp_bottleneck/ (same structure)
├── anxiety/ (same structure)
├── stress/ (same structure)
└── comprehensive/ (legacy)
```

---

## 통계적 주의사항

1. **다중비교 보정 미적용:** 54개 검정 중 3개 유의 (5.6%) - 우연 수준
2. **Bootstrap CI 우선:** Sobel test보다 Bootstrap이 더 robust
3. **효과 크기 작음:** 유의하더라도 실질적 의미 검토 필요
4. **횡단연구 한계:** 인과관계 추론 제한

---

*Report generated: 2025-12-11*
*Data: N=185 (quality-filtered from N=200)*
