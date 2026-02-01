# Supplementary Materials

## S1. 기술 사양

- 구현: Flutter Web + Firebase
- 접속: 데스크톱 브라우저만 허용
- RT 기록: `performance.now()` 기반

## S2. QC 및 전처리 기준

### S2.1 Trial‑level

| 과제 | 기준 | 규칙 |
|---|---|---|
| Stroop | RT 범위 | 200–3000 ms (`is_rt_valid`) |
| Stroop | Timeout | 보존하되 오답 처리, RT 평균에서는 제외 |
| WCST | rule/card 유효성 | colour/shape/number, 4개 기준 카드만 유지 |
| WCST | RT 최소 | < 200 ms 제거 |
| WCST | RT 유효 범위 | 200–10,000 ms (`is_rt_valid`) |
| WCST | Timeout | 오답 처리, RT 평균에서는 제외 |

### S2.2 Participant‑level

| 과제 | 기준 | 임계값 |
|---|---|---|
| 공통 | 설문 유효 | UCLA 총점 + DASS 3척도 + 성별 정보 존재 |
| 공통 | 과제 완료 | Stroop + WCST 완료 기록 |
| Stroop | 완주/정확도 | 108 trial 완료, 정확도 ≥ .70 |
| WCST | 유효 trial | ≥ 60 |
| WCST | 단일 카드 비율 | ≤ .85 |

## S3. 주요 변수 산출

### S3.1 Stroop

```
stroop_interference = mean(rt_ms | incongruent, timeout=False, rt_valid=True)
                   - mean(rt_ms | congruent, timeout=False, rt_valid=True)
```

### S3.2 WCST Perseverative Error Rate

```
wcst_perseverative_error_rate = 100 * sum(isPE) / n_trials
```

## S4. WCST Phase 정의

### S4.1 3‑phase (정석)

Rule segment는 `ruleAtThatTime` 변화로 구간을 나누며 최대 6개 category만 사용한다.

- **exploration**: rule 전환 이후 첫 정답 전
- **confirmation**: 첫 정답부터 3연속 정답 달성까지(포함)
- **exploitation**: 3연속 정답 달성 이후
- 3연속 정답이 없으면 첫 정답 이후 구간은 confirmation
- 정답이 전혀 없으면 전 구간 exploration

### S4.2 2‑phase (보조)

- **pre‑exploitation** = exploration + confirmation
- **exploitation** = 3연속 정답 달성 이후

모든 phase RT는 **all‑trials 기준**이며, `is_rt_valid` 및 비‑timeout trial만 평균에 포함한다.

## S5. WCST Phase 타당도 (OLS, DASS 통제)

공변량: DASS‑Dep/Anx/Stress + age + gender. OLS(non‑robust).

### S5.1 3‑phase 회귀

| Outcome | n | UCLA β | p |
|---|---:|---:|---:|
| exploration RT (all) | 208 | 106.173 | 0.2327 |
| confirmation RT (all) | 212 | 127.352 | 0.00934 |
| exploitation RT (all) | 212 | 27.148 | 0.4579 |
| confirmation − exploitation | 212 | 100.204 | 0.01115 |

### S5.2 2‑phase 회귀

| Outcome | n | UCLA β | p |
|---|---:|---:|---:|
| pre‑exploitation RT (all) | 212 | 122.573 | 0.01010 |
| exploitation RT (all) | 212 | 27.148 | 0.4579 |
| pre‑exploitation − exploitation | 212 | 95.425 | 0.01296 |

## S6. WCST Phase 신뢰도 (odd/even category split‑half)

| Phase | n | r | Spearman‑Brown |
|---|---:|---:|---:|
| exploration | 183 | 0.1862 | 0.3139 |
| confirmation | 210 | 0.5955 | 0.7464 |
| exploitation | 209 | 0.7484 | 0.8561 |
| pre‑exploitation | 210 | 0.6472 | 0.7858 |

---

**출력 파일**
- `outputs/stats/analysis/overall/wcst_phase_rt_ols_alltrials.csv`
- `outputs/stats/analysis/overall/wcst_phase_pre_exploit_rt_ols_alltrials.csv`
- `outputs/stats/analysis/overall/wcst_phase_split_half_reliability.csv`
