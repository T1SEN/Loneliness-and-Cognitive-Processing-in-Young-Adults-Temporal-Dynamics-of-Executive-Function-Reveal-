# Methods (Detailed)

## 2.4 측정

### 2.4.1 자기보고 척도

**UCLA Loneliness Scale (Version 3; UCLA‑LS)**. UCLA 외로움 척도 20문항 한국어판을 사용했다. 응답은 1(전혀 그렇지 않았다)–4(자주 그랬다) Likert 척도이며, 긍정 진술 9문항(1, 5, 6, 9, 10, 15, 16, 19, 20)은 역채점했다. 총점(20–80)은 모든 문항 응답을 합산해 계산했다. 미응답은 인터페이스에서 차단하여 결측 없이 수집했다.

**Depression Anxiety Stress Scales (DASS‑21)**. DASS‑21 한국어판을 사용했고, 0–3 Likert 척도로 응답했다. 우울/불안/스트레스 각 7문항 합산 후 표준 규정에 따라 2배하여 0–42 범위의 하위척도 점수를 산출했다. UCLA와 동일하게 모든 문항 응답을 요구했다.

### 2.4.2 인지 과제

**Stroop 과제**. 한국어 색 단어 4개(빨강/초록/파랑/노랑)와 중립 단어 3개(기차/학교/가방)를 4가지 잉크색으로 제시했다. 총 108 trial(일치 36, 불일치 36, 중립 36)이며, 연속 동일 자극이 나오지 않도록 무작위화했다. 각 trial은 고정주시(500 ms) → 공백(100 ms) → 자극 제시(반응 또는 3000 ms 타임아웃) 순서로 진행했다. 반응은 화면상의 색 버튼 클릭으로 기록했다.

**WCST**. 128‑card 버전 WCST를 사용했다. 기준 카드 4장(1 yellow circle, 2 black rectangles, 3 blue stars, 4 red triangles)을 화면 상단에 고정하고, 중앙에 제시된 자극 카드를 4개 버튼 중 하나로 분류하도록 했다. 규칙은 color → shape → number 순으로 진행되며, **10연속 정답** 시 규칙이 변경된다. 최대 6 category 완료 또는 128장 소진 시 종료된다. 반응 시간 제한은 없으며 각 trial의 RT를 기록했다.

## 2.5 실험 환경

실험은 Flutter 기반 웹 애플리케이션(데스크톱 전용)으로 구현되었고 Firebase를 통해 데이터가 저장되었다. RT는 `performance.now()` 기반으로 기록하여 밀리초 이하 정밀도를 확보했다.

## 2.6 절차

동의 후 인구통계(성별/나이/학력) → UCLA → DASS 순으로 설문을 완료했다. 이후 Stroop 과제, 2분 휴식, WCST 순서로 진행했다. 전체 소요 시간은 약 30–40분이었다.

## 2.7 데이터 전처리 및 QC

### 2.7.1 Trial‑level 정제

**Stroop**
- 타임아웃은 보존하되 `timeout=True`로 표시, 정확도 계산 시 오답 처리
- RT 유효 범위: 200–3000 ms (`is_rt_valid`)

**WCST**
- rule 유효성: colour/shape/number만 유지
- 선택 카드 유효성: 4개 기준 카드만 유지
- RT < 200 ms 제거
- RT 유효 범위: 200–10,000 ms (`is_rt_valid`)
- 타임아웃은 오답 처리

RT 기반 지표는 **타임아웃 제외 + 유효 RT만 사용**하며, 오류 trial도 포함한 all‑trials 기준을 기본으로 사용한다.

### 2.7.2 Participant‑level 제외 기준

다음 기준을 모두 만족하는 참가자만 포함했다.

- 설문 유효성: UCLA 총점, DASS 하위척도(우울/불안/스트레스), 성별 정보 존재
- 과제 완료: Stroop과 WCST 모두 완료 기록 존재
- Stroop QC: 108 trial 완료 + 전체 정확도 ≥ .70
- WCST QC: 유효 trial ≥ 60 + 단일 카드 선택 비율 ≤ .85

### 2.7.3 WCST phase 정의 (정석 3분할 + 2분할)

**Rule segment**는 `ruleAtThatTime` 변화로 구간을 나누며 **최대 6 category**만 사용한다.

**3‑phase (정석)**
- exploration: rule 전환 이후 **첫 정답 이전**
- confirmation: **첫 정답부터 3연속 정답 달성까지(포함)**
- exploitation: 3연속 정답 달성 이후
- 3연속 정답이 나오지 않으면, 첫 정답 이후 구간은 confirmation으로 유지
- 정답이 한 번도 없으면 전체가 exploration

**2‑phase (보조 분석)**
- pre‑exploitation = exploration + confirmation
- exploitation = 3연속 정답 달성 이후

모든 phase RT는 **all‑trials 기준**(오류 포함)으로 계산하며, 유효 RT 및 비‑timeout trial만 평균에 포함한다.

## 2.8 통계 분석

### 2.8.1 기본 회귀

모든 회귀는 **OLS(비‑robust)**로 수행했다. 공변량은 항상 DASS 하위척도(우울/불안/스트레스)와 나이, 성별을 통제했다. 주요 분석은 다음의 단계적 모델을 사용했다.

- Model 0: outcome ~ age + gender
- Model 1: Model 0 + DASS(3)
- Model 2: Model 1 + UCLA
- Model 3: Model 2 + UCLA × gender

### 2.8.2 WCST phase 타당도(보조)

WCST phase RT(3‑phase 및 2‑phase)에 대해 UCLA + DASS 통제 OLS 회귀를 수행하여 **타당도**를 평가했다. 3‑phase는 exploration/confirmation/exploitation 및 대비(confirmation‑exploitation)를, 2‑phase는 pre‑exploitation 및 대비(pre‑exploitation‑exploitation)를 보고한다.

### 2.8.3 신뢰도

WCST phase RT는 category 홀/짝 분할(split‑half)로 신뢰도를 계산했고, Spearman‑Brown 보정을 보고했다. 자세한 수치는 Supplementary에 제시했다.
