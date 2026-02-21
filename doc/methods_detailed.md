# Methods (Detailed)

## 2.4 Measures

### 2.4.1 Self-report scales

**UCLA Loneliness Scale (Version 3; UCLA-LS).** We administered the 20-item Korean version of the UCLA Loneliness Scale. Responses were collected on a 1 (never) to 4 (often) Likert scale. Nine positively worded items (items 1, 5, 6, 9, 10, 15, 16, 19, 20) were reverse-scored. The total score (range 20-80) was computed by summing all items. The interface required all items, so no item-level missing data were allowed.

**Depression Anxiety Stress Scales (DASS-21).** We used the Korean DASS-21. Items were rated on a 0-3 Likert scale. For each subscale (Depression, Anxiety, Stress), the seven items were summed and multiplied by 2 to yield the standard 0-42 subscale scores. All items were required, so no item-level missing data were allowed.

### 2.4.2 Cognitive tasks

**Stroop task.** We used four Korean color words (red, green, blue, yellow) and three neutral words (train, school, bag), each displayed in one of four ink colors. The task consisted of 108 trials (36 congruent, 36 incongruent, 36 neutral) with randomized order and no immediate repeats. Each trial followed a fixed sequence: fixation (500 ms) -> blank (100 ms) -> stimulus (until response or 3000 ms timeout). Responses were made via on-screen color buttons. Reaction time (RT) and accuracy were recorded.

**Wisconsin Card Sorting Task (WCST).** We used the standard 128-card WCST. Four key cards (1 yellow circle, 2 black rectangles, 3 blue stars, 4 red triangles) were displayed at the top of the screen, and participants selected the matching key for each stimulus card. Sorting rules progressed in a fixed order (color -> shape -> number). The rule changed after 10 consecutive correct responses. The task ended after completing 6 categories or after 128 trials. There was no time limit per trial; RT was recorded for each trial.

## 2.5 Apparatus and environment

The experiment was implemented as a Flutter web application (desktop only) with Firebase-based data storage. RTs were recorded using `performance.now()` for sub-millisecond timing precision.

## 2.6 Procedure

After consent, participants completed demographics (gender, age, education), followed by the UCLA-LS and DASS-21. They then completed the Stroop task, took a 2-minute rest, and completed the WCST. Total duration was approximately 30-40 minutes.

## 2.7 Data preprocessing and QC

### 2.7.1 Trial-level cleaning

**Stroop**
- Timeouts were retained but coded as incorrect for accuracy calculations.
- Valid RT window: 200-3000 ms (`is_rt_valid`).

**WCST**
- Valid rules: colour/shape/number only.
- Valid chosen cards: the four target key cards only.
- RT < 200 ms removed.
- Valid RT window: 200-10,000 ms (`is_rt_valid`).
- Timeouts were treated as incorrect.

For all RT-based metrics, we used **non-timeout trials with valid RTs** and included error trials (all-trials convention) unless noted otherwise.

### 2.7.2 Participant-level inclusion criteria

Participants were included only if they met all of the following:

- Valid surveys: UCLA total, all three DASS subscales, and gender present.
- Task completion: both Stroop and WCST completed.
- Stroop QC: 108 trials completed and overall accuracy >= .70.
- WCST QC: >= 60 valid trials and single-card choice proportion <= .85.

### 2.7.3 WCST phase definitions (primary 3-phase + auxiliary 2-phase)

Rule segments were defined by changes in `ruleAtThatTime` and limited to the first 6 categories.

**Three-phase (primary)**
- exploration: after a rule switch until the first correct response
- confirmation: from the first correct response through achieving 3 consecutive correct responses (inclusive)
- exploitation: after achieving 3 consecutive correct responses
- If 3 consecutive correct responses never occur, all post-first-correct trials remain confirmation.
- If there are no correct responses, the entire segment is exploration.

**Two-phase (auxiliary; rule search/application)**
- rule search (pre-exploitation) = exploration + confirmation
- rule application (exploitation) = after 3 consecutive correct responses

All phase RTs were computed on **all trials (errors included)** using valid RTs and **excluding timeouts**.

## 2.8 Statistical analysis

Descriptive sex comparisons in Table 1 were tested using Welch's independent-samples t-tests (`equal_var=False`).

### 2.8.1 Primary regressions

All regressions used **OLS (non-robust)** standard errors. Continuous predictors (UCLA, DASS-Dep/Anx/Str, age) were z-standardized before model fitting, so reported UCLA coefficients correspond to a **1-SD increase** in loneliness. We used the following hierarchical model sequence for each outcome:

- Model 0: outcome ~ age + gender
- Model 1: Model 0 + DASS(3)
- Model 2: Model 1 + UCLA
- Model 3: Model 2 + UCLA x gender

### 2.8.2 WCST phase validity (supplementary)

We evaluated WCST phase RT validity using OLS regressions with UCLA as the focal predictor and DASS/age/gender covariates. The 3-phase analysis reported exploration/confirmation/exploitation and the confirmation-exploitation contrast; the 2-phase analysis reported pre-exploitation, exploitation, and their contrast.

### 2.8.3 Reliability

WCST phase RT reliability was estimated using odd/even category split-half correlations and Spearman-Brown correction. Alternative confirmation thresholds (2 and 4 consecutive correct) were reported in the Supplementary Materials.

### 2.8.4 Stroop trial-level LMM (supplementary)

Two mixed-effects models were reported:

**(1) Full-trial LMM**
- Outcome: RT (ms)
- Fixed effects: segment (participant-level quartiles) x UCLA(z), plus condition, DASS(3), age, gender
- Random effects: participant-level random intercept + segment slope

**(2) Interference-slope LMM**
- Outcome: log(RT) on congruent/incongruent trials only
- Fixed effects: trial_scaled (within-participant 0-1) x cond_code (-0.5/+0.5) x UCLA(z), plus DASS(3), age, gender
- Random effects: preferred structure 1 + trial_scaled (convergence prioritized)

All LMMs were estimated with `statsmodels` MixedLM using maximum likelihood. For random-effect variance terms, Wald-type 95% CIs were interpreted with a non-negativity boundary, so any slightly negative lower bound from normal approximation was reported as 0. Detailed results are provided in the Supplementary Materials.
