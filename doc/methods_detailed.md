# Methods Section (Detailed)

## 2.4 Measures

### 2.4.1 Self-Report Measures

**UCLA Loneliness Scale (Version 3; UCLA-LS)**. The 20-item Korean version of the UCLA Loneliness Scale (Russell, 1996) assessed subjective feelings of loneliness and social isolation over the past week. All items were presented simultaneously on a single scrollable page in fixed order (items 1–20), with each item displayed as an individual card containing the question text and response options. Participants responded using a 4-point Likert scale with the following anchors: 1 = "전혀 그렇지 않았다" (Not at all), 2 = "거의 그렇지 않았다" (Almost never), 3 = "가끔 그랬다" (Sometimes), 4 = "자주 그랬다" (Often).

Nine items measuring positive social connection (items 1, 5, 6, 9, 10, 15, 16, 19, 20) were reverse-coded prior to scoring. Total scores were computed by summing all 20 items after reverse-coding, yielding scores ranging from 20 to 80, with higher scores indicating greater loneliness. The survey could not be submitted until all 20 items received responses; incomplete submissions were prevented at the interface level. Survey completion time was recorded from first item display to submission.

**Depression Anxiety Stress Scales (DASS-21)**. The 21-item Korean version of the DASS-21 (Lovibond & Lovibond, 1995) assessed symptoms of depression, anxiety, and stress over the past week. Items were presented in fixed order (items 1–21) on a single scrollable page using the same card-based format as the UCLA-LS. Participants responded on a 4-point scale indicating symptom frequency: 0 = "전혀 해당되지 않음" (Did not apply to me at all), 1 = "약간 또는 가끔 해당됨" (Applied to me to some degree, or some of the time), 2 = "상당히 또는 자주 해당됨" (Applied to me to a considerable degree, or a good part of time), 3 = "매우 많이 또는 거의 대부분 해당됨" (Applied to me very much, or most of the time).

Three subscales comprised seven items each: Depression (items 3, 5, 10, 13, 16, 17, 21), Anxiety (items 2, 4, 7, 9, 15, 19, 20), and Stress (items 1, 6, 8, 11, 12, 14, 18). Following standard DASS-21 scoring procedures, raw subscale scores were multiplied by 2 to correspond with the original 42-item DASS scoring, yielding subscale scores ranging from 0 to 42. As with the UCLA-LS, all 21 items required responses before submission was permitted.

### 2.4.2 Cognitive Tasks

**Stroop Task**. A computerized color-word Stroop task assessed interference control, the ability to inhibit prepotent responses when faced with conflicting information. Participants identified the ink color of Korean color words by clicking one of four on-screen response buttons labeled with Korean color names: 빨강 (red), 초록 (green), 파랑 (blue), and 노랑 (yellow). Response button order was counterbalanced across participants: approximately half viewed buttons in the standard order and half in reversed order, determined by a hash function of participant ID ensuring reproducible assignment.

The stimulus set comprised four color words and three semantically neutral words (기차 [train], 학교 [school], 가방 [bag]), each presentable in four ink colors (red, green, blue, yellow). The main task consisted of 108 trials distributed equally across three conditions (36 trials each): (a) congruent trials, where color word meaning matched ink color (e.g., "빨강" in red ink); (b) incongruent trials, where color word meaning conflicted with ink color (e.g., "빨강" in blue ink); and (c) neutral trials, where semantically neutral words appeared in colored ink (e.g., "기차" in green ink). For congruent trials, each of the four word-color combinations was repeated 9 times. For incongruent trials, each color word appeared 3 times in each of the three non-matching colors (4 words × 3 colors × 3 repetitions = 36). For neutral trials, each neutral word appeared in each of the four colors with 3 repetitions (3 words × 4 colors × 3 repetitions = 36).

Trial presentation order was pseudo-randomized with the constraint that identical stimuli (same word and same ink color) could not appear on consecutive trials. Randomization was seeded using participant ID hash to ensure reproducibility. Each trial followed a fixed temporal sequence: (1) fixation cross displayed centrally for 500 ms (font size 48 pt), (2) blank inter-stimulus interval for 100 ms, (3) stimulus word displayed in colored ink at screen center (font size 48 pt) until response or timeout. The response deadline was 3000 ms from stimulus onset; trials without response within this window were coded as timeouts.

Ten practice trials preceded the main task to familiarize participants with the response interface. Practice trials consisted of 3 congruent and 7 incongruent trials (no neutral trials), with trial order randomized. Explicit feedback was provided after each practice trial: "맞았습니다!" (Correct!) for accurate responses and "틀렸습니다!" (Incorrect!) for errors or timeouts. Feedback remained on screen until participants clicked to advance. No feedback was provided during the main 108-trial block.

Response times were recorded using the JavaScript Performance API (`html.window.performance.now()`) at both stimulus onset and button press, with RT computed as the difference. This method provides sub-millisecond temporal resolution. Response detection occurred at pointer-down event rather than pointer-up to minimize response registration latency.

**Wisconsin Card Sorting Test (WCST)**. A computerized 128-card version of the Wisconsin Card Sorting Test (Heaton et al., 1993) assessed cognitive flexibility, set-shifting ability, and the capacity to adapt behavior based on feedback. Four reference cards were displayed continuously in a horizontal row at the top of the screen: (1) one yellow circle, (2) two black rectangles, (3) three blue stars, and (4) four red triangles. Reference cards measured 120 × 120 pixels with 10-pixel margins.

On each trial, a stimulus card (200 × 200 pixels) appeared centrally below the reference cards. Stimulus cards varied along three dimensions: color (yellow, black, blue, red), shape (circle, rectangle, star, triangle), and number (1, 2, 3, 4), yielding 64 unique combinations. The card deck consisted of two complete sets of these 64 cards, shuffled once using a seeded random generator (participant ID hash) and then duplicated, for 128 total trials. Participants sorted each stimulus card by clicking one of four response buttons labeled "1번 카드," "2번 카드," "3번 카드," and "4번 카드" (Card 1, Card 2, Card 3, Card 4), corresponding to the four reference cards.

Sorting rules followed a fixed sequence: color → shape → number, repeating as categories were completed. After 10 consecutive correct responses according to the current rule, the sorting criterion changed to the next rule in the sequence without explicit notification. The test terminated when either (a) six categories were successfully completed (requiring at least 60 correct responses) or (b) all 128 cards were exhausted.

Immediate feedback was displayed for exactly 1000 ms after each response: "정답" (Correct) for responses matching the current sorting criterion, and "오답" (Incorrect) for non-matching responses. No practice trials were provided; the task began directly with experimental trials following a single instruction screen. No response time limit was imposed; participants could take unlimited time to respond, though response times were recorded for each trial.

Perseverative errors were operationally defined according to three conjunctive criteria: (1) a previous sorting rule must exist and differ from the current rule (i.e., at least one category must have been completed); (2) the selected reference card must match the stimulus card on the previous rule's dimension; and (3) exactly one reference card must match the stimulus on the previous dimension (disambiguation criterion to distinguish true perseveration from coincidental matches). Perseverative responses were defined identically but could occur on either correct or incorrect trials; perseverative errors were perseverative responses that were also incorrect.

## 2.5 Apparatus

The experiment was implemented as a web application using the Flutter framework (version 3.x, Dart programming language) with Google Firebase backend for data storage and participant authentication. The application was accessible exclusively via desktop web browsers; mobile device access was blocked at the interface level to ensure standardized display dimensions, input modalities, and response timing.

Response times were recorded using the JavaScript Performance API (`performance.now()`) with sub-millisecond resolution, implemented via Dart's `dart:html` library. This API provides high-resolution timestamps relative to the navigation start time, avoiding the precision limitations of standard JavaScript `Date.now()`. Both stimulus onset times and response registration times were captured, with reaction times computed as the difference. All timing calculations were performed client-side to eliminate network latency effects on measurement.

Response counterbalancing was implemented using deterministic hash functions of participant ID strings, ensuring: (a) reproducible randomization—the same participant would receive identical counterbalancing assignment if the experiment were repeated; (b) approximately uniform distribution across counterbalancing conditions; and (c) independence from experimenter intervention or participant characteristics.

Stimulus presentation used Flutter's widget rendering system with SVG graphics for WCST card images and text rendering for Stroop stimuli. Display refresh synchronization was handled by the browser's rendering pipeline. Response collection used pointer-down events (Stroop, WCST) to minimize input latency.

## 2.6 Procedure

All procedures were conducted online via the custom web application. After providing informed consent electronically, participants completed demographic questions (age, gender, education level) before proceeding to self-report measures. Surveys were administered in fixed order: UCLA Loneliness Scale followed by DASS-21. Both surveys implemented completion validation: participants could not proceed until all items received responses, preventing missing data. Duplicate submission was prevented by checking for existing survey documents in the database; participants who had previously completed a survey were automatically advanced to the next section.

After survey completion, participants viewed an instruction page describing both cognitive tasks, including task-specific instructions, estimated duration, and general guidelines (quiet environment, sustained attention, 2-minute rest interval between tasks). Participants confirmed readiness via a dialog acknowledging that tasks could not be paused once started.

Cognitive task order was fixed: the Stroop task was administered first and the WCST was administered last. WCST was placed last because: (a) it is the longest task (10–15 minutes), and placing it last minimizes fatigue effects on preceding tasks; (b) it requires sustained cognitive flexibility, which may be disproportionately affected by prior fatigue; and (c) fixed positioning eliminates WCST-position confounds in analyses of set-shifting performance.

Mandatory 2-minute (120-second) rest intervals with countdown timers were inserted between the Stroop task and the WCST. During rest intervals, participants viewed a rest screen with a coffee cup icon, countdown timer, and suggestion to stretch or take deep breaths ("잠시 휴식을 취하세요. 스트레칭이나 심호흡을 하시면 좋습니다."). Rest periods terminated automatically without participant input, transitioning to the next task. No rest preceded the first task or followed the final task.

The entire session lasted approximately 30–40 minutes: surveys (10–15 minutes), Stroop (5–7 minutes), WCST (10–15 minutes), plus the rest interval (2 minutes total). Upon completion of all tasks, participants viewed a completion screen and received debriefing information. Course credit was granted upon verified completion.

## 2.7 Data Preprocessing and Quality Control

**Trial-level filtering and flags**. Trial-level datasets retain all recorded trials and append quality flags rather than dropping rows. For Stroop, timeouts are preserved and marked, and RT validity is defined as 200–3000 ms. For WCST, trials are cleaned for required fields and valid condition/card values, with RT < 200 ms removed; an RT-valid flag marks 200–10,000 ms for RT-based analyses. In analysis loaders, RT-focused indices exclude timeouts (Stroop) and require valid RTs; accuracy/error indices treat timeouts as incorrect while retaining those trials.

**Participant-level exclusions**. Task-specific inclusion criteria were applied to ensure data quality:

*Stroop task*: Participants were retained if they completed all 108 main trials and overall accuracy was ≥ 70% (timeouts counted as incorrect).

*WCST*: Participants were retained if they had at least 60 valid trials after cleaning and did not select any single reference card on more than 85% of trials.

**Derived performance measures**. For Stroop, RT indices were computed on non-timeout trials with valid RTs (200–3000 ms); correct-only variants were computed separately. Accuracy metrics used all trials with timeouts coded as incorrect.

For WCST, primary dependent variables included: (a) number of categories completed (0–6); (b) total errors; (c) perseverative errors (count and percentage of total errors); (d) non-perseverative errors; (e) conceptual level responses (runs of 3+ consecutive correct responses); and (f) trials to first category completion. Failure to maintain set was computed as the number of times an error occurred after achieving 5+ consecutive correct responses. RT-based indices applied the 10,000 ms upper bound via the RT-valid flag.

## 2.8 Statistical Analyses

Hierarchical multiple regression models were estimated using OLS (non-robust) standard errors. For each outcome, Step 1 included age and gender, Step 2 added DASS-21 subscales (depression, anxiety, stress), and Step 3 added UCLA loneliness. The incremental contribution of loneliness was quantified as ΔR^2 from the Step 2 → Step 3 comparison.

Primary endpoints were pre-specified as (a) the Stroop interference RT slope (quartile-based interference trend) and (b) WCST post-shift error RT mean. Multiple-testing control was implemented using Benjamini-Hochberg FDR across seven focal outcomes (two primary endpoints plus five conventional/secondary indices). We report unadjusted p-values alongside FDR q-values. Multicollinearity was evaluated using VIF diagnostics.

Sensitivity analyses included: (a) a single-covariate DASS total model, (b) reverse entry order (UCLA before DASS) to assess shared variance, (c) winsorizing RT-based outcomes at 2.5% tails, (d) excluding participants with extreme RT variability (top 2.5% of within-person RT SD), (e) HC3-robust OLS standard errors for primary endpoints, (f) trial-level OLS with participant fixed effects testing segment × UCLA interactions, and (g) segment-count sensitivity for the interference slope (3, 5, 6 segments). Reliability of slope indices was assessed via split-half correlations and bootstrap CI-width distributions (reported in Supplementary).

### 2.8.3 Exploratory Computational Indices (optional)

In addition to the pre-specified temporal dynamics indices, we explored computational model-derived parameters as potential markers of loneliness-related cognitive differences. These included Hidden Markov Model state transition probabilities from WCST performance and distributional parameters from Stroop RTs. Given their exploratory nature, these analyses are reported separately and interpreted cautiously.

---

## Summary Tables

### Table 1. Task Parameters Overview

| Parameter | Stroop | WCST |
|-----------|--------|------|
| Total trials | 108 (+ 10 practice) | 128 max |
| Response modality | On-screen buttons | On-screen buttons |
| Response deadline | 3000 ms | None |
| Counterbalancing | Button order (2 conditions) | None |
| Rest breaks | None | None |
| Feedback | Practice only | Every trial (1 s) |
| Timing precision | performance.now() | performance.now() |

### Table 2. Trial Timing Sequences

**Stroop Task**
| Phase | Duration |
|-------|----------|
| Fixation cross | 500 ms |
| Inter-stimulus interval | 100 ms |
| Stimulus display | Until response (max 3000 ms) |

**WCST**
| Phase | Duration |
|-------|----------|
| Stimulus display | Until response (no limit) |
| Feedback | 1000 ms |

### Table 3. Self-Report Measures

| Measure | Items | Scale | Scoring |
|---------|-------|-------|---------|
| UCLA-LS | 20 | 1-4 Likert | Sum (20-80), 9 items reverse-coded |
| DASS-21 | 21 | 0-3 Likert | 3 subscales × 2 (0-42 each) |
