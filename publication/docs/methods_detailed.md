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

**Psychological Refractory Period (PRP) Task**. A dual-task paradigm assessed central bottleneck effects and attentional control capacity. The task required rapid sequential responses to two stimuli (Task 1 and Task 2), with the temporal interval between stimulus onsets (stimulus onset asynchrony; SOA) systematically varied.

Task 1 (T1) required parity judgment of single digits. Stimuli consisted of digits 1–9, displayed centrally at 48-pt bold font. Odd digits (1, 3, 5, 7, 9) required one response; even digits (2, 4, 6, 8) required another. Task 2 (T2) required color identification of a square. Stimuli consisted of an 80 × 80 pixel colored square (red or blue) displayed below the digit. Red squares required one response; blue squares required another.

Responses were collected via keyboard. T1 responses used the 'A' and 'D' keys (left and right positions on QWERTY keyboard). T2 responses used the left and right arrow keys. Key-response mappings were counterbalanced across participants in a 2 × 2 design: (a) T1 mapping reversal (odd=A/even=D vs. odd=D/even=A) and (b) T2 mapping reversal (red=left/blue=right vs. red=right/blue=left), yielding four counterbalancing conditions. Assignment to conditions was determined by hash functions of participant ID, with approximately equal distribution across conditions.

SOA varied across five levels: 50, 150, 300, 600, and 1200 ms. These values span the range from strong dual-task interference (short SOA) to minimal interference (long SOA), allowing characterization of the PRP curve. The main task comprised 120 trials following a fully crossed design: 5 SOA levels × 2 T1 parities × 2 T2 colors × 6 repetitions = 120 trials. This design ensured 24 trials per SOA level with balanced representation of all stimulus combinations. Trial order was fully randomized.

Each trial followed this temporal sequence: (1) fixation cross (60 pt, "+") displayed centrally for 500 ms; (2) T1 stimulus (digit) appeared; (3) after the designated SOA interval, T2 stimulus (colored square) appeared below T1; (4) both stimuli remained visible until responses were collected or timeouts occurred; (5) 500 ms inter-trial interval before the next trial's fixation. Response deadlines were 3000 ms for each task independently, calculated from respective stimulus onsets. If T1 response was not registered within 3000 ms of T1 onset, it was coded as timeout; similarly for T2. The trial terminated only after both responses were registered or both deadlines elapsed.

Mandatory rest breaks of 10 seconds duration were inserted after trials 30, 60, and 90 (at 25%, 50%, and 75% completion). During rest breaks, a countdown timer was displayed with the message "휴식 시간" (Rest time), and the next trial began automatically when the timer reached zero. No participant action was required to resume.

Ten practice trials (2 per SOA level) preceded the main task. Practice trials used the same stimulus and response format as main trials, with explicit feedback provided for both tasks after each trial: "T1 정답/오답/시간초과" and "T2 정답/오답/시간초과" (Correct/Incorrect/Timeout). Feedback was displayed for 1000 ms before automatic advancement to the next trial.

Response times were recorded using high-precision JavaScript timing (`performance.now()`) at stimulus onset and keypress for both tasks. Measured SOA was computed from actual T2 onset minus T1 onset timestamps and compared against nominal SOA to verify timing accuracy. Response order (whether T1 response preceded T2 response or vice versa) was recorded for each trial based on response timestamps.

## 2.5 Apparatus

The experiment was implemented as a web application using the Flutter framework (version 3.x, Dart programming language) with Google Firebase backend for data storage and participant authentication. The application was accessible exclusively via desktop web browsers; mobile device access was blocked at the interface level to ensure standardized display dimensions, input modalities, and response timing.

Response times were recorded using the JavaScript Performance API (`performance.now()`) with sub-millisecond resolution, implemented via Dart's `dart:html` library. This API provides high-resolution timestamps relative to the navigation start time, avoiding the precision limitations of standard JavaScript `Date.now()`. Both stimulus onset times and response registration times were captured, with reaction times computed as the difference. All timing calculations were performed client-side to eliminate network latency effects on measurement.

Response counterbalancing was implemented using deterministic hash functions of participant ID strings, ensuring: (a) reproducible randomization—the same participant would receive identical counterbalancing assignment if the experiment were repeated; (b) approximately uniform distribution across counterbalancing conditions; and (c) independence from experimenter intervention or participant characteristics.

Stimulus presentation used Flutter's widget rendering system with SVG graphics for WCST card images and text rendering for Stroop and PRP stimuli. Display refresh synchronization was handled by the browser's rendering pipeline. Response collection used pointer-down events (Stroop, WCST) or keyboard events (PRP) to minimize input latency.

## 2.6 Procedure

All procedures were conducted online via the custom web application. After providing informed consent electronically, participants completed demographic questions (age, gender, education level) before proceeding to self-report measures. Surveys were administered in fixed order: UCLA Loneliness Scale followed by DASS-21. Both surveys implemented completion validation: participants could not proceed until all items received responses, preventing missing data. Duplicate submission was prevented by checking for existing survey documents in the database; participants who had previously completed a survey were automatically advanced to the next section.

After survey completion, participants viewed an instruction page describing all three cognitive tasks, including task-specific instructions, estimated duration, and general guidelines (quiet environment, sustained attention, 2-minute rest intervals between tasks). Participants confirmed readiness via a dialog acknowledging that tasks could not be paused once started.

Cognitive task order was partially counterbalanced to balance practice effects against task demands. The Stroop and PRP tasks were randomly ordered across positions 1 and 2 (approximately 50% of participants completed Stroop first, 50% completed PRP first). The WCST was fixed in position 3 (final) because: (a) it is the longest task (10–15 minutes), and placing it last minimizes fatigue effects on preceding tasks; (b) it requires sustained cognitive flexibility, which may be disproportionately affected by prior fatigue; and (c) fixed positioning eliminates WCST-position confounds in analyses of set-shifting performance.

Mandatory 2-minute (120-second) rest intervals with countdown timers were inserted between consecutive tasks. During rest intervals, participants viewed a rest screen with a coffee cup icon, countdown timer, and suggestion to stretch or take deep breaths ("잠시 휴식을 취하세요. 스트레칭이나 심호흡을 하시면 좋습니다."). Rest periods terminated automatically without participant input, transitioning to the next task. No rest preceded the first task or followed the final task.

The entire session lasted approximately 40–60 minutes: surveys (10–15 minutes), Stroop (5–7 minutes), PRP (8–12 minutes), WCST (10–15 minutes), plus rest intervals (4 minutes total). Upon completion of all tasks, participants viewed a completion screen and received debriefing information. Course credit was granted upon verified completion.

## 2.7 Data Preprocessing and Quality Control

**Trial-level exclusions**. Across all tasks, trials were excluded for: (a) timeout—no response within the task-specific deadline; and (b) missing data—technical failures preventing response recording. For Stroop and PRP tasks, anticipatory responses (RT < 200 ms) were additionally excluded, as such rapid responses likely reflect premature key presses rather than genuine stimulus processing. For WCST, the anticipatory threshold was set lower (RT < 100 ms) given the self-paced response format. For PRP specifically, measured SOA was computed from actual T1 and T2 onset timestamps and compared against nominal SOA; trials with SOA measurement error exceeding ±50 ms were flagged for inspection. Upper RT limits were applied at the task level (3000 ms for Stroop and PRP, no limit for WCST) as responses beyond these limits were coded as timeouts.

**Participant-level exclusions**. Task-specific inclusion criteria were applied to ensure data quality:

*Stroop task*: Participants were retained if they achieved: (a) overall accuracy ≥ 70% across all 108 main trials, indicating task comprehension; and (b) at least 50 valid (non-excluded) trials, ensuring sufficient data for reliable RT estimation. Accuracy was computed separately for each condition to identify potential color perception difficulties or response key confusion.

*PRP task*: Participants were retained if they completed all 120 main trials and their joint accuracy (T1 and T2 both correct) was >= 50% across PRP trials (timeouts counted as incorrect). Participants showing systematic response order reversals (T2 before T1 on >30% of trials) were flagged for inspection, as this may indicate task misunderstanding or strategic reordering.

*WCST*: Participants were retained if they achieved: (a) at least 80 valid trials (≥62.5% of 128 total), ensuring sufficient data for performance indices; (b) median RT ≥ 300 ms, as faster median RTs suggest random clicking rather than deliberate card matching; and (c) proportion of trials selecting any single reference card ≤ 85%, to exclude participants using fixed response strategies rather than adapting to feedback.

**Derived performance measures**. For Stroop, the primary dependent variable was the Stroop interference effect, computed as mean RT(incongruent correct trials) − mean RT(congruent correct trials). Secondary measures included condition-specific accuracy rates and mean RTs.

For PRP, the primary dependent variable was the PRP effect magnitude, operationalized as the difference in mean T2 RT between short SOA (50 ms) and long SOA (1200 ms) conditions: PRP effect = mean RT2(SOA=50) − mean RT2(SOA=1200). This index captures the degree of response slowing attributable to central processing bottleneck at short SOAs. The slope of T2 RT across log-transformed SOA was computed as an alternative continuous measure.

For WCST, primary dependent variables included: (a) number of categories completed (0–6); (b) total errors; (c) perseverative errors (count and percentage of total errors); (d) non-perseverative errors; (e) conceptual level responses (runs of 3+ consecutive correct responses); and (f) trials to first category completion. Failure to maintain set was computed as the number of times an error occurred after achieving 5+ consecutive correct responses.

## 2.8 Statistical Analyses

### 2.8.3 Exploratory Computational Indices (optional)

In addition to the pre-specified temporal dynamics indices, we explored computational model-derived parameters as potential markers of loneliness-related cognitive differences. These included: (a) Ex-Gaussian distribution parameters (mu, sigma, tau) fitted to PRP T2 RT distributions, (b) Hidden Markov Model state transition probabilities from WCST performance, and (c) central bottleneck model fit indices from PRP. Given their exploratory nature, these analyses are reported separately and interpreted cautiously.

---

## Summary Tables

### Table 1. Task Parameters Overview

| Parameter | Stroop | WCST | PRP |
|-----------|--------|------|-----|
| Total trials | 108 (+ 10 practice) | 128 max | 120 (+ 10 practice) |
| Response modality | On-screen buttons | On-screen buttons | Keyboard |
| Response deadline | 3000 ms | None | 3000 ms per task |
| Counterbalancing | Button order (2 conditions) | None | Key mapping (4 conditions) |
| Rest breaks | None | None | Every 30 trials (10 s) |
| Feedback | Practice only | Every trial (1 s) | Practice only |
| Timing precision | performance.now() | performance.now() | performance.now() |

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

**PRP Task**
| Phase | Duration |
|-------|----------|
| Fixation cross | 500 ms |
| T1 stimulus | Until response (max 3000 ms) |
| SOA interval | 50, 150, 300, 600, or 1200 ms |
| T2 stimulus | Until response (max 3000 ms from T2 onset) |
| Inter-trial interval | 500 ms |

### Table 3. Self-Report Measures

| Measure | Items | Scale | Scoring |
|---------|-------|-------|---------|
| UCLA-LS | 20 | 1-4 Likert | Sum (20-80), 9 items reverse-coded |
| DASS-21 | 21 | 0-3 Likert | 3 subscales × 2 (0-42 each) |
