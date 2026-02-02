# Data Dictionary

This document describes the files under `data/complete_overall/` and the columns
used in the analysis pipeline. The `complete_overall` folder contains QC-passed
data used for analysis. Raw data live under `data/raw/` and should be treated
as restricted.

## Conventions

- Missing values are stored as empty cells.
- Time units are milliseconds (ms) unless otherwise noted.
- `participantId` (camelCase) is the raw identifier; `participant_id` (snake_case)
  is the standardized identifier created in preprocessing.
- Timing fields (e.g., `createdAt`, `timestamp`) can be identifying and should
  be removed or coarsened for public release.

---

## data/complete_overall/1_participants_info.csv (RESTRICTED)

Participant metadata. This file contains direct identifiers and should not be
publicly shared without de-identification.

| Column | Type | Description |
|---|---|---|
| participantId | string | Participant ID (internal). |
| studentId | string | Institutional student ID (direct identifier). |
| gender | string | Self-reported gender label (raw string). |
| age | integer | Age in years at participation. |
| birthDate | string | Birth date in YYYYMMDD format (direct identifier). |
| education | string | Self-reported education level. |
| courseName | string | Course name (direct identifier). |
| professorName | string | Instructor name (direct identifier). |
| classSection | string | Course section (direct identifier). |
| createdAt | datetime | Record creation time (UTC timestamp). |

Public release note: remove or mask all direct identifiers (`studentId`,
`birthDate`, `courseName`, `professorName`, `classSection`, `createdAt`).

---

## data/complete_overall/2_surveys_results.csv

Survey item responses and scale scores for DASS-21 and UCLA Loneliness Scale.
Each participant has two rows: one for `surveyName = dass` and one for
`surveyName = ucla`.

| Column | Type | Description |
|---|---|---|
| participantId | string | Participant ID (internal). |
| surveyName | string | `dass` or `ucla`. |
| duration_seconds | float | Completion time in seconds. |
| q1..q21 | integer | Item responses. For DASS: 0-3. For UCLA: 1-4 (q21 empty). |
| score_A | integer | DASS Anxiety subscale (sum of 7 items * 2). |
| score_S | integer | DASS Stress subscale (sum of 7 items * 2). |
| score_D | integer | DASS Depression subscale (sum of 7 items * 2). |
| score | integer | UCLA total score (20-80) after reverse scoring. |

Notes:
- For `surveyName = dass`, `score` is empty.
- For `surveyName = ucla`, `score_A`, `score_S`, and `score_D` are empty.

---

## data/complete_overall/3_cognitive_tests_summary.csv

Per-task summary statistics. Each participant has two rows:
`testName = stroop` and `testName = wcst`.

| Column | Type | Description |
|---|---|---|
| participantId | string | Participant ID (internal). |
| testName | string | `stroop` or `wcst`. |
| duration_seconds | float | Task completion time in seconds. |
| timestamp | datetime | Time the summary was saved. |
| total | integer | Stroop: total number of trials. |
| accuracy | float | Stroop: percent correct (0-100). |
| mrt_total | float | Stroop: mean RT (ms) across correct trials. |
| mrt_cong | float | Stroop: mean RT (ms) for congruent correct trials. |
| mrt_incong | float | Stroop: mean RT (ms) for incongruent correct trials. |
| stroop_effect | float | Stroop: `mrt_incong - mrt_cong` (ms). |
| totalTrialCount | integer | WCST: total trials. |
| totalCorrectCount | integer | WCST: total correct responses. |
| totalErrorCount | integer | WCST: total errors (`perseverative + non-perseverative`). |
| perseverativeErrorCount | integer | WCST: perseverative errors (incorrect PRs). |
| nonPerseverativeErrorCount | integer | WCST: non-perseverative errors. |
| perseverativeResponses | integer | WCST: perseverative responses (PRs). |
| perseverativeResponsesPercent | float | WCST: PRs as percent of total trials. |
| conceptualLevelResponses | integer | WCST: conceptual level responses (CLR). |
| conceptualLevelResponsesPercent | float | WCST: CLR percent of total trials. |
| trialsToFirstConceptualResp | integer | WCST: 1-based trial index of first CLR (0 if none). |
| trialsToFirstConceptualResp0 | integer | WCST: 0-based trial index of first CLR (-1 if none). |
| hasFirstCLR | boolean | WCST: whether a CLR occurred. |
| trialsToCompleteFirstCategory | integer | WCST: trials needed to complete first category. |
| completedCategories | integer | WCST: number of completed categories (max 6). |
| failureToMaintainSet | integer | WCST: FMS count (errors after 5+ correct within category). |
| learningEfficiencyDeltaTrials | float | WCST: mean trials (first 3 categories) minus mean trials (last 3). |
| learningToLearnHeatonClrDelta | float | WCST: mean CLR% (last 3 categories) minus mean CLR% (first 3). |
| categoryClrPercents | string | WCST: list of CLR% per category (length up to 6). |
| learningToLearn | string | WCST: legacy field (empty in current data). |

Notes:
- Stroop fields are empty for `testName = wcst`, and WCST fields are empty for
  `testName = stroop`.

---

## data/complete_overall/4a_stroop_trials.csv

Trial-level Stroop data. One row per trial.

| Column | Type | Description |
|---|---|---|
| participant_id | string | Standardized participant ID. |
| task | string | `stroop`. |
| trial_index | integer | 0-based trial index (raw log). |
| trial | integer | Legacy trial index (same as `trial_index`). |
| trial_order | integer | Standardized trial order (numeric). |
| block_index | integer | Block index (null for Stroop). |
| cond | string | Standardized condition: `congruent`, `incongruent`, `neutral`. |
| type | string | Raw condition string (same meaning as `cond`). |
| text | string | Word presented on screen. |
| letterColor | string | Ink color for the word. |
| userColor | string | Selected response color (`noResp` if timeout). |
| correct | boolean | Correct response (timeouts treated as incorrect). |
| timeout | boolean | Timeout flag (standardized). |
| is_timeout | boolean | Legacy timeout flag. |
| rt_ms | float | Reaction time in ms (standardized). |
| rt | float | Legacy RT value (ms). |
| stim_onset_ms | float | Stimulus onset time (performance.now, ms). |
| resp_time_ms | float | Response time (performance.now, ms). |
| is_rt_valid | boolean | RT within [200, 3000] ms. |
| extra | string | JSON-like task metadata (text, letterColor, userColor). |

---

## data/complete_overall/4b_wcst_trials.csv

Trial-level WCST data. One row per trial.

| Column | Type | Description |
|---|---|---|
| participant_id | string | Standardized participant ID. |
| task | string | `wcst`. |
| trial_index | integer | 0-based trial index (standardized). |
| trialIndex | integer | Legacy trial index (same meaning as `trial_index`). |
| trial_order | integer | Standardized trial order (numeric). |
| trialIndexInBlock | integer | 0-based trial index within block. |
| block_index | integer | Standardized block index (0 or 1). |
| blockIndex | integer | Legacy block index. |
| cond | string | Standardized rule: `colour`, `shape`, `number`. |
| rule | string | Standardized rule (same as `cond`). |
| ruleAtThatTime | string | Legacy rule label at the time of the trial. |
| chosen_card | string | Standardized chosen card label. |
| chosenCard | string | Legacy chosen card label. |
| cardNumber | integer | Stimulus card number (1-4). |
| cardColor | string | Stimulus card color. |
| cardShape | string | Stimulus card shape. |
| targetCardPath | string | Stimulus SVG path identifier. |
| correct | boolean | Correct response (timeouts treated as incorrect). |
| isPR | boolean | Perseverative response flag. |
| isPE | boolean | Perseverative error flag (incorrect PR). |
| isNPE | boolean | Non-perseverative error flag. |
| rt_ms | float | Reaction time in ms (standardized). |
| reactionTimeMs | float | Legacy RT value (ms). |
| stim_onset_ms | float | Stimulus onset time (performance.now, ms). |
| resp_time_ms | float | Response time (performance.now, ms). |
| timeout | boolean | Timeout flag (always false in WCST). |
| stageName | string | UI stage label at time of trial. |
| timestamp | datetime | Trial log timestamp (ISO 8601). |
| is_rt_valid | boolean | RT within [200, 10000] ms. |
| extra | string | JSON-like task metadata (rule, card features, PR/PE/NPE). |

---

## data/complete_overall/5_overall_features.csv

Participant-level derived features used in primary analyses.

| Column | Type | Description |
|---|---|---|
| participant_id | string | Standardized participant ID. |
| stroop_interference | float | Mean RT difference (incongruent - congruent), correct-only, valid RT. |
| stroop_interference_slope | float | Slope of interference across 4 within-task segments. |
| wcst_perseverative_error_rate | float | Percent of trials labeled `isPE`. |
| wcst_exploration_rt | float | Mean RT (ms), correct-only, exploration phase. |
| wcst_confirmation_rt | float | Mean RT (ms), correct-only, confirmation phase. |
| wcst_exploitation_rt | float | Mean RT (ms), correct-only, exploitation phase. |
| wcst_pre_exploitation_rt | float | Mean RT (ms), correct-only, exploration + confirmation. |
| wcst_confirmation_minus_exploitation_rt | float | `confirmation - exploitation` (ms), correct-only. |
| wcst_pre_exploitation_minus_exploitation_rt | float | `pre_exploitation - exploitation` (ms), correct-only. |
| wcst_exploration_rt_all | float | Mean RT (ms), all trials (errors included), exploration. |
| wcst_confirmation_rt_all | float | Mean RT (ms), all trials, confirmation. |
| wcst_exploitation_rt_all | float | Mean RT (ms), all trials, exploitation. |
| wcst_pre_exploitation_rt_all | float | Mean RT (ms), all trials, exploration + confirmation. |
| wcst_confirmation_minus_exploitation_rt_all | float | `confirmation - exploitation` (ms), all trials. |
| wcst_pre_exploitation_minus_exploitation_rt_all | float | `pre_exploitation - exploitation` (ms), all trials. |

Phase definitions follow the three-phase scheme in `doc/methods_detailed.md`.

---

## data/complete_overall/filtered_participant_ids.csv

List of participants who passed QC for the overall dataset.

| Column | Type | Description |
|---|---|---|
| participantId | string | Participant ID included in QC-passed analyses. |

---

## data/complete_overall/participants_completed_6_categories.csv

WCST subset with all 6 categories completed.

| Column | Type | Description |
|---|---|---|
| participant_id | string | Participant ID. |
| n_categories | integer | Number of completed categories (should be 6). |
