# Supplementary Materials

## S1. Technical Specifications

### S1.1 Development Environment

| Component | Specification |
|-----------|--------------|
| **Framework** | Flutter 3.x (Dart programming language) |
| **Backend** | Google Firebase (Firestore database, Authentication) |
| **Deployment** | Firebase Hosting (web application) |
| **Browser Compatibility** | Chrome, Firefox, Edge, Safari (desktop versions only) |

Browser metadata (user-agent strings) were not stored in the analysis dataset, so stratified comparisons by browser type were not available. Desktop-only access was enforced at the interface level.

### S1.2 Timing Precision

Response times were recorded using `html.window.performance.now()`, which provides high-resolution timestamps with sub-millisecond precision. This method returns a DOMHighResTimeStamp representing the time elapsed since the page navigation started, measured in milliseconds with microsecond precision where available. Studies have demonstrated that this timing method achieves precision comparable to laboratory software when running on modern browsers (Bridges et al., 2020; Anwyl-Irvine et al., 2021).

**Timing Variables Recorded:**
| Variable | Description |
|----------|-------------|
| `stim_onset_ms` | Absolute timestamp of stimulus appearance |
| `resp_time_ms` | Absolute timestamp of response registration |
| `rt_ms` | Computed reaction time (resp_time_ms − stim_onset_ms) |

### S1.3 Randomization Implementation

Pseudo-random number generation used Dart's `Random` class seeded with participant ID hash codes:

```dart
Random randomForShuffle = Random(participantId.hashCode);
trialList.shuffle(randomForShuffle);
```

This implementation ensures:
- (a) Reproducible randomization per participant
- (b) Different sequences across participants
- (c) Deterministic counterbalancing assignment

---

## S2. Detailed Stimulus Parameters

### S2.1 Stroop Task Stimuli

**Color Words (Korean):**

| Korean | Romanization | English | RGB Value |
|--------|-------------|---------|-----------|
| 빨강 | ppalgang | Red | Colors.red |
| 초록 | chorok | Green | Colors.green |
| 파랑 | parang | Blue | Colors.blue |
| 노랑 | norang | Yellow | Colors.yellow |

**Neutral Words (Korean):**

| Korean | Romanization | English | Semantic Category |
|--------|-------------|---------|-------------------|
| 기차 | gicha | Train | Transportation |
| 학교 | hakgyo | School | Place |
| 가방 | gabang | Bag | Object |

**Display Parameters:**

| Parameter | Value |
|-----------|-------|
| Fixation cross | "+" symbol, 48 pt font |
| Fixation duration | 500 ms |
| Inter-stimulus interval | 100 ms (blank screen) |
| Stimulus font size | 48 pt |
| Response buttons | 4 buttons, Korean color labels |
| Button counterbalancing | Standard or reversed (50/50 split) |

**Trial Distribution:**

| Condition | Combinations | Repetitions | Total Trials |
|-----------|-------------|-------------|--------------|
| Congruent | 4 (word = color) | 9 | 36 |
| Incongruent | 12 (4 words × 3 mismatched colors) | 3 | 36 |
| Neutral | 12 (3 words × 4 colors) | 3 | 36 |
| **Total** | | | **108** |

**Practice Trial Distribution:**

| Condition | Trials |
|-----------|--------|
| Congruent | 3 |
| Incongruent | 7 |
| **Total** | **10** |

### S2.2 WCST Stimuli

**Card Attribute Dimensions:**

| Dimension | Levels | Values |
|-----------|--------|--------|
| Color | 4 | Yellow, Black, Blue, Red |
| Shape | 4 | Circle, Rectangle, Star, Triangle |
| Number | 4 | 1, 2, 3, 4 |

**Total unique cards:** 4 × 4 × 4 = 64 combinations

**Reference Cards (Fixed, Left to Right):**

| Position | Description | Image File |
|----------|-------------|------------|
| 1 | One yellow circle | 1yellowcircle.svg |
| 2 | Two black rectangles | 2blackrectangle.svg |
| 3 | Three blue stars | 3bluestar.svg |
| 4 | Four red triangles | 4redtriangle.svg |

**Visual Specifications:**

| Element | Dimension |
|---------|-----------|
| Reference cards | 120 × 120 pixels |
| Reference card margin | 10 pixels |
| Stimulus card | 200 × 200 pixels |
| Card image format | SVG |
| Feedback font | 32 pt, bold |
| Feedback duration | 1000 ms |

**Sorting Rule Sequence:**
1. Color
2. Shape
3. Number
4. Color (repeat)
5. Shape (repeat)
6. Number (repeat)

**Category Completion Criterion:** 10 consecutive correct responses

---

## S3. Exclusion Criteria Details

### S3.1 Trial-Level Exclusions

Trial-level datasets retain all trials and add quality flags; exclusions below are applied in analysis loaders (WCST cleaning removes invalid records before analysis).

| Task | Criterion | Threshold | Rationale |
|------|-----------|-----------|-----------|
| All | Timeout | Flagged; excluded for RT analyses | No response indicates disengagement |
| All | Missing RT | Flagged; excluded for RT analyses | Technical failure |
| Stroop | RT validity | 200–3000 ms | Remove anticipations/late responses |
| WCST | RT minimum | ≥ 200 ms | Remove anticipations |
| WCST | RT validity (RT indices) | 200–10,000 ms | Exclude extreme lapses for RT metrics |

### S3.2 Participant-Level Exclusions

**Stroop Task:**

| Criterion | Threshold | Exclusion Rationale |
|-----------|-----------|---------------------|
| Completed trials | < 108 | Did not finish the Stroop task |
| Overall accuracy (timeout = incorrect) | < 70% | Poor task comprehension or engagement |

**WCST:**

| Criterion | Threshold | Exclusion Rationale |
|-----------|-----------|---------------------|
| Valid trials | < 60 | Insufficient data for performance indices |
| Single card selection ratio | > 85% | Fixed response strategy (not adapting to feedback) |

---

## S4. Derived Variable Computations

### S4.1 Stroop Task Variables

**Stroop Interference Effect (Primary DV):**
```
Stroop_Effect = mean(RT | cond=incongruent, timeout=False, 200<=RT<=3000)
              - mean(RT | cond=congruent, timeout=False, 200<=RT<=3000)
Stroop_Effect_Correct = mean(RT | cond=incongruent, correct=True, timeout=False, 200<=RT<=3000)
                      - mean(RT | cond=congruent, correct=True, timeout=False, 200<=RT<=3000)
```

**Stroop Facilitation Effect:**
```
Stroop_Facilitation = mean(RT | cond=neutral, timeout=False, 200<=RT<=3000)
                    - mean(RT | cond=congruent, timeout=False, 200<=RT<=3000)
Stroop_Facilitation_Correct = mean(RT | cond=neutral, correct=True, timeout=False, 200<=RT<=3000)
                           - mean(RT | cond=congruent, correct=True, timeout=False, 200<=RT<=3000)
```

**Condition-Specific Accuracy:**
```
correct = correct & ~timeout
Accuracy_cond = mean(correct | cond) × 100
```

**Condition-Specific Mean RT:**
```
MeanRT_cond = mean(RT | cond, timeout=False, 200<=RT<=3000)
MeanRT_cond_Correct = mean(RT | cond, correct=True, timeout=False, 200<=RT<=3000)
```

### S4.2 WCST Variables

All WCST indices are computed on cleaned trials (valid fields/conditions/cards, RT >= 200 ms). RT-based summaries optionally apply the 10,000 ms upper bound via the RT-valid flag.

**Categories Completed:**
```
n_categories = count(distinct completed sorting rules)
Range: 0-6
```

**Total Errors:**
```
Total_Errors = n(correct=False)
```

**Perseverative Errors:**
```
PE = Σ(is_error=True AND matches_previous_rule=True AND unambiguous=True)

Conditions for perseverative error:
1. Previous rule exists AND differs from current rule
2. Chosen card matches stimulus on previous rule dimension
3. Exactly one reference card matches on previous dimension

PE_rate = PE / Total_Errors × 100
```

**Non-Perseverative Errors:**
```
NPE = Total_Errors - PE
NPE_rate = NPE / Total_Errors × 100
```

**Perseverative Responses:**
```
PR = Σ(matches_previous_rule=True AND unambiguous=True)
PR_rate = PR / n_trials × 100
```

**Conceptual Level Responses:**
```
CLR = count(correct responses in runs of ≥3 consecutive correct)
CLR_rate = CLR / n_trials × 100
```

**Failure to Maintain Set:**
```
FMS = count(error episodes following ≥5 consecutive correct responses)
```

**Trials to First Category:**
```
Trials_First_Cat = trial_index when first category is completed
```

**Learning Efficiency:**
```
Learning_Efficiency = mean(trials_per_category | first 3 categories)
                    - mean(trials_per_category | last 3 categories)
```

---

## S4.3 Primary Conventional and Temporal Dynamics Indices

All indices below use the trial-level exclusions in S3.1. RT-based indices use task-specific RT bounds (Stroop: 200-3000 ms; WCST: 200-10,000 ms). Accuracy metrics treat timeouts as incorrect where applicable.

### S4.3.1 Conventional Indices (Traditional)

**Stroop RT interference (incongruent - congruent):**
```
RT_incong = mean(rt_ms | condition=incongruent, timeout=False, rt_valid=True)
RT_cong   = mean(rt_ms | condition=congruent, timeout=False, rt_valid=True)
stroop_rt_interference = RT_incong - RT_cong
```

**Stroop accuracy interference (incongruent - congruent):**
```
ACC_incong = mean(correct | condition=incongruent, timeout=False)
ACC_cong   = mean(correct | condition=congruent, timeout=False)
stroop_acc_interference = ACC_incong - ACC_cong
```

**WCST categories completed (0-6):**
```
wcst_categories_completed = count(rule_segments)
```
Rule segments are defined by changes in the active sorting rule; each completed category corresponds to one segment.

**WCST perseverative error rate:**
```
wcst_perseverative_error_rate = 100 * sum(isPE) / n_trials
```

### S4.3.2 Temporal Dynamics Indices

**Time-on-task drift (RT slope):**
```
RT_slope = OLS_slope(rt_ms ~ trial_order)
```
For Stroop, slopes use incongruent correct trials only. For WCST, slopes are computed within each rule segment and averaged.

**Within-task dispersion (RT SD):**
```
RT_SD_incong = SD(rt_ms | condition=incongruent, rt_valid=True)
```

**Post-perturbation recovery:**

Stroop post-error slowing:
```
PES = mean(rt_ms | prev_correct=False, rt_valid=True)
    - mean(rt_ms | prev_correct=True, rt_valid=True)
```

Stroop post-error accuracy drop:
```
post_error_acc_diff = mean(correct | prev_correct=False)
                    - mean(correct | prev_correct=True)
```

WCST post-switch RT cost (k1):
```
switch_cost_rt_k1 = rt_{switch+1} - mean(rt_{switch-3..switch-1})
```
Computed per rule switch and averaged across switches.

WCST post-error accuracy:
```
wcst_post_error_accuracy = mean(correct_{t+1} | correct_t=False)
```

WCST post-switch error rate:
```
wcst_post_switch_error_rate = 1 - mean(correct | trials in [switch, switch+4])
```

WCST trials to reacquisition:
```
wcst_trials_to_rule_reacquisition = mean(trials until 3 consecutive correct after switch)
```

---

## S5. Data Storage Schema

### S5.1 Firebase Document Structure

**Participant Document** (`/participants/{participantId}`):
```json
{
  "participantId": "string",
  "studentId": "string",
  "gender": "남자" | "여자",
  "age": number,
  "createdAt": "ISO8601 timestamp"
}
```

**Survey Document** (`/participants/{id}/surveys/{surveyName}`):
```json
{
  "surveyName": "ucla" | "dass",
  "responses": [number, ...],
  "score": number (UCLA only),
  "scores": {
    "D": number,
    "A": number,
    "S": number
  } (DASS only),
  "start_time": "ISO8601",
  "end_time": "ISO8601",
  "duration_seconds": number
}
```

**Cognitive Test Document** (`/participants/{id}/cognitive_tests/{taskName}`):
```json
{
  "task": "stroop" | "wcst",
  "trials": [/* array of trial objects */],
  "summary": {/* task-specific summary statistics */},
  "config": {
    "version": "1.0.0",
    "reverse_button": boolean (Stroop)
  },
  "session_id": "string",
  "start_time": "ISO8601",
  "end_time": "ISO8601",
  "duration_seconds": number
}
```

### S5.2 Trial-Level Data Fields

**Stroop Trial Object:**

| Field | Type | Description |
|-------|------|-------------|
| participant_id | string | Participant identifier |
| task | string | "stroop" |
| trial_index | integer | 0-107 (main trials) |
| block_index | integer | null (single block) |
| stim_onset_ms | float | Stimulus onset timestamp |
| resp_time_ms | float | Response timestamp |
| rt_ms | float | Reaction time (resp - onset) |
| correct | boolean | Response accuracy |
| timeout | boolean | No response within deadline |
| cond | string | "congruent" / "incongruent" / "neutral" |
| text | string | Stimulus word (Korean) |
| letterColor | string | Ink color key |
| userColor | string | Response color / "noResp" |

**WCST Trial Object:**

| Field | Type | Description |
|-------|------|-------------|
| participant_id | string | Participant identifier |
| task | string | "wcst" |
| trial_index | integer | 0-127 |
| block_index | integer | 0 or 1 (64 cards each) |
| stim_onset_ms | float | Stimulus onset timestamp |
| resp_time_ms | float | Response timestamp |
| rt_ms | float | Reaction time |
| correct | boolean | Response accuracy |
| timeout | boolean | Always false (no timeout) |
| cond | string | Current rule: "color" / "shape" / "number" |
| cardNumber | integer | Stimulus count (1-4) |
| cardColor | string | Stimulus color |
| cardShape | string | Stimulus shape |
| targetCardPath | string | SVG file path |
| chosenCard | string | Selected reference card |
| ruleAtThatTime | string | Active sorting rule |
| isPR | boolean | Perseverative response |
| isPE | boolean | Perseverative error |
| isNPE | boolean | Non-perseverative error |

---

## S6. Session Flow and Task Order

### S6.1 Complete Session Structure

```
1. Informed Consent
   ↓
2. Demographics (age, gender, education)
   ↓
3. UCLA Loneliness Scale (20 items)
   - Fixed item order
   - All items required before submission
   - Duration: ~5-10 minutes
   ↓
4. DASS-21 (21 items)
   - Fixed item order
   - All items required before submission
   - Duration: ~5-10 minutes
   ↓
5. Cognitive Task Introduction
   - Task descriptions
   - General instructions
   - Readiness confirmation
   ↓
6. Cognitive Tasks (Fixed Order)

   Position 1: Stroop
   ↓
   [2-minute rest interval with countdown]
   ↓
   Position 2: WCST (always last)
   ↓
7. Completion Screen
   - Debriefing information
   - Course credit confirmation
```

### S6.2 Task Order Randomization Logic

```dart
List<String> _generateTestOrder() {
  return ['/stroop', '/wcst'];
}
```

**Rationale for WCST Fixed Position:**
1. Longest task (10-15 min) - minimizes fatigue effects on earlier tasks
2. Requires sustained cognitive flexibility - most vulnerable to fatigue
3. Eliminates task-order confounds in set-shifting analyses

### S6.3 Rest Interval Specifications

| Parameter | Value |
|-----------|-------|
| Duration | 120 seconds (2 minutes) |
| Display | Countdown timer, coffee cup icon |
| Message | "잠시 휴식을 취하세요. 스트레칭이나 심호흡을 하시면 좋습니다." |
| Termination | Automatic (no user action required) |
| Timing | Between Stroop and WCST only |

---

## S7. Quality Control Procedures

### S7.1 Data Validation During Collection

| Check | Implementation | Action |
|-------|----------------|--------|
| Survey completion | All items required | Cannot submit until all answered |
| Duplicate submission | Firebase document check | Auto-advance if exists |
| Response timing | performance.now() | Log all timestamps |
| Trial data integrity | Real-time validation | Flag missing values |

### S7.2 Post-Hoc Quality Checks

These checks were used for descriptive inspection only and were not enforced as automatic exclusions in the preprocessing pipeline.

| Check | Criterion | Flag/Exclude |
|-------|-----------|--------------|
| RT distribution | Inspect for bimodality | Flag for review |
| Accuracy floor | < 50% (chance level) | Flag for review |
| Response patterns | Repeated same response | Flag for review |
| Session duration | Outside expected range | Flag for review |

---

## S8. Additional Analyses for Reviewer Response

### S8.1 Multicollinearity Diagnostics

Variance inflation factors (VIF) were low in both task samples (max = 2.59, mean = 2.11), indicating no problematic multicollinearity among UCLA and DASS covariates.

### S8.2 Primary Outcomes and FDR Control

All models used OLS with DASS subscales, age, and gender as covariates. Benjamini-Hochberg FDR was applied across seven focal outcomes.

| Outcome | n | UCLA beta | p | FDR q | Primary |
|---------|---:|----------:|---:|------:|:-------:|
| Stroop RT interference (correct) | 197 | 5.425 | 0.599 | 0.915 | No |
| Stroop accuracy interference | 197 | -0.001 | 0.784 | 0.915 | No |
| WCST categories completed | 197 | -0.023 | 0.755 | 0.915 | No |
| WCST perseverative error rate | 197 | -0.053 | 0.930 | 0.930 | No |
| Stroop interference RT slope | 197 | 19.033 | 0.0018 | 0.0125 | Yes |
| Stroop RT SD (incongruent) | 197 | 11.142 | 0.232 | 0.541 | No |
| WCST post-shift error RT mean | 192 | 202.104 | 0.029 | 0.103 | Yes |

### S8.3 Alternative DASS Specification (Total Score)

Primary endpoints remained significant under a DASS total-score covariate:

| Outcome | n | UCLA beta | p |
|---------|---:|----------:|---:|
| Stroop interference RT slope | 197 | 16.435 | 0.0049 |
| WCST post-shift error RT mean | 192 | 179.931 | 0.042 |

### S8.4 Predictor Entry Order Sensitivity

| Outcome | ΔR² (UCLA after DASS) | p | ΔR² (DASS after UCLA) | p |
|---------|----------------------:|---:|-----------------------:|---:|
| Stroop interference RT slope | 0.0482 | 0.0018 | 0.0124 | 0.463 |
| WCST post-shift error RT mean | 0.0249 | 0.029 | 0.0040 | 0.854 |

### S8.5 Slope Reliability and Bootstrap Stability

Split-half reliability for the Stroop interference slope was low (r = 0.064; Spearman-Brown = 0.120; n = 150). Bootstrap CI widths (100 resamples per participant) showed a broad distribution (n = 220; mean = 237.35, SD = 80.53, median = 220.84, IQR = [174.02, 290.70]).

### S8.6 Segment-Count Sensitivity (Interference Slope)

| Segments | UCLA beta | p |
|---------:|----------:|---:|
| 3 | 31.044 | 0.00041 |
| 4 | 19.033 | 0.0018 |
| 5 | 16.436 | 0.00082 |
| 6 | 16.266 | 0.000065 |

### S8.7 Trial-Level OLS Check

A trial-level OLS model with participant fixed effects (segment x UCLA) was not significant (beta = -0.011, SE = 1.740, p = 0.995; n_trials = 20,985; n_participants = 197). A mixed-effects model with random intercepts and random segment slopes by participant yielded the same null interaction (beta = -0.011, SE = 3.041, p = 0.997; n_trials = 20,985; n_participants = 197). When the model targeted the interference slope directly (congruent/incongruent only; condition coded -0.5/+0.5; log RT; continuous within-participant trial position), the trial_scaled x condition x UCLA interaction was significant. The full random-effects structure (1 + trial_scaled + cond_code) showed a boundary warning in statsmodels but the estimate was similar (beta = 0.0578, SE = 0.0166, p = 0.000508). The simplified structure (1 + trial_scaled) removed the warning and retained significance (beta = 0.0580, SE = 0.0167, p = 0.000522; n_trials = 13,950; n_participants = 197). An lme4 cross-check reproduced the effect for both structures (beta = 0.0578, SE = 0.0166, p = 0.000510; beta = 0.0580, SE = 0.0167, p = 0.000523).

### S8.8 RT Variability QC and Sensitivity

Participant-level RT SD distributions (valid trials only):

| Task | n | RT SD mean | SD | Min | Max | P97.5 |
|------|---:|-----------:|---:|----:|----:|------:|
| Stroop (incongruent) | 220 | 311.91 | 94.37 | 131.84 | 641.25 | 520.64 |
| WCST (all valid) | 212 | 943.88 | 372.92 | 343.70 | 2315.38 | 1858.13 |

Excluding participants above the 97.5th percentile of RT SD:

| Outcome | n | UCLA beta | p |
|---------|---:|----------:|---:|
| Stroop interference RT slope | 193 | 17.133 | 0.0038 |
| WCST post-shift error RT mean | 189 | 138.974 | 0.134 |

### S8.9 Winsorization and Robust SE

Winsorizing RT outcomes at 2.5% tails yielded similar conclusions (Stroop interference slope: beta = 17.055, p = 0.0026; WCST post-shift error RT mean: beta = 182.995, p = 0.0337). HC3-robust SEs preserved the Stroop effect (p = 0.0024) and attenuated the WCST effect (p = 0.066).

### S8.10 WCST Shift Counts

Shift-window counts (n = 212): shift trials mean = 28.51 (SD = 10.95, range = 0–92), post-shift errors mean = 10.82 (SD = 8.42, range = 0–61).

---

## References

Anwyl-Irvine, A. L., Dalmaijer, E. S., Hodges, N., & Evershed, J. K. (2021). Realistic precision and accuracy of online experiment platforms, web browsers, and devices. *Behavior Research Methods*, 53(4), 1407-1425.

Bridges, D., Pitiot, A., MacAskill, M. R., & Peirce, J. W. (2020). The timing mega-study: Comparing a range of experiment generators, both lab-based and online. *PeerJ*, 8, e9414.

Heaton, R. K., Chelune, G. J., Talley, J. L., Kay, G. G., & Curtiss, G. (1993). *Wisconsin Card Sorting Test manual: Revised and expanded*. Psychological Assessment Resources.

Lovibond, P. F., & Lovibond, S. H. (1995). The structure of negative emotional states: Comparison of the Depression Anxiety Stress Scales (DASS) with the Beck Depression and Anxiety Inventories. *Behaviour Research and Therapy*, 33(3), 335-343.

Russell, D. W. (1996). UCLA Loneliness Scale (Version 3): Reliability, validity, and factor structure. *Journal of Personality Assessment*, 66(1), 20-40.
