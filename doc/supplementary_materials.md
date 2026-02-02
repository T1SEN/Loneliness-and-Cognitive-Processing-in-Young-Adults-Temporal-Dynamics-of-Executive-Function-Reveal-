# Supplementary Materials

## S1. Technical Specifications

- Implementation: Flutter Web + Firebase
- Access: desktop browser only
- RT recording: `performance.now()` (sub-millisecond resolution)

## S2. QC and Preprocessing Criteria

### S2.1 Trial-level

| Task | Criterion | Rule |
|---|---|---|
| Stroop | RT window | 200-3000 ms (`is_rt_valid`) |
| Stroop | Timeout | retained but treated as incorrect; excluded from RT means |
| WCST | Rule/card validity | colour/shape/number only; keep 4 valid target cards |
| WCST | RT minimum | < 200 ms removed |
| WCST | RT window | 200-10,000 ms (`is_rt_valid`) |
| WCST | Timeout | treated as incorrect; excluded from RT means |

### S2.2 Participant-level

| Task | Criterion | Threshold |
|---|---|---|
| Common | Survey validity | UCLA total + all DASS subscales + gender present |
| Common | Task completion | Stroop + WCST completion record |
| Stroop | Completion/accuracy | 108 trials completed; accuracy >= .70 |
| WCST | Valid trials | >= 60 |
| WCST | Single-card ratio | <= .85 |

## S3. Variable Derivations

### S3.1 Stroop

```
stroop_interference = mean(rt_ms | incongruent, timeout=False, rt_valid=True)
                   - mean(rt_ms | congruent,   timeout=False, rt_valid=True)
```

### S3.2 WCST Perseverative Error Rate

```
wcst_perseverative_error_rate = 100 * sum(isPE) / n_trials
```

## S4. WCST Phase Definitions

### S4.1 Three-phase (primary)

Rule segments are defined by changes in `ruleAtThatTime` and use up to 6 categories.

- **exploration**: after rule switch until the first correct response
- **confirmation**: from the first correct response through achieving 3 consecutive correct responses (inclusive)
- **exploitation**: after achieving 3 consecutive correct responses
- If 3 consecutive correct responses never occur, the post-first-correct period remains confirmation.
- If there are no correct responses, the entire segment is exploration.

### S4.2 Two-phase (auxiliary; rule search/application)

- **rule search (pre-exploitation)** = exploration + confirmation
- **rule application (exploitation)** = after 3 consecutive correct responses

All phase RTs are computed on **all trials** (errors included), using valid RTs and **non-timeout** trials only.

## S5. WCST Phase Validity (OLS, DASS-controlled)

Covariates: DASS-Dep/Anx/Stress + age + gender. OLS (non-robust).

### S5.1 Three-phase regressions

| Outcome                     |   n |   UCLA beta |       p |
|:----------------------------|----:|------------:|--------:|
| Exploration RT (all)        | 208 |     106.173 | 0.23275 |
| Confirmation RT (all)       | 212 |     127.352 | 0.00934 |
| Exploitation RT (all)       | 212 |      27.148 | 0.45795 |
| Exploration - Exploitation  | 208 |      78.691 | 0.34629 |
| Confirmation - Exploitation | 212 |     100.204 | 0.01115 |
| Exploration - Confirmation  | 208 |     -28.349 | 0.72316 |

### S5.2 Two-phase regressions (rule search / rule application)

| Outcome                                  |   n |   UCLA beta |       p |
|:-----------------------------------------|----:|------------:|--------:|
| Rule search (pre-exploitation) RT (all)  | 212 |     122.573 | 0.0101  |
| Rule application (exploitation) RT (all) | 212 |      27.148 | 0.45795 |
| Rule search - Rule application           | 212 |      95.425 | 0.01296 |

### S5.3 Two-phase regressions with alternative thresholds

**confirm_len = 2**

| Outcome                                  |   n |   UCLA beta |       p |
|:-----------------------------------------|----:|------------:|--------:|
| Rule search (pre-exploitation) RT (all)  | 212 |     134.121 | 0.02495 |
| Rule application (exploitation) RT (all) | 212 |      49.979 | 0.18962 |
| Rule search - Rule application           | 212 |      84.143 | 0.06108 |

**confirm_len = 4**

| Outcome                                  |   n |   UCLA beta |       p |
|:-----------------------------------------|----:|------------:|--------:|
| Rule search (pre-exploitation) RT (all)  | 212 |     118.728 | 0.01037 |
| Rule application (exploitation) RT (all) | 212 |      22.231 | 0.5004  |
| Rule search - Rule application           | 212 |      96.497 | 0.00369 |

## S6. WCST Phase Reliability (odd/even category split-half)

| Phase                          |   n |     r |   Spearman-Brown |
|:-------------------------------|----:|------:|-----------------:|
| exploration                    | 183 | 0.186 |            0.314 |
| confirmation                   | 210 | 0.595 |            0.746 |
| exploitation                   | 209 | 0.748 |            0.856 |
| Rule search (pre-exploitation) | 210 | 0.647 |            0.786 |

### S6.1 Alternative thresholds (confirm_len = 2 / 4)

**confirm_len = 2**

| Phase                          |   n |     r |   Spearman-Brown |
|:-------------------------------|----:|------:|-----------------:|
| exploration                    | 183 | 0.186 |            0.314 |
| confirmation                   | 210 | 0.532 |            0.695 |
| exploitation                   | 209 | 0.693 |            0.818 |
| Rule search (pre-exploitation) | 210 | 0.542 |            0.703 |

**confirm_len = 4**

| Phase                          |   n |     r |   Spearman-Brown |
|:-------------------------------|----:|------:|-----------------:|
| exploration                    | 183 | 0.186 |            0.314 |
| confirmation                   | 210 | 0.61  |            0.758 |
| exploitation                   | 209 | 0.671 |            0.803 |
| Rule search (pre-exploitation) | 210 | 0.645 |            0.785 |

---

## S7. Stroop Trial-level LMM (supplementary)

Trial-level mixed models use QC-passed trials (correct, non-timeout, valid RT). Covariates: DASS-Dep/Anx/Stress, age, gender.

**Interpretation note.** S7.1 tests general within-task slowing (segment x UCLA) pooled across conditions. It does **not** test interference drift because it omits the condition-dependent interaction (segment x condition x UCLA). The primary hypothesis concerns changes in the *interference* effect over time, which is directly evaluated in S7.2 via the trial_scaled x cond x UCLA interaction. Therefore, a null S7.1 interaction does not contradict a significant S7.2 interaction.

### S7.1 Full-trial LMM (segment x UCLA)

Model: `rt_ms ~ segment * z_ucla_score + C(cond) + DASS(3) + age + gender`  
Random effects: participant-level **1 + segment**

|   n_trials |   n_participants |   segment x UCLA beta |       p |
|-----------:|-----------------:|----------------------:|--------:|
|      22544 |              212 |               -0.0466 | 0.98707 |

### S7.2 Interference-slope LMM (trial_scaled x cond x UCLA)

Model: `log_rt ~ trial_scaled * cond_code * z_ucla_score + DASS(3) + age + gender`  
Random effects (preferred): **1 + trial_scaled**

|   n_trials |   n_participants |   trial_scaled x cond x UCLA beta |      p |
|-----------:|-----------------:|----------------------------------:|-------:|
|      14986 |              212 |                            0.0509 | 0.0017 |

---

## S8. Stroop Interference Slope Sensitivity (2/3/6 segments)

We evaluated how the interference slope varies with the number of within-subject segments (OLS, DASS-controlled).

|   Segments |   n |   UCLA beta |        p |     R2 |
|-----------:|----:|------------:|---------:|-------:|
|          2 | 212 |      40.14  | 0.00159  | 0.0605 |
|          3 | 212 |      28.406 | 0.000678 | 0.0692 |
|          6 | 212 |      14.739 | 0.000106 | 0.0909 |

---

## S9. WCST Complete-6 Categories Subsample

Complete-6 participants: n = 193. Regression sample sizes may be smaller due to missing phase RTs.

### S9.1 Three-/Two-phase regressions (OLS, all trials)

| Outcome                                 |   n |   UCLA beta |       p |
|:----------------------------------------|----:|------------:|--------:|
| Exploration RT (all)                    | 191 |      60.648 | 0.43047 |
| Confirmation RT (all)                   | 191 |     100.817 | 0.03186 |
| Exploitation RT (all)                   | 191 |       3.115 | 0.90922 |
| Confirmation - Exploitation             | 191 |      97.702 | 0.00855 |
| Rule search (pre-exploitation) RT (all) | 191 |      99.038 | 0.02842 |
| Rule search - Rule application          | 191 |      95.923 | 0.00626 |

### S9.2 Reliability (odd/even split-half)

| Phase                          |   n |     r |   Spearman-Brown |
|:-------------------------------|----:|------:|-----------------:|
| exploration                    | 173 | 0.183 |            0.309 |
| confirmation                   | 193 | 0.57  |            0.726 |
| exploitation                   | 193 | 0.696 |            0.821 |
| Rule search (pre-exploitation) | 193 | 0.625 |            0.769 |

---

## S10. WCST Phase-complete Sample (N = 208)

Participants with all three phase RTs available.

| Outcome                                 |   n |   UCLA beta |       p |
|:----------------------------------------|----:|------------:|--------:|
| Exploration RT (all)                    | 208 |     107.979 | 0.22559 |
| Confirmation RT (all)                   | 208 |     142.766 | 0.00786 |
| Exploitation RT (all)                   | 208 |      33.491 | 0.37187 |
| Confirmation - Exploitation             | 208 |     109.274 | 0.0078  |
| Rule search (pre-exploitation) RT (all) | 208 |     139.564 | 0.00786 |
| Rule search - Rule application          | 208 |     106.073 | 0.00799 |

---

## S11. WCST 3-phase Threshold Sensitivity (phase-complete sample)

**confirm_len = 2**

| Outcome                     |   n |   UCLA beta |       p |
|:----------------------------|----:|------------:|--------:|
| Exploration RT (all)        | 208 |     107.979 | 0.22559 |
| Confirmation RT (all)       | 208 |     157.517 | 0.01561 |
| Exploitation RT (all)       | 208 |      49.969 | 0.20023 |
| Confirmation - Exploitation | 208 |     107.549 | 0.0297  |

**confirm_len = 4**

| Outcome                     |   n |   UCLA beta |       p |
|:----------------------------|----:|------------:|--------:|
| Exploration RT (all)        | 208 |     107.979 | 0.22559 |
| Confirmation RT (all)       | 208 |     118.09  | 0.01362 |
| Exploitation RT (all)       | 208 |      18.043 | 0.588   |
| Confirmation - Exploitation | 208 |     100.047 | 0.00458 |

---

## S12. Stroop Interference One-sample t-test

|   n |   Mean (ms) |      SD |      t |        p |
|----:|------------:|--------:|-------:|---------:|
| 212 |     139.466 | 104.264 | 19.476 | 4.97e-49 |

---

## S13. WCST Normative Statistics

|   n |   Categories M |    SD |   n_complete6 |   % complete6 |
|----:|---------------:|------:|--------------:|--------------:|
| 212 |          5.774 | 0.829 |           193 |         91.04 |

---

## S14. DASS Total Models (supplementary)

| Outcome                                |   n |   UCLA beta |        p |
|:---------------------------------------|----:|------------:|---------:|
| Stroop Interference RT                 | 212 |       2.181 | 0.81946  |
| WCST Perseverative Error Rate (%)      | 212 |      -0.203 | 0.71403  |
| Stroop Interference RT Slope           | 212 |      19.754 | 0.000511 |
| WCST Confirmation RT (all)             | 212 |     137.957 | 0.00697  |
| WCST Exploitation RT (all)             | 212 |      44.707 | 0.21261  |
| WCST Confirmation - Exploitation (all) | 212 |      93.25  | 0.01734  |

---

## S15. Reverse-order Models (UCLA entered before DASS)

| Outcome                                              |   n |   Delta R2 (UCLA first) |        p |   Delta R2 (DASS after UCLA) |      p  |
|:-----------------------------------------------------|----:|------------------------:|---------:|-----------------------------:|--------:|
| Stroop Interference RT                               | 212 |                 2e-05   | 0.95347  |                      0.00575 | 0.75315 |
| WCST Perseverative Error Rate (%)                    | 212 |                 0.01513 | 0.07504  |                      0.01208 | 0.46786 |
| Stroop Interference RT Slope                         | 212 |                 0.07501 | 5.33e-05 |                      0.01612 | 0.30216 |
| WCST Confirmation RT (ms, all trials)                | 212 |                 0.01458 | 0.07779  |                      0.02055 | 0.21946 |
| WCST Exploitation RT (ms, all trials)                | 212 |                 0.00253 | 0.45542  |                      0.01634 | 0.3076  |
| WCST Confirmation - Exploitation RT (ms, all trials) | 212 |                 0.01255 | 0.10449  |                      0.02171 | 0.20394 |

---

## S16. FDR Correction (UCLA p-values)

| Outcome                                              |   UCLA p |    FDR q |
|:-----------------------------------------------------|---------:|---------:|
| Stroop Interference RT                               | 0.62876  | 0.75451  |
| WCST Perseverative Error Rate (%)                    | 0.77503  | 0.77503  |
| Stroop Interference RT Slope                         | 0.000131 | 0.000789 |
| WCST Confirmation RT (ms, all trials)                | 0.01079  | 0.02368  |
| WCST Exploitation RT (ms, all trials)                | 0.37103  | 0.55655  |
| WCST Confirmation - Exploitation RT (ms, all trials) | 0.01184  | 0.02368  |

---

## S17. Stroop Condition Balance within 4 Segments

|   Segment |   Congruent |   Incongruent |   Neutral |
|----------:|------------:|--------------:|----------:|
|         1 |       0.337 |         0.332 |     0.331 |
|         2 |       0.323 |         0.334 |     0.342 |
|         3 |       0.334 |         0.334 |     0.331 |
|         4 |       0.338 |         0.332 |     0.329 |

---

## S18. Stroop Condition Balance within 2/3/6 Segments

### S18.1 Two segments

|   Segment |   Congruent |   Incongruent |   Neutral |
|----------:|------------:|--------------:|----------:|
|         1 |       0.33  |         0.333 |     0.336 |
|         2 |       0.336 |         0.333 |     0.33  |

### S18.2 Three segments

|   Segment |   Congruent |   Incongruent |   Neutral |
|----------:|------------:|--------------:|----------:|
|         1 |       0.331 |         0.337 |     0.332 |
|         2 |       0.329 |         0.332 |     0.339 |
|         3 |       0.34  |         0.331 |     0.329 |

### S18.3 Six segments

|   Segment |   Congruent |   Incongruent |   Neutral |
|----------:|------------:|--------------:|----------:|
|         1 |       0.341 |         0.328 |     0.331 |
|         2 |       0.321 |         0.347 |     0.332 |
|         3 |       0.329 |         0.325 |     0.346 |
|         4 |       0.329 |         0.338 |     0.332 |
|         5 |       0.345 |         0.328 |     0.328 |
|         6 |       0.335 |         0.334 |     0.331 |

---

**Output files**
- `outputs/stats/supplementary/overall/wcst_phase_rt_ols_alltrials.csv`
- `outputs/stats/supplementary/overall/wcst_phase_pre_exploit_rt_ols_alltrials.csv`
- `outputs/stats/supplementary/overall/wcst_phase_split_half_reliability.csv`
- `outputs/stats/supplementary/overall/wcst_phase_pre_exploit_rt_ols_m2_alltrials.csv`
- `outputs/stats/supplementary/overall/wcst_phase_pre_exploit_rt_ols_m4_alltrials.csv`
- `outputs/stats/supplementary/overall/wcst_phase_split_half_reliability_m2.csv`
- `outputs/stats/supplementary/overall/wcst_phase_split_half_reliability_m4.csv`
- `outputs/stats/supplementary/overall/stroop_lmm/stroop_trial_level_lmm.csv`
- `outputs/stats/supplementary/overall/stroop_lmm/stroop_interference_slope_lmm.csv`
- `outputs/stats/supplementary/overall/stroop_lmm/stroop_interference_slope_lmm_variants.csv`
- `outputs/stats/supplementary/overall/stroop_interference_slope_segment_sensitivity_2_3_6.csv`
- `outputs/stats/supplementary/overall/wcst_phase_3_2phase_6categories_ols_alltrials.csv`
- `outputs/stats/supplementary/overall/wcst_phase_3_2phase_6categories_split_half_reliability.csv`
- `outputs/stats/supplementary/overall/wcst_phase_3phase_complete_ols_alltrials.csv`
- `outputs/stats/supplementary/overall/wcst_phase_3phase_threshold_sensitivity_complete_ols_alltrials.csv`
- `outputs/stats/supplementary/overall/stroop_interference_ttest.csv`
- `outputs/stats/supplementary/overall/wcst_normative_stats.csv`
- `outputs/stats/supplementary/overall/dass_total_models.csv`
- `outputs/stats/supplementary/overall/ucla_first_model_comparison.csv`
- `outputs/stats/supplementary/overall/ucla_fdr_qvalues.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment_pivot.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment_2.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment_2_pivot.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment_3.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment_3_pivot.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment_6.csv`
- `outputs/stats/supplementary/overall/stroop_condition_balance_by_segment_6_pivot.csv`
