# QC Exclusion Report (complete -> complete/filtered)

- Generated at (UTC): 2025-12-18T05:37:05+00:00
- Input N: 205
- Excluded N: 14
- Included N: 191

## Thresholds
- age < 0
- PRP accuracy < 80%
- WCST totalErrorCount >= 60 OR perseverativeResponses >= 60
- Stroop accuracy < 93% OR mean RT > 1500ms
- Trial-derived duration > 1000s

## Counts (per rule, not unique)
- age<0: 1
- prp_accuracy<80%: 3
- wcst_totalError>=60_or_persevResp>=60: 8
- stroop_accuracy<93%_or_mrt>1500ms: 2
- trial_duration>1000s: 1

## Duration Corrections (summary duration was corrupted)

| participantId                | testName   |   duration_seconds_old |   trial_duration_s |   duration_seconds_new |    diff_s |
|:-----------------------------|:-----------|-----------------------:|-------------------:|-----------------------:|----------:|
| 3xCaqfcHxRQxuOlEzXz3PNcHODH2 | prp        |                   2445 |            295.731 |                300.057 |   2149.27 |
| s5gTyWKzfkdHXNT3qicwPmdONYv1 | stroop     |                 344749 |            166.305 |                167.36  | 344583    |

## Excluded Participants (overview)

| participantId                | gender   |   age |   ucla_score |   dass_depression |   dass_anxiety |   dass_stress |   acc_t1_prp |   acc_t2_prp |   accuracy_stroop |   mrt_total_stroop |   totalErrorCount_wcst |   perseverativeResponses_wcst | reasons                                                    |
|:-----------------------------|:---------|------:|-------------:|------------------:|---------------:|--------------:|-------------:|-------------:|------------------:|-------------------:|-----------------------:|------------------------------:|:-----------------------------------------------------------|
| 1upUBO9zBnhHcpJ1WoAnaBXWKRV2 | 남성     |    22 |           42 |                 8 |              6 |            10 |      95.8333 |      87.5    |           99.0741 |                935 |                     65 |                            90 | wcst_totalError>=60_or_persevResp>=60                      |
| 23AEyocWhqY955WHbxhHRSgQ1ms2 | 여성     |    19 |           43 |                12 |             14 |            22 |      50      |      53.3333 |           99.0741 |                800 |                     18 |                            23 | prp_accuracy<80%                                           |
| 3rlzYufVjVVLv3ZFLcpfZUtXEEi1 | 여성     |    21 |           42 |                 4 |              8 |             6 |      95      |      95.8333 |           89.8148 |                669 |                     34 |                            39 | stroop_accuracy<93%_or_mrt>1500ms                          |
| 50R0P1MIMvRvK1qpqfGROchII6R2 | 남성     |    22 |           30 |                 0 |              0 |             0 |      98.3333 |      97.5    |           98.1481 |                811 |                     54 |                            60 | wcst_totalError>=60_or_persevResp>=60                      |
| AyAnuCaltBbpEYbudoyNzTvXlcC2 | 여성     |    22 |           62 |                10 |              0 |            14 |      65      |      69.1667 |           96.2963 |               1009 |                     26 |                            29 | prp_accuracy<80%                                           |
| BbwUTWswjtPWhb5DvLAmh9Hdpyl2 | 여성     |    19 |           49 |                 8 |              0 |             2 |      96.6667 |      90      |           99.0741 |                829 |                     79 |                             2 | trial_duration>1000s;wcst_totalError>=60_or_persevResp>=60 |
| DaRehVxIOhXtEtUExdgvUl0mKcx2 | 여성     |    21 |           39 |                10 |              2 |             4 |      94.1667 |      95      |           99.0741 |               1244 |                     64 |                            62 | wcst_totalError>=60_or_persevResp>=60                      |
| Df83JuPuJkXqTGbENhM2gw5SJM43 | 여성     |    19 |           50 |                14 |             14 |            24 |      36.6667 |      36.6667 |           96.2963 |               1107 |                     12 |                            26 | prp_accuracy<80%                                           |
| EUhcW9kURScie1DwGPHqmNqotJB2 | 여성     |    19 |           36 |                 4 |              6 |             0 |      99.1667 |      95      |           99.0741 |               1137 |                     51 |                            67 | wcst_totalError>=60_or_persevResp>=60                      |
| PkpTH7pWpEOG9A86xSloOSozwTk2 | 여성     |    24 |           50 |                 2 |              0 |            10 |      98.3333 |      96.6667 |           98.1481 |               1378 |                     67 |                            63 | wcst_totalError>=60_or_persevResp>=60                      |
| qZZiI6GzgxOrpQDGu42ZA0xsGEJ2 | 여자     |    19 |           55 |                36 |             26 |            28 |      97.5    |      99.1667 |           98.1481 |                883 |                     64 |                            21 | wcst_totalError>=60_or_persevResp>=60                      |
| wdppvL4tIkVlGjou6aYzzjlaZXB2 | 남성     |    18 |           44 |                 0 |              0 |             0 |      89.1667 |      88.3333 |           96.2963 |                803 |                     75 |                            47 | wcst_totalError>=60_or_persevResp>=60                      |
| wkhxvE50megJGMA1EiROSZIJWNq1 | 여성     |   -20 |           42 |                 4 |              2 |             8 |      99.1667 |      99.1667 |          100      |                869 |                     22 |                            33 | age<0                                                      |
| yGTid3RXTdQe5JNosdSxvgktWIW2 | 여성     |    20 |           53 |                20 |             10 |            14 |      95      |      95.8333 |           91.6667 |               1553 |                     11 |                            21 | stroop_accuracy<93%_or_mrt>1500ms                          |

## Per-Participant Details

### 1upUBO9zBnhHcpJ1WoAnaBXWKRV2
- Reasons: wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=남성, age=22, birthDate=20030625, education=대학교 재학, createdAt=2025-12-07 05:02:05.310000+00:00
- Surveys: ucla_score=42.0, ucla_duration_s=62.0, dass_depression=8.0, dass_anxiety=6.0, dass_stress=10.0, dass_duration_s=53.0
- Summary: duration_seconds_prp=300.0, acc_t1_prp=95.83333333333331, acc_t2_prp=87.5, duration_seconds_stroop=168.0, accuracy_stroop=99.07407407407408, mrt_total_stroop=935.0, stroop_effect_stroop=143.0, duration_seconds_wcst=324.0, totalTrialCount_wcst=128.0, completedCategories_wcst=3.0, totalErrorCount_wcst=65.0, perseverativeResponses_wcst=90.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=0, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=647.7999999821186, prp_t1_resp_unique=2, prp_t2_resp_unique=2, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=99.07407407407408, stroop_rt_median_correct_ms=857.8000000119209, stroop_userColor_unique=4, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=49.21875, wcst_rt_median_ms=1292.7999999821186, wcst_isPE_n=62, wcst_isNPE_n=3, wcst_isPR_n=90, wcst_chosenCard_unique=4

### 23AEyocWhqY955WHbxhHRSgQ1ms2
- Reasons: prp_accuracy<80%
- Demographics: gender=여성, age=19, birthDate=20060712, education=대학교 재학, createdAt=2025-11-24 11:31:53.048000+00:00
- Surveys: ucla_score=43.0, ucla_duration_s=146.0, dass_depression=12.0, dass_anxiety=14.0, dass_stress=22.0, dass_duration_s=79.0
- Summary: duration_seconds_prp=349.0, acc_t1_prp=50.0, acc_t2_prp=53.333333333333336, duration_seconds_stroop=153.0, accuracy_stroop=99.07407407407408, mrt_total_stroop=800.0, stroop_effect_stroop=113.0, duration_seconds_wcst=236.0, totalTrialCount_wcst=85.0, completedCategories_wcst=6.0, totalErrorCount_wcst=18.0, perseverativeResponses_wcst=23.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=1, prp_t2_anticipation_n=1, prp_t2_rt_median_valid_ms=635.3999999985099, prp_t1_resp_unique=3, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=99.07407407407408, stroop_rt_median_correct_ms=769.0, stroop_userColor_unique=4, wcst_n_trials=85, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=78.82352941176471, wcst_rt_median_ms=1188.89999999851, wcst_isPE_n=8, wcst_isNPE_n=10, wcst_isPR_n=23, wcst_chosenCard_unique=4

### 3rlzYufVjVVLv3ZFLcpfZUtXEEi1
- Reasons: stroop_accuracy<93%_or_mrt>1500ms
- Demographics: gender=여성, age=21, birthDate=20040217, education=대학교 재학, createdAt=2025-12-06 15:54:24.589000+00:00
- Surveys: ucla_score=42.0, ucla_duration_s=57.0, dass_depression=4.0, dass_anxiety=8.0, dass_stress=6.0, dass_duration_s=73.0
- Summary: duration_seconds_prp=301.0, acc_t1_prp=95.0, acc_t2_prp=95.83333333333331, duration_seconds_stroop=140.0, accuracy_stroop=89.81481481481481, mrt_total_stroop=669.0, stroop_effect_stroop=105.0, duration_seconds_wcst=291.0, totalTrialCount_wcst=128.0, completedCategories_wcst=4.0, totalErrorCount_wcst=34.0, perseverativeResponses_wcst=39.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=0, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=651.3000001907349, prp_t1_resp_unique=2, prp_t2_resp_unique=2, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=89.81481481481481, stroop_rt_median_correct_ms=638.2000007629395, stroop_userColor_unique=4, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=73.4375, wcst_rt_median_ms=1086.4000000953674, wcst_isPE_n=21, wcst_isNPE_n=13, wcst_isPR_n=39, wcst_chosenCard_unique=4

### 50R0P1MIMvRvK1qpqfGROchII6R2
- Reasons: wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=남성, age=22, birthDate=20030218, education=대학교 재학, createdAt=2025-12-05 14:34:00.842000+00:00
- Surveys: ucla_score=30.0, ucla_duration_s=39.0, dass_depression=0.0, dass_anxiety=0.0, dass_stress=0.0, dass_duration_s=23.0
- Summary: duration_seconds_prp=316.0, acc_t1_prp=98.33333333333331, acc_t2_prp=97.5, duration_seconds_stroop=155.0, accuracy_stroop=98.14814814814817, mrt_total_stroop=811.0, stroop_effect_stroop=97.0, duration_seconds_wcst=279.0, totalTrialCount_wcst=128.0, completedCategories_wcst=4.0, totalErrorCount_wcst=54.0, perseverativeResponses_wcst=60.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=0, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=747.4000000953674, prp_t1_resp_unique=2, prp_t2_resp_unique=2, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=98.14814814814815, stroop_rt_median_correct_ms=782.9499998092651, stroop_userColor_unique=4, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=57.8125, wcst_rt_median_ms=1021.4999997615815, wcst_isPE_n=38, wcst_isNPE_n=16, wcst_isPR_n=60, wcst_chosenCard_unique=4

### AyAnuCaltBbpEYbudoyNzTvXlcC2
- Reasons: prp_accuracy<80%
- Demographics: gender=여성, age=22, birthDate=20030405, education=대학교 재학, createdAt=2025-11-12 11:19:26.378000+00:00
- Surveys: ucla_score=62.0, ucla_duration_s=789.0, dass_depression=10.0, dass_anxiety=0.0, dass_stress=14.0, dass_duration_s=633.0
- Summary: duration_seconds_prp=438.0, acc_t1_prp=65.0, acc_t2_prp=69.16666666666667, duration_seconds_stroop=177.0, accuracy_stroop=96.29629629629628, mrt_total_stroop=1009.0, stroop_effect_stroop=227.0, duration_seconds_wcst=376.0, totalTrialCount_wcst=112.0, completedCategories_wcst=6.0, totalErrorCount_wcst=26.0, perseverativeResponses_wcst=29.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=25, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=1321.5999999642372, prp_t1_resp_unique=3, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=96.29629629629629, stroop_rt_median_correct_ms=917.5499999821186, stroop_userColor_unique=4, wcst_n_trials=112, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=76.78571428571429, wcst_rt_median_ms=1523.3500000238419, wcst_isPE_n=13, wcst_isNPE_n=13, wcst_isPR_n=29, wcst_chosenCard_unique=4

### BbwUTWswjtPWhb5DvLAmh9Hdpyl2
- Reasons: trial_duration>1000s; wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=여성, age=19, birthDate=20061026, education=대학교 재학, createdAt=2025-11-04 04:56:57.086000+00:00
- Surveys: ucla_score=49.0, ucla_duration_s=46.0, dass_depression=8.0, dass_anxiety=0.0, dass_stress=2.0, dass_duration_s=39.0
- Summary: duration_seconds_prp=366.0, acc_t1_prp=96.66666666666669, acc_t2_prp=90.0, duration_seconds_stroop=158.0, accuracy_stroop=99.07407407407408, mrt_total_stroop=829.0, stroop_effect_stroop=165.0, duration_seconds_wcst=1069.0, totalTrialCount_wcst=128.0, completedCategories_wcst=1.0, totalErrorCount_wcst=79.0, perseverativeResponses_wcst=2.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=1, prp_t2_anticipation_n=1, prp_t2_rt_median_valid_ms=1152.8499999999767, prp_t1_resp_unique=2, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=99.07407407407408, stroop_rt_median_correct_ms=740.1000000000931, stroop_userColor_unique=4, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=38.28125, wcst_rt_median_ms=2753.0, wcst_isPE_n=1, wcst_isNPE_n=78, wcst_isPR_n=2, wcst_chosenCard_unique=4

### DaRehVxIOhXtEtUExdgvUl0mKcx2
- Reasons: wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=여성, age=21, birthDate=20040108, education=대학교 재학, createdAt=2025-11-27 12:57:32.798000+00:00
- Surveys: ucla_score=39.0, ucla_duration_s=95.0, dass_depression=10.0, dass_anxiety=2.0, dass_stress=4.0, dass_duration_s=46.0
- Summary: duration_seconds_prp=403.0, acc_t1_prp=94.16666666666669, acc_t2_prp=95.0, duration_seconds_stroop=201.0, accuracy_stroop=99.07407407407408, mrt_total_stroop=1244.0, stroop_effect_stroop=481.0, duration_seconds_wcst=402.0, totalTrialCount_wcst=128.0, completedCategories_wcst=4.0, totalErrorCount_wcst=64.0, perseverativeResponses_wcst=62.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=4, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=1450.0000000298023, prp_t1_resp_unique=3, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=99.07407407407408, stroop_rt_median_correct_ms=1132.0, stroop_userColor_unique=4, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=50.0, wcst_rt_median_ms=1647.25, wcst_isPE_n=38, wcst_isNPE_n=26, wcst_isPR_n=62, wcst_chosenCard_unique=4

### Df83JuPuJkXqTGbENhM2gw5SJM43
- Reasons: prp_accuracy<80%
- Demographics: gender=여성, age=19, birthDate=20060125, education=대학교 재학, createdAt=2025-11-27 07:20:36.324000+00:00
- Surveys: ucla_score=50.0, ucla_duration_s=226.0, dass_depression=14.0, dass_anxiety=14.0, dass_stress=24.0, dass_duration_s=79.0
- Summary: duration_seconds_prp=510.0, acc_t1_prp=36.66666666666666, acc_t2_prp=36.66666666666666, duration_seconds_stroop=187.0, accuracy_stroop=96.29629629629628, mrt_total_stroop=1107.0, stroop_effect_stroop=422.0, duration_seconds_wcst=213.0, totalTrialCount_wcst=88.0, completedCategories_wcst=6.0, totalErrorCount_wcst=12.0, perseverativeResponses_wcst=26.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=72, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=1605.5999999999767, prp_t1_resp_unique=3, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=96.29629629629629, stroop_rt_median_correct_ms=1017.0999999999767, stroop_userColor_unique=4, wcst_n_trials=88, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=86.36363636363636, wcst_rt_median_ms=1239.3500000000931, wcst_isPE_n=8, wcst_isNPE_n=4, wcst_isPR_n=26, wcst_chosenCard_unique=4

### EUhcW9kURScie1DwGPHqmNqotJB2
- Reasons: wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=여성, age=19, birthDate=20060814, education=대학교 재학, createdAt=2025-10-07 07:55:38.932000+00:00
- Surveys: ucla_score=36.0, ucla_duration_s=95.0, dass_depression=4.0, dass_anxiety=6.0, dass_stress=0.0, dass_duration_s=124.0
- Summary: duration_seconds_prp=345.0, acc_t1_prp=99.16666666666669, acc_t2_prp=95.0, duration_seconds_stroop=192.0, accuracy_stroop=99.07407407407408, mrt_total_stroop=1137.0, stroop_effect_stroop=131.0, duration_seconds_wcst=407.0, totalTrialCount_wcst=128.0, completedCategories_wcst=3.0, totalErrorCount_wcst=51.0, perseverativeResponses_wcst=67.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=3, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=989.7000000476836, prp_t1_resp_unique=2, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=99.07407407407408, stroop_rt_median_correct_ms=1081.0, stroop_userColor_unique=4, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=60.15625, wcst_rt_median_ms=1828.5, wcst_isPE_n=41, wcst_isNPE_n=10, wcst_isPR_n=67, wcst_chosenCard_unique=4

### PkpTH7pWpEOG9A86xSloOSozwTk2
- Reasons: wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=여성, age=24, birthDate=20010602, education=대학교 재학, createdAt=2025-11-30 07:26:45.259000+00:00
- Surveys: ucla_score=50.0, ucla_duration_s=59.0, dass_depression=2.0, dass_anxiety=0.0, dass_stress=10.0, dass_duration_s=39.0
- Summary: duration_seconds_prp=396.0, acc_t1_prp=98.33333333333331, acc_t2_prp=96.66666666666669, duration_seconds_stroop=219.0, accuracy_stroop=98.14814814814817, mrt_total_stroop=1378.0, stroop_effect_stroop=-8.0, duration_seconds_wcst=691.0, totalTrialCount_wcst=128.0, completedCategories_wcst=2.0, totalErrorCount_wcst=67.0, perseverativeResponses_wcst=63.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=1, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=1439.5999999046326, prp_t1_resp_unique=3, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=2, stroop_anticipation_n=0, stroop_acc_pct_trials=98.14814814814815, stroop_rt_median_correct_ms=1365.2999999523163, stroop_userColor_unique=5, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=47.65625, wcst_rt_median_ms=2452.899999976158, wcst_isPE_n=41, wcst_isNPE_n=26, wcst_isPR_n=63, wcst_chosenCard_unique=4

### qZZiI6GzgxOrpQDGu42ZA0xsGEJ2
- Reasons: wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=여자, age=19, birthDate=20061021, createdAt=2025-11-24 11:42:37.257000+00:00
- Surveys: ucla_score=55.0, ucla_duration_s=45.0, dass_depression=36.0, dass_anxiety=26.0, dass_stress=28.0, dass_duration_s=35.0
- Summary: duration_seconds_prp=307.0, acc_t1_prp=97.5, acc_t2_prp=99.16666666666669, duration_seconds_stroop=163.0, accuracy_stroop=98.14814814814817, mrt_total_stroop=883.0, stroop_effect_stroop=83.0, duration_seconds_wcst=327.0, totalTrialCount_wcst=128.0, completedCategories_wcst=1.0, totalErrorCount_wcst=64.0, perseverativeResponses_wcst=21.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=0, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=663.8999999761581, prp_t1_resp_unique=3, prp_t2_resp_unique=2, stroop_n_trials=108, stroop_timeout_n=1, stroop_anticipation_n=0, stroop_acc_pct_trials=98.14814814814815, stroop_rt_median_correct_ms=838.4000000059605, stroop_userColor_unique=5, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=50.0, wcst_rt_median_ms=1335.5999999940395, wcst_isPE_n=12, wcst_isNPE_n=52, wcst_isPR_n=21, wcst_chosenCard_unique=4

### wdppvL4tIkVlGjou6aYzzjlaZXB2
- Reasons: wcst_totalError>=60_or_persevResp>=60
- Demographics: gender=남성, age=18, birthDate=20071015, education=대학교 재학, createdAt=2025-10-17 06:48:32.354000+00:00
- Surveys: ucla_score=44.0, ucla_duration_s=105.0, dass_depression=0.0, dass_anxiety=0.0, dass_stress=0.0, dass_duration_s=52.0
- Summary: duration_seconds_prp=313.0, acc_t1_prp=89.16666666666667, acc_t2_prp=88.33333333333333, duration_seconds_stroop=152.0, accuracy_stroop=96.29629629629628, mrt_total_stroop=803.0, stroop_effect_stroop=120.0, duration_seconds_wcst=295.0, totalTrialCount_wcst=128.0, completedCategories_wcst=2.0, totalErrorCount_wcst=75.0, perseverativeResponses_wcst=47.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=1, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=601.1000000014901, prp_t1_resp_unique=3, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=96.29629629629629, stroop_rt_median_correct_ms=675.1500000003725, stroop_userColor_unique=4, wcst_n_trials=128, wcst_timeout_n=0, wcst_anticipation_n=2, wcst_acc_pct_trials=41.40625, wcst_rt_median_ms=1039.5500000026077, wcst_isPE_n=30, wcst_isNPE_n=45, wcst_isPR_n=47, wcst_chosenCard_unique=4

### wkhxvE50megJGMA1EiROSZIJWNq1
- Reasons: age<0
- Demographics: gender=여성, age=-20, birthDate=20441217, education=대학교 재학, createdAt=2025-11-21 03:29:08.999000+00:00
- Surveys: ucla_score=42.0, ucla_duration_s=294.0, dass_depression=4.0, dass_anxiety=2.0, dass_stress=8.0, dass_duration_s=204.0
- Summary: duration_seconds_prp=362.0, acc_t1_prp=99.16666666666669, acc_t2_prp=99.16666666666669, duration_seconds_stroop=162.0, accuracy_stroop=100.0, mrt_total_stroop=869.0, stroop_effect_stroop=137.0, duration_seconds_wcst=258.0, totalTrialCount_wcst=88.0, completedCategories_wcst=6.0, totalErrorCount_wcst=22.0, perseverativeResponses_wcst=33.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=0, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=1203.8000000044703, prp_t1_resp_unique=2, prp_t2_resp_unique=2, stroop_n_trials=108, stroop_timeout_n=0, stroop_anticipation_n=0, stroop_acc_pct_trials=100.0, stroop_rt_median_correct_ms=812.8499999940395, stroop_userColor_unique=4, wcst_n_trials=88, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=75.0, wcst_rt_median_ms=1458.8500000014901, wcst_isPE_n=14, wcst_isNPE_n=8, wcst_isPR_n=33, wcst_chosenCard_unique=4

### yGTid3RXTdQe5JNosdSxvgktWIW2
- Reasons: stroop_accuracy<93%_or_mrt>1500ms
- Demographics: gender=여성, age=20, birthDate=20050610, education=대학교 재학, createdAt=2025-12-08 05:22:48.212000+00:00
- Surveys: ucla_score=53.0, ucla_duration_s=189.0, dass_depression=20.0, dass_anxiety=10.0, dass_stress=14.0, dass_duration_s=262.0
- Summary: duration_seconds_prp=360.0, acc_t1_prp=95.0, acc_t2_prp=95.83333333333331, duration_seconds_stroop=239.0, accuracy_stroop=91.66666666666666, mrt_total_stroop=1553.0, stroop_effect_stroop=29.0, duration_seconds_wcst=250.0, totalTrialCount_wcst=73.0, completedCategories_wcst=6.0, totalErrorCount_wcst=11.0, perseverativeResponses_wcst=21.0
- Trial QC: prp_n_trials=120, prp_t2_timeout_n=5, prp_t2_anticipation_n=0, prp_t2_rt_median_valid_ms=1016.6499999761581, prp_t1_resp_unique=3, prp_t2_resp_unique=3, stroop_n_trials=108, stroop_timeout_n=4, stroop_anticipation_n=0, stroop_acc_pct_trials=91.66666666666666, stroop_rt_median_correct_ms=1431.600000023842, stroop_userColor_unique=5, wcst_n_trials=73, wcst_timeout_n=0, wcst_anticipation_n=0, wcst_acc_pct_trials=84.93150684931507, wcst_rt_median_ms=1968.900000095368, wcst_isPE_n=8, wcst_isNPE_n=3, wcst_isPR_n=21, wcst_chosenCard_unique=4
