#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WCST 메커니즘 종합 분석 (5-in-1)

통합 분석:
1. 피드백 민감도 (부정/긍정 피드백 후 적응)
2. 불확실성 내성 (rule change 직후 10 trials)
3. Hypervigilance 종합 지표 구축 (여성 보상 경로)
4. 보상의 장기 비용 (Hypervigilance × DASS)
5. 보상 실패 조건 탐색 (고불안/고스트레스에서 보상 붕괴)

목표: WCST 특이성 및 여성 보상 메커니즘 규명
"""

import sys
import os
from pathlib import Path
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from analysis.utils.trial_data_loader import load_wcst_trials

# UTF-8 설정
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# 경로 설정
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/wcst_mechanism_comprehensive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WCST 메커니즘 종합 분석 (5-in-1)")
print("="*80)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[데이터 로딩]")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]
master = master.rename(columns={"gender_normalized": "gender"})
master["gender"] = master["gender"].fillna("").astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

participants = master[['participant_id','gender','age']]
ucla = master[['participant_id', 'ucla_total']].dropna()
dass_df = master[['participant_id', 'dass_anxiety', 'dass_stress', 'dass_depression', 'dass_total']].dropna()

wcst_trials, _ = load_wcst_trials(use_cache=True)
if 'participantId' in wcst_trials.columns and 'participant_id' not in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

# UCLA & DASS
ucla = surveys[surveys['surveyName'] == 'ucla'][['participantId', 'score']].dropna()
ucla.columns = ['participantId', 'ucla_total']

dass = surveys[surveys['surveyName'] == 'dass'].copy()
# DASS subscales 추출 (A, S, D)
dass_scores = []
for _, row in dass.iterrows():
    pid = row['participantId']
    # score_A (anxiety), score_S (stress), score_D (depression)
    anxiety = row.get('score_A', np.nan)
    stress = row.get('score_S', np.nan)
    depression = row.get('score_D', np.nan)
    total = row.get('score', np.nan)

    # total이 NaN이면 합계 계산
    if pd.isna(total) and not pd.isna(anxiety) and not pd.isna(stress) and not pd.isna(depression):
        total = anxiety + stress + depression

    dass_scores.append({
        'participantId': pid,
        'dass_anxiety': anxiety,
        'dass_stress': stress,
        'dass_depression': depression,
        'dass_total': total
    })

dass_df = pd.DataFrame(dass_scores).dropna(subset=['dass_anxiety', 'dass_stress', 'dass_depression'])

print(f"   Participants: {len(participants)}")
print(f"   UCLA: {len(ucla)}")
print(f"   DASS: {len(dass_df)}")
print(f"   WCST trials: {len(wcst_trials)}")

# participantId normalize
if 'participant_id' in wcst_trials.columns and 'participantId' not in wcst_trials.columns:
    wcst_trials.rename(columns={'participant_id': 'participantId'}, inplace=True)

# ============================================================================
# 분석 1: 피드백 민감도
# ============================================================================
print("\n[1] 피드백 민감도 분석...")

feedback_sensitivity_list = []

for pid in wcst_trials['participantId'].unique():
    trials = wcst_trials[wcst_trials['participantId'] == pid].copy()
    trials = trials.sort_values('trialIndex').reset_index(drop=True)

    if len(trials) < 20:
        continue

    # RT 컬럼
    rt_col = 'reactionTimeMs' if 'reactionTimeMs' in trials.columns else 'rt_ms'
    trials = trials[trials[rt_col] > 0]

    # 이전 trial 피드백
    trials['prev_correct'] = trials['correct'].shift(1)

    # 부정 피드백 후 (previous trial incorrect)
    post_neg = trials[trials['prev_correct'] == False]
    # 긍정 피드백 후 (previous trial correct)
    post_pos = trials[trials['prev_correct'] == True]

    if len(post_neg) < 3 or len(post_pos) < 3:
        continue

    # RT 변화
    rt_post_neg = post_neg[rt_col].mean()
    rt_post_pos = post_pos[rt_col].mean()
    rt_change = rt_post_neg - rt_post_pos  # positive = slowing after error

    # 정확도 변화
    acc_post_neg = post_neg['correct'].mean() * 100
    acc_post_pos = post_pos['correct'].mean() * 100
    acc_change = acc_post_neg - acc_post_pos  # positive = improvement after error

    # 피드백 민감도 = RT 변화 + 정확도 변화 (적응적 조절의 종합 지표)
    # Adaptive: RT↑ (slowing) + Acc↑ (improvement)
    # Maladaptive: RT↑ but Acc↓ or flat

    feedback_sensitivity_list.append({
        'participantId': pid,
        'rt_post_negative_feedback': rt_post_neg,
        'rt_post_positive_feedback': rt_post_pos,
        'rt_feedback_sensitivity': rt_change,
        'acc_post_negative_feedback': acc_post_neg,
        'acc_post_positive_feedback': acc_post_pos,
        'acc_feedback_sensitivity': acc_change,
        'adaptive_index': acc_change - (rt_change / 100)  # 정확도 개선 - RT 비용
    })

feedback_df = pd.DataFrame(feedback_sensitivity_list)
print(f"   피드백 민감도 계산 완료: {len(feedback_df)}명")

# ============================================================================
# 분석 2: 불확실성 내성 (Rule Ambiguity)
# ============================================================================
print("\n[2] 불확실성 내성 분석 (rule change 직후)...")

# Rule change 탐지 (간단한 방법: 연속 오류 후 연속 정답 → 새 규칙 학습)
uncertainty_list = []

for pid in wcst_trials['participantId'].unique():
    trials = wcst_trials[wcst_trials['participantId'] == pid].copy()
    trials = trials.sort_values('trialIndex').reset_index(drop=True)

    if len(trials) < 30:
        continue

    rt_col = 'reactionTimeMs' if 'reactionTimeMs' in trials.columns else 'rt_ms'
    trials = trials[trials[rt_col] > 0]

    # Rule change 후보: 오류 3+ 연속 후 정답 2+ 연속
    trials['prev_correct'] = trials['correct'].shift(1)
    trials['prev2_correct'] = trials['correct'].shift(2)
    trials['prev3_correct'] = trials['correct'].shift(3)

    # Potential rule change: prev3 error, prev2 error, prev1 error → current correct
    trials['potential_rule_change'] = (
        (trials['prev3_correct'] == False) &
        (trials['prev2_correct'] == False) &
        (trials['prev_correct'] == False) &
        (trials['correct'] == True)
    )

    rule_change_indices = trials[trials['potential_rule_change']].index.tolist()

    if len(rule_change_indices) < 2:
        continue

    # Rule change 직후 10 trials의 성과
    post_rule_change_rts = []
    post_rule_change_accs = []

    for idx in rule_change_indices:
        # 직후 10 trials (충분한 trials가 있는 경우만)
        if idx + 10 < len(trials):
            post_trials = trials.iloc[idx:idx+10]
            post_rule_change_rts.extend(post_trials[rt_col].tolist())
            post_rule_change_accs.extend(post_trials['correct'].tolist())

    if len(post_rule_change_rts) < 10:
        continue

    # 안정 구간 (rule change 없는 중간 구간)
    stable_trials = trials[(trials.index > 20) & (trials.index < len(trials) - 20)]
    stable_trials = stable_trials[~stable_trials['potential_rule_change']]

    if len(stable_trials) < 10:
        continue

    # 불확실성 tolerance = (rule change 직후 RT - stable RT) / stable RT
    rt_uncertainty = np.mean(post_rule_change_rts)
    rt_stable = stable_trials[rt_col].mean()
    rt_uncertainty_cost = (rt_uncertainty - rt_stable) / rt_stable * 100  # % increase

    acc_uncertainty = np.mean(post_rule_change_accs) * 100
    acc_stable = stable_trials['correct'].mean() * 100
    acc_uncertainty_cost = acc_stable - acc_uncertainty  # % decrease

    uncertainty_list.append({
        'participantId': pid,
        'rt_uncertainty': rt_uncertainty,
        'rt_stable': rt_stable,
        'rt_uncertainty_cost_pct': rt_uncertainty_cost,
        'acc_uncertainty': acc_uncertainty,
        'acc_stable': acc_stable,
        'acc_uncertainty_cost_pct': acc_uncertainty_cost,
        'n_rule_changes_detected': len(rule_change_indices)
    })

uncertainty_df = pd.DataFrame(uncertainty_list)
print(f"   불확실성 내성 계산 완료: {len(uncertainty_df)}명")

# ============================================================================
# 분석 3: Hypervigilance 종합 지표
# ============================================================================
print("\n[3] Hypervigilance 종합 지표 구축...")

# 기존 분석에서 확인된 여성 보상 지표들:
# - PES ↓ (빠른 회복)
# - Post-error accuracy ↑
# - Error cascades ↓
# - Post-switch errors ↓

# 이미 계산한 PES 데이터 로드
try:
    pes_data = pd.read_csv(OUTPUT_DIR.parent / "post_error_slowing" / "pes_all_tasks.csv", encoding='utf-8-sig')
    pes_wcst = pes_data[pes_data['task'] == 'wcst'][['participantId', 'pes_ms', 'post_error_acc']].copy()
    print(f"   PES 데이터 로드: {len(pes_wcst)}명")
except:
    print(f"   ⚠ PES 데이터 없음, 계산 스킵")
    pes_wcst = pd.DataFrame()

# Error cascades 계산 (연속 오류 비율)
error_cascade_list = []

for pid in wcst_trials['participantId'].unique():
    trials = wcst_trials[wcst_trials['participantId'] == pid].copy()
    trials = trials.sort_values('trialIndex').reset_index(drop=True)

    if len(trials) < 20:
        continue

    # 연속 오류 탐지
    trials['prev_correct'] = trials['correct'].shift(1)
    trials['error_cascade'] = (trials['correct'] == False) & (trials['prev_correct'] == False)

    n_cascades = trials['error_cascade'].sum()
    n_errors = (trials['correct'] == False).sum()

    if n_errors > 0:
        cascade_rate = n_cascades / n_errors * 100
    else:
        cascade_rate = 0

    error_cascade_list.append({
        'participantId': pid,
        'error_cascade_rate': cascade_rate,
        'n_errors': n_errors,
        'n_cascades': n_cascades
    })

cascade_df = pd.DataFrame(error_cascade_list)
print(f"   Error cascade 계산 완료: {len(cascade_df)}명")

# Hypervigilance composite score 계산
# 높을수록 hypervigilance (z-score 기반)
from sklearn.preprocessing import StandardScaler

# Merge all components
hyper_df = feedback_df[['participantId', 'acc_feedback_sensitivity', 'adaptive_index']].copy()
hyper_df = hyper_df.merge(cascade_df[['participantId', 'error_cascade_rate']], on='participantId', how='outer')
if len(pes_wcst) > 0:
    hyper_df = hyper_df.merge(pes_wcst[['participantId', 'post_error_acc']], on='participantId', how='outer')

# Drop NaN
hyper_df = hyper_df.dropna()

if len(hyper_df) > 10:
    scaler = StandardScaler()

    # Hypervigilance components (standardized):
    # 1. High post-error accuracy (positive)
    # 2. Low error cascades (negative → reverse)
    # 3. High adaptive feedback response (positive)

    hyper_df['post_error_acc_z'] = scaler.fit_transform(hyper_df[['post_error_acc']]) if 'post_error_acc' in hyper_df.columns else 0
    hyper_df['error_cascade_z'] = -scaler.fit_transform(hyper_df[['error_cascade_rate']])  # reverse
    hyper_df['adaptive_index_z'] = scaler.fit_transform(hyper_df[['adaptive_index']])

    # Composite
    hyper_df['hypervigilance_score'] = (
        hyper_df['post_error_acc_z'] +
        hyper_df['error_cascade_z'] +
        hyper_df['adaptive_index_z']
    ) / 3

    print(f"   Hypervigilance 종합 점수 계산 완료: {len(hyper_df)}명")
    print(f"   Score 범위: {hyper_df['hypervigilance_score'].min():.2f} ~ {hyper_df['hypervigilance_score'].max():.2f}")
else:
    print(f"   ⚠ Hypervigilance 계산 실패 (N={len(hyper_df)} < 10)")

# ============================================================================
# 데이터 통합
# ============================================================================
print("\n[데이터 통합]")

# Master dataframe
master = feedback_df.copy()
master = master.merge(uncertainty_df, on='participantId', how='outer')
master = master.merge(cascade_df, on='participantId', how='outer')
if len(pes_wcst) > 0:
    master = master.merge(pes_wcst, on='participantId', how='outer')
if len(hyper_df) > 0:
    master = master.merge(hyper_df[['participantId', 'hypervigilance_score']], on='participantId', how='outer')

# Merge 인구통계
master = master.merge(ucla, on='participantId', how='left')
master = master.merge(dass_df, on='participantId', how='left')

# Drop missing
master = master.dropna(subset=['gender', 'ucla_total'])

print(f"   통합 데이터: {len(master)}명")
print(f"   Gender: {master['gender'].value_counts().to_dict()}")

# ============================================================================
# 분석 4: 보상의 장기 비용 (Hypervigilance × DASS)
# ============================================================================
print("\n[4] 보상의 장기 비용 분석 (Hypervigilance × DASS)...")

if 'hypervigilance_score' in master.columns and len(master) > 20:
    # 여성만
    females = master[master['gender'] == '여성'].copy()

    if len(females) > 10:
        # Hypervigilance vs DASS 상관
        r_anxiety, p_anxiety = pearsonr(females['hypervigilance_score'], females['dass_anxiety'])
        r_stress, p_stress = pearsonr(females['hypervigilance_score'], females['dass_stress'])
        r_depression, p_depression = pearsonr(females['hypervigilance_score'], females['dass_depression'])
        r_total, p_total = pearsonr(females['hypervigilance_score'], females['dass_total'])

        print(f"\n   여성 (N={len(females)}) Hypervigilance × DASS:")
        print(f"   - Anxiety: r={r_anxiety:+.3f}, p={p_anxiety:.4f}")
        print(f"   - Stress: r={r_stress:+.3f}, p={p_stress:.4f}")
        print(f"   - Depression: r={r_depression:+.3f}, p={p_depression:.4f}")
        print(f"   - Total: r={r_total:+.3f}, p={p_total:.4f}")

        # 비용 해석: Hypervigilance↑ → DASS↑이면 "비용 존재"
        if r_total > 0.2:
            print(f"   → 보상의 비용 존재 가능성 (hypervigilance ↑ → DASS ↑)")
        elif r_total < -0.2:
            print(f"   → 보상이 진정한 보호 효과 (hypervigilance ↑ → DASS ↓)")
        else:
            print(f"   → 무관 (독립적 경로)")

        # 저장
        cost_df = pd.DataFrame([{
            'gender': '여성',
            'n': len(females),
            'r_anxiety': r_anxiety,
            'p_anxiety': p_anxiety,
            'r_stress': r_stress,
            'p_stress': p_stress,
            'r_depression': r_depression,
            'p_depression': p_depression,
            'r_total': r_total,
            'p_total': p_total
        }])

        cost_df.to_csv(OUTPUT_DIR / "hypervigilance_cost_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"   저장: hypervigilance_cost_analysis.csv")
else:
    print(f"   ⚠ Hypervigilance 점수 없음, 스킵")

# ============================================================================
# 분석 5: 보상 실패 조건 (DASS 층화)
# ============================================================================
print("\n[5] 보상 실패 조건 탐색 (DASS 층화)...")

females = master[master['gender'] == '여성'].copy()

if len(females) > 20 and 'hypervigilance_score' in females.columns:
    # DASS median split
    stress_median = females['dass_stress'].median()
    anxiety_median = females['dass_anxiety'].median()

    # Low vs High DASS
    low_stress_fem = females[females['dass_stress'] <= stress_median]
    high_stress_fem = females[females['dass_stress'] > stress_median]

    low_anxiety_fem = females[females['dass_anxiety'] <= anxiety_median]
    high_anxiety_fem = females[females['dass_anxiety'] > anxiety_median]

    breakdown_results = []

    # Stress 층화
    if len(low_stress_fem) > 5:
        r_low, p_low = pearsonr(low_stress_fem['ucla_total'], low_stress_fem['hypervigilance_score'])
        breakdown_results.append({
            'stratification': 'Stress',
            'group': 'Low Stress',
            'n': len(low_stress_fem),
            'r': r_low,
            'p': p_low
        })

    if len(high_stress_fem) > 5:
        r_high, p_high = pearsonr(high_stress_fem['ucla_total'], high_stress_fem['hypervigilance_score'])
        breakdown_results.append({
            'stratification': 'Stress',
            'group': 'High Stress',
            'n': len(high_stress_fem),
            'r': r_high,
            'p': p_high
        })

    # Anxiety 층화
    if len(low_anxiety_fem) > 5:
        r_low, p_low = pearsonr(low_anxiety_fem['ucla_total'], low_anxiety_fem['hypervigilance_score'])
        breakdown_results.append({
            'stratification': 'Anxiety',
            'group': 'Low Anxiety',
            'n': len(low_anxiety_fem),
            'r': r_low,
            'p': p_low
        })

    if len(high_anxiety_fem) > 5:
        r_high, p_high = pearsonr(high_anxiety_fem['ucla_total'], high_anxiety_fem['hypervigilance_score'])
        breakdown_results.append({
            'stratification': 'Anxiety',
            'group': 'High Anxiety',
            'n': len(high_anxiety_fem),
            'r': r_high,
            'p': p_high
        })

    breakdown_df = pd.DataFrame(breakdown_results)

    if len(breakdown_df) > 0:
        print(f"\n   여성 보상 메커니즘의 DASS 층화 분석:")
        for _, row in breakdown_df.iterrows():
            print(f"   {row['stratification']} - {row['group']} (N={row['n']:.0f}): r={row['r']:+.3f}, p={row['p']:.4f}")

        breakdown_df.to_csv(OUTPUT_DIR / "compensation_breakdown_conditions.csv", index=False, encoding='utf-8-sig')
        print(f"   저장: compensation_breakdown_conditions.csv")
else:
    print(f"   ⚠ 여성 데이터 부족 (N={len(females)}), 스킵")

# ============================================================================
# 저장
# ============================================================================
print("\n[저장]")

output_file = OUTPUT_DIR / "wcst_mechanism_master.csv"
master.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   Master 데이터: {output_file}")

# ============================================================================
# 시각화
# ============================================================================
print("\n[시각화]")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 피드백 민감도 × UCLA
ax = axes[0, 0]
if len(master) > 10:
    males = master[master['gender'] == '남성']
    females = master[master['gender'] == '여성']

    ax.scatter(males['ucla_total'], males['acc_feedback_sensitivity'],
               alpha=0.6, c='blue', label='Males', s=50)
    ax.scatter(females['ucla_total'], females['acc_feedback_sensitivity'],
               alpha=0.6, c='red', label='Females', s=50)

    ax.set_xlabel('UCLA Loneliness', fontsize=12)
    ax.set_ylabel('Accuracy Feedback Sensitivity (%)', fontsize=12)
    ax.set_title('Feedback Sensitivity (Acc Improvement)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# 2. 불확실성 비용 × UCLA
ax = axes[0, 1]
if len(master) > 10:
    ax.scatter(males['ucla_total'], males['acc_uncertainty_cost_pct'],
               alpha=0.6, c='blue', label='Males', s=50)
    ax.scatter(females['ucla_total'], females['acc_uncertainty_cost_pct'],
               alpha=0.6, c='red', label='Females', s=50)

    ax.set_xlabel('UCLA Loneliness', fontsize=12)
    ax.set_ylabel('Uncertainty Cost (% Acc Drop)', fontsize=12)
    ax.set_title('Uncertainty Tolerance', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# 3. Error cascade × UCLA
ax = axes[0, 2]
if len(master) > 10:
    ax.scatter(males['ucla_total'], males['error_cascade_rate'],
               alpha=0.6, c='blue', label='Males', s=50)
    ax.scatter(females['ucla_total'], females['error_cascade_rate'],
               alpha=0.6, c='red', label='Females', s=50)

    ax.set_xlabel('UCLA Loneliness', fontsize=12)
    ax.set_ylabel('Error Cascade Rate (%)', fontsize=12)
    ax.set_title('Error Cascades', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# 4. Hypervigilance × UCLA
ax = axes[1, 0]
if 'hypervigilance_score' in master.columns and len(master) > 10:
    ax.scatter(males['ucla_total'], males['hypervigilance_score'],
               alpha=0.6, c='blue', label='Males', s=50)
    ax.scatter(females['ucla_total'], females['hypervigilance_score'],
               alpha=0.6, c='red', label='Females', s=50)

    # Regression lines
    if len(females.dropna(subset=['hypervigilance_score'])) > 5:
        z = np.polyfit(females['ucla_total'], females['hypervigilance_score'], 1)
        p = np.poly1d(z)
        ax.plot(females['ucla_total'].sort_values(), p(females['ucla_total'].sort_values()),
                'r--', linewidth=2, label=f'Female fit')

    ax.set_xlabel('UCLA Loneliness', fontsize=12)
    ax.set_ylabel('Hypervigilance Score (z)', fontsize=12)
    ax.set_title('Hypervigilance Composite', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# 5. Hypervigilance × DASS (여성만)
ax = axes[1, 1]
if 'hypervigilance_score' in females.columns and len(females) > 10:
    # Drop NaN for both variables
    plot_data = females[['hypervigilance_score', 'dass_total']].dropna()

    if len(plot_data) > 5:
        ax.scatter(plot_data['hypervigilance_score'], plot_data['dass_total'],
                   alpha=0.6, c='red', s=50)

        z = np.polyfit(plot_data['hypervigilance_score'], plot_data['dass_total'], 1)
        p = np.poly1d(z)
        ax.plot(plot_data['hypervigilance_score'].sort_values(),
                p(plot_data['hypervigilance_score'].sort_values()),
                'r--', linewidth=2)

    ax.set_xlabel('Hypervigilance Score', fontsize=12)
    ax.set_ylabel('DASS Total', fontsize=12)
    ax.set_title('Hypervigilance Cost (Females Only)', fontweight='bold')
    ax.grid(alpha=0.3)

# 6. 보상 breakdown (DASS 층화)
ax = axes[1, 2]
if len(breakdown_df) > 0:
    breakdown_df['label'] = breakdown_df['group']
    breakdown_df['color'] = breakdown_df['stratification'].map({'Stress': 'green', 'Anxiety': 'orange'})

    for i, row in breakdown_df.iterrows():
        ax.barh(i, row['r'], color=row['color'], alpha=0.7)
        ax.text(row['r'], i, f" p={row['p']:.3f}", va='center', fontsize=9)

    ax.set_yticks(range(len(breakdown_df)))
    ax.set_yticklabels(breakdown_df['label'])
    ax.set_xlabel('Correlation (UCLA → Hypervigilance)', fontsize=12)
    ax.set_title('Compensation Breakdown (Females)', fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
output_fig = OUTPUT_DIR / "wcst_mechanism_comprehensive.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"   저장: {output_fig}")
plt.close()

# ============================================================================
# 보고서
# ============================================================================
report_file = OUTPUT_DIR / "WCST_MECHANISM_COMPREHENSIVE_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("WCST 메커니즘 종합 분석 보고서 (5-in-1)\n")
    f.write("="*80 + "\n\n")

    f.write(f"분석 일자: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"총 참여자: {len(master)}명\n")
    f.write(f"남성: {len(master[master['gender'] == '남성'])}명\n")
    f.write(f"여성: {len(master[master['gender'] == '여성'])}명\n\n")

    f.write("="*80 + "\n")
    f.write("분석 1: 피드백 민감도\n")
    f.write("="*80 + "\n")
    f.write("목표: 부정/긍정 피드백 후 적응 능력 측정\n")
    f.write(f"측정: RT 변화, 정확도 변화, 적응 지수\n")
    f.write(f"완료: {len(feedback_df)}명\n\n")

    f.write("="*80 + "\n")
    f.write("분석 2: 불확실성 내성\n")
    f.write("="*80 + "\n")
    f.write("목표: Rule change 직후 10 trials의 성과\n")
    f.write(f"측정: RT/정확도 비용 (%)\n")
    f.write(f"완료: {len(uncertainty_df)}명\n\n")

    f.write("="*80 + "\n")
    f.write("분석 3: Hypervigilance 종합 지표\n")
    f.write("="*80 + "\n")
    f.write("구성 요소:\n")
    f.write("  - Post-error accuracy ↑\n")
    f.write("  - Error cascades ↓\n")
    f.write("  - Adaptive feedback response ↑\n")
    if 'hypervigilance_score' in master.columns:
        f.write(f"완료: {len(master.dropna(subset=['hypervigilance_score']))}명\n")
        f.write(f"Score 범위: {master['hypervigilance_score'].min():.2f} ~ {master['hypervigilance_score'].max():.2f}\n\n")

    f.write("="*80 + "\n")
    f.write("분석 4: 보상의 장기 비용 (Hypervigilance × DASS)\n")
    f.write("="*80 + "\n")
    if os.path.exists(OUTPUT_DIR / "hypervigilance_cost_analysis.csv"):
        cost_data = pd.read_csv(OUTPUT_DIR / "hypervigilance_cost_analysis.csv", encoding='utf-8-sig')
        for _, row in cost_data.iterrows():
            f.write(f"여성 (N={row['n']:.0f}):\n")
            f.write(f"  Hypervigilance × DASS Anxiety: r={row['r_anxiety']:+.3f}, p={row['p_anxiety']:.4f}\n")
            f.write(f"  Hypervigilance × DASS Stress: r={row['r_stress']:+.3f}, p={row['p_stress']:.4f}\n")
            f.write(f"  Hypervigilance × DASS Total: r={row['r_total']:+.3f}, p={row['p_total']:.4f}\n\n")

    f.write("="*80 + "\n")
    f.write("분석 5: 보상 실패 조건 (DASS 층화)\n")
    f.write("="*80 + "\n")
    if len(breakdown_df) > 0:
        for _, row in breakdown_df.iterrows():
            f.write(f"{row['stratification']} - {row['group']} (N={row['n']:.0f}): ")
            f.write(f"UCLA → Hypervigilance: r={row['r']:+.3f}, p={row['p']:.4f}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("산출물:\n")
    f.write("  - wcst_mechanism_master.csv: 통합 데이터\n")
    f.write("  - hypervigilance_cost_analysis.csv: 비용 분석\n")
    f.write("  - compensation_breakdown_conditions.csv: 보상 붕괴 조건\n")
    f.write("  - wcst_mechanism_comprehensive.png: 종합 시각화\n")
    f.write("="*80 + "\n")

print(f"\n보고서 저장: {report_file}")

print("\n" + "="*80)
print("WCST 메커니즘 종합 분석 완료!")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print("="*80)
