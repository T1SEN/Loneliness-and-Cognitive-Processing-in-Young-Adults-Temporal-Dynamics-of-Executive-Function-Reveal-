#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-Error Slowing (PES) 성별 조절 분석

목표:
1. WCST, Stroop, PRP에서 post-error slowing 계산
2. UCLA × Gender 조절 효과 검증
3. 적응적 PES vs 비효율적 PES 구분 (PES + 정확도 개선)
4. 여성의 hypervigilance 증거 vs 남성의 비효율적 보상 증거

가설:
- 남성: UCLA↑ → PES↑ but post-error accuracy ↓ (더 늦지만 틀림)
- 여성: UCLA↑ → PES↑ and post-error accuracy ↑ (늦지만 정확)
"""

import sys
import os
from pathlib import Path
import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from analysis.utils.trial_data_loader import load_wcst_trials, load_stroop_trials, load_prp_trials
# UTF-8 설정
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# 경로 설정
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/post_error_slowing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("POST-ERROR SLOWING (PES) 성별 조절 분석")
print("="*80)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1] ?????...")
master_full = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
master_full = master_full.rename(columns={'gender_normalized': 'gender'})
master_full['gender'] = master_full['gender'].fillna('').astype(str).str.strip().str.lower()
participants = master_full[['participant_id','gender','age']].rename(columns={'participant_id': 'participantId'})

if 'ucla_total' not in master_full.columns and 'ucla_score' in master_full.columns:
    master_full['ucla_total'] = master_full['ucla_score']
ucla = master_full[['participant_id', 'ucla_total']].rename(columns={'participant_id': 'participantId'}).dropna(subset=['ucla_total'])

dass = pd.DataFrame({
    'participantId': master_full['participant_id'],
    'dass_total': master_full.get('dass_depression', 0) + master_full.get('dass_anxiety', 0) + master_full.get('dass_stress', 0)
}).dropna(subset=['dass_total'])

print(f"   Participants: {len(participants)}")
print(f"   UCLA: {len(ucla)}")

# ============================================================================
# 2. WCST PES 분석
# ============================================================================
print("\n[2] WCST Post-Error Slowing 분석...")

try:
    wcst_trials, wcst_summary = load_wcst_trials(use_cache=True)
    print(f"   WCST trials: {len(wcst_trials)} | participants: {wcst_summary.get('n_participants')}")
    wcst_trials = wcst_trials.rename(columns={'participant_id': 'participantId'})

    wcst_pes_list = []
    for pid in wcst_trials['participantId'].unique():
        trials = wcst_trials[wcst_trials['participantId'] == pid].copy()
        if 'trialIndex' in trials.columns:
            trials = trials.sort_values('trialIndex').reset_index(drop=True)
        else:
            trials = trials.reset_index(drop=True)

        if 'reactionTimeMs' in trials.columns:
            rt_col = 'reactionTimeMs'
        elif 'rt_ms' in trials.columns:
            rt_col = 'rt_ms'
        else:
            continue

        trials = trials[trials[rt_col] > 0]
        if len(trials) < 10:
            continue

        trials['prev_correct'] = trials['correct'].shift(1)
        post_error = trials[trials['prev_correct'] == False]
        post_correct = trials[trials['prev_correct'] == True]
        if len(post_error) < 3 or len(post_correct) < 3:
            continue

        rt_post_error = post_error[rt_col].mean()
        rt_post_correct = post_correct[rt_col].mean()
        pes = rt_post_error - rt_post_correct
        post_error_acc = post_error['correct'].mean() * 100
        post_correct_acc = post_correct['correct'].mean() * 100

        wcst_pes_list.append({
            'participantId': pid,
            'task': 'wcst',
            'pes_ms': pes,
            'rt_post_error': rt_post_error,
            'rt_post_correct': rt_post_correct,
            'post_error_acc': post_error_acc,
            'post_correct_acc': post_correct_acc,
            'acc_improvement': post_error_acc - post_correct_acc,
            'n_post_error': len(post_error),
            'n_post_correct': len(post_correct)
        })

    wcst_pes_df = pd.DataFrame(wcst_pes_list)
except Exception as e:
    print(f'WCST PES ??: {e}')
    wcst_pes_df = pd.DataFrame()
print("\n[3] Stroop Post-Error Slowing 분석...")

try:
    stroop_trials, stroop_summary = load_stroop_trials(use_cache=True, require_correct_for_rt=False)
    print(f"   Stroop trials: {len(stroop_trials)} | participants: {stroop_summary.get('n_participants')}")

    stroop_trials = stroop_trials.rename(columns={'participant_id': 'participantId'})
    stroop_pes_list = []

    for pid in stroop_trials['participantId'].unique():
        trials = stroop_trials[stroop_trials['participantId'] == pid].copy()

        if 'trial' in trials.columns:
            trials = trials.sort_values('trial').reset_index(drop=True)
        elif 'trialIndex' in trials.columns:
            trials = trials.sort_values('trialIndex').reset_index(drop=True)
        else:
            trials = trials.reset_index(drop=True)

        rt_col = 'rt' if 'rt' in trials.columns else 'rt_ms' if 'rt_ms' in trials.columns else None
        if rt_col is None:
            continue

        if 'timeout' in trials.columns:
            trials = trials[(trials['timeout'] == False)]

        if len(trials) < 10:
            continue

        trials['prev_correct'] = trials['correct'].shift(1)
        post_error = trials[trials['prev_correct'] == False]
        post_correct = trials[trials['prev_correct'] == True]

        if len(post_error) < 3 or len(post_correct) < 3:
            continue

        rt_post_error = post_error[rt_col].mean()
        rt_post_correct = post_correct[rt_col].mean()
        pes = rt_post_error - rt_post_correct
        post_error_acc = post_error['correct'].mean() * 100
        post_correct_acc = post_correct['correct'].mean() * 100

        stroop_pes_list.append({
            'participantId': pid,
            'task': 'stroop',
            'pes_ms': pes,
            'rt_post_error': rt_post_error,
            'rt_post_correct': rt_post_correct,
            'post_error_acc': post_error_acc,
            'post_correct_acc': post_correct_acc,
            'acc_improvement': post_error_acc - post_correct_acc,
            'n_post_error': len(post_error),
            'n_post_correct': len(post_correct)
        })

    stroop_pes_df = pd.DataFrame(stroop_pes_list)
    print(f"   Stroop PES computed: {len(stroop_pes_df)} rows")

except Exception as e:
    print(f"   !!Stroop PES failed: {e}")
    import traceback
    traceback.print_exc()
    stroop_pes_df = pd.DataFrame()
# ============================================================================
# 4. PRP PES 분석
# ============================================================================
print("\n[4] PRP Post-Error Slowing 분석...")

try:
    prp_trials, prp_summary = load_prp_trials(
        use_cache=True,
        require_t1_correct=False,
        enforce_short_long_only=False,
        require_t2_correct_for_rt=False,
        drop_timeouts=True,
    )
    print(f"   PRP trials: {len(prp_trials)} | participants: {prp_summary.get('n_participants')}")

    prp_trials = prp_trials.rename(columns={'participant_id': 'participantId'})

    prp_pes_list = []

    for pid in prp_trials['participantId'].unique():
        trials = prp_trials[prp_trials['participantId'] == pid].copy()

        if 'idx' in trials.columns:
            trials = trials.sort_values('idx').reset_index(drop=True)
        else:
            trials = trials.reset_index(drop=True)

        if 't1_timeout' in trials.columns:
            trials = trials[trials['t1_timeout'] == False]

        t1_rt_col = 't1_rt' if 't1_rt' in trials.columns else 't1_rt_ms' if 't1_rt_ms' in trials.columns else None
        if t1_rt_col is None:
            continue

        trials = trials[(trials[t1_rt_col] > 0) & (trials['t2_rt'] > 0)]

        if len(trials) < 10:
            continue

        trials['prev_t1_correct'] = trials['t1_correct'].shift(1)

        post_error_t1 = trials[trials['prev_t1_correct'] == False]
        post_correct_t1 = trials[trials['prev_t1_correct'] == True]

        if len(post_error_t1) < 3 or len(post_correct_t1) < 3:
            continue

        rt_post_error = post_error_t1[t1_rt_col].mean()
        rt_post_correct = post_correct_t1[t1_rt_col].mean()
        pes = rt_post_error - rt_post_correct
        post_error_acc = post_error_t1['t1_correct'].mean() * 100
        post_correct_acc = post_correct_t1['t1_correct'].mean() * 100

        prp_pes_list.append({
            'participantId': pid,
            'task': 'prp_t1',
            'pes_ms': pes,
            'rt_post_error': rt_post_error,
            'rt_post_correct': rt_post_correct,
            'post_error_acc': post_error_acc,
            'post_correct_acc': post_correct_acc,
            'acc_improvement': post_error_acc - post_correct_acc,
            'n_post_error': len(post_error_t1),
            'n_post_correct': len(post_correct_t1)
        })

    prp_pes_df = pd.DataFrame(prp_pes_list)
    print(f"   PRP PES computed: {len(prp_pes_df)} rows")

except Exception as e:
    print(f"   !!PRP PES failed: {e}")
    import traceback
    traceback.print_exc()
    prp_pes_df = pd.DataFrame()
# ============================================================================
# 5. 통합 및 Merge
# ============================================================================
print("\n[5] 데이터 통합...")

# 세 task 합치기
all_pes_list = []
if len(wcst_pes_df) > 0:
    all_pes_list.append(wcst_pes_df)
if len(stroop_pes_df) > 0:
    all_pes_list.append(stroop_pes_df)
if len(prp_pes_df) > 0:
    all_pes_list.append(prp_pes_df)

if len(all_pes_list) == 0:
    print("   ⚠ PES 데이터가 없습니다.")
    sys.exit(1)

all_pes = pd.concat(all_pes_list, ignore_index=True)

# UCLA, DASS, 성별 merge
all_pes = all_pes.merge(ucla, on='participantId', how='left')
all_pes = all_pes.merge(dass, on='participantId', how='left')
all_pes = all_pes.merge(participants[['participantId', 'gender', 'age']], on='participantId', how='left')

# 결측치 제거
all_pes = all_pes.dropna(subset=['ucla_total', 'gender'])

print(f"   통합 데이터: {len(all_pes)} rows")
print(f"   Tasks: {all_pes['task'].value_counts().to_dict()}")
print(f"   Gender: {all_pes['gender'].value_counts().to_dict()}")

# ============================================================================
# 6. Task별 성별 조절 효과 분석
# ============================================================================
print("\n[6] Task별 UCLA × Gender → PES 분석...")

results_list = []

for task in all_pes['task'].unique():
    task_data = all_pes[all_pes['task'] == task].copy()

    # 성별 dummy
    task_data['gender_male'] = (task_data['gender'] == '남성').astype(int)

    # 전체 상관
    if len(task_data) > 10:
        r_overall, p_overall = pearsonr(task_data['ucla_total'], task_data['pes_ms'])
    else:
        r_overall, p_overall = np.nan, np.nan

    # 성별별 상관
    males = task_data[task_data['gender'] == '남성']
    females = task_data[task_data['gender'] == '여성']

    if len(males) > 5:
        r_male, p_male = pearsonr(males['ucla_total'], males['pes_ms'])
        r_male_acc, p_male_acc = pearsonr(males['ucla_total'], males['post_error_acc'])
    else:
        r_male, p_male = np.nan, np.nan
        r_male_acc, p_male_acc = np.nan, np.nan

    if len(females) > 5:
        r_fem, p_fem = pearsonr(females['ucla_total'], females['pes_ms'])
        r_fem_acc, p_fem_acc = pearsonr(females['ucla_total'], females['post_error_acc'])
    else:
        r_fem, p_fem = np.nan, np.nan
        r_fem_acc, p_fem_acc = np.nan, np.nan

    # Interaction 회귀
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    task_data['ucla_z'] = scaler.fit_transform(task_data[['ucla_total']])
    task_data['gender_male_z'] = scaler.fit_transform(task_data[['gender_male']])
    task_data['interaction'] = task_data['ucla_z'] * task_data['gender_male_z']

    # OLS
    import statsmodels.api as sm
    X = task_data[['ucla_z', 'gender_male_z', 'interaction']]
    X = sm.add_constant(X)
    y = task_data['pes_ms']

    try:
        model = sm.OLS(y, X).fit()
        beta_interaction = model.params['interaction']
        p_interaction = model.pvalues['interaction']
    except:
        beta_interaction = np.nan
        p_interaction = np.nan

    print(f"\n   {task.upper()}")
    print(f"   Overall: r={r_overall:+.3f}, p={p_overall:.4f}")
    print(f"   Males (N={len(males)}): UCLA→PES r={r_male:+.3f}, p={p_male:.4f}")
    print(f"                           UCLA→post-error acc r={r_male_acc:+.3f}, p={p_male_acc:.4f}")
    print(f"   Females (N={len(females)}): UCLA→PES r={r_fem:+.3f}, p={p_fem:.4f}")
    print(f"                             UCLA→post-error acc r={r_fem_acc:+.3f}, p={p_fem_acc:.4f}")
    print(f"   Interaction: β={beta_interaction:+.3f}, p={p_interaction:.4f}")

    results_list.append({
        'task': task,
        'n_total': len(task_data),
        'n_males': len(males),
        'n_females': len(females),
        'r_overall': r_overall,
        'p_overall': p_overall,
        'r_male_pes': r_male,
        'p_male_pes': p_male,
        'r_male_acc': r_male_acc,
        'p_male_acc': p_male_acc,
        'r_fem_pes': r_fem,
        'p_fem_pes': p_fem,
        'r_fem_acc': r_fem_acc,
        'p_fem_acc': p_fem_acc,
        'beta_interaction': beta_interaction,
        'p_interaction': p_interaction
    })

results_df = pd.DataFrame(results_list)

# ============================================================================
# 7. 저장
# ============================================================================
print("\n[7] 결과 저장...")

# PES 데이터
output_file1 = OUTPUT_DIR / "pes_all_tasks.csv"
all_pes.to_csv(output_file1, index=False, encoding='utf-8-sig')
print(f"   저장: {output_file1}")

# 통계 결과
output_file2 = OUTPUT_DIR / "pes_gender_moderation_summary.csv"
results_df.to_csv(output_file2, index=False, encoding='utf-8-sig')
print(f"   저장: {output_file2}")

# ============================================================================
# 8. 시각화
# ============================================================================
print("\n[8] 시각화...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

idx = 0
for task in ['wcst', 'stroop', 'prp_t1']:
    task_data = all_pes[all_pes['task'] == task].copy()

    if len(task_data) < 10:
        idx += 2
        continue

    # PES scatter
    ax = axes[idx]
    males = task_data[task_data['gender'] == '남성']
    females = task_data[task_data['gender'] == '여성']

    ax.scatter(males['ucla_total'], males['pes_ms'], alpha=0.6, label='Males', c='blue', s=50)
    ax.scatter(females['ucla_total'], females['pes_ms'], alpha=0.6, label='Females', c='red', s=50)

    # Regression lines
    if len(males) > 5:
        z = np.polyfit(males['ucla_total'], males['pes_ms'], 1)
        p = np.poly1d(z)
        ax.plot(males['ucla_total'].sort_values(), p(males['ucla_total'].sort_values()),
                'b--', linewidth=2)

    if len(females) > 5:
        z = np.polyfit(females['ucla_total'], females['pes_ms'], 1)
        p = np.poly1d(z)
        ax.plot(females['ucla_total'].sort_values(), p(females['ucla_total'].sort_values()),
                'r--', linewidth=2)

    ax.set_xlabel('UCLA Loneliness', fontsize=12)
    ax.set_ylabel('PES (ms)', fontsize=12)
    ax.set_title(f'{task.upper()}: UCLA × Gender → PES', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    idx += 1

    # Post-error accuracy scatter
    ax = axes[idx]
    ax.scatter(males['ucla_total'], males['post_error_acc'], alpha=0.6, label='Males', c='blue', s=50)
    ax.scatter(females['ucla_total'], females['post_error_acc'], alpha=0.6, label='Females', c='red', s=50)

    if len(males) > 5:
        z = np.polyfit(males['ucla_total'], males['post_error_acc'], 1)
        p = np.poly1d(z)
        ax.plot(males['ucla_total'].sort_values(), p(males['ucla_total'].sort_values()),
                'b--', linewidth=2)

    if len(females) > 5:
        z = np.polyfit(females['ucla_total'], females['post_error_acc'], 1)
        p = np.poly1d(z)
        ax.plot(females['ucla_total'].sort_values(), p(females['ucla_total'].sort_values()),
                'r--', linewidth=2)

    ax.set_xlabel('UCLA Loneliness', fontsize=12)
    ax.set_ylabel('Post-Error Accuracy (%)', fontsize=12)
    ax.set_title(f'{task.upper()}: UCLA × Gender → Post-Error Acc', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    idx += 1

plt.tight_layout()
output_fig = OUTPUT_DIR / "pes_gender_moderation_plots.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"   저장: {output_fig}")
plt.close()

# ============================================================================
# 9. 보고서
# ============================================================================
report_file = OUTPUT_DIR / "PES_GENDER_MODERATION_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("POST-ERROR SLOWING (PES) 성별 조절 분석 보고서\n")
    f.write("="*80 + "\n\n")

    f.write(f"분석 일자: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("1. 분석 개요\n")
    f.write("   Post-Error Slowing (PES)은 오류 후 RT가 느려지는 현상으로,\n")
    f.write("   적응적 인지 조절의 지표입니다.\n\n")

    f.write("   가설:\n")
    f.write("   - 남성: UCLA↑ → PES↑ but 정확도 개선 X (비효율적 보상)\n")
    f.write("   - 여성: UCLA↑ → PES↑ and 정확도 개선 ↑ (적응적 조절)\n\n")

    f.write("2. Task별 결과\n\n")
    for _, row in results_df.iterrows():
        f.write(f"   [{row['task'].upper()}] (N={row['n_total']:.0f})\n")
        f.write(f"   Overall: r={row['r_overall']:+.3f}, p={row['p_overall']:.4f}\n")
        f.write(f"   Males (N={row['n_males']:.0f}):\n")
        f.write(f"     UCLA → PES: r={row['r_male_pes']:+.3f}, p={row['p_male_pes']:.4f}\n")
        f.write(f"     UCLA → Post-error Acc: r={row['r_male_acc']:+.3f}, p={row['p_male_acc']:.4f}\n")
        f.write(f"   Females (N={row['n_females']:.0f}):\n")
        f.write(f"     UCLA → PES: r={row['r_fem_pes']:+.3f}, p={row['p_fem_pes']:.4f}\n")
        f.write(f"     UCLA → Post-error Acc: r={row['r_fem_acc']:+.3f}, p={row['p_fem_acc']:.4f}\n")
        f.write(f"   Interaction: β={row['beta_interaction']:+.3f}, p={row['p_interaction']:.4f}\n\n")

    f.write("="*80 + "\n")

print(f"\n[9] 보고서 저장: {report_file}")

print("\n" + "="*80)
print("PES 성별 조절 분석 완료!")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print("="*80)
