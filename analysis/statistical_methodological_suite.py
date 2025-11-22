#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통계 & 방법론 강화 분석 (4-in-1)

통합 분석:
1. Task Difficulty 비교 (WCST/Stroop/PRP)
2. 비선형 UCLA 효과 검증 (quadratic, threshold)
3. 극단 집단 비교 (상위/하위 10%)
4. WCST PE 신뢰도 분석 (split-half)
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
from sklearn.metrics import roc_curve, auc
import warnings
from analysis.utils.trial_data_loader import load_wcst_trials
warnings.filterwarnings('ignore')

# UTF-8 설정
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# 경로 설정
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/statistical_suite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("통계 & 방법론 강화 분석 (4-in-1)")
print("="*80)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[?????]")
master = load_master_dataset(use_cache=True, force_rebuild=True, merge_cognitive_summary=True)
master = master.rename(columns={"gender_normalized": "gender"})
master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
master['gender'] = master['gender'].apply(lambda x: 'male' if x == 'male' else ('female' if x == 'female' else None))

if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
if 'bottleneck' not in master.columns and 'prp_bottleneck' in master.columns:
    master['bottleneck'] = master['prp_bottleneck']
if 'interference' not in master.columns:
    if 'stroop_interference' in master.columns:
        master['interference'] = master['stroop_interference']
    elif {'rt_mean_incongruent', 'rt_mean_congruent'} <= set(master.columns):
        master['interference'] = master['rt_mean_incongruent'] - master['rt_mean_congruent']

if {'accuracy_incongruent', 'accuracy_congruent'} <= set(master.columns):
    master['accuracy_stroop'] = master[['accuracy_incongruent', 'accuracy_congruent']].mean(axis=1)
elif 'accuracy' in master.columns:
    master['accuracy_stroop'] = master['accuracy']

master = master.dropna(subset=['gender', 'ucla_total', 'pe_rate'])

print(f"   Total: {len(master)}?")
print(f"   Gender: {master['gender'].value_counts().to_dict()}")

# ============================================================================
# 분석 1: Task Difficulty 비교
# ============================================================================
print("\n[1] Task Difficulty 비교 분석...")

# Task별 정확도, RT, 변동성 계산
task_difficulty = []

# WCST
wcst_acc = master['wcst_accuracy'].mean() if 'wcst_accuracy' in master.columns else np.nan
wcst_sd = master['wcst_accuracy'].std() if 'wcst_accuracy' in master.columns else np.nan
task_difficulty.append({
    'task': 'WCST',
    'mean_accuracy': wcst_acc,
    'sd_accuracy': wcst_sd,
    'ceiling_effect': 'Yes' if wcst_acc > 95 else 'No',
    'n': master['wcst_accuracy'].notna().sum() if 'wcst_accuracy' in master.columns else 0
})

# Stroop
if 'accuracy_stroop' in master.columns:
    stroop_acc = master['accuracy_stroop'].mean()
    stroop_sd = master['accuracy_stroop'].std()
    task_difficulty.append({
        'task': 'Stroop',
        'mean_accuracy': stroop_acc,
        'sd_accuracy': stroop_sd,
        'ceiling_effect': 'Yes' if stroop_acc > 95 else 'No',
        'n': len(master.dropna(subset=['accuracy_stroop']))
    })

# Task difficulty vs Effect size 관계
# 가설: 너무 쉬운 task (ceiling)나 너무 어려운 task (floor)는 효과 작음

males = master[master['gender'] == 'male']
females = master[master['gender'] == 'female']

effect_sizes = []

if len(males) > 10 and len(females) > 10:
    # WCST PE
    r_m_pe, p_m_pe = pearsonr(males['ucla_total'], males['pe_rate'])
    r_f_pe, p_f_pe = pearsonr(females['ucla_total'], females['pe_rate'])

    effect_sizes.append({
        'task': 'WCST (PE rate)',
        'mean_accuracy': wcst_acc,
        'r_male': r_m_pe,
        'p_male': p_m_pe,
        'r_female': r_f_pe,
        'p_female': p_f_pe,
        'gender_diff': abs(r_m_pe - r_f_pe)
    })

    # Stroop
    if 'interference' in master.columns:
        males_s = males.dropna(subset=['interference'])
        females_s = females.dropna(subset=['interference'])

        if len(males_s) > 10 and len(females_s) > 10:
            r_m_s, p_m_s = pearsonr(males_s['ucla_total'], males_s['interference'])
            r_f_s, p_f_s = pearsonr(females_s['ucla_total'], females_s['interference'])

            effect_sizes.append({
                'task': 'Stroop (Interference)',
                'mean_accuracy': stroop_acc if 'accuracy_stroop' in master.columns else np.nan,
                'r_male': r_m_s,
                'p_male': p_m_s,
                'r_female': r_f_s,
                'p_female': p_f_s,
                'gender_diff': abs(r_m_s - r_f_s)
            })

task_diff_df = pd.DataFrame(task_difficulty)
effect_size_df = pd.DataFrame(effect_sizes)

print(f"\n   Task Difficulty:")
for _, row in task_diff_df.iterrows():
    print(f"   {row['task']}: Acc={row['mean_accuracy']:.1f}% (SD={row['sd_accuracy']:.1f}%), Ceiling={row['ceiling_effect']}")

print(f"\n   Effect Size vs Difficulty:")
for _, row in effect_size_df.iterrows():
    print(f"   {row['task']} (Acc={row['mean_accuracy']:.1f}%): Male r={row['r_male']:+.3f}, Female r={row['r_female']:+.3f}")

# 저장
task_diff_df.to_csv(OUTPUT_DIR / "task_difficulty_summary.csv", index=False, encoding='utf-8-sig')
effect_size_df.to_csv(OUTPUT_DIR / "task_difficulty_effect_sizes.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# 분석 2: 비선형 UCLA 효과
# ============================================================================
print("\n[2] 비선형 UCLA 효과 검증...")

# 남성만 (주요 효과가 남성에서 발견되므로)
males = master[master['gender'] == 'male']

if len(males) > 20:
    # Quadratic model
    import statsmodels.api as sm

    males['ucla_sq'] = males['ucla_total'] ** 2

    # OLS: PE ~ UCLA + UCLA^2
    X = males[['ucla_total', 'ucla_sq']]
    X = sm.add_constant(X)
    y = males['pe_rate']

    model_quad = sm.OLS(y, X).fit()

    beta_linear = model_quad.params['ucla_total']
    beta_quad = model_quad.params['ucla_sq']
    p_linear = model_quad.pvalues['ucla_total']
    p_quad = model_quad.pvalues['ucla_sq']
    r2_quad = model_quad.rsquared

    # Linear model 비교
    X_lin = males[['ucla_total']]
    X_lin = sm.add_constant(X_lin)
    model_lin = sm.OLS(y, X_lin).fit()
    r2_lin = model_lin.rsquared

    # R² increase
    r2_increase = r2_quad - r2_lin

    print(f"\n   남성 (N={len(males)}) Quadratic Model:")
    print(f"   Linear: β={beta_linear:+.3f}, p={p_linear:.4f}")
    print(f"   Quadratic: β={beta_quad:+.5f}, p={p_quad:.4f}")
    print(f"   R² (linear): {r2_lin:.3f}")
    print(f"   R² (quadratic): {r2_quad:.3f}")
    print(f"   ΔR²: {r2_increase:.3f}")

    if p_quad < 0.05:
        print(f"   → 유의한 비선형 효과 (U-shaped or inverted-U)")
    else:
        print(f"   → 선형 효과로 충분")

    # Threshold model (ROC-based optimal cutoff)
    # Binary outcome: High PE (>median) vs Low PE
    pe_median = males['pe_rate'].median()
    males['high_pe'] = (males['pe_rate'] > pe_median).astype(int)

    fpr, tpr, thresholds = roc_curve(males['high_pe'], males['ucla_total'])
    roc_auc = auc(fpr, tpr)

    # Optimal threshold (Youden index)
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n   Threshold Model (ROC):")
    print(f"   Optimal UCLA cutoff: {optimal_threshold:.1f}")
    print(f"   AUC: {roc_auc:.3f}")
    print(f"   Sensitivity: {tpr[optimal_idx]:.3f}")
    print(f"   Specificity: {1-fpr[optimal_idx]:.3f}")

    # 저장
    nonlinear_results = pd.DataFrame([{
        'gender': '남성',
        'n': len(males),
        'beta_linear': beta_linear,
        'p_linear': p_linear,
        'beta_quadratic': beta_quad,
        'p_quadratic': p_quad,
        'r2_linear': r2_lin,
        'r2_quadratic': r2_quad,
        'delta_r2': r2_increase,
        'optimal_threshold': optimal_threshold,
        'roc_auc': roc_auc
    }])

    nonlinear_results.to_csv(OUTPUT_DIR / "nonlinear_ucla_effects.csv", index=False, encoding='utf-8-sig')
else:
    print(f"   ⚠ 남성 데이터 부족 (N={len(males)}), 스킵")

# ============================================================================
# 분석 3: 극단 집단 비교
# ============================================================================
print("\n[3] 극단 집단 비교 분석...")

# 상위 10% UCLA + 상위 10% PE (vulnerable 남성)
# 하위 10% UCLA + 하위 10% PE (resilient 여성)

males = master[master['gender'] == 'male']
females = master[master['gender'] == 'female']

# Percentiles
ucla_90_m = males['ucla_total'].quantile(0.9)
ucla_10_f = females['ucla_total'].quantile(0.1)

pe_90_m = males['pe_rate'].quantile(0.9)
pe_10_f = females['pe_rate'].quantile(0.1)

# Extreme groups
vulnerable_males = males[(males['ucla_total'] >= ucla_90_m) & (males['pe_rate'] >= pe_90_m)]
resilient_females = females[(females['ucla_total'] <= ucla_10_f) & (females['pe_rate'] <= pe_10_f)]

print(f"\n   극단 집단:")
print(f"   Vulnerable Males (UCLA≥90th, PE≥90th): N={len(vulnerable_males)}")
if len(vulnerable_males) > 0:
    print(f"     UCLA 평균: {vulnerable_males['ucla_total'].mean():.1f}")
    print(f"     PE rate 평균: {vulnerable_males['pe_rate'].mean():.1f}%")

print(f"   Resilient Females (UCLA≤10th, PE≤10th): N={len(resilient_females)}")
if len(resilient_females) > 0:
    print(f"     UCLA 평균: {resilient_females['ucla_total'].mean():.1f}")
    print(f"     PE rate 평균: {resilient_females['pe_rate'].mean():.1f}%")

# 극단 집단 비교 (t-test)
if len(vulnerable_males) > 3 and len(resilient_females) > 3:
    # PE rate
    t_stat, p_val = stats.ttest_ind(vulnerable_males['pe_rate'], resilient_females['pe_rate'])
    cohen_d = (vulnerable_males['pe_rate'].mean() - resilient_females['pe_rate'].mean()) / np.sqrt(
        ((len(vulnerable_males)-1)*vulnerable_males['pe_rate'].std()**2 +
         (len(resilient_females)-1)*resilient_females['pe_rate'].std()**2) /
        (len(vulnerable_males) + len(resilient_females) - 2)
    )

    print(f"\n   Vulnerable vs Resilient (PE rate):")
    print(f"   t({len(vulnerable_males)+len(resilient_females)-2})={t_stat:.3f}, p={p_val:.4f}")
    print(f"   Cohen's d={cohen_d:.3f}")

    extreme_comparison = pd.DataFrame([{
        'comparison': 'Vulnerable Males vs Resilient Females',
        'n_vulnerable': len(vulnerable_males),
        'n_resilient': len(resilient_females),
        'mean_pe_vulnerable': vulnerable_males['pe_rate'].mean(),
        'mean_pe_resilient': resilient_females['pe_rate'].mean(),
        't_stat': t_stat,
        'p_value': p_val,
        'cohens_d': cohen_d
    }])

    extreme_comparison.to_csv(OUTPUT_DIR / "extreme_groups_comparison.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# 분석 4: WCST PE 신뢰도 분석
# ============================================================================
print("\n[4] WCST PE 신뢰도 분석 (split-half)...")

# Trial-level 데이터 필요
try:
    wcst_trials, wcst_summary = load_wcst_trials(use_cache=True)
    print(f"   WCST trials: {len(wcst_trials)} | participants: {wcst_summary.get('n_participants')}")

    reliability_list = []

    for pid in wcst_trials['participant_id'].unique():
        trials = wcst_trials[wcst_trials['participant_id'] == pid].copy()
        if 'trialIndex' in trials.columns:
            trials = trials.sort_values('trialIndex').reset_index(drop=True)
        else:
            trials = trials.reset_index(drop=True)

        if len(trials) < 20:
            continue

        pe_col = 'isPE' if 'isPE' in trials.columns else None
        if pe_col is None and 'extra' in trials.columns:
            import ast
            def parse_extra(extra_str):
                if not isinstance(extra_str, str):
                    return {}
                try:
                    return ast.literal_eval(extra_str)
                except Exception:
                    return {}
            trials['extra_dict'] = trials['extra'].apply(parse_extra)
            trials['isPE'] = trials['extra_dict'].apply(lambda x: x.get('isPE', False))
            pe_col = 'isPE'

        if pe_col is None:
            continue

        n_trials = len(trials)
        half1 = trials.iloc[:n_trials//2]
        half2 = trials.iloc[n_trials//2:]

        pe_rate_h1 = half1[pe_col].mean() * 100 if len(half1) > 0 else np.nan
        pe_rate_h2 = half2[pe_col].mean() * 100 if len(half2) > 0 else np.nan

        if not np.isnan(pe_rate_h1) and not np.isnan(pe_rate_h2):
            reliability_list.append({
                'participantId': pid,
                'pe_rate_half1': pe_rate_h1,
                'pe_rate_half2': pe_rate_h2,
                'n_trials_half1': len(half1),
                'n_trials_half2': len(half2)
            })

    reliability_df = pd.DataFrame(reliability_list)

    if len(reliability_df) > 10:
        r_split, p_split = spearmanr(reliability_df['pe_rate_half1'], reliability_df['pe_rate_half2'])
        r_corrected = (2 * r_split) / (1 + r_split)

        print(f"\\n   Split-half reliability (N={len(reliability_df)}):")
        print(f"   Spearman r: {r_split:.3f}, p={p_split:.4f}")
        print(f"   Spearman-Brown corrected: {r_corrected:.3f}")

        if r_corrected > 0.7:
            print('   Reliability: excellent')
        elif r_corrected > 0.5:
            print('   Reliability: good')
        else:
            print('   Reliability: poor')

        reliability_summary = pd.DataFrame([{
            'n': len(reliability_df),
            'split_half_r': r_split,
            'p_value': p_split,
            'spearman_brown_corrected': r_corrected,
            'interpretation': 'Excellent' if r_corrected > 0.7 else ('Good' if r_corrected > 0.5 else 'Poor')
        }])

        reliability_summary.to_csv(OUTPUT_DIR / 'wcst_pe_reliability.csv', index=False, encoding='utf-8-sig')
        reliability_df.to_csv(OUTPUT_DIR / 'wcst_pe_split_half_data.csv', index=False, encoding='utf-8-sig')
    else:
        print(f"   Reliability analysis skipped (N={len(reliability_df)} < 10)")

except Exception as e:
    print(f"   Reliability analysis failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 시각화
# ============================================================================
print("\n[시각화]")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Task Difficulty vs Effect Size
ax = axes[0, 0]
if len(effect_size_df) > 0:
    ax.scatter(effect_size_df['mean_accuracy'], effect_size_df['gender_diff'],
               s=100, alpha=0.7, c='purple')

    for i, row in effect_size_df.iterrows():
        ax.text(row['mean_accuracy'], row['gender_diff'], row['task'],
                fontsize=9, ha='left', va='bottom')

    ax.set_xlabel('Task Accuracy (%)', fontsize=12)
    ax.set_ylabel('Gender Difference (|r_male - r_female|)', fontsize=12)
    ax.set_title('Task Difficulty × Effect Size', fontweight='bold')
    ax.grid(alpha=0.3)

# 2. Nonlinear UCLA (Quadratic fit)
ax = axes[0, 1]
if len(males) > 20:
    ax.scatter(males['ucla_total'], males['pe_rate'], alpha=0.6, c='blue', s=50)

    # Quadratic fit
    x_range = np.linspace(males['ucla_total'].min(), males['ucla_total'].max(), 100)
    y_quad = model_quad.params['const'] + model_quad.params['ucla_total'] * x_range + model_quad.params['ucla_sq'] * x_range**2
    ax.plot(x_range, y_quad, 'r-', linewidth=2, label='Quadratic')

    # Linear fit
    y_lin = model_lin.params['const'] + model_lin.params['ucla_total'] * x_range
    ax.plot(x_range, y_lin, 'b--', linewidth=2, label='Linear')

    ax.set_xlabel('UCLA Loneliness', fontsize=12)
    ax.set_ylabel('PE Rate (%)', fontsize=12)
    ax.set_title('Nonlinear UCLA Effect (Males)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# 3. Extreme Groups
ax = axes[1, 0]
if len(vulnerable_males) > 0 and len(resilient_females) > 0:
    groups = ['Vulnerable\nMales', 'Resilient\nFemales']
    means = [vulnerable_males['pe_rate'].mean(), resilient_females['pe_rate'].mean()]
    sems = [vulnerable_males['pe_rate'].sem(), resilient_females['pe_rate'].sem()]

    ax.bar(groups, means, yerr=sems, alpha=0.7, color=['red', 'green'], capsize=10)
    ax.set_ylabel('PE Rate (%)', fontsize=12)
    ax.set_title('Extreme Groups Comparison', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

# 4. Split-half Reliability
ax = axes[1, 1]
if len(reliability_df) > 10:
    ax.scatter(reliability_df['pe_rate_half1'], reliability_df['pe_rate_half2'],
               alpha=0.6, s=50)

    # Diagonal line
    min_val = min(reliability_df['pe_rate_half1'].min(), reliability_df['pe_rate_half2'].min())
    max_val = max(reliability_df['pe_rate_half1'].max(), reliability_df['pe_rate_half2'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    ax.set_xlabel('PE Rate Half 1 (%)', fontsize=12)
    ax.set_ylabel('PE Rate Half 2 (%)', fontsize=12)
    ax.set_title(f'Split-Half Reliability (r={r_split:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
output_fig = OUTPUT_DIR / "statistical_suite_plots.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"   저장: {output_fig}")
plt.close()

# ============================================================================
# 보고서
# ============================================================================
report_file = OUTPUT_DIR / "STATISTICAL_SUITE_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("통계 & 방법론 강화 분석 보고서 (4-in-1)\n")
    f.write("="*80 + "\n\n")

    f.write(f"분석 일자: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("1. Task Difficulty 비교\n")
    f.write("-"*40 + "\n")
    for _, row in task_diff_df.iterrows():
        f.write(f"{row['task']}: Acc={row['mean_accuracy']:.1f}%, Ceiling={row['ceiling_effect']}\n")
    f.write("\n")

    f.write("2. 비선형 UCLA 효과\n")
    f.write("-"*40 + "\n")
    if 'nonlinear_results' in locals():
        for _, row in nonlinear_results.iterrows():
            f.write(f"Quadratic: β={row['beta_quadratic']:+.5f}, p={row['p_quadratic']:.4f}\n")
            f.write(f"ΔR²: {row['delta_r2']:.3f}\n")
            f.write(f"Optimal threshold: UCLA={row['optimal_threshold']:.1f}\n\n")

    f.write("3. 극단 집단 비교\n")
    f.write("-"*40 + "\n")
    if 'extreme_comparison' in locals():
        for _, row in extreme_comparison.iterrows():
            f.write(f"Vulnerable Males (N={row['n_vulnerable']:.0f}): PE={row['mean_pe_vulnerable']:.1f}%\n")
            f.write(f"Resilient Females (N={row['n_resilient']:.0f}): PE={row['mean_pe_resilient']:.1f}%\n")
            f.write(f"Cohen's d={row['cohens_d']:.3f}, p={row['p_value']:.4f}\n\n")

    f.write("4. WCST PE 신뢰도\n")
    f.write("-"*40 + "\n")
    if 'reliability_summary' in locals():
        for _, row in reliability_summary.iterrows():
            f.write(f"Split-half r: {row['split_half_r']:.3f}\n")
            f.write(f"Spearman-Brown corrected: {row['spearman_brown_corrected']:.3f}\n")
            f.write(f"Interpretation: {row['interpretation']}\n\n")

    f.write("="*80 + "\n")

print(f"\n보고서 저장: {report_file}")

print("\n" + "="*80)
print("통계 & 방법론 강화 분석 완료!")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print("="*80)
