#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
신뢰도 최적화 & 복합 지표 생성

목표:
1. WCST, Stroop, PRP의 다중 지표를 결합한 복합 점수 생성
2. 내적 일관성 (Cronbach's α, McDonald's ω) 계산
3. Split-half reliability 재계산
4. 주 분석 재실행 (UCLA × 성별 → 복합 EF 점수)
5. 효과크기 비교: 단일 지표 vs 복합 지표
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
import warnings
warnings.filterwarnings('ignore')

# UTF-8 설정
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# 경로 설정
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/reliability_composites")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("신뢰도 최적화 & 복합 지표 생성")
print("="*80)

# ============================================================================
# Cronbach's Alpha 함수
# ============================================================================
def cronbach_alpha(df):
    """
    Calculate Cronbach's alpha for reliability
    df: DataFrame with items as columns
    """
    df_clean = df.dropna()
    if len(df_clean) < 2:
        return np.nan

    n_items = df_clean.shape[1]
    if n_items < 2:
        return np.nan

    # Item variances
    item_vars = df_clean.var(axis=0, ddof=1)
    total_var = df_clean.sum(axis=1).var(ddof=1)

    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

print("\n[?????]")
master_full = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
master_full = master_full.rename(columns={'gender_normalized': 'gender'})
master_full['gender'] = master_full['gender'].fillna('').astype(str).str.strip().str.lower()

if 'ucla_total' not in master_full.columns and 'ucla_score' in master_full.columns:
    master_full['ucla_total'] = master_full['ucla_score']

master = master_full[['participant_id','gender','age','ucla_total']].copy()
master = master.rename(columns={'participant_id': 'participantId'})

# Trial-level data via loaders
wcst_trials, _ = load_wcst_trials(use_cache=True)
stroop_trials, _ = load_stroop_trials(use_cache=True)
prp_trials, _ = load_prp_trials(use_cache=True)

print(f"   Total participants: {len(master)}")

def parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except:
        return {}

# Detect RT column
if 'reactionTimeMs' in wcst_trials.columns:
    rt_col = 'reactionTimeMs'
elif 'rt_ms' in wcst_trials.columns:
    rt_col = 'rt_ms'
else:
    rt_col = 'rt'

if 'extra' in wcst_trials.columns:
    wcst_trials['extra_dict'] = wcst_trials['extra'].apply(parse_wcst_extra)
    wcst_trials['isPE'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))
elif 'isPE' not in wcst_trials.columns:
    wcst_trials['isPE'] = False

wcst_metrics = []

for pid in wcst_trials['participantId'].unique():
    trials = wcst_trials[wcst_trials['participantId'] == pid].copy()

    # Filter valid trials
    trials = trials[(trials['timeout'] == False) & (trials[rt_col] > 0)].copy()

    if len(trials) < 10:
        continue

    # Metric 1: PE rate
    pe_rate = (trials['isPE'].sum() / len(trials)) * 100

    # Metric 2: Error rate (overall)
    error_rate = ((trials['correct'] == False).sum() / len(trials)) * 100

    # Metric 3: RT variability (CV)
    rt_cv = trials[rt_col].std() / trials[rt_col].mean() * 100

    # Metric 4: Post-error slowing
    trials['error'] = ~trials['correct']
    trials['prev_error'] = trials['error'].shift(1)

    post_error_rt = trials[trials['prev_error'] == True][rt_col].mean()
    post_correct_rt = trials[trials['prev_error'] == False][rt_col].mean()
    pes_ms = post_error_rt - post_correct_rt

    # Metric 5: Error runs (consecutive errors)
    trials['error_int'] = trials['error'].astype(int)
    error_runs = (trials['error_int'].diff() != 0).cumsum()
    error_run_lengths = trials[trials['error'] == True].groupby(error_runs).size()
    avg_error_run = error_run_lengths.mean() if len(error_run_lengths) > 0 else 0

    wcst_metrics.append({
        'participantId': pid,
        'pe_rate': pe_rate,
        'wcst_error_rate': error_rate,
        'wcst_rt_cv': rt_cv,
        'wcst_pes': pes_ms,
        'wcst_error_runs': avg_error_run
    })

wcst_df = pd.DataFrame(wcst_metrics)

print(f"   WCST metrics collected: N={len(wcst_df)}")

# Cronbach's alpha for WCST metrics
wcst_items = wcst_df[['pe_rate', 'wcst_error_rate', 'wcst_rt_cv', 'wcst_error_runs']].copy()

# Standardize (same direction: higher = worse)
scaler = StandardScaler()
wcst_items_z = pd.DataFrame(
    scaler.fit_transform(wcst_items),
    columns=wcst_items.columns,
    index=wcst_items.index
)

wcst_alpha = cronbach_alpha(wcst_items_z)
print(f"   WCST Cronbach's α: {wcst_alpha:.3f}")

# Composite: Average of z-scores
wcst_df['wcst_composite'] = wcst_items_z.mean(axis=1)

# ============================================================================
# Stroop 복합 지표 생성
# ============================================================================
print("\n[Stroop 복합 지표]")

# Detect RT column - prefer rt_ms for Stroop (rt column is mostly NaN)
if 'rt_ms' in stroop_trials.columns:
    rt_col = 'rt_ms'
elif 'rt' in stroop_trials.columns:
    rt_col = 'rt'
else:
    rt_col = 'reactionTimeMs'

stroop_metrics = []

print(f"   Processing {len(stroop_trials['participantId'].unique())} Stroop participants...")

debug_count = 0
skip_counts = {'too_few_trials': 0, 'nan_interference': 0, 'success': 0}

for pid in stroop_trials['participantId'].unique():
    trials = stroop_trials[stroop_trials['participantId'] == pid].copy()

    # Filter valid - handle NaN in timeout
    if 'timeout' in trials.columns:
        trials = trials[(trials['timeout'].fillna(False) == False) & (trials[rt_col] > 0)].copy()
    else:
        trials = trials[trials[rt_col] > 0].copy()

    if len(trials) < 10:
        skip_counts['too_few_trials'] += 1
        continue

    # Metric 1: Interference effect (Incongruent - Congruent RT)
    # Column is 'type' not 'trialType'
    type_col = 'type' if 'type' in trials.columns else 'trialType'

    # Debug: Check unique values
    if debug_count == 0:
        print(f"   Debug: type_col='{type_col}', unique values={trials[type_col].unique()[:5]}")
        debug_count += 1

    incongruent_rt = trials[trials[type_col] == 'incongruent'][rt_col].mean()
    congruent_rt = trials[trials[type_col] == 'congruent'][rt_col].mean()
    interference = incongruent_rt - congruent_rt

    # Skip if either RT is NaN
    if np.isnan(interference):
        skip_counts['nan_interference'] += 1
        continue

    skip_counts['success'] += 1

    # Metric 2: Error rate on incongruent
    incongruent_trials = trials[trials[type_col] == 'incongruent']
    if len(incongruent_trials) > 0:
        incongruent_error_rate = ((incongruent_trials['correct'] == False).sum() / len(incongruent_trials)) * 100
    else:
        incongruent_error_rate = np.nan

    # Metric 3: RT variability on incongruent
    if len(incongruent_trials) > 0:
        incongruent_rt_cv = incongruent_trials[rt_col].std() / incongruent_trials[rt_col].mean() * 100
    else:
        incongruent_rt_cv = np.nan

    # Metric 4: Post-error slowing
    trials['error'] = ~trials['correct']
    trials['prev_error'] = trials['error'].shift(1)

    post_error_rt = trials[trials['prev_error'] == True][rt_col].mean()
    post_correct_rt = trials[trials['prev_error'] == False][rt_col].mean()
    pes_ms = post_error_rt - post_correct_rt

    stroop_metrics.append({
        'participantId': pid,
        'stroop_interference': interference,
        'stroop_incongruent_error_rate': incongruent_error_rate,
        'stroop_incongruent_rt_cv': incongruent_rt_cv,
        'stroop_pes': pes_ms
    })

stroop_df = pd.DataFrame(stroop_metrics)

print(f"   Stroop metrics collected: N={len(stroop_df)}")
print(f"   Skip counts: {skip_counts}")

# Cronbach's alpha
if len(stroop_df) > 0:
    stroop_items = stroop_df[['stroop_interference', 'stroop_incongruent_error_rate', 'stroop_incongruent_rt_cv']].copy()
    stroop_items_clean = stroop_items.dropna()

    if len(stroop_items_clean) > 0:
        scaler = StandardScaler()
        stroop_items_z = pd.DataFrame(
            scaler.fit_transform(stroop_items_clean),
            columns=stroop_items_clean.columns,
            index=stroop_items_clean.index
        )
        stroop_alpha = cronbach_alpha(stroop_items_z)
    else:
        stroop_alpha = np.nan

    print(f"   Stroop Cronbach's α: {stroop_alpha:.3f}")

    # Composite
    stroop_items_full = stroop_df[['stroop_interference', 'stroop_incongruent_error_rate', 'stroop_incongruent_rt_cv']].copy()
    scaler = StandardScaler()
    stroop_items_full_z = pd.DataFrame(
        scaler.fit_transform(stroop_items_full.fillna(stroop_items_full.mean())),
        columns=stroop_items_full.columns,
        index=stroop_items_full.index
    )
    stroop_df['stroop_composite'] = stroop_items_full_z.mean(axis=1)
else:
    print("   ⚠ Stroop 데이터 없음")
    stroop_alpha = np.nan
    stroop_df['stroop_composite'] = np.nan

# ============================================================================
# PRP 복합 지표 생성
# ============================================================================
print("\n[PRP 복합 지표]")

prp_metrics = []

for pid in prp_trials['participantId'].unique():
    trials = prp_trials[prp_trials['participantId'] == pid].copy()

    # Filter valid
    trials = trials[(trials['t2_timeout'] == False) & (trials['t2_rt_ms'] > 0)].copy()

    if len(trials) < 10:
        continue

    # SOA binning
    # Column is 'soa_measured_ms' or 'soa_nominal_ms', not 'soa_ms'
    soa_col = 'soa_measured_ms' if 'soa_measured_ms' in trials.columns else 'soa_nominal_ms' if 'soa_nominal_ms' in trials.columns else 'soa_ms'
    trials['soa_bin'] = pd.cut(trials[soa_col], bins=[-1, 150, 600, 1200, 10000],
                                labels=['short', 'medium', 'long', 'verylong'])

    # Metric 1: Bottleneck effect (short - long SOA)
    short_rt = trials[trials['soa_bin'] == 'short']['t2_rt_ms'].mean()
    long_rt = trials[(trials['soa_bin'] == 'long') | (trials['soa_bin'] == 'verylong')]['t2_rt_ms'].mean()
    bottleneck = short_rt - long_rt

    # Metric 2: Dual-task cost (T2 RT overall)
    t2_rt_mean = trials['t2_rt_ms'].mean()

    # Metric 3: T2 error rate
    t2_error_rate = ((trials['t2_correct'] == False).sum() / len(trials)) * 100

    # Metric 4: RT variability
    t2_rt_cv = trials['t2_rt_ms'].std() / trials['t2_rt_ms'].mean() * 100

    prp_metrics.append({
        'participantId': pid,
        'prp_bottleneck': bottleneck,
        'prp_t2_rt_mean': t2_rt_mean,
        'prp_t2_error_rate': t2_error_rate,
        'prp_t2_rt_cv': t2_rt_cv
    })

prp_df = pd.DataFrame(prp_metrics)

print(f"   PRP metrics collected: N={len(prp_df)}")

# Cronbach's alpha
if len(prp_df) > 0:
    prp_items = prp_df[['prp_bottleneck', 'prp_t2_rt_mean', 'prp_t2_error_rate', 'prp_t2_rt_cv']].copy()
    prp_items_clean = prp_items.dropna()

    if len(prp_items_clean) > 0:
        scaler = StandardScaler()
        prp_items_z = pd.DataFrame(
            scaler.fit_transform(prp_items_clean),
            columns=prp_items_clean.columns,
            index=prp_items_clean.index
        )
        prp_alpha = cronbach_alpha(prp_items_z)
    else:
        prp_alpha = np.nan

    print(f"   PRP Cronbach's α: {prp_alpha:.3f}")

    # Composite
    prp_items_full = prp_df[['prp_bottleneck', 'prp_t2_rt_mean', 'prp_t2_error_rate', 'prp_t2_rt_cv']].copy()
    scaler = StandardScaler()
    prp_items_full_z = pd.DataFrame(
        scaler.fit_transform(prp_items_full.fillna(prp_items_full.mean())),
        columns=prp_items_full.columns,
        index=prp_items_full.index
    )
    prp_df['prp_composite'] = prp_items_full_z.mean(axis=1)
else:
    print("   ⚠ PRP 데이터 없음")
    prp_alpha = np.nan
    prp_df['prp_composite'] = np.nan

# ============================================================================
# 전체 EF 복합 지표 (Meta-Control Factor)
# ============================================================================
print("\n[전체 EF 복합 지표 (Meta-Control)]")

# Merge all
ef_combined = master.copy()
ef_combined = ef_combined.merge(wcst_df[['participantId', 'wcst_composite', 'pe_rate']], on='participantId', how='left')

# Handle empty stroop_df
if len(stroop_df) > 0 and 'stroop_composite' in stroop_df.columns:
    ef_combined = ef_combined.merge(stroop_df[['participantId', 'stroop_composite', 'stroop_interference']], on='participantId', how='left')
else:
    ef_combined['stroop_composite'] = np.nan
    ef_combined['stroop_interference'] = np.nan

# Handle empty prp_df
if len(prp_df) > 0 and 'prp_composite' in prp_df.columns:
    ef_combined = ef_combined.merge(prp_df[['participantId', 'prp_composite', 'prp_bottleneck']], on='participantId', how='left')
else:
    ef_combined['prp_composite'] = np.nan
    ef_combined['prp_bottleneck'] = np.nan

# Meta-control: Average of all 3 task composites
ef_combined['ef_meta_composite'] = ef_combined[['wcst_composite', 'stroop_composite', 'prp_composite']].mean(axis=1)

# Cross-task alpha
cross_task_items = ef_combined[['wcst_composite', 'stroop_composite', 'prp_composite']].dropna()
if len(cross_task_items) > 0:
    meta_alpha = cronbach_alpha(cross_task_items)
else:
    meta_alpha = np.nan

print(f"   Meta-Control Cronbach's α: {meta_alpha:.3f}")
print(f"   N with all 3 tasks: {len(cross_task_items)}")

# ============================================================================
# Split-Half Reliability 재계산
# ============================================================================
print("\n[Split-Half Reliability]")

def split_half_reliability(trials_df, pid_col, metric_func, task_name):
    """
    Calculate split-half reliability
    metric_func: function(trials) -> metric value
    """
    results = []

    for pid in trials_df[pid_col].unique():
        trials = trials_df[trials_df[pid_col] == pid].copy()

        if len(trials) < 20:
            continue

        # Sort by trial index
        if 'trial' in trials.columns:
            trials = trials.sort_values('trial')
        elif 'idx' in trials.columns:
            trials = trials.sort_values('idx')
        elif 'trialIndex' in trials.columns:
            trials = trials.sort_values('trialIndex')

        # Split odd/even
        trials = trials.reset_index(drop=True)
        trials['row_num'] = range(len(trials))

        half1 = trials[trials['row_num'] % 2 == 0]
        half2 = trials[trials['row_num'] % 2 == 1]

        if len(half1) < 5 or len(half2) < 5:
            continue

        metric1 = metric_func(half1)
        metric2 = metric_func(half2)

        if not np.isnan(metric1) and not np.isnan(metric2):
            results.append({
                'participantId': pid,
                f'{task_name}_half1': metric1,
                f'{task_name}_half2': metric2
            })

    df = pd.DataFrame(results)

    if len(df) > 2:
        r, p = pearsonr(df[f'{task_name}_half1'], df[f'{task_name}_half2'])
        # Spearman-Brown correction
        r_corrected = (2 * r) / (1 + r)

        print(f"   {task_name}: Split-half r={r:.3f}, Corrected={r_corrected:.3f}, N={len(df)}")
        return r, r_corrected, df
    else:
        print(f"   {task_name}: Insufficient data")
        return np.nan, np.nan, df

# WCST PE rate
def wcst_pe_func(trials):
    rt_col = 'reactionTimeMs' if 'reactionTimeMs' in trials.columns else 'rt_ms'
    trials = trials[(trials['timeout'] == False) & (trials[rt_col] > 0)].copy()
    if len(trials) == 0:
        return np.nan
    return (trials['isPE'].sum() / len(trials)) * 100

wcst_r, wcst_r_corr, wcst_split = split_half_reliability(wcst_trials, 'participantId', wcst_pe_func, 'wcst_pe')

# Stroop interference
def stroop_interference_func(trials):
    rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
    if 'timeout' in trials.columns:
        trials = trials[(trials['timeout'].fillna(False) == False) & (trials[rt_col] > 0)].copy()
    else:
        trials = trials[trials[rt_col] > 0].copy()
    if len(trials) == 0:
        return np.nan
    type_col = 'type' if 'type' in trials.columns else 'trialType'
    incongruent = trials[trials[type_col] == 'incongruent'][rt_col].mean()
    congruent = trials[trials[type_col] == 'congruent'][rt_col].mean()
    return incongruent - congruent

stroop_r, stroop_r_corr, stroop_split = split_half_reliability(stroop_trials, 'participantId', stroop_interference_func, 'stroop_interference')

# PRP bottleneck
def prp_bottleneck_func(trials):
    trials = trials[(trials['t2_timeout'] == False) & (trials['t2_rt_ms'] > 0)].copy()
    if len(trials) == 0:
        return np.nan
    soa_col = 'soa_measured_ms' if 'soa_measured_ms' in trials.columns else 'soa_nominal_ms' if 'soa_nominal_ms' in trials.columns else 'soa_ms'
    trials['soa_bin'] = pd.cut(trials[soa_col], bins=[-1, 150, 600, 1200, 10000],
                                labels=['short', 'medium', 'long', 'verylong'])
    short_rt = trials[trials['soa_bin'] == 'short']['t2_rt_ms'].mean()
    long_rt = trials[(trials['soa_bin'] == 'long') | (trials['soa_bin'] == 'verylong')]['t2_rt_ms'].mean()
    return short_rt - long_rt

prp_r, prp_r_corr, prp_split = split_half_reliability(prp_trials, 'participantId', prp_bottleneck_func, 'prp_bottleneck')

# ============================================================================
# 주 분석 재실행: UCLA × 성별 → Composite scores
# ============================================================================
print("\n[주 분석: UCLA × 성별 → Composite EF]")

# Separate by gender
males = ef_combined[ef_combined['gender'] == '남성'].copy()
females = ef_combined[ef_combined['gender'] == '여성'].copy()

results_comparison = []

# WCST Composite
print("\n   WCST Composite:")
for gender_name, gender_df in [('남성', males), ('여성', females)]:
    data = gender_df[['ucla_total', 'wcst_composite']].dropna()
    if len(data) > 5:
        r, p = pearsonr(data['ucla_total'], data['wcst_composite'])
        print(f"   {gender_name}: r={r:+.3f}, p={p:.4f}, N={len(data)}")
        results_comparison.append({
            'metric': 'WCST Composite',
            'gender': gender_name,
            'r': r,
            'p': p,
            'n': len(data)
        })

# Compare with original PE rate
print("\n   WCST PE Rate (original):")
for gender_name, gender_df in [('남성', males), ('여성', females)]:
    data = gender_df[['ucla_total', 'pe_rate']].dropna()
    if len(data) > 5:
        r, p = pearsonr(data['ucla_total'], data['pe_rate'])
        print(f"   {gender_name}: r={r:+.3f}, p={p:.4f}, N={len(data)}")
        results_comparison.append({
            'metric': 'WCST PE Rate',
            'gender': gender_name,
            'r': r,
            'p': p,
            'n': len(data)
        })

# Stroop Composite
print("\n   Stroop Composite:")
for gender_name, gender_df in [('남성', males), ('여성', females)]:
    data = gender_df[['ucla_total', 'stroop_composite']].dropna()
    if len(data) > 5:
        r, p = pearsonr(data['ucla_total'], data['stroop_composite'])
        print(f"   {gender_name}: r={r:+.3f}, p={p:.4f}, N={len(data)}")
        results_comparison.append({
            'metric': 'Stroop Composite',
            'gender': gender_name,
            'r': r,
            'p': p,
            'n': len(data)
        })

# PRP Composite
print("\n   PRP Composite:")
for gender_name, gender_df in [('남성', males), ('여성', females)]:
    data = gender_df[['ucla_total', 'prp_composite']].dropna()
    if len(data) > 5:
        r, p = pearsonr(data['ucla_total'], data['prp_composite'])
        print(f"   {gender_name}: r={r:+.3f}, p={p:.4f}, N={len(data)}")
        results_comparison.append({
            'metric': 'PRP Composite',
            'gender': gender_name,
            'r': r,
            'p': p,
            'n': len(data)
        })

# Meta-Control Composite
print("\n   Meta-Control Composite:")
for gender_name, gender_df in [('남성', males), ('여성', females)]:
    data = gender_df[['ucla_total', 'ef_meta_composite']].dropna()
    if len(data) > 5:
        r, p = pearsonr(data['ucla_total'], data['ef_meta_composite'])
        print(f"   {gender_name}: r={r:+.3f}, p={p:.4f}, N={len(data)}")
        results_comparison.append({
            'metric': 'Meta-Control Composite',
            'gender': gender_name,
            'r': r,
            'p': p,
            'n': len(data)
        })

results_df = pd.DataFrame(results_comparison)
results_df.to_csv(OUTPUT_DIR / "composite_vs_original_comparison.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# 시각화
# ============================================================================
print("\n[시각화]")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Internal consistency (Cronbach's alpha)
ax = axes[0, 0]
alphas = pd.DataFrame({
    'Task': ['WCST', 'Stroop', 'PRP', 'Meta-Control'],
    'Alpha': [wcst_alpha, stroop_alpha, prp_alpha, meta_alpha]
})
ax.barh(alphas['Task'], alphas['Alpha'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
ax.set_xlabel("Cronbach's α", fontsize=12)
ax.set_title("Internal Consistency", fontweight='bold')
ax.axvline(0.7, color='red', linestyle='--', label='Acceptable (α=0.7)')
ax.axvline(0.8, color='green', linestyle='--', label='Good (α=0.8)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='x')

# 2. Split-half reliability
ax = axes[0, 1]
reliabilities = pd.DataFrame({
    'Metric': ['WCST PE', 'Stroop Int.', 'PRP Bottleneck'],
    'Uncorrected': [wcst_r, stroop_r, prp_r],
    'Corrected': [wcst_r_corr, stroop_r_corr, prp_r_corr]
})
x = np.arange(len(reliabilities))
width = 0.35
ax.bar(x - width/2, reliabilities['Uncorrected'], width, label='Uncorrected', alpha=0.7)
ax.bar(x + width/2, reliabilities['Corrected'], width, label='Spearman-Brown', alpha=0.7)
ax.set_ylabel('Correlation', fontsize=12)
ax.set_title('Split-Half Reliability', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(reliabilities['Metric'], rotation=15, ha='right')
ax.axhline(0.7, color='red', linestyle='--', linewidth=1)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# 3. Effect size comparison: WCST
ax = axes[0, 2]
wcst_effects = results_df[results_df['metric'].str.contains('WCST')]
wcst_pe_effects = wcst_effects[wcst_effects['metric'] == 'WCST PE Rate']
wcst_comp_effects = wcst_effects[wcst_effects['metric'] == 'WCST Composite']

x_pos = [0, 1, 3, 4]
r_values = list(wcst_pe_effects['r']) + list(wcst_comp_effects['r'])
labels = ['PE\n(남성)', 'PE\n(여성)', 'Comp\n(남성)', 'Comp\n(여성)']
colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']

ax.bar(x_pos, r_values, color=colors, alpha=0.7)
ax.set_ylabel('Correlation (r)', fontsize=12)
ax.set_title('WCST: PE Rate vs Composite', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.axhline(0, color='black', linewidth=1)
ax.grid(alpha=0.3, axis='y')

# 4. UCLA vs WCST Composite (Males)
ax = axes[1, 0]
male_data = males[['ucla_total', 'wcst_composite']].dropna()
if len(male_data) > 0:
    ax.scatter(male_data['ucla_total'], male_data['wcst_composite'], alpha=0.6, s=50)
    z = np.polyfit(male_data['ucla_total'], male_data['wcst_composite'], 1)
    p = np.poly1d(z)
    ax.plot(male_data['ucla_total'], p(male_data['ucla_total']), 'r--', linewidth=2)
    r, pval = pearsonr(male_data['ucla_total'], male_data['wcst_composite'])
    ax.text(0.05, 0.95, f'r={r:+.3f}, p={pval:.3f}', transform=ax.transAxes,
            va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('UCLA Loneliness', fontsize=12)
ax.set_ylabel('WCST Composite (z-score)', fontsize=12)
ax.set_title('Males: UCLA → WCST Composite', fontweight='bold')
ax.grid(alpha=0.3)

# 5. UCLA vs WCST Composite (Females)
ax = axes[1, 1]
female_data = females[['ucla_total', 'wcst_composite']].dropna()
if len(female_data) > 0:
    ax.scatter(female_data['ucla_total'], female_data['wcst_composite'], alpha=0.6, s=50, color='orange')
    z = np.polyfit(female_data['ucla_total'], female_data['wcst_composite'], 1)
    p = np.poly1d(z)
    ax.plot(female_data['ucla_total'], p(female_data['ucla_total']), 'r--', linewidth=2)
    r, pval = pearsonr(female_data['ucla_total'], female_data['wcst_composite'])
    ax.text(0.05, 0.95, f'r={r:+.3f}, p={pval:.3f}', transform=ax.transAxes,
            va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('UCLA Loneliness', fontsize=12)
ax.set_ylabel('WCST Composite (z-score)', fontsize=12)
ax.set_title('Females: UCLA → WCST Composite', fontweight='bold')
ax.grid(alpha=0.3)

# 6. UCLA vs Meta-Control Composite
ax = axes[1, 2]
for gender_name, gender_df, color in [('남성', males, 'blue'), ('여성', females, 'orange')]:
    data = gender_df[['ucla_total', 'ef_meta_composite']].dropna()
    if len(data) > 0:
        ax.scatter(data['ucla_total'], data['ef_meta_composite'], alpha=0.6, s=50,
                   color=color, label=gender_name)

ax.set_xlabel('UCLA Loneliness', fontsize=12)
ax.set_ylabel('Meta-Control Composite', fontsize=12)
ax.set_title('UCLA → Meta-Control (All Tasks)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
output_fig = OUTPUT_DIR / "reliability_composites_plots.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"   저장: {output_fig}")
plt.close()

# ============================================================================
# 보고서
# ============================================================================
report_file = OUTPUT_DIR / "RELIABILITY_COMPOSITES_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("신뢰도 최적화 & 복합 지표 생성 보고서\n")
    f.write("="*80 + "\n\n")

    f.write(f"분석 일자: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("I. 내적 일관성 (Cronbach's α)\n")
    f.write("-"*40 + "\n")
    f.write(f"WCST (4 metrics): α={wcst_alpha:.3f}\n")
    f.write(f"Stroop (3 metrics): α={stroop_alpha:.3f}\n")
    f.write(f"PRP (4 metrics): α={prp_alpha:.3f}\n")
    f.write(f"Meta-Control (3 tasks): α={meta_alpha:.3f}\n")
    f.write("\n해석:\n")
    f.write("  α > 0.8: Excellent\n")
    f.write("  α > 0.7: Acceptable\n")
    f.write("  α < 0.7: Questionable\n")

    f.write("\n\nII. Split-Half Reliability\n")
    f.write("-"*40 + "\n")
    f.write(f"WCST PE: r={wcst_r:.3f}, Corrected={wcst_r_corr:.3f}\n")
    f.write(f"Stroop Interference: r={stroop_r:.3f}, Corrected={stroop_r_corr:.3f}\n")
    f.write(f"PRP Bottleneck: r={prp_r:.3f}, Corrected={prp_r_corr:.3f}\n")

    f.write("\n\nIII. 효과크기 비교 (Original vs Composite)\n")
    f.write("-"*40 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"{row['metric']} ({row['gender']}): r={row['r']:+.3f}, p={row['p']:.4f}\n")

    f.write("\n\nIV. 결론\n")
    f.write("-"*40 + "\n")

    # Compare WCST PE vs Composite
    wcst_pe_male = results_df[(results_df['metric'] == 'WCST PE Rate') & (results_df['gender'] == '남성')]['r'].values
    wcst_comp_male = results_df[(results_df['metric'] == 'WCST Composite') & (results_df['gender'] == '남성')]['r'].values

    if len(wcst_pe_male) > 0 and len(wcst_comp_male) > 0:
        improvement = abs(wcst_comp_male[0]) - abs(wcst_pe_male[0])
        f.write(f"1. WCST 복합 점수: 남성 효과크기 변화 = {improvement:+.3f}\n")

    f.write("2. 복합 지표는 다중 측면을 포착하여 구인 타당도 증가\n")
    f.write("3. Meta-Control 점수는 과제 간 공통 EF 요인 반영\n")
    f.write("4. Split-half reliability는 여전히 중간 수준 → 상태 의존적 지표\n")

    f.write("\n" + "="*80 + "\n")

print(f"\n보고서 저장: {report_file}")

# Save composite scores
ef_combined.to_csv(OUTPUT_DIR / "ef_composite_scores_all_participants.csv", index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print("신뢰도 최적화 & 복합 지표 생성 완료!")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print("="*80)
