#!/usr/bin/env python3
"""
극단 집단 분석: 외로움 상위 vs 하위 집단 비교
======================================================
고외로움 집단 vs 저외로움 집단의 집행기능 차이를 검증

⚠️ WARNING: DASS-21 CONTROL MISSING ⚠️
================================================================================
This script does NOT control for DASS-21 (depression/anxiety/stress) as covariates.
Results confound loneliness effects with mood/anxiety symptoms.

DO NOT cite these results as evidence of "pure loneliness effects".

For confirmatory analysis with proper DASS control, use:
  - master_dass_controlled_analysis.py (hierarchical regression with covariates)

This script is EXPLORATORY ONLY - shows raw group differences without confound control.
================================================================================
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import statistical utilities for FDR correction
from statistical_utils import apply_multiple_comparison_correction

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("극단 집단 분석: 외로움 상/하 집단 비교")
print("=" * 80)

data_dir = Path("results")
output_dir = data_dir / "analysis_outputs"

# Load master dataset
print("\n[Step 1] Loading master dataset...")
master = pd.read_csv(output_dir / "master_dataset.csv")
print(f"Total N = {len(master)}")

# Descriptive stats for UCLA
print(f"\nUCLA Loneliness descriptive:")
print(f"  Mean = {master['ucla_total'].mean():.2f}")
print(f"  SD = {master['ucla_total'].std():.2f}")
print(f"  Median = {master['ucla_total'].median():.2f}")
print(f"  Range = {master['ucla_total'].min():.0f} - {master['ucla_total'].max():.0f}")

# ============================================================================
# Method 1: Quartile Split (상위 25% vs 하위 25%)
# ============================================================================
print("\n" + "=" * 80)
print("Method 1: Quartile Split (상위 25% vs 하위 25%)")
print("=" * 80)

q1 = master['ucla_total'].quantile(0.25)
q3 = master['ucla_total'].quantile(0.75)

low_group = master[master['ucla_total'] <= q1].copy()
high_group = master[master['ucla_total'] >= q3].copy()

print(f"\n저외로움 집단 (UCLA ≤ {q1:.0f}): N = {len(low_group)}")
print(f"  Mean UCLA = {low_group['ucla_total'].mean():.2f} (SD = {low_group['ucla_total'].std():.2f})")

print(f"\n고외로움 집단 (UCLA ≥ {q3:.0f}): N = {len(high_group)}")
print(f"  Mean UCLA = {high_group['ucla_total'].mean():.2f} (SD = {high_group['ucla_total'].std():.2f})")

# Executive function measures to compare
ef_measures = {
    'stroop_interference': 'Stroop Interference (ms)',
    'perseverative_error_rate': 'WCST Perseverative Error Rate (%)',
    'prp_bottleneck': 'PRP Bottleneck Effect (ms)'
}

# Also include DASS for comparison
dass_measures = {
    'dass_depression': 'DASS Depression',
    'dass_anxiety': 'DASS Anxiety',
    'dass_stress': 'DASS Stress'
}

all_measures = {**ef_measures, **dass_measures}

# Perform t-tests
print("\n" + "-" * 80)
print("Independent samples t-tests")
print("-" * 80)

results_q = []

for measure, label in all_measures.items():
    low_data = low_group[measure].dropna()
    high_data = high_group[measure].dropna()

    # Descriptive stats
    low_mean = low_data.mean()
    low_sd = low_data.std()
    high_mean = high_data.mean()
    high_sd = high_data.std()

    # CRITICAL FIX: Welch's t-test (does not assume equal variances)
    t_stat, p_val = stats.ttest_ind(low_data, high_data, equal_var=False)

    # Effect size (Cohen's d with pooled SD)
    pooled_sd = np.sqrt(((len(low_data) - 1) * low_sd**2 + (len(high_data) - 1) * high_sd**2) /
                         (len(low_data) + len(high_data) - 2))
    cohens_d = (high_mean - low_mean) / pooled_sd

    # Store results
    results_q.append({
        'Measure': label,
        'Low_N': len(low_data),
        'Low_Mean': low_mean,
        'Low_SD': low_sd,
        'High_N': len(high_data),
        'High_Mean': high_mean,
        'High_SD': high_sd,
        't': t_stat,
        'p': p_val,
        'Cohen_d': cohens_d
    })

    # Print
    print(f"\n{label}:")
    print(f"  Low:  M = {low_mean:7.2f}, SD = {low_sd:6.2f} (n = {len(low_data)})")
    print(f"  High: M = {high_mean:7.2f}, SD = {high_sd:6.2f} (n = {len(high_data)})")
    print(f"  t({len(low_data) + len(high_data) - 2}) = {t_stat:6.2f}, p = {p_val:.4f}, d = {cohens_d:5.2f}")

    if p_val < 0.001:
        print(f"  *** p < .001")
    elif p_val < 0.01:
        print(f"  ** p < .01")
    elif p_val < 0.05:
        print(f"  * p < .05")

# CRITICAL FIX: Apply FDR correction to all 6 tests (3 EF + 3 DASS)
results_q_df = pd.DataFrame(results_q)
print("\n" + "-" * 80)
print("다중비교 보정 (FDR) for Quartile Split")
print("-" * 80)

p_vals_q = results_q_df['p'].values
reject_q_fdr, p_adj_q_fdr = apply_multiple_comparison_correction(
    p_vals_q,
    method='fdr_bh',
    alpha=0.05
)

results_q_df['p_fdr_adjusted'] = p_adj_q_fdr
results_q_df['significant_fdr'] = reject_q_fdr

print(f"\n총 {len(p_vals_q)}개 검정에 FDR 보정 적용:")
for i, row in results_q_df.iterrows():
    sig_marker = "***" if row['p_fdr_adjusted'] < 0.001 else "**" if row['p_fdr_adjusted'] < 0.01 else "*" if row['p_fdr_adjusted'] < 0.05 else "ns"
    print(f"  {row['Measure']:35s}: p_raw = {row['p']:.4f}, p_fdr = {row['p_fdr_adjusted']:.4f} [{sig_marker}]")

# Save results
results_q_df.to_csv(output_dir / "extreme_group_quartile_results.csv", index=False, encoding='utf-8-sig')
print(f"\n[OK] Saved (with FDR): {output_dir / 'extreme_group_quartile_results.csv'}")

# ============================================================================
# Method 2: Median Split (중앙값 기준)
# ============================================================================
print("\n" + "=" * 80)
print("Method 2: Median Split (중앙값 기준)")
print("=" * 80)

median = master['ucla_total'].median()

low_group_med = master[master['ucla_total'] < median].copy()
high_group_med = master[master['ucla_total'] >= median].copy()

print(f"\n저외로움 집단 (UCLA < {median:.0f}): N = {len(low_group_med)}")
print(f"  Mean UCLA = {low_group_med['ucla_total'].mean():.2f}")

print(f"\n고외로움 집단 (UCLA ≥ {median:.0f}): N = {len(high_group_med)}")
print(f"  Mean UCLA = {high_group_med['ucla_total'].mean():.2f}")

results_med = []

for measure, label in all_measures.items():
    low_data = low_group_med[measure].dropna()
    high_data = high_group_med[measure].dropna()

    low_mean = low_data.mean()
    low_sd = low_data.std()
    high_mean = high_data.mean()
    high_sd = high_data.std()

    # CRITICAL FIX: Welch's t-test (does not assume equal variances)
    t_stat, p_val = stats.ttest_ind(low_data, high_data, equal_var=False)

    pooled_sd = np.sqrt(((len(low_data) - 1) * low_sd**2 + (len(high_data) - 1) * high_sd**2) /
                         (len(low_data) + len(high_data) - 2))
    cohens_d = (high_mean - low_mean) / pooled_sd

    results_med.append({
        'Measure': label,
        'Low_N': len(low_data),
        'Low_Mean': low_mean,
        'Low_SD': low_sd,
        'High_N': len(high_data),
        'High_Mean': high_mean,
        'High_SD': high_sd,
        't': t_stat,
        'p': p_val,
        'Cohen_d': cohens_d
    })

print("\n" + "-" * 80)
print("Results (Median Split)")
print("-" * 80)

# CRITICAL FIX: Apply FDR correction to all 6 tests
results_med_df = pd.DataFrame(results_med)
p_vals_med = results_med_df['p'].values
reject_med_fdr, p_adj_med_fdr = apply_multiple_comparison_correction(
    p_vals_med,
    method='fdr_bh',
    alpha=0.05
)

results_med_df['p_fdr_adjusted'] = p_adj_med_fdr
results_med_df['significant_fdr'] = reject_med_fdr

for i, res in results_med_df.iterrows():
    if res['Measure'] in ef_measures.values():  # Only print EF measures
        sig_marker = "*" if res['significant_fdr'] else "ns"
        print(f"\n{res['Measure']}:")
        print(f"  Low:  M = {res['Low_Mean']:7.2f}, SD = {res['Low_SD']:6.2f}")
        print(f"  High: M = {res['High_Mean']:7.2f}, SD = {res['High_SD']:6.2f}")
        print(f"  t = {res['t']:6.2f}, p_raw = {res['p']:.4f}, p_fdr = {res['p_fdr_adjusted']:.4f} [{sig_marker}], d = {res['Cohen_d']:5.2f}")

results_med_df.to_csv(output_dir / "extreme_group_median_results.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Method 3: 더 극단적 분할 (상위 30% vs 하위 30%)
# ============================================================================
print("\n" + "=" * 80)
print("Method 3: Top 30% vs Bottom 30%")
print("=" * 80)

p30 = master['ucla_total'].quantile(0.30)
p70 = master['ucla_total'].quantile(0.70)

low_group_30 = master[master['ucla_total'] <= p30].copy()
high_group_30 = master[master['ucla_total'] >= p70].copy()

print(f"\n저외로움 집단 (UCLA ≤ {p30:.0f}): N = {len(low_group_30)}")
print(f"고외로움 집단 (UCLA ≥ {p70:.0f}): N = {len(high_group_30)}")

results_30 = []

for measure, label in ef_measures.items():
    low_data = low_group_30[measure].dropna()
    high_data = high_group_30[measure].dropna()

    # CRITICAL FIX: Welch's t-test (does not assume equal variances)
    t_stat, p_val = stats.ttest_ind(low_data, high_data, equal_var=False)

    low_mean = low_data.mean()
    high_mean = high_data.mean()
    pooled_sd = np.sqrt(((len(low_data) - 1) * low_data.std()**2 +
                          (len(high_data) - 1) * high_data.std()**2) /
                         (len(low_data) + len(high_data) - 2))
    cohens_d = (high_mean - low_mean) / pooled_sd

    results_30.append({
        'Measure': label,
        't': t_stat,
        'p': p_val,
        'Cohen_d': cohens_d
    })

# CRITICAL FIX: Apply FDR correction to all 3 tests
results_30_df = pd.DataFrame(results_30)
p_vals_30 = results_30_df['p'].values
reject_30_fdr, p_adj_30_fdr = apply_multiple_comparison_correction(
    p_vals_30,
    method='fdr_bh',
    alpha=0.05
)

results_30_df['p_fdr_adjusted'] = p_adj_30_fdr
results_30_df['significant_fdr'] = reject_30_fdr

print("\n" + "-" * 80)
print("다중비교 보정 (FDR) for Top/Bottom 30% Split")
print("-" * 80)

for i, row in results_30_df.iterrows():
    sig_marker = "*" if row['significant_fdr'] else "ns"
    print(f"\n{row['Measure']}: t = {row['t']:.2f}, p_raw = {row['p']:.4f}, p_fdr = {row['p_fdr_adjusted']:.4f} [{sig_marker}], d = {row['Cohen_d']:.2f}")

# Save Method 3 results
results_30_df.to_csv(output_dir / "extreme_group_30pct_results.csv", index=False, encoding='utf-8-sig')
print(f"\n[OK] Saved (with FDR): {output_dir / 'extreme_group_30pct_results.csv'}")

# ============================================================================
# Visualization: Box plots
# ============================================================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

# Add group labels
low_group['group'] = 'Low Loneliness\n(Bottom 25%)'
high_group['group'] = 'High Loneliness\n(Top 25%)'
combined = pd.concat([low_group, high_group])

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (measure, label) in enumerate(ef_measures.items()):
    ax = axes[idx]

    # Box plot
    sns.boxplot(data=combined, x='group', y=measure, ax=ax, palette=['lightblue', 'lightcoral'])

    # Add individual points
    sns.stripplot(data=combined, x='group', y=measure, ax=ax,
                  color='black', alpha=0.3, size=4)

    # Add mean markers
    means = combined.groupby('group')[measure].mean()
    x_pos = [0, 1]
    ax.plot(x_pos, means, marker='D', color='red', markersize=8,
            linestyle='', label='Mean', zorder=10)

    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=13, fontweight='bold')

    # Add statistics text (using FDR-adjusted p-values)
    result = results_q_df[results_q_df['Measure'] == label].iloc[0]
    stats_text = f"t = {result['t']:.2f}\np_raw = {result['p']:.3f}\np_fdr = {result['p_fdr_adjusted']:.3f}\nd = {result['Cohen_d']:.2f}"

    if result['significant_fdr']:
        stats_text = "* " + stats_text
        ax.set_facecolor('#ffe6e6')  # Light red background for FDR-significant

    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "extreme_group_boxplots.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_dir / 'extreme_group_boxplots.png'}")
plt.close()

# ============================================================================
# Additional: Violin plots
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (measure, label) in enumerate(ef_measures.items()):
    ax = axes[idx]

    # Violin plot
    parts = ax.violinplot([low_group[measure].dropna(), high_group[measure].dropna()],
                           positions=[0, 1], showmeans=True, showmedians=True)

    # Color the violins
    colors = ['lightblue', 'lightcoral']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low\nLoneliness', 'High\nLoneliness'])
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "extreme_group_violinplots.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_dir / 'extreme_group_violinplots.png'}")
plt.close()

# ============================================================================
# Summary table
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Quartile Split Results (Main Analysis)")
print("=" * 80)

summary = results_q_df[results_q_df['Measure'].isin(ef_measures.values())].copy()
summary['Sig_raw'] = summary['p'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns')
summary['Sig_fdr'] = summary['p_fdr_adjusted'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns')

print("\n" + summary[['Measure', 'Low_Mean', 'High_Mean', 't', 'p', 'p_fdr_adjusted', 'Cohen_d', 'Sig_fdr']].to_string(index=False))

# Effect size interpretation
print("\n" + "=" * 80)
print("Effect Size Interpretation (Cohen's d)")
print("=" * 80)
print("  |d| < 0.2  : Negligible")
print("  |d| < 0.5  : Small")
print("  |d| < 0.8  : Medium")
print("  |d| ≥ 0.8  : Large")

print("\n" + "-" * 80)
for _, row in summary.iterrows():
    d = abs(row['Cohen_d'])
    if d < 0.2:
        interpretation = "Negligible"
    elif d < 0.5:
        interpretation = "Small"
    elif d < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"

    print(f"{row['Measure']:40s}: d = {row['Cohen_d']:6.2f} ({interpretation})")

# ============================================================================
# Final recommendations
# ============================================================================
print("\n" + "=" * 80)
print("INTERPRETATION & RECOMMENDATIONS")
print("=" * 80)

sig_count = (summary['significant_fdr']).sum()  # Use FDR-adjusted significance

if sig_count >= 2:
    print("\n[GOOD NEWS] 유의한 차이가 발견되었습니다!")
    print("  -> 극단 집단 비교에서 외로움의 효과가 나타남")
    print("  -> 이 결과는 논문에 포함할 가치가 있음")
elif sig_count == 1:
    print("\n[MIXED] 일부 지표에서만 유의한 차이")
    print("  -> 외로움 효과가 특정 집행기능에만 나타남")
    print("  -> 선택적 효과(selective effect)로 논의 가능")
else:
    print("\n[NO SIGNIFICANT DIFFERENCES]")
    print("  -> 극단 집단 비교에서도 차이가 없음")
    print("  -> 샘플 크기 증가 또는 측정 방법 재검토 필요")

print("\n다음 단계 제안:")

if sig_count > 0:
    print("1. 유의한 결과를 중심으로 논문 스토리 재구성")
    print("2. 산점도와 회귀선을 추가하여 관계 시각화")
    print("3. 조절/매개 분석으로 메커니즘 탐색")
    print("4. Q2-Q3 저널 타겟 (예: PLOS ONE, Acta Psychologica)")
else:
    print("1. 데이터 품질 재확인 (이상치, 불성실 응답)")
    print("2. 추가 데이터 수집 (N을 150명 이상으로)")
    print("3. 실험실 환경에서 일부 재검증")
    print("4. 온라인 vs 실험실 비교 연구로 전환")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print(f"\n결과 파일:")
print(f"  - extreme_group_quartile_results.csv")
print(f"  - extreme_group_median_results.csv")
print(f"  - extreme_group_boxplots.png")
print(f"  - extreme_group_violinplots.png")
print(f"\n저장 위치: {output_dir}")
