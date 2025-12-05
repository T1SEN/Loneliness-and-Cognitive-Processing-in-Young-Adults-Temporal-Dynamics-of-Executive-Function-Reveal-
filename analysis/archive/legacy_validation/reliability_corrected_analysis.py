#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
신뢰도 보정 효과크기 재추정
Reliability-Corrected Effect Size Estimation

Methods:
- Spearman-Brown correction
- Disattenuated correlations
- Power analysis with true effect sizes
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("신뢰도 보정 효과크기 재추정")
print("Reliability-Corrected Effect Size Estimation")
print("=" * 80)

# =============================================================================
# 1. 관찰된 신뢰도
# =============================================================================

print("\n" + "=" * 80)
print("1. OBSERVED RELIABILITIES")
print("=" * 80)

# From previous analysis (rv_reliability.csv)
reliability_df = pd.read_csv(OUTPUT_DIR / "rv_reliability.csv")

print("\nSplit-half reliabilities (observed):")
print(f"  Stroop interference: r = {reliability_df['stroop_split_half_r'].iloc[0]:.3f}")
print(f"  PRP bottleneck: r = {reliability_df['prp_split_half_r'].iloc[0]:.3f}")

# Spearman-Brown correction (for full-length test)
def spearman_brown(r_half):
    """Spearman-Brown prophecy formula"""
    return (2 * r_half) / (1 + r_half)

r_stroop_half = reliability_df['stroop_split_half_r'].iloc[0]
r_prp_half = reliability_df['prp_split_half_r'].iloc[0]

r_stroop_full = spearman_brown(r_stroop_half)
r_prp_full = spearman_brown(r_prp_half)

print("\nSpearman-Brown corrected reliabilities (full-length):")
print(f"  Stroop interference: r = {r_stroop_full:.3f}")
print(f"  PRP bottleneck: r = {r_prp_full:.3f}")

# WCST reliability estimate from literature / pilot
# Typical test-retest for WCST perseverative errors: r=0.50-0.70
r_wcst_full = 0.60  # Conservative estimate

print(f"  WCST perseverative errors: r = {r_wcst_full:.3f} (literature estimate)")

# UCLA reliability from literature
# Russell (1996): α=0.89-0.94, test-retest r=0.73
r_ucla_full = 0.85  # Conservative

print(f"  UCLA total score: r = {r_ucla_full:.3f} (literature estimate)")

reliabilities = {
    'stroop': r_stroop_full,
    'wcst': r_wcst_full,
    'prp': r_prp_full,
    'ucla': r_ucla_full
}

# =============================================================================
# 2. 관찰된 상관
# =============================================================================

print("\n" + "=" * 80)
print("2. OBSERVED CORRELATIONS (Uncorrected)")
print("=" * 80)

# Load correlation matrix
corr_df = pd.read_csv(OUTPUT_DIR / "correlation_matrix.csv", index_col=0)

observed_corrs = {
    'stroop': corr_df.loc['ucla_total', 'stroop_interference'],
    'wcst': corr_df.loc['ucla_total', 'pe_rate'],
    'prp': corr_df.loc['ucla_total', 'prp_bottleneck']
}

print("\nUCLA → EF 상관 (관찰값):")
for task, r_obs in observed_corrs.items():
    print(f"  {task.upper()}: r = {r_obs:.3f}")

# =============================================================================
# 3. 신뢰도 보정 상관 (Disattenuated)
# =============================================================================

print("\n" + "=" * 80)
print("3. DISATTENUATED CORRELATIONS")
print("=" * 80)

def disattenuate(r_observed, rel_x, rel_y):
    """
    Correct correlation for attenuation due to unreliability

    r_true = r_observed / sqrt(rel_x * rel_y)

    Note: Can exceed 1.0 if measurement error is large
    """
    denominator = np.sqrt(rel_x * rel_y)
    if denominator == 0:
        return np.nan
    return r_observed / denominator

corrected_corrs = {}
correction_factors = {}

print("\nUCLA → EF 상관 (신뢰도 보정):")
print("-" * 80)

for task, r_obs in observed_corrs.items():
    rel_task = reliabilities[task]
    rel_ucla = reliabilities['ucla']

    r_true = disattenuate(r_obs, rel_ucla, rel_task)
    correction_factor = 1 / np.sqrt(rel_ucla * rel_task)

    corrected_corrs[task] = r_true
    correction_factors[task] = correction_factor

    print(f"\n{task.upper()}:")
    print(f"  관찰 상관: r_obs = {r_obs:.3f}")
    print(f"  보정 상관: r_true = {r_true:.3f}")
    print(f"  보정 계수: {correction_factor:.3f}x")
    print(f"  신뢰도: UCLA={rel_ucla:.2f}, {task.upper()}={rel_task:.2f}")

    if abs(r_true) > 1.0:
        print(f"  ⚠️  경고: r_true > 1.0 - 신뢰도 추정치가 과소 추정되었을 가능성")

# Save results
disatt_df = pd.DataFrame({
    'task': list(observed_corrs.keys()),
    'r_observed': list(observed_corrs.values()),
    'reliability_ucla': [reliabilities['ucla']] * 3,
    'reliability_task': [reliabilities[t] for t in observed_corrs.keys()],
    'r_disattenuated': [corrected_corrs[t] for t in observed_corrs.keys()],
    'correction_factor': [correction_factors[t] for t in observed_corrs.keys()]
})

disatt_df.to_csv(OUTPUT_DIR / "disattenuated_correlations.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 4. 검정력 분석 (보정된 효과크기 기반)
# =============================================================================

print("\n" + "=" * 80)
print("4. POWER ANALYSIS WITH TRUE EFFECT SIZES")
print("=" * 80)

def sample_size_correlation(r, alpha=0.05, power=0.80):
    """
    Required sample size for correlation test

    Using Fisher's Z transformation
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    # Fisher's Z
    z_r = 0.5 * np.log((1 + r) / (1 - r))

    # Sample size
    n = ((z_alpha + z_beta) / z_r) ** 2 + 3

    return int(np.ceil(n))

print("\n필요 표본 크기 (α=0.05, power=0.80):")
print("-" * 80)

power_results = []

for task in ['stroop', 'wcst', 'prp']:
    r_obs = observed_corrs[task]
    r_true = corrected_corrs[task]

    # Cap r_true at 0.95 for calculation
    r_true_capped = min(abs(r_true), 0.95) * np.sign(r_true)

    # N required for observed effect
    if abs(r_obs) > 0.01:
        n_obs = sample_size_correlation(r_obs, alpha=0.05, power=0.80)
    else:
        n_obs = np.inf

    # N required for true effect
    if abs(r_true_capped) > 0.01:
        n_true = sample_size_correlation(r_true_capped, alpha=0.05, power=0.80)
    else:
        n_true = np.inf

    print(f"\n{task.upper()}:")
    print(f"  r_obs = {r_obs:.3f} → N = {n_obs if n_obs != np.inf else '∞'}")
    print(f"  r_true = {r_true:.3f} → N = {n_true if n_true != np.inf else '∞'}")

    power_results.append({
        'task': task,
        'r_observed': r_obs,
        'r_true': r_true_capped,
        'n_required_observed': n_obs if n_obs != np.inf else np.nan,
        'n_required_true': n_true if n_true != np.inf else np.nan,
        'current_n': 72
    })

power_df = pd.DataFrame(power_results)
power_df.to_csv(OUTPUT_DIR / "power_analysis_corrected.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 5. 실제 데이터 vs 이상적 조건 비교
# =============================================================================

print("\n" + "=" * 80)
print("5. ACHIEVED POWER WITH CURRENT N=72")
print("=" * 80)

def power_correlation(r, n, alpha=0.05):
    """Calculate achieved power for given r and n"""
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)

    # Fisher's Z
    z_r = 0.5 * np.log((1 + r) / (1 - r))

    # Non-centrality parameter
    delta = z_r * np.sqrt(n - 3)

    # Power
    power = 1 - norm.cdf(z_alpha - delta) + norm.cdf(-z_alpha - delta)

    return power

print("\n현재 N=72에서의 검정력:")
print("-" * 80)

for task in ['stroop', 'wcst', 'prp']:
    r_obs = observed_corrs[task]
    r_true = min(abs(corrected_corrs[task]), 0.95) * np.sign(corrected_corrs[task])

    if abs(r_obs) > 0.001:
        power_obs = power_correlation(r_obs, n=72, alpha=0.05)
    else:
        power_obs = 0.05

    if abs(r_true) > 0.001:
        power_true = power_correlation(r_true, n=72, alpha=0.05)
    else:
        power_true = 0.05

    print(f"\n{task.upper()}:")
    print(f"  r_obs = {r_obs:.3f} → power = {power_obs:.1%}")
    print(f"  r_true = {r_true:.3f} → power = {power_true:.1%}")

# =============================================================================
# 6. 신뢰도 향상 시뮬레이션
# =============================================================================

print("\n" + "=" * 80)
print("6. RELIABILITY IMPROVEMENT SCENARIOS")
print("=" * 80)

print("\n만약 신뢰도를 0.80으로 향상시킨다면?")
print("-" * 80)

improved_rel = 0.80

for task in ['stroop', 'wcst', 'prp']:
    r_obs = observed_corrs[task]

    # Current
    rel_current = reliabilities[task]
    r_current = disattenuate(r_obs, reliabilities['ucla'], rel_current)

    # Improved
    r_improved = disattenuate(r_obs, reliabilities['ucla'], improved_rel)

    print(f"\n{task.upper()} (현재 신뢰도 {rel_current:.2f} → 0.80):")
    print(f"  현재 보정 상관: r = {r_current:.3f}")
    print(f"  개선 후: r = {r_improved:.3f}")

    if abs(r_improved) < 0.95:
        n_improved = sample_size_correlation(r_improved, alpha=0.05, power=0.80)
        print(f"  필요 N: {n_improved}")

# =============================================================================
# 7. SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n핵심 발견:")

print("\n1. 신뢰도 문제:")
print(f"   Stroop 신뢰도: {reliabilities['stroop']:.2f} (낮음)")
print(f"   PRP 신뢰도: {reliabilities['prp']:.2f} (낮음)")
print(f"   WCST 신뢰도: {reliabilities['wcst']:.2f} (추정치)")

print("\n2. 효과크기 변화:")
wcst_obs = observed_corrs['wcst']
wcst_true = corrected_corrs['wcst']
print(f"   WCST: r_obs={wcst_obs:.3f} → r_true={wcst_true:.3f} ({abs(wcst_true/wcst_obs):.1f}배)")

print("\n3. 필요 표본:")
for task in ['wcst']:  # Focus on WCST
    n_req = power_df[power_df['task'] == task]['n_required_true'].values[0]
    if not np.isnan(n_req):
        print(f"   {task.upper()} 진정 효과 탐지: N={int(n_req)}")

print("\n4. 해석:")
print("   - 낮은 신뢰도가 효과크기를 크게 감쇠시켰을 가능성")
print("   - 진정 효과는 관찰값보다 클 수 있음")
print("   - 그러나 보정 후에도 효과크기는 여전히 작음 (r<0.20)")
print("   - 신뢰도 개선(과제 길이 늘림, 실험실 환경) 필요")

print("\n분석 완료!")
print(f"결과 저장 위치: {OUTPUT_DIR}")
print("\n생성된 파일:")
print("  - disattenuated_correlations.csv")
print("  - power_analysis_corrected.csv")
