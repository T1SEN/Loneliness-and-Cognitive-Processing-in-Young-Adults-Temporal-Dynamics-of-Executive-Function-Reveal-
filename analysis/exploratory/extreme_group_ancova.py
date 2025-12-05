#!/usr/bin/env python3
"""
극단집단 비교 분석 (DASS/나이/성별 통제 ANCOVA)
==================================================

UCLA 외로움 상/하위 극단집단의 EF 차이를 공변량 통제하면서 비교.

- 컷오프: 25%, 15% 둘 다 분석
- 통제변수: DASS 3개 (우울/불안/스트레스), 나이, 성별
- 결과변수: WCST 7개 + Stroop + PRP
- p-value: 미보정 (raw) p-value 출력 (탐색적 분석)

Usage:
    python -m analysis.exploratory.extreme_group_ancova
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis.preprocessing import load_master_dataset, ANALYSIS_OUTPUT_DIR
from analysis.utils.modeling import standardize_predictors, prepare_gender_variable

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "exploratory_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OUTCOME VARIABLES (High Coverage >90%)
# =============================================================================

WCST_OUTCOMES = [
    ('wcst_accuracy', 'WCST Accuracy (%)'),
    ('pe_rate', 'Perseverative Error Rate (%)'),
    ('pe_count', 'Perseverative Error Count'),
    ('wcst_mean_rt', 'WCST Mean RT (ms)'),
    ('wcst_sd_rt', 'WCST RT Variability (SD)'),
]

STROOP_OUTCOMES = [
    ('stroop_interference', 'Stroop Interference (ms)'),
    ('accuracy_incongruent', 'Stroop Accuracy Incongruent'),
    ('accuracy_congruent', 'Stroop Accuracy Congruent'),
    ('rt_mean_incongruent', 'Stroop RT Incongruent (ms)'),
    ('rt_mean_congruent', 'Stroop RT Congruent (ms)'),
]

PRP_OUTCOMES = [
    ('prp_bottleneck', 'PRP Bottleneck (ms)'),
    ('t2_rt_mean_short', 'PRP T2 RT Short SOA (ms)'),
    ('t2_rt_mean_long', 'PRP T2 RT Long SOA (ms)'),
    ('t2_rt_sd_short', 'PRP T2 RT SD Short SOA'),
    ('t2_rt_sd_long', 'PRP T2 RT SD Long SOA'),
]

ALL_OUTCOMES = WCST_OUTCOMES + STROOP_OUTCOMES + PRP_OUTCOMES


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_extreme_group_ancova(cutoff: float, master: pd.DataFrame) -> pd.DataFrame:
    """
    Run ANCOVA for extreme groups at a given cutoff.

    Args:
        cutoff: Percentile cutoff (e.g., 0.25 for top/bottom 25%)
        master: Master dataset with standardized predictors

    Returns:
        DataFrame with results
    """
    q_low = master['ucla_total'].quantile(cutoff)
    q_high = master['ucla_total'].quantile(1 - cutoff)

    # Define extreme groups
    extreme = master[
        (master['ucla_total'] <= q_low) | (master['ucla_total'] >= q_high)
    ].copy()
    extreme['group_high'] = (extreme['ucla_total'] >= q_high).astype(int)

    # Group info
    low_group = extreme[extreme['group_high'] == 0]
    high_group = extreme[extreme['group_high'] == 1]

    print(f"\n저외로움 (UCLA ≤ {q_low:.1f}): N = {len(low_group)}, UCLA M = {low_group['ucla_total'].mean():.1f}")
    print(f"고외로움 (UCLA ≥ {q_high:.1f}): N = {len(high_group)}, UCLA M = {high_group['ucla_total'].mean():.1f}")

    results = []

    for col, label in ALL_OUTCOMES:
        if col not in extreme.columns:
            print(f"  [SKIP] {col} not found in data")
            continue

        # Drop missing for this outcome
        df = extreme.dropna(subset=[col, 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male'])

        if len(df) < 20:
            print(f"  [SKIP] {col}: N = {len(df)} < 20")
            continue

        # Compute descriptive stats
        low_data = df[df['group_high'] == 0][col]
        high_data = df[df['group_high'] == 1][col]

        # ANCOVA model
        formula = f"{col} ~ group_high + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)"
        try:
            model = smf.ols(formula, data=df).fit(cov_type='HC3')

            # Cohen's d from adjusted means (approximate using residuals)
            residuals = model.resid
            pooled_sd = residuals.std()
            cohens_d = model.params['group_high'] / pooled_sd if pooled_sd > 0 else np.nan

            results.append({
                'outcome': col,
                'label': label,
                'n_low': len(low_data),
                'n_high': len(high_data),
                'low_mean': low_data.mean(),
                'low_sd': low_data.std(),
                'high_mean': high_data.mean(),
                'high_sd': high_data.std(),
                'beta': model.params['group_high'],
                'se': model.bse['group_high'],
                't': model.tvalues['group_high'],
                'p': model.pvalues['group_high'],
                'cohens_d': cohens_d,
            })
        except Exception as e:
            print(f"  [ERROR] {col}: {e}")

    if not results:
        return pd.DataFrame()

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    return df_results


def format_sig(p):
    """Format significance marker."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''


def print_results_table(df: pd.DataFrame, title: str):
    """Print results in table format."""
    print(f"\n{title}")
    print("-" * 110)

    header = f"{'변수':<28} | {'Low M (SD)':<16} | {'High M (SD)':<16} | {'β_adj':>9} | {'t':>7} | {'p':>8} | {'d':>6} | sig"
    print(header)
    print("-" * 110)

    for _, row in df.iterrows():
        # Format based on magnitude
        if abs(row['low_mean']) > 100:
            low_str = f"{row['low_mean']:.1f} ({row['low_sd']:.1f})"
            high_str = f"{row['high_mean']:.1f} ({row['high_sd']:.1f})"
        else:
            low_str = f"{row['low_mean']:.2f} ({row['low_sd']:.2f})"
            high_str = f"{row['high_mean']:.2f} ({row['high_sd']:.2f})"

        sig = format_sig(row['p'])

        line = f"{row['label']:<28} | {low_str:<16} | {high_str:<16} | {row['beta']:>9.3f} | {row['t']:>7.2f} | {row['p']:>8.4f} | {row['cohens_d']:>6.2f} | {sig}"
        print(line)

    print("-" * 110)


def main():
    """Run extreme group ANCOVA analysis."""
    print("=" * 80)
    print("극단집단 비교 (DASS/나이/성별 통제 ANCOVA)")
    print("=" * 80)

    # Load data
    print("\n[1] Loading master dataset...")
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Prepare variables
    master = standardize_predictors(master)
    master = prepare_gender_variable(master)

    print(f"Total N = {len(master)}")
    print(f"UCLA range: {master['ucla_total'].min():.0f} - {master['ucla_total'].max():.0f}")

    # Run for both cutoffs
    cutoffs = [0.25, 0.15]
    all_results = {}

    for cutoff in cutoffs:
        pct = int(cutoff * 100)
        print(f"\n{'='*80}")
        print(f"[컷오프 {pct}%] 상위 {pct}% vs 하위 {pct}%")
        print("=" * 80)

        df_results = run_extreme_group_ancova(cutoff, master)

        if len(df_results) > 0:
            # Separate by category
            wcst_cols = [c for c, _ in WCST_OUTCOMES]
            stroop_cols = [c for c, _ in STROOP_OUTCOMES]
            prp_cols = [c for c, _ in PRP_OUTCOMES]

            df_wcst = df_results[df_results['outcome'].isin(wcst_cols)]
            df_stroop = df_results[df_results['outcome'].isin(stroop_cols)]
            df_prp = df_results[df_results['outcome'].isin(prp_cols)]

            if len(df_wcst) > 0:
                print_results_table(df_wcst, f"--- WCST 변수 (컷오프 {pct}%) ---")
            if len(df_stroop) > 0:
                print_results_table(df_stroop, f"--- Stroop 변수 (컷오프 {pct}%) ---")
            if len(df_prp) > 0:
                print_results_table(df_prp, f"--- PRP 변수 (컷오프 {pct}%) ---")

            all_results[pct] = df_results

            # Save to CSV
            df_results.to_csv(OUTPUT_DIR / f"extreme_group_ancova_{pct}pct.csv",
                            index=False, encoding='utf-8-sig')

    # Summary
    print("\n" + "=" * 80)
    print("요약")
    print("=" * 80)
    print(f"\n{'컷오프':<10} | {'N (Low/High)':<15} | {'p<.05':<15}")
    print("-" * 45)

    for pct, df in all_results.items():
        n_low = df['n_low'].iloc[0] if len(df) > 0 else 0
        n_high = df['n_high'].iloc[0] if len(df) > 0 else 0
        n_sig_raw = (df['p'] < 0.05).sum()
        print(f"{pct}%{'':<7} | {n_low}/{n_high:<13} | {n_sig_raw}개")

    print("\n" + "=" * 80)
    print("분석 완료!")
    print(f"결과 저장: {OUTPUT_DIR}")
    print("=" * 80)

    return all_results


def run(analysis: str = None, verbose: bool = True):
    """
    Run extreme group ANCOVA analysis.

    This is a wrapper for main() to conform to the suite interface.

    Parameters
    ----------
    analysis : str, optional
        Not used (single analysis only). Kept for interface compatibility.
    verbose : bool
        Print progress (always True for this analysis).

    Returns
    -------
    dict
        Results dictionary from main().
    """
    return main()


if __name__ == "__main__":
    main()
