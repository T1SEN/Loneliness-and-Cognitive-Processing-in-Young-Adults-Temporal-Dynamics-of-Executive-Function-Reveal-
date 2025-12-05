"""
Bayesian Equivalence Analysis Suite (ROPE)
==========================================

베이지안 계층 모델 + ROPE 분석

연구 질문: UCLA 효과가 DASS 통제 후 실질적으로 0과 동등한지 확립할 수 있는가?

통계 방법:
- 베이지안 계층 회귀
- ROPE (실질적 동등 영역) = ±0.1 SD
- UCLA 효과의 ROPE 내 사후 확률 계산
- Bayes Factor for H0 (null effect) vs H1 (non-zero effect)

Usage:
    python -m analysis.validation.bayesian_equivalence_suite
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from analysis.preprocessing import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "bayesian_equivalence_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_master_data() -> pd.DataFrame:
    """Load and prepare master dataset."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    return master


# =============================================================================
# ROPE ANALYSIS (Frequentist Approximation)
# =============================================================================

def compute_rope_probability(
    beta: float,
    se: float,
    rope_low: float = -0.1,
    rope_high: float = 0.1,
    n_samples: int = 10000
) -> Dict:
    """
    ROPE 확률 계산 (정규 분포 근사)

    Parameters:
    -----------
    beta : float
        추정된 계수
    se : float
        표준 오차
    rope_low, rope_high : float
        ROPE 범위 (기본값: ±0.1 SD)

    Returns:
    --------
    Dict with ROPE probability and HDI
    """
    # 사후 분포 시뮬레이션 (정규 분포 근사)
    posterior = np.random.normal(beta, se, n_samples)

    # ROPE 내 확률
    in_rope = np.mean((posterior >= rope_low) & (posterior <= rope_high))

    # HDI (Highest Density Interval)
    hdi_low = np.percentile(posterior, 2.5)
    hdi_high = np.percentile(posterior, 97.5)

    # ROPE와 HDI 겹침 여부
    hdi_in_rope = (hdi_low >= rope_low) and (hdi_high <= rope_high)

    # 효과 방향 확률
    prob_positive = np.mean(posterior > 0)
    prob_negative = np.mean(posterior < 0)

    return {
        'beta': beta,
        'se': se,
        'rope_low': rope_low,
        'rope_high': rope_high,
        'p_in_rope': in_rope,
        'p_below_rope': np.mean(posterior < rope_low),
        'p_above_rope': np.mean(posterior > rope_high),
        'hdi_low': hdi_low,
        'hdi_high': hdi_high,
        'hdi_in_rope': hdi_in_rope,
        'prob_positive': prob_positive,
        'prob_negative': prob_negative,
        'conclusion': _interpret_rope(in_rope, hdi_in_rope, hdi_low, hdi_high, rope_low, rope_high)
    }


def _interpret_rope(
    p_in_rope: float,
    hdi_in_rope: bool,
    hdi_low: float,
    hdi_high: float,
    rope_low: float,
    rope_high: float
) -> str:
    """ROPE 분석 결과 해석"""
    if hdi_in_rope:
        return "PRACTICALLY EQUIVALENT (HDI entirely in ROPE)"
    elif p_in_rope > 0.95:
        return "PRACTICALLY EQUIVALENT (>95% in ROPE)"
    elif p_in_rope < 0.05:
        if hdi_low > rope_high:
            return "PRACTICALLY SIGNIFICANT (positive effect)"
        elif hdi_high < rope_low:
            return "PRACTICALLY SIGNIFICANT (negative effect)"
        else:
            return "NOT EQUIVALENT but uncertain direction"
    else:
        return "UNDECIDED (insufficient evidence)"


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_rope_equivalence(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    ROPE 동등성 분석

    각 EF 결과 변수에 대해 UCLA 효과의 실질적 동등성 검정
    """
    output_dir = OUTPUT_DIR / "rope"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[ROPE EQUIVALENCE] Testing practical equivalence of UCLA effects...")
        print("  ROPE = ±0.1 SD (small effect threshold)")

    master = load_master_data()

    import statsmodels.formula.api as smf

    ef_outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('wcst_accuracy', 'WCST Accuracy'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck', 'PRP Bottleneck')
    ]

    all_results = []
    significant_findings = []

    for outcome, label in ef_outcomes:
        if outcome not in master.columns:
            continue

        clean_data = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

        if len(clean_data) < 50:
            continue

        # 결과 변수 표준화 (ROPE를 SD 단위로 해석하기 위해)
        outcome_std = (clean_data[outcome] - clean_data[outcome].mean()) / clean_data[outcome].std()
        clean_data['outcome_z'] = outcome_std

        try:
            formula = "outcome_z ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

            beta = model.params.get('z_ucla', np.nan)
            se = model.bse.get('z_ucla', np.nan)

            # ROPE 분석
            rope_result = compute_rope_probability(beta, se, rope_low=-0.1, rope_high=0.1)

            result = {
                'outcome': outcome,
                'outcome_label': label,
                'n': len(clean_data),
                'ucla_beta': beta,
                'ucla_se': se,
                'ucla_p': model.pvalues.get('z_ucla', np.nan),
                **rope_result,
                'interaction_beta': model.params.get('z_ucla:C(gender_male)[T.1]', np.nan),
                'interaction_p': model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)
            }
            all_results.append(result)

            if verbose:
                print(f"\n  {label}:")
                print(f"    UCLA β = {beta:.4f} (SE = {se:.4f})")
                print(f"    95% HDI: [{rope_result['hdi_low']:.4f}, {rope_result['hdi_high']:.4f}]")
                print(f"    P(in ROPE): {rope_result['p_in_rope']*100:.1f}%")
                print(f"    Conclusion: {rope_result['conclusion']}")

            # 유의한 발견 기록
            if rope_result['conclusion'].startswith("PRACTICALLY EQUIVALENT"):
                significant_findings.append({
                    'analysis': 'ROPE Equivalence',
                    'outcome': label,
                    'effect': f"UCLA β={beta:.4f}, P(ROPE)={rope_result['p_in_rope']*100:.1f}%",
                    'beta': beta,
                    'p': rope_result['p_in_rope'],
                    'n': len(clean_data)
                })

        except Exception as e:
            if verbose:
                print(f"  {label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "rope_equivalence_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_bayes_factor(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Bayes Factor 분석 (BIC 근사)

    H0 (UCLA 효과 = 0) vs H1 (UCLA 효과 ≠ 0) 비교
    """
    output_dir = OUTPUT_DIR / "bayes_factor"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[BAYES FACTOR] Comparing null vs alternative models...")

    master = load_master_data()

    import statsmodels.formula.api as smf

    ef_outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('wcst_accuracy', 'WCST Accuracy'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck', 'PRP Bottleneck')
    ]

    all_results = []
    significant_findings = []

    for outcome, label in ef_outcomes:
        if outcome not in master.columns:
            continue

        clean_data = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

        if len(clean_data) < 50:
            continue

        try:
            # 전체 모델 (UCLA 포함)
            formula_full = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model_full = smf.ols(formula_full, data=clean_data).fit()

            # 축소 모델 (UCLA 제외)
            formula_null = f"{outcome} ~ C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model_null = smf.ols(formula_null, data=clean_data).fit()

            # BIC 기반 Bayes Factor 근사
            # BF01 = exp((BIC_full - BIC_null) / 2)
            # BF01 > 1: H0 지지 (UCLA 효과 없음)
            # BF01 < 1: H1 지지 (UCLA 효과 있음)
            bic_diff = model_full.bic - model_null.bic
            bf01 = np.exp(bic_diff / 2)
            bf10 = 1 / bf01

            # 해석
            if bf01 > 100:
                interpretation = "Decisive evidence for H0 (no UCLA effect)"
            elif bf01 > 10:
                interpretation = "Strong evidence for H0"
            elif bf01 > 3:
                interpretation = "Moderate evidence for H0"
            elif bf01 > 1:
                interpretation = "Weak evidence for H0"
            elif bf10 > 100:
                interpretation = "Decisive evidence for H1 (UCLA effect exists)"
            elif bf10 > 10:
                interpretation = "Strong evidence for H1"
            elif bf10 > 3:
                interpretation = "Moderate evidence for H1"
            else:
                interpretation = "Weak evidence for H1"

            result = {
                'outcome': outcome,
                'outcome_label': label,
                'n': len(clean_data),
                'bic_full': model_full.bic,
                'bic_null': model_null.bic,
                'bic_diff': bic_diff,
                'bf01': bf01,
                'bf10': bf10,
                'log_bf01': np.log10(bf01),
                'interpretation': interpretation,
                'r2_full': model_full.rsquared,
                'r2_null': model_null.rsquared,
                'delta_r2': model_full.rsquared - model_null.rsquared
            }
            all_results.append(result)

            if verbose:
                print(f"\n  {label}:")
                print(f"    BIC (with UCLA): {model_full.bic:.2f}")
                print(f"    BIC (without UCLA): {model_null.bic:.2f}")
                print(f"    BF01 = {bf01:.3f} (log10 = {np.log10(bf01):.2f})")
                print(f"    Interpretation: {interpretation}")
                print(f"    ΔR² = {(model_full.rsquared - model_null.rsquared)*100:.2f}%")

            if bf01 > 3:
                significant_findings.append({
                    'analysis': 'Bayes Factor',
                    'outcome': label,
                    'effect': interpretation,
                    'beta': bf01,
                    'p': np.nan,
                    'n': len(clean_data)
                })

        except Exception as e:
            if verbose:
                print(f"  {label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "bayes_factor_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_effect_precision(verbose: bool = True) -> pd.DataFrame:
    """
    효과 정밀도 분석

    현재 표본 크기로 탐지 가능한 최소 효과 크기 추정
    """
    output_dir = OUTPUT_DIR / "precision"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[EFFECT PRECISION] Estimating detectable effect sizes...")

    master = load_master_data()

    import statsmodels.formula.api as smf

    ef_outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('wcst_accuracy', 'WCST Accuracy'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck', 'PRP Bottleneck')
    ]

    all_results = []

    for outcome, label in ef_outcomes:
        if outcome not in master.columns:
            continue

        clean_data = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

        if len(clean_data) < 50:
            continue

        # 결과 변수 표준화
        outcome_std = (clean_data[outcome] - clean_data[outcome].mean()) / clean_data[outcome].std()
        clean_data['outcome_z'] = outcome_std

        try:
            formula = "outcome_z ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

            se = model.bse.get('z_ucla', np.nan)

            # 80% 검정력으로 탐지 가능한 최소 효과 크기 (양측 검정, α=0.05)
            # d = 2.8 * SE (근사)
            min_detectable = 2.8 * se

            # 현재 관측된 효과의 90% CI
            beta = model.params.get('z_ucla', np.nan)
            ci90_low = beta - 1.645 * se
            ci90_high = beta + 1.645 * se

            result = {
                'outcome': outcome,
                'outcome_label': label,
                'n': len(clean_data),
                'observed_beta': beta,
                'se': se,
                'min_detectable_effect': min_detectable,
                'ci90_low': ci90_low,
                'ci90_high': ci90_high,
                'effect_is_precise': min_detectable < 0.2,  # Cohen's d < 0.2 = small
                'width_90ci': ci90_high - ci90_low
            }
            all_results.append(result)

            if verbose:
                precision = "HIGH" if result['effect_is_precise'] else "LOW"
                print(f"\n  {label}:")
                print(f"    Observed β = {beta:.4f} (SE = {se:.4f})")
                print(f"    90% CI: [{ci90_low:.4f}, {ci90_high:.4f}]")
                print(f"    Min detectable effect (80% power): {min_detectable:.4f}")
                print(f"    Precision: {precision}")

        except Exception as e:
            if verbose:
                print(f"  {label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "effect_precision_results.csv", index=False, encoding='utf-8-sig')

    return results_df


# =============================================================================
# RESULTS RECORDING
# =============================================================================

def record_significant_results(findings: List[Dict], verbose: bool = True):
    """유의한 결과를 Results.md에 기록"""
    if not findings:
        return

    results_file = Path("Results.md")

    existing_content = ""
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    new_entries = []
    for finding in findings:
        p_str = f"p={finding['p']:.4f}" if not np.isnan(finding.get('p', np.nan)) else "N/A"
        beta_str = f"BF01={finding['beta']:.2f}" if 'Bayes' in finding['analysis'] else f"P(ROPE)={finding['p']*100:.1f}%"
        entry = f"| {today} | {finding['analysis']} | {finding['outcome']} | {finding['effect']} | {beta_str} | {p_str} | N={finding['n']} |"
        if entry not in existing_content:
            new_entries.append(entry)

    if new_entries:
        with open(results_file, 'a', encoding='utf-8') as f:
            for entry in new_entries:
                f.write(entry + "\n")
        if verbose:
            print(f"\n  Recorded {len(new_entries)} findings to Results.md")


# =============================================================================
# MAIN
# =============================================================================

ANALYSES = {
    'rope': ('ROPE practical equivalence test', analyze_rope_equivalence),
    'bayes_factor': ('Bayes Factor model comparison', analyze_bayes_factor),
    'precision': ('Effect precision analysis', analyze_effect_precision),
}


def run(analysis: Optional[str] = None, verbose: bool = True, record_results: bool = True) -> Dict:
    """Run Bayesian equivalence analyses."""
    if verbose:
        print("=" * 70)
        print("BAYESIAN EQUIVALENCE ANALYSIS SUITE (ROPE)")
        print("=" * 70)

    results = {}
    all_significant = []

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        desc, func = ANALYSES[analysis]
        result = func(verbose=verbose)
        if isinstance(result, tuple):
            results[analysis], findings = result
            all_significant.extend(findings)
        else:
            results[analysis] = result
    else:
        for name, (desc, func) in ANALYSES.items():
            try:
                result = func(verbose=verbose)
                if isinstance(result, tuple):
                    results[name], findings = result
                    all_significant.extend(findings)
                else:
                    results[name] = result
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if record_results and all_significant:
        record_significant_results(all_significant, verbose=verbose)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Output saved to: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Equivalence Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    args = parser.parse_args()

    run(analysis=args.analysis, verbose=not args.quiet, record_results=not args.no_record)
