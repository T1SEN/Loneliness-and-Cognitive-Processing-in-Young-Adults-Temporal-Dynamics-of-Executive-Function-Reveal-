"""
Developmental Window Analysis Suite
====================================

연령 층화 공식 분석 (조절된 조절)

연구 질문: UCLA × 성별 효과가 젊은 성인(18-24)에서 더 강한가?

통계 방법:
- 발달적 연령 그룹: 18-21, 22-25, 26-36
- 각 연령층 내 UCLA × 성별 상호작용 검정
- Johnson-Neyman 기법으로 유의한 연령 범위 식별

Usage:
    python -m analysis.advanced.developmental_window_suite
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
import statsmodels.formula.api as smf
import statsmodels.api as sm

from analysis.preprocessing import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR, find_interaction_term
)
from analysis.utils.modeling import standardize_predictors

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "developmental_window_suite"
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


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    연령 그룹 생성

    발달적 연령 그룹:
    - 18-21: 초기 성인기 (대학 초년)
    - 22-25: 중기 성인기 (대학 후반/취업 초기)
    - 26+: 후기 성인기
    """
    df = df.copy()

    # 연령 그룹 (카테고리)
    bins = [17, 21, 25, 100]
    labels = ['18-21', '22-25', '26+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    # 연령 이분법 (중위수 기준)
    age_median = df['age'].median()
    df['age_binary'] = np.where(df['age'] <= age_median, 'younger', 'older')

    # 연속형 나이 (중심화)
    df['age_centered'] = df['age'] - df['age'].mean()

    return df


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_age_stratified_effects(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    연령층별 UCLA × 성별 상호작용 검정

    각 연령 그룹(18-21, 22-25, 26+)에서 UCLA × 성별 효과 비교
    """
    output_dir = OUTPUT_DIR / "stratified"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[AGE-STRATIFIED ANALYSIS] Testing UCLA × Gender within age groups...")

    master = load_master_data()
    master = create_age_groups(master)

    # EF 결과 변수들
    ef_outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('wcst_accuracy', 'WCST Accuracy'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck', 'PRP Bottleneck')
    ]

    all_results = []
    significant_findings = []

    for age_group in ['18-21', '22-25', '26+']:
        subset = master[master['age_group'] == age_group].copy()

        if len(subset) < 20:
            if verbose:
                print(f"  {age_group}: N={len(subset)} (insufficient)")
            continue

        if verbose:
            print(f"\n  Age Group: {age_group} (N={len(subset)})")
            print("  " + "-" * 50)

        for outcome, label in ef_outcomes:
            if outcome not in subset.columns:
                continue

            clean_data = subset.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', outcome])

            if len(clean_data) < 15:
                continue

            try:
                # 연령 그룹 내에서는 나이 통제 불필요
                formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str"
                model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                result = {
                    'age_group': age_group,
                    'outcome': outcome,
                    'outcome_label': label,
                    'n': len(clean_data),
                    'n_male': (clean_data['gender_male'] == 1).sum(),
                    'n_female': (clean_data['gender_male'] == 0).sum(),
                    'ucla_beta': model.params.get('z_ucla', np.nan),
                    'ucla_se': model.bse.get('z_ucla', np.nan),
                    'ucla_p': model.pvalues.get('z_ucla', np.nan),
                }
                int_term = find_interaction_term(model.params.index)
                result.update({
                    'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
                    'interaction_se': model.bse.get(int_term, np.nan) if int_term else np.nan,
                    'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
                    'gender_beta': model.params.get('C(gender_male)[T.1]', np.nan),
                    'gender_p': model.pvalues.get('C(gender_male)[T.1]', np.nan),
                    'r_squared': model.rsquared
                })
                all_results.append(result)

                # Check significance
                if result['ucla_p'] < 0.05:
                    significant_findings.append({
                        'analysis': f'Age-Stratified ({age_group})',
                        'outcome': label,
                        'effect': 'UCLA main',
                        'beta': result['ucla_beta'],
                        'p': result['ucla_p'],
                        'n': result['n']
                    })
                    if verbose:
                        print(f"    * {label}: UCLA β={result['ucla_beta']:.3f}, p={result['ucla_p']:.4f}")

                if result['interaction_p'] < 0.05:
                    significant_findings.append({
                        'analysis': f'Age-Stratified ({age_group})',
                        'outcome': label,
                        'effect': 'UCLA x Gender',
                        'beta': result['interaction_beta'],
                        'p': result['interaction_p'],
                        'n': result['n']
                    })
                    if verbose:
                        print(f"    * {label}: UCLA×Gender β={result['interaction_beta']:.3f}, p={result['interaction_p']:.4f}")

            except Exception as e:
                if verbose:
                    print(f"    {label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "age_stratified_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_threeway_interaction(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    UCLA × 성별 × 나이 3원 상호작용 검정

    Moderated moderation: 나이가 UCLA × 성별 상호작용을 조절하는가?
    """
    output_dir = OUTPUT_DIR / "threeway"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[THREE-WAY INTERACTION] Testing UCLA × Gender × Age...")

    master = load_master_data()
    master = create_age_groups(master)

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
            # 3원 상호작용 모델
            formula = f"{outcome} ~ z_ucla * C(gender_male) * z_age + z_dass_dep + z_dass_anx + z_dass_str"
            model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

            # 2원 상호작용 모델 (비교용)
            formula_2way = f"{outcome} ~ z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_str"
            model_2way = smf.ols(formula_2way, data=clean_data).fit(cov_type='HC3')

            # 2-way와 3-way interaction 동적 탐지
            int_term = find_interaction_term(model.params.index)  # z_ucla:C(gender_male)[T.1]
            # 3-way: ucla, gender, age 모두 포함
            threeway_term = None
            for term in model.params.index:
                if 'ucla' in term and 'gender' in term and 'age' in term.lower() and term.count(':') >= 2:
                    threeway_term = term
                    break

            result = {
                'outcome': outcome,
                'outcome_label': label,
                'n': len(clean_data),
                # Main effects
                'ucla_beta': model.params.get('z_ucla', np.nan),
                'ucla_p': model.pvalues.get('z_ucla', np.nan),
                'gender_beta': model.params.get('C(gender_male)[T.1]', np.nan),
                'gender_p': model.pvalues.get('C(gender_male)[T.1]', np.nan),
                'age_beta': model.params.get('z_age', np.nan),
                'age_p': model.pvalues.get('z_age', np.nan),
                # 2-way interactions
                'ucla_gender_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
                'ucla_gender_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
                'ucla_age_beta': model.params.get('z_ucla:z_age', np.nan),
                'ucla_age_p': model.pvalues.get('z_ucla:z_age', np.nan),
                'gender_age_beta': model.params.get('C(gender_male)[T.1]:z_age', np.nan),
                'gender_age_p': model.pvalues.get('C(gender_male)[T.1]:z_age', np.nan),
                # 3-way interaction
                'threeway_beta': model.params.get(threeway_term, np.nan),
                'threeway_se': model.bse.get(threeway_term, np.nan),
                'threeway_p': model.pvalues.get(threeway_term, np.nan),
                # Model fit
                'r_squared': model.rsquared,
                'r_squared_2way': model_2way.rsquared,
                'delta_r2': model.rsquared - model_2way.rsquared
            }
            all_results.append(result)

            if verbose:
                print(f"\n  {label}:")
                print(f"    UCLA × Gender: β={result['ucla_gender_beta']:.3f}, p={result['ucla_gender_p']:.4f}")
                print(f"    UCLA × Age: β={result['ucla_age_beta']:.3f}, p={result['ucla_age_p']:.4f}")
                print(f"    UCLA × Gender × Age: β={result['threeway_beta']:.3f}, p={result['threeway_p']:.4f}")

            # Check significance
            if result['threeway_p'] < 0.05:
                significant_findings.append({
                    'analysis': 'Three-way Interaction',
                    'outcome': label,
                    'effect': 'UCLA x Gender x Age',
                    'beta': result['threeway_beta'],
                    'p': result['threeway_p'],
                    'n': result['n']
                })

            if result['ucla_age_p'] < 0.05:
                significant_findings.append({
                    'analysis': 'Three-way Interaction',
                    'outcome': label,
                    'effect': 'UCLA x Age',
                    'beta': result['ucla_age_beta'],
                    'p': result['ucla_age_p'],
                    'n': result['n']
                })

        except Exception as e:
            if verbose:
                print(f"  {label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "threeway_interaction_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_johnson_neyman(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Johnson-Neyman 유의 영역 분석

    UCLA × 성별 상호작용이 어느 연령대에서 유의한지 식별
    """
    output_dir = OUTPUT_DIR / "johnson_neyman"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[JOHNSON-NEYMAN] Identifying regions of significance...")

    master = load_master_data()

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

        clean_data = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'age', outcome])

        if len(clean_data) < 50:
            continue

        try:
            # 연령 범위 정의
            age_range = np.arange(18, 37, 1)

            # 각 연령 값에서 UCLA 효과 계산 (조건부 효과)
            conditional_effects = []

            for test_age in age_range:
                # 중심화된 나이 사용
                clean_data['age_test_centered'] = clean_data['age'] - test_age

                # UCLA × Gender × Age 상호작용 모델
                formula = f"{outcome} ~ z_ucla * C(gender_male) * age_test_centered + z_dass_dep + z_dass_anx + z_dass_str"

                try:
                    model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                    # 이 연령에서의 UCLA × Gender 조건부 효과
                    # age_test_centered = 0일 때의 interaction 계수
                    int_term = find_interaction_term(model.params.index)
                    cond_beta = model.params.get(int_term, np.nan) if int_term else np.nan
                    cond_se = model.bse.get(int_term, np.nan) if int_term else np.nan
                    cond_p = model.pvalues.get(int_term, np.nan) if int_term else np.nan

                    conditional_effects.append({
                        'age': test_age,
                        'beta': cond_beta,
                        'se': cond_se,
                        'p': cond_p,
                        'lower_ci': cond_beta - 1.96 * cond_se,
                        'upper_ci': cond_beta + 1.96 * cond_se,
                        'significant': cond_p < 0.05
                    })

                except Exception:
                    continue

            if not conditional_effects:
                continue

            cond_df = pd.DataFrame(conditional_effects)

            # 유의 영역 찾기
            sig_ages = cond_df[cond_df['significant']]['age'].values

            result = {
                'outcome': outcome,
                'outcome_label': label,
                'n': len(clean_data),
                'min_age': clean_data['age'].min(),
                'max_age': clean_data['age'].max(),
                'mean_age': clean_data['age'].mean(),
                'n_sig_ages': len(sig_ages),
                'sig_ages_range': f"{sig_ages.min():.0f}-{sig_ages.max():.0f}" if len(sig_ages) > 0 else "None",
                'nonsig_ages_range': "N/A"  # 비유의 영역
            }

            # 비유의 영역
            nonsig_ages = cond_df[~cond_df['significant']]['age'].values
            if len(nonsig_ages) > 0:
                result['nonsig_ages_range'] = f"{nonsig_ages.min():.0f}-{nonsig_ages.max():.0f}"

            all_results.append(result)

            if verbose:
                print(f"\n  {label}:")
                print(f"    Age range: {result['min_age']:.0f}-{result['max_age']:.0f}")
                if len(sig_ages) > 0:
                    print(f"    UCLA×Gender significant at ages: {result['sig_ages_range']}")
                else:
                    print(f"    UCLA×Gender NOT significant at any age")

            if len(sig_ages) > 0:
                significant_findings.append({
                    'analysis': 'Johnson-Neyman',
                    'outcome': label,
                    'effect': f'UCLA×Gender (ages {result["sig_ages_range"]})',
                    'beta': cond_df[cond_df['significant']]['beta'].mean(),
                    'p': cond_df[cond_df['significant']]['p'].min(),
                    'n': result['n']
                })

            # 조건부 효과 저장
            cond_df.to_csv(output_dir / f"jn_{outcome}.csv", index=False, encoding='utf-8-sig')

        except Exception as e:
            if verbose:
                print(f"  {label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "johnson_neyman_summary.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_simple_slopes(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Simple slopes 분석

    젊은/중간/나이든 그룹에서 각각 성별별 UCLA 기울기 계산
    """
    output_dir = OUTPUT_DIR / "simple_slopes"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[SIMPLE SLOPES] Computing UCLA slopes by age × gender...")

    master = load_master_data()
    master = create_age_groups(master)

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

        if verbose:
            print(f"\n  {label}:")

        for age_group in ['18-21', '22-25', '26+']:
            age_subset = master[master['age_group'] == age_group].copy()

            for gender, gender_label in [(0, 'Female'), (1, 'Male')]:
                subset = age_subset[age_subset['gender_male'] == gender].copy()

                clean_data = subset.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', outcome])

                if len(clean_data) < 10:
                    continue

                try:
                    formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str"
                    model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                    result = {
                        'outcome': outcome,
                        'outcome_label': label,
                        'age_group': age_group,
                        'gender': gender_label,
                        'n': len(clean_data),
                        'ucla_beta': model.params.get('z_ucla', np.nan),
                        'ucla_se': model.bse.get('z_ucla', np.nan),
                        'ucla_p': model.pvalues.get('z_ucla', np.nan),
                        'r_squared': model.rsquared
                    }
                    all_results.append(result)

                    sig = "*" if result['ucla_p'] < 0.05 else ""
                    if verbose:
                        print(f"    {age_group} {gender_label}: UCLA β={result['ucla_beta']:.3f}, p={result['ucla_p']:.4f} {sig}")

                    if result['ucla_p'] < 0.05:
                        significant_findings.append({
                            'analysis': 'Simple Slopes',
                            'outcome': f"{label} ({age_group}, {gender_label})",
                            'effect': 'UCLA',
                            'beta': result['ucla_beta'],
                            'p': result['ucla_p'],
                            'n': result['n']
                        })

                except Exception as e:
                    if verbose:
                        print(f"    {age_group} {gender_label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "simple_slopes_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


# =============================================================================
# RESULTS RECORDING
# =============================================================================

def record_significant_results(findings: List[Dict], verbose: bool = True):
    """
    유의한 결과를 Results.md에 기록
    """
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
        entry = f"| {today} | {finding['analysis']} | {finding['outcome']} | {finding['effect']} | β={finding['beta']:.3f} | p={finding['p']:.4f} | N={finding['n']} |"
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
    'stratified': ('Age-stratified UCLA × Gender effects', analyze_age_stratified_effects),
    'threeway': ('UCLA × Gender × Age three-way interaction', analyze_threeway_interaction),
    'johnson_neyman': ('Johnson-Neyman regions of significance', analyze_johnson_neyman),
    'simple_slopes': ('Simple slopes by age × gender', analyze_simple_slopes),
}


def run(analysis: Optional[str] = None, verbose: bool = True, record_results: bool = True) -> Dict:
    """Run developmental window analyses."""
    if verbose:
        print("=" * 70)
        print("DEVELOPMENTAL WINDOW ANALYSIS SUITE")
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
            if verbose:
                print(f"\n{'='*50}")
            try:
                result = func(verbose=verbose)
                if isinstance(result, tuple):
                    results[name], findings = result
                    all_significant.extend(findings)
                else:
                    results[name] = result
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    # Record significant findings
    if record_results and all_significant:
        record_significant_results(all_significant, verbose=verbose)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Output saved to: {OUTPUT_DIR}")
        if all_significant:
            print(f"Significant findings: {len(all_significant)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Developmental Window Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                       help=f"Specific analysis to run: {list(ANALYSES.keys())}")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Suppress verbose output")
    parser.add_argument('--no-record', action='store_true',
                       help="Don't record results to Results.md")
    args = parser.parse_args()

    run(analysis=args.analysis, verbose=not args.quiet, record_results=not args.no_record)
