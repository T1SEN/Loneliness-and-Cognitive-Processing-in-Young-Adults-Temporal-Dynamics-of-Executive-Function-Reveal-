"""
DASS Subscale Specificity Analysis Suite
========================================

DASS 하위척도 특이성 분석

연구 질문: UCLA-EF 혼란이 주로 우울, 불안, 스트레스 중 어디서 기인하는가?

통계 방법:
- 순차적 DASS 통제: 하위척도 하나씩 추가
- 공통성 분석 (Commonality Analysis)
- 우세성 분석 (Dominance Analysis)

Usage:
    python -m analysis.synthesis.dass_specificity_suite
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
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from analysis.preprocessing import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR, find_interaction_term
)
from analysis.utils.modeling import standardize_predictors

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "dass_specificity_suite"
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
# ANALYSES
# =============================================================================

def analyze_sequential_control(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    순차적 DASS 통제 분석

    각 하위척도를 순차적으로 추가하며 UCLA 효과 변화 추적
    """
    output_dir = OUTPUT_DIR / "sequential"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[SEQUENTIAL DASS CONTROL] Testing UCLA effect with stepwise DASS control...")

    master = load_master_data()

    ef_outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('wcst_accuracy', 'WCST Accuracy'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck', 'PRP Bottleneck')
    ]

    all_results = []
    significant_findings = []

    # 순차적 모델들
    models_specs = [
        ('model0', 'z_ucla * C(gender_male) + z_age', 'UCLA only'),
        ('model1_dep', 'z_ucla * C(gender_male) + z_age + z_dass_dep', '+ Depression'),
        ('model1_anx', 'z_ucla * C(gender_male) + z_age + z_dass_anx', '+ Anxiety'),
        ('model1_str', 'z_ucla * C(gender_male) + z_age + z_dass_str', '+ Stress'),
        ('model2_dep_anx', 'z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx', '+ Dep + Anx'),
        ('model2_dep_str', 'z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_str', '+ Dep + Str'),
        ('model2_anx_str', 'z_ucla * C(gender_male) + z_age + z_dass_anx + z_dass_str', '+ Anx + Str'),
        ('model3_full', 'z_ucla * C(gender_male) + z_age + z_dass_dep + z_dass_anx + z_dass_str', 'Full Model'),
    ]

    for outcome, label in ef_outcomes:
        if outcome not in master.columns:
            continue

        if verbose:
            print(f"\n  {label}:")
            print("  " + "-" * 60)

        clean_data = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

        if len(clean_data) < 50:
            continue

        for model_name, formula_rhs, model_label in models_specs:
            try:
                formula = f"{outcome} ~ {formula_rhs}"
                model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                result = {
                    'outcome': outcome,
                    'outcome_label': label,
                    'model': model_name,
                    'model_label': model_label,
                    'n': len(clean_data),
                    'ucla_beta': model.params.get('z_ucla', np.nan),
                    'ucla_se': model.bse.get('z_ucla', np.nan),
                    'ucla_p': model.pvalues.get('z_ucla', np.nan),
                }
                int_term = find_interaction_term(model.params.index)
                result.update({
                    'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
                    'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj
                })
                all_results.append(result)

                sig = "*" if result['ucla_p'] < 0.05 else ""
                if verbose:
                    print(f"    {model_label:20s}: UCLA β={result['ucla_beta']:.3f}, p={result['ucla_p']:.4f} {sig} (R²={result['r_squared']:.3f})")

            except Exception as e:
                if verbose:
                    print(f"    {model_label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "sequential_control_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_commonality(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    공통성 분석 (Commonality Analysis)

    UCLA와 각 DASS 하위척도의 독립적/공유 분산 분해
    """
    output_dir = OUTPUT_DIR / "commonality"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[COMMONALITY ANALYSIS] Decomposing variance contributions...")

    master = load_master_data()

    ef_outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('wcst_accuracy', 'WCST Accuracy'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck', 'PRP Bottleneck')
    ]

    all_results = []
    significant_findings = []

    predictors = ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str']

    for outcome, label in ef_outcomes:
        if outcome not in master.columns:
            continue

        clean_data = master.dropna(subset=predictors + ['z_age', 'gender_male', outcome])

        if len(clean_data) < 50:
            continue

        if verbose:
            print(f"\n  {label}:")

        # 모든 가능한 예측변수 조합의 R² 계산
        r2_dict = {}

        for r in range(1, len(predictors) + 1):
            for combo in combinations(predictors, r):
                combo_key = '+'.join(sorted(combo))
                formula = f"{outcome} ~ {' + '.join(combo)} + z_age + C(gender_male)"
                try:
                    model = smf.ols(formula, data=clean_data).fit()
                    r2_dict[combo_key] = model.rsquared
                except Exception:
                    r2_dict[combo_key] = 0

        # 기준 모델 (나이+성별만)
        base_model = smf.ols(f"{outcome} ~ z_age + C(gender_male)", data=clean_data).fit()
        r2_base = base_model.rsquared

        # 공통성 계산 (단순화된 버전)
        # 각 예측변수의 독립적 기여
        unique_contributions = {}
        for pred in predictors:
            # pred만 추가했을 때의 증분 R²
            other_preds = [p for p in predictors if p != pred]
            full_formula = f"{outcome} ~ {' + '.join(predictors)} + z_age + C(gender_male)"
            reduced_formula = f"{outcome} ~ {' + '.join(other_preds)} + z_age + C(gender_male)"

            try:
                full_model = smf.ols(full_formula, data=clean_data).fit()
                reduced_model = smf.ols(reduced_formula, data=clean_data).fit()
                unique_r2 = full_model.rsquared - reduced_model.rsquared
            except Exception:
                unique_r2 = 0

            unique_contributions[pred] = unique_r2

        # 결과 저장
        for pred, unique_r2 in unique_contributions.items():
            pred_label = {
                'z_ucla': 'UCLA',
                'z_dass_dep': 'Depression',
                'z_dass_anx': 'Anxiety',
                'z_dass_str': 'Stress'
            }.get(pred, pred)

            all_results.append({
                'outcome': outcome,
                'outcome_label': label,
                'predictor': pred,
                'predictor_label': pred_label,
                'unique_r2': unique_r2,
                'unique_r2_pct': unique_r2 * 100,
                'n': len(clean_data)
            })

            if verbose:
                print(f"    {pred_label:12s}: Unique R² = {unique_r2:.4f} ({unique_r2*100:.2f}%)")

        # UCLA vs DASS 전체 비교
        ucla_unique = unique_contributions.get('z_ucla', 0)
        dass_total_unique = sum(unique_contributions.get(p, 0) for p in ['z_dass_dep', 'z_dass_anx', 'z_dass_str'])

        if verbose:
            print(f"    ---")
            print(f"    UCLA unique:  {ucla_unique*100:.2f}%")
            print(f"    DASS total unique: {dass_total_unique*100:.2f}%")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "commonality_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_dominance(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    우세성 분석 (Dominance Analysis)

    각 DASS 하위척도가 UCLA 효과를 얼마나 감소시키는지 비교
    """
    output_dir = OUTPUT_DIR / "dominance"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[DOMINANCE ANALYSIS] Comparing DASS subscale importance...")

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

        clean_data = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

        if len(clean_data) < 50:
            continue

        if verbose:
            print(f"\n  {label}:")

        # 기준: UCLA만 있을 때의 효과
        base_formula = f"{outcome} ~ z_ucla * C(gender_male) + z_age"
        base_model = smf.ols(base_formula, data=clean_data).fit(cov_type='HC3')
        base_ucla_beta = base_model.params.get('z_ucla', np.nan)

        # 각 DASS 하위척도 추가 시 UCLA 효과 감소량
        subscales = [
            ('z_dass_dep', 'Depression'),
            ('z_dass_anx', 'Anxiety'),
            ('z_dass_str', 'Stress')
        ]

        reductions = []
        for subscale, sub_label in subscales:
            formula = f"{outcome} ~ z_ucla * C(gender_male) + z_age + {subscale}"
            try:
                model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')
                new_ucla_beta = model.params.get('z_ucla', np.nan)

                reduction = base_ucla_beta - new_ucla_beta
                reduction_pct = (reduction / abs(base_ucla_beta) * 100) if base_ucla_beta != 0 else 0

                reductions.append({
                    'outcome': outcome,
                    'outcome_label': label,
                    'subscale': subscale,
                    'subscale_label': sub_label,
                    'base_ucla_beta': base_ucla_beta,
                    'controlled_ucla_beta': new_ucla_beta,
                    'beta_reduction': reduction,
                    'reduction_pct': reduction_pct,
                    'n': len(clean_data)
                })

                if verbose:
                    print(f"    {sub_label:12s}: UCLA β reduction = {reduction:.4f} ({reduction_pct:.1f}%)")

            except Exception as e:
                if verbose:
                    print(f"    {sub_label}: ERROR - {e}")

        all_results.extend(reductions)

        # 가장 큰 감소를 보인 하위척도
        if reductions:
            max_reduction = max(reductions, key=lambda x: abs(x['reduction_pct']))
            if verbose:
                print(f"    → Primary confound: {max_reduction['subscale_label']} ({max_reduction['reduction_pct']:.1f}% reduction)")

            significant_findings.append({
                'analysis': 'Dominance Analysis',
                'outcome': label,
                'effect': f"Primary confound: {max_reduction['subscale_label']}",
                'beta': max_reduction['reduction_pct'],
                'p': np.nan,  # N/A for dominance
                'n': max_reduction['n']
            })

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "dominance_results.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_subscale_correlations(verbose: bool = True) -> pd.DataFrame:
    """
    UCLA와 DASS 하위척도 간 상관 분석

    성별별 상관 패턴 비교
    """
    output_dir = OUTPUT_DIR / "correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[UCLA-DASS CORRELATIONS] Computing subscale correlations...")

    master = load_master_data()

    variables = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
    labels = {'ucla_total': 'UCLA', 'dass_depression': 'Depression', 'dass_anxiety': 'Anxiety', 'dass_stress': 'Stress'}

    all_results = []

    for gender, gender_label in [(1, 'Male'), (0, 'Female'), (None, 'All')]:
        if gender is not None:
            subset = master[master['gender_male'] == gender].copy()
        else:
            subset = master.copy()

        if verbose:
            print(f"\n  {gender_label} (N={len(subset)}):")

        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                if var1 not in subset.columns or var2 not in subset.columns:
                    continue

                clean = subset.dropna(subset=[var1, var2])
                if len(clean) < 20:
                    continue

                r, p = stats.pearsonr(clean[var1], clean[var2])

                result = {
                    'gender': gender_label,
                    'var1': var1,
                    'var1_label': labels.get(var1, var1),
                    'var2': var2,
                    'var2_label': labels.get(var2, var2),
                    'r': r,
                    'p': p,
                    'n': len(clean)
                }
                all_results.append(result)

                if var1 == 'ucla_total':
                    sig = "*" if p < 0.05 else ""
                    if verbose:
                        print(f"    UCLA × {labels.get(var2, var2):12s}: r={r:.3f}, p={p:.4f} {sig}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "subscale_correlations.csv", index=False, encoding='utf-8-sig')

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
        entry = f"| {today} | {finding['analysis']} | {finding['outcome']} | {finding['effect']} | β={finding['beta']:.2f}% | {p_str} | N={finding['n']} |"
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
    'sequential': ('Sequential DASS subscale control', analyze_sequential_control),
    'commonality': ('Commonality analysis (variance decomposition)', analyze_commonality),
    'dominance': ('Dominance analysis (subscale importance)', analyze_dominance),
    'correlations': ('UCLA-DASS subscale correlations', analyze_subscale_correlations),
}


def run(analysis: Optional[str] = None, verbose: bool = True, record_results: bool = True) -> Dict:
    """Run DASS specificity analyses."""
    if verbose:
        print("=" * 70)
        print("DASS SUBSCALE SPECIFICITY ANALYSIS SUITE")
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
    parser = argparse.ArgumentParser(description="DASS Specificity Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    args = parser.parse_args()

    run(analysis=args.analysis, verbose=not args.quiet, record_results=not args.no_record)
