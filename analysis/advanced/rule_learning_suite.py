"""
Rule-Specific Learning Curves Analysis Suite
=============================================

WCST 규칙별 학습 곡선 분석

연구 질문: 외로운 개인이 특정 규칙 유형(색상, 모양, 숫자)에서
학습 장애를 보이는가?

통계 방법:
- 규칙 에포크별 정확도 궤적
- 규칙별 trials-to-criterion (5연속 정답) 계산
- Mixed-effects model: accuracy ~ trial_in_rule * rule_type * z_ucla * C(gender_male) + DASS controls

Usage:
    python -m analysis.advanced.rule_learning_suite
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from analysis.preprocessing import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR, find_interaction_term
)
from analysis.utils.modeling import standardize_predictors, DASS_CONTROL_FORMULA

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "rule_learning_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_wcst_extra(extra_str):
    """Parse WCST extra column."""
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}


def load_wcst_trials() -> pd.DataFrame:
    """Load WCST trial data with rule information."""
    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    # Parse extra column for isPE
    if 'extra' in df.columns:
        extra_parsed = df['extra'].apply(parse_wcst_extra)
        df['is_pe'] = extra_parsed.apply(lambda x: x.get('isPE', False))
    elif 'ispe' in df.columns:
        df['is_pe'] = df['ispe']
    else:
        df['is_pe'] = False

    # Standardize correct column
    if 'iscorrect' in df.columns:
        df['correct'] = df['iscorrect']
    elif 'is_correct' in df.columns:
        df['correct'] = df['is_correct']

    return df


def load_master_with_wcst() -> pd.DataFrame:
    """Load master dataset with WCST metrics."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
    master['gender_male'] = (master['gender'] == 'male').astype(int)

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master = standardize_predictors(master)
    return master


# =============================================================================
# RULE EPOCH EXTRACTION
# =============================================================================

def extract_rule_epochs(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rule epochs from WCST trial data.

    각 참가자별로 규칙 변경 시점을 감지하고 에포크 정보 추가.
    """
    # Find rule column
    rule_col = None
    for cand in ['ruleatthattime', 'rule_at_that_time', 'rule_at_time', 'rule']:
        if cand in trials.columns:
            rule_col = cand
            break

    if not rule_col:
        print("  WARNING: No rule column found in WCST data")
        return pd.DataFrame()

    # Sort by trial order
    trial_col = 'idx' if 'idx' in trials.columns else 'trial_index' if 'trial_index' in trials.columns else 'trialindex'
    if trial_col not in trials.columns:
        trial_col = trials.columns[0]

    trials = trials.sort_values(['participant_id', trial_col]).copy()

    # Extract epochs per participant
    epoch_data = []

    for pid, grp in trials.groupby('participant_id'):
        rules = grp[rule_col].values
        correct = grp['correct'].values
        is_pe = grp['is_pe'].values if 'is_pe' in grp.columns else np.zeros(len(grp))

        if len(rules) < 20:
            continue

        # Detect rule changes
        epoch_num = 0
        trial_in_epoch = 0

        for i in range(len(grp)):
            if i > 0 and rules[i] != rules[i-1]:
                epoch_num += 1
                trial_in_epoch = 0

            epoch_data.append({
                'participant_id': pid,
                'trial_idx': i,
                'rule': rules[i],
                'correct': correct[i],
                'is_pe': is_pe[i],
                'epoch_num': epoch_num,
                'trial_in_epoch': trial_in_epoch
            })
            trial_in_epoch += 1

    return pd.DataFrame(epoch_data)


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_rule_learning_curves(verbose: bool = True) -> pd.DataFrame:
    """
    규칙별 학습 곡선 분석

    각 규칙 유형(색상, 모양, 숫자)에서 시행에 따른 정확도 변화 분석.
    """
    output_dir = OUTPUT_DIR / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[RULE LEARNING CURVES] Analyzing rule-specific learning...")

    trials = load_wcst_trials()
    master = load_master_with_wcst()
    epochs = extract_rule_epochs(trials)

    if epochs.empty:
        print("  No epoch data available")
        return pd.DataFrame()

    # 규칙별 학습 기울기 계산
    learning_results = []

    for pid in epochs['participant_id'].unique():
        pdata = epochs[epochs['participant_id'] == pid]

        for rule in pdata['rule'].unique():
            rule_data = pdata[pdata['rule'] == rule]

            if len(rule_data) < 5:
                continue

            # 시행에 따른 정확도 기울기
            x = rule_data['trial_in_epoch'].values
            y = rule_data['correct'].astype(float).values

            if len(np.unique(x)) < 2:
                continue

            slope, intercept, r, p, se = stats.linregress(x, y)

            # Trials to criterion (5연속 정답)
            ttc = np.nan
            for i in range(len(rule_data) - 4):
                if rule_data.iloc[i:i+5]['correct'].all():
                    ttc = i
                    break

            # 에포크별 평균 정확도
            first_half_acc = rule_data.iloc[:len(rule_data)//2]['correct'].mean()
            second_half_acc = rule_data.iloc[len(rule_data)//2:]['correct'].mean()

            learning_results.append({
                'participant_id': pid,
                'rule': rule,
                'n_trials': len(rule_data),
                'learning_slope': slope,
                'learning_r': r,
                'learning_p': p,
                'trials_to_criterion': ttc,
                'first_half_accuracy': first_half_acc,
                'second_half_accuracy': second_half_acc,
                'accuracy_improvement': second_half_acc - first_half_acc,
                'overall_accuracy': rule_data['correct'].mean(),
                'pe_rate': rule_data['is_pe'].mean() * 100
            })

    learning_df = pd.DataFrame(learning_results)

    if verbose:
        print(f"  N rule-epochs: {len(learning_df)}")
        print(f"  N participants: {learning_df['participant_id'].nunique()}")

        # 규칙별 평균
        for rule in learning_df['rule'].unique():
            rule_subset = learning_df[learning_df['rule'] == rule]
            print(f"  {rule}: slope={rule_subset['learning_slope'].mean():.4f}, acc={rule_subset['overall_accuracy'].mean():.2f}")

    learning_df.to_csv(output_dir / "rule_learning_metrics.csv", index=False, encoding='utf-8-sig')
    return learning_df


def analyze_rule_ucla_interaction(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    규칙 유형 × UCLA × 성별 상호작용 분석

    DASS 통제 후 UCLA가 특정 규칙 학습에 영향을 미치는지 검정.
    """
    output_dir = OUTPUT_DIR / "ucla_interaction"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[RULE × UCLA INTERACTION] Testing rule-specific UCLA effects...")

    # Load data
    learning_df = analyze_rule_learning_curves(verbose=False)
    if learning_df.empty:
        return pd.DataFrame(), []

    master = load_master_with_wcst()

    # Keep only necessary columns from master to avoid column name conflicts
    master_cols = ['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', 'ucla_total']
    master_subset = master[[c for c in master_cols if c in master.columns]].copy()

    # Merge and prepare
    analysis_df = learning_df.merge(master_subset, on='participant_id', how='inner')

    # 규칙 유형별 분석
    results = []
    significant_findings = []

    for rule in analysis_df['rule'].unique():
        rule_data = analysis_df[analysis_df['rule'] == rule].copy()

        for outcome in ['learning_slope', 'trials_to_criterion', 'overall_accuracy', 'pe_rate']:
            clean_data = rule_data.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

            if len(clean_data) < 30:
                continue

            formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            try:
                model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                int_term = find_interaction_term(model.params.index)
                result = {
                    'rule': rule,
                    'outcome': outcome,
                    'n': len(clean_data),
                    'ucla_beta': model.params.get('z_ucla', np.nan),
                    'ucla_se': model.bse.get('z_ucla', np.nan),
                    'ucla_p': model.pvalues.get('z_ucla', np.nan),
                    'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
                    'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
                    'r_squared': model.rsquared
                }
                results.append(result)

                # Check significance
                if result['ucla_p'] < 0.05 or result['interaction_p'] < 0.05:
                    if result['ucla_p'] < 0.05:
                        significant_findings.append({
                            'analysis': 'Rule Learning',
                            'outcome': f"{rule} {outcome}",
                            'effect': 'UCLA main',
                            'beta': result['ucla_beta'],
                            'p': result['ucla_p'],
                            'n': result['n']
                        })
                    if result['interaction_p'] < 0.05:
                        significant_findings.append({
                            'analysis': 'Rule Learning',
                            'outcome': f"{rule} {outcome}",
                            'effect': 'UCLA x Gender',
                            'beta': result['interaction_beta'],
                            'p': result['interaction_p'],
                            'n': result['n']
                        })

                    if verbose:
                        sig_type = "UCLA" if result['ucla_p'] < 0.05 else "UCLA×Gender"
                        p_val = result['ucla_p'] if result['ucla_p'] < 0.05 else result['interaction_p']
                        beta = result['ucla_beta'] if result['ucla_p'] < 0.05 else result['interaction_beta']
                        print(f"  * {rule} {outcome}: {sig_type} β={beta:.3f}, p={p_val:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Error fitting {rule} {outcome}: {e}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "rule_ucla_regression.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_rule_comparison(verbose: bool = True) -> pd.DataFrame:
    """
    규칙 간 난이도 비교 분석

    어떤 규칙이 가장 어려운지, UCLA 그룹별 차이가 있는지 분석.
    """
    output_dir = OUTPUT_DIR / "rule_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[RULE COMPARISON] Comparing difficulty across rules...")

    learning_df = analyze_rule_learning_curves(verbose=False)
    if learning_df.empty:
        return pd.DataFrame()

    master = load_master_with_wcst()

    # Keep only necessary columns from master to avoid column name conflicts
    master_cols = ['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', 'ucla_total']
    master_subset = master[[c for c in master_cols if c in master.columns]].copy()

    analysis_df = learning_df.merge(master_subset, on='participant_id', how='inner')

    # UCLA tertile 생성
    analysis_df['ucla_group'] = pd.qcut(
        analysis_df['ucla_total'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    # 규칙별 통계
    comparison_results = []

    for ucla_group in ['Low', 'Medium', 'High']:
        group_data = analysis_df[analysis_df['ucla_group'] == ucla_group]

        for rule in group_data['rule'].unique():
            rule_data = group_data[group_data['rule'] == rule]

            if len(rule_data) < 5:
                continue

            comparison_results.append({
                'ucla_group': ucla_group,
                'rule': rule,
                'n': len(rule_data),
                'mean_accuracy': rule_data['overall_accuracy'].mean(),
                'std_accuracy': rule_data['overall_accuracy'].std(),
                'mean_learning_slope': rule_data['learning_slope'].mean(),
                'mean_ttc': rule_data['trials_to_criterion'].mean(),
                'mean_pe_rate': rule_data['pe_rate'].mean()
            })

    comparison_df = pd.DataFrame(comparison_results)

    if verbose and not comparison_df.empty:
        print("\n  Rule Difficulty by UCLA Group:")
        pivot = comparison_df.pivot_table(
            index='rule',
            columns='ucla_group',
            values='mean_accuracy',
            aggfunc='mean'
        )
        print(pivot.round(3).to_string())

    comparison_df.to_csv(output_dir / "rule_comparison.csv", index=False, encoding='utf-8-sig')
    return comparison_df


def analyze_first_rule_effect(verbose: bool = True) -> pd.DataFrame:
    """
    첫 번째 규칙 학습 효과 분석

    첫 규칙(보통 색상)이 가장 어렵다는 가설 검정.
    UCLA가 첫 규칙 학습에 특히 영향을 미치는지 분석.
    """
    output_dir = OUTPUT_DIR / "first_rule"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[FIRST RULE EFFECT] Analyzing first rule learning...")

    trials = load_wcst_trials()
    master = load_master_with_wcst()
    epochs = extract_rule_epochs(trials)

    if epochs.empty:
        return pd.DataFrame()

    # 첫 번째 규칙 에포크만 추출
    first_epoch = epochs[epochs['epoch_num'] == 0].copy()

    # 참가자별 첫 규칙 성과
    first_rule_results = []

    for pid in first_epoch['participant_id'].unique():
        pdata = first_epoch[first_epoch['participant_id'] == pid]

        if len(pdata) < 5:
            continue

        # 학습 기울기
        x = pdata['trial_in_epoch'].values
        y = pdata['correct'].astype(float).values

        if len(np.unique(x)) < 2:
            continue

        slope, _, r, p, _ = stats.linregress(x, y)

        # Trials to criterion
        ttc = np.nan
        for i in range(len(pdata) - 4):
            if pdata.iloc[i:i+5]['correct'].all():
                ttc = i
                break

        first_rule_results.append({
            'participant_id': pid,
            'first_rule': pdata['rule'].iloc[0],
            'n_trials': len(pdata),
            'accuracy': pdata['correct'].mean(),
            'learning_slope': slope,
            'trials_to_criterion': ttc,
            'pe_rate': pdata['is_pe'].mean() * 100
        })

    first_df = pd.DataFrame(first_rule_results)

    # Keep only necessary columns from master to avoid column name conflicts
    master_cols = ['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', 'ucla_total']
    master_subset = master[[c for c in master_cols if c in master.columns]].copy()

    analysis_df = first_df.merge(master_subset, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean first-rule accuracy: {analysis_df['accuracy'].mean():.3f}")

    # DASS-controlled regression
    results = []
    significant_findings = []

    for outcome in ['accuracy', 'learning_slope', 'trials_to_criterion', 'pe_rate']:
        clean_data = analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

        if len(clean_data) < 30:
            continue

        formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        try:
            model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

            int_term = find_interaction_term(model.params.index)
            result = {
                'outcome': f'first_rule_{outcome}',
                'n': len(clean_data),
                'ucla_beta': model.params.get('z_ucla', np.nan),
                'ucla_p': model.pvalues.get('z_ucla', np.nan),
                'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
                'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
                'r_squared': model.rsquared
            }
            results.append(result)

            if result['ucla_p'] < 0.05 or result['interaction_p'] < 0.05:
                sig_effect = 'UCLA main' if result['ucla_p'] < 0.05 else 'UCLA x Gender'
                sig_p = result['ucla_p'] if result['ucla_p'] < 0.05 else result['interaction_p']
                sig_beta = result['ucla_beta'] if result['ucla_p'] < 0.05 else result['interaction_beta']

                significant_findings.append({
                    'analysis': 'First Rule Learning',
                    'outcome': outcome,
                    'effect': sig_effect,
                    'beta': sig_beta,
                    'p': sig_p,
                    'n': result['n']
                })

                if verbose:
                    print(f"  * First rule {outcome}: {sig_effect} β={sig_beta:.3f}, p={sig_p:.4f}")

        except Exception as e:
            if verbose:
                print(f"  Error fitting first rule {outcome}: {e}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "first_rule_regression.csv", index=False, encoding='utf-8-sig')

    first_df.to_csv(output_dir / "first_rule_metrics.csv", index=False, encoding='utf-8-sig')
    return first_df


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

    # Read existing content
    existing_content = ""
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()

    # Add new findings
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    new_entries = []
    for finding in findings:
        entry = f"| {today} | Rule Learning | {finding['outcome']} | {finding['effect']} | β={finding['beta']:.3f} | p={finding['p']:.4f} | N={finding['n']} |"
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
    'learning_curves': ('Rule-specific learning curves', analyze_rule_learning_curves),
    'ucla_interaction': ('Rule × UCLA × Gender interaction', analyze_rule_ucla_interaction),
    'rule_comparison': ('Rule difficulty comparison', analyze_rule_comparison),
    'first_rule': ('First rule learning effect', analyze_first_rule_effect),
}


def run(analysis: Optional[str] = None, verbose: bool = True, record_results: bool = True) -> Dict:
    """Run rule learning analyses."""
    if verbose:
        print("=" * 70)
        print("RULE-SPECIFIC LEARNING CURVES ANALYSIS SUITE")
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
    parser = argparse.ArgumentParser(description="Rule Learning Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                       help=f"Specific analysis to run: {list(ANALYSES.keys())}")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Suppress verbose output")
    parser.add_argument('--no-record', action='store_true',
                       help="Don't record results to Results.md")
    args = parser.parse_args()

    run(analysis=args.analysis, verbose=not args.quiet, record_results=not args.no_record)
