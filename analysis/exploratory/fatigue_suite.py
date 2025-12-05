"""
Within-Block Fatigue Analysis Suite
===================================

블록 내 피로 × UCLA 상호작용 분석

연구 질문: 외로운 개인이 더 큰 피로 효과(시행에 따른 성능 저하)를 보이는가?

통계 방법:
- 과제별 시행 사분위 추출
- RT/정확도 변화 궤적 분석
- UCLA × 피로 기울기 상호작용 검정

Usage:
    python -m analysis.exploratory.fatigue_suite
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

from analysis.utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "fatigue_suite"
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


def load_trial_data(task: str) -> pd.DataFrame:
    """Load trial-level data for specific task."""
    trial_files = {
        'prp': RESULTS_DIR / '4a_prp_trials.csv',
        'stroop': RESULTS_DIR / '4c_stroop_trials.csv',
        'wcst': RESULTS_DIR / '4b_wcst_trials.csv'
    }

    if task not in trial_files:
        return pd.DataFrame()

    path = trial_files[task]
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    return df


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_fatigue_slopes(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    피로 기울기 분석

    시행에 따른 RT/정확도 변화 기울기 계산
    """
    output_dir = OUTPUT_DIR / "slopes"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[FATIGUE SLOPES] Computing performance change across trials...")

    master = load_master_data()
    all_results = []
    significant_findings = []

    for task in ['wcst', 'stroop', 'prp']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            if verbose:
                print(f"  {task.upper()}: Insufficient data")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        # Get columns
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        correct_col = None
        for c in ['correct', 'iscorrect', 'is_correct']:
            if c in trials.columns:
                correct_col = c
                break

        if correct_col is None:
            continue

        # Sort trials
        sort_cols = ['participant_id']
        for cand in ['trialindex', 'trial_index', 'timestamp', 'idx']:
            if cand in trials.columns:
                sort_cols.append(cand)
                break
        trials = trials.sort_values(sort_cols)

        # Filter valid trials
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Compute fatigue metrics per participant
        fatigue_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.reset_index(drop=True)
            n_trials = len(pdata)
            trial_nums = np.arange(n_trials)

            rts = pdata[rt_col].values
            correct = pdata[correct_col].astype(float).values

            # RT slope (positive = slowing = fatigue)
            valid_rt = ~np.isnan(rts)
            if valid_rt.sum() > 10:
                rt_slope, _, rt_r, rt_p, _ = stats.linregress(trial_nums[valid_rt], rts[valid_rt])
            else:
                rt_slope, rt_r, rt_p = np.nan, np.nan, np.nan

            # Accuracy slope (negative = decline = fatigue)
            valid_acc = ~np.isnan(correct)
            if valid_acc.sum() > 10:
                acc_slope, _, acc_r, acc_p, _ = stats.linregress(trial_nums[valid_acc], correct[valid_acc])
            else:
                acc_slope, acc_r, acc_p = np.nan, np.nan, np.nan

            # Quartile-based metrics
            q1_end = n_trials // 4
            q4_start = 3 * n_trials // 4

            q1_rt = np.nanmean(rts[:q1_end])
            q4_rt = np.nanmean(rts[q4_start:])
            q1_acc = np.nanmean(correct[:q1_end])
            q4_acc = np.nanmean(correct[q4_start:])

            fatigue_results.append({
                'participant_id': pid,
                'task': task,
                'n_trials': n_trials,
                'rt_slope': rt_slope,
                'rt_slope_r': rt_r,
                'rt_slope_p': rt_p,
                'acc_slope': acc_slope,
                'acc_slope_r': acc_r,
                'acc_slope_p': acc_p,
                'q1_rt': q1_rt,
                'q4_rt': q4_rt,
                'rt_q4_q1_diff': q4_rt - q1_rt,
                'q1_acc': q1_acc,
                'q4_acc': q4_acc,
                'acc_q4_q1_diff': q4_acc - q1_acc,
                'mean_rt': np.nanmean(rts),
                'mean_acc': np.nanmean(correct)
            })

        fatigue_df = pd.DataFrame(fatigue_results)

        if len(fatigue_df) < 20:
            continue

        # Merge with master
        merged = master.merge(fatigue_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean RT slope: {merged['rt_slope'].mean():.2f} ms/trial")
            print(f"    Mean Accuracy slope: {merged['acc_slope'].mean():.4f} /trial")

        # Regression: UCLA → Fatigue slopes
        outcomes = [
            ('rt_slope', 'RT Fatigue Slope'),
            ('rt_q4_q1_diff', 'RT Q4-Q1 Difference'),
            ('acc_slope', 'Accuracy Fatigue Slope'),
            ('acc_q4_q1_diff', 'Accuracy Q4-Q1 Difference')
        ]

        for outcome, label in outcomes:
            if outcome not in merged.columns:
                continue

            clean_data = merged.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

            if len(clean_data) < 30:
                continue

            try:
                formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                result = {
                    'task': task,
                    'outcome': outcome,
                    'outcome_label': label,
                    'n': len(clean_data),
                    'ucla_beta': model.params.get('z_ucla', np.nan),
                    'ucla_se': model.bse.get('z_ucla', np.nan),
                    'ucla_p': model.pvalues.get('z_ucla', np.nan),
                    'interaction_beta': model.params.get('z_ucla:C(gender_male)[T.1]', np.nan),
                    'interaction_p': model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan),
                    'r_squared': model.rsquared
                }
                all_results.append(result)

                sig = "*" if result['ucla_p'] < 0.05 else ""
                if verbose:
                    print(f"    UCLA → {label}: β={result['ucla_beta']:.4f}, p={result['ucla_p']:.4f} {sig}")

                if result['ucla_p'] < 0.05:
                    significant_findings.append({
                        'analysis': 'Fatigue Analysis',
                        'outcome': f"{task.upper()} {label}",
                        'effect': 'UCLA main',
                        'beta': result['ucla_beta'],
                        'p': result['ucla_p'],
                        'n': result['n']
                    })

                if result['interaction_p'] < 0.05:
                    significant_findings.append({
                        'analysis': 'Fatigue Analysis',
                        'outcome': f"{task.upper()} {label}",
                        'effect': 'UCLA x Gender',
                        'beta': result['interaction_beta'],
                        'p': result['interaction_p'],
                        'n': result['n']
                    })

            except Exception as e:
                if verbose:
                    print(f"    {label}: ERROR - {e}")

        # Save task-specific
        fatigue_df.to_csv(output_dir / f"fatigue_{task}.csv", index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "fatigue_regression.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


def analyze_fatigue_moderation(verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    피로 상태에서의 UCLA 효과 분석

    후반부 시행(피로 상태)에서 UCLA 효과가 강화되는지 검정
    """
    output_dir = OUTPUT_DIR / "moderation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[FATIGUE MODERATION] Testing UCLA effects under cognitive depletion...")

    master = load_master_data()
    all_results = []
    significant_findings = []

    for task in ['wcst', 'stroop']:
        trials = load_trial_data(task)

        if len(trials) < 100:
            continue

        if verbose:
            print(f"\n  {task.upper()}")

        # Get columns
        rt_col = 'rt_ms' if 'rt_ms' in trials.columns else 'rt'
        correct_col = None
        for c in ['correct', 'iscorrect', 'is_correct']:
            if c in trials.columns:
                correct_col = c
                break

        if correct_col is None:
            continue

        # Sort and filter
        sort_cols = ['participant_id']
        for cand in ['trialindex', 'trial_index', 'timestamp']:
            if cand in trials.columns:
                sort_cols.append(cand)
                break
        trials = trials.sort_values(sort_cols)
        trials = trials[(trials[rt_col] > 100) & (trials[rt_col] < 3000)].copy()

        # Split into early (Q1) vs late (Q4)
        early_results = []
        late_results = []

        for pid, pdata in trials.groupby('participant_id'):
            n = len(pdata)
            if n < 40:
                continue

            q1_end = n // 4
            q4_start = 3 * n // 4

            early = pdata.iloc[:q1_end]
            late = pdata.iloc[q4_start:]

            early_results.append({
                'participant_id': pid,
                'task': task,
                'phase': 'early',
                'rt': early[rt_col].mean(),
                'accuracy': early[correct_col].mean()
            })

            late_results.append({
                'participant_id': pid,
                'task': task,
                'phase': 'late',
                'rt': late[rt_col].mean(),
                'accuracy': late[correct_col].mean()
            })

        early_df = pd.DataFrame(early_results)
        late_df = pd.DataFrame(late_results)

        # Keep only necessary columns from master to avoid conflicts
        master_cols = ['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', 'ucla_total']
        master_subset = master[[c for c in master_cols if c in master.columns]].copy()

        # Merge with master and compare UCLA effects
        for phase, phase_df, phase_label in [('early', early_df, 'Early (Q1)'), ('late', late_df, 'Late (Q4)')]:
            if len(phase_df) < 20:
                continue

            merged = master_subset.merge(phase_df, on='participant_id', how='inner')

            for outcome, outcome_label in [('accuracy', 'Accuracy')]:
                clean_data = merged.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])

                if len(clean_data) < 25:
                    continue

                try:
                    formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                    model = smf.ols(formula, data=clean_data).fit(cov_type='HC3')

                    result = {
                        'task': task,
                        'phase': phase,
                        'phase_label': phase_label,
                        'outcome': outcome,
                        'n': len(clean_data),
                        'ucla_beta': model.params.get('z_ucla', np.nan),
                        'ucla_p': model.pvalues.get('z_ucla', np.nan),
                        'interaction_beta': model.params.get('z_ucla:C(gender_male)[T.1]', np.nan),
                        'interaction_p': model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)
                    }
                    all_results.append(result)

                    sig = "*" if result['ucla_p'] < 0.05 else ""
                    if verbose:
                        print(f"    {phase_label} {outcome_label}: UCLA β={result['ucla_beta']:.4f}, p={result['ucla_p']:.4f} {sig}")

                    if result['ucla_p'] < 0.05:
                        significant_findings.append({
                            'analysis': 'Fatigue Moderation',
                            'outcome': f"{task.upper()} {phase_label} {outcome_label}",
                            'effect': 'UCLA main',
                            'beta': result['ucla_beta'],
                            'p': result['ucla_p'],
                            'n': result['n']
                        })

                except Exception as e:
                    if verbose:
                        print(f"    {phase_label}: ERROR - {e}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(output_dir / "fatigue_moderation.csv", index=False, encoding='utf-8-sig')

    return results_df, significant_findings


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
    'slopes': ('Fatigue slope analysis', analyze_fatigue_slopes),
    'moderation': ('UCLA effect moderation by fatigue', analyze_fatigue_moderation),
}


def run(analysis: Optional[str] = None, verbose: bool = True, record_results: bool = True) -> Dict:
    """Run fatigue analyses."""
    if verbose:
        print("=" * 70)
        print("WITHIN-BLOCK FATIGUE ANALYSIS SUITE")
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
    parser = argparse.ArgumentParser(description="Fatigue Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    args = parser.parse_args()

    run(analysis=args.analysis, verbose=not args.quiet, record_results=not args.no_record)
