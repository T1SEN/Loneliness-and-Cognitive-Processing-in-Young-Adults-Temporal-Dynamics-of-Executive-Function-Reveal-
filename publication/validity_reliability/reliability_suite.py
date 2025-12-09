"""
Reliability Analysis Suite
==========================

Internal consistency and split-half reliability for online experiment measures.

Analyses:
1. Survey Internal Consistency (Cronbach's Alpha)
   - UCLA Loneliness Scale (20 items)
   - DASS-21 total and subscales (Depression, Anxiety, Stress)

2. Cognitive Task Reliability (Split-Half with Spearman-Brown correction)
   - Stroop: Odd/even trial interference effect correlation
   - WCST: First/second half perseverative error rate correlation
   - PRP: Odd/even trial bottleneck effect correlation

Usage:
    python -m publication.validity_reliability.reliability_suite

Output:
    results/analysis_outputs/validity_reliability/
    - reliability_results.csv
    - reliability_summary.json

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from publication.preprocessing import RESULTS_DIR, ANALYSIS_OUTPUT_DIR
from publication.preprocessing import load_stroop_trials, load_wcst_trials, load_prp_trials
from publication.preprocessing import DEFAULT_RT_MIN, STROOP_RT_MAX, PRP_RT_MAX

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "validity_reliability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cronbach_alpha(df: pd.DataFrame) -> float:
    """
    Calculate Cronbach's alpha for internal consistency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with items as columns, participants as rows

    Returns
    -------
    float
        Cronbach's alpha coefficient
    """
    df_clean = df.dropna()
    if len(df_clean) < 2:
        return np.nan

    n_items = df_clean.shape[1]
    if n_items < 2:
        return np.nan

    item_variances = df_clean.var(axis=0, ddof=1)
    total_variance = df_clean.sum(axis=1).var(ddof=1)

    if total_variance == 0:
        return np.nan

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha


def spearman_brown_correction(r: float) -> float:
    """
    Apply Spearman-Brown prophecy formula to split-half correlation.

    Parameters
    ----------
    r : float
        Split-half correlation coefficient

    Returns
    -------
    float
        Corrected reliability estimate
    """
    if np.isnan(r) or r <= -1:
        return np.nan
    return (2 * r) / (1 + r)


def interpret_alpha(alpha: float) -> str:
    """Interpret Cronbach's alpha value."""
    if np.isnan(alpha):
        return "N/A"
    if alpha >= 0.9:
        return "Excellent"
    if alpha >= 0.8:
        return "Good"
    if alpha >= 0.7:
        return "Acceptable"
    if alpha >= 0.6:
        return "Questionable"
    if alpha >= 0.5:
        return "Poor"
    return "Unacceptable"


# =============================================================================
# SURVEY RELIABILITY
# =============================================================================

def load_survey_items() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load survey item-level responses.

    Returns
    -------
    tuple
        (ucla_items, dass_items) DataFrames with item responses
    """
    surveys_path = RESULTS_DIR / "2_surveys_results.csv"
    surveys = pd.read_csv(surveys_path, encoding='utf-8-sig')

    # UCLA items (q1-q20)
    ucla_df = surveys[surveys['surveyName'] == 'ucla'].copy()
    ucla_item_cols = [f'q{i}' for i in range(1, 21)]
    ucla_items = ucla_df[['participantId'] + ucla_item_cols].dropna(subset=ucla_item_cols, how='all')

    # DASS items (q1-q21)
    dass_df = surveys[surveys['surveyName'] == 'dass'].copy()
    dass_item_cols = [f'q{i}' for i in range(1, 22)]
    dass_items = dass_df[['participantId'] + dass_item_cols].dropna(subset=dass_item_cols, how='all')

    return ucla_items, dass_items


def calculate_survey_reliability() -> pd.DataFrame:
    """
    Calculate Cronbach's alpha for UCLA and DASS scales.

    Returns
    -------
    pd.DataFrame
        Reliability results for each scale/subscale
    """
    ucla_items, dass_items = load_survey_items()

    results = []

    # UCLA total (20 items)
    # Reverse-coded items (1-indexed): 1, 5, 6, 9, 10, 15, 16, 19, 20
    # These need to be reverse-scored before calculating alpha
    ucla_item_cols = [f'q{i}' for i in range(1, 21)]
    ucla_reverse_items = ['q1', 'q5', 'q6', 'q9', 'q10', 'q15', 'q16', 'q19', 'q20']

    # Create a copy with reversed items
    ucla_for_alpha = ucla_items[ucla_item_cols].copy()
    # UCLA uses 1-4 scale, so reverse is: 5 - value
    for col in ucla_reverse_items:
        if col in ucla_for_alpha.columns:
            ucla_for_alpha[col] = 5 - ucla_for_alpha[col]

    ucla_alpha = cronbach_alpha(ucla_for_alpha)
    results.append({
        'scale': 'UCLA Loneliness Scale',
        'n_items': 20,
        'n_participants': len(ucla_items.dropna(subset=ucla_item_cols)),
        'cronbach_alpha': ucla_alpha,
        'interpretation': interpret_alpha(ucla_alpha)
    })

    # DASS-21 total (21 items)
    dass_item_cols = [f'q{i}' for i in range(1, 22)]
    dass_alpha = cronbach_alpha(dass_items[dass_item_cols])
    results.append({
        'scale': 'DASS-21 Total',
        'n_items': 21,
        'n_participants': len(dass_items.dropna(subset=dass_item_cols)),
        'cronbach_alpha': dass_alpha,
        'interpretation': interpret_alpha(dass_alpha)
    })

    # DASS subscales (based on lib/pages/survey/dass_page.dart)
    # Depression: items 3, 5, 10, 13, 16, 17, 21 (1-indexed)
    # Anxiety: items 2, 4, 7, 9, 15, 19, 20
    # Stress: items 1, 6, 8, 11, 12, 14, 18
    dass_subscales = {
        'DASS-21 Depression': [3, 5, 10, 13, 16, 17, 21],
        'DASS-21 Anxiety': [2, 4, 7, 9, 15, 19, 20],
        'DASS-21 Stress': [1, 6, 8, 11, 12, 14, 18]
    }

    for subscale_name, item_nums in dass_subscales.items():
        subscale_cols = [f'q{i}' for i in item_nums]
        subscale_alpha = cronbach_alpha(dass_items[subscale_cols])
        results.append({
            'scale': subscale_name,
            'n_items': len(item_nums),
            'n_participants': len(dass_items.dropna(subset=subscale_cols)),
            'cronbach_alpha': subscale_alpha,
            'interpretation': interpret_alpha(subscale_alpha)
        })

    return pd.DataFrame(results)


# =============================================================================
# COGNITIVE TASK RELIABILITY
# =============================================================================

def calculate_stroop_reliability() -> dict:
    """
    Calculate split-half reliability for Stroop interference effect.

    Method: Odd/even trial split, calculate interference per half, correlate.
    """
    stroop, _ = load_stroop_trials()  # Returns (df, metadata) tuple

    # Determine RT column name
    rt_col = 'rt_ms' if 'rt_ms' in stroop.columns else 'rt'
    pid_col = 'participantId' if 'participantId' in stroop.columns else 'participant_id'

    # Filter valid trials
    stroop = stroop[
        (stroop['timeout'] == False) &
        (stroop[rt_col] > DEFAULT_RT_MIN) &
        (stroop[rt_col] < STROOP_RT_MAX) &
        (stroop['correct'] == True)
    ].copy()

    # Add trial number within participant
    stroop['trial_num'] = stroop.groupby(pid_col).cumcount()
    stroop['half'] = stroop['trial_num'] % 2  # 0 = even, 1 = odd

    # Determine condition column
    cond_col = None
    for col in ['congruency', 'condition', 'type', 'cond']:
        if col in stroop.columns:
            cond_col = col
            break

    if cond_col is None:
        return {'r': np.nan, 'r_sb': np.nan, 'n': 0}

    # Calculate interference per participant per half
    interference_data = []
    for pid in stroop[pid_col].unique():
        pid_data = stroop[stroop[pid_col] == pid]

        for half in [0, 1]:
            half_data = pid_data[pid_data['half'] == half]

            # Get congruent/incongruent RTs
            cong_rt = half_data[half_data[cond_col].str.lower().str.contains('congruent') & ~half_data[cond_col].str.lower().str.contains('incongruent')][rt_col].mean()
            incong_rt = half_data[half_data[cond_col].str.lower().str.contains('incongruent')][rt_col].mean()

            if pd.notna(cong_rt) and pd.notna(incong_rt):
                interference_data.append({
                    'participantId': pid,
                    'half': half,
                    'interference': incong_rt - cong_rt
                })

    if not interference_data:
        return {'r': np.nan, 'r_sb': np.nan, 'n': 0}

    int_df = pd.DataFrame(interference_data)
    int_pivot = int_df.pivot(index='participantId', columns='half', values='interference')
    int_pivot = int_pivot.dropna()

    if len(int_pivot) < 3:
        return {'r': np.nan, 'r_sb': np.nan, 'n': len(int_pivot)}

    r, _ = stats.pearsonr(int_pivot[0], int_pivot[1])
    r_sb = spearman_brown_correction(r)

    return {'r': r, 'r_sb': r_sb, 'n': len(int_pivot)}


def calculate_wcst_reliability() -> dict:
    """
    Calculate split-half reliability for WCST perseverative error rate.

    Method: First/second half split, calculate PE rate per half, correlate.
    """
    wcst, _ = load_wcst_trials()  # Returns (df, metadata) tuple

    pid_col = 'participantId' if 'participantId' in wcst.columns else 'participant_id'

    # Filter valid trials
    wcst = wcst[wcst['timeout'] == False].copy()

    # Add trial number within participant
    wcst['trial_num'] = wcst.groupby(pid_col).cumcount()
    max_trials = wcst.groupby(pid_col)['trial_num'].transform('max')
    wcst['half'] = (wcst['trial_num'] > max_trials / 2).astype(int)

    # Calculate PE rate per participant per half
    pe_data = []
    for pid in wcst[pid_col].unique():
        pid_data = wcst[wcst[pid_col] == pid]

        for half in [0, 1]:
            half_data = pid_data[pid_data['half'] == half]

            if len(half_data) < 5:
                continue

            # Perseverative errors
            if 'is_perseverative_error' in half_data.columns:
                pe_rate = half_data['is_perseverative_error'].mean()
            elif 'errorType' in half_data.columns:
                pe_rate = (half_data['errorType'] == 'perseverative').mean()
            else:
                pe_rate = (~half_data['correct']).mean() if 'correct' in half_data.columns else np.nan

            if pd.notna(pe_rate):
                pe_data.append({
                    'participantId': pid,
                    'half': half,
                    'pe_rate': pe_rate
                })

    if not pe_data:
        return {'r': np.nan, 'r_sb': np.nan, 'n': 0}

    pe_df = pd.DataFrame(pe_data)
    pe_pivot = pe_df.pivot(index='participantId', columns='half', values='pe_rate')
    pe_pivot = pe_pivot.dropna()

    if len(pe_pivot) < 3:
        return {'r': np.nan, 'r_sb': np.nan, 'n': len(pe_pivot)}

    r, _ = stats.pearsonr(pe_pivot[0], pe_pivot[1])
    r_sb = spearman_brown_correction(r)

    return {'r': r, 'r_sb': r_sb, 'n': len(pe_pivot)}


def calculate_prp_reliability() -> dict:
    """
    Calculate split-half reliability for PRP bottleneck effect.

    Method: Odd/even trial split, calculate bottleneck per half, correlate.
    """
    prp, _ = load_prp_trials()  # Returns (df, metadata) tuple

    # Determine column names (PRP has t2_rt for Task 2 response time)
    rt_col = 't2_rt' if 't2_rt' in prp.columns else ('rt_ms' if 'rt_ms' in prp.columns else 'rt')
    pid_col = 'participantId' if 'participantId' in prp.columns else 'participant_id'
    soa_col = 'soa' if 'soa' in prp.columns else ('soa_ms' if 'soa_ms' in prp.columns else None)
    timeout_col = 't2_timeout' if 't2_timeout' in prp.columns else 'timeout'

    if soa_col is None:
        return {'r': np.nan, 'r_sb': np.nan, 'n': 0}

    # Filter valid trials (Task 2 RT)
    if timeout_col in prp.columns:
        prp = prp[prp[timeout_col] == False].copy()
    else:
        prp = prp.copy()

    prp = prp[
        (prp[rt_col] > DEFAULT_RT_MIN) &
        (prp[rt_col] < PRP_RT_MAX)
    ]

    # Add trial number within participant
    prp['trial_num'] = prp.groupby(pid_col).cumcount()
    prp['half'] = prp['trial_num'] % 2  # 0 = even, 1 = odd

    # Calculate bottleneck per participant per half
    bottleneck_data = []
    for pid in prp[pid_col].unique():
        pid_data = prp[prp[pid_col] == pid]

        for half in [0, 1]:
            half_data = pid_data[pid_data['half'] == half]

            short_soa = half_data[half_data[soa_col] <= 150][rt_col].mean()
            long_soa = half_data[half_data[soa_col] >= 1200][rt_col].mean()

            if pd.notna(short_soa) and pd.notna(long_soa):
                bottleneck_data.append({
                    'participantId': pid,
                    'half': half,
                    'bottleneck': short_soa - long_soa
                })

    if not bottleneck_data:
        return {'r': np.nan, 'r_sb': np.nan, 'n': 0}

    bn_df = pd.DataFrame(bottleneck_data)
    bn_pivot = bn_df.pivot(index='participantId', columns='half', values='bottleneck')
    bn_pivot = bn_pivot.dropna()

    if len(bn_pivot) < 3:
        return {'r': np.nan, 'r_sb': np.nan, 'n': len(bn_pivot)}

    r, _ = stats.pearsonr(bn_pivot[0], bn_pivot[1])
    r_sb = spearman_brown_correction(r)

    return {'r': r, 'r_sb': r_sb, 'n': len(bn_pivot)}


def calculate_cognitive_reliability() -> pd.DataFrame:
    """
    Calculate split-half reliability for all cognitive tasks.

    Returns
    -------
    pd.DataFrame
        Reliability results for each task
    """
    results = []

    # Stroop
    stroop_rel = calculate_stroop_reliability()
    results.append({
        'task': 'Stroop',
        'measure': 'Interference Effect',
        'method': 'Odd/Even Split-Half',
        'n_participants': stroop_rel['n'],
        'split_half_r': stroop_rel['r'],
        'spearman_brown_r': stroop_rel['r_sb'],
        'interpretation': interpret_alpha(stroop_rel['r_sb'])
    })

    # WCST
    wcst_rel = calculate_wcst_reliability()
    results.append({
        'task': 'WCST',
        'measure': 'Perseverative Error Rate',
        'method': 'First/Second Half Split',
        'n_participants': wcst_rel['n'],
        'split_half_r': wcst_rel['r'],
        'spearman_brown_r': wcst_rel['r_sb'],
        'interpretation': interpret_alpha(wcst_rel['r_sb'])
    })

    # PRP
    prp_rel = calculate_prp_reliability()
    results.append({
        'task': 'PRP',
        'measure': 'Bottleneck Effect',
        'method': 'Odd/Even Split-Half',
        'n_participants': prp_rel['n'],
        'split_half_r': prp_rel['r'],
        'spearman_brown_r': prp_rel['r_sb'],
        'interpretation': interpret_alpha(prp_rel['r_sb'])
    })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def run():
    """Run all reliability analyses."""
    print("=" * 60)
    print("RELIABILITY ANALYSIS SUITE")
    print("=" * 60)

    # Survey reliability
    print("\n[1] Survey Internal Consistency (Cronbach's Alpha)")
    print("-" * 50)
    survey_results = calculate_survey_reliability()

    for _, row in survey_results.iterrows():
        print(f"  {row['scale']}")
        print(f"    Items: {row['n_items']}, N: {row['n_participants']}")
        print(f"    Alpha: {row['cronbach_alpha']:.3f} ({row['interpretation']})")

    # Cognitive task reliability
    print("\n[2] Cognitive Task Reliability (Split-Half)")
    print("-" * 50)
    cognitive_results = calculate_cognitive_reliability()

    for _, row in cognitive_results.iterrows():
        print(f"  {row['task']} - {row['measure']}")
        print(f"    Method: {row['method']}, N: {row['n_participants']}")
        if pd.notna(row['split_half_r']):
            print(f"    r = {row['split_half_r']:.3f}, Spearman-Brown = {row['spearman_brown_r']:.3f} ({row['interpretation']})")
        else:
            print("    Insufficient data")

    # Save results
    survey_results.to_csv(OUTPUT_DIR / "survey_reliability.csv", index=False, encoding='utf-8-sig')
    cognitive_results.to_csv(OUTPUT_DIR / "cognitive_reliability.csv", index=False, encoding='utf-8-sig')

    # Summary JSON
    summary = {
        'survey_reliability': {
            'UCLA': {
                'alpha': float(survey_results[survey_results['scale'] == 'UCLA Loneliness Scale']['cronbach_alpha'].values[0]),
                'n_items': 20
            },
            'DASS_total': {
                'alpha': float(survey_results[survey_results['scale'] == 'DASS-21 Total']['cronbach_alpha'].values[0]),
                'n_items': 21
            },
            'DASS_depression': {
                'alpha': float(survey_results[survey_results['scale'] == 'DASS-21 Depression']['cronbach_alpha'].values[0]),
                'n_items': 7
            },
            'DASS_anxiety': {
                'alpha': float(survey_results[survey_results['scale'] == 'DASS-21 Anxiety']['cronbach_alpha'].values[0]),
                'n_items': 7
            },
            'DASS_stress': {
                'alpha': float(survey_results[survey_results['scale'] == 'DASS-21 Stress']['cronbach_alpha'].values[0]),
                'n_items': 7
            }
        },
        'cognitive_reliability': {
            'stroop': {
                'r_sb': float(cognitive_results[cognitive_results['task'] == 'Stroop']['spearman_brown_r'].values[0]) if not np.isnan(cognitive_results[cognitive_results['task'] == 'Stroop']['spearman_brown_r'].values[0]) else None,
                'method': 'odd_even_split_half'
            },
            'wcst': {
                'r_sb': float(cognitive_results[cognitive_results['task'] == 'WCST']['spearman_brown_r'].values[0]) if not np.isnan(cognitive_results[cognitive_results['task'] == 'WCST']['spearman_brown_r'].values[0]) else None,
                'method': 'first_second_half_split'
            },
            'prp': {
                'r_sb': float(cognitive_results[cognitive_results['task'] == 'PRP']['spearman_brown_r'].values[0]) if not np.isnan(cognitive_results[cognitive_results['task'] == 'PRP']['spearman_brown_r'].values[0]) else None,
                'method': 'odd_even_split_half'
            }
        },
        'interpretation_guide': {
            '>=0.9': 'Excellent',
            '>=0.8': 'Good',
            '>=0.7': 'Acceptable',
            '>=0.6': 'Questionable',
            '>=0.5': 'Poor',
            '<0.5': 'Unacceptable'
        },
        'note_cognitive_tasks': (
            'Cognitive task reliability (especially difference scores like Stroop interference) '
            'is typically lower (r = 0.50-0.70) than survey measures due to inherent measurement properties. '
            'Values in this range are considered acceptable for cognitive paradigms. '
            'See: Hedge et al. (2018) Behavior Research Methods.'
        )
    }

    with open(OUTPUT_DIR / "reliability_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("  - survey_reliability.csv")
    print("  - cognitive_reliability.csv")
    print("  - reliability_summary.json")
    print("=" * 60)

    return survey_results, cognitive_results


if __name__ == "__main__":
    run()
