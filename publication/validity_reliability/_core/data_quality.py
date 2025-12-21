"""
Data Quality Analysis
=====================

Online experiment data quality validation for academic publication.

Analyses:
1. Survey Response Quality
   - Response duration analysis (too fast = careless responding)
   - Straight-line response pattern detection
   - Reverse-coded item consistency check

2. Cognitive Task Quality
   - RT distribution (anticipations below task-specific RT min, timeouts)
   - Accuracy-based exclusion (below chance performance)
   - Trial count sufficiency

3. Sample Attrition Report
   - Original N -> exclusion criteria -> Final analysis N

Usage:
    python -m publication.validity_reliability.complete_overall.data_quality

Output:
    publication/data/outputs/validity_reliability/<dataset>/
    - data_quality_report.csv
    - exclusion_summary.json
    - rt_distribution.png

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
from typing import Iterable, List

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR, RAW_DIR, get_results_dir
from publication.preprocessing import ensure_participant_id
from publication.preprocessing import PRP_RT_MAX, PRP_RT_MIN, STROOP_RT_MAX, STROOP_RT_MIN, WCST_RT_MIN

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "validity_reliability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_data_dir(data_dir: Path | None, fallback_task: str) -> Path:
    if data_dir is not None:
        return data_dir
    return get_results_dir(fallback_task)


def _normalize_tasks(tasks: Iterable[str] | None) -> List[str]:
    if tasks is None:
        return ["stroop", "wcst", "prp"]
    normalized = []
    for task in tasks:
        if not task:
            continue
        normalized.append(str(task).strip().lower())
    return normalized


# =============================================================================
# SURVEY RESPONSE QUALITY
# =============================================================================

def load_survey_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load survey responses with duration info."""
    data_dir = _resolve_data_dir(data_dir, "overall")
    surveys_path = data_dir / "2_surveys_results.csv"
    surveys = pd.read_csv(surveys_path, encoding='utf-8-sig')

    ucla_df = surveys[surveys['surveyName'] == 'ucla'].copy()
    dass_df = surveys[surveys['surveyName'] == 'dass'].copy()

    return ucla_df, dass_df


def analyze_survey_duration(df: pd.DataFrame, survey_name: str, n_items: int) -> dict:
    """
    Analyze survey response duration.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data with duration_seconds column
    survey_name : str
        Name of survey (ucla or dass)
    n_items : int
        Number of items in survey

    Returns
    -------
    dict
        Duration statistics and flagged participants
    """
    df_valid = df[df['duration_seconds'].notna()].copy()

    if len(df_valid) == 0:
        return {'n_total': 0, 'n_valid_duration': 0}

    durations = df_valid['duration_seconds']
    sec_per_item = durations / n_items

    # Thresholds
    min_sec_per_item = 2.0  # Less than 2 sec/item = too fast
    max_sec_per_item = 60.0  # More than 60 sec/item = distracted

    too_fast = df_valid[sec_per_item < min_sec_per_item]['participantId'].tolist()
    too_slow = df_valid[sec_per_item > max_sec_per_item]['participantId'].tolist()

    return {
        'survey': survey_name,
        'n_items': n_items,
        'n_total': len(df),
        'n_valid_duration': len(df_valid),
        'duration_mean': durations.mean(),
        'duration_median': durations.median(),
        'duration_sd': durations.std(),
        'duration_min': durations.min(),
        'duration_max': durations.max(),
        'sec_per_item_mean': sec_per_item.mean(),
        'sec_per_item_median': sec_per_item.median(),
        'n_too_fast': len(too_fast),
        'pct_too_fast': len(too_fast) / len(df_valid) * 100 if len(df_valid) > 0 else 0,
        'too_fast_threshold': f'<{min_sec_per_item}s/item',
        'too_fast_participants': too_fast,
        'n_too_slow': len(too_slow),
        'pct_too_slow': len(too_slow) / len(df_valid) * 100 if len(df_valid) > 0 else 0,
        'too_slow_threshold': f'>{max_sec_per_item}s/item',
        'too_slow_participants': too_slow
    }


def detect_straight_line_responding(df: pd.DataFrame, item_cols: list) -> dict:
    """
    Detect straight-line (all same response) patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data with item columns
    item_cols : list
        List of item column names

    Returns
    -------
    dict
        Straight-line detection results
    """
    df_valid = df.dropna(subset=item_cols)

    if len(df_valid) == 0:
        return {'n_checked': 0, 'n_straight_line': 0}

    # Check if all responses are the same
    responses = df_valid[item_cols].values
    straight_line_mask = np.all(responses == responses[:, 0:1], axis=1)

    straight_line_participants = df_valid[straight_line_mask]['participantId'].tolist()

    return {
        'n_checked': len(df_valid),
        'n_straight_line': len(straight_line_participants),
        'pct_straight_line': len(straight_line_participants) / len(df_valid) * 100,
        'straight_line_participants': straight_line_participants
    }


def check_reverse_item_consistency(df: pd.DataFrame, item_cols: list, reverse_items: list) -> dict:
    """
    Check consistency between regular and reverse-coded items.

    A participant with very high scores on regular items and also
    high scores on reverse items (before reverse coding) may be careless.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data
    item_cols : list
        All item column names
    reverse_items : list
        Indices of reverse-coded items (0-indexed)

    Returns
    -------
    dict
        Consistency check results
    """
    df_valid = df.dropna(subset=item_cols)

    if len(df_valid) == 0:
        return {'n_checked': 0}

    responses = df_valid[item_cols].values

    # Calculate correlation between regular and reverse item means
    regular_idx = [i for i in range(len(item_cols)) if i not in reverse_items]

    if len(regular_idx) == 0 or len(reverse_items) == 0:
        return {'n_checked': len(df_valid), 'correlation': np.nan}

    regular_means = responses[:, regular_idx].mean(axis=1)
    reverse_means = responses[:, reverse_items].mean(axis=1)

    # For a valid respondent, regular and reverse item means should be NEGATIVELY correlated
    # (high loneliness on regular = low on reverse before coding)
    r, p = stats.pearsonr(regular_means, reverse_means)

    return {
        'n_checked': len(df_valid),
        'regular_reverse_correlation': r,
        'correlation_p': p,
        'expected_direction': 'negative',
        'interpretation': 'Positive correlation may indicate careless responding'
    }


def analyze_survey_quality(data_dir: Path | None = None) -> dict:
    """
    Comprehensive survey response quality analysis.

    Returns
    -------
    dict
        All survey quality metrics
    """
    ucla_df, dass_df = load_survey_data(data_dir=data_dir)

    results = {}

    # UCLA quality
    ucla_item_cols = [f'q{i}' for i in range(1, 21)]
    # UCLA reverse-coded items (0-indexed): 0, 4, 5, 8, 9, 14, 15, 18, 19 (items 1,5,6,9,10,15,16,19,20)
    ucla_reverse = [0, 4, 5, 8, 9, 14, 15, 18, 19]

    results['ucla_duration'] = analyze_survey_duration(ucla_df, 'UCLA', 20)
    results['ucla_straight_line'] = detect_straight_line_responding(ucla_df, ucla_item_cols)
    results['ucla_reverse_consistency'] = check_reverse_item_consistency(ucla_df, ucla_item_cols, ucla_reverse)

    # DASS quality
    dass_item_cols = [f'q{i}' for i in range(1, 22)]
    # DASS has no reverse-coded items

    results['dass_duration'] = analyze_survey_duration(dass_df, 'DASS', 21)
    results['dass_straight_line'] = detect_straight_line_responding(dass_df, dass_item_cols)

    return results


# =============================================================================
# COGNITIVE TASK QUALITY
# =============================================================================

def analyze_rt_quality(task_name: str, trials_df: pd.DataFrame, rt_min: float, rt_max: float) -> dict:
    """
    Analyze RT distribution and quality for a cognitive task.

    Parameters
    ----------
    task_name : str
        Name of task
    trials_df : pd.DataFrame
        Trial-level data
    rt_min : float
        Minimum valid RT (ms)
    rt_max : float
        Maximum valid RT (ms)

    Returns
    -------
    dict
        RT quality metrics
    """
    n_total = len(trials_df)

    if n_total == 0:
        return {'task': task_name, 'n_total_trials': 0}

    # Determine RT column name
    rt_col = None
    for col in ['rt_ms', 'rt', 't2_rt']:
        if col in trials_df.columns:
            rt_col = col
            break

    # Determine timeout column
    timeout_col = None
    for col in ['timeout', 't2_timeout', 'is_timeout']:
        if col in trials_df.columns:
            timeout_col = col
            break

    # Count timeouts
    if timeout_col:
        n_timeout = trials_df[timeout_col].sum()
    else:
        n_timeout = 0

    # Count anticipations (RT too fast)
    if rt_col:
        n_anticipation = (trials_df[rt_col] < rt_min).sum()
        n_slow = (trials_df[rt_col] > rt_max).sum()

        valid_rt = trials_df[(trials_df[rt_col] >= rt_min) & (trials_df[rt_col] <= rt_max)][rt_col]

        rt_stats = {
            'rt_mean': valid_rt.mean(),
            'rt_median': valid_rt.median(),
            'rt_sd': valid_rt.std(),
            'rt_min': valid_rt.min(),
            'rt_max': valid_rt.max()
        }
    else:
        n_anticipation = 0
        n_slow = 0
        rt_stats = {}

    # Count valid trials
    valid_mask = pd.Series([True] * n_total, index=trials_df.index)
    if timeout_col:
        valid_mask = valid_mask & (trials_df[timeout_col] == False)
    if rt_col:
        valid_mask = valid_mask & (trials_df[rt_col] >= rt_min) & (trials_df[rt_col] <= rt_max)

    n_valid = valid_mask.sum()

    return {
        'task': task_name,
        'n_total_trials': n_total,
        'n_timeout': n_timeout,
        'pct_timeout': n_timeout / n_total * 100,
        'n_anticipation': n_anticipation,
        'pct_anticipation': n_anticipation / n_total * 100,
        'n_slow': n_slow,
        'pct_slow': n_slow / n_total * 100,
        'n_valid': n_valid,
        'pct_valid': n_valid / n_total * 100,
        'rt_threshold_min': rt_min,
        'rt_threshold_max': rt_max,
        **rt_stats
    }


def analyze_participant_accuracy(task_name: str, trials_df: pd.DataFrame, chance_level: float = 0.5) -> dict:
    """
    Identify participants with below-chance accuracy.

    Parameters
    ----------
    task_name : str
        Name of task
    trials_df : pd.DataFrame
        Trial-level data
    chance_level : float
        Chance-level accuracy

    Returns
    -------
    dict
        Accuracy-based exclusion results
    """
    if 'correct' not in trials_df.columns:
        return {'task': task_name, 'n_participants': 0}

    # Determine participant ID column (loaders normalize to participant_id)
    pid_col = 'participant_id' if 'participant_id' in trials_df.columns else 'participantId'

    # Calculate accuracy per participant
    participant_acc = trials_df.groupby(pid_col)['correct'].mean()

    below_chance = participant_acc[participant_acc < chance_level]

    return {
        'task': task_name,
        'n_participants': len(participant_acc),
        'accuracy_mean': participant_acc.mean(),
        'accuracy_sd': participant_acc.std(),
        'accuracy_min': participant_acc.min(),
        'accuracy_max': participant_acc.max(),
        'chance_level': chance_level,
        'n_below_chance': len(below_chance),
        'pct_below_chance': len(below_chance) / len(participant_acc) * 100 if len(participant_acc) > 0 else 0,
        'below_chance_participants': below_chance.index.tolist()
    }


def analyze_trial_counts(task_name: str, trials_df: pd.DataFrame, min_trials: int = 10) -> dict:
    """
    Check if participants have sufficient trials.

    Parameters
    ----------
    task_name : str
        Name of task
    trials_df : pd.DataFrame
        Trial-level data
    min_trials : int
        Minimum required trials

    Returns
    -------
    dict
        Trial count results
    """
    # Determine participant ID column (loaders normalize to participant_id)
    pid_col = 'participant_id' if 'participant_id' in trials_df.columns else 'participantId'

    trial_counts = trials_df.groupby(pid_col).size()

    insufficient = trial_counts[trial_counts < min_trials]

    return {
        'task': task_name,
        'n_participants': len(trial_counts),
        'trials_mean': trial_counts.mean(),
        'trials_sd': trial_counts.std(),
        'trials_min': trial_counts.min(),
        'trials_max': trial_counts.max(),
        'min_required': min_trials,
        'n_insufficient': len(insufficient),
        'pct_insufficient': len(insufficient) / len(trial_counts) * 100 if len(trial_counts) > 0 else 0,
        'insufficient_participants': insufficient.index.tolist()
    }


def analyze_cognitive_quality(
    data_dir: Path | None = None,
    tasks: Iterable[str] | None = None,
) -> dict:
    """
    Comprehensive cognitive task quality analysis.

    NOTE: For quality diagnostics, we load RAW CSV data directly (without any filtering)
    to properly detect timeouts, anticipations, and other quality issues.
    Using trial loaders would apply RT filters that drop timeout trials (NaN RT).

    Returns
    -------
    dict
        All cognitive task quality metrics
    """
    results = {}

    data_dir = _resolve_data_dir(data_dir, "overall")
    task_list = _normalize_tasks(tasks)

    if "stroop" in task_list:
        # Stroop - load RAW CSV for quality analysis (bypass RT filter)
        try:
            stroop = pd.read_csv(data_dir / "4c_stroop_trials.csv", encoding="utf-8")
            stroop = ensure_participant_id(stroop)
            # Use rt_ms column if available, otherwise rt
            if "rt_ms" in stroop.columns and "rt" not in stroop.columns:
                stroop["rt"] = stroop["rt_ms"]
            elif "rt_ms" in stroop.columns:
                # Prefer rt_ms (has more data)
                stroop["rt"] = stroop["rt_ms"]
            results['stroop_rt'] = analyze_rt_quality('Stroop', stroop, STROOP_RT_MIN, STROOP_RT_MAX)
            results['stroop_accuracy'] = analyze_participant_accuracy('Stroop', stroop, 0.25)
            results['stroop_trials'] = analyze_trial_counts('Stroop', stroop, 20)
        except Exception as e:
            print(f"Stroop analysis failed: {e}")

    if "wcst" in task_list:
        # WCST - load RAW CSV for quality analysis
        try:
            wcst = pd.read_csv(data_dir / "4b_wcst_trials.csv", encoding="utf-8")
            wcst = ensure_participant_id(wcst)
            results['wcst_rt'] = analyze_rt_quality('WCST', wcst, WCST_RT_MIN, 5000)  # WCST has longer timeout
            results['wcst_accuracy'] = analyze_participant_accuracy('WCST', wcst, 0.25)  # 4 options = 25% chance
            results['wcst_trials'] = analyze_trial_counts('WCST', wcst, 30)
        except Exception as e:
            print(f"WCST analysis failed: {e}")

    if "prp" in task_list:
        # PRP - load RAW CSV for quality analysis (bypass RT filter)
        try:
            prp = pd.read_csv(data_dir / "4a_prp_trials.csv", encoding="utf-8")
            prp = ensure_participant_id(prp)
            # Use t2_rt_ms column if available
            if "t2_rt_ms" in prp.columns:
                prp["t2_rt"] = prp["t2_rt_ms"]
            # PRP uses t2_correct for Task 2 accuracy
            if 't2_correct' in prp.columns:
                prp['correct'] = prp['t2_correct']
            results['prp_rt'] = analyze_rt_quality('PRP', prp, PRP_RT_MIN, PRP_RT_MAX)
            results['prp_accuracy'] = analyze_participant_accuracy('PRP', prp, 0.5)
            results['prp_trials'] = analyze_trial_counts('PRP', prp, 20)
        except Exception as e:
            print(f"PRP analysis failed: {e}")

    return results


# =============================================================================
# SAMPLE ATTRITION
# =============================================================================

def calculate_sample_attrition(data_dir: Path | None = None) -> dict:
    """
    Calculate sample sizes at each stage of data processing.

    Returns
    -------
    dict
        Sample sizes and attrition at each stage
    """
    # Participants info (prefer raw directory for attrition)
    attrition_dir = RAW_DIR if (RAW_DIR / "1_participants_info.csv").exists() else _resolve_data_dir(data_dir, "overall")
    participants_path = attrition_dir / "1_participants_info.csv"
    participants = pd.read_csv(participants_path, encoding='utf-8-sig')
    n_registered = len(participants)

    # Survey completers
    surveys_path = attrition_dir / "2_surveys_results.csv"
    surveys = pd.read_csv(surveys_path, encoding='utf-8-sig')

    ucla_completers = surveys[surveys['surveyName'] == 'ucla']['participantId'].nunique()
    dass_completers = surveys[surveys['surveyName'] == 'dass']['participantId'].nunique()

    # Cognitive task completers
    cognitive_path = attrition_dir / "3_cognitive_tests_summary.csv"
    try:
        cognitive = pd.read_csv(cognitive_path, encoding='utf-8-sig')
        cognitive_completers = cognitive['participantId'].nunique() if 'participantId' in cognitive.columns else 0
    except Exception:
        cognitive_completers = 0

    return {
        'n_registered': n_registered,
        'n_ucla_completed': ucla_completers,
        'n_dass_completed': dass_completers,
        'n_cognitive_completed': cognitive_completers,
        'attrition_ucla': n_registered - ucla_completers,
        'attrition_dass': ucla_completers - dass_completers if ucla_completers > dass_completers else 0,
        'attrition_rate_overall': (1 - cognitive_completers / n_registered) * 100 if n_registered > 0 else 0
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_rt_distributions(
    save_path: Path,
    data_dir: Path | None = None,
    tasks: Iterable[str] | None = None,
):
    """Create RT distribution plots for all tasks."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return

    data_dir = _resolve_data_dir(data_dir, "overall")
    task_list = _normalize_tasks(tasks)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Load raw CSV data and extract RT columns
    tasks_data = []

    # Stroop
    if "stroop" in task_list:
        try:
            stroop = pd.read_csv(data_dir / "4c_stroop_trials.csv", encoding="utf-8")
            rt_col = "rt_ms" if "rt_ms" in stroop.columns else "rt"
            tasks_data.append(('Stroop', stroop, rt_col, STROOP_RT_MIN, STROOP_RT_MAX))
        except Exception:
            tasks_data.append(('Stroop', None, None, STROOP_RT_MIN, STROOP_RT_MAX))

    # WCST
    if "wcst" in task_list:
        try:
            wcst = pd.read_csv(data_dir / "4b_wcst_trials.csv", encoding="utf-8")
            rt_col = "rt_ms" if "rt_ms" in wcst.columns else "rt"
            tasks_data.append(('WCST', wcst, rt_col, WCST_RT_MIN, 5000))
        except Exception:
            tasks_data.append(('WCST', None, None, WCST_RT_MIN, 5000))

    # PRP
    if "prp" in task_list:
        try:
            prp = pd.read_csv(data_dir / "4a_prp_trials.csv", encoding="utf-8")
            rt_col = "t2_rt_ms" if "t2_rt_ms" in prp.columns else "t2_rt"
            tasks_data.append(('PRP', prp, rt_col, PRP_RT_MIN, PRP_RT_MAX))
        except Exception:
            tasks_data.append(('PRP', None, None, PRP_RT_MIN, PRP_RT_MAX))

    for ax, (task_name, trials, rt_col, rt_min, rt_max) in zip(axes, tasks_data):
        try:
            if trials is None or rt_col is None:
                raise ValueError(f"Failed to load {task_name} data")

            valid_rt = trials[(trials[rt_col] >= rt_min) & (trials[rt_col] <= rt_max)][rt_col]

            ax.hist(valid_rt, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=rt_min, color='r', linestyle='--', label=f'Min: {rt_min}ms')
            ax.axvline(x=valid_rt.median(), color='g', linestyle='-', label=f'Median: {valid_rt.median():.0f}ms')
            ax.set_xlabel('RT (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{task_name} RT Distribution')
            ax.legend()
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{task_name} (Error)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def run(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    tasks: Iterable[str] | None = None,
):
    """Run all data quality analyses."""
    data_dir = _resolve_data_dir(data_dir, "overall")
    task_list = _normalize_tasks(tasks)
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATA QUALITY SUITE")
    print("=" * 60)

    # Survey quality
    print("\n[1] Survey Response Quality")
    print("-" * 50)
    survey_quality = analyze_survey_quality(data_dir=data_dir)

    for survey in ['ucla', 'dass']:
        duration_key = f'{survey}_duration'
        straight_key = f'{survey}_straight_line'

        if duration_key in survey_quality:
            d = survey_quality[duration_key]
            print(f"\n  {survey.upper()} Survey:")
            print(f"    Duration: M={d['duration_mean']:.1f}s, Mdn={d['duration_median']:.1f}s")
            print(f"    Sec/item: M={d['sec_per_item_mean']:.2f}s")
            print(f"    Too fast (<2s/item): {d['n_too_fast']} ({d['pct_too_fast']:.1f}%)")

        if straight_key in survey_quality:
            s = survey_quality[straight_key]
            print(f"    Straight-line responding: {s['n_straight_line']} ({s['pct_straight_line']:.1f}%)")

    # Cognitive quality
    print("\n[2] Cognitive Task Quality")
    print("-" * 50)
    cognitive_quality = analyze_cognitive_quality(data_dir=data_dir, tasks=task_list)

    for task in task_list:
        rt_key = f'{task}_rt'
        acc_key = f'{task}_accuracy'

        if rt_key in cognitive_quality:
            r = cognitive_quality[rt_key]
            print(f"\n  {task.upper()}:")
            print(f"    Total trials: {r['n_total_trials']}")
            print(f"    Valid trials: {r['n_valid']} ({r['pct_valid']:.1f}%)")
            print(f"    Timeouts: {r['n_timeout']} ({r['pct_timeout']:.1f}%)")
            print(f"    Anticipations (<{r['rt_threshold_min']}ms): {r['n_anticipation']} ({r['pct_anticipation']:.1f}%)")

        if acc_key in cognitive_quality:
            a = cognitive_quality[acc_key]
            print(f"    Accuracy: M={a['accuracy_mean']:.3f}, SD={a['accuracy_sd']:.3f}")
            print(f"    Below chance: {a['n_below_chance']} ({a['pct_below_chance']:.1f}%)")

    # Sample attrition
    print("\n[3] Sample Attrition")
    print("-" * 50)
    attrition = calculate_sample_attrition(data_dir=data_dir)

    print(f"  Registered: {attrition['n_registered']}")
    print(f"  UCLA completed: {attrition['n_ucla_completed']}")
    print(f"  DASS completed: {attrition['n_dass_completed']}")
    print(f"  Cognitive completed: {attrition['n_cognitive_completed']}")
    print(f"  Overall attrition: {attrition['attrition_rate_overall']:.1f}%")

    # Save results
    all_results = {
        'survey_quality': survey_quality,
        'cognitive_quality': cognitive_quality,
        'sample_attrition': attrition
    }

    # Clean for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj

    all_results_clean = clean_for_json(all_results)

    with open(output_dir / "data_quality_report.json", 'w', encoding='utf-8') as f:
        json.dump(all_results_clean, f, indent=2, ensure_ascii=False)

    # Create summary CSV
    summary_rows = []

    # Survey summaries
    for survey in ['ucla', 'dass']:
        if f'{survey}_duration' in survey_quality:
            d = survey_quality[f'{survey}_duration']
            summary_rows.append({
                'category': 'survey',
                'measure': f'{survey.upper()}_duration',
                'metric': 'mean_seconds',
                'value': d['duration_mean'],
                'n': d['n_valid_duration']
            })
            summary_rows.append({
                'category': 'survey',
                'measure': f'{survey.upper()}_too_fast',
                'metric': 'percent',
                'value': d['pct_too_fast'],
                'n': d['n_too_fast']
            })

    # Cognitive summaries
    for task in task_list:
        if f'{task}_rt' in cognitive_quality:
            r = cognitive_quality[f'{task}_rt']
            summary_rows.append({
                'category': 'cognitive',
                'measure': f'{task.upper()}_valid_trials',
                'metric': 'percent',
                'value': r['pct_valid'],
                'n': r['n_valid']
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "data_quality_summary.csv", index=False, encoding='utf-8-sig')

    # RT distribution plot
    if HAS_MATPLOTLIB:
        plot_rt_distributions(output_dir / "rt_distributions.png", data_dir=data_dir, tasks=task_list)
        print(f"\n  Saved: rt_distributions.png")

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("  - data_quality_report.json")
    print("  - data_quality_summary.csv")
    if HAS_MATPLOTLIB:
        print("  - rt_distributions.png")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run()
