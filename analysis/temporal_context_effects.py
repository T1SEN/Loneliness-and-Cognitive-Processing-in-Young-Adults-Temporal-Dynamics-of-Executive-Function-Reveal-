"""
Temporal and Context Effects Analysis
=====================================
Analyzes whether time-of-day, day-of-week, or course context
affects the UCLA-EF relationship.

Hypotheses:
- Late testing times may amplify UCLA effects (fatigue × loneliness)
- Course context may introduce systematic variation
- Circadian effects on cognitive performance
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "temporal_context"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_temporal_features(master_df):
    """Extract temporal features from participant data."""
    participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8')
    participants = ensure_participant_id(participants)

    # Parse createdAt timestamp
    if 'createdAt' in participants.columns:
        def parse_timestamp(ts):
            if pd.isna(ts):
                return None, None, None
            try:
                # Try ISO format
                dt = pd.to_datetime(ts)
                return dt.hour, dt.dayofweek, dt
            except:
                return None, None, None

        parsed = participants['createdAt'].apply(parse_timestamp)
        participants['test_hour'] = [p[0] for p in parsed]
        participants['test_dow'] = [p[1] for p in parsed]  # 0=Monday

        # Time categories
        def categorize_hour(h):
            if pd.isna(h):
                return 'unknown'
            if 6 <= h < 12:
                return 'morning'
            elif 12 <= h < 17:
                return 'afternoon'
            elif 17 <= h < 21:
                return 'evening'
            else:
                return 'night'

        participants['time_category'] = participants['test_hour'].apply(categorize_hour)

    # Merge with master
    merged = master_df.merge(
        participants[['participant_id', 'test_hour', 'test_dow', 'time_category',
                     'courseName', 'professorName']],
        on='participant_id', how='left'
    )

    return merged


def analyze_time_of_day_effects(df, outcome):
    """Analyze time-of-day effects on EF outcomes."""
    # Filter to valid time data
    valid = df.dropna(subset=['test_hour', outcome])

    if len(valid) < 50:
        return None

    # Group statistics by time category
    time_stats = valid.groupby('time_category').agg({
        outcome: ['mean', 'std', 'count'],
        'ucla_score': 'mean'
    }).round(3)
    time_stats.columns = ['_'.join(col).rstrip('_') for col in time_stats.columns]

    # ANOVA across time categories
    groups = [valid[valid['time_category'] == cat][outcome].dropna()
              for cat in ['morning', 'afternoon', 'evening']
              if cat in valid['time_category'].values]

    if len(groups) >= 2 and all(len(g) >= 5 for g in groups):
        f_stat, p_val = stats.f_oneway(*groups)
    else:
        f_stat, p_val = np.nan, np.nan

    return {
        'time_stats': time_stats,
        'anova_f': f_stat,
        'anova_p': p_val
    }


def analyze_ucla_time_interaction(df, outcome):
    """Test UCLA × time-of-day interaction."""
    valid = df.dropna(subset=['test_hour', outcome, 'ucla_score',
                              'dass_depression', 'dass_anxiety', 'dass_stress',
                              'age', 'gender_male'])

    if len(valid) < 50:
        return None

    # Standardize
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'test_hour']:
        valid[f'z_{col}'] = (valid[col] - valid[col].mean()) / valid[col].std()

    # Model with interaction
    formula = f"{outcome} ~ z_ucla_score * z_test_hour + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"

    try:
        model = smf.ols(formula, data=valid).fit()
    except:
        return None

    return {
        'n': len(valid),
        'ucla_coef': model.params.get('z_ucla_score', np.nan),
        'ucla_p': model.pvalues.get('z_ucla_score', np.nan),
        'hour_coef': model.params.get('z_test_hour', np.nan),
        'hour_p': model.pvalues.get('z_test_hour', np.nan),
        'interaction_coef': model.params.get('z_ucla_score:z_test_hour', np.nan),
        'interaction_p': model.pvalues.get('z_ucla_score:z_test_hour', np.nan),
        'model_r2': model.rsquared
    }, model


def analyze_course_effects(df, outcome):
    """Analyze course/professor as random effect context."""
    valid = df.dropna(subset=[outcome, 'ucla_score', 'courseName'])

    if len(valid) < 50:
        return None

    # Course-level statistics
    course_stats = valid.groupby('courseName').agg({
        outcome: ['mean', 'std', 'count'],
        'ucla_score': 'mean'
    }).round(3)
    course_stats.columns = ['_'.join(col).rstrip('_') for col in course_stats.columns]

    # ICC (proportion of variance at course level)
    # Simple approach: ANOVA-based ICC
    course_groups = [valid[valid['courseName'] == c][outcome].dropna()
                    for c in valid['courseName'].unique()
                    if len(valid[valid['courseName'] == c]) >= 5]

    if len(course_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*course_groups)

        # Calculate ICC(1)
        grand_mean = valid[outcome].mean()
        ms_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in course_groups) / (len(course_groups) - 1)
        ms_within = sum(sum((x - g.mean())**2 for x in g) for g in course_groups) / (len(valid) - len(course_groups))
        n_avg = len(valid) / len(course_groups)
        icc = (ms_between - ms_within) / (ms_between + (n_avg - 1) * ms_within) if (ms_between + (n_avg - 1) * ms_within) > 0 else 0
    else:
        f_stat, p_val, icc = np.nan, np.nan, np.nan

    return {
        'course_stats': course_stats,
        'anova_f': f_stat,
        'anova_p': p_val,
        'icc': icc
    }


def main():
    print("=" * 60)
    print("Temporal and Context Effects Analysis")
    print("=" * 60)

    # Load and prepare data
    master = load_master_dataset(use_cache=True)
    df = extract_temporal_features(master)

    print(f"Dataset: N={len(df)}")
    print(f"Valid test hour: {df['test_hour'].notna().sum()}")

    # Time distribution
    print("\n[1] Testing time distribution")
    print("-" * 40)
    if 'time_category' in df.columns:
        print(df['time_category'].value_counts())

    ef_outcomes = {
        'pe_rate': 'WCST PE Rate',
        'stroop_interference': 'Stroop Interference',
        'prp_bottleneck': 'PRP Bottleneck'
    }

    all_results = []

    for outcome, label in ef_outcomes.items():
        if outcome not in df.columns:
            continue

        print(f"\n{'='*60}")
        print(f"Outcome: {label}")
        print("=" * 60)

        # Time of day effects
        print("\n[2] Time-of-day effects")
        print("-" * 40)
        tod_results = analyze_time_of_day_effects(df, outcome)
        if tod_results:
            print("\n  By time category:")
            print(tod_results['time_stats'].to_string())
            print(f"\n  ANOVA: F={tod_results['anova_f']:.2f}, p={tod_results['anova_p']:.4f}")

            tod_results['time_stats'].to_csv(
                OUTPUT_DIR / f"time_of_day_{outcome}.csv", encoding='utf-8-sig'
            )

        # UCLA × Time interaction
        print("\n[3] UCLA × Time interaction")
        print("-" * 40)
        interaction_result = analyze_ucla_time_interaction(df, outcome)
        if interaction_result:
            results, model = interaction_result
            print(f"  N: {results['n']}")
            print(f"  UCLA main effect: b={results['ucla_coef']:.4f}, p={results['ucla_p']:.4f}")
            print(f"  Hour main effect: b={results['hour_coef']:.4f}, p={results['hour_p']:.4f}")
            print(f"  UCLA × Hour interaction: b={results['interaction_coef']:.4f}, p={results['interaction_p']:.4f}")

            all_results.append({
                'outcome': outcome,
                'outcome_label': label,
                **results
            })

        # Course effects
        print("\n[4] Course context effects")
        print("-" * 40)
        course_results = analyze_course_effects(df, outcome)
        if course_results:
            print(f"  ICC (course-level variance): {course_results['icc']:.3f}")
            print(f"  Course ANOVA: F={course_results['anova_f']:.2f}, p={course_results['anova_p']:.4f}")

            course_results['course_stats'].to_csv(
                OUTPUT_DIR / f"course_effects_{outcome}.csv", encoding='utf-8-sig'
            )

    # Save all results
    if all_results:
        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / "temporal_interaction_results.csv", index=False, encoding='utf-8-sig'
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Temporal and Context Effects Analysis:

Key Questions:
1. Does time-of-day affect EF performance?
2. Does UCLA × Time interaction exist? (amplified effects at certain times)
3. Do course contexts introduce systematic variance?

""")
    if all_results:
        print("UCLA × Time Interactions:")
        for res in all_results:
            sig = "SIGNIFICANT" if res.get('interaction_p', 1) < 0.05 else "n.s."
            print(f"  - {res['outcome_label']}: b={res['interaction_coef']:.4f}, p={res['interaction_p']:.4f} ({sig})")

    print(f"""
Interpretation:
- Positive UCLA × Hour interaction: UCLA effects stronger at later times
- High ICC: Course context explains substantial variance (consider as random effect)
- Time effects may reflect fatigue, motivation, or circadian factors
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
