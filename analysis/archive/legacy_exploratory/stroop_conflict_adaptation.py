"""
Stroop Conflict Adaptation (CSE) Analysis
==========================================
Analyzes Congruency Sequence Effects (CSE) in the Stroop task:
- CSE = Gratton effect = reduced interference after incongruent trials
- Tests whether UCLA loneliness affects conflict adaptation ability

Conflict Adaptation Index:
  CSE = (cI - cC) - (iI - iC)
  where:
    cI = RT after congruent, current incongruent
    cC = RT after congruent, current congruent
    iI = RT after incongruent, current incongruent
    iC = RT after incongruent, current congruent

Hypotheses:
- CSE > 0 indicates adaptive conflict monitoring
- High UCLA may show reduced CSE (impaired cognitive flexibility)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "stroop_conflict_adaptation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_stroop_trials():
    """Load and preprocess Stroop trial data."""
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    # Determine RT column
    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    if rt_col not in df.columns:
        raise KeyError("No RT column found in Stroop trials")

    # Determine condition column
    cond_col = 'type' if 'type' in df.columns else 'cond'
    if cond_col not in df.columns:
        raise KeyError("No condition column found")

    # Create new columns instead of renaming to avoid duplicates
    df['rt'] = df[rt_col]
    df['condition'] = df[cond_col]

    # Reset index to avoid reindex issues
    df = df.reset_index(drop=True)

    # Filter valid trials
    df = df[df['timeout'] == False].reset_index(drop=True)
    df = df[df['correct'] == True].reset_index(drop=True)
    mask = (df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < STROOP_RT_MAX)
    df = df[mask].reset_index(drop=True)

    # Standardize condition names
    df['condition'] = df['condition'].str.lower()
    df = df[df['condition'].isin(['congruent', 'incongruent'])].copy()

    # Sort by participant and trial order
    df = df.sort_values(['participant_id', 'trial']).reset_index(drop=True)

    print(f"Stroop trials loaded: N={len(df)}, participants={df['participant_id'].nunique()}")
    return df


def compute_cse_metrics(df):
    """
    Compute CSE (Congruency Sequence Effects) for each participant.

    Returns DataFrame with participant-level CSE metrics.
    """
    results = []

    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid].copy()

        if len(pdata) < 10:
            continue

        # Create previous trial condition
        pdata['prev_condition'] = pdata['condition'].shift(1)
        pdata = pdata.dropna(subset=['prev_condition'])

        # Create sequence type
        pdata['sequence'] = pdata['prev_condition'] + '_' + pdata['condition']

        # Calculate mean RT for each sequence type
        seq_means = pdata.groupby('sequence')['rt'].mean()

        # Required sequences: cC, cI, iC, iI
        required = ['congruent_congruent', 'congruent_incongruent',
                   'incongruent_congruent', 'incongruent_incongruent']

        if not all(seq in seq_means.index for seq in required):
            continue

        cC = seq_means['congruent_congruent']
        cI = seq_means['congruent_incongruent']
        iC = seq_means['incongruent_congruent']
        iI = seq_means['incongruent_incongruent']

        # CSE = (cI - cC) - (iI - iC)
        # Positive CSE = reduced interference after incongruent trials
        interference_after_congruent = cI - cC
        interference_after_incongruent = iI - iC
        cse = interference_after_congruent - interference_after_incongruent

        # Count trials per sequence
        seq_counts = pdata.groupby('sequence')['rt'].count()

        results.append({
            'participant_id': pid,
            'cC_rt': cC,
            'cI_rt': cI,
            'iC_rt': iC,
            'iI_rt': iI,
            'interference_after_cong': interference_after_congruent,
            'interference_after_incong': interference_after_incongruent,
            'cse': cse,
            'n_cC': seq_counts.get('congruent_congruent', 0),
            'n_cI': seq_counts.get('congruent_incongruent', 0),
            'n_iC': seq_counts.get('incongruent_congruent', 0),
            'n_iI': seq_counts.get('incongruent_incongruent', 0),
            'n_total': len(pdata)
        })

    return pd.DataFrame(results)


def analyze_group_cse(cse_df):
    """Test if CSE is significantly different from zero at group level."""
    cse_values = cse_df['cse'].dropna()

    mean_cse = cse_values.mean()
    std_cse = cse_values.std()
    n = len(cse_values)

    # One-sample t-test against 0
    t_stat, p_value = stats.ttest_1samp(cse_values, 0)

    # Effect size (Cohen's d)
    cohens_d = mean_cse / std_cse if std_cse > 0 else 0

    return {
        'n': n,
        'mean_cse': mean_cse,
        'std_cse': std_cse,
        'sem_cse': std_cse / np.sqrt(n),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        '95%_ci_lower': mean_cse - 1.96 * std_cse / np.sqrt(n),
        '95%_ci_upper': mean_cse + 1.96 * std_cse / np.sqrt(n)
    }


def analyze_ucla_cse_relationship(master_df, cse_df):
    """
    Test whether UCLA loneliness predicts CSE (with DASS control).
    """
    merged = master_df.merge(cse_df, on='participant_id', how='inner')

    # Required columns
    required = ['ucla_score', 'cse', 'dass_depression', 'dass_anxiety',
                'dass_stress', 'age', 'gender_male']
    merged = merged.dropna(subset=required)

    if len(merged) < 30:
        return None

    # Standardize predictors
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

    # Model 1: UCLA only
    formula1 = "cse ~ z_ucla_score + z_age + C(gender_male)"
    model1 = smf.ols(formula1, data=merged).fit()

    # Model 2: UCLA + DASS controls
    formula2 = "cse ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model2 = smf.ols(formula2, data=merged).fit()

    # Model 3: UCLA × Gender interaction
    formula3 = "cse ~ z_ucla_score * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
    model3 = smf.ols(formula3, data=merged).fit()

    results = {
        'n': len(merged),
        # Model 1
        'ucla_only_coef': model1.params.get('z_ucla_score', np.nan),
        'ucla_only_se': model1.bse.get('z_ucla_score', np.nan),
        'ucla_only_p': model1.pvalues.get('z_ucla_score', np.nan),
        'ucla_only_r2': model1.rsquared,
        # Model 2 (DASS-controlled)
        'ucla_dass_coef': model2.params.get('z_ucla_score', np.nan),
        'ucla_dass_se': model2.bse.get('z_ucla_score', np.nan),
        'ucla_dass_p': model2.pvalues.get('z_ucla_score', np.nan),
        'ucla_dass_r2': model2.rsquared,
        # Model 3 (Interaction)
        'interaction_coef': model3.params.get('z_ucla_score:C(gender_male)[T.1]', np.nan),
        'interaction_p': model3.pvalues.get('z_ucla_score:C(gender_male)[T.1]', np.nan),
        'interaction_r2': model3.rsquared
    }

    # Store full model summaries
    models = {
        'model1_ucla_only': model1,
        'model2_ucla_dass': model2,
        'model3_interaction': model3
    }

    return results, models, merged


def analyze_by_ucla_group(merged_df):
    """Compare CSE between high and low UCLA groups."""
    median_ucla = merged_df['ucla_score'].median()
    merged_df['ucla_group'] = np.where(merged_df['ucla_score'] > median_ucla, 'high', 'low')

    low_cse = merged_df[merged_df['ucla_group'] == 'low']['cse']
    high_cse = merged_df[merged_df['ucla_group'] == 'high']['cse']

    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(low_cse, high_cse)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(low_cse)-1)*low_cse.std()**2 +
                          (len(high_cse)-1)*high_cse.std()**2) /
                         (len(low_cse) + len(high_cse) - 2))
    cohens_d = (low_cse.mean() - high_cse.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        'low_n': len(low_cse),
        'low_mean': low_cse.mean(),
        'low_std': low_cse.std(),
        'high_n': len(high_cse),
        'high_mean': high_cse.mean(),
        'high_std': high_cse.std(),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }


def main():
    print("=" * 60)
    print("Stroop Conflict Adaptation (CSE) Analysis")
    print("=" * 60)

    # Load trial data
    stroop_trials = load_stroop_trials()

    # Compute CSE for each participant
    print("\n[1] Computing CSE metrics per participant")
    print("-" * 40)
    cse_df = compute_cse_metrics(stroop_trials)
    print(f"  Participants with valid CSE: N={len(cse_df)}")
    cse_df.to_csv(OUTPUT_DIR / "participant_cse_metrics.csv", index=False, encoding='utf-8-sig')

    # Group-level CSE analysis
    print("\n[2] Group-level CSE (Is CSE > 0?)")
    print("-" * 40)
    group_results = analyze_group_cse(cse_df)
    print(f"  Mean CSE: {group_results['mean_cse']:.2f} ms (SE={group_results['sem_cse']:.2f})")
    print(f"  95% CI: [{group_results['95%_ci_lower']:.2f}, {group_results['95%_ci_upper']:.2f}]")
    print(f"  t({group_results['n']-1}) = {group_results['t_statistic']:.3f}, p = {group_results['p_value']:.4f}")
    print(f"  Cohen's d = {group_results['cohens_d']:.3f}")

    if group_results['p_value'] < 0.05:
        print("  -> CSE is significantly different from zero (conflict adaptation present)")
    else:
        print("  -> CSE not significantly different from zero")

    # Save group results
    pd.DataFrame([group_results]).to_csv(
        OUTPUT_DIR / "group_cse_results.csv", index=False, encoding='utf-8-sig'
    )

    # UCLA → CSE relationship
    print("\n[3] UCLA → CSE Relationship (DASS-controlled)")
    print("-" * 40)

    master = load_master_dataset(use_cache=True)
    ucla_results = analyze_ucla_cse_relationship(master, cse_df)

    if ucla_results:
        results, models, merged = ucla_results

        print(f"\n  Model 1 (UCLA only):")
        print(f"    UCLA coef: b={results['ucla_only_coef']:.3f}, p={results['ucla_only_p']:.4f}")
        print(f"    R2 = {results['ucla_only_r2']:.4f}")

        print(f"\n  Model 2 (UCLA + DASS controls):")
        print(f"    UCLA coef: b={results['ucla_dass_coef']:.3f}, p={results['ucla_dass_p']:.4f}")
        print(f"    R2 = {results['ucla_dass_r2']:.4f}")

        print(f"\n  Model 3 (UCLA × Gender interaction):")
        print(f"    Interaction: b={results['interaction_coef']:.3f}, p={results['interaction_p']:.4f}")

        # Save results
        pd.DataFrame([results]).to_csv(
            OUTPUT_DIR / "ucla_cse_regression.csv", index=False, encoding='utf-8-sig'
        )

        # Save full model summaries
        with open(OUTPUT_DIR / "regression_summaries.txt", 'w', encoding='utf-8') as f:
            f.write("UCLA → CSE Regression Models\n")
            f.write("=" * 60 + "\n\n")
            for name, model in models.items():
                f.write(f"\n{name.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(model.summary().as_text())
                f.write("\n\n")

        # Group comparison
        print("\n[4] High vs Low UCLA Group Comparison")
        print("-" * 40)
        group_comp = analyze_by_ucla_group(merged)
        print(f"  Low UCLA (N={group_comp['low_n']}): CSE = {group_comp['low_mean']:.2f} (SD={group_comp['low_std']:.2f})")
        print(f"  High UCLA (N={group_comp['high_n']}): CSE = {group_comp['high_mean']:.2f} (SD={group_comp['high_std']:.2f})")
        print(f"  t = {group_comp['t_statistic']:.3f}, p = {group_comp['p_value']:.4f}")
        print(f"  Cohen's d = {group_comp['cohens_d']:.3f}")

        pd.DataFrame([group_comp]).to_csv(
            OUTPUT_DIR / "ucla_group_comparison.csv", index=False, encoding='utf-8-sig'
        )
    else:
        print("  Insufficient data for UCLA analysis")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ucla_coef_str = f"{results['ucla_dass_coef']:.3f}" if ucla_results else 'N/A'
    ucla_p_str = f"{results['ucla_dass_p']:.4f}" if ucla_results else 'N/A'

    print(f"""
Conflict Adaptation Analysis Results:
- Valid participants: N={len(cse_df)}
- Mean CSE: {group_results['mean_cse']:.1f} ms
- CSE significantly > 0: {'Yes' if group_results['p_value'] < 0.05 else 'No'} (p={group_results['p_value']:.4f})

UCLA Effect on CSE (DASS-controlled):
- UCLA coefficient: b={ucla_coef_str}
- p-value: {ucla_p_str}

Interpretation:
- Positive CSE indicates intact conflict monitoring
- Negative UCLA effect would suggest impaired adaptation in lonely individuals
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
