"""
WCST Learning Trajectory Analysis
=================================
Analyzes trial-by-trial perseverative error (PE) rates across the task
to examine learning dynamics and whether UCLA loneliness affects
the trajectory of set-shifting improvement.

Analyses:
1. Block-wise PE rates (e.g., 20-trial blocks)
2. Growth curve modeling (linear/quadratic trends)
3. UCLA × Block interaction (DASS-controlled)
4. Individual difference patterns in learning

Hypotheses:
- PE rate should decrease across blocks (learning)
- UCLA may predict slower learning (shallower slope)
- UCLA may not affect initial performance but affect later blocks
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
import ast

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "wcst_learning_trajectory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 20  # Trials per block


def load_wcst_trials():
    """Load WCST trial data with PE flags."""
    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    # Parse isPE from extra column or use direct column
    if 'isPE' in df.columns:
        df['is_pe'] = df['isPE'].map({'True': True, 'False': False, True: True, False: False})
    elif 'extra' in df.columns:
        def parse_extra(val):
            if pd.isna(val) or not isinstance(val, str):
                return False
            try:
                d = ast.literal_eval(val)
                return d.get('isPE', False)
            except:
                return False
        df['is_pe'] = df['extra'].apply(parse_extra)
    else:
        raise KeyError("No PE information found in WCST trials")

    # Clean is_pe column
    df['is_pe'] = df['is_pe'].fillna(False).astype(bool)

    # Sort by participant and trial
    df = df.sort_values(['participant_id', 'trialIndex']).reset_index(drop=True)

    print(f"WCST trials loaded: N={len(df)}, participants={df['participant_id'].nunique()}")
    print(f"  Total PE trials: {df['is_pe'].sum()} ({df['is_pe'].mean()*100:.1f}%)")

    return df


def compute_block_pe_rates(df, block_size=BLOCK_SIZE):
    """Compute PE rate for each block within each participant."""
    results = []

    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid].copy()
        pdata = pdata.sort_values('trialIndex').reset_index(drop=True)

        n_trials = len(pdata)
        n_blocks = n_trials // block_size

        for b in range(n_blocks):
            start_idx = b * block_size
            end_idx = start_idx + block_size
            block_data = pdata.iloc[start_idx:end_idx]

            pe_rate = block_data['is_pe'].mean() * 100
            accuracy = block_data['correct'].mean() * 100

            results.append({
                'participant_id': pid,
                'block': b + 1,
                'block_start': start_idx + 1,
                'block_end': end_idx,
                'pe_rate': pe_rate,
                'accuracy': accuracy,
                'n_trials': len(block_data),
                'n_pe': block_data['is_pe'].sum()
            })

    return pd.DataFrame(results)


def fit_individual_learning_curves(block_df):
    """Fit linear learning curve for each participant."""
    results = []

    for pid in block_df['participant_id'].unique():
        pdata = block_df[block_df['participant_id'] == pid].copy()

        if len(pdata) < 3:
            continue

        # Linear regression: PE rate ~ Block
        X = sm.add_constant(pdata['block'])
        y = pdata['pe_rate']

        try:
            model = sm.OLS(y, X).fit()
            intercept = model.params['const']
            slope = model.params['block']
            r_squared = model.rsquared
        except:
            continue

        # Initial and final PE rates
        initial_pe = pdata[pdata['block'] == 1]['pe_rate'].values[0] if 1 in pdata['block'].values else np.nan
        max_block = pdata['block'].max()
        final_pe = pdata[pdata['block'] == max_block]['pe_rate'].values[0]

        results.append({
            'participant_id': pid,
            'intercept': intercept,
            'slope': slope,
            'r_squared': r_squared,
            'initial_pe': initial_pe,
            'final_pe': final_pe,
            'pe_change': final_pe - initial_pe if not np.isnan(initial_pe) else np.nan,
            'n_blocks': len(pdata)
        })

    return pd.DataFrame(results)


def analyze_group_learning(block_df):
    """Analyze group-level learning trajectory."""
    # Aggregate by block
    group_means = block_df.groupby('block').agg({
        'pe_rate': ['mean', 'std', 'count'],
        'accuracy': ['mean', 'std']
    }).reset_index()
    group_means.columns = ['block', 'pe_mean', 'pe_std', 'n',
                           'acc_mean', 'acc_std']
    group_means['pe_sem'] = group_means['pe_std'] / np.sqrt(group_means['n'])

    # Test for linear trend
    X = sm.add_constant(group_means['block'])
    model = sm.OLS(group_means['pe_mean'], X).fit()

    linear_results = {
        'intercept': model.params['const'],
        'slope': model.params['block'],
        'slope_p': model.pvalues['block'],
        'r_squared': model.rsquared
    }

    return group_means, linear_results


def analyze_ucla_learning_interaction(master_df, block_df, learning_df):
    """
    Test UCLA × Block interaction on PE rate.
    Mixed effects model: PE ~ UCLA × Block + DASS + (1|participant)
    """
    # Merge with master dataset
    merged_block = block_df.merge(
        master_df[['participant_id', 'ucla_score', 'dass_depression',
                   'dass_anxiety', 'dass_stress', 'age', 'gender_male']],
        on='participant_id', how='inner'
    )

    merged_block = merged_block.dropna()

    if len(merged_block) < 100:
        return None, None

    # Standardize
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        merged_block[f'z_{col}'] = (merged_block[col] - merged_block[col].mean()) / merged_block[col].std()

    # Center block
    merged_block['block_c'] = merged_block['block'] - merged_block['block'].mean()

    # Model 1: Main effects
    formula1 = "pe_rate ~ z_ucla_score + block_c + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model1 = smf.ols(formula1, data=merged_block).fit()

    # Model 2: UCLA × Block interaction
    formula2 = "pe_rate ~ z_ucla_score * block_c + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model2 = smf.ols(formula2, data=merged_block).fit()

    # Try mixed effects if available
    try:
        import statsmodels.regression.mixed_linear_model as mlm
        formula_mixed = "pe_rate ~ z_ucla_score * block_c + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        model_mixed = mlm.MixedLM.from_formula(
            formula_mixed, groups='participant_id', data=merged_block
        ).fit()
        mixed_available = True
    except:
        model_mixed = None
        mixed_available = False

    results = {
        'n_observations': len(merged_block),
        'n_participants': merged_block['participant_id'].nunique(),
        # Model 1
        'main_ucla_coef': model1.params.get('z_ucla_score', np.nan),
        'main_ucla_p': model1.pvalues.get('z_ucla_score', np.nan),
        'main_block_coef': model1.params.get('block_c', np.nan),
        'main_block_p': model1.pvalues.get('block_c', np.nan),
        # Model 2
        'interaction_coef': model2.params.get('z_ucla_score:block_c', np.nan),
        'interaction_p': model2.pvalues.get('z_ucla_score:block_c', np.nan),
        'interaction_r2': model2.rsquared
    }

    # Add mixed effects results if available
    if mixed_available and model_mixed:
        results['mixed_ucla_coef'] = model_mixed.fe_params.get('z_ucla_score', np.nan)
        results['mixed_ucla_p'] = model_mixed.pvalues.get('z_ucla_score', np.nan)
        results['mixed_interaction_coef'] = model_mixed.fe_params.get('z_ucla_score:block_c', np.nan)
        results['mixed_interaction_p'] = model_mixed.pvalues.get('z_ucla_score:block_c', np.nan)

    models = {'ols_main': model1, 'ols_interaction': model2}
    if mixed_available:
        models['mixed'] = model_mixed

    return results, models


def analyze_ucla_individual_slopes(master_df, learning_df):
    """Test whether UCLA predicts individual learning slopes."""
    merged = master_df.merge(learning_df, on='participant_id', how='inner')
    merged = merged.dropna(subset=['slope', 'ucla_score', 'dass_depression',
                                    'dass_anxiety', 'dass_stress', 'age', 'gender_male'])

    if len(merged) < 30:
        return None

    # Standardize
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

    # Model: Learning slope ~ UCLA + DASS + demographics
    formula = "slope ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model = smf.ols(formula, data=merged).fit()

    results = {
        'n': len(merged),
        'mean_slope': merged['slope'].mean(),
        'std_slope': merged['slope'].std(),
        'ucla_coef': model.params.get('z_ucla_score', np.nan),
        'ucla_se': model.bse.get('z_ucla_score', np.nan),
        'ucla_p': model.pvalues.get('z_ucla_score', np.nan),
        'model_r2': model.rsquared,
        'model_f_p': model.f_pvalue
    }

    return results, model


def main():
    print("=" * 60)
    print("WCST Learning Trajectory Analysis")
    print("=" * 60)

    # Load data
    wcst_trials = load_wcst_trials()

    # Compute block-wise PE rates
    print(f"\n[1] Computing block-wise PE rates (block size={BLOCK_SIZE})")
    print("-" * 40)
    block_df = compute_block_pe_rates(wcst_trials)
    print(f"  Blocks computed: {len(block_df)} ({block_df['participant_id'].nunique()} participants)")
    block_df.to_csv(OUTPUT_DIR / "block_pe_rates.csv", index=False, encoding='utf-8-sig')

    # Group-level learning
    print("\n[2] Group-level learning trajectory")
    print("-" * 40)
    group_means, linear_results = analyze_group_learning(block_df)
    print(f"  Linear trend: slope = {linear_results['slope']:.3f} (p = {linear_results['slope_p']:.4f})")
    print(f"  R2 = {linear_results['r_squared']:.4f}")
    print("\n  Block-by-block PE rates:")
    for _, row in group_means.iterrows():
        print(f"    Block {int(row['block'])}: PE = {row['pe_mean']:.1f}% (SEM = {row['pe_sem']:.1f})")

    group_means.to_csv(OUTPUT_DIR / "group_learning_trajectory.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame([linear_results]).to_csv(OUTPUT_DIR / "group_linear_trend.csv", index=False, encoding='utf-8-sig')

    # Individual learning curves
    print("\n[3] Individual learning curve parameters")
    print("-" * 40)
    learning_df = fit_individual_learning_curves(block_df)
    print(f"  Participants with valid curves: N={len(learning_df)}")
    print(f"  Mean slope: {learning_df['slope'].mean():.3f} (SD={learning_df['slope'].std():.3f})")
    print(f"  Negative slope (learning): {(learning_df['slope'] < 0).sum()} ({(learning_df['slope'] < 0).mean()*100:.1f}%)")
    learning_df.to_csv(OUTPUT_DIR / "individual_learning_curves.csv", index=False, encoding='utf-8-sig')

    # UCLA × Block interaction
    print("\n[4] UCLA × Block interaction (DASS-controlled)")
    print("-" * 40)

    master = load_master_dataset(use_cache=True)
    interaction_results, interaction_models = analyze_ucla_learning_interaction(
        master, block_df, learning_df
    )

    if interaction_results:
        print(f"  N observations: {interaction_results['n_observations']}")
        print(f"  N participants: {interaction_results['n_participants']}")
        print(f"\n  Main effect of UCLA: b={interaction_results['main_ucla_coef']:.3f}, p={interaction_results['main_ucla_p']:.4f}")
        print(f"  Main effect of Block: b={interaction_results['main_block_coef']:.3f}, p={interaction_results['main_block_p']:.4f}")
        print(f"  UCLA × Block interaction: b={interaction_results['interaction_coef']:.3f}, p={interaction_results['interaction_p']:.4f}")

        if 'mixed_interaction_p' in interaction_results:
            print(f"\n  Mixed effects model:")
            print(f"    UCLA × Block: b={interaction_results['mixed_interaction_coef']:.3f}, p={interaction_results['mixed_interaction_p']:.4f}")

        pd.DataFrame([interaction_results]).to_csv(
            OUTPUT_DIR / "ucla_block_interaction.csv", index=False, encoding='utf-8-sig'
        )

        # Save model summaries
        with open(OUTPUT_DIR / "interaction_model_summaries.txt", 'w', encoding='utf-8') as f:
            for name, model in interaction_models.items():
                f.write(f"\n{name.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(model.summary().as_text())
                f.write("\n\n")
    else:
        print("  Insufficient data for interaction analysis")

    # UCLA predicting individual slopes
    print("\n[5] UCLA predicting individual learning slopes")
    print("-" * 40)

    slope_results = analyze_ucla_individual_slopes(master, learning_df)
    if slope_results:
        results, model = slope_results
        print(f"  N participants: {results['n']}")
        print(f"  Mean learning slope: {results['mean_slope']:.3f} (SD={results['std_slope']:.3f})")
        print(f"\n  UCLA → Learning slope:")
        print(f"    b = {results['ucla_coef']:.4f} (SE = {results['ucla_se']:.4f})")
        print(f"    p = {results['ucla_p']:.4f}")
        print(f"    Model R2 = {results['model_r2']:.4f}")

        pd.DataFrame([results]).to_csv(
            OUTPUT_DIR / "ucla_slope_prediction.csv", index=False, encoding='utf-8-sig'
        )
    else:
        print("  Insufficient data")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Prepare summary strings to avoid f-string issues
    learning_trend = 'Significant learning (PE decreases over time)' if linear_results['slope_p'] < 0.05 and linear_results['slope'] < 0 else 'No significant learning trend'
    main_ucla_coef_str = f"{interaction_results['main_ucla_coef']:.3f}" if interaction_results else 'N/A'
    main_ucla_p_str = f"{interaction_results['main_ucla_p']:.4f}" if interaction_results else 'N/A'
    inter_coef_str = f"{interaction_results['interaction_coef']:.3f}" if interaction_results else 'N/A'
    inter_p_str = f"{interaction_results['interaction_p']:.4f}" if interaction_results else 'N/A'
    slope_coef_str = f"{slope_results[0]['ucla_coef']:.4f}" if slope_results else 'N/A'
    slope_p_str = f"{slope_results[0]['ucla_p']:.4f}" if slope_results else 'N/A'

    print(f"""
Learning Trajectory Results:
- Group learning trend: slope = {linear_results['slope']:.3f}, p = {linear_results['slope_p']:.4f}
- {learning_trend}

UCLA Effects (DASS-controlled):
- Main effect on PE rate: b = {main_ucla_coef_str}, p = {main_ucla_p_str}
- UCLA × Block interaction: b = {inter_coef_str}, p = {inter_p_str}
- UCLA → Individual slope: b = {slope_coef_str}, p = {slope_p_str}

Interpretation:
- Negative main block effect = learning (PE reduces over time)
- Positive UCLA × Block interaction = loneliness associated with slower learning
- Negative UCLA → slope = loneliness associated with steeper (better) learning
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
