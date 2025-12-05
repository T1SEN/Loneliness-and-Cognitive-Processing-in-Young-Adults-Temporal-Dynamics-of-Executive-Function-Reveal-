"""
PRP Response Order Analysis
===========================
Analyzes response order patterns in the PRP dual-task paradigm.

Normal: T1 → T2 order (process T1 first, then T2)
Abnormal: T2 → T1 order (premature T2 response)

Response order errors may indicate:
- Task prioritization failures
- Impulsivity
- Attentional resource allocation problems

Tests whether UCLA predicts response order errors.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, PRP_RT_MAX
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "prp_response_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prp_response_order():
    """Load PRP trials with response order information."""
    df = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    # Check for response_order column
    if 'response_order' not in df.columns:
        print("Warning: response_order column not found. Creating from RT data...")
        # Infer from RT columns
        t1_rt = 't1_rt_ms' if 't1_rt_ms' in df.columns else 't1_rt'
        t2_rt = 't2_rt_ms' if 't2_rt_ms' in df.columns else 't2_rt'

        if t1_rt in df.columns and t2_rt in df.columns:
            conditions = [
                (df[t1_rt] < df[t2_rt]),
                (df[t2_rt] < df[t1_rt]),
                (df[t1_rt].isna() & df[t2_rt].notna()),
                (df[t2_rt].isna() & df[t1_rt].notna())
            ]
            choices = ['T1T2', 'T2T1', 'T2_only', 'T1_only']
            df['response_order'] = np.select(conditions, choices, default='none')

    # Get SOA column
    soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in df.columns else 'soa'

    print(f"PRP trials loaded: N={len(df)}")
    print(f"Response order distribution:")
    print(df['response_order'].value_counts())

    return df, soa_col


def compute_response_order_metrics(df, soa_col):
    """Compute response order metrics per participant."""
    results = []

    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid]
        n_trials = len(pdata)

        if n_trials < 20:
            continue

        # Count response orders
        order_counts = pdata['response_order'].value_counts()

        n_t1t2 = order_counts.get('T1T2', 0)
        n_t2t1 = order_counts.get('T2T1', 0)
        n_t2_only = order_counts.get('T2_only', 0)
        n_t1_only = order_counts.get('T1_only', 0)
        n_none = order_counts.get('none', 0)

        # Reversal rate (T2 before T1)
        total_dual = n_t1t2 + n_t2t1
        reversal_rate = n_t2t1 / total_dual * 100 if total_dual > 0 else 0

        # By SOA
        soa_reversals = {}
        for soa in [50, 150, 300, 600, 1200]:
            soa_data = pdata[pdata[soa_col] == soa]
            if len(soa_data) > 0:
                n_rev = (soa_data['response_order'] == 'T2T1').sum()
                soa_reversals[f'reversal_soa_{soa}'] = n_rev / len(soa_data) * 100

        results.append({
            'participant_id': pid,
            'n_trials': n_trials,
            'n_t1t2': n_t1t2,
            'n_t2t1': n_t2t1,
            'n_t2_only': n_t2_only,
            'n_t1_only': n_t1_only,
            'reversal_rate': reversal_rate,
            **soa_reversals
        })

    return pd.DataFrame(results)


def analyze_reversal_by_soa(df, soa_col):
    """Analyze if reversals are more common at certain SOAs."""
    soa_stats = []

    for soa in [50, 150, 300, 600, 1200]:
        soa_data = df[df[soa_col] == soa]
        if len(soa_data) < 100:
            continue

        n_t2t1 = (soa_data['response_order'] == 'T2T1').sum()
        n_total = len(soa_data)
        rate = n_t2t1 / n_total * 100

        soa_stats.append({
            'soa': soa,
            'n_trials': n_total,
            'n_reversals': n_t2t1,
            'reversal_rate': rate
        })

    return pd.DataFrame(soa_stats)


def analyze_ucla_reversal(master_df, order_df):
    """Test UCLA → response order reversal (DASS-controlled)."""
    merged = master_df.merge(order_df, on='participant_id', how='inner')

    required = ['ucla_score', 'reversal_rate', 'dass_depression', 'dass_anxiety',
                'dass_stress', 'age', 'gender_male']
    merged = merged.dropna(subset=required)

    if len(merged) < 30:
        return None

    # Standardize
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

    # Model 1: UCLA only
    formula1 = "reversal_rate ~ z_ucla_score + z_age + C(gender_male)"
    model1 = smf.ols(formula1, data=merged).fit()

    # Model 2: UCLA + DASS
    formula2 = "reversal_rate ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    model2 = smf.ols(formula2, data=merged).fit()

    # Model 3: UCLA × Gender
    formula3 = "reversal_rate ~ z_ucla_score * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
    model3 = smf.ols(formula3, data=merged).fit()

    results = {
        'n': len(merged),
        'mean_reversal': merged['reversal_rate'].mean(),
        'std_reversal': merged['reversal_rate'].std(),
        # Model 1
        'ucla_only_coef': model1.params.get('z_ucla_score', np.nan),
        'ucla_only_p': model1.pvalues.get('z_ucla_score', np.nan),
        # Model 2
        'ucla_dass_coef': model2.params.get('z_ucla_score', np.nan),
        'ucla_dass_p': model2.pvalues.get('z_ucla_score', np.nan),
        'model2_r2': model2.rsquared,
        # Model 3
        'interaction_coef': model3.params.get('z_ucla_score:C(gender_male)[T.1]', np.nan),
        'interaction_p': model3.pvalues.get('z_ucla_score:C(gender_male)[T.1]', np.nan)
    }

    return results, {'model1': model1, 'model2': model2, 'model3': model3}


def analyze_reversal_consequences(df, soa_col):
    """Analyze if reversals affect T2 accuracy."""
    # Compare T2 accuracy for T1T2 vs T2T1 trials
    t1t2_trials = df[df['response_order'] == 'T1T2']
    t2t1_trials = df[df['response_order'] == 'T2T1']

    if len(t1t2_trials) < 100 or len(t2t1_trials) < 10:
        return None

    t1t2_acc = t1t2_trials['t2_correct'].mean() if 't2_correct' in df.columns else np.nan
    t2t1_acc = t2t1_trials['t2_correct'].mean() if 't2_correct' in df.columns else np.nan

    # T-test
    if 't2_correct' in df.columns:
        t1t2_correct = t1t2_trials['t2_correct'].astype(float)
        t2t1_correct = t2t1_trials['t2_correct'].astype(float)
        t_stat, p_val = stats.ttest_ind(t1t2_correct, t2t1_correct)
    else:
        t_stat, p_val = np.nan, np.nan

    return {
        'n_t1t2': len(t1t2_trials),
        'n_t2t1': len(t2t1_trials),
        't1t2_t2_accuracy': t1t2_acc,
        't2t1_t2_accuracy': t2t1_acc,
        'accuracy_difference': t1t2_acc - t2t1_acc,
        't_statistic': t_stat,
        'p_value': p_val
    }


def main():
    print("=" * 60)
    print("PRP Response Order Analysis")
    print("=" * 60)

    # Load data
    prp_df, soa_col = load_prp_response_order()

    # Compute metrics
    print("\n[1] Computing response order metrics per participant")
    print("-" * 40)
    order_df = compute_response_order_metrics(prp_df, soa_col)
    print(f"  Participants: N={len(order_df)}")
    print(f"  Mean reversal rate: {order_df['reversal_rate'].mean():.2f}%")
    print(f"  Reversal rate range: {order_df['reversal_rate'].min():.1f}% - {order_df['reversal_rate'].max():.1f}%")

    order_df.to_csv(OUTPUT_DIR / "response_order_metrics.csv", index=False, encoding='utf-8-sig')

    # Reversal by SOA
    print("\n[2] Response reversals by SOA")
    print("-" * 40)
    soa_stats = analyze_reversal_by_soa(prp_df, soa_col)
    if len(soa_stats) > 0:
        for _, row in soa_stats.iterrows():
            print(f"  SOA {int(row['soa'])}ms: {row['reversal_rate']:.2f}% reversals")

        soa_stats.to_csv(OUTPUT_DIR / "reversal_by_soa.csv", index=False, encoding='utf-8-sig')

    # UCLA analysis
    print("\n[3] UCLA → Response reversals (DASS-controlled)")
    print("-" * 40)

    master = load_master_dataset(use_cache=True)
    ucla_results = analyze_ucla_reversal(master, order_df)

    if ucla_results:
        results, models = ucla_results
        print(f"  N: {results['n']}")
        print(f"  Mean reversal rate: {results['mean_reversal']:.2f}%")
        print(f"\n  UCLA only: b={results['ucla_only_coef']:.4f}, p={results['ucla_only_p']:.4f}")
        print(f"  UCLA + DASS: b={results['ucla_dass_coef']:.4f}, p={results['ucla_dass_p']:.4f}")
        print(f"  UCLA × Gender: b={results['interaction_coef']:.4f}, p={results['interaction_p']:.4f}")

        pd.DataFrame([results]).to_csv(OUTPUT_DIR / "ucla_reversal_regression.csv", index=False, encoding='utf-8-sig')

        # Save model summaries
        with open(OUTPUT_DIR / "model_summaries.txt", 'w', encoding='utf-8') as f:
            for name, model in models.items():
                f.write(f"\n{name.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(model.summary().as_text())
                f.write("\n\n")

    # Reversal consequences
    print("\n[4] Consequences of response order reversals")
    print("-" * 40)
    consequences = analyze_reversal_consequences(prp_df, soa_col)
    if consequences:
        print(f"  T1T2 (normal) T2 accuracy: {consequences['t1t2_t2_accuracy']:.1%}")
        print(f"  T2T1 (reversal) T2 accuracy: {consequences['t2t1_t2_accuracy']:.1%}")
        print(f"  Difference: {consequences['accuracy_difference']:.1%}")
        print(f"  t = {consequences['t_statistic']:.2f}, p = {consequences['p_value']:.4f}")

        pd.DataFrame([consequences]).to_csv(OUTPUT_DIR / "reversal_consequences.csv", index=False, encoding='utf-8-sig')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ucla_coef_str = f"{ucla_results[0]['ucla_dass_coef']:.4f}" if ucla_results else 'N/A'
    ucla_p_str = f"{ucla_results[0]['ucla_dass_p']:.4f}" if ucla_results else 'N/A'

    print(f"""
Response Order Analysis Results:

Basic Statistics:
- Mean reversal rate (T2 before T1): {order_df['reversal_rate'].mean():.2f}%
- Reversals indicate premature T2 processing

SOA Pattern:
- Reversals typically more common at short SOAs (T1-T2 overlap)

UCLA Effects:
- UCLA -> Reversal rate (DASS-controlled): b={ucla_coef_str}, p={ucla_p_str}

Interpretation:
- Positive UCLA effect = loneliness associated with more reversals
- This could indicate impulsivity or task prioritization problems
- May reflect reduced top-down control of task sets
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
