"""
WCST (Wisconsin Card Sorting Test) Switching Analysis
=====================================================
Extract cognitive flexibility parameters from WCST trial data.

Key parameters:
- Perseverative errors: Continuing to use wrong rule
- Non-perseverative errors: Other types of errors
- Rule switching cost: RT increase after rule change
- Total errors and accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_wcst_data():
    """Load WCST trial data."""
    print("\n" + "=" * 70)
    print("WCST SWITCHING PARAMETER EXTRACTION")
    print("=" * 70)

    print("\nLoading WCST trial data...")
    df = pd.read_csv('results/4b_wcst_trials.csv')
    print(f"  Loaded {len(df)} trials from {df['participant_id'].nunique()} participants")

    return df

def extract_wcst_parameters(df):
    """
    Extract participant-level WCST parameters.

    Parameters:
    - perseverative_error_rate: isPE / total_trials
    - nonperseverative_error_rate: isNPE / total_trials
    - total_error_rate: errors / total_trials
    - rule_switch_cost: RT(post-switch) - RT(pre-switch)
    """
    print("\nExtracting WCST parameters...")

    results = []

    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid].copy()

        # Total trials
        n_trials = len(pdata)

        # Skip if no trials
        if n_trials == 0:
            continue

        # Accuracy
        accuracy = pdata['correct'].mean() if 'correct' in pdata.columns else np.nan

        # Error rates
        if 'isPE' in pdata.columns:
            persev_errors = pdata['isPE'].sum()
            persev_rate = persev_errors / n_trials if n_trials > 0 else np.nan
        else:
            persev_errors = np.nan
            persev_rate = np.nan

        if 'isNPE' in pdata.columns:
            nonpersev_errors = pdata['isNPE'].sum()
            nonpersev_rate = nonpersev_errors / n_trials if n_trials > 0 else np.nan
        else:
            nonpersev_errors = np.nan
            nonpersev_rate = np.nan

        total_error_rate = 1 - accuracy if not np.isnan(accuracy) else np.nan

        # Rule switching cost
        # Identify rule switches
        if 'ruleAtThatTime' in pdata.columns:
            pdata['prev_rule'] = pdata['ruleAtThatTime'].shift(1)
            pdata['rule_switch'] = (pdata['ruleAtThatTime'] != pdata['prev_rule']) & pdata['prev_rule'].notna()

            # Get RT for switch vs non-switch trials
            if 'rt_ms' in pdata.columns or 'reactionTimeMs' in pdata.columns:
                rt_col = 'rt_ms' if 'rt_ms' in pdata.columns else 'reactionTimeMs'

                switch_trials = pdata[pdata['rule_switch'] == True]
                nonswitch_trials = pdata[pdata['rule_switch'] == False]

                if len(switch_trials) > 0 and len(nonswitch_trials) > 0:
                    switch_rt = switch_trials[rt_col].mean()
                    nonswitch_rt = nonswitch_trials[rt_col].mean()
                    switch_cost = switch_rt - nonswitch_rt
                else:
                    switch_cost = np.nan
            else:
                switch_cost = np.nan
        else:
            switch_cost = np.nan

        results.append({
            'participant_id': pid,
            'wcst_n_trials': n_trials,
            'wcst_accuracy': accuracy,
            'wcst_persev_rate': persev_rate,
            'wcst_nonpersev_rate': nonpersev_rate,
            'wcst_total_error_rate': total_error_rate,
            'wcst_switch_cost_ms': switch_cost
        })

    results_df = pd.DataFrame(results)

    print(f"  Extracted parameters for {len(results_df)} participants")
    print(f"\nParameter statistics:")
    print(f"  Accuracy:              {results_df['wcst_accuracy'].mean():.1%} (SD: {results_df['wcst_accuracy'].std():.3f})")
    print(f"  Perseverative rate:    {results_df['wcst_persev_rate'].mean():.1%} (SD: {results_df['wcst_persev_rate'].std():.3f})")
    print(f"  Non-persev rate:       {results_df['wcst_nonpersev_rate'].mean():.1%} (SD: {results_df['wcst_nonpersev_rate'].std():.3f})")
    print(f"  Switch cost (ms):      {results_df['wcst_switch_cost_ms'].mean():.1f} (SD: {results_df['wcst_switch_cost_ms'].std():.1f})")

    return results_df

def save_parameters(params_df, output_dir='results/analysis_outputs'):
    """Save WCST parameters."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / 'wcst_switching_parameters.csv'
    params_df.to_csv(output_file, index=False)
    print(f"\n[OK] Saved WCST parameters to {output_file}")

def main():
    """Main pipeline."""

    # Load data
    df = load_wcst_data()

    # Extract parameters
    params_df = extract_wcst_parameters(df)

    # Save
    save_parameters(params_df)

    print("\n" + "=" * 70)
    print("WCST ANALYSIS COMPLETE")
    print("=" * 70)

    return params_df

if __name__ == '__main__':
    params = main()
