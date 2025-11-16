"""
Stroop DDM Data Preprocessing
=============================
Prepares trial-level Stroop data for Drift Diffusion Model (DDM) analysis using PyMC.

Input:  results/4c_stroop_trials.csv
Output: analysis/data_stroop_ddm.csv

Preprocessing steps:
1. Load raw Stroop trial data
2. Remove outlier trials (RT < 200ms or > 3000ms)
3. Exclude timeout trials
4. Code trial types: congruent=0, incongruent=1, neutral=0.5
5. Convert RT to seconds (for DDM fitting)
6. Check minimum trials per participant per condition
7. Export cleaned data for PyMC DDM modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_stroop_data(filepath):
    """Load raw Stroop trial data."""
    print(f"Loading Stroop data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} trials from {df['participant_id'].nunique()} participants")
    return df

def filter_valid_trials(df):
    """
    Remove invalid trials:
    - Missing RT
    - Timeout trials
    - Outlier RTs (< 200ms or > 3000ms)
    """
    initial_count = len(df)

    # Remove missing RT
    df = df[df['rt_ms'].notna()].copy()
    print(f"  After removing missing RT: {len(df)} trials ({initial_count - len(df)} removed)")

    # Remove timeout trials
    if 'timeout' in df.columns:
        df = df[df['timeout'] == False].copy()
        print(f"  After removing timeout: {len(df)} trials")

    # Remove outliers
    df = df[(df['rt_ms'] >= 200) & (df['rt_ms'] <= 3000)].copy()
    print(f"  After RT filtering (200-3000ms): {len(df)} trials")

    return df

def code_trial_conditions(df):
    """
    Code trial types for DDM:
    - congruent: 0 (baseline)
    - incongruent: 1 (interference condition)
    - neutral: 0.5 (intermediate)

    This allows us to model drift rate as:
    drift[trial] = drift_base + drift_interference * condition_code
    """
    condition_map = {
        'congruent': 0.0,
        'incongruent': 1.0,
        'neutral': 0.5
    }

    # Use 'type' or 'cond' column depending on what exists
    if 'type' in df.columns:
        df['condition_code'] = df['type'].map(condition_map)
        df['condition'] = df['type']
    elif 'cond' in df.columns:
        df['condition_code'] = df['cond'].map(condition_map)
        df['condition'] = df['cond']
    else:
        raise ValueError("No 'type' or 'cond' column found in data")

    print(f"\nCondition distribution:")
    print(df['condition'].value_counts().sort_index())

    return df

def convert_rt_to_seconds(df):
    """Convert RT from milliseconds to seconds (standard for DDM)."""
    df['rt_sec'] = df['rt_ms'] / 1000.0
    print(f"\nRT range: {df['rt_sec'].min():.3f}s to {df['rt_sec'].max():.3f}s")
    print(f"RT mean: {df['rt_sec'].mean():.3f}s (SD: {df['rt_sec'].std():.3f}s)")
    return df

def check_participant_trials(df, min_trials_per_condition=10):
    """
    Check that each participant has sufficient trials per condition.
    Exclude participants with too few trials.
    """
    print(f"\nChecking participant trial counts (min {min_trials_per_condition} per condition)...")

    # Count trials per participant per condition
    trial_counts = df.groupby(['participant_id', 'condition']).size().reset_index(name='n_trials')

    # Find participants with insufficient trials
    insufficient = trial_counts[trial_counts['n_trials'] < min_trials_per_condition]

    if len(insufficient) > 0:
        print(f"  Warning: {len(insufficient)} participant-condition pairs have < {min_trials_per_condition} trials")
        print(insufficient)

        # Get participants with ALL conditions having sufficient trials
        valid_participants = (
            trial_counts
            .groupby('participant_id')['n_trials']
            .min()
            .reset_index()
        )
        valid_participants = valid_participants[
            valid_participants['n_trials'] >= min_trials_per_condition
        ]['participant_id'].tolist()

        print(f"  Keeping {len(valid_participants)} participants with sufficient trials in all conditions")
        df = df[df['participant_id'].isin(valid_participants)].copy()
    else:
        print(f"  All participants have sufficient trials")

    # Print summary
    trials_per_participant = df.groupby('participant_id').size()
    print(f"\nFinal participant summary:")
    print(f"  N participants: {len(trials_per_participant)}")
    print(f"  Trials per participant: {trials_per_participant.mean():.1f} ± {trials_per_participant.std():.1f}")
    print(f"  Range: {trials_per_participant.min()} to {trials_per_participant.max()}")

    return df

def add_accuracy_indicator(df):
    """Add binary accuracy indicator (1=correct, 0=incorrect) for DDM."""
    if 'correct' not in df.columns:
        raise ValueError("No 'correct' column found in data")

    df['response'] = df['correct'].astype(int)
    accuracy = df['correct'].mean()
    print(f"\nOverall accuracy: {accuracy:.1%}")

    # Accuracy by condition
    acc_by_cond = df.groupby('condition')['correct'].mean()
    print(f"Accuracy by condition:")
    for cond, acc in acc_by_cond.items():
        print(f"  {cond}: {acc:.1%}")

    return df

def create_participant_indices(df):
    """Create integer indices for participants (required by PyMC)."""
    unique_ids = sorted(df['participant_id'].unique())
    id_to_idx = {pid: idx for idx, pid in enumerate(unique_ids)}
    df['participant_idx'] = df['participant_id'].map(id_to_idx)

    print(f"\nCreated indices for {len(unique_ids)} participants (0 to {len(unique_ids)-1})")

    return df, id_to_idx

def save_preprocessed_data(df, id_mapping, output_dir='analysis'):
    """Save preprocessed data for DDM fitting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save main data file
    data_file = output_dir / 'data_stroop_ddm.csv'
    columns_to_save = [
        'participant_id', 'participant_idx', 'trial',
        'condition', 'condition_code', 'correct', 'response',
        'rt_ms', 'rt_sec', 'text', 'letterColor'
    ]

    # Only save columns that exist
    columns_to_save = [col for col in columns_to_save if col in df.columns]
    df[columns_to_save].to_csv(data_file, index=False)
    print(f"\nSaved preprocessed data to {data_file}")
    print(f"  Shape: {df.shape}")

    # Save participant ID mapping
    mapping_file = output_dir / 'stroop_participant_mapping.csv'
    pd.DataFrame([
        {'participant_id': pid, 'participant_idx': idx}
        for pid, idx in id_mapping.items()
    ]).to_csv(mapping_file, index=False)
    print(f"Saved participant mapping to {mapping_file}")

    # Save summary statistics
    summary_file = output_dir / 'stroop_ddm_prep_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Stroop DDM Data Preprocessing Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Final dataset:\n")
        f.write(f"  Participants: {df['participant_id'].nunique()}\n")
        f.write(f"  Trials: {len(df)}\n")
        f.write(f"  Trials per participant: {len(df) / df['participant_id'].nunique():.1f}\n\n")

        f.write(f"RT statistics (seconds):\n")
        f.write(f"  Mean: {df['rt_sec'].mean():.3f}\n")
        f.write(f"  SD: {df['rt_sec'].std():.3f}\n")
        f.write(f"  Median: {df['rt_sec'].median():.3f}\n")
        f.write(f"  Min: {df['rt_sec'].min():.3f}\n")
        f.write(f"  Max: {df['rt_sec'].max():.3f}\n\n")

        f.write(f"Accuracy:\n")
        f.write(f"  Overall: {df['correct'].mean():.1%}\n")
        for cond in sorted(df['condition'].unique()):
            acc = df[df['condition'] == cond]['correct'].mean()
            f.write(f"  {cond}: {acc:.1%}\n")

        f.write(f"\nCondition distribution:\n")
        for cond, count in df['condition'].value_counts().sort_index().items():
            f.write(f"  {cond}: {count} ({count/len(df):.1%})\n")

    print(f"Saved summary to {summary_file}")

def main():
    """Main preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Stroop DDM Data Preprocessing")
    print("=" * 60 + "\n")

    # Load data
    df = load_stroop_data('results/4c_stroop_trials.csv')

    # Filter valid trials
    df = filter_valid_trials(df)

    # Code conditions
    df = code_trial_conditions(df)

    # Convert RT
    df = convert_rt_to_seconds(df)

    # Check participant trials
    df = check_participant_trials(df, min_trials_per_condition=10)

    # Add accuracy
    df = add_accuracy_indicator(df)

    # Create indices
    df, id_mapping = create_participant_indices(df)

    # Save
    save_preprocessed_data(df, id_mapping)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60 + "\n")

    return df

if __name__ == '__main__':
    df = main()
