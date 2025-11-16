"""
PRP (Psychological Refractory Period) Data Preprocessing
========================================================
Prepares trial-level PRP data for hierarchical bottleneck model analysis.

Input:  results/4a_prp_trials.csv
Output: analysis/data_prp_bottleneck.csv

The PRP paradigm measures dual-task interference:
- Two tasks (T1 and T2) presented with varying SOA (Stimulus Onset Asynchrony)
- At short SOAs, T2 RT increases due to central bottleneck
- Bottleneck reflects capacity limits in response selection

Preprocessing steps:
1. Load raw PRP trial data
2. Remove outlier trials (T2 RT < 150ms or > 3000ms)
3. Exclude trials with T1 or T2 errors (for clean bottleneck measurement)
4. Exclude timeout trials
5. Group SOA into bins: 50ms, 150ms, 600ms, 1200ms
6. Calculate PRP effect (short SOA - long SOA)
7. Export cleaned data for hierarchical modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def load_prp_data(filepath):
    """Load raw PRP trial data."""
    print(f"Loading PRP data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} trials from {df['participant_id'].nunique()} participants")
    return df

def filter_valid_trials(df):
    """
    Remove invalid trials:
    - Missing T2 RT
    - T2 timeout
    - T2 RT outliers (< 150ms or > 3000ms)
    - T1 or T2 errors (for clean bottleneck measurement)
    """
    initial_count = len(df)

    # Remove missing T2 RT
    df = df[df['t2_rt_ms'].notna()].copy()
    print(f"  After removing missing T2 RT: {len(df)} trials ({initial_count - len(df)} removed)")

    # Remove T2 timeout
    if 't2_timeout' in df.columns:
        df = df[df['t2_timeout'] == False].copy()
        print(f"  After removing T2 timeout: {len(df)} trials")

    # Remove T2 RT outliers
    df = df[(df['t2_rt_ms'] >= 150) & (df['t2_rt_ms'] <= 3000)].copy()
    print(f"  After T2 RT filtering (150-3000ms): {len(df)} trials")

    # Keep only trials where BOTH T1 and T2 were correct
    # This gives us a clean measure of processing time without error confounds
    if 't1_correct' in df.columns and 't2_correct' in df.columns:
        both_correct = (df['t1_correct'] == True) & (df['t2_correct'] == True)
        df = df[both_correct].copy()
        print(f"  After keeping only dual-correct trials: {len(df)} trials")

    return df

def categorize_soa(df):
    """
    Categorize SOA into bins.

    Standard PRP design uses 4 SOA levels:
    - Short: 50-100ms (maximum interference)
    - Medium-short: 150-200ms
    - Medium-long: 300-600ms
    - Long: 1000-1200ms (minimal interference)
    """
    print("\nCategorizing SOA levels...")

    # Use soa_nominal_ms if available, otherwise soa
    if 'soa_nominal_ms' in df.columns:
        soa_col = 'soa_nominal_ms'
    elif 'soa' in df.columns:
        soa_col = 'soa'
    else:
        raise ValueError("No SOA column found in data")

    df['soa_ms'] = df[soa_col]

    # Create categorical SOA bins
    df['soa_category'] = pd.cut(
        df['soa_ms'],
        bins=[0, 100, 200, 700, 1500],
        labels=['short', 'medium_short', 'medium_long', 'long']
    )

    # Also keep numeric SOA for modeling
    df['soa_sec'] = df['soa_ms'] / 1000.0

    print(f"\nSOA distribution:")
    print(df['soa_category'].value_counts().sort_index())

    return df

def convert_rt_to_seconds(df):
    """Convert T2 RT from milliseconds to seconds."""
    df['t2_rt_sec'] = df['t2_rt_ms'] / 1000.0

    if 't1_rt_ms' in df.columns:
        df['t1_rt_sec'] = df['t1_rt_ms'] / 1000.0
        print(f"\nT1 RT range: {df['t1_rt_sec'].min():.3f}s to {df['t1_rt_sec'].max():.3f}s")
        print(f"T1 RT mean: {df['t1_rt_sec'].mean():.3f}s (SD: {df['t1_rt_sec'].std():.3f}s)")

    print(f"\nT2 RT range: {df['t2_rt_sec'].min():.3f}s to {df['t2_rt_sec'].max():.3f}s")
    print(f"T2 RT mean: {df['t2_rt_sec'].mean():.3f}s (SD: {df['t2_rt_sec'].std():.3f}s)")

    return df

def check_participant_trials(df, min_trials_per_soa=5):
    """
    Check that each participant has sufficient trials per SOA condition.
    """
    print(f"\nChecking participant trial counts (min {min_trials_per_soa} per SOA)...")

    # Count trials per participant per SOA
    trial_counts = df.groupby(['participant_id', 'soa_category']).size().reset_index(name='n_trials')

    # Find participants with insufficient trials
    insufficient = trial_counts[trial_counts['n_trials'] < min_trials_per_soa]

    if len(insufficient) > 0:
        print(f"  Warning: {len(insufficient)} participant-SOA pairs have < {min_trials_per_soa} trials")

        # Get participants with ALL SOA conditions having sufficient trials
        valid_participants = (
            trial_counts
            .groupby('participant_id')['n_trials']
            .min()
            .reset_index()
        )
        valid_participants = valid_participants[
            valid_participants['n_trials'] >= min_trials_per_soa
        ]['participant_id'].tolist()

        print(f"  Keeping {len(valid_participants)} participants with sufficient trials in all SOA conditions")
        df = df[df['participant_id'].isin(valid_participants)].copy()
    else:
        print(f"  All participants have sufficient trials")

    # Print summary
    trials_per_participant = df.groupby('participant_id').size()
    print(f"\nFinal participant summary:")
    print(f"  N participants: {len(trials_per_participant)}")
    print(f"  Trials per participant: {trials_per_participant.mean():.1f} +/- {trials_per_participant.std():.1f}")
    print(f"  Range: {trials_per_participant.min()} to {trials_per_participant.max()}")

    return df

def calculate_prp_effect(df):
    """
    Calculate PRP effect for each participant.

    PRP effect = T2_RT(short SOA) - T2_RT(long SOA)

    Positive values indicate bottleneck interference.
    """
    print("\nCalculating PRP effects...")

    prp_effects = []

    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid]

        # Get mean T2 RT for each SOA category
        soa_means = pdata.groupby('soa_category')['t2_rt_sec'].mean()

        if 'short' in soa_means.index and 'long' in soa_means.index:
            prp_effect = soa_means['short'] - soa_means['long']

            prp_effects.append({
                'participant_id': pid,
                't2_rt_short': soa_means['short'],
                't2_rt_long': soa_means['long'],
                'prp_effect': prp_effect
            })

    prp_df = pd.DataFrame(prp_effects)

    print(f"  PRP effect calculated for {len(prp_df)} participants")
    print(f"  Mean PRP effect: {prp_df['prp_effect'].mean():.3f}s ({prp_df['prp_effect'].mean()*1000:.1f}ms)")
    print(f"  SD: {prp_df['prp_effect'].std():.3f}s ({prp_df['prp_effect'].std()*1000:.1f}ms)")

    # Check if PRP effect is positive (expected)
    positive_prp = (prp_df['prp_effect'] > 0).sum()
    print(f"  Participants with positive PRP effect: {positive_prp}/{len(prp_df)} ({positive_prp/len(prp_df):.1%})")

    return prp_df

def create_participant_indices(df):
    """Create integer indices for participants."""
    unique_ids = sorted(df['participant_id'].unique())
    id_to_idx = {pid: idx for idx, pid in enumerate(unique_ids)}
    df['participant_idx'] = df['participant_id'].map(id_to_idx)

    print(f"\nCreated indices for {len(unique_ids)} participants (0 to {len(unique_ids)-1})")

    return df, id_to_idx

def plot_prp_effect(df, output_dir='results/analysis_outputs'):
    """
    Plot T2 RT as a function of SOA to visualize PRP effect.
    """
    print("\nGenerating PRP effect plot...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Calculate mean T2 RT for each SOA
    soa_means = df.groupby('soa_category')['t2_rt_ms'].agg(['mean', 'sem']).reset_index()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = {'short': 0, 'medium_short': 1, 'medium_long': 2, 'long': 3}
    soa_means['x_pos'] = soa_means['soa_category'].map(x_pos)
    soa_means = soa_means.sort_values('x_pos')

    ax.errorbar(
        soa_means['x_pos'],
        soa_means['mean'],
        yerr=soa_means['sem'],
        marker='o',
        markersize=10,
        linewidth=2,
        capsize=5,
        color='darkblue'
    )

    ax.set_xlabel('SOA Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('T2 Reaction Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('PRP Effect: T2 RT as Function of SOA', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Short\n(50-100ms)', 'Med-Short\n(150-200ms)',
                        'Med-Long\n(300-600ms)', 'Long\n(1000-1200ms)'])
    ax.grid(True, alpha=0.3)

    # Add annotation
    if len(soa_means) >= 2:
        prp_effect = soa_means.iloc[0]['mean'] - soa_means.iloc[-1]['mean']
        ax.text(0.05, 0.95, f'PRP Effect: {prp_effect:.1f} ms',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_file = output_dir / 'prp_effect_by_soa.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {plot_file}")
    plt.close()

def save_preprocessed_data(df, prp_effects, id_mapping, output_dir='analysis'):
    """Save preprocessed data for bottleneck modeling."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save trial-level data
    data_file = output_dir / 'data_prp_bottleneck.csv'
    columns_to_save = [
        'participant_id', 'participant_idx', 'trial_index',
        'soa_ms', 'soa_sec', 'soa_category',
        't1_rt_ms', 't1_rt_sec', 't2_rt_ms', 't2_rt_sec',
        't1_correct', 't2_correct'
    ]

    # Only save columns that exist
    columns_to_save = [col for col in columns_to_save if col in df.columns]
    df[columns_to_save].to_csv(data_file, index=False)
    print(f"\nSaved trial-level data to {data_file}")
    print(f"  Shape: {df.shape}")

    # Save PRP effects
    prp_file = output_dir / 'prp_effects_summary.csv'
    prp_effects.to_csv(prp_file, index=False)
    print(f"Saved PRP effects to {prp_file}")

    # Save participant ID mapping
    mapping_file = output_dir / 'prp_participant_mapping.csv'
    pd.DataFrame([
        {'participant_id': pid, 'participant_idx': idx}
        for pid, idx in id_mapping.items()
    ]).to_csv(mapping_file, index=False)
    print(f"Saved participant mapping to {mapping_file}")

    # Save summary statistics
    summary_file = output_dir / 'prp_bottleneck_prep_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("PRP Bottleneck Data Preprocessing Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Final dataset:\n")
        f.write(f"  Participants: {df['participant_id'].nunique()}\n")
        f.write(f"  Trials: {len(df)}\n")
        f.write(f"  Trials per participant: {len(df) / df['participant_id'].nunique():.1f}\n\n")

        f.write(f"T2 RT statistics (seconds):\n")
        f.write(f"  Mean: {df['t2_rt_sec'].mean():.3f}\n")
        f.write(f"  SD: {df['t2_rt_sec'].std():.3f}\n")
        f.write(f"  Median: {df['t2_rt_sec'].median():.3f}\n")
        f.write(f"  Min: {df['t2_rt_sec'].min():.3f}\n")
        f.write(f"  Max: {df['t2_rt_sec'].max():.3f}\n\n")

        f.write(f"SOA distribution:\n")
        for soa, count in df['soa_category'].value_counts().sort_index().items():
            f.write(f"  {soa}: {count} ({count/len(df):.1%})\n")

        f.write(f"\nPRP Effect:\n")
        f.write(f"  Mean: {prp_effects['prp_effect'].mean():.3f}s ({prp_effects['prp_effect'].mean()*1000:.1f}ms)\n")
        f.write(f"  SD: {prp_effects['prp_effect'].std():.3f}s ({prp_effects['prp_effect'].std()*1000:.1f}ms)\n")
        f.write(f"  Positive effects: {(prp_effects['prp_effect'] > 0).sum()}/{len(prp_effects)}\n")

    print(f"Saved summary to {summary_file}")

def main():
    """Main preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("PRP Bottleneck Data Preprocessing")
    print("=" * 60 + "\n")

    # Load data
    df = load_prp_data('results/4a_prp_trials.csv')

    # Filter valid trials
    df = filter_valid_trials(df)

    # Categorize SOA
    df = categorize_soa(df)

    # Convert RT
    df = convert_rt_to_seconds(df)

    # Check participant trials
    df = check_participant_trials(df, min_trials_per_soa=5)

    # Calculate PRP effects
    prp_effects = calculate_prp_effect(df)

    # Create indices
    df, id_mapping = create_participant_indices(df)

    # Plot
    plot_prp_effect(df)

    # Save
    save_preprocessed_data(df, prp_effects, id_mapping)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60 + "\n")

    return df, prp_effects

if __name__ == '__main__':
    df, prp_effects = main()
