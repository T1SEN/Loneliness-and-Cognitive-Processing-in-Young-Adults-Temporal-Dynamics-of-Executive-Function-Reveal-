"""
Network Visualization: Cross-Task EF Correlations

OBJECTIVE:
Create network graphs showing correlations among EF metrics across tasks,
separately for males and females, to visualize gender-specific patterns.

RATIONALE:
- Network structure may differ by gender
- Reveals which EF components cluster together
- Shows task-general vs task-specific impairments
- Visual aid for understanding cross-task integration

METRICS INCLUDED:
- WCST: PE rate, accuracy, switch costs
- Stroop: Interference, facilitation, RT
- PRP: Bottleneck effect, T2 RT, dual-task costs
- UCLA: Loneliness (as central node)
- DASS: Depression, Anxiety, Stress
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/network_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9
np.random.seed(42)

print("="*80)
print("NETWORK VISUALIZATION: CROSS-TASK CORRELATIONS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")

# Load master
master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if not master_path.exists():
    print("ERROR: master_dataset.csv not found")
    sys.exit(1)

master = pd.read_csv(master_path, encoding='utf-8-sig')

# Load participants
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

master = master.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')

# Normalize gender
gender_map = {'남성': 'male', '여성': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Find PE column
for col in ['pe_rate', 'perseverative_error_rate']:
    if col in master.columns:
        master = master.rename(columns={col: 'pe_rate'})
        break

print(f"  Loaded N={len(master)}")
print(f"  Columns: {len(master.columns)}")

# ============================================================================
# 2. SELECT KEY METRICS FOR NETWORK
# ============================================================================
print("\n[2/4] Selecting key metrics...")

# Define metrics of interest
metrics = {
    'UCLA': 'ucla_total',
    'DASS-D': 'dass_depression',
    'DASS-A': 'dass_anxiety',
    'DASS-S': 'dass_stress',
    'WCST PE': 'pe_rate',
    'Stroop Int': 'stroop_interference',
    'PRP Bottleneck': 'prp_bottleneck'
}

# Check which metrics are available
available_metrics = {}
for name, col in metrics.items():
    if col in master.columns:
        available_metrics[name] = col

print(f"  Available metrics: {len(available_metrics)}")
for name in available_metrics.keys():
    print(f"    - {name}")

# ============================================================================
# 3. COMPUTE CORRELATION MATRICES BY GENDER
# ============================================================================
print("\n[3/4] Computing correlation matrices...")

def compute_correlation_network(data, metrics_dict):
    """Compute correlation matrix for selected metrics"""

    # Extract columns
    metric_names = list(metrics_dict.keys())
    metric_cols = [metrics_dict[name] for name in metric_names]

    df_subset = data[metric_cols].copy()
    df_subset.columns = metric_names

    # Compute correlations
    corr_matrix = df_subset.corr(method='pearson')

    # P-values
    n = len(df_subset.dropna())
    pval_matrix = pd.DataFrame(np.ones_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)

    for i, col1 in enumerate(metric_names):
        for j, col2 in enumerate(metric_names):
            if i < j:  # Upper triangle only
                valid_data = df_subset[[col1, col2]].dropna()
                if len(valid_data) >= 10:
                    r, p = stats.pearsonr(valid_data[col1], valid_data[col2])
                    pval_matrix.loc[col1, col2] = p
                    pval_matrix.loc[col2, col1] = p

    return corr_matrix, pval_matrix, n

# Males
males = master[master['gender_male'] == 1]
if len(males) >= 20:
    corr_male, pval_male, n_male = compute_correlation_network(males, available_metrics)
    print(f"\n  Males: N={n_male}")
else:
    corr_male, pval_male, n_male = None, None, 0
    print("\n  Males: Insufficient N")

# Females
females = master[master['gender_male'] == 0]
if len(females) >= 20:
    corr_female, pval_female, n_female = compute_correlation_network(females, available_metrics)
    print(f"  Females: N={n_female}")
else:
    corr_female, pval_female, n_female = None, None, 0
    print("  Females: Insufficient N")

# ============================================================================
# 4. VISUALIZE NETWORKS
# ============================================================================
print("\n[4/4] Creating network visualizations...")

def plot_correlation_network(corr_matrix, pval_matrix, title, filename, threshold=0.3):
    """Plot correlation network as heatmap with significance markers"""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask non-significant correlations
    mask = np.abs(corr_matrix) < threshold

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               vmin=-1, vmax=1, square=True, linewidths=1,
               cbar_kws={'label': 'Correlation (r)'},
               mask=mask, ax=ax)

    # Add significance stars
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if not mask.iloc[i, j]:
                p = pval_matrix.iloc[i, j]
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = ''

                if sig:
                    ax.text(j + 0.7, i + 0.3, sig, color='black', fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot males
if corr_male is not None:
    plot_correlation_network(
        corr_male, pval_male,
        f'Male Cross-Task Network (N={n_male})',
        OUTPUT_DIR / 'network_males.png',
        threshold=0.25
    )
    print("  ✓ Male network saved")

# Plot females
if corr_female is not None:
    plot_correlation_network(
        corr_female, pval_female,
        f'Female Cross-Task Network (N={n_female})',
        OUTPUT_DIR / 'network_females.png',
        threshold=0.25
    )
    print("  ✓ Female network saved")

# Comparison plot (side-by-side)
if corr_male is not None and corr_female is not None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Males
    ax = axes[0]
    mask_m = np.abs(corr_male) < 0.25
    sns.heatmap(corr_male, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               vmin=-1, vmax=1, square=True, linewidths=1,
               cbar_kws={'label': 'r'}, mask=mask_m, ax=ax)
    ax.set_title(f'Males (N={n_male})', fontsize=14, fontweight='bold')

    # Females
    ax = axes[1]
    mask_f = np.abs(corr_female) < 0.25
    sns.heatmap(corr_female, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               vmin=-1, vmax=1, square=True, linewidths=1,
               cbar_kws={'label': 'r'}, mask=mask_f, ax=ax)
    ax.set_title(f'Females (N={n_female})', fontsize=14, fontweight='bold')

    plt.suptitle('Gender Comparison: Cross-Task Correlation Networks', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'network_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Comparison network saved")

# Save correlation matrices
if corr_male is not None:
    corr_male.to_csv(OUTPUT_DIR / 'correlation_matrix_males.csv', encoding='utf-8-sig')
    pval_male.to_csv(OUTPUT_DIR / 'pvalue_matrix_males.csv', encoding='utf-8-sig')

if corr_female is not None:
    corr_female.to_csv(OUTPUT_DIR / 'correlation_matrix_females.csv', encoding='utf-8-sig')
    pval_female.to_csv(OUTPUT_DIR / 'pvalue_matrix_females.csv', encoding='utf-8-sig')

# Report
with open(OUTPUT_DIR / "NETWORK_VISUALIZATION_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CROSS-TASK CORRELATION NETWORKS\n")
    f.write("="*80 + "\n\n")

    f.write("OBJECTIVE\n")
    f.write("-"*80 + "\n")
    f.write("Visualize correlation patterns among EF metrics across tasks,\n")
    f.write("separately for males and females.\n\n")

    f.write("METRICS INCLUDED\n")
    f.write("-"*80 + "\n")
    for name in available_metrics.keys():
        f.write(f"  - {name}\n")
    f.write("\n")

    f.write("SAMPLE SIZES\n")
    f.write("-"*80 + "\n")
    f.write(f"Males: N={n_male}\n")
    f.write(f"Females: N={n_female}\n\n")

    if corr_male is not None and corr_female is not None:
        f.write("KEY DIFFERENCES (Males vs Females)\n")
        f.write("-"*80 + "\n")

        # Find largest differences
        diff_matrix = corr_male - corr_female

        # Get top 5 differences
        diff_flat = []
        for i in range(len(diff_matrix)):
            for j in range(i+1, len(diff_matrix)):
                diff_flat.append({
                    'metric1': diff_matrix.index[i],
                    'metric2': diff_matrix.columns[j],
                    'r_male': corr_male.iloc[i, j],
                    'r_female': corr_female.iloc[i, j],
                    'difference': diff_matrix.iloc[i, j]
                })

        diff_df = pd.DataFrame(diff_flat)
        diff_df = diff_df.sort_values('difference', key=abs, ascending=False)

        f.write("\nTop 5 Gender Differences:\n")
        for _, row in diff_df.head(5).iterrows():
            f.write(f"\n{row['metric1']} - {row['metric2']}:\n")
            f.write(f"  Male r={row['r_male']:.3f}, Female r={row['r_female']:.3f}\n")
            f.write(f"  Difference: {row['difference']:.3f}\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Visualizations saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ Network Visualization Complete!")
print("="*80)
print(f"\nFiles saved:")
print(f"  - network_males.png")
print(f"  - network_females.png")
print(f"  - network_comparison.png")
