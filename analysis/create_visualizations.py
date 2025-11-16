"""
Visualization Generation for Advanced Insights
==============================================
Creates publication-quality plots for key findings:
1. Vulnerability profile comparisons
2. DASS stratification effects
3. ROC curve for UCLA cutoff
4. Cross-task correlations
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

from data_loader_utils import load_master_dataset

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("results/analysis_outputs/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Load data
print("Loading data...")
master = load_master_dataset()
print(f"  Loaded {len(master)} participants\n")

# ============================================================================
# 1. Vulnerability Profile Comparison
# ============================================================================

print("1. Creating Vulnerability Profile Comparison Plot...")

males = master[master['gender_male'] == 1].copy()
males = males.dropna(subset=['ucla_total', 'pe_rate', 'dass_stress', 'dass_anxiety', 'dass_depression'])

if len(males) >= 20:
    # Define vulnerability profiles
    ucla_median = males['ucla_total'].median()
    pe_75 = males['pe_rate'].quantile(0.75)
    pe_25 = males['pe_rate'].quantile(0.25)

    def assign_profile(row):
        if row['ucla_total'] > ucla_median:
            if row['pe_rate'] > pe_75:
                return 'Vulnerable'
            elif row['pe_rate'] < pe_25:
                return 'Resilient'
            else:
                return 'Moderate'
        else:
            return 'Control'

    males['profile'] = males.apply(assign_profile, axis=1)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, (component, title) in enumerate([
        ('dass_stress', 'DASS Stress'),
        ('dass_anxiety', 'DASS Anxiety'),
        ('dass_depression', 'DASS Depression')
    ]):
        ax = axes[idx]

        # Filter to key groups
        plot_data = males[males['profile'].isin(['Vulnerable', 'Resilient', 'Control'])].copy()

        # Boxplot
        sns.boxplot(data=plot_data, x='profile', y=component, ax=ax,
                    order=['Vulnerable', 'Resilient', 'Control'],
                    palette={'Vulnerable': '#d62728', 'Resilient': '#2ca02c', 'Control': '#7f7f7f'})

        ax.set_xlabel('Profile', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{title} Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add sample sizes
        for i, profile in enumerate(['Vulnerable', 'Resilient', 'Control']):
            n = len(plot_data[plot_data['profile'] == profile])
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n}',
                   ha='center', va='top', fontsize=10)

    plt.suptitle('Vulnerability Profile Comparisons (Males Only)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "vulnerability_profiles_comparison.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# 2. DASS Stratification Effects
# ============================================================================

print("\n2. Creating DASS Stratification Effects Plot...")

males_clean = master[(master['gender_male'] == 1)].dropna(subset=['ucla_total', 'pe_rate', 'dass_stress'])

if len(males_clean) >= 20:
    # Stratify by DASS stress
    stress_median = males_clean['dass_stress'].median()
    males_clean['stress_group'] = males_clean['dass_stress'].apply(
        lambda x: 'High DASS Stress' if x > stress_median else 'Low DASS Stress'
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'Low DASS Stress': '#e74c3c', 'High DASS Stress': '#3498db'}

    for group_name in ['Low DASS Stress', 'High DASS Stress']:
        group_data = males_clean[males_clean['stress_group'] == group_name]

        if len(group_data) >= 10:
            # Scatter
            ax.scatter(group_data['ucla_total'], group_data['pe_rate'],
                      alpha=0.5, s=80, color=colors[group_name], label=group_name)

            # Regression line
            x = group_data['ucla_total'].values
            y = group_data['pe_rate'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color=colors[group_name], linewidth=2.5, linestyle='--')

            # Calculate correlation
            r, p_val = stats.pearsonr(x, y)

            # Add correlation text
            y_pos = ax.get_ylim()[1] * (0.95 if group_name == 'Low DASS Stress' else 0.88)
            ax.text(0.05, y_pos, f'{group_name}: r = {r:.3f}, p = {p_val:.4f}',
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor=colors[group_name], alpha=0.2))

    ax.set_xlabel('UCLA Loneliness Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('WCST Perseverative Error Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('DASS Stress Buffers Loneliness → PE Effect (Males Only)',
                fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "dass_stratification_effect.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# 3. ROC Curve for UCLA Cutoff
# ============================================================================

print("\n3. Creating ROC Curve Plot...")

if len(males_clean) >= 20:
    pe_75 = males_clean['pe_rate'].quantile(0.75)
    males_clean['high_pe'] = (males_clean['pe_rate'] > pe_75).astype(int)

    if males_clean['high_pe'].sum() >= 5:
        fpr, tpr, thresholds = roc_curve(males_clean['high_pe'], males_clean['ucla_total'])
        roc_auc_val = auc(fpr, tpr)

        # Find optimal cutoff
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_cutoff = thresholds[optimal_idx]

        fig, ax = plt.subplots(figsize=(8, 8))

        # ROC curve
        ax.plot(fpr, tpr, color='#e74c3c', linewidth=3,
               label=f'ROC Curve (AUC = {roc_auc_val:.3f})')

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], color='gray', linewidth=2, linestyle='--',
               label='Random Classifier')

        # Optimal point
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', markersize=12,
               color='#2ecc71', markeredgecolor='black', markeredgewidth=2,
               label=f'Optimal Cutoff = {optimal_cutoff:.1f}')

        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title(f'ROC Curve: UCLA Predicts High PE Risk (Males)\nSensitivity = {tpr[optimal_idx]:.2%}, Specificity = {(1-fpr[optimal_idx]):.2%}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect('equal')

        plt.tight_layout()
        output_file = OUTPUT_DIR / "roc_curve_ucla_cutoff.png"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

# ============================================================================
# 4. Cross-Task Correlation: WCST PE × Stroop σ
# ============================================================================

print("\n4. Creating Cross-Task Correlation Plot...")

# Load Stroop Ex-Gaussian parameters
try:
    from data_loader_utils import load_exgaussian_params
    stroop_exg = load_exgaussian_params('stroop')
    master_plot = master.merge(stroop_exg[['participant_id', 'sigma']],
                               on='participant_id', how='left')
except:
    print("  Warning: Could not load Stroop Ex-Gaussian parameters")
    master_plot = master.copy()

males_corr = master_plot[(master_plot['gender_male'] == 1)].dropna(subset=['pe_rate', 'sigma'])

if len(males_corr) >= 10:
    fig, ax = plt.subplots(figsize=(10, 8))

    x = males_corr['sigma'].values
    y = males_corr['pe_rate'].values

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=100, color='#9b59b6', edgecolors='black', linewidth=1)

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color='#e74c3c', linewidth=3, linestyle='--',
           label='Linear Fit')

    # Calculate correlation
    r, p_val = stats.pearsonr(x, y)

    # Add correlation text
    ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p_val:.4f}\nN = {len(males_corr)}',
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    ax.set_xlabel('Stroop RT Variability (σ, ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('WCST Perseverative Error Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Cross-Task Correlation: RT Variability Predicts Set-Shifting\n(Males Only)',
                fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "cross_task_correlation_pe_sigma.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 80)
print(f"\nAll plots saved to: {OUTPUT_DIR}")
print("\nGenerated visualizations:")
print("  1. Vulnerability profile comparisons (DASS scores)")
print("  2. DASS stratification effects (interaction plot)")
print("  3. ROC curve for UCLA cutoff determination")
print("  4. Cross-task correlation (WCST PE × Stroop σ)")
print("\n" + "=" * 80)
