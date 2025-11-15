"""
Publication-Quality Figures
============================

Generates high-quality figures for manuscript submission.

Figures:
1. Main Effect: WCST Gender Moderation
2. Trial-Level Dynamics
3. DASS Stratification (KEY FINDING)
4. Summary Forest Plot

Author: Research Team
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/publication_figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/4] Loading data...")

master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))
participants = pd.read_csv(Path("results/1_participants_info.csv"), encoding='utf-8-sig')

if 'participantId' in participants.columns:
    if 'participant_id' in participants.columns:
        participants.drop(columns=['participantId'], inplace=True)
    else:
        participants.rename(columns={'participantId': 'participant_id'}, inplace=True)

master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

# Handle Korean gender
if 'gender' in master.columns:
    master['gender_male'] = 0
    master.loc[master['gender'] == '남성', 'gender_male'] = 1
    master.loc[master['gender'].str.lower() == 'male', 'gender_male'] = 1
    master['gender_label'] = master['gender_male'].map({0: 'Female', 1: 'Male'})

master = master.dropna(subset=['ucla_total', 'pe_rate', 'gender_male']).copy()

# Load DASS stratification results
dass_strat = pd.read_csv(Path("results/analysis_outputs/mechanism_analysis/dass_stratified_moderation.csv"))

print()

# ============================================================================
# FIGURE 1: MAIN EFFECT (WCST GENDER MODERATION)
# ============================================================================

print("[2/4] Figure 1: Main Effect...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, outcome, ylabel in zip(axes, ['pe_rate', 'wcst_accuracy'],
                                ['Perseverative Error Rate (%)', 'WCST Accuracy (%)']):

    # Scatter plot
    for gender, color, label in [(0, '#E74C3C', 'Female'), (1, '#3498DB', 'Male')]:
        data = master[master['gender_male'] == gender]

        ax.scatter(data['ucla_total'], data[outcome],
                   alpha=0.6, s=80, color=color, label=label,
                   edgecolors='white', linewidth=0.5)

        # Regression line
        if len(data) > 5:
            z = np.polyfit(data['ucla_total'].dropna(), data[outcome].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

    ax.set_xlabel('UCLA Loneliness Score', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.legend(frameon=True, loc='best', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Fig1_Main_Effect.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Fig1_Main_Effect.pdf", bbox_inches='tight')
plt.close()
print("  ✓ Saved: Fig1_Main_Effect (PNG + PDF)")

# ============================================================================
# FIGURE 2: TRIAL-LEVEL DYNAMICS
# ============================================================================

print("[3/4] Figure 2: Trial-Level Dynamics...")

# Load trial-level data
error_chains = pd.read_csv(Path("results/analysis_outputs/wcst_trial_dynamics/error_chains.csv"))
post_shift = pd.read_csv(Path("results/analysis_outputs/wcst_trial_dynamics/post_shift_errors.csv"))

# Merge with master for gender labels
error_chains = error_chains.merge(master[['participant_id', 'gender_label']], on='participant_id', how='left')
post_shift = post_shift.merge(master[['participant_id', 'gender_label']], on='participant_id', how='left')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Error Chain Length
ax = axes[0]
for gender, color in [('Female', '#E74C3C'), ('Male', '#3498DB')]:
    data = error_chains[error_chains['gender_label'] == gender]
    ax.scatter(data['ucla_total'], data['mean_chain_length'],
               alpha=0.6, s=80, color=color, label=gender,
               edgecolors='white', linewidth=0.5)

    if len(data) > 5:
        z = np.polyfit(data['ucla_total'].dropna(), data['mean_chain_length'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

ax.set_xlabel('UCLA Loneliness Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Error Chain Length', fontsize=12, fontweight='bold')
ax.set_title('Error Perseveration', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Post-Shift Errors
ax = axes[1]
for gender, color in [('Female', '#E74C3C'), ('Male', '#3498DB')]:
    data = post_shift[post_shift['gender_label'] == gender]
    ax.scatter(data['ucla_total'], data['mean_post_shift_pe_rate'],
               alpha=0.6, s=80, color=color, label=gender,
               edgecolors='white', linewidth=0.5)

    if len(data) > 5:
        z = np.polyfit(data['ucla_total'].dropna(), data['mean_post_shift_pe_rate'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--', alpha=0.8)

ax.set_xlabel('UCLA Loneliness Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Post-Shift Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Rule-Switch Recovery', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Fig2_Trial_Dynamics.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Fig2_Trial_Dynamics.pdf", bbox_inches='tight')
plt.close()
print("  ✓ Saved: Fig2_Trial_Dynamics (PNG + PDF)")

# ============================================================================
# FIGURE 3: DASS STRATIFICATION (KEY FINDING)
# ============================================================================

print("[4/4] Figure 3: DASS Stratification...")

# Filter significant results
sig_strat = dass_strat[dass_strat['interaction_pval'] < 0.05].copy()

if len(sig_strat) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(sig_strat))

    # Plot effect sizes with error bars (approximated from p-values)
    colors = ['#27AE60' if row['stratum'] == 'Low' else '#E67E22' for _, row in sig_strat.iterrows()]

    bars = ax.barh(y_pos, sig_strat['interaction_beta'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)

    # Add vertical line at 0
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Labels
    labels = [f"{row['dass_measure'].split('_')[1].title()} {row['stratum']}\n({row['outcome']})"
              for _, row in sig_strat.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel('Gender Moderation Effect (β)', fontsize=12, fontweight='bold')
    ax.set_title('Gender × UCLA Interaction by DASS Stratum', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add p-values
    for i, (_, row) in enumerate(sig_strat.iterrows()):
        ax.text(row['interaction_beta'] + 0.2, i, f"p={row['interaction_pval']:.3f}",
                va='center', fontsize=9, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#27AE60', label='Low DASS'),
                       Patch(facecolor='#E67E22', label='High DASS')]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig3_DASS_Stratification.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Fig3_DASS_Stratification.pdf", bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: Fig3_DASS_Stratification (PNG + PDF)")
else:
    print("  ! No significant DASS stratification results to plot")

# ============================================================================
# FIGURE 4: FOREST PLOT (ALL METRICS)
# ============================================================================

print("[Bonus] Figure 4: Forest Plot...")

all_metrics = pd.read_csv(Path("results/analysis_outputs/gender_comprehensive/all_metrics_moderation.csv"))

# Select key metrics
key_metrics = all_metrics[all_metrics['outcome'].isin([
    'pe_rate', 'wcst_accuracy', 'wcst_npe_rate',
    'stroop_interference', 'prp_bottleneck'
])].copy()

key_metrics = key_metrics.sort_values('interaction_pval')

fig, ax = plt.subplots(figsize=(10, 7))

y_pos = np.arange(len(key_metrics))

# Plot with error bars
ax.errorbar(key_metrics['interaction_beta'], y_pos,
            xerr=[key_metrics['interaction_beta'] - key_metrics['boot_ci_lower'],
                  key_metrics['boot_ci_upper'] - key_metrics['interaction_beta']],
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='#2C3E50', ecolor='#95A5A6', linewidth=2)

# Color significant ones
sig_mask = key_metrics['interaction_pval'] < 0.05
ax.scatter(key_metrics.loc[sig_mask, 'interaction_beta'],
           y_pos[sig_mask.values],
           s=150, color='#E74C3C', zorder=5, marker='D',
           label='p < 0.05')

# Vertical line at 0
ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(key_metrics['outcome'].str.replace('_', ' ').str.title(), fontsize=10)
ax.set_xlabel('Gender Moderation Effect (β) with 95% Bootstrap CI', fontsize=12, fontweight='bold')
ax.set_title('Summary: Gender × UCLA Interactions Across EF Metrics', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Fig4_Forest_Plot.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Fig4_Forest_Plot.pdf", bbox_inches='tight')
plt.close()
print("  ✓ Saved: Fig4_Forest_Plot (PNG + PDF)")

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("="*80)
print("PUBLICATION FIGURES COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated figures:")
print("  - Fig1_Main_Effect.png/pdf (Gender moderation scatter plots)")
print("  - Fig2_Trial_Dynamics.png/pdf (Error chains + post-shift errors)")
print("  - Fig3_DASS_Stratification.png/pdf (KEY FINDING)")
print("  - Fig4_Forest_Plot.png/pdf (Summary across all metrics)")
print()
print("All figures saved at 300 DPI in both PNG and PDF formats.")
print()
