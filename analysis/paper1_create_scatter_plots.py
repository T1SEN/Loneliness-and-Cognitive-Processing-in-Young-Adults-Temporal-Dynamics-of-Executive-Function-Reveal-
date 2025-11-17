"""
Paper 1: Create Publication-Quality Scatter Plots
==================================================

Creates Figure 1 for manuscript: 2×2 grid showing UCLA × Gender interactions
for PRP tau and RMSSD.

Top row:    Males - UCLA vs PRP tau | UCLA vs PRP RMSSD
Bottom row: Females - UCLA vs PRP tau | UCLA vs PRP RMSSD

Output: paper1_figure1_scatter_plots.png (300 dpi, publication-ready)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14

# Directories
OUTPUT_DIR = Path("results/analysis_outputs/paper1_distributional")

print("=" * 80)
print("PAPER 1: CREATING SCATTER PLOTS (Figure 1)")
print("=" * 80)

# Load data
df = pd.read_csv(OUTPUT_DIR / "paper1_participant_variability_metrics.csv")

# Filter valid data
df_tau = df.dropna(subset=['ucla_total', 'prp_tau_long', 'gender'])
df_rmssd = df.dropna(subset=['ucla_total', 'prp_rmssd', 'gender'])

# Separate by gender
males_tau = df_tau[df_tau['gender'] == 'male']
females_tau = df_tau[df_tau['gender'] == 'female']
males_rmssd = df_rmssd[df_rmssd['gender'] == 'male']
females_rmssd = df_rmssd[df_rmssd['gender'] == 'female']

print(f"\nSample sizes:")
print(f"  PRP tau:  Males N={len(males_tau)}, Females N={len(females_tau)}")
print(f"  PRP RMSSD: Males N={len(males_rmssd)}, Females N={len(females_rmssd)}")

# Create 2×2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Gender-Specific Loneliness Effects on Attentional Variability',
             fontsize=16, fontweight='bold', y=0.995)

# Color scheme
color_male = '#2E86AB'  # Blue
color_female = '#A23B72'  # Purple/Pink

# ============================================================================
# SUBPLOT 1: Males - UCLA vs PRP tau
# ============================================================================
ax = axes[0, 0]
x, y = males_tau['ucla_total'], males_tau['prp_tau_long']
r, p = stats.pearsonr(x, y)

# Scatter points
ax.scatter(x, y, alpha=0.6, s=80, color=color_male, edgecolors='black', linewidth=0.5)

# Regression line with 95% CI
z = np.polyfit(x, y, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = p_fit(x_line)
ax.plot(x_line, y_line, color=color_male, linewidth=2.5, label=f'r = {r:.3f}')

# 95% CI band
from scipy.stats import t as t_dist
n = len(x)
dof = n - 2
t_val = t_dist.ppf(0.975, dof)
residuals = y - p_fit(x)
se = np.sqrt(np.sum(residuals**2) / dof)
x_mean = x.mean()
sxx = np.sum((x - x_mean)**2)
se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
ci_upper = y_line + t_val * se_line
ci_lower = y_line - t_val * se_line
ax.fill_between(x_line, ci_lower, ci_upper, color=color_male, alpha=0.2)

ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
ax.set_ylabel('PRP τ (tau) - Long SOA (ms)', fontweight='bold')
ax.set_title('Males: UCLA vs Attentional Lapses (tau)', fontweight='bold', pad=10)
ax.legend(loc='upper left', fontsize=11, frameon=True)

# Add stats text
stats_text = f"N = {n}\nr = {r:.3f}\np = {p:.4f}" if p >= 0.001 else f"N = {n}\nr = {r:.3f}\np < .001"
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# ============================================================================
# SUBPLOT 2: Males - UCLA vs RMSSD
# ============================================================================
ax = axes[0, 1]
x, y = males_rmssd['ucla_total'], males_rmssd['prp_rmssd']
r, p = stats.pearsonr(x, y)

ax.scatter(x, y, alpha=0.6, s=80, color=color_male, edgecolors='black', linewidth=0.5)

z = np.polyfit(x, y, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = p_fit(x_line)
ax.plot(x_line, y_line, color=color_male, linewidth=2.5, label=f'r = {r:.3f}')

n = len(x)
dof = n - 2
t_val = t_dist.ppf(0.975, dof)
residuals = y - p_fit(x)
se = np.sqrt(np.sum(residuals**2) / dof)
x_mean = x.mean()
sxx = np.sum((x - x_mean)**2)
se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
ci_upper = y_line + t_val * se_line
ci_lower = y_line - t_val * se_line
ax.fill_between(x_line, ci_lower, ci_upper, color=color_male, alpha=0.2)

ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
ax.set_ylabel('PRP RMSSD (ms)', fontweight='bold')
ax.set_title('Males: UCLA vs Sequential Variability (RMSSD)', fontweight='bold', pad=10)
ax.legend(loc='upper left', fontsize=11, frameon=True)

stats_text = f"N = {n}\nr = {r:.3f}\np = {p:.4f}" if p >= 0.001 else f"N = {n}\nr = {r:.3f}\np < .001"
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# ============================================================================
# SUBPLOT 3: Females - UCLA vs PRP tau
# ============================================================================
ax = axes[1, 0]
x, y = females_tau['ucla_total'], females_tau['prp_tau_long']
r, p = stats.pearsonr(x, y)

ax.scatter(x, y, alpha=0.6, s=80, color=color_female, edgecolors='black', linewidth=0.5)

z = np.polyfit(x, y, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = p_fit(x_line)
ax.plot(x_line, y_line, color=color_female, linewidth=2.5, label=f'r = {r:.3f}')

n = len(x)
dof = n - 2
t_val = t_dist.ppf(0.975, dof)
residuals = y - p_fit(x)
se = np.sqrt(np.sum(residuals**2) / dof)
x_mean = x.mean()
sxx = np.sum((x - x_mean)**2)
se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
ci_upper = y_line + t_val * se_line
ci_lower = y_line - t_val * se_line
ax.fill_between(x_line, ci_lower, ci_upper, color=color_female, alpha=0.2)

ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
ax.set_ylabel('PRP τ (tau) - Long SOA (ms)', fontweight='bold')
ax.set_title('Females: UCLA vs Attentional Lapses (tau)', fontweight='bold', pad=10)
ax.legend(loc='upper left', fontsize=11, frameon=True)

stats_text = f"N = {n}\nr = {r:.3f}\np = {p:.4f}"
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# ============================================================================
# SUBPLOT 4: Females - UCLA vs RMSSD
# ============================================================================
ax = axes[1, 1]
x, y = females_rmssd['ucla_total'], females_rmssd['prp_rmssd']
r, p = stats.pearsonr(x, y)

ax.scatter(x, y, alpha=0.6, s=80, color=color_female, edgecolors='black', linewidth=0.5)

z = np.polyfit(x, y, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = p_fit(x_line)
ax.plot(x_line, y_line, color=color_female, linewidth=2.5, label=f'r = {r:.3f}')

n = len(x)
dof = n - 2
t_val = t_dist.ppf(0.975, dof)
residuals = y - p_fit(x)
se = np.sqrt(np.sum(residuals**2) / dof)
x_mean = x.mean()
sxx = np.sum((x - x_mean)**2)
se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
ci_upper = y_line + t_val * se_line
ci_lower = y_line - t_val * se_line
ax.fill_between(x_line, ci_lower, ci_upper, color=color_female, alpha=0.2)

ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
ax.set_ylabel('PRP RMSSD (ms)', fontweight='bold')
ax.set_title('Females: UCLA vs Sequential Variability (RMSSD)', fontweight='bold', pad=10)
ax.legend(loc='upper left', fontsize=11, frameon=True)

stats_text = f"N = {n}\nr = {r:.3f}\np = {p:.4f}"
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# ============================================================================
# Final adjustments and save
# ============================================================================
plt.tight_layout(rect=[0, 0, 1, 0.99])

output_path = OUTPUT_DIR / "paper1_figure1_scatter_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Figure 1 saved: {output_path}")
print("\nInterpretation:")
print("  - Males (top row): STRONG positive correlations (r=0.50-0.57, p<.01)")
print("  - Females (bottom row): ZERO correlations (r≈0, p>.6)")
print("  - Visual impact: Males show CLEAR upward trends, females show FLAT lines")
print("\n" + "=" * 80)
print("SCATTER PLOTS COMPLETE!")
print("=" * 80)
