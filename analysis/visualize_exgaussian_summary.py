"""
Ex-Gaussian Findings Visualization
===================================
Creates a summary plot of key Ex-Gaussian findings from Phase 6.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results/analysis_outputs/mechanism_analysis/exgaussian")
OUTPUT_DIR = Path("results/analysis_outputs/mechanism_analysis/exgaussian")

print("=" * 80)
print("VISUALIZING EX-GAUSSIAN FINDINGS")
print("=" * 80)

# Load correlation results
prp_corr = pd.read_csv(RESULTS_DIR / "prp_exgaussian_gender_correlations.csv", encoding='utf-8-sig')
prp_bottleneck = pd.read_csv(RESULTS_DIR / "prp_exgaussian_bottleneck_correlations.csv", encoding='utf-8-sig')

# Filter significant effects
sig_corr = prp_corr[prp_corr['p'] < 0.05].copy()
sig_bottleneck = prp_bottleneck[prp_bottleneck['p'] < 0.05].copy()

print(f"\n[1] Significant SOA-specific effects: {len(sig_corr)}")
print(sig_corr[['gender', 'soa', 'parameter', 'r', 'p']])

print(f"\n[2] Significant bottleneck effects: {len(sig_bottleneck)}")
print(sig_bottleneck[['gender', 'parameter', 'r', 'p']])

# Create summary figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PRP Ex-Gaussian Decomposition - Key Findings', fontsize=16, fontweight='bold')

# Plot 1: Males - SOA-specific effects
ax = axes[0, 0]
male_data = sig_corr[sig_corr['gender'] == 'male'].copy()
if len(male_data) > 0:
    male_data['label'] = male_data['soa'] + '_' + male_data['parameter']
    bars = ax.barh(male_data['label'], male_data['r'], color=['red' if r > 0 else 'blue' for r in male_data['r']])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Correlation (r)', fontsize=11)
    ax.set_title('Males: SOA-Specific Effects (p<0.05)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.1, 0.7)

    # Add p-values
    for i, (idx, row) in enumerate(male_data.iterrows()):
        p_str = f"p={row['p']:.3f}"
        if row['p'] < 0.01:
            p_str += "**"
        elif row['p'] < 0.05:
            p_str += "*"
        ax.text(row['r'] + 0.02, i, p_str, va='center', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No significant effects', ha='center', va='center', transform=ax.transAxes)

# Plot 2: Females - SOA-specific effects
ax = axes[0, 1]
female_data = sig_corr[sig_corr['gender'] == 'female'].copy()
if len(female_data) > 0:
    female_data['label'] = female_data['soa'] + '_' + female_data['parameter']
    bars = ax.barh(female_data['label'], female_data['r'], color=['red' if r > 0 else 'blue' for r in female_data['r']])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Correlation (r)', fontsize=11)
    ax.set_title('Females: SOA-Specific Effects (p<0.05)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 0.5)

    # Add p-values
    for i, (idx, row) in enumerate(female_data.iterrows()):
        p_str = f"p={row['p']:.3f}"
        if row['p'] < 0.01:
            p_str += "**"
        elif row['p'] < 0.05:
            p_str += "*"
        x_offset = 0.02 if row['r'] > 0 else -0.02
        ha = 'left' if row['r'] > 0 else 'right'
        ax.text(row['r'] + x_offset, i, p_str, va='center', ha=ha, fontsize=9)
else:
    ax.text(0.5, 0.5, 'No significant effects', ha='center', va='center', transform=ax.transAxes)

# Plot 3: Males - Bottleneck effects
ax = axes[1, 0]
male_bottleneck = sig_bottleneck[sig_bottleneck['gender'] == 'male'].copy()
if len(male_bottleneck) > 0:
    bars = ax.barh(male_bottleneck['parameter'], male_bottleneck['r'],
                   color=['red' if r > 0 else 'blue' for r in male_bottleneck['r']])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Correlation (r)', fontsize=11)
    ax.set_title('Males: Bottleneck Effects (p<0.05)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.1, 0.7)

    # Add p-values
    for i, (idx, row) in enumerate(male_bottleneck.iterrows()):
        p_str = f"p={row['p']:.3f}"
        if row['p'] < 0.01:
            p_str += "**"
        elif row['p'] < 0.05:
            p_str += "*"
        ax.text(row['r'] + 0.02, i, p_str, va='center', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No significant effects', ha='center', va='center', transform=ax.transAxes)

# Plot 4: Interpretation text
ax = axes[1, 1]
ax.axis('off')

interpretation = """
KEY FINDINGS:

MALES (Vulnerability Pattern):
• Long SOA τ: r=0.578, p=0.002**
  → More attentional lapses

• σ bottleneck: r=0.511, p=0.006**
  → Greater dual-task variability

• μ bottleneck: r=0.479, p=0.012*
  → Slower central processing

FEMALES (Compensation Pattern):
• Short SOA τ: r=-0.384, p=0.009**
  → FEWER lapses (hypervigilance)

• Long SOA σ: r=0.393, p=0.008**
  → Increased variability when relaxed

INTERPRETATION:
μ = Routine processing speed
σ = Processing variability
τ = Attentional lapses (slow tail)

Males: Loneliness → lapses + variability
Females: Loneliness → hypervigilance
"""

ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exgaussian_summary.png", dpi=300, bbox_inches='tight')
print(f"\n✅ Summary plot saved: {OUTPUT_DIR / 'exgaussian_summary.png'}")

print("\n" + "="*80)
print("EX-GAUSSIAN VISUALIZATION COMPLETE!")
print("="*80)
