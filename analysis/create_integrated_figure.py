"""
Create Integrated Figure: Papers 1-3 Summary Visualization

DESIGN:
- 3-panel figure (horizontal layout)
- Panel A: Paper 1 - Gender-stratified scatter (UCLA vs PRP tau)
- Panel B: Paper 2 - Network diagram with bridge centrality highlighted
- Panel C: Paper 3 - Profile comparison heatmap

TARGET: Publication-ready composite figure (300 dpi)
"""

import sys
from pathlib import Path
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*100)
print("CREATING INTEGRATED PAPERS 1-3 FIGURE")
print("="*100)

# === LOAD DATA ===
print("\n[1] Loading data...")

# Paper 1 data
paper1_dir = Path("results/analysis_outputs/paper1_distributional")
paper1_data = pd.read_csv(paper1_dir / "paper1_participant_variability_metrics.csv", encoding='utf-8-sig')

# Paper 2 edges
paper2_dir = Path("results/analysis_outputs/paper2_network")
paper2_edges = pd.read_csv(paper2_dir / "paper2_network_edges_overall.csv", encoding='utf-8-sig')
paper2_bridge = pd.read_csv(paper2_dir / "paper2_bridge_centrality.csv", encoding='utf-8-sig')

# Paper 3 profiles
paper3_dir = Path("results/analysis_outputs/paper3_profiles")
paper3_assignments = pd.read_csv(paper3_dir / "paper3_profile_assignments.csv", encoding='utf-8-sig')

# Merge for scatter plot using master
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
master = master.rename(columns={'gender_normalized': 'gender'})
master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']

scatter_df = master[['participant_id', 'gender', 'ucla_total']].dropna(subset=['gender', 'ucla_total'])
scatter_df = scatter_df.merge(paper1_data[['participant_id', 'prp_tau_long']], on='participant_id')
scatter_df = scatter_df.dropna()

print(f"  Loaded N={len(scatter_df)} for scatter plot")

# === CREATE FIGURE ===
print("\n[2] Creating 3-panel figure...")

fig = plt.figure(figsize=(18, 6))

# === PANEL A: PAPER 1 SCATTER ===
ax1 = plt.subplot(1, 3, 1)

males = scatter_df[scatter_df['gender'] == 'male']
females = scatter_df[scatter_df['gender'] == 'female']

# Scatter points
ax1.scatter(males['ucla_total'], males['prp_tau_long'],
           c='#3498db', s=80, alpha=0.7, edgecolors='black', linewidth=0.5, label='Males')
ax1.scatter(females['ucla_total'], females['prp_tau_long'],
           c='#e74c3c', s=80, alpha=0.7, edgecolors='black', linewidth=0.5, label='Females')

# Regression lines
from sklearn.linear_model import LinearRegression

# Males
X_m = males['ucla_total'].values.reshape(-1, 1)
y_m = males['prp_tau_long'].values
reg_m = LinearRegression().fit(X_m, y_m)
x_range_m = np.linspace(males['ucla_total'].min(), males['ucla_total'].max(), 100)
y_pred_m = reg_m.predict(x_range_m.reshape(-1, 1))
ax1.plot(x_range_m, y_pred_m, color='#3498db', linewidth=2.5, linestyle='--')

# Females
X_f = females['ucla_total'].values.reshape(-1, 1)
y_f = females['prp_tau_long'].values
reg_f = LinearRegression().fit(X_f, y_f)
x_range_f = np.linspace(females['ucla_total'].min(), females['ucla_total'].max(), 100)
y_pred_f = reg_f.predict(x_range_f.reshape(-1, 1))
ax1.plot(x_range_f, y_pred_f, color='#e74c3c', linewidth=2.5, linestyle='--')

# Statistics
r_m, p_m = stats.pearsonr(males['ucla_total'], males['prp_tau_long'])
r_f, p_f = stats.pearsonr(females['ucla_total'], females['prp_tau_long'])

# Add stats text
ax1.text(0.05, 0.95, f'Males: r={r_m:.2f}, p={p_m:.3f}',
        transform=ax1.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(0.05, 0.85, f'Females: r={r_f:.2f}, p={p_f:.3f}',
        transform=ax1.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_xlabel('UCLA Loneliness Scale', fontsize=13, fontweight='bold')
ax1.set_ylabel('PRP Tau (ms)\nExecutive Lapses', fontsize=13, fontweight='bold')
ax1.set_title('A. Paper 1: Gender Moderation', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# === PANEL B: PAPER 2 NETWORK (simplified) ===
ax2 = plt.subplot(1, 3, 2)

# Node positions (manually arranged for clarity)
node_positions = {
    'UCLA': (0.5, 0.9),
    'DASS_Dep': (0.2, 0.7),
    'DASS_Anx': (0.5, 0.7),
    'DASS_Str': (0.8, 0.7),  # Bridge node - highlighted
    'PRP_tau': (0.2, 0.3),
    'PRP_RMSSD': (0.5, 0.3),
    'PRP_RT': (0.8, 0.3),
    'WCST_tau': (0.2, 0.1),
    'WCST_PE': (0.5, 0.1),
    'Stroop_Int': (0.7, 0.1),
    'Stroop_tau': (0.9, 0.1)
}

# Draw edges (only strongest ones for clarity)
strong_edges = paper2_edges.head(15)  # Top 15 edges

for _, edge in strong_edges.iterrows():
    n1, n2 = edge['node1'], edge['node2']
    if n1 in node_positions and n2 in node_positions:
        x1, y1 = node_positions[n1]
        x2, y2 = node_positions[n2]

        # Edge color based on bridge involvement
        if 'DASS_Str' in [n1, n2]:
            color = '#e67e22'  # Orange for bridge edges
            alpha = 0.8
            width = 2.5
        else:
            color = 'gray'
            alpha = 0.4
            width = 1.5

        ax2.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=width, zorder=1)

# Draw nodes
for node, (x, y) in node_positions.items():
    # Node size based on bridge centrality
    bridge_row = paper2_bridge[paper2_bridge['node'] == node]
    if len(bridge_row) > 0 and bridge_row.iloc[0]['bridge_strength'] > 0.10:
        size = 800  # Large for top bridge
        color = '#e67e22'  # Orange
        edgecolor = 'black'
        linewidth = 3
    elif node in ['UCLA', 'DASS_Dep', 'DASS_Anx', 'DASS_Str']:
        size = 400  # Medium for mood
        color = '#9b59b6'  # Purple
        edgecolor = 'black'
        linewidth = 1.5
    else:
        size = 300  # Small for EF
        color = '#2ecc71'  # Green
        edgecolor = 'black'
        linewidth = 1.5

    ax2.scatter(x, y, s=size, c=color, edgecolors=edgecolor, linewidth=linewidth, zorder=2)

    # Node labels
    ax2.text(x, y, node, fontsize=9, ha='center', va='center', fontweight='bold', zorder=3)

ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.05, 1.0)
ax2.axis('off')
ax2.set_title('B. Paper 2: Network Structure\n(DASS_Str = Bridge Node)',
             fontsize=14, fontweight='bold', pad=15)

# Legend
mood_patch = mpatches.Patch(color='#9b59b6', label='Mood Cluster')
ef_patch = mpatches.Patch(color='#2ecc71', label='EF Cluster')
bridge_patch = mpatches.Patch(color='#e67e22', label='Bridge Node')
ax2.legend(handles=[mood_patch, ef_patch, bridge_patch], loc='upper left', fontsize=10)

# === PANEL C: PAPER 3 PROFILES HEATMAP ===
ax3 = plt.subplot(1, 3, 3)

# Load profile data
paper1_with_profiles = paper1_data.merge(paper3_assignments[['participant_id', 'profile']], on='participant_id')

# Compute profile means for key variables
profile_means = paper1_with_profiles.groupby('profile').agg({
    'prp_tau_long': 'mean',
    'prp_rmssd': 'mean',
    'wcst_tau': 'mean',
    'wcst_rmssd': 'mean',
    'stroop_tau_incong': 'mean'
}).T

# Normalize for heatmap (z-score)
from sklearn.preprocessing import StandardScaler
profile_means_z = pd.DataFrame(
    StandardScaler().fit_transform(profile_means),
    index=profile_means.index,
    columns=profile_means.columns
)

# Rename for clarity
profile_means_z.index = ['PRP Tau', 'PRP RMSSD', 'WCST Tau', 'WCST RMSSD', 'Stroop Tau']
profile_means_z.columns = [f'Profile {int(c)}' for c in profile_means_z.columns]

# Heatmap
sns.heatmap(profile_means_z, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
           cbar_kws={'label': 'Z-score'}, linewidths=1, linecolor='black',
           ax=ax3, vmin=-1, vmax=1)

ax3.set_xlabel('Latent Profile', fontsize=13, fontweight='bold')
ax3.set_ylabel('EF Variability Metric', fontsize=13, fontweight='bold')
ax3.set_title('C. Paper 3: Latent Profiles\n(k=2 GMM)', fontsize=14, fontweight='bold', pad=15)
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

# === OVERALL TITLE ===
fig.suptitle('Integrated Multi-Method Analysis: Loneliness and Executive Function Variability',
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = OUTPUT_DIR / "integrated_figure_papers123.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n  Saved: {output_path.name}")

print("\n" + "="*100)
print("FIGURE CREATION COMPLETE")
print("="*100)
print("\nINTEGRATED FIGURE SUMMARY:")
print("  Panel A: Gender moderation (Paper 1)")
print(f"    - Males: r={r_m:.3f}, p={p_m:.4f}")
print(f"    - Females: r={r_f:.3f}, p={p_f:.4f}")
print(f"  Panel B: Network structure (Paper 2)")
print(f"    - Top bridge: DASS_Str")
print(f"    - Mood cluster: UCLA + DASS (purple)")
print(f"    - EF cluster: variability metrics (green)")
print(f"  Panel C: Latent profiles (Paper 3)")
print(f"    - Profile 1: High variability")
print(f"    - Profile 2: Low variability")
print("="*100)
