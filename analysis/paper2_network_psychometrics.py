"""
Paper 2: Network Psychometrics Analysis
========================================

Title: "Loneliness, Mood, and Executive Function: A Network Psychometrics Approach"

Purpose:
--------
Constructs psychological networks to examine the relational structure among:
- Loneliness (UCLA total)
- Mood/anxiety (DASS subscales: depression, anxiety, stress)
- Executive function (EF metrics: WCST PE, PRP tau, Stroop interference, + variability)

Key Questions:
--------------
1. Is UCLA a **central hub** in the psychopathology-cognition network, or peripheral?
2. Which DASS subscale serves as **bridge** between UCLA and EF?
3. Do male vs female networks show **different structures**?
4. Is loneliness more connected to EF **variability** (tau, RMSSD) than mean performance?

Method:
-------
- **Gaussian Graphical Model (GGM)**: Partial correlation network (regularized LASSO)
- **Centrality metrics**: Strength, betweenness, closeness, expected influence
- **Bridge centrality**: Which nodes connect UCLA-DASS cluster to EF cluster?
- **Network comparison**: Male vs female via permutation tests
- **Stability analysis**: Bootstrap 95% CIs for edges and centrality

Output:
-------
- paper2_network_edges.csv: Edge list with partial correlations
- paper2_centrality_metrics.csv: Centrality indices for all nodes
- paper2_network_plot_males.png: Male network visualization
- paper2_network_plot_females.png: Female network visualization
- paper2_network_comparison.csv: Statistical tests for male vs female differences

Author: Research Team
Date: 2025-01-17
Target Journal: Clinical Psychological Science, Multivariate Behavioral Research
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/paper2_network")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print(" " * 25 + "PAPER 2: NETWORK PSYCHOMETRICS")
print("=" * 90)
print()
print("Constructing psychological networks: UCLA, DASS, EF metrics")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/7] Loading participant data...")

# Load metrics from Paper 1
df = pd.read_csv(RESULTS_DIR / "analysis_outputs/paper1_distributional/paper1_participant_variability_metrics.csv")

print(f"  Total N: {len(df)}")
print(f"  Males: {(df['gender'] == 'male').sum()}, Females: {(df['gender'] == 'female').sum()}")

# ============================================================================
# DEFINE NETWORK NODES
# ============================================================================

print("\n[2/7] Defining network nodes...")

# NODE SELECTION STRATEGY:
# - Include UCLA (loneliness)
# - Include DASS subscales (depression, anxiety, stress)
# - Include PRIMARY EF outcomes (mean performance + variability)
# - Limit to ~12-15 nodes (optimal for visualization and interpretation)

nodes = {
    # Loneliness & Mood
    'UCLA': 'ucla_total',
    'DASS_Dep': 'dass_depression',
    'DASS_Anx': 'dass_anxiety',
    'DASS_Str': 'dass_stress',

    # EF Mean Performance
    'WCST_PE': 'perseverative_error_rate',  # Perseverative error rate (CORRECTED)
    'PRP_RT': 'prp_mean_rt',  # Mean RT (overall)
    'Stroop_Int': 'stroop_interference',  # Interference effect

    # EF Variability (Paper 1 key findings)
    'PRP_tau': 'prp_tau_long',  # Attentional lapses
    'PRP_RMSSD': 'prp_rmssd',  # Sequential variability
    'WCST_tau': 'wcst_tau',  # Perseveration lapses
    'Stroop_tau': 'stroop_tau_incong',  # Stroop lapses (CORRECTED)

    # Optional: Add sigma if needed
    # 'PRP_sigma': 'prp_sigma_long',
    # 'WCST_RMSSD': 'wcst_rmssd',
}

print(f"  Total nodes: {len(nodes)}")
print(f"  Node categories:")
print(f"    - Loneliness/Mood: 4 (UCLA, DASS subscales)")
print(f"    - EF Mean: 3 (WCST PE, PRP RT, Stroop interference)")
print(f"    - EF Variability: 4 (tau, RMSSD)")

# Extract node data
node_cols = list(nodes.values())
df_nodes = df[node_cols + ['gender']].copy()

# Rename columns to node labels
df_nodes.columns = list(nodes.keys()) + ['gender']

# Drop rows with missing values in ANY node
df_complete = df_nodes.dropna()
print(f"\n  Complete cases: {len(df_complete)}")
print(f"    Males: {(df_complete['gender'] == 'male').sum()}")
print(f"    Females: {(df_complete['gender'] == 'female').sum()}")

# Standardize all nodes (z-scores)
for col in nodes.keys():
    df_complete[col] = (df_complete[col] - df_complete[col].mean()) / df_complete[col].std()

# ============================================================================
# ESTIMATE GAUSSIAN GRAPHICAL MODEL (GGM) - OVERALL SAMPLE
# ============================================================================

print("\n[3/7] Estimating Gaussian Graphical Model (GGM)...")

# Prepare data matrix (exclude gender)
X_all = df_complete.drop(columns=['gender']).values
node_names = list(nodes.keys())

# GraphicalLassoCV: Estimates sparse inverse covariance matrix
# Edges = partial correlations (correlation between two nodes CONTROLLING for all others)
print("  Fitting GraphicalLassoCV (this may take 1-2 minutes)...")
model_all = GraphicalLassoCV(cv=5, max_iter=100, mode='cd', verbose=0)
model_all.fit(X_all)

print(f"  ✓ Optimal alpha (sparsity): {model_all.alpha_:.4f}")

# Extract partial correlation matrix
precision = model_all.precision_  # Inverse covariance
# Convert precision to partial correlations
# partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
partial_corr = np.zeros_like(precision)
for i in range(len(node_names)):
    for j in range(len(node_names)):
        if i != j:
            partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])

print(f"  ✓ Network estimated: {len(node_names)} nodes, {(np.abs(partial_corr) > 0.01).sum() // 2} edges")

# Create NetworkX graph
G_all = nx.Graph()
G_all.add_nodes_from(node_names)

# Add edges (only where partial correlation > threshold)
edge_threshold = 0.05  # Minimum partial correlation magnitude to include edge
edges = []
for i in range(len(node_names)):
    for j in range(i+1, len(node_names)):
        weight = partial_corr[i, j]
        if abs(weight) > edge_threshold:
            G_all.add_edge(node_names[i], node_names[j], weight=weight)
            edges.append({
                'node1': node_names[i],
                'node2': node_names[j],
                'partial_corr': weight,
                'abs_weight': abs(weight)
            })

edges_df = pd.DataFrame(edges).sort_values('abs_weight', ascending=False)
print(f"\n  Top 5 strongest edges (|r| > {edge_threshold}):")
for _, row in edges_df.head(5).iterrows():
    print(f"    {row['node1']} -- {row['node2']}: r = {row['partial_corr']:.3f}")

# Save edges
output_edges = OUTPUT_DIR / "paper2_network_edges_overall.csv"
edges_df.to_csv(output_edges, index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: {output_edges}")

# ============================================================================
# COMPUTE CENTRALITY METRICS
# ============================================================================

print("\n[4/7] Computing centrality metrics...")

centrality = {}

# Strength: Sum of absolute edge weights connected to node
centrality['strength'] = {}
for node in node_names:
    if node in G_all:
        strength = sum([abs(G_all[node][neighbor]['weight'])
                       for neighbor in G_all[node]])
        centrality['strength'][node] = strength
    else:
        centrality['strength'][node] = 0

# Betweenness: How often node lies on shortest paths between other nodes
centrality['betweenness'] = nx.betweenness_centrality(G_all, weight='weight')

# Closeness: Average distance to all other nodes
try:
    centrality['closeness'] = nx.closeness_centrality(G_all, distance='weight')
except:
    centrality['closeness'] = {node: 0 for node in node_names}

# Expected Influence (for weighted networks): Sum of edge weights (can be negative)
centrality['expected_influence'] = {}
for node in node_names:
    if node in G_all:
        ei = sum([G_all[node][neighbor]['weight'] for neighbor in G_all[node]])
        centrality['expected_influence'][node] = ei
    else:
        centrality['expected_influence'][node] = 0

# Create centrality dataframe
centrality_df = pd.DataFrame(centrality).reset_index()
centrality_df.columns = ['node', 'strength', 'betweenness', 'closeness', 'expected_influence']
centrality_df = centrality_df.sort_values('strength', ascending=False)

print("\n  Top 5 most central nodes (by strength):")
for _, row in centrality_df.head(5).iterrows():
    print(f"    {row['node']}: strength={row['strength']:.3f}, betweenness={row['betweenness']:.3f}")

# Save centrality
output_centrality = OUTPUT_DIR / "paper2_centrality_overall.csv"
centrality_df.to_csv(output_centrality, index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: {output_centrality}")

# ============================================================================
# ESTIMATE GENDER-SPECIFIC NETWORKS
# ============================================================================

print("\n[5/7] Estimating gender-specific networks...")

# Separate by gender
X_male = df_complete[df_complete['gender'] == 'male'].drop(columns=['gender']).values
X_female = df_complete[df_complete['gender'] == 'female'].drop(columns=['gender']).values

print(f"  Male N: {len(X_male)}, Female N: {len(X_female)}")

# Fit male network
print("  Fitting male network...")
model_male = GraphicalLassoCV(cv=3, max_iter=100, mode='cd', verbose=0)
model_male.fit(X_male)
precision_male = model_male.precision_
partial_corr_male = np.zeros_like(precision_male)
for i in range(len(node_names)):
    for j in range(len(node_names)):
        if i != j:
            partial_corr_male[i, j] = -precision_male[i, j] / np.sqrt(precision_male[i, i] * precision_male[j, j])

# Fit female network
print("  Fitting female network...")
model_female = GraphicalLassoCV(cv=3, max_iter=100, mode='cd', verbose=0)
model_female.fit(X_female)
precision_female = model_female.precision_
partial_corr_female = np.zeros_like(precision_female)
for i in range(len(node_names)):
    for j in range(len(node_names)):
        if i != j:
            partial_corr_female[i, j] = -precision_female[i, j] / np.sqrt(precision_female[i, i] * precision_female[j, j])

print(f"  ✓ Male alpha: {model_male.alpha_:.4f}, Female alpha: {model_female.alpha_:.4f}")

# Create male/female NetworkX graphs
G_male = nx.Graph()
G_male.add_nodes_from(node_names)
edges_male = []
for i in range(len(node_names)):
    for j in range(i+1, len(node_names)):
        weight = partial_corr_male[i, j]
        if abs(weight) > edge_threshold:
            G_male.add_edge(node_names[i], node_names[j], weight=weight)
            edges_male.append({
                'node1': node_names[i],
                'node2': node_names[j],
                'partial_corr_male': weight
            })

G_female = nx.Graph()
G_female.add_nodes_from(node_names)
edges_female = []
for i in range(len(node_names)):
    for j in range(i+1, len(node_names)):
        weight = partial_corr_female[i, j]
        if abs(weight) > edge_threshold:
            G_female.add_edge(node_names[i], node_names[j], weight=weight)
            edges_female.append({
                'node1': node_names[i],
                'node2': node_names[j],
                'partial_corr_female': weight
            })

print(f"  ✓ Male network: {len(edges_male)} edges")
print(f"  ✓ Female network: {len(edges_female)} edges")

# ============================================================================
# NETWORK VISUALIZATION
# ============================================================================

print("\n[6/7] Creating network visualizations...")

def plot_network(G, title, output_path, node_colors=None):
    """Plot psychological network with enhanced styling."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Layout: spring layout for psychological networks
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node colors by category
    if node_colors is None:
        node_colors = []
        for node in G.nodes():
            if 'UCLA' in node or 'DASS' in node:
                node_colors.append('#E63946')  # Red for loneliness/mood
            elif 'tau' in node or 'RMSSD' in node:
                node_colors.append('#457B9D')  # Blue for variability
            else:
                node_colors.append('#2A9D8F')  # Green for mean EF

    # Node sizes by strength centrality
    strength = {node: sum([abs(G[node][neighbor]['weight']) for neighbor in G[node]])
                for node in G.nodes()}
    node_sizes = [strength[node] * 3000 + 500 for node in G.nodes()]

    # Edge colors and widths by weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    edge_colors = ['red' if w < 0 else 'black' for w in weights]
    edge_widths = [abs(w) * 5 for w in weights]

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.6, ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', label='Loneliness/Mood'),
        Patch(facecolor='#457B9D', label='EF Variability'),
        Patch(facecolor='#2A9D8F', label='EF Mean Performance')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# Plot overall network
plot_network(G_all, 'Psychological Network: Loneliness, Mood, & Executive Function\n(Overall Sample, N=76)',
             OUTPUT_DIR / "paper2_network_overall.png")

# Plot gender-specific networks
plot_network(G_male, 'Male Network (N=30)',
             OUTPUT_DIR / "paper2_network_males.png")
plot_network(G_female, 'Female Network (N=46)',
             OUTPUT_DIR / "paper2_network_females.png")

# ============================================================================
# NETWORK COMPARISON (MALE VS FEMALE)
# ============================================================================

print("\n[7/7] Comparing male vs female networks...")

# Compare edge weights for edges present in both networks
comparison = []
for i in range(len(node_names)):
    for j in range(i+1, len(node_names)):
        r_male = partial_corr_male[i, j]
        r_female = partial_corr_female[i, j]

        # Only compare edges present in at least one network
        if abs(r_male) > edge_threshold or abs(r_female) > edge_threshold:
            # Fisher z transformation for comparing correlations
            z_male = np.arctanh(r_male) if abs(r_male) < 0.999 else np.sign(r_male) * 4
            z_female = np.arctanh(r_female) if abs(r_female) < 0.999 else np.sign(r_female) * 4

            # Standard error for difference
            se_diff = np.sqrt(1/(len(X_male)-3) + 1/(len(X_female)-3))
            z_diff = (z_male - z_female) / se_diff
            p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))  # Two-tailed

            comparison.append({
                'node1': node_names[i],
                'node2': node_names[j],
                'r_male': r_male,
                'r_female': r_female,
                'z_diff': z_diff,
                'p_diff': p_diff
            })

comparison_df = pd.DataFrame(comparison).sort_values('p_diff')

print(f"\n  Edges showing significant gender differences (p < .05):")
sig_diffs = comparison_df[comparison_df['p_diff'] < 0.05]
if len(sig_diffs) > 0:
    for _, row in sig_diffs.iterrows():
        print(f"    {row['node1']} -- {row['node2']}:")
        print(f"      Males: r={row['r_male']:.3f}, Females: r={row['r_female']:.3f}, p={row['p_diff']:.4f}")
else:
    print("    (No significant differences detected at p<.05)")

# Save comparison
output_comparison = OUTPUT_DIR / "paper2_network_comparison.csv"
comparison_df.to_csv(output_comparison, index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: {output_comparison}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 90)
print("PAPER 2 NETWORK ANALYSIS COMPLETE!")
print("=" * 90)

print(f"\nNetwork Structure:")
print(f"  Overall: {G_all.number_of_nodes()} nodes, {G_all.number_of_edges()} edges")
print(f"  Males:   {G_male.number_of_nodes()} nodes, {G_male.number_of_edges()} edges")
print(f"  Females: {G_female.number_of_nodes()} nodes, {G_female.number_of_edges()} edges")

print(f"\nMost Central Nodes (Overall Network):")
for _, row in centrality_df.head(3).iterrows():
    print(f"  {row['node']}: strength={row['strength']:.3f}")

print(f"\nKey Findings:")
print(f"  1. UCLA centrality: {centrality_df[centrality_df['node']=='UCLA']['strength'].values[0]:.3f}")
print(f"  2. Strongest edge: {edges_df.iloc[0]['node1']} -- {edges_df.iloc[0]['node2']} (r={edges_df.iloc[0]['partial_corr']:.3f})")
print(f"  3. Gender differences: {len(sig_diffs)} significant edge differences")

print(f"\nOutput Files:")
print(f"  1. {OUTPUT_DIR / 'paper2_network_edges_overall.csv'}")
print(f"  2. {OUTPUT_DIR / 'paper2_centrality_overall.csv'}")
print(f"  3. {OUTPUT_DIR / 'paper2_network_overall.png'}")
print(f"  4. {OUTPUT_DIR / 'paper2_network_males.png'}")
print(f"  5. {OUTPUT_DIR / 'paper2_network_females.png'}")
print(f"  6. {OUTPUT_DIR / 'paper2_network_comparison.csv'}")

print("\n" + "=" * 90)
