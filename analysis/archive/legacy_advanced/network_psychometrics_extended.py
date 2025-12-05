"""
Extended Network Psychometrics Analysis
=======================================
UCLA-DASS-EF 간 네트워크 분석 심화

Features:
1. 부분상관 네트워크 (Partial correlation network)
2. 성별 비교 네트워크 구조
3. Bridge centrality 분석
4. Bootstrap edge stability

DASS-21 Control: N/A (네트워크 구조 분석)
"""

import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Add parent to path for imports
_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/network_psychometrics_extended.py')
sys.path.insert(0, str(_this_file.parent))
from utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "network_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EXTENDED NETWORK PSYCHOMETRICS ANALYSIS")
print("=" * 70)


# ============================================================================
# Helper functions for network analysis
# ============================================================================

def partial_correlation_matrix(df, method='pearson'):
    """
    Compute partial correlation matrix.
    Partial correlation controls for all other variables.
    """
    # Correlation matrix
    corr = df.corr(method=method)

    # Precision matrix (inverse of correlation matrix)
    try:
        precision = inv(corr.values)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        precision = np.linalg.pinv(corr.values)

    # Partial correlations from precision matrix
    # pcor_ij = -precision_ij / sqrt(precision_ii * precision_jj)
    n = len(corr)
    pcor = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                pcor[i, j] = 1.0
            else:
                denom = np.sqrt(precision[i, i] * precision[j, j])
                if denom > 0:
                    pcor[i, j] = -precision[i, j] / denom
                else:
                    pcor[i, j] = 0

    return pd.DataFrame(pcor, index=corr.index, columns=corr.columns)


def compute_centrality_measures(adj_matrix, threshold=0.1):
    """
    Compute various centrality measures from adjacency matrix.
    """
    # Create graph from adjacency matrix
    G = nx.Graph()

    nodes = adj_matrix.columns
    G.add_nodes_from(nodes)

    # Add edges with weights above threshold
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:
                weight = abs(adj_matrix.iloc[i, j])
                if weight > threshold:
                    G.add_edge(node_i, node_j, weight=weight)

    # Compute centrality measures
    centrality = {}

    # Strength (sum of absolute edge weights)
    strength = {}
    for node in nodes:
        strength[node] = sum(abs(adj_matrix.loc[node, :].drop(node)))
    centrality['strength'] = strength

    # Betweenness centrality
    if G.number_of_edges() > 0:
        centrality['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    else:
        centrality['betweenness'] = {node: 0 for node in nodes}

    # Closeness centrality
    if G.number_of_edges() > 0:
        centrality['closeness'] = nx.closeness_centrality(G, distance='weight')
    else:
        centrality['closeness'] = {node: 0 for node in nodes}

    # Expected influence (sum of signed edge weights)
    expected_influence = {}
    for node in nodes:
        expected_influence[node] = sum(adj_matrix.loc[node, :].drop(node))
    centrality['expected_influence'] = expected_influence

    return centrality, G


def compute_bridge_centrality(adj_matrix, communities, threshold=0.1):
    """
    Compute bridge centrality - importance of nodes connecting different communities.
    """
    nodes = adj_matrix.columns
    bridge_strength = {}

    for node in nodes:
        node_community = communities.get(node, 'unknown')
        bridge_edges = 0

        for other_node in nodes:
            if node == other_node:
                continue

            other_community = communities.get(other_node, 'unknown')
            weight = abs(adj_matrix.loc[node, other_node])

            # Edge is a bridge if it connects different communities
            if node_community != other_community and weight > threshold:
                bridge_edges += weight

        bridge_strength[node] = bridge_edges

    return bridge_strength


def bootstrap_edge_stability(df, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap confidence intervals for edge weights.
    """
    n = len(df)
    variables = df.columns

    # Store bootstrap results
    boot_edges = {(v1, v2): [] for i, v1 in enumerate(variables)
                  for j, v2 in enumerate(variables) if i < j}

    for _ in range(n_bootstrap):
        # Resample
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_df = df.iloc[boot_idx]

        # Compute partial correlations
        try:
            boot_pcor = partial_correlation_matrix(boot_df)

            for (v1, v2) in boot_edges.keys():
                boot_edges[(v1, v2)].append(boot_pcor.loc[v1, v2])
        except Exception:
            continue

    # Compute CIs
    alpha = (1 - ci) / 2
    edge_ci = {}

    for (v1, v2), values in boot_edges.items():
        if len(values) > 10:
            edge_ci[(v1, v2)] = {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, alpha * 100),
                'ci_upper': np.percentile(values, (1 - alpha) * 100)
            }

    return edge_ci


def network_comparison_test(df1, df2, n_permutations=1000):
    """
    Permutation test for comparing network structures between two groups.
    Tests global strength and maximum edge difference.
    """
    # Observed networks
    pcor1 = partial_correlation_matrix(df1)
    pcor2 = partial_correlation_matrix(df2)

    # Observed statistics
    obs_global_strength_diff = abs(pcor1.abs().sum().sum() - pcor2.abs().sum().sum())
    obs_max_edge_diff = abs(pcor1 - pcor2).max().max()

    # Combine data for permutation
    combined = pd.concat([df1, df2], ignore_index=True)
    n1, n2 = len(df1), len(df2)

    # Permutation distribution
    perm_global_strength = []
    perm_max_edge = []

    for _ in range(n_permutations):
        # Random permutation
        perm_idx = np.random.permutation(len(combined))
        perm_df1 = combined.iloc[perm_idx[:n1]]
        perm_df2 = combined.iloc[perm_idx[n1:]]

        try:
            perm_pcor1 = partial_correlation_matrix(perm_df1)
            perm_pcor2 = partial_correlation_matrix(perm_df2)

            perm_global_strength.append(
                abs(perm_pcor1.abs().sum().sum() - perm_pcor2.abs().sum().sum())
            )
            perm_max_edge.append(
                abs(perm_pcor1 - perm_pcor2).max().max()
            )
        except Exception:
            continue

    # P-values
    p_global = (np.sum(np.array(perm_global_strength) >= obs_global_strength_diff) + 1) / (len(perm_global_strength) + 1)
    p_edge = (np.sum(np.array(perm_max_edge) >= obs_max_edge_diff) + 1) / (len(perm_max_edge) + 1)

    return {
        'global_strength_diff': obs_global_strength_diff,
        'p_global_strength': p_global,
        'max_edge_diff': obs_max_edge_diff,
        'p_max_edge': p_edge
    }


# ============================================================================
# Load and prepare data
# ============================================================================
print("\n[1] Loading and preparing data...")

df = load_master_dataset()

# Select network variables - use available EF metrics
# stroop_interference may not be computed yet, so compute it if needed
if 'stroop_interference' not in df.columns:
    # Try to compute from mrt columns
    if 'mrt_incong' in df.columns and 'mrt_cong' in df.columns:
        df['stroop_interference'] = df['mrt_incong'] - df['mrt_cong']
        print("   Computed stroop_interference from mrt columns")
    elif 'rt_mean_incongruent' in df.columns and 'rt_mean_congruent' in df.columns:
        df['stroop_interference'] = df['rt_mean_incongruent'] - df['rt_mean_congruent']
        print("   Computed stroop_interference from rt_mean columns")

network_vars_candidates = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress',
                           'pe_rate', 'prp_bottleneck', 'stroop_interference', 'wcst_accuracy']

# Filter to available variables with sufficient data (at least 50 valid cases)
network_vars = []
for v in network_vars_candidates:
    if v in df.columns and df[v].notna().sum() >= 50:
        network_vars.append(v)

print(f"   Network variables: {network_vars}")

# Clean data - use pairwise complete cases approach
# First, check which variables have enough overlap
network_df = df[['participant_id', 'gender'] + network_vars].copy()

# For each variable, check non-missing count
print("   Data availability:")
for v in network_vars:
    n_valid = network_df[v].notna().sum()
    print(f"     {v}: {n_valid} valid")

# Use listwise deletion for core analysis
network_df = network_df.dropna(subset=network_vars + ['gender'])
print(f"   Complete cases (listwise): {len(network_df)}")
print(f"   Males: {(network_df['gender'] == 'male').sum()}")
print(f"   Females: {(network_df['gender'] == 'female').sum()}")

# Define communities for bridge centrality
communities = {
    'ucla_score': 'Loneliness',
    'dass_depression': 'Mood',
    'dass_anxiety': 'Mood',
    'dass_stress': 'Mood',
    'pe_rate': 'EF',
    'prp_bottleneck': 'EF',
    'stroop_interference': 'EF'
}


# ============================================================================
# 2. Full Sample Network
# ============================================================================
print("\n[2] Computing full sample network...")

# Pearson correlation matrix
corr_matrix = network_df[network_vars].corr()
corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix_full.csv", encoding='utf-8-sig')

# Partial correlation matrix
pcor_matrix = partial_correlation_matrix(network_df[network_vars])
pcor_matrix.to_csv(OUTPUT_DIR / "partial_correlation_matrix_full.csv", encoding='utf-8-sig')
print(f"   Saved: partial_correlation_matrix_full.csv")

# Centrality measures
centrality, G_full = compute_centrality_measures(pcor_matrix)

centrality_df = pd.DataFrame(centrality)
centrality_df.index.name = 'variable'
centrality_df.to_csv(OUTPUT_DIR / "centrality_measures_full.csv", encoding='utf-8-sig')
print(f"   Saved: centrality_measures_full.csv")

# Bridge centrality
bridge = compute_bridge_centrality(pcor_matrix, communities)
bridge_df = pd.DataFrame({'variable': bridge.keys(), 'bridge_strength': bridge.values()})
bridge_df.to_csv(OUTPUT_DIR / "bridge_centrality_full.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: bridge_centrality_full.csv")


# ============================================================================
# 3. Gender-specific Networks
# ============================================================================
print("\n[3] Computing gender-specific networks...")

gender_networks = {}
gender_centrality = {}

for gender in ['male', 'female']:
    gender_data = network_df[network_df['gender'] == gender][network_vars]

    if len(gender_data) < 10:
        print(f"   Skipping {gender}: insufficient data")
        continue

    # Partial correlation matrix
    pcor = partial_correlation_matrix(gender_data)
    gender_networks[gender] = pcor
    pcor.to_csv(OUTPUT_DIR / f"partial_correlation_matrix_{gender}.csv", encoding='utf-8-sig')

    # Centrality
    cent, _ = compute_centrality_measures(pcor)
    gender_centrality[gender] = cent

    print(f"   {gender.capitalize()}: N = {len(gender_data)}")

# Save gender centrality comparison
if len(gender_centrality) == 2:
    comparison_data = []
    for var in network_vars:
        row = {'variable': var}
        for gender in ['male', 'female']:
            for measure in ['strength', 'expected_influence', 'betweenness', 'closeness']:
                row[f'{measure}_{gender}'] = gender_centrality[gender][measure].get(var, np.nan)
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(OUTPUT_DIR / "centrality_gender_comparison.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: centrality_gender_comparison.csv")


# ============================================================================
# 4. Network Comparison Test (Male vs Female)
# ============================================================================
print("\n[4] Testing network differences between genders...")

male_data = network_df[network_df['gender'] == 'male'][network_vars]
female_data = network_df[network_df['gender'] == 'female'][network_vars]

if len(male_data) >= 15 and len(female_data) >= 15:
    comparison_results = network_comparison_test(male_data, female_data, n_permutations=500)

    print(f"   Global strength difference: {comparison_results['global_strength_diff']:.3f}")
    print(f"   p-value (global): {comparison_results['p_global_strength']:.4f}")
    print(f"   Maximum edge difference: {comparison_results['max_edge_diff']:.3f}")
    print(f"   p-value (edge): {comparison_results['p_max_edge']:.4f}")

    # Save results
    pd.DataFrame([comparison_results]).to_csv(
        OUTPUT_DIR / "network_comparison_test.csv", index=False, encoding='utf-8-sig'
    )
    print(f"   Saved: network_comparison_test.csv")
else:
    comparison_results = None
    print("   Insufficient data for network comparison test")


# ============================================================================
# 5. Edge-wise Comparison
# ============================================================================
print("\n[5] Computing edge-wise differences between genders...")

if len(gender_networks) == 2:
    edge_diff = gender_networks['male'] - gender_networks['female']
    edge_diff.to_csv(OUTPUT_DIR / "edge_differences_male_minus_female.csv", encoding='utf-8-sig')

    # Identify largest differences
    edge_list = []
    for i, v1 in enumerate(network_vars):
        for j, v2 in enumerate(network_vars):
            if i < j:
                edge_list.append({
                    'node1': v1,
                    'node2': v2,
                    'pcor_male': gender_networks['male'].loc[v1, v2],
                    'pcor_female': gender_networks['female'].loc[v1, v2],
                    'difference': edge_diff.loc[v1, v2],
                    'abs_difference': abs(edge_diff.loc[v1, v2])
                })

    edge_list_df = pd.DataFrame(edge_list).sort_values('abs_difference', ascending=False)
    edge_list_df.to_csv(OUTPUT_DIR / "edge_comparison_list.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: edge_comparison_list.csv")

    print("\n   Top 5 edge differences (Male - Female):")
    for _, row in edge_list_df.head(5).iterrows():
        print(f"     {row['node1']} -- {row['node2']}: Δ = {row['difference']:.3f} (M={row['pcor_male']:.3f}, F={row['pcor_female']:.3f})")


# ============================================================================
# 6. Bootstrap Edge Stability (Full Sample)
# ============================================================================
print("\n[6] Computing bootstrap edge stability (this may take a moment)...")

edge_ci = bootstrap_edge_stability(network_df[network_vars], n_bootstrap=500)

edge_stability = []
for (v1, v2), ci in edge_ci.items():
    edge_stability.append({
        'node1': v1,
        'node2': v2,
        'mean_pcor': ci['mean'],
        'ci_lower': ci['ci_lower'],
        'ci_upper': ci['ci_upper'],
        'significant': (ci['ci_lower'] > 0) or (ci['ci_upper'] < 0)
    })

edge_stability_df = pd.DataFrame(edge_stability)
edge_stability_df.to_csv(OUTPUT_DIR / "edge_bootstrap_stability.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: edge_bootstrap_stability.csv")

n_significant = edge_stability_df['significant'].sum()
print(f"   Significant edges (CI excludes 0): {n_significant}/{len(edge_stability_df)}")


# ============================================================================
# 7. Visualization
# ============================================================================
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 7a. Correlation heatmap (full sample)
ax = axes[0, 0]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title('Correlation Matrix (Full Sample)', fontweight='bold')

# 7b. Partial correlation heatmap (full sample)
ax = axes[0, 1]
mask = np.triu(np.ones_like(pcor_matrix, dtype=bool), k=1)
sns.heatmap(pcor_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title('Partial Correlation Matrix (Full Sample)', fontweight='bold')

# 7c. Centrality comparison
ax = axes[0, 2]
cent_df = centrality_df.reset_index()
cent_df_melted = cent_df.melt(id_vars='variable', var_name='measure', value_name='centrality')
sns.barplot(data=cent_df_melted, x='variable', y='centrality', hue='measure', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Centrality Measures (Full Sample)', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)

# 7d. Network graph (full sample)
ax = axes[1, 0]
pos = nx.spring_layout(G_full, k=2, iterations=50, seed=42)

# Node colors by community
node_colors = [{'Loneliness': 'red', 'Mood': 'blue', 'EF': 'green'}.get(communities.get(n, 'unknown'), 'gray')
               for n in G_full.nodes()]

# Edge weights
edge_weights = [G_full[u][v]['weight'] * 3 for u, v in G_full.edges()]

nx.draw(G_full, pos, ax=ax, with_labels=True, node_color=node_colors,
        node_size=1000, font_size=8, font_weight='bold',
        edge_color='gray', width=edge_weights, alpha=0.7)
ax.set_title('Network Graph (Full Sample)\nRed=Loneliness, Blue=Mood, Green=EF', fontweight='bold')

# 7e. Gender comparison - edge differences
ax = axes[1, 1]
if len(gender_networks) == 2:
    mask = np.triu(np.ones_like(edge_diff, dtype=bool), k=1)
    sns.heatmap(edge_diff, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-0.5, vmax=0.5, ax=ax, square=True)
    ax.set_title('Edge Differences (Male - Female)', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
    ax.set_title('Edge Differences')

# 7f. Bridge centrality
ax = axes[1, 2]
bridge_sorted = bridge_df.sort_values('bridge_strength', ascending=True)
colors = [{'Loneliness': 'red', 'Mood': 'blue', 'EF': 'green'}.get(communities.get(v, 'unknown'), 'gray')
          for v in bridge_sorted['variable']]
ax.barh(range(len(bridge_sorted)), bridge_sorted['bridge_strength'], color=colors, alpha=0.7)
ax.set_yticks(range(len(bridge_sorted)))
ax.set_yticklabels(bridge_sorted['variable'])
ax.set_xlabel('Bridge Strength')
ax.set_title('Bridge Centrality\n(Connections between communities)', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "network_analysis_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: network_analysis_plots.png")


# ============================================================================
# 8. Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("NETWORK PSYCHOMETRICS SUMMARY REPORT")
print("=" * 70)

print("\n1. Network Structure (Full Sample):")
print(f"   N = {len(network_df)}")
print(f"   Variables: {len(network_vars)}")
print(f"   Edges (|pcor| > 0.1): {G_full.number_of_edges()}")

print("\n2. Most Central Nodes (by strength):")
strength_sorted = sorted(centrality['strength'].items(), key=lambda x: -x[1])
for node, strength in strength_sorted[:3]:
    print(f"   - {node}: {strength:.3f}")

print("\n3. Bridge Nodes (connecting communities):")
bridge_sorted_dict = sorted(bridge.items(), key=lambda x: -x[1])
for node, bridge_val in bridge_sorted_dict[:3]:
    print(f"   - {node}: {bridge_val:.3f} (community: {communities.get(node, 'unknown')})")

print("\n4. Gender Comparison:")
if comparison_results:
    print(f"   Global strength difference: {comparison_results['global_strength_diff']:.3f}")
    sig_global = "Significant" if comparison_results['p_global_strength'] < 0.05 else "Not significant"
    print(f"   {sig_global} (p = {comparison_results['p_global_strength']:.4f})")

print("\n5. Key Edge Differences (Male vs Female):")
if len(gender_networks) == 2:
    for _, row in edge_list_df.head(3).iterrows():
        print(f"   - {row['node1']} -- {row['node2']}: Δ = {row['difference']:.3f}")

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("=" * 70)

# Save report
with open(OUTPUT_DIR / "network_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write("NETWORK PSYCHOMETRICS SUMMARY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"1. Network Structure (Full Sample):\n")
    f.write(f"   N = {len(network_df)}\n")
    f.write(f"   Variables: {len(network_vars)}\n")
    f.write(f"   Edges (|pcor| > 0.1): {G_full.number_of_edges()}\n\n")

    f.write(f"2. Centrality Rankings:\n")
    for measure in ['strength', 'expected_influence', 'betweenness']:
        f.write(f"   {measure.capitalize()}:\n")
        sorted_cent = sorted(centrality[measure].items(), key=lambda x: -x[1])
        for node, val in sorted_cent[:3]:
            f.write(f"     - {node}: {val:.3f}\n")

    f.write(f"\n3. Bridge Centrality:\n")
    for node, bridge_val in bridge_sorted_dict:
        f.write(f"   - {node}: {bridge_val:.3f}\n")

    if comparison_results:
        f.write(f"\n4. Gender Network Comparison:\n")
        f.write(f"   Global strength difference: {comparison_results['global_strength_diff']:.3f}, p = {comparison_results['p_global_strength']:.4f}\n")
        f.write(f"   Max edge difference: {comparison_results['max_edge_diff']:.3f}, p = {comparison_results['p_max_edge']:.4f}\n")

print("\nNetwork analysis complete!")
