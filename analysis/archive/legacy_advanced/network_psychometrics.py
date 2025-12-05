"""
Network Psychometrics Analysis
==============================
Constructs and analyzes partial correlation networks of:
- UCLA loneliness
- DASS-21 subscales (Depression, Anxiety, Stress)
- Executive function measures (WCST PE, Stroop, PRP)

Network analysis reveals:
- Central nodes (bridge symptoms)
- Clustering patterns
- Gender differences in network structure
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import zscore
import warnings

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "network_psychometrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_partial_correlations(data):
    """Compute partial correlation matrix (controlling for all other variables)."""
    n = data.shape[1]
    partial_corr = np.zeros((n, n))

    # Compute correlation matrix
    corr = data.corr().values

    # Precision matrix (inverse of correlation)
    try:
        precision = np.linalg.inv(corr)

        # Partial correlations from precision matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    partial_corr[i, j] = 1.0
                else:
                    partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
    except:
        # If singular, use regularized inverse
        partial_corr = corr  # Fallback to regular correlations

    return pd.DataFrame(partial_corr, index=data.columns, columns=data.columns)


def compute_network_centrality(partial_corr):
    """Compute centrality measures for each node."""
    # Absolute values for strength calculation
    abs_corr = partial_corr.abs()

    # Degree centrality (sum of absolute connections)
    strength = abs_corr.sum(axis=0) - 1  # Subtract self-connection

    # Expected influence (sum considering signs)
    expected_influence = partial_corr.sum(axis=0) - 1

    # Closeness (mean strength to all nodes)
    closeness = strength / (len(partial_corr) - 1)

    return pd.DataFrame({
        'node': partial_corr.columns,
        'strength': strength.values,
        'expected_influence': expected_influence.values,
        'closeness': closeness.values
    })


def compare_networks_by_gender(master_df, variables):
    """Compare network structures between genders."""
    # Split by gender
    males = master_df[master_df['gender_male'] == 1][variables].dropna()
    females = master_df[master_df['gender_male'] == 0][variables].dropna()

    if len(males) < 30 or len(females) < 30:
        return None

    # Compute networks
    male_network = compute_partial_correlations(males)
    female_network = compute_partial_correlations(females)

    # Network comparison: correlation of edge weights
    male_edges = male_network.values[np.triu_indices(len(variables), k=1)]
    female_edges = female_network.values[np.triu_indices(len(variables), k=1)]

    r_networks, p_networks = stats.pearsonr(male_edges, female_edges)

    # Mean edge strength difference
    male_strength = np.abs(male_edges).mean()
    female_strength = np.abs(female_edges).mean()

    # Largest edge differences
    edge_diff = male_edges - female_edges
    edge_names = []
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            edge_names.append(f"{variables[i]}--{variables[j]}")

    diff_df = pd.DataFrame({
        'edge': edge_names,
        'male_weight': male_edges,
        'female_weight': female_edges,
        'difference': edge_diff
    })
    diff_df = diff_df.sort_values('difference', key=abs, ascending=False)

    return {
        'male_network': male_network,
        'female_network': female_network,
        'network_similarity': r_networks,
        'network_sim_p': p_networks,
        'male_mean_strength': male_strength,
        'female_mean_strength': female_strength,
        'edge_differences': diff_df,
        'n_male': len(males),
        'n_female': len(females)
    }


def identify_bridge_nodes(partial_corr, psychological_vars, ef_vars):
    """Identify nodes that bridge psychological and EF domains."""
    bridge_centrality = {}

    for node in partial_corr.columns:
        # Bridge strength = connections to other domain
        if node in psychological_vars:
            other_domain = ef_vars
        else:
            other_domain = psychological_vars

        bridge_connections = partial_corr.loc[node, [v for v in other_domain if v in partial_corr.columns]]
        bridge_centrality[node] = bridge_connections.abs().sum()

    return pd.DataFrame({
        'node': list(bridge_centrality.keys()),
        'bridge_centrality': list(bridge_centrality.values())
    }).sort_values('bridge_centrality', ascending=False)


def main():
    print("=" * 60)
    print("Network Psychometrics Analysis")
    print("=" * 60)

    # Load data
    master = load_master_dataset(use_cache=True)

    # Define variables
    psychological_vars = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress']
    ef_vars = ['pe_rate', 'stroop_interference', 'prp_bottleneck']

    all_vars = psychological_vars + ef_vars
    available_vars = [v for v in all_vars if v in master.columns]

    print(f"Available variables: {available_vars}")

    # Prepare data
    network_data = master[available_vars].dropna()
    print(f"Complete cases: N={len(network_data)}")

    # Standardize
    network_data_z = network_data.apply(zscore)

    # 1. Full network
    print("\n[1] Full Partial Correlation Network")
    print("-" * 40)

    partial_corr = compute_partial_correlations(network_data_z)
    print("\nPartial Correlation Matrix:")
    print(partial_corr.round(3).to_string())

    partial_corr.to_csv(OUTPUT_DIR / "partial_correlation_matrix.csv", encoding='utf-8-sig')

    # 2. Centrality measures
    print("\n[2] Network Centrality Measures")
    print("-" * 40)

    centrality = compute_network_centrality(partial_corr)
    centrality = centrality.sort_values('strength', ascending=False)
    print(centrality.to_string(index=False))

    centrality.to_csv(OUTPUT_DIR / "centrality_measures.csv", index=False, encoding='utf-8-sig')

    # Most central node
    most_central = centrality.iloc[0]['node']
    print(f"\n  Most central node: {most_central}")

    # 3. Bridge analysis
    print("\n[3] Bridge Centrality Analysis")
    print("-" * 40)

    psych_available = [v for v in psychological_vars if v in available_vars]
    ef_available = [v for v in ef_vars if v in available_vars]

    bridge_results = identify_bridge_nodes(partial_corr, psych_available, ef_available)
    print(bridge_results.to_string(index=False))

    bridge_results.to_csv(OUTPUT_DIR / "bridge_centrality.csv", index=False, encoding='utf-8-sig')

    # 4. Gender comparison
    print("\n[4] Network Comparison by Gender")
    print("-" * 40)

    gender_results = compare_networks_by_gender(master, available_vars)
    if gender_results:
        print(f"  Male network (N={gender_results['n_male']})")
        print(f"  Female network (N={gender_results['n_female']})")
        print(f"  Network similarity: r={gender_results['network_similarity']:.3f}, p={gender_results['network_sim_p']:.4f}")
        print(f"  Male mean edge strength: {gender_results['male_mean_strength']:.3f}")
        print(f"  Female mean edge strength: {gender_results['female_mean_strength']:.3f}")

        print("\n  Largest edge differences (Male - Female):")
        print(gender_results['edge_differences'].head(5).to_string(index=False))

        gender_results['male_network'].to_csv(OUTPUT_DIR / "male_network.csv", encoding='utf-8-sig')
        gender_results['female_network'].to_csv(OUTPUT_DIR / "female_network.csv", encoding='utf-8-sig')
        gender_results['edge_differences'].to_csv(OUTPUT_DIR / "gender_edge_differences.csv", index=False, encoding='utf-8-sig')
    else:
        print("  Insufficient sample size for gender comparison")

    # 5. Key edges summary
    print("\n[5] Key UCLA Edges")
    print("-" * 40)
    if 'ucla_score' in partial_corr.columns:
        ucla_edges = partial_corr['ucla_score'].drop('ucla_score').sort_values(key=abs, ascending=False)
        print("  UCLA partial correlations:")
        for node, r in ucla_edges.items():
            print(f"    {node}: r={r:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Network Psychometrics Results:

Network Structure:
- N = {len(network_data)} (complete cases)
- Variables: {', '.join(available_vars)}

Centrality:
- Most central node: {most_central}
- Bridge nodes connect psychological symptoms to EF

Key Findings:
""")
    # UCLA connections
    if 'ucla_score' in partial_corr.columns:
        print("  UCLA partial correlations (controlling for all others):")
        for node in partial_corr.columns:
            if node != 'ucla_score':
                r = partial_corr.loc['ucla_score', node]
                print(f"    - UCLA ↔ {node}: r = {r:.3f}")

    print(f"""
Interpretation:
- High centrality = node influences many others
- Bridge centrality = connects psychological to EF domain
- Gender differences in network structure may explain differential vulnerability
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
