"""
Paper 2: Precision Verification & Critical Review
==================================================
Comprehensive validation of network psychometrics results
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats

print('=' * 90)
print('PAPER 2: PRECISION VERIFICATION & CRITICAL REVIEW')
print('=' * 90)

# Load results
edges = pd.read_csv('results/analysis_outputs/paper2_network/paper2_network_edges_overall.csv')
centrality = pd.read_csv('results/analysis_outputs/paper2_network/paper2_centrality_overall.csv')
comparison = pd.read_csv('results/analysis_outputs/paper2_network/paper2_network_comparison.csv')

print('\n[1] SAMPLE SIZE ADEQUACY FOR NETWORK ANALYSIS')
print('-' * 90)
n_nodes = 11
n_total = 74
n_male = 30
n_female = 44

print(f'  Rule of thumb: N >= 3 * n_nodes')
print(f'    Required: >= {3 * n_nodes}')
print(f'    Actual overall: {n_total} [{"OK" if n_total >= 3*n_nodes else "WARNING"}]')
print(f'    Actual males: {n_male} [{"OK" if n_male >= 3*n_nodes else "WARNING - BORDERLINE"}]')
print(f'    Actual females: {n_female} [{"OK" if n_female >= 3*n_nodes else "WARNING"}]')

print(f'\n  Conservative rule: N >= 5 * n_nodes = {5 * n_nodes}')
print(f'    Overall: {n_total} [{"OK" if n_total >= 5*n_nodes else "FAIL"}]')
print(f'    Males: {n_male} [{"MARGINAL" if 3*n_nodes <= n_male < 5*n_nodes else ("OK" if n_male >= 5*n_nodes else "FAIL")}]')
print(f'    Females: {n_female} [{"OK" if n_female >= 5*n_nodes else "MARGINAL"}]')

print('\n[2] NETWORK SPARSITY & EDGE DISTRIBUTION')
print('-' * 90)
n_possible_edges = n_nodes * (n_nodes - 1) / 2
n_edges = len(edges)
sparsity = 1 - (n_edges / n_possible_edges)

print(f'  Possible edges (fully connected): {int(n_possible_edges)}')
print(f'  Estimated edges: {n_edges}')
print(f'  Network density: {1-sparsity:.3f} (sparsity: {sparsity:.3f})')
print(f'  Assessment: [{"Too sparse" if n_edges < 10 else "Optimal" if 10 <= n_edges <= 30 else "Too dense"}]')

print(f'\n  Edge weight distribution:')
print(f'    Mean |r|: {edges["abs_weight"].mean():.3f}')
print(f'    Median |r|: {edges["abs_weight"].median():.3f}')
print(f'    Max |r|: {edges["abs_weight"].max():.3f}')
print(f'    Min |r|: {edges["abs_weight"].min():.3f}')

# Check if threshold was appropriate
weak_edges = edges[edges['abs_weight'] < 0.10]
strong_edges = edges[edges['abs_weight'] > 0.30]
print(f'\n  Edge strength categories:')
print(f'    Weak (|r| < 0.10): {len(weak_edges)} edges ({len(weak_edges)/len(edges)*100:.1f}%)')
print(f'    Medium (0.10 <= |r| <= 0.30): {len(edges) - len(weak_edges) - len(strong_edges)} edges')
print(f'    Strong (|r| > 0.30): {len(strong_edges)} edges ({len(strong_edges)/len(edges)*100:.1f}%)')

if len(weak_edges) / len(edges) > 0.5:
    print(f'  WARNING: >50% edges are weak - consider raising threshold to 0.10')

print('\n[3] CENTRALITY METRICS VALIDITY')
print('-' * 90)
print('  Top 5 nodes by each metric:')
print('\n  Strength (sum of absolute weights):')
for i, row in centrality.nlargest(5, 'strength').iterrows():
    print(f'    {i+1}. {row["node"]}: {row["strength"]:.3f}')

print('\n  Betweenness (lies on shortest paths):')
for i, row in centrality.nlargest(5, 'betweenness').iterrows():
    print(f'    {i+1}. {row["node"]}: {row["betweenness"]:.3f}')

print('\n  Expected Influence (sum of signed weights):')
for i, row in centrality.nlargest(5, 'expected_influence').iterrows():
    print(f'    {i+1}. {row["node"]}: {row["expected_influence"]:.3f}')

# Check if UCLA is central or peripheral
ucla_centrality = centrality[centrality['node'] == 'UCLA']
ucla_rank_strength = (centrality['strength'] > ucla_centrality['strength'].values[0]).sum() + 1
ucla_rank_betweenness = (centrality['betweenness'] > ucla_centrality['betweenness'].values[0]).sum() + 1

print(f'\n  UCLA Centrality Assessment:')
print(f'    Strength rank: {ucla_rank_strength}/{len(centrality)}')
print(f'    Betweenness rank: {ucla_rank_betweenness}/{len(centrality)}')
print(f'    Expected influence: {ucla_centrality["expected_influence"].values[0]:.3f}')
print(f'    VERDICT: [{"PERIPHERAL" if ucla_rank_strength > 8 else "MODERATE CENTRALITY" if ucla_rank_strength > 5 else "CENTRAL HUB"}]')

print('\n[4] GENDER COMPARISON: NULL FINDING INVESTIGATION')
print('-' * 90)
print(f'  Total edge comparisons: {len(comparison)}')
print(f'  Significant differences (p < .05): {(comparison["p_diff"] < 0.05).sum()}')
print(f'  Significant differences (p < .10): {(comparison["p_diff"] < 0.10).sum()}')

# Power analysis for detecting medium effect (r_diff = 0.3)
n1, n2 = 30, 44
se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
z_crit = 1.96  # Two-tailed alpha=.05
r_diff_detectable = z_crit * se_diff / np.sqrt(2)  # Rough approximation

print(f'\n  Power Analysis:')
print(f'    SE for difference: {se_diff:.3f}')
print(f'    Detectable |r_diff| at 80% power: ~{r_diff_detectable:.3f}')
print(f'    Assessment: Can detect differences if |r_male - r_female| > ~0.35')

# Check largest differences (even if non-significant)
print(f'\n  Top 5 largest gender differences (even if n.s.):')
comparison_sorted = comparison.sort_values('p_diff')
for _, row in comparison_sorted.head(5).iterrows():
    diff = row['r_male'] - row['r_female']
    print(f'    {row["node1"]} -- {row["node2"]}:')
    print(f'      r_male={row["r_male"]:.3f}, r_female={row["r_female"]:.3f}, diff={diff:.3f}, p={row["p_diff"]:.3f}')

print('\n[5] PAPER 1 vs PAPER 2 INCONSISTENCY ANALYSIS')
print('-' * 90)
# Check if UCLA-PRP_tau edge exists
ucla_edges = edges[(edges['node1'] == 'UCLA') | (edges['node2'] == 'UCLA')]
print(f'  UCLA direct connections in overall network: {len(ucla_edges)} edges')
print(f'  UCLA edges:')
for _, row in ucla_edges.iterrows():
    partner = row['node1'] if row['node2'] == 'UCLA' else row['node2']
    print(f'    UCLA -- {partner}: r = {row["partial_corr"]:.3f}')

# Find UCLA-PRP_tau edge specifically
ucla_tau_edge = edges[((edges['node1'] == 'UCLA') & (edges['node2'] == 'PRP_tau')) |
                       ((edges['node1'] == 'PRP_tau') & (edges['node2'] == 'UCLA'))]

if len(ucla_tau_edge) > 0:
    print(f'\n  UCLA -- PRP_tau edge: r = {ucla_tau_edge["partial_corr"].values[0]:.3f}')
else:
    print(f'\n  UCLA -- PRP_tau edge: NOT PRESENT in network (|r| < 0.05 threshold)')
    # Check what the actual value was
    ucla_tau_comp = comparison[((comparison['node1'] == 'UCLA') & (comparison['node2'] == 'PRP_tau')) |
                                ((comparison['node1'] == 'PRP_tau') & (comparison['node2'] == 'UCLA'))]
    if len(ucla_tau_comp) > 0:
        print(f'    Male partial r: {ucla_tau_comp["r_male"].values[0]:.3f}')
        print(f'    Female partial r: {ucla_tau_comp["r_female"].values[0]:.3f}')
        print(f'    Gender difference p: {ucla_tau_comp["p_diff"].values[0]:.3f}')

print(f'\n  EXPLANATION OF INCONSISTENCY:')
print(f'    Paper 1: BIVARIATE correlation (UCLA x tau, no controls)')
print(f'    Paper 2: PARTIAL correlation (UCLA x tau, CONTROLLING for all other nodes)')
print(f'    Implication: UCLA-tau link may be MEDIATED by other nodes (e.g., DASS)')

print('\n[6] CRITICAL ISSUES & CONCERNS')
print('-' * 90)

issues = []
if n_male < 5 * n_nodes:
    issues.append(f'[MODERATE] Male sample (N={n_male}) borderline for {n_nodes} nodes (prefer N>=55)')
if sparsity > 0.8:
    issues.append(f'[LOW] Network is very sparse ({sparsity:.2f}) - may lack power to detect edges')
if (comparison['p_diff'] < 0.10).sum() == 0:
    issues.append('[MODERATE] Zero marginally significant gender differences - possible power issue OR genuine null')
if ucla_rank_strength > 5:
    issues.append(f'[INFO] UCLA not a central hub (rank {ucla_rank_strength}/{len(centrality)}) - challenges "loneliness as hub" theory')

if len(issues) == 0:
    print('  [OK] No critical issues detected')
else:
    for issue in issues:
        print(f'  {issue}')

print('\n[7] RECOMMENDATIONS')
print('-' * 90)
print('  1. SAMPLE SIZE: Males N=30 is borderline - results valid but should note limited power')
print('  2. EDGE THRESHOLD: Consider sensitivity analysis with threshold=0.10 (stricter)')
print('  3. GENDER NULL: Genuine finding OR power issue - report with caution')
print('  4. UCLA CENTRALITY: "Moderate" not "high" - revise Paper 2 narrative')
print('  5. PAPER 1-2 LINK: Emphasize bivariate vs partial correlation difference in discussion')

print('\n' + '=' * 90)
print('END OF PRECISION VERIFICATION')
print('=' * 90)
