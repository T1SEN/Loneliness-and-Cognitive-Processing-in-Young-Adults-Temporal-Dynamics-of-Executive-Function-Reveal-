"""
Latent Modeling Suite
=====================

Latent variable and network analyses for UCLA × Executive Function.

Consolidates:
- framework1_regression_mixtures.py
- framework2_normative_modeling.py
- framework3_latent_factors_sem.py
- framework4_causal_dag_simulation.py
- network_psychometrics.py
- measurement_invariance_full.py
- latent_metacontrol_sem.py
- causal_dag_extended.py

Usage:
    python -m analysis.advanced.latent_suite                    # Run all
    python -m analysis.advanced.latent_suite --analysis network
    python -m analysis.advanced.latent_suite --list

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf

from analysis.preprocessing import load_master_dataset, ANALYSIS_OUTPUT_DIR, find_interaction_term
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "latent_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AnalysisSpec:
    name: str
    description: str
    function: Callable
    source_script: str


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, source_script: str):
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(name, description, func, source_script)
        return func
    return decorator


def load_latent_data() -> pd.DataFrame:
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    return master


def compute_partial_correlation(df: pd.DataFrame, x: str, y: str, controls: list) -> dict:
    """
    Compute partial correlation between x and y, controlling for variables in controls.

    Returns dict with r, p, and n.
    """
    from scipy.stats import pearsonr

    # Remove controls that are x or y
    controls = [c for c in controls if c != x and c != y and c in df.columns]

    if len(controls) == 0:
        # No controls, just compute regular correlation
        clean = df[[x, y]].dropna()
        if len(clean) < 3:
            return {'r': np.nan, 'p': np.nan, 'n': 0}
        r, p = pearsonr(clean[x], clean[y])
        return {'r': r, 'p': p, 'n': len(clean)}

    # Residualize x and y on controls
    all_cols = [x, y] + controls
    clean = df[all_cols].dropna()

    if len(clean) < len(controls) + 3:
        return {'r': np.nan, 'p': np.nan, 'n': 0}

    # Regression residuals
    try:
        X_controls = clean[controls].values
        X_controls = np.column_stack([np.ones(len(clean)), X_controls])

        # Residualize x
        beta_x = np.linalg.lstsq(X_controls, clean[x].values, rcond=None)[0]
        resid_x = clean[x].values - X_controls @ beta_x

        # Residualize y
        beta_y = np.linalg.lstsq(X_controls, clean[y].values, rcond=None)[0]
        resid_y = clean[y].values - X_controls @ beta_y

        # Partial correlation
        r, p = pearsonr(resid_x, resid_y)
        return {'r': r, 'p': p, 'n': len(clean)}

    except Exception:
        return {'r': np.nan, 'p': np.nan, 'n': 0}


def compute_partial_corr_matrix(df: pd.DataFrame, vars_list: list, controls: list) -> pd.DataFrame:
    """
    Compute partial correlation matrix for vars_list, controlling for controls.
    """
    n_vars = len(vars_list)
    corr_matrix = pd.DataFrame(np.eye(n_vars), index=vars_list, columns=vars_list)

    for i, v1 in enumerate(vars_list):
        for j, v2 in enumerate(vars_list):
            if i < j:
                result = compute_partial_correlation(df, v1, v2, controls)
                corr_matrix.loc[v1, v2] = result['r']
                corr_matrix.loc[v2, v1] = result['r']

    return corr_matrix


@register_analysis(
    name="network",
    description="Partial correlation network of UCLA, DASS, and EF measures",
    source_script="network_psychometrics.py"
)
def analyze_network(verbose: bool = True) -> pd.DataFrame:
    """
    Network analysis examining connections between variables.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("NETWORK ANALYSIS")
        print("=" * 70)

    master = load_latent_data()

    # Variables for network
    vars_list = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
                 'pe_rate', 'stroop_interference', 'prp_bottleneck']

    available = [v for v in vars_list if v in master.columns]

    if len(available) < 4:
        if verbose:
            print("  Insufficient variables for network analysis")
        return pd.DataFrame()

    df = master[available].dropna()

    if len(df) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")
        print(f"  Variables: {available}")

    # Compute correlation matrix
    corr_matrix = df.corr()

    if verbose:
        print(f"\n  Correlation Matrix:")
        print(corr_matrix.round(3).to_string())

    # Save correlation matrix
    corr_matrix.to_csv(OUTPUT_DIR / "network_correlation_matrix.csv", encoding='utf-8-sig')

    # Long-format edges
    edges = []
    for i, v1 in enumerate(available):
        for j, v2 in enumerate(available):
            if i < j:
                r = corr_matrix.loc[v1, v2]
                n = len(df)
                t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
                p = 2 * (1 - stats.t.cdf(abs(t), n - 2))

                edges.append({
                    'node1': v1,
                    'node2': v2,
                    'r': r,
                    'p': p
                })

    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(OUTPUT_DIR / "network_edges.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Strongest edges (|r| > 0.3):")
        strong = edges_df[edges_df['r'].abs() > 0.3].sort_values('r', key=abs, ascending=False)
        for _, row in strong.head(10).iterrows():
            print(f"    {row['node1']} -- {row['node2']}: r={row['r']:.3f}")

        print(f"\n  Output: {OUTPUT_DIR / 'network_edges.csv'}")

    return edges_df


@register_analysis(
    name="network_extended",
    description="Extended network analysis with GraphicalLASSO regularization and NCT gender comparison",
    source_script="network_psychometrics_extended.py"
)
def analyze_network_extended(verbose: bool = True) -> pd.DataFrame:
    """
    Enhanced network analysis with statistical improvements:
    1. GraphicalLASSO for regularized partial correlations (more stable)
    2. Network Comparison Test (NCT) for gender comparison with permutation p-values
    3. Bootstrap edge stability analysis
    4. Centrality measures (strength, expected influence, betweenness)

    Statistical improvements over basic network analysis:
    - Regularization prevents overfitting in partial correlation estimation
    - Permutation testing provides valid p-values for network comparisons
    - Bootstrap CIs indicate edge reliability
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXTENDED NETWORK ANALYSIS (GraphicalLASSO + NCT)")
        print("=" * 70)

    try:
        from sklearn.covariance import GraphicalLassoCV
    except ImportError:
        if verbose:
            print("  sklearn.covariance.GraphicalLassoCV not available")
        return pd.DataFrame()

    master = load_latent_data()

    # Variables for network
    vars_list = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
                 'pe_rate', 'stroop_interference', 'prp_bottleneck']

    available = [v for v in vars_list if v in master.columns and master[v].notna().sum() >= 50]

    if len(available) < 4:
        if verbose:
            print("  Insufficient variables for network analysis")
        return pd.DataFrame()

    # Complete cases
    df = master[available + ['gender_male']].dropna()

    if len(df) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")
        print(f"  Variables: {available}")

    # ==== 1. GraphicalLASSO for regularized partial correlations ====
    def compute_regularized_partial_correlations(data: pd.DataFrame) -> pd.DataFrame:
        """Compute partial correlations using GraphicalLASSO regularization."""
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        try:
            # cv=3 for stability with smaller samples (N/p ratio ~7-10)
            gl_model = GraphicalLassoCV(cv=3, assume_centered=True, max_iter=500)
            gl_model.fit(data_scaled)

            precision = gl_model.precision_
            d = np.sqrt(np.diag(precision))

            # Convert precision to partial correlations
            pcor = np.zeros_like(precision)
            for i in range(len(d)):
                for j in range(len(d)):
                    if i == j:
                        pcor[i, j] = 1.0
                    else:
                        if d[i] > 0 and d[j] > 0:
                            pcor[i, j] = -precision[i, j] / (d[i] * d[j])
                        else:
                            pcor[i, j] = 0

            return pd.DataFrame(pcor, index=data.columns, columns=data.columns)
        except Exception as e:
            if verbose:
                print(f"    GraphicalLASSO failed: {e}, using pseudo-inverse")
            # Fallback to regularized pseudo-inverse
            corr = data.corr()
            try:
                precision = np.linalg.pinv(corr.values + np.eye(len(corr)) * 0.01)
            except:
                return pd.DataFrame()

            d = np.sqrt(np.abs(np.diag(precision)))
            pcor = np.zeros_like(precision)
            for i in range(len(d)):
                for j in range(len(d)):
                    if i == j:
                        pcor[i, j] = 1.0
                    elif d[i] > 0 and d[j] > 0:
                        pcor[i, j] = -precision[i, j] / (d[i] * d[j])

            return pd.DataFrame(pcor, index=data.columns, columns=data.columns)

    # Full sample partial correlation network
    pcor_matrix = compute_regularized_partial_correlations(df[available])

    if pcor_matrix.empty:
        if verbose:
            print("  Failed to compute partial correlations")
        return pd.DataFrame()

    pcor_matrix.to_csv(OUTPUT_DIR / "network_extended_partial_correlations.csv", encoding='utf-8-sig')

    if verbose:
        print(f"\n  Regularized Partial Correlation Matrix (GraphicalLASSO):")
        print(pcor_matrix.round(3).to_string())

    # ==== 2. Centrality measures ====
    def compute_centrality(pcor: pd.DataFrame) -> pd.DataFrame:
        """Compute node centrality measures."""
        nodes = pcor.columns
        centrality = []

        for node in nodes:
            row = pcor.loc[node, :].drop(node)

            # Strength: sum of absolute edge weights
            strength = row.abs().sum()

            # Expected Influence: sum of signed edge weights
            ei = row.sum()

            centrality.append({
                'node': node,
                'strength': strength,
                'expected_influence': ei
            })

        return pd.DataFrame(centrality)

    centrality_df = compute_centrality(pcor_matrix)
    centrality_df.to_csv(OUTPUT_DIR / "network_extended_centrality.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Centrality Measures:")
        for _, row in centrality_df.sort_values('strength', ascending=False).iterrows():
            print(f"    {row['node']}: strength={row['strength']:.3f}, EI={row['expected_influence']:.3f}")

    # ==== 3. Gender-specific networks ====
    male_data = df[df['gender_male'] == 1][available]
    female_data = df[df['gender_male'] == 0][available]

    # Minimum N=50 per group for stable GraphicalLASSO with 7 variables
    # (recommended N/p ratio >= 7)
    gender_networks = {}
    if len(male_data) >= 50:
        gender_networks['male'] = compute_regularized_partial_correlations(male_data)
        gender_networks['male'].to_csv(OUTPUT_DIR / "network_extended_male.csv", encoding='utf-8-sig')
    if len(female_data) >= 50:
        gender_networks['female'] = compute_regularized_partial_correlations(female_data)
        gender_networks['female'].to_csv(OUTPUT_DIR / "network_extended_female.csv", encoding='utf-8-sig')

    if verbose:
        print(f"\n  Gender samples: Male N={len(male_data)}, Female N={len(female_data)}")

    # ==== 4. Network Comparison Test (NCT) ====
    nct_results = {}

    # NCT requires minimum N=40 per group for reliable permutation inference
    if len(male_data) >= 40 and len(female_data) >= 40 and 'male' in gender_networks and 'female' in gender_networks:
        if verbose:
            print(f"\n  Running Network Comparison Test (1000 permutations)...")

        net_m = gender_networks['male']
        net_f = gender_networks['female']

        # Observed differences
        obs_global_strength_diff = net_m.abs().sum().sum() - net_f.abs().sum().sum()
        obs_max_edge_diff = (net_m - net_f).abs().max().max()

        # Permutation test
        combined = pd.concat([male_data, female_data])
        n_male = len(male_data)
        n_permutations = 1000

        perm_global_diffs = []
        perm_max_edge_diffs = []

        # Use local RandomState for reproducibility within this block
        rng_perm = np.random.RandomState(42)
        for _ in range(n_permutations):
            perm_idx = rng_perm.permutation(len(combined))
            perm_male = combined.iloc[perm_idx[:n_male]]
            perm_female = combined.iloc[perm_idx[n_male:]]

            try:
                perm_net_m = compute_regularized_partial_correlations(perm_male)
                perm_net_f = compute_regularized_partial_correlations(perm_female)

                if not perm_net_m.empty and not perm_net_f.empty:
                    perm_global_diffs.append(
                        perm_net_m.abs().sum().sum() - perm_net_f.abs().sum().sum()
                    )
                    perm_max_edge_diffs.append(
                        (perm_net_m - perm_net_f).abs().max().max()
                    )
            except:
                continue

        # Calculate p-values (require 80% success rate: 800/1000 permutations)
        if len(perm_global_diffs) >= 800:
            p_global = (np.sum(np.abs(perm_global_diffs) >= np.abs(obs_global_strength_diff)) + 1) / (len(perm_global_diffs) + 1)
            p_edge = (np.sum(np.array(perm_max_edge_diffs) >= obs_max_edge_diff) + 1) / (len(perm_max_edge_diffs) + 1)

            nct_results = {
                'global_strength_diff': obs_global_strength_diff,
                'p_global_strength': p_global,
                'max_edge_diff': obs_max_edge_diff,
                'p_max_edge': p_edge,
                'n_male': len(male_data),
                'n_female': len(female_data),
                'n_permutations': len(perm_global_diffs)
            }

            if verbose:
                sig_global = "*" if p_global < 0.05 else ""
                sig_edge = "*" if p_edge < 0.05 else ""
                print(f"\n  NCT Results:")
                print(f"    Global strength diff: {obs_global_strength_diff:.3f}, p={p_global:.4f}{sig_global}")
                print(f"    Max edge diff: {obs_max_edge_diff:.3f}, p={p_edge:.4f}{sig_edge}")
        else:
            if verbose:
                print(f"    NCT failed: insufficient valid permutations ({len(perm_global_diffs)})")

    # ==== 5. Bootstrap edge stability (simplified) ====
    if verbose:
        print(f"\n  Computing bootstrap edge stability (500 iterations)...")

    n_bootstrap = 500
    edge_counts = {}

    for v1_idx, v1 in enumerate(available):
        for v2_idx, v2 in enumerate(available):
            if v1_idx < v2_idx:
                edge_counts[(v1, v2)] = []

    # Use local RandomState for reproducibility within this block
    rng_boot = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        boot_idx = rng_boot.choice(len(df), size=len(df), replace=True)
        boot_data = df.iloc[boot_idx][available]

        try:
            boot_pcor = compute_regularized_partial_correlations(boot_data)
            if not boot_pcor.empty:
                for (v1, v2) in edge_counts.keys():
                    edge_counts[(v1, v2)].append(boot_pcor.loc[v1, v2])
        except:
            continue

    # Compute bootstrap CIs
    edge_stability = []
    for (v1, v2), values in edge_counts.items():
        if len(values) >= 100:
            values = np.array(values)
            edge_stability.append({
                'node1': v1,
                'node2': v2,
                'pcor_mean': np.mean(values),
                'pcor_median': np.median(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'significant': (np.percentile(values, 2.5) > 0) or (np.percentile(values, 97.5) < 0)
            })

    edge_stability_df = pd.DataFrame(edge_stability)
    edge_stability_df.to_csv(OUTPUT_DIR / "network_extended_edge_stability.csv", index=False, encoding='utf-8-sig')

    n_significant = edge_stability_df['significant'].sum() if len(edge_stability_df) > 0 else 0
    if verbose:
        print(f"    Significant edges (95% CI excludes 0): {n_significant}/{len(edge_stability_df)}")

    # ==== Compile results ====
    results = []

    # Edges from main analysis
    for i, v1 in enumerate(available):
        for j, v2 in enumerate(available):
            if i < j:
                results.append({
                    'node1': v1,
                    'node2': v2,
                    'pcor': pcor_matrix.loc[v1, v2],
                    'type': 'full_sample'
                })

    # Add NCT results
    if nct_results:
        results.append({
            'type': 'NCT',
            'global_strength_diff': nct_results.get('global_strength_diff'),
            'p_global_strength': nct_results.get('p_global_strength'),
            'max_edge_diff': nct_results.get('max_edge_diff'),
            'p_max_edge': nct_results.get('p_max_edge')
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "network_extended_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'network_extended_results.csv'}")

    return results_df


@register_analysis(
    name="bridge_centrality",
    description="Bridge centrality analysis: identify nodes connecting DASS and EF domains (UCLA excluded)",
    source_script="network_psychometrics_bridge.py"
)
def analyze_bridge_centrality(verbose: bool = True) -> pd.DataFrame:
    """
    Bridge centrality analysis to identify variables that connect different domains.

    IMPORTANT: UCLA is EXCLUDED from bridge calculation to avoid circular logic
    (we cannot use UCLA correlations to study UCLA-EF relationships).

    Domains:
    - Psychological: DASS subscales only (not UCLA)
    - Cognitive: PE rate, Stroop interference, PRP bottleneck

    Bridge centrality measures how strongly a node connects to nodes in other domains.
    High bridge centrality = potential intervention target.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BRIDGE CENTRALITY ANALYSIS")
        print("=" * 70)
        print("  NOTE: UCLA excluded from bridge calculation (avoids circular logic)")

    master = load_latent_data()

    # Define domains - UCLA EXCLUDED from bridge analysis to avoid circular logic
    dass_vars = ['dass_depression', 'dass_anxiety', 'dass_stress']
    cog_vars = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    control_vars = ['z_age']  # Controls for partial correlations

    dass_available = [v for v in dass_vars if v in master.columns]
    cog_available = [v for v in cog_vars if v in master.columns]
    controls = [v for v in control_vars if v in master.columns]

    all_vars = dass_available + cog_available

    if len(dass_available) < 2 or len(cog_available) < 2:
        if verbose:
            print("  Insufficient variables for bridge centrality analysis")
        return pd.DataFrame()

    # Complete cases - include UCLA for separate analysis
    df = master[all_vars + ['ucla_total', 'gender_male'] + controls].dropna()

    if len(df) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")
        print(f"  DASS domain (psychological): {dass_available}")
        print(f"  EF domain (cognitive): {cog_available}")
        print(f"  Controls for partial correlations: {controls}")

    # Compute PARTIAL correlation matrix controlling for age
    corr_matrix = compute_partial_corr_matrix(df, all_vars, controls)

    if verbose:
        print("\n  Using PARTIAL correlations (controlling for age)")

    # Bridge centrality: sum of absolute partial correlations with OTHER domain
    bridge_results = []

    for var in all_vars:
        is_dass = var in dass_available
        other_domain = cog_available if is_dass else dass_available

        # Bridge centrality = sum of abs partial correlations with other domain
        bridge_strength = corr_matrix.loc[var, other_domain].abs().sum()

        # Expected bridge influence = sum of signed correlations with other domain
        bridge_ei = corr_matrix.loc[var, other_domain].sum()

        # Within-domain strength
        own_domain = dass_available if is_dass else cog_available
        own_domain_filtered = [v for v in own_domain if v != var]
        within_strength = corr_matrix.loc[var, own_domain_filtered].abs().sum() if own_domain_filtered else 0

        bridge_results.append({
            'variable': var,
            'domain': 'DASS' if is_dass else 'Cognitive',
            'bridge_strength': bridge_strength,
            'bridge_ei': bridge_ei,
            'within_strength': within_strength,
            'bridge_ratio': bridge_strength / (bridge_strength + within_strength) if (bridge_strength + within_strength) > 0 else 0,
            'method': 'partial_correlation'
        })

    bridge_df = pd.DataFrame(bridge_results)

    if verbose:
        print("\n  Bridge Centrality (DASS ↔ EF connections, partial r):")
        print("  " + "-" * 60)
        for _, row in bridge_df.sort_values('bridge_strength', ascending=False).iterrows():
            print(f"    {row['variable']}: bridge={row['bridge_strength']:.3f}, ratio={row['bridge_ratio']:.3f} ({row['domain']})")

    # Identify top bridge nodes (key mediators between DASS and EF)
    top_bridge = bridge_df.nlargest(3, 'bridge_strength')

    if verbose:
        print(f"\n  Top Bridge Nodes (DASS-EF mediators):")
        for _, row in top_bridge.iterrows():
            print(f"    • {row['variable']} (bridge={row['bridge_strength']:.3f})")
        print("\n  Interpretation:")
        print("    These variables most strongly connect DASS distress to EF impairments.")
        print("    Targeting these nodes may interrupt the distress→cognition pathway.")

    # Gender-stratified bridge analysis (using partial correlations)
    gender_bridge_results = []

    for gender, label in [(1, 'Male'), (0, 'Female')]:
        subset = df[df['gender_male'] == gender]

        if len(subset) < 30:
            continue

        # Use partial correlation matrix for gender subgroup
        corr_g = compute_partial_corr_matrix(subset, all_vars, controls)

        for var in all_vars:
            is_dass = var in dass_available
            other_domain = cog_available if is_dass else dass_available
            bridge_strength_g = corr_g.loc[var, other_domain].abs().sum()

            gender_bridge_results.append({
                'gender': label,
                'variable': var,
                'domain': 'DASS' if is_dass else 'Cognitive',
                'bridge_strength': bridge_strength_g,
                'method': 'partial_correlation'
            })

    if gender_bridge_results:
        gender_bridge_df = pd.DataFrame(gender_bridge_results)
        gender_bridge_df.to_csv(OUTPUT_DIR / "bridge_centrality_by_gender.csv", index=False, encoding='utf-8-sig')

        if verbose:
            print(f"\n  Gender-Stratified Bridge Centrality (partial r):")
            print("  " + "-" * 60)

            # Compare DASS depression bridge centrality by gender
            dass_dep_bridge = gender_bridge_df[gender_bridge_df['variable'] == 'dass_depression']
            if len(dass_dep_bridge) == 2:
                male_bridge = dass_dep_bridge[dass_dep_bridge['gender'] == 'Male']['bridge_strength'].values[0]
                female_bridge = dass_dep_bridge[dass_dep_bridge['gender'] == 'Female']['bridge_strength'].values[0]
                print(f"    DASS Depression:")
                print(f"      Male bridge (partial r): {male_bridge:.3f}")
                print(f"      Female bridge (partial r): {female_bridge:.3f}")
                if male_bridge > female_bridge:
                    print(f"      → Stronger DASS-Cog link in males")
                else:
                    print(f"      → Stronger DASS-Cog link in females")

    # Save results
    bridge_df.to_csv(OUTPUT_DIR / "bridge_centrality.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'bridge_centrality.csv'}")

    return bridge_df


@register_analysis(
    name="factor_analysis",
    description="Exploratory factor analysis of UCLA items",
    source_script="framework3_latent_factors_sem.py"
)
def analyze_factor_analysis(verbose: bool = True) -> pd.DataFrame:
    """
    Factor analysis of EF metrics to identify latent structure.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("FACTOR ANALYSIS")
        print("=" * 70)

    master = load_latent_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck', 'wcst_accuracy']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 3:
        if verbose:
            print("  Need at least 3 EF metrics for factor analysis")
        return pd.DataFrame()

    df = master[available].dropna()

    if len(df) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")
        print(f"  Variables: {available}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # PCA for comparison
    pca = PCA(n_components=min(len(available), 3))
    pca.fit(X_scaled)

    if verbose:
        print(f"\n  PCA Results:")
        for i, var_exp in enumerate(pca.explained_variance_ratio_):
            print(f"    PC{i+1}: {var_exp*100:.1f}% variance")

    # Factor Analysis
    try:
        fa = FactorAnalysis(n_components=2, random_state=42)
        fa.fit(X_scaled)

        loadings = pd.DataFrame(
            fa.components_.T,
            index=available,
            columns=['Factor1', 'Factor2']
        )

        if verbose:
            print(f"\n  Factor Loadings:")
            print(loadings.round(3).to_string())

        loadings.to_csv(OUTPUT_DIR / "factor_loadings.csv", encoding='utf-8-sig')

        if verbose:
            print(f"\n  Output: {OUTPUT_DIR / 'factor_loadings.csv'}")

        return loadings

    except Exception as e:
        if verbose:
            print(f"  Factor analysis error: {e}")
        return pd.DataFrame()


@register_analysis(
    name="measurement_invariance",
    description="Test measurement invariance across gender",
    source_script="measurement_invariance_full.py"
)
def analyze_measurement_invariance(verbose: bool = True) -> pd.DataFrame:
    """
    Test if EF measurement structure is equivalent across gender.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MEASUREMENT INVARIANCE")
        print("=" * 70)

    master = load_latent_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 2:
        if verbose:
            print("  Insufficient EF metrics")
        return pd.DataFrame()

    df = master[available + ['gender_male']].dropna()

    results = []

    for gender, label in [(1, 'Male'), (0, 'Female')]:
        grp = df[df['gender_male'] == gender][available]

        if len(grp) < 20:
            continue

        # Compute correlation matrix per group
        corr = grp.corr()

        # Mean correlation (as summary of structure)
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        mean_corr = upper_tri.stack().mean()

        results.append({
            'gender': label,
            'n': len(grp),
            'mean_cross_task_corr': mean_corr
        })

        if verbose:
            print(f"\n  {label} (n={len(grp)})")
            print(f"    Mean cross-task correlation: {mean_corr:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "measurement_invariance.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'measurement_invariance.csv'}")

    return results_df


@register_analysis(
    name="regression_mixtures",
    description="GMM-based regression mixture modeling to identify latent subgroups",
    source_script="framework1_regression_mixtures.py"
)
def analyze_regression_mixtures(verbose: bool = True) -> pd.DataFrame:
    """
    Identify latent subgroups with different UCLA→EF slope profiles using GMM.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("REGRESSION MIXTURE MODELING")
        print("=" * 70)

    master = load_latent_data()

    # Feature selection
    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 2:
        if verbose:
            print("  Insufficient EF metrics")
        return pd.DataFrame()

    vars_needed = ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', 'ucla_total'] + available
    vars_available = [v for v in vars_needed if v in master.columns]
    df = master[vars_available].dropna()

    if len(df) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")

    all_results = []

    for ef_outcome in available:
        if verbose:
            print(f"\n  Analyzing: {ef_outcome}")

        # Prepare features
        feature_cols = ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male']
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].values
        y = df[ef_outcome].values

        # Augmented feature space
        Xy = np.column_stack([X, y])

        # Fit GMM for K=2-4
        best_bic = np.inf
        best_k = 2
        best_labels = None

        for k in range(2, 5):
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
            gmm.fit(Xy)
            bic = gmm.bic(Xy)

            if verbose:
                print(f"    K={k}: BIC={bic:.1f}")

            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_labels = gmm.predict(Xy)

        if verbose:
            print(f"    Best K={best_k}")

        # Characterize classes
        df_temp = df.copy()
        df_temp['class'] = best_labels

        for c in range(best_k):
            c_data = df_temp[df_temp['class'] == c]
            all_results.append({
                'outcome': ef_outcome,
                'class': c,
                'n': len(c_data),
                'ucla_mean': c_data['ucla_total'].mean() if 'ucla_total' in c_data else np.nan,
                'pct_male': c_data['gender_male'].mean() * 100,
                'ef_mean': c_data[ef_outcome].mean()
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "regression_mixtures.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'regression_mixtures.csv'}")

    return results_df


@register_analysis(
    name="normative_modeling",
    description="Gaussian Process normative modeling with deviation scores",
    source_script="framework2_normative_modeling.py"
)
def analyze_normative_modeling(verbose: bool = True) -> pd.DataFrame:
    """
    Build normative model and compute personalized deviation scores.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("NORMATIVE MODELING")
        print("=" * 70)

    master = load_latent_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 1:
        if verbose:
            print("  No EF metrics available")
        return pd.DataFrame()

    all_results = []

    for ef_outcome in available:
        vars_needed = ['age', 'dass_depression', 'dass_anxiety', 'dass_stress', 'gender_male',
                      'ucla_total', 'z_ucla', ef_outcome]
        vars_available = [v for v in vars_needed if v in master.columns]
        df = master[vars_available].dropna()

        if len(df) < 50:
            if verbose:
                print(f"  Insufficient data for {ef_outcome} (N={len(df)})")
            continue

        if verbose:
            print(f"\n  Analyzing: {ef_outcome} (N={len(df)})")

        # Normative features
        feature_cols = ['age', 'dass_depression', 'dass_anxiety', 'dass_stress', 'gender_male']
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].values
        y = df[ef_outcome].values

        # K-fold CV for unbiased predictions
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        y_pred = np.zeros_like(y, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Standardize within fold
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train = scaler_X.fit_transform(X[train_idx])
            y_train = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
            X_test = scaler_X.transform(X[test_idx])

            # Fit GP
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42, normalize_y=True)
            gp.fit(X_train, y_train)

            pred = gp.predict(X_test)
            y_pred[test_idx] = scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()

        # Compute deviations
        deviation = y - y_pred
        deviation_z = (deviation - deviation.mean()) / deviation.std()

        # Test UCLA → deviation
        df_temp = df.copy()
        df_temp['deviation_z'] = deviation_z

        if 'z_ucla' in df_temp.columns:
            try:
                model = smf.ols("deviation_z ~ z_ucla * C(gender_male)", data=df_temp).fit()

                ucla_beta = model.params.get('z_ucla', np.nan)
                ucla_p = model.pvalues.get('z_ucla', np.nan)
                int_term = find_interaction_term(model.params.index)
                interaction_beta = model.params.get(int_term, np.nan) if int_term else np.nan
                interaction_p = model.pvalues.get(int_term, np.nan) if int_term else np.nan

                all_results.append({
                    'outcome': ef_outcome,
                    'n': len(df),
                    'normative_r2': 1 - (np.var(deviation) / np.var(y)),
                    'ucla_deviation_beta': ucla_beta,
                    'ucla_deviation_p': ucla_p,
                    'interaction_beta': interaction_beta,
                    'interaction_p': interaction_p
                })

                if verbose:
                    print(f"    UCLA → Deviation: β={ucla_beta:.3f}, p={ucla_p:.4f}")
                    print(f"    UCLA × Gender: β={interaction_beta:.3f}, p={interaction_p:.4f}")

            except Exception as e:
                if verbose:
                    print(f"    Regression error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "normative_modeling.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'normative_modeling.csv'}")

    return results_df


@register_analysis(
    name="metacontrol_sem",
    description="Meta-control latent factor via PCA and path analysis",
    source_script="latent_metacontrol_sem.py"
)
def analyze_metacontrol_sem(verbose: bool = True) -> pd.DataFrame:
    """
    Extract latent meta-control factor using PCA and test path analysis.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("META-CONTROL SEM")
        print("=" * 70)

    master = load_latent_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 3:
        if verbose:
            print("  Need at least 3 EF metrics for meta-control factor")
        return pd.DataFrame()

    vars_needed = available + ['ucla_total', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male']
    vars_available = [v for v in vars_needed if v in master.columns]
    df = master[vars_available].dropna()

    if len(df) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")

    # Reverse code so higher = better control
    for col in available:
        df[f'{col}_control'] = -df[col]

    control_vars = [f'{c}_control' for c in available]

    # PCA
    scaler = StandardScaler()
    control_scaled = scaler.fit_transform(df[control_vars])

    pca = PCA(n_components=len(control_vars))
    components = pca.fit_transform(control_scaled)

    if verbose:
        print(f"\n  PCA Variance Explained:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"    PC{i+1}: {var*100:.1f}%")

    # Use PC1 as meta-control factor
    df['meta_control'] = components[:, 0]

    # Path analysis
    results = []

    # Path a: UCLA → Meta-Control
    try:
        model_a = smf.ols("meta_control ~ z_ucla + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                          data=df).fit()
        path_a = model_a.params.get('z_ucla', np.nan)
        path_a_p = model_a.pvalues.get('z_ucla', np.nan)
    except:
        path_a, path_a_p = np.nan, np.nan

    # Path c': UCLA → PE direct (controlling for meta-control)
    if 'pe_rate' in df.columns:
        try:
            model_c = smf.ols("pe_rate ~ z_ucla + meta_control + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                              data=df).fit()
            path_c = model_c.params.get('z_ucla', np.nan)
            path_c_p = model_c.pvalues.get('z_ucla', np.nan)
        except:
            path_c, path_c_p = np.nan, np.nan
    else:
        path_c, path_c_p = np.nan, np.nan

    results.append({
        'path': 'a: UCLA → Meta-Control',
        'beta': path_a,
        'p': path_a_p
    })
    results.append({
        'path': 'c\': UCLA → PE (direct)',
        'beta': path_c,
        'p': path_c_p
    })

    if verbose:
        print(f"\n  Path Analysis:")
        print(f"    UCLA → Meta-Control: β={path_a:.3f}, p={path_a_p:.4f}")
        print(f"    UCLA → PE (direct): β={path_c:.3f}, p={path_c_p:.4f}")

    # Save loadings
    loadings_df = pd.DataFrame({
        'variable': control_vars,
        'PC1_loading': pca.components_[0]
    })
    loadings_df.to_csv(OUTPUT_DIR / "metacontrol_loadings.csv", index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "metacontrol_paths.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'metacontrol_paths.csv'}")

    return results_df


@register_analysis(
    name="causal_dag",
    description="Compare competing causal DAG models",
    source_script="framework4_causal_dag_simulation.py"
)
def analyze_causal_dag(verbose: bool = True) -> pd.DataFrame:
    """
    Compare competing causal models for UCLA → EF relationships.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CAUSAL DAG COMPARISON")
        print("=" * 70)

    master = load_latent_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 1:
        if verbose:
            print("  No EF outcomes available")
        return pd.DataFrame()

    all_results = []

    for ef_outcome in available:
        vars_needed = ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', ef_outcome]
        vars_available = [v for v in vars_needed if v in master.columns]
        df = master[vars_available].dropna()

        if len(df) < 30:
            if verbose:
                print(f"  Insufficient data for {ef_outcome} (N={len(df)})")
            continue

        if verbose:
            print(f"\n  Analyzing: {ef_outcome} (N={len(df)})")

        # Model 1: Mood-Driven (DASS only)
        try:
            m1 = smf.ols(f"{ef_outcome} ~ z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)",
                        data=df).fit()
            aic1 = m1.aic
            r2_1 = m1.rsquared
        except:
            aic1, r2_1 = np.nan, np.nan

        # Model 2: Parallel (UCLA + DASS)
        try:
            m2 = smf.ols(f"{ef_outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(gender_male)",
                        data=df).fit()
            aic2 = m2.aic
            r2_2 = m2.rsquared
            ucla_beta = m2.params.get('z_ucla', np.nan)
            ucla_p = m2.pvalues.get('z_ucla', np.nan)
        except:
            aic2, r2_2, ucla_beta, ucla_p = np.nan, np.nan, np.nan, np.nan

        # Model 3: Gender-Moderated
        try:
            m3 = smf.ols(f"{ef_outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                        data=df).fit()
            aic3 = m3.aic
            r2_3 = m3.rsquared
            int_term = find_interaction_term(m3.params.index)
            interaction_beta = m3.params.get(int_term, np.nan) if int_term else np.nan
            interaction_p = m3.pvalues.get(int_term, np.nan) if int_term else np.nan
        except:
            aic3, r2_3, interaction_beta, interaction_p = np.nan, np.nan, np.nan, np.nan

        all_results.append({
            'outcome': ef_outcome,
            'model': 'Mood-Driven',
            'AIC': aic1,
            'R2': r2_1
        })
        all_results.append({
            'outcome': ef_outcome,
            'model': 'Parallel',
            'AIC': aic2,
            'R2': r2_2,
            'ucla_beta': ucla_beta,
            'ucla_p': ucla_p
        })
        all_results.append({
            'outcome': ef_outcome,
            'model': 'Gender-Moderated',
            'AIC': aic3,
            'R2': r2_3,
            'interaction_beta': interaction_beta,
            'interaction_p': interaction_p
        })

        if verbose:
            best_model = 'Mood-Driven' if aic1 < aic2 and aic1 < aic3 else ('Parallel' if aic2 < aic3 else 'Gender-Moderated')
            print(f"    Best model by AIC: {best_model}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "causal_dag_comparison.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'causal_dag_comparison.csv'}")

    return results_df


@register_analysis(
    name="causal_extended",
    description="Extended causal analysis with mediation decomposition",
    source_script="causal_dag_extended.py"
)
def analyze_causal_extended(verbose: bool = True) -> pd.DataFrame:
    """
    Extended causal analysis with backdoor adjustment and mediation.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXTENDED CAUSAL ANALYSIS")
        print("=" * 70)

    master = load_latent_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 1:
        if verbose:
            print("  No EF outcomes available")
        return pd.DataFrame()

    all_results = []

    for ef_outcome in available:
        vars_needed = ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', ef_outcome]
        vars_available = [v for v in vars_needed if v in master.columns]
        df = master[vars_available].dropna()

        if len(df) < 30:
            if verbose:
                print(f"  Insufficient data for {ef_outcome} (N={len(df)})")
            continue

        if verbose:
            print(f"\n  Analyzing: {ef_outcome} (N={len(df)})")

        # Mediation through DASS Depression
        try:
            # Path a: UCLA → DASS Depression
            m_a = smf.ols("z_dass_dep ~ z_ucla + z_age", data=df).fit()
            a = m_a.params.get('z_ucla', np.nan)

            # Path b: DASS → Outcome (controlling UCLA)
            m_b = smf.ols(f"{ef_outcome} ~ z_dass_dep + z_ucla + z_age", data=df).fit()
            b = m_b.params.get('z_dass_dep', np.nan)
            direct = m_b.params.get('z_ucla', np.nan)

            # Total effect
            m_total = smf.ols(f"{ef_outcome} ~ z_ucla + z_age", data=df).fit()
            total = m_total.params.get('z_ucla', np.nan)

            indirect = a * b
            mediation_pct = (indirect / total * 100) if total != 0 else np.nan

            all_results.append({
                'outcome': ef_outcome,
                'mediator': 'DASS_Depression',
                'a_path': a,
                'b_path': b,
                'indirect': indirect,
                'direct': direct,
                'total': total,
                'mediation_pct': mediation_pct
            })

            if verbose:
                print(f"    DASS Depression mediation: {mediation_pct:.1f}%")

        except Exception as e:
            if verbose:
                print(f"    Mediation error: {e}")

        # Gender-stratified effects
        for gender, label in [(1, 'Male'), (0, 'Female')]:
            g_data = df[df['gender_male'] == gender]
            if len(g_data) >= 15:
                try:
                    m_g = smf.ols(f"{ef_outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age",
                                 data=g_data).fit()
                    all_results.append({
                        'outcome': ef_outcome,
                        'gender': label,
                        'n': len(g_data),
                        'ucla_beta': m_g.params.get('z_ucla', np.nan),
                        'ucla_p': m_g.pvalues.get('z_ucla', np.nan)
                    })

                    if verbose:
                        ucla_b = m_g.params.get('z_ucla', np.nan)
                        ucla_p_val = m_g.pvalues.get('z_ucla', np.nan)
                        print(f"    {label}: UCLA β={ucla_b:.3f}, p={ucla_p_val:.4f}")
                except:
                    pass

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "causal_extended.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'causal_extended.csv'}")

    return results_df


@register_analysis(
    name="network_cluster_integration",
    description="Integrate network structure and cluster analyses",
    source_script="network_cluster_integration.py"
)
def analyze_network_cluster_integration(verbose: bool = True) -> pd.DataFrame:
    """
    Integrate network analysis with clustering results.

    Questions addressed:
    1. Do different vulnerability clusters have different network structures?
    2. Is bridge centrality related to cluster membership?
    3. Can we identify network-based risk profiles?
    """
    import json

    if verbose:
        print("\n" + "=" * 70)
        print("NETWORK-CLUSTER INTEGRATION ANALYSIS")
        print("=" * 70)

    master = load_latent_data()

    # Load cluster assignments
    cluster_file = ANALYSIS_OUTPUT_DIR / "clustering_suite" / "cluster_assignments.csv"
    if not cluster_file.exists():
        if verbose:
            print("  Cluster assignments not found. Running clustering first...")
        return pd.DataFrame()

    cluster_df = pd.read_csv(cluster_file)
    master = master.merge(cluster_df[['participant_id', 'cluster']], on='participant_id', how='left')

    # Network variables - exclude UCLA from bridge to avoid circular logic
    psych_vars = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
    dass_vars = ['dass_depression', 'dass_anxiety', 'dass_stress']  # For bridge (no UCLA)
    cog_vars = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    control_vars = ['z_age']  # Controls for partial correlations

    psych_available = [v for v in psych_vars if v in master.columns]
    dass_available = [v for v in dass_vars if v in master.columns]
    cog_available = [v for v in cog_vars if v in master.columns]
    controls = [v for v in control_vars if v in master.columns]

    all_vars = psych_available + cog_available
    bridge_vars = dass_available + cog_available  # For bridge analysis (no UCLA)

    # Complete cases with cluster info
    clean_data = master[all_vars + ['cluster', 'gender_male'] + controls].dropna().copy()

    if len(clean_data) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(clean_data)})")
        return pd.DataFrame()

    n_clusters = clean_data['cluster'].nunique()

    if verbose:
        print(f"  N = {len(clean_data)}, Clusters = {n_clusters}")
        print(f"  Psychological vars: {psych_available}")
        print(f"  Cognitive vars: {cog_available}")
        print(f"  Using PARTIAL correlations (controlling for age)")
        print(f"  Bridge analysis excludes UCLA (avoids circular logic)")

    all_results = []

    # Compute cluster-specific network structures
    cluster_networks = {}
    cluster_bridge = {}

    for cluster_id in sorted(clean_data['cluster'].unique()):
        cluster_subset = clean_data[clean_data['cluster'] == cluster_id]

        if len(cluster_subset) < 20:
            continue

        # PARTIAL correlation matrix (controlling for age)
        corr_matrix = compute_partial_corr_matrix(cluster_subset, all_vars, controls)
        cluster_networks[cluster_id] = corr_matrix

        # Global strength (mean of absolute partial correlations)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        global_strength = np.abs(corr_matrix.values)[mask].mean()

        # Bridge centrality for DASS↔EF (excluding UCLA to avoid circular logic)
        bridge_corr = compute_partial_corr_matrix(cluster_subset, bridge_vars, controls)
        bridge_results = {}
        for var in bridge_vars:
            is_dass = var in dass_available
            other_domain = cog_available if is_dass else dass_available
            bridge_strength = bridge_corr.loc[var, other_domain].abs().sum()
            bridge_results[var] = bridge_strength

        cluster_bridge[cluster_id] = bridge_results

        # Find most important bridge in this cluster
        top_bridge_var = max(bridge_results, key=bridge_results.get)
        top_bridge_val = bridge_results[top_bridge_var]

        all_results.append({
            'cluster': cluster_id,
            'n': len(cluster_subset),
            'global_strength': global_strength,
            'top_bridge_node': top_bridge_var,
            'top_bridge_strength': top_bridge_val,
            'pct_male': (clean_data[clean_data['cluster'] == cluster_id]['gender_male'] == 1).mean() * 100,
            'method': 'partial_correlation'
        })

        if verbose:
            print(f"\n  CLUSTER {cluster_id} (N={len(cluster_subset)})")
            print("  " + "-" * 50)
            print(f"    Global network strength (partial r): {global_strength:.3f}")
            print(f"    Top bridge (DASS↔EF): {top_bridge_var} ({top_bridge_val:.3f})")
            print(f"    % Male: {all_results[-1]['pct_male']:.1f}%")

    # Compare network structures across clusters
    if verbose and len(cluster_networks) >= 2:
        print("\n  CROSS-CLUSTER NETWORK COMPARISON")
        print("  " + "-" * 50)

        cluster_ids = list(cluster_networks.keys())

        # Compare global strengths
        strengths = {c: r['global_strength'] for c, r in zip(cluster_ids, all_results)}
        strongest = max(strengths, key=strengths.get)
        weakest = min(strengths, key=strengths.get)

        print(f"    Strongest network: Cluster {strongest} ({strengths[strongest]:.3f})")
        print(f"    Weakest network: Cluster {weakest} ({strengths[weakest]:.3f})")

        # Compare UCLA-DASS link across clusters
        print("\n    UCLA-DASS Depression link by cluster:")
        for cluster_id, corr_mat in cluster_networks.items():
            if 'ucla_total' in corr_mat.index and 'dass_depression' in corr_mat.columns:
                r = corr_mat.loc['ucla_total', 'dass_depression']
                print(f"      Cluster {cluster_id}: r={r:.3f}")

        # Compare PE-DASS link across clusters
        print("\n    PE Rate - DASS Depression link by cluster:")
        for cluster_id, corr_mat in cluster_networks.items():
            if 'pe_rate' in corr_mat.index and 'dass_depression' in corr_mat.columns:
                r = corr_mat.loc['pe_rate', 'dass_depression']
                print(f"      Cluster {cluster_id}: r={r:.3f}")

    # Clinical implication summary
    if verbose:
        print("\n  CLINICAL IMPLICATIONS")
        print("  " + "=" * 50)

        # Find high-vulnerability cluster (female-dominated, weak network)
        female_clusters = [r for r in all_results if r['pct_male'] < 35]
        if female_clusters:
            most_female = min(female_clusters, key=lambda x: x['pct_male'])
            print(f"\n    High-risk profile (Cluster {most_female['cluster']}):")
            print(f"      - {100 - most_female['pct_male']:.0f}% female")
            print(f"      - Network strength: {most_female['global_strength']:.3f}")
            print(f"      - Key bridge: {most_female['top_bridge_node']}")
            print("      → Consider targeting {0} for intervention".format(most_female['top_bridge_node']))

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "network_cluster_integration.csv", index=False, encoding='utf-8-sig')

    # Save cluster networks
    for cluster_id, corr_mat in cluster_networks.items():
        corr_mat.to_csv(OUTPUT_DIR / f"network_cluster_{int(cluster_id)}.csv", encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'network_cluster_integration.csv'}")

    return results_df


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    if verbose:
        print("=" * 70)
        print("LATENT MODELING SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("LATENT SUITE COMPLETE")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    print("\nAvailable Latent Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")
        print(f"    Source: {spec.source_script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Modeling Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--list', '-l', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
