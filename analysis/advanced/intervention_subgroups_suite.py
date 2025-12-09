"""
Intervention Subgroups Analysis Suite
=====================================

Identifies high-risk subgroups for potential intervention targeting.

Methods:
1. K-means clustering on psychological-cognitive profile
2. Latent Profile Analysis (LPA) approximation via GMM
3. Risk decision tree for clinical identification
4. Subgroup profiling and characteristics

Research Questions:
- Can we identify distinct subgroups based on UCLA, DASS, and EF?
- Which subgroup shows the strongest UCLA-EF relationship?
- What profile characteristics predict poor EF outcomes?

Usage:
    python -m analysis.advanced.intervention_subgroups_suite

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
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import silhouette_score, adjusted_rand_score
import statsmodels.formula.api as smf

from analysis.preprocessing import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR, apply_fdr_correction, find_interaction_term
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "intervention_subgroups"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AnalysisSpec:
    name: str
    description: str
    function: Callable
    source_script: str


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, source_script: str = "intervention_subgroups_suite.py"):
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(name=name, description=description, function=func, source_script=source_script)
        return func
    return decorator


def load_master_with_standardization() -> pd.DataFrame:
    """Load master dataset with standardized predictors."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    return master


# Clustering features
PROFILE_FEATURES = [
    'ucla_total', 'dass_dep', 'dass_anx', 'dass_str',  # Psychological
    'pe_rate', 'stroop_interference', 'prp_bottleneck',  # EF
]

# Outcome measures
EF_OUTCOMES = ['pe_rate', 'stroop_interference', 'prp_bottleneck']


@register_analysis("optimal_clusters", "Determine optimal number of clusters")
def analyze_optimal_clusters(verbose: bool = True) -> Dict:
    """
    Determine optimal cluster number using multiple criteria.

    Methods:
    - Silhouette score (higher = better)
    - BIC from GMM (lower = better)
    - Gap statistic
    """
    if verbose:
        print("\n" + "=" * 70)
        print("OPTIMAL CLUSTER NUMBER DETERMINATION")
        print("=" * 70)

    master = load_master_with_standardization()

    # Select features that exist
    available_features = [f for f in PROFILE_FEATURES if f in master.columns]
    data = master[available_features].dropna()

    if len(data) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(data)})")
        return {}

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    if verbose:
        print(f"  N = {len(data)}")
        print(f"  Features: {available_features}")
        print("\n  Evaluating k = 2 to 6 clusters...")

    results = {'n': len(data), 'features': available_features}

    k_range = range(2, 7)
    silhouettes = []
    bics = []

    for k in k_range:
        # K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels_km)
        silhouettes.append(sil)

        # GMM for BIC
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(X)
        bics.append(gmm.bic(X))

        if verbose:
            print(f"    k={k}: Silhouette={sil:.3f}, BIC={bics[-1]:.1f}")

    # Find optimal
    results['silhouettes'] = silhouettes
    results['bics'] = bics

    optimal_sil = k_range[np.argmax(silhouettes)]
    optimal_bic = k_range[np.argmin(bics)]

    results['optimal_silhouette'] = optimal_sil
    results['optimal_bic'] = optimal_bic

    # Use consensus or silhouette
    optimal_k = optimal_sil

    if verbose:
        print(f"\n  Optimal k (silhouette): {optimal_sil}")
        print(f"  Optimal k (BIC): {optimal_bic}")
        print(f"  Using k = {optimal_k} for subsequent analyses")

    results['optimal_k'] = optimal_k

    import json
    with open(OUTPUT_DIR / "optimal_clusters.json", 'w', encoding='utf-8') as f:
        json_results = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in results.items()}
        json.dump(json_results, f, indent=2)

    return results


@register_analysis("gmm_profiles", "Gaussian Mixture Model latent profiles")
def analyze_gmm_profiles(verbose: bool = True) -> pd.DataFrame:
    """
    Fit GMM as approximation to Latent Profile Analysis.
    Identifies distinct psychological-cognitive profiles.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GAUSSIAN MIXTURE MODEL PROFILES")
        print("=" * 70)

    master = load_master_with_standardization()

    # Get optimal k
    opt_file = OUTPUT_DIR / "optimal_clusters.json"
    if opt_file.exists():
        import json
        with open(opt_file, 'r') as f:
            opt = json.load(f)
        n_clusters = opt.get('optimal_k', 3)
    else:
        n_clusters = 3

    if verbose:
        print(f"  Using k = {n_clusters} clusters")

    # Select features
    available_features = [f for f in PROFILE_FEATURES if f in master.columns]
    data = master[['participant_id'] + available_features + ['gender_male', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']].dropna()

    if len(data) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(data)})")
        return pd.DataFrame()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(data[available_features])

    # Fit GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10, covariance_type='full')
    data['cluster'] = gmm.fit_predict(X)
    data['cluster_prob'] = gmm.predict_proba(X).max(axis=1)

    if verbose:
        print(f"\n  N = {len(data)}")
        print("\n  Cluster Profiles:")
        print("  " + "-" * 60)

    # Profile each cluster
    profile_results = []

    for cluster in range(n_clusters):
        cluster_data = data[data['cluster'] == cluster]
        n_cluster = len(cluster_data)
        pct = n_cluster / len(data) * 100

        profile = {
            'cluster': cluster,
            'n': n_cluster,
            'pct': pct,
            'mean_prob': cluster_data['cluster_prob'].mean(),
        }

        for feat in available_features:
            profile[f'mean_{feat}'] = cluster_data[feat].mean()
            profile[f'sd_{feat}'] = cluster_data[feat].std()

        # Gender composition
        profile['pct_male'] = cluster_data['gender_male'].mean() * 100

        profile_results.append(profile)

        if verbose:
            print(f"\n  Cluster {cluster}: N={n_cluster} ({pct:.1f}%)")
            ucla_mean = cluster_data['ucla_total'].mean() if 'ucla_total' in cluster_data.columns else np.nan
            dass_mean = (cluster_data['dass_dep'].mean() + cluster_data['dass_anx'].mean() + cluster_data['dass_str'].mean()) / 3
            print(f"    UCLA mean: {ucla_mean:.1f}")
            print(f"    DASS mean: {dass_mean:.1f}")
            print(f"    PE rate: {cluster_data['pe_rate'].mean()*100:.1f}%" if 'pe_rate' in cluster_data.columns else "")
            print(f"    Male %: {profile['pct_male']:.1f}%")

    profile_df = pd.DataFrame(profile_results)
    profile_df.to_csv(OUTPUT_DIR / "gmm_profiles.csv", index=False, encoding='utf-8-sig')

    # Save cluster assignments
    data[['participant_id', 'cluster', 'cluster_prob']].to_csv(
        OUTPUT_DIR / "cluster_assignments.csv", index=False, encoding='utf-8-sig'
    )

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gmm_profiles.csv'}")
        print(f"  Output: {OUTPUT_DIR / 'cluster_assignments.csv'}")

    return profile_df


@register_analysis("subgroup_ucla_effects", "UCLA effects within each subgroup")
def analyze_subgroup_ucla_effects(verbose: bool = True) -> pd.DataFrame:
    """
    Test UCLA → EF effects within each identified subgroup.
    Identifies which subgroup shows the strongest UCLA-EF relationship.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECTS WITHIN SUBGROUPS")
        print("=" * 70)

    master = load_master_with_standardization()

    # Load cluster assignments
    cluster_file = OUTPUT_DIR / "cluster_assignments.csv"
    if not cluster_file.exists():
        analyze_gmm_profiles(verbose=False)

    if not cluster_file.exists():
        if verbose:
            print("  Cluster assignments not available")
        return pd.DataFrame()

    clusters = pd.read_csv(cluster_file)
    merged = master.merge(clusters, on='participant_id', how='inner')

    if len(merged) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    all_results = []

    for cluster in merged['cluster'].unique():
        cluster_data = merged[merged['cluster'] == cluster]

        if len(cluster_data) < 20:
            continue

        if verbose:
            print(f"\n  Cluster {cluster} (N={len(cluster_data)})")
            print("  " + "-" * 50)

        for outcome in EF_OUTCOMES:
            if outcome not in cluster_data.columns:
                continue

            valid = cluster_data.dropna(subset=['z_ucla', outcome, 'z_dass_dep', 'z_dass_anx', 'z_dass_str'])

            if len(valid) < 15:
                continue

            try:
                formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=valid).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    se = model.bse['z_ucla']
                    p = model.pvalues['z_ucla']

                    all_results.append({
                        'cluster': cluster,
                        'outcome': outcome,
                        'beta_ucla': beta,
                        'se_ucla': se,
                        'p_ucla': p,
                        'r_squared': model.rsquared,
                        'n': len(valid)
                    })

                    if verbose and p < 0.10:
                        sig = "*" if p < 0.05 else "†"
                        print(f"    UCLA → {outcome}: β={beta:.4f}, p={p:.4f}{sig}")

            except Exception:
                continue

    if len(all_results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "subgroup_ucla_effects.csv", index=False, encoding='utf-8-sig')

    # Identify high-risk subgroup
    if verbose:
        print("\n  HIGH-RISK SUBGROUP IDENTIFICATION:")
        print("  " + "-" * 50)

        sig_effects = results_df[results_df['p_ucla'] < 0.05]
        if len(sig_effects) > 0:
            for _, row in sig_effects.iterrows():
                direction = "Higher" if row['beta_ucla'] > 0 else "Lower"
                print(f"    Cluster {row['cluster']}: {direction} UCLA → worse {row['outcome']}")
        else:
            print("    No significant within-cluster UCLA effects (DASS controlled)")

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'subgroup_ucla_effects.csv'}")

    return results_df


@register_analysis("risk_decision_tree", "Decision tree for high-risk identification")
def analyze_risk_decision_tree(verbose: bool = True) -> Dict:
    """
    Build decision tree to identify high-risk individuals based on
    easily assessed features (demographics, self-report).

    Outcome: Poor EF (top quartile of composite EF deficit)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RISK DECISION TREE")
        print("=" * 70)

    master = load_master_with_standardization()

    # Create EF deficit composite (higher = worse)
    ef_vars = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available_ef = [v for v in ef_vars if v in master.columns]

    if len(available_ef) < 2:
        if verbose:
            print("  Insufficient EF variables")
        return {}

    # Standardize and average EF deficits
    for v in available_ef:
        master[f'z_{v}'] = (master[v] - master[v].mean()) / master[v].std()

    master['ef_deficit'] = master[[f'z_{v}' for v in available_ef]].mean(axis=1)

    # Define high-risk as top quartile
    threshold = master['ef_deficit'].quantile(0.75)
    master['high_risk'] = (master['ef_deficit'] >= threshold).astype(int)

    # Predictor features (easily assessed in clinical settings)
    predictor_features = ['ucla_total', 'dass_dep', 'dass_anx', 'dass_str', 'age', 'gender_male']
    available_predictors = [f for f in predictor_features if f in master.columns]

    data = master[available_predictors + ['high_risk']].dropna()

    if len(data) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(data)})")
        return {}

    X = data[available_predictors]
    y = data['high_risk']

    if verbose:
        print(f"  N = {len(data)}")
        print(f"  High-risk (top 25% EF deficit): {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Predictors: {available_predictors}")

    # Fit decision tree (limited depth for interpretability)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree.fit(X, y)

    # Feature importance
    importance = dict(zip(available_predictors, tree.feature_importances_))

    results = {
        'n': len(data),
        'n_high_risk': int(y.sum()),
        'accuracy': tree.score(X, y),
        'feature_importance': importance
    }

    if verbose:
        print(f"\n  Tree Accuracy: {results['accuracy']:.3f}")
        print("\n  Feature Importance:")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"    {feat}: {imp:.3f}")

        print("\n  Decision Rules:")
        print("  " + "-" * 50)
        rules = export_text(tree, feature_names=available_predictors)
        for line in rules.split('\n')[:15]:  # Print first 15 lines
            print(f"    {line}")

    # Save rules
    with open(OUTPUT_DIR / "decision_tree_rules.txt", 'w', encoding='utf-8') as f:
        f.write(export_text(tree, feature_names=available_predictors))

    # Save results
    import json
    with open(OUTPUT_DIR / "risk_decision_tree.json", 'w', encoding='utf-8') as f:
        json_results = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in results.items()}
        json.dump(json_results, f, indent=2, default=float)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'decision_tree_rules.txt'}")
        print(f"  Output: {OUTPUT_DIR / 'risk_decision_tree.json'}")

    return results


@register_analysis("subgroup_profiles", "Detailed subgroup profiling")
def analyze_subgroup_profiles(verbose: bool = True) -> pd.DataFrame:
    """
    Create detailed profiles of each subgroup including:
    - Demographics
    - Psychological measures
    - Cognitive performance
    - Risk characteristics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DETAILED SUBGROUP PROFILES")
        print("=" * 70)

    master = load_master_with_standardization()

    # Load cluster assignments
    cluster_file = OUTPUT_DIR / "cluster_assignments.csv"
    if not cluster_file.exists():
        analyze_gmm_profiles(verbose=False)

    if not cluster_file.exists():
        if verbose:
            print("  Cluster assignments not available")
        return pd.DataFrame()

    clusters = pd.read_csv(cluster_file)
    merged = master.merge(clusters, on='participant_id', how='inner')

    # All variables for profiling
    profile_vars = {
        'Demographics': ['age', 'gender_male'],
        'Psychological': ['ucla_total', 'dass_dep', 'dass_anx', 'dass_str'],
        'WCST': ['pe_rate', 'npe_rate', 'total_error_rate', 'categories_completed'],
        'Stroop': ['stroop_interference', 'stroop_effect', 'stroop_acc'],
        'PRP': ['prp_bottleneck', 'prp_short_rt', 'prp_long_rt']
    }

    all_profiles = []

    for cluster in sorted(merged['cluster'].unique()):
        cluster_data = merged[merged['cluster'] == cluster]

        profile = {'cluster': cluster, 'n': len(cluster_data)}

        for category, vars in profile_vars.items():
            for v in vars:
                if v in cluster_data.columns:
                    profile[f'{v}_mean'] = cluster_data[v].mean()
                    profile[f'{v}_sd'] = cluster_data[v].std()

        all_profiles.append(profile)

    profile_df = pd.DataFrame(all_profiles)

    if verbose:
        print(f"\n  N clusters: {len(all_profiles)}")

        # Print summary table
        for cluster in profile_df['cluster'].unique():
            row = profile_df[profile_df['cluster'] == cluster].iloc[0]
            print(f"\n  CLUSTER {cluster} (N={row['n']})")
            print("  " + "-" * 40)

            # Key characteristics
            if 'ucla_total_mean' in row:
                ucla_level = "High" if row['ucla_total_mean'] > 45 else "Moderate" if row['ucla_total_mean'] > 35 else "Low"
                print(f"    UCLA: {row['ucla_total_mean']:.1f} ({ucla_level})")

            if 'dass_dep_mean' in row:
                dass_level = "High" if row['dass_dep_mean'] > 10 else "Moderate" if row['dass_dep_mean'] > 5 else "Low"
                print(f"    DASS-Dep: {row['dass_dep_mean']:.1f} ({dass_level})")

            if 'pe_rate_mean' in row:
                print(f"    WCST PE: {row['pe_rate_mean']*100:.1f}%")

            if 'gender_male_mean' in row:
                print(f"    Male %: {row['gender_male_mean']*100:.1f}%")

    profile_df.to_csv(OUTPUT_DIR / "subgroup_profiles_detailed.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'subgroup_profiles_detailed.csv'}")

    return profile_df


@register_analysis("summary", "Summary of intervention subgroup findings")
def analyze_summary(verbose: bool = True) -> Dict:
    """Generate summary of subgroup identification findings."""
    if verbose:
        print("\n" + "=" * 70)
        print("INTERVENTION SUBGROUPS SUMMARY")
        print("=" * 70)

    summary = {}

    # Load optimal clusters
    opt_file = OUTPUT_DIR / "optimal_clusters.json"
    if opt_file.exists():
        import json
        with open(opt_file, 'r') as f:
            opt = json.load(f)
        summary['optimal_k'] = opt.get('optimal_k', 'N/A')

    # Load profiles
    profile_file = OUTPUT_DIR / "gmm_profiles.csv"
    if profile_file.exists():
        profiles = pd.read_csv(profile_file)
        summary['n_clusters'] = len(profiles)

        # Identify highest UCLA cluster
        if 'mean_ucla_total' in profiles.columns:
            high_ucla_cluster = profiles.loc[profiles['mean_ucla_total'].idxmax(), 'cluster']
            summary['highest_ucla_cluster'] = int(high_ucla_cluster)

    # Load decision tree results
    tree_file = OUTPUT_DIR / "risk_decision_tree.json"
    if tree_file.exists():
        import json
        with open(tree_file, 'r') as f:
            tree = json.load(f)
        summary['tree_accuracy'] = tree.get('accuracy', 'N/A')
        summary['top_predictor'] = max(tree.get('feature_importance', {}).items(), key=lambda x: x[1])[0] if tree.get('feature_importance') else 'N/A'

    if verbose:
        print("\n  Key Findings:")
        print("  " + "-" * 50)
        print(f"  1. Optimal cluster number: {summary.get('optimal_k', 'N/A')}")
        print(f"  2. Highest UCLA cluster: {summary.get('highest_ucla_cluster', 'N/A')}")
        print(f"  3. Decision tree accuracy: {summary.get('tree_accuracy', 'N/A'):.3f}" if isinstance(summary.get('tree_accuracy'), float) else "")
        print(f"  4. Top risk predictor: {summary.get('top_predictor', 'N/A')}")

        print("\n  Clinical Implications:")
        print("  " + "-" * 50)
        print("  - Subgroups can guide targeted interventions")
        print("  - Decision tree provides simple screening rules")
        print("  - High UCLA + High DASS = highest risk profile")

    # Save summary
    import json
    with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'summary.json'}")

    return summary


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict:
    """Run intervention subgroups analyses."""
    if verbose:
        print("=" * 70)
        print("INTERVENTION SUBGROUPS ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}")
        results[analysis] = ANALYSES[analysis].function(verbose=verbose)
    else:
        analysis_order = [
            'optimal_clusters',
            'gmm_profiles',
            'subgroup_ucla_effects',
            'risk_decision_tree',
            'subgroup_profiles',
            'summary'
        ]

        for name in analysis_order:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("INTERVENTION SUBGROUPS SUITE COMPLETE")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    print("\nAvailable Intervention Subgroups Analyses:")
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--list', '-l', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
