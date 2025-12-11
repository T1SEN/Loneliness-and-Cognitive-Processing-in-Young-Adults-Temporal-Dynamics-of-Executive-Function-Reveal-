"""
Publication Network Analysis Suite
==================================

Implements domain-level Gaussian Graphical Models (GGM) for:
- Overall sample (UCLA + DASS + EF nodes)
- Gender-stratified networks (male/female)
- Bootstrap edge stability
- Permutation-based gender edge difference testing

Usage:
    python -m publication.network_analysis --analysis domain_gender

Programmatic:
    from publication.network_analysis.network_suite import run
    run(analysis="domain_gender")
"""

from __future__ import annotations

import json
import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

from publication.preprocessing import (
    load_master_dataset,
    prepare_gender_variable,
    safe_zscore,
)
from ._config import BASE_OUTPUT, NetworkVariableSet, VARIABLE_SETS


# =============================================================================
# HELPER DATA STRUCTURES
# =============================================================================

AVAILABLE_ANALYSES = {
    "domain_overall": "Domain-level network (overall sample only)",
    "domain_gender": "Domain-level network + gender comparison + permutations",
}


@dataclass
class NetworkResults:
    """Convenience structure for returning subset-level outputs."""

    edges: pd.DataFrame
    centrality: pd.DataFrame
    partial_correlations: pd.DataFrame
    metadata: Dict
    bootstrap: pd.DataFrame


# =============================================================================
# CORE UTILITIES
# =============================================================================

def prepare_network_frame(
    variable_set: str = "domain_level",
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, List[str], NetworkVariableSet]:
    """
    Load the master dataset and return a standardized frame with requested nodes.
    """
    if variable_set not in VARIABLE_SETS:
        raise ValueError(f"Unknown variable set '{variable_set}'. Options: {list(VARIABLE_SETS.keys())}")

    config = VARIABLE_SETS[variable_set]
    master = load_master_dataset(
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        merge_cognitive_summary=True,
        merge_trial_features=True,
    )
    master = prepare_gender_variable(master)

    master = config.apply_aliases(master)
    available_cols = config.available_columns(master.columns)
    if not available_cols:
        raise ValueError(
            f"No columns from {config.name} set found in master dataset."
            " Check preprocessing outputs."
        )

    standardized = pd.DataFrame(index=master.index)
    for col in available_cols:
        standardized[col] = safe_zscore(master[col])

    analysis_frame = pd.concat(
        [
            master[["participant_id", "gender_male"]].reset_index(drop=True),
            standardized.reset_index(drop=True),
        ],
        axis=1,
    )
    return analysis_frame, available_cols, config


def _precision_to_partial(precision: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.diag(precision))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(diag, diag)
        partial = -precision / denom
    np.fill_diagonal(partial, 1.0)
    partial = np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0)
    return partial


def _rank_transform(data: np.ndarray) -> np.ndarray:
    """Convert columns to ranked (Spearman) scores with zero mean and unit variance."""
    ranked = np.apply_along_axis(rankdata, 0, data)
    ranked = ranked.astype(float)
    col_mean = ranked.mean(axis=0)
    col_std = ranked.std(axis=0, ddof=1)
    col_std[col_std == 0] = 1.0
    ranked = (ranked - col_mean) / col_std
    return ranked


def estimate_glasso_network(
    matrix: pd.DataFrame,
    alpha: Optional[float] = None,
    use_cv: bool = True,
    cv_folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-4,
    correlation: str = "pearson",
) -> Dict:
    """
    Fit Graphical LASSO (optionally with CV to pick alpha).
    """
    n_samples = len(matrix)
    if n_samples < 10:
        raise ValueError(f"Need at least 10 rows to estimate network, got {n_samples}.")

    cv = min(cv_folds, n_samples - 1)
    cv = max(cv, 2)

    method = correlation.lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"Unknown correlation method '{correlation}'. Use 'pearson' or 'spearman'.")

    data = matrix.values.astype(float)
    if method == "spearman":
        data = _rank_transform(data)
    if use_cv and alpha is None:
        model = GraphicalLassoCV(
            alphas=10,
            cv=cv,
            max_iter=max_iter,
            tol=tol,
            n_refinements=4,
        ).fit(data)
        selected_alpha = float(model.alpha_)
        used_cv = True
        n_iter = int(model.n_iter_)
        objective = model.score(data)
        cv_alphas = None
        if hasattr(model, "cv_alphas_") and model.cv_alphas_ is not None:
            try:
                cv_alphas = model.cv_alphas_.tolist()
            except Exception:
                cv_alphas = None
    else:
        selected_alpha = float(alpha or 0.05)
        model = GraphicalLasso(
            alpha=selected_alpha,
            max_iter=max_iter,
            tol=tol,
        ).fit(data)
        used_cv = False
        n_iter = int(model.n_iter_)
        objective = model.score(data)
        cv_alphas = None

    precision = model.precision_.copy()
    covariance = model.covariance_.copy()
    partial = _precision_to_partial(precision)

    return {
        "precision": precision,
        "covariance": covariance,
        "partial_corr": partial,
        "alpha": selected_alpha,
        "used_cv": used_cv,
        "cv_alphas": cv_alphas,
        "n_iter": n_iter,
        "objective": objective,
    }


def partial_to_edge_df(
    partial_corr: np.ndarray,
    columns: List[str],
    config: NetworkVariableSet,
) -> pd.DataFrame:
    """Convert partial correlation matrix to edge list."""
    records: List[Dict[str, object]] = []
    p = len(columns)
    for i in range(p):
        for j in range(i + 1, p):
            weight = partial_corr[i, j]
            if np.isclose(weight, 0.0, atol=1e-6):
                continue
            node_i = columns[i]
            node_j = columns[j]
            records.append({
                "node_i": node_i,
                "node_j": node_j,
                "node_i_label": config.get_label(node_i),
                "node_j_label": config.get_label(node_j),
                "community_i": config.get_community(node_i),
                "community_j": config.get_community(node_j),
                "weight": float(weight),
                "abs_weight": float(abs(weight)),
                "sign": "positive" if weight > 0 else "negative",
            })
    return pd.DataFrame(records)


def build_graph(edge_df: pd.DataFrame, columns: List[str], config: NetworkVariableSet) -> nx.Graph:
    G = nx.Graph()
    for node in columns:
        G.add_node(
            node,
            label=config.get_label(node),
            community=config.get_community(node),
        )

    for _, row in edge_df.iterrows():
        G.add_edge(
            row["node_i"],
            row["node_j"],
            weight=row["weight"],
            abs_weight=row["abs_weight"],
        )

    for u, v, data in G.edges(data=True):
        data["distance"] = 1.0 / max(abs(data["weight"]), 1e-6)

    return G


def compute_centrality(
    graph: nx.Graph,
    columns: List[str],
    config: NetworkVariableSet,
) -> pd.DataFrame:
    """Compute strength/bridge centrality metrics."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    strength = {node: 0.0 for node in graph.nodes()}
    for node in graph.nodes():
        node_strength = 0.0
        for _, attrs in graph[node].items():
            node_strength += abs(attrs.get("weight", 0.0))
        strength[node] = node_strength

    distance_weighted = graph.copy()
    for u, v, data in distance_weighted.edges(data=True):
        data["distance"] = 1.0 / max(abs(data.get("weight", 0.0)), 1e-6)

    closeness = nx.closeness_centrality(distance_weighted, distance="distance")
    betweenness = nx.betweenness_centrality(distance_weighted, weight="distance", normalized=True)

    try:
        eigenvector = nx.eigenvector_centrality_numpy(graph, weight="abs_weight")
    except nx.NetworkXException:
        eigenvector = {node: 0.0 for node in graph.nodes()}

    bridge_strength = {}
    bridge_ratio = {}
    for node in graph.nodes():
        node_comm = config.get_community(node)
        cross = 0.0
        total = 0.0
        for neighbor, attrs in graph[node].items():
            w = abs(attrs.get("weight", 0.0))
            total += w
            if config.get_community(neighbor) != node_comm:
                cross += w
        bridge_strength[node] = cross
        bridge_ratio[node] = cross / total if total > 0 else 0.0

    rows = []
    for node in columns:
        rows.append({
            "node": node,
            "label": config.get_label(node),
            "community": config.get_community(node),
            "strength": strength.get(node, 0.0),
            "strength_z": np.nan,
            "betweenness": betweenness.get(node, 0.0),
            "closeness": closeness.get(node, 0.0),
            "eigenvector": eigenvector.get(node, 0.0),
            "bridge_strength": bridge_strength.get(node, 0.0),
            "bridge_ratio": bridge_ratio.get(node, 0.0),
            "degree": graph.degree(node),
        })

    centrality_df = pd.DataFrame(rows)
    if not centrality_df.empty:
        centrality_df["strength_z"] = centrality_df["strength"].pipe(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) else 0.0
        )

    return centrality_df


def bootstrap_edge_stability(
    matrix: pd.DataFrame,
    columns: List[str],
    config: NetworkVariableSet,
    alpha: float,
    correlation: str,
    n_iter: int = 500,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Non-parametric bootstrap resampling of network edges."""
    if n_iter <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    records: List[Dict[str, object]] = []
    valid_iters = 0

    for i in range(n_iter):
        try:
            sample = matrix.sample(
                frac=sample_fraction,
                replace=True,
                random_state=rng.integers(0, 1_000_000),
            )
            result = estimate_glasso_network(
                sample,
                alpha=alpha,
                use_cv=False,
                correlation=correlation,
            )
            edges = partial_to_edge_df(result["partial_corr"], columns, config)
            valid_iters += 1
            for _, edge in edges.iterrows():
                records.append({
                    "node_i": edge["node_i"],
                    "node_j": edge["node_j"],
                    "weight": edge["weight"],
                    "abs_weight": edge["abs_weight"],
                    "sign": 1 if edge["weight"] > 0 else -1,
                })
        except Exception:
            continue

    if not records or valid_iters == 0:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["node_i", "node_j"])
        .agg(
            n_present=("weight", "count"),
            mean_weight=("weight", "mean"),
            mean_abs_weight=("abs_weight", "mean"),
            sign_consistency=("sign", lambda x: abs(x.mean())),
        )
        .reset_index()
    )
    summary["presence_rate"] = summary["n_present"] / valid_iters
    summary["n_bootstrap"] = valid_iters
    summary["node_i_label"] = summary["node_i"].map(config.get_label)
    summary["node_j_label"] = summary["node_j"].map(config.get_label)
    return summary


def compare_group_edges(
    edges_a: pd.DataFrame,
    edges_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    config: NetworkVariableSet,
) -> pd.DataFrame:
    """Merge edge sets and compute differences between two groups."""
    a = edges_a[["node_i", "node_j", "weight"]].rename(columns={"weight": f"weight_{label_a}"})
    b = edges_b[["node_i", "node_j", "weight"]].rename(columns={"weight": f"weight_{label_b}"})
    merged = pd.merge(a, b, on=["node_i", "node_j"], how="outer")
    merged[f"weight_{label_a}"] = merged[f"weight_{label_a}"].fillna(0.0)
    merged[f"weight_{label_b}"] = merged[f"weight_{label_b}"].fillna(0.0)
    merged["weight_diff"] = merged[f"weight_{label_a}"] - merged[f"weight_{label_b}"]
    merged["abs_diff"] = merged["weight_diff"].abs()
    merged["node_i_label"] = merged["node_i"].map(config.get_label)
    merged["node_j_label"] = merged["node_j"].map(config.get_label)
    merged["community_i"] = merged["node_i"].map(config.get_community)
    merged["community_j"] = merged["node_j"].map(config.get_community)
    return merged


def permutation_edge_test(
    frame: pd.DataFrame,
    columns: List[str],
    config: NetworkVariableSet,
    alpha: float,
    n_permutations: int,
    observed_diff: Optional[pd.DataFrame] = None,
    correlation: str = "pearson",
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation test for gender edge differences."""
    if n_permutations <= 0:
        return pd.DataFrame()

    clean = frame.dropna(subset=columns).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    records: List[Dict[str, object]] = []
    valid = 0

    for _ in range(n_permutations):
        perm_labels = rng.permutation(clean["gender_male"].values)
        perm_df = clean.copy()
        perm_df["perm_gender"] = perm_labels

        male = perm_df[perm_df["perm_gender"] == 1]
        female = perm_df[perm_df["perm_gender"] == 0]
        if len(male) < config.min_n or len(female) < config.min_n:
            continue

        try:
            male_net = estimate_glasso_network(male[columns], alpha=alpha, use_cv=False, correlation=correlation)
            female_net = estimate_glasso_network(female[columns], alpha=alpha, use_cv=False, correlation=correlation)
        except Exception:
            continue

        male_edges = partial_to_edge_df(male_net["partial_corr"], columns, config)
        female_edges = partial_to_edge_df(female_net["partial_corr"], columns, config)
        diff_df = compare_group_edges(male_edges, female_edges, "male", "female", config)
        valid += 1

        for _, row in diff_df.iterrows():
            records.append({
                "node_i": row["node_i"],
                "node_j": row["node_j"],
                "weight_diff": row["weight_diff"],
            })

    if not records or valid == 0:
        return pd.DataFrame()

    perm_df = pd.DataFrame(records)
    summary = (
        perm_df.groupby(["node_i", "node_j"])["weight_diff"]
        .apply(list)
        .reset_index()
        .rename(columns={"weight_diff": "perm_distribution"})
    )
    summary["n_permutations"] = valid
    summary["node_i_label"] = summary["node_i"].map(config.get_label)
    summary["node_j_label"] = summary["node_j"].map(config.get_label)

    if observed_diff is not None and not observed_diff.empty:
        merged = pd.merge(
            observed_diff[["node_i", "node_j", "weight_diff", "abs_diff"]],
            summary,
            on=["node_i", "node_j"],
            how="left",
        )

        def compute_p(row: pd.Series) -> float:
            dist = row.get("perm_distribution", [])
            if not isinstance(dist, list) or len(dist) == 0:
                return np.nan
            threshold = abs(row["weight_diff"])
            exceed = sum(abs(val) >= threshold for val in dist)
            return (exceed + 1) / (len(dist) + 1)

        merged["p_value"] = merged.apply(compute_p, axis=1)
        merged["n_permutations"] = valid
        merged["abs_obs_diff"] = merged["abs_diff"]
        return merged

    return summary


# =============================================================================
# SUBSET RUNNERS
# =============================================================================

def run_subset_network(
    frame: pd.DataFrame,
    columns: List[str],
    config: NetworkVariableSet,
    subset_label: str,
    subset_mask: Optional[pd.Series],
    output_dir: str,
    alpha: Optional[float],
    use_cv: bool,
    cv_folds: int,
    bootstrap_iter: int,
    bootstrap_fraction: float,
    correlation: str,
) -> Optional[NetworkResults]:
    """Estimate network for a specific subset and persist outputs."""
    subset = frame if subset_mask is None else frame[subset_mask.values]
    subset = subset.dropna(subset=columns)
    n = len(subset)
    if n < config.min_n:
        print(f"[WARN] {subset_label}: insufficient rows (n={n} < min_n={config.min_n}). Skipping.")
        return None

    matrix = subset[columns]
    result = estimate_glasso_network(
        matrix,
        alpha=alpha,
        use_cv=use_cv,
        cv_folds=cv_folds,
        correlation=correlation,
    )
    edges = partial_to_edge_df(result["partial_corr"], columns, config)
    graph = build_graph(edges, columns, config)
    centrality = compute_centrality(graph, columns, config)
    partial_df = pd.DataFrame(result["partial_corr"], index=columns, columns=columns)

    subset_dir = BASE_OUTPUT / output_dir / subset_label
    subset_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(subset_dir / "edges.csv", index=False, encoding="utf-8-sig")
    centrality.to_csv(subset_dir / "centrality.csv", index=False, encoding="utf-8-sig")
    partial_df.to_csv(subset_dir / "partial_correlations.csv", encoding="utf-8-sig")

    metadata = {
        "subset": subset_label,
        "n": n,
        "alpha": result["alpha"],
        "used_cv": result["used_cv"],
        "cv_alphas": result["cv_alphas"],
        "objective": result["objective"],
        "n_iter": result["n_iter"],
        "correlation_method": correlation,
        "variables": columns,
        "labels": {col: config.get_label(col) for col in columns},
        "communities": {col: config.get_community(col) for col in columns},
    }
    (subset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    bootstrap_df = pd.DataFrame()
    if bootstrap_iter > 0:
        bootstrap_df = bootstrap_edge_stability(
            matrix,
            columns,
            config,
            alpha=result["alpha"],
            correlation=correlation,
            n_iter=bootstrap_iter,
            sample_fraction=bootstrap_fraction,
            random_state=42,
        )
        if not bootstrap_df.empty:
            bootstrap_df.to_csv(subset_dir / "bootstrap_edge_stability.csv", index=False, encoding="utf-8-sig")

    return NetworkResults(
        edges=edges,
        centrality=centrality,
        partial_correlations=partial_df,
        metadata=metadata,
        bootstrap=bootstrap_df,
    )


def run_gender_comparison(
    frame: pd.DataFrame,
    columns: List[str],
    config: NetworkVariableSet,
    output_dir: str,
    base_alpha: Optional[float],
    use_cv: bool,
    cv_folds: int,
    bootstrap_iter: int,
    bootstrap_fraction: float,
    n_permutations: int,
    correlation: str,
) -> Dict[str, object]:
    """Run overall + gender-specific networks and comparisons."""
    results: Dict[str, object] = {}

    overall = run_subset_network(
        frame,
        columns,
        config,
        subset_label="overall",
        subset_mask=None,
        output_dir=output_dir,
        alpha=base_alpha,
        use_cv=use_cv and base_alpha is None,
        cv_folds=cv_folds,
        bootstrap_iter=bootstrap_iter,
        bootstrap_fraction=bootstrap_fraction,
        correlation=correlation,
    )
    if overall is None:
        return results

    results["overall"] = overall
    group_alpha = overall.metadata.get("alpha", base_alpha or 0.05)

    male_mask = frame["gender_male"] == 1
    female_mask = frame["gender_male"] == 0

    male = run_subset_network(
        frame,
        columns,
        config,
        subset_label="male",
        subset_mask=male_mask,
        output_dir=output_dir,
        alpha=group_alpha,
        use_cv=False,
        cv_folds=cv_folds,
        bootstrap_iter=bootstrap_iter,
        bootstrap_fraction=bootstrap_fraction,
        correlation=correlation,
    )
    female = run_subset_network(
        frame,
        columns,
        config,
        subset_label="female",
        subset_mask=female_mask,
        output_dir=output_dir,
        alpha=group_alpha,
        use_cv=False,
        cv_folds=cv_folds,
        bootstrap_iter=bootstrap_iter,
        bootstrap_fraction=bootstrap_fraction,
        correlation=correlation,
    )

    if male and female:
        diff_dir = BASE_OUTPUT / output_dir / "gender_comparison"
        diff_dir.mkdir(parents=True, exist_ok=True)
        diff_df = compare_group_edges(male.edges, female.edges, "male", "female", config)
        diff_df.to_csv(diff_dir / "edge_differences.csv", index=False, encoding="utf-8-sig")
        results["edge_difference"] = diff_df

        perm_df = permutation_edge_test(
            frame,
            columns,
            config,
            alpha=group_alpha,
            n_permutations=n_permutations,
            observed_diff=diff_df,
            correlation=correlation,
        )
        if not perm_df.empty and n_permutations > 0:
            perm_df.to_csv(diff_dir / "permutation_results.csv", index=False, encoding="utf-8-sig")
            results["permutation"] = perm_df
    else:
        print("[INFO] Gender-specific networks unavailable; skipping edge comparison.")

    return results


# =============================================================================
# PUBLIC API
# =============================================================================

def run(
    analysis: str = "domain_gender",
    variable_set: str = "domain_level",
    bootstrap_iter: int = 500,
    bootstrap_fraction: float = 0.8,
    gender_permutations: int = 500,
    alpha: Optional[float] = None,
    use_cv: bool = True,
    cv_folds: int = 5,
    force_rebuild: bool = False,
    verbose: bool = True,
    correlation: str = "pearson",
) -> Dict[str, object]:
    """
    Entry point for network analyses.
    """
    if analysis not in AVAILABLE_ANALYSES:
        raise ValueError(f"Unknown analysis '{analysis}'. Options: {list(AVAILABLE_ANALYSES.keys())}")

    if verbose:
        print("=" * 70)
        print("PUBLICATION NETWORK ANALYSIS SUITE")
        print("=" * 70)
        print(f"Analysis: {analysis} | Variable set: {variable_set}")

    frame, columns, config = prepare_network_frame(
        variable_set=variable_set,
        use_cache=not force_rebuild,
        force_rebuild=force_rebuild,
    )

    output_dir = config.name
    results: Dict[str, object] = {}

    if analysis == "domain_overall":
        overall = run_subset_network(
            frame,
            columns,
            config,
            subset_label="overall",
            subset_mask=None,
            output_dir=output_dir,
            alpha=alpha,
            use_cv=use_cv and alpha is None,
            cv_folds=cv_folds,
            bootstrap_iter=bootstrap_iter,
            bootstrap_fraction=bootstrap_fraction,
            correlation=correlation,
        )
        if overall:
            results["overall"] = overall
    elif analysis == "domain_gender":
        results["gender"] = run_gender_comparison(
            frame,
            columns,
            config,
            output_dir=output_dir,
            base_alpha=alpha,
            use_cv=use_cv,
            cv_folds=cv_folds,
            bootstrap_iter=bootstrap_iter,
            bootstrap_fraction=bootstrap_fraction,
            n_permutations=gender_permutations,
            correlation=correlation,
        )

    if verbose:
        print("\nCompleted. Outputs saved to:", BASE_OUTPUT / output_dir)

    return results


def list_analyses() -> None:
    """Print available analyses."""
    print("\nAvailable network analyses:")
    for key, desc in AVAILABLE_ANALYSES.items():
        print(f"  {key:<16} - {desc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publication Network Analysis Suite")
    parser.add_argument("--analysis", "-a", type=str, default="domain_gender", help="Analysis key (see list).")
    parser.add_argument("--variable-set", "-v", type=str, default="domain_level", help="Variable set to use.")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations per subset.")
    parser.add_argument("--bootstrap-frac", type=float, default=0.8, help="Bootstrap sample fraction.")
    parser.add_argument("--permutations", type=int, default=500, help="Gender permutation iterations.")
    parser.add_argument("--alpha", type=float, default=None, help="Manual GraphicalLasso alpha (skip CV).")
    parser.add_argument("--no-cv", action="store_true", help="Disable GraphicalLassoCV (use alpha only).")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds when applicable.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of master dataset cache.")
    parser.add_argument(
        "--correlation",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation metric used before GraphicalLasso.",
    )
    parser.add_argument("--list", action="store_true", help="List available analyses and exit.")

    args = parser.parse_args()

    if args.list:
        list_analyses()
        sys.exit(0)

    run(
        analysis=args.analysis,
        variable_set=args.variable_set,
        bootstrap_iter=args.bootstrap,
        bootstrap_fraction=args.bootstrap_frac,
        gender_permutations=args.permutations,
        alpha=args.alpha,
        use_cv=not args.no_cv and args.alpha is None,
        cv_folds=args.cv_folds,
        force_rebuild=args.force_rebuild,
        verbose=True,
        correlation=args.correlation,
    )
