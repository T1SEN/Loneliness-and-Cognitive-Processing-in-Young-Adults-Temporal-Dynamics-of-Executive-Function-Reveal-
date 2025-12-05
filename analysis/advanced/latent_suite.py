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

from analysis.utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR
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
                interaction_beta = model.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
                interaction_p = model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)

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
            interaction_beta = m3.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
            interaction_p = m3.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)
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
