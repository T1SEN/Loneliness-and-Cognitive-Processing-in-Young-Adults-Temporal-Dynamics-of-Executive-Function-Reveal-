"""
Clustering Analysis Suite
=========================

Subgroup identification and vulnerability clustering for UCLA × Executive Function.

Consolidates all 5 analyses:
- ef_vulnerability_clustering.py
- composite_vulnerability_indices.py
- error_burst_clustering.py
- attentional_lapse_mixture.py (gmm_profiles)
- gendered_temporal_vulnerability.py

Usage:
    python -m analysis.advanced.clustering_suite                    # Run all
    python -m analysis.advanced.clustering_suite --analysis vulnerability
    python -m analysis.advanced.clustering_suite --list

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from analysis.utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR, RESULTS_DIR
from analysis.utils.modeling import standardize_predictors
import statsmodels.formula.api as smf

np.random.seed(42)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "clustering_suite"
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


def load_clustering_data() -> pd.DataFrame:
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
    name="vulnerability",
    description="K-means clustering to identify EF vulnerability profiles",
    source_script="ef_vulnerability_clustering.py"
)
def analyze_vulnerability_clusters(verbose: bool = True) -> pd.DataFrame:
    """
    Identify EF vulnerability profiles via clustering.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("VULNERABILITY CLUSTERING")
        print("=" * 70)

    master = load_clustering_data()

    # EF metrics for clustering
    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 2:
        if verbose:
            print("  Insufficient EF metrics")
        return pd.DataFrame()

    df = master[available + ['participant_id', 'ucla_total', 'gender_male']].dropna(subset=available)

    if len(df) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")
        print(f"  Features: {available}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df[available])

    # K-means clustering (k=3)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    if verbose:
        print(f"\n  Cluster sizes:")
        for c in range(k):
            n_c = (df['cluster'] == c).sum()
            print(f"    Cluster {c}: n={n_c}")

    # Cluster characteristics
    cluster_stats = []

    for c in range(k):
        c_data = df[df['cluster'] == c]

        stats_row = {'cluster': c, 'n': len(c_data)}

        for col in available:
            stats_row[f'{col}_mean'] = c_data[col].mean()
            stats_row[f'{col}_sd'] = c_data[col].std()

        stats_row['ucla_mean'] = c_data['ucla_total'].mean()
        stats_row['pct_male'] = c_data['gender_male'].mean() * 100

        cluster_stats.append(stats_row)

        if verbose:
            print(f"\n    Cluster {c}:")
            print(f"      UCLA mean: {stats_row['ucla_mean']:.1f}")
            print(f"      % Male: {stats_row['pct_male']:.1f}%")

    cluster_df = pd.DataFrame(cluster_stats)
    cluster_df.to_csv(OUTPUT_DIR / "vulnerability_clusters.csv", index=False, encoding='utf-8-sig')

    # Save individual assignments
    df[['participant_id', 'cluster', 'ucla_total', 'gender_male']].to_csv(
        OUTPUT_DIR / "cluster_assignments.csv", index=False, encoding='utf-8-sig'
    )

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'vulnerability_clusters.csv'}")

    return cluster_df


@register_analysis(
    name="composite_index",
    description="Create composite vulnerability index from EF metrics",
    source_script="composite_vulnerability_indices.py"
)
def analyze_composite_index(verbose: bool = True) -> pd.DataFrame:
    """
    Create a composite EF vulnerability index.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("COMPOSITE VULNERABILITY INDEX")
        print("=" * 70)

    master = load_clustering_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 2:
        if verbose:
            print("  Insufficient EF metrics")
        return pd.DataFrame()

    df = master[available + ['participant_id', 'ucla_total', 'gender_male',
                             'z_dass_dep', 'z_dass_anx', 'z_dass_str']].dropna(subset=available)

    if len(df) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")

    # Standardize and create composite (mean of z-scores)
    scaler = StandardScaler()

    for col in available:
        df[f'{col}_z'] = scaler.fit_transform(df[[col]])

    z_cols = [f'{c}_z' for c in available]
    df['ef_vulnerability_index'] = df[z_cols].mean(axis=1)

    if verbose:
        print(f"\n  Vulnerability Index Stats:")
        print(f"    Mean: {df['ef_vulnerability_index'].mean():.3f}")
        print(f"    SD: {df['ef_vulnerability_index'].std():.3f}")
        print(f"    Range: [{df['ef_vulnerability_index'].min():.2f}, {df['ef_vulnerability_index'].max():.2f}]")

    # Test UCLA correlation with index
    r, p = stats.pearsonr(df['ucla_total'], df['ef_vulnerability_index'])

    if verbose:
        sig = "*" if p < 0.05 else ""
        print(f"\n  UCLA vs Vulnerability Index: r={r:.3f}, p={p:.4f}{sig}")

    # By gender
    for gender, label in [(1, 'Male'), (0, 'Female')]:
        g_data = df[df['gender_male'] == gender]
        if len(g_data) >= 10:
            r_g, p_g = stats.pearsonr(g_data['ucla_total'], g_data['ef_vulnerability_index'])
            if verbose:
                sig = "*" if p_g < 0.05 else ""
                print(f"    {label}: r={r_g:.3f}, p={p_g:.4f}{sig}")

    results = df[['participant_id', 'ef_vulnerability_index', 'ucla_total', 'gender_male']]
    results.to_csv(OUTPUT_DIR / "composite_vulnerability_index.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'composite_vulnerability_index.csv'}")

    return results


@register_analysis(
    name="gmm_profiles",
    description="Gaussian Mixture Model for latent performance profiles",
    source_script="attentional_lapse_mixture.py"
)
def analyze_gmm_profiles(verbose: bool = True) -> pd.DataFrame:
    """
    Use GMM to identify latent performance profiles.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GMM LATENT PROFILES")
        print("=" * 70)

    master = load_clustering_data()

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available = [c for c in ef_cols if c in master.columns]

    if len(available) < 2:
        if verbose:
            print("  Insufficient EF metrics")
        return pd.DataFrame()

    df = master[available + ['participant_id', 'ucla_total', 'gender_male']].dropna(subset=available)

    if len(df) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(df)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[available])

    # Fit GMM with 2-4 components, select best by BIC
    best_bic = np.inf
    best_k = 2
    best_gmm = None

    for k in range(2, 5):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm.fit(X)
        bic = gmm.bic(X)

        if verbose:
            print(f"  k={k}: BIC={bic:.1f}")

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    if verbose:
        print(f"\n  Best k={best_k}")

    df['profile'] = best_gmm.predict(X)
    df['profile_prob'] = best_gmm.predict_proba(X).max(axis=1)

    # Profile characteristics
    profile_stats = []

    for p in range(best_k):
        p_data = df[df['profile'] == p]

        stats_row = {
            'profile': p,
            'n': len(p_data),
            'pct': len(p_data) / len(df) * 100,
            'ucla_mean': p_data['ucla_total'].mean(),
            'pct_male': p_data['gender_male'].mean() * 100
        }
        profile_stats.append(stats_row)

        if verbose:
            print(f"\n  Profile {p} (n={len(p_data)}, {stats_row['pct']:.1f}%):")
            print(f"    UCLA mean: {stats_row['ucla_mean']:.1f}")
            print(f"    % Male: {stats_row['pct_male']:.1f}%")

    profile_df = pd.DataFrame(profile_stats)
    profile_df.to_csv(OUTPUT_DIR / "gmm_profiles.csv", index=False, encoding='utf-8-sig')

    df[['participant_id', 'profile', 'profile_prob', 'ucla_total', 'gender_male']].to_csv(
        OUTPUT_DIR / "gmm_assignments.csv", index=False, encoding='utf-8-sig'
    )

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gmm_profiles.csv'}")

    return profile_df


@register_analysis(
    name="error_burst",
    description="Error burst clustering and temporal patterns",
    source_script="error_burst_clustering.py"
)
def analyze_error_burst(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze error clustering patterns (error bursts vs random errors).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ERROR BURST CLUSTERING")
        print("=" * 70)

    master = load_clustering_data()

    # Load trial data
    trial_files = {
        'wcst': RESULTS_DIR / '4b_wcst_trials.csv',
        'stroop': RESULTS_DIR / '4c_stroop_trials.csv'
    }

    all_results = []

    for task, filepath in trial_files.items():
        if not filepath.exists():
            if verbose:
                print(f"  {task.upper()}: No trial file")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        trials = pd.read_csv(filepath, encoding='utf-8')
        trials.columns = trials.columns.str.lower()

        # Handle duplicate participant_id columns
        if 'participantid' in trials.columns and 'participant_id' in trials.columns:
            trials = trials.drop(columns=['participantid'])
        elif 'participantid' in trials.columns:
            trials = trials.rename(columns={'participantid': 'participant_id'})

        acc_col = 'correct' if 'correct' in trials.columns else 'is_correct'
        if acc_col not in trials.columns:
            continue

        # Compute error burst metrics per participant
        burst_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 20:
                continue

            pdata = pdata.sort_values(pdata.columns[0])
            errors = (pdata[acc_col] == False).astype(int).values

            n_errors = errors.sum()
            if n_errors < 3:
                continue

            # Compute run lengths
            runs = []
            current_run = 1
            for i in range(1, len(errors)):
                if errors[i] == errors[i-1]:
                    current_run += 1
                else:
                    runs.append((errors[i-1], current_run))
                    current_run = 1
            runs.append((errors[-1], current_run))

            # Error runs only
            error_runs = [r[1] for r in runs if r[0] == 1]
            max_error_run = max(error_runs) if error_runs else 0
            mean_error_run = np.mean(error_runs) if error_runs else 0

            # Transition probabilities
            n_post_error = 0
            n_error_after_error = 0
            for i in range(1, len(errors)):
                if errors[i-1] == 1:
                    n_post_error += 1
                    if errors[i] == 1:
                        n_error_after_error += 1

            p_error_after_error = n_error_after_error / n_post_error if n_post_error > 0 else np.nan

            burst_results.append({
                'participant_id': pid,
                'n_errors': n_errors,
                'max_error_run': max_error_run,
                'mean_error_run': mean_error_run,
                'p_error_after_error': p_error_after_error
            })

        if len(burst_results) < 20:
            continue

        burst_df = pd.DataFrame(burst_results)
        merged = master.merge(burst_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean max error run: {merged['max_error_run'].mean():.2f}")

        # Test UCLA effect on error clustering
        try:
            formula = "p_error_after_error ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            merged_clean = merged.dropna(subset=['p_error_after_error'])
            if len(merged_clean) >= 20:
                model = smf.ols(formula, data=merged_clean).fit()

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    UCLA -> P(Error|Error): β={beta:.3f}, p={p:.4f}{sig}")

                    all_results.append({
                        'task': task,
                        'metric': 'p_error_after_error',
                        'beta_ucla': beta,
                        'p_ucla': p,
                        'n': len(merged_clean)
                    })

        except Exception as e:
            if verbose:
                print(f"    Regression error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "error_burst_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'error_burst_results.csv'}")

    return results_df


@register_analysis(
    name="gendered_vulnerability",
    description="Temporal trajectory of gender-specific vulnerability",
    source_script="gendered_temporal_vulnerability.py"
)
def analyze_gendered_vulnerability(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze whether male-specific vulnerability emerges early or late in session.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENDERED TEMPORAL VULNERABILITY")
        print("=" * 70)

    master = load_clustering_data()

    trial_files = {
        'wcst': RESULTS_DIR / '4b_wcst_trials.csv'
    }

    all_results = []

    for task, filepath in trial_files.items():
        if not filepath.exists():
            if verbose:
                print(f"  {task.upper()}: No trial file")
            continue

        if verbose:
            print(f"\n  {task.upper()}")
            print("  " + "-" * 50)

        trials = pd.read_csv(filepath, encoding='utf-8')
        trials.columns = trials.columns.str.lower()

        # Handle duplicate participant_id columns
        if 'participantid' in trials.columns and 'participant_id' in trials.columns:
            trials = trials.drop(columns=['participantid'])
        elif 'participantid' in trials.columns:
            trials = trials.rename(columns={'participantid': 'participant_id'})

        acc_col = 'correct' if 'correct' in trials.columns else 'is_correct'
        if acc_col not in trials.columns:
            continue

        # Compute epoch metrics per participant
        epoch_results = []

        for pid, pdata in trials.groupby('participant_id'):
            if len(pdata) < 30:
                continue

            pdata = pdata.sort_values(pdata.columns[0]).reset_index(drop=True)
            n_trials = len(pdata)

            # Divide into thirds
            third = n_trials // 3
            early = pdata.iloc[:third]
            late = pdata.iloc[2*third:]

            error_rate_early = (early[acc_col] == False).mean() * 100
            error_rate_late = (late[acc_col] == False).mean() * 100
            error_slope = error_rate_late - error_rate_early

            epoch_results.append({
                'participant_id': pid,
                'error_rate_early': error_rate_early,
                'error_rate_late': error_rate_late,
                'error_slope': error_slope
            })

        if len(epoch_results) < 20:
            continue

        epoch_df = pd.DataFrame(epoch_results)
        merged = master.merge(epoch_df, on='participant_id', how='inner')

        if len(merged) < 20:
            continue

        if verbose:
            print(f"    N = {len(merged)}")

        # Test UCLA × Gender × Epoch interaction (using slope)
        try:
            formula = "error_slope ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit()

            if 'z_ucla' in model.params:
                beta_ucla = model.params['z_ucla']
                p_ucla = model.pvalues['z_ucla']

            interaction_key = 'z_ucla:C(gender_male)[T.1]'
            if interaction_key in model.params:
                beta_interaction = model.params[interaction_key]
                p_interaction = model.pvalues[interaction_key]
            else:
                beta_interaction, p_interaction = np.nan, np.nan

            if verbose:
                sig_u = "*" if p_ucla < 0.05 else ""
                sig_i = "*" if p_interaction < 0.05 else ""
                print(f"    UCLA -> Error Slope: β={beta_ucla:.3f}, p={p_ucla:.4f}{sig_u}")
                print(f"    UCLA × Gender: β={beta_interaction:.3f}, p={p_interaction:.4f}{sig_i}")

            all_results.append({
                'task': task,
                'beta_ucla': beta_ucla,
                'p_ucla': p_ucla,
                'beta_interaction': beta_interaction,
                'p_interaction': p_interaction,
                'n': len(merged)
            })

            # Gender-stratified
            for gender, label in [(1, 'Male'), (0, 'Female')]:
                g_data = merged[merged['gender_male'] == gender]
                if len(g_data) >= 15:
                    r, p = stats.pearsonr(g_data['z_ucla'], g_data['error_slope'])
                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"      {label}: UCLA-slope r={r:.3f}, p={p:.4f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    Regression error: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "gendered_temporal_vulnerability.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gendered_temporal_vulnerability.csv'}")

    return results_df


@register_analysis(
    name="cross_task_profile",
    description="Cross-task vulnerability profile clustering by UCLA group",
    source_script="NEW - cross_task_profile_clustering.py"
)
def analyze_cross_task_profiles(verbose: bool = True) -> pd.DataFrame:
    """
    과제 간 취약성 프로파일 군집화

    연구 질문: 과제 간 성능 패턴에 기반한 UCLA 관련 취약성 하위유형을 식별할 수 있는가?

    통계 방법:
    - 모든 EF 지표 표준화 (z-점수)
    - K-means 또는 계층적 군집화
    - 다항 로지스틱 회귀: cluster ~ UCLA * Gender + DASS
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-TASK VULNERABILITY PROFILE CLUSTERING")
        print("=" * 70)

    output_dir = OUTPUT_DIR / "cross_task_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    master = load_clustering_data()

    # EF 지표들
    ef_features = ['pe_rate', 'wcst_accuracy', 'stroop_interference', 'prp_bottleneck']
    available_features = [f for f in ef_features if f in master.columns]

    if len(available_features) < 3:
        if verbose:
            print("  Insufficient EF features for cross-task profiling")
        return pd.DataFrame()

    # 완전한 데이터만 사용
    clean_data = master.dropna(subset=available_features + ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str']).copy()

    if len(clean_data) < 50:
        if verbose:
            print(f"  Insufficient data (N={len(clean_data)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(clean_data)}")
        print(f"  Features: {available_features}")

    # 특성 표준화
    scaler = StandardScaler()
    X = scaler.fit_transform(clean_data[available_features])

    # 최적 클러스터 수 결정 (실루엣 점수)
    from sklearn.metrics import silhouette_score

    best_k = 3
    best_score = -1

    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    if verbose:
        print(f"  Optimal clusters: {best_k} (silhouette={best_score:.3f})")

    # 최종 클러스터링
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clean_data['cluster'] = kmeans.fit_predict(X)

    # 클러스터 특성 분석
    cluster_profiles = []
    for cluster_id in range(best_k):
        cluster_data = clean_data[clean_data['cluster'] == cluster_id]

        profile = {
            'cluster': cluster_id,
            'n': len(cluster_data),
            'pct': len(cluster_data) / len(clean_data) * 100,
            'mean_ucla': cluster_data['ucla_total'].mean(),
            'mean_dass_dep': cluster_data['dass_depression'].mean(),
            'pct_male': (cluster_data['gender_male'] == 1).mean() * 100
        }

        for feat in available_features:
            profile[f'mean_{feat}'] = cluster_data[feat].mean()
            profile[f'z_{feat}'] = (cluster_data[feat].mean() - clean_data[feat].mean()) / clean_data[feat].std()

        cluster_profiles.append(profile)

        if verbose:
            print(f"\n  Cluster {cluster_id}: N={profile['n']} ({profile['pct']:.1f}%)")
            print(f"    UCLA: {profile['mean_ucla']:.1f}, Male: {profile['pct_male']:.1f}%")
            for feat in available_features:
                z_val = profile[f'z_{feat}']
                indicator = "HIGH" if z_val > 0.5 else "LOW" if z_val < -0.5 else "avg"
                print(f"    {feat}: z={z_val:.2f} ({indicator})")

    profiles_df = pd.DataFrame(cluster_profiles)
    profiles_df.to_csv(output_dir / "cluster_profiles.csv", index=False, encoding='utf-8-sig')

    # UCLA × 클러스터 관계 분석
    # 다항 로지스틱 회귀: cluster ~ UCLA * Gender + DASS
    if verbose:
        print("\n  Testing UCLA prediction of cluster membership:")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        # 예측 변수 준비
        pred_vars = ['z_ucla', 'gender_male', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
        X_pred = clean_data[pred_vars].values
        y = clean_data['cluster'].values

        # 다항 로지스틱
        lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
        lr.fit(X_pred, y)

        # 각 클러스터에 대한 UCLA 계수
        for cluster_id in range(best_k):
            ucla_coef = lr.coef_[cluster_id, 0]  # z_ucla는 첫 번째 변수
            if verbose:
                print(f"    Cluster {cluster_id}: UCLA coefficient = {ucla_coef:.3f}")

    except Exception as e:
        if verbose:
            print(f"    Multinomial regression error: {e}")

    # UCLA tertile별 클러스터 분포
    clean_data['ucla_tertile'] = pd.qcut(clean_data['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

    tertile_distribution = pd.crosstab(
        clean_data['ucla_tertile'],
        clean_data['cluster'],
        normalize='index'
    ) * 100

    if verbose:
        print("\n  Cluster distribution by UCLA tertile (%):")
        print(tertile_distribution.round(1).to_string())

    # 카이제곱 검정
    contingency = pd.crosstab(clean_data['ucla_tertile'], clean_data['cluster'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    if verbose:
        sig = "*" if p < 0.05 else ""
        print(f"\n  Chi-square test: χ²={chi2:.2f}, p={p:.4f} {sig}")

    # 결과 저장
    clean_data[['participant_id', 'cluster', 'ucla_tertile'] + available_features].to_csv(
        output_dir / "participant_clusters.csv", index=False, encoding='utf-8-sig'
    )

    tertile_distribution.to_csv(output_dir / "ucla_cluster_distribution.csv", encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {output_dir}")

    return profiles_df


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    if verbose:
        print("=" * 70)
        print("CLUSTERING ANALYSIS SUITE")
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
        print("CLUSTERING SUITE COMPLETE")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    print("\nAvailable Clustering Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")
        print(f"    Source: {spec.source_script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--list', '-l', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
