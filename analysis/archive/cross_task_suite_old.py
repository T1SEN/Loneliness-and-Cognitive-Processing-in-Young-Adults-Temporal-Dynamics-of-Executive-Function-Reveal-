"""
Cross-Task Analysis Suite
=========================

Unified cross-task analyses for UCLA × Executive Function research.

Consolidates:
- cross_task_consistency.py
- cross_task_integration.py
- cross_task_meta_control.py
- cross_task_order_effects.py
- task_order_effects.py
- age_gam_developmental_windows.py
- age_gender_ucla_threeway.py
- dass_anxiety_mask_hypothesis.py
- dose_response_threshold_analysis.py
- extreme_group_analysis.py
- hidden_patterns_analysis.py
- nonlinear_gender_effects.py
- nonlinear_threshold_analysis.py
- residual_ucla_analysis.py
- temporal_context_effects.py
- ucla_nonlinear_effects.py

Usage:
    python -m analysis.exploratory.cross_task_suite                    # Run all
    python -m analysis.exploratory.cross_task_suite --analysis consistency
    python -m analysis.exploratory.cross_task_suite --list

    from analysis.exploratory import cross_task_suite
    cross_task_suite.run('consistency')
    cross_task_suite.run()  # All analyses

NOTE: These are EXPLORATORY analyses. For confirmatory results, use:
    analysis/gold_standard/pipeline.py

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors
from analysis.preprocessing import (
    safe_zscore,
    prepare_gender_variable,
    find_interaction_term
)

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "cross_task_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable
    source_script: str


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, source_script: str):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            source_script=source_script
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cross_task_data() -> pd.DataFrame:
    """Load and prepare master dataset for cross-task analyses."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors
    master = standardize_predictors(master)

    return master


# =============================================================================
# UTILITIES
# =============================================================================

def residualize_for_covariates(
    df: pd.DataFrame,
    target_cols: List[str],
    covars: List[str]
) -> pd.DataFrame:
    """
    Regress out covariates from each target column and return residuals.
    """
    resid_df = pd.DataFrame(index=df.index)
    for col in target_cols:
        cols_needed = [col] + covars
        sub = df.dropna(subset=cols_needed)
        if len(sub) < 10:
            continue
        formula = f"{col} ~ " + " + ".join(covars)
        model = smf.ols(formula, data=sub).fit()
        resid_df.loc[sub.index, f"{col}_resid"] = model.resid
    return resid_df


# =============================================================================
# ANALYSIS 1: CROSS-TASK CONSISTENCY
# =============================================================================

@register_analysis(
    name="consistency",
    description="Within-person variability across EF tasks (CV, range)",
    source_script="cross_task_consistency.py"
)
def analyze_consistency(verbose: bool = True) -> pd.DataFrame:
    """
    Research Question: Are lonely individuals inconsistent across tasks
    (high cross-task variability) or consistently impaired?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-TASK CONSISTENCY ANALYSIS")
        print("=" * 70)

    master = load_cross_task_data()

    # EF metrics to analyze
    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]
    available_cols = [c for c in ef_cols if c in master.columns]

    if len(available_cols) < 2:
        print("  ERROR: Need at least 2 EF metrics")
        return pd.DataFrame()

    if verbose:
        print(f"  EF metrics: {available_cols}")

    # Standardize EF metrics using NaN-safe z-score
    for col in available_cols:
        if master[col].notna().sum() > 5:
            master[f'{col}_z'] = safe_zscore(master[col])

    z_cols = [f'{c}_z' for c in available_cols]

    # Compute cross-task metrics per participant
    results = []

    for _, row in master.iterrows():
        pid = row.get("participant_id")

        ef_values = []
        for col in z_cols:
            if col in row and pd.notna(row[col]):
                ef_values.append(row[col])

        if len(ef_values) < 2:
            continue

        ef_mean = np.mean(ef_values)
        ef_sd = np.std(ef_values)
        cross_task_cv = ef_sd / abs(ef_mean) if ef_mean != 0 else np.nan
        cross_task_range = max(ef_values) - min(ef_values)

        results.append({
            "participant_id": pid,
            "cross_task_cv": cross_task_cv,
            "cross_task_range": cross_task_range,
            "cross_task_mean": ef_mean,
            "cross_task_sd": ef_sd,
            "n_tasks": len(ef_values),
            "ucla_total": row.get("ucla_total"),
            "gender_male": row.get("gender_male"),
            "z_ucla": row.get("z_ucla"),
            "z_dass_dep": row.get("z_dass_dep"),
            "z_dass_anx": row.get("z_dass_anx"),
            "z_dass_str": row.get("z_dass_str"),
            "z_age": row.get("z_age"),
        })

    ct_df = pd.DataFrame(results)

    if verbose:
        print(f"  Participants with >= 2 tasks: {len(ct_df)}")

    # DASS/나이/성별 통제 회귀로 UCLA 효과 검정
    if len(ct_df) > 20:
        model_df = ct_df.dropna(subset=[
            'cross_task_cv', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'
        ])
        if len(model_df) >= 20:
            formula = "cross_task_cv ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=model_df).fit(cov_type='HC3')
            if verbose:
                print("\n  UCLA effect on cross-task CV (DASS/age/gender controlled):")
                print(f"    UCLA beta = {model.params.get('z_ucla', np.nan):.3f}, p = {model.pvalues.get('z_ucla', np.nan):.4f}")
                int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
                if int_term is not None and int_term in model.params:
                    print(f"    UCLA x Gender beta = {model.params[int_term]:.3f}, p = {model.pvalues[int_term]:.4f}")

    ct_df.to_csv(OUTPUT_DIR / "cross_task_consistency.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'cross_task_consistency.csv'}")

    return ct_df


# =============================================================================
# ANALYSIS 2: CROSS-TASK CORRELATIONS
# =============================================================================

@register_analysis(
    name="correlations",
    description="Correlation matrix between EF tasks",
    source_script="cross_task_integration.py"
)
def analyze_correlations(verbose: bool = True) -> pd.DataFrame:
    """
    Examine cross-task correlations to assess shared variance.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-TASK CORRELATIONS")
        print("=" * 70)

    master = load_cross_task_data()

    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck", "wcst_accuracy"]
    available_cols = [c for c in ef_cols if c in master.columns]

    if len(available_cols) < 2:
        print("  ERROR: Need at least 2 EF metrics")
        return pd.DataFrame()

    if verbose:
        print(f"  EF metrics: {available_cols}")

    covars = ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
    base_df = master.dropna(subset=available_cols + covars).copy()

    # 부분상관: 공변량 잔차화 후 상관
    resid_df = residualize_for_covariates(base_df, available_cols, covars)
    resid_cols = [f"{c}_resid" for c in available_cols if f"{c}_resid" in resid_df.columns]
    corr_matrix = resid_df[resid_cols].corr()

    if verbose and not corr_matrix.empty:
        print(f"\n  Partial Correlation Matrix (covariates removed):")
        print(corr_matrix.round(3).to_string())

    # Long-format results
    results = []
    for i, col1 in enumerate(resid_cols):
        for j, col2 in enumerate(resid_cols):
            if i < j:
                df_pair = resid_df[[col1, col2]].dropna()
                n = len(df_pair)

                if n > 5:
                    r, p = stats.pearsonr(df_pair[col1], df_pair[col2])

                    results.append({
                        'task1': col1.replace('_resid', ''),
                        'task2': col2.replace('_resid', ''),
                        'partial_r': r,
                        'p': p,
                        'n': n
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"  {col1.replace('_resid','')} x {col2.replace('_resid','')}: r={r:.3f}, p={p:.4f}, n={n}{sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "cross_task_correlations.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'cross_task_correlations.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: META-CONTROL FACTOR (PCA)
# =============================================================================

@register_analysis(
    name="meta_control",
    description="PCA-based meta-control factor across EF tasks",
    source_script="cross_task_meta_control.py"
)
def analyze_meta_control(verbose: bool = True) -> pd.DataFrame:
    """
    Extract a latent 'meta-control' factor via PCA across EF tasks.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("META-CONTROL FACTOR (PCA)")
        print("=" * 70)

    master = load_cross_task_data()

    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]
    available_cols = [c for c in ef_cols if c in master.columns]

    if len(available_cols) < 2:
        print("  ERROR: Need at least 2 EF metrics")
        return pd.DataFrame()

    # Standardize + 공변량 잔차화 후 PCA
    df_pca = master[available_cols + ['participant_id', 'ucla_total', 'gender_male',
                                       'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']].dropna(
        subset=available_cols + ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
    ).copy()

    if len(df_pca) < 20:
        print(f"  ERROR: Insufficient data (N={len(df_pca)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df_pca)}")
        print(f"  EF metrics: {available_cols}")

    # 공변량 잔차화 후 스케일링
    resid = residualize_for_covariates(df_pca, available_cols, ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])
    resid_cols = [f"{c}_resid" for c in available_cols if f"{c}_resid" in resid.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(resid[resid_cols])

    # PCA
    pca = PCA(n_components=min(len(available_cols), 3))
    pca_scores = pca.fit_transform(X_scaled)

    df_pca['meta_control'] = pca_scores[:, 0]  # First PC

    if verbose:
        print(f"\n  PCA Results:")
        print(f"    PC1 explained variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    PC1 loadings:")
        for col, loading in zip(available_cols, pca.components_[0]):
            print(f"      {col}: {loading:.3f}")

    # Standardize for regression using NaN-safe z-score
    df_pca['z_ucla'] = safe_zscore(df_pca['ucla_total'])

    # Test UCLA effect on meta-control (DASS-controlled)
    try:
        formula = "meta_control ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=df_pca).fit()

        if verbose:
            print(f"\n  Meta-control ~ UCLA (DASS-controlled):")
            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']
                print(f"    UCLA main effect: beta={beta:.3f}, p={p:.4f}")

            int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
            if int_term is not None and int_term in model.params:
                beta = model.params[int_term]
                p = model.pvalues[int_term]
                print(f"    UCLA x Gender: beta={beta:.3f}, p={p:.4f}")

    except Exception as e:
        if verbose:
            print(f"  Regression error: {e}")

    # Save results
    pca_loadings = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca.components_))],
        'explained_variance': pca.explained_variance_ratio_,
        **{col: pca.components_[:, i] for i, col in enumerate(available_cols)}
    })
    pca_loadings.to_csv(OUTPUT_DIR / "meta_control_loadings.csv", index=False, encoding='utf-8-sig')

    df_pca[['participant_id', 'meta_control', 'ucla_total', 'gender_male']].to_csv(
        OUTPUT_DIR / "meta_control_scores.csv", index=False, encoding='utf-8-sig'
    )

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'meta_control_loadings.csv'}")
        print(f"  Output: {OUTPUT_DIR / 'meta_control_scores.csv'}")

    return pca_loadings


# =============================================================================
# ANALYSIS 4: TASK ORDER EFFECTS
# =============================================================================

@register_analysis(
    name="order_effects",
    description="Test if task order affects EF outcomes or UCLA effects",
    source_script="task_order_effects.py"
)
def analyze_order_effects(verbose: bool = True) -> pd.DataFrame:
    """
    Test if task presentation order affects results.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TASK ORDER EFFECTS")
        print("=" * 70)

    master = load_cross_task_data()

    # Check if task order info exists
    order_cols = ['task_order', 'stroop_order', 'wcst_order', 'prp_order']
    available_order = [c for c in order_cols if c in master.columns]

    if len(available_order) == 0:
        if verbose:
            print("  No task order information available in dataset.")
            print("  Creating simulated analysis based on participant sequence.")

        # Use participant_id order as proxy
        master['session_order'] = master.groupby('participant_id').cumcount()

    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]
    available_ef = [c for c in ef_cols if c in master.columns]

    if verbose:
        print(f"  EF metrics: {available_ef}")

    results = []

    for outcome in available_ef:
        df = master.dropna(subset=[outcome, 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

        if len(df) < 30:
            continue

        # Split into early vs late participants (proxy order)
        median_idx = len(df) // 2
        df['session_half'] = np.where(df.index < df.index[median_idx], 'First', 'Second')

        # 공변량 통제 회귀: session_half 효과와 UCLA 상호작용 포함
        df['session_half'] = df['session_half'].astype('category')
        formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(session_half)"
        model = smf.ols(formula, data=df).fit(cov_type='HC3')

        int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
        results.append({
            'outcome': outcome,
            'n': len(df),
            'ucla_beta': model.params.get('z_ucla', np.nan),
            'ucla_p': model.pvalues.get('z_ucla', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
            'session_half_effect': model.params.get('C(session_half)[T.Second]', np.nan),
            'session_half_p': model.pvalues.get('C(session_half)[T.Second]', np.nan),
        })

        if verbose:
            print(f"  {outcome}: UCLA beta={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "order_effects.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'order_effects.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 5: AGE GAM / DEVELOPMENTAL WINDOWS
# =============================================================================

@register_analysis(
    name="age_gam",
    description="Polynomial age effects on UCLA→EF relationship",
    source_script="age_gam_developmental_windows.py"
)
def analyze_age_gam(verbose: bool = True) -> pd.DataFrame:
    """
    Tests age as continuous predictor using polynomial terms.
    Examines UCLA slope variation across age.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("AGE GAM / DEVELOPMENTAL WINDOWS ANALYSIS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['pe_rate', 'age', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    # Mean-center age and create polynomial terms
    master['age_mc'] = master['age'] - master['age'].mean()
    master['age_mc2'] = master['age_mc'] ** 2

    if verbose:
        print(f"  N = {len(master)}")
        print(f"  Age range: {master['age'].min():.0f}-{master['age'].max():.0f}")

    # Polynomial regression: PE ~ Age + Age² + UCLA × Gender × Age
    formula = ("pe_rate ~ age_mc + age_mc2 + z_ucla * C(gender_male) * age_mc + "
               "z_dass_dep + z_dass_anx + z_dass_str")
    model = smf.ols(formula, data=master).fit(cov_type='HC3')

    results = pd.DataFrame({
        'parameter': model.params.index,
        'coefficient': model.params.values,
        'std_err': model.bse.values,
        'p_value': model.pvalues.values
    })

    if verbose:
        print("\n  Key results:")
        for term in ['age_mc', 'age_mc2', 'z_ucla:age_mc']:
            if term in model.params:
                print(f"    {term}: β={model.params[term]:.4f}, p={model.pvalues[term]:.4f}")

    results.to_csv(OUTPUT_DIR / "age_gam_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'age_gam_results.csv'}")

    return results


# =============================================================================
# ANALYSIS 6: THREE-WAY INTERACTION (Age × Gender × UCLA)
# =============================================================================

@register_analysis(
    name="threeway_interaction",
    description="Age × Gender × UCLA three-way interaction on WCST PE",
    source_script="age_gender_ucla_threeway.py"
)
def analyze_threeway_interaction(verbose: bool = True) -> pd.DataFrame:
    """
    Tests whether male vulnerability to UCLA→PE is age-dependent.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("AGE × GENDER × UCLA THREE-WAY INTERACTION")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['pe_rate', 'age', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    master['age_centered'] = master['age'] - master['age'].mean()
    master['age_median_split'] = (master['age'] >= master['age'].median()).astype(int)
    master['age_group'] = master['age_median_split'].map({0: 'Younger', 1: 'Older'})

    if verbose:
        print(f"  N = {len(master)}")
        print(f"  Age median: {master['age'].median():.1f}")

    # Model with 3-way interaction
    formula_3way = 'pe_rate ~ z_ucla * C(gender_male) * age_centered + z_dass_dep + z_dass_anx + z_dass_str'
    model_3way = smf.ols(formula_3way, data=master).fit(cov_type='HC3')

    # Stratified analysis
    results = []
    for age_group in ['Younger', 'Older']:
        subset = master[master['age_group'] == age_group].copy()
        if len(subset) < 15:
            continue

        formula = 'pe_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str'
        model = smf.ols(formula, data=subset).fit(cov_type='HC3')

        int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
        results.append({
            'age_group': age_group,
            'n': len(subset),
            'ucla_beta': model.params.get('z_ucla', np.nan),
            'ucla_p': model.pvalues.get('z_ucla', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan
        })

        if verbose:
            int_beta = model.params.get(int_term, np.nan) if int_term else np.nan
            int_p = model.pvalues.get(int_term, np.nan) if int_term else np.nan
            print(f"  {age_group} (N={len(subset)}): UCLA×Gender β={int_beta:.4f}, p={int_p:.4f}")

    # Three-way interaction term - dynamic detection
    threeway_candidates = [k for k in model_3way.params.index
                          if 'ucla' in k.lower() and 'gender' in k.lower() and 'age' in k.lower() and ':' in k]
    if threeway_candidates and len(threeway_candidates) >= 1:
        threeway_term = threeway_candidates[0]
        if verbose:
            print(f"\n  3-way interaction: β={model_3way.params[threeway_term]:.4f}, p={model_3way.pvalues[threeway_term]:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "threeway_interaction_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'threeway_interaction_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 7: ANXIETY MASK HYPOTHESIS
# =============================================================================

@register_analysis(
    name="anxiety_mask",
    description="Tests whether anxiety masks UCLA effects on EF",
    source_script="dass_anxiety_mask_hypothesis.py"
)
def analyze_anxiety_mask(verbose: bool = True) -> pd.DataFrame:
    """
    Tests whether high anxiety masks loneliness→EF effects.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DASS ANXIETY MASK HYPOTHESIS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['pe_rate', 'dass_anxiety', 'z_ucla', 'z_dass_dep', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    # Median split on anxiety
    anxiety_median = master['dass_anxiety'].median()
    master['high_anxiety'] = (master['dass_anxiety'] > anxiety_median).astype(int)
    master['z_anxiety'] = (master['dass_anxiety'] - master['dass_anxiety'].mean()) / master['dass_anxiety'].std()

    if verbose:
        print(f"  N = {len(master)}")
        print(f"  Anxiety median: {anxiety_median:.1f}")

    # 3-way interaction: UCLA × Gender × Anxiety
    formula_3way = 'pe_rate ~ z_ucla * C(gender_male) * z_anxiety + z_dass_dep + z_dass_anx + z_dass_str + z_age'
    model_3way = smf.ols(formula_3way, data=master).fit(cov_type='HC3')

    # Stratified by anxiety
    results = []
    for anx_group, label in [(0, 'Low Anxiety'), (1, 'High Anxiety')]:
        subset = master[master['high_anxiety'] == anx_group].copy()
        if len(subset) < 15:
            continue

        # Keep continuous anxiety control within strata to avoid residual confounding
        formula = 'pe_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age'
        model = smf.ols(formula, data=subset).fit(cov_type='HC3')

        int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
        results.append({
            'anxiety_group': label,
            'n': len(subset),
            'ucla_beta': model.params.get('z_ucla', np.nan),
            'ucla_p': model.pvalues.get('z_ucla', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan
        })

        if verbose:
            int_beta = model.params.get(int_term, np.nan) if int_term else np.nan
            int_p = model.pvalues.get(int_term, np.nan) if int_term else np.nan
            print(f"  {label} (N={len(subset)}): UCLA×Gender β={int_beta:.4f}, p={int_p:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "anxiety_mask_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'anxiety_mask_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 8: DOSE-RESPONSE THRESHOLD
# =============================================================================

@register_analysis(
    name="dose_response",
    description="Tests linear vs threshold effects in UCLA→PE relationship",
    source_script="dose_response_threshold_analysis.py"
)
def analyze_dose_response(verbose: bool = True) -> pd.DataFrame:
    """
    Tests linearity vs threshold effects using piecewise regression and ROC.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DOSE-RESPONSE THRESHOLD ANALYSIS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['pe_rate', 'ucla_total', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(master)}")

    results = []

    # Linear model
    formula_linear = "pe_rate ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_linear = smf.ols(formula_linear, data=master).fit()
    results.append({
        'model': 'Linear',
        'aic': model_linear.aic,
        'bic': model_linear.bic,
        'r_squared': model_linear.rsquared,
        'ucla_beta': model_linear.params.get('z_ucla', np.nan),
        'ucla_p': model_linear.pvalues.get('z_ucla', np.nan)
    })

    # Quadratic model
    master['z_ucla_sq'] = master['z_ucla'] ** 2
    formula_quad = "pe_rate ~ z_ucla + z_ucla_sq + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_quad = smf.ols(formula_quad, data=master).fit()
    results.append({
        'model': 'Quadratic',
        'aic': model_quad.aic,
        'bic': model_quad.bic,
        'r_squared': model_quad.rsquared,
        'ucla_beta': model_quad.params.get('z_ucla', np.nan),
        'ucla_p': model_quad.pvalues.get('z_ucla', np.nan)
    })

    # ROC analysis (predict high PE from UCLA)
    pe_75 = master['pe_rate'].quantile(0.75)
    master['high_pe'] = (master['pe_rate'] > pe_75).astype(int)

    if master['high_pe'].sum() >= 5:
        fpr, tpr, thresholds = roc_curve(master['high_pe'], master['ucla_total'])
        roc_auc_val = auc(fpr, tpr)
        youden_idx = np.argmax(tpr - fpr)
        optimal_cutoff = thresholds[youden_idx]

        results.append({
            'model': 'ROC_optimal',
            'optimal_ucla_cutoff': optimal_cutoff,
            'roc_auc': roc_auc_val,
            'sensitivity': tpr[youden_idx],
            'specificity': 1 - fpr[youden_idx]
        })

        if verbose:
            print(f"\n  ROC Analysis:")
            print(f"    Optimal UCLA cutoff: {optimal_cutoff:.1f}")
            print(f"    AUC: {roc_auc_val:.3f}")

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n  Model comparison (lowest AIC = best):")
        for _, row in results_df[results_df['aic'].notna()].iterrows():
            print(f"    {row['model']}: AIC={row['aic']:.1f}, R²={row['r_squared']:.3f}")

    results_df.to_csv(OUTPUT_DIR / "dose_response_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'dose_response_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 9: EXTREME GROUPS COMPARISON
# =============================================================================

@register_analysis(
    name="extreme_groups",
    description="Compares high vs low UCLA groups on EF metrics",
    source_script="extreme_group_analysis.py"
)
def analyze_extreme_groups(verbose: bool = True) -> pd.DataFrame:
    """
    Compares top 25% vs bottom 25% UCLA groups on EF.
    NOTE: Does NOT control for DASS - exploratory only.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXTREME GROUP ANALYSIS (Quartile Split)")
        print("⚠️ WARNING: No DASS control - exploratory only")
        print("=" * 70)

    master = load_cross_task_data()

    q1 = master['ucla_total'].quantile(0.25)
    q3 = master['ucla_total'].quantile(0.75)

    low_group = master[master['ucla_total'] <= q1].copy()
    high_group = master[master['ucla_total'] >= q3].copy()

    if verbose:
        print(f"  Low UCLA (≤{q1:.0f}): N={len(low_group)}")
        print(f"  High UCLA (≥{q3:.0f}): N={len(high_group)}")

    ef_measures = {
        'stroop_interference': 'Stroop Interference (ms)',
        'pe_rate': 'WCST PE Rate (%)',
        'prp_bottleneck': 'PRP Bottleneck (ms)'
    }

    results = []
    for measure, label in ef_measures.items():
        if measure not in master.columns:
            continue

        low_data = low_group[measure].dropna()
        high_data = high_group[measure].dropna()

        if len(low_data) < 5 or len(high_data) < 5:
            continue

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(low_data, high_data, equal_var=False)

        # Cohen's d
        pooled_sd = np.sqrt(((len(low_data) - 1) * low_data.std()**2 +
                            (len(high_data) - 1) * high_data.std()**2) /
                           (len(low_data) + len(high_data) - 2))
        cohens_d = (high_data.mean() - low_data.mean()) / pooled_sd if pooled_sd > 0 else 0

        results.append({
            'measure': label,
            'low_mean': low_data.mean(),
            'low_sd': low_data.std(),
            'high_mean': high_data.mean(),
            'high_sd': high_data.std(),
            't': t_stat,
            'p': p_val,
            'cohens_d': cohens_d
        })

        if verbose:
            sig = "*" if p_val < 0.05 else ""
            print(f"  {label}: t={t_stat:.2f}, p={p_val:.4f}, d={cohens_d:.2f}{sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "extreme_groups_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'extreme_groups_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 10: HIDDEN PATTERNS (Double Dissociation)
# =============================================================================

@register_analysis(
    name="hidden_patterns",
    description="Gender-specific task vulnerability patterns",
    source_script="hidden_patterns_analysis.py"
)
def analyze_hidden_patterns(verbose: bool = True) -> pd.DataFrame:
    """
    Tests for double dissociation: Males→WCST, Females→Stroop vulnerability.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HIDDEN PATTERNS: DOUBLE DISSOCIATION")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(master)}")

    tasks = {
        'WCST_PE': 'pe_rate',
        'Stroop_Interference': 'stroop_interference',
        'PRP_Bottleneck': 'prp_bottleneck'
    }

    results = []
    for task_name, metric in tasks.items():
        if metric not in master.columns or master[metric].notna().sum() < 30:
            continue

        formula = f"{metric} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=master.dropna(subset=[metric])).fit(cov_type='HC3')

        int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
        results.append({
            'task': task_name,
            'metric': metric,
            'n': model.nobs,
            'ucla_main_beta': model.params.get('z_ucla', np.nan),
            'ucla_main_p': model.pvalues.get('z_ucla', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan
        })

        if verbose:
            int_beta = model.params.get(int_term, np.nan) if int_term else np.nan
            print(f"  {task_name}: UCLA β={model.params.get('z_ucla', np.nan):.3f}, Interaction β={int_beta:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "hidden_patterns_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'hidden_patterns_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 11: NONLINEAR GENDER EFFECTS
# =============================================================================

@register_analysis(
    name="nonlinear_gender",
    description="Quadratic UCLA effects by gender",
    source_script="nonlinear_gender_effects.py"
)
def analyze_nonlinear_gender(verbose: bool = True) -> pd.DataFrame:
    """
    Tests quadratic UCLA × Gender interactions.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("NONLINEAR GENDER × UCLA EFFECTS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['pe_rate', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    master['z_ucla_sq'] = master['z_ucla'] ** 2

    if verbose:
        print(f"  N = {len(master)}")

    # Quadratic interaction model
    formula = "pe_rate ~ z_ucla + z_ucla_sq + z_ucla:C(gender_male) + z_ucla_sq:C(gender_male) + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model = smf.ols(formula, data=master).fit(cov_type='HC3')

    # Gender-stratified models
    results = []
    for gender, label in [(0, 'Female'), (1, 'Male')]:
        subset = master[master['gender_male'] == gender].copy()
        if len(subset) < 15:
            continue

        formula_strat = "pe_rate ~ z_ucla + z_ucla_sq + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model_strat = smf.ols(formula_strat, data=subset).fit(cov_type='HC3')

        results.append({
            'gender': label,
            'n': len(subset),
            'linear_beta': model_strat.params.get('z_ucla', np.nan),
            'linear_p': model_strat.pvalues.get('z_ucla', np.nan),
            'quadratic_beta': model_strat.params.get('z_ucla_sq', np.nan),
            'quadratic_p': model_strat.pvalues.get('z_ucla_sq', np.nan)
        })

        if verbose:
            print(f"  {label} (N={len(subset)}): Linear β={model_strat.params.get('z_ucla', np.nan):.3f}, Quadratic β={model_strat.params.get('z_ucla_sq', np.nan):.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "nonlinear_gender_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'nonlinear_gender_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 12: NONLINEAR THRESHOLD DETECTION
# =============================================================================

@register_analysis(
    name="nonlinear_threshold",
    description="Detects threshold effects in UCLA→EF",
    source_script="nonlinear_threshold_analysis.py"
)
def analyze_nonlinear_threshold(verbose: bool = True) -> pd.DataFrame:
    """
    Tests for threshold/breakpoint in UCLA→PE relationship.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("NONLINEAR THRESHOLD DETECTION")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['pe_rate', 'ucla_total', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(master)}")

    # Test candidate thresholds
    thresholds = [35, 40, 45, 50]
    results = []

    for threshold in thresholds:
        master[f'above_{threshold}'] = (master['ucla_total'] > threshold).astype(int)
        master[f'ucla_x_above_{threshold}'] = master['z_ucla'] * master[f'above_{threshold}']

        formula = f"pe_rate ~ z_ucla + above_{threshold} + ucla_x_above_{threshold} + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        try:
            model = smf.ols(formula, data=master).fit()
            results.append({
                'threshold': threshold,
                'aic': model.aic,
                'bic': model.bic,
                'r_squared': model.rsquared,
                'slope_change_beta': model.params.get(f'ucla_x_above_{threshold}', np.nan),
                'slope_change_p': model.pvalues.get(f'ucla_x_above_{threshold}', np.nan)
            })

            if verbose:
                print(f"  Threshold {threshold}: AIC={model.aic:.1f}, slope_change β={model.params.get(f'ucla_x_above_{threshold}', np.nan):.3f}")
        except Exception as e:
            if verbose:
                print(f"  Threshold {threshold}: Error - {e}")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        best_idx = results_df['aic'].idxmin()
        if verbose:
            print(f"\n  Best threshold: {results_df.loc[best_idx, 'threshold']} (lowest AIC)")

    results_df.to_csv(OUTPUT_DIR / "nonlinear_threshold_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'nonlinear_threshold_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 13: RESIDUALIZED UCLA ANALYSIS
# =============================================================================

@register_analysis(
    name="residual_ucla",
    description="UCLA effects after residualizing for DASS",
    source_script="residual_ucla_analysis.py"
)
def analyze_residual_ucla(verbose: bool = True) -> pd.DataFrame:
    """
    Tests UCLA effects using DASS-residualized UCLA scores.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RESIDUALIZED UCLA ANALYSIS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(master)}")

    # Residualize UCLA for DASS
    formula_resid = "ucla_total ~ dass_depression + dass_anxiety + dass_stress"
    model_resid = smf.ols(formula_resid, data=master).fit()
    master['ucla_residual'] = model_resid.resid
    master['z_ucla_resid'] = safe_zscore(master['ucla_residual'])

    if verbose:
        print(f"  UCLA ~ DASS: R² = {model_resid.rsquared:.3f}")
        print(f"  UCLA residual variance explained by DASS: {model_resid.rsquared*100:.1f}%")

    # Test residualized UCLA on EF
    outcomes = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    results = []

    for outcome in outcomes:
        if outcome not in master.columns:
            continue

        valid = master.dropna(subset=[outcome]).copy()
        if len(valid) < 20:
            continue

        formula = f"{outcome} ~ z_ucla_resid * C(gender_male) + z_age"
        model = smf.ols(formula, data=valid).fit(cov_type='HC3')

        # Find interaction term dynamically
        int_term = find_interaction_term(model.params.index, 'ucla_resid', 'gender')
        results.append({
            'outcome': outcome,
            'n': len(valid),
            'ucla_resid_beta': model.params.get('z_ucla_resid', np.nan),
            'ucla_resid_p': model.pvalues.get('z_ucla_resid', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan
        })

        if verbose:
            print(f"  {outcome}: UCLA_resid β={model.params.get('z_ucla_resid', np.nan):.3f}, p={model.pvalues.get('z_ucla_resid', np.nan):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "residual_ucla_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'residual_ucla_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 14: TEMPORAL CONTEXT EFFECTS
# =============================================================================

@register_analysis(
    name="temporal_context",
    description="Time-of-day and context effects on UCLA-EF relationship",
    source_script="temporal_context_effects.py"
)
def analyze_temporal_context(verbose: bool = True) -> pd.DataFrame:
    """
    Tests whether time-of-day affects UCLA→EF relationships.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEMPORAL CONTEXT EFFECTS")
        print("=" * 70)

    master = load_cross_task_data()

    # Check for timestamp data
    from analysis.preprocessing import RESULTS_DIR
    try:
        participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8')
        if 'createdAt' in participants.columns:
            participants['test_hour'] = pd.to_datetime(participants['createdAt'], errors='coerce').dt.hour
            participants = participants.rename(columns={'participantId': 'participant_id'})
            master = master.merge(participants[['participant_id', 'test_hour']], on='participant_id', how='left')
    except Exception:
        pass

    if 'test_hour' not in master.columns or master['test_hour'].isna().all():
        if verbose:
            print("  No temporal data available. Using simulated session order.")
        master['session_order'] = range(len(master))
        master['session_half'] = np.where(master['session_order'] < len(master) // 2, 'First', 'Second')
    else:
        # Categorize time
        def categorize_hour(h):
            if pd.isna(h):
                return 'unknown'
            if 6 <= h < 12:
                return 'morning'
            elif 12 <= h < 17:
                return 'afternoon'
            else:
                return 'evening'
        master['time_category'] = master['test_hour'].apply(categorize_hour)

    master = master.dropna(subset=['pe_rate', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(master)}")

    results = []

    # Test time effect on PE
    if 'time_category' in master.columns:
        time_groups = master.groupby('time_category')['pe_rate'].agg(['mean', 'std', 'count'])
        if verbose:
            print("\n  PE by time of day:")
            print(time_groups.to_string())
        results.append({'analysis': 'time_descriptives', 'data': time_groups.to_dict()})
    elif 'session_half' in master.columns:
        formula = "pe_rate ~ z_ucla * C(gender_male) + C(session_half) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=master).fit(cov_type='HC3')

        results.append({
            'analysis': 'session_half_effect',
            'n': len(master),
            'session_half_beta': model.params.get('C(session_half)[T.Second]', np.nan),
            'session_half_p': model.pvalues.get('C(session_half)[T.Second]', np.nan)
        })

        if verbose:
            print(f"  Session half effect: β={model.params.get('C(session_half)[T.Second]', np.nan):.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "temporal_context_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'temporal_context_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 15: NONLINEAR UCLA EFFECTS
# =============================================================================

@register_analysis(
    name="nonlinear_ucla",
    description="Tests quadratic/cubic UCLA effects on EF",
    source_script="ucla_nonlinear_effects.py"
)
def analyze_nonlinear_ucla(verbose: bool = True) -> pd.DataFrame:
    """
    Tests for quadratic and cubic UCLA effects across EF tasks.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("NONLINEAR UCLA EFFECTS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    master['z_ucla_sq'] = master['z_ucla'] ** 2
    master['z_ucla_cu'] = master['z_ucla'] ** 3

    if verbose:
        print(f"  N = {len(master)}")

    outcomes = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    results = []

    for outcome in outcomes:
        if outcome not in master.columns:
            continue

        valid = master.dropna(subset=[outcome]).copy()
        if len(valid) < 30:
            continue

        # Linear model
        formula_linear = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model_linear = smf.ols(formula_linear, data=valid).fit()

        # Quadratic model
        formula_quad = f"{outcome} ~ z_ucla + z_ucla_sq + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model_quad = smf.ols(formula_quad, data=valid).fit()

        # Compare models
        results.append({
            'outcome': outcome,
            'n': len(valid),
            'linear_aic': model_linear.aic,
            'linear_r2': model_linear.rsquared,
            'quad_aic': model_quad.aic,
            'quad_r2': model_quad.rsquared,
            'quad_beta': model_quad.params.get('z_ucla_sq', np.nan),
            'quad_p': model_quad.pvalues.get('z_ucla_sq', np.nan),
            'best_model': 'Quadratic' if model_quad.aic < model_linear.aic else 'Linear'
        })

        if verbose:
            best = 'Quadratic' if model_quad.aic < model_linear.aic else 'Linear'
            print(f"  {outcome}: Linear AIC={model_linear.aic:.1f}, Quadratic AIC={model_quad.aic:.1f} → Best: {best}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "nonlinear_ucla_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'nonlinear_ucla_results.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run cross-task analyses.
    """
    if verbose:
        print("=" * 70)
        print("CROSS-TASK ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Running: {spec.name}")
            try:
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-TASK SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable Cross-Task Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")
        print(f"    Source: {spec.source_script}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Task Analysis Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                        help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
