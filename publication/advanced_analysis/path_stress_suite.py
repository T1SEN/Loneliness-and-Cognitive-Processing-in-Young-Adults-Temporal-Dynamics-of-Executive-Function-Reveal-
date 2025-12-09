"""
Path Model Comparison Suite - STRESS Version
=============================================

경쟁 경로모형(Competing Path Models) 비교 분석 - 스트레스(Stress) 버전

네 가지 인과 모형을 비교하여 외로움-스트레스-인지기능 관계의 구조를 탐색:
- Model 1 (C→L→S): 인지기능 → 외로움 → 스트레스
- Model 2 (C→S→L): 인지기능 → 스트레스 → 외로움
- Model 3 (L→S→C): 외로움 → 스트레스 → 인지기능
- Model 4 (L→C→S): 외로움 → 인지기능 → 스트레스

Usage:
    python -m publication.advanced_analysis.path_stress_suite
    python -m publication.advanced_analysis.path_stress_suite --analysis path_comparison
    python -m publication.advanced_analysis.path_stress_suite --list

    from publication.advanced_analysis import path_stress_suite
    path_stress_suite.run()

NOTE: 횡단 데이터로는 인과 방향을 확정할 수 없음. 모형 적합도 비교만 가능.

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
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Publication preprocessing imports
from publication.preprocessing import (
    load_master_dataset,
    standardize_predictors,
    prepare_gender_variable,
)

# Shared utilities from _utils
from ._utils import (
    BASE_OUTPUT,
    SEMOPY_AVAILABLE,
    create_ef_composite,
    fit_path_model_semopy,
    extract_path_coefficients,
)

np.random.seed(42)

# Output directory
OUTPUT_DIR = BASE_OUTPUT / "path_stress"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
        )
        return func
    return decorator


# =============================================================================
# MODEL SPECIFICATIONS - STRESS
# =============================================================================

# Model 1: C → L → S (Cognition → Loneliness → Stress)
MODEL_1_CLS = """
z_ucla ~ c1*ef_composite + z_age + gender_male
z_dass_str ~ d1*z_ucla + z_age + gender_male
"""

# Model 2: C → S → L (Cognition → Stress → Loneliness)
MODEL_2_CSL = """
z_dass_str ~ c2*ef_composite + z_age + gender_male
z_ucla ~ d2*z_dass_str + z_age + gender_male
"""

# Model 3: L → S → C (Loneliness → Stress → Cognition)
MODEL_3_LSC = """
z_dass_str ~ c3*z_ucla + z_age + gender_male
ef_composite ~ d3*z_dass_str + z_age + gender_male
"""

# Model 4: L → C → S (Loneliness → Cognition → Stress)
MODEL_4_LCS = """
ef_composite ~ c4*z_ucla + z_age + gender_male
z_dass_str ~ d4*ef_composite + z_age + gender_male
"""

MODEL_SPECS = {
    'Model1_CLS': {
        'spec': MODEL_1_CLS,
        'path': 'C → L → S',
        'description': '인지기능 → 외로움 → 스트레스',
        'exogenous': 'ef_composite',
        'mediator': 'z_ucla',
        'endogenous': 'z_dass_str',
        'a_label': 'c1',
        'b_label': 'd1',
    },
    'Model2_CSL': {
        'spec': MODEL_2_CSL,
        'path': 'C → S → L',
        'description': '인지기능 → 스트레스 → 외로움',
        'exogenous': 'ef_composite',
        'mediator': 'z_dass_str',
        'endogenous': 'z_ucla',
        'a_label': 'c2',
        'b_label': 'd2',
    },
    'Model3_LSC': {
        'spec': MODEL_3_LSC,
        'path': 'L → S → C',
        'description': '외로움 → 스트레스 → 인지기능',
        'exogenous': 'z_ucla',
        'mediator': 'z_dass_str',
        'endogenous': 'ef_composite',
        'a_label': 'c3',
        'b_label': 'd3',
    },
    'Model4_LCS': {
        'spec': MODEL_4_LCS,
        'path': 'L → C → S',
        'description': '외로움 → 인지기능 → 스트레스',
        'exogenous': 'z_ucla',
        'mediator': 'ef_composite',
        'endogenous': 'z_dass_str',
        'a_label': 'c4',
        'b_label': 'd4',
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_path_data() -> pd.DataFrame:
    """
    Load and prepare data for path analysis.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - z_ucla: Standardized UCLA loneliness
        - z_dass_str: Standardized DASS stress
        - ef_composite: Mean of z-scored EF metrics
        - z_age, gender_male: Covariates
    """
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    # Handle UCLA column naming
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors
    master = standardize_predictors(master)

    # Create EF composite
    master = create_ef_composite(master)

    # Keep only complete cases
    required_cols = ['z_ucla', 'z_dass_str', 'ef_composite', 'z_age', 'gender_male']
    available_cols = [c for c in required_cols if c in master.columns]

    if len(available_cols) < len(required_cols):
        missing = set(required_cols) - set(available_cols)
        raise ValueError(f"Missing required columns: {missing}")

    df = master[required_cols].dropna()

    print(f"[DATA] Loaded {len(df)} complete cases for path analysis (Stress)")
    print(f"       Males: {df['gender_male'].sum()}, Females: {len(df) - df['gender_male'].sum()}")

    return df


# =============================================================================
# MAIN ANALYSES
# =============================================================================

@register_analysis(
    name="path_comparison",
    description="Compare 4 competing causal models: C→L→S, C→S→L, L→S→C, L→C→S"
)
def analyze_path_comparison(verbose: bool = True) -> pd.DataFrame:
    """Fit and compare four competing path models."""
    if verbose:
        print("\n" + "=" * 60)
        print("PATH MODEL COMPARISON (STRESS)")
        print("=" * 60)
        print("\nComparing 4 competing causal structures:")
        for name, info in MODEL_SPECS.items():
            print(f"  {name}: {info['description']} ({info['path']})")

    df = load_path_data()
    results = []
    coefficients = []

    for model_name, model_info in MODEL_SPECS.items():
        if verbose:
            print(f"\n[FITTING] {model_name}: {model_info['path']}")

        result = fit_path_model_semopy(model_info['spec'], df, model_name)
        coefs = extract_path_coefficients(result, model_info)

        fit_row = {
            'model': model_name,
            'path': model_info['path'],
            'description': model_info['description'],
            'n': result.get('n_obs'),
            'aic': result.get('aic'),
            'bic': result.get('bic'),
            'cfi': result.get('cfi'),
            'rmsea': result.get('rmsea'),
            'chi2': result.get('chi2'),
            'dof': result.get('dof'),
            'a_path': coefs['a_coef'],
            'b_path': coefs['b_coef'],
            'indirect': coefs['indirect_effect'],
            'a_p': coefs['a_p'],
            'b_p': coefs['b_p'],
        }
        results.append(fit_row)

        coef_row = {
            'model': model_name,
            'path': model_info['path'],
            'exogenous': model_info['exogenous'],
            'mediator': model_info['mediator'],
            'endogenous': model_info['endogenous'],
            'a_coef': coefs['a_coef'],
            'a_se': coefs['a_se'],
            'a_p': coefs['a_p'],
            'b_coef': coefs['b_coef'],
            'b_se': coefs['b_se'],
            'b_p': coefs['b_p'],
            'indirect': coefs['indirect_effect'],
        }
        coefficients.append(coef_row)

    comparison_df = pd.DataFrame(results)
    coef_df = pd.DataFrame(coefficients)

    # Rank models
    comparison_df['rank_aic'] = comparison_df['aic'].rank()
    comparison_df['rank_bic'] = comparison_df['bic'].rank()
    if comparison_df['cfi'].notna().any():
        comparison_df['rank_cfi'] = comparison_df['cfi'].rank(ascending=False)
        comparison_df['rank_rmsea'] = comparison_df['rmsea'].rank()
        comparison_df['rank_composite'] = comparison_df[['rank_aic', 'rank_bic', 'rank_cfi', 'rank_rmsea']].mean(axis=1)
    else:
        comparison_df['rank_composite'] = comparison_df[['rank_aic', 'rank_bic']].mean(axis=1)

    comparison_df = comparison_df.sort_values('rank_composite')

    if verbose:
        print("\n" + "-" * 60)
        print("MODEL FIT COMPARISON")
        print("-" * 60)
        for _, row in comparison_df.iterrows():
            print(f"\n{row['model']}: {row['path']}")
            print(f"  AIC: {row['aic']:.2f}" if pd.notna(row['aic']) else "  AIC: N/A")
            print(f"  BIC: {row['bic']:.2f}" if pd.notna(row['bic']) else "  BIC: N/A")
            if pd.notna(row['cfi']):
                print(f"  CFI: {row['cfi']:.3f}, RMSEA: {row['rmsea']:.3f}")
            print(f"  a-path: {row['a_path']:.3f} (p={row['a_p']:.4f})" if pd.notna(row['a_path']) else "  a-path: N/A")
            print(f"  b-path: {row['b_path']:.3f} (p={row['b_p']:.4f})" if pd.notna(row['b_path']) else "  b-path: N/A")
            print(f"  Indirect: {row['indirect']:.3f}" if pd.notna(row['indirect']) else "  Indirect: N/A")

        best = comparison_df.iloc[0]
        print("\n" + "=" * 60)
        print(f"BEST FIT: {best['model']} ({best['path']})")
        print("=" * 60)

    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding='utf-8-sig')
    coef_df.to_csv(OUTPUT_DIR / "path_coefficients.csv", index=False, encoding='utf-8-sig')

    # Visualize
    if verbose:
        plot_model_comparison(comparison_df)

    return comparison_df


@register_analysis(
    name="indirect_effects",
    description="Bootstrap confidence intervals for indirect effects"
)
def analyze_indirect_effects(verbose: bool = True, n_bootstrap: int = 5000) -> pd.DataFrame:
    """Bootstrap CIs for indirect effects in each model."""
    if verbose:
        print("\n" + "=" * 60)
        print("BOOTSTRAP INDIRECT EFFECTS (STRESS)")
        print(f"n_bootstrap = {n_bootstrap}")
        print("=" * 60)

    df = load_path_data()
    n = len(df)
    results = []

    for model_name, model_info in MODEL_SPECS.items():
        if verbose:
            print(f"\n[BOOTSTRAP] {model_name}: {model_info['path']}")

        indirect_boots = []
        for i in range(n_bootstrap):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{n_bootstrap}")

            boot_idx = np.random.choice(n, size=n, replace=True)
            boot_df = df.iloc[boot_idx].reset_index(drop=True)

            try:
                result = fit_path_model_semopy(model_info['spec'], boot_df, model_name)
                coefs = extract_path_coefficients(result, model_info)
                indirect = coefs['indirect_effect']
                if not np.isnan(indirect):
                    indirect_boots.append(indirect)
            except Exception:
                continue

        if len(indirect_boots) < n_bootstrap * 0.5:
            print(f"  [WARNING] Only {len(indirect_boots)} successful bootstraps")

        indirect_boots = np.array(indirect_boots)
        point_est = np.mean(indirect_boots)
        se = np.std(indirect_boots, ddof=1)
        ci_lower = np.percentile(indirect_boots, 2.5)
        ci_upper = np.percentile(indirect_boots, 97.5)
        significant = (ci_lower > 0) or (ci_upper < 0)

        results.append({
            'model': model_name,
            'path': model_info['path'],
            'description': model_info['description'],
            'indirect_point': point_est,
            'indirect_se': se,
            'ci_lower_95': ci_lower,
            'ci_upper_95': ci_upper,
            'significant_95': significant,
            'n_valid_boots': len(indirect_boots),
        })

        if verbose:
            sig_str = "***" if significant else ""
            print(f"  Indirect: {point_est:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] {sig_str}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "indirect_effects.csv", index=False, encoding='utf-8-sig')

    # Visualize
    if verbose:
        plot_indirect_effects(results_df)

    return results_df


@register_analysis(
    name="gender_moderated",
    description="Test path models separately by gender"
)
def analyze_gender_moderated(verbose: bool = True) -> pd.DataFrame:
    """Fit all 4 models separately for males and females."""
    if verbose:
        print("\n" + "=" * 60)
        print("GENDER-MODERATED PATH ANALYSIS (STRESS)")
        print("=" * 60)

    df = load_path_data()
    df_male = df[df['gender_male'] == 1].copy()
    df_female = df[df['gender_male'] == 0].copy()

    if verbose:
        print(f"\n[DATA] Males: n={len(df_male)}, Females: n={len(df_female)}")

    results = []

    for gender, gender_df in [('male', df_male), ('female', df_female)]:
        for model_name, model_info in MODEL_SPECS.items():
            if verbose:
                print(f"\n[FITTING] {gender.upper()} - {model_name}")

            spec_no_gender = model_info['spec'].replace('+ gender_male', '')
            result = fit_path_model_semopy(spec_no_gender, gender_df, f"{model_name}_{gender}")
            coefs = extract_path_coefficients(result, model_info)

            results.append({
                'gender': gender,
                'model': model_name,
                'path': model_info['path'],
                'n': len(gender_df),
                'aic': result.get('aic'),
                'bic': result.get('bic'),
                'cfi': result.get('cfi'),
                'rmsea': result.get('rmsea'),
                'a_path': coefs['a_coef'],
                'b_path': coefs['b_coef'],
                'indirect': coefs['indirect_effect'],
                'a_p': coefs['a_p'],
                'b_p': coefs['b_p'],
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "gender_paths.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print("\n" + "-" * 60)
        print("GENDER-SPECIFIC RESULTS")
        print("-" * 60)
        for model_name in MODEL_SPECS.keys():
            male_row = results_df[(results_df['model'] == model_name) & (results_df['gender'] == 'male')].iloc[0]
            female_row = results_df[(results_df['model'] == model_name) & (results_df['gender'] == 'female')].iloc[0]
            print(f"\n{model_name}: {MODEL_SPECS[model_name]['path']}")
            print(f"  Male   (n={int(male_row['n'])}): a={male_row['a_path']:.3f}, b={male_row['b_path']:.3f}, indirect={male_row['indirect']:.3f}")
            print(f"  Female (n={int(female_row['n'])}): a={female_row['a_path']:.3f}, b={female_row['b_path']:.3f}, indirect={female_row['indirect']:.3f}")

    # Visualize
    plot_gender_comparison(results_df)

    return results_df


@register_analysis(
    name="gender_indirect_comparison",
    description="Bootstrap comparison of indirect effects between genders (z-test)"
)
def analyze_gender_indirect_comparison(verbose: bool = True, n_bootstrap: int = 2000) -> pd.DataFrame:
    """
    성별별 간접효과 부트스트랩 후 차이 유의성 평가 (STRESS 버전).

    Parameters
    ----------
    verbose : bool
        Print progress
    n_bootstrap : int
        Number of bootstrap samples (default: 2000)

    Returns
    -------
    pd.DataFrame
        Gender comparison results with z-test statistics
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GENDER INDIRECT EFFECT COMPARISON (STRESS)")
        print(f"n_bootstrap = {n_bootstrap}")
        print("=" * 60)

    df = load_path_data()

    # Split by gender
    df_male = df[df['gender_male'] == 1].copy()
    df_female = df[df['gender_male'] == 0].copy()
    n_male, n_female = len(df_male), len(df_female)

    if verbose:
        print(f"\n[DATA] Males: n={n_male}, Females: n={n_female}")

    results = []

    for model_name, model_info in MODEL_SPECS.items():
        if verbose:
            print(f"\n[BOOTSTRAP] {model_name}: {model_info['path']}")

        # Remove gender covariate for within-gender analysis
        spec_no_gender = model_info['spec'].replace('+ gender_male', '')

        # Bootstrap for males
        male_a_boots = []
        male_b_boots = []
        male_indirect_boots = []

        for i in range(n_bootstrap):
            if verbose and (i + 1) % 500 == 0:
                print(f"  Male: {i + 1}/{n_bootstrap}")

            boot_idx = np.random.choice(n_male, size=n_male, replace=True)
            boot_df = df_male.iloc[boot_idx].reset_index(drop=True)

            try:
                result = fit_path_model_semopy(spec_no_gender, boot_df, f"{model_name}_male_boot")
                coefs = extract_path_coefficients(result, model_info)

                if not np.isnan(coefs['a_coef']):
                    male_a_boots.append(coefs['a_coef'])
                if not np.isnan(coefs['b_coef']):
                    male_b_boots.append(coefs['b_coef'])
                if not np.isnan(coefs['indirect_effect']):
                    male_indirect_boots.append(coefs['indirect_effect'])
            except Exception:
                continue

        # Bootstrap for females
        female_a_boots = []
        female_b_boots = []
        female_indirect_boots = []

        for i in range(n_bootstrap):
            if verbose and (i + 1) % 500 == 0:
                print(f"  Female: {i + 1}/{n_bootstrap}")

            boot_idx = np.random.choice(n_female, size=n_female, replace=True)
            boot_df = df_female.iloc[boot_idx].reset_index(drop=True)

            try:
                result = fit_path_model_semopy(spec_no_gender, boot_df, f"{model_name}_female_boot")
                coefs = extract_path_coefficients(result, model_info)

                if not np.isnan(coefs['a_coef']):
                    female_a_boots.append(coefs['a_coef'])
                if not np.isnan(coefs['b_coef']):
                    female_b_boots.append(coefs['b_coef'])
                if not np.isnan(coefs['indirect_effect']):
                    female_indirect_boots.append(coefs['indirect_effect'])
            except Exception:
                continue

        # Convert to arrays
        male_a_boots = np.array(male_a_boots)
        male_b_boots = np.array(male_b_boots)
        male_indirect_boots = np.array(male_indirect_boots)
        female_a_boots = np.array(female_a_boots)
        female_b_boots = np.array(female_b_boots)
        female_indirect_boots = np.array(female_indirect_boots)

        # Compute statistics for each path
        def compute_gender_diff_stats(male_boots, female_boots, path_name):
            """Compute gender difference statistics using z-test."""
            if len(male_boots) < 100 or len(female_boots) < 100:
                return {
                    'path': path_name,
                    'male_mean': np.nan,
                    'female_mean': np.nan,
                    'diff': np.nan,
                    'z_stat': np.nan,
                    'p_value': np.nan,
                }

            male_mean = np.mean(male_boots)
            male_se = np.std(male_boots, ddof=1)
            female_mean = np.mean(female_boots)
            female_se = np.std(female_boots, ddof=1)

            diff = male_mean - female_mean

            # Z-test for difference of means
            se_diff = np.sqrt(male_se**2 + female_se**2)

            if se_diff > 0:
                z_stat = diff / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = np.nan
                p_value = np.nan

            return {
                'path': path_name,
                'male_mean': male_mean,
                'male_se': male_se,
                'male_ci_lower': np.percentile(male_boots, 2.5),
                'male_ci_upper': np.percentile(male_boots, 97.5),
                'female_mean': female_mean,
                'female_se': female_se,
                'female_ci_lower': np.percentile(female_boots, 2.5),
                'female_ci_upper': np.percentile(female_boots, 97.5),
                'diff': diff,
                'se_diff': se_diff,
                'z_stat': z_stat,
                'p_value': p_value,
            }

        # a-path comparison
        a_stats = compute_gender_diff_stats(male_a_boots, female_a_boots, 'a-path')
        a_stats['model'] = model_name
        a_stats['model_path'] = model_info['path']
        a_stats['exog_to_med'] = f"{model_info['exogenous']} → {model_info['mediator']}"
        results.append(a_stats)

        # b-path comparison
        b_stats = compute_gender_diff_stats(male_b_boots, female_b_boots, 'b-path')
        b_stats['model'] = model_name
        b_stats['model_path'] = model_info['path']
        b_stats['exog_to_med'] = f"{model_info['mediator']} → {model_info['endogenous']}"
        results.append(b_stats)

        # indirect effect comparison
        ind_stats = compute_gender_diff_stats(male_indirect_boots, female_indirect_boots, 'indirect')
        ind_stats['model'] = model_name
        ind_stats['model_path'] = model_info['path']
        ind_stats['exog_to_med'] = 'a × b'
        results.append(ind_stats)

        if verbose:
            print(f"\n  {model_name} Gender Differences:")
            print(f"    a-path: Male={a_stats['male_mean']:.3f}, Female={a_stats['female_mean']:.3f}, z={a_stats['z_stat']:.2f}, p={a_stats['p_value']:.4f}")
            print(f"    b-path: Male={b_stats['male_mean']:.3f}, Female={b_stats['female_mean']:.3f}, z={b_stats['z_stat']:.2f}, p={b_stats['p_value']:.4f}")
            print(f"    indirect: Male={ind_stats['male_mean']:.3f}, Female={ind_stats['female_mean']:.3f}, z={ind_stats['z_stat']:.2f}, p={ind_stats['p_value']:.4f}")

            # Flag significant differences
            if a_stats['p_value'] < 0.05:
                print(f"    *** SIGNIFICANT a-path gender difference! ***")
            if ind_stats['p_value'] < 0.05:
                print(f"    *** SIGNIFICANT indirect effect gender difference! ***")

    results_df = pd.DataFrame(results)

    # Reorder columns
    col_order = ['model', 'model_path', 'path', 'exog_to_med',
                 'male_mean', 'male_se', 'male_ci_lower', 'male_ci_upper',
                 'female_mean', 'female_se', 'female_ci_lower', 'female_ci_upper',
                 'diff', 'se_diff', 'z_stat', 'p_value']
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    # Save
    results_df.to_csv(OUTPUT_DIR / "gender_indirect_comparison.csv", index=False, encoding='utf-8-sig')

    # Summary of significant findings
    if verbose:
        sig = results_df[results_df['p_value'] < 0.05]
        if len(sig) > 0:
            print("\n" + "=" * 60)
            print("SIGNIFICANT GENDER DIFFERENCES (p < 0.05)")
            print("=" * 60)
            for _, row in sig.iterrows():
                print(f"\n{row['model']} {row['path']}: {row['exog_to_med']}")
                print(f"  Male: {row['male_mean']:.3f} [{row['male_ci_lower']:.3f}, {row['male_ci_upper']:.3f}]")
                print(f"  Female: {row['female_mean']:.3f} [{row['female_ci_lower']:.3f}, {row['female_ci_upper']:.3f}]")
                print(f"  Difference: {row['diff']:.3f}, z={row['z_stat']:.2f}, p={row['p_value']:.4f}")
        else:
            print("\n[INFO] No significant gender differences found (all p > 0.05)")

    # Create visualization
    plot_gender_indirect_comparison(results_df)

    return results_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_model_comparison(df: pd.DataFrame):
    """Plot fit indices comparison across models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    models = df['model'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)]

    # AIC
    ax = axes[0, 0]
    aic_vals = df['aic'].fillna(0).tolist()
    if df['aic'].notna().any():
        ax.bar(models, aic_vals, color=colors)
        ax.set_title('AIC (lower = better)', fontsize=12, fontweight='bold')
        ax.set_ylabel('AIC')
    else:
        ax.text(0.5, 0.5, 'AIC not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('AIC', fontsize=12)
    ax.tick_params(axis='x', rotation=15)

    # BIC
    ax = axes[0, 1]
    bic_vals = df['bic'].fillna(0).tolist()
    if df['bic'].notna().any():
        ax.bar(models, bic_vals, color=colors)
        ax.set_title('BIC (lower = better)', fontsize=12, fontweight='bold')
        ax.set_ylabel('BIC')
    else:
        ax.text(0.5, 0.5, 'BIC not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('BIC', fontsize=12)
    ax.tick_params(axis='x', rotation=15)

    # CFI (if available)
    ax = axes[1, 0]
    if 'cfi' in df.columns and df['cfi'].notna().any():
        cfi_vals = df['cfi'].fillna(0).tolist()
        ax.bar(models, cfi_vals, color=colors)
        ax.axhline(0.95, color='red', linestyle='--', label='Good fit (0.95)')
        ax.axhline(0.90, color='orange', linestyle='--', label='Acceptable (0.90)')
        ax.set_title('CFI (higher = better)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CFI')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'CFI not available\n(saturated model or OLS)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('CFI', fontsize=12)
    ax.tick_params(axis='x', rotation=15)

    # Indirect Effects
    ax = axes[1, 1]
    indirect_vals = df['indirect'].fillna(0).tolist()
    if df['indirect'].notna().any():
        ax.bar(models, indirect_vals, color=colors)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Indirect Effect (a × b)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Indirect Effect')
    else:
        ax.text(0.5, 0.5, 'Indirect effects not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Indirect Effect', fontsize=12)
    ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Path Model Comparison (STRESS)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {VIS_DIR / 'model_comparison.png'}")


def plot_indirect_effects(df: pd.DataFrame):
    """Plot bootstrap CIs for indirect effects."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(df))

    # Plot point estimates and CIs
    for i, (_, row) in enumerate(df.iterrows()):
        color = 'darkgreen' if row['significant_95'] else 'gray'

        # CI bar
        ax.plot([row['ci_lower_95'], row['ci_upper_95']], [i, i],
                color=color, linewidth=2, solid_capstyle='butt')

        # Point estimate
        ax.scatter(row['indirect_point'], i, color=color, s=100, zorder=5)

    # Reference line at 0
    ax.axvline(0, color='red', linestyle='--', linewidth=1)

    # Labels
    ax.set_yticks(y_pos)
    labels = [f"{row['model']}\n{row['path']}" for _, row in df.iterrows()]
    ax.set_yticklabels(labels)

    ax.set_xlabel('Indirect Effect (a × b)', fontsize=12)
    ax.set_title('Bootstrap 95% CIs for Indirect Effects (STRESS)\n(Green = significant, Gray = non-significant)',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIS_DIR / "indirect_effects_ci.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {VIS_DIR / 'indirect_effects_ci.png'}")


def plot_gender_comparison(df: pd.DataFrame):
    """Plot gender-stratified path coefficients."""
    n_models = len(MODEL_SPECS)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for i, model_name in enumerate(MODEL_SPECS.keys()):
        ax = axes[i]

        male_rows = df[(df['model'] == model_name) & (df['gender'] == 'male')]
        female_rows = df[(df['model'] == model_name) & (df['gender'] == 'female')]

        if len(male_rows) == 0 or len(female_rows) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        male = male_rows.iloc[0]
        female = female_rows.iloc[0]

        x = np.arange(3)  # a-path, b-path, indirect
        width = 0.35

        male_vals = [male['a_path'], male['b_path'], male['indirect']]
        female_vals = [female['a_path'], female['b_path'], female['indirect']]

        ax.bar(x - width/2, male_vals, width, label='Male', color='#1f77b4')
        ax.bar(x + width/2, female_vals, width, label='Female', color='#ff7f0e')

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['a-path', 'b-path', 'Indirect'])
        ax.set_title(f"{model_name}\n{MODEL_SPECS[model_name]['path']}", fontsize=10, fontweight='bold')
        ax.legend()

    plt.suptitle('Gender-Stratified Path Coefficients (STRESS)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "gender_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {VIS_DIR / 'gender_comparison.png'}")


def plot_gender_indirect_comparison(df: pd.DataFrame):
    """Plot bootstrap gender comparison results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: All models, a-path only
    ax = axes[0]
    a_path_df = df[df['path'] == 'a-path'].copy()

    if len(a_path_df) > 0:
        y_pos = np.arange(len(a_path_df))

        for i, (_, row) in enumerate(a_path_df.iterrows()):
            # Male CI
            ax.plot([row['male_ci_lower'], row['male_ci_upper']], [i - 0.15, i - 0.15],
                    color='#1f77b4', linewidth=2, solid_capstyle='butt')
            ax.scatter(row['male_mean'], i - 0.15, color='#1f77b4', s=80, zorder=5, label='Male' if i == 0 else '')

            # Female CI
            ax.plot([row['female_ci_lower'], row['female_ci_upper']], [i + 0.15, i + 0.15],
                    color='#ff7f0e', linewidth=2, solid_capstyle='butt')
            ax.scatter(row['female_mean'], i + 0.15, color='#ff7f0e', s=80, zorder=5, label='Female' if i == 0 else '')

            # Add p-value annotation
            if row['p_value'] < 0.05:
                ax.annotate(f"p={row['p_value']:.3f}*", (max(row['male_ci_upper'], row['female_ci_upper']) + 0.05, i),
                           fontsize=8, color='red')

        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['model']}\n{row['model_path']}" for _, row in a_path_df.iterrows()])
        ax.set_xlabel('a-path Coefficient', fontsize=12)
        ax.set_title('a-path: Gender Comparison\n(Exogenous -> Mediator)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')

    # Right: Indirect effects comparison
    ax = axes[1]
    ind_df = df[df['path'] == 'indirect'].copy()

    if len(ind_df) > 0:
        y_pos = np.arange(len(ind_df))

        for i, (_, row) in enumerate(ind_df.iterrows()):
            # Male CI
            ax.plot([row['male_ci_lower'], row['male_ci_upper']], [i - 0.15, i - 0.15],
                    color='#1f77b4', linewidth=2, solid_capstyle='butt')
            ax.scatter(row['male_mean'], i - 0.15, color='#1f77b4', s=80, zorder=5)

            # Female CI
            ax.plot([row['female_ci_lower'], row['female_ci_upper']], [i + 0.15, i + 0.15],
                    color='#ff7f0e', linewidth=2, solid_capstyle='butt')
            ax.scatter(row['female_mean'], i + 0.15, color='#ff7f0e', s=80, zorder=5)

            # Add p-value annotation
            if row['p_value'] < 0.05:
                ax.annotate(f"p={row['p_value']:.3f}*", (max(row['male_ci_upper'], row['female_ci_upper']) + 0.02, i),
                           fontsize=8, color='red')

        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['model']}\n{row['model_path']}" for _, row in ind_df.iterrows()])
        ax.set_xlabel('Indirect Effect (a x b)', fontsize=12)
        ax.set_title('Indirect Effects: Gender Comparison', fontsize=12, fontweight='bold')

    plt.suptitle('Bootstrap Gender Comparison - STRESS (Blue=Male, Orange=Female)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "gender_indirect_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {VIS_DIR / 'gender_indirect_comparison.png'}")


def generate_summary_report(verbose: bool = True):
    """Generate human-readable summary report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PATH MODEL COMPARISON SUMMARY (STRESS)")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("연구 질문: 외로움, 스트레스, 인지기능 간의 인과 구조 중")
    report_lines.append("          어떤 모형이 데이터에 가장 적합한가?")
    report_lines.append("")
    report_lines.append("비교 모형:")
    report_lines.append("  1) C → L → S: 인지기능 → 외로움 → 스트레스")
    report_lines.append("  2) C → S → L: 인지기능 → 스트레스 → 외로움")
    report_lines.append("  3) L → S → C: 외로움 → 스트레스 → 인지기능")
    report_lines.append("  4) L → C → S: 외로움 → 인지기능 → 스트레스")
    report_lines.append("")

    # Load results
    try:
        comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
        indirect = pd.read_csv(OUTPUT_DIR / "indirect_effects.csv")

        report_lines.append("-" * 70)
        report_lines.append("모형 적합도 비교")
        report_lines.append("-" * 70)

        for _, row in comparison.iterrows():
            report_lines.append(f"\n{row['model']}: {row['path']}")
            report_lines.append(f"  AIC: {row['aic']:.2f}, BIC: {row['bic']:.2f}")
            if pd.notna(row['cfi']):
                report_lines.append(f"  CFI: {row['cfi']:.3f}, RMSEA: {row['rmsea']:.3f}")
            report_lines.append(f"  간접효과: {row['indirect']:.4f}")

        best = comparison.iloc[0]
        report_lines.append(f"\n** 최적 모형: {best['model']} ({best['path']})")

        report_lines.append("\n" + "-" * 70)
        report_lines.append("부트스트랩 간접효과 (95% CI)")
        report_lines.append("-" * 70)

        for _, row in indirect.iterrows():
            sig = "유의" if row['significant_95'] else "비유의"
            report_lines.append(f"\n{row['model']}: {row['indirect_point']:.4f} [{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] ({sig})")

        report_lines.append("\n" + "-" * 70)
        report_lines.append("주의사항")
        report_lines.append("-" * 70)
        report_lines.append("- 횡단 데이터로 인과 방향 확정 불가")
        report_lines.append("- 네 모형은 관찰적으로 동치 - 적합도 차이는 상대적 해석만 가능")
        report_lines.append("- 표본 크기가 SEM에 다소 제한적")

    except FileNotFoundError:
        report_lines.append("[분석 결과 파일 없음 - 먼저 분석을 실행하세요]")

    report_text = "\n".join(report_lines)

    with open(OUTPUT_DIR / "summary_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)

    if verbose:
        print(report_text)

    print(f"\n[SAVED] {OUTPUT_DIR / 'summary_report.txt'}")


# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True, **kwargs) -> Dict[str, Any]:
    """Run path comparison suite (Stress)."""
    results = {}

    if analysis:
        if analysis not in ANALYSES:
            available = ', '.join(ANALYSES.keys())
            raise ValueError(f"Unknown analysis: {analysis}. Available: {available}")

        spec = ANALYSES[analysis]
        if verbose:
            print(f"\n[RUNNING] {spec.name}: {spec.description}")

        if analysis == 'indirect_effects':
            n_boot = kwargs.get('n_bootstrap', 5000)
            results[analysis] = spec.function(verbose=verbose, n_bootstrap=n_boot)
        else:
            results[analysis] = spec.function(verbose=verbose)
    else:
        order = ['path_comparison', 'indirect_effects', 'gender_moderated', 'gender_indirect_comparison']
        for name in order:
            if name in ANALYSES:
                spec = ANALYSES[name]
                if verbose:
                    print(f"\n[RUNNING] {spec.name}: {spec.description}")
                try:
                    if name == 'indirect_effects':
                        n_boot = kwargs.get('n_bootstrap', 5000)
                        results[name] = spec.function(verbose=verbose, n_bootstrap=n_boot)
                    elif name == 'gender_indirect_comparison':
                        n_boot = kwargs.get('n_bootstrap', 2000)
                        results[name] = spec.function(verbose=verbose, n_bootstrap=n_boot)
                    else:
                        results[name] = spec.function(verbose=verbose)
                except Exception as e:
                    print(f"[ERROR] {name}: {e}")
                    results[name] = None

        # Generate summary report after all analyses
        generate_summary_report(verbose=verbose)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable analyses:")
    for name, spec in ANALYSES.items():
        print(f"  {name}: {spec.description}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path Model Comparison Suite (Stress)')
    parser.add_argument('--analysis', type=str, help='Specific analysis to run')
    parser.add_argument('--list', action='store_true', help='List available analyses')
    parser.add_argument('--n-bootstrap', type=int, default=5000, help='Bootstrap iterations')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet, n_bootstrap=args.n_bootstrap)
