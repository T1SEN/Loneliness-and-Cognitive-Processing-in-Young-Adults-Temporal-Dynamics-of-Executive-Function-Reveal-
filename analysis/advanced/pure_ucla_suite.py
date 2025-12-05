"""
Pure UCLA Effect Analysis Suite
================================

DASS와 독립적인 "순수 외로움" 효과 분석.

Analyses:
1. residualized_ucla: DASS 분산 제거 후 순수 UCLA 효과
2. subscale_comparison: DASS 하위척도 개별 통제 비교
3. bayesian_comparison: Bayesian 모델 비교 (UCLA 효과 유무)
4. effect_decomposition: UCLA 효과 분해 (DASS 공유 vs 고유)

Usage:
    python -m analysis.advanced.pure_ucla_suite                    # Run all
    python -m analysis.advanced.pure_ucla_suite --analysis residualized_ucla
    python -m analysis.advanced.pure_ucla_suite --list

    from analysis.advanced import pure_ucla_suite
    pure_ucla_suite.run('residualized_ucla')
    pure_ucla_suite.run()  # All analyses

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
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "pure_ucla"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable
    priority: int


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str, priority: int = 1):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            priority=priority
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_analysis_data() -> pd.DataFrame:
    """Load and prepare master dataset."""
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


# =============================================================================
# EF OUTCOMES
# =============================================================================

EF_OUTCOMES = [
    ('wcst_pe_rate', 'WCST Perseverative Error Rate'),
    ('wcst_accuracy', 'WCST Accuracy'),
    ('stroop_interference', 'Stroop Interference Effect'),
    ('prp_bottleneck', 'PRP Bottleneck Effect'),
]


def get_available_outcomes(df: pd.DataFrame) -> List[tuple]:
    """Get list of available EF outcomes in the dataset."""
    available = []
    for col, label in EF_OUTCOMES:
        if col in df.columns and df[col].notna().sum() > 20:
            available.append((col, label))
    return available


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="residualized_ucla",
    description="Test UCLA effect after residualizing out DASS variance",
    priority=1
)
def analyze_residualized_ucla(verbose: bool = True) -> pd.DataFrame:
    """
    Residualize UCLA by DASS to get "pure loneliness" predictor.

    1. UCLA_pure = residual(UCLA ~ DASS_dep + DASS_anx + DASS_str)
    2. Test UCLA_pure effects on EF outcomes
    3. Compare to original UCLA effects

    If UCLA_pure is significant, loneliness has unique effect beyond mood.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RESIDUALIZED UCLA ANALYSIS")
        print("=" * 70)
        print("\n  Creating UCLA_pure by regressing out DASS variance...")

    df = load_analysis_data()

    # Check required columns
    required = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if verbose:
            print(f"  Missing columns: {missing}")
        return pd.DataFrame()

    # Residualize UCLA
    ucla_formula = "ucla_total ~ dass_depression + dass_anxiety + dass_stress"
    ucla_model = smf.ols(ucla_formula, data=df).fit()

    df['ucla_residual'] = ucla_model.resid
    df['z_ucla_pure'] = (df['ucla_residual'] - df['ucla_residual'].mean()) / df['ucla_residual'].std()

    # Calculate variance explained
    ucla_var_total = df['ucla_total'].var()
    ucla_var_residual = df['ucla_residual'].var()
    dass_explained_pct = (1 - ucla_var_residual / ucla_var_total) * 100

    if verbose:
        print(f"\n  UCLA-DASS Regression:")
        print(f"    R² = {ucla_model.rsquared:.3f}")
        print(f"    DASS explains {dass_explained_pct:.1f}% of UCLA variance")
        print(f"    UCLA_pure retains {100 - dass_explained_pct:.1f}% of variance")

    # Test both original and pure UCLA on EF outcomes
    outcomes = get_available_outcomes(df)

    if len(outcomes) == 0:
        if verbose:
            print("  No EF outcomes available")
        return pd.DataFrame()

    all_results = []

    if verbose:
        print(f"\n  Comparing UCLA vs UCLA_pure effects:")
        print(f"    (Controlling for gender and age)")

    for outcome_col, outcome_label in outcomes:
        outcome_df = df.dropna(subset=[outcome_col])

        if len(outcome_df) < 30:
            continue

        # Original UCLA (no DASS control - for comparison)
        formula_orig = f"{outcome_col} ~ z_ucla * C(gender_male) + z_age"
        model_orig = smf.ols(formula_orig, data=outcome_df).fit(cov_type='HC3')

        # UCLA with DASS control (standard)
        formula_dass = f"{outcome_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model_dass = smf.ols(formula_dass, data=outcome_df).fit(cov_type='HC3')

        # UCLA_pure (residualized)
        formula_pure = f"{outcome_col} ~ z_ucla_pure * C(gender_male) + z_age"
        model_pure = smf.ols(formula_pure, data=outcome_df).fit(cov_type='HC3')

        result = {
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': len(outcome_df),
            'dass_explains_pct': dass_explained_pct,
        }

        # Extract coefficients for each model
        for model, prefix in [(model_orig, 'ucla_orig'), (model_dass, 'ucla_dass'), (model_pure, 'ucla_pure')]:
            ucla_param = 'z_ucla' if 'ucla' in prefix and 'pure' not in prefix else 'z_ucla_pure'
            if ucla_param in model.params:
                result[f'{prefix}_beta'] = model.params[ucla_param]
                result[f'{prefix}_se'] = model.bse[ucla_param]
                result[f'{prefix}_p'] = model.pvalues[ucla_param]
                result[f'{prefix}_r2'] = model.rsquared

        # Effect size change
        if 'ucla_orig_beta' in result and 'ucla_dass_beta' in result:
            result['beta_reduction_pct'] = (1 - abs(result['ucla_dass_beta']) / abs(result['ucla_orig_beta'])) * 100 if result['ucla_orig_beta'] != 0 else 0

        all_results.append(result)

        if verbose:
            print(f"\n    {outcome_label}:")
            orig_sig = "*" if result.get('ucla_orig_p', 1) < 0.05 else ""
            dass_sig = "*" if result.get('ucla_dass_p', 1) < 0.05 else ""
            pure_sig = "*" if result.get('ucla_pure_p', 1) < 0.05 else ""
            print(f"      UCLA (no DASS control): β={result.get('ucla_orig_beta', np.nan):.4f}, p={result.get('ucla_orig_p', np.nan):.4f} {orig_sig}")
            print(f"      UCLA (DASS controlled): β={result.get('ucla_dass_beta', np.nan):.4f}, p={result.get('ucla_dass_p', np.nan):.4f} {dass_sig}")
            print(f"      UCLA_pure (residual):   β={result.get('ucla_pure_beta', np.nan):.4f}, p={result.get('ucla_pure_p', np.nan):.4f} {pure_sig}")

    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "residualized_effects.csv", index=False, encoding='utf-8-sig')

    # Save UCLA residualization model
    with open(OUTPUT_DIR / "ucla_residualization.txt", 'w', encoding='utf-8') as f:
        f.write("UCLA RESIDUALIZATION (UCLA ~ DASS)\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(ucla_model.summary()))

    # Visualization
    _plot_residualized_comparison(results_df)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'residualized_effects.csv'}")
        print(f"  Figure: {FIGURES_DIR / 'residualized_comparison.png'}")

    return results_df


def _plot_residualized_comparison(df: pd.DataFrame):
    """Plot comparison of UCLA effects: Original vs DASS-controlled vs Pure."""
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    outcomes = df['outcome_label'].values
    x_pos = np.arange(len(outcomes))
    width = 0.25

    # Extract betas
    orig_betas = df['ucla_orig_beta'].values
    dass_betas = df['ucla_dass_beta'].values
    pure_betas = df['ucla_pure_beta'].values

    # Plot bars
    ax.bar(x_pos - width, orig_betas, width, label='UCLA (no control)', color='#3498db', alpha=0.8)
    ax.bar(x_pos, dass_betas, width, label='UCLA (DASS covariate)', color='#e74c3c', alpha=0.8)
    ax.bar(x_pos + width, pure_betas, width, label='UCLA_pure (residual)', color='#2ecc71', alpha=0.8)

    # Add significance markers
    for i, (orig_p, dass_p, pure_p) in enumerate(zip(df['ucla_orig_p'], df['ucla_dass_p'], df['ucla_pure_p'])):
        y_max = max(abs(orig_betas[i]), abs(dass_betas[i]), abs(pure_betas[i]))
        if orig_p < 0.05:
            ax.text(i - width, orig_betas[i] + 0.01 * np.sign(orig_betas[i]), '*', ha='center', fontsize=14)
        if dass_p < 0.05:
            ax.text(i, dass_betas[i] + 0.01 * np.sign(dass_betas[i]), '*', ha='center', fontsize=14)
        if pure_p < 0.05:
            ax.text(i + width, pure_betas[i] + 0.01 * np.sign(pure_betas[i]), '*', ha='center', fontsize=14)

    ax.set_ylabel('UCLA β', fontsize=12)
    ax.set_title('UCLA Effect Comparison: Original vs DASS-Controlled vs Pure', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([o.replace(' ', '\n') for o in outcomes], fontsize=9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residualized_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


@register_analysis(
    name="subscale_comparison",
    description="Compare UCLA effects with each DASS subscale controlled separately",
    priority=1
)
def analyze_subscale_comparison(verbose: bool = True) -> pd.DataFrame:
    """
    Test which DASS subscale most attenuates UCLA effect.

    Models:
    1. No control: UCLA only
    2. Depression only: UCLA + DASS_dep
    3. Anxiety only: UCLA + DASS_anx
    4. Stress only: UCLA + DASS_str
    5. Full DASS: UCLA + DASS_dep + DASS_anx + DASS_str

    Answer: Which subscale "explains away" the most UCLA variance?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DASS SUBSCALE COMPARISON ANALYSIS")
        print("=" * 70)

    df = load_analysis_data()

    outcomes = get_available_outcomes(df)

    if len(outcomes) == 0:
        if verbose:
            print("  No EF outcomes available")
        return pd.DataFrame()

    all_results = []

    model_specs = {
        'no_control': "{outcome} ~ z_ucla * C(gender_male) + z_age",
        'dep_only': "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_age",
        'anx_only': "{outcome} ~ z_ucla * C(gender_male) + z_dass_anx + z_age",
        'str_only': "{outcome} ~ z_ucla * C(gender_male) + z_dass_str + z_age",
        'full_dass': "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
    }

    if verbose:
        print("\n  Testing UCLA effects with different DASS controls:")

    for outcome_col, outcome_label in outcomes:
        outcome_df = df.dropna(subset=[outcome_col])

        if len(outcome_df) < 30:
            continue

        result = {
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': len(outcome_df)
        }

        no_control_beta = None

        for model_name, formula_template in model_specs.items():
            formula = formula_template.format(outcome=outcome_col)

            try:
                model = smf.ols(formula, data=outcome_df).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    result[f'{model_name}_beta'] = beta
                    result[f'{model_name}_p'] = p
                    result[f'{model_name}_r2'] = model.rsquared

                    if model_name == 'no_control':
                        no_control_beta = beta

            except Exception:
                pass

        # Calculate attenuation percentages
        if no_control_beta is not None and no_control_beta != 0:
            for model_name in ['dep_only', 'anx_only', 'str_only', 'full_dass']:
                if f'{model_name}_beta' in result:
                    attenuation = (1 - abs(result[f'{model_name}_beta']) / abs(no_control_beta)) * 100
                    result[f'{model_name}_attenuation_pct'] = attenuation

        all_results.append(result)

        if verbose:
            print(f"\n    {outcome_label}:")
            for model_name in model_specs.keys():
                if f'{model_name}_beta' in result:
                    beta = result[f'{model_name}_beta']
                    p = result[f'{model_name}_p']
                    sig = "*" if p < 0.05 else ""
                    atten = f" ({result.get(f'{model_name}_attenuation_pct', 0):.1f}% attenuation)" if model_name != 'no_control' else ""
                    print(f"      {model_name}: β={beta:.4f}, p={p:.4f} {sig}{atten}")

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "subscale_comparison.csv", index=False, encoding='utf-8-sig')

        # Find which subscale causes most attenuation
        attenuation_cols = [c for c in results_df.columns if 'attenuation_pct' in c]
        if attenuation_cols:
            mean_attenuations = {col.replace('_attenuation_pct', ''): results_df[col].mean()
                                for col in attenuation_cols}

            if verbose:
                print(f"\n  Mean Attenuation by DASS Subscale:")
                for subscale, atten in sorted(mean_attenuations.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {subscale}: {atten:.1f}%")

        # Visualization
        _plot_subscale_comparison(results_df)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'subscale_comparison.csv'}")

    return results_df


def _plot_subscale_comparison(df: pd.DataFrame):
    """Plot DASS subscale comparison."""
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('DASS Subscale Effect on UCLA', fontsize=14, fontweight='bold')

    # Panel 1: Beta coefficients by control type
    ax1 = axes[0]
    outcomes = df['outcome_label'].values
    x_pos = np.arange(len(outcomes))
    width = 0.15

    controls = ['no_control', 'dep_only', 'anx_only', 'str_only', 'full_dass']
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']

    for i, (control, color) in enumerate(zip(controls, colors)):
        col = f'{control}_beta'
        if col in df.columns:
            betas = df[col].values
            ax1.bar(x_pos + i * width, betas, width, label=control.replace('_', ' ').title(), color=color, alpha=0.8)

    ax1.set_ylabel('UCLA β')
    ax1.set_title('UCLA Effect by DASS Control')
    ax1.set_xticks(x_pos + width * 2)
    ax1.set_xticklabels([o[:15] + '...' if len(o) > 15 else o for o in outcomes], rotation=15)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Attenuation percentages
    ax2 = axes[1]
    attenuation_data = []
    for control in ['dep_only', 'anx_only', 'str_only', 'full_dass']:
        col = f'{control}_attenuation_pct'
        if col in df.columns:
            attenuation_data.append({
                'control': control.replace('_only', '').replace('_', ' ').title(),
                'mean_attenuation': df[col].mean(),
                'sem': df[col].sem()
            })

    if attenuation_data:
        atten_df = pd.DataFrame(attenuation_data)
        bars = ax2.bar(atten_df['control'], atten_df['mean_attenuation'],
                       yerr=atten_df['sem'], capsize=5, color=['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71'])
        ax2.set_ylabel('Mean % Attenuation of UCLA Effect')
        ax2.set_title('Which DASS Subscale Attenuates UCLA Most?')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "subscale_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


@register_analysis(
    name="effect_decomposition",
    description="Decompose UCLA effect into DASS-shared vs unique components",
    priority=1
)
def analyze_effect_decomposition(verbose: bool = True) -> pd.DataFrame:
    """
    Decompose UCLA effect using commonality analysis.

    Components:
    1. UCLA unique: Variance explained by UCLA alone (after DASS)
    2. DASS unique: Variance explained by DASS alone (after UCLA)
    3. Shared: Variance explained by both UCLA and DASS
    """
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA EFFECT DECOMPOSITION")
        print("=" * 70)

    df = load_analysis_data()

    outcomes = get_available_outcomes(df)

    if len(outcomes) == 0:
        if verbose:
            print("  No EF outcomes available")
        return pd.DataFrame()

    all_results = []

    if verbose:
        print("\n  Decomposing variance into UCLA-unique, DASS-unique, and shared:")

    for outcome_col, outcome_label in outcomes:
        outcome_df = df.dropna(subset=[outcome_col]).copy()

        if len(outcome_df) < 30:
            continue

        # Base model (gender + age only)
        formula_base = f"{outcome_col} ~ C(gender_male) + z_age"
        model_base = smf.ols(formula_base, data=outcome_df).fit()
        r2_base = model_base.rsquared

        # UCLA only
        formula_ucla = f"{outcome_col} ~ z_ucla + C(gender_male) + z_age"
        model_ucla = smf.ols(formula_ucla, data=outcome_df).fit()
        r2_ucla = model_ucla.rsquared

        # DASS only
        formula_dass = f"{outcome_col} ~ z_dass_dep + z_dass_anx + z_dass_str + C(gender_male) + z_age"
        model_dass = smf.ols(formula_dass, data=outcome_df).fit()
        r2_dass = model_dass.rsquared

        # Full model
        formula_full = f"{outcome_col} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + C(gender_male) + z_age"
        model_full = smf.ols(formula_full, data=outcome_df).fit()
        r2_full = model_full.rsquared

        # Commonality analysis
        ucla_increment = r2_ucla - r2_base  # UCLA contribution over base
        dass_increment = r2_dass - r2_base  # DASS contribution over base
        full_increment = r2_full - r2_base  # Full model contribution over base

        # Unique contributions
        ucla_unique = r2_full - r2_dass  # UCLA adds to DASS model
        dass_unique = r2_full - r2_ucla  # DASS adds to UCLA model
        shared = ucla_increment + dass_increment - full_increment  # Shared variance

        # Total variance explained (R² of full model)
        total_explained = full_increment

        result = {
            'outcome': outcome_col,
            'outcome_label': outcome_label,
            'n': len(outcome_df),
            'r2_base': r2_base,
            'r2_ucla': r2_ucla,
            'r2_dass': r2_dass,
            'r2_full': r2_full,
            'ucla_increment': ucla_increment,
            'dass_increment': dass_increment,
            'ucla_unique': ucla_unique,
            'dass_unique': dass_unique,
            'shared': max(0, shared),  # Can be slightly negative due to rounding
            'total_explained': total_explained,
            # Proportions (of total explained)
            'ucla_unique_pct': ucla_unique / total_explained * 100 if total_explained > 0 else 0,
            'dass_unique_pct': dass_unique / total_explained * 100 if total_explained > 0 else 0,
            'shared_pct': max(0, shared) / total_explained * 100 if total_explained > 0 else 0,
        }

        all_results.append(result)

        if verbose:
            print(f"\n    {outcome_label}:")
            print(f"      Total R² beyond base: {total_explained*100:.2f}%")
            print(f"      UCLA unique: {ucla_unique*100:.3f}% ({result['ucla_unique_pct']:.1f}% of explained)")
            print(f"      DASS unique: {dass_unique*100:.3f}% ({result['dass_unique_pct']:.1f}% of explained)")
            print(f"      Shared: {max(0, shared)*100:.3f}% ({result['shared_pct']:.1f}% of explained)")

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "effect_decomposition.csv", index=False, encoding='utf-8-sig')

        # Summary statistics
        if verbose:
            print(f"\n  Overall Summary:")
            print(f"    Mean UCLA unique: {results_df['ucla_unique_pct'].mean():.1f}%")
            print(f"    Mean DASS unique: {results_df['dass_unique_pct'].mean():.1f}%")
            print(f"    Mean Shared: {results_df['shared_pct'].mean():.1f}%")

        # Visualization
        _plot_decomposition(results_df)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'effect_decomposition.csv'}")

    return results_df


def _plot_decomposition(df: pd.DataFrame):
    """Plot variance decomposition."""
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('UCLA-DASS Variance Decomposition', fontsize=14, fontweight='bold')

    # Panel 1: Stacked bar chart
    ax1 = axes[0]
    outcomes = df['outcome_label'].values
    x_pos = np.arange(len(outcomes))

    ucla_unique = df['ucla_unique_pct'].values
    dass_unique = df['dass_unique_pct'].values
    shared = df['shared_pct'].values

    ax1.bar(x_pos, ucla_unique, label='UCLA Unique', color='#3498db')
    ax1.bar(x_pos, shared, bottom=ucla_unique, label='Shared', color='#f39c12')
    ax1.bar(x_pos, dass_unique, bottom=ucla_unique + shared, label='DASS Unique', color='#e74c3c')

    ax1.set_ylabel('% of Explained Variance')
    ax1.set_title('Variance Decomposition by Outcome')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([o[:12] + '...' if len(o) > 12 else o for o in outcomes], rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Pie chart of overall mean
    ax2 = axes[1]
    mean_ucla = df['ucla_unique_pct'].mean()
    mean_dass = df['dass_unique_pct'].mean()
    mean_shared = df['shared_pct'].mean()

    sizes = [mean_ucla, mean_shared, mean_dass]
    labels = [f'UCLA Unique\n({mean_ucla:.1f}%)',
              f'Shared\n({mean_shared:.1f}%)',
              f'DASS Unique\n({mean_dass:.1f}%)']
    colors = ['#3498db', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
            startangle=90, shadow=True)
    ax2.set_title('Mean Variance Decomposition\nAcross All Outcomes')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "decomposition.png", dpi=300, bbox_inches='tight')
    plt.close()


@register_analysis(
    name="bayesian_comparison",
    description="Bayesian model comparison for UCLA effect (requires PyMC)",
    priority=2
)
def analyze_bayesian_comparison(verbose: bool = True) -> pd.DataFrame:
    """
    Bayesian model comparison to test evidence for UCLA effect.

    Compares:
    - Null model: No UCLA effect
    - UCLA model: UCLA effect present

    Uses WAIC/LOO for model comparison.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BAYESIAN MODEL COMPARISON")
        print("=" * 70)

    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        if verbose:
            print("  PyMC or ArviZ not available - skipping Bayesian analysis")
            print("  Install with: pip install pymc arviz")
        return pd.DataFrame()

    df = load_analysis_data()

    outcomes = get_available_outcomes(df)

    if len(outcomes) == 0:
        if verbose:
            print("  No EF outcomes available")
        return pd.DataFrame()

    all_results = []

    for outcome_col, outcome_label in outcomes[:2]:  # Limit to first 2 for speed
        outcome_df = df.dropna(subset=[outcome_col, 'ucla_total', 'gender_male', 'age']).copy()

        if len(outcome_df) < 50:
            continue

        if verbose:
            print(f"\n  {outcome_label} (N={len(outcome_df)}):")
            print(f"    Fitting Bayesian models...")

        # Standardize outcome
        y = outcome_df[outcome_col].values
        y_std = (y - y.mean()) / y.std()

        # Predictors
        x_ucla = outcome_df['z_ucla'].values
        x_gender = outcome_df['gender_male'].values
        x_age = outcome_df['z_age'].values
        x_dass = outcome_df[['z_dass_dep', 'z_dass_anx', 'z_dass_str']].values

        try:
            # Model 1: Null (no UCLA effect)
            with pm.Model() as null_model:
                # Priors
                intercept = pm.Normal('intercept', 0, 1)
                beta_gender = pm.Normal('beta_gender', 0, 1)
                beta_age = pm.Normal('beta_age', 0, 1)
                beta_dass = pm.Normal('beta_dass', 0, 1, shape=3)
                sigma = pm.HalfNormal('sigma', 1)

                # Likelihood
                mu = intercept + beta_gender * x_gender + beta_age * x_age + pm.math.dot(x_dass, beta_dass)
                y_obs = pm.Normal('y', mu=mu, sigma=sigma, observed=y_std)

                # Sample
                trace_null = pm.sample(1000, tune=500, cores=1, progressbar=False, random_seed=42)

            # Model 2: UCLA effect
            with pm.Model() as ucla_model:
                # Priors
                intercept = pm.Normal('intercept', 0, 1)
                beta_ucla = pm.Normal('beta_ucla', 0, 1)
                beta_gender = pm.Normal('beta_gender', 0, 1)
                beta_age = pm.Normal('beta_age', 0, 1)
                beta_dass = pm.Normal('beta_dass', 0, 1, shape=3)
                sigma = pm.HalfNormal('sigma', 1)

                # Likelihood
                mu = intercept + beta_ucla * x_ucla + beta_gender * x_gender + beta_age * x_age + pm.math.dot(x_dass, beta_dass)
                y_obs = pm.Normal('y', mu=mu, sigma=sigma, observed=y_std)

                # Sample
                trace_ucla = pm.sample(1000, tune=500, cores=1, progressbar=False, random_seed=42)

            # Model comparison using LOO
            compare = az.compare({'null': trace_null, 'ucla': trace_ucla}, ic='loo')

            # Extract comparison metrics
            best_model = compare.index[0]
            loo_diff = compare.loc['ucla', 'd_loo'] if best_model == 'null' else -compare.loc['null', 'd_loo']

            # UCLA posterior
            ucla_posterior = trace_ucla.posterior['beta_ucla'].values.flatten()
            ucla_mean = ucla_posterior.mean()
            ucla_hdi = az.hdi(trace_ucla.posterior['beta_ucla'], hdi_prob=0.95)
            ucla_hdi_lower = float(ucla_hdi.values[0])
            ucla_hdi_upper = float(ucla_hdi.values[1])

            # Probability of direction
            prob_positive = (ucla_posterior > 0).mean()
            prob_direction = max(prob_positive, 1 - prob_positive)

            result = {
                'outcome': outcome_col,
                'outcome_label': outcome_label,
                'n': len(outcome_df),
                'best_model': best_model,
                'loo_difference': loo_diff,
                'ucla_posterior_mean': ucla_mean,
                'ucla_hdi_lower': ucla_hdi_lower,
                'ucla_hdi_upper': ucla_hdi_upper,
                'prob_direction': prob_direction,
                'ucla_sig_bayesian': 0 > ucla_hdi_lower or 0 < ucla_hdi_upper  # HDI excludes 0
            }

            all_results.append(result)

            if verbose:
                print(f"    Best model: {best_model} (ΔLOO = {loo_diff:.2f})")
                print(f"    UCLA posterior: {ucla_mean:.4f} [95% HDI: {ucla_hdi_lower:.4f}, {ucla_hdi_upper:.4f}]")
                print(f"    P(direction): {prob_direction:.3f}")

            # Save trace summary
            summary = az.summary(trace_ucla)
            summary.to_csv(OUTPUT_DIR / f"bayesian_{outcome_col}_summary.csv", encoding='utf-8-sig')

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "bayesian_comparison.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'bayesian_comparison.csv'}")

    return results_df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run pure UCLA effect analyses.
    """
    if verbose:
        print("=" * 70)
        print("PURE UCLA EFFECT ANALYSIS SUITE")
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
        for priority in [1, 2, 3]:
            priority_analyses = {k: v for k, v in ANALYSES.items() if v.priority == priority}
            for name, spec in priority_analyses.items():
                try:
                    results[name] = spec.function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    # Generate summary
    if verbose:
        _generate_summary_report(results)

    if verbose:
        print("\n" + "=" * 70)
        print("PURE UCLA SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def _generate_summary_report(results: Dict[str, pd.DataFrame]):
    """Generate summary report."""
    lines = [
        "=" * 80,
        "PURE UCLA EFFECT ANALYSIS - SUMMARY REPORT",
        "=" * 80,
        "",
        "KEY QUESTION: Does UCLA have effects independent of DASS?",
        "",
    ]

    # Check residualized results
    if 'residualized_ucla' in results and len(results['residualized_ucla']) > 0:
        res_df = results['residualized_ucla']
        lines.append("RESIDUALIZED UCLA RESULTS:")
        for _, row in res_df.iterrows():
            pure_p = row.get('ucla_pure_p', np.nan)
            sig = "*" if pure_p < 0.05 else ""
            lines.append(f"  {row['outcome_label']}: UCLA_pure β={row.get('ucla_pure_beta', np.nan):.4f}, p={pure_p:.4f} {sig}")

    # Check decomposition
    if 'effect_decomposition' in results and len(results['effect_decomposition']) > 0:
        dec_df = results['effect_decomposition']
        lines.append("\nVARIANCE DECOMPOSITION (mean across outcomes):")
        lines.append(f"  UCLA Unique: {dec_df['ucla_unique_pct'].mean():.1f}%")
        lines.append(f"  DASS Unique: {dec_df['dass_unique_pct'].mean():.1f}%")
        lines.append(f"  Shared: {dec_df['shared_pct'].mean():.1f}%")

    lines.extend([
        "",
        "=" * 80,
        "INTERPRETATION:",
        "",
        "If UCLA_pure shows significant effects, loneliness has unique variance",
        "beyond general emotional distress (DASS).",
        "",
        "If UCLA_unique is substantial (>20%), loneliness is a distinct construct",
        "worth studying separately from depression/anxiety/stress.",
        "",
        "=" * 80
    ])

    summary_text = "\n".join(lines)

    with open(OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)


def list_analyses():
    """List available analyses."""
    print("\nAvailable Pure UCLA Analyses:")
    print("-" * 60)
    for priority in [1, 2, 3]:
        priority_analyses = {k: v for k, v in ANALYSES.items() if v.priority == priority}
        if priority_analyses:
            label = {1: 'High', 2: 'Medium', 3: 'Low'}[priority]
            print(f"\n  Priority {priority} ({label}):")
            for name, spec in priority_analyses.items():
                print(f"    {name}")
                print(f"      {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure UCLA Effect Analysis Suite")
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
