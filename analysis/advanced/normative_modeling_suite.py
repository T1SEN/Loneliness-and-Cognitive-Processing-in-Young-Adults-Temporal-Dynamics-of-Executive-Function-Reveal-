"""
Normative Modeling Suite
========================

Personalized deviation analysis: Does UCLA predict EF impairment BEYOND
what is expected given demographics and mood?

Research Question:
------------------
Traditional regression: "Does UCLA predict EF level?"
Normative modeling: "Does UCLA predict deviation from EXPECTED EF?"

Approach:
---------
Stage 1: Build normative model EF ~ f(Age, DASS, Gender) using Gaussian Process
Stage 2: Compute cross-validated residuals (observed - predicted)
Stage 3: Test Residuals ~ UCLA * Gender (DASS already controlled in Stage 1)

Advantages:
-----------
1. Clinically interpretable: Identifies "at-risk" individuals with excessive impairment
2. Non-linear age effects via Gaussian Process kernels
3. Personalized: Each person has their own expected EF level
4. No double-dipping: Cross-validation prevents overfitting

Restored from: analysis/archive/legacy_advanced/framework2_normative_modeling.py

Usage:
    python -m analysis.advanced.normative_modeling_suite
    python -m analysis.advanced.normative_modeling_suite --analysis pe_rate
    python -m analysis.advanced.normative_modeling_suite --list

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
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

# Project imports
from analysis.preprocessing import (
    load_master_dataset, ANALYSIS_OUTPUT_DIR
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "normative_modeling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
N_FOLDS = 10
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

# EF outcomes to analyze
EF_OUTCOMES = {
    'pe_rate': 'WCST Perseverative Error Rate (%)',
    'prp_bottleneck': 'PRP Bottleneck Effect (ms)',
    'stroop_interference': 'Stroop Interference (ms)'
}


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
            function=func
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_normative_data() -> pd.DataFrame:
    """Load and prepare master dataset for normative modeling."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender
    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    # UCLA aliases
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']
    elif 'ucla_score' not in master.columns and 'ucla_total' in master.columns:
        master['ucla_score'] = master['ucla_total']

    master['gender_male'] = (master['gender'] == 'male').astype(int)
    master = standardize_predictors(master)

    # Ensure DASS z-scores
    for src, dst in [('dass_depression', 'z_dass_dep'), ('dass_anxiety', 'z_dass_anx'), ('dass_stress', 'z_dass_str')]:
        if src in master.columns and dst not in master.columns:
            std = master[src].std()
            master[dst] = (master[src] - master[src].mean()) / std if std > 0 else 0

    return master


# =============================================================================
# GAUSSIAN PROCESS NORMATIVE MODEL
# =============================================================================

def build_gp_normative_model(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """
    Build Gaussian Process normative model.

    The GP captures non-linear relationships between demographics/mood and EF,
    providing both predictions and uncertainty estimates.

    Parameters
    ----------
    X : array, shape (n, p)
        Features (standardized)
    y : array, shape (n,)
        Outcome (standardized)

    Returns
    -------
    GaussianProcessRegressor : fitted model
    """
    # RBF kernel captures smooth non-linear relationships
    # ConstantKernel allows for amplitude scaling
    # WhiteKernel accounts for observation noise
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=RANDOM_STATE,
        normalize_y=True
    )

    gp.fit(X, y)
    return gp


def compute_normative_deviations(
    df: pd.DataFrame,
    ef_outcome: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute personalized normative deviations using cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with required columns
    ef_outcome : str
        Name of EF outcome variable
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - {ef_outcome}_predicted: Expected value from normative model
        - {ef_outcome}_deviation: Raw deviation (observed - predicted)
        - {ef_outcome}_deviation_z: Z-scored deviation
    """
    if verbose:
        print(f"\n  Building Normative Model: {EF_OUTCOMES.get(ef_outcome, ef_outcome)}")

    # Features for normative model (NOT including UCLA - that's what we test later)
    feature_cols = ['age', 'dass_depression', 'dass_anxiety', 'dass_stress', 'gender_male']

    # Check for required columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_cols].values
    y = df[ef_outcome].values

    if verbose:
        print(f"    {N_FOLDS}-fold cross-validation...")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    y_pred = np.zeros_like(y, dtype=float)
    y_pred_std = np.full_like(y, np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Fold-specific scaling to prevent data leakage
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X[train_idx])
        y_train = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
        X_test = scaler_X.transform(X[test_idx])

        # Fit GP
        model = build_gp_normative_model(X_train, y_train)

        # Predict with uncertainty
        pred_mean, pred_std = model.predict(X_test, return_std=True)

        # Inverse transform back to original scale
        y_pred[test_idx] = scaler_y.inverse_transform(pred_mean.reshape(-1, 1)).ravel()
        if pred_std is not None:
            y_pred_std[test_idx] = pred_std * scaler_y.scale_[0]

    # Compute deviations
    deviation = y - y_pred
    deviation_sd = np.std(deviation)
    deviation_z = deviation / deviation_sd if deviation_sd > 0 else deviation

    df = df.copy()
    df[f'{ef_outcome}_predicted'] = y_pred
    df[f'{ef_outcome}_predicted_sd'] = y_pred_std
    df[f'{ef_outcome}_deviation'] = deviation
    df[f'{ef_outcome}_deviation_z'] = deviation_z

    # Model performance
    r2 = 1 - (np.var(deviation) / np.var(y))
    rmse = np.sqrt(np.mean(deviation**2))

    if verbose:
        print(f"    Cross-validated R²: {r2:.3f}")
        print(f"    RMSE: {rmse:.3f}")
        print(f"    Deviation SD: {deviation_sd:.3f}")

    return df


# =============================================================================
# UCLA DEVIATION ANALYSIS
# =============================================================================

def test_ucla_deviation_effects(
    df: pd.DataFrame,
    ef_outcome: str,
    verbose: bool = True
) -> Dict:
    """
    Test if UCLA predicts normative deviations.

    Since DASS is already controlled in Stage 1 (normative model),
    we only need UCLA * Gender here. However, we include DASS
    explicitly for robustness per CLAUDE.md guidelines.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with deviation scores
    ef_outcome : str
        EF outcome name

    Returns
    -------
    dict : Regression results
    """
    if verbose:
        print(f"\n  Testing UCLA → Normative Deviations")

    deviation_col = f'{ef_outcome}_deviation_z'

    # Model 1: UCLA main effect only
    model1 = smf.ols(f'{deviation_col} ~ z_ucla', data=df).fit(cov_type='HC3')

    # Model 2: UCLA + Gender
    model2 = smf.ols(f'{deviation_col} ~ z_ucla + C(gender_male)', data=df).fit(cov_type='HC3')

    # Model 3: UCLA * Gender (full model)
    model3 = smf.ols(f'{deviation_col} ~ z_ucla * C(gender_male)', data=df).fit(cov_type='HC3')

    # Model 4: With explicit DASS control (robustness check - should be similar to Model 3)
    has_dass = all(c in df.columns for c in ['z_dass_dep', 'z_dass_anx', 'z_dass_str'])
    if has_dass:
        model4 = smf.ols(
            f'{deviation_col} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str',
            data=df
        ).fit(cov_type='HC3')
    else:
        model4 = model3

    if verbose:
        print(f"\n    Hierarchical Model Comparison:")
        print(f"    Model 1 (UCLA only): R² = {model1.rsquared:.3f}")
        print(f"    Model 2 (+Gender): R² = {model2.rsquared:.3f}, ΔR² = {model2.rsquared - model1.rsquared:.3f}")
        print(f"    Model 3 (UCLA×Gender): R² = {model3.rsquared:.3f}, ΔR² = {model3.rsquared - model2.rsquared:.3f}")
        if has_dass:
            print(f"    Model 4 (+DASS): R² = {model4.rsquared:.3f}")

    # Extract key coefficients from full model
    ucla_beta = model3.params.get('z_ucla', np.nan)
    ucla_p = model3.pvalues.get('z_ucla', np.nan)
    ucla_se = model3.bse.get('z_ucla', np.nan)

    gender_beta = model3.params.get('C(gender_male)[T.1]', np.nan)
    gender_p = model3.pvalues.get('C(gender_male)[T.1]', np.nan)

    interaction_beta = model3.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
    interaction_p = model3.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)
    interaction_se = model3.bse.get('z_ucla:C(gender_male)[T.1]', np.nan)

    if verbose:
        print(f"\n    Final Model Coefficients (Model 3):")
        sig_ucla = "*" if ucla_p < 0.05 else ""
        sig_int = "*" if interaction_p < 0.05 else ""
        print(f"      UCLA main: β = {ucla_beta:.3f}, SE = {ucla_se:.3f}, p = {ucla_p:.4f}{sig_ucla}")
        print(f"      Gender main: β = {gender_beta:.3f}, p = {gender_p:.4f}")
        print(f"      UCLA×Gender: β = {interaction_beta:.3f}, SE = {interaction_se:.3f}, p = {interaction_p:.4f}{sig_int}")

    return {
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'model4': model4,
        'ucla_beta': ucla_beta,
        'ucla_se': ucla_se,
        'ucla_p': ucla_p,
        'gender_beta': gender_beta,
        'gender_p': gender_p,
        'interaction_beta': interaction_beta,
        'interaction_se': interaction_se,
        'interaction_p': interaction_p,
        'r2_final': model3.rsquared
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_deviation_regression(
    df: pd.DataFrame,
    ef_outcome: str,
    n_iterations: int = N_BOOTSTRAP,
    verbose: bool = True
) -> Dict:
    """Bootstrap CIs for UCLA → deviation regression."""
    if verbose:
        print(f"\n  Bootstrap CIs (N = {n_iterations})")

    deviation_col = f'{ef_outcome}_deviation_z'
    n = len(df)

    ucla_betas = []
    interaction_betas = []

    for i in range(n_iterations):
        df_boot = df.sample(n=n, replace=True, random_state=RANDOM_STATE + i)

        try:
            model = smf.ols(
                f'{deviation_col} ~ z_ucla * C(gender_male)',
                data=df_boot
            ).fit()

            ucla_betas.append(model.params.get('z_ucla', np.nan))
            interaction_betas.append(model.params.get('z_ucla:C(gender_male)[T.1]', np.nan))
        except:
            continue

    ucla_betas = np.array([b for b in ucla_betas if not np.isnan(b)])
    interaction_betas = np.array([b for b in interaction_betas if not np.isnan(b)])

    if verbose and len(ucla_betas) > 0:
        ucla_ci = (np.percentile(ucla_betas, 2.5), np.percentile(ucla_betas, 97.5))
        int_ci = (np.percentile(interaction_betas, 2.5), np.percentile(interaction_betas, 97.5))
        print(f"    UCLA: β = {np.mean(ucla_betas):.3f} [{ucla_ci[0]:.3f}, {ucla_ci[1]:.3f}]")
        print(f"    UCLA×Gender: β = {np.mean(interaction_betas):.3f} [{int_ci[0]:.3f}, {int_ci[1]:.3f}]")

    return {
        'ucla_beta_mean': np.mean(ucla_betas) if len(ucla_betas) > 0 else np.nan,
        'ucla_beta_ci_lower': np.percentile(ucla_betas, 2.5) if len(ucla_betas) > 0 else np.nan,
        'ucla_beta_ci_upper': np.percentile(ucla_betas, 97.5) if len(ucla_betas) > 0 else np.nan,
        'interaction_beta_mean': np.mean(interaction_betas) if len(interaction_betas) > 0 else np.nan,
        'interaction_beta_ci_lower': np.percentile(interaction_betas, 2.5) if len(interaction_betas) > 0 else np.nan,
        'interaction_beta_ci_upper': np.percentile(interaction_betas, 97.5) if len(interaction_betas) > 0 else np.nan
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_normative_model(
    df: pd.DataFrame,
    ef_outcome: str,
    verbose: bool = True
) -> None:
    """Create visualization for normative modeling results."""
    if verbose:
        print(f"\n  Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Normative curve by gender
    ax = axes[0, 0]
    for gender_val, gender_label, color in [(0, 'Female', '#E69F00'), (1, 'Male', '#56B4E9')]:
        gdata = df[df['gender_male'] == gender_val].sort_values('age')

        ax.scatter(gdata['age'], gdata[ef_outcome], alpha=0.3, s=30, color=color,
                  label=f'{gender_label} (observed)')
        ax.plot(gdata['age'], gdata[f'{ef_outcome}_predicted'], color=color,
               linewidth=2.5, label=f'{gender_label} (normative)')

    ax.set_xlabel('Age (years)', fontsize=11)
    ax.set_ylabel(EF_OUTCOMES.get(ef_outcome, ef_outcome), fontsize=11)
    ax.set_title('A) Normative Curves by Age and Gender', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    # Panel B: Deviations by UCLA tertile and gender
    ax = axes[0, 1]
    df['ucla_tertile'] = pd.qcut(df['ucla_score'], q=3, labels=['Low', 'Medium', 'High'])
    df['gender_label'] = df['gender_male'].map({0: 'Female', 1: 'Male'})

    sns.boxplot(
        data=df, x='ucla_tertile', y=f'{ef_outcome}_deviation_z',
        hue='gender_label', ax=ax, palette=['#E69F00', '#56B4E9']
    )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('UCLA Loneliness Tertile', fontsize=11)
    ax.set_ylabel('Normative Deviation (Z-score)', fontsize=11)
    ax.set_title('B) Deviations by UCLA Level and Gender', fontsize=12, fontweight='bold')
    ax.legend(title='Gender', frameon=False)
    ax.grid(axis='y', alpha=0.3)

    # Panel C: Scatter plot of UCLA vs deviation by gender
    ax = axes[1, 0]
    for gender_val, gender_label, color in [(0, 'Female', '#E69F00'), (1, 'Male', '#56B4E9')]:
        gdata = df[df['gender_male'] == gender_val]

        ax.scatter(gdata['ucla_score'], gdata[f'{ef_outcome}_deviation_z'],
                  alpha=0.5, s=50, color=color, label=gender_label)

        # Regression line
        if len(gdata) >= 10:
            z = np.polyfit(gdata['ucla_score'], gdata[f'{ef_outcome}_deviation_z'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(gdata['ucla_score'].min(), gdata['ucla_score'].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--')

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('UCLA Loneliness Score', fontsize=11)
    ax.set_ylabel('Normative Deviation (Z-score)', fontsize=11)
    ax.set_title('C) UCLA Predicts Excessive Impairment?', fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Panel D: Distribution of deviations
    ax = axes[1, 1]
    for gender_val, gender_label, color in [(0, 'Female', '#E69F00'), (1, 'Male', '#56B4E9')]:
        gdata = df[df['gender_male'] == gender_val]
        ax.hist(gdata[f'{ef_outcome}_deviation_z'], bins=20, alpha=0.5, color=color,
               label=gender_label, density=True)

    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Expected (normative)')
    ax.set_xlabel('Normative Deviation (Z-score)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('D) Deviation Distributions', fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'normative_model_{ef_outcome}.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"    Saved: {OUTPUT_DIR / f'normative_model_{ef_outcome}.png'}")


# =============================================================================
# INDIVIDUAL ANALYSES
# =============================================================================

def analyze_outcome(
    ef_outcome: str,
    verbose: bool = True
) -> Dict:
    """Run complete normative modeling analysis for one EF outcome."""
    if verbose:
        print("\n" + "=" * 70)
        print(f"NORMATIVE MODELING: {EF_OUTCOMES.get(ef_outcome, ef_outcome)}")
        print("=" * 70)

    # Load data
    df = load_normative_data()

    # Required columns
    required = ['participant_id', 'age', 'gender_male', 'ucla_score', 'z_ucla',
                'dass_depression', 'dass_anxiety', 'dass_stress', ef_outcome]

    available = [c for c in required if c in df.columns]
    if len(available) < len(required):
        missing = set(required) - set(available)
        if verbose:
            print(f"  Missing columns: {missing}")
        return {}

    df_complete = df[required].dropna()

    if len(df_complete) < 30:
        if verbose:
            print(f"  Insufficient data (N = {len(df_complete)})")
        return {}

    if verbose:
        print(f"\n  Sample: N = {len(df_complete)}")
        print(f"    Female: {(df_complete['gender_male'] == 0).sum()}")
        print(f"    Male: {(df_complete['gender_male'] == 1).sum()}")

    # Stage 1: Build normative model and compute deviations
    df_complete = compute_normative_deviations(df_complete, ef_outcome, verbose=verbose)

    # Stage 2: Test UCLA effects on deviations
    results = test_ucla_deviation_effects(df_complete, ef_outcome, verbose=verbose)

    # Stage 3: Bootstrap CIs
    bootstrap_results = bootstrap_deviation_regression(df_complete, ef_outcome, verbose=verbose)

    # Visualize
    visualize_normative_model(df_complete, ef_outcome, verbose=verbose)

    # Save individual deviation scores
    deviation_df = df_complete[[
        'participant_id', ef_outcome,
        f'{ef_outcome}_predicted',
        f'{ef_outcome}_deviation',
        f'{ef_outcome}_deviation_z'
    ]]
    deviation_df.to_csv(
        OUTPUT_DIR / f'normative_deviations_{ef_outcome}.csv',
        index=False, encoding='utf-8-sig'
    )

    if verbose:
        print(f"\n  Saved: {OUTPUT_DIR / f'normative_deviations_{ef_outcome}.csv'}")

    return {
        'ef_outcome': ef_outcome,
        'n': len(df_complete),
        **{k: v for k, v in results.items() if not isinstance(v, smf.ols)},
        **bootstrap_results
    }


# Register individual analyses
for outcome, description in EF_OUTCOMES.items():
    def make_analysis(o):
        def analysis_func(verbose: bool = True):
            return analyze_outcome(o, verbose=verbose)
        return analysis_func

    register_analysis(
        name=outcome,
        description=f"Normative modeling for {description}"
    )(make_analysis(outcome))


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict:
    """Run normative modeling analyses."""
    if verbose:
        print("=" * 70)
        print("NORMATIVE MODELING SUITE")
        print("=" * 70)
        print("\nApproach: Does UCLA predict deviation from EXPECTED EF?")
        print("Stage 1: EF ~ f(Age, DASS, Gender) via Gaussian Process")
        print("Stage 2: Deviation ~ UCLA * Gender")

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        results[analysis] = spec.function(verbose=verbose)
    else:
        # Run all EF outcomes
        all_results = []
        for name, spec in ANALYSES.items():
            try:
                result = spec.function(verbose=verbose)
                if result:
                    all_results.append(result)
                    results[name] = result
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

        # Save summary table
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(
                OUTPUT_DIR / 'normative_modeling_summary.csv',
                index=False, encoding='utf-8-sig'
            )
            if verbose:
                print(f"\n  Saved: {OUTPUT_DIR / 'normative_modeling_summary.csv'}")

        # Generate interpretation guide
        _write_interpretation_guide(all_results)

    if verbose:
        print("\n" + "=" * 70)
        print("NORMATIVE MODELING SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def _write_interpretation_guide(results: list) -> None:
    """Write interpretation guide."""
    with open(OUTPUT_DIR / 'interpretation_guide.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("NORMATIVE MODELING INTERPRETATION GUIDE\n")
        f.write("=" * 70 + "\n\n")

        f.write("RESEARCH QUESTION:\n")
        f.write("-" * 70 + "\n")
        f.write("Does loneliness predict EF impairment BEYOND what is expected\n")
        f.write("given age, mood (DASS), and gender?\n\n")

        f.write("APPROACH:\n")
        f.write("-" * 70 + "\n")
        f.write("Stage 1: Build normative model EF ~ f(Age, DASS, Gender)\n")
        f.write("Stage 2: Compute personalized deviations (Observed - Expected)\n")
        f.write("Stage 3: Test Deviation ~ UCLA * Gender\n\n")

        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write("- Positive UCLA beta: Loneliness predicts WORSE-than-expected EF\n")
        f.write("- Positive Interaction beta: Effect stronger in males\n")
        f.write("- Negative beta: Loneliness predicts BETTER-than-expected EF\n")
        f.write("- Beta near 0: No UCLA effect beyond demographics/mood\n\n")

        f.write("KEY RESULTS:\n")
        f.write("-" * 70 + "\n")
        for res in results:
            if not res:
                continue
            f.write(f"\n{EF_OUTCOMES.get(res['ef_outcome'], res['ef_outcome'])} (N={res['n']}):\n")
            f.write(f"  UCLA main: beta = {res.get('ucla_beta', np.nan):.3f}, p = {res.get('ucla_p', np.nan):.4f}\n")
            f.write(f"  UCLA x Gender: beta = {res.get('interaction_beta', np.nan):.3f}, p = {res.get('interaction_p', np.nan):.4f}\n")
            f.write(f"  Bootstrap 95% CI (UCLA): [{res.get('ucla_beta_ci_lower', np.nan):.3f}, {res.get('ucla_beta_ci_upper', np.nan):.3f}]\n")
            f.write(f"  Bootstrap 95% CI (Interaction): [{res.get('interaction_beta_ci_lower', np.nan):.3f}, {res.get('interaction_beta_ci_upper', np.nan):.3f}]\n")


def list_analyses():
    """List available analyses."""
    print("\nAvailable Normative Modeling Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normative Modeling Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                        help="Specific outcome to analyze")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
