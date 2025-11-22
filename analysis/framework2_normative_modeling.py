"""
Framework 2: Normative Modeling
=================================

Tests if loneliness predicts deviations from normative expectations.

Research Question:
------------------
"Does loneliness explain EF impairment BEYOND what is expected given
 age, mood (DASS), and gender?"

Conceptual Approach:
--------------------
Traditional regression asks: "Does UCLA predict EF level?"
Normative modeling asks: "Does UCLA predict deviation from expected EF?"

This reframes the question as:
- Stage 1: Build normative model EF ~ f(Age, DASS, Gender)
- Stage 2: Compute residuals (observed - predicted)
- Stage 3: Test Residuals ~ UCLA * Gender

Advantages:
-----------
1. Clinically interpretable: "Excessive impairment beyond baseline"
2. Non-linear age effects (GAM captures developmental curves)
3. Personalized: Each person has their own "expected" EF level
4. Early deviation detection: Identifies at-risk individuals

Methods:
--------
- Generalized Additive Models (GAM) for smooth non-linear curves
- Gaussian Process (GP) as Bayesian alternative
- 10-fold cross-validation for unbiased normative estimates
- Bootstrap confidence intervals for deviation scores

Author: Research Analysis Pipeline
Date: 2024-11-17
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

try:
    from pygam import GAM, s, f, l
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    print("⚠️  pygam not available, will use Gaussian Process instead")

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.publication_helpers import (
    set_publication_style,
    save_publication_figure,
    bootstrap_ci,
    format_pvalue,
    format_ci,
    load_master_dataset,
)

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/framework2_normative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 10
N_BOOTSTRAP = 2000

EF_OUTCOMES = {
    'wcst_pe_rate': 'WCST Perseverative Error Rate',
    'prp_bottleneck': 'PRP Bottleneck Effect (ms)',
    'stroop_interference': 'Stroop Interference (ms)'
}

# ============================================================================
# Step 1: Load Data
# ============================================================================

def load_normative_data():
    """
    Load master dataset for normative modeling.
    """
    print("=" * 70)
    print("LOADING DATA FOR NORMATIVE MODELING (centralized master dataset)")
    print("=" * 70)

    master = load_master_dataset(
        use_cache=True,
        force_rebuild=False,
        add_standardized=True
    )

    if 'ucla_total' in master.columns and 'ucla_score' not in master.columns:
        master = master.rename(columns={'ucla_total': 'ucla_score'})
    if 'z_ucla_score' in master.columns:
        master['z_ucla'] = master['z_ucla_score']
    elif 'ucla_score' in master.columns:
        master['z_ucla'] = (master['ucla_score'] - master['ucla_score'].mean()) / master['ucla_score'].std()

    print(f"\\nFinal sample size: N = {len(master)}")
    print(f"  Female: {(master['gender_male'] == 0).sum()}")
    print(f"  Male: {(master['gender_male'] == 1).sum()}")
    print(f"  Age range: {master['age'].min():.1f} - {master['age'].max():.1f} years")

    return master



# ============================================================================
# Step 2: Build Normative Models
# ============================================================================

def build_gam_normative_model(X, y, feature_names):
    """
    Build GAM normative model with smooth terms for continuous predictors.

    Parameters
    ----------
    X : array, shape (n, p)
        Features: [age, dass_dep, dass_anx, dass_str, gender_male]
    y : array, shape (n,)
        Outcome variable
    feature_names : list
        Names of features

    Returns
    -------
    model : fitted GAM
    """
    if not PYGAM_AVAILABLE:
        raise ImportError("pygam is required for GAM modeling")

    # Build GAM formula
    # s() = smooth spline for continuous
    # f() = factor (categorical) for gender

    # Assuming feature order: age, dass_dep, dass_anx, dass_str, gender_male
    gam = GAM(s(0) + s(1) + s(2) + s(3) + f(4))

    gam.gridsearch(X, y, progress=False)

    return gam


def build_gp_normative_model(X, y):
    """
    Build Gaussian Process normative model (Bayesian alternative).

    Parameters
    ----------
    X : array, shape (n, p)
        Features (standardized)
    y : array, shape (n,)
        Outcome (standardized)

    Returns
    -------
    model : fitted GP
    """
    # RBF kernel captures smooth non-linear relationships
    # ConstantKernel allows for amplitude scaling
    # WhiteKernel accounts for noise

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=RANDOM_STATE,
        normalize_y=True
    )

    gp.fit(X, y)

    return gp


def compute_normative_deviations(df, ef_outcome, method='gp'):
    """
    Compute personalized normative deviations using cross-validation (no data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset
    ef_outcome : str
        Name of EF outcome variable
    method : str, 'gam' or 'gp'
        Modeling method

    Returns
    -------
    df : pd.DataFrame
        Original df with added columns:
        - {ef_outcome}_predicted
        - {ef_outcome}_deviation
        - {ef_outcome}_deviation_z
    """
    print(f"\n{'=' * 70}")
    print(f"BUILDING NORMATIVE MODEL: {EF_OUTCOMES[ef_outcome]}")
    print(f"Method: {method.upper()}")
    print(f"{'=' * 70}")

    feature_cols = ['age', 'dass_depression', 'dass_anxiety', 'dass_stress', 'gender_male']
    X = df[feature_cols].values
    y = df[ef_outcome].values

    print(f"\nPerforming {N_FOLDS}-fold cross-validation...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    y_pred = np.zeros_like(y, dtype=float)
    y_pred_std = np.full_like(y, np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}...", end=' ')

        # Fold-specific scaling to prevent leakage
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X[train_idx])
        y_train = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
        X_test = scaler_X.transform(X[test_idx])

        if method == 'gam' and PYGAM_AVAILABLE:
            model = build_gam_normative_model(X_train, y_train, feature_cols)
            pred_mean = model.predict(X_test)
            pred_std = np.full_like(pred_mean, np.nan)
        else:
            model = build_gp_normative_model(X_train, y_train)
            pred_mean, pred_std = model.predict(X_test, return_std=True)

        # Inverse-transform predictions back to original scale
        y_pred[test_idx] = scaler_y.inverse_transform(pred_mean.reshape(-1, 1)).ravel()
        if pred_std is not None and not np.all(np.isnan(pred_std)):
            y_pred_std[test_idx] = pred_std * scaler_y.scale_[0]

        print("done")

    deviation = y - y_pred
    deviation_sd = np.std(deviation)
    deviation_z = deviation / deviation_sd

    df[f'{ef_outcome}_predicted'] = y_pred
    df[f'{ef_outcome}_predicted_sd'] = y_pred_std
    df[f'{ef_outcome}_deviation'] = deviation
    df[f'{ef_outcome}_deviation_z'] = deviation_z

    r2 = 1 - (np.var(deviation) / np.var(y))
    rmse = np.sqrt(np.mean(deviation**2))

    print(f"\nNormative Model Performance:")
    print(f"  R^2 (cross-validated): {r2:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Deviation SD: {deviation_sd:.3f}")

    return df



# ============================================================================
# Step 3: Test UCLA as Predictor of Deviations
# ============================================================================

def test_ucla_deviation_effects(df, ef_outcome):
    """
    Hierarchical regression: Do deviations ~ UCLA * Gender?

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
    print(f"\n{'=' * 70}")
    print(f"TESTING UCLA → NORMATIVE DEVIATIONS")
    print(f"{'=' * 70}")

    deviation_col = f'{ef_outcome}_deviation_z'

    # Model 1: Intercept only (should be ~0)
    model1 = smf.ols(f'{deviation_col} ~ 1', data=df).fit()

    # Model 2: UCLA main effect
    model2 = smf.ols(f'{deviation_col} ~ z_ucla', data=df).fit()

    # Model 3: UCLA + Gender
    model3 = smf.ols(f'{deviation_col} ~ z_ucla + C(gender_male)', data=df).fit()

    # Model 4: UCLA * Gender interaction
    model4 = smf.ols(f'{deviation_col} ~ z_ucla * C(gender_male)', data=df).fit()

    # Model comparison
    print("\nHierarchical Regression Results:")
    print("-" * 70)
    print(f"Model 1 (Intercept): R² = {model1.rsquared:.3f}")
    print(f"  Intercept β = {model1.params['Intercept']:.3f}, {format_pvalue(model1.pvalues['Intercept'])}")

    print(f"\nModel 2 (+UCLA): R² = {model2.rsquared:.3f}, ΔR² = {model2.rsquared - model1.rsquared:.3f}")
    print(f"  UCLA β = {model2.params['z_ucla']:.3f}, {format_pvalue(model2.pvalues['z_ucla'])}")

    print(f"\nModel 3 (+Gender): R² = {model3.rsquared:.3f}, ΔR² = {model3.rsquared - model2.rsquared:.3f}")

    print(f"\nModel 4 (UCLA×Gender): R² = {model4.rsquared:.3f}, ΔR² = {model4.rsquared - model3.rsquared:.3f}")

    # Extract key coefficients from Model 4
    ucla_beta = model4.params.get('z_ucla', np.nan)
    ucla_p = model4.pvalues.get('z_ucla', np.nan)

    gender_beta = model4.params.get('C(gender_male)[T.1]', np.nan)
    gender_p = model4.pvalues.get('C(gender_male)[T.1]', np.nan)

    interaction_beta = model4.params.get('z_ucla:C(gender_male)[T.1]', np.nan)
    interaction_p = model4.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)

    print(f"\nFinal Model Coefficients:")
    print(f"  UCLA main effect: β = {ucla_beta:.3f}, {format_pvalue(ucla_p)}")
    print(f"  Gender main effect: β = {gender_beta:.3f}, {format_pvalue(gender_p)}")
    print(f"  UCLA × Gender: β = {interaction_beta:.3f}, {format_pvalue(interaction_p)}")

    return {
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'model4': model4,
        'ucla_beta': ucla_beta,
        'ucla_p': ucla_p,
        'gender_beta': gender_beta,
        'gender_p': gender_p,
        'interaction_beta': interaction_beta,
        'interaction_p': interaction_p,
        'r2_final': model4.rsquared,
        'delta_r2_interaction': model4.rsquared - model3.rsquared
    }


# ============================================================================
# Step 4: Visualization
# ============================================================================

def visualize_normative_model(df, ef_outcome, results):
    """
    Create publication-quality normative modeling visualizations.
    """
    set_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Age effect (normative curve)
    ax = axes[0, 0]

    for gender_val, gender_label, color in [(0, 'Female', '#E69F00'), (1, 'Male', '#56B4E9')]:
        df_gender = df[df['gender_male'] == gender_val].copy()

        # Sort by age for plotting
        df_gender = df_gender.sort_values('age')

        # Observed data
        ax.scatter(df_gender['age'], df_gender[ef_outcome],
                  alpha=0.3, s=30, color=color, label=f'{gender_label} (observed)')

        # Predicted (normative) curve
        ax.plot(df_gender['age'], df_gender[f'{ef_outcome}_predicted'],
               color=color, linewidth=2.5, label=f'{gender_label} (normative)')

    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel(EF_OUTCOMES[ef_outcome], fontsize=12)
    ax.set_title('A) Normative Curves by Age and Gender', fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    # Panel B: Deviations by UCLA (scatter)
    ax = axes[0, 1]

    # Tertile split for visualization
    df['ucla_tertile'] = pd.qcut(df['ucla_score'], q=3, labels=['Low', 'Medium', 'High'])

    sns.boxplot(data=df, x='ucla_tertile', y=f'{ef_outcome}_deviation_z',
               hue='gender', ax=ax, palette=['#E69F00', '#56B4E9'])

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Expected (normative)')
    ax.set_xlabel('UCLA Loneliness Tertile', fontsize=12)
    ax.set_ylabel('Normative Deviation (Z-score)', fontsize=12)
    ax.set_title('B) Deviations by UCLA Level and Gender', fontsize=13, fontweight='bold')
    ax.legend(frameon=False, title='Gender')
    ax.grid(axis='y', alpha=0.3)

    # Panel C: UCLA × Deviation regression
    ax = axes[1, 0]

    for gender_val, gender_label, color in [(0, 'Female', '#E69F00'), (1, 'Male', '#56B4E9')]:
        df_gender = df[df['gender_male'] == gender_val].copy()

        ax.scatter(df_gender['ucla_score'], df_gender[f'{ef_outcome}_deviation_z'],
                  alpha=0.5, s=50, color=color, label=gender_label)

        # Regression line
        z = np.polyfit(df_gender['ucla_score'], df_gender[f'{ef_outcome}_deviation_z'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_gender['ucla_score'].min(), df_gender['ucla_score'].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--')

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('UCLA Loneliness Score', fontsize=12)
    ax.set_ylabel('Normative Deviation (Z-score)', fontsize=12)
    ax.set_title('C) UCLA Predicts Excessive Impairment?', fontsize=13, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Panel D: Percentile bands
    ax = axes[1, 1]

    # Focus on one gender for clarity (males, as they show effects)
    df_male = df[df['gender_male'] == 1].copy().sort_values('age')

    # Compute percentile bands
    age_bins = pd.cut(df_male['age'], bins=5)
    percentiles = df_male.groupby(age_bins)[ef_outcome].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack()

    age_centers = [interval.mid for interval in percentiles.index]

    ax.fill_between(age_centers, percentiles[0.05], percentiles[0.95],
                    alpha=0.2, color='#56B4E9', label='5th-95th percentile')
    ax.fill_between(age_centers, percentiles[0.25], percentiles[0.75],
                    alpha=0.4, color='#56B4E9', label='25th-75th percentile')
    ax.plot(age_centers, percentiles[0.50], color='#0072B2', linewidth=2.5, label='Median')

    # Overlay high-UCLA individuals
    df_high_ucla = df_male[df_male['ucla_score'] > df_male['ucla_score'].quantile(0.75)]
    ax.scatter(df_high_ucla['age'], df_high_ucla[ef_outcome],
              color='red', s=80, alpha=0.7, marker='x', linewidths=2, label='High UCLA (>75th %ile)')

    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel(EF_OUTCOMES[ef_outcome], fontsize=12)
    ax.set_title('D) Normative Percentiles for Males', fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_publication_figure(fig, OUTPUT_DIR / f'normative_model_{ef_outcome}', formats=['png', 'pdf'])
    plt.close()


# ============================================================================
# Step 5: Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_deviation_regression(df, ef_outcome, n_iterations=N_BOOTSTRAP):
    """
    Bootstrap CIs for UCLA → deviation regression coefficients.
    """
    print(f"\n{'=' * 70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (N = {n_iterations})")
    print(f"{'=' * 70}")

    deviation_col = f'{ef_outcome}_deviation_z'

    np.random.seed(RANDOM_STATE)
    n = len(df)

    ucla_betas = []
    interaction_betas = []

    for i in range(n_iterations):
        if (i + 1) % 500 == 0:
            print(f"  Iteration {i+1}/{n_iterations}...")

        # Resample
        df_boot = df.sample(n=n, replace=True, random_state=RANDOM_STATE+i)

        # Fit model
        try:
            model = smf.ols(f'{deviation_col} ~ z_ucla * C(gender_male)', data=df_boot).fit()

            ucla_betas.append(model.params.get('z_ucla', np.nan))
            interaction_betas.append(model.params.get('z_ucla:C(gender_male)[T.1]', np.nan))
        except:
            # Skip if model fails
            continue

    ucla_betas = np.array([b for b in ucla_betas if not np.isnan(b)])
    interaction_betas = np.array([b for b in interaction_betas if not np.isnan(b)])

    print(f"\nBootstrap Results:")
    print(f"  UCLA main effect: β = {np.mean(ucla_betas):.3f}, {format_ci(np.percentile(ucla_betas, 2.5), np.percentile(ucla_betas, 97.5), decimals=3)}")
    print(f"  UCLA × Gender: β = {np.mean(interaction_betas):.3f}, {format_ci(np.percentile(interaction_betas, 2.5), np.percentile(interaction_betas, 97.5), decimals=3)}")

    return {
        'ucla_beta_mean': np.mean(ucla_betas),
        'ucla_beta_ci_lower': np.percentile(ucla_betas, 2.5),
        'ucla_beta_ci_upper': np.percentile(ucla_betas, 97.5),
        'interaction_beta_mean': np.mean(interaction_betas),
        'interaction_beta_ci_lower': np.percentile(interaction_betas, 2.5),
        'interaction_beta_ci_upper': np.percentile(interaction_betas, 97.5)
    }


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    """
    Execute complete Normative Modeling analysis.
    """
    print("\n" + "=" * 70)
    print("FRAMEWORK 2: NORMATIVE MODELING")
    print("=" * 70)

    # Step 1: Load data
    df = load_normative_data()

    # Summary of results
    all_results = []

    # Analyze each EF outcome
    for ef_outcome in EF_OUTCOMES.keys():
        if ef_outcome not in df.columns:
            print(f"\n⚠️  Skipping {ef_outcome} (not in dataset)")
            continue

        # Drop missing
        df_complete = df[['participant_id', 'age', 'gender', 'gender_male',
                         'ucla_score', 'z_ucla',
                         'dass_depression', 'dass_anxiety', 'dass_stress',
                         ef_outcome]].dropna()

        if len(df_complete) < 30:
            print(f"\n⚠️  Insufficient data for {ef_outcome} (N = {len(df_complete)})")
            continue

        print(f"\n\n{'=' * 70}")
        print(f"ANALYZING: {EF_OUTCOMES[ef_outcome]}")
        print(f"N = {len(df_complete)}")
        print(f"{'=' * 70}")

        # Step 2: Build normative model & compute deviations
        method = 'gp'  # Use Gaussian Process (available in scikit-learn)
        df_complete = compute_normative_deviations(df_complete, ef_outcome, method=method)

        # Step 3: Test UCLA → deviations
        results = test_ucla_deviation_effects(df_complete, ef_outcome)

        # Step 4: Visualize
        visualize_normative_model(df_complete, ef_outcome, results)

        # Step 5: Bootstrap CIs
        bootstrap_results = bootstrap_deviation_regression(df_complete, ef_outcome, n_iterations=N_BOOTSTRAP)

        # Combine results
        all_results.append({
            'ef_outcome': ef_outcome,
            'n': len(df_complete),
            **results,
            **bootstrap_results
        })

        # Save individual deviation scores
        deviation_df = df_complete[['participant_id', ef_outcome,
                                   f'{ef_outcome}_predicted',
                                   f'{ef_outcome}_deviation',
                                   f'{ef_outcome}_deviation_z']]
        deviation_df.to_csv(OUTPUT_DIR / f'normative_deviations_{ef_outcome}.csv', index=False)
        print(f"\n✓ Saved: {OUTPUT_DIR / f'normative_deviations_{ef_outcome}.csv'}")

    # ========================================================================
    # Save Summary Table
    # ========================================================================

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'normative_modeling_summary.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'normative_modeling_summary.csv'}")

    # ========================================================================
    # Generate Interpretation
    # ========================================================================

    with open(OUTPUT_DIR / 'interpretation_guide.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAMEWORK 2: NORMATIVE MODELING\n")
        f.write("Interpretation Guide\n")
        f.write("=" * 70 + "\n\n")

        f.write("RESEARCH QUESTION:\n")
        f.write("-" * 70 + "\n")
        f.write("Does loneliness predict EF impairment BEYOND what is expected\n")
        f.write("given age, mood (DASS), and gender?\n\n")

        f.write("APPROACH:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Built normative models: EF ~ f(Age, DASS, Gender)\n")
        f.write("2. Computed personalized deviations (Observed - Expected)\n")
        f.write("3. Tested: Deviation ~ UCLA * Gender\n\n")

        f.write("KEY RESULTS:\n")
        f.write("-" * 70 + "\n")
        for res in all_results:
            f.write(f"\n{EF_OUTCOMES[res['ef_outcome']]} (N={res['n']}):\n")
            f.write(f"  Normative model R²: {1 - res['r2_final']:.3f}\n")
            f.write(f"  UCLA → Deviation: β = {res['ucla_beta']:.3f}, p = {res['ucla_p']:.3f}\n")
            f.write(f"  UCLA × Gender: β = {res['interaction_beta']:.3f}, p = {res['interaction_p']:.3f}\n")
            f.write(f"  Bootstrap 95% CI (UCLA): [{res['ucla_beta_ci_lower']:.3f}, {res['ucla_beta_ci_upper']:.3f}]\n")
            f.write(f"  Bootstrap 95% CI (Interaction): [{res['interaction_beta_ci_lower']:.3f}, {res['interaction_beta_ci_upper']:.3f}]\n")

        f.write("\n\nINTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write("- Positive UCLA β: Loneliness predicts WORSE-than-expected EF\n")
        f.write("- Positive Interaction β: Effect stronger in males\n")
        f.write("- Negative UCLA β: Loneliness predicts BETTER-than-expected EF (unexpected)\n")
        f.write("- β near 0: No additional UCLA effect beyond demographics/mood\n\n")

        f.write("CLINICAL IMPLICATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write("If UCLA β > 0 in males:\n")
        f.write("  → Lonely men show 'excessive' EF impairment not explained by mood\n")
        f.write("  → Early risk marker for cognitive vulnerability\n")
        f.write("  → Personalized normative approach identifies at-risk individuals\n\n")

        f.write("COMPARISON TO FRAMEWORK 1:\n")
        f.write("-" * 70 + "\n")
        f.write("- Framework 1 (Mixtures): Found latent subgroups with different slopes\n")
        f.write("- Framework 2 (Normative): Quantifies how much each person deviates\n")
        f.write("- Both converge if: High-deviation individuals = Vulnerable mixture class\n\n")

    print(f"\n✓ Saved: {OUTPUT_DIR / 'interpretation_guide.txt'}")

    print("\n" + "=" * 70)
    print("✓ FRAMEWORK 2 COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
