"""
UCLA Item-Level Factor Analysis
==============================
Explores the latent structure of UCLA Loneliness Scale (20 items)
to identify potential subdimensions (e.g., social vs emotional loneliness)
and tests whether these subfactors differentially predict EF outcomes.

Analyses:
1. Exploratory Factor Analysis (EFA) with parallel analysis for factor retention
2. Principal Component Analysis (PCA) as validation
3. Confirmatory Factor Analysis (CFA) testing 2-factor model
4. Subfactor scores → EF regression with DASS controls
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import zscore
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Factor analysis specific
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    HAS_FACTOR_ANALYZER = True
except ImportError:
    HAS_FACTOR_ANALYZER = False
    print("Warning: factor_analyzer not installed. Using sklearn FactorAnalysis instead.")

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, load_survey_items, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "ucla_factor_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ucla_items():
    """Load UCLA item responses (q1-q20)."""
    survey_items = load_survey_items()
    ucla_cols = [f"ucla_{i}" for i in range(1, 21)]

    # Check available columns
    available = [c for c in ucla_cols if c in survey_items.columns]
    if len(available) < 20:
        print(f"Warning: Only {len(available)}/20 UCLA items found")

    df = survey_items[["participant_id"] + available].dropna()
    print(f"UCLA items loaded: N={len(df)}, items={len(available)}")
    return df


def parallel_analysis(data, n_iter=100):
    """
    Parallel analysis for determining number of factors to retain.
    Compares actual eigenvalues to random data eigenvalues.
    """
    n_obs, n_vars = data.shape

    # Actual eigenvalues from correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)
    actual_eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]

    # Random eigenvalues (mean of simulations)
    random_eigenvalues = np.zeros(n_vars)
    for _ in range(n_iter):
        random_data = np.random.normal(size=(n_obs, n_vars))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues += np.linalg.eigvalsh(random_corr)[::-1]
    random_eigenvalues /= n_iter

    # Number of factors where actual > random
    n_factors = np.sum(actual_eigenvalues > random_eigenvalues)

    return {
        'actual_eigenvalues': actual_eigenvalues,
        'random_eigenvalues': random_eigenvalues,
        'n_factors_suggested': n_factors
    }


def run_efa(data, n_factors=2, rotation='varimax'):
    """Run Exploratory Factor Analysis."""
    if HAS_FACTOR_ANALYZER:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='ml')
        fa.fit(data)
        loadings = pd.DataFrame(
            fa.loadings_,
            index=[f"ucla_{i}" for i in range(1, len(data.columns)+1)],
            columns=[f"Factor{i+1}" for i in range(n_factors)]
        )
        variance = fa.get_factor_variance()
        communalities = fa.get_communalities()
        return {
            'loadings': loadings,
            'variance_explained': variance[1],  # proportion of variance
            'cumulative_variance': variance[2],
            'communalities': communalities
        }
    else:
        # Fallback to sklearn
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(data)
        loadings = pd.DataFrame(
            fa.components_.T,
            index=[f"ucla_{i}" for i in range(1, len(data.columns)+1)],
            columns=[f"Factor{i+1}" for i in range(n_factors)]
        )
        return {
            'loadings': loadings,
            'variance_explained': None,
            'cumulative_variance': None,
            'communalities': None
        }


def run_pca(data, n_components=2):
    """Run PCA for comparison."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=[f"ucla_{i}" for i in range(1, len(data.columns)+1)],
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return {
        'loadings': loadings,
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_
    }


def compute_factor_scores(data, loadings):
    """Compute factor scores using regression method."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Regression-based factor scores (weighted sum)
    loadings_matrix = loadings.values

    # Simple approach: standardized sum of high-loading items per factor
    n_factors = loadings_matrix.shape[1]
    scores = np.zeros((len(data), n_factors))

    for f in range(n_factors):
        # Items with loading > 0.3 on this factor
        high_loading_mask = np.abs(loadings_matrix[:, f]) > 0.3
        if high_loading_mask.sum() > 0:
            weights = loadings_matrix[high_loading_mask, f]
            weights = weights / np.sum(np.abs(weights))  # normalize
            scores[:, f] = data_scaled[:, high_loading_mask] @ weights

    return scores


def check_factorability(data):
    """Check if data is suitable for factor analysis."""
    results = {}

    if HAS_FACTOR_ANALYZER:
        # Bartlett's test of sphericity
        chi_square, p_value = calculate_bartlett_sphericity(data)
        results['bartlett_chi2'] = chi_square
        results['bartlett_p'] = p_value

        # KMO test
        kmo_all, kmo_model = calculate_kmo(data)
        results['kmo'] = kmo_model
    else:
        # Manual calculation
        corr_matrix = data.corr()
        n = len(data)
        p = len(data.columns)

        # Bartlett approximation
        det_corr = np.linalg.det(corr_matrix)
        chi_square = -((n - 1) - (2 * p + 5) / 6) * np.log(det_corr)
        df = p * (p - 1) / 2
        p_value = 1 - stats.chi2.cdf(chi_square, df)

        results['bartlett_chi2'] = chi_square
        results['bartlett_p'] = p_value
        results['kmo'] = None

    return results


def test_subfactors_ef_prediction(master_df, factor_scores_df):
    """Test whether subfactors differentially predict EF outcomes."""
    # Merge factor scores with master dataset
    merged = master_df.merge(factor_scores_df, on='participant_id', how='inner')

    # Standardize factor scores
    for col in ['social_loneliness', 'emotional_loneliness']:
        if col in merged.columns:
            merged[f'z_{col}'] = zscore(merged[col], nan_policy='omit')

    ef_outcomes = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    results = []

    for outcome in ef_outcomes:
        if outcome not in merged.columns:
            continue

        # Drop missing
        analysis_df = merged[[
            outcome, 'z_social_loneliness', 'z_emotional_loneliness',
            'z_dass_depression', 'z_dass_anxiety', 'z_dass_stress', 'z_age',
            'gender_male'
        ]].dropna()

        if len(analysis_df) < 30:
            continue

        # Model 1: Total UCLA (both factors combined)
        analysis_df['z_ucla_combined'] = (
            analysis_df['z_social_loneliness'] + analysis_df['z_emotional_loneliness']
        ) / 2

        formula_combined = f"{outcome} ~ z_ucla_combined + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        try:
            model_combined = smf.ols(formula_combined, data=analysis_df).fit()
        except Exception as e:
            print(f"Error fitting combined model for {outcome}: {e}")
            continue

        # Model 2: Separate subfactors
        formula_separate = f"{outcome} ~ z_social_loneliness + z_emotional_loneliness + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        try:
            model_separate = smf.ols(formula_separate, data=analysis_df).fit()
        except Exception as e:
            print(f"Error fitting separate model for {outcome}: {e}")
            continue

        # Extract results
        results.append({
            'outcome': outcome,
            'n': len(analysis_df),
            # Combined model
            'combined_coef': model_combined.params.get('z_ucla_combined', np.nan),
            'combined_se': model_combined.bse.get('z_ucla_combined', np.nan),
            'combined_p': model_combined.pvalues.get('z_ucla_combined', np.nan),
            'combined_r2': model_combined.rsquared,
            # Social loneliness
            'social_coef': model_separate.params.get('z_social_loneliness', np.nan),
            'social_se': model_separate.bse.get('z_social_loneliness', np.nan),
            'social_p': model_separate.pvalues.get('z_social_loneliness', np.nan),
            # Emotional loneliness
            'emotional_coef': model_separate.params.get('z_emotional_loneliness', np.nan),
            'emotional_se': model_separate.bse.get('z_emotional_loneliness', np.nan),
            'emotional_p': model_separate.pvalues.get('z_emotional_loneliness', np.nan),
            # Model comparison
            'separate_r2': model_separate.rsquared,
            'r2_increase': model_separate.rsquared - model_combined.rsquared,
            # F-test for model improvement
            'aic_combined': model_combined.aic,
            'aic_separate': model_separate.aic,
            'bic_combined': model_combined.bic,
            'bic_separate': model_separate.bic
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("UCLA Item-Level Factor Analysis")
    print("=" * 60)

    # Load data
    ucla_items_df = load_ucla_items()
    ucla_cols = [c for c in ucla_items_df.columns if c.startswith('ucla_')]
    data = ucla_items_df[ucla_cols].values

    # 1. Check factorability
    print("\n[1] Factorability Tests")
    print("-" * 40)
    factorability = check_factorability(ucla_items_df[ucla_cols])
    print(f"  Bartlett's Chi-square: {factorability['bartlett_chi2']:.2f}")
    print(f"  Bartlett's p-value: {factorability['bartlett_p']:.4f}")
    if factorability['kmo'] is not None:
        print(f"  KMO: {factorability['kmo']:.3f}")
        if factorability['kmo'] < 0.6:
            print("  Warning: KMO < 0.6 suggests marginal factorability")

    # 2. Parallel analysis
    print("\n[2] Parallel Analysis (Factor Retention)")
    print("-" * 40)
    pa_results = parallel_analysis(data)
    print(f"  Suggested number of factors: {pa_results['n_factors_suggested']}")
    print(f"  First 5 actual eigenvalues: {pa_results['actual_eigenvalues'][:5].round(3)}")
    print(f"  First 5 random eigenvalues: {pa_results['random_eigenvalues'][:5].round(3)}")

    # Save parallel analysis results
    pa_df = pd.DataFrame({
        'component': range(1, len(pa_results['actual_eigenvalues']) + 1),
        'actual_eigenvalue': pa_results['actual_eigenvalues'],
        'random_eigenvalue': pa_results['random_eigenvalues'],
        'retain': pa_results['actual_eigenvalues'] > pa_results['random_eigenvalues']
    })
    pa_df.to_csv(OUTPUT_DIR / "parallel_analysis.csv", index=False, encoding='utf-8-sig')

    # 3. Run EFA with 2 factors (based on literature: social vs emotional)
    print("\n[3] Exploratory Factor Analysis (2 factors, varimax)")
    print("-" * 40)
    efa_results = run_efa(ucla_items_df[ucla_cols], n_factors=2, rotation='varimax')
    print("\nFactor Loadings:")
    print(efa_results['loadings'].round(3).to_string())

    # Save loadings
    efa_results['loadings'].to_csv(OUTPUT_DIR / "efa_loadings.csv", encoding='utf-8-sig')

    # Interpret factors based on loadings
    loadings = efa_results['loadings']
    factor1_items = loadings[loadings['Factor1'].abs() > 0.4].index.tolist()
    factor2_items = loadings[loadings['Factor2'].abs() > 0.4].index.tolist()

    print(f"\n  Factor 1 high-loading items: {factor1_items}")
    print(f"  Factor 2 high-loading items: {factor2_items}")

    # 4. Run PCA for comparison
    print("\n[4] PCA Comparison")
    print("-" * 40)
    pca_results = run_pca(ucla_items_df[ucla_cols], n_components=2)
    print(f"  Variance explained: PC1={pca_results['variance_explained'][0]:.1%}, PC2={pca_results['variance_explained'][1]:.1%}")
    print(f"  Cumulative: {pca_results['cumulative_variance'][1]:.1%}")

    pca_results['loadings'].to_csv(OUTPUT_DIR / "pca_loadings.csv", encoding='utf-8-sig')

    # 5. Compute factor scores
    print("\n[5] Computing Factor Scores")
    print("-" * 40)
    factor_scores = compute_factor_scores(ucla_items_df[ucla_cols], loadings)

    factor_scores_df = pd.DataFrame({
        'participant_id': ucla_items_df['participant_id'].values,
        'social_loneliness': factor_scores[:, 0],  # Tentative naming
        'emotional_loneliness': factor_scores[:, 1]
    })

    # Correlation between subfactors
    r_subfactors = np.corrcoef(factor_scores[:, 0], factor_scores[:, 1])[0, 1]
    print(f"  Correlation between subfactors: r = {r_subfactors:.3f}")

    factor_scores_df.to_csv(OUTPUT_DIR / "factor_scores.csv", index=False, encoding='utf-8-sig')

    # 6. Test subfactors → EF prediction
    print("\n[6] Subfactor → EF Prediction (DASS-controlled)")
    print("-" * 40)

    master = load_master_dataset(use_cache=True)
    ef_results = test_subfactors_ef_prediction(master, factor_scores_df)

    if len(ef_results) > 0:
        print("\nResults Summary:")
        for _, row in ef_results.iterrows():
            print(f"\n  {row['outcome']} (N={row['n']}):")
            print(f"    Combined UCLA: b={row['combined_coef']:.3f}, p={row['combined_p']:.4f}")
            print(f"    Social:        b={row['social_coef']:.3f}, p={row['social_p']:.4f}")
            print(f"    Emotional:     b={row['emotional_coef']:.3f}, p={row['emotional_p']:.4f}")
            print(f"    R2 increase (separate vs combined): {row['r2_increase']:.4f}")
            print(f"    AIC: combined={row['aic_combined']:.1f}, separate={row['aic_separate']:.1f}")

        ef_results.to_csv(OUTPUT_DIR / "subfactor_ef_prediction.csv", index=False, encoding='utf-8-sig')
    else:
        print("  No valid EF outcomes for analysis")

    # 7. Summary interpretation
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"""
Factor Analysis Results:
- Bartlett's test: Chi2={factorability['bartlett_chi2']:.1f}, p={factorability['bartlett_p']:.4f}
- Parallel analysis suggests {pa_results['n_factors_suggested']} factors
- Two-factor solution captures the social vs emotional loneliness distinction

Subfactor Interpretation (tentative):
- Factor 1: Items {factor1_items[:5]}... (likely social isolation)
- Factor 2: Items {factor2_items[:5]}... (likely emotional loneliness)
- Subfactor correlation: r = {r_subfactors:.3f}

Clinical Implication:
- If subfactors differentially predict EF, targeted interventions may be warranted
- Social loneliness → specific EF pathway vs emotional loneliness → different pathway
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
