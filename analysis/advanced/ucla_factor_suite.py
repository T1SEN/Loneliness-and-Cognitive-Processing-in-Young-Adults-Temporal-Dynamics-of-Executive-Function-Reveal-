"""
UCLA Item-Level Factor Analysis Suite
=====================================

Explores the latent structure of UCLA Loneliness Scale (20 items)
to identify potential subdimensions (e.g., social vs emotional loneliness)
and tests whether these subfactors differentially predict EF outcomes.

Key Research Questions:
1. Does UCLA have a unidimensional or multidimensional structure?
2. Which loneliness facet drives the UCLA × Gender interaction on WCST PE?
3. Do social vs emotional loneliness affect different EF components?

UCLA Loneliness Scale Structure (Russell, 1996):
- 20 items rated 1-4 (Never, Rarely, Sometimes, Often)
- Known to have 2-3 factor structure:
  - Factor 1: Intimate/Emotional Loneliness (feeling lack of close relationships)
  - Factor 2: Social Loneliness (feeling isolated from social network)
  - Factor 3: (sometimes) Collective Loneliness (not belonging to groups)

Analyses:
- factorability: Check if data suitable for factor analysis (Bartlett, KMO)
- parallel_analysis: Determine optimal number of factors
- efa: Exploratory Factor Analysis with varimax rotation
- factor_scores: Compute subfactor scores for each participant
- subfactor_ef: Test subfactor effects on EF (DASS-controlled)
- subfactor_gender: Test subfactor × Gender interactions
- summary: Summary report

Usage:
    python -m analysis.advanced.ucla_factor_suite
    python -m analysis.advanced.ucla_factor_suite --analysis efa
    python -m analysis.advanced.ucla_factor_suite --list

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
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Try to import factor_analyzer for more sophisticated analysis
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    HAS_FACTOR_ANALYZER = True
except ImportError:
    HAS_FACTOR_ANALYZER = False

# Project imports
from analysis.preprocessing import (
    load_master_dataset, RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    find_interaction_term, apply_fdr_correction, safe_zscore
)
from analysis.preprocessing.loaders import load_survey_items, ensure_participant_id
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "ucla_factor_analysis"
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


def register_analysis(name: str, description: str, source_script: str = "ucla_factor_suite.py"):
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

def load_ucla_items() -> pd.DataFrame:
    """
    Load UCLA item responses (q1-q20) from survey data.

    Returns DataFrame with participant_id and ucla_1 through ucla_20.
    """
    # Load raw survey data
    survey_path = RESULTS_DIR / '2_surveys_results.csv'
    if not survey_path.exists():
        raise FileNotFoundError(f"Survey results not found: {survey_path}")

    surveys = pd.read_csv(survey_path, encoding='utf-8')
    surveys = ensure_participant_id(surveys)

    # Filter to UCLA rows
    ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()

    if len(ucla_data) == 0:
        raise ValueError("No UCLA survey data found")

    # Extract q1-q20 columns
    item_cols = [f'q{i}' for i in range(1, 21)]
    available_cols = [c for c in item_cols if c in ucla_data.columns]

    if len(available_cols) < 20:
        print(f"  Warning: Only {len(available_cols)}/20 UCLA items found")

    # Rename to ucla_1, ucla_2, etc.
    rename_dict = {f'q{i}': f'ucla_{i}' for i in range(1, 21)}
    ucla_data = ucla_data.rename(columns=rename_dict)

    # Select relevant columns
    ucla_cols = [f'ucla_{i}' for i in range(1, 21)]
    result = ucla_data[['participant_id'] + ucla_cols].copy()

    # Drop rows with missing items
    result = result.dropna(subset=ucla_cols)

    print(f"  UCLA items loaded: N={len(result)}, items=20")

    return result


def load_master_with_standardization() -> pd.DataFrame:
    """Load master dataset with standardized predictors."""
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
# FACTOR ANALYSIS UTILITIES
# =============================================================================

def check_factorability(data: pd.DataFrame) -> Dict:
    """Check if data is suitable for factor analysis."""
    results = {}

    if HAS_FACTOR_ANALYZER:
        chi_square, p_value = calculate_bartlett_sphericity(data)
        results['bartlett_chi2'] = chi_square
        results['bartlett_p'] = p_value

        kmo_all, kmo_model = calculate_kmo(data)
        results['kmo'] = kmo_model
        results['kmo_per_item'] = kmo_all
    else:
        # Manual Bartlett calculation
        corr_matrix = data.corr()
        n = len(data)
        p = len(data.columns)

        det_corr = np.linalg.det(corr_matrix.values)
        if det_corr > 0:
            chi_square = -((n - 1) - (2 * p + 5) / 6) * np.log(det_corr)
            df = p * (p - 1) / 2
            p_value = 1 - stats.chi2.cdf(chi_square, df)
        else:
            chi_square = np.inf
            p_value = 0.0

        results['bartlett_chi2'] = chi_square
        results['bartlett_p'] = p_value
        results['kmo'] = None

    return results


def parallel_analysis(data: np.ndarray, n_iter: int = 100) -> Dict:
    """
    Parallel analysis for determining number of factors.
    Compares actual eigenvalues to random data eigenvalues.
    """
    n_obs, n_vars = data.shape

    # Actual eigenvalues
    corr_matrix = np.corrcoef(data, rowvar=False)
    actual_eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]

    # Random eigenvalues (mean of simulations)
    random_eigenvalues = np.zeros(n_vars)
    for _ in range(n_iter):
        random_data = np.random.normal(size=(n_obs, n_vars))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues += np.linalg.eigvalsh(random_corr)[::-1]
    random_eigenvalues /= n_iter

    n_factors = int(np.sum(actual_eigenvalues > random_eigenvalues))

    return {
        'actual_eigenvalues': actual_eigenvalues,
        'random_eigenvalues': random_eigenvalues,
        'n_factors_suggested': n_factors
    }


def run_efa(data: pd.DataFrame, n_factors: int = 2, rotation: str = 'varimax') -> Dict:
    """Run Exploratory Factor Analysis."""
    item_cols = [c for c in data.columns if c.startswith('ucla_')]

    if HAS_FACTOR_ANALYZER:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='ml')
        fa.fit(data[item_cols])

        loadings = pd.DataFrame(
            fa.loadings_,
            index=item_cols,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )
        variance = fa.get_factor_variance()
        communalities = fa.get_communalities()

        return {
            'loadings': loadings,
            'variance_explained': variance[1],
            'cumulative_variance': variance[2],
            'communalities': dict(zip(item_cols, communalities))
        }
    else:
        # sklearn fallback
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(data[item_cols])

        loadings = pd.DataFrame(
            fa.components_.T,
            index=item_cols,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )

        return {
            'loadings': loadings,
            'variance_explained': None,
            'cumulative_variance': None,
            'communalities': None
        }


def compute_factor_scores(data: pd.DataFrame, loadings: pd.DataFrame, threshold: float = 0.3) -> np.ndarray:
    """
    Compute factor scores using weighted sum of high-loading items.

    Items with |loading| > threshold are included with their loading as weight.
    """
    item_cols = loadings.index.tolist()
    data_items = data[item_cols].values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_items)

    n_factors = loadings.shape[1]
    scores = np.zeros((len(data), n_factors))

    for f in range(n_factors):
        factor_loadings = loadings.iloc[:, f].values
        high_loading_mask = np.abs(factor_loadings) > threshold

        if high_loading_mask.sum() > 0:
            weights = factor_loadings[high_loading_mask]
            weights = weights / np.sum(np.abs(weights))  # normalize
            scores[:, f] = data_scaled[:, high_loading_mask] @ weights

    return scores


# =============================================================================
# ANALYSES
# =============================================================================

@register_analysis(
    name="factorability",
    description="Check if UCLA data is suitable for factor analysis"
)
def analyze_factorability(verbose: bool = True) -> Dict:
    """Check factorability using Bartlett's test and KMO."""
    if verbose:
        print("\n" + "=" * 70)
        print("FACTORABILITY TESTS")
        print("=" * 70)

    ucla_items = load_ucla_items()
    item_cols = [c for c in ucla_items.columns if c.startswith('ucla_')]

    results = check_factorability(ucla_items[item_cols])

    if verbose:
        print(f"\n  N = {len(ucla_items)}")
        print(f"\n  Bartlett's Test of Sphericity:")
        print(f"    Chi-square: {results['bartlett_chi2']:.2f}")
        print(f"    p-value: {results['bartlett_p']:.6f}")

        if results['bartlett_p'] < 0.05:
            print("    -> Correlation matrix is NOT identity matrix (good for FA)")
        else:
            print("    -> WARNING: Data may not be suitable for factor analysis")

        if results['kmo'] is not None:
            print(f"\n  Kaiser-Meyer-Olkin (KMO):")
            print(f"    Overall KMO: {results['kmo']:.3f}")

            if results['kmo'] >= 0.9:
                interpretation = "Marvelous"
            elif results['kmo'] >= 0.8:
                interpretation = "Meritorious"
            elif results['kmo'] >= 0.7:
                interpretation = "Middling"
            elif results['kmo'] >= 0.6:
                interpretation = "Mediocre"
            else:
                interpretation = "Miserable (not recommended)"

            print(f"    Interpretation: {interpretation}")

    # Save results
    results_df = pd.DataFrame([{
        'test': 'Bartlett',
        'statistic': results['bartlett_chi2'],
        'p_value': results['bartlett_p']
    }, {
        'test': 'KMO',
        'statistic': results.get('kmo', np.nan),
        'p_value': np.nan
    }])
    results_df.to_csv(OUTPUT_DIR / "factorability_tests.csv", index=False, encoding='utf-8-sig')

    return results


@register_analysis(
    name="parallel_analysis",
    description="Parallel analysis for factor retention"
)
def analyze_parallel_analysis(verbose: bool = True) -> Dict:
    """Determine optimal number of factors using parallel analysis."""
    if verbose:
        print("\n" + "=" * 70)
        print("PARALLEL ANALYSIS (FACTOR RETENTION)")
        print("=" * 70)

    ucla_items = load_ucla_items()
    item_cols = [c for c in ucla_items.columns if c.startswith('ucla_')]
    data = ucla_items[item_cols].values

    results = parallel_analysis(data, n_iter=100)

    if verbose:
        print(f"\n  N = {len(ucla_items)}, Items = {len(item_cols)}")
        print(f"\n  Suggested number of factors: {results['n_factors_suggested']}")
        print(f"\n  First 5 eigenvalues:")
        print(f"    Actual: {results['actual_eigenvalues'][:5].round(3)}")
        print(f"    Random: {results['random_eigenvalues'][:5].round(3)}")

        # Show which factors to retain
        print(f"\n  Retain factors where Actual > Random:")
        for i in range(min(5, len(results['actual_eigenvalues']))):
            actual = results['actual_eigenvalues'][i]
            random = results['random_eigenvalues'][i]
            retain = "RETAIN" if actual > random else "drop"
            print(f"    Factor {i+1}: {actual:.3f} vs {random:.3f} -> {retain}")

    # Save results
    pa_df = pd.DataFrame({
        'factor': range(1, len(results['actual_eigenvalues']) + 1),
        'actual_eigenvalue': results['actual_eigenvalues'],
        'random_eigenvalue': results['random_eigenvalues'],
        'retain': results['actual_eigenvalues'] > results['random_eigenvalues']
    })
    pa_df.to_csv(OUTPUT_DIR / "parallel_analysis.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'parallel_analysis.csv'}")

    return results


@register_analysis(
    name="efa",
    description="Exploratory Factor Analysis with varimax rotation"
)
def analyze_efa(verbose: bool = True) -> Dict:
    """Run EFA with 2 and 3 factor solutions."""
    if verbose:
        print("\n" + "=" * 70)
        print("EXPLORATORY FACTOR ANALYSIS")
        print("=" * 70)

    ucla_items = load_ucla_items()
    item_cols = [c for c in ucla_items.columns if c.startswith('ucla_')]

    all_results = {}

    for n_factors in [2, 3]:
        if verbose:
            print(f"\n  {n_factors}-FACTOR SOLUTION (varimax rotation):")
            print("  " + "-" * 50)

        efa_results = run_efa(ucla_items, n_factors=n_factors)
        all_results[f'{n_factors}_factor'] = efa_results

        loadings = efa_results['loadings']

        if verbose:
            print("\n  Factor Loadings (|loading| > 0.3 highlighted):")
            # Format loadings for display
            for item in loadings.index:
                row_str = f"    {item}: "
                for col in loadings.columns:
                    val = loadings.loc[item, col]
                    marker = "*" if abs(val) > 0.3 else " "
                    row_str += f"{val:7.3f}{marker} "
                print(row_str)

            # Identify high-loading items per factor
            print(f"\n  High-loading items (|loading| > 0.4):")
            for f in range(n_factors):
                col = f'Factor{f+1}'
                high_items = loadings[loadings[col].abs() > 0.4].index.tolist()
                print(f"    Factor {f+1}: {high_items}")

            if efa_results['variance_explained'] is not None:
                print(f"\n  Variance explained:")
                for f in range(n_factors):
                    print(f"    Factor {f+1}: {efa_results['variance_explained'][f]*100:.1f}%")
                print(f"    Total: {efa_results['cumulative_variance'][-1]*100:.1f}%")

        # Save loadings
        loadings.to_csv(OUTPUT_DIR / f"efa_loadings_{n_factors}f.csv", encoding='utf-8-sig')

    if verbose:
        print(f"\n  Outputs: {OUTPUT_DIR / 'efa_loadings_*.csv'}")

    return all_results


@register_analysis(
    name="factor_scores",
    description="Compute UCLA subfactor scores for each participant"
)
def analyze_factor_scores(verbose: bool = True) -> pd.DataFrame:
    """Compute factor scores based on 2-factor EFA solution."""
    if verbose:
        print("\n" + "=" * 70)
        print("COMPUTING UCLA SUBFACTOR SCORES")
        print("=" * 70)

    ucla_items = load_ucla_items()

    # Load or run EFA
    loadings_file = OUTPUT_DIR / "efa_loadings_2f.csv"
    if not loadings_file.exists():
        analyze_efa(verbose=False)

    loadings = pd.read_csv(loadings_file, index_col=0)

    # Compute factor scores
    scores = compute_factor_scores(ucla_items, loadings, threshold=0.3)

    # Create DataFrame with interpretable names
    # Based on typical UCLA factor structure:
    # Factor 1 often captures "social network" loneliness
    # Factor 2 often captures "emotional/intimate" loneliness
    scores_df = pd.DataFrame({
        'participant_id': ucla_items['participant_id'].values,
        'ucla_factor1': scores[:, 0],
        'ucla_factor2': scores[:, 1],
        'ucla_social': scores[:, 0],  # Alias
        'ucla_emotional': scores[:, 1],  # Alias
    })

    # Add z-scored versions
    scores_df['z_ucla_social'] = safe_zscore(scores_df['ucla_social'])
    scores_df['z_ucla_emotional'] = safe_zscore(scores_df['ucla_emotional'])

    if verbose:
        print(f"\n  N = {len(scores_df)}")
        print(f"\n  Factor Score Descriptives:")
        print(f"    Social loneliness: M={scores_df['ucla_social'].mean():.3f}, SD={scores_df['ucla_social'].std():.3f}")
        print(f"    Emotional loneliness: M={scores_df['ucla_emotional'].mean():.3f}, SD={scores_df['ucla_emotional'].std():.3f}")

        # Correlation between subfactors
        r = scores_df['ucla_social'].corr(scores_df['ucla_emotional'])
        print(f"\n  Subfactor correlation: r = {r:.3f}")

        # Correlation with total score
        # Add UCLA total from items
        item_cols = [c for c in ucla_items.columns if c.startswith('ucla_')]
        ucla_total = ucla_items[item_cols].sum(axis=1)
        scores_df['ucla_total_from_items'] = ucla_total.values

        r_social = scores_df['ucla_social'].corr(scores_df['ucla_total_from_items'])
        r_emotional = scores_df['ucla_emotional'].corr(scores_df['ucla_total_from_items'])
        print(f"\n  Correlation with total score:")
        print(f"    Social: r = {r_social:.3f}")
        print(f"    Emotional: r = {r_emotional:.3f}")

    scores_df.to_csv(OUTPUT_DIR / "factor_scores.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'factor_scores.csv'}")

    return scores_df


@register_analysis(
    name="subfactor_ef",
    description="Test subfactor effects on EF outcomes (DASS-controlled)"
)
def analyze_subfactor_ef(verbose: bool = True) -> pd.DataFrame:
    """Test whether subfactors differentially predict EF outcomes."""
    if verbose:
        print("\n" + "=" * 70)
        print("SUBFACTOR → EF PREDICTION (DASS-CONTROLLED)")
        print("=" * 70)

    # Load factor scores
    scores_file = OUTPUT_DIR / "factor_scores.csv"
    if not scores_file.exists():
        analyze_factor_scores(verbose=False)
    scores = pd.read_csv(scores_file)

    # Load master data
    master = load_master_with_standardization()

    # Merge
    merged = master.merge(scores[['participant_id', 'z_ucla_social', 'z_ucla_emotional']],
                          on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  N = {len(merged)}")

    # EF outcomes to test
    ef_outcomes = [
        ('pe_rate', 'WCST PE Rate'),
        ('stroop_interference', 'Stroop Interference'),
        ('prp_bottleneck', 'PRP Bottleneck'),
    ]

    # Alternative column names
    alt_names = {
        'pe_rate': ['wcst_pe', 'wcst_pe_rate'],
        'stroop_interference': ['stroop_effect'],
        'prp_bottleneck': ['prp_effect'],
    }

    results = []

    for outcome_col, outcome_name in ef_outcomes:
        # Try to find the column
        actual_col = outcome_col
        if outcome_col not in merged.columns:
            for alt in alt_names.get(outcome_col, []):
                if alt in merged.columns:
                    actual_col = alt
                    break
            else:
                if verbose:
                    print(f"\n  {outcome_name}: Column not found")
                continue

        if verbose:
            print(f"\n  {outcome_name} ({actual_col}):")
            print("  " + "-" * 50)

        try:
            # Model 1: Combined (z_social + z_emotional as single predictor)
            merged['z_ucla_combined'] = (merged['z_ucla_social'] + merged['z_ucla_emotional']) / 2

            formula_combined = f"{actual_col} ~ z_ucla_combined * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model_combined = smf.ols(formula_combined, data=merged).fit(cov_type='HC3')

            # Model 2: Separate subfactors
            formula_separate = f"{actual_col} ~ z_ucla_social * C(gender_male) + z_ucla_emotional * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model_separate = smf.ols(formula_separate, data=merged).fit(cov_type='HC3')

            result_row = {
                'outcome': outcome_name,
                'outcome_col': actual_col,
                'n': int(model_combined.nobs),
                # Combined model
                'combined_beta': model_combined.params.get('z_ucla_combined', np.nan),
                'combined_p': model_combined.pvalues.get('z_ucla_combined', np.nan),
                'combined_r2': model_combined.rsquared,
                # Separate - Social
                'social_beta': model_separate.params.get('z_ucla_social', np.nan),
                'social_p': model_separate.pvalues.get('z_ucla_social', np.nan),
                # Separate - Emotional
                'emotional_beta': model_separate.params.get('z_ucla_emotional', np.nan),
                'emotional_p': model_separate.pvalues.get('z_ucla_emotional', np.nan),
                # Model comparison
                'separate_r2': model_separate.rsquared,
                'r2_improvement': model_separate.rsquared - model_combined.rsquared,
                'aic_combined': model_combined.aic,
                'aic_separate': model_separate.aic,
            }

            results.append(result_row)

            if verbose:
                print(f"    Combined UCLA: beta={result_row['combined_beta']:.4f}, p={result_row['combined_p']:.4f}")
                print(f"    Social:        beta={result_row['social_beta']:.4f}, p={result_row['social_p']:.4f}")
                print(f"    Emotional:     beta={result_row['emotional_beta']:.4f}, p={result_row['emotional_p']:.4f}")
                print(f"    R2 improvement: {result_row['r2_improvement']:.4f}")

                # Check if separate model is better
                if result_row['aic_separate'] < result_row['aic_combined']:
                    print(f"    -> Separate model preferred (lower AIC)")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

    if len(results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "subfactor_ef_prediction.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'subfactor_ef_prediction.csv'}")

    return results_df


@register_analysis(
    name="subfactor_gender",
    description="Test subfactor × Gender interactions on EF"
)
def analyze_subfactor_gender(verbose: bool = True) -> pd.DataFrame:
    """
    Test whether subfactor × Gender interactions explain the UCLA × Gender effect.

    Key question: Which loneliness facet drives the significant UCLA × Gender
    interaction on WCST PE (p=0.025)?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SUBFACTOR × GENDER INTERACTIONS")
        print("=" * 70)

    # Load factor scores
    scores_file = OUTPUT_DIR / "factor_scores.csv"
    if not scores_file.exists():
        analyze_factor_scores(verbose=False)
    scores = pd.read_csv(scores_file)

    # Load master data
    master = load_master_with_standardization()

    # Merge
    merged = master.merge(scores[['participant_id', 'z_ucla_social', 'z_ucla_emotional']],
                          on='participant_id', how='inner')

    if len(merged) < 30:
        if verbose:
            print(f"  Insufficient data (N={len(merged)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  N = {len(merged)}")
        males = merged['gender_male'].sum()
        females = len(merged) - males
        print(f"  Males: {males}, Females: {females}")

    # Primary outcome: WCST PE (where UCLA × Gender was significant)
    pe_col = None
    for col in ['pe_rate', 'wcst_pe', 'wcst_pe_rate']:
        if col in merged.columns:
            pe_col = col
            break

    if pe_col is None:
        if verbose:
            print("  WCST PE column not found")
        return pd.DataFrame()

    results = []

    # Test each subfactor × Gender interaction
    for subfactor in ['z_ucla_social', 'z_ucla_emotional']:
        subfactor_name = subfactor.replace('z_ucla_', '').title()

        try:
            formula = f"{pe_col} ~ {subfactor} * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=merged).fit(cov_type='HC3')

            # Find interaction term
            int_term = find_interaction_term(model.params.index, subfactor)

            result_row = {
                'subfactor': subfactor_name,
                'outcome': 'WCST PE Rate',
                'n': int(model.nobs),
                'main_beta': model.params.get(subfactor, np.nan),
                'main_p': model.pvalues.get(subfactor, np.nan),
            }

            if int_term:
                result_row['interaction_beta'] = model.params[int_term]
                result_row['interaction_p'] = model.pvalues[int_term]
            else:
                result_row['interaction_beta'] = np.nan
                result_row['interaction_p'] = np.nan

            results.append(result_row)

            if verbose:
                print(f"\n  {subfactor_name} Loneliness:")
                print(f"    Main effect: beta={result_row['main_beta']:.4f}, p={result_row['main_p']:.4f}")
                if pd.notna(result_row['interaction_p']):
                    sig = "*" if result_row['interaction_p'] < 0.05 else ""
                    print(f"    × Gender: beta={result_row['interaction_beta']:.4f}, p={result_row['interaction_p']:.4f}{sig}")

        except Exception as e:
            if verbose:
                print(f"\n  {subfactor_name}: Error - {e}")

    # Gender-stratified analysis
    if verbose:
        print("\n  GENDER-STRATIFIED ANALYSIS:")
        print("  " + "-" * 50)

    for gender_val, gender_name in [(0, 'Female'), (1, 'Male')]:
        subset = merged[merged['gender_male'] == gender_val]

        if len(subset) < 15:
            continue

        if verbose:
            print(f"\n  {gender_name} (N={len(subset)}):")

        for subfactor in ['z_ucla_social', 'z_ucla_emotional']:
            subfactor_name = subfactor.replace('z_ucla_', '').title()

            try:
                formula = f"{pe_col} ~ {subfactor} + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=subset).fit(cov_type='HC3')

                beta = model.params.get(subfactor, np.nan)
                p = model.pvalues.get(subfactor, np.nan)

                results.append({
                    'subfactor': f"{subfactor_name} ({gender_name})",
                    'outcome': 'WCST PE Rate',
                    'n': int(model.nobs),
                    'main_beta': beta,
                    'main_p': p,
                    'interaction_beta': np.nan,
                    'interaction_p': np.nan,
                })

                if verbose:
                    sig = "*" if p < 0.05 else ""
                    print(f"    {subfactor_name}: beta={beta:.4f}, p={p:.4f}{sig}")

            except Exception as e:
                if verbose:
                    print(f"    {subfactor_name}: Error - {e}")

    if len(results) == 0:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "subfactor_gender_interactions.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'subfactor_gender_interactions.csv'}")

    return results_df


@register_analysis(
    name="summary",
    description="Summary report and interpretation"
)
def analyze_summary(verbose: bool = True) -> Dict:
    """Generate summary report."""
    if verbose:
        print("\n" + "=" * 70)
        print("UCLA FACTOR ANALYSIS SUMMARY")
        print("=" * 70)

    summary = {}

    # Load parallel analysis
    pa_file = OUTPUT_DIR / "parallel_analysis.csv"
    if pa_file.exists():
        pa = pd.read_csv(pa_file)
        n_factors = int(pa['retain'].sum())
        summary['n_factors_suggested'] = n_factors

    # Load EFA loadings
    loadings_file = OUTPUT_DIR / "efa_loadings_2f.csv"
    if loadings_file.exists():
        loadings = pd.read_csv(loadings_file, index_col=0)

        # High-loading items
        f1_items = loadings[loadings['Factor1'].abs() > 0.4].index.tolist()
        f2_items = loadings[loadings['Factor2'].abs() > 0.4].index.tolist()

        summary['factor1_items'] = f1_items
        summary['factor2_items'] = f2_items

    # Load subfactor EF results
    ef_file = OUTPUT_DIR / "subfactor_ef_prediction.csv"
    if ef_file.exists():
        ef = pd.read_csv(ef_file)
        sig_social = ef[ef['social_p'] < 0.05] if 'social_p' in ef.columns else pd.DataFrame()
        sig_emotional = ef[ef['emotional_p'] < 0.05] if 'emotional_p' in ef.columns else pd.DataFrame()

        summary['significant_social_effects'] = sig_social['outcome'].tolist() if len(sig_social) > 0 else []
        summary['significant_emotional_effects'] = sig_emotional['outcome'].tolist() if len(sig_emotional) > 0 else []

    # Load gender interaction results
    gender_file = OUTPUT_DIR / "subfactor_gender_interactions.csv"
    if gender_file.exists():
        gender = pd.read_csv(gender_file)
        sig_int = gender[gender['interaction_p'] < 0.05] if 'interaction_p' in gender.columns else pd.DataFrame()
        summary['significant_gender_interactions'] = sig_int['subfactor'].tolist() if len(sig_int) > 0 else []

    if verbose:
        print(f"\n  FACTOR STRUCTURE:")
        if 'n_factors_suggested' in summary:
            print(f"    Parallel analysis suggests: {summary['n_factors_suggested']} factors")
        if 'factor1_items' in summary:
            print(f"    Factor 1 (Social): {summary['factor1_items']}")
            print(f"    Factor 2 (Emotional): {summary['factor2_items']}")

        print(f"\n  SUBFACTOR → EF EFFECTS:")
        if summary.get('significant_social_effects'):
            print(f"    Social loneliness: {summary['significant_social_effects']}")
        else:
            print(f"    Social loneliness: No significant effects")

        if summary.get('significant_emotional_effects'):
            print(f"    Emotional loneliness: {summary['significant_emotional_effects']}")
        else:
            print(f"    Emotional loneliness: No significant effects")

        print(f"\n  GENDER INTERACTIONS:")
        if summary.get('significant_gender_interactions'):
            print(f"    Significant: {summary['significant_gender_interactions']}")
        else:
            print(f"    No significant subfactor × Gender interactions")

        print(f"\n  INTERPRETATION:")
        if not summary.get('significant_social_effects') and not summary.get('significant_emotional_effects'):
            print("    Neither loneliness subfactor shows significant EF effects after DASS control")
            print("    -> Consistent with total UCLA findings: DASS confounds all UCLA-EF associations")
        else:
            if summary.get('significant_social_effects'):
                print(f"    Social loneliness drives effects on: {summary['significant_social_effects']}")
            if summary.get('significant_emotional_effects'):
                print(f"    Emotional loneliness drives effects on: {summary['significant_emotional_effects']}")

    # Save summary
    import json
    with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'summary.json'}")

    return summary


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict:
    """Run UCLA factor analyses."""
    if verbose:
        print("=" * 70)
        print("UCLA ITEM-LEVEL FACTOR ANALYSIS SUITE")
        print("=" * 70)

    if not HAS_FACTOR_ANALYZER:
        print("\nNote: factor_analyzer package not installed. Using sklearn fallback.")
        print("For full functionality: pip install factor-analyzer\n")

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        analysis_order = [
            'factorability',
            'parallel_analysis',
            'efa',
            'factor_scores',
            'subfactor_ef',
            'subfactor_gender',
            'summary',
        ]

        for name in analysis_order:
            if name in ANALYSES:
                try:
                    results[name] = ANALYSES[name].function(verbose=verbose)
                except Exception as e:
                    print(f"  ERROR in {name}: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("UCLA FACTOR ANALYSIS SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable UCLA Factor Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCLA Factor Analysis Suite")
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
