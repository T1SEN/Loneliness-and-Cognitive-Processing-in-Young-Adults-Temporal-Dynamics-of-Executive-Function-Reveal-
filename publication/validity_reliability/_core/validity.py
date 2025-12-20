"""
Validity Analysis
=================

Construct, convergent/discriminant, and criterion validity for online experiment measures.

Analyses:
1. Construct Validity (Factor Structure)
   - KMO and Bartlett's test for factorability
   - Exploratory Factor Analysis (EFA) for UCLA scale

2. Convergent/Discriminant Validity
   - UCLA vs DASS subscale correlation matrix
   - Expected: UCLA-Depression high, UCLA-Anxiety moderate, UCLA-Stress moderate

3. Criterion Validity
   - UCLA prediction of EF task performance (R-squared)

Usage:
    python -m publication.validity_reliability.complete_overall.validity

Output:
    publication/data/outputs/validity_reliability/<dataset>/
    - validity_correlation_matrix.csv
    - efa_loadings.csv
    - validity_summary.json

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import statsmodels.formula.api as smf

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR, get_results_dir
from publication.preprocessing import (
    ensure_participant_id,
    load_master_dataset,
    safe_zscore,
    prepare_gender_variable,
)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "validity_reliability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _resolve_data_dir(data_dir: Path | None, task: str) -> Path:
    if data_dir is not None:
        return data_dir
    return get_results_dir(task)


def _load_filtered_ids(data_dir: Path) -> set[str] | None:
    ids_path = data_dir / "filtered_participant_ids.csv"
    if ids_path.exists():
        df = pd.read_csv(ids_path, encoding="utf-8-sig")
        for col in ("participantId", "participant_id"):
            if col in df.columns:
                return set(df[col].dropna().astype(str))
        if df.columns.tolist():
            return set(df[df.columns[0]].dropna().astype(str))
    participants_path = data_dir / "1_participants_info.csv"
    if participants_path.exists():
        df = pd.read_csv(participants_path, encoding="utf-8-sig")
        df = ensure_participant_id(df)
        return set(df["participant_id"].dropna().astype(str))
    return None

def kmo_test(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    """
    Calculate Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with items as columns

    Returns
    -------
    tuple
        (overall_kmo, individual_kmo_per_item)
    """
    df_clean = df.dropna()
    corr_matrix = df_clean.corr()

    # Partial correlation matrix
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        partial_corr = np.zeros_like(corr_matrix)
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if i != j:
                    partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
    except np.linalg.LinAlgError:
        return np.nan, np.array([])

    corr_sq = np.square(corr_matrix.values)
    partial_sq = np.square(partial_corr)

    np.fill_diagonal(corr_sq, 0)
    np.fill_diagonal(partial_sq, 0)

    sum_corr = corr_sq.sum()
    sum_partial = partial_sq.sum()

    overall_kmo = sum_corr / (sum_corr + sum_partial) if (sum_corr + sum_partial) > 0 else np.nan

    # Individual KMO
    individual_kmo = []
    for i in range(len(corr_matrix)):
        row_corr = corr_sq[i, :].sum()
        row_partial = partial_sq[i, :].sum()
        kmo_i = row_corr / (row_corr + row_partial) if (row_corr + row_partial) > 0 else np.nan
        individual_kmo.append(kmo_i)

    return overall_kmo, np.array(individual_kmo)


def bartlett_test(df: pd.DataFrame) -> tuple[float, float]:
    """
    Bartlett's test of sphericity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with items as columns

    Returns
    -------
    tuple
        (chi_square, p_value)
    """
    df_clean = df.dropna()
    n = len(df_clean)
    p = df_clean.shape[1]
    corr_matrix = df_clean.corr()

    det = np.linalg.det(corr_matrix)
    if det <= 0:
        return np.nan, np.nan

    chi_sq = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    df_chi = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi_sq, df_chi)

    return chi_sq, p_value


def interpret_kmo(kmo: float) -> str:
    """Interpret KMO value."""
    if np.isnan(kmo):
        return "N/A"
    if kmo >= 0.9:
        return "Marvelous"
    if kmo >= 0.8:
        return "Meritorious"
    if kmo >= 0.7:
        return "Middling"
    if kmo >= 0.6:
        return "Mediocre"
    if kmo >= 0.5:
        return "Miserable"
    return "Unacceptable"


# =============================================================================
# CONSTRUCT VALIDITY (EFA)
# =============================================================================

def load_ucla_items(data_dir: Path | None = None, task: str = "overall") -> pd.DataFrame:
    """Load UCLA item-level responses."""
    data_dir = _resolve_data_dir(data_dir, task)
    surveys_path = data_dir / "2_surveys_results.csv"
    surveys = pd.read_csv(surveys_path, encoding='utf-8-sig')

    ucla_df = surveys[surveys['surveyName'] == 'ucla'].copy()
    ucla_item_cols = [f'q{i}' for i in range(1, 21)]
    ucla_items = ucla_df[['participantId'] + ucla_item_cols].dropna(subset=ucla_item_cols, how='all')
    filtered_ids = _load_filtered_ids(data_dir)
    if filtered_ids:
        ucla_items = ucla_items[ucla_items["participantId"].astype(str).isin(filtered_ids)]

    return ucla_items


def perform_efa(df: pd.DataFrame, n_factors: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Exploratory Factor Analysis with varimax rotation.

    Parameters
    ----------
    df : pd.DataFrame
        Item-level data
    n_factors : int
        Number of factors to extract

    Returns
    -------
    tuple
        (loadings, variance_explained)
    """
    df_clean = df.dropna()

    # Try factor_analyzer first (supports varimax rotation)
    try:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
        fa.fit(df_clean)
        loadings = fa.loadings_  # shape: (n_items, n_factors)
        variance_explained = fa.get_factor_variance()[0]  # eigenvalues
        return loadings, variance_explained
    except ImportError:
        pass

    # Fallback: sklearn FactorAnalysis without rotation
    # Note: sklearn's FactorAnalysis does not support rotation parameter
    from sklearn.decomposition import FactorAnalysis
    import warnings
    warnings.warn(
        "factor_analyzer not installed. Using sklearn FactorAnalysis without varimax rotation. "
        "Install factor_analyzer for proper EFA: pip install factor_analyzer",
        UserWarning
    )

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa.fit(df_clean)

    loadings = fa.components_.T  # shape: (n_items, n_factors)
    variance_explained = np.sum(fa.components_ ** 2, axis=1)  # per factor

    return loadings, variance_explained


def calculate_construct_validity(data_dir: Path | None = None, task: str = "overall") -> dict:
    """
    Assess construct validity through factorability tests and EFA.

    Returns
    -------
    dict
        Factorability and EFA results
    """
    ucla_items = load_ucla_items(data_dir=data_dir, task=task)
    item_cols = [f'q{i}' for i in range(1, 21)]
    items_df = ucla_items[item_cols].copy()

    # Reverse-code items before EFA (same as reliability.py)
    # UCLA reverse-coded items (1-indexed): 1, 5, 6, 9, 10, 15, 16, 19, 20
    ucla_reverse_items = ['q1', 'q5', 'q6', 'q9', 'q10', 'q15', 'q16', 'q19', 'q20']
    for col in ucla_reverse_items:
        if col in items_df.columns:
            items_df[col] = 5 - items_df[col]

    # Factorability tests
    kmo, kmo_individual = kmo_test(items_df)
    bartlett_chi, bartlett_p = bartlett_test(items_df)

    # EFA with 2 factors (theoretical: social vs emotional loneliness)
    try:
        loadings, variance_explained = perform_efa(items_df, n_factors=2)

        # Create loadings dataframe
        loadings_df = pd.DataFrame(
            loadings,
            index=item_cols,
            columns=['Factor1', 'Factor2']
        )
        loadings_df['communality'] = np.sum(loadings ** 2, axis=1)
    except Exception as e:
        print(f"EFA failed: {e}")
        loadings_df = None
        variance_explained = None

    return {
        'kmo': kmo,
        'kmo_interpretation': interpret_kmo(kmo),
        'bartlett_chi_sq': bartlett_chi,
        'bartlett_p': bartlett_p,
        'bartlett_significant': bartlett_p < 0.05 if pd.notna(bartlett_p) else None,
        'loadings_df': loadings_df,
        'variance_explained': variance_explained,
        'n_participants': len(items_df.dropna())
    }


# =============================================================================
# CONVERGENT/DISCRIMINANT VALIDITY
# =============================================================================

def calculate_convergent_discriminant_validity(task: str = "overall") -> pd.DataFrame:
    """
    Calculate correlation matrix between UCLA and DASS subscales.

    Returns
    -------
    pd.DataFrame
        Correlation matrix with p-values
    """
    master = load_master_dataset(task=task)

    # Ensure required columns
    required_cols = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
    for col in required_cols:
        if col not in master.columns:
            if col == 'ucla_total' and 'ucla_score' in master.columns:
                master['ucla_total'] = master['ucla_score']
            else:
                raise ValueError(f"Missing column: {col}")

    # Select variables
    validity_vars = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
    validity_df = master[validity_vars].dropna()

    # Calculate correlations with p-values
    n_vars = len(validity_vars)
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                r, p = stats.pearsonr(validity_df[validity_vars[i]], validity_df[validity_vars[j]])
                corr_matrix[i, j] = r
                p_matrix[i, j] = p

    # Create results dataframe
    results = []
    var_labels = {
        'ucla_total': 'UCLA Loneliness',
        'dass_depression': 'DASS Depression',
        'dass_anxiety': 'DASS Anxiety',
        'dass_stress': 'DASS Stress'
    }

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            results.append({
                'variable_1': var_labels[validity_vars[i]],
                'variable_2': var_labels[validity_vars[j]],
                'r': corr_matrix[i, j],
                'p_value': p_matrix[i, j],
                'significant': p_matrix[i, j] < 0.05,
                'n': len(validity_df)
            })

    return pd.DataFrame(results)


# =============================================================================
# CRITERION VALIDITY
# =============================================================================

def calculate_criterion_validity(task: str = "overall") -> pd.DataFrame:
    """
    Assess UCLA prediction of EF task outcomes.

    Returns
    -------
    pd.DataFrame
        R-squared values for UCLA predicting each EF outcome
    """
    master = load_master_dataset(task=task, merge_cognitive_summary=True)
    master = prepare_gender_variable(master)

    # Ensure ucla column
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors
    master['z_ucla'] = safe_zscore(master['ucla_total'])
    master['z_age'] = safe_zscore(master['age'])
    master['z_dass_dep'] = safe_zscore(master['dass_depression'])
    master['z_dass_anx'] = safe_zscore(master['dass_anxiety'])
    master['z_dass_str'] = safe_zscore(master['dass_stress'])

    # Define outcomes
    outcomes = []
    if 'pe_rate' in master.columns:
        outcomes.append(('pe_rate', 'WCST Perseverative Error'))
    if 'prp_bottleneck' in master.columns:
        outcomes.append(('prp_bottleneck', 'PRP Bottleneck Effect'))
    if 'stroop_interference' in master.columns:
        outcomes.append(('stroop_interference', 'Stroop Interference'))

    results = []

    for outcome_var, outcome_label in outcomes:
        df_valid = master.dropna(subset=[outcome_var, 'z_ucla', 'z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_str'])

        if len(df_valid) < 10:
            continue

        # Model without UCLA (baseline)
        try:
            formula_baseline = f"{outcome_var} ~ z_age + z_dass_dep + z_dass_anx + z_dass_str"
            model_baseline = smf.ols(formula_baseline, data=df_valid).fit()
            r2_baseline = model_baseline.rsquared
        except Exception:
            r2_baseline = np.nan

        # Model with UCLA
        try:
            formula_ucla = f"{outcome_var} ~ z_ucla + z_age + z_dass_dep + z_dass_anx + z_dass_str"
            model_ucla = smf.ols(formula_ucla, data=df_valid).fit()
            r2_ucla = model_ucla.rsquared
            ucla_beta = model_ucla.params.get('z_ucla', np.nan)
            ucla_p = model_ucla.pvalues.get('z_ucla', np.nan)
        except Exception:
            r2_ucla = np.nan
            ucla_beta = np.nan
            ucla_p = np.nan

        # Delta R-squared
        delta_r2 = r2_ucla - r2_baseline if pd.notna(r2_ucla) and pd.notna(r2_baseline) else np.nan

        results.append({
            'outcome': outcome_label,
            'outcome_variable': outcome_var,
            'n': len(df_valid),
            'r2_baseline': r2_baseline,
            'r2_with_ucla': r2_ucla,
            'delta_r2': delta_r2,
            'ucla_beta': ucla_beta,
            'ucla_p': ucla_p,
            'ucla_significant': ucla_p < 0.05 if pd.notna(ucla_p) else None
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def run(
    task: str = "overall",
    data_dir: Path | None = None,
    output_dir: Path | None = None,
):
    """Run all validity analyses."""
    data_dir = _resolve_data_dir(data_dir, task)
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VALIDITY ANALYSIS SUITE")
    print("=" * 60)

    # Construct validity (EFA)
    print("\n[1] Construct Validity (Factorability & EFA)")
    print("-" * 50)
    construct_results = calculate_construct_validity(data_dir=data_dir, task=task)

    print(f"  KMO: {construct_results['kmo']:.3f} ({construct_results['kmo_interpretation']})")
    bartlett_p = construct_results['bartlett_p']
    if pd.notna(bartlett_p) and bartlett_p < 0.001:
        p_str = "p < 0.001"
    elif pd.notna(bartlett_p):
        p_str = f"p = {bartlett_p:.3f}"
    else:
        p_str = "p = N/A"
    print(f"  Bartlett's Test: chi-sq = {construct_results['bartlett_chi_sq']:.2f}, {p_str}")
    print(f"  N participants: {construct_results['n_participants']}")

    if construct_results['loadings_df'] is not None:
        print("\n  Factor Loadings (2-factor solution):")
        loadings_df = construct_results['loadings_df']
        print(loadings_df.to_string())
        loadings_df.to_csv(output_dir / "efa_loadings.csv", encoding='utf-8-sig')
        print(f"\n  Saved: efa_loadings.csv")

    # Convergent/Discriminant validity
    print("\n[2] Convergent/Discriminant Validity")
    print("-" * 50)
    convergent_results = calculate_convergent_discriminant_validity(task=task)

    print("  UCLA vs DASS Correlations:")
    for _, row in convergent_results.iterrows():
        sig_marker = "*" if row['significant'] else ""
        print(f"    {row['variable_1']} - {row['variable_2']}: r = {row['r']:.3f}{sig_marker}")

    convergent_results.to_csv(output_dir / "validity_correlation_matrix.csv", index=False, encoding='utf-8-sig')
    print(f"\n  Saved: validity_correlation_matrix.csv")

    # Criterion validity
    print("\n[3] Criterion Validity (UCLA -> EF Prediction)")
    print("-" * 50)
    criterion_results = calculate_criterion_validity(task=task)

    print("  UCLA Prediction of Executive Function:")
    for _, row in criterion_results.iterrows():
        sig_marker = "*" if row['ucla_significant'] else ""
        print(f"    {row['outcome']}: beta = {row['ucla_beta']:.3f}, p = {row['ucla_p']:.3f}{sig_marker}")
        print(f"      R2 change = {row['delta_r2']:.4f}")

    criterion_results.to_csv(output_dir / "criterion_validity.csv", index=False, encoding='utf-8-sig')
    print(f"\n  Saved: criterion_validity.csv")

    # Summary JSON
    summary = {
        'construct_validity': {
            'kmo': construct_results['kmo'],
            'kmo_interpretation': construct_results['kmo_interpretation'],
            'bartlett_chi_sq': construct_results['bartlett_chi_sq'],
            'bartlett_p': construct_results['bartlett_p'],
            'bartlett_significant': construct_results['bartlett_significant'],
            'efa_factors': 2,
            'n_participants': construct_results['n_participants']
        },
        'convergent_discriminant_validity': {
            row['variable_1'] + '_vs_' + row['variable_2']: {
                'r': row['r'],
                'p': row['p_value'],
                'significant': row['significant']
            }
            for _, row in convergent_results.iterrows()
        },
        'criterion_validity': {
            row['outcome_variable']: {
                'beta': row['ucla_beta'],
                'p': row['ucla_p'],
                'delta_r2': row['delta_r2'],
                'significant': row['ucla_significant']
            }
            for _, row in criterion_results.iterrows()
        },
        'interpretation': {
            'kmo_threshold': '>=0.6 for adequate sampling',
            'bartlett_threshold': 'p<0.05 indicates factorability',
            'convergent_validity': 'High UCLA-Depression correlation expected (r > 0.5)',
            'discriminant_validity': 'UCLA should differ from depression (not identical construct)'
        }
    }

    with open(output_dir / "validity_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("  - efa_loadings.csv")
    print("  - validity_correlation_matrix.csv")
    print("  - criterion_validity.csv")
    print("  - validity_summary.json")
    print("=" * 60)

    return construct_results, convergent_results, criterion_results


if __name__ == "__main__":
    run()
