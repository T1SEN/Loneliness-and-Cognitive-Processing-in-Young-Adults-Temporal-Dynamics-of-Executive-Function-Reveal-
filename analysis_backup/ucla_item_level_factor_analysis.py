"""
UCLA Item-Level Factor Analysis
================================
Identifies which facet of loneliness (emotional vs social) predicts PE

Research Questions:
1. Does UCLA have 2-3 distinct factors?
2. Which factor predicts PE in males?
3. Do factors differ by gender?

Expected: Emotional loneliness > Social loneliness for males
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from data_loader_utils import load_participants, load_wcst_summary

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/ucla_facets")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("UCLA ITEM-LEVEL FACTOR ANALYSIS")
print("=" * 80)
print("\nResearch Question: Which loneliness facet predicts PE?")
print("  - Social loneliness (lack of companionship)")
print("  - Emotional loneliness (lack of intimacy)")
print("  - Relational connectedness (reverse-scored items)\n")

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load surveys
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")

# Normalize column names
if 'participant_id' in surveys.columns:
    surveys = surveys.drop(columns=['participant_id'])
if 'participantId' in surveys.columns:
    surveys = surveys.rename(columns={'participantId': 'participant_id'})

# Get UCLA items
if 'surveyName' in surveys.columns:
    ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
elif 'survey' in surveys.columns:
    ucla_data = surveys[surveys['survey'].str.lower() == 'ucla'].copy()
else:
    raise KeyError("No survey name column found")

print(f"  Found {len(ucla_data)} UCLA responses")

# UCLA items are already in wide format (q1-q20/q21 columns)
# Extract item columns
item_cols_raw = [col for col in ucla_data.columns if col.startswith('q') and col[1:].isdigit()]
print(f"  Found UCLA item columns: {item_cols_raw}")

# Select participant_id + item columns
ucla_wide = ucla_data[['participant_id'] + item_cols_raw].copy()

print(f"  UCLA wide format: {ucla_wide.shape[0]} participants × {len(item_cols_raw)} items")

# Load WCST PE data
wcst = load_wcst_summary()

# Load participant info
participants = load_participants()

# Merge
master = participants.merge(wcst[['participant_id', 'pe_rate']], on='participant_id', how='inner')
master = master.merge(ucla_wide, on='participant_id', how='inner')

# Create gender dummy
gender_lower = master['gender'].astype(str).str.lower()
master['gender_male'] = (
    gender_lower.str.contains('male', na=False) |
    gender_lower.str.contains('남', na=False) |
    (gender_lower == 'm')
).astype(int)

# Drop missing
master = master.dropna(subset=['pe_rate', 'gender_male']).copy()

print(f"\n  Final dataset: N={len(master)}")
print(f"    Males: {master['gender_male'].sum()}")
print(f"    Females: {(1-master['gender_male']).sum()}\n")

# ============================================================================
# Exploratory Factor Analysis
# ============================================================================

print("=" * 80)
print("EXPLORATORY FACTOR ANALYSIS")
print("=" * 80)

# Get UCLA item columns (exclude participant_id, pe_rate, gender, etc.)
item_cols = [col for col in master.columns if col not in
             ['participant_id', 'age', 'gender', 'education', 'pe_rate', 'gender_male']]

print(f"\nFound {len(item_cols)} UCLA items")

# Extract items matrix
X = master[item_cols].values

# Check for missing values
n_missing = np.isnan(X).sum()
if n_missing > 0:
    print(f"  Warning: {n_missing} missing values, using median imputation")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test 2-factor and 3-factor solutions
results_fa = []

for n_factors in [2, 3]:
    print(f"\n{n_factors}-Factor Solution:")
    print("-" * 60)

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factor_scores = fa.fit_transform(X_scaled)

    # Get loadings
    loadings = fa.components_.T

    # Variance explained (approximation)
    var_explained = np.var(factor_scores, axis=0)
    total_var = np.sum(var_explained)
    prop_var = var_explained / total_var

    print(f"  Variance explained: {prop_var * 100}")

    # Add factor scores to master
    for i in range(n_factors):
        master[f'factor{i+1}_{n_factors}f'] = factor_scores[:, i]

    # Print top loadings for each factor
    for i in range(n_factors):
        print(f"\n  Factor {i+1} (variance: {prop_var[i]:.1%}):")
        # Get top 5 items by absolute loading
        top_idx = np.argsort(np.abs(loadings[:, i]))[-5:][::-1]
        for idx in top_idx:
            if idx < len(item_cols):
                item_name = str(item_cols[idx])[:50]  # Truncate long names
                loading = loadings[idx, i]
                print(f"    {loading:+.3f}  {item_name}")

    # Store loadings
    loadings_df = pd.DataFrame(loadings,
                                columns=[f'Factor{i+1}' for i in range(n_factors)])
    if len(item_cols) == len(loadings):
        loadings_df['Item'] = item_cols
    else:
        loadings_df['Item'] = [f'Item{i+1}' for i in range(len(loadings))]

    results_fa.append({
        'n_factors': n_factors,
        'loadings': loadings_df,
        'var_explained': prop_var
    })

# Save loadings
for res in results_fa:
    n_f = res['n_factors']
    output_file = OUTPUT_DIR / f"ucla_factor_loadings_{n_f}factor.csv"
    res['loadings'].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved loadings: {output_file}")

# ============================================================================
# Test Factor Effects on PE
# ============================================================================

print("\n\n" + "=" * 80)
print("FACTOR EFFECTS ON PE")
print("=" * 80)

# Use 2-factor solution as primary
factor_cols_2f = ['factor1_2f', 'factor2_2f']
factor_cols_3f = ['factor1_3f', 'factor2_3f', 'factor3_3f']

results_list = []

print("\n2-FACTOR SOLUTION:")
print("-" * 80)

for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
    df_sub = master[master['gender_male'] == gender_val].copy()
    print(f"\n{gender_name} (N={len(df_sub)}):")

    for i, factor_col in enumerate(factor_cols_2f, 1):
        if factor_col in df_sub.columns:
            # Correlation
            valid_data = df_sub[[factor_col, 'pe_rate']].dropna()
            if len(valid_data) >= 10:
                r, p = stats.pearsonr(valid_data[factor_col], valid_data['pe_rate'])
                print(f"  Factor {i} × PE:  r = {r:+.3f}, p = {p:.4f}{'  ⭐' if p < 0.05 else ''}")

                results_list.append({
                    'solution': '2-factor',
                    'factor': f'Factor{i}',
                    'gender': gender_name,
                    'n': len(valid_data),
                    'r': r,
                    'p': p
                })

print("\n\n3-FACTOR SOLUTION:")
print("-" * 80)

for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
    df_sub = master[master['gender_male'] == gender_val].copy()
    print(f"\n{gender_name} (N={len(df_sub)}):")

    for i, factor_col in enumerate(factor_cols_3f, 1):
        if factor_col in df_sub.columns:
            # Correlation
            valid_data = df_sub[[factor_col, 'pe_rate']].dropna()
            if len(valid_data) >= 10:
                r, p = stats.pearsonr(valid_data[factor_col], valid_data['pe_rate'])
                print(f"  Factor {i} × PE:  r = {r:+.3f}, p = {p:.4f}{'  ⭐' if p < 0.05 else ''}")

                results_list.append({
                    'solution': '3-factor',
                    'factor': f'Factor{i}',
                    'gender': gender_name,
                    'n': len(valid_data),
                    'r': r,
                    'p': p
                })

# Save results
if results_list:
    results_df = pd.DataFrame(results_list)
    output_file = OUTPUT_DIR / "factor_pe_correlations.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n\nSaved correlation results: {output_file}")

# ============================================================================
# Compare Total Score vs Factor Scores
# ============================================================================

print("\n\n" + "=" * 80)
print("TOTAL SCORE VS FACTOR SCORES COMPARISON")
print("=" * 80)

# Calculate UCLA total from items
master['ucla_total_items'] = master[item_cols].sum(axis=1)

print("\nMales:")
print("-" * 60)
males = master[master['gender_male'] == 1].dropna(subset=['ucla_total_items', 'pe_rate'])

r_total, p_total = stats.pearsonr(males['ucla_total_items'], males['pe_rate'])
print(f"  UCLA Total × PE:     r = {r_total:+.3f}, p = {p_total:.4f}")

for i, factor_col in enumerate(factor_cols_2f, 1):
    if factor_col in males.columns:
        valid = males[[factor_col, 'pe_rate']].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[factor_col], valid['pe_rate'])
            print(f"  Factor {i} × PE:        r = {r:+.3f}, p = {p:.4f}")

print("\nFemales:")
print("-" * 60)
females = master[master['gender_male'] == 0].dropna(subset=['ucla_total_items', 'pe_rate'])

r_total, p_total = stats.pearsonr(females['ucla_total_items'], females['pe_rate'])
print(f"  UCLA Total × PE:     r = {r_total:+.3f}, p = {p_total:.4f}")

for i, factor_col in enumerate(factor_cols_2f, 1):
    if factor_col in females.columns:
        valid = females[[factor_col, 'pe_rate']].dropna()
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[factor_col], valid['pe_rate'])
            print(f"  Factor {i} × PE:        r = {r:+.3f}, p = {p:.4f}")

# ============================================================================
# Save Factor Scores
# ============================================================================

# Save participant-level factor scores
factor_scores_df = master[['participant_id', 'gender_male', 'pe_rate', 'ucla_total_items'] +
                           factor_cols_2f + factor_cols_3f].copy()

output_file = OUTPUT_DIR / "participant_factor_scores.csv"
factor_scores_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nSaved participant factor scores: {output_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:")
print("-" * 80)

# Find strongest factor for males
males_results = [r for r in results_list if r['gender'] == 'Male' and r['solution'] == '2-factor']
if males_results:
    males_results_sorted = sorted(males_results, key=lambda x: abs(x['r']), reverse=True)
    strongest = males_results_sorted[0]
    print(f"\n✓ Strongest factor for MALES:")
    print(f"  {strongest['factor']}: r = {strongest['r']:+.3f}, p = {strongest['p']:.4f}")

# Find strongest factor for females
females_results = [r for r in results_list if r['gender'] == 'Female' and r['solution'] == '2-factor']
if females_results:
    females_results_sorted = sorted(females_results, key=lambda x: abs(x['r']), reverse=True)
    strongest = females_results_sorted[0]
    print(f"\n✓ Strongest factor for FEMALES:")
    print(f"  {strongest['factor']}: r = {strongest['r']:+.3f}, p = {strongest['p']:.4f}")

print("\n" + "=" * 80)
