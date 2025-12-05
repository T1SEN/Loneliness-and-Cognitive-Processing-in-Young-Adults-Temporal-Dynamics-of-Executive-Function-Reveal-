"""
Cross-Task Suite - Age and Gender Analyses
==========================================

Part of the cross_task analysis suite.
"""

from analysis.exploratory.cross_task._common import (
    AnalysisSpec,
    register_analysis,
    load_cross_task_data,
    residualize_for_covariates,
    OUTPUT_DIR,
    np, pd, stats, smf,
    PCA, StandardScaler, roc_curve, auc,
    safe_zscore, find_interaction_term,
)

# Analysis registry for this module
ANALYSES = {}

def _register(name, description, source_script):
    return register_analysis(ANALYSES, name, description, source_script)

# =============================================================================
# ANALYSIS 5: AGE GAM / DEVELOPMENTAL WINDOWS
# =============================================================================

@_register(
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

@_register(
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

@_register(
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

