"""
Cross-Task Suite - Nonlinear Effect Analyses
============================================

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
# ANALYSIS 8: DOSE-RESPONSE THRESHOLD
# =============================================================================

@_register(
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

@_register(
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

@_register(
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

@_register(
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

@_register(
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

