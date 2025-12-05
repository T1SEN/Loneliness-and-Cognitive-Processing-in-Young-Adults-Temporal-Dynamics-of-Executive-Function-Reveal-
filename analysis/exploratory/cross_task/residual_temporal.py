"""
Cross-Task Suite - Residual and Temporal Analyses
=================================================

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

# ANALYSIS 13: RESIDUALIZED UCLA ANALYSIS
# =============================================================================

@_register(
    name="residual_ucla",
    description="UCLA effects after residualizing for DASS",
    source_script="residual_ucla_analysis.py"
)
def analyze_residual_ucla(verbose: bool = True) -> pd.DataFrame:
    """
    Tests UCLA effects using DASS-residualized UCLA scores.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("RESIDUALIZED UCLA ANALYSIS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(master)}")

    # Residualize UCLA for DASS
    formula_resid = "ucla_total ~ dass_depression + dass_anxiety + dass_stress"
    model_resid = smf.ols(formula_resid, data=master).fit()
    master['ucla_residual'] = model_resid.resid
    master['z_ucla_resid'] = safe_zscore(master['ucla_residual'])

    if verbose:
        print(f"  UCLA ~ DASS: R² = {model_resid.rsquared:.3f}")
        print(f"  UCLA residual variance explained by DASS: {model_resid.rsquared*100:.1f}%")

    # Test residualized UCLA on EF
    outcomes = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    results = []

    for outcome in outcomes:
        if outcome not in master.columns:
            continue

        valid = master.dropna(subset=[outcome]).copy()
        if len(valid) < 20:
            continue

        formula = f"{outcome} ~ z_ucla_resid * C(gender_male) + z_age"
        model = smf.ols(formula, data=valid).fit(cov_type='HC3')

        # Find interaction term dynamically
        int_term = find_interaction_term(model.params.index, 'ucla_resid', 'gender')
        results.append({
            'outcome': outcome,
            'n': len(valid),
            'ucla_resid_beta': model.params.get('z_ucla_resid', np.nan),
            'ucla_resid_p': model.pvalues.get('z_ucla_resid', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan
        })

        if verbose:
            print(f"  {outcome}: UCLA_resid β={model.params.get('z_ucla_resid', np.nan):.3f}, p={model.pvalues.get('z_ucla_resid', np.nan):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "residual_ucla_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'residual_ucla_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 14: TEMPORAL CONTEXT EFFECTS
# =============================================================================

@_register(
    name="temporal_context",
    description="Time-of-day and context effects on UCLA-EF relationship",
    source_script="temporal_context_effects.py"
)
def analyze_temporal_context(verbose: bool = True) -> pd.DataFrame:
    """
    Tests whether time-of-day affects UCLA→EF relationships.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEMPORAL CONTEXT EFFECTS")
        print("=" * 70)

    master = load_cross_task_data()

    # Check for timestamp data
    from analysis.preprocessing import RESULTS_DIR
    try:
        participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8')
        if 'createdAt' in participants.columns:
            participants['test_hour'] = pd.to_datetime(participants['createdAt'], errors='coerce').dt.hour
            participants = participants.rename(columns={'participantId': 'participant_id'})
            master = master.merge(participants[['participant_id', 'test_hour']], on='participant_id', how='left')
    except Exception:
        pass

    if 'test_hour' not in master.columns or master['test_hour'].isna().all():
        if verbose:
            print("  No temporal data available. Using simulated session order.")
        master['session_order'] = range(len(master))
        master['session_half'] = np.where(master['session_order'] < len(master) // 2, 'First', 'Second')
    else:
        # Categorize time
        def categorize_hour(h):
            if pd.isna(h):
                return 'unknown'
            if 6 <= h < 12:
                return 'morning'
            elif 12 <= h < 17:
                return 'afternoon'
            else:
                return 'evening'
        master['time_category'] = master['test_hour'].apply(categorize_hour)

    master = master.dropna(subset=['pe_rate', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(master)}")

    results = []

    # Test time effect on PE
    if 'time_category' in master.columns:
        time_groups = master.groupby('time_category')['pe_rate'].agg(['mean', 'std', 'count'])
        if verbose:
            print("\n  PE by time of day:")
            print(time_groups.to_string())
        results.append({'analysis': 'time_descriptives', 'data': time_groups.to_dict()})
    elif 'session_half' in master.columns:
        formula = "pe_rate ~ z_ucla * C(gender_male) + C(session_half) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=master).fit(cov_type='HC3')

        results.append({
            'analysis': 'session_half_effect',
            'n': len(master),
            'session_half_beta': model.params.get('C(session_half)[T.Second]', np.nan),
            'session_half_p': model.pvalues.get('C(session_half)[T.Second]', np.nan)
        })

        if verbose:
            print(f"  Session half effect: β={model.params.get('C(session_half)[T.Second]', np.nan):.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "temporal_context_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'temporal_context_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 15: NONLINEAR UCLA EFFECTS
# =============================================================================

@_register(
    name="nonlinear_ucla",
    description="Tests quadratic/cubic UCLA effects on EF",
    source_script="ucla_nonlinear_effects.py"
)
def analyze_nonlinear_ucla(verbose: bool = True) -> pd.DataFrame:
    """
    Tests for quadratic and cubic UCLA effects across EF tasks.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("NONLINEAR UCLA EFFECTS")
        print("=" * 70)

    master = load_cross_task_data()
    master = master.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

    if len(master) < 30:
        print(f"  ERROR: Insufficient data (N={len(master)})")
        return pd.DataFrame()

    master['z_ucla_sq'] = master['z_ucla'] ** 2
    master['z_ucla_cu'] = master['z_ucla'] ** 3

    if verbose:
        print(f"  N = {len(master)}")

    outcomes = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    results = []

    for outcome in outcomes:
        if outcome not in master.columns:
            continue

        valid = master.dropna(subset=[outcome]).copy()
        if len(valid) < 30:
            continue

        # Linear model
        formula_linear = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model_linear = smf.ols(formula_linear, data=valid).fit()

        # Quadratic model
        formula_quad = f"{outcome} ~ z_ucla + z_ucla_sq + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model_quad = smf.ols(formula_quad, data=valid).fit()

        # Compare models
        results.append({
            'outcome': outcome,
            'n': len(valid),
            'linear_aic': model_linear.aic,
            'linear_r2': model_linear.rsquared,
            'quad_aic': model_quad.aic,
            'quad_r2': model_quad.rsquared,
            'quad_beta': model_quad.params.get('z_ucla_sq', np.nan),
            'quad_p': model_quad.pvalues.get('z_ucla_sq', np.nan),
            'best_model': 'Quadratic' if model_quad.aic < model_linear.aic else 'Linear'
        })

        if verbose:
            best = 'Quadratic' if model_quad.aic < model_linear.aic else 'Linear'
            print(f"  {outcome}: Linear AIC={model_linear.aic:.1f}, Quadratic AIC={model_quad.aic:.1f} → Best: {best}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "nonlinear_ucla_results.csv", index=False, encoding='utf-8-sig')
    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'nonlinear_ucla_results.csv'}")

    return results_df


