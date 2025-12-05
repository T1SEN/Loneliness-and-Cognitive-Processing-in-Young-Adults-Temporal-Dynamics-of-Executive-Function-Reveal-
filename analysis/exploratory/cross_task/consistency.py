"""
Cross-Task Suite - Consistency Analyses
=======================================

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
# ANALYSIS 1: CROSS-TASK CONSISTENCY
# =============================================================================

@_register(
    name="consistency",
    description="Within-person variability across EF tasks (CV, range)",
    source_script="cross_task_consistency.py"
)
def analyze_consistency(verbose: bool = True) -> pd.DataFrame:
    """
    Research Question: Are lonely individuals inconsistent across tasks
    (high cross-task variability) or consistently impaired?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-TASK CONSISTENCY ANALYSIS")
        print("=" * 70)

    master = load_cross_task_data()

    # EF metrics to analyze
    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]
    available_cols = [c for c in ef_cols if c in master.columns]

    if len(available_cols) < 2:
        print("  ERROR: Need at least 2 EF metrics")
        return pd.DataFrame()

    if verbose:
        print(f"  EF metrics: {available_cols}")

    # Standardize EF metrics using NaN-safe z-score
    for col in available_cols:
        if master[col].notna().sum() > 5:
            master[f'{col}_z'] = safe_zscore(master[col])

    z_cols = [f'{c}_z' for c in available_cols]

    # Compute cross-task metrics per participant
    results = []

    for _, row in master.iterrows():
        pid = row.get("participant_id")

        ef_values = []
        for col in z_cols:
            if col in row and pd.notna(row[col]):
                ef_values.append(row[col])

        if len(ef_values) < 2:
            continue

        ef_mean = np.mean(ef_values)
        ef_sd = np.std(ef_values)
        cross_task_cv = ef_sd / abs(ef_mean) if ef_mean != 0 else np.nan
        cross_task_range = max(ef_values) - min(ef_values)

        results.append({
            "participant_id": pid,
            "cross_task_cv": cross_task_cv,
            "cross_task_range": cross_task_range,
            "cross_task_mean": ef_mean,
            "cross_task_sd": ef_sd,
            "n_tasks": len(ef_values),
            "ucla_total": row.get("ucla_total"),
            "gender_male": row.get("gender_male"),
            "z_ucla": row.get("z_ucla"),
            "z_dass_dep": row.get("z_dass_dep"),
            "z_dass_anx": row.get("z_dass_anx"),
            "z_dass_str": row.get("z_dass_str"),
            "z_age": row.get("z_age"),
        })

    ct_df = pd.DataFrame(results)

    if verbose:
        print(f"  Participants with >= 2 tasks: {len(ct_df)}")

    # DASS/나이/성별 통제 회귀로 UCLA 효과 검정
    if len(ct_df) > 20:
        model_df = ct_df.dropna(subset=[
            'cross_task_cv', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'
        ])
        if len(model_df) >= 20:
            formula = "cross_task_cv ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=model_df).fit(cov_type='HC3')
            if verbose:
                print("\n  UCLA effect on cross-task CV (DASS/age/gender controlled):")
                print(f"    UCLA beta = {model.params.get('z_ucla', np.nan):.3f}, p = {model.pvalues.get('z_ucla', np.nan):.4f}")
                int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
                if int_term is not None and int_term in model.params:
                    print(f"    UCLA x Gender beta = {model.params[int_term]:.3f}, p = {model.pvalues[int_term]:.4f}")

    ct_df.to_csv(OUTPUT_DIR / "cross_task_consistency.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'cross_task_consistency.csv'}")

    return ct_df


# =============================================================================
# ANALYSIS 2: CROSS-TASK CORRELATIONS
# =============================================================================

@_register(
    name="correlations",
    description="Correlation matrix between EF tasks",
    source_script="cross_task_integration.py"
)
def analyze_correlations(verbose: bool = True) -> pd.DataFrame:
    """
    Examine cross-task correlations to assess shared variance.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-TASK CORRELATIONS")
        print("=" * 70)

    master = load_cross_task_data()

    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck", "wcst_accuracy"]
    available_cols = [c for c in ef_cols if c in master.columns]

    if len(available_cols) < 2:
        print("  ERROR: Need at least 2 EF metrics")
        return pd.DataFrame()

    if verbose:
        print(f"  EF metrics: {available_cols}")

    covars = ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
    base_df = master.dropna(subset=available_cols + covars).copy()

    # 부분상관: 공변량 잔차화 후 상관
    resid_df = residualize_for_covariates(base_df, available_cols, covars)
    resid_cols = [f"{c}_resid" for c in available_cols if f"{c}_resid" in resid_df.columns]
    corr_matrix = resid_df[resid_cols].corr()

    if verbose and not corr_matrix.empty:
        print(f"\n  Partial Correlation Matrix (covariates removed):")
        print(corr_matrix.round(3).to_string())

    # Long-format results
    results = []
    for i, col1 in enumerate(resid_cols):
        for j, col2 in enumerate(resid_cols):
            if i < j:
                df_pair = resid_df[[col1, col2]].dropna()
                n = len(df_pair)

                if n > 5:
                    r, p = stats.pearsonr(df_pair[col1], df_pair[col2])

                    results.append({
                        'task1': col1.replace('_resid', ''),
                        'task2': col2.replace('_resid', ''),
                        'partial_r': r,
                        'p': p,
                        'n': n
                    })

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"  {col1.replace('_resid','')} x {col2.replace('_resid','')}: r={r:.3f}, p={p:.4f}, n={n}{sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "cross_task_correlations.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'cross_task_correlations.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: META-CONTROL FACTOR (PCA)
# =============================================================================

@_register(
    name="meta_control",
    description="PCA-based meta-control factor across EF tasks",
    source_script="cross_task_meta_control.py"
)
def analyze_meta_control(verbose: bool = True) -> pd.DataFrame:
    """
    Extract a latent 'meta-control' factor via PCA across EF tasks.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("META-CONTROL FACTOR (PCA)")
        print("=" * 70)

    master = load_cross_task_data()

    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]
    available_cols = [c for c in ef_cols if c in master.columns]

    if len(available_cols) < 2:
        print("  ERROR: Need at least 2 EF metrics")
        return pd.DataFrame()

    # Standardize + 공변량 잔차화 후 PCA
    df_pca = master[available_cols + ['participant_id', 'ucla_total', 'gender_male',
                                       'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']].dropna(
        subset=available_cols + ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']
    ).copy()

    if len(df_pca) < 20:
        print(f"  ERROR: Insufficient data (N={len(df_pca)})")
        return pd.DataFrame()

    if verbose:
        print(f"  N = {len(df_pca)}")
        print(f"  EF metrics: {available_cols}")

    # 공변량 잔차화 후 스케일링
    resid = residualize_for_covariates(df_pca, available_cols, ['z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age'])
    resid_cols = [f"{c}_resid" for c in available_cols if f"{c}_resid" in resid.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(resid[resid_cols])

    # PCA
    pca = PCA(n_components=min(len(available_cols), 3))
    pca_scores = pca.fit_transform(X_scaled)

    df_pca['meta_control'] = pca_scores[:, 0]  # First PC

    if verbose:
        print(f"\n  PCA Results:")
        print(f"    PC1 explained variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    PC1 loadings:")
        for col, loading in zip(available_cols, pca.components_[0]):
            print(f"      {col}: {loading:.3f}")

    # Standardize for regression using NaN-safe z-score
    df_pca['z_ucla'] = safe_zscore(df_pca['ucla_total'])

    # Test UCLA effect on meta-control (DASS-controlled)
    try:
        formula = "meta_control ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=df_pca).fit()

        if verbose:
            print(f"\n  Meta-control ~ UCLA (DASS-controlled):")
            if 'z_ucla' in model.params:
                beta = model.params['z_ucla']
                p = model.pvalues['z_ucla']
                print(f"    UCLA main effect: beta={beta:.3f}, p={p:.4f}")

            int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
            if int_term is not None and int_term in model.params:
                beta = model.params[int_term]
                p = model.pvalues[int_term]
                print(f"    UCLA x Gender: beta={beta:.3f}, p={p:.4f}")

    except Exception as e:
        if verbose:
            print(f"  Regression error: {e}")

    # Save results
    pca_loadings = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca.components_))],
        'explained_variance': pca.explained_variance_ratio_,
        **{col: pca.components_[:, i] for i, col in enumerate(available_cols)}
    })
    pca_loadings.to_csv(OUTPUT_DIR / "meta_control_loadings.csv", index=False, encoding='utf-8-sig')

    df_pca[['participant_id', 'meta_control', 'ucla_total', 'gender_male']].to_csv(
        OUTPUT_DIR / "meta_control_scores.csv", index=False, encoding='utf-8-sig'
    )

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'meta_control_loadings.csv'}")
        print(f"  Output: {OUTPUT_DIR / 'meta_control_scores.csv'}")

    return pca_loadings


# =============================================================================
# ANALYSIS 4: TASK ORDER EFFECTS
# =============================================================================

@_register(
    name="order_effects",
    description="Test if task order affects EF outcomes or UCLA effects",
    source_script="task_order_effects.py"
)
def analyze_order_effects(verbose: bool = True) -> pd.DataFrame:
    """
    Test if task presentation order affects results.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TASK ORDER EFFECTS")
        print("=" * 70)

    master = load_cross_task_data()

    # Check if task order info exists
    order_cols = ['task_order', 'stroop_order', 'wcst_order', 'prp_order']
    available_order = [c for c in order_cols if c in master.columns]

    if len(available_order) == 0:
        if verbose:
            print("  No task order information available in dataset.")
            print("  Creating simulated analysis based on participant sequence.")

        # Use participant_id order as proxy
        master['session_order'] = master.groupby('participant_id').cumcount()

    ef_cols = ["pe_rate", "stroop_interference", "prp_bottleneck"]
    available_ef = [c for c in ef_cols if c in master.columns]

    if verbose:
        print(f"  EF metrics: {available_ef}")

    results = []

    for outcome in available_ef:
        df = master.dropna(subset=[outcome, 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']).copy()

        if len(df) < 30:
            continue

        # Split into early vs late participants (proxy order)
        median_idx = len(df) // 2
        df['session_half'] = np.where(df.index < df.index[median_idx], 'First', 'Second')

        # 공변량 통제 회귀: session_half 효과와 UCLA 상호작용 포함
        df['session_half'] = df['session_half'].astype('category')
        formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age + C(session_half)"
        model = smf.ols(formula, data=df).fit(cov_type='HC3')

        int_term = find_interaction_term(model.params.index, 'ucla', 'gender')
        results.append({
            'outcome': outcome,
            'n': len(df),
            'ucla_beta': model.params.get('z_ucla', np.nan),
            'ucla_p': model.pvalues.get('z_ucla', np.nan),
            'interaction_beta': model.params.get(int_term, np.nan) if int_term else np.nan,
            'interaction_p': model.pvalues.get(int_term, np.nan) if int_term else np.nan,
            'session_half_effect': model.params.get('C(session_half)[T.Second]', np.nan),
            'session_half_p': model.pvalues.get('C(session_half)[T.Second]', np.nan),
        })

        if verbose:
            print(f"  {outcome}: UCLA beta={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "order_effects.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'order_effects.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 5: AGE GAM / DEVELOPMENTAL WINDOWS
# =============================================================================

