"""
Gold Standard Analysis Pipeline
================================

Unified runner for all publication-ready confirmatory analyses.

All analyses:
- Control for DASS-21 subscales (depression, anxiety, stress)
- Control for age
- Test UCLA × Gender interactions
- Use HC3 robust standard errors
- Apply FDR correction for multiple comparisons

Usage:
    # Run all analyses
    python -m analysis.gold_standard.pipeline

    # Run from code
    from analysis.gold_standard import pipeline
    pipeline.run()
    pipeline.run(analyses=['wcst_pe', 'stroop_interference'])
    pipeline.run(tier=1)  # Only Tier 1 confirmatory family

Output:
    results/gold_standard/
    ├── summary_results.csv          # All model coefficients
    ├── hierarchical_comparison.csv  # Model comparison (ΔR²)
    ├── significant_effects.csv      # Effects surviving FDR correction
    ├── model_comparison_plot.png    # Visualization
    └── FINAL_REPORT.txt             # Human-readable summary
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Project imports
from analysis.preprocessing import load_master_dataset, RESULTS_DIR
from analysis.preprocessing import load_prp_trials
from analysis.preprocessing import (
    safe_zscore,
    prepare_gender_variable,
    find_interaction_term
)
from sklearn.decomposition import PCA

# Paths
THIS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = THIS_DIR / "analyses.yml"
OUTPUT_DIR = Path("results/gold_standard")

# Standard formula template (MANDATORY for all Gold Standard analyses)
FORMULA_TEMPLATE = "{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"

# Hierarchical model formulas
HIERARCHICAL_FORMULAS = {
    "model0": "{outcome} ~ z_age + C(gender_male)",
    "model1": "{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str",
    "model2": "{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla",
    "model3": "{outcome} ~ z_age + C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_ucla * C(gender_male)",
}


def load_config() -> Dict[str, Any]:
    """Load analysis configuration from YAML."""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _normalize_tier_filter(tier: Optional[Union[int, Iterable[int]]]) -> Optional[set]:
    """
    Normalize tier input into a set of ints or None.

    Accepts a single int or an iterable of ints. Strings are not treated as iterables.
    """
    if tier is None:
        return None
    if isinstance(tier, Iterable) and not isinstance(tier, (str, bytes)):
        return {int(t) for t in tier}
    return {int(tier)}


def compute_prp_extended_metrics(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extended PRP metrics: SOA slope, CV at short SOA.

    Adds columns:
    - prp_soa_slope: Linear slope of T2 RT across SOA bins
    - prp_cv_short: Coefficient of variation at short SOA
    """
    print("  [COMPUTE] PRP extended metrics (SOA slope, CV)...")

    try:
        prp_trials, _ = load_prp_trials(use_cache=True)

        # Compute per participant
        results = []
        for pid in prp_trials['participant_id'].unique():
            pdata = prp_trials[prp_trials['participant_id'] == pid]

            if len(pdata) < 20:
                continue

            # SOA slope via linear regression
            soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in pdata.columns else 'soa'
            rt_col = 't2_rt_ms' if 't2_rt_ms' in pdata.columns else 't2_rt'

            valid = pdata.dropna(subset=[soa_col, rt_col])
            if len(valid) >= 10:
                slope, _, _, _, _ = linregress(valid[soa_col], valid[rt_col])
                prp_soa_slope = slope
            else:
                prp_soa_slope = np.nan

            # CV at short SOA (with guard against division by zero)
            short_trials = pdata[(pdata[soa_col] <= 150) & (pdata[rt_col] > 0)]
            short_mean = short_trials[rt_col].mean() if len(short_trials) >= 5 else 0
            if len(short_trials) >= 5 and short_mean > 0:
                prp_cv_short = short_trials[rt_col].std() / short_mean
            else:
                prp_cv_short = np.nan

            results.append({
                'participant_id': pid,
                'prp_soa_slope': prp_soa_slope,
                'prp_cv_short': prp_cv_short
            })

        prp_metrics = pd.DataFrame(results)
        master = master.merge(prp_metrics, on='participant_id', how='left')
        print(f"    Added prp_soa_slope, prp_cv_short for {len(prp_metrics)} participants")

    except Exception as e:
        print(f"    WARNING: Could not compute PRP extended metrics: {e}")
        master['prp_soa_slope'] = np.nan
        master['prp_cv_short'] = np.nan

    return master


def compute_exgaussian_params(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ex-Gaussian parameters (mu, sigma, tau) for PRP task.

    Adds columns:
    - prp_mu: Ex-Gaussian mu (mean of Gaussian component)
    - prp_sigma: Ex-Gaussian sigma (SD of Gaussian component)
    - prp_tau: Ex-Gaussian tau (exponential tail parameter)
    """
    print("  [COMPUTE] Ex-Gaussian parameters (mu, sigma, tau)...")

    try:
        prp_trials, _ = load_prp_trials(use_cache=True)
        rt_col = 't2_rt_ms' if 't2_rt_ms' in prp_trials.columns else 't2_rt'

        # Simple method: tau approximated by skewness-based formula
        results = []
        for pid in prp_trials['participant_id'].unique():
            pdata = prp_trials[prp_trials['participant_id'] == pid]
            rts = pdata[rt_col].dropna()

            if len(rts) < 20:
                continue

            # Method of moments approximation
            m = rts.mean()
            s = rts.std()
            skew = ((rts - m) ** 3).mean() / (s ** 3)

            # Tau from skewness: tau ≈ s * (skew/2)^(1/3) for positive skew
            if skew > 0:
                tau = s * ((skew / 2) ** (1/3))
            else:
                tau = 0

            # mu = mean - tau, sigma = sqrt(var - tau^2)
            mu = m - tau
            sigma_sq = s**2 - tau**2
            sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else s

            results.append({
                'participant_id': pid,
                'prp_mu': mu,
                'prp_sigma': sigma,
                'prp_tau': tau
            })

        exgauss_df = pd.DataFrame(results)
        master = master.merge(exgauss_df, on='participant_id', how='left')
        print(f"    Added prp_mu, prp_sigma, prp_tau for {len(exgauss_df)} participants")

    except Exception as e:
        print(f"    WARNING: Could not compute Ex-Gaussian params: {e}")
        master['prp_mu'] = np.nan
        master['prp_sigma'] = np.nan
        master['prp_tau'] = np.nan

    return master


def compute_meta_control_score(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute meta-control factor via PCA across EF tasks.

    Adds column:
    - meta_control_score: First principal component across standardized EF metrics
    """
    print("  [COMPUTE] Meta-control factor (PCA)...")

    ef_cols = ['pe_rate', 'stroop_interference', 'prp_bottleneck']
    available_cols = [c for c in ef_cols if c in master.columns]

    if len(available_cols) < 2:
        print(f"    WARNING: Need at least 2 EF metrics, found {len(available_cols)}")
        master['meta_control_score'] = np.nan
        return master

    try:
        # Get complete cases
        df_pca = master.dropna(subset=available_cols).copy()

        if len(df_pca) < 20:
            print(f"    WARNING: Insufficient data for PCA (N={len(df_pca)})")
            master['meta_control_score'] = np.nan
            return master

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(df_pca[available_cols])

        # PCA
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X)

        df_pca['meta_control_score'] = scores[:, 0]

        # Merge back
        master = master.merge(
            df_pca[['participant_id', 'meta_control_score']],
            on='participant_id',
            how='left'
        )

        print(f"    Added meta_control_score for {len(df_pca)} participants")
        print(f"    PC1 explained variance: {pca.explained_variance_ratio_[0]*100:.1f}%")

    except Exception as e:
        print(f"    WARNING: Could not compute meta-control score: {e}")
        master['meta_control_score'] = np.nan

    return master


def prepare_data(force_rebuild: bool = False) -> pd.DataFrame:
    """Load and prepare master dataset with standardized predictors."""
    print("[DATA] Loading master dataset...")

    master = load_master_dataset(use_cache=not force_rebuild, force_rebuild=force_rebuild, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    # Ensure ucla_total exists
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Required columns
    required = ['participant_id', 'ucla_total', 'gender_male', 'age',
                'dass_depression', 'dass_anxiety', 'dass_stress']

    missing = [col for col in required if col not in master.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean and standardize using NaN-safe z-score
    df = master.dropna(subset=required).copy()

    df['z_age'] = safe_zscore(df['age'])
    df['z_ucla'] = safe_zscore(df['ucla_total'])
    df['z_dass_dep'] = safe_zscore(df['dass_depression'])
    df['z_dass_anx'] = safe_zscore(df['dass_anxiety'])
    df['z_dass_str'] = safe_zscore(df['dass_stress'])

    print(f"  Total N = {len(df)}")
    print(f"    Males: {(df['gender_male'] == 1).sum()}")
    print(f"    Females: {(df['gender_male'] == 0).sum()}")

    # Compute derived variables for extended analyses
    print("\n[DATA] Computing derived variables...")
    df = compute_prp_extended_metrics(df)
    df = compute_exgaussian_params(df)
    df = compute_meta_control_score(df)

    return df


def run_hierarchical_regression(
    data: pd.DataFrame,
    outcome: str,
    label: str,
    robust: str = "HC3",
    min_n: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Run hierarchical regression with 4 models.

    Model 0: Covariates only (age, gender)
    Model 1: + DASS subscales
    Model 2: + UCLA
    Model 3: + UCLA × Gender
    """
    df = data.dropna(subset=[outcome]).copy()

    if len(df) < min_n:
        print(f"  [SKIP] {label}: N={len(df)} < {min_n}")
        return None

    models = {}
    for model_name, formula_template in HIERARCHICAL_FORMULAS.items():
        formula = formula_template.format(outcome=outcome)
        models[model_name] = smf.ols(formula, data=df).fit(cov_type=robust)

    # HC3-based Wald tests for key increments
    ucla_wald = models['model2'].wald_test('z_ucla = 0', use_f=True)

    # Dynamically detect interaction term for Wald test
    int_term_for_wald = find_interaction_term(models['model3'].params.index, 'ucla', 'gender')
    if int_term_for_wald is not None:
        interaction_wald = models['model3'].wald_test(f'{int_term_for_wald} = 0', use_f=True)
        interaction_wald_p = float(interaction_wald.pvalue)
    else:
        interaction_wald_p = np.nan
        warnings.warn("Interaction term not found for Wald test")

    # Model comparisons
    anova_1v0 = anova_lm(models['model0'], models['model1'])
    anova_2v1 = anova_lm(models['model1'], models['model2'])
    anova_3v2 = anova_lm(models['model2'], models['model3'])

    results = {
        'outcome': label,
        'outcome_column': outcome,
        'n': len(df),

        # R² values
        'model0_r2': models['model0'].rsquared,
        'model1_r2': models['model1'].rsquared,
        'model2_r2': models['model2'].rsquared,
        'model3_r2': models['model3'].rsquared,

        # AIC
        'model0_aic': models['model0'].aic,
        'model1_aic': models['model1'].aic,
        'model2_aic': models['model2'].aic,
        'model3_aic': models['model3'].aic,

        # ΔR² and tests
        'delta_r2_dass': models['model1'].rsquared - models['model0'].rsquared,
        'p_dass': anova_1v0['Pr(>F)'][1],

        'delta_r2_ucla': models['model2'].rsquared - models['model1'].rsquared,
        'p_ucla': float(ucla_wald.pvalue),  # HC3 Wald test (robust to heteroscedasticity)
        'p_ucla_ols': anova_2v1['Pr(>F)'][1],  # OLS-based (for reference)

        'delta_r2_interaction': models['model3'].rsquared - models['model2'].rsquared,
        'p_interaction': interaction_wald_p,  # HC3 Wald test (robust)
        'p_interaction_ols': anova_3v2['Pr(>F)'][1],  # OLS-based (for reference)

        # UCLA main effect (from Model 2)
        'ucla_beta': models['model2'].params.get('z_ucla', np.nan),
        'ucla_se': models['model2'].bse.get('z_ucla', np.nan),
        'ucla_p': models['model2'].pvalues.get('z_ucla', np.nan),
    }

    # Interaction term (from Model 3) - dynamic detection
    int_term = find_interaction_term(models['model3'].params.index, 'ucla', 'gender')
    if int_term is not None and int_term in models['model3'].params:
        results['interaction_beta'] = models['model3'].params[int_term]
        results['interaction_se'] = models['model3'].bse[int_term]
        results['interaction_p'] = models['model3'].pvalues[int_term]
        results['interaction_term'] = int_term  # Track actual term name
    else:
        results['interaction_beta'] = np.nan
        results['interaction_se'] = np.nan
        results['interaction_p'] = np.nan
        results['interaction_term'] = None

    # Gender-stratified effects
    for gender, gender_val, label in [('female', 0, 'female'), ('male', 1, 'male')]:
        subset = df[df['gender_male'] == gender_val]
        if len(subset) >= 15:
            formula = f"{outcome} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            m = smf.ols(formula, data=subset).fit(cov_type=robust)
            results[f'{label}_ucla_beta'] = m.params.get('z_ucla', np.nan)
            results[f'{label}_ucla_p'] = m.pvalues.get('z_ucla', np.nan)
        else:
            results[f'{label}_ucla_beta'] = np.nan
            results[f'{label}_ucla_p'] = np.nan

    # Diagnostics: VIF, heteroscedasticity, residual normality (sampled)
    try:
        X = df[['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']].dropna()
        vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        results['vif_max'] = max(vif_vals) if vif_vals else np.nan
    except Exception:
        results['vif_max'] = np.nan

    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(models['model3'].resid, models['model3'].model.exog)
        results['bp_p'] = bp_p
    except Exception:
        results['bp_p'] = np.nan

    try:
        resid_sample = models['model3'].resid
        if len(resid_sample) > 5000:
            resid_sample = np.random.default_rng(42).choice(resid_sample, size=5000, replace=False)
        from scipy.stats import shapiro
        _, sw_p = shapiro(resid_sample)
        results['shapiro_p'] = sw_p
    except Exception:
        results['shapiro_p'] = np.nan

    return results


def apply_fdr_by_tier(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FDR correction within each tier family.

    If no tier is present, falls back to single-family correction across all rows.
    """
    df = results_df.copy()

    tier_values = df['tier'].dropna().unique() if 'tier' in df.columns else []
    tier_values = list(tier_values)

    for col in ['p_ucla', 'p_interaction']:
        adj = np.full(len(df), np.nan, dtype=float)

        # No tier metadata -> apply once across all finite p-values
        if not tier_values:
            pvals = df[col].to_numpy()
            mask = np.isfinite(pvals)
            if mask.sum() > 0:
                adj[mask] = multipletests(pvals[mask], method='fdr_bh')[1]
            df[f'{col}_fdr'] = adj
            continue

        # Tier-aware correction
        for tier_val in tier_values:
            mask = (df['tier'] == tier_val) & np.isfinite(df[col])
            if mask.sum() > 0:
                tier_adj = multipletests(df.loc[mask, col], method='fdr_bh')[1]
                adj[mask] = tier_adj

        df[f'{col}_fdr'] = adj

    return df


def create_visualization(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create model comparison visualization."""
    n_plots = min(len(results_df), 4)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(results_df.head(4).iterrows()):
        ax = axes[i]

        r2_values = [row['model0_r2'], row['model1_r2'], row['model2_r2'], row['model3_r2']]
        model_labels = ['Covariates\nOnly', '+ DASS', '+ UCLA', '+ UCLA×Gender']
        colors = ['#95A5A6', '#3498DB', '#E74C3C', '#2ECC71']

        bars = ax.bar(range(4), r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        delta_r2 = [0, row['delta_r2_dass'], row['delta_r2_ucla'], row['delta_r2_interaction']]
        p_vals = [1.0, row['p_dass'], row['p_ucla'], row['p_interaction']]

        for j, (bar, delta, p) in enumerate(zip(bars, delta_r2, p_vals)):
            if j > 0:
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.text(j, r2_values[j] + 0.01, f'ΔR²={delta:.3f}\n{sig}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(4))
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title(f"{row['outcome']}\n(N={row['n']:.0f})", fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(r2_values) * 1.25)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for i in range(n_plots, 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results_df: pd.DataFrame, output_path: Path) -> None:
    """Generate human-readable final report with tiered families."""

    tier_values = sorted(results_df['tier'].dropna().unique()) if 'tier' in results_df.columns else []

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GOLD STANDARD ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("METHODOLOGY\n")
        f.write("-" * 40 + "\n")
        f.write("All analyses control for:\n")
        f.write("  - DASS-21 subscales (depression, anxiety, stress)\n")
        f.write("  - Age\n")
        f.write("  - Gender (main effect + interaction with UCLA)\n")
        f.write("  - HC3 robust standard errors\n")
        f.write("  - FDR correction within hypothesis families (tier-aware)\n")
        if tier_values:
            f.write("Tier definition:\n")
            f.write("  - Tier 1: Core confirmatory EF metrics (WCST, Stroop, PRP bottleneck)\n")
            f.write("  - Tier 2: Mechanistic/derived metrics (PRP slope/CV, ex-Gaussian, meta-control)\n")
        f.write("\n")

        f.write("RESULTS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total hypotheses tested: {len(results_df)}\n")

        if tier_values:
            for tier_val in tier_values:
                tdf = results_df[results_df['tier'] == tier_val]
                main_p = (tdf['p_ucla'] < 0.05).sum()
                main_q = (tdf['p_ucla_fdr'] < 0.05).sum()
                int_p = (tdf['p_interaction'] < 0.05).sum()
                int_q = (tdf['p_interaction_fdr'] < 0.05).sum()
                f.write(f"Tier {tier_val} (n={len(tdf)}): main p<0.05={main_p}, q<0.05={main_q}; "
                        f"interaction p<0.05={int_p}, q<0.05={int_q}\n")
        else:
            main_p = (results_df['p_ucla'] < 0.05).sum()
            main_q = (results_df['p_ucla_fdr'] < 0.05).sum()
            int_p = (results_df['p_interaction'] < 0.05).sum()
            int_q = (results_df['p_interaction_fdr'] < 0.05).sum()
            f.write(f"UCLA main effects: p<0.05={main_p}, q<0.05={main_q}\n")
            f.write(f"UCLA × Gender interactions: p<0.05={int_p}, q<0.05={int_q}\n")
        f.write("\n")

        # Significant results (FDR) by tier
        if tier_values:
            for tier_val in tier_values:
                tdf = results_df[results_df['tier'] == tier_val]
                sig_main = tdf[tdf['p_ucla_fdr'] < 0.05]
                sig_int = tdf[tdf['p_interaction_fdr'] < 0.05]
                f.write(f"TIER {tier_val} SIGNIFICANT (q<0.05)\n")
                if len(sig_main) == 0 and len(sig_int) == 0:
                    f.write("  - None\n\n")
                    continue

                if len(sig_main) > 0:
                    f.write("  UCLA main effects:\n")
                    for _, row in sig_main.iterrows():
                        f.write(f"    - {row['outcome']}: β={row['ucla_beta']:.3f}, q={row['p_ucla_fdr']:.4f} "
                                f"(p={row['p_ucla']:.4f})\n")
                if len(sig_int) > 0:
                    f.write("  UCLA × Gender interactions:\n")
                    for _, row in sig_int.iterrows():
                        f.write(f"    - {row['outcome']}: β={row['interaction_beta']:.3f}, q={row['p_interaction_fdr']:.4f} "
                                f"(p={row['p_interaction']:.4f})\n")
                        f.write(f"        Female: β={row['female_ucla_beta']:.3f}, p={row['female_ucla_p']:.4f}\n")
                        f.write(f"        Male: β={row['male_ucla_beta']:.3f}, p={row['male_ucla_p']:.4f}\n")
                f.write("\n")
        else:
            sig_main = results_df[results_df['p_ucla_fdr'] < 0.05]
            sig_int = results_df[results_df['p_interaction_fdr'] < 0.05]
            if len(sig_main) > 0:
                f.write("SIGNIFICANT UCLA MAIN EFFECTS (q<0.05):\n")
                for _, row in sig_main.iterrows():
                    f.write(f"  - {row['outcome']}: β={row['ucla_beta']:.3f}, q={row['p_ucla_fdr']:.4f}\n")
                f.write("\n")

            if len(sig_int) > 0:
                f.write("SIGNIFICANT UCLA × GENDER INTERACTIONS (q<0.05):\n")
                for _, row in sig_int.iterrows():
                    f.write(f"  - {row['outcome']}: β={row['interaction_beta']:.3f}, q={row['p_interaction_fdr']:.4f}\n")
                    f.write(f"      Female: β={row['female_ucla_beta']:.3f}, p={row['female_ucla_p']:.4f}\n")
                    f.write(f"      Male: β={row['male_ucla_beta']:.3f}, p={row['male_ucla_p']:.4f}\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n")

        has_main = (results_df['p_ucla_fdr'] < 0.05).any()
        has_int = (results_df['p_interaction_fdr'] < 0.05).any()

        if not has_main and not has_int:
            f.write("No UCLA effects survived FDR within tiers after DASS control.\n")
            f.write("Loneliness effects appear confounded with mood/anxiety.\n")
        elif has_int:
            f.write("UCLA × Gender interactions detected after FDR.\n")
            f.write("Loneliness effects are gender-specific within at least one tier.\n")
        else:
            f.write("UCLA main effects survived DASS control after FDR.\n")
            f.write("Evidence supports loneliness effects beyond mood/anxiety in the tested tier(s).\n")


def run(
    analyses: Optional[List[str]] = None,
    tier: Optional[Union[int, Iterable[int]]] = None,
    force_rebuild: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run Gold Standard analyses.

    Args:
        analyses: List of analysis names to run (default: all)
        tier: Tier(s) to run (1 = core confirmatory, 2 = derived/mechanistic)
        force_rebuild: Force rebuild of master dataset
        verbose: Print progress

    Returns:
        DataFrame with all results
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("GOLD STANDARD ANALYSIS PIPELINE")
        print("=" * 80)

    # Load config
    config = load_config()

    # Optional filters
    tier_filter = _normalize_tier_filter(tier)
    if analyses:
        config = {k: v for k, v in config.items() if k in analyses}
    if tier_filter is not None:
        config = {
            k: v for k, v in config.items()
            if v is not None and v.get('tier') in tier_filter
        }

    if verbose:
        print(f"\nAnalyses to run: {list(config.keys())}")
        if tier_filter is not None:
            print(f"Tier filter applied: {sorted(tier_filter)}")

    if not config:
        print("No analyses to run after filtering.")
        return pd.DataFrame()

    # Load data
    df = prepare_data(force_rebuild=force_rebuild)

    # Run analyses
    all_results = []

    if verbose:
        print("\n[ANALYSES] Running hierarchical regressions...")

    for name, spec in config.items():
        if spec is None:
            continue

        outcome_col = spec.get('outcome_column')
        description = spec.get('description', name)

        if outcome_col not in df.columns:
            if verbose:
                # Improved skip warnings based on source
                source = spec.get('source_script', '')
                if spec.get('requires_pca'):
                    print(f"  [SKIP] {name}: PCA column '{outcome_col}' not found. Run loneliness_exec_models.py first.")
                elif 'exgaussian' in source.lower():
                    print(f"  [SKIP] {name}: Ex-Gaussian column '{outcome_col}' not found. Run prp_exgaussian_dass_controlled.py first.")
                else:
                    print(f"  [SKIP] {name}: column '{outcome_col}' not found")
            continue

        if verbose:
            print(f"  Running: {name}...")

        result = run_hierarchical_regression(
            data=df,
            outcome=outcome_col,
            label=description,
            robust=spec.get('robust', 'HC3'),
            min_n=spec.get('min_n', 30)
        )

        if result:
            result['analysis_name'] = name
            result['hypothesis'] = spec.get('hypothesis', '')
            result['tier'] = spec.get('tier')
            all_results.append(result)

    if not all_results:
        print("No analyses completed successfully.")
        return pd.DataFrame()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Apply FDR correction (by tier family)
    results_df = apply_fdr_by_tier(results_df)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "summary_results.csv", index=False, encoding='utf-8-sig')

    # Save hierarchical comparison (Bug 3 fix)
    hier_cols = ['analysis_name', 'tier', 'outcome', 'n',
                 'model0_r2', 'model1_r2', 'model2_r2', 'model3_r2',
                 'delta_r2_dass', 'delta_r2_ucla', 'delta_r2_interaction',
                 'p_dass', 'p_ucla', 'p_interaction']
    hier_df = results_df[[c for c in hier_cols if c in results_df.columns]]
    hier_df.to_csv(OUTPUT_DIR / "hierarchical_comparison.csv", index=False, encoding='utf-8-sig')

    # Save significant effects (FDR-corrected) (Bug 3 fix)
    sig_mask = (
        (results_df['p_ucla_fdr'] < 0.05) |
        (results_df['p_interaction_fdr'] < 0.05)
    )
    sig_df = results_df[sig_mask]
    sig_df.to_csv(OUTPUT_DIR / "significant_effects.csv", index=False, encoding='utf-8-sig')

    # Create visualization
    create_visualization(results_df, OUTPUT_DIR / "model_comparison_plot.png")

    # Generate report
    generate_report(results_df, OUTPUT_DIR / "FINAL_REPORT.txt")

    if verbose:
        print(f"\n[DONE] Results saved to: {OUTPUT_DIR}")
        print("  - summary_results.csv")
        print("  - hierarchical_comparison.csv")
        print("  - significant_effects.csv")
        print("  - model_comparison_plot.png")
        print("  - FINAL_REPORT.txt")
        print(f"\n  NOTE: ΔR² p-values from anova_lm() use OLS SEs (not HC3).")

    return results_df


if __name__ == "__main__":
    run()
