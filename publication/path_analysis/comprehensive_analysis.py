"""
Comprehensive Path Analysis
===========================

Additional analyses for path models:
1. Gender-stratified analysis (separate models for males/females)
2. Model fit comparison (AIC, BIC, R²)
3. Fisher's z-test for gender coefficient comparison
4. Summary tables for publication
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from publication.preprocessing import (
    load_master_dataset,
    standardize_predictors,
    prepare_gender_variable,
)
from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR
from publication.advanced_analysis._utils import create_ef_composite

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "path_analysis" / "comprehensive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DASS_COLS = {
    'depression': 'z_dass_dep',
    'anxiety': 'z_dass_anx',
    'stress': 'z_dass_str',
}

MODEL_SPECS = {
    'model1_ucla_dass_ef': {
        'name': 'UCLA → DASS → EF',
        'stage1': '{dass_col} ~ z_ucla + z_age',
        'stage2': 'ef_composite ~ z_ucla + {dass_col} + z_age',
        'a_path': ('z_ucla', 'stage1'),
        'b_path': ('{dass_col}', 'stage2'),
    },
    'model2_ucla_ef_dass': {
        'name': 'UCLA → EF → DASS',
        'stage1': 'ef_composite ~ z_ucla + z_age',
        'stage2': '{dass_col} ~ z_ucla + ef_composite + z_age',
        'a_path': ('z_ucla', 'stage1'),
        'b_path': ('ef_composite', 'stage2'),
    },
    'model3_dass_ucla_ef': {
        'name': 'DASS → UCLA → EF',
        'stage1': 'z_ucla ~ {dass_col} + z_age',
        'stage2': 'ef_composite ~ z_ucla + {dass_col} + z_age',
        'a_path': ('{dass_col}', 'stage1'),
        'b_path': ('z_ucla', 'stage2'),
    },
    'model4_ef_dass_ucla': {
        'name': 'EF → DASS → UCLA',
        'stage1': '{dass_col} ~ ef_composite + z_age',
        'stage2': 'z_ucla ~ ef_composite + {dass_col} + z_age',
        'a_path': ('ef_composite', 'stage1'),
        'b_path': ('{dass_col}', 'stage2'),
    },
}


def load_data() -> pd.DataFrame:
    """Load and prepare data."""
    df = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
    df = prepare_gender_variable(df)
    df = standardize_predictors(df)
    df = create_ef_composite(df)

    required = ['ef_composite', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str',
                'z_age', 'gender_male']
    df = df[required].dropna()
    return df


def fit_mediation_model(
    df: pd.DataFrame,
    stage1_formula: str,
    stage2_formula: str,
    a_term: str,
    b_term: str,
    n_bootstrap: int = 5000,
) -> Dict:
    """Fit mediation model and compute indirect effect with bootstrap CI."""

    stage1 = smf.ols(stage1_formula, data=df).fit()
    stage2 = smf.ols(stage2_formula, data=df).fit()

    a = stage1.params.get(a_term, np.nan)
    a_se = stage1.bse.get(a_term, np.nan)
    a_p = stage1.pvalues.get(a_term, np.nan)

    b = stage2.params.get(b_term, np.nan)
    b_se = stage2.bse.get(b_term, np.nan)
    b_p = stage2.pvalues.get(b_term, np.nan)

    indirect = a * b

    # Sobel test
    sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z))) if not np.isnan(sobel_z) else np.nan

    # Bootstrap
    n = len(df)
    rng = np.random.default_rng(42)
    boot_indirect = []

    for _ in range(n_bootstrap):
        sample = df.sample(n=n, replace=True, random_state=int(rng.integers(0, 1e9)))
        try:
            s1 = smf.ols(stage1_formula, data=sample).fit()
            s2 = smf.ols(stage2_formula, data=sample).fit()
            a_boot = s1.params.get(a_term, np.nan)
            b_boot = s2.params.get(b_term, np.nan)
            if np.isfinite(a_boot) and np.isfinite(b_boot):
                boot_indirect.append(a_boot * b_boot)
        except:
            continue

    boot_indirect = np.array(boot_indirect)
    ci_low = np.percentile(boot_indirect, 2.5) if len(boot_indirect) > 0 else np.nan
    ci_high = np.percentile(boot_indirect, 97.5) if len(boot_indirect) > 0 else np.nan
    significant = (ci_low > 0) or (ci_high < 0) if np.isfinite(ci_low) and np.isfinite(ci_high) else False

    return {
        'n': len(df),
        'a_path': a,
        'a_se': a_se,
        'a_p': a_p,
        'b_path': b,
        'b_se': b_se,
        'b_p': b_p,
        'indirect': indirect,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant': significant,
        'sobel_z': sobel_z,
        'sobel_p': sobel_p,
        'stage1_r2': stage1.rsquared,
        'stage2_r2': stage2.rsquared,
        'stage1_aic': stage1.aic,
        'stage2_aic': stage2.aic,
        'stage1_bic': stage1.bic,
        'stage2_bic': stage2.bic,
    }


def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> Tuple[float, float]:
    """Fisher's z-test for comparing two correlations/betas."""
    z1 = 0.5 * np.log((1 + r1) / (1 - r1)) if abs(r1) < 1 else np.nan
    z2 = 0.5 * np.log((1 + r2) / (1 - r2)) if abs(r2) < 1 else np.nan

    if np.isnan(z1) or np.isnan(z2):
        return np.nan, np.nan

    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z_diff = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z_diff)))
    return z_diff, p


def run_comprehensive_analysis(verbose: bool = True) -> Dict:
    """Run all comprehensive analyses."""

    df = load_data()
    df_male = df[df['gender_male'] == 1].copy()
    df_female = df[df['gender_male'] == 0].copy()

    if verbose:
        print(f"\n{'='*70}")
        print("COMPREHENSIVE PATH ANALYSIS")
        print(f"{'='*70}")
        print(f"Total N = {len(df)} (Male = {len(df_male)}, Female = {len(df_female)})")

    all_results = []

    for dass_name, dass_col in DASS_COLS.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"DASS: {dass_name.upper()}")
            print(f"{'='*70}")

        for model_key, model_spec in MODEL_SPECS.items():
            stage1_f = model_spec['stage1'].format(dass_col=dass_col)
            stage2_f = model_spec['stage2'].format(dass_col=dass_col)
            a_term = model_spec['a_path'][0].format(dass_col=dass_col)
            b_term = model_spec['b_path'][0].format(dass_col=dass_col)

            # Full sample
            full_res = fit_mediation_model(df, stage1_f, stage2_f, a_term, b_term)

            # Gender-stratified
            male_res = fit_mediation_model(df_male, stage1_f, stage2_f, a_term, b_term)
            female_res = fit_mediation_model(df_female, stage1_f, stage2_f, a_term, b_term)

            # Fisher's z-test for a-path and b-path
            a_z, a_p = fisher_z_test(male_res['a_path'], len(df_male),
                                      female_res['a_path'], len(df_female))
            b_z, b_p = fisher_z_test(male_res['b_path'], len(df_male),
                                      female_res['b_path'], len(df_female))

            result_row = {
                'dass': dass_name,
                'model': model_spec['name'],
                'model_key': model_key,
                # Full sample
                'full_n': full_res['n'],
                'full_a': full_res['a_path'],
                'full_b': full_res['b_path'],
                'full_indirect': full_res['indirect'],
                'full_ci_low': full_res['ci_low'],
                'full_ci_high': full_res['ci_high'],
                'full_sig': full_res['significant'],
                'full_sobel_z': full_res['sobel_z'],
                'full_sobel_p': full_res['sobel_p'],
                'full_aic': full_res['stage1_aic'] + full_res['stage2_aic'],
                'full_bic': full_res['stage1_bic'] + full_res['stage2_bic'],
                'full_r2_stage1': full_res['stage1_r2'],
                'full_r2_stage2': full_res['stage2_r2'],
                # Male
                'male_n': male_res['n'],
                'male_a': male_res['a_path'],
                'male_a_p': male_res['a_p'],
                'male_b': male_res['b_path'],
                'male_b_p': male_res['b_p'],
                'male_indirect': male_res['indirect'],
                'male_ci_low': male_res['ci_low'],
                'male_ci_high': male_res['ci_high'],
                'male_sig': male_res['significant'],
                # Female
                'female_n': female_res['n'],
                'female_a': female_res['a_path'],
                'female_a_p': female_res['a_p'],
                'female_b': female_res['b_path'],
                'female_b_p': female_res['b_p'],
                'female_indirect': female_res['indirect'],
                'female_ci_low': female_res['ci_low'],
                'female_ci_high': female_res['ci_high'],
                'female_sig': female_res['significant'],
                # Gender comparison
                'a_path_z_diff': a_z,
                'a_path_p_diff': a_p,
                'b_path_z_diff': b_z,
                'b_path_p_diff': b_p,
            }
            all_results.append(result_row)

            if verbose:
                sig_mark = "✅" if full_res['significant'] else "❌"
                print(f"\n[{model_spec['name']}]")
                print(f"  Full: indirect={full_res['indirect']:.3f}, "
                      f"CI=[{full_res['ci_low']:.3f}, {full_res['ci_high']:.3f}] {sig_mark}")
                print(f"  Male: indirect={male_res['indirect']:.3f}, "
                      f"CI=[{male_res['ci_low']:.3f}, {male_res['ci_high']:.3f}] "
                      f"{'✅' if male_res['significant'] else '❌'}")
                print(f"  Female: indirect={female_res['indirect']:.3f}, "
                      f"CI=[{female_res['ci_low']:.3f}, {female_res['ci_high']:.3f}] "
                      f"{'✅' if female_res['significant'] else '❌'}")
                if a_p < 0.05 or b_p < 0.05:
                    print(f"  ⚠️ Gender difference: a-path p={a_p:.3f}, b-path p={b_p:.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "comprehensive_results.csv", index=False, encoding='utf-8-sig')

    # Summary tables
    _create_summary_tables(results_df, verbose)

    return {'results': results_df}


def _create_summary_tables(df: pd.DataFrame, verbose: bool = True):
    """Create publication-ready summary tables."""

    # Table 1: Indirect effects by DASS subscale and model
    summary_indirect = df.pivot_table(
        index='model',
        columns='dass',
        values=['full_indirect', 'full_sig'],
        aggfunc='first'
    )
    summary_indirect.to_csv(OUTPUT_DIR / "summary_indirect_effects.csv", encoding='utf-8-sig')

    # Table 2: Gender-stratified results
    gender_summary = []
    for _, row in df.iterrows():
        gender_summary.append({
            'DASS': row['dass'],
            'Model': row['model'],
            'Male β': f"{row['male_indirect']:.3f}",
            'Male CI': f"[{row['male_ci_low']:.3f}, {row['male_ci_high']:.3f}]",
            'Male Sig': '✅' if row['male_sig'] else '❌',
            'Female β': f"{row['female_indirect']:.3f}",
            'Female CI': f"[{row['female_ci_low']:.3f}, {row['female_ci_high']:.3f}]",
            'Female Sig': '✅' if row['female_sig'] else '❌',
            'Diff p': f"{row['a_path_p_diff']:.3f}" if not np.isnan(row['a_path_p_diff']) else 'N/A',
        })
    gender_df = pd.DataFrame(gender_summary)
    gender_df.to_csv(OUTPUT_DIR / "gender_stratified_summary.csv", index=False, encoding='utf-8-sig')

    # Table 3: Model fit comparison
    fit_summary = df[['dass', 'model', 'full_aic', 'full_bic',
                      'full_r2_stage1', 'full_r2_stage2']].copy()
    fit_summary['total_r2'] = (fit_summary['full_r2_stage1'] + fit_summary['full_r2_stage2']) / 2
    fit_summary.to_csv(OUTPUT_DIR / "model_fit_comparison.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY: SIGNIFICANT INDIRECT EFFECTS")
        print(f"{'='*70}")
        sig_results = df[df['full_sig'] | df['male_sig'] | df['female_sig']]
        if len(sig_results) == 0:
            print("No significant indirect effects found.")
        else:
            for _, row in sig_results.iterrows():
                print(f"\n{row['dass'].upper()} - {row['model']}:")
                if row['full_sig']:
                    print(f"  Full: β={row['full_indirect']:.3f} [{row['full_ci_low']:.3f}, {row['full_ci_high']:.3f}] ✅")
                if row['male_sig']:
                    print(f"  Male: β={row['male_indirect']:.3f} [{row['male_ci_low']:.3f}, {row['male_ci_high']:.3f}] ✅")
                if row['female_sig']:
                    print(f"  Female: β={row['female_indirect']:.3f} [{row['female_ci_low']:.3f}, {row['female_ci_high']:.3f}] ✅")

        print(f"\n{'='*70}")
        print("MODEL FIT COMPARISON (by AIC - lower is better)")
        print(f"{'='*70}")
        for dass_name in DASS_COLS.keys():
            dass_df = df[df['dass'] == dass_name].sort_values('full_aic')
            print(f"\n{dass_name.upper()}:")
            for _, row in dass_df.iterrows():
                print(f"  {row['model']}: AIC={row['full_aic']:.1f}, BIC={row['full_bic']:.1f}")


if __name__ == "__main__":
    run_comprehensive_analysis(verbose=True)
