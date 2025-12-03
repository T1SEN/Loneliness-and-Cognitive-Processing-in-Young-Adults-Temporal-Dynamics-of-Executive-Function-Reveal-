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
from typing import Dict, List, Optional, Any
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# Project imports
from analysis.utils.data_loader_utils import load_master_dataset

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


def prepare_data(force_rebuild: bool = False) -> pd.DataFrame:
    """Load and prepare master dataset with standardized predictors."""
    print("[DATA] Loading master dataset...")

    master = load_master_dataset(use_cache=not force_rebuild, force_rebuild=force_rebuild, merge_cognitive_summary=True)

    # Normalize gender
    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

    master['gender_male'] = (master['gender'] == 'male').astype(int)

    # Ensure ucla_total exists
    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Required columns
    required = ['participant_id', 'ucla_total', 'gender_male', 'age',
                'dass_depression', 'dass_anxiety', 'dass_stress']

    missing = [col for col in required if col not in master.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean and standardize
    df = master.dropna(subset=required).copy()

    scaler = StandardScaler()
    df['z_age'] = scaler.fit_transform(df[['age']])
    df['z_ucla'] = scaler.fit_transform(df[['ucla_total']])
    df['z_dass_dep'] = scaler.fit_transform(df[['dass_depression']])
    df['z_dass_anx'] = scaler.fit_transform(df[['dass_anxiety']])
    df['z_dass_str'] = scaler.fit_transform(df[['dass_stress']])

    print(f"  Total N = {len(df)}")
    print(f"    Males: {(df['gender_male'] == 1).sum()}")
    print(f"    Females: {(df['gender_male'] == 0).sum()}")

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
        'p_ucla': anova_2v1['Pr(>F)'][1],

        'delta_r2_interaction': models['model3'].rsquared - models['model2'].rsquared,
        'p_interaction': anova_3v2['Pr(>F)'][1],

        # UCLA main effect (from Model 2)
        'ucla_beta': models['model2'].params.get('z_ucla', np.nan),
        'ucla_se': models['model2'].bse.get('z_ucla', np.nan),
        'ucla_p': models['model2'].pvalues.get('z_ucla', np.nan),
    }

    # Interaction term (from Model 3)
    int_term = 'z_ucla:C(gender_male)[T.1]'
    if int_term in models['model3'].params:
        results['interaction_beta'] = models['model3'].params[int_term]
        results['interaction_se'] = models['model3'].bse[int_term]
        results['interaction_p'] = models['model3'].pvalues[int_term]
    else:
        results['interaction_beta'] = np.nan
        results['interaction_se'] = np.nan
        results['interaction_p'] = np.nan

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

    return results


def apply_fdr_correction(results_df: pd.DataFrame) -> pd.DataFrame:
    """Apply FDR correction to p-values."""
    df = results_df.copy()

    for col in ['p_ucla', 'p_interaction']:
        pvals = df[col].to_numpy()
        mask = np.isfinite(pvals)
        adj = np.full_like(pvals, np.nan, dtype=float)
        if mask.sum() > 0:
            adj[mask] = multipletests(pvals[mask], method='fdr_bh')[1]
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
    """Generate human-readable final report."""
    survived = results_df[results_df['ucla_p'] < 0.05]
    sig_interaction = results_df[results_df['interaction_p'] < 0.05]

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
        f.write("  - FDR correction for multiple comparisons\n\n")

        f.write("RESULTS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total hypotheses tested: {len(results_df)}\n")
        f.write(f"UCLA main effects surviving p < 0.05: {len(survived)}\n")
        f.write(f"UCLA × Gender interactions p < 0.05: {len(sig_interaction)}\n\n")

        if len(survived) > 0:
            f.write("SIGNIFICANT UCLA MAIN EFFECTS:\n")
            for _, row in survived.iterrows():
                f.write(f"  - {row['outcome']}: β={row['ucla_beta']:.3f}, p={row['ucla_p']:.4f}\n")
            f.write("\n")

        if len(sig_interaction) > 0:
            f.write("SIGNIFICANT UCLA × GENDER INTERACTIONS:\n")
            for _, row in sig_interaction.iterrows():
                f.write(f"  - {row['outcome']}: β={row['interaction_beta']:.3f}, p={row['interaction_p']:.4f}\n")
                f.write(f"      Female: β={row['female_ucla_beta']:.3f}, p={row['female_ucla_p']:.4f}\n")
                f.write(f"      Male: β={row['male_ucla_beta']:.3f}, p={row['male_ucla_p']:.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n")

        if len(survived) == 0 and len(sig_interaction) == 0:
            f.write("No UCLA effects survived DASS control.\n")
            f.write("Loneliness effects appear confounded with mood/anxiety.\n")
        elif len(sig_interaction) > 0:
            f.write("UCLA × Gender interactions detected.\n")
            f.write("Loneliness effects are gender-specific.\n")
        else:
            f.write("UCLA main effects survived DASS control.\n")
            f.write("Pure loneliness effects beyond mood are supported.\n")


def run(
    analyses: Optional[List[str]] = None,
    force_rebuild: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run Gold Standard analyses.

    Args:
        analyses: List of analysis names to run (default: all)
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
    if analyses:
        config = {k: v for k, v in config.items() if k in analyses}

    if verbose:
        print(f"\nAnalyses to run: {list(config.keys())}")

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
            all_results.append(result)

    if not all_results:
        print("No analyses completed successfully.")
        return pd.DataFrame()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Apply FDR correction
    results_df = apply_fdr_correction(results_df)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "summary_results.csv", index=False, encoding='utf-8-sig')

    # Save hierarchical comparison (Bug 3 fix)
    hier_cols = ['analysis_name', 'outcome', 'n',
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
