"""
DASS-21 Mediation Analysis

Tests whether DASS-21 (depression, anxiety, stress) mediates the relationship
between UCLA loneliness and executive function outcomes.

Approach:
1. Baron & Kenny 4-step method
2. Bootstrap confidence intervals for indirect effects
3. Multiple outcomes: HMM lapse recovery, RT variability, error rates

Author: Claude Code
Date: 2025-12-03
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.utils.data_loader_utils import load_master_dataset

# Output directory
OUTPUT_DIR = Path("results/analysis_outputs/mediation_dass")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("DASS-21 MEDIATION ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading data...")

df_master = load_master_dataset()
print(f"  Master dataset: {len(df_master)} participants")

# Load HMM results
hmm_df = pd.read_csv("results/analysis_outputs/hmm_attentional_states/hmm_results_raw.csv")
print(f"  HMM results: {len(hmm_df)} participants")

# Load other outcome measures from cognitive summary
cognitive_df = pd.read_csv("results/1_participants_info.csv")
print(f"  Cognitive data: {len(cognitive_df)} participants")

# ============================================================================
# STEP 2: Prepare Master Dataset
# ============================================================================
print("\nSTEP 2: Preparing merged dataset...")

# Merge all data sources
df = df_master.copy()

# Merge HMM metrics
if len(hmm_df) > 0:
    hmm_cols = ['participant_id', 'lapsed_occupancy', 'p_lapsed_to_focused',
                'p_focused_to_lapsed', 'rt_difference']
    df = df.merge(hmm_df[hmm_cols], on='participant_id', how='left')

print(f"  Final dataset: {len(df)} participants")

# Create composite DASS total score
df['dass_total'] = df['dass_depression'] + df['dass_anxiety'] + df['dass_stress']

# Ensure gender coding
if 'gender_male' not in df.columns:
    df['gender_male'] = (df['gender'].str.lower() == 'male').astype(int)

# Standardize for effect size interpretation
for col in ['ucla_total', 'dass_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    if col in df.columns:
        df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()

# ============================================================================
# STEP 3: Define Mediation Function
# ============================================================================
print("\nSTEP 3: Defining mediation analysis functions...")

def baron_kenny_mediation(df, x_var, m_var, y_var, covariates=None):
    """
    Perform Baron & Kenny 4-step mediation analysis.

    Steps:
    1. Total effect: Y ~ X (c path)
    2. M ~ X (a path)
    3. Y ~ X + M (c' and b paths)
    4. Test indirect effect: a * b

    Returns dict with all path coefficients and p-values
    """
    results = {
        'x_var': x_var,
        'm_var': m_var,
        'y_var': y_var,
        'n': len(df.dropna(subset=[x_var, m_var, y_var]))
    }

    # Prepare covariate string
    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(covariates)

    # Drop missing values
    df_clean = df.dropna(subset=[x_var, m_var, y_var])
    if covariates:
        # Only drop NA on actual column names, not formula terms like C(gender_male)
        actual_cols = [c for c in covariates if not c.startswith('C(')]
        if actual_cols:
            df_clean = df_clean.dropna(subset=actual_cols)

    if len(df_clean) < 20:
        results['error'] = 'Insufficient N'
        return results

    try:
        # Step 1: Total effect (c path)
        formula_total = f"{y_var} ~ {x_var}{cov_str}"
        model_total = smf.ols(formula_total, data=df_clean).fit()
        results['c_coef'] = model_total.params[x_var]
        results['c_se'] = model_total.bse[x_var]
        results['c_p'] = model_total.pvalues[x_var]
        results['c_ci_lower'] = model_total.conf_int().loc[x_var, 0]
        results['c_ci_upper'] = model_total.conf_int().loc[x_var, 1]

        # Step 2: a path (X → M)
        formula_a = f"{m_var} ~ {x_var}{cov_str}"
        model_a = smf.ols(formula_a, data=df_clean).fit()
        results['a_coef'] = model_a.params[x_var]
        results['a_se'] = model_a.bse[x_var]
        results['a_p'] = model_a.pvalues[x_var]
        results['a_ci_lower'] = model_a.conf_int().loc[x_var, 0]
        results['a_ci_upper'] = model_a.conf_int().loc[x_var, 1]

        # Step 3: c' and b paths (Y ~ X + M)
        formula_mediated = f"{y_var} ~ {x_var} + {m_var}{cov_str}"
        model_mediated = smf.ols(formula_mediated, data=df_clean).fit()

        # c' path (direct effect)
        results['c_prime_coef'] = model_mediated.params[x_var]
        results['c_prime_se'] = model_mediated.bse[x_var]
        results['c_prime_p'] = model_mediated.pvalues[x_var]
        results['c_prime_ci_lower'] = model_mediated.conf_int().loc[x_var, 0]
        results['c_prime_ci_upper'] = model_mediated.conf_int().loc[x_var, 1]

        # b path (M → Y, controlling for X)
        results['b_coef'] = model_mediated.params[m_var]
        results['b_se'] = model_mediated.bse[m_var]
        results['b_p'] = model_mediated.pvalues[m_var]
        results['b_ci_lower'] = model_mediated.conf_int().loc[m_var, 0]
        results['b_ci_upper'] = model_mediated.conf_int().loc[m_var, 1]

        # Indirect effect (a * b) - Sobel test
        indirect_effect = results['a_coef'] * results['b_coef']
        sobel_se = np.sqrt(results['b_coef']**2 * results['a_se']**2 +
                          results['a_coef']**2 * results['b_se']**2)
        sobel_z = indirect_effect / sobel_se
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        results['indirect_effect'] = indirect_effect
        results['indirect_se'] = sobel_se
        results['indirect_z'] = sobel_z
        results['indirect_p'] = sobel_p

        # Proportion mediated
        if results['c_coef'] != 0:
            results['proportion_mediated'] = indirect_effect / results['c_coef']
        else:
            results['proportion_mediated'] = np.nan

        results['error'] = None

    except Exception as e:
        results['error'] = str(e)

    return results

def bootstrap_indirect_effect(df, x_var, m_var, y_var, covariates=None, n_bootstrap=1000, random_state=42):
    """
    Bootstrap confidence intervals for indirect effect
    """
    np.random.seed(random_state)

    # Prepare data
    df_clean = df.dropna(subset=[x_var, m_var, y_var])
    if covariates:
        # Only drop NA on actual column names, not formula terms
        actual_cols = [c for c in covariates if not c.startswith('C(')]
        if actual_cols:
            df_clean = df_clean.dropna(subset=actual_cols)

    if len(df_clean) < 20:
        return None, None, None

    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(covariates)

    indirect_effects = []

    for i in range(n_bootstrap):
        # Resample with replacement
        df_boot = df_clean.sample(n=len(df_clean), replace=True, random_state=random_state+i)

        try:
            # a path
            formula_a = f"{m_var} ~ {x_var}{cov_str}"
            model_a = smf.ols(formula_a, data=df_boot).fit()
            a_coef = model_a.params[x_var]

            # b path
            formula_b = f"{y_var} ~ {x_var} + {m_var}{cov_str}"
            model_b = smf.ols(formula_b, data=df_boot).fit()
            b_coef = model_b.params[m_var]

            # Indirect effect
            indirect = a_coef * b_coef
            indirect_effects.append(indirect)
        except:
            continue

    if len(indirect_effects) < n_bootstrap * 0.5:
        return None, None, None

    # Compute percentile CI
    ci_lower = np.percentile(indirect_effects, 2.5)
    ci_upper = np.percentile(indirect_effects, 97.5)
    mean_indirect = np.mean(indirect_effects)

    return mean_indirect, ci_lower, ci_upper

# ============================================================================
# STEP 4: Run Mediation Analyses
# ============================================================================
print("\nSTEP 4: Running mediation analyses...")

# Define outcome variables to test
outcomes = [
    ('p_lapsed_to_focused', 'HMM: P(Lapse→Focus)'),
    ('lapsed_occupancy', 'HMM: Lapsed Occupancy'),
    ('p_focused_to_lapsed', 'HMM: P(Focus→Lapse)'),
    ('rt_difference', 'HMM: RT Difference (Lapsed-Focused)')
]

# Define mediators to test
mediators = [
    ('z_dass_total', 'DASS Total'),
    ('z_dass_depression', 'DASS Depression'),
    ('z_dass_anxiety', 'DASS Anxiety'),
    ('z_dass_stress', 'DASS Stress')
]

# Covariates
covariates = ['z_age', 'C(gender_male)']

all_results = []

for outcome_var, outcome_label in outcomes:
    if outcome_var not in df.columns:
        print(f"  Skipping {outcome_label} (not available)")
        continue

    print(f"\n  Analyzing: {outcome_label}")

    for mediator_var, mediator_label in mediators:
        print(f"    Mediator: {mediator_label}")

        # Baron & Kenny
        bk_result = baron_kenny_mediation(
            df,
            x_var='z_ucla_total',
            m_var=mediator_var,
            y_var=outcome_var,
            covariates=covariates
        )

        if bk_result.get('error'):
            print(f"      ERROR: {bk_result['error']}")
            continue

        # Bootstrap
        boot_mean, boot_ci_lower, boot_ci_upper = bootstrap_indirect_effect(
            df,
            x_var='z_ucla_total',
            m_var=mediator_var,
            y_var=outcome_var,
            covariates=covariates,
            n_bootstrap=1000
        )

        bk_result['outcome_label'] = outcome_label
        bk_result['mediator_label'] = mediator_label
        bk_result['boot_indirect'] = boot_mean
        bk_result['boot_ci_lower'] = boot_ci_lower
        bk_result['boot_ci_upper'] = boot_ci_upper

        all_results.append(bk_result)

        # Print key results
        print(f"      Total effect (c): β={bk_result['c_coef']:.4f}, p={bk_result['c_p']:.4f}")
        print(f"      Direct effect (c'): β={bk_result['c_prime_coef']:.4f}, p={bk_result['c_prime_p']:.4f}")
        print(f"      Indirect effect (a×b): {bk_result['indirect_effect']:.4f}, p={bk_result['indirect_p']:.4f}")
        if bk_result['proportion_mediated'] is not None and not np.isnan(bk_result['proportion_mediated']):
            print(f"      Proportion mediated: {bk_result['proportion_mediated']*100:.1f}%")
        if boot_ci_lower is not None:
            sig_boot = "***" if boot_ci_lower * boot_ci_upper > 0 else ""
            print(f"      Bootstrap 95% CI: [{boot_ci_lower:.4f}, {boot_ci_upper:.4f}] {sig_boot}")

# Save all results
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "mediation_results_all.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STEP 5: Visualization - Path Diagrams
# ============================================================================
print("\nSTEP 5: Creating visualizations...")

# Focus on key finding: P(Lapse→Focus) with DASS Total
key_results = results_df[
    (results_df['outcome_label'] == 'HMM: P(Lapse→Focus)') &
    (results_df['mediator_label'] == 'DASS Total')
]

if len(key_results) > 0:
    key_result = key_results.iloc[0]

    # Create path diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Boxes
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2)

    # X (UCLA)
    ax.text(2, 5, 'UCLA\nLoneliness', ha='center', va='center', fontsize=14, fontweight='bold', bbox=box_props)

    # M (DASS)
    ax.text(5, 8, 'DASS-21\nTotal', ha='center', va='center', fontsize=14, fontweight='bold', bbox=box_props)

    # Y (Outcome)
    ax.text(8, 5, 'P(Lapse→Focus)\nRecovery', ha='center', va='center', fontsize=14, fontweight='bold', bbox=box_props)

    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')

    # a path (X → M)
    ax.annotate('', xy=(4.2, 7.5), xytext=(2.8, 5.5), arrowprops=arrow_props)
    a_sig = "***" if key_result['a_p'] < 0.001 else "**" if key_result['a_p'] < 0.01 else "*" if key_result['a_p'] < 0.05 else ""
    ax.text(3, 6.8, f"a = {key_result['a_coef']:.3f}{a_sig}\np={key_result['a_p']:.4f}",
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # b path (M → Y)
    ax.annotate('', xy=(7.2, 6.5), xytext=(5.8, 7.5), arrowprops=arrow_props)
    b_sig = "***" if key_result['b_p'] < 0.001 else "**" if key_result['b_p'] < 0.01 else "*" if key_result['b_p'] < 0.05 else ""
    ax.text(6.5, 7.5, f"b = {key_result['b_coef']:.3f}{b_sig}\np={key_result['b_p']:.4f}",
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # c' path (direct effect, X → Y)
    ax.annotate('', xy=(7.2, 5), xytext=(2.8, 5), arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    c_prime_sig = "***" if key_result['c_prime_p'] < 0.001 else "**" if key_result['c_prime_p'] < 0.01 else "*" if key_result['c_prime_p'] < 0.05 else ""
    ax.text(5, 4.2, f"c' = {key_result['c_prime_coef']:.3f}{c_prime_sig} (direct)\np={key_result['c_prime_p']:.4f}",
            ha='center', fontsize=11, color='blue', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # Total effect (in text box)
    c_sig = "***" if key_result['c_p'] < 0.001 else "**" if key_result['c_p'] < 0.01 else "*" if key_result['c_p'] < 0.05 else ""
    ax.text(5, 2, f"Total effect (c) = {key_result['c_coef']:.3f}{c_sig}, p={key_result['c_p']:.4f}",
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

    # Indirect effect
    indirect_sig = ""
    if key_result['boot_ci_lower'] is not None and key_result['boot_ci_upper'] is not None:
        if key_result['boot_ci_lower'] * key_result['boot_ci_upper'] > 0:
            indirect_sig = " ***"
    ax.text(5, 1, f"Indirect effect (a×b) = {key_result['indirect_effect']:.4f}{indirect_sig}\n" +
                  f"Sobel p={key_result['indirect_p']:.4f}\n" +
                  f"Bootstrap 95% CI: [{key_result['boot_ci_lower']:.4f}, {key_result['boot_ci_upper']:.4f}]",
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))

    # Title
    ax.text(5, 9.5, 'DASS-21 Mediation: UCLA → P(Lapse→Focus)', ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "path_diagram_lapse_recovery.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: path_diagram_lapse_recovery.png")

# ============================================================================
# STEP 6: Summary Table
# ============================================================================
print("\nSTEP 6: Creating summary tables...")

# Filter for DASS Total mediator only (simplest interpretation)
dass_total_results = results_df[results_df['mediator_label'] == 'DASS Total'].copy()

# Create summary table
summary_data = []
for _, row in dass_total_results.iterrows():
    # Determine mediation type
    c_sig = row['c_p'] < 0.05
    c_prime_sig = row['c_prime_p'] < 0.05
    indirect_sig = (row['boot_ci_lower'] is not None and
                    row['boot_ci_upper'] is not None and
                    row['boot_ci_lower'] * row['boot_ci_upper'] > 0)

    if c_sig and not c_prime_sig and indirect_sig:
        mediation_type = "Full Mediation"
    elif c_sig and c_prime_sig and indirect_sig:
        mediation_type = "Partial Mediation"
    elif not c_sig:
        mediation_type = "No Total Effect"
    else:
        mediation_type = "No Mediation"

    summary_data.append({
        'Outcome': row['outcome_label'],
        'Total Effect (c)': f"{row['c_coef']:.4f} (p={row['c_p']:.4f})",
        'Direct Effect (c\')': f"{row['c_prime_coef']:.4f} (p={row['c_prime_p']:.4f})",
        'Indirect Effect (a×b)': f"{row['indirect_effect']:.4f} (p={row['indirect_p']:.4f})",
        'Bootstrap 95% CI': f"[{row['boot_ci_lower']:.4f}, {row['boot_ci_upper']:.4f}]" if row['boot_ci_lower'] is not None else "N/A",
        'Mediation Type': mediation_type,
        'Proportion Mediated': f"{row['proportion_mediated']*100:.1f}%" if row['proportion_mediated'] is not None and not np.isnan(row['proportion_mediated']) else "N/A"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(OUTPUT_DIR / "mediation_summary_table.csv", index=False, encoding='utf-8-sig')

print("\nSummary Table (DASS Total as Mediator):")
print(summary_df.to_string(index=False))

# ============================================================================
# STEP 7: Summary Report
# ============================================================================
print("\nSTEP 7: Generating summary report...")

summary_lines = [
    "=" * 80,
    "DASS-21 MEDIATION ANALYSIS - SUMMARY REPORT",
    "=" * 80,
    "",
    "RESEARCH QUESTION:",
    "Does DASS-21 (depression, anxiety, stress) mediate the relationship between",
    "UCLA loneliness and executive function outcomes (especially HMM lapse recovery)?",
    "",
    "=" * 80,
    "METHODOLOGY",
    "=" * 80,
    "1. Baron & Kenny 4-step approach:",
    "   - Step 1: Total effect (c path): Y ~ X",
    "   - Step 2: a path: M ~ X",
    "   - Step 3: Direct effect (c' path) + b path: Y ~ X + M",
    "   - Step 4: Indirect effect (a×b) with Sobel test",
    "",
    "2. Bootstrap confidence intervals (1000 iterations)",
    "3. Covariates: Age, Gender",
    "",
    f"Sample size: {len(df)} participants",
    "",
    "=" * 80,
    "KEY FINDINGS - DASS TOTAL AS MEDIATOR",
    "=" * 80,
]

for _, row in summary_df.iterrows():
    summary_lines.extend([
        f"\n{row['Outcome']}:",
        f"  Total effect: {row['Total Effect (c)']}",
        f"  Direct effect: {row['Direct Effect (c\')']}",
        f"  Indirect effect: {row['Indirect Effect (a×b)']}",
        f"  Bootstrap 95% CI: {row['Bootstrap 95% CI']}",
        f"  Mediation type: {row['Mediation Type']}",
        f"  Proportion mediated: {row['Proportion Mediated']}"
    ])

summary_lines.extend([
    "",
    "=" * 80,
    "INTERPRETATION",
    "=" * 80,
    "",
    "Mediation Types:",
    "  - FULL MEDIATION: Total effect significant, direct effect NS, indirect significant",
    "    → DASS-21 completely explains UCLA → Outcome relationship",
    "",
    "  - PARTIAL MEDIATION: Total, direct, and indirect all significant",
    "    → DASS-21 explains some but not all of UCLA effect",
    "",
    "  - NO MEDIATION: Total effect significant, but indirect NS",
    "    → UCLA effect is independent of DASS-21",
    "",
    "Bootstrap CI Interpretation:",
    "  - If CI excludes zero → Significant indirect effect",
    "  - If CI includes zero → No mediation",
    "",
    "=" * 80,
    "FILES GENERATED",
    "=" * 80,
    "CSV Files:",
    "  - mediation_results_all.csv (all mediator × outcome combinations)",
    "  - mediation_summary_table.csv (DASS Total only, formatted)",
    "",
    "Figures:",
    "  - path_diagram_lapse_recovery.png (key finding visualization)",
    "",
    "=" * 80,
    "END OF REPORT",
    "=" * 80
])

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nDASS-21 Mediation Analysis COMPLETE!")
