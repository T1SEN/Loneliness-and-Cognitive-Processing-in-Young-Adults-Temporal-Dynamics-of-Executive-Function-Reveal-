"""
Multivariate Executive Function Analysis
==========================================
Tests UCLA × Gender effects on the COMPLETE EF profile (WCST + PRP + Stroop) simultaneously.

Advantages over univariate tests:
1. Single omnibus test → no multiple comparison penalty
2. Captures covariance structure across EF tasks
3. Shows whether loneliness affects "meta-control" as a unified construct
4. Canonical weights reveal which EF dimensions drive the effect

Method: MANOVA with [WCST_PE, PRP_tau/bottleneck, Stroop_interference] as multivariate outcome

CRITICAL: All models control for DASS-21 subscales.

Author: Research Team
Date: 2025-01-16
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.multivariate.manova import MANOVA
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from analysis.utils.data_loader_utils import load_master_dataset
warnings.filterwarnings('ignore')

np.random.seed(42)

# Constants
MIN_N_REGRESSION = 30  # Minimum sample size for regression models

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/multivariate_ef_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MULTIVARIATE EXECUTIVE FUNCTION ANALYSIS")
print("=" * 80)
print("\nPurpose: Test UCLA × Gender on EF profile (WCST + PRP + Stroop) simultaneously")
print("Method: MANOVA with DASS-21 control\n")

# Load master dataset
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
master['gender_male'] = (master['gender'] == 'male').astype(int)
print(f"Loaded master dataset: {len(master)} participants")

print(f"\nData overview:")
print(f"  Total N: {len(master)}")
if 'gender' in master.columns:
    print(f"  Males: {sum(master['gender']=='male')}, Females: {sum(master['gender']=='female')}")

# Identify available EF outcomes
ef_candidates = {
    'wcst': ['pe_rate', 'pe_rate', 'pe_rate'],
    'prp': ['prp_tau_long', 'prp_bottleneck', 'prp_bottleneck_effect'],
    'stroop': ['stroop_interference', 'stroop_effect', 'interference_rt']
}

ef_outcomes = {}

for task, possible_names in ef_candidates.items():
    for name in possible_names:
        if name in master.columns:
            ef_outcomes[task] = name
            break

print(f"\nIdentified EF outcomes:")
for task, col in ef_outcomes.items():
    print(f"  {task.upper()}: {col}")

if len(ef_outcomes) < 2:
    print("\nERROR: Need at least 2 EF outcomes for multivariate analysis.")
    print(f"Available columns: {[col for col in master.columns if 'wcst' in col or 'prp' in col or 'stroop' in col]}")
    sys.exit(1)

# Collect outcome column names
outcome_cols = list(ef_outcomes.values())
print(f"\nMultivariate outcome vector: {outcome_cols}")

# Required predictors
required_base = ['participant_id', 'ucla_total', 'gender_male', 'age',
                 'dass_depression', 'dass_anxiety', 'dass_stress']

missing_cols = [col for col in required_base if col not in master.columns]
if missing_cols:
    print(f"\nERROR: Missing required columns: {missing_cols}")
    sys.exit(1)

# Clean data - require all EF outcomes + covariates
required_all = required_base + outcome_cols
master_clean = master.dropna(subset=required_all).copy()

print(f"\nAfter requiring all EF outcomes + covariates: N = {len(master_clean)}")

if len(master_clean) < MIN_N_REGRESSION:
    print(f"ERROR: Insufficient data (N={len(master_clean)} < {MIN_N_REGRESSION}).")
    print("Regression models require at least 30 participants for stable estimates.")
    sys.exit(1)

# Standardize all variables
print("\nStandardizing variables...")
scaler = StandardScaler()

master_clean['z_age'] = scaler.fit_transform(master_clean[['age']])
master_clean['z_ucla'] = scaler.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_str'] = scaler.fit_transform(master_clean[['dass_stress']])

# Standardize EF outcomes for comparability
for col in outcome_cols:
    master_clean[f'z_{col}'] = scaler.fit_transform(master_clean[[col]])

# Create standardized outcome names
z_outcome_cols = [f'z_{col}' for col in outcome_cols]

print("=" * 80)
print("STEP 1: Multivariate Regression (Individual Outcomes)")
print("=" * 80)
print("\nBefore MANOVA, run univariate models for reference...\n")

univariate_results = []

for outcome in outcome_cols:
    z_outcome = f'z_{outcome}'
    df = master_clean[[z_outcome, 'z_ucla', 'gender_male', 'z_dass_dep',
                       'z_dass_anx', 'z_dass_str', 'z_age']].dropna()

    if len(df) < 30:
        continue

    formula = f"{z_outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model = smf.ols(formula, data=df).fit()

    int_term = 'z_ucla:C(gender_male)[T.1]'
    ucla_main = model.params.get('z_ucla', np.nan)
    ucla_main_p = model.pvalues.get('z_ucla', np.nan)

    if int_term in model.params:
        interaction = model.params[int_term]
        interaction_p = model.pvalues[int_term]
    else:
        interaction = np.nan
        interaction_p = np.nan

    print(f"{outcome}:")
    print(f"  UCLA main: β={ucla_main:.4f}, p={ucla_main_p:.4f}")
    print(f"  UCLA × Gender: β={interaction:.4f}, p={interaction_p:.4f}")
    print(f"  R² = {model.rsquared:.3f}\n")

    univariate_results.append({
        'outcome': outcome,
        'ucla_beta': ucla_main,
        'ucla_p': ucla_main_p,
        'interaction_beta': interaction,
        'interaction_p': interaction_p,
        'r_squared': model.rsquared
    })

univariate_df = pd.DataFrame(univariate_results)
univariate_df.to_csv(OUTPUT_DIR / "univariate_ef_models.csv",
                     index=False, encoding='utf-8-sig')
print(f"✓ Saved: univariate_ef_models.csv")

print("\n" + "=" * 80)
print("STEP 2: MANOVA - Multivariate Test")
print("=" * 80)

# Prepare data for MANOVA
manova_df = master_clean[['participant_id', 'z_ucla', 'gender_male', 'z_age',
                          'z_dass_dep', 'z_dass_anx', 'z_dass_str'] + z_outcome_cols].copy()

# MANOVA requires numeric gender for formula
manova_df['gender_male_num'] = manova_df['gender_male'].astype(float)

# Build formula
# Format: "z_outcome1 + z_outcome2 + z_outcome3 ~ predictors"
manova_formula = " + ".join(z_outcome_cols) + " ~ z_ucla * gender_male_num + z_dass_dep + z_dass_anx + z_dass_str + z_age"

print(f"\nMANOVA formula:")
print(f"  {manova_formula}\n")

# Initialize variables before try block to avoid NameError on failure
manova_summary_str = "MANOVA computation failed (see error message above)"

try:
    manova_model = MANOVA.from_formula(manova_formula, data=manova_df)
    manova_results = manova_model.mv_test()

    print("=" * 80)
    print("MANOVA RESULTS")
    print("=" * 80)
    print(manova_results)

    # Extract test statistics
    # MANOVA.mv_test() returns a MVTestResults object
    # We'll parse the summary table

    manova_summary_str = str(manova_results)

    # Save full MANOVA output
    with open(OUTPUT_DIR / "manova_full_output.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Formula: {manova_formula}\n")
        f.write(f"N = {len(manova_df)}\n")
        f.write(f"Outcomes: {', '.join(outcome_cols)}\n\n")
        f.write(manova_summary_str)

    print(f"\n✓ Saved: manova_full_output.txt")

    # Extract specific test statistics (Pillai's trace, Wilks' lambda, etc.)
    # Note: statsmodels MANOVA output structure can be complex
    # We'll extract manually from results

    print("\n" + "=" * 80)
    print("KEY EFFECTS")
    print("=" * 80)

    # The mv_test() method returns results for each term
    # We care about:
    # 1. UCLA main effect
    # 2. Gender main effect
    # 3. UCLA × Gender interaction

    # Parse results object safely
    results_dict = manova_model.mv_test().results

    manova_summary_rows = []

    for term_name, term_results in results_dict.items():
        # Initialize all variables with defaults FIRST
        wilks_stat = np.nan
        wilks_F = np.nan
        wilks_p = np.nan
        pillai_stat = np.nan
        pillai_F = np.nan
        pillai_p = np.nan

        try:
            # Statsmodels MANOVA returns nested dict structure
            if isinstance(term_results, dict):
                # Extract Wilks' lambda if available
                if "Wilks' lambda" in term_results:
                    wilks_dict = term_results["Wilks' lambda"]
                    if isinstance(wilks_dict, dict):
                        wilks_stat = wilks_dict.get('stat', np.nan)
                        wilks_F = wilks_dict.get('F Value', np.nan)
                        wilks_p = wilks_dict.get('Pr > F', np.nan)

                # Extract Pillai's trace if available
                if "Pillai's trace" in term_results:
                    pillai_dict = term_results["Pillai's trace"]
                    if isinstance(pillai_dict, dict):
                        pillai_stat = pillai_dict.get('stat', np.nan)
                        pillai_F = pillai_dict.get('F Value', np.nan)
                        pillai_p = pillai_dict.get('Pr > F', np.nan)

        except (KeyError, IndexError, TypeError, AttributeError) as e:
            print(f"    Warning: Could not parse results for '{term_name}': {e}")
            # All values remain np.nan (already initialized)

        manova_summary_rows.append({
            'term': term_name,
            'wilks_stat': wilks_stat,
            'wilks_F': wilks_F,
            'wilks_p': wilks_p,
            'pillai_stat': pillai_stat,
            'pillai_F': pillai_F,
            'pillai_p': pillai_p
        })

    manova_summary_df = pd.DataFrame(manova_summary_rows)
    manova_summary_df.to_csv(OUTPUT_DIR / "manova_test_statistics.csv",
                             index=False, encoding='utf-8-sig')
    print(f"✓ Saved: manova_test_statistics.csv")

except Exception as e:
    print(f"\nWARNING: MANOVA failed with error: {e}")
    print("This can happen with small N or perfect multicollinearity.")
    print("Proceeding with alternative multivariate approach...\n")

    manova_results = None

print("\n" + "=" * 80)
print("STEP 3: Canonical Correlation Analysis (Alternative Multivariate Approach)")
print("=" * 80)

# Canonical correlation: Which linear combination of EF outcomes
# correlates most strongly with UCLA × Gender?

from sklearn.cross_decomposition import CCA

# Prepare predictors (UCLA, Gender, UCLA×Gender)
X = manova_df[['z_ucla', 'gender_male_num']].copy()
X['ucla_x_gender'] = X['z_ucla'] * X['gender_male_num']

# Outcomes
Y = manova_df[z_outcome_cols].copy()

# Remove missing
XY = pd.concat([X, Y], axis=1).dropna()
X_clean = XY[['z_ucla', 'gender_male_num', 'ucla_x_gender']].values
Y_clean = XY[z_outcome_cols].values

if len(XY) >= 30:
    print(f"\nRunning CCA with N={len(XY)}")

    # Fit CCA (n_components = min of predictors/outcomes)
    n_components = min(X_clean.shape[1], Y_clean.shape[1])
    cca = CCA(n_components=n_components)
    cca.fit(X_clean, Y_clean)

    # Transform data
    X_c, Y_c = cca.transform(X_clean, Y_clean)

    # Canonical correlations
    canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]

    print(f"\nCanonical correlations:")
    for i, r in enumerate(canonical_corrs):
        print(f"  Component {i+1}: r = {r:.4f} (r² = {r**2:.4f})")

    # Y loadings (EF outcome weights on first canonical variate)
    y_loadings = cca.y_weights_[:, 0]  # First canonical variate

    print(f"\nEF outcome loadings on first canonical variate:")
    for outcome, loading in zip(outcome_cols, y_loadings):
        print(f"  {outcome}: {loading:.4f}")

    # Save canonical results
    canonical_df = pd.DataFrame({
        'outcome': outcome_cols,
        'canonical_loading': y_loadings
    })
    canonical_df.to_csv(OUTPUT_DIR / "canonical_weights.csv",
                        index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved: canonical_weights.csv")

else:
    print(f"Insufficient data for CCA (N={len(XY)})")

print("\n" + "=" * 80)
print("STEP 4: Visualizations")
print("=" * 80)

# 4A: Heatmap of EF profiles by Gender × UCLA (median split)
print("\nCreating EF profile heatmap...")

# Median split UCLA
ucla_median = master_clean['ucla_total'].median()
master_clean['ucla_group'] = master_clean['ucla_total'].apply(
    lambda x: 'High Loneliness' if x >= ucla_median else 'Low Loneliness'
)

master_clean['gender_label'] = master_clean['gender_male'].map({1: 'Male', 0: 'Female'})

# Compute group means for z-scored outcomes
group_means = master_clean.groupby(['gender_label', 'ucla_group'])[z_outcome_cols].mean()

# Reshape for heatmap
heatmap_data = group_means.T
heatmap_data.columns = [f"{gender}_{ucla}" for gender, ucla in heatmap_data.columns]
heatmap_data.index = outcome_cols

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, cbar_kws={'label': 'Z-scored EF Impairment'},
            linewidths=0.5, ax=ax)
ax.set_title('Executive Function Profile by Gender × Loneliness Group', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender × Loneliness Group', fontsize=12)
ax.set_ylabel('EF Outcome', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ef_profile_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: ef_profile_heatmap.png")

# 4B: Correlation matrix of EF outcomes
print("\nCreating EF correlation matrix...")

ef_corr = master_clean[outcome_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(ef_corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix: EF Outcomes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ef_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: ef_correlation_matrix.png")

# Save correlation matrix as CSV
ef_corr.to_csv(OUTPUT_DIR / "ef_correlation_matrix.csv", encoding='utf-8-sig')

print("\n" + "=" * 80)
print("STEP 5: Effect Size Summary")
print("=" * 80)

# Compute multivariate effect size (if MANOVA succeeded)
# Otherwise, summarize univariate effect sizes

effect_summary = univariate_df.copy()

# Add significance flags
effect_summary['ucla_sig'] = (effect_summary['ucla_p'] < 0.05).map({True: '*', False: ''})
effect_summary['interaction_sig'] = (effect_summary['interaction_p'] < 0.05).map({True: '*', False: ''})

# Reorder columns
effect_summary = effect_summary[['outcome', 'ucla_beta', 'ucla_p', 'ucla_sig',
                                 'interaction_beta', 'interaction_p', 'interaction_sig', 'r_squared']]

effect_summary.to_csv(OUTPUT_DIR / "multivariate_effect_sizes.csv",
                      index=False, encoding='utf-8-sig')
print(f"✓ Saved: multivariate_effect_sizes.csv")

# Print summary table
print("\nEffect Size Summary:")
print(effect_summary.to_string(index=False))

print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

# Create comprehensive report
report_path = OUTPUT_DIR / "MULTIVARIATE_EF_REPORT.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("MULTIVARIATE EXECUTIVE FUNCTION ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 80 + "\n")
    f.write("Test whether UCLA loneliness × Gender affects the ENTIRE EF profile\n")
    f.write("(WCST + PRP + Stroop) as a unified construct, rather than individual tasks.\n\n")

    f.write("ADVANTAGES OVER UNIVARIATE TESTS\n")
    f.write("-" * 80 + "\n")
    f.write("1. Single omnibus test → no multiple comparison penalty\n")
    f.write("2. Captures covariance structure across tasks\n")
    f.write("3. Tests 'meta-control' as a unified latent factor\n")
    f.write("4. Higher statistical power for multivariate effects\n\n")

    f.write("METHOD\n")
    f.write("-" * 80 + "\n")
    f.write(f"N = {len(master_clean)}\n")
    f.write(f"EF Outcomes: {', '.join(outcome_cols)}\n")
    f.write(f"Predictors: UCLA × Gender + DASS(3) + Age\n\n")

    if manova_results is not None:
        f.write("MANOVA Results:\n")
        f.write(manova_summary_str + "\n\n")
    else:
        f.write("MANOVA: Not computed (see canonical correlation instead)\n\n")

    f.write("UNIVARIATE RESULTS (for comparison)\n")
    f.write("-" * 80 + "\n")
    for _, row in effect_summary.iterrows():
        f.write(f"{row['outcome']}:\n")
        f.write(f"  UCLA main: β={row['ucla_beta']:.4f}, p={row['ucla_p']:.4f} {row['ucla_sig']}\n")
        f.write(f"  UCLA × Gender: β={row['interaction_beta']:.4f}, p={row['interaction_p']:.4f} {row['interaction_sig']}\n\n")

    f.write("INTERPRETATION\n")
    f.write("-" * 80 + "\n")

    sig_interactions = effect_summary[effect_summary['interaction_p'] < 0.05]

    if len(sig_interactions) >= 2:
        f.write(f"✓ {len(sig_interactions)}/{len(effect_summary)} outcomes show significant UCLA × Gender.\n")
        f.write("  → Multivariate effect likely driven by these tasks:\n")
        for _, row in sig_interactions.iterrows():
            f.write(f"     - {row['outcome']} (β={row['interaction_beta']:.3f}, p={row['interaction_p']:.3f})\n")
        f.write("\n  CONCLUSION: UCLA × Gender affects multiple EF domains simultaneously.\n")
        f.write("  This supports a general 'meta-control' disruption rather than task-specific deficits.\n")

    elif len(sig_interactions) == 1:
        f.write(f"✓ 1/{len(effect_summary)} outcome shows significant UCLA × Gender:\n")
        f.write(f"   - {sig_interactions.iloc[0]['outcome']}\n\n")
        f.write("  CONCLUSION: Effect is task-specific, not a general multivariate pattern.\n")
        f.write("  MANOVA may not provide additional value beyond univariate tests.\n")

    else:
        f.write("✗ No significant UCLA × Gender interactions in any outcome.\n\n")
        f.write("  CONCLUSION: Current sample shows no multivariate loneliness × gender effect on EF.\n")
        f.write("  Main effects may be driven by DASS (affective distress) rather than UCLA.\n")

    f.write("\nTHEORETICAL IMPLICATIONS\n")
    f.write("-" * 80 + "\n")
    if len(sig_interactions) >= 2:
        f.write("Multiple EF tasks show coordinated vulnerability to loneliness (male-specific).\n")
        f.write("This suggests a DOMAIN-GENERAL meta-control mechanism, not isolated deficits.\n\n")
        f.write("Possible mechanisms:\n")
        f.write("  - Reduced proactive/sustained attention affecting all controlled tasks\n")
        f.write("  - Working memory disruption cascading across EF domains\n")
        f.write("  - Motivational/energetic depletion impacting effortful control globally\n")
    else:
        f.write("Limited multivariate effect suggests TASK-SPECIFIC vulnerabilities.\n")
        f.write("Each EF domain may have independent pathways to loneliness impairment.\n")

print(f"✓ Saved: MULTIVARIATE_EF_REPORT.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("\nKey outputs:")
print("  1. manova_full_output.txt - Complete MANOVA results")
print("  2. multivariate_effect_sizes.csv - Effect size summary table")
print("  3. canonical_weights.csv - EF outcome loadings")
print("  4. ef_profile_heatmap.png - Visual comparison by gender×loneliness")
print("  5. MULTIVARIATE_EF_REPORT.txt - Interpretation summary")
print("\nNext steps:")
print("  → If MANOVA shows significant multivariate effect, cite for multiple comparison defense")
print("  → Use canonical weights to interpret which EF tasks drive the effect")
print("  → Compare with univariate results to assess domain-general vs task-specific patterns")
