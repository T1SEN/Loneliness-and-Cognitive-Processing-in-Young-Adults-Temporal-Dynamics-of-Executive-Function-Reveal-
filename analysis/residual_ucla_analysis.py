"""
Residual UCLA Analysis
======================
Tests whether UCLA loneliness effects persist after removing variance explained by DASS-21.

This analysis addresses: "Is the UCLA x Gender interaction driven by pure social loneliness,
or is it confounded with affective distress (depression/anxiety/stress)?"

Method:
1. Regress UCLA on DASS subscales → extract residuals (UCLA_resid)
2. Re-run key models with UCLA_resid instead of UCLA
3. Compare effect sizes: original UCLA vs residual UCLA

If UCLA_resid x Gender remains significant → "social-cognitive loneliness" independent of dysphoria
If it disappears → current effects are primarily affective/dysphoric loneliness

CRITICAL: This provides more intuitive interpretation than commonality analysis for reviewers.

Author: Research Team
Date: 2025-01-16
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Constants
MIN_N_REGRESSION = 30  # Minimum sample size for regression models

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/residual_ucla_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("RESIDUAL UCLA ANALYSIS")
print("=" * 80)
print("\nPurpose: Test UCLA effects after removing DASS-explained variance")
print("Method: UCLA_resid = residuals(UCLA ~ DASS_dep + DASS_anx + DASS_str)")
print("\nLoading data...")

master = load_master_dataset(use_cache=True)

if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']

# Normalize columns
master.columns = master.columns.str.lower()
if 'participantid' in master.columns:
    master.rename(columns={'participantid': 'participant_id'}, inplace=True)

# Ensure gender coding
gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female', 'M': 'male', 'F': 'female'}
if 'gender' in master.columns:
    master['gender'] = master['gender'].map(gender_map).fillna(master['gender'])
    master['gender_male'] = (master['gender'] == 'male').astype(int)

print(f"\nData overview:")
print(f"  Total N: {len(master)}")
if 'gender' in master.columns:
    print(f"  Males: {sum(master['gender']=='male')}, Females: {sum(master['gender']=='female')}")

# Check required columns
required_base = ['participant_id', 'ucla_total', 'gender_male', 'age',
                 'dass_depression', 'dass_anxiety', 'dass_stress']

missing_cols = [col for col in required_base if col not in master.columns]
if missing_cols:
    print(f"\nERROR: Missing required columns: {missing_cols}")
    print(f"Available columns: {list(master.columns)}")
    sys.exit(1)

# Clean data
master_clean = master.dropna(subset=required_base).copy()
print(f"\nAfter dropping missing covariates: N = {len(master_clean)}")

if len(master_clean) < MIN_N_REGRESSION:
    print(f"ERROR: Insufficient data (N={len(master_clean)} < {MIN_N_REGRESSION}).")
    print("Regression models require at least 30 participants for stable estimates.")
    sys.exit(1)

# Standardize predictors
print("\nStandardizing predictors...")
scaler = StandardScaler()
master_clean['z_age'] = scaler.fit_transform(master_clean[['age']])
master_clean['z_ucla'] = scaler.fit_transform(master_clean[['ucla_total']])
master_clean['z_dass_dep'] = scaler.fit_transform(master_clean[['dass_depression']])
master_clean['z_dass_anx'] = scaler.fit_transform(master_clean[['dass_anxiety']])
master_clean['z_dass_str'] = scaler.fit_transform(master_clean[['dass_stress']])

print("=" * 80)
print("STEP 1: Extract Residual UCLA (removing DASS variance)")
print("=" * 80)

# Regress UCLA on DASS subscales
ucla_on_dass = smf.ols("z_ucla ~ z_dass_dep + z_dass_anx + z_dass_str",
                       data=master_clean).fit()

print("\nUCLA ~ DASS regression:")
print(f"  R² = {ucla_on_dass.rsquared:.3f}")
print(f"  DASS explains {ucla_on_dass.rsquared*100:.1f}% of UCLA variance")
print(f"\n{ucla_on_dass.summary()}")

# Extract residuals
master_clean['ucla_resid_raw'] = ucla_on_dass.resid

# Standardize for interpretability (residuals already centered at 0 by OLS)
# This divides by SD to get z-scores (mean=0, SD=1)
resid_std = master_clean['ucla_resid_raw'].std()
if resid_std == 0 or np.isnan(resid_std):
    print("\nERROR: UCLA residuals have zero or undefined standard deviation.")
    print("This indicates all residuals are identical (no variance).")
    print("Cannot proceed with standardization.")
    sys.exit(1)

master_clean['z_ucla_resid'] = (
    master_clean['ucla_resid_raw'] / resid_std
)

# Note: OLS residuals already have mean=0 by construction,
# so this is equivalent to StandardScaler but more transparent

print(f"\nResidual UCLA statistics:")
print(f"  Mean: {master_clean['ucla_resid_raw'].mean():.6f} (should be ~0)")
print(f"  SD: {master_clean['ucla_resid_raw'].std():.3f}")
print(f"  Range: [{master_clean['ucla_resid_raw'].min():.2f}, {master_clean['ucla_resid_raw'].max():.2f}]")

# Check correlation between UCLA_resid and DASS (should be near 0)
corr_resid_dass_dep = master_clean['z_ucla_resid'].corr(master_clean['z_dass_dep'])
corr_resid_dass_anx = master_clean['z_ucla_resid'].corr(master_clean['z_dass_anx'])
corr_resid_dass_str = master_clean['z_ucla_resid'].corr(master_clean['z_dass_str'])

print(f"\nOrthogonality check (UCLA_resid vs DASS):")
print(f"  r(UCLA_resid, DASS_dep) = {corr_resid_dass_dep:.4f}")
print(f"  r(UCLA_resid, DASS_anx) = {corr_resid_dass_anx:.4f}")
print(f"  r(UCLA_resid, DASS_str) = {corr_resid_dass_str:.4f}")
print(f"  (All should be near 0 - orthogonal by construction)")

print("\n" + "=" * 80)
print("STEP 2: Re-run Key Models with Residual UCLA")
print("=" * 80)

# Define key outcomes
outcomes = []
if 'pe_rate' in master_clean.columns:
    outcomes.append(('pe_rate', 'WCST Perseverative Error Rate'))
if 'prp_tau_long' in master_clean.columns:
    outcomes.append(('prp_tau_long', 'PRP Tau (Long SOA)'))
elif 'prp_bottleneck' in master_clean.columns:
    outcomes.append(('prp_bottleneck', 'PRP Bottleneck Effect'))
if 'stroop_interference' in master_clean.columns:
    outcomes.append(('stroop_interference', 'Stroop Interference'))

if not outcomes:
    print("WARNING: No EF outcomes found. Checking available columns...")
    ef_candidates = [col for col in master_clean.columns if any(
        kw in col.lower() for kw in ['pe', 'wcst', 'prp', 'stroop', 'bottleneck', 'tau', 'interference']
    )]
    print(f"Potential EF columns: {ef_candidates}")

    # Use available columns
    for col in ef_candidates[:3]:  # Take first 3 available
        outcomes.append((col, col.replace('_', ' ').title()))

print(f"\nAnalyzing {len(outcomes)} outcome measures:")
for outcome, label in outcomes:
    print(f"  - {label} ({outcome})")

# Store results
comparison_results = []

for outcome_col, outcome_label in outcomes:
    print(f"\n{'='*80}")
    print(f"Outcome: {outcome_label}")
    print(f"{'='*80}")

    # Filter to non-missing
    df_outcome = master_clean.dropna(subset=[outcome_col]).copy()

    if len(df_outcome) < 30:
        print(f"  SKIP: Insufficient data (N={len(df_outcome)})")
        continue

    print(f"  N = {len(df_outcome)}")

    # Standardize outcome for effect size comparison
    df_outcome['z_outcome'] = scaler.fit_transform(df_outcome[[outcome_col]])

    # Model A: Original UCLA (with DASS control)
    formula_original = "z_outcome ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_original = smf.ols(formula_original, data=df_outcome).fit()

    # Model B: Residual UCLA (still with DASS control for fair comparison)
    formula_residual = "z_outcome ~ z_ucla_resid * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
    model_residual = smf.ols(formula_residual, data=df_outcome).fit()

    # Extract key coefficients
    int_term_orig = 'z_ucla:C(gender_male)[T.1]'
    int_term_resid = 'z_ucla_resid:C(gender_male)[T.1]'

    # Original model
    ucla_main_orig = model_original.params.get('z_ucla', np.nan)
    ucla_main_p_orig = model_original.pvalues.get('z_ucla', np.nan)

    if int_term_orig in model_original.params:
        interaction_orig = model_original.params[int_term_orig]
        interaction_p_orig = model_original.pvalues[int_term_orig]
    else:
        interaction_orig = np.nan
        interaction_p_orig = np.nan

    # Residual model
    ucla_main_resid = model_residual.params.get('z_ucla_resid', np.nan)
    ucla_main_p_resid = model_residual.pvalues.get('z_ucla_resid', np.nan)

    if int_term_resid in model_residual.params:
        interaction_resid = model_residual.params[int_term_resid]
        interaction_p_resid = model_residual.pvalues[int_term_resid]
    else:
        interaction_resid = np.nan
        interaction_p_resid = np.nan

    print(f"\n  ORIGINAL UCLA MODEL:")
    print(f"    UCLA main effect: β={ucla_main_orig:.4f}, p={ucla_main_p_orig:.4f}")
    print(f"    UCLA × Gender:    β={interaction_orig:.4f}, p={interaction_p_orig:.4f}")
    print(f"    R² = {model_original.rsquared:.3f}")

    print(f"\n  RESIDUAL UCLA MODEL:")
    print(f"    UCLA_resid main effect: β={ucla_main_resid:.4f}, p={ucla_main_p_resid:.4f}")
    print(f"    UCLA_resid × Gender:    β={interaction_resid:.4f}, p={interaction_p_resid:.4f}")
    print(f"    R² = {model_residual.rsquared:.3f}")

    # Interpretation
    if not np.isnan(interaction_p_orig) and not np.isnan(interaction_p_resid):
        if interaction_p_orig < 0.05 and interaction_p_resid < 0.05:
            interpretation = "SURVIVES: Interaction persists after removing DASS variance"
        elif interaction_p_orig < 0.05 and interaction_p_resid >= 0.05:
            interpretation = "ELIMINATED: Interaction disappears when DASS removed"
        elif interaction_p_orig >= 0.05:
            interpretation = "NOT SIGNIFICANT: No interaction in original model"
        else:
            interpretation = "EMERGED: Interaction appears only after removing DASS"
    else:
        interpretation = "INSUFFICIENT DATA"

    print(f"\n  INTERPRETATION: {interpretation}")

    # Store results
    comparison_results.append({
        'outcome': outcome_label,
        'outcome_var': outcome_col,
        'n': len(df_outcome),

        'ucla_main_beta': ucla_main_orig,
        'ucla_main_p': ucla_main_p_orig,
        'ucla_interaction_beta': interaction_orig,
        'ucla_interaction_p': interaction_p_orig,
        'ucla_model_r2': model_original.rsquared,

        'ucla_resid_main_beta': ucla_main_resid,
        'ucla_resid_main_p': ucla_main_p_resid,
        'ucla_resid_interaction_beta': interaction_resid,
        'ucla_resid_interaction_p': interaction_p_resid,
        'ucla_resid_model_r2': model_residual.rsquared,

        'interaction_beta_change': interaction_resid - interaction_orig if not np.isnan(interaction_resid) else np.nan,
        'interaction_p_change': interaction_p_resid - interaction_p_orig if not np.isnan(interaction_p_resid) else np.nan,
        'interpretation': interpretation
    })

print("\n" + "=" * 80)
print("STEP 3: Save Results")
print("=" * 80)

# Convert to DataFrame
results_df = pd.DataFrame(comparison_results)

# Save detailed results
results_df.to_csv(OUTPUT_DIR / "original_vs_residual_comparison.csv",
                  index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: original_vs_residual_comparison.csv ({len(results_df)} outcomes)")

# Create summary report
report_path = OUTPUT_DIR / "RESIDUAL_UCLA_REPORT.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RESIDUAL UCLA ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 80 + "\n")
    f.write("Test whether UCLA loneliness effects persist after removing variance\n")
    f.write("explained by DASS-21 (depression/anxiety/stress).\n\n")

    f.write("This addresses the question: Are we measuring 'social-cognitive loneliness'\n")
    f.write("or 'affective/dysphoric loneliness'?\n\n")

    f.write("METHOD\n")
    f.write("-" * 80 + "\n")
    f.write(f"1. Regressed UCLA on DASS subscales (R² = {ucla_on_dass.rsquared:.3f})\n")
    f.write(f"   → DASS explains {ucla_on_dass.rsquared*100:.1f}% of UCLA variance\n\n")

    f.write("2. Extracted residuals → UCLA_resid (orthogonal to DASS by construction)\n\n")

    f.write("3. Re-ran key models:\n")
    f.write("   Original:  outcome ~ UCLA × Gender + DASS + Age\n")
    f.write("   Residual:  outcome ~ UCLA_resid × Gender + DASS + Age\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-" * 80 + "\n\n")

    for _, row in results_df.iterrows():
        f.write(f"{row['outcome']}\n")
        f.write(f"  N = {row['n']}\n")
        f.write(f"  Original UCLA × Gender:  β={row['ucla_interaction_beta']:.4f}, p={row['ucla_interaction_p']:.4f}\n")
        f.write(f"  Residual UCLA × Gender:  β={row['ucla_resid_interaction_beta']:.4f}, p={row['ucla_resid_interaction_p']:.4f}\n")
        f.write(f"  → {row['interpretation']}\n\n")

    f.write("\nINTERPRETATION GUIDE\n")
    f.write("-" * 80 + "\n")
    f.write("SURVIVES: Effect independent of affective distress (pure social loneliness)\n")
    f.write("ELIMINATED: Effect was confounded with depression/anxiety/stress\n")
    f.write("EMERGED: Suppression effect - DASS was masking true social loneliness effect\n\n")

    f.write("THEORETICAL IMPLICATIONS\n")
    f.write("-" * 80 + "\n")

    survived = results_df[results_df['interpretation'].str.contains('SURVIVES', na=False)]
    eliminated = results_df[results_df['interpretation'].str.contains('ELIMINATED', na=False)]

    if len(survived) > 0:
        f.write(f"✓ {len(survived)} outcome(s) show UCLA × Gender interaction independent of DASS.\n")
        f.write("  → Supports 'social-cognitive loneliness' mechanism distinct from dysphoria.\n\n")

    if len(eliminated) > 0:
        f.write(f"✗ {len(eliminated)} outcome(s) lose UCLA × Gender interaction after removing DASS.\n")
        f.write("  → These effects are primarily affective/emotional distress, not social isolation.\n\n")

    if len(survived) == 0 and len(eliminated) == 0:
        f.write("No significant UCLA × Gender interactions found in either model.\n")
        f.write("Current sample may lack power to detect residual effects.\n\n")

    f.write("\nCONCLUSION\n")
    f.write("-" * 80 + "\n")
    if len(survived) > len(eliminated):
        f.write("Predominant pattern: UCLA × Gender effects SURVIVE removal of DASS variance.\n")
        f.write("Interpretation: These are true social-cognitive loneliness effects,\n")
        f.write("not simply dysphoria/mood disturbance.\n")
    elif len(eliminated) > len(survived):
        f.write("Predominant pattern: UCLA × Gender effects ELIMINATED after removing DASS.\n")
        f.write("Interpretation: Current 'loneliness' effects are largely affective/dysphoric,\n")
        f.write("heavily overlapping with depression and anxiety.\n")
    else:
        f.write("Mixed pattern: Effects vary by outcome measure.\n")
        f.write("Some EF domains show pure social loneliness sensitivity,\n")
        f.write("others are confounded with general affective distress.\n")

print(f"✓ Saved: RESIDUAL_UCLA_REPORT.txt")

# Save residual UCLA scores for potential reuse
residual_scores = master_clean[['participant_id', 'ucla_total', 'ucla_resid_raw',
                                 'z_ucla', 'z_ucla_resid']].copy()
residual_scores.to_csv(OUTPUT_DIR / "ucla_residual_scores.csv",
                       index=False, encoding='utf-8-sig')
print(f"✓ Saved: ucla_residual_scores.csv ({len(residual_scores)} participants)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("\nKey outputs:")
print("  1. original_vs_residual_comparison.csv - Coefficient comparison table")
print("  2. RESIDUAL_UCLA_REPORT.txt - Interpretation summary")
print("  3. ucla_residual_scores.csv - Residual UCLA scores for reuse")
print("\nNext steps:")
print("  → Review RESIDUAL_UCLA_REPORT.txt for theoretical interpretation")
print("  → If interactions SURVIVE, cite as evidence for social-specific mechanism")
print("  → If ELIMINATED, discuss affective confounding in limitations")
