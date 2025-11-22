"""
Synthesis Analysis 8: Robust Standard Errors Sensitivity Check
===============================================================

Purpose:
--------
Address borderline variance homogeneity for WCST PE (variance ratio=1.68).
Compare standard OLS with HC3 robust standard errors to verify that the
UCLA × Gender interaction remains significant.

Statistical Approach:
--------------------
- Standard OLS regression (same as Script 4)
- HC3 heteroskedasticity-consistent standard errors (robust)
- Compare p-values and confidence intervals
- Verify interaction significance is not artifact of heteroskedasticity

Output:
-------
- robust_se_comparison.csv: Side-by-side OLS vs HC3 results
- robust_se_sensitivity_report.txt: Interpretation and conclusion

Author: Research Team
Date: 2025-01-17
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
import warnings
from data_loader_utils import load_master_dataset
warnings.filterwarnings('ignore')

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/synthesis_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SYNTHESIS ANALYSIS 8: ROBUST STANDARD ERRORS SENSITIVITY CHECK")
print("=" * 80)
print()
print("Verifying WCST PE × Gender interaction with HC3 robust standard errors")
print()

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("[1/4] Loading data...")

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
master = master.rename(columns={'gender_normalized': 'gender'})
master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Rename PE if needed
for col in ['pe_rate', 'perseverative_error_rate']:
    if col in master.columns and col != 'pe_rate':
        master = master.rename(columns={col: 'pe_rate'})
        break

# Required columns
required_cols = ['participant_id', 'pe_rate', 'ucla_total', 'gender_male',
                'dass_depression', 'dass_anxiety', 'dass_stress', 'age']

master_clean = master.dropna(subset=required_cols).copy()

print(f"  Total N = {len(master_clean)}")
print(f"    Females: {(master_clean['gender_male'] == 0).sum()}")
print(f"    Males: {(master_clean['gender_male'] == 1).sum()}")
print()

# Standardize predictors
for var in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    master_clean[f'z_{var}'] = (master_clean[var] - master_clean[var].mean()) / master_clean[var].std()

# Rename for formula
master_clean['z_ucla'] = master_clean['z_ucla_total']
master_clean['z_dass_dep'] = master_clean['z_dass_depression']
master_clean['z_dass_anx'] = master_clean['z_dass_anxiety']
master_clean['z_dass_str'] = master_clean['z_dass_stress']

# ============================================================================
# STANDARD OLS REGRESSION
# ============================================================================

print("[2/4] Running standard OLS regression...")

formula = "pe_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
model_ols = smf.ols(formula, data=master_clean).fit()

# Extract interaction term
interaction_idx = [i for i, name in enumerate(model_ols.params.index)
                  if 'z_ucla:C(gender_male)[T.1]' in name][0]

ols_beta = model_ols.params.iloc[interaction_idx]
ols_se = model_ols.bse.iloc[interaction_idx]
ols_t = model_ols.tvalues.iloc[interaction_idx]
ols_p = model_ols.pvalues.iloc[interaction_idx]
ols_ci_lower = model_ols.conf_int().iloc[interaction_idx, 0]
ols_ci_upper = model_ols.conf_int().iloc[interaction_idx, 1]

print(f"\n  Standard OLS:")
print(f"    β = {ols_beta:.3f}")
print(f"    SE = {ols_se:.3f}")
print(f"    t = {ols_t:.3f}")
print(f"    p = {ols_p:.4f}")
print(f"    95% CI = [{ols_ci_lower:.3f}, {ols_ci_upper:.3f}]")

# ============================================================================
# HC3 ROBUST STANDARD ERRORS
# ============================================================================

print("\n[3/4] Running HC3 robust standard errors...")

# Fit with HC3 covariance
model_hc3 = model_ols.get_robustcov_results(cov_type='HC3')

hc3_beta = model_hc3.params[interaction_idx]
hc3_se = model_hc3.bse[interaction_idx]
hc3_t = model_hc3.tvalues[interaction_idx]
hc3_p = model_hc3.pvalues[interaction_idx]
hc3_ci = model_hc3.conf_int()
hc3_ci_lower = hc3_ci[interaction_idx, 0]
hc3_ci_upper = hc3_ci[interaction_idx, 1]

print(f"\n  HC3 Robust SE:")
print(f"    β = {hc3_beta:.3f}")
print(f"    SE = {hc3_se:.3f}")
print(f"    t = {hc3_t:.3f}")
print(f"    p = {hc3_p:.4f}")
print(f"    95% CI = [{hc3_ci_lower:.3f}, {hc3_ci_upper:.3f}]")

# Comparison
se_ratio = hc3_se / ols_se
p_ratio = hc3_p / ols_p

print(f"\n  Comparison:")
print(f"    SE ratio (HC3/OLS): {se_ratio:.3f}")
print(f"    p-value ratio (HC3/OLS): {p_ratio:.3f}")

if hc3_p < 0.05:
    print(f"    ✓ Interaction REMAINS significant with robust SE (p={hc3_p:.4f})")
else:
    print(f"    ✗ Interaction becomes NON-significant with robust SE (p={hc3_p:.4f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[4/4] Saving results...")

comparison_results = pd.DataFrame({
    'method': ['Standard OLS', 'HC3 Robust'],
    'beta': [ols_beta, hc3_beta],
    'se': [ols_se, hc3_se],
    't_value': [ols_t, hc3_t],
    'p_value': [ols_p, hc3_p],
    'ci_lower': [ols_ci_lower, hc3_ci_lower],
    'ci_upper': [ols_ci_upper, hc3_ci_upper],
    'significant_05': [ols_p < 0.05, hc3_p < 0.05]
})

comparison_results.to_csv(OUTPUT_DIR / "robust_se_comparison.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Saved: robust_se_comparison.csv")

# ============================================================================
# GENERATE REPORT
# ============================================================================

report_lines = [
    "=" * 80,
    "ROBUST STANDARD ERRORS SENSITIVITY CHECK REPORT",
    "=" * 80,
    "",
    "Analysis: WCST PE × Gender Interaction with DASS Control",
    f"Sample Size: N={len(master_clean)} (Female={master_clean['gender_male'].value_counts()[0]}, Male={master_clean['gender_male'].value_counts()[1]})",
    "Model: pe_rate ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age",
    "",
    "=" * 80,
    "RESULTS COMPARISON",
    "=" * 80,
    "",
    "1. STANDARD OLS:",
    f"   β = {ols_beta:.3f}",
    f"   SE = {ols_se:.3f}",
    f"   t = {ols_t:.3f}",
    f"   p = {ols_p:.4f} {'*' if ols_p < 0.05 else 'ns'}",
    f"   95% CI = [{ols_ci_lower:.3f}, {ols_ci_upper:.3f}]",
    "",
    "2. HC3 ROBUST STANDARD ERRORS:",
    f"   β = {hc3_beta:.3f}",
    f"   SE = {hc3_se:.3f}",
    f"   t = {hc3_t:.3f}",
    f"   p = {hc3_p:.4f} {'*' if hc3_p < 0.05 else 'ns'}",
    f"   95% CI = [{hc3_ci_lower:.3f}, {hc3_ci_upper:.3f}]",
    "",
    "3. COMPARISON:",
    f"   SE Inflation: {((hc3_se - ols_se) / ols_se * 100):.1f}%",
    f"   p-value Change: {ols_p:.4f} → {hc3_p:.4f}",
    "",
    "=" * 80,
    "INTERPRETATION",
    "=" * 80,
    "",
]

# Add interpretation
if hc3_p < 0.05:
    report_lines.extend([
        "VERDICT: ✓ INTERACTION ROBUST",
        "",
        "The UCLA × Gender interaction remains statistically significant even with",
        "HC3 heteroskedasticity-consistent standard errors. This confirms that the",
        "finding is NOT an artifact of the borderline variance heterogeneity",
        f"(variance ratio = 1.68).",
        "",
        f"The robust standard error is {se_ratio:.2f}× larger than the OLS SE,",
        f"which increases the p-value by {((hc3_p / ols_p - 1) * 100):.1f}%. However, the effect",
        "remains significant at the α=0.05 level.",
        "",
        "CONCLUSION: The male-specific loneliness-EF vulnerability is a ROBUST finding",
        "that survives correction for potential heteroskedasticity. The interaction",
        "effect can be confidently reported in publications.",
    ])
else:
    report_lines.extend([
        "VERDICT: ✗ INTERACTION SENSITIVE TO HETEROSKEDASTICITY",
        "",
        "The UCLA × Gender interaction becomes NON-significant when using HC3 robust",
        "standard errors. This suggests that the borderline variance heterogeneity",
        f"(variance ratio = 1.68) may have inflated the Type I error rate.",
        "",
        f"The robust standard error is {se_ratio:.2f}× larger than the OLS SE,",
        f"which increases the p-value to {hc3_p:.4f} (above α=0.05).",
        "",
        "CONCLUSION: The interaction finding is FRAGILE and should be interpreted",
        "with caution. Consider reporting as exploratory and prioritize replication.",
    ])

report_lines.extend([
    "",
    "=" * 80,
    "RECOMMENDATIONS",
    "=" * 80,
    "",
])

if hc3_p < 0.05:
    report_lines.extend([
        "1. ✓ Report HC3 robust results alongside OLS in manuscript",
        "2. ✓ Emphasize robustness to heteroskedasticity assumptions",
        "3. ✓ Maintain 'Ready for publication' status",
        "4. ~ Still recommend replication (exploratory gender finding)",
    ])
else:
    report_lines.extend([
        "1. ⚠ Report HC3 robust results as primary analysis",
        "2. ⚠ Downgrade finding to 'exploratory/preliminary'",
        "3. ⚠ Emphasize need for replication",
        "4. ⚠ Consider bootstrapped confidence intervals as alternative",
    ])

report_lines.extend([
    "",
    "=" * 80,
    "Generated by: synthesis_8_robust_sensitivity.py",
    "Date: 2025-01-17",
    "=" * 80,
])

report_text = "\n".join(report_lines)

# Save report
report_path = OUTPUT_DIR / "robust_se_sensitivity_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"  ✓ Saved: robust_se_sensitivity_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SYNTHESIS ANALYSIS 8 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - robust_se_comparison.csv")
print("  - robust_se_sensitivity_report.txt")
print()

if hc3_p < 0.05:
    print("FINAL VERDICT: ✓ INTERACTION ROBUST TO HETEROSKEDASTICITY")
    print(f"  OLS p={ols_p:.4f} → HC3 p={hc3_p:.4f} (still significant)")
else:
    print("FINAL VERDICT: ✗ INTERACTION SENSITIVE TO HETEROSKEDASTICITY")
    print(f"  OLS p={ols_p:.4f} → HC3 p={hc3_p:.4f} (no longer significant)")

print()
