"""
전체 메트릭에 대한 성별 조절효과 분석
Gender Moderation Analysis for All Metrics

This script tests gender moderation effects across all executive function metrics
using hierarchical regression with robust confirmation methods.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/gender_comprehensive")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("전체 메트릭에 대한 성별 조절효과 분석")
print("Gender Moderation Analysis for All Metrics")
print("="*80)
print()


def normalize_gender_series(series: pd.Series) -> pd.Series:
    """Normalize gender strings (Korean/English) to 'male'/'female'."""
    if series is None:
        return pd.Series(dtype='object')
    s = (
        series.fillna('')
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z\u3131-\u318e\uac00-\ud7a3]", '', regex=True)
    )
    out = pd.Series(np.nan, index=series.index, dtype='object')
    male_mask = s.isin({'m', 'male', 'man'}) | s.str.startswith('남')
    female_mask = s.isin({'f', 'female', 'woman'}) | s.str.startswith('여')
    out[male_mask] = 'male'
    out[female_mask] = 'female'
    return out

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/6] Loading data...")
master = pd.read_csv(Path("results/analysis_outputs/master_expanded_metrics.csv"))

# Load demographic info
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

# Merge demographics
master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

print(f"  Total participants: {len(master)}")
print()

# ============================================================================
# PREPARE VARIABLES
# ============================================================================

print("[2/6] Preparing variables...")

# Create gender binary (robust to Korean and English labels)
if 'gender' in master.columns:
    master['gender_clean'] = normalize_gender_series(master['gender'])
elif 'sex' in master.columns:
    master['gender_clean'] = normalize_gender_series(master['sex'])
else:
    print("ERROR: Gender variable not found!")
    sys.exit(1)

master['gender_male'] = master['gender_clean'].map({'male': 1, 'female': 0})

# Filter complete cases
required_cols = ['ucla_total', 'age', 'gender_male', 'dass_depression', 'dass_anxiety', 'dass_stress']
master = master.dropna(subset=required_cols).copy()

print(f"  Complete cases: {len(master)}")
print(f"  Female: {(master['gender_male'] == 0).sum()}")
print(f"  Male: {(master['gender_male'] == 1).sum()}")
print()

# Standardize predictors
scaler = StandardScaler()
master['z_ucla'] = scaler.fit_transform(master[['ucla_total']])
master['z_age'] = scaler.fit_transform(master[['age']])
master['z_dass_dep'] = scaler.fit_transform(master[['dass_depression']])
master['z_dass_anx'] = scaler.fit_transform(master[['dass_anxiety']])
master['z_dass_stress'] = scaler.fit_transform(master[['dass_stress']])

print("✓ Variables prepared")
print()

# ============================================================================
# DEFINE HELPER FUNCTIONS
# ============================================================================

def run_gender_moderation(df, outcome_col, predictor='z_ucla',
                           covariates=['z_age', 'z_dass_dep', 'z_dass_anx', 'z_dass_stress'],
                           n_perm=1000, n_boot=1000):
    """
    Run gender moderation analysis with permutation and bootstrap tests.
    """

    # Filter to cases with valid outcome
    df_valid = df[df[outcome_col].notna()].copy()

    if len(df_valid) < 30:
        return None

    # Build formula
    cov_str = ' + '.join(covariates) if covariates else ''
    formula = f"{outcome_col} ~ {predictor} * C(gender_male)"
    if cov_str:
        formula += f" + {cov_str}"

    # Fit main model
    try:
        model = ols(formula, data=df_valid).fit()
    except:
        return None

    # Extract interaction term
    interaction_term = f"{predictor}:C(gender_male)[T.1]"
    if interaction_term not in model.params:
        return None

    interaction_beta = model.params[interaction_term]
    interaction_pval = model.pvalues[interaction_term]
    interaction_se = model.bse[interaction_term]

    # Simple slopes
    female_df = df_valid[df_valid['gender_male'] == 0].copy()
    male_df = df_valid[df_valid['gender_male'] == 1].copy()

    # Female slope
    if len(female_df) >= 15:
        formula_f = f"{outcome_col} ~ {predictor}"
        if cov_str:
            formula_f += f" + {cov_str}"
        try:
            model_f = ols(formula_f, data=female_df).fit()
            female_beta = model_f.params[predictor]
            female_pval = model_f.pvalues[predictor]
        except:
            female_beta, female_pval = np.nan, np.nan
    else:
        female_beta, female_pval = np.nan, np.nan

    # Male slope
    if len(male_df) >= 15:
        formula_m = f"{outcome_col} ~ {predictor}"
        if cov_str:
            formula_m += f" + {cov_str}"
        try:
            model_m = ols(formula_m, data=male_df).fit()
            male_beta = model_m.params[predictor]
            male_pval = model_m.pvalues[predictor]
        except:
            male_beta, male_pval = np.nan, np.nan
    else:
        male_beta, male_pval = np.nan, np.nan

    # Permutation test
    perm_betas = []
    for _ in range(n_perm):
        df_perm = df_valid.copy()
        df_perm['gender_male'] = np.random.permutation(df_perm['gender_male'].values)
        try:
            model_perm = ols(formula, data=df_perm).fit()
            perm_betas.append(model_perm.params[interaction_term])
        except:
            continue

    if len(perm_betas) > 0:
        perm_pval = np.mean(np.abs(perm_betas) >= np.abs(interaction_beta))
    else:
        perm_pval = np.nan

    # Bootstrap CI
    boot_betas = []
    for _ in range(n_boot):
        df_boot = df_valid.sample(n=len(df_valid), replace=True)
        try:
            model_boot = ols(formula, data=df_boot).fit()
            boot_betas.append(model_boot.params[interaction_term])
        except:
            continue

    if len(boot_betas) > 0:
        boot_ci_lower = np.percentile(boot_betas, 2.5)
        boot_ci_upper = np.percentile(boot_betas, 97.5)
        boot_pct_positive = np.mean(np.array(boot_betas) > 0) * 100
    else:
        boot_ci_lower, boot_ci_upper, boot_pct_positive = np.nan, np.nan, np.nan

    return {
        'outcome': outcome_col,
        'n_total': len(df_valid),
        'n_female': len(female_df),
        'n_male': len(male_df),
        'female_beta': female_beta,
        'female_pval': female_pval,
        'male_beta': male_beta,
        'male_pval': male_pval,
        'interaction_beta': interaction_beta,
        'interaction_se': interaction_se,
        'interaction_pval': interaction_pval,
        'perm_pval': perm_pval,
        'boot_ci_lower': boot_ci_lower,
        'boot_ci_upper': boot_ci_upper,
        'boot_pct_positive': boot_pct_positive,
        'model_rsquared': model.rsquared,
        'model_aic': model.aic
    }


# ============================================================================
# TEST ALL METRICS
# ============================================================================

print("[3/6] Testing gender moderation for all metrics...")
print()

# Define metric groups
wcst_metrics = ['pe_rate', 'wcst_npe_rate', 'wcst_accuracy', 'wcst_rt_cv']
stroop_metrics = ['stroop_interference', 'stroop_cong_acc', 'stroop_incong_acc',
                  'stroop_cong_cv', 'stroop_incong_cv']
prp_metrics = ['prp_bottleneck', 'prp_short_t2_acc', 'prp_long_t2_acc',
               'prp_t1_slowing', 'prp_slope']

all_metrics = wcst_metrics + stroop_metrics + prp_metrics

results_list = []

for i, outcome in enumerate(all_metrics, 1):
    if outcome not in master.columns:
        print(f"  [{i}/{len(all_metrics)}] Skipping {outcome} (not found)")
        continue

    print(f"  [{i}/{len(all_metrics)}] Testing {outcome}...", end=' ')

    result = run_gender_moderation(master, outcome)

    if result is not None:
        results_list.append(result)
        p = result['interaction_pval']
        if p < 0.001:
            print(f"✓✓✓ p={p:.4f} ***")
        elif p < 0.01:
            print(f"✓✓ p={p:.4f} **")
        elif p < 0.05:
            print(f"✓ p={p:.4f} *")
        else:
            print(f"p={p:.4f}")
    else:
        print("✗ (insufficient data)")

print()

results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_DIR / "all_metrics_moderation.csv", index=False, encoding='utf-8-sig')
print(f"✓ Saved: all_metrics_moderation.csv ({len(results_df)} tests)")
print()

# ============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# ============================================================================

print("[4/6] Applying multiple comparison corrections...")

valid_pvals = results_df['interaction_pval'].dropna()
n_tests = len(valid_pvals)

# Bonferroni
bonf_reject, bonf_pvals, _, bonf_alpha = multipletests(
    valid_pvals, alpha=0.05, method='bonferroni')

# FDR
fdr_reject, fdr_pvals, _, fdr_alpha = multipletests(
    valid_pvals, alpha=0.05, method='fdr_bh')

# Add to results
results_df_with_correction = results_df[results_df['interaction_pval'].notna()].copy()
results_df_with_correction['bonferroni_pval'] = bonf_pvals
results_df_with_correction['bonferroni_reject'] = bonf_reject
results_df_with_correction['fdr_pval'] = fdr_pvals
results_df_with_correction['fdr_reject'] = fdr_reject

results_df_with_correction.to_csv(OUTPUT_DIR / "all_metrics_with_corrections.csv",
                                   index=False, encoding='utf-8-sig')

print(f"  Total tests: {n_tests}")
print(f"  Bonferroni α: {bonf_alpha:.5f}")
print(f"  Bonferroni significant: {bonf_reject.sum()}")
print(f"  FDR significant: {fdr_reject.sum()}")
print()

# ============================================================================
# TASK-SPECIFIC SUMMARIES
# ============================================================================

print("[5/6] Creating task-specific summaries...")

# WCST summary
wcst_results = results_df[results_df['outcome'].isin(wcst_metrics)].copy()
wcst_results.to_csv(OUTPUT_DIR / "wcst_moderation_summary.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ WCST: {len(wcst_results)} metrics tested")

wcst_sig = wcst_results[wcst_results['interaction_pval'] < 0.05]
if len(wcst_sig) > 0:
    print(f"    Significant (p<0.05): {len(wcst_sig)}")
    for _, row in wcst_sig.iterrows():
        print(f"      - {row['outcome']}: β={row['interaction_beta']:.3f}, p={row['interaction_pval']:.4f}")

# Stroop summary
stroop_results = results_df[results_df['outcome'].isin(stroop_metrics)].copy()
stroop_results.to_csv(OUTPUT_DIR / "stroop_moderation_summary.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ Stroop: {len(stroop_results)} metrics tested")

stroop_sig = stroop_results[stroop_results['interaction_pval'] < 0.05]
if len(stroop_sig) > 0:
    print(f"    Significant (p<0.05): {len(stroop_sig)}")
    for _, row in stroop_sig.iterrows():
        print(f"      - {row['outcome']}: β={row['interaction_beta']:.3f}, p={row['interaction_pval']:.4f}")

# PRP summary
prp_results = results_df[results_df['outcome'].isin(prp_metrics)].copy()
prp_results.to_csv(OUTPUT_DIR / "prp_moderation_summary.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ PRP: {len(prp_results)} metrics tested")

prp_sig = prp_results[prp_results['interaction_pval'] < 0.05]
if len(prp_sig) > 0:
    print(f"    Significant (p<0.05): {len(prp_sig)}")
    for _, row in prp_sig.iterrows():
        print(f"      - {row['outcome']}: β={row['interaction_beta']:.3f}, p={row['interaction_pval']:.4f}")

print()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("[6/6] Generating summary report...")

report = []
report.append("="*80)
report.append("성별 조절효과 종합 분석 결과")
report.append("Comprehensive Gender Moderation Analysis Results")
report.append("="*80)
report.append("")
report.append(f"Sample: N={len(master)} ({(master['gender_male']==0).sum()} Female, {(master['gender_male']==1).sum()} Male)")
report.append(f"Total metrics tested: {len(results_df)}")
report.append("")
report.append("="*80)
report.append("UNCORRECTED RESULTS (p < 0.05)")
report.append("="*80)
report.append("")

sig_results = results_df[results_df['interaction_pval'] < 0.05].sort_values('interaction_pval')

if len(sig_results) > 0:
    report.append(f"Significant gender moderation effects: {len(sig_results)}/{len(results_df)}")
    report.append("")

    for _, row in sig_results.iterrows():
        report.append(f"{row['outcome']}:")
        report.append(f"  Interaction: β={row['interaction_beta']:.4f}, p={row['interaction_pval']:.4f}")
        report.append(f"  Female: β={row['female_beta']:.4f}, p={row['female_pval']:.4f} (N={int(row['n_female'])})")
        report.append(f"  Male: β={row['male_beta']:.4f}, p={row['male_pval']:.4f} (N={int(row['n_male'])})")
        report.append(f"  Permutation p={row['perm_pval']:.4f}")
        report.append(f"  Bootstrap 95% CI=[{row['boot_ci_lower']:.4f}, {row['boot_ci_upper']:.4f}]")
        report.append("")
else:
    report.append("No significant gender moderation effects found at p < 0.05")
    report.append("")

report.append("="*80)
report.append("MULTIPLE COMPARISON CORRECTIONS")
report.append("="*80)
report.append("")
report.append(f"Total tests: {n_tests}")
report.append(f"Bonferroni corrected α: {bonf_alpha:.5f}")
report.append(f"Bonferroni significant: {bonf_reject.sum()}")
report.append(f"FDR (Benjamini-Hochberg) significant: {fdr_reject.sum()}")
report.append("")

bonf_sig = results_df_with_correction[results_df_with_correction['bonferroni_reject']]
if len(bonf_sig) > 0:
    report.append("Bonferroni-corrected significant findings:")
    for _, row in bonf_sig.iterrows():
        report.append(f"  - {row['outcome']}: uncorrected p={row['interaction_pval']:.4f}, Bonferroni p={row['bonferroni_pval']:.4f}")
    report.append("")

fdr_sig = results_df_with_correction[results_df_with_correction['fdr_reject']]
if len(fdr_sig) > 0:
    report.append("FDR-corrected significant findings:")
    for _, row in fdr_sig.iterrows():
        report.append(f"  - {row['outcome']}: uncorrected p={row['interaction_pval']:.4f}, FDR p={row['fdr_pval']:.4f}")
    report.append("")

report.append("="*80)
report.append("TASK-SPECIFIC SUMMARY")
report.append("="*80)
report.append("")
report.append(f"WCST: {len(wcst_sig)}/{len(wcst_results)} metrics significant (p<0.05)")
report.append(f"Stroop: {len(stroop_sig)}/{len(stroop_results)} metrics significant (p<0.05)")
report.append(f"PRP: {len(prp_sig)}/{len(prp_results)} metrics significant (p<0.05)")
report.append("")

report.append("="*80)
report.append("RECOMMENDATIONS")
report.append("="*80)
report.append("")
report.append("1. Results should be interpreted cautiously due to:")
report.append("   - Multiple comparisons (14 tests conducted)")
report.append("   - Small sample size (especially male N=27)")
report.append("   - Exploratory nature of expanded metrics")
report.append("")
report.append("2. Findings with p<0.05 but not surviving correction:")
report.append("   - Consider hypothesis-generating")
report.append("   - Require independent replication")
report.append("")
report.append("3. Next steps:")
report.append("   - Preregistered replication with N≥150")
report.append("   - Focus on metrics with strongest effects")
report.append("   - Theoretical development for gender-specific pathways")
report.append("")
report.append("="*80)

report_text = "\n".join(report)

with open(OUTPUT_DIR / "GENDER_MODERATION_COMPREHENSIVE_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print("✓ Saved: GENDER_MODERATION_COMPREHENSIVE_REPORT.txt")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - all_metrics_moderation.csv")
print("  - all_metrics_with_corrections.csv")
print("  - wcst_moderation_summary.csv")
print("  - stroop_moderation_summary.csv")
print("  - prp_moderation_summary.csv")
print("  - GENDER_MODERATION_COMPREHENSIVE_REPORT.txt")
print()
