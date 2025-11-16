"""
Combined: Threshold Analysis + Cross-Task Correlations
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

from data_loader_utils import load_master_dataset, load_exgaussian_params

# ============================================================================
# PART 1: THRESHOLD ANALYSIS
# ============================================================================

OUTPUT_DIR1 = Path("results/analysis_outputs/threshold_analysis")
OUTPUT_DIR1.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("PART 1: DOSE-RESPONSE THRESHOLD ANALYSIS")
print("=" * 80)

master = load_master_dataset()
master = master.dropna(subset=['ucla_total', 'gender_male', 'pe_rate']).copy()

print(f"\nLoaded {len(master)} participants")

results_threshold = []

for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
    df_sub = master[master['gender_male'] == gender_val].copy()

    if len(df_sub) < 20:
        continue

    print(f"\n{gender_name.upper()} (N={len(df_sub)})")
    print("-" * 60)

    # Linear model
    linear_model = smf.ols("pe_rate ~ ucla_total + age", data=df_sub).fit()
    print(f"Linear Model: R² = {linear_model.rsquared:.3f}, AIC = {linear_model.aic:.1f}")

    # Quadratic model
    df_sub['ucla_sq'] = df_sub['ucla_total'] ** 2
    quad_model = smf.ols("pe_rate ~ ucla_total + ucla_sq + age", data=df_sub).fit()
    print(f"Quadratic Model: R² = {quad_model.rsquared:.3f}, AIC = {quad_model.aic:.1f}")

    # ROC analysis
    pe_75 = df_sub['pe_rate'].quantile(0.75)
    df_sub['high_pe'] = (df_sub['pe_rate'] > pe_75).astype(int)

    if df_sub['high_pe'].sum() >= 5:
        fpr, tpr, thresholds = roc_curve(df_sub['high_pe'], df_sub['ucla_total'])
        roc_auc_val = auc(fpr, tpr)

        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_cutoff = thresholds[optimal_idx]

        print(f"ROC AUC: {roc_auc_val:.3f}")
        print(f"Optimal UCLA cutoff: {optimal_cutoff:.1f}")
        print(f"  Sensitivity: {tpr[optimal_idx]:.2%}")
        print(f"  Specificity: {(1-fpr[optimal_idx]):.2%}")

        results_threshold.append({
            'gender': gender_name,
            'n': len(df_sub),
            'linear_r2': linear_model.rsquared,
            'linear_aic': linear_model.aic,
            'quad_r2': quad_model.rsquared,
            'quad_aic': quad_model.aic,
            'roc_auc': roc_auc_val,
            'optimal_cutoff': optimal_cutoff
        })

if results_threshold:
    threshold_df = pd.DataFrame(results_threshold)
    threshold_file = OUTPUT_DIR1 / "threshold_analysis_results.csv"
    threshold_df.to_csv(threshold_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {threshold_file}")

# ============================================================================
# PART 2: CROSS-TASK CORRELATIONS
# ============================================================================

OUTPUT_DIR2 = Path("results/analysis_outputs/cross_task_correlations")
OUTPUT_DIR2.mkdir(exist_ok=True, parents=True)

print("\n\n" + "=" * 80)
print("PART 2: CROSS-TASK CORRELATIONS")
print("=" * 80)

# Load Ex-Gaussian params
try:
    prp_exg = load_exgaussian_params('prp')
    master = master.merge(prp_exg[['participant_id', 'long_tau', 'long_sigma']],
                          on='participant_id', how='left')
    master = master.rename(columns={'long_tau': 'prp_tau_long', 'long_sigma': 'prp_sigma_long'})
except:
    print("  Warning: Could not load PRP Ex-Gaussian")

try:
    stroop_exg = load_exgaussian_params('stroop')
    master = master.merge(stroop_exg[['participant_id', 'tau', 'sigma']],
                          on='participant_id', how='left', suffixes=('', '_stroop'))
except:
    print("  Warning: Could not load Stroop Ex-Gaussian")

correlation_pairs = [
    ('pe_rate', 'prp_tau_long', 'WCST PE × PRP τ'),
    ('pe_rate', 'prp_sigma_long', 'WCST PE × PRP σ'),
    ('pe_rate', 't2_rt_mean_long', 'WCST PE × PRP T2 RT (long)'),
    ('pe_rate', 'wcst_sd_rt', 'WCST PE × WCST RT SD'),
    ('pe_rate', 'tau', 'WCST PE × Stroop τ'),
    ('pe_rate', 'sigma', 'WCST PE × Stroop σ'),
    ('pe_rate', 'stroop_interference', 'WCST PE × Stroop Interference'),
]

corr_results = []

print("\n")
for var1, var2, label in correlation_pairs:
    if var1 not in master.columns or var2 not in master.columns:
        continue

    print(f"{label}")
    print("   " + "-" * 60)

    for gender_name, gender_val in [('Male', 1), ('Female', 0)]:
        df_sub = master[master['gender_male'] == gender_val].copy()
        df_clean = df_sub[[var1, var2, 'ucla_total']].dropna()

        if len(df_clean) < 10:
            continue

        r, p = stats.pearsonr(df_clean[var1], df_clean[var2])

        # Partial correlation (control UCLA)
        from scipy.stats import pearsonr

        def partial_corr(x, y, z):
            rx = x - np.polyval(np.polyfit(z, x, 1), z)
            ry = y - np.polyval(np.polyfit(z, y, 1), z)
            return pearsonr(rx, ry)

        r_partial, p_partial = partial_corr(df_clean[var1].values, df_clean[var2].values,
                                             df_clean['ucla_total'].values)

        print(f"   {gender_name}: r = {r:.3f}, p = {p:.4f}, partial r = {r_partial:.3f}, p = {p_partial:.4f}")

        corr_results.append({
            'variable_1': var1,
            'variable_2': var2,
            'label': label,
            'gender': gender_name,
            'n': len(df_clean),
            'r': r,
            'p': p,
            'r_partial': r_partial,
            'p_partial': p_partial
        })

    print()

if corr_results:
    corr_df = pd.DataFrame(corr_results)
    corr_file = OUTPUT_DIR2 / "cross_task_correlations.csv"
    corr_df.to_csv(corr_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {corr_file}")

    # Significant correlations
    sig_corr = corr_df[corr_df['p'] < 0.05]
    if len(sig_corr) > 0:
        print("\nSIGNIFICANT CORRELATIONS (p < 0.05):")
        for _, row in sig_corr.iterrows():
            print(f"  {row['label']} ({row['gender']}): r = {row['r']:.3f}, p = {row['p']:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
