"""
PRP Comprehensive Analysis with DASS Controls
==============================================
Statistically rigorous analysis of Psychological Refractory Period (PRP) task
examining UCLA loneliness × Gender effects on dual-task coordination,
with MANDATORY control for DASS-21 subscales.

Key Analyses:
1. Bottleneck effect (T2 RT: short SOA - long SOA)
2. SOA slope (linear trend across SOA bins)
3. Dual-task variability (CV, IQR by SOA condition)
4. Error cascades (T1 error → T2 error patterns)
5. Post-error adjustments (RT changes after errors)
6. Gender moderation effects (parallel to WCST findings)

CRITICAL: ALL regression models control for DASS depression, anxiety, stress.

Author: Automated analysis pipeline
Date: 2025-01-16
"""

import sys
import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from analysis.utils.trial_data_loader import load_prp_trials

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/prp_comprehensive_dass_controlled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRP COMPREHENSIVE ANALYSIS - DASS CONTROLLED")
print("=" * 80)
print("\n⚠️  CRITICAL: All models control for DASS-21 subscales (depression, anxiety, stress)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

trials, _ = load_prp_trials(
    use_cache=True,
    rt_min=200,
    rt_max=5000,
    require_t1_correct=False,
    require_t2_correct_for_rt=False,
    enforce_short_long_only=False,
    drop_timeouts=True,
)
trials.columns = trials.columns.str.lower()

# Load master dataset (UCLA, DASS, demographics)
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]
master = master.rename(columns={"gender_normalized": "gender"})
master["gender"] = master["gender"].fillna("").astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

print(f"  Master dataset: N={len(master)}")
print(f"    Males: {sum(master['gender']=='male')}, Females: {sum(master['gender']=='female')}")
print(f"  Trial data: {len(trials)} trials")

# ============================================================================
# CLEAN TRIAL DATA
# ============================================================================
print("\n[2] Cleaning trial data...")

# Identify RT columns
rt_col_t1 = 't1_rt_ms' if 't1_rt_ms' in trials.columns else 't1_rt'
rt_col_t2 = 't2_rt_ms' if 't2_rt_ms' in trials.columns else 't2_rt'
soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in trials.columns else 'soa'

# Filter valid trials
trials_clean = trials[
    (trials['t1_correct'].notna()) &
    (trials['t2_correct'].notna()) &
    (trials[rt_col_t1].notna()) &
    (trials[rt_col_t2].notna()) &
    (trials[rt_col_t1] > 200) &  # Remove too-fast RTs
    (trials[rt_col_t1] < 5000) &  # Remove too-slow RTs
    (trials[rt_col_t2] > 200) &
    (trials[rt_col_t2] < 5000)
].copy()

trials_clean['t1_rt'] = trials_clean[rt_col_t1]
trials_clean['t2_rt'] = trials_clean[rt_col_t2]
trials_clean['soa'] = trials_clean[soa_col]

# SOA categorization (consistent with project standards)
def categorize_soa(soa):
    if soa <= 150:
        return 'short'
    elif 300 <= soa <= 600:
        return 'medium'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'

trials_clean['soa_cat'] = trials_clean['soa'].apply(categorize_soa)
trials_clean = trials_clean[trials_clean['soa_cat'] != 'other']

# Add trial index within participant
trials_clean = trials_clean.sort_values(['participant_id', 'idx' if 'idx' in trials_clean.columns else 'trial_index'])
trials_clean['trial_num'] = trials_clean.groupby('participant_id').cumcount() + 1

print(f"  Valid trials after cleaning: {len(trials_clean)}")
print(f"    Short SOA (≤150ms): {sum(trials_clean['soa_cat']=='short')}")
print(f"    Medium SOA (300-600ms): {sum(trials_clean['soa_cat']=='medium')}")
print(f"    Long SOA (≥1200ms): {sum(trials_clean['soa_cat']=='long')}")

# ============================================================================
# METRIC 1: BOTTLENECK EFFECT (Primary DV)
# ============================================================================
print("\n[3] Computing bottleneck effect (T2 RT: short - long SOA)...")

bottleneck_df = trials_clean.groupby(['participant_id', 'soa_cat'])['t2_rt'].mean().unstack(fill_value=np.nan)
bottleneck_df = bottleneck_df.reset_index()
bottleneck_df['prp_bottleneck_computed'] = bottleneck_df['short'] - bottleneck_df['long']

# Also compute mean T2 RT across all SOAs
mean_t2_rt = trials_clean.groupby('participant_id')['t2_rt'].mean().reset_index()
mean_t2_rt.columns = ['participant_id', 't2_rt_mean']

bottleneck_df = bottleneck_df.merge(mean_t2_rt, on='participant_id', how='left')

print(f"  Bottleneck effect computed for N={len(bottleneck_df)} participants")
print(f"    Mean bottleneck: {bottleneck_df['prp_bottleneck_computed'].mean():.1f} ms")
print(f"    SD: {bottleneck_df['prp_bottleneck_computed'].std():.1f} ms")

# ============================================================================
# METRIC 2: SOA SLOPE (Linear trend)
# ============================================================================
print("\n[4] Computing SOA slope (linear trend in T2 RT)...")

def compute_soa_slope(group):
    """Compute linear slope of T2 RT across SOA bins."""
    soa_means = group.groupby('soa_cat')['t2_rt'].mean()

    # Map SOA categories to numeric values for regression
    soa_numeric = {'short': 1, 'medium': 2, 'long': 3}

    x = [soa_numeric[cat] for cat in soa_means.index if cat in soa_numeric]
    y = [soa_means[cat] for cat in soa_means.index if cat in soa_numeric]

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        return pd.Series({'soa_slope': slope, 'soa_intercept': intercept})
    else:
        return pd.Series({'soa_slope': np.nan, 'soa_intercept': np.nan})

soa_slope_df = trials_clean.groupby('participant_id').apply(compute_soa_slope).reset_index()

print(f"  SOA slope computed for N={len(soa_slope_df)} participants")
print(f"    Mean slope: {soa_slope_df['soa_slope'].mean():.1f} ms/step")

# ============================================================================
# METRIC 3: DUAL-TASK VARIABILITY (CV, IQR by SOA)
# ============================================================================
print("\n[5] Computing dual-task RT variability...")

def compute_variability_metrics(group):
    """Compute CV and IQR of T2 RT for each SOA condition."""
    metrics = {}

    for soa_cat in ['short', 'medium', 'long']:
        soa_trials = group[group['soa_cat'] == soa_cat]
        if len(soa_trials) >= 5:
            rt_vals = soa_trials['t2_rt'].values
            metrics[f't2_cv_{soa_cat}'] = (np.std(rt_vals) / np.mean(rt_vals)) if np.mean(rt_vals) > 0 else np.nan
            metrics[f't2_iqr_{soa_cat}'] = np.percentile(rt_vals, 75) - np.percentile(rt_vals, 25)
        else:
            metrics[f't2_cv_{soa_cat}'] = np.nan
            metrics[f't2_iqr_{soa_cat}'] = np.nan

    # Overall T2 variability
    rt_vals_all = group['t2_rt'].values
    if len(rt_vals_all) >= 10:
        metrics['t2_cv_overall'] = np.std(rt_vals_all) / np.mean(rt_vals_all)
        metrics['t2_iqr_overall'] = np.percentile(rt_vals_all, 75) - np.percentile(rt_vals_all, 25)
    else:
        metrics['t2_cv_overall'] = np.nan
        metrics['t2_iqr_overall'] = np.nan

    return pd.Series(metrics)

variability_df = trials_clean.groupby('participant_id').apply(compute_variability_metrics).reset_index()

print(f"  Variability metrics computed for N={len(variability_df)} participants")
print(f"    Mean CV (overall): {variability_df['t2_cv_overall'].mean():.3f}")
print(f"    Mean IQR (overall): {variability_df['t2_iqr_overall'].mean():.1f} ms")

# ============================================================================
# METRIC 4: ERROR CASCADES (T1 error → T2 error)
# ============================================================================
print("\n[6] Computing error cascade patterns...")

def compute_error_cascades(group):
    """Compute error cascade metrics."""
    metrics = {}

    # T1 error rate
    metrics['t1_error_rate'] = 1 - group['t1_correct'].mean()

    # T2 error rate
    metrics['t2_error_rate'] = 1 - group['t2_correct'].mean()

    # Cascade rate: P(T2 error | T1 error)
    t1_errors = group[group['t1_correct'] == False]
    if len(t1_errors) >= 3:
        metrics['cascade_rate'] = 1 - t1_errors['t2_correct'].mean()
        metrics['cascade_n'] = len(t1_errors)
    else:
        metrics['cascade_rate'] = np.nan
        metrics['cascade_n'] = 0

    # Non-cascade T2 error rate: P(T2 error | T1 correct)
    t1_correct = group[group['t1_correct'] == True]
    if len(t1_correct) >= 10:
        metrics['t2_error_rate_no_cascade'] = 1 - t1_correct['t2_correct'].mean()
    else:
        metrics['t2_error_rate_no_cascade'] = np.nan

    # Cascade inflation: (Cascade rate - Base T2 error rate)
    if not np.isnan(metrics['cascade_rate']) and not np.isnan(metrics['t2_error_rate_no_cascade']):
        metrics['cascade_inflation'] = metrics['cascade_rate'] - metrics['t2_error_rate_no_cascade']
    else:
        metrics['cascade_inflation'] = np.nan

    return pd.Series(metrics)

cascade_df = trials_clean.groupby('participant_id').apply(compute_error_cascades).reset_index()

print(f"  Error cascades computed for N={len(cascade_df)} participants")
print(f"    Mean cascade rate: {cascade_df['cascade_rate'].mean():.3f}")
print(f"    Mean cascade inflation: {cascade_df['cascade_inflation'].mean():.3f}")

# ============================================================================
# METRIC 5: POST-ERROR ADJUSTMENTS
# ============================================================================
print("\n[7] Computing post-error RT adjustments...")

def compute_post_error_adjustments(group):
    """Compute post-error slowing (PES) for T1 and T2."""
    group = group.sort_values('trial_num')

    metrics = {}

    # Post-T1-error T2 RT
    t1_error_trials = group[group['t1_correct'] == False].index
    post_t1_error = group.loc[t1_error_trials]

    # Post-T1-correct T2 RT
    t1_correct_trials = group[group['t1_correct'] == True].index
    post_t1_correct = group.loc[t1_correct_trials]

    if len(post_t1_error) >= 3 and len(post_t1_correct) >= 10:
        metrics['t2_rt_post_t1_error'] = post_t1_error['t2_rt'].mean()
        metrics['t2_rt_post_t1_correct'] = post_t1_correct['t2_rt'].mean()
        metrics['t2_pes_t1_error'] = metrics['t2_rt_post_t1_error'] - metrics['t2_rt_post_t1_correct']
    else:
        metrics['t2_rt_post_t1_error'] = np.nan
        metrics['t2_rt_post_t1_correct'] = np.nan
        metrics['t2_pes_t1_error'] = np.nan

    # For next-trial effects, we need to shift
    group['t1_correct_prev'] = group['t1_correct'].shift(1)
    group['t2_correct_prev'] = group['t2_correct'].shift(1)

    # T1 RT after T1 error vs correct (classic PES)
    prev_t1_error = group[group['t1_correct_prev'] == False]
    prev_t1_correct = group[group['t1_correct_prev'] == True]

    if len(prev_t1_error) >= 3 and len(prev_t1_correct) >= 10:
        metrics['t1_rt_post_error'] = prev_t1_error['t1_rt'].mean()
        metrics['t1_rt_post_correct'] = prev_t1_correct['t1_rt'].mean()
        metrics['t1_pes'] = metrics['t1_rt_post_error'] - metrics['t1_rt_post_correct']
    else:
        metrics['t1_rt_post_error'] = np.nan
        metrics['t1_rt_post_correct'] = np.nan
        metrics['t1_pes'] = np.nan

    return pd.Series(metrics)

pes_df = trials_clean.groupby('participant_id').apply(compute_post_error_adjustments).reset_index()

print(f"  Post-error adjustments computed for N={len(pes_df)} participants")
print(f"    Mean T1 PES: {pes_df['t1_pes'].mean():.1f} ms")
print(f"    Mean T2 PES (post-T1-error): {pes_df['t2_pes_t1_error'].mean():.1f} ms")

# ============================================================================
# METRIC 6: TEMPORAL DYNAMICS (Early vs Late trials)
# ============================================================================
print("\n[8] Computing temporal dynamics (learning/fatigue)...")

def compute_temporal_dynamics(group):
    """Compare early vs late trials to detect learning or fatigue."""
    group = group.sort_values('trial_num')

    n_trials = len(group)
    if n_trials < 20:
        return pd.Series({
            't2_rt_early': np.nan, 't2_rt_late': np.nan, 't2_rt_drift': np.nan,
            'acc_t2_early': np.nan, 'acc_t2_late': np.nan, 'acc_t2_drift': np.nan
        })

    # Split into thirds
    third = n_trials // 3
    early = group.iloc[:third]
    late = group.iloc[-third:]

    metrics = {
        't2_rt_early': early['t2_rt'].mean(),
        't2_rt_late': late['t2_rt'].mean(),
        't2_rt_drift': late['t2_rt'].mean() - early['t2_rt'].mean(),
        'acc_t2_early': early['t2_correct'].mean(),
        'acc_t2_late': late['t2_correct'].mean(),
        'acc_t2_drift': late['t2_correct'].mean() - early['t2_correct'].mean()
    }

    return pd.Series(metrics)

temporal_df = trials_clean.groupby('participant_id').apply(compute_temporal_dynamics).reset_index()

print(f"  Temporal dynamics computed for N={len(temporal_df)} participants")
print(f"    Mean RT drift (late - early): {temporal_df['t2_rt_drift'].mean():.1f} ms")
print(f"    Mean accuracy drift: {temporal_df['acc_t2_drift'].mean():.3f}")

# ============================================================================
# MERGE ALL METRICS
# ============================================================================
print("\n[9] Merging all metrics into master dataset...")

# Merge all computed metrics
prp_metrics = bottleneck_df.copy()
prp_metrics = prp_metrics.merge(soa_slope_df, on='participant_id', how='outer')
prp_metrics = prp_metrics.merge(variability_df, on='participant_id', how='outer')
prp_metrics = prp_metrics.merge(cascade_df, on='participant_id', how='outer')
prp_metrics = prp_metrics.merge(pes_df, on='participant_id', how='outer')
prp_metrics = prp_metrics.merge(temporal_df, on='participant_id', how='outer')

# Merge with master dataset
analysis_df = master.merge(prp_metrics, on='participant_id', how='inner')

# Standardize key variables for regression
for var in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    if var in analysis_df.columns:
        analysis_df[f'z_{var}'] = (analysis_df[var] - analysis_df[var].mean()) / analysis_df[var].std()

print(f"  Final analysis dataset: N={len(analysis_df)}")
print(f"    Complete cases for main analysis: {analysis_df[['prp_bottleneck_computed', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'gender']].dropna().shape[0]}")

# Save merged dataset
analysis_df.to_csv(OUTPUT_DIR / "prp_metrics_master.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STATISTICAL ANALYSIS: UCLA × GENDER EFFECTS WITH DASS CONTROLS
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS: UCLA × GENDER EFFECTS (DASS CONTROLLED)")
print("=" * 80)

# List of DVs to test
dvs = [
    ('prp_bottleneck_computed', 'PRP Bottleneck (short - long SOA)'),
    ('t2_rt_mean', 'Mean T2 RT'),
    ('soa_slope', 'SOA Slope'),
    ('t2_cv_overall', 'T2 RT Variability (CV)'),
    ('t2_iqr_overall', 'T2 RT Variability (IQR)'),
    ('t2_cv_short', 'T2 CV at Short SOA'),
    ('cascade_rate', 'Error Cascade Rate'),
    ('cascade_inflation', 'Cascade Inflation'),
    ('t1_pes', 'T1 Post-Error Slowing'),
    ('t2_pes_t1_error', 'T2 RT After T1 Error'),
    ('t2_rt_drift', 'T2 RT Drift (fatigue)'),
]

results_list = []

for dv_col, dv_name in dvs:
    if dv_col not in analysis_df.columns:
        continue

    # Drop missing values for this DV
    df_clean = analysis_df.dropna(subset=[dv_col, 'z_ucla_total', 'z_dass_depression', 'z_dass_anxiety', 'z_dass_stress', 'z_age', 'gender_male'])

    if len(df_clean) < 20:
        continue

    print(f"\n{'='*80}")
    print(f"DV: {dv_name} (N={len(df_clean)})")
    print(f"{'='*80}")

    # Model 1: UCLA main effect (DASS controlled)
    try:
        formula1 = f"{dv_col} ~ z_ucla_total + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
        model1 = smf.ols(formula1, data=df_clean).fit()

        print(f"\nModel 1: UCLA Main Effect (DASS controlled)")
        print(f"  UCLA β = {model1.params['z_ucla_total']:.3f}, p = {model1.pvalues['z_ucla_total']:.4f}")

        results_list.append({
            'dv': dv_name,
            'model': 'UCLA_main',
            'n': len(df_clean),
            'beta_ucla': model1.params['z_ucla_total'],
            'se_ucla': model1.bse['z_ucla_total'],
            'p_ucla': model1.pvalues['z_ucla_total'],
            'r2': model1.rsquared,
            'r2_adj': model1.rsquared_adj,
        })
    except Exception as e:
        print(f"  Model 1 failed: {e}")

    # Model 2: UCLA × Gender interaction (DASS controlled)
    try:
        formula2 = f"{dv_col} ~ z_ucla_total * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
        model2 = smf.ols(formula2, data=df_clean).fit()

        print(f"\nModel 2: UCLA × Gender Interaction (DASS controlled)")
        print(f"  UCLA β = {model2.params['z_ucla_total']:.3f}, p = {model2.pvalues['z_ucla_total']:.4f}")
        print(f"  Gender β = {model2.params['C(gender_male)[T.1]']:.3f}, p = {model2.pvalues['C(gender_male)[T.1]']:.4f}")

        interaction_term = 'z_ucla_total:C(gender_male)[T.1]'
        if interaction_term in model2.params:
            print(f"  UCLA × Gender β = {model2.params[interaction_term]:.3f}, p = {model2.pvalues[interaction_term]:.4f}")

            results_list.append({
                'dv': dv_name,
                'model': 'UCLA_x_Gender',
                'n': len(df_clean),
                'beta_ucla': model2.params['z_ucla_total'],
                'se_ucla': model2.bse['z_ucla_total'],
                'p_ucla': model2.pvalues['z_ucla_total'],
                'beta_gender': model2.params['C(gender_male)[T.1]'],
                'p_gender': model2.pvalues['C(gender_male)[T.1]'],
                'beta_interaction': model2.params[interaction_term],
                'se_interaction': model2.bse[interaction_term],
                'p_interaction': model2.pvalues[interaction_term],
                'r2': model2.rsquared,
                'r2_adj': model2.rsquared_adj,
            })

        # Gender-stratified effects
        print(f"\nGender-Stratified Effects:")
        for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
            df_gender = df_clean[df_clean['gender_male'] == gender_val]
            if len(df_gender) >= 10:
                formula_strat = f"{dv_col} ~ z_ucla_total + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"
                model_strat = smf.ols(formula_strat, data=df_gender).fit()

                print(f"  {gender_label} (N={len(df_gender)}): UCLA β = {model_strat.params['z_ucla_total']:.3f}, p = {model_strat.pvalues['z_ucla_total']:.4f}")

                results_list.append({
                    'dv': dv_name,
                    'model': f'UCLA_{gender_label}',
                    'n': len(df_gender),
                    'beta_ucla': model_strat.params['z_ucla_total'],
                    'se_ucla': model_strat.bse['z_ucla_total'],
                    'p_ucla': model_strat.pvalues['z_ucla_total'],
                    'r2': model_strat.rsquared,
                    'r2_adj': model_strat.rsquared_adj,
                })

    except Exception as e:
        print(f"  Model 2 failed: {e}")

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_DIR / "prp_dass_controlled_regression_results.csv", index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("SUMMARY OF SIGNIFICANT EFFECTS (p < 0.05)")
print("=" * 80)

sig_effects = results_df[results_df['p_ucla'] < 0.05]
if len(sig_effects) > 0:
    print(sig_effects[['dv', 'model', 'n', 'beta_ucla', 'p_ucla', 'r2_adj']].to_string(index=False))
else:
    print("  No significant UCLA effects after DASS control")

print("\n" + "=" * 80)
print("INTERACTION EFFECTS (p < 0.10)")
print("=" * 80)

interaction_effects = results_df[
    (results_df['model'] == 'UCLA_x_Gender') &
    (results_df['p_interaction'].notna()) &
    (results_df['p_interaction'] < 0.10)
]
if len(interaction_effects) > 0:
    print(interaction_effects[['dv', 'n', 'beta_interaction', 'p_interaction', 'r2_adj']].to_string(index=False))
else:
    print("  No significant UCLA × Gender interactions")

# ============================================================================
# FINAL SUMMARY REPORT
# ============================================================================
summary_report = f"""
PRP COMPREHENSIVE ANALYSIS - FINAL SUMMARY
==========================================

Data Summary:
- Total participants: {len(analysis_df)}
- Males: {sum(analysis_df['gender_male']==1)}, Females: {sum(analysis_df['gender_male']==0)}
- Total valid trials analyzed: {len(trials_clean)}

Key Metrics Computed:
1. Bottleneck Effect: T2 RT (short SOA - long SOA)
   - Mean: {analysis_df['prp_bottleneck_computed'].mean():.1f} ms (SD: {analysis_df['prp_bottleneck_computed'].std():.1f})

2. SOA Slope: Linear trend in T2 RT across SOA bins
   - Mean: {analysis_df['soa_slope'].mean():.1f} ms/step

3. Dual-Task Variability (CV, IQR)
   - Mean CV: {analysis_df['t2_cv_overall'].mean():.3f}
   - Mean IQR: {analysis_df['t2_iqr_overall'].mean():.1f} ms

4. Error Cascades
   - Mean cascade rate: {analysis_df['cascade_rate'].mean():.3f}
   - Mean cascade inflation: {analysis_df['cascade_inflation'].mean():.3f}

5. Post-Error Adjustments
   - T1 PES: {analysis_df['t1_pes'].mean():.1f} ms
   - T2 RT after T1 error: {analysis_df['t2_pes_t1_error'].mean():.1f} ms

6. Temporal Dynamics
   - RT drift (late - early): {analysis_df['t2_rt_drift'].mean():.1f} ms

Statistical Analysis:
- ALL models controlled for DASS depression, anxiety, stress, and age
- Tested {len(dvs)} dependent variables
- Significant UCLA main effects (p < 0.05): {len(sig_effects)} / {len(results_df[results_df['model']=='UCLA_main'])}
- Significant UCLA × Gender interactions (p < 0.10): {len(interaction_effects)}

Files Generated:
1. prp_metrics_master.csv - All computed metrics merged with master dataset
2. prp_dass_controlled_regression_results.csv - Regression results for all DVs

Next Steps:
- If interactions found: Simple slopes analysis at ±1 SD UCLA
- If variability effects: Ex-Gaussian decomposition (μ, σ, τ)
- If error cascades: Trial-level mixed-effects models
- Cross-task integration: Compare PRP patterns with WCST gender effects
"""

with open(OUTPUT_DIR / "PRP_COMPREHENSIVE_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("\n" + "=" * 80)
print("✅ PRP COMPREHENSIVE ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nKey files:")
print("  - prp_metrics_master.csv")
print("  - prp_dass_controlled_regression_results.csv")
print("  - PRP_COMPREHENSIVE_SUMMARY.txt")
print("\n" + "=" * 80)
