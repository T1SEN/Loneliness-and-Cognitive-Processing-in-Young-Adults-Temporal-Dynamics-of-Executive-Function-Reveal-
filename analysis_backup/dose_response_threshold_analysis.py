"""
Dose-Response Threshold Analysis
==================================
Tests linearity vs threshold effects in UCLA → WCST PE relationship

Analyses:
1. Spline regression with 3-5 knots to identify inflection points
2. Piecewise regression testing candidate thresholds (35, 40, 45, 50)
3. ROC analysis to find optimal UCLA cutoff predicting high PE (>75th percentile)
4. Model comparison: Linear vs quadratic vs threshold (AIC/BIC)

Goal: Identify clinical cutoff values for UCLA loneliness scores
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings

from data_loader_utils import ensure_participant_id, load_participants, normalize_gender_series
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/threshold_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

THRESHOLD_CANDIDATES = [35, 40, 45, 50]

print("=" * 80)
print("DOSE-RESPONSE THRESHOLD ANALYSIS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading data...")

# Participant info and surveys
participants = load_participants()

surveys = ensure_participant_id(pd.read_csv(RESULTS_DIR / "2_surveys_results.csv"))

# UCLA scores
if 'surveyName' in surveys.columns:
    ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
elif 'survey' in surveys.columns:
    ucla_data = surveys[surveys['survey'].str.lower() == 'ucla'].copy()
else:
    raise KeyError("No survey name column found")
ucla_scores = ucla_data.groupby('participant_id')['score'].sum().reset_index()
ucla_scores.columns = ['participant_id', 'ucla_total']

# Merge with demographics
master = participants[['participant_id', 'age', 'gender', 'education']].merge(
    ucla_scores, on='participant_id', how='inner'
)
master['age'] = pd.to_numeric(master['age'], errors='coerce')

# WCST data for PE rate
wcst_trials = ensure_participant_id(pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv"))

import ast
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_dict'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

wcst_summary = wcst_trials.groupby('participant_id').agg(
    pe_count=('is_pe', 'sum'),
    total_trials=('is_pe', 'count'),
    wcst_accuracy=('correct', lambda x: (x.sum() / len(x)) * 100)
).reset_index()
wcst_summary['pe_rate'] = (wcst_summary['pe_count'] / wcst_summary['total_trials']) * 100

# Merge WCST
master = master.merge(wcst_summary[['participant_id', 'pe_rate', 'wcst_accuracy']],
                      on='participant_id', how='left')

# Create gender dummy
master['gender'] = normalize_gender_series(master['gender'])
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Drop missing
master = master.dropna(subset=['ucla_total', 'gender_male', 'pe_rate']).copy()

print(f"  Loaded {len(master)} participants with complete data")
print(f"  Males: {master['gender_male'].sum()}, Females: {(1-master['gender_male']).sum()}\n")

# ============================================================================
# Analysis by Gender
# ============================================================================

all_results = []

for gender_name, gender_val in [('Male', 1), ('Female', 0), ('All', None)]:
    if gender_val is None:
        df_sub = master.copy()
    else:
        df_sub = master[master['gender_male'] == gender_val].copy()

    if len(df_sub) < 20:
        print(f"\nSkipping {gender_name}: insufficient data (N={len(df_sub)})")
        continue

    print("\n" + "=" * 80)
    print(f"{gender_name.upper()} (N={len(df_sub)})")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Baseline Linear Model
    # ------------------------------------------------------------------------
    print("\n1. LINEAR MODEL")
    print("   " + "-" * 60)

    df_sub['z_ucla'] = (df_sub['ucla_total'] - df_sub['ucla_total'].mean()) / df_sub['ucla_total'].std()
    df_sub['z_age'] = (df_sub['age'] - df_sub['age'].mean()) / df_sub['age'].std()

    linear_model = smf.ols("pe_rate ~ z_ucla + z_age", data=df_sub).fit()
    print(f"   R² = {linear_model.rsquared:.4f}, AIC = {linear_model.aic:.2f}, BIC = {linear_model.bic:.2f}")
    print(f"   β_UCLA = {linear_model.params['z_ucla']:.3f}, p = {linear_model.pvalues['z_ucla']:.4f}")

    all_results.append({
        'gender': gender_name,
        'model': 'Linear',
        'n': len(df_sub),
        'r_squared': linear_model.rsquared,
        'aic': linear_model.aic,
        'bic': linear_model.bic,
        'ucla_coef': linear_model.params['z_ucla'],
        'ucla_p': linear_model.pvalues['z_ucla']
    })

    # ------------------------------------------------------------------------
    # 2. Quadratic Model
    # ------------------------------------------------------------------------
    print("\n2. QUADRATIC MODEL")
    print("   " + "-" * 60)

    df_sub['z_ucla_sq'] = df_sub['z_ucla'] ** 2
    quad_model = smf.ols("pe_rate ~ z_ucla + z_ucla_sq + z_age", data=df_sub).fit()
    print(f"   R² = {quad_model.rsquared:.4f}, AIC = {quad_model.aic:.2f}, BIC = {quad_model.bic:.2f}")
    print(f"   β_UCLA² = {quad_model.params.get('z_ucla_sq', np.nan):.3f}, p = {quad_model.pvalues.get('z_ucla_sq', np.nan):.4f}")

    all_results.append({
        'gender': gender_name,
        'model': 'Quadratic',
        'n': len(df_sub),
        'r_squared': quad_model.rsquared,
        'aic': quad_model.aic,
        'bic': quad_model.bic,
        'ucla_coef': quad_model.params['z_ucla'],
        'ucla_p': quad_model.pvalues['z_ucla']
    })

    # ------------------------------------------------------------------------
    # 3. Piecewise Regression (Candidate Thresholds)
    # ------------------------------------------------------------------------
    print("\n3. PIECEWISE REGRESSION")
    print("   " + "-" * 60)

    best_threshold = None
    best_aic = np.inf

    for threshold in THRESHOLD_CANDIDATES:
        # Convert raw threshold to z-score for consistent comparison
        z_threshold = (threshold - df_sub['ucla_total'].mean()) / df_sub['ucla_total'].std()
        df_sub[f'ucla_above_{threshold}'] = (df_sub['z_ucla'] > z_threshold).astype(int)
        df_sub[f'ucla_x_above_{threshold}'] = df_sub['z_ucla'] * df_sub[f'ucla_above_{threshold}']

        try:
            formula = f"pe_rate ~ z_ucla + ucla_above_{threshold} + ucla_x_above_{threshold} + z_age"
            piecewise_model = smf.ols(formula, data=df_sub).fit()

            print(f"\n   Threshold = {threshold}:")
            print(f"      R² = {piecewise_model.rsquared:.4f}, AIC = {piecewise_model.aic:.2f}, BIC = {piecewise_model.bic:.2f}")

            # Slopes below and above threshold
            slope_below = piecewise_model.params['z_ucla']
            slope_above = slope_below + piecewise_model.params.get(f'ucla_x_above_{threshold}', 0)
            print(f"      Slope below {threshold}: {slope_below:.3f}")
            print(f"      Slope above {threshold}: {slope_above:.3f}")

            all_results.append({
                'gender': gender_name,
                'model': f'Piecewise_{threshold}',
                'n': len(df_sub),
                'r_squared': piecewise_model.rsquared,
                'aic': piecewise_model.aic,
                'bic': piecewise_model.bic,
                'threshold': threshold,
                'slope_below': slope_below,
                'slope_above': slope_above
            })

            if piecewise_model.aic < best_aic:
                best_aic = piecewise_model.aic
                best_threshold = threshold

        except Exception as e:
            print(f"   Threshold = {threshold}: Error - {e}")

    if best_threshold:
        print(f"\n   Best threshold (lowest AIC): {best_threshold}")

    # ------------------------------------------------------------------------
    # 4. ROC Analysis (Optimal Cutoff)
    # ------------------------------------------------------------------------
    print("\n4. ROC ANALYSIS (Predicting High PE)")
    print("   " + "-" * 60)

    # Define high PE (>75th percentile)
    pe_75 = df_sub['pe_rate'].quantile(0.75)
    df_sub['high_pe'] = (df_sub['pe_rate'] > pe_75).astype(int)

    if df_sub['high_pe'].sum() >= 5:
        fpr, tpr, thresholds = roc_curve(df_sub['high_pe'], df_sub['ucla_total'])
        roc_auc_val = auc(fpr, tpr)

        # Youden index (optimal cutoff)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_cutoff = thresholds[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]

        print(f"   High PE threshold: >{pe_75:.1f}%")
        print(f"   ROC AUC: {roc_auc_val:.3f}")
        print(f"   Optimal UCLA cutoff (Youden): {optimal_cutoff:.1f}")
        print(f"      Sensitivity: {optimal_sensitivity:.2%}")
        print(f"      Specificity: {optimal_specificity:.2%}")

        all_results.append({
            'gender': gender_name,
            'model': 'ROC_optimal',
            'n': len(df_sub),
            'high_pe_threshold': pe_75,
            'roc_auc': roc_auc_val,
            'optimal_ucla_cutoff': optimal_cutoff,
            'sensitivity': optimal_sensitivity,
            'specificity': optimal_specificity
        })

    else:
        print(f"   Insufficient high PE cases (N={df_sub['high_pe'].sum()})")

    # ------------------------------------------------------------------------
    # 5. Model Comparison
    # ------------------------------------------------------------------------
    print("\n5. MODEL COMPARISON")
    print("   " + "-" * 60)

    comparison = [
        ('Linear', linear_model.aic, linear_model.bic, linear_model.rsquared),
        ('Quadratic', quad_model.aic, quad_model.bic, quad_model.rsquared)
    ]

    if best_threshold:
        formula_best = f"pe_rate ~ z_ucla + ucla_above_{best_threshold} + ucla_x_above_{best_threshold} + z_age"
        best_piecewise = smf.ols(formula_best, data=df_sub).fit()
        comparison.append((f'Piecewise_{best_threshold}', best_piecewise.aic,
                           best_piecewise.bic, best_piecewise.rsquared))

    comparison_df = pd.DataFrame(comparison, columns=['Model', 'AIC', 'BIC', 'R²'])
    comparison_df = comparison_df.sort_values('AIC')

    print(comparison_df.to_string(index=False))

    best_model = comparison_df.iloc[0]['Model']
    print(f"\n   Best model (lowest AIC): {best_model}")

# ============================================================================
# Save Results
# ============================================================================

print("\n\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Save detailed results
detail_file = OUTPUT_DIR / "threshold_analysis_detailed.csv"
results_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
print(f"\nDetailed results: {detail_file}")

# Summary by gender
summary_rows = []
for gender in ['Male', 'Female', 'All']:
    gender_data = results_df[results_df['gender'] == gender]

    if len(gender_data) == 0:
        continue

    # Best model
    linear = gender_data[gender_data['model'] == 'Linear']
    quad = gender_data[gender_data['model'] == 'Quadratic']
    roc = gender_data[gender_data['model'] == 'ROC_optimal']

    best_aic_row = gender_data.dropna(subset=['aic']).sort_values('aic').iloc[0] if len(gender_data.dropna(subset=['aic'])) > 0 else None

    summary_rows.append({
        'gender': gender,
        'n': linear.iloc[0]['n'] if len(linear) > 0 else np.nan,
        'linear_r2': linear.iloc[0]['r_squared'] if len(linear) > 0 else np.nan,
        'linear_aic': linear.iloc[0]['aic'] if len(linear) > 0 else np.nan,
        'quad_r2': quad.iloc[0]['r_squared'] if len(quad) > 0 else np.nan,
        'quad_aic': quad.iloc[0]['aic'] if len(quad) > 0 else np.nan,
        'best_model': best_aic_row['model'] if best_aic_row is not None else np.nan,
        'best_aic': best_aic_row['aic'] if best_aic_row is not None else np.nan,
        'roc_optimal_cutoff': roc.iloc[0]['optimal_ucla_cutoff'] if len(roc) > 0 else np.nan,
        'roc_auc': roc.iloc[0]['roc_auc'] if len(roc) > 0 else np.nan
    })

summary_df = pd.DataFrame(summary_rows)

summary_file = OUTPUT_DIR / "threshold_analysis_summary.csv"
summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f"Summary: {summary_file}")

# ============================================================================
# Generate Text Report
# ============================================================================

male_data = results_df[results_df['gender'] == 'Male']
male_roc = male_data[male_data['model'] == 'ROC_optimal']
male_best = male_data.dropna(subset=['aic']).sort_values('aic').iloc[0] if len(male_data.dropna(subset=['aic'])) > 0 else None

male_sample = male_data.iloc[0]['n'] if len(male_data) > 0 else 'N/A'
linear_row = male_data[male_data['model'] == 'Linear']
quad_row = male_data[male_data['model'] == 'Quadratic']
male_linear_r2 = f"{linear_row.iloc[0]['r_squared']:.3f}" if not linear_row.empty else 'N/A'
male_linear_aic = f"{linear_row.iloc[0]['aic']:.1f}" if not linear_row.empty else 'N/A'
male_quadratic_r2 = f"{quad_row.iloc[0]['r_squared']:.3f}" if not quad_row.empty else 'N/A'
male_quadratic_aic = f"{quad_row.iloc[0]['aic']:.1f}" if not quad_row.empty else 'N/A'
male_best_model = male_best['model'] if male_best is not None else 'N/A'
male_best_aic = f"{male_best['aic']:.1f}" if male_best is not None else 'N/A'
male_cutoff = f"{male_roc.iloc[0]['optimal_ucla_cutoff']:.1f}" if len(male_roc) > 0 else 'N/A'
male_pe_threshold = f"{male_roc.iloc[0]['high_pe_threshold']:.1f}" if len(male_roc) > 0 else 'N/A'
male_sensitivity = f"{male_roc.iloc[0]['sensitivity']:.1%}" if len(male_roc) > 0 else 'N/A'
male_specificity = f"{male_roc.iloc[0]['specificity']:.1%}" if len(male_roc) > 0 else 'N/A'
male_auc = f"{male_roc.iloc[0]['roc_auc']:.3f}" if len(male_roc) > 0 else 'N/A'
interpretation = (
    "Linear relationship appears adequate"
    if male_best and 'Linear' in male_best['model']
    else "Non-linear relationship detected"
)
clinical_cutoff_text = (
    f"Males with UCLA > {male_cutoff} are at elevated risk for high perseverative errors"
    if male_cutoff != 'N/A'
    else "Insufficient data to define a UCLA cutoff for males"
)

report_text = f"""
DOSE-RESPONSE THRESHOLD ANALYSIS REPORT
========================================

MALE FINDINGS (Primary Interest):
----------------------------------
Sample: N = {male_sample}

Model Comparison:
- Linear:     R² = {male_linear_r2}, AIC = {male_linear_aic}
- Quadratic:  R² = {male_quadratic_r2}, AIC = {male_quadratic_aic}

Best Model: {male_best_model} (AIC = {male_best_aic})

Clinical Cutoff (ROC Analysis):
- Optimal UCLA threshold: {male_cutoff}
- Predicting high PE (> {male_pe_threshold}%)
- Sensitivity: {male_sensitivity}
- Specificity: {male_specificity}
- ROC AUC: {male_auc}

INTERPRETATION:
---------------
{interpretation}

Clinical implication:
- {clinical_cutoff_text}
- This cutoff may be useful for screening/intervention targeting

"""

report_file = OUTPUT_DIR / "THRESHOLD_ANALYSIS_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"Report: {report_file}")

print("\n" + "=" * 80)
print("DOSE-RESPONSE THRESHOLD ANALYSIS COMPLETE")
print("=" * 80)
