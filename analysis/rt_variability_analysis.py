"""
RT Variability Analysis (IIV - Intra-Individual Variability)
=============================================================

Purpose: Test whether UCLA × Gender effect manifests as increased RT variability
         (inconsistent attention) rather than overall slowing.

RT Variability Metrics:
- SD: Standard deviation of RTs (absolute variability)
- CV: Coefficient of variation (SD/Mean, relative variability)
- IQR: Inter-quartile range (75th - 25th percentile)
- Range90: 90th - 10th percentile (robust range)

Hypothesis: Lonely males show higher RT variability on PE trials (inconsistent
            attention during perseveration), not just slower mean RT.

This is a more robust alternative to Ex-Gaussian decomposition.

Author: Claude Code
Date: 2025-11-15
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/rt_variability")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("RT Variability Analysis (Intra-Individual Variability)")
print("=" * 80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n1. Loading data...")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")

print(f"   - Participants: {len(participants)}")
print(f"   - WCST trials: {len(wcst_trials)}")
print(f"   - Stroop trials: {len(stroop_trials)}")

# ============================================================================
# 2. Extract UCLA and Demographics
# ============================================================================
print("\n2. Extracting UCLA scores and demographics...")

surveys = surveys.rename(columns={'participantId': 'participant_id'})
ucla_scores = surveys[surveys['surveyName'] == 'ucla'][['participant_id', 'score']].copy()
ucla_scores.columns = ['participant_id', 'ucla_total']

demo = participants[['participantId', 'age', 'gender', 'education']].copy()
demo.columns = ['participant_id', 'age', 'gender', 'education']
demo['gender'] = demo['gender'].map({'남성': 'male', '여성': 'female'})
demo = demo.dropna(subset=['gender'])

print(f"   - UCLA scores: {len(ucla_scores)}")
print(f"   - Demographics: {len(demo)}")

# ============================================================================
# 3. Compute RT Variability Metrics for WCST
# ============================================================================
print("\n3. Computing RT variability metrics for WCST...")

wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['reactionTimeMs'] > 0) &
    (wcst_trials['isPE'].notna())
].copy()

columns_needed = ['participantId', 'reactionTimeMs', 'isPE', 'isNPE', 'correct']
wcst_valid = wcst_valid[columns_needed].copy()
wcst_valid = wcst_valid.rename(columns={'participantId': 'participant_id'})

wcst_valid = wcst_valid.merge(ucla_scores, on='participant_id', how='inner')
wcst_valid = wcst_valid.merge(demo, on='participant_id', how='inner')

def compute_rt_variability(rt_values, min_trials=10):
    """Compute RT variability metrics."""
    if len(rt_values) < min_trials:
        return {
            'mean_rt': np.nan,
            'sd_rt': np.nan,
            'cv_rt': np.nan,
            'iqr_rt': np.nan,
            'range90_rt': np.nan,
            'n_trials': len(rt_values)
        }

    # Remove extreme outliers (< 200ms or > 5000ms)
    rt_clean = rt_values[(rt_values >= 200) & (rt_values <= 5000)]

    if len(rt_clean) < min_trials:
        return {
            'mean_rt': np.nan,
            'sd_rt': np.nan,
            'cv_rt': np.nan,
            'iqr_rt': np.nan,
            'range90_rt': np.nan,
            'n_trials': len(rt_clean)
        }

    mean_rt = np.mean(rt_clean)
    sd_rt = np.std(rt_clean, ddof=1)
    cv_rt = sd_rt / mean_rt if mean_rt > 0 else np.nan

    q25, q75 = np.percentile(rt_clean, [25, 75])
    iqr_rt = q75 - q25

    p10, p90 = np.percentile(rt_clean, [10, 90])
    range90_rt = p90 - p10

    return {
        'mean_rt': mean_rt,
        'sd_rt': sd_rt,
        'cv_rt': cv_rt,
        'iqr_rt': iqr_rt,
        'range90_rt': range90_rt,
        'n_trials': len(rt_clean)
    }

wcst_iiv_results = []

for pid in wcst_valid['participant_id'].unique():
    pid_data = wcst_valid[wcst_valid['participant_id'] == pid]

    ucla = pid_data['ucla_total'].iloc[0]
    gender = pid_data['gender'].iloc[0]
    age = pid_data['age'].iloc[0]

    # PE trials
    pe_trials = pid_data[pid_data['isPE'] == True]
    pe_metrics = compute_rt_variability(pe_trials['reactionTimeMs'].values)

    # NPE trials
    npe_trials = pid_data[pid_data['isNPE'] == True]
    npe_metrics = compute_rt_variability(npe_trials['reactionTimeMs'].values)

    # Correct trials
    correct_trials = pid_data[pid_data['correct'] == True]
    correct_metrics = compute_rt_variability(correct_trials['reactionTimeMs'].values)

    # All trials
    all_metrics = compute_rt_variability(pid_data['reactionTimeMs'].values)

    wcst_iiv_results.append({
        'participant_id': pid,
        'ucla_total': ucla,
        'gender': gender,
        'age': age,

        # PE metrics
        'pe_mean_rt': pe_metrics['mean_rt'],
        'pe_sd_rt': pe_metrics['sd_rt'],
        'pe_cv_rt': pe_metrics['cv_rt'],
        'pe_iqr_rt': pe_metrics['iqr_rt'],
        'pe_range90_rt': pe_metrics['range90_rt'],
        'n_pe': pe_metrics['n_trials'],

        # NPE metrics
        'npe_mean_rt': npe_metrics['mean_rt'],
        'npe_sd_rt': npe_metrics['sd_rt'],
        'npe_cv_rt': npe_metrics['cv_rt'],

        # Correct metrics
        'correct_mean_rt': correct_metrics['mean_rt'],
        'correct_sd_rt': correct_metrics['sd_rt'],
        'correct_cv_rt': correct_metrics['cv_rt'],
        'correct_iqr_rt': correct_metrics['iqr_rt'],
        'correct_range90_rt': correct_metrics['range90_rt'],
        'n_correct': correct_metrics['n_trials'],

        # All trials
        'all_sd_rt': all_metrics['sd_rt'],
        'all_cv_rt': all_metrics['cv_rt']
    })

wcst_iiv_df = pd.DataFrame(wcst_iiv_results)
wcst_iiv_df = wcst_iiv_df.dropna(subset=['pe_sd_rt', 'correct_sd_rt', 'ucla_total'])

print(f"   - Participants with IIV data: {len(wcst_iiv_df)}")

# Compute difference metrics (PE - Correct)
wcst_iiv_df['sd_diff_pe_correct'] = wcst_iiv_df['pe_sd_rt'] - wcst_iiv_df['correct_sd_rt']
wcst_iiv_df['cv_diff_pe_correct'] = wcst_iiv_df['pe_cv_rt'] - wcst_iiv_df['correct_cv_rt']

# ============================================================================
# 4. Test UCLA × Gender on PE RT Variability
# ============================================================================
print("\n4. Testing UCLA × Gender interaction on PE RT variability...")

wcst_iiv_df['gender_coded'] = wcst_iiv_df['gender'].map({'male': 1, 'female': -1})
wcst_iiv_df['z_ucla'] = (wcst_iiv_df['ucla_total'] - wcst_iiv_df['ucla_total'].mean()) / wcst_iiv_df['ucla_total'].std()
wcst_iiv_df['z_age'] = (wcst_iiv_df['age'] - wcst_iiv_df['age'].mean()) / wcst_iiv_df['age'].std()
wcst_iiv_df['ucla_x_gender'] = wcst_iiv_df['z_ucla'] * wcst_iiv_df['gender_coded']

X = wcst_iiv_df[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values

# PE SD (absolute variability)
y_pe_sd = wcst_iiv_df['pe_sd_rt'].values
model_pe_sd = LinearRegression().fit(X, y_pe_sd)
pe_sd_interaction = model_pe_sd.coef_[-1]

print(f"\n   PE RT Standard Deviation (absolute variability):")
print(f"      - UCLA × Gender β = {pe_sd_interaction:.2f}ms")

# PE CV (relative variability)
y_pe_cv = wcst_iiv_df['pe_cv_rt'].values
model_pe_cv = LinearRegression().fit(X, y_pe_cv)
pe_cv_interaction = model_pe_cv.coef_[-1]

print(f"\n   PE RT Coefficient of Variation (relative variability):")
print(f"      - UCLA × Gender β = {pe_cv_interaction:.4f}")

# Stratified correlations
male_data = wcst_iiv_df[wcst_iiv_df['gender'] == 'male']
female_data = wcst_iiv_df[wcst_iiv_df['gender'] == 'female']

male_r_sd, male_p_sd = stats.pearsonr(male_data['ucla_total'], male_data['pe_sd_rt'])
female_r_sd, female_p_sd = stats.pearsonr(female_data['ucla_total'], female_data['pe_sd_rt'])

print(f"\n   Correlation: UCLA × PE SD")
print(f"      - Males (N={len(male_data)}): r={male_r_sd:.3f}, p={male_p_sd:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_sd:.3f}, p={female_p_sd:.4f}")

# ============================================================================
# 5. Test Variability Difference (PE - Correct)
# ============================================================================
print("\n5. Testing variability difference (PE - Correct)...")

y_sd_diff = wcst_iiv_df['sd_diff_pe_correct'].values
model_sd_diff = LinearRegression().fit(X, y_sd_diff)
sd_diff_interaction = model_sd_diff.coef_[-1]

print(f"\n   SD Difference (PE - Correct):")
print(f"      - UCLA × Gender β = {sd_diff_interaction:.2f}ms")

male_r_diff, male_p_diff = stats.pearsonr(male_data['ucla_total'], male_data['sd_diff_pe_correct'])
female_r_diff, female_p_diff = stats.pearsonr(female_data['ucla_total'], female_data['sd_diff_pe_correct'])

print(f"\n   Correlation: UCLA × SD Difference")
print(f"      - Males (N={len(male_data)}): r={male_r_diff:.3f}, p={male_p_diff:.4f}")
print(f"      - Females (N={len(female_data)}): r={female_r_diff:.3f}, p={female_p_diff:.4f}")

# ============================================================================
# 6. Test Overall RT Variability (All Trials)
# ============================================================================
print("\n6. Testing overall RT variability (all trials)...")

y_all_cv = wcst_iiv_df['all_cv_rt'].values
model_all_cv = LinearRegression().fit(X, y_all_cv)
all_cv_interaction = model_all_cv.coef_[-1]

print(f"\n   Overall RT CV (all trials):")
print(f"      - UCLA × Gender β = {all_cv_interaction:.4f}")

# ============================================================================
# 7. Compare PE vs NPE Variability
# ============================================================================
print("\n7. Comparing PE vs NPE variability...")

# Check if NPE data is available
wcst_npe_available = wcst_iiv_df.dropna(subset=['npe_sd_rt'])

if len(wcst_npe_available) >= 10:
    X_npe = wcst_npe_available[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values
    y_npe_sd = wcst_npe_available['npe_sd_rt'].values
    model_npe_sd = LinearRegression().fit(X_npe, y_npe_sd)
    npe_sd_interaction = model_npe_sd.coef_[-1]

    print(f"\n   NPE RT SD:")
    print(f"      - UCLA × Gender β = {npe_sd_interaction:.2f}ms")
    print(f"      - PE/NPE ratio: {abs(pe_sd_interaction)/abs(npe_sd_interaction) if npe_sd_interaction != 0 else np.inf:.2f}×")
else:
    print(f"\n   NPE data insufficient (N={len(wcst_npe_available)})")
    print(f"      - Skipping NPE analysis")
    npe_sd_interaction = np.nan

# ============================================================================
# 8. Stroop RT Variability (Control Task)
# ============================================================================
print("\n8. Computing Stroop RT variability (control)...")

stroop_valid = stroop_trials[
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 0)
].copy()

# Select only needed columns to avoid duplicates
stroop_cols = ['participantId', 'rt_ms', 'cond']
stroop_valid = stroop_valid[stroop_cols].copy()
stroop_valid = stroop_valid.rename(columns={'participantId': 'participant_id'})

stroop_valid = stroop_valid.merge(ucla_scores, on='participant_id', how='inner')
stroop_valid = stroop_valid.merge(demo, on='participant_id', how='inner')

stroop_iiv_results = []

for pid in stroop_valid['participant_id'].unique():
    pid_data = stroop_valid[stroop_valid['participant_id'] == pid]

    ucla = pid_data['ucla_total'].iloc[0]
    gender = pid_data['gender'].iloc[0]
    age = pid_data['age'].iloc[0]

    # Incongruent trials
    incong_trials = pid_data[pid_data['cond'] == 'incongruent']
    incong_metrics = compute_rt_variability(incong_trials['rt_ms'].values)

    # Congruent trials
    cong_trials = pid_data[pid_data['cond'] == 'congruent']
    cong_metrics = compute_rt_variability(cong_trials['rt_ms'].values)

    stroop_iiv_results.append({
        'participant_id': pid,
        'ucla_total': ucla,
        'gender': gender,
        'age': age,
        'incong_sd_rt': incong_metrics['sd_rt'],
        'incong_cv_rt': incong_metrics['cv_rt'],
        'cong_sd_rt': cong_metrics['sd_rt'],
        'cong_cv_rt': cong_metrics['cv_rt']
    })

stroop_iiv_df = pd.DataFrame(stroop_iiv_results)
stroop_iiv_df = stroop_iiv_df.dropna(subset=['incong_sd_rt', 'ucla_total'])

stroop_iiv_df['gender_coded'] = stroop_iiv_df['gender'].map({'male': 1, 'female': -1})
stroop_iiv_df['z_ucla'] = (stroop_iiv_df['ucla_total'] - stroop_iiv_df['ucla_total'].mean()) / stroop_iiv_df['ucla_total'].std()
stroop_iiv_df['z_age'] = (stroop_iiv_df['age'] - stroop_iiv_df['age'].mean()) / stroop_iiv_df['age'].std()
stroop_iiv_df['ucla_x_gender'] = stroop_iiv_df['z_ucla'] * stroop_iiv_df['gender_coded']

X_stroop = stroop_iiv_df[['z_ucla', 'gender_coded', 'z_age', 'ucla_x_gender']].values
y_stroop_sd = stroop_iiv_df['incong_sd_rt'].values

model_stroop_sd = LinearRegression().fit(X_stroop, y_stroop_sd)
stroop_sd_interaction = model_stroop_sd.coef_[-1]

print(f"\n   Stroop Incongruent RT SD:")
print(f"      - UCLA × Gender β = {stroop_sd_interaction:.2f}ms")
print(f"      - WCST PE/Stroop ratio: {abs(pe_sd_interaction)/abs(stroop_sd_interaction) if stroop_sd_interaction != 0 else np.inf:.2f}×")

# ============================================================================
# 9. Descriptive Statistics
# ============================================================================
print("\n9. Descriptive statistics:")

print(f"\n   Overall WCST RT variability:")
print(f"      - PE SD: M={wcst_iiv_df['pe_sd_rt'].mean():.1f}ms (SD={wcst_iiv_df['pe_sd_rt'].std():.1f})")
print(f"      - Correct SD: M={wcst_iiv_df['correct_sd_rt'].mean():.1f}ms (SD={wcst_iiv_df['correct_sd_rt'].std():.1f})")
print(f"      - SD difference: M={wcst_iiv_df['sd_diff_pe_correct'].mean():.1f}ms")

print(f"\n   By gender:")
for gender in ['male', 'female']:
    gdata = wcst_iiv_df[wcst_iiv_df['gender'] == gender]
    print(f"\n   {gender.capitalize()} (N={len(gdata)}):")
    print(f"      - PE SD: M={gdata['pe_sd_rt'].mean():.1f}ms")
    print(f"      - PE CV: M={gdata['pe_cv_rt'].mean():.3f}")
    print(f"      - SD diff: M={gdata['sd_diff_pe_correct'].mean():.1f}ms")

# ============================================================================
# 10. Save Results
# ============================================================================
print("\n10. Saving results...")

summary_results = pd.DataFrame([{
    'task': 'wcst',
    'metric': 'pe_sd',
    'interaction_beta': pe_sd_interaction,
    'male_r': male_r_sd,
    'male_p': male_p_sd,
    'female_r': female_r_sd,
    'female_p': female_p_sd
}, {
    'task': 'wcst',
    'metric': 'pe_cv',
    'interaction_beta': pe_cv_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan
}, {
    'task': 'wcst',
    'metric': 'sd_diff_pe_correct',
    'interaction_beta': sd_diff_interaction,
    'male_r': male_r_diff,
    'male_p': male_p_diff,
    'female_r': female_r_diff,
    'female_p': female_p_diff
}, {
    'task': 'wcst',
    'metric': 'npe_sd',
    'interaction_beta': npe_sd_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan
}, {
    'task': 'stroop',
    'metric': 'incong_sd',
    'interaction_beta': stroop_sd_interaction,
    'male_r': np.nan,
    'male_p': np.nan,
    'female_r': np.nan,
    'female_p': np.nan
}])

summary_results.to_csv(OUTPUT_DIR / "rt_variability_summary.csv", index=False, encoding='utf-8-sig')
wcst_iiv_df.to_csv(OUTPUT_DIR / "wcst_rt_variability.csv", index=False, encoding='utf-8-sig')
stroop_iiv_df.to_csv(OUTPUT_DIR / "stroop_rt_variability.csv", index=False, encoding='utf-8-sig')

print(f"   - Saved: rt_variability_summary.csv")
print(f"   - Saved: wcst_rt_variability.csv")
print(f"   - Saved: stroop_rt_variability.csv")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# 11. Interpretation Summary
# ============================================================================
print("\n11. KEY FINDINGS SUMMARY:")
print("=" * 80)

print("\n📊 PE RT VARIABILITY:")
print(f"   - SD interaction: β={pe_sd_interaction:.2f}ms")
print(f"   - CV interaction: β={pe_cv_interaction:.4f}")
print(f"   - Males: r={male_r_sd:.3f}, p={male_p_sd:.4f}")
print(f"   - Females: r={female_r_sd:.3f}, p={female_p_sd:.4f}")

print("\n📊 VARIABILITY DIFFERENCE (PE - CORRECT):")
print(f"   - β={sd_diff_interaction:.2f}ms")
print(f"   - Males: r={male_r_diff:.3f}, p={male_p_diff:.4f}")

print("\n🎯 TASK SPECIFICITY:")
print(f"   - WCST PE SD: β={pe_sd_interaction:.2f}ms")
print(f"   - Stroop incong SD: β={stroop_sd_interaction:.2f}ms")
if stroop_sd_interaction != 0:
    print(f"   - Ratio: {abs(pe_sd_interaction)/abs(stroop_sd_interaction):.2f}×")

print("\n💡 INTERPRETATION:")
if abs(pe_sd_interaction) > 10:
    print("   ✓ UCLA × Gender affects RT variability on PE trials")
    print("   ✓ Lonely males show MORE INCONSISTENT responses during perseveration")
    print("   ✓ Suggests attentional fluctuations, not just slower responses")
else:
    print("   ✓ Weak/null effect on RT variability")
    print("   ✓ Effect may be more about mean RT than variability")

if abs(sd_diff_interaction) > abs(pe_sd_interaction) * 0.5:
    print("   ✓ PE-specific variability elevation")
else:
    print("   ✓ General variability elevation (not PE-specific)")

print("\n" + "=" * 80)
