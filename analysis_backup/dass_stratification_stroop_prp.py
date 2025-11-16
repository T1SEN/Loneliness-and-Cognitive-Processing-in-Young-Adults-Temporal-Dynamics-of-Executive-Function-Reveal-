"""
DASS Stratification for Stroop & PRP
=====================================
Tests anxiety buffering hypothesis in Stroop and PRP.

WCST finding: Low Anxiety showed strongest effects (p=0.008)
Tests if same pattern applies to Stroop and PRP.

Analyses:
1. Median split on DASS Anxiety
2. Median split on DASS Stress
3. Gender × UCLA effects in each stratum
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/dass_stratification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DASS STRATIFICATION - STROOP & PRP")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================

print("\n[1] Loading data...")

master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
master.columns = master.columns.str.lower()

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
participants.columns = participants.columns.str.lower()
if 'participantid' in participants.columns and 'participant_id' in participants.columns:
    participants = participants.drop(columns=['participantid'])
elif 'participantid' in participants.columns:
    participants.rename(columns={'participantid': 'participant_id'}, inplace=True)

master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')
gender_map = {'남성': 'male', '여성': 'female', 'male': 'male', 'female': 'female'}
master['gender'] = master['gender'].map(gender_map)

print(f"  Master: N={len(master)}")

# ============================================================================
# DASS Anxiety Stratification
# ============================================================================

print("\n[2] DASS Anxiety stratification...")

median_anxiety = master['dass_anxiety'].median()
print(f"  Median anxiety: {median_anxiety}")

low_anxiety = master[master['dass_anxiety'] <= median_anxiety].copy()
high_anxiety = master[master['dass_anxiety'] > median_anxiety].copy()

print(f"  Low anxiety: N={len(low_anxiety)}")
print(f"  High anxiety: N={len(high_anxiety)}")

anxiety_results = []

for stratum, data in [('Low Anxiety', low_anxiety), ('High Anxiety', high_anxiety)]:
    for gender in ['male', 'female']:
        gender_data = data[data['gender'] == gender]

        if len(gender_data) < 10:
            continue

        # Stroop
        valid_stroop = gender_data.dropna(subset=['ucla_total', 'stroop_interference'])
        if len(valid_stroop) >= 5:
            r_stroop, p_stroop = pearsonr(valid_stroop['ucla_total'], valid_stroop['stroop_interference'])
        else:
            r_stroop, p_stroop = np.nan, np.nan

        # PRP
        valid_prp = gender_data.dropna(subset=['ucla_total', 'prp_bottleneck'])
        if len(valid_prp) >= 5:
            r_prp, p_prp = pearsonr(valid_prp['ucla_total'], valid_prp['prp_bottleneck'])
        else:
            r_prp, p_prp = np.nan, np.nan

        anxiety_results.append({
            'stratum': stratum,
            'gender': gender,
            'n': len(gender_data),
            'stroop_r': r_stroop,
            'stroop_p': p_stroop,
            'prp_r': r_prp,
            'prp_p': p_prp
        })

anxiety_df = pd.DataFrame(anxiety_results)
print(anxiety_df)

anxiety_df.to_csv(OUTPUT_DIR / "anxiety_stratification.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# DASS Stress Stratification
# ============================================================================

print("\n[3] DASS Stress stratification...")

median_stress = master['dass_stress'].median()
print(f"  Median stress: {median_stress}")

low_stress = master[master['dass_stress'] <= median_stress].copy()
high_stress = master[master['dass_stress'] > median_stress].copy()

print(f"  Low stress: N={len(low_stress)}")
print(f"  High stress: N={len(high_stress)}")

stress_results = []

for stratum, data in [('Low Stress', low_stress), ('High Stress', high_stress)]:
    for gender in ['male', 'female']:
        gender_data = data[data['gender'] == gender]

        if len(gender_data) < 10:
            continue

        # Stroop
        valid_stroop = gender_data.dropna(subset=['ucla_total', 'stroop_interference'])
        if len(valid_stroop) >= 5:
            r_stroop, p_stroop = pearsonr(valid_stroop['ucla_total'], valid_stroop['stroop_interference'])
        else:
            r_stroop, p_stroop = np.nan, np.nan

        # PRP
        valid_prp = gender_data.dropna(subset=['ucla_total', 'prp_bottleneck'])
        if len(valid_prp) >= 5:
            r_prp, p_prp = pearsonr(valid_prp['ucla_total'], valid_prp['prp_bottleneck'])
        else:
            r_prp, p_prp = np.nan, np.nan

        stress_results.append({
            'stratum': stratum,
            'gender': gender,
            'n': len(gender_data),
            'stroop_r': r_stroop,
            'stroop_p': p_stroop,
            'prp_r': r_prp,
            'prp_p': p_prp
        })

stress_df = pd.DataFrame(stress_results)
print(stress_df)

stress_df.to_csv(OUTPUT_DIR / "stress_stratification.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Summary & Comparison to WCST
# ============================================================================

print("\n" + "="*80)
print("DASS STRATIFICATION - KEY FINDINGS")
print("="*80)

print(f"""
1. ANXIETY STRATIFICATION:

Stroop:
{anxiety_df[['stratum', 'gender', 'n', 'stroop_r', 'stroop_p']].to_string(index=False)}

PRP:
{anxiety_df[['stratum', 'gender', 'n', 'prp_r', 'prp_p']].to_string(index=False)}

2. STRESS STRATIFICATION:

Stroop:
{stress_df[['stratum', 'gender', 'n', 'stroop_r', 'stroop_p']].to_string(index=False)}

PRP:
{stress_df[['stratum', 'gender', 'n', 'prp_r', 'prp_p']].to_string(index=False)}

3. Comparison to WCST:
   - WCST: Low Anxiety p=0.008**, High Anxiety p=0.125 (ns)
   - Anxiety buffering: High anxiety protects against loneliness effects
   - Current analysis tests if same pattern holds for Stroop/PRP

4. Interpretation:
   - If Low Anxiety shows stronger effects → Same mechanism as WCST
   - If High Anxiety shows stronger effects → Different mechanism
   - If no difference → Anxiety-independent

5. Files Generated:
   - anxiety_stratification.csv
   - stress_stratification.csv
""")

print("\n✅ DASS stratification analysis complete!")
print("="*80)
