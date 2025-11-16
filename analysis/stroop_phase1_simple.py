"""
Stroop Phase 1 Simple Analysis
===============================
Quick analysis of block fatigue, timeout patterns, and ratio interference.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_phase1_simple")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STROOP PHASE 1 SIMPLE ANALYSIS")
print("="*80)

# Load master dataset and participants for gender
try:
    master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
    participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')

    # Normalize participant_id
    if 'participantId' in participants.columns:
        participants = participants.rename(columns={'participantId': 'participant_id'})

    # Merge gender into master
    master = master.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')

    print(f"\nMaster dataset loaded: {len(master)} rows")
except Exception as e:
    print(f"\n⚠ Error loading master: {e}")
    master = None

# Load trials
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')

# Filter valid
stroop_valid = stroop_trials[
    (stroop_trials['cond'].notna()) &
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 0) &
    (stroop_trials['rt_ms'] < 5000)
].copy()

print(f"Valid trials: {len(stroop_valid)}")

# ==== PHASE 1-2: BLOCK FATIGUE (QUARTILE ANALYSIS) ====
print("\n" + "="*80)
print("PHASE 1-2: BLOCK FATIGUE (Quartile Analysis)")
print("="*80)

stroop_valid = stroop_valid.sort_values(['participant_id', 'trial_index'])

# Create quartiles
quartile_data = []

for pid in stroop_valid['participant_id'].unique():
    pid_data = stroop_valid[stroop_valid['participant_id'] == pid]

    # Divide into 4 quartiles
    n_trials = len(pid_data)
    quartile_size = n_trials // 4

    for q in range(4):
        start_idx = q * quartile_size
        end_idx = (q + 1) * quartile_size if q < 3 else n_trials
        q_data = pid_data.iloc[start_idx:end_idx]

        cong_rt = q_data[q_data['cond'] == 'congruent']['rt_ms'].mean()
        incong_rt = q_data[q_data['cond'] == 'incongruent']['rt_ms'].mean()

        if pd.notna(cong_rt) and pd.notna(incong_rt):
            quartile_data.append({
                'participant_id': pid,
                'quartile': q + 1,  # 1-4
                'interference': incong_rt - cong_rt
            })

quartile_df = pd.DataFrame(quartile_data)
print(f"Quartile data: {len(quartile_df)} rows")

# Merge with master if available
ucla_col = 'ucla_total' if (master is not None and 'ucla_total' in master.columns) else ('ucla_score' if (master is not None and 'ucla_score' in master.columns) else None)

if ucla_col is not None:
    quartile_df = quartile_df.merge(master[['participant_id', ucla_col, 'gender']], on='participant_id', how='left')
    quartile_df = quartile_df.rename(columns={ucla_col: 'ucla_score'})

    # Test: does interference increase over quartiles × UCLA × gender?
    quartile_df['z_ucla'] = (quartile_df['ucla_score'] - quartile_df['ucla_score'].mean()) / quartile_df['ucla_score'].std()

    formula = 'interference ~ quartile * z_ucla * C(gender)'
    try:
        model = smf.ols(formula, data=quartile_df.dropna()).fit()
        print("\nQuartile × UCLA × Gender Model:")
        print(f"  Quartile × UCLA: p={model.pvalues.get('quartile:z_ucla', np.nan):.3f}")
        print(f"  Quartile × Gender: p={model.pvalues.get('quartile:C(gender)[T.여성]', np.nan):.3f}")

        # Save
        pd.DataFrame({
            'term': model.params.index,
            'coef': model.params.values,
            'p': model.pvalues.values
        }).to_csv(OUTPUT_DIR / "quartile_model.csv", index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"⚠ Quartile model failed: {e}")

# ==== PHASE 1-3: TIMEOUT/SLOW TRIALS ====
print("\n" + "="*80)
print("PHASE 1-3: TIMEOUT/SLOW TRIALS")
print("="*80)

# Slow threshold: 2000ms
slow_thresh = 2000

timeout_data = []

for pid in stroop_valid['participant_id'].unique():
    pid_data = stroop_valid[stroop_valid['participant_id'] == pid]

    for cond in ['congruent', 'incongruent']:
        cond_data = pid_data[pid_data['cond'] == cond]
        n_total = len(cond_data)
        n_slow = (cond_data['rt_ms'] > slow_thresh).sum()

        if n_total > 0:
            timeout_data.append({
                'participant_id': pid,
                'condition': cond,
                'slow_rate': n_slow / n_total * 100
            })

timeout_df = pd.DataFrame(timeout_data)
timeout_pivot = timeout_df.pivot_table(index='participant_id', columns='condition', values='slow_rate').reset_index()
timeout_pivot.columns.name = None

print(f"Timeout data: {len(timeout_pivot)} rows")

if ucla_col is not None:
    timeout_pivot = timeout_pivot.merge(master[['participant_id', ucla_col, 'gender']], on='participant_id', how='left')
    timeout_pivot = timeout_pivot.rename(columns={ucla_col: 'ucla_score'})
    timeout_pivot['z_ucla'] = (timeout_pivot['ucla_score'] - timeout_pivot['ucla_score'].mean()) / timeout_pivot['ucla_score'].std()

    # Test incongruent slow rate
    formula = 'incongruent ~ z_ucla * C(gender)'
    try:
        model = smf.ols(formula, data=timeout_pivot.dropna()).fit()
        print("\nIncongruent Slow Rate Model:")
        print(f"  UCLA: p={model.pvalues.get('z_ucla', np.nan):.3f}")
        print(f"  UCLA × Gender: p={model.pvalues.get('z_ucla:C(gender)[T.여성]', np.nan):.3f}")

        pd.DataFrame({
            'term': model.params.index,
            'coef': model.params.values,
            'p': model.pvalues.values
        }).to_csv(OUTPUT_DIR / "timeout_model.csv", index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"⚠ Timeout model failed: {e}")

# ==== PHASE 1-4: RATIO INTERFERENCE ====
print("\n" + "="*80)
print("PHASE 1-4: RATIO INTERFERENCE")
print("="*80)

# Calculate RTs
cond_rts = stroop_valid.groupby(['participant_id', 'cond'])['rt_ms'].mean().unstack().reset_index()
cond_rts['ratio'] = cond_rts['incongruent'] / cond_rts['congruent']
cond_rts['log_ratio'] = np.log(cond_rts['ratio'])

print(f"Ratio data: {len(cond_rts)} rows")
print(f"Mean ratio: {cond_rts['ratio'].mean():.3f}")

if ucla_col is not None:
    cond_rts = cond_rts.merge(master[['participant_id', ucla_col, 'gender']], on='participant_id', how='left')
    cond_rts = cond_rts.rename(columns={ucla_col: 'ucla_score'})
    cond_rts['z_ucla'] = (cond_rts['ucla_score'] - cond_rts['ucla_score'].mean()) / cond_rts['ucla_score'].std()

    formula = 'ratio ~ z_ucla * C(gender)'
    try:
        model = smf.ols(formula, data=cond_rts.dropna()).fit()
        print("\nRatio Interference Model:")
        print(f"  UCLA: p={model.pvalues.get('z_ucla', np.nan):.3f}")
        print(f"  UCLA × Gender: p={model.pvalues.get('z_ucla:C(gender)[T.여성]', np.nan):.3f}")

        pd.DataFrame({
            'term': model.params.index,
            'coef': model.params.values,
            'p': model.pvalues.values
        }).to_csv(OUTPUT_DIR / "ratio_model.csv", index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"⚠ Ratio model failed: {e}")

print("\n✓ Phase 1 complete!")
print(f"✓ Results saved to: {OUTPUT_DIR}")
