"""
Reliability-Corrected Effect Analysis
=====================================
Corrects UCLA → EF correlations for measurement error using:
1. Split-half reliability estimates
2. Spearman-Brown corrected reliability
3. Attenuation correction formula

Observed correlation (r_xy) underestimates true correlation due to measurement error.
True correlation: r_true = r_xy / sqrt(r_xx * r_yy)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX, PRP_RT_MAX
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "reliability_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_ucla_reliability(master_df):
    """
    Compute UCLA internal consistency (Cronbach's alpha).
    Uses item-level data if available.
    """
    # Check for item columns
    ucla_items = [f'ucla_{i}' for i in range(1, 21)]
    available_items = [c for c in ucla_items if c in master_df.columns]

    if len(available_items) < 10:
        print("Warning: UCLA items not available, using default reliability estimate")
        return 0.92  # Literature-based estimate

    item_data = master_df[available_items].dropna()

    if len(item_data) < 30:
        return 0.92

    # Cronbach's alpha
    n_items = len(available_items)
    item_vars = item_data.var()
    total_var = item_data.sum(axis=1).var()

    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)

    return alpha


def compute_ef_reliability_splithalf(task='stroop'):
    """
    Compute split-half reliability for EF measures.
    Splits trials into odd/even and correlates.
    """
    if task == 'stroop':
        df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
        df = ensure_participant_id(df)
        rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
        cond_col = 'type' if 'type' in df.columns else 'cond'

        df = df[df['timeout'] == False].copy()
        df = df[df['correct'] == True].copy()
        df = df[(df[rt_col] > DEFAULT_RT_MIN) & (df[rt_col] < STROOP_RT_MAX)].copy()

        # Compute interference for odd and even trials
        results = []
        for pid in df['participant_id'].unique():
            pdata = df[df['participant_id'] == pid].copy()
            pdata['trial_num'] = range(len(pdata))
            pdata['split'] = pdata['trial_num'] % 2  # 0=even, 1=odd

            for split in [0, 1]:
                split_data = pdata[pdata['split'] == split]
                cong = split_data[split_data[cond_col].str.lower() == 'congruent'][rt_col].mean()
                incong = split_data[split_data[cond_col].str.lower() == 'incongruent'][rt_col].mean()

                if not np.isnan(cong) and not np.isnan(incong):
                    results.append({
                        'participant_id': pid,
                        'split': split,
                        'interference': incong - cong
                    })

        results_df = pd.DataFrame(results)
        even = results_df[results_df['split'] == 0].set_index('participant_id')['interference']
        odd = results_df[results_df['split'] == 1].set_index('participant_id')['interference']

        common_pids = even.index.intersection(odd.index)
        if len(common_pids) < 30:
            return None

        r_split, _ = stats.pearsonr(even.loc[common_pids], odd.loc[common_pids])
        # Spearman-Brown correction
        r_sb = 2 * r_split / (1 + r_split)

        return {
            'task': 'stroop',
            'measure': 'interference',
            'r_split_half': r_split,
            'r_spearman_brown': r_sb,
            'n': len(common_pids)
        }

    elif task == 'wcst':
        df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
        df = ensure_participant_id(df)

        # Parse isPE
        if 'isPE' in df.columns:
            df['is_pe'] = df['isPE'].map({'True': True, 'False': False, True: True, False: False})
        else:
            return None

        results = []
        for pid in df['participant_id'].unique():
            pdata = df[df['participant_id'] == pid].sort_values('trialIndex').reset_index()
            pdata['trial_num'] = range(len(pdata))
            pdata['split'] = pdata['trial_num'] % 2

            for split in [0, 1]:
                split_data = pdata[pdata['split'] == split]
                pe_rate = split_data['is_pe'].mean() * 100 if len(split_data) > 0 else np.nan

                results.append({
                    'participant_id': pid,
                    'split': split,
                    'pe_rate': pe_rate
                })

        results_df = pd.DataFrame(results)
        even = results_df[results_df['split'] == 0].set_index('participant_id')['pe_rate']
        odd = results_df[results_df['split'] == 1].set_index('participant_id')['pe_rate']

        common_pids = even.index.intersection(odd.index)
        if len(common_pids) < 30:
            return None

        r_split, _ = stats.pearsonr(even.loc[common_pids].dropna(), odd.loc[common_pids].dropna())
        r_sb = 2 * r_split / (1 + r_split)

        return {
            'task': 'wcst',
            'measure': 'pe_rate',
            'r_split_half': r_split,
            'r_spearman_brown': r_sb,
            'n': len(common_pids)
        }

    elif task == 'prp':
        df = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8')
        df = ensure_participant_id(df)

        rt_col = 't2_rt_ms' if 't2_rt_ms' in df.columns else 't2_rt'
        soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in df.columns else 'soa'

        df = df[df['t2_timeout'] == False].copy()
        df = df[(df[rt_col] > DEFAULT_RT_MIN) & (df[rt_col] < PRP_RT_MAX)].copy()

        results = []
        for pid in df['participant_id'].unique():
            pdata = df[df['participant_id'] == pid].copy()
            pdata['trial_num'] = range(len(pdata))
            pdata['split'] = pdata['trial_num'] % 2

            for split in [0, 1]:
                split_data = pdata[pdata['split'] == split]
                short_rt = split_data[split_data[soa_col] <= 150][rt_col].mean()
                long_rt = split_data[split_data[soa_col] >= 1200][rt_col].mean()
                bottleneck = short_rt - long_rt if not np.isnan(short_rt) and not np.isnan(long_rt) else np.nan

                results.append({
                    'participant_id': pid,
                    'split': split,
                    'bottleneck': bottleneck
                })

        results_df = pd.DataFrame(results)
        even = results_df[results_df['split'] == 0].set_index('participant_id')['bottleneck']
        odd = results_df[results_df['split'] == 1].set_index('participant_id')['bottleneck']

        common_pids = even.dropna().index.intersection(odd.dropna().index)
        if len(common_pids) < 30:
            return None

        r_split, _ = stats.pearsonr(even.loc[common_pids], odd.loc[common_pids])
        r_sb = 2 * r_split / (1 + r_split)

        return {
            'task': 'prp',
            'measure': 'bottleneck',
            'r_split_half': r_split,
            'r_spearman_brown': r_sb,
            'n': len(common_pids)
        }

    return None


def correct_for_attenuation(r_observed, reliability_x, reliability_y):
    """
    Correct correlation for measurement error.
    r_true = r_observed / sqrt(reliability_x * reliability_y)
    """
    if reliability_x <= 0 or reliability_y <= 0:
        return np.nan

    r_true = r_observed / np.sqrt(reliability_x * reliability_y)

    # Cap at 1.0
    return min(abs(r_true), 1.0) * np.sign(r_observed)


def compute_corrected_effects(master_df, ucla_reliability, ef_reliabilities):
    """Compute observed and corrected UCLA-EF correlations."""
    results = []

    for task, reliability_info in ef_reliabilities.items():
        if reliability_info is None:
            continue

        measure = reliability_info['measure']
        ef_reliability = reliability_info['r_spearman_brown']

        if measure not in master_df.columns:
            continue

        # Observed correlation
        valid = master_df[['ucla_score', measure]].dropna()
        if len(valid) < 30:
            continue

        r_observed, p_observed = stats.pearsonr(valid['ucla_score'], valid[measure])

        # Corrected correlation
        r_corrected = correct_for_attenuation(r_observed, ucla_reliability, ef_reliability)

        # Confidence interval for corrected correlation (approximate)
        n = len(valid)
        se_r = np.sqrt((1 - r_observed**2) / (n - 2))
        ci_low_obs = r_observed - 1.96 * se_r
        ci_high_obs = r_observed + 1.96 * se_r

        # Scale CI for correction (rough approximation)
        correction_factor = 1 / np.sqrt(ucla_reliability * ef_reliability)
        ci_low_corr = ci_low_obs * correction_factor
        ci_high_corr = ci_high_obs * correction_factor

        results.append({
            'task': task,
            'ef_measure': measure,
            'n': n,
            'ucla_reliability': ucla_reliability,
            'ef_reliability': ef_reliability,
            'r_observed': r_observed,
            'p_observed': p_observed,
            'r_corrected': r_corrected,
            'ci_low_observed': ci_low_obs,
            'ci_high_observed': ci_high_obs,
            'ci_low_corrected': ci_low_corr,
            'ci_high_corrected': ci_high_corr,
            'correction_factor': correction_factor
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Reliability-Corrected Effect Analysis")
    print("=" * 60)

    # Load data
    master = load_master_dataset(use_cache=True)

    # 1. UCLA reliability
    print("\n[1] UCLA Loneliness Scale Reliability")
    print("-" * 40)
    ucla_rel = compute_ucla_reliability(master)
    print(f"  UCLA Cronbach's alpha: {ucla_rel:.3f}")

    # 2. EF reliabilities
    print("\n[2] EF Measure Split-Half Reliabilities")
    print("-" * 40)

    ef_reliabilities = {}
    for task in ['stroop', 'wcst', 'prp']:
        rel = compute_ef_reliability_splithalf(task)
        ef_reliabilities[task] = rel
        if rel:
            print(f"  {task.upper()} {rel['measure']}:")
            print(f"    Split-half r: {rel['r_split_half']:.3f}")
            print(f"    Spearman-Brown corrected: {rel['r_spearman_brown']:.3f}")
            print(f"    N: {rel['n']}")

    # Save reliability estimates
    rel_df = pd.DataFrame([r for r in ef_reliabilities.values() if r is not None])
    rel_df['ucla_alpha'] = ucla_rel
    rel_df.to_csv(OUTPUT_DIR / "reliability_estimates.csv", index=False, encoding='utf-8-sig')

    # 3. Corrected correlations
    print("\n[3] Attenuation-Corrected UCLA-EF Correlations")
    print("-" * 40)

    corrected = compute_corrected_effects(master, ucla_rel, ef_reliabilities)

    if len(corrected) > 0:
        print("\n" + "=" * 80)
        print(f"{'Task':<10} {'Measure':<20} {'r_obs':>8} {'p_obs':>8} {'r_corr':>8} {'Factor':>8}")
        print("=" * 80)
        for _, row in corrected.iterrows():
            print(f"{row['task']:<10} {row['ef_measure']:<20} {row['r_observed']:>8.3f} {row['p_observed']:>8.4f} {row['r_corrected']:>8.3f} {row['correction_factor']:>8.2f}")

        corrected.to_csv(OUTPUT_DIR / "corrected_correlations.csv", index=False, encoding='utf-8-sig')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Reliability-Corrected Effect Analysis:

Measurement Reliability:
- UCLA Loneliness: alpha = {ucla_rel:.3f}
""")
    for task, rel in ef_reliabilities.items():
        if rel:
            print(f"- {task.upper()} {rel['measure']}: r_SB = {rel['r_spearman_brown']:.3f}")

    print(f"""
Attenuation Correction Formula:
  r_true = r_observed / sqrt(reliability_X * reliability_Y)

Interpretation:
- Observed correlations underestimate true relationships
- Low reliability → larger correction needed
- Corrected r > |1.0| is capped (indicates very low observed r or reliability)
- Corrected effects represent disattenuated true population correlations
""")

    if len(corrected) > 0:
        print("\nCorrected UCLA-EF Correlations:")
        for _, row in corrected.iterrows():
            change = ((row['r_corrected'] / row['r_observed']) - 1) * 100 if row['r_observed'] != 0 else 0
            print(f"  - {row['task'].upper()}: r_obs={row['r_observed']:.3f} → r_corr={row['r_corrected']:.3f} ({change:+.0f}% change)")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
