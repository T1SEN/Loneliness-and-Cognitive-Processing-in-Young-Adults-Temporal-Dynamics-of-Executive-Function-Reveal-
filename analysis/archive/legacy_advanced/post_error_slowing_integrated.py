"""
Post-Error Slowing (PES) Integrated Analysis
============================================
Analyzes post-error slowing across all three cognitive tasks
to assess behavioral adjustment and error monitoring.

Post-Error Slowing (PES):
- RT on trial N+1 after an error vs. RT after correct trials
- Positive PES indicates adaptive behavioral adjustment
- Reflects error monitoring and cognitive control

Analyses:
1. PES calculation per task (Stroop, WCST, PRP)
2. Cross-task meta-analysis of UCLA effects
3. DASS-controlled regression
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import ast

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX, PRP_RT_MAX
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "post_error_slowing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_pes_stroop():
    """Compute PES for Stroop task."""
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    df['rt'] = df[rt_col]
    df = df.reset_index(drop=True)

    # Filter timeouts and invalid RTs
    df = df[df['timeout'] == False].reset_index(drop=True)
    mask = (df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < STROOP_RT_MAX)
    df = df[mask].reset_index(drop=True)

    results = []
    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid].sort_values('trial').reset_index(drop=True)

        if len(pdata) < 10:
            continue

        # Create previous trial info
        pdata['prev_correct'] = pdata['correct'].shift(1)
        pdata['prev_rt'] = pdata['rt'].shift(1)
        pdata = pdata.dropna(subset=['prev_correct'])

        # Only look at current correct trials for clean RT measurement
        current_correct = pdata[pdata['correct'] == True]

        rt_after_error = current_correct[current_correct['prev_correct'] == False]['rt']
        rt_after_correct = current_correct[current_correct['prev_correct'] == True]['rt']

        if len(rt_after_error) >= 3 and len(rt_after_correct) >= 10:
            pes = rt_after_error.mean() - rt_after_correct.mean()
            pes_robust = rt_after_error.median() - rt_after_correct.median()

            results.append({
                'participant_id': pid,
                'task': 'stroop',
                'pes': pes,
                'pes_robust': pes_robust,
                'rt_after_error': rt_after_error.mean(),
                'rt_after_correct': rt_after_correct.mean(),
                'n_post_error': len(rt_after_error),
                'n_post_correct': len(rt_after_correct)
            })

    return pd.DataFrame(results)


def compute_pes_wcst():
    """Compute PES for WCST task."""
    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'reactionTimeMs'
    df = df.rename(columns={rt_col: 'rt'})

    # Filter invalid RTs
    df = df[(df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < 10000)].copy()

    results = []
    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid].sort_values('trialIndex').reset_index(drop=True)

        if len(pdata) < 10:
            continue

        pdata['prev_correct'] = pdata['correct'].shift(1)
        pdata = pdata.dropna(subset=['prev_correct'])

        # Current correct trials only
        current_correct = pdata[pdata['correct'] == True]

        rt_after_error = current_correct[current_correct['prev_correct'] == False]['rt']
        rt_after_correct = current_correct[current_correct['prev_correct'] == True]['rt']

        if len(rt_after_error) >= 3 and len(rt_after_correct) >= 10:
            pes = rt_after_error.mean() - rt_after_correct.mean()
            pes_robust = rt_after_error.median() - rt_after_correct.median()

            results.append({
                'participant_id': pid,
                'task': 'wcst',
                'pes': pes,
                'pes_robust': pes_robust,
                'rt_after_error': rt_after_error.mean(),
                'rt_after_correct': rt_after_correct.mean(),
                'n_post_error': len(rt_after_error),
                'n_post_correct': len(rt_after_correct)
            })

    return pd.DataFrame(results)


def compute_pes_prp():
    """Compute PES for PRP task (based on T2 errors)."""
    df = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    rt_col = 't2_rt_ms' if 't2_rt_ms' in df.columns else 't2_rt'
    df = df.rename(columns={rt_col: 'rt'})

    # Filter invalid RTs
    df = df[(df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < PRP_RT_MAX)].copy()
    df = df[df['t2_timeout'] == False].copy()

    results = []
    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid].copy()

        # Use idx for trial order
        if 'idx' in pdata.columns:
            pdata = pdata.sort_values('idx').reset_index(drop=True)
        else:
            continue

        if len(pdata) < 10:
            continue

        pdata['prev_t2_correct'] = pdata['t2_correct'].shift(1)
        pdata = pdata.dropna(subset=['prev_t2_correct'])

        current_correct = pdata[pdata['t2_correct'] == True]

        rt_after_error = current_correct[current_correct['prev_t2_correct'] == False]['rt']
        rt_after_correct = current_correct[current_correct['prev_t2_correct'] == True]['rt']

        if len(rt_after_error) >= 3 and len(rt_after_correct) >= 10:
            pes = rt_after_error.mean() - rt_after_correct.mean()
            pes_robust = rt_after_error.median() - rt_after_correct.median()

            results.append({
                'participant_id': pid,
                'task': 'prp',
                'pes': pes,
                'pes_robust': pes_robust,
                'rt_after_error': rt_after_error.mean(),
                'rt_after_correct': rt_after_correct.mean(),
                'n_post_error': len(rt_after_error),
                'n_post_correct': len(rt_after_correct)
            })

    return pd.DataFrame(results)


def run_meta_analysis(effect_sizes, se_values):
    """Run fixed-effects meta-analysis."""
    weights = 1 / (se_values ** 2)
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    pooled_se = np.sqrt(1 / np.sum(weights))

    z = pooled_effect / pooled_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    # Heterogeneity (Q statistic)
    q = np.sum(weights * (effect_sizes - pooled_effect) ** 2)
    df = len(effect_sizes) - 1
    p_het = 1 - stats.chi2.cdf(q, df) if df > 0 else 1
    i_squared = max(0, (q - df) / q * 100) if q > 0 else 0

    return {
        'pooled_effect': pooled_effect,
        'pooled_se': pooled_se,
        'z': z,
        'p': p,
        'q': q,
        'i_squared': i_squared,
        'p_heterogeneity': p_het
    }


def analyze_ucla_pes(master_df, pes_df):
    """Analyze UCLA → PES relationship per task."""
    results = []

    for task in pes_df['task'].unique():
        task_pes = pes_df[pes_df['task'] == task].copy()
        merged = master_df.merge(task_pes, on='participant_id', how='inner')

        required = ['ucla_score', 'pes', 'dass_depression', 'dass_anxiety',
                    'dass_stress', 'age', 'gender_male']
        merged = merged.dropna(subset=required)

        if len(merged) < 30:
            continue

        for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
            merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

        formula = "pes ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        try:
            model = smf.ols(formula, data=merged).fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            import warnings
            warnings.warn(f"OLS fitting failed for {task}: {e}")
            continue

        results.append({
            'task': task,
            'n': len(merged),
            'mean_pes': merged['pes'].mean(),
            'std_pes': merged['pes'].std(),
            'ucla_coef': model.params.get('z_ucla_score', np.nan),
            'ucla_se': model.bse.get('z_ucla_score', np.nan),
            'ucla_p': model.pvalues.get('z_ucla_score', np.nan),
            'model_r2': model.rsquared
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Post-Error Slowing (PES) Integrated Analysis")
    print("=" * 60)

    # Compute PES for each task
    print("\n[1] Computing PES per task")
    print("-" * 40)

    stroop_pes = compute_pes_stroop()
    print(f"  Stroop: N={len(stroop_pes)}, Mean PES={stroop_pes['pes'].mean():.1f} ms")

    wcst_pes = compute_pes_wcst()
    print(f"  WCST: N={len(wcst_pes)}, Mean PES={wcst_pes['pes'].mean():.1f} ms")

    prp_pes = compute_pes_prp()
    print(f"  PRP: N={len(prp_pes)}, Mean PES={prp_pes['pes'].mean():.1f} ms")

    # Combine all PES data
    all_pes = pd.concat([stroop_pes, wcst_pes, prp_pes], ignore_index=True)
    all_pes.to_csv(OUTPUT_DIR / "pes_by_task.csv", index=False, encoding='utf-8-sig')

    # Test if PES > 0 for each task
    print("\n[2] PES significance tests (vs 0)")
    print("-" * 40)

    task_tests = []
    for task in ['stroop', 'wcst', 'prp']:
        task_data = all_pes[all_pes['task'] == task]['pes'].dropna()
        if len(task_data) < 10:
            continue

        t_stat, p_val = stats.ttest_1samp(task_data, 0)
        d = task_data.mean() / task_data.std() if task_data.std() > 0 else 0

        print(f"  {task.upper()}: M={task_data.mean():.1f}, t({len(task_data)-1})={t_stat:.2f}, p={p_val:.4f}, d={d:.2f}")

        task_tests.append({
            'task': task,
            'n': len(task_data),
            'mean': task_data.mean(),
            'std': task_data.std(),
            't': t_stat,
            'p': p_val,
            'cohens_d': d
        })

    pd.DataFrame(task_tests).to_csv(OUTPUT_DIR / "pes_significance_tests.csv", index=False, encoding='utf-8-sig')

    # UCLA → PES analysis
    print("\n[3] UCLA → PES (DASS-controlled)")
    print("-" * 40)

    master = load_master_dataset(use_cache=True)
    ucla_results = analyze_ucla_pes(master, all_pes)

    if len(ucla_results) > 0:
        for _, row in ucla_results.iterrows():
            sig = "*" if row['ucla_p'] < 0.05 else ""
            print(f"  {row['task'].upper()}: b={row['ucla_coef']:.3f}, p={row['ucla_p']:.4f}{sig}")

        ucla_results.to_csv(OUTPUT_DIR / "ucla_pes_regression.csv", index=False, encoding='utf-8-sig')

        # Meta-analysis across tasks
        print("\n[4] Meta-analysis of UCLA → PES effects")
        print("-" * 40)

        valid_effects = ucla_results.dropna(subset=['ucla_coef', 'ucla_se'])
        if len(valid_effects) >= 2:
            meta = run_meta_analysis(
                valid_effects['ucla_coef'].values,
                valid_effects['ucla_se'].values
            )
            print(f"  Pooled effect: b={meta['pooled_effect']:.4f} (SE={meta['pooled_se']:.4f})")
            print(f"  z={meta['z']:.3f}, p={meta['p']:.4f}")
            print(f"  Heterogeneity: I²={meta['i_squared']:.1f}%, p={meta['p_heterogeneity']:.4f}")

            pd.DataFrame([meta]).to_csv(OUTPUT_DIR / "meta_analysis_results.csv", index=False, encoding='utf-8-sig')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Post-Error Slowing Results:
- Stroop PES: {stroop_pes['pes'].mean():.1f} ms (N={len(stroop_pes)})
- WCST PES: {wcst_pes['pes'].mean():.1f} ms (N={len(wcst_pes)})
- PRP PES: {prp_pes['pes'].mean():.1f} ms (N={len(prp_pes)})

UCLA Effects (DASS-controlled):
""")
    if len(ucla_results) > 0:
        for _, row in ucla_results.iterrows():
            sig = "SIGNIFICANT" if row['ucla_p'] < 0.05 else "n.s."
            print(f"  - {row['task'].upper()}: b={row['ucla_coef']:.3f}, p={row['ucla_p']:.4f} ({sig})")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
