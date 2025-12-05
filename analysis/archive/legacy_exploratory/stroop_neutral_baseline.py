"""
Stroop Neutral Condition Analysis
=================================
Analyzes the neutral condition in the Stroop task to separate:
- Facilitation effect: Neutral RT - Congruent RT
- Interference effect: Incongruent RT - Neutral RT

Traditional Stroop effect (Incongruent - Congruent) confounds these two processes.
Neutral trials (non-color words or symbols) provide a baseline.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "stroop_neutral"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_stroop_by_condition():
    """Load Stroop trials and compute metrics by condition."""
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    cond_col = 'type' if 'type' in df.columns else 'cond'

    df['rt'] = df[rt_col]
    df['condition'] = df[cond_col].str.lower()
    df = df.reset_index(drop=True)

    # Check if neutral condition exists
    conditions = df['condition'].unique()
    print(f"Available conditions: {conditions}")

    has_neutral = 'neutral' in conditions

    # Filter valid trials
    df = df[df['timeout'] == False].reset_index(drop=True)
    df = df[df['correct'] == True].reset_index(drop=True)
    mask = (df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < STROOP_RT_MAX)
    df = df[mask].reset_index(drop=True)

    # Compute participant-level metrics
    results = []
    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid]

        cong_rt = pdata[pdata['condition'] == 'congruent']['rt'].mean()
        incong_rt = pdata[pdata['condition'] == 'incongruent']['rt'].mean()
        neutral_rt = pdata[pdata['condition'] == 'neutral']['rt'].mean() if has_neutral else np.nan

        cong_n = len(pdata[pdata['condition'] == 'congruent'])
        incong_n = len(pdata[pdata['condition'] == 'incongruent'])
        neutral_n = len(pdata[pdata['condition'] == 'neutral']) if has_neutral else 0

        if cong_n < 10 or incong_n < 10:
            continue

        # Traditional Stroop effect
        stroop_effect = incong_rt - cong_rt

        # Decomposed effects (if neutral available)
        if has_neutral and not np.isnan(neutral_rt) and neutral_n >= 10:
            facilitation = neutral_rt - cong_rt  # Speed-up for congruent
            interference = incong_rt - neutral_rt  # Slow-down for incongruent
        else:
            facilitation = np.nan
            interference = np.nan

        results.append({
            'participant_id': pid,
            'cong_rt': cong_rt,
            'incong_rt': incong_rt,
            'neutral_rt': neutral_rt,
            'stroop_effect': stroop_effect,
            'facilitation': facilitation,
            'interference': interference,
            'n_cong': cong_n,
            'n_incong': incong_n,
            'n_neutral': neutral_n
        })

    return pd.DataFrame(results), has_neutral


def analyze_decomposed_effects(stroop_df):
    """Analyze facilitation and interference at group level."""
    results = {}

    # Facilitation
    fac = stroop_df['facilitation'].dropna()
    if len(fac) >= 10:
        t_fac, p_fac = stats.ttest_1samp(fac, 0)
        results['facilitation'] = {
            'n': len(fac),
            'mean': fac.mean(),
            'std': fac.std(),
            't': t_fac,
            'p': p_fac,
            'cohens_d': fac.mean() / fac.std() if fac.std() > 0 else 0
        }

    # Interference
    intf = stroop_df['interference'].dropna()
    if len(intf) >= 10:
        t_int, p_int = stats.ttest_1samp(intf, 0)
        results['interference'] = {
            'n': len(intf),
            'mean': intf.mean(),
            'std': intf.std(),
            't': t_int,
            'p': p_int,
            'cohens_d': intf.mean() / intf.std() if intf.std() > 0 else 0
        }

    # Compare facilitation vs interference
    if len(fac) >= 10 and len(intf) >= 10:
        t_comp, p_comp = stats.ttest_rel(intf, fac)  # paired t-test
        results['comparison'] = {
            't': t_comp,
            'p': p_comp,
            'interference_larger': intf.mean() > fac.mean()
        }

    return results


def analyze_ucla_decomposed(master_df, stroop_df):
    """Test UCLA → facilitation/interference separately."""
    merged = master_df.merge(stroop_df, on='participant_id', how='inner')

    required = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'gender_male']
    merged = merged.dropna(subset=required)

    # Standardize
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

    results = []
    models = {}

    outcomes = {
        'stroop_effect': 'Traditional Stroop Effect',
        'facilitation': 'Facilitation (Neutral - Congruent)',
        'interference': 'Interference (Incongruent - Neutral)'
    }

    for outcome, label in outcomes.items():
        if outcome not in merged.columns:
            continue

        valid = merged.dropna(subset=[outcome])
        if len(valid) < 30:
            continue

        formula = f"{outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        try:
            model = smf.ols(formula, data=valid).fit()
            models[outcome] = model

            results.append({
                'outcome': outcome,
                'outcome_label': label,
                'n': len(valid),
                'ucla_coef': model.params.get('z_ucla_score', np.nan),
                'ucla_se': model.bse.get('z_ucla_score', np.nan),
                'ucla_p': model.pvalues.get('z_ucla_score', np.nan),
                'model_r2': model.rsquared
            })
        except:
            pass

    return pd.DataFrame(results), models


def main():
    print("=" * 60)
    print("Stroop Neutral Condition Analysis")
    print("=" * 60)

    # Load data
    stroop_df, has_neutral = load_stroop_by_condition()
    print(f"\nParticipants: N={len(stroop_df)}")
    print(f"Neutral condition available: {has_neutral}")

    stroop_df.to_csv(OUTPUT_DIR / "stroop_by_condition.csv", index=False, encoding='utf-8-sig')

    # Basic statistics
    print("\n[1] Condition RT means")
    print("-" * 40)
    print(f"  Congruent: {stroop_df['cong_rt'].mean():.1f} ms (SD={stroop_df['cong_rt'].std():.1f})")
    print(f"  Incongruent: {stroop_df['incong_rt'].mean():.1f} ms (SD={stroop_df['incong_rt'].std():.1f})")
    if has_neutral:
        valid_neutral = stroop_df['neutral_rt'].dropna()
        if len(valid_neutral) > 0:
            print(f"  Neutral: {valid_neutral.mean():.1f} ms (SD={valid_neutral.std():.1f})")

    print(f"\n  Traditional Stroop effect: {stroop_df['stroop_effect'].mean():.1f} ms")

    if has_neutral:
        # Decomposed effects
        print("\n[2] Decomposed Effects Analysis")
        print("-" * 40)
        decomp_results = analyze_decomposed_effects(stroop_df)

        if 'facilitation' in decomp_results:
            fac = decomp_results['facilitation']
            sig_fac = "*" if fac['p'] < 0.05 else ""
            print(f"  Facilitation: {fac['mean']:.1f} ms, t({fac['n']-1})={fac['t']:.2f}, p={fac['p']:.4f}{sig_fac}")

        if 'interference' in decomp_results:
            intf = decomp_results['interference']
            sig_int = "*" if intf['p'] < 0.05 else ""
            print(f"  Interference: {intf['mean']:.1f} ms, t({intf['n']-1})={intf['t']:.2f}, p={intf['p']:.4f}{sig_int}")

        if 'comparison' in decomp_results:
            comp = decomp_results['comparison']
            print(f"\n  Interference vs Facilitation: t={comp['t']:.2f}, p={comp['p']:.4f}")
            print(f"  Interference larger: {comp['interference_larger']}")

        pd.DataFrame([decomp_results.get('facilitation', {}),
                     decomp_results.get('interference', {})]).to_csv(
            OUTPUT_DIR / "decomposed_effects.csv", index=False, encoding='utf-8-sig'
        )
    else:
        print("\n[2] Decomposed Effects Analysis")
        print("-" * 40)
        print("  Skipped: No neutral condition in data")

    # UCLA analysis
    print("\n[3] UCLA → Stroop components (DASS-controlled)")
    print("-" * 40)

    master = load_master_dataset(use_cache=True)
    ucla_results, models = analyze_ucla_decomposed(master, stroop_df)

    if len(ucla_results) > 0:
        for _, row in ucla_results.iterrows():
            sig = "*" if row['ucla_p'] < 0.05 else ""
            print(f"  {row['outcome_label']}: b={row['ucla_coef']:.3f}, p={row['ucla_p']:.4f}{sig}")

        ucla_results.to_csv(OUTPUT_DIR / "ucla_stroop_regression.csv", index=False, encoding='utf-8-sig')

        # Save model summaries
        with open(OUTPUT_DIR / "model_summaries.txt", 'w', encoding='utf-8') as f:
            for name, model in models.items():
                f.write(f"\n{name.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(model.summary().as_text())
                f.write("\n\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Stroop Neutral Condition Analysis:

Condition Means:
- Congruent: {stroop_df['cong_rt'].mean():.0f} ms
- Neutral: {stroop_df['neutral_rt'].mean():.0f} ms (if available)
- Incongruent: {stroop_df['incong_rt'].mean():.0f} ms

Decomposition:
- Traditional effect = Incongruent - Congruent = {stroop_df['stroop_effect'].mean():.0f} ms
- Facilitation = Neutral - Congruent (speed-up from congruency)
- Interference = Incongruent - Neutral (slow-down from conflict)

UCLA Effects:
""")
    if len(ucla_results) > 0:
        for _, row in ucla_results.iterrows():
            sig = "SIGNIFICANT" if row['ucla_p'] < 0.05 else "n.s."
            print(f"  - {row['outcome_label']}: b={row['ucla_coef']:.3f}, p={row['ucla_p']:.4f} ({sig})")

    print(f"""
Interpretation:
- If UCLA → Interference (not Facilitation): Conflict-specific effect
- If UCLA → Facilitation (not Interference): Baseline processing effect
- If UCLA → Both equally: General slowing
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
