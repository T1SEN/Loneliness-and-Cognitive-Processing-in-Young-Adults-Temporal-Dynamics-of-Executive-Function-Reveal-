"""
WCST Error Type Decomposition
=============================
Decomposes WCST errors into distinct types:
- Perseverative Errors (PE): Stuck on previous rule
- Perseverative Responses (PR): Response matching old rule (correct or error)
- Non-Perseverative Errors (NPE): Random/other errors

Tests whether UCLA differentially predicts each error type.
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
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "wcst_error_decomposition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_wcst_with_error_types():
    """Load WCST trials with error type flags."""
    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    # Parse error type flags
    for col in ['isPE', 'isPR', 'isNPE']:
        if col in df.columns:
            df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
            df[col] = df[col].fillna(False)

    # Calculate participant-level metrics
    results = []
    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid]
        n_trials = len(pdata)

        if n_trials < 20:
            continue

        # Count error types
        n_pe = pdata['isPE'].sum() if 'isPE' in pdata.columns else 0
        n_pr = pdata['isPR'].sum() if 'isPR' in pdata.columns else 0
        n_npe = pdata['isNPE'].sum() if 'isNPE' in pdata.columns else 0
        n_errors = (~pdata['correct']).sum()

        results.append({
            'participant_id': pid,
            'n_trials': n_trials,
            'n_errors': n_errors,
            'n_pe': n_pe,
            'n_pr': n_pr,
            'n_npe': n_npe,
            'pe_rate': n_pe / n_trials * 100,
            'pr_rate': n_pr / n_trials * 100,
            'npe_rate': n_npe / n_trials * 100,
            'error_rate': n_errors / n_trials * 100,
            'pe_proportion': n_pe / n_errors * 100 if n_errors > 0 else 0,
            'npe_proportion': n_npe / n_errors * 100 if n_errors > 0 else 0
        })

    return pd.DataFrame(results)


def analyze_error_type_correlations(error_df, master_df):
    """Analyze correlations between error types and UCLA/DASS."""
    merged = master_df.merge(error_df, on='participant_id', how='inner')

    predictors = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress']
    outcomes = ['pe_rate', 'pr_rate', 'npe_rate', 'pe_proportion', 'npe_proportion']

    corr_results = []
    for pred in predictors:
        for out in outcomes:
            if pred in merged.columns and out in merged.columns:
                valid = merged[[pred, out]].dropna()
                if len(valid) > 20:
                    r, p = stats.pearsonr(valid[pred], valid[out])
                    corr_results.append({
                        'predictor': pred,
                        'outcome': out,
                        'r': r,
                        'p': p,
                        'n': len(valid)
                    })

    return pd.DataFrame(corr_results)


def analyze_ucla_error_types(master_df, error_df):
    """Regression analysis: UCLA → each error type (DASS-controlled)."""
    merged = master_df.merge(error_df, on='participant_id', how='inner')

    required = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'gender_male']
    merged = merged.dropna(subset=required)

    # Standardize
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

    outcomes = {
        'pe_rate': 'Perseverative Error Rate',
        'npe_rate': 'Non-Perseverative Error Rate',
        'pr_rate': 'Perseverative Response Rate',
        'pe_proportion': 'PE as % of All Errors',
        'npe_proportion': 'NPE as % of All Errors'
    }

    results = []
    models = {}

    for outcome, label in outcomes.items():
        if outcome not in merged.columns:
            continue

        formula = f"{outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        try:
            model = smf.ols(formula, data=merged).fit()
            models[outcome] = model

            results.append({
                'outcome': outcome,
                'outcome_label': label,
                'n': int(model.nobs),
                'ucla_coef': model.params.get('z_ucla_score', np.nan),
                'ucla_se': model.bse.get('z_ucla_score', np.nan),
                'ucla_p': model.pvalues.get('z_ucla_score', np.nan),
                'dass_dep_coef': model.params.get('z_dass_depression', np.nan),
                'dass_dep_p': model.pvalues.get('z_dass_depression', np.nan),
                'dass_anx_coef': model.params.get('z_dass_anxiety', np.nan),
                'dass_anx_p': model.pvalues.get('z_dass_anxiety', np.nan),
                'model_r2': model.rsquared,
                'model_f_p': model.f_pvalue
            })
        except Exception as e:
            print(f"Error fitting model for {outcome}: {e}")

    return pd.DataFrame(results), models


def compare_pe_vs_npe_effects(results_df):
    """Test if UCLA effect differs between PE and NPE."""
    pe_row = results_df[results_df['outcome'] == 'pe_rate']
    npe_row = results_df[results_df['outcome'] == 'npe_rate']

    if len(pe_row) == 0 or len(npe_row) == 0:
        return None

    pe_row = pe_row.iloc[0]
    npe_row = npe_row.iloc[0]

    # Z-test for difference in coefficients
    diff = pe_row['ucla_coef'] - npe_row['ucla_coef']
    se_diff = np.sqrt(pe_row['ucla_se']**2 + npe_row['ucla_se']**2)
    z = diff / se_diff if se_diff > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        'pe_coef': pe_row['ucla_coef'],
        'npe_coef': npe_row['ucla_coef'],
        'difference': diff,
        'se_diff': se_diff,
        'z': z,
        'p': p,
        'ucla_effect_larger_for': 'PE' if pe_row['ucla_coef'] > npe_row['ucla_coef'] else 'NPE'
    }


def analyze_error_sequences(df):
    """Analyze error clustering/sequences."""
    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    # Parse isPE
    if 'isPE' in df.columns:
        df['isPE'] = df['isPE'].map({'True': True, 'False': False, True: True, False: False})
        df['isPE'] = df['isPE'].fillna(False)

    results = []
    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid].sort_values('trialIndex')

        if len(pdata) < 20:
            continue

        # Count consecutive PE runs
        pe_series = pdata['isPE'].astype(int).values
        runs = []
        current_run = 0
        for val in pe_series:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        n_runs = len(runs)
        max_run = max(runs) if runs else 0
        mean_run = np.mean(runs) if runs else 0

        # PE-after-PE probability (perseveration momentum)
        pdata['prev_pe'] = pdata['isPE'].shift(1)
        pe_after_pe = pdata[pdata['prev_pe'] == True]['isPE'].mean() if pdata['prev_pe'].sum() > 0 else np.nan

        results.append({
            'participant_id': pid,
            'n_pe_runs': n_runs,
            'max_pe_run': max_run,
            'mean_pe_run': mean_run,
            'pe_after_pe_prob': pe_after_pe
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("WCST Error Type Decomposition")
    print("=" * 60)

    # Load error type data
    print("\n[1] Loading WCST error type data")
    print("-" * 40)
    error_df = load_wcst_with_error_types()
    print(f"  Participants: N={len(error_df)}")
    print(f"  Mean PE rate: {error_df['pe_rate'].mean():.1f}%")
    print(f"  Mean NPE rate: {error_df['npe_rate'].mean():.1f}%")
    print(f"  Mean PR rate: {error_df['pr_rate'].mean():.1f}%")

    error_df.to_csv(OUTPUT_DIR / "error_type_metrics.csv", index=False, encoding='utf-8-sig')

    # Correlations
    print("\n[2] Error type correlations with UCLA/DASS")
    print("-" * 40)
    master = load_master_dataset(use_cache=True)
    corr_df = analyze_error_type_correlations(error_df, master)

    if len(corr_df) > 0:
        print("\n  UCLA correlations:")
        ucla_corr = corr_df[corr_df['predictor'] == 'ucla_score']
        for _, row in ucla_corr.iterrows():
            sig = "*" if row['p'] < 0.05 else ""
            print(f"    {row['outcome']}: r={row['r']:.3f}, p={row['p']:.4f}{sig}")

        corr_df.to_csv(OUTPUT_DIR / "error_type_correlations.csv", index=False, encoding='utf-8-sig')

    # Regression analysis
    print("\n[3] UCLA → Error Types (DASS-controlled)")
    print("-" * 40)
    reg_results, models = analyze_ucla_error_types(master, error_df)

    if len(reg_results) > 0:
        print("\n  UCLA effects on error types:")
        for _, row in reg_results.iterrows():
            sig = "*" if row['ucla_p'] < 0.05 else ""
            print(f"    {row['outcome_label']}: b={row['ucla_coef']:.4f}, p={row['ucla_p']:.4f}{sig}")

        reg_results.to_csv(OUTPUT_DIR / "ucla_error_regression.csv", index=False, encoding='utf-8-sig')

        # Save model summaries
        with open(OUTPUT_DIR / "regression_summaries.txt", 'w', encoding='utf-8') as f:
            for name, model in models.items():
                f.write(f"\n{name.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(model.summary().as_text())
                f.write("\n\n")

    # Compare PE vs NPE
    print("\n[4] Comparing UCLA effect: PE vs NPE")
    print("-" * 40)
    comparison = compare_pe_vs_npe_effects(reg_results)
    if comparison:
        print(f"  PE coefficient: {comparison['pe_coef']:.4f}")
        print(f"  NPE coefficient: {comparison['npe_coef']:.4f}")
        print(f"  Difference: {comparison['difference']:.4f}")
        print(f"  z = {comparison['z']:.2f}, p = {comparison['p']:.4f}")
        print(f"  UCLA effect larger for: {comparison['ucla_effect_larger_for']}")

        pd.DataFrame([comparison]).to_csv(OUTPUT_DIR / "pe_vs_npe_comparison.csv", index=False, encoding='utf-8-sig')

    # Error sequences
    print("\n[5] Error Clustering Analysis")
    print("-" * 40)
    sequence_df = analyze_error_sequences(master)
    if len(sequence_df) > 0:
        print(f"  Mean max PE run: {sequence_df['max_pe_run'].mean():.2f}")
        print(f"  P(PE | previous PE): {sequence_df['pe_after_pe_prob'].mean():.3f}")

        # Merge with UCLA and test
        seq_merged = master.merge(sequence_df, on='participant_id', how='inner')
        if len(seq_merged) > 30:
            r, p = stats.pearsonr(
                seq_merged['ucla_score'].dropna(),
                seq_merged['max_pe_run'].dropna()
            )
            print(f"  UCLA × max_pe_run: r={r:.3f}, p={p:.4f}")

        sequence_df.to_csv(OUTPUT_DIR / "error_sequences.csv", index=False, encoding='utf-8-sig')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Error Type Decomposition Results:

Error Types:
- PE (Perseverative Error): Stuck on previous rule → reflects cognitive rigidity
- NPE (Non-Perseverative Error): Random errors → reflects general confusion
- PR (Perseverative Response): Old rule response (may be correct) → habit strength

UCLA Effects (DASS-controlled):
""")
    if len(reg_results) > 0:
        for _, row in reg_results.iterrows():
            sig = "SIGNIFICANT" if row['ucla_p'] < 0.05 else "n.s."
            print(f"  - {row['outcome_label']}: b={row['ucla_coef']:.4f}, p={row['ucla_p']:.4f} ({sig})")

    print(f"""
Interpretation:
- If UCLA → PE (but not NPE): Loneliness specifically impairs cognitive flexibility
- If UCLA → both PE and NPE: General cognitive impairment
- If UCLA → NPE only: Attention/engagement issues rather than flexibility
""")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
