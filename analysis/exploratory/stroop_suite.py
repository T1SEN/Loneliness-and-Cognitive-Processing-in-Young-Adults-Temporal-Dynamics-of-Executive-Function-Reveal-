"""
Stroop Exploratory Analysis Suite
=================================

Unified exploratory analyses for the Stroop task.

Consolidates:
- stroop_conflict_adaptation.py
- stroop_neutral_baseline.py
- stroop_exgaussian_decomposition.py
- stroop_post_error_adjustments.py
- stroop_cse_conflict_adaptation.py

Usage:
    python -m analysis.exploratory.stroop_suite
    python -m analysis.exploratory.stroop_suite --analysis conflict_adaptation

NOTE: These are EXPLORATORY analyses.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import exponnorm
from scipy.optimize import minimize
import statsmodels.formula.api as smf

from analysis.utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, STROOP_RT_MAX
)
from analysis.utils.modeling import standardize_predictors

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "stroop_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_stroop_trials() -> pd.DataFrame:
    """Load Stroop trial data."""
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    # Standardize RT column
    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    df['rt'] = df[rt_col]

    # Standardize congruency - actual data uses 'type' column with values: congruent, incongruent, neutral
    if 'type' in df.columns:
        df['congruent'] = df['type'].str.lower() == 'congruent'
        df['incongruent'] = df['type'].str.lower() == 'incongruent'
        df['is_neutral'] = df['type'].str.lower() == 'neutral'
    elif 'congruency' in df.columns:
        df['congruent'] = df['congruency'].str.lower() == 'congruent'
        df['incongruent'] = df['congruency'].str.lower() == 'incongruent'
        df['is_neutral'] = df['congruency'].str.lower() == 'neutral'
    elif 'iscongruent' in df.columns:
        # Boolean column: True/1 = congruent, False/0 = incongruent
        df['congruent'] = df['iscongruent'].astype(bool)
        df['incongruent'] = ~df['congruent']
        df['is_neutral'] = False

    # Filter valid trials
    df = df[
        (df['rt'] > DEFAULT_RT_MIN) &
        (df['rt'] < STROOP_RT_MAX) &
        (df['correct'] == 1)
    ].copy()

    return df


def load_master_with_stroop() -> pd.DataFrame:
    """Load master dataset with Stroop metrics."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    if 'gender_normalized' in master.columns:
        master['gender'] = master['gender_normalized'].fillna('').str.strip().str.lower()
    else:
        master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
    master['gender_male'] = (master['gender'] == 'male').astype(int)

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    master = standardize_predictors(master)
    return master


# =============================================================================
# ANALYSES
# =============================================================================

ANALYSES = {}


def analyze_conflict_adaptation(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze Congruency Sequence Effects (Gratton effect).

    Tests whether current trial congruency effect depends on previous trial.
    """
    output_dir = OUTPUT_DIR / "conflict_adaptation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[CONFLICT ADAPTATION] Analyzing congruency sequence effects...")

    trials = load_stroop_trials()
    master = load_master_with_stroop()

    # Sort and add previous trial info
    trials = trials.sort_values(['participant_id', 'idx' if 'idx' in trials.columns else 'trial'])
    trials['prev_congruent'] = trials.groupby('participant_id')['congruent'].shift(1)

    # Compute CSE per participant
    cse_results = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid].dropna(subset=['prev_congruent'])

        if len(pdata) < 20:
            continue

        # Current congruency effect after congruent vs incongruent
        after_cong = pdata[pdata['prev_congruent'] == True]
        after_incong = pdata[pdata['prev_congruent'] == False]

        # Congruency effect = Incongruent RT - Congruent RT
        ce_after_cong = after_cong[after_cong['incongruent'] == True]['rt'].mean() - after_cong[after_cong['congruent'] == True]['rt'].mean()
        ce_after_incong = after_incong[after_incong['incongruent'] == True]['rt'].mean() - after_incong[after_incong['congruent'] == True]['rt'].mean()

        cse = ce_after_cong - ce_after_incong  # Gratton effect

        cse_results.append({
            'participant_id': pid,
            'ce_after_congruent': ce_after_cong,
            'ce_after_incongruent': ce_after_incong,
            'cse': cse,
            'n_trials': len(pdata)
        })

    cse_df = pd.DataFrame(cse_results)
    analysis_df = cse_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean CSE (Gratton): {analysis_df['cse'].mean():.1f} ms")

    # DASS-controlled regression
    if len(analysis_df) >= 30:
        formula = "cse ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'cse'])).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → CSE: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

        pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'p': model.pvalues.values
        }).to_csv(output_dir / "cse_regression.csv", index=False, encoding='utf-8-sig')

    cse_df.to_csv(output_dir / "cse_metrics.csv", index=False, encoding='utf-8-sig')
    return cse_df


def analyze_neutral_baseline(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze facilitation vs interference using neutral baseline.

    Decomposes Stroop effect into:
    - Facilitation: Neutral - Congruent
    - Interference: Incongruent - Neutral
    """
    output_dir = OUTPUT_DIR / "neutral_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[NEUTRAL BASELINE] Analyzing facilitation vs interference...")

    trials = load_stroop_trials()
    master = load_master_with_stroop()

    # Check for neutral trials (is_neutral already set in load_stroop_trials)
    if 'is_neutral' not in trials.columns or not trials['is_neutral'].any():
        if verbose:
            print("  No neutral trials found. Skipping.")
        return pd.DataFrame()

    # Compute metrics per participant
    results = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid]

        rt_cong = pdata[pdata['congruent'] == True]['rt'].mean()
        rt_incong = pdata[pdata['incongruent'] == True]['rt'].mean()
        rt_neutral = pdata[pdata['is_neutral'] == True]['rt'].mean()

        if pd.isna(rt_neutral):
            continue

        results.append({
            'participant_id': pid,
            'rt_congruent': rt_cong,
            'rt_incongruent': rt_incong,
            'rt_neutral': rt_neutral,
            'facilitation': rt_neutral - rt_cong,
            'interference': rt_incong - rt_neutral,
            'stroop_effect': rt_incong - rt_cong
        })

    if not results:
        if verbose:
            print("  Insufficient neutral trial data.")
        return pd.DataFrame()

    baseline_df = pd.DataFrame(results)

    # Keep only necessary columns from master to avoid conflicts
    master_cols = ['participant_id', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'gender_male', 'ucla_total']
    master_subset = master[[c for c in master_cols if c in master.columns]].copy()

    analysis_df = baseline_df.merge(master_subset, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean facilitation: {analysis_df['facilitation'].mean():.1f} ms")
        print(f"  Mean interference: {analysis_df['interference'].mean():.1f} ms")

    # DASS-controlled regression on decomposed effects
    regression_results = []
    if len(analysis_df) >= 30:
        for outcome, label in [('facilitation', 'Facilitation'), ('interference', 'Interference'),
                               ('stroop_effect', 'Traditional Stroop')]:
            valid = analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', outcome])
            if len(valid) < 20:
                continue

            formula = f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=valid).fit(cov_type='HC3')

            regression_results.append({
                'outcome': label,
                'n': len(valid),
                'ucla_beta': model.params.get('z_ucla', np.nan),
                'ucla_p': model.pvalues.get('z_ucla', np.nan),
                'interaction_beta': model.params.get('z_ucla:C(gender_male)[T.1]', np.nan),
                'interaction_p': model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan)
            })

            if verbose:
                sig = "*" if model.pvalues.get('z_ucla', 1) < 0.05 else ""
                print(f"  UCLA → {label}: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}{sig}")

        if regression_results:
            pd.DataFrame(regression_results).to_csv(output_dir / "ucla_decomposed_regression.csv",
                                                     index=False, encoding='utf-8-sig')

    baseline_df.to_csv(output_dir / "neutral_baseline_metrics.csv", index=False, encoding='utf-8-sig')
    return baseline_df


def analyze_post_error(verbose: bool = True) -> pd.DataFrame:
    """Analyze post-error slowing in Stroop task."""
    output_dir = OUTPUT_DIR / "post_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[POST-ERROR] Analyzing post-error adjustments...")

    # Load all trials (including errors)
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    df['rt'] = df[rt_col]
    df = df[(df['rt'] > DEFAULT_RT_MIN) & (df['rt'] < STROOP_RT_MAX)].copy()

    master = load_master_with_stroop()

    # Sort and add previous trial info
    df = df.sort_values(['participant_id', 'idx' if 'idx' in df.columns else 'trial'])
    df['prev_correct'] = df.groupby('participant_id')['correct'].shift(1)

    # Compute PES
    pes_results = []
    for pid in df['participant_id'].unique():
        pdata = df[(df['participant_id'] == pid) & (df['correct'] == 1)].dropna(subset=['prev_correct'])

        if len(pdata) < 20:
            continue

        rt_after_error = pdata[pdata['prev_correct'] == 0]['rt'].mean()
        rt_after_correct = pdata[pdata['prev_correct'] == 1]['rt'].mean()
        pes = rt_after_error - rt_after_correct if not pd.isna(rt_after_error) else np.nan

        pes_results.append({
            'participant_id': pid,
            'pes': pes,
            'n_post_error': (pdata['prev_correct'] == 0).sum(),
            'n_post_correct': (pdata['prev_correct'] == 1).sum()
        })

    pes_df = pd.DataFrame(pes_results)
    analysis_df = pes_df.merge(master, on='participant_id', how='inner')

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean PES: {analysis_df['pes'].mean():.1f} ms")

    # DASS-controlled regression
    if len(analysis_df) >= 30:
        formula = "pes ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df.dropna(subset=['z_ucla', 'z_dass_dep', 'pes'])).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA → PES: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")

    pes_df.to_csv(output_dir / "stroop_pes_metrics.csv", index=False, encoding='utf-8-sig')
    return pes_df


def analyze_cse(verbose: bool = True) -> pd.DataFrame:
    """
    Detailed Congruency Sequence Effect (CSE) analysis.

    Tests whether lonely males show impaired trial-to-trial cognitive control adjustments.
    Computes 2×2 sequence effects (cC, cI, iC, iI) and tests UCLA × Gender interaction.

    Source: stroop_cse_conflict_adaptation.py
    """
    output_dir = OUTPUT_DIR / "cse"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[CSE] Analyzing detailed congruency sequence effects...")

    # Load trial data
    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8')
    df = ensure_participant_id(df)
    df.columns = df.columns.str.lower()

    rt_col = 'rt_ms' if 'rt_ms' in df.columns else 'rt'
    df['rt'] = df[rt_col]

    # Find trial index column
    trial_idx_col = None
    for cand in ['trialindex', 'trial', 'idx']:
        if cand in df.columns:
            trial_idx_col = cand
            break
    if not trial_idx_col:
        trial_idx_col = df.columns[0]  # Fallback

    # Standardize type column
    if 'type' not in df.columns:
        for cand in ['condition', 'cond']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'type'})
                break

    # Filter valid trials
    valid = df[
        (df['correct'] == True) &
        (df['rt'] > 200) &
        (df['rt'] < 3000) &
        (df['type'].isin(['congruent', 'incongruent']))
    ].copy()

    valid = valid.sort_values(['participant_id', trial_idx_col])
    valid['is_incongruent'] = (valid['type'] == 'incongruent').astype(int)
    valid['prev_is_incongruent'] = valid.groupby('participant_id')['is_incongruent'].shift(1)

    cse_data = valid.dropna(subset=['prev_is_incongruent']).copy()

    # Compute CSE per participant
    cse_results = []
    for pid in cse_data['participant_id'].unique():
        group = cse_data[cse_data['participant_id'] == pid]

        if len(group) < 20:
            continue

        # 2×2 cell means
        cC = group[(group['prev_is_incongruent'] == 0) & (group['is_incongruent'] == 0)]['rt'].mean()
        cI = group[(group['prev_is_incongruent'] == 0) & (group['is_incongruent'] == 1)]['rt'].mean()
        iC = group[(group['prev_is_incongruent'] == 1) & (group['is_incongruent'] == 0)]['rt'].mean()
        iI = group[(group['prev_is_incongruent'] == 1) & (group['is_incongruent'] == 1)]['rt'].mean()

        if pd.notna([cC, cI, iC, iI]).all():
            interference_after_congruent = cI - cC
            interference_after_incongruent = iI - iC
            cse = interference_after_congruent - interference_after_incongruent

            cse_results.append({
                'participant_id': pid,
                'cC': cC, 'cI': cI, 'iC': iC, 'iI': iI,
                'interference_after_congruent': interference_after_congruent,
                'interference_after_incongruent': interference_after_incongruent,
                'cse': cse,
                'n_trials': len(group)
            })

    cse_df = pd.DataFrame(cse_results)
    master = load_master_with_stroop()
    analysis_df = cse_df.merge(master, on='participant_id', how='inner')

    required = ['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', 'cse']
    analysis_df = analysis_df.dropna(subset=[c for c in required if c in analysis_df.columns])

    if verbose:
        print(f"  N participants: {len(analysis_df)}")
        print(f"  Mean CSE: {analysis_df['cse'].mean():.1f} ms (SD={analysis_df['cse'].std():.1f})")

    # Gender-stratified correlations
    results_rows = []
    for gender_val, label in [(1, 'Male'), (0, 'Female')]:
        subset = analysis_df[analysis_df['gender_male'] == gender_val]
        if len(subset) >= 10:
            r, p = stats.pearsonr(subset['ucla_total'], subset['cse'])
            results_rows.append({'gender': label, 'n': len(subset), 'r': r, 'p': p})
            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"  {label} (N={len(subset)}): r={r:.3f}, p={p:.4f}{sig}")

    # DASS-controlled regression
    if len(analysis_df) >= 30:
        formula = "cse ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
        model = smf.ols(formula, data=analysis_df).fit(cov_type='HC3')

        if verbose:
            print(f"  UCLA main: β={model.params.get('z_ucla', np.nan):.3f}, p={model.pvalues.get('z_ucla', np.nan):.4f}")
            print(f"  UCLA×Gender: β={model.params.get('z_ucla:C(gender_male)[T.1]', np.nan):.3f}, p={model.pvalues.get('z_ucla:C(gender_male)[T.1]', np.nan):.4f}")

        pd.DataFrame({
            'predictor': model.params.index,
            'beta': model.params.values,
            'se': model.bse.values,
            'p': model.pvalues.values
        }).to_csv(output_dir / "cse_regression.csv", index=False, encoding='utf-8-sig')

    cse_df.to_csv(output_dir / "cse_participant_scores.csv", index=False, encoding='utf-8-sig')
    if results_rows:
        pd.DataFrame(results_rows).to_csv(output_dir / "cse_gender_correlations.csv", index=False, encoding='utf-8-sig')

    return cse_df


def _fit_exgaussian(rts: np.ndarray) -> dict:
    """
    Fit Ex-Gaussian distribution using MLE.

    Returns mu, sigma, tau parameters.
    """
    if len(rts) < 20:
        return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}

    rts = np.array(rts)

    # Method of moments initial estimates
    m = np.mean(rts)
    s = np.std(rts)
    skew = np.mean(((rts - m) / s) ** 3) if s > 0 else 0

    tau_init = max(10, (abs(skew) / 2) ** (1/3) * s) if skew > 0 else 50
    mu_init = max(100, m - tau_init)
    sigma_init = max(10, np.sqrt(max(0, s**2 - tau_init**2)))

    def neg_loglik(params):
        mu, sigma, tau = params
        if sigma <= 0 or tau <= 0:
            return 1e10
        K = tau / sigma
        try:
            loglik = np.sum(exponnorm.logpdf(rts, K, loc=mu, scale=sigma))
            return -loglik if np.isfinite(loglik) else 1e10
        except:
            return 1e10

    try:
        result = minimize(
            neg_loglik,
            x0=[mu_init, sigma_init, tau_init],
            method='L-BFGS-B',
            bounds=[(100, 2000), (5, 500), (5, 1000)]
        )

        if result.success:
            mu, sigma, tau = result.x
            return {'mu': mu, 'sigma': sigma, 'tau': tau, 'n': len(rts)}
    except:
        pass

    return {'mu': np.nan, 'sigma': np.nan, 'tau': np.nan, 'n': len(rts)}


def analyze_exgaussian(verbose: bool = True) -> pd.DataFrame:
    """
    Ex-Gaussian RT decomposition for Stroop task.

    Decomposes RT distributions into:
    - μ (mu): Gaussian mean (routine processing speed)
    - σ (sigma): Gaussian SD (processing variability)
    - τ (tau): Exponential component (attentional lapses)

    Source: stroop_exgaussian_decomposition.py
    """
    output_dir = OUTPUT_DIR / "exgaussian"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[EXGAUSSIAN] Fitting Ex-Gaussian distributions...")

    trials = load_stroop_trials()
    master = load_master_with_stroop()

    # Fit for each participant × condition
    results = []
    for pid in trials['participant_id'].unique():
        pdata = trials[trials['participant_id'] == pid]

        row = {'participant_id': pid}

        for condition, is_incong in [('congruent', False), ('incongruent', True)]:
            if 'incongruent' in pdata.columns:
                cond_trials = pdata[pdata['incongruent'] == is_incong]
            else:
                cond_trials = pdata[pdata['type'] == condition] if 'type' in pdata.columns else pdata

            params = _fit_exgaussian(cond_trials['rt'].values)
            row[f'{condition}_mu'] = params['mu']
            row[f'{condition}_sigma'] = params['sigma']
            row[f'{condition}_tau'] = params['tau']
            row[f'{condition}_n'] = params['n']

        results.append(row)

    exgauss_df = pd.DataFrame(results)

    # Add interference effects
    exgauss_df['mu_interference'] = exgauss_df['incongruent_mu'] - exgauss_df['congruent_mu']
    exgauss_df['sigma_interference'] = exgauss_df['incongruent_sigma'] - exgauss_df['congruent_sigma']
    exgauss_df['tau_interference'] = exgauss_df['incongruent_tau'] - exgauss_df['congruent_tau']

    analysis_df = exgauss_df.merge(master, on='participant_id', how='inner')

    valid_count = analysis_df['congruent_mu'].notna().sum()
    if verbose:
        print(f"  Fitted N={valid_count} participants")
        print(f"  Mean congruent μ: {analysis_df['congruent_mu'].mean():.1f} ms")
        print(f"  Mean congruent σ: {analysis_df['congruent_sigma'].mean():.1f} ms")
        print(f"  Mean congruent τ: {analysis_df['congruent_tau'].mean():.1f} ms")

    # Gender-stratified correlations
    corr_results = []
    for gender_val, gender_label in [(1, 'male'), (0, 'female')]:
        subset = analysis_df[analysis_df['gender_male'] == gender_val]

        for condition in ['congruent', 'incongruent']:
            for param in ['mu', 'sigma', 'tau']:
                col = f'{condition}_{param}'
                valid = subset.dropna(subset=['ucla_total', col])

                if len(valid) >= 10:
                    r, p = stats.pearsonr(valid['ucla_total'], valid[col])
                    corr_results.append({
                        'gender': gender_label,
                        'condition': condition,
                        'parameter': param,
                        'n': len(valid),
                        'r': r,
                        'p': p,
                        'mean': valid[col].mean(),
                        'sd': valid[col].std()
                    })

    corr_df = pd.DataFrame(corr_results)

    if verbose and len(corr_df) > 0:
        sig_effects = corr_df[corr_df['p'] < 0.10]
        if len(sig_effects) > 0:
            print("  Marginal effects (p < 0.10):")
            for _, row in sig_effects.iterrows():
                print(f"    {row['gender']} {row['condition']} {row['parameter']}: r={row['r']:.3f}, p={row['p']:.4f}")

    # === DASS-controlled regression for Ex-Gaussian parameters ===
    regression_results = {}
    required_cols = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age', 'gender_male']

    if all(col in analysis_df.columns for col in required_cols):
        # Standardize predictors
        reg_df = analysis_df.copy()
        for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
            if reg_df[col].std() > 0:
                reg_df[f'z_{col}'] = (reg_df[col] - reg_df[col].mean()) / reg_df[col].std()

        # Rename for formula
        reg_df = reg_df.rename(columns={
            'z_ucla_total': 'z_ucla',
            'z_dass_depression': 'z_dass_dep',
            'z_dass_anxiety': 'z_dass_anx',
            'z_dass_stress': 'z_dass_str'
        })

        # Run regression for tau_interference (primary Ex-Gaussian parameter)
        for dv_name in ['tau_interference', 'mu_interference', 'sigma_interference']:
            if dv_name in reg_df.columns:
                valid_df = reg_df.dropna(subset=['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age', dv_name])

                if len(valid_df) >= 20:
                    formula = f"{dv_name} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                    try:
                        model = smf.ols(formula, data=valid_df).fit(cov_type='HC3')

                        if verbose and dv_name == 'tau_interference':
                            print(f"\n  DASS-Controlled Regression ({dv_name}):")
                            print(f"    N={int(model.nobs)}, R²={model.rsquared:.3f}")
                            print(f"    UCLA main: β={model.params.get('z_ucla', 0):.3f}, p={model.pvalues.get('z_ucla', 1):.4f}")
                            ucla_gender = 'z_ucla:C(gender_male)[T.True]'
                            if ucla_gender in model.params:
                                print(f"    UCLA×Gender: β={model.params[ucla_gender]:.3f}, p={model.pvalues[ucla_gender]:.4f}")

                        regression_results[dv_name] = {
                            'coefficients': model.params.to_dict(),
                            'pvalues': model.pvalues.to_dict(),
                            'r_squared': model.rsquared,
                            'n': int(model.nobs)
                        }
                    except Exception as e:
                        if verbose:
                            print(f"  Regression failed for {dv_name}: {e}")

        # Save regression results
        if regression_results:
            reg_summary = []
            for dv, res in regression_results.items():
                for coef, val in res['coefficients'].items():
                    reg_summary.append({
                        'outcome': dv,
                        'predictor': coef,
                        'beta': val,
                        'p_value': res['pvalues'].get(coef, np.nan),
                        'r_squared': res['r_squared'],
                        'n': res['n']
                    })
            pd.DataFrame(reg_summary).to_csv(
                output_dir / "exgaussian_dass_regression.csv",
                index=False, encoding='utf-8-sig'
            )

    exgauss_df.to_csv(output_dir / "exgaussian_parameters.csv", index=False, encoding='utf-8-sig')
    if len(corr_df) > 0:
        corr_df.to_csv(output_dir / "exgaussian_correlations.csv", index=False, encoding='utf-8-sig')

    return exgauss_df


ANALYSES = {
    'conflict_adaptation': ('Congruency Sequence Effects (Gratton)', analyze_conflict_adaptation),
    'neutral_baseline': ('Facilitation vs Interference', analyze_neutral_baseline),
    'post_error': ('Post-Error Slowing', analyze_post_error),
    'cse': ('Detailed CSE Analysis', analyze_cse),
    'exgaussian': ('Ex-Gaussian RT Decomposition', analyze_exgaussian),
}


def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run Stroop exploratory analyses."""
    if verbose:
        print("=" * 70)
        print("STROOP EXPLORATORY ANALYSIS SUITE")
        print("=" * 70)

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        desc, func = ANALYSES[analysis]
        results[analysis] = func(verbose=verbose)
    else:
        for name, (desc, func) in ANALYSES.items():
            if verbose:
                print(f"\n[{name.upper()}] {desc}")
            try:
                results[name] = func(verbose=verbose)
            except Exception as e:
                print(f"  ERROR: {e}")

    if verbose:
        print(f"\nOutput: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stroop Exploratory Suite")
    parser.add_argument('--analysis', '-a', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()
    run(analysis=args.analysis, verbose=not args.quiet)
