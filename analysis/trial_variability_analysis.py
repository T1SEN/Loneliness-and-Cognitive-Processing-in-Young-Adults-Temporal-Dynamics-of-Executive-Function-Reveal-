#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trial-level Variability Analysis
IIV (intra-individual variability) & Post-error slowing
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Trial-Level Variability Analysis")
print("IIV & Post-Error Slowing")
print("=" * 80)

# =============================================================================
# 1. 데이터 로딩
# =============================================================================

master = pd.read_csv(OUTPUT_DIR / "master_dataset.csv")
master = master.rename(columns={'pe_rate': 'perseverative_error_rate'})

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

master = master.merge(
    participants[['participant_id', 'age', 'gender']],
    on='participant_id',
    how='left'
)

master['gender_male'] = (master['gender'] == '남성').astype(int)

# Load trial-level data
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")

# Handle duplicate columns
if 'participantId' in stroop_trials.columns and 'participant_id' in stroop_trials.columns:
    stroop_trials = stroop_trials.drop(columns=['participantId'])
elif 'participantId' in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})

if 'participantId' in prp_trials.columns and 'participant_id' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participantId'])
elif 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.rename(columns={'participantId': 'participant_id'})

print(f"\n샘플 크기: N = {len(master)}")
print(f"Stroop trials: {len(stroop_trials)}")
print(f"PRP trials: {len(prp_trials)}")

# =============================================================================
# 2. INTRA-INDIVIDUAL VARIABILITY (IIV)
# =============================================================================

print("\n" + "=" * 80)
print("1. INTRA-INDIVIDUAL VARIABILITY (IIV)")
print("=" * 80)

# Stroop IIV
stroop_clean = stroop_trials[
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 0)
].copy()

print(f"\nStroop: {len(stroop_clean)} valid trials from {stroop_clean['participant_id'].nunique()} participants")

# Calculate IIV (SD of RT) and CV (coefficient of variation)
stroop_iiv = stroop_clean.groupby('participant_id')['rt_ms'].agg([
    ('mean_rt', 'mean'),
    ('sd_rt', 'std'),
    ('n_trials', 'count')
]).reset_index()

stroop_iiv['cv_rt'] = stroop_iiv['sd_rt'] / stroop_iiv['mean_rt']
stroop_iiv.columns = ['participant_id', 'stroop_mean_rt', 'stroop_iiv', 'stroop_n_trials', 'stroop_cv']

# PRP IIV
prp_clean = prp_trials[
    (prp_trials['t1_timeout'] == False) &
    (prp_trials['t2_timeout'] == False) &
    (prp_trials['t2_rt_ms'] > 0)
].copy()

print(f"PRP: {len(prp_clean)} valid trials from {prp_clean['participant_id'].nunique()} participants")

prp_iiv = prp_clean.groupby('participant_id')['t2_rt_ms'].agg([
    ('mean_rt', 'mean'),
    ('sd_rt', 'std'),
    ('n_trials', 'count')
]).reset_index()

prp_iiv['cv_rt'] = prp_iiv['sd_rt'] / prp_iiv['mean_rt']
prp_iiv.columns = ['participant_id', 'prp_mean_rt', 'prp_iiv', 'prp_n_trials', 'prp_cv']

# Merge with master
master_iiv = master.merge(stroop_iiv, on='participant_id', how='left')
master_iiv = master_iiv.merge(prp_iiv, on='participant_id', how='left')

# =============================================================================
# 3. UCLA와 IIV 상관
# =============================================================================

print("\n" + "=" * 80)
print("2. UCLA와 IIV 상관")
print("=" * 80)

iiv_corr_results = []

for iiv_var, iiv_label in [
    ('stroop_iiv', 'Stroop IIV (SD)'),
    ('stroop_cv', 'Stroop CV'),
    ('prp_iiv', 'PRP IIV (SD)'),
    ('prp_cv', 'PRP CV')
]:
    df_clean = master_iiv[['ucla_total', iiv_var]].dropna()

    if len(df_clean) > 10:
        r, p = pearsonr(df_clean['ucla_total'], df_clean[iiv_var])
        rho, p_spear = spearmanr(df_clean['ucla_total'], df_clean[iiv_var])

        print(f"\n{iiv_label}:")
        print(f"  N = {len(df_clean)}")
        print(f"  Pearson: r = {r:.3f}, p = {p:.4f}")
        print(f"  Spearman: rho = {rho:.3f}, p = {p_spear:.4f}")

        if p < 0.05:
            print(f"  *** 유의함 (p < 0.05) ***")
        elif p < 0.10:
            print(f"  ** Marginal (p < 0.10) **")

        iiv_corr_results.append({
            'variable': iiv_label,
            'n': len(df_clean),
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spear
        })

iiv_corr_df = pd.DataFrame(iiv_corr_results)
iiv_corr_df.to_csv(OUTPUT_DIR / "iiv_correlations.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 4. 공변량 통제 회귀
# =============================================================================

print("\n" + "=" * 80)
print("3. IIV 회귀 분석 (DASS 통제)")
print("=" * 80)

iiv_regression_results = []

for iiv_var, iiv_label in [
    ('stroop_cv', 'Stroop CV'),
    ('prp_cv', 'PRP CV')
]:
    df_reg = master_iiv[
        ['participant_id', iiv_var, 'ucla_total', 'dass_depression',
         'dass_anxiety', 'dass_stress', 'age', 'gender_male']
    ].dropna()

    if len(df_reg) > 30:
        print(f"\n{iiv_label} (N={len(df_reg)}):")

        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_reg['z_ucla'] = scaler.fit_transform(df_reg[['ucla_total']])
        df_reg['z_dass_dep'] = scaler.fit_transform(df_reg[['dass_depression']])
        df_reg['z_dass_anx'] = scaler.fit_transform(df_reg[['dass_anxiety']])
        df_reg['z_dass_stress'] = scaler.fit_transform(df_reg[['dass_stress']])

        # Model 1: UCLA only
        formula1 = f"{iiv_var} ~ z_ucla + age + gender_male"
        model1 = smf.ols(formula1, data=df_reg).fit()

        # Model 2: UCLA + DASS
        formula2 = f"{iiv_var} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress + age + gender_male"
        model2 = smf.ols(formula2, data=df_reg).fit()

        print(f"  Model 1 (UCLA only): R² = {model1.rsquared:.4f}")
        print(f"    UCLA: β = {model1.params['z_ucla']:.4f}, p = {model1.pvalues['z_ucla']:.4f}")

        print(f"  Model 2 (UCLA + DASS): R² = {model2.rsquared:.4f}")
        print(f"    UCLA: β = {model2.params['z_ucla']:.4f}, p = {model2.pvalues['z_ucla']:.4f}")
        print(f"    ΔR² = {model2.rsquared - model1.rsquared:.4f}")

        iiv_regression_results.append({
            'outcome': iiv_label,
            'n': len(df_reg),
            'model1_r2': model1.rsquared,
            'model1_ucla_beta': model1.params['z_ucla'],
            'model1_ucla_p': model1.pvalues['z_ucla'],
            'model2_r2': model2.rsquared,
            'model2_ucla_beta': model2.params['z_ucla'],
            'model2_ucla_p': model2.pvalues['z_ucla'],
            'delta_r2': model2.rsquared - model1.rsquared
        })

iiv_reg_df = pd.DataFrame(iiv_regression_results)
iiv_reg_df.to_csv(OUTPUT_DIR / "iiv_regression.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 5. POST-ERROR SLOWING
# =============================================================================

print("\n" + "=" * 80)
print("4. POST-ERROR SLOWING")
print("=" * 80)

def calc_post_error_slowing(df, rt_col='rt_ms', correct_col='correct'):
    """Calculate post-error slowing per participant"""
    df = df.sort_values(['participant_id', 'trialIndex']).copy()

    # Previous trial correctness
    df['prev_correct'] = df.groupby('participant_id')[correct_col].shift(1)

    # Remove first trial of each participant
    df_valid = df[df['prev_correct'].notna()].copy()

    # Calculate mean RT after error vs after correct
    post_error_rt = df_valid[df_valid['prev_correct'] == False][rt_col].mean()
    post_correct_rt = df_valid[df_valid['prev_correct'] == True][rt_col].mean()

    pes = post_error_rt - post_correct_rt

    return pes, post_error_rt, post_correct_rt

# Stroop PES
print("\nStroop Post-Error Slowing:")
print("-" * 80)

stroop_pes_list = []

for pid in stroop_clean['participant_id'].unique():
    p_trials = stroop_clean[stroop_clean['participant_id'] == pid].copy()

    if len(p_trials) < 10:
        continue

    # Sort by trial index (use available column)
    trial_col = 'trial_index' if 'trial_index' in p_trials.columns else 'trialIndex' if 'trialIndex' in p_trials.columns else 'trial'
    if trial_col in p_trials.columns:
        p_trials = p_trials.sort_values(trial_col).copy()
    p_trials['prev_correct'] = p_trials['correct'].shift(1)

    # Remove first trial
    p_trials_valid = p_trials[p_trials['prev_correct'].notna()].copy()

    if len(p_trials_valid[p_trials_valid['prev_correct'] == False]) < 3:
        continue

    post_error = p_trials_valid[p_trials_valid['prev_correct'] == False]['rt_ms'].mean()
    post_correct = p_trials_valid[p_trials_valid['prev_correct'] == True]['rt_ms'].mean()

    pes = post_error - post_correct

    stroop_pes_list.append({
        'participant_id': pid,
        'stroop_pes': pes,
        'post_error_rt': post_error,
        'post_correct_rt': post_correct,
        'n_errors': len(p_trials_valid[p_trials_valid['prev_correct'] == False])
    })

stroop_pes_df = pd.DataFrame(stroop_pes_list)

print(f"Participants with PES data: {len(stroop_pes_df)}")
print(f"Mean PES: {stroop_pes_df['stroop_pes'].mean():.2f} ms (SD={stroop_pes_df['stroop_pes'].std():.2f})")

# Merge with master
master_pes = master_iiv.merge(stroop_pes_df[['participant_id', 'stroop_pes']], on='participant_id', how='left')

# PRP PES
print("\nPRP Post-Error Slowing:")
print("-" * 80)

prp_pes_list = []

for pid in prp_clean['participant_id'].unique():
    p_trials = prp_clean[prp_clean['participant_id'] == pid].copy()

    if len(p_trials) < 10:
        continue

    # Need to check T1 or T2 correct - use T2
    if 't2_correct' not in p_trials.columns:
        continue

    # Use available trial index column
    trial_col_prp = 'trial_index' if 'trial_index' in p_trials.columns else 'idx' if 'idx' in p_trials.columns else 'trial'
    if trial_col_prp in p_trials.columns:
        p_trials = p_trials.sort_values(trial_col_prp).copy()
    p_trials['prev_t2_correct'] = p_trials['t2_correct'].shift(1)

    p_trials_valid = p_trials[p_trials['prev_t2_correct'].notna()].copy()

    if len(p_trials_valid[p_trials_valid['prev_t2_correct'] == False]) < 3:
        continue

    post_error = p_trials_valid[p_trials_valid['prev_t2_correct'] == False]['t2_rt_ms'].mean()
    post_correct = p_trials_valid[p_trials_valid['prev_t2_correct'] == True]['t2_rt_ms'].mean()

    pes = post_error - post_correct

    prp_pes_list.append({
        'participant_id': pid,
        'prp_pes': pes,
        'prp_post_error_rt': post_error,
        'prp_post_correct_rt': post_correct,
        'prp_n_errors': len(p_trials_valid[p_trials_valid['prev_t2_correct'] == False])
    })

prp_pes_df = pd.DataFrame(prp_pes_list)

if len(prp_pes_df) > 0:
    print(f"Participants with PES data: {len(prp_pes_df)}")
    print(f"Mean PES: {prp_pes_df['prp_pes'].mean():.2f} ms (SD={prp_pes_df['prp_pes'].std():.2f})")

    master_pes = master_pes.merge(prp_pes_df[['participant_id', 'prp_pes']], on='participant_id', how='left')

# =============================================================================
# 6. UCLA와 PES 상관
# =============================================================================

print("\n" + "=" * 80)
print("5. UCLA와 POST-ERROR SLOWING 상관")
print("=" * 80)

pes_corr_results = []

for pes_var, pes_label in [
    ('stroop_pes', 'Stroop PES'),
    ('prp_pes', 'PRP PES')
]:
    if pes_var in master_pes.columns:
        df_clean = master_pes[['ucla_total', pes_var]].dropna()

        if len(df_clean) > 10:
            r, p = pearsonr(df_clean['ucla_total'], df_clean[pes_var])
            rho, p_spear = spearmanr(df_clean['ucla_total'], df_clean[pes_var])

            print(f"\n{pes_label}:")
            print(f"  N = {len(df_clean)}")
            print(f"  Pearson: r = {r:.3f}, p = {p:.4f}")
            print(f"  Spearman: rho = {rho:.3f}, p = {p_spear:.4f}")

            if p < 0.05:
                print(f"  *** 유의함 (p < 0.05) ***")
            elif p < 0.10:
                print(f"  ** Marginal (p < 0.10) **")

            pes_corr_results.append({
                'variable': pes_label,
                'n': len(df_clean),
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spear
            })

if len(pes_corr_results) > 0:
    pes_corr_df = pd.DataFrame(pes_corr_results)
    pes_corr_df.to_csv(OUTPUT_DIR / "pes_correlations.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 7. SAVE VARIABILITY MEASURES
# =============================================================================

# Save all variability measures
variability_df = master_pes[[
    'participant_id', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
    'stroop_iiv', 'stroop_cv', 'prp_iiv', 'prp_cv', 'stroop_pes'
] + (['prp_pes'] if 'prp_pes' in master_pes.columns else [])].copy()

variability_df.to_csv(OUTPUT_DIR / "trial_variability_measures.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 8. SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n주요 발견:")

print("\n1. IIV (변동성):")
sig_iiv = iiv_corr_df[iiv_corr_df['pearson_p'] < 0.10] if len(iiv_corr_results) > 0 else pd.DataFrame()
if len(sig_iiv) > 0:
    for _, row in sig_iiv.iterrows():
        print(f"   {row['variable']}: r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f}")
else:
    print("   유의한 상관 없음 (모두 p > 0.10)")

print("\n2. Post-Error Slowing:")
if len(pes_corr_results) > 0:
    sig_pes = pes_corr_df[pes_corr_df['pearson_p'] < 0.10]
    if len(sig_pes) > 0:
        for _, row in sig_pes.iterrows():
            print(f"   {row['variable']}: r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f}")
    else:
        print("   유의한 상관 없음 (모두 p > 0.10)")
else:
    print("   PES 데이터 부족")

print("\n3. 회귀 분석 (DASS 통제):")
if len(iiv_regression_results) > 0:
    for _, row in iiv_reg_df.iterrows():
        print(f"   {row['outcome']}:")
        print(f"     UCLA (단독): β = {row['model1_ucla_beta']:.4f}, p = {row['model1_ucla_p']:.4f}")
        print(f"     UCLA (DASS 통제): β = {row['model2_ucla_beta']:.4f}, p = {row['model2_ucla_p']:.4f}")

print("\n해석:")
print("  - Trial-level 변동성과 외로움의 관계 탐색")
print("  - 평균 RT 외에 변동성/오류 후 조절도 외로움의 지표일 가능성")
print("  - 현재 데이터에서는 뚜렷한 신호 발견 어려움")

print("\n분석 완료!")
print(f"결과 저장 위치: {OUTPUT_DIR}")
print("\n생성된 파일:")
print("  - iiv_correlations.csv")
print("  - iiv_regression.csv")
if len(pes_corr_results) > 0:
    print("  - pes_correlations.csv")
print("  - trial_variability_measures.csv")
