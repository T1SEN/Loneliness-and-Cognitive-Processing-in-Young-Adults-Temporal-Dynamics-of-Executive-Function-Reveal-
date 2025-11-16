#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
나머지 가설 재현성 검증 (DASS 통제 포함)
Remaining Hypotheses Verification with DASS Controls

검증 항목:
1. PRP Ex-Gaussian: 남성 τ↑, 여성 τ↓
2. Post-Error Slowing: 남성 PES↑
3. Error Cascades: 여성 cascades↓
4. Stroop CSE: UCLA × Gender (Null)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/replication_verification")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("나머지 가설 재현성 검증 (DASS 통제 포함)")
print("Remaining Hypotheses Verification with DASS Controls")
print("=" * 80)

# =============================================================================
# 공통 데이터 로딩
# =============================================================================

print("\n데이터 로딩...")

# Master dataset
master = pd.read_csv("results/analysis_outputs/master_dataset.csv")

# Gender 정규화
def normalize_gender(val):
    if not isinstance(val, str):
        return None
    val = val.strip().lower()
    if any(x in val for x in ['남성', '남자', '남', 'male', 'm']):
        return 'male'
    elif any(x in val for x in ['여성', '여자', '여', 'female', 'f']):
        return 'female'
    else:
        return None

master['gender_normalized'] = master['gender'].apply(normalize_gender)
master['gender_male'] = (master['gender_normalized'] == 'male').astype(int)

# Z-score 표준화
def zscore(series):
    s = pd.to_numeric(series, errors='coerce')
    std = s.std(ddof=0)
    if std is None or std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std

for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    if col in master.columns:
        master[f'z_{col}'] = zscore(master[col])

print(f"  Master dataset: N={len(master)}")

# 재현 결과 저장
verification_results = []

def add_result(hypothesis, metric, claimed_r, actual_r, claimed_p, actual_p,
               claimed_beta=None, actual_beta=None, tier=2):
    """재현 결과 기록"""
    # 상관계수 비교 (r)
    if claimed_r is not None and actual_r is not None:
        r_diff_pct = abs((actual_r - claimed_r) / claimed_r * 100) if claimed_r != 0 else None
        r_match = r_diff_pct <= 20 if r_diff_pct is not None else None  # 20% 허용
        sign_match = (np.sign(claimed_r) == np.sign(actual_r))
    else:
        r_diff_pct = None
        r_match = None
        sign_match = None

    # p-value 비교
    p_match = abs(claimed_p - actual_p) <= 0.03 if (claimed_p is not None and actual_p is not None) else None

    # 전체 재현 판정
    if r_match is not None and p_match is not None and sign_match is not None:
        replicated = r_match and p_match and sign_match
    elif p_match is not None:  # r 없이 p만 있는 경우 (interaction)
        replicated = p_match
    else:
        replicated = None

    verification_results.append({
        'hypothesis': hypothesis,
        'metric': metric,
        'tier': tier,
        'claimed_r': claimed_r,
        'actual_r': actual_r,
        'claimed_p': claimed_p,
        'actual_p': actual_p,
        'claimed_beta': claimed_beta,
        'actual_beta': actual_beta,
        'r_diff_pct': r_diff_pct,
        'p_match': p_match,
        'sign_match': sign_match,
        'replicated': replicated
    })

# =============================================================================
# [1] PRP Ex-Gaussian: 남성 τ↑, 여성 τ↓
# =============================================================================

print("\n" + "=" * 80)
print("[1] PRP Ex-Gaussian: 남성 τ↑, 여성 τ↓")
print("=" * 80)

try:
    # Ex-Gaussian 파라미터 로드
    exg_path = Path("results/analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv")

    if not exg_path.exists():
        print(f"  ⚠️  파일 없음: {exg_path}")
        print("  스크립트를 먼저 실행해야 합니다:")
        print("  python analysis/prp_exgaussian_decomposition.py")
    else:
        exg_prp = pd.read_csv(exg_path)

        # BOM 제거
        if exg_prp.columns[0].startswith('\ufeff'):
            exg_prp.columns = [exg_prp.columns[0].replace('\ufeff', '')] + list(exg_prp.columns[1:])

        # Master와 병합
        master_exg = master.merge(exg_prp, on='participant_id', how='inner', suffixes=('', '_exg'))

        # UCLA 컬럼 확인
        if 'ucla_total' not in master_exg.columns and 'ucla_total_exg' in master_exg.columns:
            master_exg['ucla_total'] = master_exg['ucla_total_exg']

        print(f"  N={len(master_exg)}")

        # τ long 컬럼 찾기
        tau_cols = [c for c in master_exg.columns if 'tau' in c.lower() and 'long' in c.lower()]

        if not tau_cols:
            print(f"  사용 가능한 컬럼: {[c for c in master_exg.columns if 'tau' in c.lower()]}")
            tau_col = 'long_tau' if 'long_tau' in master_exg.columns else None
        else:
            tau_col = tau_cols[0]

        if tau_col and tau_col in master_exg.columns:
            # 필수 컬럼 확인
            required = ['ucla_total', 'gender_male', 'dass_depression', 'dass_anxiety', 'dass_stress', tau_col]
            master_exg_clean = master_exg.dropna(subset=required).copy()

            # Z-scores
            for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']:
                master_exg_clean[f'z_{col}'] = zscore(master_exg_clean[col])

            # 성별 분리
            males = master_exg_clean[master_exg_clean['gender_male'] == 1]
            females = master_exg_clean[master_exg_clean['gender_male'] == 0]

            print(f"  남성: N={len(males)}, 여성: N={len(females)}")

            # === Method 1: Simple Correlation (원본 방식) ===
            print("\n  [Method 1] Simple Correlation (DASS 통제 없음)")

            if len(males) >= 10:
                r_male_simple, p_male_simple = pearsonr(males['ucla_total'], males[tau_col])
                print(f"    남성: r={r_male_simple:.3f}, p={p_male_simple:.4f}")

            if len(females) >= 10:
                r_female_simple, p_female_simple = pearsonr(females['ucla_total'], females[tau_col])
                print(f"    여성: r={r_female_simple:.3f}, p={p_female_simple:.4f}")

            # === Method 2: Partial Correlation (DASS 통제) ===
            print("\n  [Method 2] Partial Correlation (DASS 통제 포함)")

            # 남성: Partial correlation
            if len(males) >= 15:
                formula_male = f"{tau_col} ~ z_ucla_total + z_dass_depression + z_dass_anxiety + z_dass_stress"
                try:
                    model_male = smf.ols(formula_male, data=males).fit(cov_type='HC3')
                    beta_male = model_male.params['z_ucla_total']
                    p_male = model_male.pvalues['z_ucla_total']

                    # Partial r 계산 (beta를 r로 근사)
                    # r_partial ≈ t / sqrt(t^2 + df)
                    t_male = model_male.tvalues['z_ucla_total']
                    df_male = model_male.df_resid
                    r_male_partial = t_male / np.sqrt(t_male**2 + df_male)

                    print(f"    남성 (partial r): r={r_male_partial:.3f}, p={p_male:.4f}, β={beta_male:.2f}")
                    print(f"    주장: r≈0.578, p≈0.002")

                    add_result(
                        hypothesis="PRP Ex-Gaussian: Male τ",
                        metric="Male UCLA×τ_long partial correlation",
                        claimed_r=0.578,
                        actual_r=r_male_partial,
                        claimed_p=0.002,
                        actual_p=p_male,
                        tier=1
                    )
                except Exception as e:
                    print(f"    남성 모델 오류: {e}")

            # 여성: Partial correlation
            if len(females) >= 15:
                formula_female = f"{tau_col} ~ z_ucla_total + z_dass_depression + z_dass_anxiety + z_dass_stress"
                try:
                    model_female = smf.ols(formula_female, data=females).fit(cov_type='HC3')
                    beta_female = model_female.params['z_ucla_total']
                    p_female = model_female.pvalues['z_ucla_total']

                    t_female = model_female.tvalues['z_ucla_total']
                    df_female = model_female.df_resid
                    r_female_partial = t_female / np.sqrt(t_female**2 + df_female)

                    print(f"    여성 (partial r): r={r_female_partial:.3f}, p={p_female:.4f}, β={beta_female:.2f}")
                    print(f"    주장: r≈-0.384, p≈0.009")

                    add_result(
                        hypothesis="PRP Ex-Gaussian: Female τ",
                        metric="Female UCLA×τ_long partial correlation",
                        claimed_r=-0.384,
                        actual_r=r_female_partial,
                        claimed_p=0.009,
                        actual_p=p_female,
                        tier=1
                    )
                except Exception as e:
                    print(f"    여성 모델 오류: {e}")
        else:
            print(f"  τ_long 컬럼을 찾을 수 없음")

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# [2] Post-Error Slowing: 남성 PES↑
# =============================================================================

print("\n" + "=" * 80)
print("[2] Post-Error Slowing: 남성 PES↑ (비효율적)")
print("=" * 80)

try:
    # WCST trial 데이터 로드 (fresh copy)
    wcst_trials_pes = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

    # participantId와 participant_id 중복 제거
    if 'participantId' in wcst_trials_pes.columns and 'participant_id' in wcst_trials_pes.columns:
        wcst_trials_pes = wcst_trials_pes.drop(columns=['participantId'])
    elif 'participantId' in wcst_trials_pes.columns:
        wcst_trials_pes = wcst_trials_pes.rename(columns={'participantId': 'participant_id'})

    # RT 컬럼
    rt_col = 'rt_ms' if 'rt_ms' in wcst_trials_pes.columns else 'reactionTimeMs'

    # Previous trial correct
    wcst_trials_pes = wcst_trials_pes.sort_values(['participant_id', 'trialIndex'])
    wcst_trials_pes['prev_correct'] = wcst_trials_pes.groupby('participant_id')['correct'].shift(1)

    # PES 계산
    pes_results = []
    for pid, group in wcst_trials_pes.groupby('participant_id'):
        post_correct = group[group['prev_correct'] == True][rt_col].mean()
        post_error = group[group['prev_correct'] == False][rt_col].mean()

        n_post_correct = (group['prev_correct'] == True).sum()
        n_post_error = (group['prev_correct'] == False).sum()

        if pd.notna(post_correct) and pd.notna(post_error) and n_post_error >= 3:
            pes = post_error - post_correct
            pes_results.append({
                'participant_id': pid,
                'wcst_pes': pes,
                'post_correct_rt': post_correct,
                'post_error_rt': post_error,
                'n_post_error': n_post_error
            })

    pes_df = pd.DataFrame(pes_results)
    master_pes = master.merge(pes_df, on='participant_id', how='inner')

    print(f"  N={len(master_pes)}")

    # 필수 컬럼
    required = ['ucla_total', 'gender_male', 'dass_depression', 'dass_anxiety', 'dass_stress', 'wcst_pes']
    master_pes = master_pes.dropna(subset=required).copy()

    # Z-scores
    for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']:
        master_pes[f'z_{col}'] = zscore(master_pes[col])

    # 성별 분리
    males_pes = master_pes[master_pes['gender_male'] == 1]
    females_pes = master_pes[master_pes['gender_male'] == 0]

    print(f"  남성: N={len(males_pes)}, 여성: N={len(females_pes)}")

    # === Method 1: Simple Correlation ===
    print("\n  [Method 1] Simple Correlation (DASS 통제 없음)")

    if len(males_pes) >= 10:
        r_male_pes_simple, p_male_pes_simple = pearsonr(males_pes['ucla_total'], males_pes['wcst_pes'])
        print(f"    남성: r={r_male_pes_simple:.3f}, p={p_male_pes_simple:.4f}")

    # === Method 2: Partial Correlation ===
    print("\n  [Method 2] Partial Correlation (DASS 통제 포함)")

    if len(males_pes) >= 15:
        formula_pes = "wcst_pes ~ z_ucla_total + z_dass_depression + z_dass_anxiety + z_dass_stress"
        model_pes = smf.ols(formula_pes, data=males_pes).fit(cov_type='HC3')

        beta_pes = model_pes.params['z_ucla_total']
        p_pes = model_pes.pvalues['z_ucla_total']

        t_pes = model_pes.tvalues['z_ucla_total']
        df_pes = model_pes.df_resid
        r_pes_partial = t_pes / np.sqrt(t_pes**2 + df_pes)

        print(f"    남성 (partial r): r={r_pes_partial:.3f}, p={p_pes:.4f}, β={beta_pes:.2f}")
        print(f"    주장: r≈0.422, p≈0.018")

        add_result(
            hypothesis="Post-Error Slowing: Male",
            metric="Male UCLA×PES partial correlation",
            claimed_r=0.422,
            actual_r=r_pes_partial,
            claimed_p=0.018,
            actual_p=p_pes,
            tier=2
        )

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# [3] Error Cascades: 여성 cascades↓ (보호 효과)
# =============================================================================

print("\n" + "=" * 80)
print("[3] Error Cascades: 여성 cascades↓ (Hypervigilance)")
print("=" * 80)

try:
    import ast

    # WCST trials (fresh load)
    wcst_trials_casc = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

    # participantId와 participant_id 중복 제거
    if 'participantId' in wcst_trials_casc.columns and 'participant_id' in wcst_trials_casc.columns:
        wcst_trials_casc = wcst_trials_casc.drop(columns=['participantId'])
    elif 'participantId' in wcst_trials_casc.columns:
        wcst_trials_casc = wcst_trials_casc.rename(columns={'participantId': 'participant_id'})

    # Parse extra for isPE
    wcst_trials_casc['extra_dict'] = wcst_trials_casc['extra'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
    )
    wcst_trials_casc['is_pe'] = wcst_trials_casc['extra_dict'].apply(lambda x: x.get('isPE', False))

    # Error cascades 계산
    cascade_results = []

    for pid, group in wcst_trials_casc.groupby('participant_id'):
        group = group.sort_values('trialIndex').reset_index(drop=True)

        # PE run lengths
        pe_runs = []
        current_run = 0
        for is_pe in group['is_pe']:
            if is_pe:
                current_run += 1
            else:
                if current_run > 1:  # cascade = 2+ consecutive PEs
                    pe_runs.append(current_run)
                current_run = 0
        if current_run > 1:
            pe_runs.append(current_run)

        mean_cascade = np.mean(pe_runs) if pe_runs else 0
        max_cascade = np.max(pe_runs) if pe_runs else 0
        n_cascades = len(pe_runs)

        cascade_results.append({
            'participant_id': pid,
            'mean_pe_cascade': mean_cascade,
            'max_pe_cascade': max_cascade,
            'n_cascades': n_cascades
        })

    cascade_df = pd.DataFrame(cascade_results)
    master_cascade = master.merge(cascade_df, on='participant_id', how='inner')

    print(f"  N={len(master_cascade)}")

    # 필수 컬럼
    required = ['ucla_total', 'gender_male', 'dass_depression', 'dass_anxiety', 'dass_stress', 'mean_pe_cascade']
    master_cascade = master_cascade.dropna(subset=required).copy()

    # Z-scores
    for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']:
        master_cascade[f'z_{col}'] = zscore(master_cascade[col])

    # 성별 분리
    males_casc = master_cascade[master_cascade['gender_male'] == 1]
    females_casc = master_cascade[master_cascade['gender_male'] == 0]

    print(f"  남성: N={len(males_casc)}, 여성: N={len(females_casc)}")

    # === Method 1: Simple Correlation ===
    print("\n  [Method 1] Simple Correlation (DASS 통제 없음)")

    if len(females_casc) >= 10:
        r_female_casc_simple, p_female_casc_simple = pearsonr(
            females_casc['ucla_total'],
            females_casc['mean_pe_cascade']
        )
        print(f"    여성: r={r_female_casc_simple:.3f}, p={p_female_casc_simple:.4f}")

    # === Method 2: Partial Correlation ===
    print("\n  [Method 2] Partial Correlation (DASS 통제 포함)")

    if len(females_casc) >= 15:
        formula_casc = "mean_pe_cascade ~ z_ucla_total + z_dass_depression + z_dass_anxiety + z_dass_stress"
        model_casc = smf.ols(formula_casc, data=females_casc).fit(cov_type='HC3')

        beta_casc = model_casc.params['z_ucla_total']
        p_casc = model_casc.pvalues['z_ucla_total']

        t_casc = model_casc.tvalues['z_ucla_total']
        df_casc = model_casc.df_resid
        r_casc_partial = t_casc / np.sqrt(t_casc**2 + df_casc)

        print(f"    여성 (partial r): r={r_casc_partial:.3f}, p={p_casc:.4f}, β={beta_casc:.3f}")
        print(f"    주장: r≈-0.389, p≈0.007")

        add_result(
            hypothesis="Error Cascades: Female",
            metric="Female UCLA×Cascades partial correlation",
            claimed_r=-0.389,
            actual_r=r_casc_partial,
            claimed_p=0.007,
            actual_p=p_casc,
            tier=2
        )

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# [4] Stroop CSE: UCLA × Gender (Null 예상)
# =============================================================================

print("\n" + "=" * 80)
print("[4] Stroop CSE: UCLA × Gender (동적 조절, Null 예상)")
print("=" * 80)

try:
    # Stroop trials 로드 (fresh)
    stroop_trials_cse = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")

    # participantId와 participant_id 중복 제거
    if 'participantId' in stroop_trials_cse.columns and 'participant_id' in stroop_trials_cse.columns:
        stroop_trials_cse = stroop_trials_cse.drop(columns=['participantId'])
    elif 'participantId' in stroop_trials_cse.columns:
        stroop_trials_cse = stroop_trials_cse.rename(columns={'participantId': 'participant_id'})

    # 필터링
    stroop_valid = stroop_trials_cse[
        (stroop_trials_cse['timeout'] == False) &
        (stroop_trials_cse['correct'] == True) &
        (stroop_trials_cse['rt'] >= 200) &
        (stroop_trials_cse['rt'] <= 3000)
    ].copy()

    # type 컬럼
    type_col = 'type' if 'type' in stroop_valid.columns else 'trialType'

    # Congruent/Incongruent만
    stroop_valid = stroop_valid[stroop_valid[type_col].isin(['congruent', 'incongruent'])].copy()

    # Previous trial type
    stroop_valid = stroop_valid.sort_values(['participant_id', 'trialIndex'])
    stroop_valid['is_incongruent'] = (stroop_valid[type_col] == 'incongruent').astype(int)
    stroop_valid['prev_is_incongruent'] = stroop_valid.groupby('participant_id')['is_incongruent'].shift(1)

    # Drop first trials
    cse_data = stroop_valid.dropna(subset=['prev_is_incongruent']).copy()

    # CSE 계산
    cse_results = []

    for pid, group in cse_data.groupby('participant_id'):
        # 4 cells
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
                'stroop_cse': cse,
                'n_trials': len(group)
            })

    cse_df = pd.DataFrame(cse_results)
    master_cse = master.merge(cse_df, on='participant_id', how='inner')

    print(f"  N={len(master_cse)}")

    # 필수 컬럼
    required = ['ucla_total', 'gender_male', 'dass_depression', 'dass_anxiety', 'dass_stress', 'stroop_cse']
    master_cse = master_cse.dropna(subset=required).copy()

    # Z-scores
    for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']:
        master_cse[f'z_{col}'] = zscore(master_cse[col])

    print(f"  분석 샘플: N={len(master_cse)}")

    # === Interaction Model (DASS 통제 없음) ===
    print("\n  [Method 1] Interaction Model (DASS 통제 없음)")

    formula_cse_simple = "stroop_cse ~ z_ucla_total * C(gender_male)"
    model_cse_simple = smf.ols(formula_cse_simple, data=master_cse).fit()
    beta_cse_simple = model_cse_simple.params.get('z_ucla_total:C(gender_male)[T.1]', np.nan)
    p_cse_simple = model_cse_simple.pvalues.get('z_ucla_total:C(gender_male)[T.1]', np.nan)

    print(f"    UCLA × Gender: β={beta_cse_simple:.3f}, p={p_cse_simple:.4f}")

    # === Interaction Model (DASS 통제 포함) ===
    print("\n  [Method 2] Interaction Model (DASS 통제 포함)")

    formula_cse = "stroop_cse ~ z_ucla_total * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress"
    model_cse = smf.ols(formula_cse, data=master_cse).fit(cov_type='HC3')

    beta_cse = model_cse.params.get('z_ucla_total:C(gender_male)[T.1]', np.nan)
    p_cse = model_cse.pvalues.get('z_ucla_total:C(gender_male)[T.1]', np.nan)

    print(f"    UCLA × Gender: β={beta_cse:.3f}, p={p_cse:.4f}")
    print(f"    주장: p≈0.209 (NS)")

    add_result(
        hypothesis="Stroop CSE (Null)",
        metric="UCLA×Gender→CSE interaction",
        claimed_r=None,
        actual_r=None,
        claimed_p=0.209,
        actual_p=p_cse,
        claimed_beta=None,
        actual_beta=beta_cse,
        tier=3
    )

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 최종 요약
# =============================================================================

print("\n" + "=" * 80)
print("재현성 검증 요약 (DASS 통제 포함)")
print("=" * 80)

results_df = pd.DataFrame(verification_results)

if len(results_df) > 0:
    results_df.to_csv(OUTPUT_DIR / "remaining_hypotheses_results.csv", index=False, encoding='utf-8-sig')

    # 요약
    total = len(results_df)
    replicated = results_df['replicated'].sum()
    failed = (results_df['replicated'] == False).sum()
    unclear = results_df['replicated'].isna().sum()

    print(f"\n총 검증 항목: {total}")
    print(f"  ✅ 재현 성공: {replicated} ({replicated/total*100:.1f}%)")
    print(f"  ❌ 재현 실패: {failed} ({failed/total*100:.1f}%)")
    print(f"  ⚠️  판정 불가: {unclear} ({unclear/total*100:.1f}%)")

    # 상세 출력
    print("\n상세 결과:")
    for _, row in results_df.iterrows():
        status = "✅" if row['replicated'] == True else ("❌" if row['replicated'] == False else "⚠️")
        print(f"\n{status} {row['hypothesis']}")

        if row['claimed_r'] is not None:
            print(f"  주장: r={row['claimed_r']:.3f}, p={row['claimed_p']:.4f}")
            print(f"  실제: r={row['actual_r']:.3f}, p={row['actual_p']:.4f}")
            if row['r_diff_pct'] is not None:
                print(f"  차이: r {row['r_diff_pct']:.1f}%, p-diff={abs(row['claimed_p']-row['actual_p']):.4f}")
        else:
            print(f"  주장: p={row['claimed_p']:.4f}")
            print(f"  실제: p={row['actual_p']:.4f}")

    # 재현 실패 분석
    if failed > 0:
        print("\n⚠️  재현 실패 항목:")
        failures = results_df[results_df['replicated'] == False]
        for _, row in failures.iterrows():
            print(f"  - {row['hypothesis']}: {row['metric']}")

print(f"\n✅ 결과 저장: {OUTPUT_DIR / 'remaining_hypotheses_results.csv'}")

print("\n" + "=" * 80)
print("검증 완료!")
print("=" * 80)
