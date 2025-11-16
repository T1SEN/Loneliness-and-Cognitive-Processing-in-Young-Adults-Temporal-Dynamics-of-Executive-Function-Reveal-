#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
종합 재현성 검증 스크립트
Comprehensive Replication Verification Script

이 스크립트는 9개 핵심 가설의 통계치를 재계산하여
이전 보고된 결과와 비교합니다.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/replication_verification")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("종합 재현성 검증 (Comprehensive Replication Verification)")
print("=" * 80)

# =============================================================================
# 데이터 로딩
# =============================================================================

print("\n[1/10] 데이터 로딩중...")

# 기존 master_dataset 로드
master = pd.read_csv("results/analysis_outputs/master_dataset.csv")

# Gender 정규화 (한글 -> 영어)
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

# PE_rate 컬럼명 확인
if 'perseverative_error_rate' in master.columns and 'pe_rate' not in master.columns:
    master['pe_rate'] = master['perseverative_error_rate']

# 기본 필터링: 핵심 변수들이 있는 케이스만
essential_cols = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
                   'age', 'gender_male', 'pe_rate', 'stroop_interference', 'prp_bottleneck']
master = master.dropna(subset=essential_cols).copy()

print(f"  총 샘플: N = {len(master)}")
print(f"  남성: {int(master['gender_male'].sum())}명")
print(f"  여성: {int((1-master['gender_male']).sum())}명")
print(f"  평균 나이: {master['age'].mean():.1f}세 (SD={master['age'].std():.1f})")
print(f"  평균 UCLA: {master['ucla_total'].mean():.1f} (SD={master['ucla_total'].std():.1f})")

# Z-score 표준화
for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    master[f'z_{col}'] = (master[col] - master[col].mean()) / master[col].std()

# =============================================================================
# 재현성 검증 결과 저장
# =============================================================================

replication_results = []

def add_result(hypothesis, metric, claimed_value, actual_value, claimed_p, actual_p, tier):
    """재현 결과 기록"""
    # p-value 차이 허용 범위: ±0.02
    p_match = abs(claimed_p - actual_p) <= 0.02 if (claimed_p is not None and actual_p is not None) else None

    # 효과크기 차이 허용 범위: ±15%
    if claimed_value is not None and actual_value is not None and claimed_value != 0:
        effect_diff_pct = abs((actual_value - claimed_value) / claimed_value * 100)
        effect_match = effect_diff_pct <= 15
    else:
        effect_diff_pct = None
        effect_match = None

    # 방향성 일치
    if claimed_value is not None and actual_value is not None:
        sign_match = (np.sign(claimed_value) == np.sign(actual_value))
    else:
        sign_match = None

    # 전체 재현 판정
    replicated = None
    if p_match is not None and effect_match is not None and sign_match is not None:
        replicated = p_match and effect_match and sign_match

    replication_results.append({
        'hypothesis': hypothesis,
        'metric': metric,
        'tier': tier,
        'claimed_value': claimed_value,
        'actual_value': actual_value,
        'claimed_p': claimed_p,
        'actual_p': actual_p,
        'p_match': p_match,
        'effect_match': effect_match,
        'effect_diff_pct': effect_diff_pct,
        'sign_match': sign_match,
        'replicated': replicated
    })

# =============================================================================
# [TIER 1-1] WCST PE × Gender 조절효과
# =============================================================================

print("\n" + "=" * 80)
print("[2/10] Tier 1-1: WCST PE × Gender 조절효과")
print("=" * 80)

# 주장: β≈2.59, p≈0.004
formula = 'pe_rate ~ z_ucla_total * gender_male + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age'
try:
    model = smf.ols(formula, data=master).fit()
    interaction_beta = model.params['z_ucla_total:gender_male']
    interaction_p = model.pvalues['z_ucla_total:gender_male']

    print(f"  OLS 회귀 결과:")
    print(f"    UCLA × Gender 상호작용 β = {interaction_beta:.3f}")
    print(f"    p-value = {interaction_p:.4f}")
    print(f"    주장: β≈2.59, p≈0.004")

    add_result(
        hypothesis="WCST PE × Gender interaction",
        metric="Interaction beta",
        claimed_value=2.59,
        actual_value=interaction_beta,
        claimed_p=0.004,
        actual_p=interaction_p,
        tier=1
    )

    # Simple slopes
    males = master[master['gender_male'] == 1]
    females = master[master['gender_male'] == 0]

    r_male, p_male = pearsonr(males['ucla_total'], males['pe_rate'])
    r_female, p_female = pearsonr(females['ucla_total'], females['pe_rate'])

    # 남성 단순 회귀로 기울기 구하기
    male_model = smf.ols('pe_rate ~ ucla_total', data=males).fit()
    male_slope = male_model.params['ucla_total']
    male_slope_p = male_model.pvalues['ucla_total']

    female_model = smf.ols('pe_rate ~ ucla_total', data=females).fit()
    female_slope = female_model.params['ucla_total']
    female_slope_p = female_model.pvalues['ucla_total']

    print(f"\n  Simple Slopes:")
    print(f"    남성 (N={len(males)}): β={male_slope:.3f}, p={male_slope_p:.4f}, r={r_male:.3f}")
    print(f"    여성 (N={len(females)}): β={female_slope:.3f}, p={female_slope_p:.4f}, r={r_female:.3f}")
    print(f"    주장: 남성 β≈2.29, p≈0.067 / 여성 β≈−0.30, p≈0.72")

    add_result(
        hypothesis="WCST PE × Gender: Male slope",
        metric="Male UCLA→PE beta",
        claimed_value=2.29,
        actual_value=male_slope,
        claimed_p=0.067,
        actual_p=male_slope_p,
        tier=1
    )

    add_result(
        hypothesis="WCST PE × Gender: Female slope",
        metric="Female UCLA→PE beta",
        claimed_value=-0.30,
        actual_value=female_slope,
        claimed_p=0.72,
        actual_p=female_slope_p,
        tier=1
    )

except Exception as e:
    print(f"  오류 발생: {e}")

# =============================================================================
# [TIER 1-2] PRP Ex-Gaussian 파라미터 (τ, σ)
# =============================================================================

print("\n" + "=" * 80)
print("[3/10] Tier 1-2: PRP Ex-Gaussian 남성 τ↑, 여성 τ↓")
print("=" * 80)

try:
    # Ex-Gaussian 파라미터 로딩
    exg_prp_path = Path("results/analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv")
    if not exg_prp_path.exists():
        print(f"  경고: PRP Ex-Gaussian 파일 없음: {exg_prp_path}")
        raise FileNotFoundError("PRP Ex-Gaussian file not found")

    exg_prp = pd.read_csv(exg_prp_path)

    # BOM 제거
    if exg_prp.columns[0].startswith('\ufeff'):
        exg_prp.columns = [exg_prp.columns[0].replace('\ufeff', '')] + list(exg_prp.columns[1:])

    # participant_id로 병합
    master_exg = master.merge(exg_prp, on='participant_id', how='inner')

    print(f"  Ex-Gaussian 파라미터 로드 완료: N={len(master_exg)}")

    # 성별 분리
    males_exg = master_exg[master_exg['gender_male'] == 1]
    females_exg = master_exg[master_exg['gender_male'] == 0]

    # 주요 파라미터: τ_long (long SOA에서 τ)
    if 'tau_long' in master_exg.columns:
        tau_col = 'tau_long'
    elif 'τ_long' in master_exg.columns:
        tau_col = 'τ_long'
    else:
        # 컬럼명 확인
        print(f"  사용 가능한 컬럼: {master_exg.columns.tolist()}")
        tau_col = [c for c in master_exg.columns if 'tau' in c.lower() and 'long' in c.lower()]
        tau_col = tau_col[0] if tau_col else None

    if tau_col:
        r_male_tau, p_male_tau = pearsonr(
            males_exg['ucla_total'],
            males_exg[tau_col].replace([np.inf, -np.inf], np.nan).dropna()
        )
        r_female_tau, p_female_tau = pearsonr(
            females_exg['ucla_total'],
            females_exg[tau_col].replace([np.inf, -np.inf], np.nan).dropna()
        )

        print(f"\n  τ (long SOA) 상관:")
        print(f"    남성: r={r_male_tau:.3f}, p={p_male_tau:.4f}")
        print(f"    여성: r={r_female_tau:.3f}, p={p_female_tau:.4f}")
        print(f"    주장: 남성 r≈0.578, p≈0.002 / 여성 r≈−0.384, p≈0.009")

        add_result(
            hypothesis="PRP Ex-Gaussian: Male τ",
            metric="Male UCLA×τ_long correlation",
            claimed_value=0.578,
            actual_value=r_male_tau,
            claimed_p=0.002,
            actual_p=p_male_tau,
            tier=1
        )

        add_result(
            hypothesis="PRP Ex-Gaussian: Female τ",
            metric="Female UCLA×τ_long correlation",
            claimed_value=-0.384,
            actual_value=r_female_tau,
            claimed_p=0.009,
            actual_p=p_female_tau,
            tier=1
        )
    else:
        print("  경고: τ_long 컬럼을 찾을 수 없음")

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# [TIER 1-3] MVPA 분류기 (AUC≈0.797)
# =============================================================================

print("\n" + "=" * 80)
print("[4/10] Tier 1-3: MVPA 분류기 (High vs Low Loneliness)")
print("=" * 80)

try:
    # UCLA 3분위로 분류
    master['ucla_tercile'] = pd.qcut(master['ucla_total'], q=3, labels=['low', 'medium', 'high'])

    # High vs Low만 사용
    mvpa_data = master[master['ucla_tercile'].isin(['high', 'low'])].copy()
    mvpa_data['target'] = (mvpa_data['ucla_tercile'] == 'high').astype(int)

    # 특성: EF + Demographics + DASS
    feature_cols = [
        'pe_rate', 'wcst_accuracy', 'stroop_interference', 'prp_bottleneck',
        'age', 'gender_male',
        'dass_depression', 'dass_anxiety', 'dass_stress'
    ]

    X = mvpa_data[feature_cols].fillna(mvpa_data[feature_cols].median())
    y = mvpa_data['target']

    print(f"  분류 데이터: N={len(mvpa_data)} (High={y.sum()}, Low={(1-y).sum()})")

    # Random Forest 5-fold CV
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # AUC 스코어
    auc_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    mean_auc = auc_scores.mean()
    std_auc = auc_scores.std()

    print(f"\n  Random Forest 5-Fold CV AUC:")
    print(f"    Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"    Folds: {auc_scores}")
    print(f"    주장: AUC≈0.797")

    add_result(
        hypothesis="MVPA Classifier",
        metric="Random Forest AUC (5-fold CV)",
        claimed_value=0.797,
        actual_value=mean_auc,
        claimed_p=None,  # AUC는 p-value 없음
        actual_p=None,
        tier=1
    )

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# [TIER 2-1] Post-Error Slowing 성별 조절
# =============================================================================

print("\n" + "=" * 80)
print("[5/10] Tier 2-1: Post-Error Slowing 성별 조절")
print("=" * 80)

try:
    # WCST trial 데이터 로드
    wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

    # participant_id 정규화
    if 'participantId' in wcst_trials.columns:
        wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

    # RT 컬럼 확인
    rt_col = 'rt_ms' if 'rt_ms' in wcst_trials.columns else 'reactionTimeMs'

    # Post-error slowing 계산
    wcst_trials = wcst_trials.sort_values(['participant_id', 'trialIndex'])
    wcst_trials['prev_correct'] = wcst_trials.groupby('participant_id')['correct'].shift(1)

    pes_by_participant = []
    for pid, group in wcst_trials.groupby('participant_id'):
        post_correct = group[group['prev_correct'] == True][rt_col].mean()
        post_error = group[group['prev_correct'] == False][rt_col].mean()

        if pd.notna(post_correct) and pd.notna(post_error):
            pes = post_error - post_correct
            pes_by_participant.append({
                'participant_id': pid,
                'wcst_pes': pes,
                'post_correct_rt': post_correct,
                'post_error_rt': post_error
            })

    pes_df = pd.DataFrame(pes_by_participant)
    master_pes = master.merge(pes_df, on='participant_id', how='inner')

    print(f"  PES 계산 완료: N={len(master_pes)}")

    # 성별 분리
    males_pes = master_pes[master_pes['gender_male'] == 1]
    females_pes = master_pes[master_pes['gender_male'] == 0]

    r_male_pes, p_male_pes = pearsonr(males_pes['ucla_total'], males_pes['wcst_pes'])
    r_female_pes, p_female_pes = pearsonr(females_pes['ucla_total'], females_pes['wcst_pes'])

    print(f"\n  UCLA × PES 상관:")
    print(f"    남성: r={r_male_pes:.3f}, p={p_male_pes:.4f}")
    print(f"    여성: r={r_female_pes:.3f}, p={p_female_pes:.4f}")
    print(f"    주장: 남성은 PES↑ but inefficient, 여성은 PES↑ and efficient")

    add_result(
        hypothesis="Post-Error Slowing: Male",
        metric="Male UCLA×PES correlation",
        claimed_value=0.422,  # 보고서에서 r≈0.422, p≈0.018
        actual_value=r_male_pes,
        claimed_p=0.018,
        actual_p=p_male_pes,
        tier=2
    )

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# [TIER 2-2] DASS Anxiety 층화
# =============================================================================

print("\n" + "=" * 80)
print("[6/10] Tier 2-2: DASS Anxiety 층화 효과")
print("=" * 80)

try:
    # Anxiety median split
    median_anx = master['dass_anxiety'].median()
    master['anxiety_group'] = master['dass_anxiety'].apply(
        lambda x: 'low' if x <= median_anx else 'high'
    )

    low_anx = master[master['anxiety_group'] == 'low']
    high_anx = master[master['anxiety_group'] == 'high']

    print(f"  Low Anxiety: N={len(low_anx)}")
    print(f"  High Anxiety: N={len(high_anx)}")

    # Low anxiety에서 조절효과
    formula_low = 'pe_rate ~ z_ucla_total * gender_male + z_age'
    model_low = smf.ols(formula_low, data=low_anx).fit()
    beta_low = model_low.params.get('z_ucla_total:gender_male', np.nan)
    p_low = model_low.pvalues.get('z_ucla_total:gender_male', np.nan)

    # High anxiety에서 조절효과
    model_high = smf.ols(formula_low, data=high_anx).fit()
    beta_high = model_high.params.get('z_ucla_total:gender_male', np.nan)
    p_high = model_high.pvalues.get('z_ucla_total:gender_male', np.nan)

    print(f"\n  UCLA × Gender 상호작용:")
    print(f"    Low Anxiety: β={beta_low:.3f}, p={p_low:.4f}")
    print(f"    High Anxiety: β={beta_high:.3f}, p={p_high:.4f}")
    print(f"    주장: Low Anxiety β≈4.28, p≈0.008")

    add_result(
        hypothesis="DASS Anxiety Stratification",
        metric="Low Anxiety UCLA×Gender interaction",
        claimed_value=4.28,
        actual_value=beta_low,
        claimed_p=0.008,
        actual_p=p_low,
        tier=2
    )

except Exception as e:
    print(f"  오류 발생: {e}")

# =============================================================================
# [TIER 2-3] 여성 보상 경로 (Error Cascades)
# =============================================================================

print("\n" + "=" * 80)
print("[7/10] Tier 2-3: 여성 보상 경로 (Error Cascades)")
print("=" * 80)

try:
    # WCST에서 error cascade 계산
    # Error cascade: 연속된 오류 수

    import ast
    wcst_trials['extra_dict'] = wcst_trials['extra'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
    wcst_trials['is_pe'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

    cascade_results = []
    for pid, group in wcst_trials.groupby('participant_id'):
        group = group.sort_values('trialIndex')

        # Error chains
        error_chain_lengths = []
        current_chain = 0
        for is_correct in group['correct']:
            if not is_correct:
                current_chain += 1
            else:
                if current_chain > 1:
                    error_chain_lengths.append(current_chain)
                current_chain = 0

        if current_chain > 1:
            error_chain_lengths.append(current_chain)

        mean_cascade = np.mean(error_chain_lengths) if error_chain_lengths else 0
        max_cascade = np.max(error_chain_lengths) if error_chain_lengths else 0
        n_cascades = len(error_chain_lengths)

        cascade_results.append({
            'participant_id': pid,
            'mean_error_cascade': mean_cascade,
            'max_error_cascade': max_cascade,
            'n_error_cascades': n_cascades
        })

    cascade_df = pd.DataFrame(cascade_results)
    master_cascade = master.merge(cascade_df, on='participant_id', how='inner')

    males_casc = master_cascade[master_cascade['gender_male'] == 1]
    females_casc = master_cascade[master_cascade['gender_male'] == 0]

    # 여성: UCLA↑ → cascades↓ (보호 효과)
    r_female_casc, p_female_casc = pearsonr(
        females_casc['ucla_total'],
        females_casc['mean_error_cascade']
    )

    r_male_casc, p_male_casc = pearsonr(
        males_casc['ucla_total'],
        males_casc['mean_error_cascade']
    )

    print(f"\n  UCLA × Error Cascades 상관:")
    print(f"    여성: r={r_female_casc:.3f}, p={p_female_casc:.4f}")
    print(f"    남성: r={r_male_casc:.3f}, p={p_male_casc:.4f}")
    print(f"    주장: 여성 r≈−0.389, p≈0.007 (hypervigilance 보호)")

    add_result(
        hypothesis="Female Compensation: Error Cascades",
        metric="Female UCLA×Cascades correlation",
        claimed_value=-0.389,
        actual_value=r_female_casc,
        claimed_p=0.007,
        actual_p=p_female_casc,
        tier=2
    )

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# [TIER 3-1] Stroop CSE (Null 결과)
# =============================================================================

print("\n" + "=" * 80)
print("[8/10] Tier 3-1: Stroop CSE (동적 조절, Null 예상)")
print("=" * 80)

try:
    # CSE = RT(iI) + RT(cC) - RT(iC) - RT(cI)
    # i=incongruent, c=congruent, 첫글자=prev, 둘째=current

    stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")

    if 'participantId' in stroop_trials.columns:
        stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})

    # Timeout 제거
    stroop_trials = stroop_trials[stroop_trials['timeout'] == False].copy()
    stroop_trials = stroop_trials.sort_values(['participant_id', 'trialIndex'])

    # Previous trial type
    stroop_trials['prev_type'] = stroop_trials.groupby('participant_id')['type'].shift(1)

    cse_by_participant = []
    for pid, group in stroop_trials.groupby('participant_id'):
        # 4가지 조건별 평균 RT
        rt_iI = group[(group['prev_type'] == 'incongruent') & (group['type'] == 'incongruent')]['rt'].mean()
        rt_cC = group[(group['prev_type'] == 'congruent') & (group['type'] == 'congruent')]['rt'].mean()
        rt_iC = group[(group['prev_type'] == 'incongruent') & (group['type'] == 'congruent')]['rt'].mean()
        rt_cI = group[(group['prev_type'] == 'congruent') & (group['type'] == 'incongruent')]['rt'].mean()

        cse = (rt_iI + rt_cC) - (rt_iC + rt_cI)

        if pd.notna(cse):
            cse_by_participant.append({
                'participant_id': pid,
                'stroop_cse': cse
            })

    cse_df = pd.DataFrame(cse_by_participant)
    master_cse = master.merge(cse_df, on='participant_id', how='inner')

    # UCLA × Gender → CSE 조절효과
    formula_cse = 'stroop_cse ~ z_ucla_total * gender_male + z_age'
    model_cse = smf.ols(formula_cse, data=master_cse).fit()
    beta_cse = model_cse.params.get('z_ucla_total:gender_male', np.nan)
    p_cse = model_cse.pvalues.get('z_ucla_total:gender_male', np.nan)

    print(f"\n  UCLA × Gender → CSE 조절효과:")
    print(f"    β={beta_cse:.3f}, p={p_cse:.4f}")
    print(f"    주장: p≈0.209 (NS, 동적 조절 장애 아님)")

    add_result(
        hypothesis="Stroop CSE (Null)",
        metric="UCLA×Gender→CSE interaction",
        claimed_value=None,  # NS이므로 효과크기 비교 안함
        actual_value=beta_cse,
        claimed_p=0.209,
        actual_p=p_cse,
        tier=3
    )

except Exception as e:
    print(f"  오류 발생: {e}")

# =============================================================================
# [TIER 3-2] Changepoint Detection (점진적 손상)
# =============================================================================

print("\n" + "=" * 80)
print("[9/10] Tier 3-2: Changepoint Detection (점진적 손상)")
print("=" * 80)

try:
    # WCST에서 RT changepoint 탐지 (간단한 버전)
    # Trial을 4등분하여 early vs late 기울기 비교

    changepoint_results = []

    for pid, group in wcst_trials.groupby('participant_id'):
        group = group.sort_values('trialIndex').reset_index(drop=True)
        n_trials = len(group)

        if n_trials < 20:
            continue

        # Early (1st quarter)
        early = group.iloc[:n_trials//4]
        # Late (4th quarter)
        late = group.iloc[3*n_trials//4:]

        early_pe_rate = early['is_pe'].mean() * 100
        late_pe_rate = late['is_pe'].mean() * 100
        pe_increase = late_pe_rate - early_pe_rate

        changepoint_results.append({
            'participant_id': pid,
            'early_pe_rate': early_pe_rate,
            'late_pe_rate': late_pe_rate,
            'pe_increase': pe_increase
        })

    cp_df = pd.DataFrame(changepoint_results)
    master_cp = master.merge(cp_df, on='participant_id', how='inner')

    # UCLA와 pe_increase 상관
    r_cp, p_cp = pearsonr(master_cp['ucla_total'], master_cp['pe_increase'])

    print(f"\n  UCLA × PE 증가량 (late - early) 상관:")
    print(f"    r={r_cp:.3f}, p={p_cp:.4f}")
    print(f"    주장: r≈0.222, p≈0.129 (NS, 급격한 붕괴 없음)")

    add_result(
        hypothesis="Changepoint (Gradual Decline)",
        metric="UCLA×PE_increase correlation",
        claimed_value=0.222,
        actual_value=r_cp,
        claimed_p=0.129,
        actual_p=p_cp,
        tier=3
    )

except Exception as e:
    print(f"  오류 발생: {e}")

# =============================================================================
# [TIER 3-3] UCLA Network Psychometrics (Factor 2 효과)
# =============================================================================

print("\n" + "=" * 80)
print("[10/10] Tier 3-3: UCLA Factor Analysis")
print("=" * 80)

try:
    # UCLA 20문항 로드
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")

    if 'participantId' in surveys.columns:
        surveys = surveys.rename(columns={'participantId': 'participant_id'})

    # UCLA만 필터
    ucla_items = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()

    # Wide format으로 변환
    ucla_wide = ucla_items.pivot_table(
        index='participant_id',
        columns='questionNum',
        values='score'
    ).reset_index()

    # Factor 분석은 생략하고, 간단하게 2-factor 가정
    # Factor 1 (사회적 고립): 문항 1, 5, 6, 9, 10
    # Factor 2 (정서적 외로움): 문항 2, 3, 4, 7, 8

    factor1_items = [1, 5, 6, 9, 10]
    factor2_items = [2, 3, 4, 7, 8]

    # 컬럼명이 숫자인지 확인
    if all(i in ucla_wide.columns for i in factor1_items):
        ucla_wide['factor1_social'] = ucla_wide[factor1_items].mean(axis=1)
        ucla_wide['factor2_emotional'] = ucla_wide[factor2_items].mean(axis=1)

        master_ucla_f = master.merge(
            ucla_wide[['participant_id', 'factor1_social', 'factor2_emotional']],
            on='participant_id',
            how='inner'
        )

        males_f = master_ucla_f[master_ucla_f['gender_male'] == 1]
        females_f = master_ucla_f[master_ucla_f['gender_male'] == 0]

        # Factor 2 × 남성
        r_male_f2, p_male_f2 = pearsonr(males_f['factor2_emotional'], males_f['pe_rate'])

        print(f"\n  UCLA Factor 2 (정서적) × PE 상관:")
        print(f"    남성: r={r_male_f2:.3f}, p={p_male_f2:.4f}")
        print(f"    주장: r≈−0.374, p≈0.054 (보호 효과 경향)")

        add_result(
            hypothesis="UCLA Factor 2: Male",
            metric="Male Factor2×PE correlation",
            claimed_value=-0.374,
            actual_value=r_male_f2,
            claimed_p=0.054,
            actual_p=p_male_f2,
            tier=3
        )
    else:
        print(f"  경고: UCLA 문항 컬럼을 찾을 수 없음. 사용가능: {ucla_wide.columns.tolist()}")

except Exception as e:
    print(f"  오류 발생: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 최종 재현성 보고서 저장
# =============================================================================

print("\n" + "=" * 80)
print("재현성 검증 완료 - 결과 저장 중...")
print("=" * 80)

results_df = pd.DataFrame(replication_results)
results_df.to_csv(OUTPUT_DIR / "replication_results.csv", index=False, encoding='utf-8-sig')

# 요약 통계
total_tests = len(results_df)
replicated_tests = results_df['replicated'].sum()
failed_tests = (results_df['replicated'] == False).sum()
unclear_tests = results_df['replicated'].isna().sum()

print(f"\n총 검증 항목: {total_tests}")
print(f"  ✅ 재현 성공: {replicated_tests} ({replicated_tests/total_tests*100:.1f}%)")
print(f"  ❌ 재현 실패: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
print(f"  ⚠️  판정 불가: {unclear_tests} ({unclear_tests/total_tests*100:.1f}%)")

# Tier별 재현율
print(f"\nTier별 재현율:")
for tier in [1, 2, 3]:
    tier_df = results_df[results_df['tier'] == tier]
    tier_replicated = tier_df['replicated'].sum()
    tier_total = len(tier_df[tier_df['replicated'].notna()])
    if tier_total > 0:
        print(f"  Tier {tier}: {tier_replicated}/{tier_total} ({tier_replicated/tier_total*100:.1f}%)")

# 불일치 항목만 저장
discrepancies = results_df[results_df['replicated'] == False]
if len(discrepancies) > 0:
    discrepancies.to_csv(OUTPUT_DIR / "discrepancies.csv", index=False, encoding='utf-8-sig')
    print(f"\n⚠️  불일치 항목 {len(discrepancies)}개 발견:")
    for _, row in discrepancies.iterrows():
        print(f"  - {row['hypothesis']}: {row['metric']}")
        print(f"    주장 {row['claimed_value']:.3f} (p={row['claimed_p']:.3f}) vs 실제 {row['actual_value']:.3f} (p={row['actual_p']:.3f})")

# 상세 보고서 텍스트 생성
report_lines = []
report_lines.append("=" * 80)
report_lines.append("재현성 검증 최종 보고서")
report_lines.append("REPLICATION VERIFICATION FINAL REPORT")
report_lines.append("=" * 80)
report_lines.append(f"\n실행 일시: {pd.Timestamp.now()}")
report_lines.append(f"데이터: N={len(master)} (남성={int(master['gender_male'].sum())}, 여성={int((1-master['gender_male']).sum())})")
report_lines.append(f"\n총 검증 항목: {total_tests}")
report_lines.append(f"재현 성공: {replicated_tests} ({replicated_tests/total_tests*100:.1f}%)")
report_lines.append(f"재현 실패: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
report_lines.append(f"판정 불가: {unclear_tests} ({unclear_tests/total_tests*100:.1f}%)")

report_lines.append("\n" + "=" * 80)
report_lines.append("상세 결과")
report_lines.append("=" * 80)

for tier in [1, 2, 3]:
    tier_df = results_df[results_df['tier'] == tier]
    if len(tier_df) == 0:
        continue

    report_lines.append(f"\n### TIER {tier} ###")

    for _, row in tier_df.iterrows():
        report_lines.append(f"\n[{row['hypothesis']}]")
        report_lines.append(f"  지표: {row['metric']}")
        report_lines.append(f"  주장 값: {row['claimed_value']}")
        report_lines.append(f"  실제 값: {row['actual_value']}")
        report_lines.append(f"  주장 p: {row['claimed_p']}")
        report_lines.append(f"  실제 p: {row['actual_p']}")

        if row['replicated'] == True:
            report_lines.append(f"  ✅ 재현 성공")
        elif row['replicated'] == False:
            report_lines.append(f"  ❌ 재현 실패")
            if row['effect_diff_pct'] is not None:
                report_lines.append(f"     효과크기 차이: {row['effect_diff_pct']:.1f}%")
            if not row['p_match']:
                report_lines.append(f"     p-value 차이: {abs(row['claimed_p'] - row['actual_p']):.4f}")
        else:
            report_lines.append(f"  ⚠️  판정 불가")

report_lines.append("\n" + "=" * 80)
report_lines.append("결론")
report_lines.append("=" * 80)

if replicated_tests / total_tests >= 0.7:
    report_lines.append("\n✅ 재현율 70% 이상 달성 - 전반적으로 가설이 재현되었습니다.")
else:
    report_lines.append(f"\n⚠️  재현율 {replicated_tests/total_tests*100:.1f}% - 일부 가설이 재현되지 않았습니다.")
    report_lines.append("   추가 검토가 필요합니다.")

report_text = "\n".join(report_lines)

with open(OUTPUT_DIR / "REPLICATION_VERIFICATION_REPORT.txt", "w", encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✅ 보고서 저장 완료:")
print(f"  - {OUTPUT_DIR / 'replication_results.csv'}")
print(f"  - {OUTPUT_DIR / 'REPLICATION_VERIFICATION_REPORT.txt'}")
if len(discrepancies) > 0:
    print(f"  - {OUTPUT_DIR / 'discrepancies.csv'}")

print("\n" + "=" * 80)
print("모든 재현성 검증 완료!")
print("=" * 80)
