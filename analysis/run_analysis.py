#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
외로움 × 집행기능 분석 파이프라인
====================================
IRB 계획에 따른 핵심 가설 검증:
1. UCLA 외로움 척도가 각 집행기능 지표를 예측하는가? (DASS-21 통제 후)
2. 세 과제(Stroop, WCST, PRP)에 걸친 공통 '메타-통제' 요인이 존재하는가?
3. 외로움이 이 공통 요인을 예측하는가?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 경로 설정 및 데이터 로드
# ============================================================================
print("=" * 70)
print("외로움 × 집행기능 분석 파이프라인 시작")
print("=" * 70)

data_dir = Path("results")
output_dir = data_dir / "analysis_outputs"
output_dir.mkdir(exist_ok=True)

print("\n[1단계] CSV 파일 로딩 중...")
participants = pd.read_csv(data_dir / "1_participants_info.csv")
surveys = pd.read_csv(data_dir / "2_surveys_results.csv")
cognitive = pd.read_csv(data_dir / "3_cognitive_tests_summary.csv")
prp_trials = pd.read_csv(data_dir / "4a_prp_trials.csv")
wcst_trials = pd.read_csv(data_dir / "4b_wcst_trials.csv")
stroop_trials = pd.read_csv(data_dir / "4c_stroop_trials.csv")

print(f"   - Participants: {len(participants):,} rows")
print(f"   - Survey rows: {len(surveys):,} rows")
print(f"   - Cognitive: {len(cognitive):,} rows")
print(f"   - PRP trials: {len(prp_trials):,} rows")
print(f"   - WCST trials: {len(wcst_trials):,} rows")
print(f"   - Stroop trials: {len(stroop_trials):,} rows")

# ============================================================================
# 1. 데이터 전처리: 참가자별 집계
# ============================================================================
print("\n[2단계] 참가자별 집행기능 지표 계산 중...")

# ─────────────────────────────────────────────────────────────────────────
# 1.1 Stroop 간섭 효과 (Incongruent RT - Congruent RT)
# ─────────────────────────────────────────────────────────────────────────
stroop_correct = stroop_trials[stroop_trials['correct'] == True].copy()
stroop_summary = stroop_correct.groupby(['participant_id', 'cond']).agg(
    mean_rt=('rt_ms', 'mean'),
    accuracy=('correct', 'size')
).reset_index()

stroop_pivot = stroop_summary.pivot(
    index='participant_id',
    columns='cond',
    values='mean_rt'
).reset_index()

stroop_pivot['stroop_interference'] = (
    stroop_pivot.get('incongruent', 0) - stroop_pivot.get('congruent', 0)
)

# ─────────────────────────────────────────────────────────────────────────
# 1.2 WCST 보속 오류 비율
# ─────────────────────────────────────────────────────────────────────────
wcst_summary = wcst_trials.groupby('participant_id').agg(
    total_trials=('trial_index', 'count'),
    perseverative_errors=('extra', lambda x: sum(
        pd.Series(x).apply(lambda y: eval(y).get('isPE', False) if isinstance(y, str) else False)
    )),
    total_errors=('correct', lambda x: (~x).sum())
).reset_index()

wcst_summary['perseverative_error_rate'] = (
    wcst_summary['perseverative_errors'] / wcst_summary['total_trials'] * 100
)

# ─────────────────────────────────────────────────────────────────────────
# 1.3 PRP 병목 효과 (짧은 SOA에서의 T2 RT 지연)
# ─────────────────────────────────────────────────────────────────────────
prp_correct = prp_trials[
    (prp_trials['t2_correct'] == True) &
    (prp_trials['t2_rt_ms'].notna()) &
    (prp_trials['t2_rt_ms'] > 0)
].copy()

# SOA를 3단계로 분류: short (≤150), medium (300-600), long (≥1200)
prp_correct['soa_bin'] = pd.cut(
    prp_correct['soa_nominal_ms'],
    bins=[0, 150, 600, 9999],
    labels=['short', 'medium', 'long']
)

prp_summary = prp_correct.groupby(['participant_id', 'soa_bin']).agg(
    mean_t2_rt=('t2_rt_ms', 'mean')
).reset_index()

prp_pivot = prp_summary.pivot(
    index='participant_id',
    columns='soa_bin',
    values='mean_t2_rt'
).reset_index()

prp_pivot['prp_bottleneck'] = (
    prp_pivot.get('short', 0) - prp_pivot.get('long', 0)
)

# ─────────────────────────────────────────────────────────────────────────
# 1.4 설문 데이터 (UCLA 외로움, DASS-21)
# ─────────────────────────────────────────────────────────────────────────
surveys_clean = surveys[['participant_id', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']].copy()

# ============================================================================
# 2. 마스터 데이터프레임 생성
# ============================================================================
print("\n[3단계] 마스터 데이터프레임 생성 중...")

master = participants[['participant_id', 'age', 'gender']].copy()
master = master.merge(surveys_clean, on='participant_id', how='left')
master = master.merge(stroop_pivot[['participant_id', 'stroop_interference']], on='participant_id', how='left')
master = master.merge(wcst_summary[['participant_id', 'perseverative_error_rate']], on='participant_id', how='left')
master = master.merge(prp_pivot[['participant_id', 'prp_bottleneck']], on='participant_id', how='left')

# 결측치 제거
print(f"\n   • 병합 전 참가자 수: {len(master)}")
master_complete = master.dropna(subset=[
    'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
    'stroop_interference', 'perseverative_error_rate', 'prp_bottleneck'
])
print(f"   • 병합 후 완전 데이터: {len(master_complete)}명")

if len(master_complete) < 20:
    print("\n⚠️  경고: 완전 데이터가 20명 미만입니다. 분석 결과의 신뢰도가 낮을 수 있습니다.")
    exit(1)

# 저장
master_complete.to_csv(output_dir / "master_dataset.csv", index=False, encoding='utf-8-sig')
print(f"\n   ✓ 마스터 데이터셋 저장 완료: {output_dir / 'master_dataset.csv'}")

# ============================================================================
# 3. 기술통계
# ============================================================================
print("\n" + "=" * 70)
print("기술통계")
print("=" * 70)

desc_stats = master_complete[[
    'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
    'stroop_interference', 'perseverative_error_rate', 'prp_bottleneck'
]].describe()

print("\n", desc_stats.round(2))

desc_stats.to_csv(output_dir / "descriptive_statistics.csv", encoding='utf-8-sig')

# ============================================================================
# 4. 상관 분석
# ============================================================================
print("\n" + "=" * 70)
print("상관 분석 (Pearson)")
print("=" * 70)

corr_vars = [
    'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress',
    'stroop_interference', 'perseverative_error_rate', 'prp_bottleneck'
]

corr_matrix = master_complete[corr_vars].corr()
print("\n", corr_matrix.round(3))

corr_matrix.to_csv(output_dir / "correlation_matrix.csv", encoding='utf-8-sig')

# p-value 계산
def calc_pval(x, y):
    valid = pd.notna(x) & pd.notna(y)
    if valid.sum() < 3:
        return np.nan
    return stats.pearsonr(x[valid], y[valid])[1]

pval_matrix = pd.DataFrame(
    [[calc_pval(master_complete[v1], master_complete[v2]) for v2 in corr_vars] for v1 in corr_vars],
    index=corr_vars,
    columns=corr_vars
)

print("\n[p-values]")
print(pval_matrix.round(4))
pval_matrix.to_csv(output_dir / "correlation_pvalues.csv", encoding='utf-8-sig')

# ============================================================================
# 5. 위계적 회귀분석 (핵심 가설 검증)
# ============================================================================
print("\n" + "=" * 70)
print("위계적 회귀분석: UCLA 외로움의 고유 효과")
print("=" * 70)

from sklearn.linear_model import LinearRegression
from scipy.stats import f as f_dist

def hierarchical_regression(df, outcome, covariates, predictor):
    """
    위계적 회귀분석 수행

    Model 1: outcome ~ covariates (DASS-21 subscales)
    Model 2: outcome ~ covariates + predictor (UCLA loneliness)

    반환: ΔR², ΔF, p-value
    """
    # 결측치 제거
    vars_needed = [outcome] + covariates + [predictor]
    df_clean = df[vars_needed].dropna()

    if len(df_clean) < 20:
        return {
            'outcome': outcome,
            'n': len(df_clean),
            'r2_model1': np.nan,
            'r2_model2': np.nan,
            'delta_r2': np.nan,
            'delta_f': np.nan,
            'p_value': np.nan,
            'ucla_coef': np.nan,
            'ucla_se': np.nan,
            'ucla_t': np.nan,
            'ucla_p': np.nan
        }

    y = df_clean[outcome].values
    X1 = df_clean[covariates].values
    X2 = df_clean[covariates + [predictor]].values

    # Model 1: 통제변인만
    model1 = LinearRegression().fit(X1, y)
    y_pred1 = model1.predict(X1)
    ss_res1 = np.sum((y - y_pred1) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_1 = 1 - (ss_res1 / ss_tot)

    # Model 2: 통제변인 + UCLA
    model2 = LinearRegression().fit(X2, y)
    y_pred2 = model2.predict(X2)
    ss_res2 = np.sum((y - y_pred2) ** 2)
    r2_2 = 1 - (ss_res2 / ss_tot)

    # ΔR² 검정
    delta_r2 = r2_2 - r2_1
    n = len(df_clean)
    k1 = X1.shape[1]
    k2 = X2.shape[1]

    delta_f = (delta_r2 / (k2 - k1)) / ((1 - r2_2) / (n - k2 - 1))
    p_value = 1 - f_dist.cdf(delta_f, k2 - k1, n - k2 - 1)

    # UCLA 계수의 t 검정
    # 잔차 표준오차
    mse = ss_res2 / (n - k2 - 1)
    # X'X의 역행렬
    XtX_inv = np.linalg.inv(X2.T @ X2)
    se_coefs = np.sqrt(np.diag(XtX_inv * mse))

    # UCLA는 마지막 계수
    ucla_coef = model2.coef_[-1]
    ucla_se = se_coefs[-1]
    ucla_t = ucla_coef / ucla_se
    ucla_p = 2 * (1 - stats.t.cdf(abs(ucla_t), n - k2 - 1))

    return {
        'outcome': outcome,
        'n': n,
        'r2_model1': r2_1,
        'r2_model2': r2_2,
        'delta_r2': delta_r2,
        'delta_f': delta_f,
        'p_value': p_value,
        'ucla_coef': ucla_coef,
        'ucla_se': ucla_se,
        'ucla_t': ucla_t,
        'ucla_p': ucla_p
    }

# 세 개의 집행기능 지표에 대해 각각 위계적 회귀 실행
outcomes = [
    ('stroop_interference', 'Stroop 간섭 효과 (ms)'),
    ('perseverative_error_rate', 'WCST 보속 오류율 (%)'),
    ('prp_bottleneck', 'PRP 병목 효과 (ms)')
]

covariates = ['dass_depression', 'dass_anxiety', 'dass_stress']
predictor = 'ucla_total'

results = []
for outcome_var, outcome_label in outcomes:
    print(f"\n▶ {outcome_label}")
    result = hierarchical_regression(master_complete, outcome_var, covariates, predictor)
    results.append(result)

    print(f"   N = {result['n']}")
    print(f"   Model 1 R² = {result['r2_model1']:.4f}")
    print(f"   Model 2 R² = {result['r2_model2']:.4f}")
    print(f"   ΔR² = {result['delta_r2']:.4f}, F = {result['delta_f']:.2f}, p = {result['p_value']:.4f}")
    print(f"   UCLA 계수 = {result['ucla_coef']:.4f}, SE = {result['ucla_se']:.4f}, t = {result['ucla_t']:.2f}, p = {result['ucla_p']:.4f}")

    if result['p_value'] < 0.05:
        print(f"   ✓ 유의함! (p < .05)")
    else:
        print(f"   × 유의하지 않음 (p ≥ .05)")

results_df = pd.DataFrame(results)
results_df.to_csv(output_dir / "hierarchical_regression_results.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ 위계적 회귀 결과 저장: {output_dir / 'hierarchical_regression_results.csv'}")

# ============================================================================
# 6. 공통 요인 분석 (PCA)
# ============================================================================
print("\n" + "=" * 70)
print("공통 '메타-통제' 요인 추출 (PCA)")
print("=" * 70)

# 세 지표를 표준화
ef_vars = ['stroop_interference', 'perseverative_error_rate', 'prp_bottleneck']
ef_data = master_complete[ef_vars].dropna()

if len(ef_data) < 20:
    print("⚠️  경고: PCA를 위한 데이터가 부족합니다.")
else:
    scaler = StandardScaler()
    ef_scaled = scaler.fit_transform(ef_data)

    pca = PCA(n_components=1)
    pca.fit(ef_scaled)

    print(f"\n제1 주성분 설명 분산: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"\n각 과제의 적재치 (loadings):")
    loadings = pd.DataFrame({
        'Task': ef_vars,
        'Loading': pca.components_[0]
    })
    print(loadings.to_string(index=False))

    loadings.to_csv(output_dir / "pca_loadings.csv", index=False, encoding='utf-8-sig')

    # 공통 요인 점수 계산
    meta_control_scores = pca.transform(ef_scaled)

    # 마스터 데이터에 추가
    ef_data_with_id = master_complete.loc[ef_data.index, ['participant_id']].copy()
    ef_data_with_id['meta_control_factor'] = meta_control_scores

    # UCLA와의 상관
    merged = ef_data_with_id.merge(
        master_complete[['participant_id', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']],
        on='participant_id'
    )

    r, p = stats.pearsonr(merged['meta_control_factor'], merged['ucla_total'])
    print(f"\n공통 요인 vs UCLA 외로움: r = {r:.3f}, p = {p:.4f}")

    # 공통 요인에 대한 위계적 회귀
    print("\n▶ 공통 요인에 대한 위계적 회귀:")
    merged_for_reg = merged.dropna()
    result_meta = hierarchical_regression(
        merged_for_reg,
        'meta_control_factor',
        covariates,
        predictor
    )

    print(f"   N = {result_meta['n']}")
    print(f"   Model 1 R² = {result_meta['r2_model1']:.4f}")
    print(f"   Model 2 R² = {result_meta['r2_model2']:.4f}")
    print(f"   ΔR² = {result_meta['delta_r2']:.4f}, F = {result_meta['delta_f']:.2f}, p = {result_meta['p_value']:.4f}")
    print(f"   UCLA 계수 = {result_meta['ucla_coef']:.4f}, SE = {result_meta['ucla_se']:.4f}, t = {result_meta['ucla_t']:.2f}, p = {result_meta['ucla_p']:.4f}")

    if result_meta['p_value'] < 0.05:
        print(f"   ✓ 유의함! (p < .05)")
    else:
        print(f"   × 유의하지 않음 (p ≥ .05)")

    # 결과 저장
    ef_data_with_id.to_csv(output_dir / "meta_control_scores.csv", index=False, encoding='utf-8-sig')

    # 최종 결과에 추가
    results.append(result_meta)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "hierarchical_regression_results_with_meta.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# 7. 최종 요약
# ============================================================================
print("\n" + "=" * 70)
print("분석 완료 요약")
print("=" * 70)

print(f"\n✓ 총 {len(master_complete)}명의 완전 데이터로 분석 완료")
print(f"✓ 결과 파일이 {output_dir} 폴더에 저장되었습니다.")
print("\n주요 결과 파일:")
print("   1. master_dataset.csv - 참가자별 최종 데이터셋")
print("   2. descriptive_statistics.csv - 기술통계")
print("   3. correlation_matrix.csv - 상관행렬")
print("   4. hierarchical_regression_results.csv - 위계적 회귀 결과")
print("   5. pca_loadings.csv - 공통 요인 적재치")
print("   6. meta_control_scores.csv - 참가자별 메타-통제 요인 점수")

print("\n" + "=" * 70)
print("다음 단계 제안:")
print("=" * 70)
print("1. 결과 파일을 Excel로 열어 주요 p-value 확인")
print("2. 유의한 효과가 나온 지표에 대해 산점도 작성")
print("3. 공통 요인의 적재치 패턴 해석 (속도 vs 갈등 해결 vs 전략적 조절)")
print("4. Q1/Q2 저널 타겟 논문 작성 시작")
print("   - 추천 저널: Cognitive, Affective, & Behavioral Neuroscience (Q1)")
print("   - 또는: Journal of Experimental Psychology: General (Q1)")
print("   - 또는: Psychological Science (Q1)")
print("=" * 70)
