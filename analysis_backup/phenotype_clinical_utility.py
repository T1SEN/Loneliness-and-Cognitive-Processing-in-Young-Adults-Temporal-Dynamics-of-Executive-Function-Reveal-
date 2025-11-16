#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
표현형 & 임상 활용 분석 (2-in-1)

통합 분석:
1. 성별 Stratified LPA (Latent Profile Analysis)
2. 위험 예측 모델 (Clinical Risk Prediction)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# UTF-8 설정
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# 경로 설정
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/phenotype_clinical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("표현형 & 임상 활용 분석 (2-in-1)")
print("="*80)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[데이터 로딩]")
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
cognitive = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv", encoding='utf-8-sig')

# UCLA
ucla = surveys[surveys['surveyName'] == 'ucla'][['participantId', 'score']].dropna()
ucla.columns = ['participantId', 'ucla_total']

# WCST
wcst = cognitive[cognitive['testName'] == 'wcst'].copy()
wcst['pe_rate'] = (wcst['perseverativeErrorCount'] / wcst['totalTrialCount']) * 100

# Master
master = participants[['participantId', 'gender', 'age']].copy()
master = master.merge(ucla, on='participantId', how='left')
master = master.merge(wcst[['participantId', 'pe_rate', 'accuracy', 'totalTrialCount']], on='participantId', how='left')

# PES 데이터 로드 (이미 계산됨)
try:
    pes_data = pd.read_csv(RESULTS_DIR.parent / "results/analysis_outputs/post_error_slowing/pes_all_tasks.csv", encoding='utf-8-sig')
    pes_wcst = pes_data[pes_data['task'] == 'wcst'][['participantId', 'pes_ms', 'post_error_acc']].copy()
    master = master.merge(pes_wcst, on='participantId', how='left')
except:
    print("   ⚠ PES 데이터 없음")

master = master.dropna(subset=['gender', 'ucla_total', 'pe_rate'])

print(f"   Total: {len(master)}명")
print(f"   Gender: {master['gender'].value_counts().to_dict()}")

# ============================================================================
# 분석 1: 성별 Stratified LPA
# ============================================================================
print("\n[1] 성별 Stratified LPA...")

# 남성과 여성을 따로 clustering
males = master[master['gender'] == '남성'].copy()
females = master[master['gender'] == '여성'].copy()

# Clustering variables
cluster_vars = ['ucla_total', 'pe_rate']
if 'pes_ms' in master.columns:
    cluster_vars.append('pes_ms')
if 'post_error_acc' in master.columns:
    cluster_vars.append('post_error_acc')

lpa_results = []

for gender_name, gender_df in [('남성', males), ('여성', females)]:
    if len(gender_df) < 10:
        print(f"   {gender_name}: N={len(gender_df)} < 10, 스킵")
        continue

    # Subset with non-missing values
    gender_cluster = gender_df[cluster_vars].dropna()

    if len(gender_cluster) < 10:
        print(f"   {gender_name}: 유효 데이터 {len(gender_cluster)} < 10, 스킵")
        continue

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(gender_cluster)

    # K-means (k=2, 3, 4 시도)
    best_k = 2
    best_silhouette = -1

    for k in range(2, min(5, len(gender_cluster))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Silhouette score
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X_scaled, labels)

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    # Final clustering with best k
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(X_scaled)

    # Add to dataframe
    gender_cluster['cluster'] = labels_final

    # Cluster centers (denormalized)
    centers_scaled = kmeans_final.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    print(f"\n   {gender_name} (N={len(gender_cluster)}):")
    print(f"   Best k: {best_k}, Silhouette: {best_silhouette:.3f}")

    for i in range(best_k):
        cluster_data = gender_cluster[gender_cluster['cluster'] == i]
        n_cluster = len(cluster_data)

        print(f"\n   Cluster {i+1} (N={n_cluster}, {n_cluster/len(gender_cluster)*100:.1f}%):")
        for j, var in enumerate(cluster_vars):
            print(f"     {var}: {centers[i, j]:.2f}")

        # 저장
        lpa_results.append({
            'gender': gender_name,
            'cluster': i+1,
            'n': n_cluster,
            'percent': n_cluster/len(gender_cluster)*100,
            **{f'{var}_mean': centers[i, j] for j, var in enumerate(cluster_vars)}
        })

    # Cluster assignments 저장
    cluster_assignments = gender_df.copy()
    cluster_assignments = cluster_assignments.merge(
        gender_cluster[['cluster']],
        left_index=True,
        right_index=True,
        how='left'
    )

    output_file = OUTPUT_DIR / f"lpa_clusters_{gender_name}.csv"
    cluster_assignments.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"   저장: {output_file}")

lpa_df = pd.DataFrame(lpa_results)
lpa_df.to_csv(OUTPUT_DIR / "lpa_summary.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# 분석 2: 위험 예측 모델
# ============================================================================
print("\n[2] 위험 예측 모델 (Clinical Utility)...")

# 목표: UCLA + 기본 변수로 "High PE risk" 예측
# Binary outcome: PE rate > median (high risk) vs ≤median (low risk)

pe_median = master['pe_rate'].median()
master['high_pe_risk'] = (master['pe_rate'] > pe_median).astype(int)

print(f"   PE median: {pe_median:.2f}%")
print(f"   High risk: {master['high_pe_risk'].sum()}명 ({master['high_pe_risk'].mean()*100:.1f}%)")

# Features
feature_cols = ['ucla_total', 'age']
if 'pes_ms' in master.columns:
    feature_cols.append('pes_ms')

# Gender dummy
master['gender_male'] = (master['gender'] == '남성').astype(int)
feature_cols.append('gender_male')

# Drop missing
model_data = master[feature_cols + ['high_pe_risk']].dropna()

if len(model_data) < 20:
    print(f"   ⚠ 데이터 부족 (N={len(model_data)}), 스킵")
else:
    X = model_data[feature_cols]
    y = model_data['high_pe_risk']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n   모델 학습 (N={len(model_data)}, Features={len(feature_cols)})...")

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')

    print(f"\n   Logistic Regression (5-fold CV):")
    print(f"   Mean AUC: {cv_scores.mean():.3f} (SD={cv_scores.std():.3f})")

    # Train on all data for coefficients
    lr.fit(X_scaled, y)

    # Coefficients
    coefs = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lr.coef_[0],
        'odds_ratio': np.exp(lr.coef_[0])
    })

    print(f"\n   Feature Importance (Odds Ratios):")
    for _, row in coefs.iterrows():
        print(f"     {row['feature']}: OR={row['odds_ratio']:.3f} (β={row['coefficient']:+.3f})")

    coefs.to_csv(OUTPUT_DIR / "risk_model_coefficients.csv", index=False, encoding='utf-8-sig')

    # Random Forest (더 복잡한 패턴 탐지)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
    cv_scores_rf = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc')

    print(f"\n   Random Forest (5-fold CV):")
    print(f"   Mean AUC: {cv_scores_rf.mean():.3f} (SD={cv_scores_rf.std():.3f})")

    # Feature importance
    rf.fit(X_scaled, y)
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n   RF Feature Importance:")
    for _, row in importances.iterrows():
        print(f"     {row['feature']}: {row['importance']:.3f}")

    importances.to_csv(OUTPUT_DIR / "risk_model_rf_importance.csv", index=False, encoding='utf-8-sig')

    # 모델 성능 요약
    model_performance = pd.DataFrame([
        {'model': 'Logistic Regression', 'mean_auc': cv_scores.mean(), 'sd_auc': cv_scores.std()},
        {'model': 'Random Forest', 'mean_auc': cv_scores_rf.mean(), 'sd_auc': cv_scores_rf.std()}
    ])

    model_performance.to_csv(OUTPUT_DIR / "risk_model_performance.csv", index=False, encoding='utf-8-sig')

    # Clinical utility: Cutoff recommendations
    # Predict probabilities
    y_pred_proba = lr.predict_proba(X_scaled)[:, 1]

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Youden index
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n   Clinical Cutoff (Youden index):")
    print(f"   Optimal probability threshold: {optimal_threshold:.3f}")
    print(f"   Sensitivity: {tpr[optimal_idx]:.3f}")
    print(f"   Specificity: {1-fpr[optimal_idx]:.3f}")

# ============================================================================
# 시각화
# ============================================================================
print("\n[시각화]")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. LPA Clusters (남성)
ax = axes[0, 0]
if len(lpa_df[lpa_df['gender'] == '남성']) > 0:
    male_lpa = lpa_df[lpa_df['gender'] == '남성']

    for i, row in male_lpa.iterrows():
        ax.scatter(row['ucla_total_mean'], row['pe_rate_mean'],
                   s=row['n']*10, alpha=0.7, label=f"Cluster {row['cluster']:.0f}")

    ax.set_xlabel('UCLA Loneliness (Cluster Mean)', fontsize=12)
    ax.set_ylabel('PE Rate (Cluster Mean)', fontsize=12)
    ax.set_title('LPA Clusters - Males', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# 2. LPA Clusters (여성)
ax = axes[0, 1]
if len(lpa_df[lpa_df['gender'] == '여성']) > 0:
    female_lpa = lpa_df[lpa_df['gender'] == '여성']

    for i, row in female_lpa.iterrows():
        ax.scatter(row['ucla_total_mean'], row['pe_rate_mean'],
                   s=row['n']*10, alpha=0.7, label=f"Cluster {row['cluster']:.0f}")

    ax.set_xlabel('UCLA Loneliness (Cluster Mean)', fontsize=12)
    ax.set_ylabel('PE Rate (Cluster Mean)', fontsize=12)
    ax.set_title('LPA Clusters - Females', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# 3. Risk Model Coefficients
ax = axes[1, 0]
if 'coefs' in locals():
    ax.barh(coefs['feature'], coefs['coefficient'], alpha=0.7, color='purple')
    ax.set_xlabel('Coefficient (β)', fontsize=12)
    ax.set_title('Risk Model Coefficients (Logistic Regression)', fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(alpha=0.3, axis='x')

# 4. RF Feature Importance
ax = axes[1, 1]
if 'importances' in locals():
    ax.barh(importances['feature'], importances['importance'], alpha=0.7, color='green')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Random Forest Feature Importance', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
output_fig = OUTPUT_DIR / "phenotype_clinical_plots.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"   저장: {output_fig}")
plt.close()

# ============================================================================
# 보고서
# ============================================================================
report_file = OUTPUT_DIR / "PHENOTYPE_CLINICAL_REPORT.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("표현형 & 임상 활용 분석 보고서 (2-in-1)\n")
    f.write("="*80 + "\n\n")

    f.write(f"분석 일자: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("1. 성별 Stratified LPA\n")
    f.write("-"*40 + "\n")
    for gender_name in ['남성', '여성']:
        gender_lpa = lpa_df[lpa_df['gender'] == gender_name]
        if len(gender_lpa) > 0:
            f.write(f"\n{gender_name}:\n")
            for _, row in gender_lpa.iterrows():
                f.write(f"  Cluster {row['cluster']:.0f} (N={row['n']:.0f}, {row['percent']:.1f}%):\n")
                f.write(f"    UCLA: {row['ucla_total_mean']:.1f}\n")
                f.write(f"    PE rate: {row['pe_rate_mean']:.1f}%\n")

    f.write("\n\n2. 위험 예측 모델\n")
    f.write("-"*40 + "\n")
    if 'model_performance' in locals():
        for _, row in model_performance.iterrows():
            f.write(f"{row['model']}: AUC={row['mean_auc']:.3f} (SD={row['sd_auc']:.3f})\n")

    if 'coefs' in locals():
        f.write(f"\nTop Predictors (Logistic Regression):\n")
        coefs_sorted = coefs.sort_values('odds_ratio', ascending=False)
        for _, row in coefs_sorted.iterrows():
            f.write(f"  {row['feature']}: OR={row['odds_ratio']:.3f}\n")

    f.write("\n" + "="*80 + "\n")

print(f"\n보고서 저장: {report_file}")

print("\n" + "="*80)
print("표현형 & 임상 활용 분석 완료!")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print("="*80)
