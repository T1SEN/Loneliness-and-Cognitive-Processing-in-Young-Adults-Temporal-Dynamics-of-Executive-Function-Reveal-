#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Latent Profile Analysis
UCLA + DASS로 profiles 추출하고 Profile별 EF 비교
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Latent Profile Analysis")
print("UCLA + DASS 기반 심리적 프로파일 추출 및 EF 비교")
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

master_clean = master.dropna().copy()

print(f"\n샘플 크기: N = {len(master_clean)}")

# =============================================================================
# 2. Latent Profile Analysis - 모델 선택
# =============================================================================

print("\n" + "=" * 80)
print("1. MODEL SELECTION (BIC)")
print("=" * 80)

# Profile variables
profile_vars = ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
X = master_clean[profile_vars].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit models with different k
bics = []
aics = []
models = {}

print("\nFitting Gaussian Mixture Models...")
print("-" * 80)

for k in range(2, 7):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='full',
        random_state=42,
        n_init=10
    )
    gmm.fit(X_scaled)

    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)

    bics.append((k, bic))
    aics.append((k, aic))
    models[k] = gmm

    print(f"k={k}: BIC={bic:.1f}, AIC={aic:.1f}")

# Best model
best_k_bic = min(bics, key=lambda x: x[1])[0]
best_k_aic = min(aics, key=lambda x: x[1])[0]

print(f"\n최적 모델:")
print(f"  BIC 기준: k = {best_k_bic}")
print(f"  AIC 기준: k = {best_k_aic}")

# Use BIC (more conservative)
best_k = best_k_bic
gmm_best = models[best_k]

# Save model selection results
model_selection_df = pd.DataFrame({
    'k': [k for k, _ in bics],
    'BIC': [bic for _, bic in bics],
    'AIC': [aic for _, aic in aics]
})
model_selection_df.to_csv(OUTPUT_DIR / "lpa_model_selection.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 3. 프로파일 할당
# =============================================================================

print("\n" + "=" * 80)
print(f"2. PROFILE ASSIGNMENT (k={best_k})")
print("=" * 80)

# Predict profiles
labels = gmm_best.predict(X_scaled)
probabilities = gmm_best.predict_proba(X_scaled)

# Add to dataframe
master_clean['profile'] = labels
master_clean['profile_prob'] = probabilities.max(axis=1)

# Profile sizes
print("\nProfile sizes:")
for i in range(best_k):
    n = (labels == i).sum()
    pct = n / len(labels) * 100
    print(f"  Profile {i+1}: N={n} ({pct:.1f}%)")

# Average classification probability
print(f"\n평균 분류 확률: {master_clean['profile_prob'].mean():.3f}")

# =============================================================================
# 4. 프로파일 특성화
# =============================================================================

print("\n" + "=" * 80)
print("3. PROFILE CHARACTERIZATION")
print("=" * 80)

# Compute profile means (original scale)
profile_summary = master_clean.groupby('profile')[profile_vars].agg(['mean', 'std'])

print("\nProfile means and SDs:")
print("=" * 80)

for i in range(best_k):
    print(f"\nProfile {i+1}:")
    print("-" * 80)
    for var in profile_vars:
        mean_val = profile_summary.loc[i, (var, 'mean')]
        std_val = profile_summary.loc[i, (var, 'std')]
        print(f"  {var:20s}: M={mean_val:6.2f}, SD={std_val:5.2f}")

# Flatten and save
profile_means = master_clean.groupby('profile')[profile_vars].mean().reset_index()
profile_means.to_csv(OUTPUT_DIR / "lpa_profile_means.csv", index=False, encoding='utf-8-sig')

# Label profiles based on characteristics
profile_labels = {}
for i in range(best_k):
    ucla = profile_means.loc[profile_means['profile'] == i, 'ucla_total'].values[0]
    dass_dep = profile_means.loc[profile_means['profile'] == i, 'dass_depression'].values[0]

    if ucla > master_clean['ucla_total'].median():
        lone_label = "고외로움"
    else:
        lone_label = "저외로움"

    if dass_dep > master_clean['dass_depression'].median():
        dep_label = "고우울"
    else:
        dep_label = "저우울"

    profile_labels[i] = f"{lone_label}-{dep_label}"

print("\n\nProfile labels (자동 명명):")
for i, label in profile_labels.items():
    print(f"  Profile {i+1}: {label}")

master_clean['profile_label'] = master_clean['profile'].map(profile_labels)

# =============================================================================
# 5. Profile별 EF 비교
# =============================================================================

print("\n" + "=" * 80)
print("4. EXECUTIVE FUNCTION BY PROFILE")
print("=" * 80)

ef_vars = ['stroop_interference', 'perseverative_error_rate', 'prp_bottleneck']

ef_results = []

print("\nOne-way ANOVA:")
print("-" * 80)

for ef_var in ef_vars:
    # Extract groups
    groups = [master_clean[master_clean['profile'] == i][ef_var].dropna()
              for i in range(best_k)]

    # ANOVA
    f_stat, p_val = f_oneway(*groups)

    print(f"\n{ef_var}:")
    print(f"  F({best_k-1}, {len(master_clean)-best_k}) = {f_stat:.3f}, p = {p_val:.4f}")

    if p_val < 0.05:
        print(f"  *** 유의한 차이 (p < 0.05) ***")
    elif p_val < 0.10:
        print(f"  ** Marginal (p < 0.10) **")

    # Effect size (eta-squared)
    ss_between = sum(len(g) * (g.mean() - master_clean[ef_var].mean())**2 for g in groups)
    ss_total = ((master_clean[ef_var] - master_clean[ef_var].mean())**2).sum()
    eta_sq = ss_between / ss_total

    print(f"  η² = {eta_sq:.3f}")

    # Profile means
    for i in range(best_k):
        mean_val = groups[i].mean()
        std_val = groups[i].std()
        print(f"    Profile {i+1} ({profile_labels[i]}): M={mean_val:.2f}, SD={std_val:.2f}")

    ef_results.append({
        'ef_variable': ef_var,
        'F_statistic': f_stat,
        'p_value': p_val,
        'eta_squared': eta_sq
    })

ef_anova_df = pd.DataFrame(ef_results)
ef_anova_df.to_csv(OUTPUT_DIR / "lpa_ef_anova.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 6. Post-hoc 검정 (Tukey HSD)
# =============================================================================

if best_k > 2:
    print("\n" + "=" * 80)
    print("5. POST-HOC TESTS (Pairwise Comparisons)")
    print("=" * 80)

    from scipy.stats import ttest_ind

    posthoc_results = []

    for ef_var in ef_vars:
        print(f"\n{ef_var}:")
        print("-" * 80)

        # Pairwise t-tests
        for i in range(best_k):
            for j in range(i+1, best_k):
                group_i = master_clean[master_clean['profile'] == i][ef_var].dropna()
                group_j = master_clean[master_clean['profile'] == j][ef_var].dropna()

                t_stat, p_val = ttest_ind(group_i, group_j)

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(group_i) - 1) * group_i.std()**2 +
                     (len(group_j) - 1) * group_j.std()**2) /
                    (len(group_i) + len(group_j) - 2)
                )
                cohens_d = (group_i.mean() - group_j.mean()) / pooled_std

                print(f"  {profile_labels[i]} vs {profile_labels[j]}:")
                print(f"    t = {t_stat:.3f}, p = {p_val:.4f}, d = {cohens_d:.3f}")

                posthoc_results.append({
                    'ef_variable': ef_var,
                    'profile_1': i,
                    'profile_1_label': profile_labels[i],
                    'profile_2': j,
                    'profile_2_label': profile_labels[j],
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d
                })

    posthoc_df = pd.DataFrame(posthoc_results)
    posthoc_df.to_csv(OUTPUT_DIR / "lpa_posthoc.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# 7. 시각화
# =============================================================================

print("\n" + "=" * 80)
print("6. VISUALIZATION")
print("=" * 80)

# Plot 1: Profile means (radar/heatmap)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap of standardized means
profile_z_scores = master_clean.groupby('profile')[profile_vars].apply(
    lambda x: (x.mean() - master_clean[profile_vars].mean()) / master_clean[profile_vars].std()
)

sns.heatmap(
    profile_z_scores.T,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    ax=axes[0],
    cbar_kws={'label': 'Z-score'}
)
axes[0].set_title('Profile Characteristics (Standardized)', fontsize=12)
axes[0].set_xlabel('Profile')
axes[0].set_ylabel('Variable')

# EF by profile (boxplot)
ef_long = master_clean.melt(
    id_vars=['profile', 'profile_label'],
    value_vars=ef_vars,
    var_name='EF_Task',
    value_name='Score'
)

# Just plot one EF for clarity
wcst_data = master_clean[['profile_label', 'perseverative_error_rate']]
wcst_data = wcst_data.sort_values('profile_label')

sns.boxplot(
    data=wcst_data,
    x='profile_label',
    y='perseverative_error_rate',
    ax=axes[1],
    palette='Set2'
)
axes[1].set_title('WCST Perseverative Errors by Profile', fontsize=12)
axes[1].set_xlabel('Profile')
axes[1].set_ylabel('Perseverative Error Rate (%)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lpa_visualization.png", dpi=300, bbox_inches='tight')
print(f"\n시각화 저장: {OUTPUT_DIR / 'lpa_visualization.png'}")

# Plot 2: All EF tasks
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ef_var in enumerate(ef_vars):
    ef_data = master_clean[['profile_label', ef_var]].copy()
    ef_data = ef_data.sort_values('profile_label')

    sns.boxplot(
        data=ef_data,
        x='profile_label',
        y=ef_var,
        ax=axes[idx],
        palette='Set2'
    )
    axes[idx].set_title(ef_var.replace('_', ' ').title(), fontsize=12)
    axes[idx].set_xlabel('Profile')
    axes[idx].set_ylabel('Score')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lpa_ef_all_tasks.png", dpi=300, bbox_inches='tight')
print(f"시각화 저장: {OUTPUT_DIR / 'lpa_ef_all_tasks.png'}")

# =============================================================================
# 8. SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n최적 프로파일 수: k = {best_k}")

print("\n프로파일 특성:")
for i in range(best_k):
    n = (master_clean['profile'] == i).sum()
    label = profile_labels[i]
    ucla_m = profile_means.loc[profile_means['profile'] == i, 'ucla_total'].values[0]
    dass_m = profile_means.loc[profile_means['profile'] == i, 'dass_depression'].values[0]

    print(f"\n  Profile {i+1} ({label}): N={n}")
    print(f"    UCLA: {ucla_m:.1f}")
    print(f"    DASS-Depression: {dass_m:.1f}")

print("\nEF 차이:")
for _, row in ef_anova_df.iterrows():
    var = row['ef_variable']
    p = row['p_value']
    eta = row['eta_squared']

    if p < 0.05:
        sig_label = "***"
    elif p < 0.10:
        sig_label = "**"
    else:
        sig_label = ""

    print(f"  {var}: p={p:.4f}, η²={eta:.3f} {sig_label}")

print("\n해석:")
if any(ef_anova_df['p_value'] < 0.05):
    print("  - 프로파일 간 EF 차이 발견")
    print("  - '외로움+정서' 조합이 EF에 영향")
else:
    print("  - 프로파일 간 EF 차이 미미")
    print("  - 외로움/우울 조합보다 다른 요인이 중요할 가능성")

print("\n분석 완료!")
print(f"결과 저장 위치: {OUTPUT_DIR}")
print("\n생성된 파일:")
print("  - lpa_model_selection.csv")
print("  - lpa_profile_means.csv")
print("  - lpa_ef_anova.csv")
if best_k > 2:
    print("  - lpa_posthoc.csv")
print("  - lpa_visualization.png")
print("  - lpa_ef_all_tasks.png")

# Save profile assignments
master_clean[['participant_id', 'profile', 'profile_label', 'profile_prob']].to_csv(
    OUTPUT_DIR / "lpa_profile_assignments.csv",
    index=False,
    encoding='utf-8-sig'
)
print("  - lpa_profile_assignments.csv")
