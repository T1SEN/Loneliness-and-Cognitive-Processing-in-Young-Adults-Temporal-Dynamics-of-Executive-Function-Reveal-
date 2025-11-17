"""
Loneliness Classification Model
================================

╔════════════════════════════════════════════════════════════════════════╗
║                     *** METHODOLOGICAL NOTE ***                        ║
╟────────────────────────────────────────────────────────────────────────╢
║ This is a PREDICTIVE analysis, NOT confirmatory hypothesis testing.   ║
║                                                                        ║
║ DASS is included as a FEATURE (predictor), NOT a covariate.          ║
║                                                                        ║
║ Why? We test if EF patterns can predict loneliness status.           ║
║ Including DASS as a feature tests the incremental predictive value   ║
║ of EF beyond mood/anxiety context.                                   ║
║                                                                        ║
║ This differs from regression models where DASS MUST be controlled    ║
║ as a covariate (see other scripts).                                  ║
║                                                                        ║
║ See CLAUDE.md "DASS-21 Covariate Control Requirement" for rationale. ║
╚════════════════════════════════════════════════════════════════════════╝

Research Question: "Can we identify high-lonely individuals from their
executive function signature alone?"

Method:
1. Define groups: UCLA top 25% (High) vs bottom 25% (Low)
2. Features: EF metrics (WCST PE, PRP tau/mu/sigma, Stroop, RT variability, PES)
   + DASS subscales (as features, not covariates)
3. Models: Logistic Regression + Random Forest with 5-fold stratified CV
4. Compare: Overall sample vs Male-only vs Female-only

Interpretation:
- AUC > 0.65: Meaningful individual-difference signal exists
- Male AUC >> Female AUC: Male-specific predictive signature

Author: Research Team
Date: 2025-01-16
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Constants
MIN_N_ML = 20  # Minimum sample size for ML/classification (smaller OK than regression)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/loneliness_classification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LONELINESS CLASSIFICATION MODEL")
print("=" * 80)
print("\nPurpose: Predict High vs Low loneliness from EF patterns")
print("Method: Logistic Regression + Random Forest with 5-fold CV\n")

# Load master dataset
try:
    master = pd.read_csv(RESULTS_DIR / "analysis_outputs/master_dataset.csv", encoding='utf-8-sig')
    print(f"Loaded master dataset: {len(master)} participants")
except FileNotFoundError:
    print("ERROR: master_dataset.csv not found.")
    sys.exit(1)

# Normalize columns
master.columns = master.columns.str.lower()
if 'participantid' in master.columns:
    master.rename(columns={'participantid': 'participant_id'}, inplace=True)

# Ensure gender coding
gender_map = {'남성': 'male', '여성': 'female', 'Male': 'male', 'Female': 'female', 'M': 'male', 'F': 'female'}
if 'gender' in master.columns:
    master['gender'] = master['gender'].map(gender_map).fillna(master['gender'])
    master['gender_male'] = (master['gender'] == 'male').astype(int)

print(f"\nData overview:")
print(f"  Total N: {len(master)}")
if 'gender' in master.columns:
    print(f"  Males: {sum(master['gender']=='male')}, Females: {sum(master['gender']=='female')}")

# Check for UCLA
if 'ucla_total' not in master.columns:
    print("ERROR: 'ucla_total' column not found.")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 1: Define High vs Low Loneliness Groups")
print("=" * 80)

# Compute quartiles
ucla_25th = master['ucla_total'].quantile(0.25)
ucla_75th = master['ucla_total'].quantile(0.75)

print(f"\nUCLA distribution:")
print(f"  25th percentile: {ucla_25th:.1f}")
print(f"  Median: {master['ucla_total'].median():.1f}")
print(f"  75th percentile: {ucla_75th:.1f}")

# Create groups
master['loneliness_group'] = np.nan
master.loc[master['ucla_total'] <= ucla_25th, 'loneliness_group'] = 0  # Low
master.loc[master['ucla_total'] >= ucla_75th, 'loneliness_group'] = 1  # High

# Filter to extreme groups only
extreme_groups = master[master['loneliness_group'].notna()].copy()

print(f"\nExtreme groups:")
print(f"  Low loneliness (≤{ucla_25th:.1f}): N={sum(extreme_groups['loneliness_group']==0)}")
print(f"  High loneliness (≥{ucla_75th:.1f}): N={sum(extreme_groups['loneliness_group']==1)}")

if len(extreme_groups) < 20:
    print("ERROR: Insufficient data for classification (N < 20).")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 2: Select EF Features")
print("=" * 80)

# Define candidate features (all EF-related metrics)
candidate_features = []

# WCST features
wcst_features = ['pe_rate', 'wcst_pe_rate', 'perseverative_error_rate',
                 'wcst_accuracy', 'wcst_mean_rt']
candidate_features.extend([f for f in wcst_features if f in extreme_groups.columns])

# PRP features
prp_features = ['prp_bottleneck', 'prp_tau_long', 'prp_mu_long', 'prp_sigma_long',
                'prp_tau_short', 'prp_mu_short', 'prp_sigma_short',
                't2_rt_mean_short', 't2_rt_mean_long', 't2_rt_sd']
candidate_features.extend([f for f in prp_features if f in extreme_groups.columns])

# Stroop features
stroop_features = ['stroop_interference', 'stroop_effect',
                   'stroop_congruent_rt', 'stroop_incongruent_rt',
                   'stroop_incongruent_acc', 'stroop_congruent_acc']
candidate_features.extend([f for f in stroop_features if f in extreme_groups.columns])

# Variability features
variability_features = ['rt_cv', 'rt_sd', 'pes', 'post_error_slowing']
candidate_features.extend([f for f in variability_features if f in extreme_groups.columns])

# Remove duplicates
candidate_features = list(set(candidate_features))

print(f"\nCandidate EF features found: {len(candidate_features)}")
for feat in sorted(candidate_features):
    print(f"  - {feat}")

# Optional: Include DASS as features (not as control)
include_dass = True
if include_dass and all(col in extreme_groups.columns for col in ['dass_depression', 'dass_anxiety', 'dass_stress']):
    candidate_features.extend(['dass_depression', 'dass_anxiety', 'dass_stress'])
    print(f"\nIncluding DASS as features (+3) → Total: {len(candidate_features)}")

# Clean data: drop rows with missing features
feature_df = extreme_groups[['participant_id', 'loneliness_group', 'gender_male'] + candidate_features].copy()
feature_df = feature_df.dropna(subset=candidate_features + ['loneliness_group'])

print(f"\nAfter dropping missing features: N={len(feature_df)}")
print(f"  Low: {sum(feature_df['loneliness_group']==0)}, High: {sum(feature_df['loneliness_group']==1)}")

if len(feature_df) < 20:
    print("ERROR: Insufficient complete data (N < 20).")
    sys.exit(1)

# Prepare X (features) and y (target)
X = feature_df[candidate_features].values
y = feature_df['loneliness_group'].values
gender_labels = feature_df['gender_male'].values

print(f"\nFinal dataset: N={len(X)}, Features={X.shape[1]}")

print("\n" + "=" * 80)
print("STEP 3: Train Classification Models with Cross-Validation")
print("=" * 80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5,
                                            random_state=42, class_weight='balanced')
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scoring metrics
scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']

results_summary = []

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 40)

    # Overall performance
    cv_results = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring,
                                return_train_score=False)

    mean_auc = cv_results['test_roc_auc'].mean()
    std_auc = cv_results['test_roc_auc'].std()
    mean_acc = cv_results['test_accuracy'].mean()
    mean_prec = cv_results['test_precision'].mean()
    mean_rec = cv_results['test_recall'].mean()
    mean_f1 = cv_results['test_f1'].mean()

    print(f"  Overall (5-fold CV):")
    print(f"    AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"    Accuracy: {mean_acc:.3f}")
    print(f"    Precision: {mean_prec:.3f}")
    print(f"    Recall: {mean_rec:.3f}")
    print(f"    F1: {mean_f1:.3f}")

    results_summary.append({
        'model': model_name,
        'subset': 'Overall',
        'n': len(X),
        'auc': mean_auc,
        'auc_std': std_auc,
        'accuracy': mean_acc,
        'precision': mean_prec,
        'recall': mean_rec,
        'f1': mean_f1
    })

    # Gender-stratified performance
    for gender, gender_label in [(1, 'Male'), (0, 'Female')]:
        gender_idx = gender_labels == gender
        X_gender = X_scaled[gender_idx]
        y_gender = y[gender_idx]

        if len(y_gender) < 10 or len(np.unique(y_gender)) < 2:
            print(f"  {gender_label}: Insufficient data (N={len(y_gender)})")
            continue

        # Use fewer folds for small N
        n_splits_gender = min(3, len(y_gender) // 2)
        cv_gender = StratifiedKFold(n_splits=n_splits_gender, shuffle=True, random_state=42)

        cv_results_gender = cross_validate(model, X_gender, y_gender, cv=cv_gender,
                                           scoring=scoring, return_train_score=False)

        mean_auc_g = cv_results_gender['test_roc_auc'].mean()
        std_auc_g = cv_results_gender['test_roc_auc'].std()
        mean_acc_g = cv_results_gender['test_accuracy'].mean()

        print(f"  {gender_label} only (N={len(y_gender)}):")
        print(f"    AUC: {mean_auc_g:.3f} ± {std_auc_g:.3f}")
        print(f"    Accuracy: {mean_acc_g:.3f}")

        results_summary.append({
            'model': model_name,
            'subset': gender_label,
            'n': len(y_gender),
            'auc': mean_auc_g,
            'auc_std': std_auc_g,
            'accuracy': mean_acc_g,
            'precision': cv_results_gender['test_precision'].mean(),
            'recall': cv_results_gender['test_recall'].mean(),
            'f1': cv_results_gender['test_f1'].mean()
        })

# Save performance summary
results_df = pd.DataFrame(results_summary)
results_df.to_csv(OUTPUT_DIR / "classification_performance.csv",
                  index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: classification_performance.csv")

print("\n" + "=" * 80)
print("STEP 4: Feature Importance")
print("=" * 80)

# Fit models on full data for feature importance
for model_name, model in models.items():
    print(f"\n{model_name} - Feature Importance:")
    print("-" * 40)

    model.fit(X_scaled, y)

    if hasattr(model, 'coef_'):
        # Logistic regression coefficients
        importances = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # Random forest importances
        importances = model.feature_importances_
    else:
        # Permutation importance (fallback)
        perm_imp = permutation_importance(model, X_scaled, y, n_repeats=10,
                                         random_state=42, n_jobs=-1)
        importances = perm_imp.importances_mean

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': candidate_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(importance_df.head(10).to_string(index=False))

    # Save
    importance_df.to_csv(OUTPUT_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.csv",
                        index=False, encoding='utf-8-sig')

print(f"\n✓ Saved: feature_importance_*.csv files")

print("\n" + "=" * 80)
print("STEP 5: Visualizations")
print("=" * 80)

# 5A: ROC Curves (Overall + Gender-stratified)
print("\nCreating ROC curves...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (subset_name, subset_mask) in enumerate([
    ('Overall', np.ones(len(X), dtype=bool)),
    ('Male', gender_labels == 1),
    ('Female', gender_labels == 0)
]):

    ax = axes[idx]

    X_subset = X_scaled[subset_mask]
    y_subset = y[subset_mask]

    if len(y_subset) < 10 or len(np.unique(y_subset)) < 2:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
        ax.set_title(f'{subset_name} (N={len(y_subset)})')
        continue

    for model_name, model in models.items():
        # Fit and predict probabilities
        model.fit(X_subset, y_subset)
        y_pred_proba = model.predict_proba(X_subset)[:, 1]

        # Compute ROC
        fpr, tpr, _ = roc_curve(y_subset, y_pred_proba)
        auc = roc_auc_score(y_subset, y_pred_proba)

        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Chance', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'{subset_name} (N={len(y_subset)})', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: roc_curves.png")

# 5B: Feature Importance Bar Plot (Random Forest)
print("\nCreating feature importance plot...")

rf_model = models['Random Forest']
rf_model.fit(X_scaled, y)
importances_rf = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'feature': candidate_features,
    'importance': importances_rf
}).sort_values('importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis', ax=ax)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Top 15 Features: Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_barplot.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: feature_importance_barplot.png")

# 5C: Confusion Matrices (Overall)
print("\nCreating confusion matrices...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (model_name, model) in enumerate(models.items()):
    ax = axes[idx]

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'],
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: confusion_matrices.png")

print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

# Create comprehensive report
report_path = OUTPUT_DIR / "CLASSIFICATION_REPORT.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("LONELINESS CLASSIFICATION REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-" * 80 + "\n")
    f.write("Can executive function patterns predict high vs low loneliness?\n")
    f.write("If so, is this predictive signature gender-specific?\n\n")

    f.write("METHOD\n")
    f.write("-" * 80 + "\n")
    f.write(f"Target: High loneliness (UCLA ≥{ucla_75th:.1f}) vs Low (UCLA ≤{ucla_25th:.1f})\n")
    f.write(f"N: {len(feature_df)} participants (extremes only)\n")
    f.write(f"Features: {len(candidate_features)} EF metrics\n")
    f.write(f"Models: Logistic Regression + Random Forest\n")
    f.write(f"Validation: 5-fold stratified cross-validation\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-" * 80 + "\n\n")

    for _, row in results_df.iterrows():
        f.write(f"{row['model']} - {row['subset']} (N={row['n']}):\n")
        f.write(f"  AUC: {row['auc']:.3f} ± {row['auc_std']:.3f}\n")
        f.write(f"  Accuracy: {row['accuracy']:.3f}\n")
        f.write(f"  F1: {row['f1']:.3f}\n\n")

    f.write("INTERPRETATION GUIDE\n")
    f.write("-" * 80 + "\n")
    f.write("AUC = 0.50: Random chance (no signal)\n")
    f.write("AUC = 0.60-0.65: Weak predictive signal\n")
    f.write("AUC = 0.65-0.75: Moderate signal (meaningful individual differences)\n")
    f.write("AUC > 0.75: Strong signal (potential screening utility)\n\n")

    # Determine overall interpretation
    overall_auc = results_df[(results_df['subset'] == 'Overall') & (results_df['model'] == 'Random Forest')]['auc'].values
    if len(overall_auc) > 0:
        overall_auc = overall_auc[0]
    else:
        overall_auc = np.nan

    male_auc = results_df[(results_df['subset'] == 'Male') & (results_df['model'] == 'Random Forest')]['auc'].values
    female_auc = results_df[(results_df['subset'] == 'Female') & (results_df['model'] == 'Random Forest')]['auc'].values

    if len(male_auc) > 0:
        male_auc = male_auc[0]
    else:
        male_auc = np.nan

    if len(female_auc) > 0:
        female_auc = female_auc[0]
    else:
        female_auc = np.nan

    f.write("CONCLUSION\n")
    f.write("-" * 80 + "\n")

    if not np.isnan(overall_auc):
        if overall_auc >= 0.65:
            f.write(f"✓ Overall AUC = {overall_auc:.3f} → MEANINGFUL predictive signal exists.\n")
            f.write("  EF patterns can discriminate high vs low loneliness above chance.\n\n")
        else:
            f.write(f"✗ Overall AUC = {overall_auc:.3f} → WEAK/NO predictive signal.\n")
            f.write("  Current EF metrics do not reliably predict loneliness status.\n\n")

    if not np.isnan(male_auc) and not np.isnan(female_auc):
        if male_auc > female_auc + 0.10:
            f.write(f"✓ MALE-SPECIFIC SIGNATURE: Male AUC ({male_auc:.3f}) >> Female AUC ({female_auc:.3f})\n")
            f.write("  Loneliness → EF impairment is more predictable in males.\n")
            f.write("  Supports gender-differentiated vulnerability hypothesis.\n\n")
        elif female_auc > male_auc + 0.10:
            f.write(f"✓ FEMALE-SPECIFIC SIGNATURE: Female AUC ({female_auc:.3f}) >> Male AUC ({male_auc:.3f})\n")
            f.write("  Loneliness → EF impairment is more predictable in females.\n\n")
        else:
            f.write(f"○ No clear gender difference: Male AUC={male_auc:.3f}, Female AUC={female_auc:.3f}\n")
            f.write("  Similar predictive patterns across genders.\n\n")

    f.write("LIMITATIONS\n")
    f.write("-" * 80 + "\n")
    f.write("- Small sample size limits generalizability\n")
    f.write("- Extreme group design (top/bottom 25%) may inflate AUC\n")
    f.write("- Cross-validation on same dataset (not external validation)\n")
    f.write("- Predictive model ≠ causal mechanism\n\n")

    f.write("NEXT STEPS\n")
    f.write("-" * 80 + "\n")
    f.write("- If AUC > 0.65: Cite in Discussion as evidence for measurable EF signature\n")
    f.write("- Examine feature importance to identify which EF metrics drive classification\n")
    f.write("- If gender-specific: Connect to interaction findings from regression models\n")
    f.write("- Future: External validation in independent sample\n")

print(f"✓ Saved: CLASSIFICATION_REPORT.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("\nKey outputs:")
print("  1. classification_performance.csv - AUC/accuracy by model and gender")
print("  2. feature_importance_*.csv - Which EF metrics predict loneliness")
print("  3. roc_curves.png - ROC curves for overall + gender subsets")
print("  4. feature_importance_barplot.png - Top 15 predictive features")
print("  5. CLASSIFICATION_REPORT.txt - Interpretation summary")
print("\nKey takeaway:")
if not np.isnan(overall_auc):
    if overall_auc >= 0.65:
        print(f"  → AUC = {overall_auc:.3f}: Meaningful predictive signal detected!")
    else:
        print(f"  → AUC = {overall_auc:.3f}: Weak predictive signal.")
else:
    print("  → Check results CSV for AUC values")
