#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCLA Loneliness Facet Analysis

Decomposes UCLA-20 scale into subscales to test construct specificity:
1. Factor analysis to extract social vs. emotional loneliness dimensions
2. Test which facets predict EF performance (by gender)
3. Compare predictive power of facets vs. total score

UCLA-20 typically yields 2-3 factors:
- Social loneliness (lack of social network)
- Emotional loneliness (lack of intimate connections)
- (Sometimes) Isolation/belonging
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Unicode handling for Windows
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/ucla_facets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("UCLA LONELINESS FACET ANALYSIS")
print("Item-Level Factor Extraction and EF Prediction")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1] Loading data...")

participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")
participants = participants.rename(columns={'participantId': 'participant_id'})

surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})

cognitive = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv")
cognitive = cognitive.rename(columns={'participantId': 'participant_id'})

# Load trial data for custom EF metrics (all already have participant_id column)
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")

wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")

print(f"  Loaded {len(participants)} participants")

# ============================================================================
# 2. EXTRACT UCLA ITEM-LEVEL RESPONSES
# ============================================================================

print("\n[2] Extracting UCLA item-level responses...")

ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()

# Extract item columns q1-q20
item_cols_original = [f'q{i}' for i in range(1, 21)]
ucla_wide = ucla_data[['participant_id'] + item_cols_original + ['score']].copy()

# Rename to ucla_item_N
rename_dict = {f'q{i}': f'ucla_item_{i}' for i in range(1, 21)}
ucla_wide = ucla_wide.rename(columns=rename_dict)
ucla_wide = ucla_wide.rename(columns={'score': 'ucla_total'})

# Convert to numeric
item_cols = [f'ucla_item_{i}' for i in range(1, 21)]
for col in item_cols:
    ucla_wide[col] = pd.to_numeric(ucla_wide[col], errors='coerce')

# Drop rows with missing items or total
ucla_wide = ucla_wide.dropna(subset=item_cols + ['ucla_total'])

print(f"  UCLA item-level data for {len(ucla_wide)} participants")
print(f"  Mean total score: {ucla_wide['ucla_total'].mean():.2f}, SD: {ucla_wide['ucla_total'].std():.2f}")

# ============================================================================
# 3. FACTOR ANALYSIS: Extract Loneliness Facets
# ============================================================================

print("\n[3] Performing factor analysis...")

# Remove rows with missing values
ucla_items_complete = ucla_wide[item_cols].dropna()

print(f"  Complete cases: {len(ucla_items_complete)}")

# Standardize items
scaler = StandardScaler()
ucla_items_scaled = scaler.fit_transform(ucla_items_complete)

# --- PCA for initial exploration ---
print("\n  Principal Component Analysis (for exploration):")
pca = PCA(n_components=5)
pca_scores = pca.fit_transform(ucla_items_scaled)

print("    Explained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"      PC{i+1}: {var:.3f} ({var*100:.1f}%)")

cumulative_var = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_var >= 0.70) + 1  # 70% threshold
print(f"    Components to retain (70% variance): {n_components}")

# --- Factor Analysis with 2 factors (typical for UCLA) ---
print("\n  Factor Analysis (2 factors):")
fa = FactorAnalysis(n_components=2, random_state=42)
fa_scores = fa.fit_transform(ucla_items_scaled)

# Add factor scores back to dataframe
ucla_wide_complete = ucla_wide.loc[ucla_items_complete.index].copy()
ucla_wide_complete['factor1'] = fa_scores[:, 0]
ucla_wide_complete['factor2'] = fa_scores[:, 1]

# Examine loadings to interpret factors
loadings = pd.DataFrame(
    fa.components_.T,
    columns=['Factor1', 'Factor2'],
    index=[f'Item {i+1}' for i in range(len(item_cols))]
)

print("\n  Factor Loadings (top 5 per factor):")
print("\n    FACTOR 1 (top loadings):")
top_f1 = loadings['Factor1'].abs().nlargest(5)
for item, loading in top_f1.items():
    actual_loading = loadings.loc[item, 'Factor1']
    print(f"      {item}: {actual_loading:>6.3f}")

print("\n    FACTOR 2 (top loadings):")
top_f2 = loadings['Factor2'].abs().nlargest(5)
for item, loading in top_f2.items():
    actual_loading = loadings.loc[item, 'Factor2']
    print(f"      {item}: {actual_loading:>6.3f}")

# Save loadings
loadings.to_csv(OUTPUT_DIR / "ucla_factor_loadings.csv", encoding='utf-8-sig')
print("\n✓ Saved: ucla_factor_loadings.csv")

# Based on typical UCLA structure:
# Factor 1: Social loneliness (items about social network)
# Factor 2: Emotional loneliness (items about intimacy/closeness)
# We'll label them descriptively

ucla_wide_complete['social_loneliness'] = ucla_wide_complete['factor1']
ucla_wide_complete['emotional_loneliness'] = ucla_wide_complete['factor2']

# ============================================================================
# 4. COMPUTE EF METRICS
# ============================================================================

print("\n[4] Computing EF metrics...")

# --- Stroop interference ---
stroop_valid = stroop_trials[
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 200) &
    (stroop_trials['rt_ms'] < 5000)
].copy()

stroop_summary = stroop_valid.groupby(['participant_id', 'cond']).agg(
    mean_rt=('rt_ms', 'mean')
).reset_index()

stroop_wide = stroop_summary.pivot(index='participant_id', columns='cond', values='mean_rt').reset_index()
if 'congruent' in stroop_wide.columns and 'incongruent' in stroop_wide.columns:
    stroop_wide['stroop_interference'] = stroop_wide['incongruent'] - stroop_wide['congruent']
else:
    stroop_wide['stroop_interference'] = np.nan

# --- WCST perseverative errors ---
def parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_parsed'] = wcst_trials['extra'].apply(parse_wcst_extra)
wcst_trials['is_pe'] = wcst_trials['extra_parsed'].apply(lambda x: x.get('isPE', False))

wcst_valid = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['rt_ms'] > 0)
].copy()

wcst_pe_rate = (
    wcst_valid.groupby('participant_id')
    .agg(
        total_trials=('trial_index', 'count'),
        pe_count=('is_pe', 'sum')
    )
    .reset_index()
)
wcst_pe_rate['wcst_pe_rate'] = (wcst_pe_rate['pe_count'] / wcst_pe_rate['total_trials']) * 100

# --- PRP bottleneck ---
prp_valid = prp_trials[
    (prp_trials['t1_timeout'] == False) &
    (prp_trials['t2_timeout'] == False) &
    (prp_trials['t2_rt_ms'] > 200)
].copy()

prp_valid['soa_bin'] = pd.cut(
    prp_valid['soa_ms'],
    bins=[-np.inf, 150, 1200, np.inf],
    labels=['short', 'medium', 'long']
)

prp_summary = prp_valid.groupby(['participant_id', 'soa_bin']).agg(
    mean_t2_rt=('t2_rt_ms', 'mean')
).reset_index()

prp_wide = prp_summary.pivot(index='participant_id', columns='soa_bin', values='mean_t2_rt').reset_index()
if 'short' in prp_wide.columns and 'long' in prp_wide.columns:
    prp_wide['prp_bottleneck'] = prp_wide['short'] - prp_wide['long']
else:
    prp_wide['prp_bottleneck'] = np.nan

print(f"  Stroop interference: {stroop_wide['stroop_interference'].notna().sum()} participants")
print(f"  WCST PE rate: {len(wcst_pe_rate)} participants")
print(f"  PRP bottleneck: {prp_wide['prp_bottleneck'].notna().sum()} participants")

# ============================================================================
# 5. MERGE MASTER DATASET
# ============================================================================

print("\n[5] Merging master dataset...")

master = ucla_wide_complete[['participant_id', 'ucla_total', 'social_loneliness', 'emotional_loneliness']].copy()
master = master.merge(participants[['participant_id', 'gender', 'age']], on='participant_id', how='left')

# Recode gender from Korean to English
master['gender'] = master['gender'].map({'남성': 'male', '여성': 'female'})

master = master.merge(stroop_wide[['participant_id', 'stroop_interference']], on='participant_id', how='left')
master = master.merge(wcst_pe_rate[['participant_id', 'wcst_pe_rate']], on='participant_id', how='left')
master = master.merge(prp_wide[['participant_id', 'prp_bottleneck']], on='participant_id', how='left')

# Standardize predictors
scaler = StandardScaler()
master['z_ucla_total'] = scaler.fit_transform(master[['ucla_total']])
master['z_social'] = scaler.fit_transform(master[['social_loneliness']])
master['z_emotional'] = scaler.fit_transform(master[['emotional_loneliness']])

print(f"  Final N = {len(master)}")
print(f"  EF metrics available:")
print(f"    Stroop: {master['stroop_interference'].notna().sum()}")
print(f"    WCST: {master['wcst_pe_rate'].notna().sum()}")
print(f"    PRP: {master['prp_bottleneck'].notna().sum()}")

# ============================================================================
# 6. FACET-SPECIFIC PREDICTION OF EF
# ============================================================================

print("\n[6] Testing facet-specific EF predictions...")

def test_facet_predictions(df, ef_metric, ef_label):
    """Compare total score vs. facets in predicting EF"""

    print(f"\n  {ef_label.upper()}")
    print("  " + "-"*70)

    # Remove missing
    df_clean = df[['z_ucla_total', 'z_social', 'z_emotional', ef_metric, 'gender']].dropna()

    if len(df_clean) < 20:
        print(f"    Insufficient data (N={len(df_clean)})")
        return None

    print(f"    N = {len(df_clean)}")

    # Correlations
    r_total, p_total = stats.pearsonr(df_clean['z_ucla_total'], df_clean[ef_metric])
    r_social, p_social = stats.pearsonr(df_clean['z_social'], df_clean[ef_metric])
    r_emotional, p_emotional = stats.pearsonr(df_clean['z_emotional'], df_clean[ef_metric])

    print(f"\n    Zero-order correlations:")
    print(f"      Total score:         r={r_total:>6.3f}, p={p_total:.4f} {'***' if p_total < 0.05 else 'ns'}")
    print(f"      Social loneliness:   r={r_social:>6.3f}, p={p_social:.4f} {'***' if p_social < 0.05 else 'ns'}")
    print(f"      Emotional loneliness: r={r_emotional:>6.3f}, p={p_emotional:.4f} {'***' if p_emotional < 0.05 else 'ns'}")

    # Regression: EF ~ Social + Emotional (both facets simultaneously)
    X = df_clean[['z_social', 'z_emotional']].values
    X = np.column_stack([np.ones(len(X)), X])
    Y = df_clean[ef_metric].values

    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]

    # R-squared for facet model
    y_pred = X @ coeffs
    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2_facets = 1 - (ss_res / ss_tot)

    # R-squared for total score model
    X_total = np.column_stack([np.ones(len(df_clean)), df_clean['z_ucla_total']])
    coeffs_total = np.linalg.lstsq(X_total, Y, rcond=None)[0]
    y_pred_total = X_total @ coeffs_total
    ss_res_total = np.sum((Y - y_pred_total) ** 2)
    r2_total = 1 - (ss_res_total / ss_tot)

    print(f"\n    Simultaneous regression (EF ~ Social + Emotional):")
    print(f"      Social β:     {coeffs[1]:>6.3f}")
    print(f"      Emotional β:  {coeffs[2]:>6.3f}")
    print(f"      R² (facets):  {r2_facets:.4f}")
    print(f"      R² (total):   {r2_total:.4f}")
    print(f"      ΔR²:          {r2_facets - r2_total:.4f} {'(facets better)' if r2_facets > r2_total else '(total better)'}")

    # Gender-stratified
    print(f"\n    Gender-stratified correlations:")
    for gender in ['male', 'female']:
        subset = df_clean[df_clean['gender'] == gender]
        if len(subset) > 10:
            r_s, p_s = stats.pearsonr(subset['z_social'], subset[ef_metric])
            r_e, p_e = stats.pearsonr(subset['z_emotional'], subset[ef_metric])
            print(f"      {gender.capitalize():8s} (N={len(subset):2d}): Social r={r_s:>6.3f} (p={p_s:.3f}), Emotional r={r_e:>6.3f} (p={p_e:.3f})")

    return {
        'ef_metric': ef_label,
        'n': len(df_clean),
        'r_total': r_total,
        'p_total': p_total,
        'r_social': r_social,
        'p_social': p_social,
        'r_emotional': r_emotional,
        'p_emotional': p_emotional,
        'social_beta': coeffs[1],
        'emotional_beta': coeffs[2],
        'r2_facets': r2_facets,
        'r2_total': r2_total,
        'delta_r2': r2_facets - r2_total
    }

results_list = []

stroop_result = test_facet_predictions(master, 'stroop_interference', 'Stroop Interference')
if stroop_result:
    results_list.append(stroop_result)

wcst_result = test_facet_predictions(master, 'wcst_pe_rate', 'WCST PE Rate')
if wcst_result:
    results_list.append(wcst_result)

prp_result = test_facet_predictions(master, 'prp_bottleneck', 'PRP Bottleneck')
if prp_result:
    results_list.append(prp_result)

results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_DIR / "facet_predictions_summary.csv", index=False, encoding='utf-8-sig')
print("\n✓ Saved: facet_predictions_summary.csv")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n[7] Creating visualizations...")

# --- Scree plot ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('UCLA Loneliness Scale - Scree Plot', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ucla_scree_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ucla_scree_plot.png")

# --- Factor loadings heatmap ---
fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Loading'}, ax=ax)
ax.set_title('UCLA Item Factor Loadings', fontsize=14, fontweight='bold')
ax.set_xlabel('Factor', fontsize=12)
ax.set_ylabel('Item', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ucla_factor_loadings_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ucla_factor_loadings_heatmap.png")

# --- Facet correlations with EF ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('UCLA Facets vs. Executive Function Performance', fontsize=16, fontweight='bold')

ef_metrics = [
    ('stroop_interference', 'Stroop Interference (ms)'),
    ('wcst_pe_rate', 'WCST PE Rate (%)'),
    ('prp_bottleneck', 'PRP Bottleneck (ms)')
]

for idx, (ef_metric, ef_label) in enumerate(ef_metrics):
    ax = axes[idx]

    df_plot = master[['z_social', 'z_emotional', ef_metric, 'gender']].dropna()

    if len(df_plot) > 10:
        # Plot both facets
        ax.scatter(df_plot['z_social'], df_plot[ef_metric], alpha=0.5, s=60, label='Social', edgecolors='black', linewidths=0.5, color='blue')
        ax.scatter(df_plot['z_emotional'], df_plot[ef_metric], alpha=0.5, s=60, label='Emotional', edgecolors='black', linewidths=0.5, color='red', marker='^')

        # Fit lines
        for facet, color in [('z_social', 'blue'), ('z_emotional', 'red')]:
            if len(df_plot) > 5:
                z = np.polyfit(df_plot[facet], df_plot[ef_metric], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df_plot[facet].min(), df_plot[facet].max(), 100)
                ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Loneliness Facet (z-scored)', fontsize=11)
        ax.set_ylabel(ef_label, fontsize=11)
        ax.set_title(ef_label.split('(')[0], fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'facet_ef_scatterplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: facet_ef_scatterplots.png")

# --- Comparison: Total vs Facets R² ---
if len(results_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width/2, results_df['r2_total'], width, label='Total Score', alpha=0.7, color='gray', edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, results_df['r2_facets'], width, label='Facets (Social + Emotional)', alpha=0.7, color='green', edgecolor='black', linewidth=1.5)

    ax.set_xlabel('EF Metric', fontsize=12)
    ax.set_ylabel('R² (Variance Explained)', fontsize=12)
    ax.set_title('Predictive Power: UCLA Total vs. Facets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['ef_metric'], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'facet_total_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: facet_total_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("UCLA FACET ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")

print(f"\n  FACTOR STRUCTURE:")
print(f"    • 2 factors extracted")
print(f"    • Factor 1 (Social Loneliness) - top items: {', '.join([i.split()[1] for i in top_f1.index[:3]])}")
print(f"    • Factor 2 (Emotional Loneliness) - top items: {', '.join([i.split()[1] for i in top_f2.index[:3]])}")

if len(results_df) > 0:
    print(f"\n  FACET-SPECIFIC EF PREDICTIONS:")
    for _, row in results_df.iterrows():
        print(f"\n    {row['ef_metric']}:")
        print(f"      Social loneliness:    r={row['r_social']:>6.3f}, p={row['p_social']:.4f}, β={row['social_beta']:>6.3f}")
        print(f"      Emotional loneliness: r={row['r_emotional']:>6.3f}, p={row['p_emotional']:.4f}, β={row['emotional_beta']:>6.3f}")
        print(f"      R² improvement: {row['delta_r2']:>+.4f} {'(facets better)' if row['delta_r2'] > 0 else '(total better)'}")

print("\nOUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}/")
print("  - ucla_factor_loadings.csv")
print("  - facet_predictions_summary.csv")
print("  - ucla_scree_plot.png")
print("  - ucla_factor_loadings_heatmap.png")
print("  - facet_ef_scatterplots.png")
print("  - facet_total_comparison.png")

print("\n" + "="*80)
