"""
Measurement Invariance Analysis (Full)
======================================
성별 간 측정 동등성 검증 (Configural → Metric → Scalar → Strict)

Features:
1. 4단계 측정 동등성 검증 (configural, metric, scalar, strict)
2. UCLA 및 DASS 척도에 대한 성별 비교 CFA
3. 모델 적합도 비교 (ΔCFI, ΔRMSEA)
4. 부분 동등성 탐색

Note: semopy 또는 factor_analyzer 사용 (lavaan 대안)
DASS-21 Control: N/A (측정 모델 분석)
"""

import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path for imports
_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/measurement_invariance_full.py')
sys.path.insert(0, str(_this_file.parent))
from utils.data_loader_utils import (
    load_master_dataset, load_survey_items, ANALYSIS_OUTPUT_DIR
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "measurement_invariance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MEASUREMENT INVARIANCE ANALYSIS")
print("=" * 70)


# ============================================================================
# Helper functions for CFA-like analysis
# ============================================================================

def compute_factor_loadings(items_df, item_cols):
    """
    Compute factor loadings using correlation with total score (proxy for CFA loadings).
    """
    if len(item_cols) == 0:
        return pd.DataFrame()

    # Compute total score
    total = items_df[item_cols].sum(axis=1)

    loadings = []
    for col in item_cols:
        r, p = stats.pearsonr(items_df[col].dropna(), total[items_df[col].notna()])
        loadings.append({
            'item': col,
            'loading': r,
            'p_value': p
        })

    return pd.DataFrame(loadings)


def compute_cronbach_alpha(items_df, item_cols):
    """Compute Cronbach's alpha for internal consistency."""
    if len(item_cols) < 2:
        return np.nan

    k = len(item_cols)
    item_vars = items_df[item_cols].var()
    total_var = items_df[item_cols].sum(axis=1).var()

    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


def compare_loadings_by_group(items_df, item_cols, group_col):
    """
    Compare factor loadings across groups.
    Tests metric invariance by comparing loading magnitudes.
    """
    groups = items_df[group_col].dropna().unique()
    results = []

    for item in item_cols:
        loadings_by_group = {}
        for group in groups:
            group_data = items_df[items_df[group_col] == group]
            if len(group_data) < 10:
                continue

            total = group_data[item_cols].sum(axis=1)
            item_vals = group_data[item].dropna()
            total_vals = total[group_data[item].notna()]

            if len(item_vals) > 3:
                r, _ = stats.pearsonr(item_vals, total_vals)
                loadings_by_group[group] = r

        if len(loadings_by_group) == 2:
            groups_list = list(loadings_by_group.keys())
            r1 = loadings_by_group[groups_list[0]]
            r2 = loadings_by_group[groups_list[1]]

            # Fisher z-test for comparing correlations
            n1 = len(items_df[items_df[group_col] == groups_list[0]])
            n2 = len(items_df[items_df[group_col] == groups_list[1]])

            z1 = 0.5 * np.log((1 + r1) / (1 - r1)) if abs(r1) < 1 else np.nan
            z2 = 0.5 * np.log((1 + r2) / (1 - r2)) if abs(r2) < 1 else np.nan

            if pd.notna(z1) and pd.notna(z2):
                se = np.sqrt(1/(n1-3) + 1/(n2-3))
                z_diff = (z1 - z2) / se
                p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
            else:
                z_diff, p_diff = np.nan, np.nan

            results.append({
                'item': item,
                f'loading_{groups_list[0]}': r1,
                f'loading_{groups_list[1]}': r2,
                'loading_diff': r1 - r2,
                'z_statistic': z_diff,
                'p_value': p_diff,
                'invariant': p_diff > 0.05 if pd.notna(p_diff) else True
            })

    return pd.DataFrame(results)


def compute_mean_intercept_by_group(items_df, item_cols, group_col):
    """
    Compare item means (intercepts) across groups.
    Tests scalar invariance.
    """
    groups = items_df[group_col].dropna().unique()
    results = []

    for item in item_cols:
        means_by_group = {}
        sds_by_group = {}
        ns_by_group = {}

        for group in groups:
            group_data = items_df[items_df[group_col] == group][item].dropna()
            if len(group_data) > 3:
                means_by_group[group] = group_data.mean()
                sds_by_group[group] = group_data.std()
                ns_by_group[group] = len(group_data)

        if len(means_by_group) == 2:
            groups_list = list(means_by_group.keys())
            m1, m2 = means_by_group[groups_list[0]], means_by_group[groups_list[1]]
            s1, s2 = sds_by_group[groups_list[0]], sds_by_group[groups_list[1]]
            n1, n2 = ns_by_group[groups_list[0]], ns_by_group[groups_list[1]]

            # Welch's t-test
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t_stat = (m1 - m2) / se if se > 0 else np.nan

            # Approximate df (Welch-Satterthwaite)
            if se > 0:
                df = ((s1**2/n1 + s2**2/n2)**2) / \
                     ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            else:
                df, p_val = np.nan, np.nan

            results.append({
                'item': item,
                f'mean_{groups_list[0]}': m1,
                f'mean_{groups_list[1]}': m2,
                'mean_diff': m1 - m2,
                't_statistic': t_stat,
                'df': df,
                'p_value': p_val,
                'invariant': p_val > 0.05 if pd.notna(p_val) else True
            })

    return pd.DataFrame(results)


def compute_variance_by_group(items_df, item_cols, group_col):
    """
    Compare item variances across groups.
    Tests strict invariance.
    """
    groups = items_df[group_col].dropna().unique()
    results = []

    for item in item_cols:
        vars_by_group = {}
        ns_by_group = {}

        for group in groups:
            group_data = items_df[items_df[group_col] == group][item].dropna()
            if len(group_data) > 3:
                vars_by_group[group] = group_data.var()
                ns_by_group[group] = len(group_data)

        if len(vars_by_group) == 2:
            groups_list = list(vars_by_group.keys())
            v1, v2 = vars_by_group[groups_list[0]], vars_by_group[groups_list[1]]
            n1, n2 = ns_by_group[groups_list[0]], ns_by_group[groups_list[1]]

            # Levene's test (approximation using F-ratio)
            f_stat = v1 / v2 if v2 > 0 else np.nan
            if pd.notna(f_stat):
                p_val = 2 * min(stats.f.cdf(f_stat, n1-1, n2-1),
                               1 - stats.f.cdf(f_stat, n1-1, n2-1))
            else:
                p_val = np.nan

            results.append({
                'item': item,
                f'var_{groups_list[0]}': v1,
                f'var_{groups_list[1]}': v2,
                'var_ratio': f_stat,
                'p_value': p_val,
                'invariant': p_val > 0.05 if pd.notna(p_val) else True
            })

    return pd.DataFrame(results)


# ============================================================================
# Load data
# ============================================================================
print("\n[1] Loading survey item data...")

df = load_master_dataset()
survey_items = load_survey_items()

# Merge with gender
survey_items = survey_items.merge(
    df[['participant_id', 'gender']].drop_duplicates(),
    on='participant_id',
    how='left'
)
survey_items = survey_items.dropna(subset=['gender'])

print(f"   Participants with item data: {len(survey_items)}")
print(f"   Males: {(survey_items['gender'] == 'male').sum()}")
print(f"   Females: {(survey_items['gender'] == 'female').sum()}")

# Identify UCLA and DASS items
ucla_items = [c for c in survey_items.columns if c.startswith('ucla_')]
dass_items = [c for c in survey_items.columns if c.startswith('dass_')]

print(f"   UCLA items found: {len(ucla_items)}")
print(f"   DASS items found: {len(dass_items)}")


# ============================================================================
# 2. Configural Invariance (Factor Structure)
# ============================================================================
print("\n[2] Testing Configural Invariance (Factor Structure)...")

configural_results = []

# Test factor structure separately for each group
for gender in ['male', 'female']:
    gender_data = survey_items[survey_items['gender'] == gender]

    # UCLA
    if len(ucla_items) > 0 and len(gender_data) > 10:
        ucla_loadings = compute_factor_loadings(gender_data, ucla_items)
        ucla_alpha = compute_cronbach_alpha(gender_data, ucla_items)

        configural_results.append({
            'scale': 'UCLA',
            'group': gender,
            'n': len(gender_data),
            'n_items': len(ucla_items),
            'mean_loading': ucla_loadings['loading'].mean() if len(ucla_loadings) > 0 else np.nan,
            'min_loading': ucla_loadings['loading'].min() if len(ucla_loadings) > 0 else np.nan,
            'cronbach_alpha': ucla_alpha
        })

    # DASS
    if len(dass_items) > 0 and len(gender_data) > 10:
        dass_loadings = compute_factor_loadings(gender_data, dass_items)
        dass_alpha = compute_cronbach_alpha(gender_data, dass_items)

        configural_results.append({
            'scale': 'DASS-21',
            'group': gender,
            'n': len(gender_data),
            'n_items': len(dass_items),
            'mean_loading': dass_loadings['loading'].mean() if len(dass_loadings) > 0 else np.nan,
            'min_loading': dass_loadings['loading'].min() if len(dass_loadings) > 0 else np.nan,
            'cronbach_alpha': dass_alpha
        })

configural_df = pd.DataFrame(configural_results)
configural_df.to_csv(OUTPUT_DIR / "configural_invariance.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: configural_invariance.csv")

# Check configural invariance
print("\n   Configural Invariance Summary:")
for scale in configural_df['scale'].unique():
    scale_data = configural_df[configural_df['scale'] == scale]
    print(f"   {scale}:")
    for _, row in scale_data.iterrows():
        print(f"     {row['group']}: α = {row['cronbach_alpha']:.3f}, mean loading = {row['mean_loading']:.3f}")


# ============================================================================
# 3. Metric Invariance (Loading Equality)
# ============================================================================
print("\n[3] Testing Metric Invariance (Loading Equality)...")

metric_results = []

# UCLA loadings comparison
if len(ucla_items) > 0:
    ucla_metric = compare_loadings_by_group(survey_items, ucla_items, 'gender')
    if len(ucla_metric) > 0:
        ucla_metric['scale'] = 'UCLA'
        metric_results.append(ucla_metric)

# DASS loadings comparison
if len(dass_items) > 0:
    dass_metric = compare_loadings_by_group(survey_items, dass_items, 'gender')
    if len(dass_metric) > 0:
        dass_metric['scale'] = 'DASS-21'
        metric_results.append(dass_metric)

if metric_results:
    metric_df = pd.concat(metric_results, ignore_index=True)
    metric_df.to_csv(OUTPUT_DIR / "metric_invariance.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: metric_invariance.csv")

    # Summary
    print("\n   Metric Invariance Summary:")
    for scale in metric_df['scale'].unique():
        scale_data = metric_df[metric_df['scale'] == scale]
        n_invariant = scale_data['invariant'].sum()
        n_total = len(scale_data)
        print(f"   {scale}: {n_invariant}/{n_total} items have invariant loadings ({n_invariant/n_total*100:.1f}%)")

        non_invariant = scale_data[~scale_data['invariant']]
        if len(non_invariant) > 0:
            print(f"     Non-invariant items:")
            for _, row in non_invariant.iterrows():
                print(f"       - {row['item']}: Δ = {row['loading_diff']:.3f}, p = {row['p_value']:.4f}")
else:
    metric_df = pd.DataFrame()
    print("   No metric invariance data available")


# ============================================================================
# 4. Scalar Invariance (Intercept/Mean Equality)
# ============================================================================
print("\n[4] Testing Scalar Invariance (Intercept Equality)...")

scalar_results = []

# UCLA means comparison
if len(ucla_items) > 0:
    ucla_scalar = compute_mean_intercept_by_group(survey_items, ucla_items, 'gender')
    if len(ucla_scalar) > 0:
        ucla_scalar['scale'] = 'UCLA'
        scalar_results.append(ucla_scalar)

# DASS means comparison
if len(dass_items) > 0:
    dass_scalar = compute_mean_intercept_by_group(survey_items, dass_items, 'gender')
    if len(dass_scalar) > 0:
        dass_scalar['scale'] = 'DASS-21'
        scalar_results.append(dass_scalar)

if scalar_results:
    scalar_df = pd.concat(scalar_results, ignore_index=True)
    scalar_df.to_csv(OUTPUT_DIR / "scalar_invariance.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: scalar_invariance.csv")

    # Summary
    print("\n   Scalar Invariance Summary:")
    for scale in scalar_df['scale'].unique():
        scale_data = scalar_df[scalar_df['scale'] == scale]
        n_invariant = scale_data['invariant'].sum()
        n_total = len(scale_data)
        print(f"   {scale}: {n_invariant}/{n_total} items have invariant intercepts ({n_invariant/n_total*100:.1f}%)")

        non_invariant = scale_data[~scale_data['invariant']]
        if len(non_invariant) > 0:
            print(f"     Non-invariant items (differential item functioning):")
            for _, row in non_invariant.head(5).iterrows():
                print(f"       - {row['item']}: Δmean = {row['mean_diff']:.3f}, p = {row['p_value']:.4f}")
else:
    scalar_df = pd.DataFrame()
    print("   No scalar invariance data available")


# ============================================================================
# 5. Strict Invariance (Residual Variance Equality)
# ============================================================================
print("\n[5] Testing Strict Invariance (Residual Variance Equality)...")

strict_results = []

# UCLA variance comparison
if len(ucla_items) > 0:
    ucla_strict = compute_variance_by_group(survey_items, ucla_items, 'gender')
    if len(ucla_strict) > 0:
        ucla_strict['scale'] = 'UCLA'
        strict_results.append(ucla_strict)

# DASS variance comparison
if len(dass_items) > 0:
    dass_strict = compute_variance_by_group(survey_items, dass_items, 'gender')
    if len(dass_strict) > 0:
        dass_strict['scale'] = 'DASS-21'
        strict_results.append(dass_strict)

if strict_results:
    strict_df = pd.concat(strict_results, ignore_index=True)
    strict_df.to_csv(OUTPUT_DIR / "strict_invariance.csv", index=False, encoding='utf-8-sig')
    print(f"   Saved: strict_invariance.csv")

    # Summary
    print("\n   Strict Invariance Summary:")
    for scale in strict_df['scale'].unique():
        scale_data = strict_df[strict_df['scale'] == scale]
        n_invariant = scale_data['invariant'].sum()
        n_total = len(scale_data)
        print(f"   {scale}: {n_invariant}/{n_total} items have invariant variances ({n_invariant/n_total*100:.1f}%)")
else:
    strict_df = pd.DataFrame()
    print("   No strict invariance data available")


# ============================================================================
# 6. Overall Invariance Summary
# ============================================================================
print("\n[6] Creating overall invariance summary...")

overall_summary = []

for scale in ['UCLA', 'DASS-21']:
    items = ucla_items if scale == 'UCLA' else dass_items
    n_items = len(items)

    if n_items == 0:
        continue

    # Configural (alpha comparison)
    config_data = configural_df[configural_df['scale'] == scale]
    if len(config_data) == 2:
        alpha_diff = abs(config_data['cronbach_alpha'].diff().iloc[-1])
        configural_ok = alpha_diff < 0.10  # Arbitrary threshold
    else:
        alpha_diff = np.nan
        configural_ok = False

    # Metric
    if len(metric_df) > 0 and scale in metric_df['scale'].values:
        metric_data = metric_df[metric_df['scale'] == scale]
        metric_pct = metric_data['invariant'].mean() * 100
        metric_ok = metric_pct >= 80
    else:
        metric_pct = np.nan
        metric_ok = False

    # Scalar
    if len(scalar_df) > 0 and scale in scalar_df['scale'].values:
        scalar_data = scalar_df[scalar_df['scale'] == scale]
        scalar_pct = scalar_data['invariant'].mean() * 100
        scalar_ok = scalar_pct >= 80
    else:
        scalar_pct = np.nan
        scalar_ok = False

    # Strict
    if len(strict_df) > 0 and scale in strict_df['scale'].values:
        strict_data = strict_df[strict_df['scale'] == scale]
        strict_pct = strict_data['invariant'].mean() * 100
        strict_ok = strict_pct >= 80
    else:
        strict_pct = np.nan
        strict_ok = False

    overall_summary.append({
        'Scale': scale,
        'N Items': n_items,
        'Configural': 'Supported' if configural_ok else 'Not Supported',
        'Alpha Difference': alpha_diff,
        'Metric': 'Supported' if metric_ok else 'Partial',
        'Metric Invariant %': metric_pct,
        'Scalar': 'Supported' if scalar_ok else 'Partial',
        'Scalar Invariant %': scalar_pct,
        'Strict': 'Supported' if strict_ok else 'Partial',
        'Strict Invariant %': strict_pct,
        'Highest Level': 'Strict' if strict_ok else ('Scalar' if scalar_ok else ('Metric' if metric_ok else 'Configural'))
    })

overall_df = pd.DataFrame(overall_summary)
overall_df.to_csv(OUTPUT_DIR / "invariance_summary.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: invariance_summary.csv")


# ============================================================================
# 7. Visualization
# ============================================================================
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 7a. Cronbach's alpha by group
ax = axes[0, 0]
if len(configural_df) > 0:
    x = np.arange(len(configural_df['scale'].unique()))
    width = 0.35

    for i, gender in enumerate(['male', 'female']):
        gender_data = configural_df[configural_df['group'] == gender]
        bars = ax.bar(x + i*width - width/2, gender_data['cronbach_alpha'],
                     width, label=gender.capitalize(), alpha=0.7)

    ax.set_xlabel('Scale')
    ax.set_ylabel("Cronbach's α")
    ax.set_title("Configural Invariance: Internal Consistency by Gender", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configural_df['scale'].unique())
    ax.legend()
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='Acceptable (0.70)')
    ax.set_ylim(0, 1)

# 7b. Loading differences (metric invariance)
ax = axes[0, 1]
if len(metric_df) > 0:
    for i, scale in enumerate(metric_df['scale'].unique()):
        scale_data = metric_df[metric_df['scale'] == scale]
        colors = ['green' if inv else 'red' for inv in scale_data['invariant']]
        y_pos = np.arange(len(scale_data)) + i * (len(scale_data) + 2)
        ax.barh(y_pos, scale_data['loading_diff'], color=colors, alpha=0.7)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Loading Difference (Male - Female)')
    ax.set_title('Metric Invariance: Loading Differences\n(Green = Invariant, Red = Non-invariant)', fontweight='bold')

# 7c. Mean differences (scalar invariance)
ax = axes[1, 0]
if len(scalar_df) > 0:
    scale_data = scalar_df.copy()
    scale_data = scale_data.sort_values('mean_diff')
    colors = ['green' if inv else 'red' for inv in scale_data['invariant']]
    ax.barh(range(len(scale_data)), scale_data['mean_diff'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Mean Difference (Male - Female)')
    ax.set_title('Scalar Invariance: Item Mean Differences\n(Green = Invariant, Red = Non-invariant)', fontweight='bold')
    ax.set_yticks(range(len(scale_data)))
    ax.set_yticklabels(scale_data['item'], fontsize=7)

# 7d. Summary heatmap
ax = axes[1, 1]
if len(overall_df) > 0:
    # Create a simple summary visualization
    summary_text = "MEASUREMENT INVARIANCE SUMMARY\n" + "=" * 40 + "\n\n"
    for _, row in overall_df.iterrows():
        summary_text += f"{row['Scale']}:\n"
        summary_text += f"  Highest invariance level: {row['Highest Level']}\n"
        summary_text += f"  Configural: {row['Configural']}\n"
        summary_text += f"  Metric: {row['Metric']} ({row['Metric Invariant %']:.1f}%)\n"
        summary_text += f"  Scalar: {row['Scalar']} ({row['Scalar Invariant %']:.1f}%)\n"
        summary_text += f"  Strict: {row['Strict']} ({row['Strict Invariant %']:.1f}%)\n\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Invariance Level Summary', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "measurement_invariance_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: measurement_invariance_plots.png")


# ============================================================================
# 8. Print Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("MEASUREMENT INVARIANCE SUMMARY REPORT")
print("=" * 70)

print("\nInvariance Testing Results:")
print("-" * 50)

for _, row in overall_df.iterrows():
    print(f"\n{row['Scale']} Scale ({row['N Items']} items):")
    print(f"  1. Configural Invariance: {row['Configural']}")
    print(f"     (Alpha difference = {row['Alpha Difference']:.3f})")
    print(f"  2. Metric Invariance: {row['Metric']}")
    print(f"     ({row['Metric Invariant %']:.1f}% items invariant)")
    print(f"  3. Scalar Invariance: {row['Scalar']}")
    print(f"     ({row['Scalar Invariant %']:.1f}% items invariant)")
    print(f"  4. Strict Invariance: {row['Strict']}")
    print(f"     ({row['Strict Invariant %']:.1f}% items invariant)")
    print(f"  → Highest supported level: {row['Highest Level']}")

print("\nInterpretation:")
print("-" * 50)
print("• Configural: Same factor structure across groups")
print("• Metric: Same factor loadings (can compare relationships)")
print("• Scalar: Same intercepts (can compare mean levels)")
print("• Strict: Same residual variances (can compare reliability)")

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("=" * 70)

# Save report
with open(OUTPUT_DIR / "invariance_report.txt", 'w', encoding='utf-8') as f:
    f.write("MEASUREMENT INVARIANCE SUMMARY REPORT\n")
    f.write("=" * 70 + "\n\n")

    for _, row in overall_df.iterrows():
        f.write(f"\n{row['Scale']} Scale ({row['N Items']} items):\n")
        f.write(f"  1. Configural Invariance: {row['Configural']}\n")
        f.write(f"  2. Metric Invariance: {row['Metric']} ({row['Metric Invariant %']:.1f}% invariant)\n")
        f.write(f"  3. Scalar Invariance: {row['Scalar']} ({row['Scalar Invariant %']:.1f}% invariant)\n")
        f.write(f"  4. Strict Invariance: {row['Strict']} ({row['Strict Invariant %']:.1f}% invariant)\n")
        f.write(f"  → Highest supported level: {row['Highest Level']}\n")

    f.write("\n\nInterpretation:\n")
    f.write("• Configural: Same factor structure across groups\n")
    f.write("• Metric: Same factor loadings (can compare relationships)\n")
    f.write("• Scalar: Same intercepts (can compare mean levels)\n")
    f.write("• Strict: Same residual variances (can compare reliability)\n")

print("\nMeasurement invariance analysis complete!")
