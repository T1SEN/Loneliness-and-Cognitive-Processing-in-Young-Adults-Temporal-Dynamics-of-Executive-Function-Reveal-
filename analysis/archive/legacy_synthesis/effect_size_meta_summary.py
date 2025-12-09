"""
Effect Size Meta-Summary Analysis
=================================
모든 분석 결과의 효과 크기를 통합하여 종합표 및 Forest Plot 생성

Features:
- Cohen's d, η², standardized β, r 통합
- Bootstrap 95% CI 계산
- APA 형식 테이블 자동 생성
- Forest plot 시각화

DASS-21 Control: 기존 분석 결과를 통합하므로 원본 분석의 DASS 통제 여부를 표시
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
_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/effect_size_meta_summary.py')
sys.path.insert(0, str(_this_file.parent))
from utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "effect_size_summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EFFECT SIZE META-SUMMARY ANALYSIS")
print("=" * 70)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def eta_squared_from_f(f_stat, df1, df2):
    """Calculate partial eta squared from F statistic."""
    return (df1 * f_stat) / (df1 * f_stat + df2)


def r_to_d(r):
    """Convert correlation r to Cohen's d."""
    if abs(r) >= 1:
        return np.nan
    return 2 * r / np.sqrt(1 - r**2)


def d_to_r(d):
    """Convert Cohen's d to correlation r."""
    return d / np.sqrt(d**2 + 4)


def bootstrap_ci(data, stat_func, n_bootstrap=5000, ci=0.95):
    """Bootstrap confidence interval for any statistic."""
    boot_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])


def format_pvalue(p, threshold=0.001):
    """Format p-value in APA style."""
    if p < threshold:
        return f"< {threshold}"
    return f"= {p:.3f}"


def format_ci(lower, upper, decimals=2):
    """Format confidence interval in APA style."""
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def interpret_effect_size(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# 1. Load existing analysis results
# ============================================================================
print("\n[1] Loading existing analysis results...")

effect_sizes = []

# 1a. Forest plot effect sizes (already computed)
forest_path = ANALYSIS_OUTPUT_DIR / "synthesis_analysis" / "forest_plot_effect_sizes.csv"
if forest_path.exists():
    forest_df = pd.read_csv(forest_path)
    print(f"   Loaded forest_plot_effect_sizes.csv: {len(forest_df)} rows")

    for _, row in forest_df.iterrows():
        effect_sizes.append({
            'outcome': row['outcome'],
            'effect_type': row['effect_type'],
            'effect_label': row['effect_label'],
            'analysis_type': row['analysis'],
            'effect_metric': 'β (standardized)' if 'Regression' in row['analysis'] else 'r',
            'effect_value': row['beta'],
            'se': row.get('se', np.nan),
            'ci_lower': row['ci_lower'],
            'ci_upper': row['ci_upper'],
            'p_value': row['p_value'],
            'n': row['n'],
            'r2': row.get('r2', np.nan),
            'dass_controlled': 'DASS-controlled' in str(row['analysis']),
            'source': 'synthesis_analysis'
        })

# 1b. Hierarchical regression results
hier_path = ANALYSIS_OUTPUT_DIR / "master_dass_controlled" / "hierarchical_regression_results.csv"
if hier_path.exists():
    hier_df = pd.read_csv(hier_path)
    print(f"   Loaded hierarchical_regression_results.csv: {len(hier_df)} rows")

    for _, row in hier_df.iterrows():
        # Model comparison F-test effect (delta R²)
        if pd.notna(row.get('delta_r2_3v2')):
            effect_sizes.append({
                'outcome': row['outcome'],
                'effect_type': 'Interaction ΔR²',
                'effect_label': f"{row['outcome']}: UCLA × Gender ΔR²",
                'analysis_type': 'Hierarchical Model Comparison',
                'effect_metric': 'ΔR²',
                'effect_value': row['delta_r2_3v2'],
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': row['p_3v2'],
                'n': row['n'],
                'r2': row['model3_r2'],
                'dass_controlled': True,
                'source': 'master_dass_controlled'
            })

        # Interaction beta
        if pd.notna(row.get('interaction_beta')):
            effect_sizes.append({
                'outcome': row['outcome'],
                'effect_type': 'Interaction β',
                'effect_label': f"{row['outcome']}: UCLA × Gender β",
                'analysis_type': 'Hierarchical Regression Model 3',
                'effect_metric': 'β (unstandardized)',
                'effect_value': row['interaction_beta'],
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': row['interaction_p'],
                'n': row['n'],
                'r2': row['model3_r2'],
                'dass_controlled': True,
                'source': 'master_dass_controlled'
            })

# 1c. Gender stratified coefficients
gender_path = ANALYSIS_OUTPUT_DIR / "synthesis_analysis" / "gender_stratified_coefficients.csv"
if gender_path.exists():
    gender_df = pd.read_csv(gender_path)
    print(f"   Loaded gender_stratified_coefficients.csv: {len(gender_df)} rows")

    for _, row in gender_df.iterrows():
        effect_sizes.append({
            'outcome': row['outcome'],
            'effect_type': f"{row['gender']} UCLA β",
            'effect_label': f"{row['outcome']}: {row['gender']} UCLA Effect",
            'analysis_type': 'Gender-Stratified Regression',
            'effect_metric': 'β (unstandardized)',
            'effect_value': row['beta_ucla'],
            'se': row['se_ucla'],
            'ci_lower': row['ci_lower'],
            'ci_upper': row['ci_upper'],
            'p_value': row['p_ucla'],
            'n': row['n'],
            'r2': row['r2'],
            'dass_controlled': True,
            'source': 'synthesis_analysis'
        })

# 1d. PRP comprehensive DASS controlled
prp_path = ANALYSIS_OUTPUT_DIR / "prp_comprehensive_dass_controlled" / "prp_dass_controlled_regression_results.csv"
if prp_path.exists():
    prp_df = pd.read_csv(prp_path)
    print(f"   Loaded prp_dass_controlled_regression_results.csv: {len(prp_df)} rows")

    for _, row in prp_df.iterrows():
        effect_sizes.append({
            'outcome': row.get('outcome', 'PRP'),
            'effect_type': row.get('effect_type', 'Regression'),
            'effect_label': row.get('effect_label', f"PRP: {row.get('predictor', 'UCLA')}"),
            'analysis_type': 'PRP Comprehensive DASS-Controlled',
            'effect_metric': 'β',
            'effect_value': row.get('beta', row.get('coefficient', np.nan)),
            'se': row.get('se', row.get('std_err', np.nan)),
            'ci_lower': row.get('ci_lower', np.nan),
            'ci_upper': row.get('ci_upper', np.nan),
            'p_value': row.get('p_value', row.get('pvalue', np.nan)),
            'n': row.get('n', np.nan),
            'r2': row.get('r2', np.nan),
            'dass_controlled': True,
            'source': 'prp_comprehensive'
        })

print(f"\n   Total effect sizes collected: {len(effect_sizes)}")


# ============================================================================
# 2. Compute from master dataset for key effects
# ============================================================================
print("\n[2] Computing additional effect sizes from master dataset...")

df = load_master_dataset()

# Define key outcome variables
outcomes = {
    'pe_rate': 'WCST PE Rate (%)',
    'wcst_accuracy': 'WCST Accuracy (%)',
    'prp_bottleneck': 'PRP Bottleneck (ms)',
    'stroop_interference': 'Stroop Interference (ms)'
}

# 2a. Cohen's d for high vs low UCLA groups (median split)
ucla_median = df['ucla_score'].median()
df['ucla_group'] = df['ucla_score'].apply(lambda x: 'High' if x > ucla_median else 'Low')

for col, label in outcomes.items():
    if col not in df.columns:
        continue

    valid = df.dropna(subset=[col, 'ucla_group'])
    high_group = valid[valid['ucla_group'] == 'High'][col]
    low_group = valid[valid['ucla_group'] == 'Low'][col]

    if len(high_group) < 5 or len(low_group) < 5:
        continue

    d = cohens_d(high_group, low_group)

    # Bootstrap CI for Cohen's d
    combined = np.concatenate([high_group.values, low_group.values])
    n_high, n_low = len(high_group), len(low_group)

    boot_d = []
    for _ in range(5000):
        boot_combined = np.random.choice(combined, size=len(combined), replace=True)
        boot_high = boot_combined[:n_high]
        boot_low = boot_combined[n_high:]
        boot_d.append(cohens_d(pd.Series(boot_high), pd.Series(boot_low)))

    ci_lower, ci_upper = np.percentile(boot_d, [2.5, 97.5])

    # t-test for p-value
    t_stat, p_val = stats.ttest_ind(high_group, low_group)

    effect_sizes.append({
        'outcome': label,
        'effect_type': "Cohen's d (High vs Low UCLA)",
        'effect_label': f"{label}: High vs Low UCLA (median split)",
        'analysis_type': 'Group Comparison (Median Split)',
        'effect_metric': "Cohen's d",
        'effect_value': d,
        'se': np.nan,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_val,
        'n': len(valid),
        'r2': np.nan,
        'dass_controlled': False,
        'source': 'computed_group_comparison'
    })

# 2b. Pearson correlations with bootstrap CI
for col, label in outcomes.items():
    if col not in df.columns or 'ucla_score' not in df.columns:
        continue

    valid = df.dropna(subset=[col, 'ucla_score'])
    if len(valid) < 10:
        continue

    r, p = stats.pearsonr(valid['ucla_score'], valid[col])

    # Bootstrap CI for correlation
    def corr_func(data):
        mid = len(data) // 2
        return np.corrcoef(data[:mid], data[mid:])[0, 1]

    combined = np.column_stack([valid['ucla_score'].values, valid[col].values]).flatten()

    boot_r = []
    for _ in range(5000):
        idx = np.random.choice(len(valid), size=len(valid), replace=True)
        boot_x = valid['ucla_score'].iloc[idx].values
        boot_y = valid[col].iloc[idx].values
        boot_r.append(np.corrcoef(boot_x, boot_y)[0, 1])

    ci_lower, ci_upper = np.percentile(boot_r, [2.5, 97.5])

    effect_sizes.append({
        'outcome': label,
        'effect_type': 'Correlation r',
        'effect_label': f"{label}: UCLA Correlation (Full Sample)",
        'analysis_type': 'Pearson Correlation',
        'effect_metric': 'r',
        'effect_value': r,
        'se': np.nan,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p,
        'n': len(valid),
        'r2': r**2,
        'dass_controlled': False,
        'source': 'computed_correlation'
    })

# 2c. Gender-specific Cohen's d
for gender in ['male', 'female']:
    gender_df = df[df['gender'] == gender]

    if len(gender_df) < 20:
        continue

    ucla_med_g = gender_df['ucla_score'].median()
    gender_df = gender_df.copy()
    gender_df['ucla_group'] = gender_df['ucla_score'].apply(lambda x: 'High' if x > ucla_med_g else 'Low')

    for col, label in outcomes.items():
        if col not in gender_df.columns:
            continue

        valid = gender_df.dropna(subset=[col, 'ucla_group'])
        high_group = valid[valid['ucla_group'] == 'High'][col]
        low_group = valid[valid['ucla_group'] == 'Low'][col]

        if len(high_group) < 3 or len(low_group) < 3:
            continue

        d = cohens_d(high_group, low_group)
        t_stat, p_val = stats.ttest_ind(high_group, low_group)

        # Bootstrap CI
        combined = np.concatenate([high_group.values, low_group.values])
        n_high, n_low = len(high_group), len(low_group)

        boot_d = []
        for _ in range(2000):
            boot_combined = np.random.choice(combined, size=len(combined), replace=True)
            boot_high = boot_combined[:n_high]
            boot_low = boot_combined[n_high:]
            bd = cohens_d(pd.Series(boot_high), pd.Series(boot_low))
            if not np.isnan(bd):
                boot_d.append(bd)

        if len(boot_d) > 100:
            ci_lower, ci_upper = np.percentile(boot_d, [2.5, 97.5])
        else:
            ci_lower, ci_upper = np.nan, np.nan

        effect_sizes.append({
            'outcome': label,
            'effect_type': f"Cohen's d ({gender.capitalize()})",
            'effect_label': f"{label}: High vs Low UCLA ({gender.capitalize()} only)",
            'analysis_type': f'Group Comparison ({gender.capitalize()} Median Split)',
            'effect_metric': "Cohen's d",
            'effect_value': d,
            'se': np.nan,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_val,
            'n': len(valid),
            'r2': np.nan,
            'dass_controlled': False,
            'source': f'computed_group_{gender}'
        })

print(f"   Total effect sizes after computation: {len(effect_sizes)}")


# ============================================================================
# 3. Create comprehensive summary table
# ============================================================================
print("\n[3] Creating comprehensive effect size summary table...")

effects_df = pd.DataFrame(effect_sizes)

# Add interpretation column
effects_df['interpretation'] = effects_df['effect_value'].apply(
    lambda x: interpret_effect_size(x) if pd.notna(x) else ''
)

# Add significance flag
effects_df['significant'] = effects_df['p_value'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
)

# Format for display
effects_df['effect_formatted'] = effects_df.apply(
    lambda row: f"{row['effect_value']:.3f}" if pd.notna(row['effect_value']) else '',
    axis=1
)

effects_df['ci_formatted'] = effects_df.apply(
    lambda row: format_ci(row['ci_lower'], row['ci_upper'])
    if pd.notna(row['ci_lower']) and pd.notna(row['ci_upper']) else '',
    axis=1
)

effects_df['p_formatted'] = effects_df['p_value'].apply(
    lambda p: format_pvalue(p) if pd.notna(p) else ''
)

# Sort by outcome and effect type
effects_df = effects_df.sort_values(['outcome', 'effect_type', 'dass_controlled'],
                                     ascending=[True, True, False])

# Save full table
effects_df.to_csv(OUTPUT_DIR / "effect_size_comprehensive.csv",
                  index=False, encoding='utf-8-sig')
print(f"   Saved: effect_size_comprehensive.csv ({len(effects_df)} rows)")


# ============================================================================
# 4. Create APA-style summary table
# ============================================================================
print("\n[4] Creating APA-style summary table...")

apa_table = effects_df[[
    'outcome', 'effect_type', 'effect_metric', 'effect_formatted',
    'ci_formatted', 'p_formatted', 'significant', 'n', 'dass_controlled', 'interpretation'
]].copy()

apa_table.columns = [
    'Outcome', 'Effect Type', 'Metric', 'Effect Size',
    '95% CI', 'p-value', 'Sig.', 'N', 'DASS Controlled', 'Interpretation'
]

apa_table.to_csv(OUTPUT_DIR / "effect_size_apa_table.csv",
                 index=False, encoding='utf-8-sig')
print(f"   Saved: effect_size_apa_table.csv")


# ============================================================================
# 5. Create Forest Plot
# ============================================================================
print("\n[5] Creating Forest Plot...")

# Filter for main effects (DASS-controlled where available)
plot_effects = effects_df[
    (effects_df['effect_metric'].isin(['β (standardized)', 'r', "Cohen's d"])) &
    (effects_df['ci_lower'].notna()) &
    (effects_df['ci_upper'].notna())
].copy()

# Take only key effects for readability
key_effects = plot_effects[
    plot_effects['effect_type'].str.contains('Interaction|Male|Female|Correlation', case=False)
].copy()

if len(key_effects) > 0:
    fig, ax = plt.subplots(figsize=(12, max(8, len(key_effects) * 0.4)))

    # Sort by effect size
    key_effects = key_effects.sort_values('effect_value')
    y_pos = np.arange(len(key_effects))

    # Plot
    colors = key_effects['p_value'].apply(
        lambda p: '#2ecc71' if p < 0.05 else '#95a5a6'
    )

    ax.hlines(y_pos, key_effects['ci_lower'], key_effects['ci_upper'],
              colors=colors, linewidth=2, alpha=0.7)
    ax.scatter(key_effects['effect_value'], y_pos, c=colors, s=100, zorder=3)

    # Reference line at 0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(key_effects['effect_label'], fontsize=9)
    ax.set_xlabel('Effect Size (standardized β / r / d)', fontsize=11)
    ax.set_title('Forest Plot: Effect Sizes with 95% CI\n(Green = p < .05, Gray = p ≥ .05)',
                 fontsize=13, fontweight='bold')

    # Add significance markers
    for i, row in enumerate(key_effects.itertuples()):
        sig_marker = '***' if row.p_value < 0.001 else ('**' if row.p_value < 0.01 else ('*' if row.p_value < 0.05 else ''))
        ax.annotate(sig_marker, xy=(row.ci_upper + 0.02, i), fontsize=10,
                   color='#e74c3c' if sig_marker else 'gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forest_plot_comprehensive.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: forest_plot_comprehensive.png")
else:
    print("   No suitable effects for forest plot")


# ============================================================================
# 6. Create outcome-specific summary
# ============================================================================
print("\n[6] Creating outcome-specific summary...")

outcome_summary = []
for outcome in effects_df['outcome'].unique():
    outcome_effects = effects_df[effects_df['outcome'] == outcome]

    # Get key statistics
    dass_controlled = outcome_effects[outcome_effects['dass_controlled'] == True]
    not_controlled = outcome_effects[outcome_effects['dass_controlled'] == False]

    # Significant effects
    sig_effects = outcome_effects[outcome_effects['p_value'] < 0.05]

    outcome_summary.append({
        'Outcome': outcome,
        'Total Effects': len(outcome_effects),
        'DASS-Controlled': len(dass_controlled),
        'Not Controlled': len(not_controlled),
        'Significant (p<.05)': len(sig_effects),
        'Largest Effect': outcome_effects['effect_value'].abs().max() if len(outcome_effects) > 0 else np.nan,
        'Mean |Effect|': outcome_effects['effect_value'].abs().mean() if len(outcome_effects) > 0 else np.nan
    })

outcome_summary_df = pd.DataFrame(outcome_summary)
outcome_summary_df.to_csv(OUTPUT_DIR / "outcome_effect_summary.csv",
                          index=False, encoding='utf-8-sig')
print(f"   Saved: outcome_effect_summary.csv")


# ============================================================================
# 7. Effect Size Conversion Table
# ============================================================================
print("\n[7] Creating effect size conversion table...")

conversion_table = effects_df[effects_df['effect_value'].notna()].copy()

# Add converted effect sizes
conversion_table['r_equivalent'] = conversion_table.apply(
    lambda row: row['effect_value'] if row['effect_metric'] == 'r'
    else d_to_r(row['effect_value']) if row['effect_metric'] == "Cohen's d"
    else np.nan,
    axis=1
)

conversion_table['d_equivalent'] = conversion_table.apply(
    lambda row: row['effect_value'] if row['effect_metric'] == "Cohen's d"
    else r_to_d(row['effect_value']) if row['effect_metric'] == 'r'
    else np.nan,
    axis=1
)

conversion_table = conversion_table[[
    'outcome', 'effect_type', 'effect_metric', 'effect_value',
    'r_equivalent', 'd_equivalent', 'p_value', 'n'
]]

conversion_table.to_csv(OUTPUT_DIR / "effect_size_conversions.csv",
                        index=False, encoding='utf-8-sig')
print(f"   Saved: effect_size_conversions.csv")


# ============================================================================
# 8. Print Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("EFFECT SIZE META-SUMMARY REPORT")
print("=" * 70)

print(f"\nTotal effect sizes collected: {len(effects_df)}")
print(f"  - DASS-controlled: {len(effects_df[effects_df['dass_controlled']])}")
print(f"  - Not controlled: {len(effects_df[~effects_df['dass_controlled']])}")

print(f"\nSignificance summary:")
print(f"  - p < .001 (***): {len(effects_df[effects_df['p_value'] < 0.001])}")
print(f"  - p < .01 (**):   {len(effects_df[(effects_df['p_value'] >= 0.001) & (effects_df['p_value'] < 0.01)])}")
print(f"  - p < .05 (*):    {len(effects_df[(effects_df['p_value'] >= 0.01) & (effects_df['p_value'] < 0.05)])}")
print(f"  - p >= .05:       {len(effects_df[effects_df['p_value'] >= 0.05])}")

print(f"\nEffect size interpretation distribution:")
interp_counts = effects_df['interpretation'].value_counts()
for interp, count in interp_counts.items():
    if interp:
        print(f"  - {interp}: {count}")

print(f"\nKey findings (DASS-controlled, p < .05):")
key_findings = effects_df[
    (effects_df['dass_controlled'] == True) &
    (effects_df['p_value'] < 0.05)
].sort_values('p_value')

for _, row in key_findings.head(10).iterrows():
    print(f"  • {row['effect_label']}")
    print(f"    Effect = {row['effect_value']:.3f}, p = {row['p_value']:.4f}")

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("=" * 70)

# Save summary report as text
with open(OUTPUT_DIR / "effect_size_summary_report.txt", 'w', encoding='utf-8') as f:
    f.write("EFFECT SIZE META-SUMMARY REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total effect sizes collected: {len(effects_df)}\n")
    f.write(f"  - DASS-controlled: {len(effects_df[effects_df['dass_controlled']])}\n")
    f.write(f"  - Not controlled: {len(effects_df[~effects_df['dass_controlled']])}\n\n")

    f.write("Significance summary:\n")
    f.write(f"  - p < .001 (***): {len(effects_df[effects_df['p_value'] < 0.001])}\n")
    f.write(f"  - p < .01 (**):   {len(effects_df[(effects_df['p_value'] >= 0.001) & (effects_df['p_value'] < 0.01)])}\n")
    f.write(f"  - p < .05 (*):    {len(effects_df[(effects_df['p_value'] >= 0.01) & (effects_df['p_value'] < 0.05)])}\n")
    f.write(f"  - p >= .05:       {len(effects_df[effects_df['p_value'] >= 0.05])}\n\n")

    f.write("Key findings (DASS-controlled, p < .05):\n")
    for _, row in key_findings.iterrows():
        f.write(f"  - {row['effect_label']}\n")
        f.write(f"    Effect = {row['effect_value']:.3f}, 95% CI {row['ci_formatted']}, p {row['p_formatted']}\n")

print("\nEffect size meta-summary analysis complete!")
