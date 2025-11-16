"""
Gender-Specific Pattern Summary
================================
Integrates all Stroop & PRP analyses to identify gender-specific vulnerabilities.

Compiles:
1. Learning curves (temporal dynamics)
2. Error cascades
3. RT variability
4. Gratton effect
5. Overall gender pattern
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results/analysis_outputs")
OUTPUT_DIR = Path("results/analysis_outputs/gender_pattern_summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GENDER-SPECIFIC PATTERN SUMMARY")
print("=" * 80)

# ============================================================================
# Collect All Gender-Stratified Results
# ============================================================================

summary = []

# 1. Stroop Learning Curves
try:
    stroop_lc = pd.read_csv(RESULTS_DIR / "stroop_deep_dive/learning_curves/gender_stratified_slopes.csv")
    for _, row in stroop_lc.iterrows():
        summary.append({
            'task': 'Stroop',
            'analysis': 'Learning Curves',
            'metric': 'Interference Slope',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['r'],
            'p': row['p'],
            'interpretation': 'Positive = interference increases over time'
        })
except:
    print("  Warning: Stroop learning curves not found")

# 2. PRP Learning Curves
try:
    prp_lc = pd.read_csv(RESULTS_DIR / "prp_deep_dive/learning_curves/gender_stratified_slopes.csv")
    for _, row in prp_lc.iterrows():
        summary.append({
            'task': 'PRP',
            'analysis': 'Learning Curves',
            'metric': 'Bottleneck Slope',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['r'],
            'p': row['p'],
            'interpretation': 'Negative = bottleneck decreases (learning)'
        })
except:
    print("  Warning: PRP learning curves not found")

# 3. Stroop Error Cascades
try:
    stroop_ec = pd.read_csv(RESULTS_DIR / "stroop_deep_dive/error_cascades/gender_stratified_correlations.csv")
    for _, row in stroop_ec.iterrows():
        summary.append({
            'task': 'Stroop',
            'analysis': 'Error Cascades',
            'metric': 'Max Error Run',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['max_error_run_r'],
            'p': row['max_error_run_p'],
            'interpretation': 'Negative = shorter error runs (WCST pattern)'
        })
        summary.append({
            'task': 'Stroop',
            'analysis': 'Error Cascades',
            'metric': 'Max Slow Run',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['max_slow_run_r'],
            'p': row['max_slow_run_p'],
            'interpretation': 'Slow response clustering'
        })
except:
    print("  Warning: Stroop error cascades not found")

# 4. PRP Dual-Error Cascades
try:
    prp_ec = pd.read_csv(RESULTS_DIR / "prp_deep_dive/error_cascades/gender_stratified_correlations.csv")
    for _, row in prp_ec.iterrows():
        summary.append({
            'task': 'PRP',
            'analysis': 'Dual-Error Cascades',
            'metric': 'Max Dual-Error Run',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['max_dual_error_run_r'],
            'p': row['max_dual_error_run_p'],
            'interpretation': 'Coordination breakdown clustering'
        })
        summary.append({
            'task': 'PRP',
            'analysis': 'Dual-Error Cascades',
            'metric': 'Max T2-Error Run',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['max_t2_error_run_r'],
            'p': row['max_t2_error_run_p'],
            'interpretation': 'T2-specific error clustering'
        })
except:
    print("  Warning: PRP dual-error cascades not found")

# 5. Stroop RT Variability
try:
    stroop_var = pd.read_csv(RESULTS_DIR / "stroop_deep_dive/rt_variability/gender_stratified_correlations.csv")
    for _, row in stroop_var.iterrows():
        summary.append({
            'task': 'Stroop',
            'analysis': 'RT Variability',
            'metric': 'Incongruent RT SD',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['incongruent_rt_sd_r'],
            'p': row['incongruent_rt_sd_p'],
            'interpretation': 'Attentional inconsistency'
        })
except:
    print("  Warning: Stroop RT variability not found")

# 6. PRP RT Variability
try:
    prp_var = pd.read_csv(RESULTS_DIR / "prp_deep_dive/rt_variability/gender_stratified_correlations.csv")
    for _, row in prp_var.iterrows():
        summary.append({
            'task': 'PRP',
            'analysis': 'RT Variability',
            'metric': 'T2 RT SD (short SOA)',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['t2_rt_sd_short_r'],
            'p': row['t2_rt_sd_short_p'],
            'interpretation': 'Coordination instability'
        })
        summary.append({
            'task': 'PRP',
            'analysis': 'RT Variability',
            'metric': 'T2 RT SD (long SOA)',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['t2_rt_sd_long_r'],
            'p': row['t2_rt_sd_long_p'],
            'interpretation': 'Baseline variability'
        })
except:
    print("  Warning: PRP RT variability not found")

# 7. Gratton Effect
try:
    gratton = pd.read_csv(RESULTS_DIR / "stroop_deep_dive/gratton_effect/gender_stratified_correlations.csv")
    for _, row in gratton.iterrows():
        summary.append({
            'task': 'Stroop',
            'analysis': 'Gratton Effect',
            'metric': 'Adaptation Magnitude',
            'gender': row['gender'],
            'n': row['n'],
            'r': row['adaptation_magnitude_r'],
            'p': row['adaptation_magnitude_p'],
            'interpretation': 'Negative = reduced conflict adaptation'
        })
except:
    print("  Warning: Gratton effect not found")

# Create summary dataframe
summary_df = pd.DataFrame(summary)

print(f"\n[1] Collected {len(summary_df)} gender-stratified results")

# Save
summary_df.to_csv(OUTPUT_DIR / "all_gender_effects.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Identify Significant Effects
# ============================================================================

print("\n[2] Identifying significant effects (p < 0.10)...")

sig_effects = summary_df[summary_df['p'] < 0.10].sort_values('p')

print(sig_effects[['task', 'analysis', 'metric', 'gender', 'r', 'p']])

sig_effects.to_csv(OUTPUT_DIR / "significant_effects.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Male vs Female Pattern Comparison
# ============================================================================

print("\n[3] Male vs Female pattern comparison...")

comparison = []
for (task, analysis, metric), group in summary_df.groupby(['task', 'analysis', 'metric']):
    if len(group) != 2:
        continue

    male_row = group[group['gender'] == 'male'].iloc[0] if len(group[group['gender'] == 'male']) > 0 else None
    female_row = group[group['gender'] == 'female'].iloc[0] if len(group[group['gender'] == 'female']) > 0 else None

    if male_row is None or female_row is None:
        continue

    male_r = male_row['r']
    female_r = female_row['r']
    male_p = male_row['p']
    female_p = female_row['p']

    # Determine pattern
    if male_p < 0.10 and female_p >= 0.10:
        pattern = "Male-specific"
    elif female_p < 0.10 and male_p >= 0.10:
        pattern = "Female-specific"
    elif male_p < 0.10 and female_p < 0.10:
        if np.sign(male_r) == np.sign(female_r):
            pattern = "Both (same direction)"
        else:
            pattern = "Both (opposite direction)"
    else:
        pattern = "Neither significant"

    comparison.append({
        'task': task,
        'analysis': analysis,
        'metric': metric,
        'male_r': male_r,
        'male_p': male_p,
        'female_r': female_r,
        'female_p': female_p,
        'pattern': pattern
    })

comparison_df = pd.DataFrame(comparison)

print(comparison_df[['task', 'metric', 'male_r', 'female_r', 'pattern']])

comparison_df.to_csv(OUTPUT_DIR / "male_vs_female_comparison.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# Summary Report
# ============================================================================

print("\n" + "="*80)
print("GENDER-SPECIFIC PATTERNS - SUMMARY")
print("="*80)

print(f"""
1. Total Effects Analyzed: {len(summary_df)}
   - Stroop: {len(summary_df[summary_df['task']=='Stroop'])}
   - PRP: {len(summary_df[summary_df['task']=='PRP'])}

2. Significant Effects (p < 0.10): {len(sig_effects)}

3. Pattern Distribution:
{comparison_df['pattern'].value_counts().to_string()}

4. Male-Specific Vulnerabilities:
{comparison_df[comparison_df['pattern']=='Male-specific'][['task', 'metric', 'male_r', 'male_p']].to_string(index=False)}

5. Female-Specific Patterns:
{comparison_df[comparison_df['pattern']=='Female-specific'][['task', 'metric', 'female_r', 'female_p']].to_string(index=False)}

6. Files Generated:
   - all_gender_effects.csv ({len(summary_df)} rows)
   - significant_effects.csv ({len(sig_effects)} rows)
   - male_vs_female_comparison.csv ({len(comparison_df)} rows)
""")

print("\n✅ Gender pattern summary complete!")
print("="*80)
