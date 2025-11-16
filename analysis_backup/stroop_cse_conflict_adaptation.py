"""
Stroop Congruency Sequence Effect (CSE) - Conflict Adaptation Analysis

OBJECTIVE:
Test whether lonely males show impaired trial-to-trial cognitive control adjustments,
even if overall Stroop interference is null. CSE measures dynamic conflict adaptation.

RATIONALE:
- Mean Stroop interference has been consistently null in this sample
- CSE tests PROACTIVE CONTROL: adjusting strategy based on previous trial
- Gratton et al. (1992): Interference reduced after incongruent trials
- Hypothesis: Lonely males show reduced CSE (impaired adaptation)

CSE CALCULATION:
CSE = (cI - cC) - (iI - iC)
  where:
    cI = RT on incongruent trial after congruent trial
    cC = RT on congruent trial after congruent trial
    iI = RT on incongruent trial after incongruent trial
    iC = RT on congruent trial after incongruent trial

EXPECTED PATTERN:
- Typical CSE < 0 (negative): Interference smaller after incongruent trials
- Impaired CSE ≈ 0: No adaptation, flat interference regardless of N-1 trial
- UCLA × Gender → CSE: Lonely males show reduced (less negative) CSE

OUTPUTS:
1. Participant-level CSE scores
2. Gender-stratified UCLA correlations with CSE
3. Visualization of 2×2 interaction patterns
4. Report documenting conflict adaptation findings
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/stroop_cse")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*80)
print("STROOP CONGRUENCY SEQUENCE EFFECT (CSE) - CONFLICT ADAPTATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Load trial-level Stroop data
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')

# Normalize participant ID
if 'participantId' in stroop_trials.columns and 'participant_id' not in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})
elif 'participantId' in stroop_trials.columns and 'participant_id' in stroop_trials.columns:
    stroop_trials = stroop_trials.drop(columns=['participantId'])

# Load master dataset for UCLA and demographics
master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if not master_path.exists():
    print("ERROR: master_dataset.csv not found")
    sys.exit(1)

master = pd.read_csv(master_path, encoding='utf-8-sig')

# Load participants for gender if needed
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

# Merge gender into master if missing
if 'gender' not in master.columns:
    master = master.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')

# Normalize gender
gender_map = {'남성': 'male', '여성': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

print(f"  Loaded {len(stroop_trials)} Stroop trials")
print(f"  Loaded {len(master)} participants")

# ============================================================================
# 2. PREPARE TRIAL-LEVEL DATA FOR CSE
# ============================================================================
print("\n[2/5] Computing trial N-1 congruency sequences...")

# Ensure trials are sorted by participant and trial index
if 'timestamp' in stroop_trials.columns:
    stroop_trials = stroop_trials.sort_values(['participant_id', 'timestamp'])
elif 'trialIndex' in stroop_trials.columns:
    stroop_trials = stroop_trials.sort_values(['participant_id', 'trialIndex'])

# Determine congruency (use 'cond' or 'type' column)
if 'cond' in stroop_trials.columns:
    condition_col = 'cond'
elif 'type' in stroop_trials.columns:
    condition_col = 'type'
else:
    print("ERROR: No congruency column found in Stroop data")
    sys.exit(1)

# Normalize 'correct' column name
if 'isCorrect' not in stroop_trials.columns and 'correct' in stroop_trials.columns:
    stroop_trials['isCorrect'] = stroop_trials['correct']

# Filter valid trials (correct responses, no timeouts, valid RTs)
valid_stroop = stroop_trials[
    (stroop_trials['isCorrect'] == True) &
    (stroop_trials['timeout'] == False) &
    (stroop_trials['rt_ms'] > 200) &
    (stroop_trials['rt_ms'] < 3000)
].copy()

# Keep only congruent and incongruent trials (exclude neutral)
valid_stroop = valid_stroop[valid_stroop[condition_col].isin(['congruent', 'incongruent'])].copy()

print(f"  Valid trials after filtering: {len(valid_stroop)}")

# Code congruency as binary (1 = incongruent, 0 = congruent)
valid_stroop['is_incongruent'] = (valid_stroop[condition_col] == 'incongruent').astype(int)

# Compute N-1 congruency within each participant
valid_stroop['prev_is_incongruent'] = valid_stroop.groupby('participant_id')['is_incongruent'].shift(1)

# Drop first trial of each participant (no N-1 trial)
cse_data = valid_stroop.dropna(subset=['prev_is_incongruent']).copy()

print(f"  Trials with N-1 congruency coded: {len(cse_data)}")

# ============================================================================
# 3. CALCULATE CSE FOR EACH PARTICIPANT
# ============================================================================
print("\n[3/5] Calculating CSE scores per participant...")

cse_results = []

for pid, group in cse_data.groupby('participant_id'):
    # 2×2 conditions
    cC = group[(group['prev_is_incongruent'] == 0) & (group['is_incongruent'] == 0)]['rt_ms'].mean()  # congruent after congruent
    cI = group[(group['prev_is_incongruent'] == 0) & (group['is_incongruent'] == 1)]['rt_ms'].mean()  # incongruent after congruent
    iC = group[(group['prev_is_incongruent'] == 1) & (group['is_incongruent'] == 0)]['rt_ms'].mean()  # congruent after incongruent
    iI = group[(group['prev_is_incongruent'] == 1) & (group['is_incongruent'] == 1)]['rt_ms'].mean()  # incongruent after incongruent

    # Check if all conditions have data
    if pd.notna([cC, cI, iC, iI]).all():
        # Classic CSE = (cI - cC) - (iI - iC)
        # Negative CSE = reduced interference after incongruent trials (good adaptation)
        interference_after_congruent = cI - cC
        interference_after_incongruent = iI - iC
        cse = interference_after_congruent - interference_after_incongruent

        cse_results.append({
            'participant_id': pid,
            'cC': cC,
            'cI': cI,
            'iC': iC,
            'iI': iI,
            'interference_after_congruent': interference_after_congruent,
            'interference_after_incongruent': interference_after_incongruent,
            'cse': cse,
            'n_trials': len(group)
        })

cse_df = pd.DataFrame(cse_results)

print(f"  CSE computed for {len(cse_df)} participants")
print(f"  Mean CSE: {cse_df['cse'].mean():.1f} ms (SD={cse_df['cse'].std():.1f})")

# ============================================================================
# 4. MERGE WITH UCLA AND TEST CORRELATIONS
# ============================================================================
print("\n[4/5] Testing UCLA × Gender → CSE...")

# Merge CSE with master
merge_cols = ['participant_id', 'ucla_total', 'gender', 'gender_male']
if 'age' in master.columns:
    merge_cols.append('age')

analysis_data = cse_df.merge(
    master[merge_cols],
    on='participant_id',
    how='inner'
)

# Drop missing UCLA
analysis_data = analysis_data.dropna(subset=['ucla_total', 'gender_male'])

print(f"  Final N={len(analysis_data)}")
print(f"    Males: {(analysis_data['gender_male'] == 1).sum()}")
print(f"    Females: {(analysis_data['gender_male'] == 0).sum()}")

# Overall correlation: UCLA → CSE
r_overall, p_overall = stats.pearsonr(analysis_data['ucla_total'], analysis_data['cse'])
print(f"\n  Overall UCLA → CSE: r={r_overall:.3f}, p={p_overall:.4f}")

# Gender-stratified correlations
males = analysis_data[analysis_data['gender_male'] == 1]
females = analysis_data[analysis_data['gender_male'] == 0]

if len(males) >= 10:
    r_male, p_male = stats.pearsonr(males['ucla_total'], males['cse'])
    print(f"  Males (N={len(males)}): r={r_male:.3f}, p={p_male:.4f}")
else:
    r_male, p_male = np.nan, np.nan
    print(f"  Males: Insufficient N")

if len(females) >= 10:
    r_female, p_female = stats.pearsonr(females['ucla_total'], females['cse'])
    print(f"  Females (N={len(females)}): r={r_female:.3f}, p={p_female:.4f}")
else:
    r_female, p_female = np.nan, np.nan
    print(f"  Females: Insufficient N")

# Interaction test: UCLA × Gender → CSE
analysis_data['z_ucla'] = (analysis_data['ucla_total'] - analysis_data['ucla_total'].mean()) / analysis_data['ucla_total'].std()

formula = 'cse ~ z_ucla * gender_male'
model = smf.ols(formula, data=analysis_data).fit()

print(f"\n  Interaction Model: CSE ~ UCLA × Gender")
print(f"    UCLA main effect: β={model.params['z_ucla']:.2f}, p={model.pvalues['z_ucla']:.4f}")
print(f"    Gender main effect: β={model.params['gender_male']:.2f}, p={model.pvalues['gender_male']:.4f}")
print(f"    UCLA × Gender: β={model.params['z_ucla:gender_male']:.2f}, p={model.pvalues['z_ucla:gender_male']:.4f}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[5/5] Creating visualizations...")

# Figure 1: 2×2 Interaction plot (mean RTs by N-1 and N congruency)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, gender_val, gender_label, subset in [
    (axes[0], 1, 'Males', males),
    (axes[1], 0, 'Females', females)
]:
    if len(subset) < 5:
        ax.text(0.5, 0.5, 'Insufficient N', ha='center', va='center', fontsize=12)
        ax.set_title(f'{gender_label} (N={len(subset)})', fontweight='bold')
        continue

    # Aggregate across participants for visualization
    mean_cC = subset['cC'].mean()
    mean_cI = subset['cI'].mean()
    mean_iC = subset['iC'].mean()
    mean_iI = subset['iI'].mean()

    se_cC = subset['cC'].sem()
    se_cI = subset['cI'].sem()
    se_iC = subset['iC'].sem()
    se_iI = subset['iI'].sem()

    x = [1, 2]
    after_congruent = [mean_cC, mean_cI]
    after_incongruent = [mean_iC, mean_iI]
    se_congruent = [se_cC, se_cI]
    se_incongruent = [se_iC, se_iI]

    ax.errorbar(x, after_congruent, yerr=se_congruent, marker='o', linewidth=2,
                label='After Congruent', color='#3498DB', markersize=8, capsize=5)
    ax.errorbar(x, after_incongruent, yerr=se_incongruent, marker='s', linewidth=2,
                label='After Incongruent', color='#E74C3C', markersize=8, capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(['Congruent\nTrial (N)', 'Incongruent\nTrial (N)'])
    ax.set_ylabel('Mean RT (ms)', fontweight='bold')
    ax.set_title(f'{gender_label} (N={len(subset)})', fontweight='bold', pad=10)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(alpha=0.3)

    # Annotate CSE
    mean_cse = subset['cse'].mean()
    ax.text(0.98, 0.02, f'Mean CSE = {mean_cse:.1f} ms',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

plt.suptitle('Congruency Sequence Effect: Trial N-1 × Trial N Interaction',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'cse_interaction_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ CSE interaction pattern saved")

# Figure 2: UCLA × CSE scatterplots by gender
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, gender_val, gender_label, subset, r_val, p_val in [
    (axes[0], 1, 'Males', males, r_male, p_male),
    (axes[1], 0, 'Females', females, r_female, p_female)
]:
    if len(subset) < 10:
        ax.text(0.5, 0.5, 'Insufficient N', ha='center', va='center', fontsize=12)
        ax.set_title(f'{gender_label} (N={len(subset)})', fontweight='bold')
        continue

    ax.scatter(subset['ucla_total'], subset['cse'], alpha=0.7, s=80, edgecolors='black')

    # Regression line
    z = np.polyfit(subset['ucla_total'], subset['cse'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

    ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
    ax.set_ylabel('CSE (ms)', fontweight='bold')
    ax.set_title(f'{gender_label} (N={len(subset)})', fontweight='bold', pad=10)
    ax.grid(alpha=0.3)

    # Add correlation stats
    if pd.notna(r_val) and pd.notna(p_val):
        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(0.05, 0.95, f'r = {r_val:.3f}\np = {p_val:.4f} {sig_marker}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)

    # Add reference line at CSE=0
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, label='CSE = 0 (no adaptation)')

plt.suptitle('UCLA Loneliness → Congruency Sequence Effect',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'ucla_cse_scatterplots.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ UCLA × CSE scatterplots saved")

# Figure 3: Distribution of CSE by gender
fig, ax = plt.subplots(figsize=(10, 6))

if len(males) >= 5:
    ax.hist(males['cse'], bins=15, alpha=0.6, label=f'Males (N={len(males)})', color='#3498DB', edgecolor='black')
if len(females) >= 5:
    ax.hist(females['cse'], bins=15, alpha=0.6, label=f'Females (N={len(females)})', color='#E74C3C', edgecolor='black')

ax.axvline(0, color='black', linestyle='--', linewidth=2, label='CSE = 0 (no adaptation)')
ax.set_xlabel('CSE (ms)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Congruency Sequence Effect by Gender', fontweight='bold', pad=15)
ax.legend(loc='upper right', frameon=True)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cse_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ CSE distribution plot saved")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

# Save participant-level CSE data
analysis_data.to_csv(OUTPUT_DIR / 'cse_participant_scores.csv', index=False, encoding='utf-8-sig')

# Save summary statistics
summary_stats = pd.DataFrame({
    'Group': ['Overall', 'Males', 'Females'],
    'N': [len(analysis_data), len(males), len(females)],
    'Mean_CSE': [
        analysis_data['cse'].mean(),
        males['cse'].mean() if len(males) > 0 else np.nan,
        females['cse'].mean() if len(females) > 0 else np.nan
    ],
    'SD_CSE': [
        analysis_data['cse'].std(),
        males['cse'].std() if len(males) > 0 else np.nan,
        females['cse'].std() if len(females) > 0 else np.nan
    ],
    'UCLA_CSE_r': [r_overall, r_male, r_female],
    'UCLA_CSE_p': [p_overall, p_male, p_female]
})
summary_stats.to_csv(OUTPUT_DIR / 'cse_summary_stats.csv', index=False, encoding='utf-8-sig')

# Save regression model results
model_results = pd.DataFrame({
    'Predictor': ['Intercept', 'UCLA (z)', 'Gender (male)', 'UCLA × Gender'],
    'Coefficient': [
        model.params['Intercept'],
        model.params['z_ucla'],
        model.params['gender_male'],
        model.params['z_ucla:gender_male']
    ],
    'SE': [
        model.bse['Intercept'],
        model.bse['z_ucla'],
        model.bse['gender_male'],
        model.bse['z_ucla:gender_male']
    ],
    'p_value': [
        model.pvalues['Intercept'],
        model.pvalues['z_ucla'],
        model.pvalues['gender_male'],
        model.pvalues['z_ucla:gender_male']
    ],
    'CI_lower': model.conf_int()[0].values,
    'CI_upper': model.conf_int()[1].values
})
model_results.to_csv(OUTPUT_DIR / 'cse_regression_model.csv', index=False, encoding='utf-8-sig')

# Generate report
with open(OUTPUT_DIR / "CSE_CONFLICT_ADAPTATION_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("STROOP CONGRUENCY SEQUENCE EFFECT (CSE) - CONFLICT ADAPTATION ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write("OBJECTIVE\n")
    f.write("-"*80 + "\n")
    f.write("Test whether lonely individuals (especially males) show impaired trial-to-trial\n")
    f.write("cognitive control adjustments using the Congruency Sequence Effect (CSE).\n\n")

    f.write("BACKGROUND\n")
    f.write("-"*80 + "\n")
    f.write("CSE (Gratton et al., 1992) measures dynamic conflict adaptation:\n")
    f.write("- Negative CSE = Reduced interference after incongruent trials (good adaptation)\n")
    f.write("- CSE ≈ 0 = No adaptation, flat interference regardless of N-1 trial\n")
    f.write("- Positive CSE = Increased interference after incongruent trials (maladaptive)\n\n")

    f.write("CSE Formula: (cI - cC) - (iI - iC)\n")
    f.write("  where:\n")
    f.write("    cI = RT incongruent trial after congruent trial\n")
    f.write("    cC = RT congruent trial after congruent trial\n")
    f.write("    iI = RT incongruent trial after incongruent trial\n")
    f.write("    iC = RT congruent trial after incongruent trial\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"Total N = {len(analysis_data)}\n")
    f.write(f"  Males: {len(males)}\n")
    f.write(f"  Females: {len(females)}\n\n")

    f.write("OVERALL CSE PATTERN\n")
    f.write("-"*80 + "\n")
    f.write(f"Mean CSE = {analysis_data['cse'].mean():.1f} ms (SD = {analysis_data['cse'].std():.1f})\n")

    if analysis_data['cse'].mean() < 0:
        f.write("\n✓ Negative CSE indicates overall conflict adaptation is present\n")
    elif analysis_data['cse'].mean() > 0:
        f.write("\n✗ Positive CSE indicates maladaptive pattern (unusual)\n")
    else:
        f.write("\n~ CSE ≈ 0 indicates minimal conflict adaptation\n")

    f.write("\n")

    f.write("GENDER DIFFERENCES IN CSE\n")
    f.write("-"*80 + "\n")
    if len(males) >= 5 and len(females) >= 5:
        t_stat, t_p = stats.ttest_ind(males['cse'], females['cse'])
        f.write(f"Males: M = {males['cse'].mean():.1f} ms (SD = {males['cse'].std():.1f})\n")
        f.write(f"Females: M = {females['cse'].mean():.1f} ms (SD = {females['cse'].std():.1f})\n")
        f.write(f"Independent t-test: t = {t_stat:.2f}, p = {t_p:.4f}\n\n")

        if t_p < 0.05:
            if males['cse'].mean() > females['cse'].mean():
                f.write("✓ Males show significantly REDUCED conflict adaptation (less negative CSE)\n")
            else:
                f.write("✓ Females show significantly REDUCED conflict adaptation (less negative CSE)\n")
        else:
            f.write("~ No significant gender difference in baseline CSE\n")

    f.write("\n")

    f.write("UCLA → CSE CORRELATIONS\n")
    f.write("-"*80 + "\n")
    f.write(f"Overall: r = {r_overall:.3f}, p = {p_overall:.4f}\n")

    if pd.notna(r_male) and pd.notna(p_male):
        f.write(f"Males (N={len(males)}): r = {r_male:.3f}, p = {p_male:.4f}\n")
        if p_male < 0.05 and r_male > 0:
            f.write("  ✓ Higher loneliness → REDUCED adaptation (less negative CSE)\n")
        elif p_male < 0.05 and r_male < 0:
            f.write("  ✗ Higher loneliness → ENHANCED adaptation (more negative CSE) [unexpected]\n")
        else:
            f.write("  ~ No significant UCLA → CSE relationship in males\n")

    if pd.notna(r_female) and pd.notna(p_female):
        f.write(f"Females (N={len(females)}): r = {r_female:.3f}, p = {p_female:.4f}\n")
        if p_female < 0.05 and r_female > 0:
            f.write("  ✓ Higher loneliness → REDUCED adaptation (less negative CSE)\n")
        elif p_female < 0.05 and r_female < 0:
            f.write("  ✗ Higher loneliness → ENHANCED adaptation (more negative CSE) [unexpected]\n")
        else:
            f.write("  ~ No significant UCLA → CSE relationship in females\n")

    f.write("\n")

    f.write("INTERACTION MODEL: CSE ~ UCLA × Gender\n")
    f.write("-"*80 + "\n")
    f.write(f"UCLA main effect: β = {model.params['z_ucla']:.2f}, p = {model.pvalues['z_ucla']:.4f}\n")
    f.write(f"Gender main effect: β = {model.params['gender_male']:.2f}, p = {model.pvalues['gender_male']:.4f}\n")
    f.write(f"UCLA × Gender: β = {model.params['z_ucla:gender_male']:.2f}, p = {model.pvalues['z_ucla:gender_male']:.4f}\n")
    f.write(f"Model R² = {model.rsquared:.3f}\n\n")

    if model.pvalues['z_ucla:gender_male'] < 0.05:
        if model.params['z_ucla:gender_male'] > 0:
            f.write("✓ SIGNIFICANT INTERACTION: Loneliness predicts REDUCED adaptation in males\n")
            f.write("  (or ENHANCED adaptation in females)\n")
        else:
            f.write("✓ SIGNIFICANT INTERACTION: Loneliness predicts ENHANCED adaptation in males\n")
            f.write("  (or REDUCED adaptation in females)\n")
    else:
        f.write("~ No significant UCLA × Gender interaction\n")
        f.write("  Loneliness → CSE relationship similar across genders\n")

    f.write("\n")

    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")

    # Determine key finding
    if model.pvalues['z_ucla:gender_male'] < 0.05:
        f.write("FINDING: Gender-specific loneliness effects on conflict adaptation detected.\n\n")
    elif p_male < 0.05 or p_female < 0.05:
        f.write("FINDING: Loneliness affects conflict adaptation, but not gender-specifically.\n\n")
    else:
        f.write("FINDING: No significant loneliness → CSE relationship detected.\n\n")

    f.write("Theoretical implications:\n")
    f.write("- CSE tests PROACTIVE control (anticipatory strategy adjustment)\n")
    f.write("- Distinct from REACTIVE control (mean interference)\n")
    f.write("- Impaired CSE suggests reduced cognitive flexibility\n")
    f.write("- Can explain null mean Stroop findings if baseline is low\n\n")

    f.write("LIMITATIONS\n")
    f.write("-"*80 + "\n")
    f.write("1. CSE requires sufficient trials in all 4 cells (cC, cI, iC, iI)\n")
    f.write("2. Practice effects may reduce CSE magnitude over task duration\n")
    f.write("3. Assumes stable conflict adaptation strategy across all trials\n")
    f.write("4. Does not distinguish proactive from reactive control mechanisms\n\n")

    f.write("NEXT STEPS\n")
    f.write("-"*80 + "\n")
    f.write("1. Test CSE changes over time (early vs late blocks)\n")
    f.write("2. Link CSE to PRP bottleneck (cross-task control flexibility)\n")
    f.write("3. Test CSE × DASS anxiety (anxiety-driven hypervigilance)\n")
    f.write("4. Trial-level MVPA to predict vulnerability moments\n\n")

    f.write("="*80 + "\n")
    f.write("OUTPUT FILES\n")
    f.write("="*80 + "\n")
    f.write("1. cse_participant_scores.csv - Individual CSE scores\n")
    f.write("2. cse_summary_stats.csv - Descriptive statistics by gender\n")
    f.write("3. cse_regression_model.csv - UCLA × Gender interaction model\n")
    f.write("4. cse_interaction_pattern.png - 2×2 RT pattern visualization\n")
    f.write("5. ucla_cse_scatterplots.png - Loneliness correlations\n")
    f.write("6. cse_distribution.png - CSE distribution by gender\n")
    f.write("7. CSE_CONFLICT_ADAPTATION_REPORT.txt - This report\n\n")

    f.write(f"Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print("\n" + "="*80)
print("✓ STROOP CSE ANALYSIS COMPLETE!")
print("="*80)
print(f"\nKey Findings:")
print(f"  Mean CSE: {analysis_data['cse'].mean():.1f} ms")
print(f"  Overall UCLA → CSE: r={r_overall:.3f}, p={p_overall:.4f}")
if pd.notna(r_male):
    print(f"  Males UCLA → CSE: r={r_male:.3f}, p={p_male:.4f}")
if pd.notna(r_female):
    print(f"  Females UCLA → CSE: r={r_female:.3f}, p={p_female:.4f}")
print(f"  Interaction p-value: {model.pvalues['z_ucla:gender_male']:.4f}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
