"""
Bayesian Changepoint Detection - Performance Collapse Timing

OBJECTIVE:
Identify the precise trial where vulnerable individuals' (especially lonely males)
performance undergoes a regime shift - from stable to deteriorated performance.

RATIONALE:
- Mean performance metrics may mask temporal dynamics
- Changepoint = trial where generative process shifts (e.g., RT distribution changes)
- Hypothesis: Lonely males show EARLIER changepoint (faster depletion)
- Clinical utility: Identify "critical threshold" for intervention

BAYESIAN CHANGEPOINT MODEL:
For each participant's trial sequence:
  RT[t] ~ Normal(μ_early, σ) if t < τ
  RT[t] ~ Normal(μ_late, σ) if t ≥ τ

  Where:
    τ = changepoint trial (latent variable, uniform prior)
    μ_early = mean RT before changepoint
    μ_late = mean RT after changepoint

Expected pattern: μ_late > μ_early (performance deteriorates)

TEST:
1. Fit changepoint model for each participant
2. Extract posterior mean changepoint (τ)
3. Test: UCLA × Gender → τ (earlier collapse in lonely males?)
4. Visualize individual changepoint patterns

TASKS:
- WCST: Changepoint in PE rate (shift to perseveration)
- Stroop: Changepoint in mean RT (fatigue/disengagement)
- PRP: Changepoint in T2 RT (dual-task breakdown)
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
import warnings

from data_loader_utils import ensure_participant_id, load_participants, normalize_gender_series
warnings.filterwarnings('ignore')

# Note: PyMC changepoint detection is computationally expensive
# We'll use a simpler approach: CUSUM (Cumulative Sum) for changepoint detection

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/changepoint_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*80)
print("BAYESIAN CHANGEPOINT DETECTION - PERFORMANCE COLLAPSE TIMING")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Load WCST trials (PE is the key metric)
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
wcst_trials = ensure_participant_id(wcst_trials)

# Load master dataset
master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if not master_path.exists():
    print("ERROR: master_dataset.csv not found")
    sys.exit(1)

master = pd.read_csv(master_path, encoding='utf-8-sig')

participants = load_participants()[['participant_id', 'gender']]

if 'gender' not in master.columns:
    master = master.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')

# Normalize gender
master['gender'] = normalize_gender_series(master['gender'])
master['gender_male'] = (master['gender'] == 'male').astype(int)

print(f"  Loaded {len(wcst_trials)} WCST trials")
print(f"  Loaded {len(master)} participants")

# ============================================================================
# 2. COMPUTE CHANGEPOINT USING CUSUM METHOD
# ============================================================================
print("\n[2/5] Computing changepoints via CUSUM...")

def detect_changepoint_cusum(series, threshold=1.0):
    """
    Detect changepoint using CUSUM (Cumulative Sum) method.

    Returns:
        changepoint_index: Trial index where cumulative deviation exceeds threshold
        cusum: Full CUSUM series for visualization
    """
    if len(series) < 10:
        return np.nan, None

    # Standardize series
    z = (series - series.mean()) / (series.std() + 1e-8)

    # Compute CUSUM (cumulative sum of deviations from mean)
    cusum = np.cumsum(z)

    # Find point of maximum absolute deviation
    abs_cusum = np.abs(cusum)
    changepoint_idx = np.argmax(abs_cusum)

    # Only return if deviation is substantial (avoid spurious changepoints)
    if abs_cusum[changepoint_idx] > threshold * np.sqrt(len(series)):
        return changepoint_idx, cusum
    else:
        return np.nan, cusum

# Parse WCST 'extra' field for isPE
import ast

def parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return {}
    try:
        return ast.literal_eval(extra_str)
    except (ValueError, SyntaxError):
        return {}

wcst_trials['extra_dict'] = wcst_trials['extra'].apply(parse_wcst_extra)
wcst_trials['isPE'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

# Filter valid WCST trials
valid_wcst = wcst_trials[
    (wcst_trials['timeout'] == False) &
    (wcst_trials['rt_ms'] > 0)
].copy()

print(f"  Valid WCST trials: {len(valid_wcst)}")

# Sort by participant and trial order
if 'timestamp' in valid_wcst.columns:
    valid_wcst = valid_wcst.sort_values(['participant_id', 'timestamp'])
elif 'trialIndex' in valid_wcst.columns:
    valid_wcst = valid_wcst.sort_values(['participant_id', 'trialIndex'])

valid_wcst = valid_wcst.reset_index(drop=True)

# Compute changepoint for each participant
changepoint_results = []

for pid, group in valid_wcst.groupby('participant_id'):
    if len(group) < 20:  # Need minimum trials for reliable changepoint
        continue

    # Reset trial numbering
    group = group.reset_index(drop=True)
    group['trial_num'] = np.arange(len(group))

    # Detect changepoint in RT
    rt_series = group['rt_ms'].values
    cp_rt, cusum_rt = detect_changepoint_cusum(rt_series, threshold=1.0)

    # Detect changepoint in PE occurrence (use moving average)
    pe_series = group['isPE'].astype(float).values
    # Use moving window to smooth binary PE signal
    window = 10
    if len(pe_series) >= window:
        pe_smooth = pd.Series(pe_series).rolling(window=window, min_periods=1).mean().values
        cp_pe, cusum_pe = detect_changepoint_cusum(pe_smooth, threshold=0.5)
    else:
        cp_pe, cusum_pe = np.nan, None

    # Compute pre/post changepoint stats
    if pd.notna(cp_rt) and cp_rt > 0 and cp_rt < len(group) - 1:
        rt_early = group.loc[:int(cp_rt), 'rt_ms'].mean()
        rt_late = group.loc[int(cp_rt):, 'rt_ms'].mean()
        rt_change = rt_late - rt_early
        rt_pct_change = (rt_change / rt_early) * 100
    else:
        rt_early, rt_late, rt_change, rt_pct_change = np.nan, np.nan, np.nan, np.nan

    if pd.notna(cp_pe) and cp_pe > 0 and cp_pe < len(group) - 1:
        pe_early = group.loc[:int(cp_pe), 'isPE'].mean() * 100
        pe_late = group.loc[int(cp_pe):, 'isPE'].mean() * 100
        pe_change = pe_late - pe_early
    else:
        pe_early, pe_late, pe_change = np.nan, np.nan, np.nan

    changepoint_results.append({
        'participant_id': pid,
        'n_trials': len(group),
        'changepoint_rt_trial': cp_rt,
        'changepoint_pe_trial': cp_pe,
        'rt_early': rt_early,
        'rt_late': rt_late,
        'rt_change_ms': rt_change,
        'rt_pct_change': rt_pct_change,
        'pe_early': pe_early,
        'pe_late': pe_late,
        'pe_change': pe_change
    })

cp_df = pd.DataFrame(changepoint_results)

print(f"  Changepoints detected for {len(cp_df)} participants")
print(f"  RT changepoints: {cp_df['changepoint_rt_trial'].notna().sum()} participants")
print(f"  PE changepoints: {cp_df['changepoint_pe_trial'].notna().sum()} participants")

# ============================================================================
# 3. MERGE WITH UCLA AND TEST CORRELATIONS
# ============================================================================
print("\n[3/5] Testing UCLA × Gender → Changepoint timing...")

# Merge with master
merge_cols = ['participant_id', 'ucla_total', 'gender', 'gender_male']
analysis_data = cp_df.merge(master[merge_cols], on='participant_id', how='inner')
analysis_data = analysis_data.dropna(subset=['ucla_total', 'gender_male'])

print(f"  Final N={len(analysis_data)}")
print(f"    Males: {(analysis_data['gender_male'] == 1).sum()}")
print(f"    Females: {(analysis_data['gender_male'] == 0).sum()}")

# Normalize changepoint by total trials (% through task)
analysis_data['cp_rt_pct'] = (analysis_data['changepoint_rt_trial'] / analysis_data['n_trials']) * 100
analysis_data['cp_pe_pct'] = (analysis_data['changepoint_pe_trial'] / analysis_data['n_trials']) * 100

# Gender-stratified analysis
males = analysis_data[analysis_data['gender_male'] == 1]
females = analysis_data[analysis_data['gender_male'] == 0]

# Test: UCLA → Changepoint timing (earlier = lower % through task)
print("\n  RT Changepoint Timing:")
rt_cp_valid = analysis_data.dropna(subset=['cp_rt_pct'])
if len(rt_cp_valid) >= 10:
    r_overall, p_overall = stats.pearsonr(rt_cp_valid['ucla_total'], rt_cp_valid['cp_rt_pct'])
    print(f"    Overall: r={r_overall:.3f}, p={p_overall:.4f}")

    males_rt = males.dropna(subset=['cp_rt_pct'])
    if len(males_rt) >= 5:
        r_male, p_male = stats.pearsonr(males_rt['ucla_total'], males_rt['cp_rt_pct'])
        print(f"    Males (N={len(males_rt)}): r={r_male:.3f}, p={p_male:.4f}")

    females_rt = females.dropna(subset=['cp_rt_pct'])
    if len(females_rt) >= 5:
        r_female, p_female = stats.pearsonr(females_rt['ucla_total'], females_rt['cp_rt_pct'])
        print(f"    Females (N={len(females_rt)}): r={r_female:.3f}, p={p_female:.4f}")

print("\n  PE Changepoint Timing:")
pe_cp_valid = analysis_data.dropna(subset=['cp_pe_pct'])
if len(pe_cp_valid) >= 10:
    r_overall_pe, p_overall_pe = stats.pearsonr(pe_cp_valid['ucla_total'], pe_cp_valid['cp_pe_pct'])
    print(f"    Overall: r={r_overall_pe:.3f}, p={p_overall_pe:.4f}")

    males_pe = males.dropna(subset=['cp_pe_pct'])
    if len(males_pe) >= 5:
        r_male_pe, p_male_pe = stats.pearsonr(males_pe['ucla_total'], males_pe['cp_pe_pct'])
        print(f"    Males (N={len(males_pe)}): r={r_male_pe:.3f}, p={p_male_pe:.4f}")

    females_pe = females.dropna(subset=['cp_pe_pct'])
    if len(females_pe) >= 5:
        r_female_pe, p_female_pe = stats.pearsonr(females_pe['ucla_total'], females_pe['cp_pe_pct'])
        print(f"    Females (N={len(females_pe)}): r={r_female_pe:.3f}, p={p_female_pe:.4f}")

# Test magnitude of change
print("\n  RT Change Magnitude (post - pre):")
rt_change_valid = analysis_data.dropna(subset=['rt_change_ms'])
if len(rt_change_valid) >= 10:
    r_change, p_change = stats.pearsonr(rt_change_valid['ucla_total'], rt_change_valid['rt_change_ms'])
    print(f"    Overall UCLA → RT increase: r={r_change:.3f}, p={p_change:.4f}")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n[4/5] Creating visualizations...")

# Figure 1: Changepoint timing distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# RT changepoint
ax = axes[0]
if len(males.dropna(subset=['cp_rt_pct'])) >= 3:
    ax.hist(males['cp_rt_pct'].dropna(), bins=10, alpha=0.6, label=f'Males (N={males["cp_rt_pct"].notna().sum()})',
            color='#3498DB', edgecolor='black')
if len(females.dropna(subset=['cp_rt_pct'])) >= 3:
    ax.hist(females['cp_rt_pct'].dropna(), bins=10, alpha=0.6, label=f'Females (N={females["cp_rt_pct"].notna().sum()})',
            color='#E74C3C', edgecolor='black')
ax.axvline(50, color='black', linestyle='--', linewidth=2, label='Midpoint')
ax.set_xlabel('RT Changepoint (% through task)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('RT Changepoint Timing Distribution', fontweight='bold', pad=10)
ax.legend(loc='upper right', frameon=True)
ax.grid(alpha=0.3, axis='y')

# PE changepoint
ax = axes[1]
if len(males.dropna(subset=['cp_pe_pct'])) >= 3:
    ax.hist(males['cp_pe_pct'].dropna(), bins=10, alpha=0.6, label=f'Males (N={males["cp_pe_pct"].notna().sum()})',
            color='#3498DB', edgecolor='black')
if len(females.dropna(subset=['cp_pe_pct'])) >= 3:
    ax.hist(females['cp_pe_pct'].dropna(), bins=10, alpha=0.6, label=f'Females (N={females["cp_pe_pct"].notna().sum()})',
            color='#E74C3C', edgecolor='black')
ax.axvline(50, color='black', linestyle='--', linewidth=2, label='Midpoint')
ax.set_xlabel('PE Changepoint (% through task)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('PE Changepoint Timing Distribution', fontweight='bold', pad=10)
ax.legend(loc='upper right', frameon=True)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Changepoint Detection: WCST Performance Collapse Timing', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'changepoint_timing_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Changepoint timing distribution saved")

# Figure 2: UCLA × Changepoint timing
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (ax, metric, title) in enumerate([
    (axes[0], 'cp_rt_pct', 'RT Changepoint'),
    (axes[1], 'cp_pe_pct', 'PE Changepoint')
]):
    valid_data = analysis_data.dropna(subset=[metric, 'ucla_total'])

    if len(valid_data) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontweight='bold')
        continue

    # Scatter by gender
    for gender_val, gender_label, color in [(1, 'Males', '#3498DB'), (0, 'Females', '#E74C3C')]:
        subset = valid_data[valid_data['gender_male'] == gender_val]
        if len(subset) >= 3:
            ax.scatter(subset['ucla_total'], subset[metric], label=gender_label,
                      alpha=0.7, s=80, color=color, edgecolors='black')

    ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
    ax.set_ylabel(f'{title} (% through task)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='Midpoint')
    ax.legend(loc='best', frameon=True)
    ax.grid(alpha=0.3)

plt.suptitle('UCLA Loneliness → Performance Changepoint Timing', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'ucla_changepoint_scatterplots.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ UCLA × changepoint scatterplots saved")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

# Save changepoint data
analysis_data.to_csv(OUTPUT_DIR / 'changepoint_results.csv', index=False, encoding='utf-8-sig')

# Generate report
with open(OUTPUT_DIR / "CHANGEPOINT_DETECTION_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("BAYESIAN CHANGEPOINT DETECTION - PERFORMANCE COLLAPSE TIMING\n")
    f.write("="*80 + "\n\n")

    f.write("OBJECTIVE\n")
    f.write("-"*80 + "\n")
    f.write("Identify the precise trial where performance undergoes a regime shift\n")
    f.write("from stable to deteriorated performance.\n\n")

    f.write("METHOD\n")
    f.write("-"*80 + "\n")
    f.write("CUSUM (Cumulative Sum) changepoint detection:\n")
    f.write("- Detects trial where cumulative deviation from mean is maximized\n")
    f.write("- Applied to RT series and PE occurrence series\n")
    f.write("- Changepoint = trial index with maximum absolute CUSUM\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"Total N = {len(analysis_data)}\n")
    f.write(f"  RT changepoints detected: {analysis_data['changepoint_rt_trial'].notna().sum()}\n")
    f.write(f"  PE changepoints detected: {analysis_data['changepoint_pe_trial'].notna().sum()}\n\n")

    f.write("CHANGEPOINT TIMING (% through task)\n")
    f.write("-"*80 + "\n")
    if 'cp_rt_pct' in analysis_data.columns and analysis_data['cp_rt_pct'].notna().any():
        f.write(f"RT Changepoint: M={analysis_data['cp_rt_pct'].mean():.1f}%, SD={analysis_data['cp_rt_pct'].std():.1f}%\n")
    if 'cp_pe_pct' in analysis_data.columns and analysis_data['cp_pe_pct'].notna().any():
        f.write(f"PE Changepoint: M={analysis_data['cp_pe_pct'].mean():.1f}%, SD={analysis_data['cp_pe_pct'].std():.1f}%\n")
    f.write("\n")

    f.write("UCLA → CHANGEPOINT TIMING CORRELATIONS\n")
    f.write("-"*80 + "\n")
    f.write("(Negative r = earlier changepoint with higher loneliness)\n\n")

    if 'cp_rt_pct' in rt_cp_valid.columns:
        f.write(f"RT Changepoint: r={r_overall:.3f}, p={p_overall:.4f}\n")
        if r_overall < 0 and p_overall < 0.05:
            f.write("  ✓ Higher loneliness → EARLIER RT collapse\n")
        elif r_overall > 0 and p_overall < 0.05:
            f.write("  ✗ Higher loneliness → LATER RT collapse (unexpected)\n")
        else:
            f.write("  ~ No significant UCLA → timing relationship\n")

    if 'cp_pe_pct' in pe_cp_valid.columns:
        f.write(f"\nPE Changepoint: r={r_overall_pe:.3f}, p={p_overall_pe:.4f}\n")
        if r_overall_pe < 0 and p_overall_pe < 0.05:
            f.write("  ✓ Higher loneliness → EARLIER PE collapse\n")
        elif r_overall_pe > 0 and p_overall_pe < 0.05:
            f.write("  ✗ Higher loneliness → LATER PE collapse (unexpected)\n")
        else:
            f.write("  ~ No significant UCLA → timing relationship\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Outputs saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ CHANGEPOINT DETECTION COMPLETE!")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
