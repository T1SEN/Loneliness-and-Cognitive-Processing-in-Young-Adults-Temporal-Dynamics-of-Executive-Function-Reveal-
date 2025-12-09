"""
Tier 2.2: Hidden Markov Model (HMM) for Attentional States

Identifies discrete "on-task" (focused) vs "off-task" (lapsed) states from RT sequences
and tests whether UCLA loneliness predicts state occupancy and transition dynamics.

Author: Claude Code
Date: 2025-12-03
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_wcst_trials, load_prp_trials, load_stroop_trials

# Output directory
OUTPUT_DIR = Path("results/analysis_outputs/hmm_attentional_states")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("TIER 2.2: HIDDEN MARKOV MODEL (HMM) - ATTENTIONAL STATES")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading data...")

# Load master dataset (demographics, UCLA, DASS)
df_master = load_master_dataset()
print(f"  Master dataset: {len(df_master)} participants")

# Load trial data - WCST has most trials (~128 per participant), best for HMM
wcst_df, wcst_info = load_wcst_trials()
print(f"  WCST: {len(wcst_df)} trials from {wcst_df['participant_id'].nunique()} participants")

# ============================================================================
# STEP 2: Fit 2-State Gaussian HMM per Participant
# ============================================================================
print("\nSTEP 2: Fitting 2-state Gaussian HMM per participant...")

def fit_hmm_2state(rt_sequence, n_iter=100, random_state=42):
    """
    Fit 2-state Gaussian HMM to RT sequence.

    Returns:
    - model: Fitted HMM model
    - state_sequence: Most likely state sequence (0 or 1)
    - log_likelihood: Model log-likelihood
    - converged: Whether model converged
    """
    if len(rt_sequence) < 10:
        return None, None, None, False

    # Reshape for hmmlearn (needs column vector)
    X = rt_sequence.reshape(-1, 1)

    # Initialize 2-state Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        init_params="stmc"  # Initialize: startprob, transmat, means, covars
    )

    try:
        # Fit model
        model.fit(X)

        # Predict most likely state sequence
        state_sequence = model.predict(X)

        # Compute log-likelihood
        log_likelihood = model.score(X)

        return model, state_sequence, log_likelihood, model.monitor_.converged

    except Exception as e:
        print(f"    HMM fitting error: {e}")
        return None, None, None, False

hmm_results = []

for pid in wcst_df['participant_id'].unique():
    if pd.isna(pid):
        continue

    pid_data = wcst_df[wcst_df['participant_id'] == pid].copy()

    # Sort by trial index
    if 'trialIndex' in pid_data.columns:
        pid_data = pid_data.sort_values('trialIndex')

    # Get RT sequence
    rt_sequence = pid_data['rt_ms'].values

    # Remove NaN values
    valid_mask = ~np.isnan(rt_sequence)
    rt_sequence = rt_sequence[valid_mask]

    if len(rt_sequence) < 20:  # Need at least 20 trials for stable HMM
        continue

    # Fit HMM
    model, state_seq, log_lik, converged = fit_hmm_2state(rt_sequence, n_iter=100, random_state=42)

    if model is None:
        continue

    # Extract state statistics
    # Identify which state is "focused" (lower mean RT) vs "lapsed" (higher mean RT)
    state_means = model.means_.flatten()
    focused_state = np.argmin(state_means)
    lapsed_state = np.argmax(state_means)

    # State occupancy (% trials in each state)
    focused_occupancy = np.mean(state_seq == focused_state)
    lapsed_occupancy = np.mean(state_seq == lapsed_state)

    # Transition probabilities
    transmat = model.transmat_
    p_focused_to_lapsed = transmat[focused_state, lapsed_state]
    p_lapsed_to_focused = transmat[lapsed_state, focused_state]
    p_focused_stay = transmat[focused_state, focused_state]
    p_lapsed_stay = transmat[lapsed_state, lapsed_state]

    # State means and variances
    focused_mean = state_means[focused_state]
    lapsed_mean = state_means[lapsed_state]
    focused_std = np.sqrt(model.covars_[focused_state].flatten()[0])
    lapsed_std = np.sqrt(model.covars_[lapsed_state].flatten()[0])

    hmm_results.append({
        'participant_id': pid,
        'n_trials': len(rt_sequence),
        'log_likelihood': log_lik,
        'converged': converged,
        'focused_state_id': focused_state,
        'lapsed_state_id': lapsed_state,
        'focused_occupancy': focused_occupancy,
        'lapsed_occupancy': lapsed_occupancy,
        'p_focused_to_lapsed': p_focused_to_lapsed,
        'p_lapsed_to_focused': p_lapsed_to_focused,
        'p_focused_stay': p_focused_stay,
        'p_lapsed_stay': p_lapsed_stay,
        'focused_mean_rt': focused_mean,
        'lapsed_mean_rt': lapsed_mean,
        'focused_sd_rt': focused_std,
        'lapsed_sd_rt': lapsed_std,
        'rt_difference': lapsed_mean - focused_mean  # How much slower in lapsed state
    })

hmm_df = pd.DataFrame(hmm_results)
print(f"  Fitted HMM for {len(hmm_df)} participants")
print(f"  Converged models: {hmm_df['converged'].sum()} ({hmm_df['converged'].mean()*100:.1f}%)")

# Save raw HMM results
hmm_df.to_csv(OUTPUT_DIR / "hmm_results_raw.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STEP 3: Descriptive Statistics
# ============================================================================
print("\nSTEP 3: Descriptive statistics...")

desc_stats = {
    'Metric': [],
    'Mean': [],
    'SD': [],
    'Min': [],
    'Max': []
}

for metric in ['lapsed_occupancy', 'p_focused_to_lapsed', 'p_lapsed_to_focused',
               'focused_mean_rt', 'lapsed_mean_rt', 'rt_difference']:
    desc_stats['Metric'].append(metric)
    desc_stats['Mean'].append(hmm_df[metric].mean())
    desc_stats['SD'].append(hmm_df[metric].std())
    desc_stats['Min'].append(hmm_df[metric].min())
    desc_stats['Max'].append(hmm_df[metric].max())

desc_df = pd.DataFrame(desc_stats)
print("\nHMM State Metrics:")
print(desc_df.round(3))

desc_df.to_csv(OUTPUT_DIR / "hmm_descriptive_stats.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STEP 4: Merge with UCLA/DASS
# ============================================================================
print("\nSTEP 4: Merging with UCLA/DASS...")

df = hmm_df.merge(df_master, on='participant_id', how='inner')
print(f"  Merged dataset: {len(df)} participants")

# Ensure required columns
if 'gender_male' not in df.columns:
    df['gender_male'] = (df['gender'].str.lower() == 'male').astype(int)

# Standardize predictors
for col in ['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    if col in df.columns:
        df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()

# Save merged data
df.to_csv(OUTPUT_DIR / "hmm_merged_with_predictors.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STEP 5: Regression Analysis - UCLA Predicting HMM Metrics
# ============================================================================
print("\nSTEP 5: Regression analysis - UCLA predicting HMM metrics...")

outcomes = [
    ('lapsed_occupancy', 'Lapsed State Occupancy (%)'),
    ('p_focused_to_lapsed', 'P(Focus → Lapse)'),
    ('p_lapsed_to_focused', 'P(Lapse → Focus)'),
    ('p_lapsed_stay', 'P(Lapse → Lapse)'),
    ('rt_difference', 'RT Difference (Lapsed - Focused)')
]

regression_results = []

for outcome_var, outcome_label in outcomes:
    print(f"\n  {outcome_label}:")

    # Formula: outcome ~ UCLA + Gender + DASS + age
    formula = f"{outcome_var} ~ z_ucla_total + C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

    try:
        model = smf.ols(formula, data=df).fit()

        # Extract UCLA coefficient
        ucla_coef = model.params.get('z_ucla_total', np.nan)
        ucla_se = model.bse.get('z_ucla_total', np.nan)
        ucla_t = model.tvalues.get('z_ucla_total', np.nan)
        ucla_p = model.pvalues.get('z_ucla_total', np.nan)
        ucla_ci = model.conf_int().loc['z_ucla_total'] if 'z_ucla_total' in model.conf_int().index else [np.nan, np.nan]

        regression_results.append({
            'outcome': outcome_var,
            'outcome_label': outcome_label,
            'n': len(df),
            'ucla_beta': ucla_coef,
            'ucla_se': ucla_se,
            'ucla_t': ucla_t,
            'ucla_p': ucla_p,
            'ucla_ci_lower': ucla_ci[0],
            'ucla_ci_upper': ucla_ci[1],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj
        })

        sig = "***" if ucla_p < 0.001 else "**" if ucla_p < 0.01 else "*" if ucla_p < 0.05 else ""
        print(f"    UCLA β={ucla_coef:.4f}, p={ucla_p:.4f} {sig}")

        # Save full model summary
        with open(OUTPUT_DIR / f"regression_{outcome_var}.txt", 'w', encoding='utf-8') as f:
            f.write(str(model.summary()))

    except Exception as e:
        print(f"    ERROR: {e}")

regression_df = pd.DataFrame(regression_results)
regression_df.to_csv(OUTPUT_DIR / "regression_results.csv", index=False, encoding='utf-8-sig')

# ============================================================================
# STEP 6: Visualization - State Occupancy by UCLA
# ============================================================================
print("\nSTEP 6: Creating visualizations...")

# Add UCLA tertiles
df['ucla_tertile'] = pd.qcut(df['ucla_total'], q=3, labels=['Low', 'Medium', 'High'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('HMM Attentional State Metrics by UCLA Loneliness', fontsize=16, fontweight='bold')

# Plot 1: Lapsed occupancy by UCLA tertile
ax1 = axes[0, 0]
sns.boxplot(data=df, x='ucla_tertile', y='lapsed_occupancy', palette='RdYlGn_r', ax=ax1)
ax1.set_xlabel('UCLA Tertile', fontsize=12)
ax1.set_ylabel('Lapsed State Occupancy (%)', fontsize=12)
ax1.set_title('Time Spent in Lapsed State', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Lapsed occupancy vs UCLA scatter
ax2 = axes[0, 1]
ax2.scatter(df['ucla_total'], df['lapsed_occupancy'], alpha=0.5, s=50)
try:
    z = np.polyfit(df['ucla_total'], df['lapsed_occupancy'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ucla_total'].min(), df['ucla_total'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r--', linewidth=2)
    r, p_val = stats.pearsonr(df['ucla_total'], df['lapsed_occupancy'])
    ax2.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3f}', transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
except:
    pass
ax2.set_xlabel('UCLA Loneliness Score', fontsize=12)
ax2.set_ylabel('Lapsed State Occupancy (%)', fontsize=12)
ax2.set_title('Lapsed Occupancy vs UCLA', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: P(Focus → Lapse) vs UCLA
ax3 = axes[0, 2]
ax3.scatter(df['ucla_total'], df['p_focused_to_lapsed'], alpha=0.5, s=50, color='orange')
try:
    z = np.polyfit(df['ucla_total'], df['p_focused_to_lapsed'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ucla_total'].min(), df['ucla_total'].max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', linewidth=2)
    r, p_val = stats.pearsonr(df['ucla_total'], df['p_focused_to_lapsed'])
    ax3.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3f}', transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
except:
    pass
ax3.set_xlabel('UCLA Loneliness Score', fontsize=12)
ax3.set_ylabel('P(Focus → Lapse)', fontsize=12)
ax3.set_title('Transition to Lapse State', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: P(Lapse → Focus) vs UCLA
ax4 = axes[1, 0]
ax4.scatter(df['ucla_total'], df['p_lapsed_to_focused'], alpha=0.5, s=50, color='green')
try:
    z = np.polyfit(df['ucla_total'], df['p_lapsed_to_focused'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ucla_total'].min(), df['ucla_total'].max(), 100)
    ax4.plot(x_line, p(x_line), 'r--', linewidth=2)
    r, p_val = stats.pearsonr(df['ucla_total'], df['p_lapsed_to_focused'])
    ax4.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3f}', transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
except:
    pass
ax4.set_xlabel('UCLA Loneliness Score', fontsize=12)
ax4.set_ylabel('P(Lapse → Focus)', fontsize=12)
ax4.set_title('Recovery from Lapse State', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: RT difference (Lapsed - Focused) vs UCLA
ax5 = axes[1, 1]
ax5.scatter(df['ucla_total'], df['rt_difference'], alpha=0.5, s=50, color='purple')
try:
    z = np.polyfit(df['ucla_total'], df['rt_difference'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ucla_total'].min(), df['ucla_total'].max(), 100)
    ax5.plot(x_line, p(x_line), 'r--', linewidth=2)
    r, p_val = stats.pearsonr(df['ucla_total'], df['rt_difference'])
    ax5.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3f}', transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
except:
    pass
ax5.set_xlabel('UCLA Loneliness Score', fontsize=12)
ax5.set_ylabel('RT Difference (ms)', fontsize=12)
ax5.set_title('Lapsed State Slowness', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: State means by UCLA tertile
ax6 = axes[1, 2]
focused_means = df.groupby('ucla_tertile')['focused_mean_rt'].mean()
lapsed_means = df.groupby('ucla_tertile')['lapsed_mean_rt'].mean()
x_pos = np.arange(len(focused_means))
width = 0.35
ax6.bar(x_pos - width/2, focused_means, width, label='Focused State', color='#2ecc71')
ax6.bar(x_pos + width/2, lapsed_means, width, label='Lapsed State', color='#e74c3c')
ax6.set_xlabel('UCLA Tertile', fontsize=12)
ax6.set_ylabel('Mean RT (ms)', fontsize=12)
ax6.set_title('State RTs by UCLA', fontsize=13, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(['Low', 'Medium', 'High'])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hmm_metrics_by_ucla.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: hmm_metrics_by_ucla.png")

# ============================================================================
# STEP 7: Example State Sequences Visualization
# ============================================================================

# Select 3 example participants (low, medium, high UCLA)
df_sorted = df.sort_values('ucla_total')
example_pids = [
    df_sorted.iloc[0]['participant_id'],  # Low UCLA
    df_sorted.iloc[len(df_sorted)//2]['participant_id'],  # Medium UCLA
    df_sorted.iloc[-1]['participant_id']  # High UCLA
]

fig, axes = plt.subplots(3, 1, figsize=(16, 10))
fig.suptitle('Example HMM State Sequences by UCLA Loneliness', fontsize=16, fontweight='bold')

for idx, (pid, ax) in enumerate(zip(example_pids, axes)):
    # Get participant data
    pid_data = wcst_df[wcst_df['participant_id'] == pid].copy()
    if 'trialIndex' in pid_data.columns:
        pid_data = pid_data.sort_values('trialIndex')

    rt_sequence = pid_data['rt_ms'].values
    valid_mask = ~np.isnan(rt_sequence)
    rt_sequence = rt_sequence[valid_mask]

    if len(rt_sequence) < 20:
        continue

    # Fit HMM
    model, state_seq, _, _ = fit_hmm_2state(rt_sequence, random_state=42)

    if model is None:
        continue

    # Get UCLA score
    ucla_score = df[df['participant_id'] == pid]['ucla_total'].values[0]
    lapsed_occ = df[df['participant_id'] == pid]['lapsed_occupancy'].values[0]

    # Plot RT with state coloring
    trial_nums = np.arange(len(rt_sequence))

    # Color-code by state
    state_means = model.means_.flatten()
    focused_state = np.argmin(state_means)

    colors = ['#2ecc71' if s == focused_state else '#e74c3c' for s in state_seq]

    ax.scatter(trial_nums, rt_sequence, c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Trial Number', fontsize=11)
    ax.set_ylabel('Reaction Time (ms)', fontsize=11)

    tertile_label = ['Low', 'Medium', 'High'][idx]
    ax.set_title(f'UCLA {tertile_label} (Score={ucla_score:.1f}, Lapsed Occupancy={lapsed_occ*100:.1f}%)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Focused State'),
        Patch(facecolor='#e74c3c', label='Lapsed State')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "example_state_sequences.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: example_state_sequences.png")

# ============================================================================
# STEP 8: Summary Report
# ============================================================================
print("\nSTEP 8: Generating summary report...")

summary_lines = [
    "=" * 80,
    "TIER 2.2: HIDDEN MARKOV MODEL (HMM) - ATTENTIONAL STATES SUMMARY",
    "=" * 80,
    "",
    "RESEARCH QUESTION:",
    "Does UCLA loneliness predict attentional state dynamics (lapse frequency,",
    "transition probabilities) as measured by 2-state Gaussian HMM?",
    "",
    "=" * 80,
    "DATA SUMMARY",
    "=" * 80,
    f"Total participants with HMM fits: {len(hmm_df)}",
    f"Converged models: {hmm_df['converged'].sum()} ({hmm_df['converged'].mean()*100:.1f}%)",
    f"Mean trials per participant: {hmm_df['n_trials'].mean():.1f} (SD={hmm_df['n_trials'].std():.1f})",
    "",
    "=" * 80,
    "HMM STATE CHARACTERISTICS (Mean ± SD)",
    "=" * 80,
    f"Lapsed state occupancy: {hmm_df['lapsed_occupancy'].mean()*100:.1f}% ± {hmm_df['lapsed_occupancy'].std()*100:.1f}%",
    f"Focused mean RT: {hmm_df['focused_mean_rt'].mean():.1f} ± {hmm_df['focused_mean_rt'].std():.1f} ms",
    f"Lapsed mean RT: {hmm_df['lapsed_mean_rt'].mean():.1f} ± {hmm_df['lapsed_mean_rt'].std():.1f} ms",
    f"RT difference (Lapsed - Focused): {hmm_df['rt_difference'].mean():.1f} ± {hmm_df['rt_difference'].std():.1f} ms",
    "",
    f"P(Focus → Lapse): {hmm_df['p_focused_to_lapsed'].mean():.3f} ± {hmm_df['p_focused_to_lapsed'].std():.3f}",
    f"P(Lapse → Focus): {hmm_df['p_lapsed_to_focused'].mean():.3f} ± {hmm_df['p_lapsed_to_focused'].std():.3f}",
    f"P(Focus → Focus): {hmm_df['p_focused_stay'].mean():.3f} ± {hmm_df['p_focused_stay'].std():.3f}",
    f"P(Lapse → Lapse): {hmm_df['p_lapsed_stay'].mean():.3f} ± {hmm_df['p_lapsed_stay'].std():.3f}",
    "",
    "=" * 80,
    "REGRESSION RESULTS - UCLA PREDICTING HMM METRICS (DASS-controlled)",
    "=" * 80,
]

for _, row in regression_df.iterrows():
    sig = "***" if row['ucla_p'] < 0.001 else "**" if row['ucla_p'] < 0.01 else "*" if row['ucla_p'] < 0.05 else ""
    summary_lines.append(
        f"{row['outcome_label']}: β={row['ucla_beta']:.4f}, p={row['ucla_p']:.4f} {sig} (R²={row['adj_r_squared']:.3f})"
    )

summary_lines.extend([
    "",
    "=" * 80,
    "INTERPRETATION",
    "=" * 80,
    "",
    "The HMM identifies two latent attentional states from RT sequences:",
    "  1. FOCUSED state: Lower mean RT, sustained attention",
    "  2. LAPSED state: Higher mean RT, attentional lapses/mind-wandering",
    "",
    "Key Hypotheses:",
    "  - If UCLA → HIGHER lapsed occupancy: More time spent 'zoned out'",
    "  - If UCLA → HIGHER P(Focus → Lapse): More frequent transitions to lapses",
    "  - If UCLA → LOWER P(Lapse → Focus): Slower recovery from lapses",
    "  - If UCLA → LARGER RT difference: Deeper lapses when off-task",
    "",
    "=" * 80,
    "FILES GENERATED",
    "=" * 80,
    "CSV Files:",
    "  - hmm_results_raw.csv (per-participant HMM metrics)",
    "  - hmm_merged_with_predictors.csv (merged with UCLA/DASS)",
    "  - hmm_descriptive_stats.csv (summary statistics)",
    "  - regression_results.csv (all regression coefficients)",
    "",
    "Text Files:",
    "  - regression_[outcome].txt (full regression outputs for each outcome)",
    "",
    "Figures:",
    "  - hmm_metrics_by_ucla.png (6-panel figure with all key metrics)",
    "  - example_state_sequences.png (state sequences for 3 example participants)",
    "",
    "=" * 80,
    "END OF REPORT",
    "=" * 80
])

summary_text = "\n".join(summary_lines)
print(summary_text)

# Save summary
with open(OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nTier 2.2 HMM Attentional States Analysis COMPLETE!")
