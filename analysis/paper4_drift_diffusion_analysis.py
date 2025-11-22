"""
Paper 4: Drift Diffusion Model (DDM) Analysis

RESEARCH QUESTION:
  Does loneliness affect decision QUALITY (drift rate) or response CAUTION (boundary)?

METHOD: EZ-Diffusion Model (Wagenmakers et al., 2007)
  - Inputs: Mean RT, Variance RT, Accuracy
  - Outputs: v (drift rate), a (boundary), t (non-decision time)

THEORETICAL PREDICTIONS:
  - UCLA → v (drift): Information processing quality
  - UCLA → a (boundary): Speed-accuracy tradeoff (impulsivity)
  - UCLA → t (non-decision): Perceptual/motor delays

ADVANTAGES:
  - Decomposes RT into cognitive components
  - No complex fitting required (closed-form solution)
  - Complements Paper 1 (tau = mixture of v, a, t effects)
"""

import sys
from pathlib import Path
import pandas as pd
from data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_stroop_trials, load_prp_trials, load_wcst_trials
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/paper4_ddm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*100)
print("PAPER 4: DRIFT DIFFUSION MODEL ANALYSIS")
print("="*100)

# === EZ-DIFFUSION FUNCTIONS ===
def ez_diffusion(mean_rt, var_rt, prop_correct, s=0.1):
    """
    EZ-Diffusion Model (Wagenmakers et al., 2007)

    Parameters:
    -----------
    mean_rt : float
        Mean RT for correct trials (in seconds)
    var_rt : float
        Variance of RT for correct trials (in seconds^2)
    prop_correct : float
        Proportion of correct responses (0-1)
    s : float
        Scaling parameter (default 0.1)

    Returns:
    --------
    dict with keys: v (drift rate), a (boundary), t (non-decision time)
    """

    # Handle edge cases (relax constraints slightly)
    if prop_correct <= 0.51 or prop_correct >= 0.99:
        return {'v': np.nan, 'a': np.nan, 't': np.nan, 'valid': False}

    if mean_rt <= 0 or var_rt <= 0:
        return {'v': np.nan, 'a': np.nan, 't': np.nan, 'valid': False}

    # Compute intermediate values
    try:
        # Logit transform
        L = np.log(prop_correct / (1 - prop_correct))

        # Compute x (edge correction parameter)
        x = L * (L * prop_correct**2 - L * prop_correct + prop_correct - 0.5) / var_rt

        # Drift rate v
        v = np.sign(prop_correct - 0.5) * s * x**0.25

        # Boundary separation a
        s2 = s**2
        a = s2 * L / v

        # Non-decision time t
        y = -v * a / s2
        MDT = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
        t = mean_rt - MDT

        # Check validity
        if t < 0 or t > mean_rt:
            return {'v': np.nan, 'a': np.nan, 't': np.nan, 'valid': False}

        return {'v': v, 'a': a, 't': t, 'valid': True}

    except:
        return {'v': np.nan, 'a': np.nan, 't': np.nan, 'valid': False}


# === LOAD DATA ===
print("\n[1] Loading trial-level data...")

# Participants and surveys
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
master = master.rename(columns={'gender_normalized': 'gender'})
master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()

if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']

demo = master[['participant_id', 'age', 'gender', 'ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress']].dropna(subset=['gender']).copy()
demo['gender_male'] = (demo['gender'] == 'male').astype(int)

print(f"Demographics: N={len(demo)} ({demo['gender_male'].sum()} males, {len(demo) - demo['gender_male'].sum()} females)")

# === COMPUTE DDM PARAMETERS FOR EACH TASK ===
print("\n[2] Computing DDM parameters...")

ddm_results = []

# === STROOP TASK ===
print("\n[2.1] Stroop task...")
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')

# Drop null participant_id column and rename
if 'participant_id' in stroop_trials.columns:
    stroop_trials = stroop_trials.drop(columns=['participant_id'])
if 'participantId' in stroop_trials.columns:
    stroop_trials = stroop_trials.rename(columns={'participantId': 'participant_id'})

# Filter valid trials
# Check column names
if 'rt' in stroop_trials.columns:
    rt_col_stroop = 'rt'
    correct_col = 'correct'
    type_col = 'type'
else:
    print("ERROR: Stroop columns not found")
    rt_col_stroop = None

if rt_col_stroop:
    stroop_clean = stroop_trials[
        (stroop_trials[rt_col_stroop] > 200) &
        (stroop_trials[rt_col_stroop] < 5000)
    ].copy().reset_index(drop=True)
else:
    stroop_clean = pd.DataFrame()

print(f"  Stroop trials: {len(stroop_clean)} valid")

# Compute per participant for INCONGRUENT trials only (harder condition)
for pid in demo['participant_id']:
    subset = stroop_clean[
        (stroop_clean['participant_id'] == pid) &
        (stroop_clean[type_col] == 'incongruent')
    ]

    if len(subset) < 10:  # Need minimum trials
        continue

    # Convert RT to seconds
    rts = subset[rt_col_stroop].values / 1000.0
    correct = subset[correct_col].astype(bool).values

    # Compute statistics
    mean_rt = np.mean(rts[correct == 1])
    var_rt = np.var(rts[correct == 1], ddof=1)
    prop_correct = np.mean(correct)

    # Fit EZ-diffusion
    params = ez_diffusion(mean_rt, var_rt, prop_correct)

    if params['valid']:
        ddm_results.append({
            'participant_id': pid,
            'task': 'stroop',
            'condition': 'incongruent',
            'v': params['v'],
            'a': params['a'],
            't': params['t'],
            'mean_rt': mean_rt,
            'var_rt': var_rt,
            'accuracy': prop_correct,
            'n_trials': len(subset)
        })

print(f"  Stroop DDM: {len([r for r in ddm_results if r['task'] == 'stroop'])} participants")

# === PRP TASK ===
print("\n[2.2] PRP task...")
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')

if 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.drop(columns=['participant_id'], errors='ignore')
if 'participantId' in prp_trials.columns:
    prp_trials = prp_trials.rename(columns={'participantId': 'participant_id'})

# Use t2_rt_ms column
rt_col = 't2_rt_ms' if 't2_rt_ms' in prp_trials.columns else 't2_rt'

prp_clean = prp_trials[
    (prp_trials[rt_col] > 200) &
    (prp_trials[rt_col] < 5000) &
    (prp_trials['t2_timeout'] == False)
].copy().reset_index(drop=True)

print(f"  PRP trials: {len(prp_clean)} valid")

# Compute per participant for LONG SOA (easier condition for baseline)
for pid in demo['participant_id']:
    subset = prp_clean[
        (prp_clean['participant_id'] == pid) &
        (prp_clean['soa'] >= 1200)  # Long SOA
    ]

    if len(subset) < 10:
        continue

    rts = subset[rt_col].values / 1000.0
    correct = (subset['t2_error'] == 0).values

    mean_rt = np.mean(rts[correct])
    var_rt = np.var(rts[correct], ddof=1)
    prop_correct = np.mean(correct)

    params = ez_diffusion(mean_rt, var_rt, prop_correct)

    if params['valid']:
        ddm_results.append({
            'participant_id': pid,
            'task': 'prp',
            'condition': 'long_soa',
            'v': params['v'],
            'a': params['a'],
            't': params['t'],
            'mean_rt': mean_rt,
            'var_rt': var_rt,
            'accuracy': prop_correct,
            'n_trials': len(subset)
        })

print(f"  PRP DDM: {len([r for r in ddm_results if r['task'] == 'prp'])} participants")

# === WCST TASK ===
print("\n[2.3] WCST task...")
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')

# Drop null participant_id column and rename
if 'participant_id' in wcst_trials.columns:
    wcst_trials = wcst_trials.drop(columns=['participant_id'])
if 'participantId' in wcst_trials.columns:
    wcst_trials = wcst_trials.rename(columns={'participantId': 'participant_id'})

# Check column names
if 'reactionTimeMs' in wcst_trials.columns:
    rt_col_wcst = 'reactionTimeMs'
    correct_col_wcst = 'correct'
else:
    rt_col_wcst = None

if rt_col_wcst:
    wcst_clean = wcst_trials[
        (wcst_trials[rt_col_wcst] > 200) &
        (wcst_trials[rt_col_wcst] < 10000)
    ].copy().reset_index(drop=True)
else:
    wcst_clean = pd.DataFrame()

print(f"  WCST trials: {len(wcst_clean)} valid")

for pid in demo['participant_id']:
    subset = wcst_clean[wcst_clean['participant_id'] == pid]

    if len(subset) < 20:  # WCST needs more trials
        continue

    rts = subset[rt_col_wcst].values / 1000.0
    correct = subset[correct_col_wcst].astype(bool).values

    mean_rt = np.mean(rts[correct == 1])
    var_rt = np.var(rts[correct == 1], ddof=1)
    prop_correct = np.mean(correct)

    params = ez_diffusion(mean_rt, var_rt, prop_correct)

    if params['valid']:
        ddm_results.append({
            'participant_id': pid,
            'task': 'wcst',
            'condition': 'overall',
            'v': params['v'],
            'a': params['a'],
            't': params['t'],
            'mean_rt': mean_rt,
            'var_rt': var_rt,
            'accuracy': prop_correct,
            'n_trials': len(subset)
        })

print(f"  WCST DDM: {len([r for r in ddm_results if r['task'] == 'wcst'])} participants")

# Convert to DataFrame
ddm_df = pd.DataFrame(ddm_results)

# Merge with demographics
ddm_df = ddm_df.merge(demo, on='participant_id', how='inner')

print(f"\nTotal DDM estimates: {len(ddm_df)} (across {ddm_df['participant_id'].nunique()} participants)")

# Save
ddm_df.to_csv(OUTPUT_DIR / "paper4_ddm_parameters.csv", index=False, encoding='utf-8-sig')
print(f"Saved: paper4_ddm_parameters.csv")

# === ANALYZE UCLA EFFECTS ===
print("\n" + "="*100)
print("UCLA EFFECTS ON DDM PARAMETERS")
print("="*100)

results_summary = []

for task in ['stroop', 'prp', 'wcst']:
    task_data = ddm_df[ddm_df['task'] == task].copy()

    if len(task_data) < 20:
        print(f"\n[{task.upper()}] SKIPPED (N={len(task_data)} too small)")
        continue

    print(f"\n[{task.upper()}] N={len(task_data)}")
    print("-"*100)

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    task_data[['z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str', 'z_age']] = scaler.fit_transform(
        task_data[['ucla_total', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']]
    )

    for param in ['v', 'a', 't']:
        print(f"\n  [{param.upper()}] Drift={param=='v'}, Boundary={param=='a'}, NonDecision={param=='t'}")

        # Overall correlation
        r_all, p_all = stats.pearsonr(task_data['ucla_total'], task_data[param])
        print(f"    Overall: r={r_all:.3f}, p={p_all:.4f}")

        # Gender-stratified
        males = task_data[task_data['gender_male'] == 1]
        females = task_data[task_data['gender_male'] == 0]

        if len(males) >= 10 and len(females) >= 10:
            r_m, p_m = stats.pearsonr(males['ucla_total'], males[param])
            r_f, p_f = stats.pearsonr(females['ucla_total'], females[param])

            # Fisher z-test
            z_m = np.arctanh(r_m)
            z_f = np.arctanh(r_f)
            se_diff = np.sqrt(1/(len(males)-3) + 1/(len(females)-3))
            z_diff = (z_m - z_f) / se_diff
            p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

            print(f"    Males (N={len(males)}): r={r_m:.3f}, p={p_m:.4f}")
            print(f"    Females (N={len(females)}): r={r_f:.3f}, p={p_f:.4f}")
            print(f"    Gender difference: z={z_diff:.3f}, p={p_diff:.4f}")

            results_summary.append({
                'task': task,
                'parameter': param,
                'r_all': r_all,
                'p_all': p_all,
                'r_male': r_m,
                'p_male': p_m,
                'r_female': r_f,
                'p_female': p_f,
                'fisher_z': z_diff,
                'fisher_p': p_diff
            })
        else:
            print(f"    Insufficient data for gender stratification (M={len(males)}, F={len(females)})")

# Save results
results_df = pd.DataFrame(results_summary)
results_df.to_csv(OUTPUT_DIR / "paper4_ddm_ucla_correlations.csv", index=False, encoding='utf-8-sig')
print(f"\nSaved: paper4_ddm_ucla_correlations.csv")

# === VISUALIZATION ===
print("\n" + "="*100)
print("CREATING VISUALIZATIONS")
print("="*100)

# Create 3x3 grid (3 tasks × 3 parameters)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

tasks = ['stroop', 'prp', 'wcst']
params = ['v', 'a', 't']
param_labels = ['Drift Rate (v)', 'Boundary (a)', 'Non-Decision Time (t)']

for i, task in enumerate(tasks):
    task_data = ddm_df[ddm_df['task'] == task]

    for j, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i, j]

        if len(task_data) < 20:
            ax.text(0.5, 0.5, 'Insufficient\nData', ha='center', va='center', fontsize=14)
            ax.set_title(f'{task.upper()}: {label}', fontweight='bold')
            ax.axis('off')
            continue

        # Scatter by gender
        males = task_data[task_data['gender_male'] == 1]
        females = task_data[task_data['gender_male'] == 0]

        ax.scatter(males['ucla_total'], males[param], c='#3498db', s=60, alpha=0.6, label='Males')
        ax.scatter(females['ucla_total'], females[param], c='#e74c3c', s=60, alpha=0.6, label='Females')

        # Regression lines
        if len(males) >= 10:
            z = np.polyfit(males['ucla_total'], males[param], 1)
            p = np.poly1d(z)
            x_range = np.linspace(males['ucla_total'].min(), males['ucla_total'].max(), 100)
            ax.plot(x_range, p(x_range), color='#3498db', linestyle='--', linewidth=2)

        if len(females) >= 10:
            try:
                z = np.polyfit(females['ucla_total'], females[param], 1)
                p = np.poly1d(z)
                x_range = np.linspace(females['ucla_total'].min(), females['ucla_total'].max(), 100)
                ax.plot(x_range, p(x_range), color='#e74c3c', linestyle='--', linewidth=2)
            except:
                pass  # Skip if regression fails

        ax.set_xlabel('UCLA Loneliness' if i == 2 else '', fontsize=11)
        ax.set_ylabel(label if j == 0 else '', fontsize=11)
        ax.set_title(f'{task.upper()}: {label}', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)

        if i == 0 and j == 2:
            ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "paper4_ddm_scatter_grid.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved: paper4_ddm_scatter_grid.png")

# === SUMMARY REPORT ===
print("\n" + "="*100)
print("GENERATING SUMMARY REPORT")
print("="*100)

report_lines = [
    "="*100,
    "PAPER 4: DRIFT DIFFUSION MODEL ANALYSIS - SUMMARY REPORT",
    "="*100,
    "",
    "METHOD: EZ-Diffusion Model (Wagenmakers et al., 2007)",
    "-"*100,
    "Decomposes RT into:",
    "  - v (drift rate): Information accumulation rate → Decision QUALITY",
    "  - a (boundary): Response threshold → Speed-accuracy TRADEOFF",
    "  - t (non-decision time): Perceptual/motor processing → DELAYS",
    "",
    f"SAMPLE SIZE:",
    f"  Stroop: {len(ddm_df[ddm_df['task'] == 'stroop'])} participants",
    f"  PRP: {len(ddm_df[ddm_df['task'] == 'prp'])} participants",
    f"  WCST: {len(ddm_df[ddm_df['task'] == 'wcst'])} participants",
    "",
    "KEY FINDINGS:",
    "-"*100
]

# Add significant findings
for idx, row in results_df.iterrows():
    if row['fisher_p'] < 0.05:
        report_lines.append(
            f"  [{row['task'].upper()} - {row['parameter'].upper()}] SIGNIFICANT GENDER MODERATION"
        )
        report_lines.append(
            f"    Males: r={row['r_male']:.3f}, p={row['p_male']:.4f}"
        )
        report_lines.append(
            f"    Females: r={row['r_female']:.3f}, p={row['p_female']:.4f}"
        )
        report_lines.append(
            f"    Fisher z={row['fisher_z']:.3f}, p={row['fisher_p']:.4f}"
        )
        report_lines.append("")

if not any(results_df['fisher_p'] < 0.05):
    report_lines.append("  No significant gender moderation effects detected (all p > .05)")

report_lines.extend([
    "",
    "OUTPUTS:",
    "-"*100,
    "  1. paper4_ddm_parameters.csv - DDM estimates (v, a, t) for all participants",
    "  2. paper4_ddm_ucla_correlations.csv - UCLA correlation results",
    "  3. paper4_ddm_scatter_grid.png - 3x3 visualization (300 dpi)",
    "",
    "="*100,
    "END OF REPORT",
    "="*100
])

report_text = "\n".join(report_lines)
with open(OUTPUT_DIR / "PAPER4_DDM_SUMMARY.txt", "w", encoding='utf-8') as f:
    f.write(report_text)

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print("\nFiles created:")
print("  1. paper4_ddm_parameters.csv")
print("  2. paper4_ddm_ucla_correlations.csv")
print("  3. paper4_ddm_scatter_grid.png")
print("  4. PAPER4_DDM_SUMMARY.txt")
print("="*100)
