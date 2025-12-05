"""
Moderated Mediation Analysis: τ Mediates UCLA → WCST PE (Moderated by Gender)

Research Question:
Does Ex-Gaussian τ (attentional lapses) mediate the UCLA → WCST PE relationship,
and is this mediation pathway moderated by gender?

Theoretical Model:
    UCLA → τ (a path) → WCST PE (b path)
    Direct: UCLA → WCST PE (c' path)
    Total: UCLA → WCST PE (c path)
    Indirect effect: a × b
    Moderation: Does gender moderate paths a, b, or both?

Hypotheses:
1. In MALES: Significant indirect effect (UCLA → τ↑ → PE↑)
2. In FEMALES: Null or negative indirect effect (UCLA → τ↓ → PE↓)
3. Moderated mediation index significant (gender moderates indirect effect)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from analysis.utils.data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/tau_mediation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*80)
print("MODERATED MEDIATION: τ MEDIATES UCLA → WCST PE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

# Demographics and survey scores from master
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
    master['ucla_total'] = master['ucla_score']

participants = master[['participant_id', 'gender', 'age']].copy()
participants['gender_male'] = (participants['gender'] == 'male').astype(int)

ucla_data = master[['participant_id', 'ucla_total']].dropna()
dass_cols = ['dass_total', 'dass_depression', 'dass_anxiety', 'dass_stress']
if 'dass_total' not in master.columns and {'dass_depression', 'dass_anxiety', 'dass_stress'} <= set(master.columns):
    master['dass_total'] = master['dass_depression'] + master['dass_anxiety'] + master['dass_stress']
dass_data = master[['participant_id', 'dass_total']].dropna()

# Ex-Gaussian parameters (PRP τ)
exg_path = RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv"
if not exg_path.exists():
    # Try alternative path
    exg_path = RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/exgaussian_parameters.csv"

if exg_path.exists():
    exg_data = pd.read_csv(exg_path, encoding='utf-8-sig')
    if 'participantId' in exg_data.columns and 'participant_id' not in exg_data.columns:
        exg_data.rename(columns={'participantId': 'participant_id'}, inplace=True)
    print(f"  Loaded Ex-Gaussian data: {len(exg_data)} rows")
else:
    print(f"  ERROR: Ex-Gaussian file not found at {exg_path}")
    sys.exit(1)

# WCST PE rates
wcst_summary = master

# Select PE column (try multiple possible names)
pe_col = None
for col_name in ['pe_rate', 'pe_rate', 'pe_rate']:
    if col_name in wcst_summary.columns:
        pe_col = col_name
        break

if pe_col is None:
    print("  ERROR: Could not find PE rate column")
    print(f"  Available columns: {wcst_summary.columns.tolist()}")
    sys.exit(1)

wcst_data = wcst_summary[['participant_id', pe_col]].copy()
wcst_data.columns = ['participant_id', 'pe_rate']

print(f"  Loaded WCST PE data: {len(wcst_data)} participants")

# ============================================================================
# 2. MERGE AND PREPARE MEDIATION DATASET
# ============================================================================
print("\n[2/7] Preparing mediation dataset...")

# Identify τ columns in Ex-Gaussian data
tau_cols = [col for col in exg_data.columns if 'tau' in col.lower() or 'τ' in col]
print(f"  Available τ columns: {tau_cols}")

# Use overall τ or long SOA τ (most stable estimate)
if 'tau_long' in exg_data.columns:
    tau_col = 'tau_long'
elif 'tau_overall' in exg_data.columns:
    tau_col = 'tau_overall'
elif 'tau' in exg_data.columns:
    tau_col = 'tau'
elif len(tau_cols) > 0:
    tau_col = tau_cols[0]
else:
    print("  ERROR: No tau column found in Ex-Gaussian data")
    sys.exit(1)

print(f"  Using τ column: {tau_col}")

# Select mediator
mediator_data = exg_data[['participant_id', tau_col]].copy()
mediator_data.columns = ['participant_id', 'tau']

# Merge all
master = ucla_data.merge(mediator_data, on='participant_id', how='inner')
master = master.merge(wcst_data, on='participant_id', how='inner')
master = master.merge(dass_data, on='participant_id', how='left')
master = master.merge(participants[['participant_id', 'gender_male']], on='participant_id', how='left')

# Drop missing
master = master.dropna(subset=['ucla_total', 'tau', 'pe_rate', 'gender_male'])

print(f"\n  Complete cases: N={len(master)}")
print(f"    Males: {(master['gender_male'] == 1).sum()}")
print(f"    Females: {(master['gender_male'] == 0).sum()}")

# Z-score predictors for interpretability
master['z_ucla'] = (master['ucla_total'] - master['ucla_total'].mean()) / master['ucla_total'].std()
master['z_tau'] = (master['tau'] - master['tau'].mean()) / master['tau'].std()
master['z_dass'] = (master['dass_total'] - master['dass_total'].mean()) / master['dass_total'].std() if 'dass_total' in master.columns else 0

# ============================================================================
# 3. TEST PATHS SEPARATELY BY GENDER
# ============================================================================
print("\n[3/7] Testing mediation paths by gender...")

results_by_gender = []

for gender, label in [(1, 'Male'), (0, 'Female')]:
    subset = master[master['gender_male'] == gender].copy()
    n = len(subset)

    if n < 10:
        print(f"\n  {label}: N={n} (too small, skipping)")
        continue

    print(f"\n  {label} (N={n}):")

    # Path a: UCLA → τ
    r_a, p_a = stats.pearsonr(subset['ucla_total'], subset['tau'])
    model_a = smf.ols('tau ~ ucla_total', data=subset).fit()
    a_path = model_a.params['ucla_total']
    a_se = model_a.bse['ucla_total']

    print(f"    Path a (UCLA → τ): β={a_path:.3f}, SE={a_se:.3f}, r={r_a:.3f}, p={p_a:.3f}")

    # Path b: τ → PE (controlling UCLA)
    model_b = smf.ols('pe_rate ~ tau + ucla_total', data=subset).fit()
    b_path = model_b.params['tau']
    b_se = model_b.bse['tau']

    print(f"    Path b (τ → PE|UCLA): β={b_path:.3f}, SE={b_se:.3f}")

    # Path c: UCLA → PE (total effect)
    model_c = smf.ols('pe_rate ~ ucla_total', data=subset).fit()
    c_path = model_c.params['ucla_total']
    c_se = model_c.bse['ucla_total']
    c_p = model_c.pvalues['ucla_total']

    print(f"    Path c (UCLA → PE total): β={c_path:.3f}, SE={c_se:.3f}, p={c_p:.3f}")

    # Path c': UCLA → PE (direct, controlling τ)
    cprime_path = model_b.params['ucla_total']
    cprime_se = model_b.bse['ucla_total']

    print(f"    Path c' (UCLA → PE direct): β={cprime_path:.3f}, SE={cprime_se:.3f}")

    # Indirect effect: a × b
    indirect = a_path * b_path

    # Bootstrap CI for indirect effect
    def bootstrap_indirect(data, indices):
        """Bootstrap function for indirect effect"""
        d = data.iloc[indices]
        # Path a
        m_a = smf.ols('tau ~ ucla_total', data=d).fit()
        a = m_a.params['ucla_total']
        # Path b
        m_b = smf.ols('pe_rate ~ tau + ucla_total', data=d).fit()
        b = m_b.params['tau']
        return a * b

    # Bootstrap 10,000 samples
    n_boot = 10000
    boot_samples = np.random.choice(len(subset), size=(n_boot, len(subset)), replace=True)
    boot_indirects = []

    for boot_idx in boot_samples:
        try:
            indirect_boot = bootstrap_indirect(subset, boot_idx)
            boot_indirects.append(indirect_boot)
        except:
            pass

    boot_indirects = np.array(boot_indirects)
    indirect_ci_lower = np.percentile(boot_indirects, 2.5)
    indirect_ci_upper = np.percentile(boot_indirects, 97.5)
    indirect_sig = 'Yes' if (indirect_ci_lower > 0 and indirect_ci_upper > 0) or (indirect_ci_lower < 0 and indirect_ci_upper < 0) else 'No'

    print(f"    Indirect effect: {indirect:.4f}, 95% CI [{indirect_ci_lower:.4f}, {indirect_ci_upper:.4f}] {indirect_sig}")

    # Proportion mediated
    if abs(c_path) > 0.001:
        prop_mediated = indirect / c_path
    else:
        prop_mediated = np.nan

    print(f"    Proportion mediated: {prop_mediated:.1%}" if not np.isnan(prop_mediated) else "    Proportion mediated: N/A")

    results_by_gender.append({
        'Gender': label,
        'N': n,
        'a_path': a_path,
        'a_se': a_se,
        'a_p': p_a,
        'b_path': b_path,
        'b_se': b_se,
        'c_path': c_path,
        'c_p': c_p,
        'c_prime': cprime_path,
        'indirect': indirect,
        'indirect_ci_lower': indirect_ci_lower,
        'indirect_ci_upper': indirect_ci_upper,
        'indirect_sig': indirect_sig,
        'prop_mediated': prop_mediated
    })

results_df = pd.DataFrame(results_by_gender)

# ============================================================================
# 4. MODERATED MEDIATION: TEST GENDER AS MODERATOR
# ============================================================================
print("\n[4/7] Testing moderated mediation (gender as moderator)...")

# Path a moderation: UCLA × Gender → τ
model_a_mod = smf.ols('tau ~ ucla_total * gender_male', data=master).fit()
a_interaction = model_a_mod.params['ucla_total:gender_male']
a_interaction_p = model_a_mod.pvalues['ucla_total:gender_male']

print(f"\n  Path a moderation (UCLA × Gender → τ):")
print(f"    Interaction: β={a_interaction:.3f}, p={a_interaction_p:.3f}")

# Path b moderation: τ × Gender → PE (controlling UCLA)
model_b_mod = smf.ols('pe_rate ~ tau * gender_male + ucla_total', data=master).fit()
b_interaction = model_b_mod.params['tau:gender_male']
b_interaction_p = model_b_mod.pvalues['tau:gender_male']

print(f"\n  Path b moderation (τ × Gender → PE|UCLA):")
print(f"    Interaction: β={b_interaction:.3f}, p={b_interaction_p:.3f}")

# Index of moderated mediation (difference in indirect effects)
if len(results_df) == 2:
    male_indirect = results_df[results_df['Gender'] == 'Male']['indirect'].values[0]
    female_indirect = results_df[results_df['Gender'] == 'Female']['indirect'].values[0]
    moderated_mediation_index = male_indirect - female_indirect

    print(f"\n  Index of Moderated Mediation:")
    print(f"    Male indirect: {male_indirect:.4f}")
    print(f"    Female indirect: {female_indirect:.4f}")
    print(f"    Difference: {moderated_mediation_index:.4f}")

# ============================================================================
# 5. CONTROL FOR DASS
# ============================================================================
print("\n[5/7] Testing mediation controlling for DASS...")

if 'dass_total' in master.columns and master['dass_total'].notna().sum() > 50:
    for gender, label in [(1, 'Male'), (0, 'Female')]:
        subset = master[(master['gender_male'] == gender) & master['dass_total'].notna()].copy()

        if len(subset) < 10:
            continue

        # Path a: UCLA → τ (controlling DASS)
        model_a_dass = smf.ols('tau ~ ucla_total + dass_total', data=subset).fit()
        a_dass = model_a_dass.params['ucla_total']
        a_dass_p = model_a_dass.pvalues['ucla_total']

        # Path b: τ → PE (controlling UCLA + DASS)
        model_b_dass = smf.ols('pe_rate ~ tau + ucla_total + dass_total', data=subset).fit()
        b_dass = model_b_dass.params['tau']

        # Indirect
        indirect_dass = a_dass * b_dass

        print(f"\n  {label} (controlling DASS):")
        print(f"    Path a: β={a_dass:.3f}, p={a_dass_p:.3f}")
        print(f"    Path b: β={b_dass:.3f}")
        print(f"    Indirect: {indirect_dass:.4f}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n[6/7] Creating visualizations...")

# Plot 1: Path diagram with coefficients
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (gender, label, color) in enumerate([(1, 'Male', '#3498DB'), (0, 'Female', '#E74C3C')]):
    ax = axes[idx]

    if len(results_df[results_df['Gender'] == label]) == 0:
        continue

    row = results_df[results_df['Gender'] == label].iloc[0]

    # Draw path diagram
    # Positions: UCLA (left), τ (top middle), PE (right)
    ucla_pos = (0.1, 0.5)
    tau_pos = (0.5, 0.8)
    pe_pos = (0.9, 0.5)

    # Draw nodes
    for pos, text in [(ucla_pos, 'UCLA\nLoneliness'), (tau_pos, 'τ\n(Lapses)'), (pe_pos, 'WCST\nPE')]:
        circle = plt.Circle(pos, 0.08, color=color, alpha=0.3, zorder=1)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)

    # Draw arrows
    # Path a: UCLA → τ
    ax.annotate('', xy=tau_pos, xytext=ucla_pos,
                arrowprops=dict(arrowstyle='->', lw=2 if row['a_p'] < 0.05 else 1,
                              color=color, alpha=1 if row['a_p'] < 0.05 else 0.5))
    ax.text(0.3, 0.7, f"a = {row['a_path']:.3f}\n{'*' if row['a_p'] < 0.05 else 'ns'}",
           fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Path b: τ → PE
    ax.annotate('', xy=pe_pos, xytext=tau_pos,
                arrowprops=dict(arrowstyle='->', lw=2, color=color))
    ax.text(0.7, 0.7, f"b = {row['b_path']:.3f}",
           fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Path c': UCLA → PE (direct)
    ax.annotate('', xy=pe_pos, xytext=ucla_pos,
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', linestyle='--', alpha=0.5))
    ax.text(0.5, 0.35, f"c' = {row['c_prime']:.3f}",
           fontsize=8, ha='center', color='gray',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Indirect effect annotation
    indirect_text = f"Indirect: {row['indirect']:.4f}\n95% CI [{row['indirect_ci_lower']:.4f}, {row['indirect_ci_upper']:.4f}]\n{row['indirect_sig']}"
    ax.text(0.5, 0.1, indirect_text, ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow' if row['indirect_sig'] == 'Yes' else 'white', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'{label} (N={row["N"]})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mediation_path_diagrams.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Scatter plots showing paths
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for gender, label, marker, color in [(1, 'Male', 's', '#3498DB'), (0, 'Female', 'o', '#E74C3C')]:
    subset = master[master['gender_male'] == gender]

    # Path a: UCLA → τ
    axes[0, 0].scatter(subset['ucla_total'], subset['tau'],
                      alpha=0.6, label=label, marker=marker, s=80, color=color)

    # Path b: τ → PE
    axes[0, 1].scatter(subset['tau'], subset['pe_rate'],
                      alpha=0.6, label=label, marker=marker, s=80, color=color)

    # Path c: UCLA → PE (total)
    axes[1, 0].scatter(subset['ucla_total'], subset['pe_rate'],
                      alpha=0.6, label=label, marker=marker, s=80, color=color)

# Regression lines
for gender, color in [(1, '#3498DB'), (0, '#E74C3C')]:
    subset = master[master['gender_male'] == gender]

    # Path a
    z_a = np.polyfit(subset['ucla_total'], subset['tau'], 1)
    p_a = np.poly1d(z_a)
    x_a = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
    axes[0, 0].plot(x_a, p_a(x_a), color=color, linewidth=2, alpha=0.7)

    # Path c
    z_c = np.polyfit(subset['ucla_total'], subset['pe_rate'], 1)
    p_c = np.poly1d(z_c)
    x_c = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
    axes[1, 0].plot(x_c, p_c(x_c), color=color, linewidth=2, alpha=0.7)

axes[0, 0].set_xlabel('UCLA Loneliness')
axes[0, 0].set_ylabel('τ (Attentional Lapses)')
axes[0, 0].set_title('Path a: UCLA → τ')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_xlabel('τ (Attentional Lapses)')
axes[0, 1].set_ylabel('WCST PE Rate')
axes[0, 1].set_title('Path b: τ → PE')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].set_xlabel('UCLA Loneliness')
axes[1, 0].set_ylabel('WCST PE Rate')
axes[1, 0].set_title('Path c: UCLA → PE (Total Effect)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Summary table in bottom right
axes[1, 1].axis('off')
if len(results_df) > 0:
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['Gender'],
            f"{row['indirect']:.4f}",
            f"[{row['indirect_ci_lower']:.3f}, {row['indirect_ci_upper']:.3f}]",
            row['indirect_sig']
        ])

    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Gender', 'Indirect', '95% CI', 'Sig'],
                            cellLoc='center', loc='center',
                            bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1, 1].set_title('Indirect Effects Summary', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mediation_scatter_plots.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n[7/7] Saving results...")

results_df.to_csv(OUTPUT_DIR / "moderated_mediation_results.csv", index=False, encoding='utf-8-sig')

# Summary report
with open(OUTPUT_DIR / "MODERATED_MEDIATION_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("MODERATED MEDIATION: τ MEDIATES UCLA → WCST PE\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-"*80 + "\n")
    f.write("Does Ex-Gaussian τ (attentional lapses) mediate the UCLA → WCST PE\n")
    f.write("relationship, and is this pathway moderated by gender?\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"N = {len(master)}\n")
    f.write(f"Males: {(master['gender_male'] == 1).sum()}\n")
    f.write(f"Females: {(master['gender_male'] == 0).sum()}\n\n")

    f.write("GENDER-STRATIFIED RESULTS\n")
    f.write("-"*80 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("MODERATION TESTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Path a moderation (UCLA × Gender → τ): β={a_interaction:.3f}, p={a_interaction_p:.3f}\n")
    f.write(f"Path b moderation (τ × Gender → PE): β={b_interaction:.3f}, p={b_interaction_p:.3f}\n\n")

    if len(results_df) == 2:
        f.write(f"Index of Moderated Mediation: {moderated_mediation_index:.4f}\n")
        f.write(f"  (Male indirect - Female indirect)\n\n")

    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")

    if len(results_df) == 2:
        male_row = results_df[results_df['Gender'] == 'Male'].iloc[0]
        female_row = results_df[results_df['Gender'] == 'Female'].iloc[0]

        if male_row['indirect_sig'] == 'Yes':
            f.write(f"[OK] MALES: Significant mediation (indirect = {male_row['indirect']:.4f})\n")
            f.write(f"  UCLA -> tau_up -> PE_up pathway confirmed\n")
            f.write(f"  Mechanism: Attentional lapses mediate loneliness -> perseveration\n\n")
        else:
            f.write(f"[NS] MALES: Non-significant mediation (indirect = {male_row['indirect']:.4f})\n\n")

        if female_row['indirect_sig'] == 'Yes':
            f.write(f"[OK] FEMALES: Significant mediation (indirect = {female_row['indirect']:.4f})\n")
            if female_row['indirect'] < 0:
                f.write(f"  UCLA -> tau_down -> PE_down pathway (protective!)\n\n")
            else:
                f.write(f"  UCLA -> tau_up -> PE_up pathway\n\n")
        else:
            f.write(f"[NS] FEMALES: Non-significant mediation (indirect = {female_row['indirect']:.4f})\n\n")

    f.write("="*80 + "\n")
    f.write(f"Full results saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("[OK] Moderated Mediation Analysis Complete!")
print("="*80)
print(f"\nKey Finding: {'GENDER-SPECIFIC MEDIATION' if len(results_df) == 2 else 'See results for details'}")
if len(results_df) == 2:
    for _, row in results_df.iterrows():
        sig_marker = "[SIGNIFICANT]" if row['indirect_sig'] == 'Yes' else "[Not significant]"
        print(f"  {row['Gender']}: Indirect = {row['indirect']:.4f} {sig_marker}")
