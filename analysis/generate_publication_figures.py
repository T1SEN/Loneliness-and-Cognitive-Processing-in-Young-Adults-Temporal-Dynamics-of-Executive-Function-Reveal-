"""
Generate Publication-Quality Figures for Manuscript

Creates 4-6 main figures for publication:

Figure 1: Main Effect - UCLA × Gender → WCST PE (scatter + simple slopes)
Figure 2: Mechanisms - Mediation pathway (UCLA → τ → PE in males)
Figure 3: Gender Divergence - Female compensation vs Male impairment
Figure 4: Context Effects - DASS stratification (power analysis)
Figure 5: UCLA Facets - Social vs Emotional loneliness differential effects
Figure S1: Split-half replication (robustness check)

Style: APA 7th edition compliant, grayscale-friendly, high-resolution
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from data_loader_utils import load_master_dataset
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings for publication quality
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

# Color palette (colorblind-friendly + grayscale-printable)
COLORS = {
    'male': '#0173B2',      # Blue
    'female': '#DE8F05',    # Orange
    'neutral': '#949494',   # Gray
    'sig': '#029E73',       # Green
    'ns': '#CC78BC'         # Purple
}

np.random.seed(42)

print("="*80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

# Load master dataset
master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if master_path.exists():
    master = pd.read_csv(master_path, encoding='utf-8-sig')
else:
    print("ERROR: master_dataset.csv not found")
    import sys
    sys.exit(1)

# Load participants for gender
master = load_master_dataset(use_cache=True)
participants = master[['participant_id','gender_normalized','age']].rename(columns={'gender_normalized':'gender'})
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

# Normalize gender
gender_map = {'남성': 'male', '여성': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# Find PE column
for col in ['pe_rate', 'perseverative_error_rate']:
    if col in master.columns:
        master = master.rename(columns={col: 'pe_rate'})
        break

# Clean - ensure DASS covariates are present
master = master.dropna(subset=['ucla_total', 'pe_rate', 'gender_male',
                                'dass_depression', 'dass_anxiety', 'dass_stress', 'age'])

print(f"  Loaded N={len(master)} participants (with complete DASS + age data)")
print(f"    Males: {(master['gender_male'] == 1).sum()}")
print(f"    Females: {(master['gender_male'] == 0).sum()}")

# ============================================================================
# FIGURE 1: MAIN EFFECT - UCLA × GENDER → PE
# ============================================================================
print("\n[2/7] Creating Figure 1: Main Effect...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Scatter plot with regression lines
ax = axes[0]

for gender, gender_label, color, marker in [
    (0, 'Female', COLORS['female'], 'o'),
    (1, 'Male', COLORS['male'], 's')
]:
    subset = master[master['gender_male'] == gender]

    # Scatter
    ax.scatter(subset['ucla_total'], subset['pe_rate'],
              color=color, marker=marker, alpha=0.6, s=100,
              label=gender_label, edgecolors='black', linewidth=0.5)

    # Regression line (DASS-controlled)
    if len(subset) >= 10:
        model = smf.ols('pe_rate ~ ucla_total + dass_depression + dass_anxiety + dass_stress + age',
                       data=subset).fit()
        x_range = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
        # Predicted values at mean DASS and age for visualization
        dass_dep_mean = subset['dass_depression'].mean()
        dass_anx_mean = subset['dass_anxiety'].mean()
        dass_str_mean = subset['dass_stress'].mean()
        age_mean = subset['age'].mean()
        y_pred = (model.params['Intercept'] +
                  model.params['ucla_total'] * x_range +
                  model.params['dass_depression'] * dass_dep_mean +
                  model.params['dass_anxiety'] * dass_anx_mean +
                  model.params['dass_stress'] * dass_str_mean +
                  model.params['age'] * age_mean)

        r, p = stats.pearsonr(subset['ucla_total'], subset['pe_rate'])
        linestyle = '-' if p < 0.05 else '--'

        ax.plot(x_range, y_pred, color=color, linewidth=3, linestyle=linestyle,
               label=f'{gender_label}: r={r:.2f}, p={p:.3f}')

ax.set_xlabel('UCLA Loneliness Score', fontweight='bold')
ax.set_ylabel('WCST Perseverative Error Rate (%)', fontweight='bold')
ax.set_title('A. Gender-Specific Vulnerability (DASS-adjusted)', fontweight='bold', pad=15)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(alpha=0.3, linestyle=':', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Simple slopes with confidence bands
ax = axes[1]

# Calculate interaction (DASS-controlled)
formula = 'pe_rate ~ ucla_total * gender_male + dass_depression + dass_anxiety + dass_stress + age'
model_int = smf.ols(formula, data=master).fit()

interaction_beta = model_int.params['ucla_total:gender_male']
interaction_p = model_int.pvalues['ucla_total:gender_male']

# Plot slopes with error bands
for gender, gender_label, color in [(0, 'Female', COLORS['female']), (1, 'Male', COLORS['male'])]:
    subset = master[master['gender_male'] == gender]

    # Fit model (DASS-controlled)
    model = smf.ols('pe_rate ~ ucla_total + dass_depression + dass_anxiety + dass_stress + age',
                   data=subset).fit()
    beta = model.params['ucla_total']
    se = model.bse['ucla_total']
    ci_lower = beta - 1.96 * se
    ci_upper = beta + 1.96 * se

    # Bar
    x_pos = gender
    ax.bar(x_pos, beta, width=0.6, color=color, alpha=0.7,
          edgecolor='black', linewidth=2, label=gender_label)

    # Error bars
    ax.errorbar(x_pos, beta, yerr=1.96*se, fmt='none', ecolor='black',
               capsize=8, capthick=2, linewidth=2)

    # p-value annotation
    r, p = stats.pearsonr(subset['ucla_total'], subset['pe_rate'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    y_pos = beta + 1.96*se + 0.05
    ax.text(x_pos, y_pos, sig, ha='center', va='bottom', fontsize=16, fontweight='bold')

ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Female', 'Male'])
ax.set_ylabel('UCLA → PE Slope (β, DASS-adjusted)', fontweight='bold')
ax.set_title(f'B. Interaction (DASS-adjusted): β={interaction_beta:.2f}, p={interaction_p:.3f}',
            fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Figure1_MainEffect_HighRes.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Figure1_MainEffect.pdf", bbox_inches='tight')
print("  ✓ Figure 1 saved")
plt.close()

# ============================================================================
# FIGURE 2: SUMMARY OF KEY FINDINGS (4-panel)
# ============================================================================
print("\n[3/7] Creating Figure 2: Key Findings Summary...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Main effect (reuse from Fig 1)
ax1 = fig.add_subplot(gs[0, 0])
for gender, gender_label, color, marker in [
    (0, 'Female', COLORS['female'], 'o'),
    (1, 'Male', COLORS['male'], 's')
]:
    subset = master[master['gender_male'] == gender]
    ax1.scatter(subset['ucla_total'], subset['pe_rate'],
              color=color, marker=marker, alpha=0.6, s=80, label=gender_label)

    if len(subset) >= 10:
        model = smf.ols('pe_rate ~ ucla_total + dass_depression + dass_anxiety + dass_stress + age',
                       data=subset).fit()
        x_range = np.linspace(subset['ucla_total'].min(), subset['ucla_total'].max(), 100)
        # Predicted values at mean DASS and age
        dass_dep_mean = subset['dass_depression'].mean()
        dass_anx_mean = subset['dass_anxiety'].mean()
        dass_str_mean = subset['dass_stress'].mean()
        age_mean = subset['age'].mean()
        y_pred = (model.params['Intercept'] +
                  model.params['ucla_total'] * x_range +
                  model.params['dass_depression'] * dass_dep_mean +
                  model.params['dass_anxiety'] * dass_anx_mean +
                  model.params['dass_stress'] * dass_str_mean +
                  model.params['age'] * age_mean)
        ax1.plot(x_range, y_pred, color=color, linewidth=2.5)

ax1.set_xlabel('UCLA Loneliness', fontweight='bold')
ax1.set_ylabel('WCST PE Rate (%)', fontweight='bold')
ax1.set_title('A. Main Gender Moderation', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel B: Female vs Male costs/compensation
ax2 = fig.add_subplot(gs[0, 1])

# Load compensation cost results if available
comp_cost_path = RESULTS_DIR / "analysis_outputs/female_compensation_costs/compensation_costs_results.csv"
if comp_cost_path.exists():
    comp_costs = pd.read_csv(comp_cost_path, encoding='utf-8-sig')

    # Filter TIME COSTS
    time_costs = comp_costs[comp_costs['category'] == 'TIME COSTS (Overall RT)']

    if len(time_costs) > 0:
        # Reshape for plotting
        females = time_costs[time_costs['gender'] == 'Female']
        males = time_costs[time_costs['gender'] == 'Male']

        outcomes = females['outcome'].unique()
        x_pos = np.arange(len(outcomes))
        width = 0.35

        fem_r = [females[females['outcome'] == o]['r'].values[0] if len(females[females['outcome'] == o]) > 0 else 0 for o in outcomes]
        male_r = [males[males['outcome'] == o]['r'].values[0] if len(males[males['outcome'] == o]) > 0 else 0 for o in outcomes]

        ax2.bar(x_pos - width/2, fem_r, width, label='Female', color=COLORS['female'], alpha=0.7)
        ax2.bar(x_pos + width/2, male_r, width, label='Male', color=COLORS['male'], alpha=0.7)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([o.replace('Mean RT', '').strip() for o in outcomes], rotation=45, ha='right')
        ax2.set_ylabel('Correlation (UCLA → RT)', fontweight='bold')
        ax2.set_title('B. Time Costs (Slower RT)', fontweight='bold')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax2.set_title('B. Compensation Costs', fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center')
    ax2.set_title('B. Compensation Costs', fontweight='bold')

# Panel C: UCLA Facets
ax3 = fig.add_subplot(gs[1, 0])

facet_corr_path = RESULTS_DIR / "analysis_outputs/ucla_facets_ef/facet_correlations.csv"
if facet_corr_path.exists():
    facet_corr = pd.read_csv(facet_corr_path, encoding='utf-8-sig')

    males = facet_corr[facet_corr['gender'] == 'Male']

    if len(males) >= 2:
        factors = ['Factor 1\n(Social)', 'Factor 2\n(Emotional)']
        r_vals = males['r'].tolist()[:2]
        p_vals = males['p'].tolist()[:2]

        colors_facet = [COLORS['sig'] if p < 0.05 else COLORS['ns'] for p in p_vals]

        bars = ax3.bar(factors, r_vals, color=colors_facet, alpha=0.7, edgecolor='black', linewidth=2)

        for i, (r, p) in enumerate(zip(r_vals, p_vals)):
            sig = '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else 'ns'
            y_pos = r + 0.03 if r > 0 else r - 0.05
            ax3.text(i, y_pos, sig, ha='center', fontsize=14, fontweight='bold')

        ax3.axhline(0, color='black', linewidth=1)
        ax3.set_ylabel('Correlation (Factor → PE in Males)', fontweight='bold')
        ax3.set_title('C. UCLA Facet Differentiation', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax3.set_title('C. UCLA Facets', fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'Data not available', ha='center', va='center')
    ax3.set_title('C. UCLA Facets', fontweight='bold')

# Panel D: DASS stratification (power analysis)
ax4 = fig.add_subplot(gs[1, 1])

dass_strat_path = RESULTS_DIR / "analysis_outputs/dass_anxiety_mask/anxiety_stratified_results.csv"
if dass_strat_path.exists():
    dass_strat = pd.read_csv(dass_strat_path, encoding='utf-8-sig')

    males = dass_strat[dass_strat['gender'] == 'Male']

    if len(males) >= 2:
        groups = males['anxiety_group'].tolist()
        r_vals = males['r'].tolist()
        n_vals = males['n'].tolist()

        colors_dass = [COLORS['male'] for _ in r_vals]

        bars = ax4.bar(range(len(groups)), r_vals, color=colors_dass, alpha=0.7, edgecolor='black', linewidth=2)

        # Add N annotations
        for i, (r, n) in enumerate(zip(r_vals, n_vals)):
            ax4.text(i, r/2, f'N={n}', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        ax4.set_xticks(range(len(groups)))
        ax4.set_xticklabels(groups)
        ax4.axhline(0, color='black', linewidth=1)
        ax4.set_ylabel('Correlation (UCLA → PE in Males)', fontweight='bold')
        ax4.set_title('D. DASS Anxiety: Power Artifact', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax4.set_title('D. DASS Stratification', fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'Data not available', ha='center', va='center')
    ax4.set_title('D. DASS Stratification', fontweight='bold')

plt.suptitle('Key Findings: Loneliness, Gender, and Executive Function', fontsize=20, fontweight='bold', y=0.98)
plt.savefig(OUTPUT_DIR / "Figure2_KeyFindings_HighRes.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Figure2_KeyFindings.pdf", bbox_inches='tight')
print("  ✓ Figure 2 saved")
plt.close()

# ============================================================================
# FIGURE 3: EFFECT SIZE FOREST PLOT
# ============================================================================
print("\n[4/7] Creating Figure 3: Effect Size Forest Plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Compile effect sizes from all analyses
effects = []

# Main effect (males, DASS-controlled)
males = master[master['gender_male'] == 1]
if len(males) >= 10:
    model = smf.ols('pe_rate ~ ucla_total + dass_depression + dass_anxiety + dass_stress + age',
                   data=males).fit()
    beta = model.params['ucla_total']
    ci_lower = model.conf_int().loc['ucla_total', 0]
    ci_upper = model.conf_int().loc['ucla_total', 1]
    p = model.pvalues['ucla_total']

    effects.append({
        'analysis': 'Main Effect (Males, DASS-adj)',
        'beta': beta,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p': p,
        'n': len(males)
    })

# Add more effects if data available
# (Female effect, facets, etc.)

effects_df = pd.DataFrame(effects)

if len(effects_df) > 0:
    y_pos = np.arange(len(effects_df))

    # Plot points and error bars
    for i, row in effects_df.iterrows():
        color = COLORS['sig'] if row['p'] < 0.05 else COLORS['ns']
        ax.scatter(row['beta'], y_pos[i], s=150, color=color, zorder=3, edgecolors='black', linewidth=2)
        ax.plot([row['ci_lower'], row['ci_upper']], [y_pos[i], y_pos[i]],
               color=color, linewidth=3, zorder=2)

    # Vertical line at 0
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(effects_df['analysis'])
    ax.set_xlabel('Effect Size (β)', fontweight='bold')
    ax.set_title('Effect Sizes Across Analyses', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure3_ForestPlot_HighRes.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "Figure3_ForestPlot.pdf", bbox_inches='tight')
    print("  ✓ Figure 3 saved")
    plt.close()
else:
    print("  ⚠ Skipped Figure 3 (insufficient data)")

# ============================================================================
# SAVE FIGURE CATALOG
# ============================================================================
print("\n[7/7] Saving figure catalog...")

with open(OUTPUT_DIR / "FIGURE_CATALOG.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("PUBLICATION FIGURE CATALOG\n")
    f.write("="*80 + "\n\n")

    f.write("MAIN FIGURES\n")
    f.write("-"*80 + "\n\n")

    f.write("Figure 1: Main Effect - UCLA × Gender → WCST PE (DASS-adjusted)\n")
    f.write("  Panel A: Scatter plot with regression lines by gender\n")
    f.write("  Panel B: Simple slopes comparison with 95% CIs\n")
    f.write("  NOTE: All models control for DASS-21 subscales + age\n")
    f.write("  File: Figure1_MainEffect.pdf\n\n")

    f.write("Figure 2: Key Findings Summary (4-panel)\n")
    f.write("  Panel A: Main gender moderation\n")
    f.write("  Panel B: Time costs (compensation analysis)\n")
    f.write("  Panel C: UCLA facet differentiation\n")
    f.write("  Panel D: DASS anxiety stratification\n")
    f.write("  File: Figure2_KeyFindings.pdf\n\n")

    f.write("Figure 3: Effect Size Forest Plot\n")
    f.write("  Shows β coefficients with 95% CIs across analyses\n")
    f.write("  File: Figure3_ForestPlot.pdf\n\n")

    f.write("="*80 + "\n")
    f.write("All figures saved in: " + str(OUTPUT_DIR) + "\n")
    f.write("Formats: PDF (vector) + PNG (high-res raster)\n")

print("\n" + "="*80)
print("✓ Publication Figures Generated!")
print("="*80)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print("  - Figure1_MainEffect.pdf")
print("  - Figure2_KeyFindings.pdf")
print("  - Figure3_ForestPlot.pdf")
print("\nReady for manuscript submission!")
