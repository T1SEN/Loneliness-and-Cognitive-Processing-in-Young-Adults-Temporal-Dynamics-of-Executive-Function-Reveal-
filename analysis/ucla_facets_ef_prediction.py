"""
UCLA Facet-Specific EF Prediction Analysis

BACKGROUND:
UCLA Loneliness Scale can be decomposed into multiple factors:
- Factor 1: Typically reflects "Social Isolation" (lack of companionship, network size)
- Factor 2: Typically reflects "Emotional Loneliness" (subjective feelings, distress)

RESEARCH QUESTION:
Do these facets differentially predict EF impairment, especially in males?

HYPOTHESES:
H1: Social Isolation (Factor 1) drives male EF impairment more than Emotional Loneliness
    Rationale: Social isolation → fewer social interactions → cognitive under-stimulation
H2: Factor × Gender interactions differ between Factor 1 vs Factor 2
H3: Mediation pathway (Factor → τ → PE) stronger for Factor 1 in males

ANALYTIC STRATEGY:
1. Compare Factor 1 vs Factor 2 as predictors of WCST PE (by gender)
2. Test which factor shows stronger gender moderation
3. Test mediation via PRP τ for each factor separately
4. Report differential predictive validity
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
OUTPUT_DIR = Path("results/analysis_outputs/ucla_facets_ef")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
np.random.seed(42)

print("="*80)
print("UCLA FACET-SPECIFIC EF PREDICTION ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Factor scores
factor_scores = pd.read_csv(RESULTS_DIR / "analysis_outputs/ucla_facets/participant_factor_scores.csv",
                           encoding='utf-8-sig')

print(f"  Loaded factor scores: N={len(factor_scores)}")
print(f"  Columns: {factor_scores.columns.tolist()}")

# Load Ex-Gaussian τ for mediation
exg_path = RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv"
if not exg_path.exists():
    exg_path = RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/exgaussian_parameters.csv"

if exg_path.exists():
    exg_data = pd.read_csv(exg_path, encoding='utf-8-sig')
    if 'participantId' in exg_data.columns:
        exg_data = exg_data.rename(columns={'participantId': 'participant_id'})

    # Find τ column
    tau_col = None
    for col in ['tau_long', 'tau_overall', 'tau']:
        if col in exg_data.columns:
            tau_col = col
            break

    if tau_col:
        tau_data = exg_data[['participant_id', tau_col]].copy()
        tau_data.columns = ['participant_id', 'tau']
        print(f"  Loaded τ data: N={len(tau_data)}")
    else:
        tau_data = None
        print("  Warning: No τ column found")
else:
    tau_data = None
    print("  Warning: Ex-Gaussian file not found")

# Merge
master = factor_scores.copy()
if tau_data is not None:
    master = master.merge(tau_data, on='participant_id', how='left')

# Drop missing
master = master.dropna(subset=['factor1_2f', 'factor2_2f', 'pe_rate', 'gender_male'])

n_total = len(master)
n_male = (master['gender_male'] == 1).sum()
n_female = (master['gender_male'] == 0).sum()

print(f"\n  Complete cases: N={n_total}")
print(f"    Males: {n_male}")
print(f"    Females: {n_female}")

# ============================================================================
# 2. CORRELATIONS BY GENDER
# ============================================================================
print("\n[2/5] Testing Factor 1 vs Factor 2 correlations with PE...")

results_corr = []

for factor, label in [('factor1_2f', 'Factor 1 (Social Isolation)'),
                      ('factor2_2f', 'Factor 2 (Emotional Loneliness)')]:

    print(f"\n{label}:")
    print("-"*80)

    for gender, gender_label in [(0, 'Female'), (1, 'Male')]:
        subset = master[master['gender_male'] == gender].copy()

        if len(subset) < 10:
            print(f"  {gender_label}: N={len(subset)} - too small")
            continue

        # Correlation
        r, p = stats.pearsonr(subset[factor], subset['pe_rate'])

        # Regression
        model = smf.ols(f'pe_rate ~ {factor}', data=subset).fit()
        beta = model.params[factor]
        beta_p = model.pvalues[factor]

        print(f"  {gender_label} (N={len(subset)}): r={r:.3f}, p={p:.4f}, β={beta:.3f}")

        results_corr.append({
            'factor': label,
            'gender': gender_label,
            'n': len(subset),
            'r': r,
            'p': p,
            'beta': beta,
            'beta_p': beta_p
        })

# ============================================================================
# 3. INTERACTION MODELS (Factor × Gender)
# ============================================================================
print("\n[3/5] Testing Factor × Gender interactions...")

results_interaction = []

for factor, label in [('factor1_2f', 'Factor 1'), ('factor2_2f', 'Factor 2')]:

    # Interaction model
    formula = f'pe_rate ~ {factor} * gender_male'
    model = smf.ols(formula, data=master).fit()

    interaction_coef = model.params[f'{factor}:gender_male']
    interaction_p = model.pvalues[f'{factor}:gender_male']

    # Simple slopes
    males = master[master['gender_male'] == 1]
    females = master[master['gender_male'] == 0]

    if len(males) > 5:
        male_model = smf.ols(f'pe_rate ~ {factor}', data=males).fit()
        male_slope = male_model.params[factor]
        male_p = male_model.pvalues[factor]
    else:
        male_slope, male_p = np.nan, np.nan

    if len(females) > 5:
        female_model = smf.ols(f'pe_rate ~ {factor}', data=females).fit()
        female_slope = female_model.params[factor]
        female_p = female_model.pvalues[factor]
    else:
        female_slope, female_p = np.nan, np.nan

    print(f"\n{label}:")
    print(f"  Interaction: β={interaction_coef:.3f}, p={interaction_p:.4f}")
    print(f"  Male slope: β={male_slope:.3f}, p={male_p:.4f}")
    print(f"  Female slope: β={female_slope:.3f}, p={female_p:.4f}")

    results_interaction.append({
        'factor': label,
        'interaction_beta': interaction_coef,
        'interaction_p': interaction_p,
        'male_slope': male_slope,
        'male_p': male_p,
        'female_slope': female_slope,
        'female_p': female_p
    })

# ============================================================================
# 4. MEDIATION ANALYSIS (Factor → τ → PE) BY GENDER
# ============================================================================
print("\n[4/5] Testing mediation pathways (Factor → τ → PE)...")

if tau_data is not None and 'tau' in master.columns:

    master_tau = master[master['tau'].notna()].copy()

    print(f"\n  Mediation sample: N={len(master_tau)} (with τ data)")

    results_mediation = []

    for factor, label in [('factor1_2f', 'Factor 1'), ('factor2_2f', 'Factor 2')]:
        print(f"\n{label}:")
        print("-"*80)

        for gender, gender_label in [(1, 'Male'), (0, 'Female')]:
            subset = master_tau[master_tau['gender_male'] == gender].copy()

            if len(subset) < 10:
                print(f"  {gender_label}: N={len(subset)} - too small for mediation")
                continue

            # Path a: Factor → τ
            r_a, p_a = stats.pearsonr(subset[factor], subset['tau'])
            model_a = smf.ols(f'tau ~ {factor}', data=subset).fit()
            a_path = model_a.params[factor]
            a_p = model_a.pvalues[factor]

            # Path b: τ → PE (controlling Factor)
            model_b = smf.ols(f'pe_rate ~ tau + {factor}', data=subset).fit()
            b_path = model_b.params['tau']
            b_p = model_b.pvalues['tau']

            # Path c: Factor → PE (total)
            model_c = smf.ols(f'pe_rate ~ {factor}', data=subset).fit()
            c_path = model_c.params[factor]
            c_p = model_c.pvalues[factor]

            # Path c': Factor → PE (direct)
            cprime_path = model_b.params[factor]
            cprime_p = model_b.pvalues[factor]

            # Indirect effect
            indirect = a_path * b_path

            print(f"  {gender_label} (N={len(subset)}):")
            print(f"    Path a ({factor} → τ): β={a_path:.3f}, p={a_p:.4f}")
            print(f"    Path b (τ → PE): β={b_path:.3f}, p={b_p:.4f}")
            print(f"    Indirect (a×b): {indirect:.4f}")
            print(f"    Total effect (c): β={c_path:.3f}, p={c_p:.4f}")
            print(f"    Direct effect (c'): β={cprime_path:.3f}, p={cprime_p:.4f}")

            results_mediation.append({
                'factor': label,
                'gender': gender_label,
                'n': len(subset),
                'a_path': a_path,
                'a_p': a_p,
                'b_path': b_path,
                'b_p': b_p,
                'indirect': indirect,
                'c_path': c_path,
                'c_p': c_p,
                'cprime': cprime_path
            })

    mediation_df = pd.DataFrame(results_mediation)
    mediation_df.to_csv(OUTPUT_DIR / "facet_mediation_results.csv", index=False, encoding='utf-8-sig')

else:
    print("\n  Skipping mediation analysis (no τ data)")
    mediation_df = None

# ============================================================================
# 5. VISUALIZATIONS & SAVE
# ============================================================================
print("\n[5/5] Creating visualizations and saving results...")

corr_df = pd.DataFrame(results_corr)
interaction_df = pd.DataFrame(results_interaction)

corr_df.to_csv(OUTPUT_DIR / "facet_correlations.csv", index=False, encoding='utf-8-sig')
interaction_df.to_csv(OUTPUT_DIR / "facet_interactions.csv", index=False, encoding='utf-8-sig')

# Plot 1: Comparison of Factor 1 vs Factor 2 correlations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (factor_name, factor_label) in enumerate([('Factor 1 (Social Isolation)', 'Factor 1'),
                                                     ('Factor 2 (Emotional Loneliness)', 'Factor 2')]):

    ax = axes[idx]

    factor_data = corr_df[corr_df['factor'] == factor_name]

    if len(factor_data) == 0:
        continue

    # Bar plot
    x_pos = np.arange(len(factor_data))
    colors = ['#E74C3C' if g == 'Female' else '#3498DB' for g in factor_data['gender']]

    bars = ax.bar(x_pos, factor_data['r'], color=colors, alpha=0.7, edgecolor='black')

    # Add significance stars
    for i, (_, row) in enumerate(factor_data.iterrows()):
        if row['p'] < 0.001:
            sig = '***'
        elif row['p'] < 0.01:
            sig = '**'
        elif row['p'] < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        y_pos = row['r'] + 0.02 if row['r'] > 0 else row['r'] - 0.04
        ax.text(i, y_pos, sig, ha='center', va='bottom' if row['r'] > 0 else 'top',
               fontweight='bold')

    ax.set_xlabel('Gender')
    ax.set_ylabel('Correlation with PE Rate (r)')
    ax.set_title(f'{factor_label} → WCST PE')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(factor_data['gender'])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # Add n
    for i, (_, row) in enumerate(factor_data.iterrows()):
        ax.text(i, -0.15, f"N={row['n']}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "facet_comparison_correlations.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Interaction effects comparison
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(interaction_df))
colors = ['#9B59B6', '#E67E22']

bars = ax.bar(x_pos, interaction_df['interaction_beta'], color=colors, alpha=0.7, edgecolor='black')

# Significance
for i, (_, row) in enumerate(interaction_df.iterrows()):
    if row['interaction_p'] < 0.05:
        sig = '**' if row['interaction_p'] < 0.01 else '*'
    else:
        sig = 'ns'

    y_pos = row['interaction_beta'] + 0.5
    ax.text(i, y_pos, sig, ha='center', fontweight='bold')

    # p-value
    ax.text(i, row['interaction_beta'] / 2, f"p={row['interaction_p']:.3f}",
           ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.set_xlabel('UCLA Factor')
ax.set_ylabel('Interaction β (Factor × Gender → PE)')
ax.set_title('Gender Moderation Strength by UCLA Facet')
ax.set_xticks(x_pos)
ax.set_xticklabels(interaction_df['factor'])
ax.grid(axis='y', alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "facet_interaction_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Report
with open(OUTPUT_DIR / "UCLA_FACETS_EF_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("UCLA FACET-SPECIFIC EF PREDICTION ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-"*80 + "\n")
    f.write("Do Social Isolation (Factor 1) vs Emotional Loneliness (Factor 2)\n")
    f.write("differentially predict EF impairment in males vs females?\n\n")

    f.write("SAMPLE\n")
    f.write("-"*80 + "\n")
    f.write(f"N = {n_total}\n")
    f.write(f"  Males: {n_male}\n")
    f.write(f"  Females: {n_female}\n\n")

    f.write("CORRELATIONS WITH WCST PE\n")
    f.write("-"*80 + "\n\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\n")

    f.write("INTERACTION EFFECTS\n")
    f.write("-"*80 + "\n\n")
    f.write(interaction_df.to_string(index=False))
    f.write("\n\n")

    if mediation_df is not None:
        f.write("MEDIATION PATHWAYS (Factor → τ → PE)\n")
        f.write("-"*80 + "\n\n")
        f.write(mediation_df.to_string(index=False))
        f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")

    # Determine which factor is stronger
    factor1_male_r = corr_df[(corr_df['factor'] == 'Factor 1 (Social Isolation)') &
                             (corr_df['gender'] == 'Male')]
    factor2_male_r = corr_df[(corr_df['factor'] == 'Factor 2 (Emotional Loneliness)') &
                             (corr_df['gender'] == 'Male')]

    if len(factor1_male_r) > 0 and len(factor2_male_r) > 0:
        f1_r = factor1_male_r['r'].values[0]
        f1_p = factor1_male_r['p'].values[0]
        f2_r = factor2_male_r['r'].values[0]
        f2_p = factor2_male_r['p'].values[0]

        f.write(f"\nMale vulnerability:\n")
        f.write(f"  Factor 1 (Social Isolation): r={f1_r:.3f}, p={f1_p:.3f}\n")
        f.write(f"  Factor 2 (Emotional Loneliness): r={f2_r:.3f}, p={f2_p:.3f}\n\n")

        if abs(f1_r) > abs(f2_r) and f1_p < 0.10:
            f.write("CONCLUSION: Social Isolation (Factor 1) is the PRIMARY driver.\n")
            f.write("  Implication: Lack of social network size/companionship matters more\n")
            f.write("  than subjective feelings of loneliness for male EF impairment.\n")
        elif abs(f2_r) > abs(f1_r) and f2_p < 0.10:
            f.write("CONCLUSION: Emotional Loneliness (Factor 2) is the PRIMARY driver.\n")
            f.write("  Implication: Subjective distress matters more than objective isolation.\n")
        else:
            f.write("CONCLUSION: Both facets contribute similarly.\n")
            f.write("  Implication: Total UCLA score captures the relevant construct.\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Full results saved to: {OUTPUT_DIR}\n")

print("\n" + "="*80)
print("✓ UCLA Facet Analysis Complete!")
print("="*80)

# Print key finding
if len(factor1_male_r) > 0 and len(factor2_male_r) > 0:
    f1_r = factor1_male_r['r'].values[0]
    f2_r = factor2_male_r['r'].values[0]
    print(f"\nKey Finding:")
    print(f"  Factor 1 (Social Isolation) - Male r={f1_r:.3f}")
    print(f"  Factor 2 (Emotional Loneliness) - Male r={f2_r:.3f}")
    print(f"  Stronger predictor: {'Factor 1' if abs(f1_r) > abs(f2_r) else 'Factor 2'}")
