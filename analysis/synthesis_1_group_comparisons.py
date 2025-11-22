"""
Synthesis Analysis 1: Group Comparison Visualizations
=======================================================

Purpose:
--------
2×2×2 group comparisons (UCLA High/Low × DASS High/Low × Gender) for three EF outcomes:
1. WCST Perseverative Error Rate
2. PRP τ (tau) at long SOA
3. Stroop Interference

Statistical Approach:
--------------------
- UCLA: Quartile split (Q1 vs Q3)
- DASS: Median split (low vs high)
- Gender: Male vs Female
- ANCOVA with age as covariate (NOT raw t-tests)
- FDR correction (Benjamini-Hochberg) for multiple comparisons
- Bootstrap 95% confidence intervals

Output:
-------
- group_comparison_statistics.csv: Descriptive stats and effect sizes
- group_comparison_results.csv: ANCOVA results with FDR correction
- figure_group_comparisons.png: 3-panel grouped bar plot

Author: Research Team
Date: 2025-01-17
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
from scipy.optimize import minimize
from scipy.stats import exponnorm
import ast
import warnings
warnings.filterwarnings('ignore')

from data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials

# Import local utilities
import sys
sys.path.append('analysis')
from statistical_utils import cohen_d, bootstrap_ci, apply_multiple_comparison_correction

np.random.seed(42)

# Directories
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/synthesis_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SYNTHESIS ANALYSIS 1: GROUP COMPARISONS")
print("=" * 80)
print()
print("2×2×2 Design: UCLA (High/Low) × DASS (High/Low) × Gender (M/F)")
print("Outcomes: WCST PE Rate, PRP τ(long SOA), Stroop Interference")
print()

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("[1/5] Loading data...")

# Load master dataset via shared loader
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

master = master.rename(columns={"gender_normalized": "gender"})
master["gender"] = master["gender"].fillna("").astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

# Rename PE if needed
for col in ["pe_rate", "perseverative_error_rate"]:
    if col in master.columns and col != "pe_rate":
        master = master.rename(columns={col: "pe_rate"})
        break

print(f"  Total N = {len(master)}")

# ============================================================================
# COMPUTE PRP TAU (LONG SOA) FROM TRIAL DATA
# ============================================================================

print("[2/5] Computing PRP τ(long SOA) from trial-level data...")

prp_trials, _ = load_prp_trials(
    use_cache=True,
    rt_min=200,
    rt_max=5000,
    require_t1_correct=False,
    require_t2_correct_for_rt=False,
    enforce_short_long_only=False,
    drop_timeouts=True,
)
if "soa_nominal_ms" not in prp_trials.columns and "soa" in prp_trials.columns:
    prp_trials["soa_nominal_ms"] = prp_trials["soa"]
if "t2_rt" not in prp_trials.columns and "t2_rt_ms" in prp_trials.columns:
    prp_trials["t2_rt"] = prp_trials["t2_rt_ms"]

# Filter valid trials
prp_trials = prp_trials[
    (prp_trials['t1_correct'] == True) &
    (prp_trials['t2_rt'] > 0) &
    (prp_trials['t2_rt'] < 5000)
].copy()

# SOA categorization
def categorize_soa(soa):
    if soa <= 150:
        return 'short'
    elif 300 <= soa <= 600:
        return 'medium'
    elif soa >= 1200:
        return 'long'
    else:
        return 'other'

soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in prp_trials.columns else 'soa'
prp_trials['soa_cat'] = prp_trials[soa_col].apply(categorize_soa)

# Focus on long SOA
prp_long = prp_trials[prp_trials['soa_cat'] == 'long'].copy()

# Ensure participant_id is properly named and remove duplicates
if 'participantId' in prp_long.columns and 'participant_id' not in prp_long.columns:
    prp_long = prp_long.rename(columns={'participantId': 'participant_id'})

# Remove duplicate columns if any
prp_long = prp_long.loc[:, ~prp_long.columns.duplicated()]

# Debug: Check if participant_id exists
if 'participant_id' not in prp_long.columns:
    print(f"ERROR: participant_id column not found. Available columns: {prp_long.columns.tolist()}")
    sys.exit(1)

# Ex-Gaussian fitting function
def fit_exgaussian(rts):
    """Fit Ex-Gaussian distribution to RTs, return tau parameter"""
    if len(rts) < 20:
        return np.nan

    rts = np.array(rts)
    rts = rts[(rts > 100) & (rts < 3000)]  # Remove extreme outliers

    if len(rts) < 20:
        return np.nan

    # Initial parameter estimates
    m = np.mean(rts)
    s = np.std(rts)
    skew = stats.skew(rts)

    tau_init = max(10, (abs(skew) / 2) ** (1/3) * s)
    mu_init = max(100, m - tau_init)
    sigma_init = max(10, np.sqrt(max(0, s**2 - tau_init**2)))

    def neg_loglik(params):
        mu, sigma, tau = params
        if sigma <= 0 or tau <= 0:
            return 1e10
        K = tau / sigma
        try:
            loglik = np.sum(exponnorm.logpdf(rts, K, loc=mu, scale=sigma))
            return -loglik
        except:
            return 1e10

    try:
        result = minimize(
            neg_loglik,
            x0=[mu_init, sigma_init, tau_init],
            method='L-BFGS-B',
            bounds=[(100, 2000), (5, 500), (5, 1000)]
        )

        if result.success:
            mu, sigma, tau = result.x
            return tau
        else:
            return np.nan
    except:
        return np.nan

# Compute tau for each participant at long SOA
print("  Fitting Ex-Gaussian to extract τ (this may take a moment)...")
tau_results = []

# Get unique participant IDs
if len(prp_long) == 0:
    print("  ⚠ No PRP trials at long SOA found")
    tau_df = pd.DataFrame(columns=['participant_id', 'tau_long_soa', 'n_trials'])
else:
    grouped = prp_long.groupby('participant_id')
    for pid in grouped.groups.keys():
        pid_data = prp_long[prp_long['participant_id'] == pid]
        rts = pid_data['t2_rt'].values

        tau = fit_exgaussian(rts)
        tau_results.append({
            'participant_id': pid,
            'tau_long_soa': tau,
            'n_trials': len(rts)
        })

tau_df = pd.DataFrame(tau_results)
print(f"  τ computed for {len(tau_df)} participants")

if len(tau_df) > 0 and 'tau_long_soa' in tau_df.columns:
    print(f"  Valid τ values: {tau_df['tau_long_soa'].notna().sum()}")
else:
    print(f"  ⚠ No τ values computed (empty or missing column)")

# Merge tau into master
if len(tau_df) > 0 and 'participant_id' in tau_df.columns and 'tau_long_soa' in tau_df.columns:
    master = master.merge(tau_df[['participant_id', 'tau_long_soa']], on='participant_id', how='left')
else:
    master['tau_long_soa'] = np.nan

# ============================================================================
# CREATE GROUP ASSIGNMENTS
# ============================================================================

print("[3/5] Creating group assignments...")

# Required columns
required_cols = ['participant_id', 'ucla_total', 'dass_total', 'gender_male', 'age',
                'pe_rate', 'stroop_interference', 'tau_long_soa']

# Create working dataset
df = master.copy()

# Calculate DASS total if not present
if 'dass_total' not in df.columns:
    if all(col in df.columns for col in ['dass_depression', 'dass_anxiety', 'dass_stress']):
        df['dass_total'] = df['dass_depression'] + df['dass_anxiety'] + df['dass_stress']
    else:
        print("ERROR: Cannot compute DASS total")
        sys.exit(1)

# Group assignments
# UCLA: Q1 vs Q3
ucla_q1 = df['ucla_total'].quantile(0.25)
ucla_q3 = df['ucla_total'].quantile(0.75)

df['ucla_group'] = 'middle'
df.loc[df['ucla_total'] <= ucla_q1, 'ucla_group'] = 'low'
df.loc[df['ucla_total'] >= ucla_q3, 'ucla_group'] = 'high'

# DASS: Median split
dass_median = df['dass_total'].median()
df['dass_group'] = df['dass_total'].apply(lambda x: 'low' if x <= dass_median else 'high')

# Gender: already coded
df['gender_group'] = df['gender_male'].map({0: 'female', 1: 'male'})

# Create combined group label
df['combined_group'] = (df['ucla_group'] + '_' +
                        df['dass_group'] + '_' +
                        df['gender_group'])

# Filter to only Q1 and Q3 groups (exclude middle)
df_groups = df[df['ucla_group'].isin(['low', 'high'])].copy()

print(f"  Total N for group analysis = {len(df_groups)}")
print(f"    UCLA Low (Q1): {(df_groups['ucla_group'] == 'low').sum()}")
print(f"    UCLA High (Q3): {(df_groups['ucla_group'] == 'high').sum()}")
print(f"    DASS Low: {(df_groups['dass_group'] == 'low').sum()}")
print(f"    DASS High: {(df_groups['dass_group'] == 'high').sum()}")
print(f"    Male: {(df_groups['gender_male'] == 1).sum()}")
print(f"    Female: {(df_groups['gender_male'] == 0).sum()}")
print()

# ============================================================================
# COMPUTE GROUP STATISTICS
# ============================================================================

print("[4/5] Computing group statistics...")

outcomes = {
    'pe_rate': 'WCST PE Rate (%)',
    'tau_long_soa': 'PRP τ (long SOA, ms)',
    'stroop_interference': 'Stroop Interference (ms)'
}

group_stats = []

for outcome_col, outcome_label in outcomes.items():
    # Drop missing for this outcome
    df_outcome = df_groups.dropna(subset=[outcome_col, 'age']).copy()

    if len(df_outcome) < 20:
        print(f"  ⚠ Skipping {outcome_label}: insufficient data (N={len(df_outcome)})")
        continue

    print(f"\n  Analyzing: {outcome_label}")
    print(f"    N = {len(df_outcome)}")

    # Compute statistics for each of the 8 groups
    for ucla_val in ['low', 'high']:
        for dass_val in ['low', 'high']:
            for gender_val in ['female', 'male']:
                group_data = df_outcome[
                    (df_outcome['ucla_group'] == ucla_val) &
                    (df_outcome['dass_group'] == dass_val) &
                    (df_outcome['gender_group'] == gender_val)
                ]

                if len(group_data) >= 3:
                    values = group_data[outcome_col].values

                    # Bootstrap CI for mean
                    mean_est, ci_lower, ci_upper = bootstrap_ci(
                        values,
                        np.mean,
                        n_bootstrap=10000,
                        confidence_level=0.95
                    )

                    group_stats.append({
                        'outcome': outcome_label,
                        'outcome_col': outcome_col,
                        'ucla_group': ucla_val,
                        'dass_group': dass_val,
                        'gender': gender_val,
                        'n': len(values),
                        'mean': mean_est,
                        'sd': np.std(values, ddof=1),
                        'median': np.median(values),
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    })

group_stats_df = pd.DataFrame(group_stats)
group_stats_df.to_csv(OUTPUT_DIR / "group_comparison_statistics.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ Saved: group_comparison_statistics.csv")

# ============================================================================
# ANCOVA TESTS WITH FDR CORRECTION
# ============================================================================

print("\n[5/5] Running ANCOVA tests with FDR correction...")

ancova_results = []

for outcome_col, outcome_label in outcomes.items():
    # Prepare data
    df_outcome = df_groups.dropna(subset=[outcome_col, 'age']).copy()

    if len(df_outcome) < 20:
        continue

    print(f"\n  {outcome_label}:")

    # Encode categorical variables
    df_outcome['ucla_high'] = (df_outcome['ucla_group'] == 'high').astype(int)
    df_outcome['dass_high'] = (df_outcome['dass_group'] == 'high').astype(int)

    # Full 3-way ANCOVA model
    formula = f"{outcome_col} ~ C(ucla_high) * C(dass_high) * C(gender_male) + age"

    try:
        model = smf.ols(formula, data=df_outcome).fit()

        # Extract key effects
        effects = []

        # Main effects
        for term in ['C(ucla_high)[T.1]', 'C(dass_high)[T.1]', 'C(gender_male)[T.1]']:
            if term in model.params:
                effects.append({
                    'outcome': outcome_label,
                    'effect': term.replace('C(', '').replace('[T.1]', '').replace(')', ''),
                    'beta': model.params[term],
                    'se': model.bse[term],
                    'p_value': model.pvalues[term]
                })

        # 2-way interactions
        for term in model.params.index:
            if ':' in term and term.count(':') == 1:
                effects.append({
                    'outcome': outcome_label,
                    'effect': term.replace('C(', '').replace('[T.1]', '').replace(')', ''),
                    'beta': model.params[term],
                    'se': model.bse[term],
                    'p_value': model.pvalues[term]
                })

        # 3-way interaction
        three_way_terms = [t for t in model.params.index if t.count(':') == 2]
        for term in three_way_terms:
            effects.append({
                'outcome': outcome_label,
                'effect': term.replace('C(', '').replace('[T.1]', '').replace(')', ''),
                'beta': model.params[term],
                'se': model.bse[term],
                'p_value': model.pvalues[term]
            })

        for eff in effects:
            eff['n'] = len(df_outcome)
            eff['r2'] = model.rsquared
            ancova_results.append(eff)

        print(f"    Model R² = {model.rsquared:.3f}")
        print(f"    N = {len(df_outcome)}")

    except Exception as e:
        print(f"    ⚠ Error fitting model: {e}")
        continue

# Convert to DataFrame
ancova_df = pd.DataFrame(ancova_results)

# Apply FDR correction across all tests
if len(ancova_df) > 0:
    p_values = ancova_df['p_value'].values
    reject, p_adjusted = apply_multiple_comparison_correction(
        p_values,
        method='fdr_bh',
        alpha=0.05
    )

    ancova_df['p_adjusted_fdr'] = p_adjusted
    ancova_df['significant_fdr'] = reject

    # Sort by p-value
    ancova_df = ancova_df.sort_values('p_value')

    ancova_df.to_csv(OUTPUT_DIR / "group_comparison_ancova.csv", index=False, encoding='utf-8-sig')
    print(f"\n  ✓ Saved: group_comparison_ancova.csv")
    print(f"\n  Significant effects after FDR correction:")
    sig_effects = ancova_df[ancova_df['significant_fdr'] == True]
    if len(sig_effects) > 0:
        for _, row in sig_effects.iterrows():
            print(f"    {row['outcome']} - {row['effect']}: β={row['beta']:.3f}, p={row['p_value']:.4f} (FDR p={row['p_adjusted_fdr']:.4f})")
    else:
        print("    No effects survived FDR correction")

# ============================================================================
# VISUALIZATION: 3-PANEL GROUPED BAR PLOTS
# ============================================================================

print("\n[6/6] Creating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (outcome_col, outcome_label) in enumerate(outcomes.items()):
    ax = axes[idx]

    # Get data for this outcome
    outcome_data = group_stats_df[group_stats_df['outcome_col'] == outcome_col]

    if len(outcome_data) == 0:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
        ax.set_title(outcome_label, fontweight='bold', fontsize=12)
        continue

    # Create group labels for x-axis
    outcome_data['x_label'] = (outcome_data['ucla_group'].str.capitalize() + ' UCLA\n' +
                               outcome_data['dass_group'].str.capitalize() + ' DASS')

    # Separate by gender
    female_data = outcome_data[outcome_data['gender'] == 'female']
    male_data = outcome_data[outcome_data['gender'] == 'male']

    # X positions
    x_labels = outcome_data['x_label'].unique()
    x_pos = np.arange(len(x_labels))
    width = 0.35

    # Plot bars for each gender
    for i, label in enumerate(x_labels):
        f_row = female_data[female_data['x_label'] == label]
        m_row = male_data[male_data['x_label'] == label]

        if len(f_row) > 0:
            f_mean = f_row['mean'].values[0]
            f_ci_low = f_row['ci_lower'].values[0]
            f_ci_high = f_row['ci_upper'].values[0]

            ax.bar(i - width/2, f_mean, width,
                  color='#DE8F05', alpha=0.7, edgecolor='black', linewidth=1.5,
                  label='Female' if i == 0 else '')
            ax.errorbar(i - width/2, f_mean,
                       yerr=[[f_mean - f_ci_low], [f_ci_high - f_mean]],
                       fmt='none', ecolor='black', capsize=5, capthick=2)

        if len(m_row) > 0:
            m_mean = m_row['mean'].values[0]
            m_ci_low = m_row['ci_lower'].values[0]
            m_ci_high = m_row['ci_upper'].values[0]

            ax.bar(i + width/2, m_mean, width,
                  color='#0173B2', alpha=0.7, edgecolor='black', linewidth=1.5,
                  label='Male' if i == 0 else '')
            ax.errorbar(i + width/2, m_mean,
                       yerr=[[m_mean - m_ci_low], [m_ci_high - m_mean]],
                       fmt='none', ecolor='black', capsize=5, capthick=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(outcome_label, fontweight='bold', fontsize=11)
    ax.set_title(outcome_label, fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, frameon=True, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_group_comparisons.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure_group_comparisons.png")
plt.close()

# ============================================================================
# DONE
# ============================================================================

print()
print("=" * 80)
print("SYNTHESIS ANALYSIS 1 COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  - group_comparison_statistics.csv")
print("  - group_comparison_ancova.csv")
print("  - figure_group_comparisons.png")
print()
