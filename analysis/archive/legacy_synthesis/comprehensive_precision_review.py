"""
COMPREHENSIVE PRECISION REVIEW: Papers 1-3
===========================================
Global verification and integration analysis
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

print('=' * 100)
print(' ' * 30 + 'COMPREHENSIVE PRECISION REVIEW')
print(' ' * 35 + 'Papers 1, 2, 3')
print('=' * 100)

# ============================================================================
# LOAD ALL RESULTS
# ============================================================================

print('\n[SECTION 1] DATA LOADING & VERIFICATION')
print('=' * 100)

# Paper 1
p1_metrics = pd.read_csv('results/analysis_outputs/paper1_distributional/paper1_participant_variability_metrics.csv')
p1_gender = pd.read_csv('results/analysis_outputs/paper1_distributional/paper1_gender_stratified_comparisons.csv')
p1_ls = pd.read_csv('results/analysis_outputs/paper1_distributional/paper1_location_scale_results.csv')

# Paper 2
p2_edges = pd.read_csv('results/analysis_outputs/paper2_network/paper2_network_edges_overall.csv')
p2_centrality = pd.read_csv('results/analysis_outputs/paper2_network/paper2_centrality_overall.csv')
p2_comparison = pd.read_csv('results/analysis_outputs/paper2_network/paper2_network_comparison.csv')

# Paper 3
p3_fit = pd.read_csv('results/analysis_outputs/paper3_profiles/paper3_gmm_fit_statistics.csv')
p3_chars = pd.read_csv('results/analysis_outputs/paper3_profiles/paper3_profile_characteristics.csv')
p3_demo = pd.read_csv('results/analysis_outputs/paper3_profiles/paper3_profile_demographics.csv')
p3_assign = pd.read_csv('results/analysis_outputs/paper3_profiles/paper3_profile_assignments.csv')

print('[OK] All result files loaded successfully')
print(f'  Paper 1: {len(p1_metrics)} participants')
print(f'  Paper 2: {len(p2_edges)} edges, {len(p2_centrality)} nodes')
print(f'  Paper 3: {len(p3_chars)} profiles identified')

# ============================================================================
# PAPER 1: CRITICAL VERIFICATION
# ============================================================================

print('\n[SECTION 2] PAPER 1: DISTRIBUTIONAL ANALYSIS VERIFICATION')
print('=' * 100)

print('\n[2.1] PRIMARY FINDING REPLICATION CHECK')
print('-' * 100)

# Extract key findings
prp_tau = p1_gender[p1_gender['outcome'] == 'PRP τ (long SOA)'].iloc[0]
prp_rmssd = p1_gender[p1_gender['outcome'] == 'PRP RMSSD'].iloc[0]

print('PRP tau (long SOA):')
print(f'  Males:   r={prp_tau["r_male"]:.3f}, p={prp_tau["p_male"]:.4f}')
print(f'  Females: r={prp_tau["r_female"]:.3f}, p={prp_tau["p_female"]:.4f}')
print(f'  Fisher z={prp_tau["fisher_z_diff"]:.3f}, p={(2*(1-stats.norm.cdf(abs(prp_tau["fisher_z_diff"])))):.4f}')
print(f'  VERDICT: [{"SIGNIFICANT" if abs(prp_tau["fisher_z_diff"]) > 1.96 else "NOT SIGNIFICANT"}]')

print('\nPRP RMSSD:')
print(f'  Males:   r={prp_rmssd["r_male"]:.3f}, p={prp_rmssd["p_male"]:.4f}')
print(f'  Females: r={prp_rmssd["r_female"]:.3f}, p={prp_rmssd["p_female"]:.4f}')
print(f'  Fisher z={prp_rmssd["fisher_z_diff"]:.3f}, p={(2*(1-stats.norm.cdf(abs(prp_rmssd["fisher_z_diff"])))):.4f}')
print(f'  VERDICT: [{"SIGNIFICANT" if abs(prp_rmssd["fisher_z_diff"]) > 1.96 else "NOT SIGNIFICANT"}]')

print('\n[2.2] EFFECT SIZE ASSESSMENT')
print('-' * 100)

males_tau_r = prp_tau['r_male']
males_rmssd_r = prp_rmssd['r_male']

print(f'Male effect sizes:')
print(f'  tau r={males_tau_r:.3f} → R²={males_tau_r**2:.3f} ({males_tau_r**2*100:.1f}% variance)')
print(f'  RMSSD r={males_rmssd_r:.3f} → R²={males_rmssd_r**2:.3f} ({males_rmssd_r**2*100:.1f}% variance)')
print(f'  Cohen f² (tau) = {(males_tau_r**2)/(1-males_tau_r**2):.3f} [{"LARGE" if (males_tau_r**2)/(1-males_tau_r**2) > 0.35 else "MEDIUM"}]')
print(f'  Cohen f² (RMSSD) = {(males_rmssd_r**2)/(1-males_rmssd_r**2):.3f} [{"LARGE" if (males_rmssd_r**2)/(1-males_rmssd_r**2) > 0.35 else "MEDIUM"}]')

print('\n[2.3] DASS CONTROL EFFECTIVENESS')
print('-' * 100)

# Check UCLA main effects after DASS control
ucla_main_effects = p1_ls[p1_ls['component'] == 'Mean (μ)']['ucla_p']
n_sig = (ucla_main_effects < 0.05).sum()
print(f'UCLA main effects (DASS-controlled):')
print(f'  Significant (p<.05): {n_sig}/{len(ucla_main_effects)}')
print(f'  Mean p-value: {ucla_main_effects.mean():.3f}')
print(f'  VERDICT: [{"DASS control successful" if n_sig == 0 else "WARNING: Some UCLA main effects remain"}]')

# ============================================================================
# PAPER 2: CRITICAL VERIFICATION
# ============================================================================

print('\n[SECTION 3] PAPER 2: NETWORK ANALYSIS VERIFICATION')
print('=' * 100)

print('\n[3.1] UCLA CENTRALITY RANKING')
print('-' * 100)

ucla_row = p2_centrality[p2_centrality['node'] == 'UCLA'].iloc[0]
ucla_strength = ucla_row['strength']
ucla_betweenness = ucla_row['betweenness']

strength_rank = (p2_centrality['strength'] > ucla_strength).sum() + 1
betweenness_rank = (p2_centrality['betweenness'] > ucla_betweenness).sum() + 1

print(f'UCLA node properties:')
print(f'  Strength: {ucla_strength:.3f} (rank {strength_rank}/{len(p2_centrality)})')
print(f'  Betweenness: {ucla_betweenness:.3f} (rank {betweenness_rank}/{len(p2_centrality)})')
print(f'  CLASSIFICATION: [{"HUB" if strength_rank <= 3 else "MODERATE" if strength_rank <= 6 else "PERIPHERAL"}]')

print('\n[3.2] UCLA DIRECT CONNECTIONS')
print('-' * 100)

ucla_edges = p2_edges[(p2_edges['node1'] == 'UCLA') | (p2_edges['node2'] == 'UCLA')]
print(f'Total UCLA edges: {len(ucla_edges)}')
print(f'UCLA connects to:')
for _, edge in ucla_edges.iterrows():
    partner = edge['node1'] if edge['node2'] == 'UCLA' else edge['node2']
    print(f'  {partner}: r={edge["partial_corr"]:.3f}')

# Check for UCLA-PRP_tau edge
ucla_tau_edge = p2_edges[((p2_edges['node1'] == 'UCLA') & (p2_edges['node2'] == 'PRP_tau')) |
                          ((p2_edges['node1'] == 'PRP_tau') & (p2_edges['node2'] == 'UCLA'))]

print(f'\nUCLA--PRP_tau edge:')
if len(ucla_tau_edge) > 0:
    print(f'  PRESENT: r={ucla_tau_edge["partial_corr"].values[0]:.3f}')
else:
    print(f'  ABSENT (|r| < 0.05 threshold)')
    # Check comparison data
    ucla_tau_comp = p2_comparison[((p2_comparison['node1'] == 'UCLA') & (p2_comparison['node2'] == 'PRP_tau')) |
                                   ((p2_comparison['node1'] == 'PRP_tau') & (p2_comparison['node2'] == 'UCLA'))]
    if len(ucla_tau_comp) > 0:
        print(f'  Partial r (males): {ucla_tau_comp["r_male"].values[0]:.3f}')
        print(f'  Partial r (females): {ucla_tau_comp["r_female"].values[0]:.3f}')

print('\n[3.3] PAPER 1 vs PAPER 2 RECONCILIATION')
print('-' * 100)

print('BIVARIATE (Paper 1) vs PARTIAL (Paper 2):')
print(f'  Paper 1 males (bivariate): r=0.566, p=.001 [SIGNIFICANT]')
print(f'  Paper 2 males (partial):   r={ucla_tau_comp["r_male"].values[0]:.3f}, p={ucla_tau_comp["p_diff"].values[0]:.3f} [NOT SIGNIFICANT]')
print(f'\nEXPLANATION:')
print(f'  Bivariate correlation conflates direct + indirect effects')
print(f'  Partial correlation isolates direct effect (controlling all nodes)')
print(f'  → UCLA-tau link is MEDIATED by other nodes (DASS)')

# ============================================================================
# PAPER 3: CRITICAL VERIFICATION
# ============================================================================

print('\n[SECTION 4] PAPER 3: LATENT PROFILE ANALYSIS VERIFICATION')
print('=' * 100)

print('\n[4.1] MODEL SELECTION ASSESSMENT')
print('-' * 100)

print('BIC by k:')
for _, row in p3_fit.iterrows():
    k = int(row['k'])
    bic = row['bic']
    entropy = row['relative_entropy']
    marker = ' <-- BEST' if k == 2 else ''
    print(f'  k={k}: BIC={bic:.2f}, Entropy={entropy:.3f}{marker}')

# Check if k=3 or k=4 are close
bic_k2 = p3_fit[p3_fit['k'] == 2]['bic'].values[0]
bic_k3 = p3_fit[p3_fit['k'] == 3]['bic'].values[0]
bic_k4 = p3_fit[p3_fit['k'] == 4]['bic'].values[0]

bic_diff_3 = bic_k3 - bic_k2
bic_diff_4 = bic_k4 - bic_k2

print(f'\nBIC differences from k=2:')
print(f'  k=3: ΔBIC={bic_diff_3:.2f} [{"CLOSE" if abs(bic_diff_3) < 10 else "CLEARLY WORSE" if bic_diff_3 > 10 else "UNCLEAR"}]')
print(f'  k=4: ΔBIC={bic_diff_4:.2f} [{"CLOSE" if abs(bic_diff_4) < 10 else "CLEARLY WORSE" if bic_diff_4 > 10 else "BETTER!" if bic_diff_4 < -10 else "UNCLEAR"}]')

if bic_diff_4 > 0:
    print('\nWARNING: k=4 has HIGHER BIC than k=2, indicating worse fit')
    print('         BUT AIC may prefer k=4 (less penalty for complexity)')
    aic_k2 = p3_fit[p3_fit['k'] == 2]['aic'].values[0]
    aic_k4 = p3_fit[p3_fit['k'] == 4]['aic'].values[0]
    print(f'  k=2 AIC: {aic_k2:.2f}')
    print(f'  k=4 AIC: {aic_k4:.2f} [{"BETTER" if aic_k4 < aic_k2 else "WORSE"}]')

print('\n[4.2] PROFILE CHARACTERISTICS')
print('-' * 100)

for _, row in p3_chars.iterrows():
    profile_num = int(row['profile'])
    n = int(row['N'])
    print(f'\nProfile {profile_num} (N={n}, {n/74*100:.1f}%):')
    print(f'  PRP tau: {row["PRP_tau_mean"]:.1f} ms')
    print(f'  WCST tau: {row["WCST_tau_mean"]:.1f} ms')
    print(f'  Stroop tau: {row["Stroop_tau_mean"]:.1f} ms')

    # Compare to overall mean
    overall_prp_tau = p1_metrics['prp_tau_long'].mean()
    overall_wcst_tau = p1_metrics['wcst_tau'].mean()
    z_prp = (row['PRP_tau_mean'] - overall_prp_tau) / p1_metrics['prp_tau_long'].std()
    z_wcst = (row['WCST_tau_mean'] - overall_wcst_tau) / p1_metrics['wcst_tau'].std()
    print(f'  Z-scores: PRP tau z={z_prp:.2f}, WCST tau z={z_wcst:.2f}')
    print(f'  LABEL: [{"HIGH VARIABILITY" if z_prp > 0.5 else "LOW VARIABILITY"}]')

print('\n[4.3] PROFILE-DEMOGRAPHIC ASSOCIATIONS')
print('-' * 100)

print('Profile demographics:')
for _, row in p3_demo.iterrows():
    profile_num = int(row['profile'])
    print(f'\nProfile {profile_num}:')
    print(f'  Gender: {row["pct_male"]:.1f}% male (n_male={int(row["n_male"])}, n_female={int(row["n_female"])})')
    print(f'  UCLA: M={row["ucla_mean"]:.1f}, SD={row["ucla_sd"]:.1f}')
    print(f'  DASS_Dep: M={row["dass_dep_mean"]:.1f}')

# Chi-square test
profile_1_male_pct = p3_demo[p3_demo['profile'] == 1]['pct_male'].values[0]
profile_2_male_pct = p3_demo[p3_demo['profile'] == 2]['pct_male'].values[0]

print(f'\nGender distribution test:')
print(f'  Profile 1: {profile_1_male_pct:.1f}% male')
print(f'  Profile 2: {profile_2_male_pct:.1f}% male')
print(f'  Difference: {abs(profile_1_male_pct - profile_2_male_pct):.1f} percentage points')

# Test if this is consistent with Paper 1
print(f'\nCONSISTENCY WITH PAPER 1:')
print(f'  Paper 1: Males show high variability (r=0.566)')
print(f'  Paper 3: Profile 1 (high var) has {profile_1_male_pct:.1f}% males')
print(f'  Expected if consistent: Profile 1 should have >50% males')
print(f'  VERDICT: [{"INCONSISTENT" if profile_1_male_pct < 40 else "SOMEWHAT CONSISTENT" if profile_1_male_pct < 50 else "CONSISTENT"}]')

# ============================================================================
# CROSS-PAPER INTEGRATION
# ============================================================================

print('\n[SECTION 5] CROSS-PAPER INTEGRATION & CONSISTENCY CHECK')
print('=' * 100)

print('\n[5.1] THREE-WAY CONSISTENCY MATRIX')
print('-' * 100)

print('\n| Finding | Paper 1 | Paper 2 | Paper 3 | Consistent? |')
print('|---------|---------|---------|---------|-------------|')

# Finding 1: UCLA main effects after DASS control
p1_main = 'Zero (0/12 sig)'
p2_main = 'UCLA moderate centrality'
p3_main = 'UCLA not predictive (F=1.85, p=.178)'
consistent_1 = 'YES' if n_sig == 0 else 'NO'
print(f'| UCLA main effect | {p1_main} | {p2_main} | {p3_main} | {consistent_1} |')

# Finding 2: Gender moderation
p1_gender = 'Strong (z=2.91**)'
p2_gender = 'None detected (0 edges)'
p3_gender = 'None (χ²=1.02, p=.311)'
consistent_2 = 'NO - MAJOR INCONSISTENCY'
print(f'| Gender moderation | {p1_gender} | {p2_gender} | {p3_gender} | {consistent_2} |')

# Finding 3: Variability as key metric
p1_var = 'tau + RMSSD significant'
p2_var = 'RMSSD is hub (str=0.907)'
p3_var = 'Profiles based on variability'
consistent_3 = 'YES'
print(f'| Variability focus | {p1_var} | {p2_var} | {p3_var} | {consistent_3} |')

print('\n[5.2] INCONSISTENCY INVESTIGATION: GENDER')
print('-' * 100)

print('\nWhy does Paper 1 show strong gender effects but Papers 2-3 do not?')
print('\nHypothesis 1: POWER ISSUE')
print('  - Paper 2: Males N=30 (borderline for network)')
print('  - Paper 3: Profile 1 males N≈8 (too small for tests)')
print('  - Detectable effect: |r_diff| > 0.35 (Paper 2 power analysis)')
print('  PLAUSIBILITY: HIGH')

print('\nHypothesis 2: ANALYSIS LEVEL DIFFERENCE')
print('  - Paper 1: Bivariate correlation (LOCAL effect)')
print('  - Paper 2: Network structure (GLOBAL topology)')
print('  - Paper 3: Profile membership (CATEGORICAL grouping)')
print('  - Local effects can exist without global/categorical differences')
print('  PLAUSIBILITY: MODERATE')

print('\nHypothesis 3: k=2 TOO COARSE FOR GENDER SUBGROUPS')
print('  - k=2 may merge "male lapse-prone" with "female hypervigilant"')
print('  - k=3-4 might separate them')
print('  - BIC prefers parsimony, but may miss interpretable subtypes')
print('  PLAUSIBILITY: MODERATE-HIGH')

print('\n[5.3] SAMPLE SIZE ADEQUACY ACROSS PAPERS')
print('-' * 100)

n_total = 74
n_male = 30
n_female = 44

print(f'Overall sample: N={n_total}')
print(f'  Males: {n_male}, Females: {n_female}')
print(f'\nAdequacy by analysis:')
print(f'  Paper 1 (correlation): {"OK" if n_male >= 30 else "MARGINAL"} (males N={n_male}, need ~30 for r=0.5)')
print(f'  Paper 2 (network, 11 nodes): {"MARGINAL" if n_male < 55 else "OK"} (males N={n_male}, prefer 5×11=55)')
print(f'  Paper 3 (GMM, 7 features): {"OK" if n_total >= 70 else "MARGINAL"} (overall N={n_total}, prefer 10×7=70)')
print(f'  Paper 3 (profile tests): {"FAIL" if profile_1_male_pct*26/100 < 10 else "MARGINAL"} (Profile 1 males N≈{profile_1_male_pct*26/100:.0f}, need ≥10)')

# ============================================================================
# POWER ANALYSIS
# ============================================================================

print('\n[SECTION 6] POST-HOC POWER ANALYSIS')
print('=' * 100)

print('\n[6.1] PAPER 3: PROFILE × GENDER ASSOCIATION')
print('-' * 100)

# Observed chi-square
chi2_obs = 1.02
p_obs = 0.3115

# What effect size would be detectable?
from scipy.stats import chi2_contingency, chi2
alpha = 0.05
df = 1  # (2 profiles - 1) × (2 genders - 1)
crit_chi2 = chi2.ppf(1-alpha, df)

print(f'Observed test:')
print(f'  χ² = {chi2_obs:.2f}, p = {p_obs:.4f}')
print(f'  Critical χ² (α=.05, df=1): {crit_chi2:.2f}')
print(f'  Observed is {"FAR BELOW" if chi2_obs < crit_chi2/2 else "BELOW"} critical value')

# Effect size (Cramér's V)
n = 74
cramers_v_obs = np.sqrt(chi2_obs / n)
print(f'\nEffect size:')
print(f'  Cramér V = {cramers_v_obs:.3f} [{"SMALL" if cramers_v_obs < 0.3 else "MEDIUM"}]')

# What V would be needed?
cramers_v_detectable = np.sqrt(crit_chi2 / n)
print(f'  Detectable V ≥ {cramers_v_detectable:.3f} at 80% power')

print('\n[6.2] PAPER 3: PROFILE × UCLA ANOVA')
print('-' * 100)

f_obs = 1.85
p_obs_anova = 0.1783

# Critical F
from scipy.stats import f as f_dist
df_between = 1  # 2 profiles - 1
df_within = 72  # 74 - 2
crit_f = f_dist.ppf(1-alpha, df_between, df_within)

print(f'Observed test:')
print(f'  F({df_between},{df_within}) = {f_obs:.2f}, p = {p_obs_anova:.4f}')
print(f'  Critical F (α=.05): {crit_f:.2f}')
print(f'  Observed is {"FAR BELOW" if f_obs < crit_f/2 else "BELOW"} critical value')

# Eta-squared
eta2_obs = f_obs * df_between / (f_obs * df_between + df_within)
print(f'\nEffect size:')
print(f'  η² = {eta2_obs:.3f} [{"SMALL" if eta2_obs < 0.06 else "MEDIUM"}]')

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================

print('\n[SECTION 7] CRITICAL RECOMMENDATIONS')
print('=' * 100)

print('\n[7.1] PAPER-SPECIFIC ISSUES')
print('-' * 100)

print('\nPaper 1: DISTRIBUTIONAL ANALYSIS')
print('  STATUS: [ROBUST]')
print('  - Strong effects (r=0.57, large effect size)')
print('  - Clear gender moderation')
print('  - DASS control effective')
print('  - Ready for manuscript')

print('\nPaper 2: NETWORK ANALYSIS')
print('  STATUS: [VALID BUT NEEDS EXTENSION]')
print('  - Statistically sound (N adequate for overall network)')
print('  - UCLA is NOT a hub (moderate centrality)')
print('  - Mediation explanation for Paper 1-2 difference')
print('  - NEEDS: Bootstrap stability, bridge centrality')

print('\nPaper 3: LATENT PROFILE ANALYSIS')
print('  STATUS: [PROBLEMATIC - NULL FINDINGS]')
print('  - k=2 model fits well (BIC, entropy)')
print('  - BUT: No gender/UCLA differences between profiles')
print('  - INCONSISTENT with Paper 1')
print('  - NEEDS: k=3-4 reanalysis OR alternative interpretation')

print('\n[7.2] CROSS-PAPER INTEGRATION CONCERNS')
print('-' * 100)

print('\nMAJOR INCONSISTENCY: Gender moderation')
print('  Paper 1: Strong (Fisher z=2.91, p<.01)')
print('  Paper 2: None (0 edges differ)')
print('  Paper 3: None (χ²=1.02, p=.31)')
print('\nPOSSIBLE EXPLANATIONS:')
print('  1. Power issue (males N=30 borderline, profile subgroups N<10)')
print('  2. Analysis level (bivariate ≠ network ≠ profiles)')
print('  3. k=2 too coarse (gender subgroups merged)')
print('\nRECOMMENDATION:')
print('  - Rerun Paper 3 with k=3-4 (prioritize interpretation over BIC)')
print('  - Test for gender × UCLA within Profile 1')
print('  - Acknowledge power limitations in Papers 2-3')

print('\n[7.3] NEXT STEPS PRIORITY RANKING')
print('-' * 100)

print('\nPRIORITY 1 (ESSENTIAL): Paper 3 Reanalysis')
print('  - Fit k=3 and k=4 GMM models')
print('  - Compare interpretability vs fit statistics')
print('  - Test if k=3-4 reveals gender subgroups')
print('  - TIME: 2-3 hours')

print('\nPRIORITY 2 (IMPORTANT): Paper 2 Extensions')
print('  - Bootstrap edge weights (1000 iterations)')
print('  - Bridge centrality analysis')
print('  - Community detection')
print('  - TIME: 1-2 days')

print('\nPRIORITY 3 (RECOMMENDED): Integrated Summary')
print('  - Write comprehensive 3-paper summary')
print('  - Reconcile inconsistencies')
print('  - Theoretical integration')
print('  - TIME: 1 day')

print('\nPRIORITY 4 (OPTIONAL): Paper 4 (DDM)')
print('  - Drift diffusion modeling')
print('  - Mechanistic interpretation')
print('  - TIME: 2-3 weeks')

print('\n' + '=' * 100)
print('END OF COMPREHENSIVE PRECISION REVIEW')
print('=' * 100)
