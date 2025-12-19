"""
WCST Hidden Markov Model Modeling (DASS-Controlled)
==================================================

Analysis of Hidden Markov Model attentional states with proper DASS control.

Background:
-----------
Initial HMM analysis found UCLA predicted Lapse->Focus transition (p=0.010),
but this became non-significant with full DASS control (p=0.78).

This suite re-examines the HMM lapse effect with:
1. Proper DASS integration at the model level
2. Gender-stratified HMM fitting
3. Mediation analysis: UCLA -> DASS -> Lapse -> EF
4. State characteristic analysis
5. Recovery dynamics analysis

Analyses:
---------
1. dass_controlled_hmm: HMM with proper DASS covariates
2. gender_stratified: Separate HMM models for males/females
3. mediation_pathway: UCLA -> DASS -> Lapse -> PE mediation
4. state_characteristics: Detailed state-specific analysis
5. recovery_dynamics: Lapse recovery patterns by UCLA

Usage:
    python -m publication.mechanism_analysis -a wcst_hmm_modeling_dass
    python -m publication.mechanism_analysis -a wcst_hmm_modeling_dass --sub dass_controlled_hmm
    python -m publication.mechanism_analysis -a wcst_hmm_modeling_dass --sub gender_stratified

    from publication.mechanism_analysis.wcst_hmm_modeling_dass import run
    run('dass_controlled_hmm')

Author: Research Team
Date: 2025-12
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import ast
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Publication imports
from publication.preprocessing import (
    load_master_dataset,
    load_wcst_trials as load_wcst_trials_shared,
    find_interaction_term,
    standardize_predictors,
    prepare_gender_variable,
    DEFAULT_RT_MIN,
    DEFAULT_RT_MAX,
)
from ._utils import BASE_OUTPUT

np.random.seed(42)

# Output directory
OUTPUT_DIR = BASE_OUTPUT / "wcst_hmm_modeling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check for hmmlearn
try:
    from hmmlearn import hmm as hmm_module
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARNING] hmmlearn not available - some analyses will be skipped")
    print("         Install with: pip install hmmlearn")


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable


ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, description: str):
    """Decorator to register an analysis function."""
    def decorator(func: Callable):
        ANALYSES[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_hmm_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for HMM analysis."""
    master = load_master_dataset(task="wcst", merge_cognitive_summary=True)
    master = prepare_gender_variable(master)
    master = standardize_predictors(master)

    trials, _ = load_wcst_trials_shared()
    trials.columns = trials.columns.str.lower()

    if 'rt_ms' in trials.columns:
        trials['rt'] = trials['rt_ms']
    elif 'reactiontimems' in trials.columns:
        trials['rt'] = trials['reactiontimems']

    if 'rt' in trials.columns:
        trials = trials[(trials['rt'] > DEFAULT_RT_MIN) & (trials['rt'] < DEFAULT_RT_MAX)].copy()

    if 'is_pe' not in trials.columns and 'extra' in trials.columns:
        def _parse_extra(extra_str):
            if not isinstance(extra_str, str):
                return {}
            try:
                return ast.literal_eval(extra_str)
            except (ValueError, SyntaxError):
                return {}
        trials['extra_dict'] = trials['extra'].apply(_parse_extra)
        trials['is_pe'] = trials['extra_dict'].apply(lambda x: x.get('isPE', False))

    sort_cols = ['participant_id']
    for cand in ['trialindex', 'trial_index', 'timestamp']:
        if cand in trials.columns:
            sort_cols.append(cand)
            break
    trials = trials.sort_values(sort_cols).reset_index(drop=True)
    trials['trial_num'] = trials.groupby('participant_id').cumcount() + 1

    return master, trials


def fit_hmm_per_participant(trials: pd.DataFrame, n_states: int = 2) -> pd.DataFrame:
    """
    Fit 2-state HMM for each participant.

    Returns DataFrame with:
    - lapse_occupancy: % time in lapse state
    - trans_to_lapse: P(Focus -> Lapse)
    - trans_to_focus: P(Lapse -> Focus)
    - stay_lapse: P(Lapse -> Lapse)
    """
    if not HMM_AVAILABLE:
        return pd.DataFrame()

    hmm_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 50:
            continue

        pdata = pdata.sort_values('trial_num')
        rts = pdata['rt'].values.reshape(-1, 1)

        try:
            model = hmm_module.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            model.fit(rts)

            states = model.predict(rts)
            means = model.means_.flatten()

            # Identify lapse state (higher RT mean)
            lapse_state = np.argmax(means)
            focus_state = 1 - lapse_state

            # State occupancy
            lapse_occupancy = (states == lapse_state).mean() * 100

            # Transition probabilities
            trans_matrix = model.transmat_

            trans_to_lapse = trans_matrix[focus_state, lapse_state]
            trans_to_focus = trans_matrix[lapse_state, focus_state]
            stay_lapse = trans_matrix[lapse_state, lapse_state]
            stay_focus = trans_matrix[focus_state, focus_state]

            # State means
            lapse_rt_mean = means[lapse_state]
            focus_rt_mean = means[focus_state]

            # Count state transitions
            state_changes = np.sum(np.diff(states) != 0)

            hmm_results.append({
                'participant_id': pid,
                'lapse_occupancy': lapse_occupancy,
                'trans_to_lapse': trans_to_lapse,
                'trans_to_focus': trans_to_focus,
                'stay_lapse': stay_lapse,
                'stay_focus': stay_focus,
                'lapse_rt_mean': lapse_rt_mean,
                'focus_rt_mean': focus_rt_mean,
                'rt_diff': lapse_rt_mean - focus_rt_mean,
                'state_changes': state_changes,
                'n_trials': len(pdata)
            })

        except Exception:
            continue

    return pd.DataFrame(hmm_results)


def get_dass_controlled_formula(outcome: str) -> str:
    """Get DASS-controlled regression formula per CLAUDE.md."""
    return f"{outcome} ~ z_ucla * C(gender_male) + z_dass_dep + z_dass_anx + z_dass_str + z_age"


# =============================================================================
# ANALYSIS 1: DASS-CONTROLLED HMM
# =============================================================================

@register_analysis(
    name="dass_controlled_hmm",
    description="HMM state analysis with proper DASS covariates"
)
def analyze_dass_controlled_hmm(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze HMM states with full DASS control.

    This is the definitive test of whether UCLA predicts attentional states
    after controlling for mood.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: DASS-CONTROLLED HMM")
        print("=" * 70)

    if not HMM_AVAILABLE:
        if verbose:
            print("  hmmlearn not available - skipping")
        return pd.DataFrame()

    master, trials = load_hmm_data()

    if verbose:
        print("\n  Fitting 2-state HMM per participant...")

    hmm_df = fit_hmm_per_participant(trials)

    if len(hmm_df) < 20:
        if verbose:
            print(f"  Insufficient HMM fits ({len(hmm_df)} participants)")
        return pd.DataFrame()

    merged = master.merge(hmm_df, on='participant_id', how='inner')

    if verbose:
        print(f"\n  N = {len(merged)}")
        print(f"  Mean lapse occupancy: {merged['lapse_occupancy'].mean():.1f}%")
        print(f"  Mean P(Lapse->Focus): {merged['trans_to_focus'].mean():.3f}")

    all_results = []

    # Logit transform probabilities for regression
    epsilon = 1e-4
    for prob_col in ['lapse_occupancy', 'trans_to_focus', 'trans_to_lapse', 'stay_lapse']:
        if prob_col == 'lapse_occupancy':
            prob = np.clip(merged[prob_col] / 100.0, epsilon, 1 - epsilon)
        else:
            prob = np.clip(merged[prob_col], epsilon, 1 - epsilon)
        merged[f'{prob_col}_logit'] = np.log(prob / (1 - prob))

    # Test key metrics
    key_metrics = [
        ('lapse_occupancy_logit', 'Lapse Occupancy'),
        ('trans_to_focus_logit', 'P(Lapse->Focus)'),
        ('stay_lapse_logit', 'P(Lapse->Lapse)')
    ]

    if verbose:
        print("\n  DASS-Controlled Regression Results:")
        print("  " + "-" * 60)

    for metric, label in key_metrics:
        merged_clean = merged.dropna(subset=[metric])
        if len(merged_clean) < 20:
            continue

        try:
            formula = get_dass_controlled_formula(metric)
            model = smf.wls(
                formula, data=merged_clean,
                weights=merged_clean['n_trials']
            ).fit(cov_type='HC3')

            int_term = find_interaction_term(model.params.index)
            terms_to_check = ['z_ucla', 'C(gender_male)[T.1]']
            if int_term:
                terms_to_check.append(int_term)

            for term in terms_to_check:
                if term in model.params:
                    beta = model.params[term]
                    se = model.bse[term]
                    p = model.pvalues[term]

                    term_label = (term
                        .replace('z_ucla', 'UCLA')
                        .replace('C(gender_male)[T.1]', 'Male')
                        .replace(':C(gender_male)[T.1]', ' x Male')
                        .replace(':', ' x '))

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    {label} ~ {term_label}: beta={beta:.4f}, SE={se:.4f}, p={p:.4f}{sig}")

                    all_results.append({
                        'outcome': label,
                        'term': term_label,
                        'beta': beta,
                        'se': se,
                        'p': p,
                        'n': len(merged_clean),
                        'r2': model.rsquared
                    })

        except Exception as e:
            if verbose:
                print(f"    Error for {label}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "dass_controlled_hmm_results.csv", index=False, encoding='utf-8-sig')
    hmm_df.to_csv(OUTPUT_DIR / "hmm_states_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'dass_controlled_hmm_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 2: GENDER-STRATIFIED HMM
# =============================================================================

@register_analysis(
    name="gender_stratified",
    description="Separate HMM analysis for males and females"
)
def analyze_gender_stratified(verbose: bool = True) -> pd.DataFrame:
    """
    Fit and analyze HMM separately for each gender.

    This tests whether UCLA-lapse relationships differ fundamentally
    between males and females.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: GENDER-STRATIFIED HMM")
        print("=" * 70)

    if not HMM_AVAILABLE:
        if verbose:
            print("  hmmlearn not available - skipping")
        return pd.DataFrame()

    master, trials = load_hmm_data()

    # Merge gender info with trials
    trials_with_gender = trials.merge(
        master[['participant_id', 'gender_male', 'z_ucla', 'z_dass_dep', 'z_dass_anx', 'z_dass_str']],
        on='participant_id',
        how='inner'
    )

    all_results = []

    for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
        if verbose:
            print(f"\n  {gender_label}")
            print("  " + "-" * 50)

        gender_trials = trials_with_gender[trials_with_gender['gender_male'] == gender_val].copy()
        gender_master = master[master['gender_male'] == gender_val].copy()

        hmm_df = fit_hmm_per_participant(gender_trials)

        if len(hmm_df) < 15:
            if verbose:
                print(f"    Insufficient data ({len(hmm_df)} participants)")
            continue

        merged = gender_master.merge(hmm_df, on='participant_id', how='inner')

        if verbose:
            print(f"    N = {len(merged)}")
            print(f"    Mean lapse occupancy: {merged['lapse_occupancy'].mean():.1f}%")

        # Test UCLA effects (DASS-controlled, no gender interaction since stratified)
        for metric in ['lapse_occupancy', 'trans_to_focus']:
            merged_clean = merged.dropna(subset=[metric])
            if len(merged_clean) < 15:
                continue

            try:
                # Simpler formula without gender (already stratified)
                formula = f"{metric} ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_str + z_age"
                model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

                if 'z_ucla' in model.params:
                    beta = model.params['z_ucla']
                    p = model.pvalues['z_ucla']

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    {metric} ~ UCLA: beta={beta:.4f}, p={p:.4f}{sig}")

                    all_results.append({
                        'gender': gender_label,
                        'outcome': metric,
                        'beta_ucla': beta,
                        'p_ucla': p,
                        'n': len(merged_clean),
                        'r2': model.rsquared
                    })

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")

        # Simple correlation for interpretation
        if len(merged) >= 10:
            r, p = stats.pearsonr(merged['z_ucla'], merged['lapse_occupancy'])
            if verbose:
                sig = "*" if p < 0.05 else ""
                print(f"    UCLA-Lapse correlation: r={r:.3f}, p={p:.4f}{sig}")

            all_results.append({
                'gender': gender_label,
                'outcome': 'lapse_occupancy_corr',
                'beta_ucla': r,
                'p_ucla': p,
                'n': len(merged),
                'r2': r**2
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "gender_stratified_hmm_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'gender_stratified_hmm_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 3: MEDIATION PATHWAY
# =============================================================================

@register_analysis(
    name="mediation_pathway",
    description="UCLA -> DASS -> Lapse -> PE mediation analysis"
)
def analyze_mediation_pathway(verbose: bool = True) -> pd.DataFrame:
    """
    Test mediation: UCLA -> DASS -> Lapse Occupancy -> PE Rate

    Even if UCLA doesn't directly predict lapse states, it might work
    through DASS (mood) as a mediator.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: MEDIATION PATHWAY")
        print("=" * 70)
        print("  Testing: UCLA -> DASS -> Lapse -> PE")

    if not HMM_AVAILABLE:
        if verbose:
            print("  hmmlearn not available - skipping")
        return pd.DataFrame()

    master, trials = load_hmm_data()
    hmm_df = fit_hmm_per_participant(trials)

    if len(hmm_df) < 20:
        if verbose:
            print(f"  Insufficient HMM fits ({len(hmm_df)} participants)")
        return pd.DataFrame()

    merged = master.merge(hmm_df, on='participant_id', how='inner')

    # Need PE rate
    if 'pe_rate' not in merged.columns:
        if verbose:
            print("  Computing PE rate from trials...")
        pe_rates = trials.groupby('participant_id')['is_pe'].mean() * 100
        pe_rates = pe_rates.reset_index()
        pe_rates.columns = ['participant_id', 'pe_rate']
        merged = merged.merge(pe_rates, on='participant_id', how='left')

    merged_clean = merged.dropna(subset=['z_ucla', 'dass_depression', 'lapse_occupancy', 'pe_rate'])

    if len(merged_clean) < 30:
        if verbose:
            print(f"  Insufficient complete cases ({len(merged_clean)})")
        return pd.DataFrame()

    if verbose:
        print(f"\n  N = {len(merged_clean)}")

    results = []

    # Path a: UCLA -> DASS (Depression as primary mediator)
    model_a = smf.ols('dass_depression ~ z_ucla', data=merged_clean).fit()
    a_coef = model_a.params['z_ucla']
    a_p = model_a.pvalues['z_ucla']

    if verbose:
        print(f"\n  Path a (UCLA -> DASS): beta={a_coef:.3f}, p={a_p:.4f}")

    results.append({
        'path': 'a (UCLA -> DASS)',
        'beta': a_coef,
        'p': a_p
    })

    # Path b: DASS -> Lapse (controlling for UCLA)
    model_b = smf.ols('lapse_occupancy ~ dass_depression + z_ucla', data=merged_clean).fit()
    b_coef = model_b.params['dass_depression']
    b_p = model_b.pvalues['dass_depression']

    if verbose:
        print(f"  Path b (DASS -> Lapse | UCLA): beta={b_coef:.3f}, p={b_p:.4f}")

    results.append({
        'path': 'b (DASS -> Lapse)',
        'beta': b_coef,
        'p': b_p
    })

    # Path c: Lapse -> PE (controlling for DASS)
    model_c = smf.ols('pe_rate ~ lapse_occupancy + dass_depression + z_ucla', data=merged_clean).fit()
    c_coef = model_c.params['lapse_occupancy']
    c_p = model_c.pvalues['lapse_occupancy']

    if verbose:
        print(f"  Path c (Lapse -> PE | DASS, UCLA): beta={c_coef:.3f}, p={c_p:.4f}")

    results.append({
        'path': 'c (Lapse -> PE)',
        'beta': c_coef,
        'p': c_p
    })

    # Total indirect effect: a * b * c
    indirect_effect = a_coef * b_coef * c_coef

    if verbose:
        print(f"\n  Indirect Effect (a*b*c): {indirect_effect:.4f}")

    # Bootstrap CI for indirect effect
    n_boot = 1000
    indirect_boots = []

    for _ in range(n_boot):
        boot_sample = merged_clean.sample(n=len(merged_clean), replace=True)

        try:
            ma = smf.ols('dass_depression ~ z_ucla', data=boot_sample).fit()
            mb = smf.ols('lapse_occupancy ~ dass_depression + z_ucla', data=boot_sample).fit()
            mc = smf.ols('pe_rate ~ lapse_occupancy + dass_depression + z_ucla', data=boot_sample).fit()

            indirect = ma.params['z_ucla'] * mb.params['dass_depression'] * mc.params['lapse_occupancy']
            indirect_boots.append(indirect)
        except:
            continue

    if len(indirect_boots) > 100:
        ci_lower = np.percentile(indirect_boots, 2.5)
        ci_upper = np.percentile(indirect_boots, 97.5)
        sig_mediation = (ci_lower > 0) or (ci_upper < 0)

        if verbose:
            sig = "*" if sig_mediation else ""
            print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]{sig}")

        results.append({
            'path': 'indirect (a*b*c)',
            'beta': indirect_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': sig_mediation
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "mediation_pathway_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'mediation_pathway_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 4: STATE CHARACTERISTICS
# =============================================================================

@register_analysis(
    name="state_characteristics",
    description="Detailed state-specific analysis"
)
def analyze_state_characteristics(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze characteristics of the lapse vs focus states.

    Does UCLA predict the nature of states (RT difference, variability)?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: STATE CHARACTERISTICS")
        print("=" * 70)

    if not HMM_AVAILABLE:
        if verbose:
            print("  hmmlearn not available - skipping")
        return pd.DataFrame()

    master, trials = load_hmm_data()
    hmm_df = fit_hmm_per_participant(trials)

    if len(hmm_df) < 20:
        if verbose:
            print(f"  Insufficient data ({len(hmm_df)} participants)")
        return pd.DataFrame()

    merged = master.merge(hmm_df, on='participant_id', how='inner')

    if verbose:
        print(f"\n  N = {len(merged)}")
        print(f"  Mean Focus RT: {merged['focus_rt_mean'].mean():.0f} ms")
        print(f"  Mean Lapse RT: {merged['lapse_rt_mean'].mean():.0f} ms")
        print(f"  Mean RT Difference: {merged['rt_diff'].mean():.0f} ms")

    all_results = []

    # Test UCLA effects on state characteristics
    for metric in ['rt_diff', 'state_changes', 'lapse_rt_mean', 'focus_rt_mean']:
        merged_clean = merged.dropna(subset=[metric])
        if len(merged_clean) < 20:
            continue

        try:
            formula = get_dass_controlled_formula(metric)
            model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

            int_term = find_interaction_term(model.params.index)
            terms_to_check = ['z_ucla']
            if int_term:
                terms_to_check.append(int_term)

            for term in terms_to_check:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = (term
                        .replace('z_ucla', 'UCLA')
                        .replace(':C(gender_male)[T.1]', ' x Male')
                        .replace('C(gender_male)[T.1]', 'Male')
                        .replace(':', ' x '))

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    {metric} ~ {label}: beta={beta:.3f}, p={p:.4f}{sig}")

                    all_results.append({
                        'outcome': metric,
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged_clean)
                    })

        except Exception as e:
            if verbose:
                print(f"    Error for {metric}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "state_characteristics_results.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'state_characteristics_results.csv'}")

    return results_df


# =============================================================================
# ANALYSIS 5: RECOVERY DYNAMICS
# =============================================================================

@register_analysis(
    name="recovery_dynamics",
    description="Lapse recovery patterns by UCLA"
)
def analyze_recovery_dynamics(verbose: bool = True) -> pd.DataFrame:
    """
    Analyze how quickly participants recover from lapse states.

    Does UCLA predict recovery speed after controlling for DASS?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS: RECOVERY DYNAMICS")
        print("=" * 70)

    if not HMM_AVAILABLE:
        if verbose:
            print("  hmmlearn not available - skipping")
        return pd.DataFrame()

    master, trials = load_hmm_data()

    recovery_results = []

    for pid, pdata in trials.groupby('participant_id'):
        if len(pdata) < 50:
            continue

        pdata = pdata.sort_values('trial_num')
        rts = pdata['rt'].values.reshape(-1, 1)

        try:
            model = hmm_module.GaussianHMM(
                n_components=2,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            model.fit(rts)
            states = model.predict(rts)
            means = model.means_.flatten()
            lapse_state = np.argmax(means)

            # Find lapse episodes and their durations
            lapse_durations = []
            current_duration = 0

            for state in states:
                if state == lapse_state:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        lapse_durations.append(current_duration)
                    current_duration = 0

            if current_duration > 0:
                lapse_durations.append(current_duration)

            if len(lapse_durations) > 0:
                recovery_results.append({
                    'participant_id': pid,
                    'mean_lapse_duration': np.mean(lapse_durations),
                    'max_lapse_duration': np.max(lapse_durations),
                    'n_lapse_episodes': len(lapse_durations),
                    'n_trials': len(pdata)
                })

        except:
            continue

    if len(recovery_results) < 20:
        if verbose:
            print(f"  Insufficient data ({len(recovery_results)} participants)")
        return pd.DataFrame()

    recovery_df = pd.DataFrame(recovery_results)
    merged = master.merge(recovery_df, on='participant_id', how='inner')

    if verbose:
        print(f"\n  N = {len(merged)}")
        print(f"  Mean lapse duration: {merged['mean_lapse_duration'].mean():.2f} trials")
        print(f"  Mean lapse episodes: {merged['n_lapse_episodes'].mean():.1f}")

    all_results = []

    for metric in ['mean_lapse_duration', 'max_lapse_duration', 'n_lapse_episodes']:
        merged_clean = merged.dropna(subset=[metric])
        if len(merged_clean) < 20:
            continue

        try:
            formula = get_dass_controlled_formula(metric)
            model = smf.ols(formula, data=merged_clean).fit(cov_type='HC3')

            int_term = find_interaction_term(model.params.index)
            terms_to_check = ['z_ucla']
            if int_term:
                terms_to_check.append(int_term)

            for term in terms_to_check:
                if term in model.params:
                    beta = model.params[term]
                    p = model.pvalues[term]

                    label = (term
                        .replace('z_ucla', 'UCLA')
                        .replace(':C(gender_male)[T.1]', ' x Male')
                        .replace('C(gender_male)[T.1]', 'Male')
                        .replace(':', ' x '))

                    if verbose:
                        sig = "*" if p < 0.05 else ""
                        print(f"    {metric} ~ {label}: beta={beta:.3f}, p={p:.4f}{sig}")

                    all_results.append({
                        'outcome': metric,
                        'term': label,
                        'beta': beta,
                        'p': p,
                        'n': len(merged_clean)
                    })

        except Exception as e:
            if verbose:
                print(f"    Error for {metric}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "recovery_dynamics_results.csv", index=False, encoding='utf-8-sig')
    recovery_df.to_csv(OUTPUT_DIR / "recovery_by_participant.csv", index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\n  Output: {OUTPUT_DIR / 'recovery_dynamics_results.csv'}")

    return results_df


# =============================================================================
# SUMMARY VISUALIZATION
# =============================================================================

def create_summary_visualization(verbose: bool = True) -> None:
    """Create summary figure for HMM mechanism analysis."""
    if verbose:
        print("\n" + "=" * 70)
        print("CREATING SUMMARY VISUALIZATION")
        print("=" * 70)

    result_files = {
        'DASS-Controlled': OUTPUT_DIR / 'dass_controlled_hmm_results.csv',
        'Gender-Stratified': OUTPUT_DIR / 'gender_stratified_hmm_results.csv',
        'State Characteristics': OUTPUT_DIR / 'state_characteristics_results.csv',
        'Recovery Dynamics': OUTPUT_DIR / 'recovery_dynamics_results.csv'
    }

    all_effects = []

    for analysis_name, filepath in result_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Get UCLA main effects
            if 'term' in df.columns:
                ucla_effects = df[df['term'].str.contains('UCLA', case=False, na=False)]
                for _, row in ucla_effects.iterrows():
                    all_effects.append({
                        'Analysis': analysis_name,
                        'Outcome': row.get('outcome', ''),
                        'Term': row['term'],
                        'Beta': row['beta'],
                        'P-value': row['p']
                    })
            elif 'beta_ucla' in df.columns:
                for _, row in df.iterrows():
                    all_effects.append({
                        'Analysis': analysis_name,
                        'Outcome': row.get('outcome', row.get('gender', '')),
                        'Term': 'UCLA',
                        'Beta': row['beta_ucla'],
                        'P-value': row['p_ucla']
                    })

    if len(all_effects) == 0:
        if verbose:
            print("  No results to visualize")
        return

    effects_df = pd.DataFrame(all_effects)

    fig, ax = plt.subplots(figsize=(12, max(6, len(effects_df) * 0.35)))

    y_pos = range(len(effects_df))
    colors = ['#E74C3C' if p < 0.05 else '#3498DB' for p in effects_df['P-value']]

    ax.barh(y_pos, effects_df['Beta'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    labels = [f"{row['Analysis']}: {row['Outcome']} ({row['Term']})" for _, row in effects_df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Coefficient (Beta)')
    ax.set_title('HMM Mechanism Analysis: UCLA Effects (DASS-Controlled)\n(Red = p < 0.05)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hmm_mechanism_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {OUTPUT_DIR / 'hmm_mechanism_summary.png'}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(analysis: Optional[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run HMM mechanism analyses.

    Parameters
    ----------
    analysis : str, optional
        Specific analysis to run. If None, runs all.
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict
        Results from all analyses.
    """
    if verbose:
        print("=" * 70)
        print("WCST HMM MODELING SUITE")
        print("=" * 70)
        print("\nTesting: Is there ANY unique UCLA effect on attentional dynamics")
        print("after controlling for DASS (mood)?")

    results = {}

    if analysis:
        if analysis not in ANALYSES:
            raise ValueError(f"Unknown analysis: {analysis}. Available: {list(ANALYSES.keys())}")
        spec = ANALYSES[analysis]
        if verbose:
            print(f"\nRunning: {spec.name}")
        results[analysis] = spec.function(verbose=verbose)
    else:
        for name, spec in ANALYSES.items():
            try:
                if verbose:
                    print(f"\n--- Running: {spec.description} ---")
                results[name] = spec.function(verbose=verbose)
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

        create_summary_visualization(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("HMM MODELING SUITE COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 70)

    return results


def list_analyses():
    """List available analyses."""
    print("\nAvailable HMM Modeling Analyses:")
    print("-" * 60)
    for name, spec in ANALYSES.items():
        print(f"  {name}")
        print(f"    {spec.description}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WCST HMM Modeling Suite (DASS-Controlled)")
    parser.add_argument('--analysis', '-a', type=str, default=None,
                        help="Specific analysis to run")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available analyses")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Suppress output")
    args = parser.parse_args()

    if args.list:
        list_analyses()
    else:
        run(analysis=args.analysis, verbose=not args.quiet)
