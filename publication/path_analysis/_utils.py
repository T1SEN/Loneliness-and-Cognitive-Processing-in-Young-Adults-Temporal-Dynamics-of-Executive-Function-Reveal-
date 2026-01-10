"""
Shared utilities for EF-focused path analyses.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from semopy import Model as SemopyModel
    from semopy import calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    SemopyModel = None
    calc_stats = None

from publication.preprocessing import (
    load_master_dataset,
    standardize_predictors,
    prepare_gender_variable,
)
from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR, VALID_TASKS

BASE_OUTPUT = ANALYSIS_OUTPUT_DIR / "path_analysis"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)


def _validate_task(task: str) -> str:
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {sorted(VALID_TASKS)}")
    return task


def get_output_dir(task: str, analysis_name: str) -> Path:
    task = _validate_task(task)
    output_dir = BASE_OUTPUT / task / analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Default EF targets analyzed separately
EF_TARGETS: List[Dict[str, str]] = [
    {'col': 'pe_rate', 'label': 'WCST Perseverative Error Rate'},
    {'col': 'stroop_interference', 'label': 'Stroop Interference Effect'},
    {'col': 'prp_bottleneck', 'label': 'PRP Bottleneck Effect'},
    {'col': 'prp_cb_base', 'label': 'PRP CBT Base (Plateau RT2)'},
    {'col': 'prp_cb_bottleneck', 'label': 'PRP CBT Bottleneck Duration'},
    {'col': 'prp_cb_r_squared', 'label': 'PRP CBT R-squared'},
    {'col': 'prp_cb_rmse', 'label': 'PRP CBT RMSE'},
    {'col': 'prp_cb_slope', 'label': 'PRP CBT Short-SOA Slope'},
    {'col': 'prp_exg_short_mu', 'label': 'PRP Ex-Gaussian mu (Short SOA)'},
    {'col': 'prp_exg_short_sigma', 'label': 'PRP Ex-Gaussian sigma (Short SOA)'},
    {'col': 'prp_exg_short_tau', 'label': 'PRP Ex-Gaussian tau (Short SOA)'},
    {'col': 'prp_exg_long_mu', 'label': 'PRP Ex-Gaussian mu (Long SOA)'},
    {'col': 'prp_exg_long_sigma', 'label': 'PRP Ex-Gaussian sigma (Long SOA)'},
    {'col': 'prp_exg_long_tau', 'label': 'PRP Ex-Gaussian tau (Long SOA)'},
    {'col': 'prp_exg_overall_mu', 'label': 'PRP Ex-Gaussian mu (Overall)'},
    {'col': 'prp_exg_overall_sigma', 'label': 'PRP Ex-Gaussian sigma (Overall)'},
    {'col': 'prp_exg_overall_tau', 'label': 'PRP Ex-Gaussian tau (Overall)'},
    {'col': 'prp_exg_mu_bottleneck', 'label': 'PRP Ex-Gaussian mu (Bottleneck)'},
    {'col': 'prp_exg_sigma_bottleneck', 'label': 'PRP Ex-Gaussian sigma (Bottleneck)'},
    {'col': 'prp_exg_tau_bottleneck', 'label': 'PRP Ex-Gaussian tau (Bottleneck)'},
    {'col': 'stroop_exg_congruent_mu', 'label': 'Stroop Ex-Gaussian mu (Congruent)'},
    {'col': 'stroop_exg_congruent_sigma', 'label': 'Stroop Ex-Gaussian sigma (Congruent)'},
    {'col': 'stroop_exg_congruent_tau', 'label': 'Stroop Ex-Gaussian tau (Congruent)'},
    {'col': 'stroop_exg_incongruent_mu', 'label': 'Stroop Ex-Gaussian mu (Incongruent)'},
    {'col': 'stroop_exg_incongruent_sigma', 'label': 'Stroop Ex-Gaussian sigma (Incongruent)'},
    {'col': 'stroop_exg_incongruent_tau', 'label': 'Stroop Ex-Gaussian tau (Incongruent)'},
    {'col': 'stroop_exg_neutral_mu', 'label': 'Stroop Ex-Gaussian mu (Neutral)'},
    {'col': 'stroop_exg_neutral_sigma', 'label': 'Stroop Ex-Gaussian sigma (Neutral)'},
    {'col': 'stroop_exg_neutral_tau', 'label': 'Stroop Ex-Gaussian tau (Neutral)'},
    {'col': 'stroop_exg_mu_interference', 'label': 'Stroop Ex-Gaussian mu (Interference)'},
    {'col': 'stroop_exg_sigma_interference', 'label': 'Stroop Ex-Gaussian sigma (Interference)'},
    {'col': 'stroop_exg_tau_interference', 'label': 'Stroop Ex-Gaussian tau (Interference)'},
    {'col': 'wcst_hmm_lapse_occupancy', 'label': 'WCST HMM Lapse Occupancy (%)'},
    {'col': 'wcst_hmm_trans_to_lapse', 'label': 'WCST HMM P(Focus->Lapse)'},
    {'col': 'wcst_hmm_trans_to_focus', 'label': 'WCST HMM P(Lapse->Focus)'},
    {'col': 'wcst_hmm_stay_lapse', 'label': 'WCST HMM P(Lapse->Lapse)'},
    {'col': 'wcst_hmm_stay_focus', 'label': 'WCST HMM P(Focus->Focus)'},
    {'col': 'wcst_hmm_lapse_rt_mean', 'label': 'WCST HMM Lapse RT Mean'},
    {'col': 'wcst_hmm_focus_rt_mean', 'label': 'WCST HMM Focus RT Mean'},
    {'col': 'wcst_hmm_rt_diff', 'label': 'WCST HMM RT Difference'},
    {'col': 'wcst_hmm_state_changes', 'label': 'WCST HMM State Changes'},
    {'col': 'wcst_rl_alpha', 'label': 'WCST RL alpha'},
    {'col': 'wcst_rl_beta', 'label': 'WCST RL beta'},
    {'col': 'wcst_rl_alpha_pos', 'label': 'WCST RL alpha (pos)'},
    {'col': 'wcst_rl_alpha_neg', 'label': 'WCST RL alpha (neg)'},
    {'col': 'wcst_rl_alpha_asymmetry', 'label': 'WCST RL alpha asymmetry'},
    {'col': 'wcst_rl_beta_asym', 'label': 'WCST RL beta (asym)'},
    {'col': 'wcst_wsls_p_stay_win', 'label': 'WCST WSLS P(stay|win)'},
    {'col': 'wcst_wsls_p_shift_lose', 'label': 'WCST WSLS P(shift|lose)'},
    {'col': 'wcst_brl_hazard', 'label': 'WCST Bayesian RL hazard'},
    {'col': 'wcst_brl_noise', 'label': 'WCST Bayesian RL noise'},
    {'col': 'wcst_brl_beta', 'label': 'WCST Bayesian RL beta'},
]


@dataclass
class PathModelResult:
    stage1: pd.DataFrame
    stage2: pd.DataFrame
    path_effects: pd.DataFrame
    fit_stats: Dict[str, float]
    bootstrap: pd.DataFrame
    sobel: pd.DataFrame
    diagnostics: Dict[str, float]
    description: str
    ef_label: str
    ef_col: str


PATH_SPECS: List[Dict] = [
    {
        'name': 'loneliness_to_dass_to_ef',
        'description': 'Loneliness -> DASS -> {ef_label}',
        'stage1_label': 'Loneliness predicting DASS',
        'stage2_label': '{ef_label} outcome',
        'stage1_formula': lambda dass_col, ef_col: (
            f"{dass_col} ~ z_ucla * C(gender_male) + z_age"
        ),
        'stage2_formula': lambda dass_col, ef_col: (
            f"{ef_col} ~ z_ucla * C(gender_male) + "
            f"{dass_col} * C(gender_male) + z_age"
        ),
        'paths': [
            {'path': "UCLA -> {dass_label}", 'model': 'stage1', 'term': 'z_ucla', 'role': 'a'},
            {'path': "{dass_label} -> {ef_label}", 'model': 'stage2', 'term': '{dass_col}', 'role': 'b'},
            {'path': "UCLA -> {ef_label} (direct)", 'model': 'stage2', 'term': 'z_ucla', 'role': 'direct'},
        ],
        'sem_model': lambda dass_col, ef_col: (
            f"{dass_col} ~ c1*z_ucla + c2*gender_male + c3*z_age + c4*ucla_male\n"
            f"{ef_col} ~ b1*z_ucla + b2*{dass_col} + b3*gender_male + "
            f"b4*z_age + b5*ucla_male + b6*dass_male"
        ),
        'indirect': {
            'stage_a': 'stage1',
            'term_a': 'z_ucla',
            'stage_b': 'stage2',
            'term_b': '{dass_col}',
        },
    },
    {
        'name': 'loneliness_to_ef_to_dass',
        'description': 'Loneliness -> {ef_label} -> DASS',
        'stage1_label': 'Loneliness predicting {ef_label}',
        'stage2_label': '{dass_label} outcome',
        'stage1_formula': lambda dass_col, ef_col: (
            f"{ef_col} ~ z_ucla * C(gender_male) + z_age"
        ),
        'stage2_formula': lambda dass_col, ef_col: (
            f"{dass_col} ~ {ef_col} * C(gender_male) + "
            f"z_ucla * C(gender_male) + z_age"
        ),
        'paths': [
            {'path': "UCLA -> {ef_label}", 'model': 'stage1', 'term': 'z_ucla', 'role': 'a'},
            {'path': "{ef_label} -> {dass_label}", 'model': 'stage2', 'term': '{ef_col}', 'role': 'b'},
            {'path': "UCLA -> {dass_label} (direct)", 'model': 'stage2', 'term': 'z_ucla', 'role': 'direct'},
        ],
        'sem_model': lambda dass_col, ef_col: (
            f"{ef_col} ~ c1*z_ucla + c2*gender_male + c3*z_age + c4*ucla_male\n"
            f"{dass_col} ~ b1*{ef_col} + b2*z_ucla + b3*gender_male + "
            f"b4*z_age + b5*ef_male + b6*ucla_male"
        ),
        'indirect': {
            'stage_a': 'stage1',
            'term_a': 'z_ucla',
            'stage_b': 'stage2',
            'term_b': '{ef_col}',
        },
    },
    {
        'name': 'dass_to_loneliness_to_ef',
        'description': '{dass_label} -> Loneliness -> {ef_label}',
        'stage1_label': '{dass_label} predicting loneliness',
        'stage2_label': '{ef_label} outcome',
        'stage1_formula': lambda dass_col, ef_col: (
            f"z_ucla ~ {dass_col} * C(gender_male) + z_age"
        ),
        'stage2_formula': lambda dass_col, ef_col: (
            f"{ef_col} ~ z_ucla * C(gender_male) + "
            f"{dass_col} * C(gender_male) + z_age"
        ),
        'paths': [
            {'path': "{dass_label} -> Loneliness", 'model': 'stage1', 'term': '{dass_col}', 'role': 'a'},
            {'path': "Loneliness -> {ef_label}", 'model': 'stage2', 'term': 'z_ucla', 'role': 'b'},
            {'path': "{dass_label} -> {ef_label} (direct)", 'model': 'stage2', 'term': '{dass_col}', 'role': 'direct'},
        ],
        'sem_model': lambda dass_col, ef_col: (
            f"z_ucla ~ c1*{dass_col} + c2*gender_male + c3*z_age + c4*dass_male\n"
            f"{ef_col} ~ b1*z_ucla + b2*{dass_col} + b3*gender_male + "
            f"b4*z_age + b5*ucla_male + b6*dass_male"
        ),
        'indirect': {
            'stage_a': 'stage1',
            'term_a': '{dass_col}',
            'stage_b': 'stage2',
            'term_b': 'z_ucla',
        },
    },
]


def load_common_path_data(
    target_col: str,
    task: str,
    min_n: int = 50,
    ef_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load master dataset with predictors needed for path analysis.
    """
    task = _validate_task(task)
    master = load_master_dataset(task=task, merge_cognitive_summary=True)
    master = prepare_gender_variable(master)
    master = standardize_predictors(master)

    ef_cols = ef_cols or [spec['col'] for spec in EF_TARGETS]

    required_base = [
        'z_ucla',
        target_col,
        'z_age',
        'gender_male',
    ]

    missing = [col for col in required_base if col not in master.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    available_ef = [col for col in ef_cols if col in master.columns]
    keep_cols = required_base + available_ef

    df = master[keep_cols].dropna(subset=required_base)
    if len(df) < min_n:
        raise ValueError(f"Insufficient data for {target_col}: N={len(df)} < {min_n}")
    return df


def _summarize_model(model, model_name: str) -> pd.DataFrame:
    records = []
    for term in model.params.index:
        records.append({
            'model': model_name,
            'term': term,
            'beta': model.params[term],
            'se': model.bse[term],
            't': model.tvalues[term],
            'p': model.pvalues[term],
        })
    return pd.DataFrame(records)


def _find_interaction_term(model, base_term: str) -> Optional[str]:
    candidates = [
        f"{base_term}:C(gender_male)[T.1]",
        f"C(gender_male)[T.1]:{base_term}",
    ]
    for cand in candidates:
        if cand in model.params.index:
            return cand
    return None


def _gender_effect_stats(model, base_term: str) -> Dict[str, float]:
    base = model.params.get(base_term, np.nan)
    base_se = model.bse.get(base_term, np.nan)
    inter_term = _find_interaction_term(model, base_term)
    cov = model.cov_params()
    female_beta = base
    female_se = base_se

    female_p = model.pvalues.get(base_term, np.nan)

    if inter_term:
        inter = model.params.get(inter_term, 0.0)
        inter_se = model.bse.get(inter_term, np.nan)
        cov_val = 0.0
        try:
            cov_val = cov.loc[base_term, inter_term]
        except Exception:
            cov_val = 0.0
        male_beta = base + inter
        male_var = (base_se ** 2) + (inter_se ** 2) + 2 * cov_val
        male_se = np.sqrt(max(male_var, 0.0))
    else:
        male_beta = base
        male_se = base_se
        male_p = female_p

    # Wald test for male effect (base + interaction)
    if inter_term:
        try:
            contrast = np.zeros(len(model.params))
            idx_base = model.params.index.get_loc(base_term)
            idx_int = model.params.index.get_loc(inter_term)
            contrast[idx_base] = 1
            contrast[idx_int] = 1
            wald = model.wald_test(contrast, use_f=False)
            male_p = float(wald.pvalue)
        except Exception:
            pass

    return {
        'female_beta': female_beta,
        'female_se': female_se,
        'female_p': female_p,
        'male_beta': male_beta,
        'male_se': male_se,
        'male_p': male_p,
    }


def _sobel_test(a: float, sa: float, b: float, sb: float) -> Dict[str, float]:
    if any(np.isnan(val) for val in [a, sa, b, sb]):
        return {'z': np.nan, 'p': np.nan}
    se = np.sqrt((b ** 2) * (sa ** 2) + (a ** 2) * (sb ** 2))
    if se == 0 or np.isnan(se):
        return {'z': np.nan, 'p': np.nan}
    z_val = (a * b) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
    return {'z': z_val, 'p': p_val}


def _compute_vif_table(model) -> pd.DataFrame:
    exog = getattr(model.model, "exog", None)
    exog_names = list(getattr(model.model, "exog_names", []))
    if exog is None or len(exog_names) == 0:
        return pd.DataFrame(columns=["term", "vif"])

    X = pd.DataFrame(exog, columns=exog_names)
    drop_cols = [c for c in X.columns if c.lower() in {"intercept", "const"}]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    variances = X.var(axis=0)
    keep_cols = variances[variances > 1e-12].index.tolist()
    X = X[keep_cols]
    if X.shape[1] < 2:
        return pd.DataFrame(columns=["term", "vif"])

    vifs = []
    for i, col in enumerate(X.columns):
        try:
            vif_val = variance_inflation_factor(X.values, i)
        except Exception:
            vif_val = np.nan
        vifs.append({"term": col, "vif": float(vif_val)})
    return pd.DataFrame(vifs)


def _summarize_vif(vif_df: pd.DataFrame) -> Dict[str, float]:
    if vif_df.empty or "vif" not in vif_df.columns:
        return {"vif_max": np.nan, "vif_mean": np.nan}
    return {
        "vif_max": float(vif_df["vif"].max()),
        "vif_mean": float(vif_df["vif"].mean()),
    }


def _bootstrap_indirect_effects(
    df: pd.DataFrame,
    stage1_formula: str,
    stage2_formula: str,
    term_a: str,
    term_b: str,
    n_bootstrap: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    female_vals = []
    male_vals = []

    for _ in range(n_bootstrap):
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
        try:
            stage1 = smf.ols(stage1_formula, data=sample).fit(cov_type='HC3')
            stage2 = smf.ols(stage2_formula, data=sample).fit(cov_type='HC3')
        except Exception:
            continue

        a_stats = _gender_effect_stats(stage1, term_a)
        b_stats = _gender_effect_stats(stage2, term_b)

        female_vals.append(a_stats['female_beta'] * b_stats['female_beta'])
        male_vals.append(a_stats['male_beta'] * b_stats['male_beta'])

    def summarize(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {'mean': np.nan, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
        arr = np.array(vals)
        return {
            'mean': float(np.mean(arr)),
            'se': float(np.std(arr, ddof=1)),
            'ci_lower': float(np.percentile(arr, 2.5)),
            'ci_upper': float(np.percentile(arr, 97.5)),
        }

    female_stats = summarize(female_vals)
    male_stats = summarize(male_vals)

    female_stats['significant'] = female_stats['ci_lower'] > 0 or female_stats['ci_upper'] < 0
    male_stats['significant'] = male_stats['ci_lower'] > 0 or male_stats['ci_upper'] < 0

    return pd.DataFrame([
        {'gender': 'Female', **female_stats},
        {'gender': 'Male', **male_stats},
    ])


def fit_all_path_models(
    df: pd.DataFrame,
    dass_col: str,
    dass_label: str,
    output_dir: Path,
    verbose: bool = True,
    n_bootstrap: int = 5000,
    bootstrap_seed: int = 42,
    ef_targets: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Dict[str, PathModelResult]]:
    """
    Fit gender-moderated path models for each EF outcome separately.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_defs = ef_targets or EF_TARGETS
    all_results: Dict[str, Dict[str, PathModelResult]] = {}

    base_cols = ['z_ucla', dass_col, 'z_age', 'gender_male']

    for ef_spec in target_defs:
        ef_col = ef_spec['col']
        ef_label = ef_spec['label']

        if ef_col not in df.columns:
            if verbose:
                print(f"[WARN] Skipping {ef_label} ({ef_col}): column missing.")
            continue

        ef_df = df[base_cols + [ef_col]].dropna()
        if len(ef_df) < 50:
            if verbose:
                print(f"[WARN] Skipping {ef_label} ({ef_col}): insufficient data (N={len(ef_df)}).")
            continue

        ef_dir = output_dir / ef_col
        ef_dir.mkdir(parents=True, exist_ok=True)
        ef_results: Dict[str, PathModelResult] = {}

        for spec in PATH_SPECS:
            spec_dir = ef_dir / spec['name']
            spec_dir.mkdir(parents=True, exist_ok=True)

            spec_df = ef_df.copy()
            spec_df['ucla_male'] = spec_df['z_ucla'] * spec_df['gender_male']
            spec_df['ef_male'] = spec_df[ef_col] * spec_df['gender_male']
            spec_df['dass_male'] = spec_df[dass_col] * spec_df['gender_male']

            stage1_formula = spec['stage1_formula'](dass_col, ef_col)
            stage2_formula = spec['stage2_formula'](dass_col, ef_col)

            stage1_label = spec['stage1_label'].format(dass_label=dass_label, ef_label=ef_label)
            stage2_label = spec['stage2_label'].format(dass_label=dass_label, ef_label=ef_label)

            stage1_model = smf.ols(stage1_formula, data=spec_df).fit(cov_type='HC3')
            stage2_model = smf.ols(stage2_formula, data=spec_df).fit(cov_type='HC3')

            stage1_df = _summarize_model(stage1_model, stage1_label)
            stage2_df = _summarize_model(stage2_model, stage2_label)

            vif_stage1 = _compute_vif_table(stage1_model)
            vif_stage2 = _compute_vif_table(stage2_model)
            if not vif_stage1.empty:
                vif_stage1.to_csv(spec_dir / "vif_stage1.csv", index=False, encoding='utf-8-sig')
            if not vif_stage2.empty:
                vif_stage2.to_csv(spec_dir / "vif_stage2.csv", index=False, encoding='utf-8-sig')

            effects = []
            for path_info in spec['paths']:
                model_ref = stage1_model if path_info['model'] == 'stage1' else stage2_model
                term = path_info['term'].format(dass_col=dass_col, ef_col=ef_col)
                stats_g = _gender_effect_stats(model_ref, term)
                effects.append({
                    'path': path_info['path'].format(dass_label=dass_label, ef_label=ef_label),
                    'female_beta': stats_g['female_beta'],
                    'female_se': stats_g['female_se'],
                    'male_beta': stats_g['male_beta'],
                    'male_se': stats_g['male_se'],
                })

            effects_df = pd.DataFrame(effects)

            stage1_df.to_csv(spec_dir / "stage1_model.csv", index=False, encoding='utf-8-sig')
            stage2_df.to_csv(spec_dir / "stage2_model.csv", index=False, encoding='utf-8-sig')
            effects_df.to_csv(spec_dir / "gender_path_effects.csv", index=False, encoding='utf-8-sig')

            fit_stats = {
                'stage1_aic': stage1_model.aic,
                'stage1_bic': stage1_model.bic,
                'stage2_aic': stage2_model.aic,
                'stage2_bic': stage2_model.bic,
            }

            if SEMOPY_AVAILABLE and spec.get('sem_model'):
                try:
                    sem_spec = spec['sem_model'](dass_col, ef_col)
                    sem_model = SemopyModel(sem_spec)
                    sem_model.fit(spec_df, obj='MLW')
                    sem_stats = calc_stats(sem_model)
                    fit_stats.update({
                        'sem_cfi': sem_stats.get('CFI'),
                        'sem_tli': sem_stats.get('TLI'),
                        'sem_rmsea': sem_stats.get('RMSEA'),
                        'sem_aic': sem_stats.get('AIC'),
                        'sem_bic': sem_stats.get('BIC'),
                    })
                except Exception:
                    fit_stats.update({
                        'sem_cfi': np.nan,
                        'sem_tli': np.nan,
                        'sem_rmsea': np.nan,
                        'sem_aic': np.nan,
                        'sem_bic': np.nan,
                    })
            else:
                fit_stats.update({
                    'sem_cfi': np.nan,
                    'sem_tli': np.nan,
                    'sem_rmsea': np.nan,
                    'sem_aic': np.nan,
                    'sem_bic': np.nan,
                })

            sobel_rows = []
            indirect_cfg = spec.get('indirect')
            if indirect_cfg:
                term_a = indirect_cfg['term_a'].format(dass_col=dass_col, ef_col=ef_col)
                term_b = indirect_cfg['term_b'].format(dass_col=dass_col, ef_col=ef_col)
                model_a = stage1_model if indirect_cfg['stage_a'] == 'stage1' else stage2_model
                model_b = stage1_model if indirect_cfg['stage_b'] == 'stage1' else stage2_model

                stats_a = _gender_effect_stats(model_a, term_a)
                stats_b = _gender_effect_stats(model_b, term_b)

                female_sobel = _sobel_test(
                    stats_a['female_beta'], stats_a['female_se'],
                    stats_b['female_beta'], stats_b['female_se']
                )
                male_sobel = _sobel_test(
                    stats_a['male_beta'], stats_a['male_se'],
                    stats_b['male_beta'], stats_b['male_se']
                )
                sobel_rows.extend([
                    {
                        'gender': 'Female',
                        'z': female_sobel['z'],
                        'p': female_sobel['p'],
                    },
                    {
                        'gender': 'Male',
                        'z': male_sobel['z'],
                        'p': male_sobel['p'],
                    },
                ])
            sobel_df = pd.DataFrame(sobel_rows)
            if not sobel_df.empty:
                sobel_df.to_csv(spec_dir / "sobel_tests.csv", index=False, encoding='utf-8-sig')

            bootstrap_df = pd.DataFrame()
            if indirect_cfg:
                term_a = indirect_cfg['term_a'].format(dass_col=dass_col, ef_col=ef_col)
                term_b = indirect_cfg['term_b'].format(dass_col=dass_col, ef_col=ef_col)
                bootstrap_df = _bootstrap_indirect_effects(
                    spec_df,
                    stage1_formula,
                    stage2_formula,
                    term_a,
                    term_b,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
                bootstrap_df.to_csv(spec_dir / "bootstrap_indirect.csv", index=False, encoding='utf-8-sig')

            diagnostics = {}
            for label, model in [('stage1', stage1_model), ('stage2', stage2_model)]:
                resid = model.resid
                bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
                try:
                    shapiro_p = stats.shapiro(resid)[1]
                except Exception:
                    shapiro_p = np.nan
                diagnostics[f'{label}_bp_p'] = bp_p
                diagnostics[f'{label}_shapiro_p'] = shapiro_p
            stage1_vif_summary = _summarize_vif(vif_stage1)
            stage2_vif_summary = _summarize_vif(vif_stage2)
            diagnostics.update({
                'stage1_vif_max': stage1_vif_summary['vif_max'],
                'stage1_vif_mean': stage1_vif_summary['vif_mean'],
                'stage2_vif_max': stage2_vif_summary['vif_max'],
                'stage2_vif_mean': stage2_vif_summary['vif_mean'],
            })

            if verbose:
                print(f"\n[{spec['description'].format(dass_label=dass_label, ef_label=ef_label)}]")
                print(f"  Outcome: {ef_label} (N={len(spec_df)})")
                for row in effects:
                    female = row['female_beta']
                    male = row['male_beta']
                    print(f"  {row['path']}: female={female:.3f}, male={male:.3f}")

            ef_results[spec['name']] = PathModelResult(
                stage1=stage1_df,
                stage2=stage2_df,
                path_effects=effects_df,
                fit_stats=fit_stats,
                bootstrap=bootstrap_df,
                sobel=sobel_df,
                diagnostics=diagnostics,
                description=spec['description'].format(dass_label=dass_label, ef_label=ef_label),
                ef_label=ef_label,
                ef_col=ef_col,
            )

        if ef_results:
            all_results[ef_col] = ef_results

    return all_results
