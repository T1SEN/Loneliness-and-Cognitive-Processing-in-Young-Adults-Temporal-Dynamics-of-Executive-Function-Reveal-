"""
Common data loading utilities for all analysis scripts
Handles column naming inconsistencies and data preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
import re
from typing import Optional

RESULTS_DIR = Path("results")


MALE_TOKENS_EXACT = {"m", "male", "man", "men", "boy", "boys"}
FEMALE_TOKENS_EXACT = {"f", "female", "woman", "women", "girl", "girls"}
MALE_TOKENS_CONTAINS = {"남성", "남자", "남", "남학생"}
FEMALE_TOKENS_CONTAINS = {"여성", "여자", "여", "여학생"}

PARTICIPANT_ID_ALIASES = {"participant_id", "participantId", "participantid"}


def ensure_participant_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is exactly one 'participant_id' column.
    Prefers an existing participant_id column, otherwise renames common aliases.
    """
    canonical = "participant_id"
    if canonical not in df.columns:
        for col in df.columns:
            if col in PARTICIPANT_ID_ALIASES and col != canonical:
                df = df.rename(columns={col: canonical})
                break
    if canonical not in df.columns:
        raise KeyError("No participant id column found in dataframe.")

    aliases = [col for col in df.columns if col in PARTICIPANT_ID_ALIASES and col != canonical]
    if aliases:
        df = df.drop(columns=aliases)
    return df


def _normalize_gender_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^a-z\u3131-\u318e\uac00-\ud7a3]", "", cleaned)
    return cleaned


def normalize_gender_value(value: object) -> Optional[str]:
    """
    Normalize arbitrary gender text (Korean/English) to 'male'/'female'.
    Returns None if the value cannot be mapped.
    """
    token = _normalize_gender_string(value)
    if not token:
        return None
    if token in FEMALE_TOKENS_EXACT or any(t and t in token for t in FEMALE_TOKENS_CONTAINS):
        return "female"
    if token in MALE_TOKENS_EXACT or any(t and t in token for t in MALE_TOKENS_CONTAINS):
        return "male"
    return None


def normalize_gender_series(series: pd.Series) -> pd.Series:
    mapped = series.apply(normalize_gender_value)
    return pd.Series(mapped, index=series.index, dtype="object")

def load_participants():
    """Load and normalize participant info"""
    df = pd.read_csv(RESULTS_DIR / "1_participants_info.csv")

    df = ensure_participant_id(df)
    df["gender"] = normalize_gender_series(df["gender"])

    return df[['participant_id', 'age', 'gender', 'education']]

def load_ucla_scores():
    """Load UCLA loneliness scores"""
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")

    surveys = ensure_participant_id(surveys)

    # Get UCLA data
    if 'surveyName' in surveys.columns:
        ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
    else:
        raise KeyError("No survey name column found")

    ucla_scores = ucla_data.groupby('participant_id')['score'].sum().reset_index()
    ucla_scores.columns = ['participant_id', 'ucla_total']

    return ucla_scores

def load_dass_scores():
    """Load DASS component scores (Depression, Anxiety, Stress)"""
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv")

    surveys = ensure_participant_id(surveys)

    # Get DASS data
    if 'surveyName' in surveys.columns:
        dass_data = surveys[surveys['surveyName'].str.lower().str.contains('dass')].copy()
    else:
        return pd.DataFrame(columns=['participant_id', 'dass_depression', 'dass_anxiety', 'dass_stress'])

    if len(dass_data) == 0:
        return pd.DataFrame(columns=['participant_id', 'dass_depression', 'dass_anxiety', 'dass_stress'])

    # Try to use pre-computed scores if available
    if all(col in dass_data.columns for col in ['score_D', 'score_A', 'score_S']):
        dass_summary = dass_data.groupby('participant_id').agg({
            'score_D': 'first',
            'score_A': 'first',
            'score_S': 'first'
        }).reset_index()
        dass_summary.columns = ['participant_id', 'dass_depression', 'dass_anxiety', 'dass_stress']
    else:
        # Fallback: compute from items
        dass_scores = dass_data.groupby(['participant_id', 'questionText'])['score'].sum().unstack(fill_value=0)

        dep_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ['meaningless', 'nothing', 'enthused', 'worth', 'positive', 'initiative', 'future'])]
        anx_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ['breathing', 'trembling', 'worried', 'panic', 'heart', 'scared', 'dry'])]
        stress_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ['wind down', 'over-react', 'nervous', 'agitated', 'relax', 'intolerant', 'touchy'])]

        dass_summary = pd.DataFrame()
        dass_summary['participant_id'] = dass_scores.index
        dass_summary['dass_depression'] = dass_scores[dep_items].sum(axis=1).values if dep_items else 0
        dass_summary['dass_anxiety'] = dass_scores[anx_items].sum(axis=1).values if anx_items else 0
        dass_summary['dass_stress'] = dass_scores[stress_items].sum(axis=1).values if stress_items else 0
        dass_summary = dass_summary.reset_index(drop=True)

    return dass_summary

def load_wcst_summary():
    """Load WCST summary metrics including PE rate"""
    wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv")

    wcst_trials = ensure_participant_id(wcst_trials)

    # Parse extra field for PE
    def _parse_wcst_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except (ValueError, SyntaxError):
            return {}

    wcst_trials['extra_dict'] = wcst_trials['extra'].apply(_parse_wcst_extra)
    wcst_trials['is_pe'] = wcst_trials['extra_dict'].apply(lambda x: x.get('isPE', False))

    # Determine RT column
    rt_col = 'rt_ms' if 'rt_ms' in wcst_trials.columns else 'reactionTimeMs'

    wcst_summary = wcst_trials.groupby('participant_id').agg(
        pe_count=('is_pe', 'sum'),
        total_trials=('is_pe', 'count'),
        wcst_accuracy=('correct', lambda x: (x.sum() / len(x)) * 100),
        wcst_mean_rt=(rt_col, 'mean'),
        wcst_sd_rt=(rt_col, 'std')
    ).reset_index()
    wcst_summary['pe_rate'] = (wcst_summary['pe_count'] / wcst_summary['total_trials']) * 100

    return wcst_summary

def load_prp_summary():
    """Load PRP summary metrics including bottleneck effect"""
    prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv")

    prp_trials = ensure_participant_id(prp_trials)

    # Filter valid trials
    prp_trials = prp_trials[
        (prp_trials['t1_correct'] == True) &
        (prp_trials['t2_rt'] > 0) &
        (prp_trials['t2_rt'] < 5000)  # Remove outliers
    ].copy()

    # Bin SOA
    def bin_soa(soa):
        if soa <= 150:
            return 'short'
        elif soa >= 1200:
            return 'long'
        else:
            return 'other'

    prp_trials['soa_bin'] = prp_trials['soa'].apply(bin_soa)
    prp_trials = prp_trials[prp_trials['soa_bin'].isin(['short', 'long'])].copy()

    # Calculate by SOA
    prp_summary = prp_trials.groupby(['participant_id', 'soa_bin']).agg(
        t2_rt_mean=('t2_rt', 'mean'),
        t2_rt_sd=('t2_rt', 'std'),
        n_trials=('t2_rt', 'count')
    ).reset_index()

    # Pivot to wide format
    prp_wide = prp_summary.pivot(index='participant_id', columns='soa_bin', values=['t2_rt_mean', 't2_rt_sd'])
    prp_wide.columns = ['_'.join(col).rstrip('_') for col in prp_wide.columns.values]
    prp_wide = prp_wide.reset_index()

    # Calculate bottleneck effect
    if 't2_rt_mean_short' in prp_wide.columns and 't2_rt_mean_long' in prp_wide.columns:
        prp_wide['prp_bottleneck'] = prp_wide['t2_rt_mean_short'] - prp_wide['t2_rt_mean_long']

    return prp_wide

def load_stroop_summary():
    """Load Stroop summary metrics including interference"""
    stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv")

    stroop_trials = ensure_participant_id(stroop_trials)

    # Filter valid trials
    stroop_trials = stroop_trials[stroop_trials['timeout'] == False].copy()

    # Calculate by condition
    stroop_summary = stroop_trials.groupby(['participant_id', 'type']).agg(
        rt_mean=('rt', 'mean'),
        accuracy=('correct', 'mean')
    ).reset_index()

    # Pivot to wide
    stroop_wide = stroop_summary.pivot(index='participant_id', columns='type', values=['rt_mean', 'accuracy'])
    stroop_wide.columns = ['_'.join(col).rstrip('_') for col in stroop_wide.columns.values]
    stroop_wide = stroop_wide.reset_index()

    # Calculate interference
    if 'rt_mean_incongruent' in stroop_wide.columns and 'rt_mean_congruent' in stroop_wide.columns:
        stroop_wide['stroop_interference'] = stroop_wide['rt_mean_incongruent'] - stroop_wide['rt_mean_congruent']

    return stroop_wide

def load_master_dataset():
    """Load complete master dataset with all measures"""
    # Load all components
    participants = load_participants()
    ucla = load_ucla_scores()
    dass = load_dass_scores()
    wcst = load_wcst_summary()
    prp = load_prp_summary()
    stroop = load_stroop_summary()

    # Merge all
    master = participants.merge(ucla, on='participant_id', how='inner')
    master = master.merge(dass, on='participant_id', how='left')
    master = master.merge(wcst, on='participant_id', how='left')
    master = master.merge(prp, on='participant_id', how='left')
    master = master.merge(stroop, on='participant_id', how='left')

    master["gender_normalized"] = normalize_gender_series(master["gender"])
    master["gender_male"] = master["gender_normalized"].map({"male": 1, "female": 0})
    print(
        "  Gender distribution: Male={}, Female={}, Unknown={}".format(
            int(master["gender_male"].fillna(0).sum()),
            int((master["gender_male"] == 0).sum()),
            int(master["gender_male"].isna().sum()),
        )
    )

    # Standardize predictors
    standardize_mapping = {
        'ucla_total': 'z_ucla',
        'dass_depression': 'z_depression',
        'dass_anxiety': 'z_anxiety',
        'dass_stress': 'z_stress',
        'age': 'z_age'
    }
    for col, z_col in standardize_mapping.items():
        if col in master.columns:
            master[z_col] = (master[col] - master[col].mean()) / master[col].std()

    return master

def load_exgaussian_params(task='stroop'):
    """
    Load Ex-Gaussian parameters
    task: 'stroop' or 'prp'
    """
    if task == 'stroop':
        file_path = "results/analysis_outputs/mechanism_analysis/exgaussian/exgaussian_parameters.csv"
    elif task == 'prp':
        file_path = "results/analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv"
    else:
        raise ValueError("task must be 'stroop' or 'prp'")

    df = pd.read_csv(file_path)

    # Clean BOM if present
    if df.columns[0].startswith('\ufeff'):
        df.columns = [df.columns[0].replace('\ufeff', '')] + list(df.columns[1:])

    if 'participant_id' in df.columns and df['participant_id'].dtype == 'O':
        df['participant_id'] = df['participant_id'].str.replace('\ufeff', '')

    return df
