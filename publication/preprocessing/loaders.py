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

from .constants import (
    RESULTS_DIR,
    ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN,
    DEFAULT_RT_MAX,
    PRP_RT_MAX,
    STROOP_RT_MAX,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
    MASTER_CACHE_PATH,
    STANDARDIZE_COLS,
    MALE_TOKENS_EXACT,
    FEMALE_TOKENS_EXACT,
    MALE_TOKENS_CONTAINS,
    FEMALE_TOKENS_CONTAINS,
    PARTICIPANT_ID_ALIASES,
)


def ensure_participant_id(df: pd.DataFrame, warn_threshold: float = 1.0) -> pd.DataFrame:
    """
    Ensure there is exactly one 'participant_id' column.
    Prefers an existing participant_id column, otherwise renames common aliases.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    warn_threshold : float
        Warn if more than this percentage of participant_id values are NaN (default: 1%)

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized participant_id column
    """
    import warnings

    canonical = "participant_id"
    if canonical not in df.columns:
        for col in df.columns:
            if col in PARTICIPANT_ID_ALIASES and col != canonical:
                df = df.rename(columns={col: canonical})
                break
    if canonical not in df.columns:
        raise KeyError("No participant id column found in dataframe.")

    # Validate that participant_id column has actual data
    missing_count = df[canonical].isna().sum()
    missing_pct = missing_count / len(df) * 100 if len(df) > 0 else 0

    if missing_pct > warn_threshold:
        warnings.warn(
            f"participant_id column has {missing_pct:.1f}% missing values ({missing_count}/{len(df)} rows). "
            "This may cause silent data loss in downstream analyses.",
            UserWarning
        )

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
    """Load and normalize participant info."""
    df = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding="utf-8")

    df = ensure_participant_id(df)
    df["gender"] = normalize_gender_series(df["gender"])

    return df[['participant_id', 'age', 'gender', 'education']]

def load_ucla_scores():
    """Load UCLA loneliness scores."""
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding="utf-8")

    surveys = ensure_participant_id(surveys)

    # Get UCLA data
    if 'surveyName' in surveys.columns:
        ucla_data = surveys[surveys['surveyName'].str.lower() == 'ucla'].copy()
    else:
        raise KeyError("No survey name column found")

    # 점수가 중복 기록될 경우(재시행) 합산 대신 한 건만 사용
    ucla_data['score'] = pd.to_numeric(ucla_data['score'], errors='coerce')
    ucla_data = ucla_data.dropna(subset=['score'])
    # CSV 순서 기준 최신 기록 유지
    ucla_data = ucla_data.drop_duplicates(subset=['participant_id'], keep='last')

    ucla_scores = ucla_data[['participant_id', 'score']].rename(columns={'score': 'ucla_total'})
    ucla_scores.columns = ['participant_id', 'ucla_total']

    return ucla_scores

def load_dass_scores():
    """Load DASS component scores (Depression, Anxiety, Stress)."""
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding="utf-8")

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
        # UCLA와 동일하게 최신 기록 유지
        for col in ['score_D', 'score_A', 'score_S']:
            dass_data[col] = pd.to_numeric(dass_data[col], errors='coerce')

        # 시간 기준 정렬 (CSV 순서가 섞여 있을 수 있으므로)
        sort_cols = ['participant_id']
        if 'createdAt' in dass_data.columns:
            sort_cols.append('createdAt')
        elif 'created_at' in dass_data.columns:
            sort_cols.append('created_at')
        dass_data = dass_data.sort_values(sort_cols)

        dass_summary = dass_data.drop_duplicates(
            subset=['participant_id'],
            keep='last'
        )[['participant_id', 'score_D', 'score_A', 'score_S']].copy()
        dass_summary.columns = ['participant_id', 'dass_depression', 'dass_anxiety', 'dass_stress']
    else:
        # Fallback: compute from items
        if 'score' not in dass_data.columns:
            raise KeyError("DASS survey responses missing 'score' column; cannot compute component scores.")
        dass_data['score'] = pd.to_numeric(dass_data['score'], errors='coerce')
        dass_data = dass_data.dropna(subset=['score'])
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


def load_survey_items():
    """
    Load raw UCLA/DASS item responses from 2_surveys_results.csv and return wide format.
    Columns are renamed to ucla_1..ucla_20 and dass_1..dass_21 when available.
    """
    surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding="utf-8")
    surveys = ensure_participant_id(surveys)

    def _extract_items(df: pd.DataFrame, prefix: str, n_items: int) -> pd.DataFrame:
        cols = [f"q{i}" for i in range(1, n_items + 1) if f"q{i}" in df.columns]
        if not cols:
            return pd.DataFrame(columns=["participant_id"])
        tmp = df[["participant_id"] + cols].copy()
        # Keep first non-null per participant
        tmp = tmp.groupby("participant_id").first().reset_index()
        rename_map = {c: f"{prefix}_{c.lstrip('q')}" for c in cols}
        tmp = tmp.rename(columns=rename_map)
        # Numeric coercion
        for c in rename_map.values():
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        return tmp

    ucla_df = pd.DataFrame(columns=["participant_id"])
    dass_df = pd.DataFrame(columns=["participant_id"])

    if "surveyName" in surveys.columns:
        ucla_df = surveys[surveys["surveyName"].str.lower() == "ucla"].copy()
        dass_df = surveys[surveys["surveyName"].str.lower().str.contains("dass")].copy()

    ucla_items = _extract_items(ucla_df, "ucla", 20)
    dass_items = _extract_items(dass_df, "dass", 21)

    survey_items = ucla_items.merge(dass_items, on="participant_id", how="outer")
    return survey_items

def load_wcst_summary(rt_min: int = DEFAULT_RT_MIN, rt_max: Optional[int] = None):
    """Load WCST summary metrics including PE rate.

    Parameters
    ----------
    rt_min : int
        Minimum RT threshold (default: 100ms, removes anticipatory responses)
    rt_max : int or None
        Maximum RT threshold (default: None, no upper bound since WCST has no timeout)
    """
    wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding="utf-8")

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

    # Apply RT filtering (WCST has no timeout, so default rt_max=None)
    before_filter = len(wcst_trials)
    mask = wcst_trials[rt_col] > rt_min
    if rt_max is not None:
        mask = mask & (wcst_trials[rt_col] < rt_max)
    wcst_trials = wcst_trials[mask].copy()
    after_filter = len(wcst_trials)
    if before_filter != after_filter:
        print(f"  WCST RT filtering (>{rt_min}ms): {before_filter} -> {after_filter} trials ({(before_filter - after_filter) / before_filter * 100:.1f}% removed)")

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
    """Load PRP summary metrics including bottleneck effect."""
    prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding="utf-8")

    prp_trials = ensure_participant_id(prp_trials)

    # Normalize RT column name - prefer _ms columns (have more data)
    # But backfill NaN in _ms with legacy values (early participants)
    rt_col = 't2_rt_ms' if 't2_rt_ms' in prp_trials.columns else ('t2_rt' if 't2_rt' in prp_trials.columns else None)
    if rt_col is None:
        raise KeyError("PRP trials missing T2 RT column ('t2_rt' or 't2_rt_ms').")
    if rt_col != 't2_rt':
        # Fill NaN in t2_rt_ms with legacy t2_rt values (early participants)
        if 't2_rt' in prp_trials.columns:
            prp_trials[rt_col] = prp_trials[rt_col].fillna(prp_trials['t2_rt'])
            prp_trials = prp_trials.drop(columns=['t2_rt'])
        prp_trials = prp_trials.rename(columns={rt_col: 't2_rt'})

    # Normalize SOA column name - prefer soa_nominal_ms (has more data)
    # But backfill NaN in _ms with legacy values (early participants)
    soa_col = None
    for cand in ['soa_nominal_ms', 'soa_ms', 'soa']:
        if cand in prp_trials.columns:
            soa_col = cand
            break
    if soa_col is None:
        raise KeyError("PRP trials missing SOA column ('soa', 'soa_ms', or 'soa_nominal_ms').")
    if soa_col != 'soa':
        # Fill NaN in soa_nominal_ms with legacy soa values (early participants)
        if 'soa' in prp_trials.columns:
            prp_trials[soa_col] = prp_trials[soa_col].fillna(prp_trials['soa'])
            prp_trials = prp_trials.drop(columns=['soa'])
        prp_trials = prp_trials.rename(columns={soa_col: 'soa'})

    # Normalize correctness/timeout columns to avoid silent NaN-based row drops
    prp_trials['t1_correct'] = prp_trials['t1_correct'].fillna(False).astype(bool)
    if 't2_correct' in prp_trials.columns:
        prp_trials['t2_correct'] = prp_trials['t2_correct'].fillna(False).astype(bool)
    else:
        # Legacy exports omitted t2_correct entirely; assume entries were valid
        prp_trials['t2_correct'] = True
    prp_trials['t2_timeout'] = prp_trials.get('t2_timeout', False)
    if isinstance(prp_trials['t2_timeout'], pd.Series):
        prp_trials['t2_timeout'] = prp_trials['t2_timeout'].fillna(False).astype(bool)

    # Filter valid trials for RT (정답 + 시간내 반응 + 합리적 RT 범위)
    prp_rt = prp_trials[
        (prp_trials['t1_correct'] == True) &
        (prp_trials['t2_correct'] == True) &
        (prp_trials['t2_timeout'] == False) &
        (prp_trials['t2_rt'] > DEFAULT_RT_MIN) &
        (prp_trials['t2_rt'] < PRP_RT_MAX)
    ].copy()

    # Bin SOA
    def bin_soa(soa):
        if soa <= DEFAULT_SOA_SHORT:
            return 'short'
        elif soa >= DEFAULT_SOA_LONG:
            return 'long'
        else:
            return 'other'

    prp_rt['soa_bin'] = prp_rt['soa'].apply(bin_soa)
    prp_rt = prp_rt[prp_rt['soa_bin'].isin(['short', 'long'])].copy()

    # Calculate by SOA
    prp_summary = prp_rt.groupby(['participant_id', 'soa_bin']).agg(
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
    """Load Stroop summary metrics including interference."""
    stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding="utf-8")

    stroop_trials = ensure_participant_id(stroop_trials)

    # Determine RT column name (prefer rt_ms which has more data)
    # But backfill NaN in _ms with legacy values (early participants)
    rt_col = 'rt_ms' if 'rt_ms' in stroop_trials.columns else ('rt' if 'rt' in stroop_trials.columns else None)
    if rt_col is None:
        raise KeyError("Stroop trials missing RT column ('rt' or 'rt_ms').")
    if rt_col != 'rt':
        # Fill NaN in rt_ms with legacy rt values (early participants)
        if 'rt' in stroop_trials.columns:
            stroop_trials[rt_col] = stroop_trials[rt_col].fillna(stroop_trials['rt'])
            stroop_trials = stroop_trials.drop(columns=['rt'])
        stroop_trials = stroop_trials.rename(columns={rt_col: 'rt'})
        rt_col = 'rt'

    # Determine condition column
    cond_col = 'type' if 'type' in stroop_trials.columns else ('condition' if 'condition' in stroop_trials.columns else ('cond' if 'cond' in stroop_trials.columns else None))
    if cond_col is None:
        raise KeyError("Stroop trials missing condition column ('type', 'condition', or 'cond').")

    # Accuracy: 모든 trial 포함(타임아웃은 False로 집계되므로 분모 포함)
    stroop_trials['correct'] = stroop_trials['correct'].fillna(False).astype(bool)
    if 'timeout' in stroop_trials.columns:
        stroop_trials['timeout'] = stroop_trials['timeout'].fillna(False).astype(bool)
    acc_summary = stroop_trials.groupby(['participant_id', cond_col]).agg(
        accuracy=('correct', 'mean')
    ).reset_index()

    # RT: 정답 + 시간내 반응 + 합리적 RT 범위
    rt_trials = stroop_trials[
        ((stroop_trials['timeout'] == False) if 'timeout' in stroop_trials.columns else True) &
        (stroop_trials['correct'] == True) &
        (stroop_trials[rt_col] > DEFAULT_RT_MIN) &
        (stroop_trials[rt_col] < STROOP_RT_MAX)
    ].copy()

    rt_summary = rt_trials.groupby(['participant_id', cond_col]).agg(
        rt_mean=(rt_col, 'mean')
    ).reset_index()

    stroop_summary = acc_summary.merge(rt_summary, on=['participant_id', cond_col], how='left')

    # Pivot to wide
    stroop_wide = stroop_summary.pivot(index='participant_id', columns=cond_col, values=['rt_mean', 'accuracy'])
    stroop_wide.columns = ['_'.join(col).rstrip('_') for col in stroop_wide.columns.values]
    stroop_wide = stroop_wide.reset_index()

    # Calculate interference
    if 'rt_mean_incongruent' in stroop_wide.columns and 'rt_mean_congruent' in stroop_wide.columns:
        stroop_wide['stroop_interference'] = stroop_wide['rt_mean_incongruent'] - stroop_wide['rt_mean_congruent']

    return stroop_wide

def load_master_dataset(
    use_cache: bool = True,
    force_rebuild: bool = False,
    cache_path: Path = MASTER_CACHE_PATH,
    add_standardized: bool = True,
    merge_cognitive_summary: bool = True
) -> pd.DataFrame:
    """
    Build or load a unified master dataset shared by all analyses.

    Standardized definitions:
      - Gender: normalized via normalize_gender_series; gender_male = 1 (male), 0 (female).
      - PRP bottleneck: mean T2 RT (short SOA <= DEFAULT_SOA_SHORT) minus mean T2 RT (long SOA >= DEFAULT_SOA_LONG);
        include trials with t1_correct == True, t2_correct == True, timeout == False, DEFAULT_RT_MIN < t2_rt < PRP_RT_MAX.
      - Stroop interference: mean RT (incongruent) - mean RT (congruent); RT는 정답·시간내 반응만 사용, RT 범위 [DEFAULT_RT_MIN, STROOP_RT_MAX],
        정확도는 timeout 포함 전체 분모.
      - WCST perseverative error rate: proportion of trials flagged isPE per participant.

    Parameters
    ----------
    use_cache : bool
        If True, load cached parquet when available.
    force_rebuild : bool
        If True, ignore cache and rebuild from source CSVs.
    cache_path : Path
        Location to store/read the master parquet.
    add_standardized : bool
        If True, add z-scored versions of STANDARDIZE_COLS (prefixed with 'z_').
    merge_cognitive_summary : bool
        If True, merge cognitive summary file (3_cognitive_tests_summary.csv) when present.
    """
    cache_path = Path(cache_path)

    if use_cache and not force_rebuild:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        csv_fallback = cache_path.with_suffix(".csv")
        if csv_fallback.exists():
            return pd.read_csv(csv_fallback)

    participants = load_participants()
    participants["gender_normalized"] = normalize_gender_series(participants["gender"])
    participants["gender_male"] = participants["gender_normalized"].map({"male": 1, "female": 0})

    def _log_merge(df: pd.DataFrame, df_name: str, merge_df: pd.DataFrame, merge_name: str, key: str = "participant_id") -> pd.DataFrame:
        """Merge with audit logging to catch silent row drops."""
        before = len(df)
        merged = df.merge(merge_df, on=key, how="left")
        after = len(merged)
        print(f"  Merge {df_name} + {merge_name}: {before} -> {after} rows (left join on '{key}')")
        return merged

    ucla = load_ucla_scores().rename(columns={"ucla_total": "ucla_score"})
    if "ucla_total" not in ucla.columns and "ucla_score" in ucla.columns:
        ucla["ucla_total"] = ucla["ucla_score"]

    dass = load_dass_scores()
    survey_items = load_survey_items()
    wcst = load_wcst_summary()
    prp = load_prp_summary()
    stroop = load_stroop_summary()

    master = participants.merge(ucla, on="participant_id", how="inner")
    print(f"  Merge participants + UCLA (inner): {len(participants)} -> {len(master)} rows")
    master = _log_merge(master, "master", dass, "dass")
    if not survey_items.empty:
        master = _log_merge(master, "master", survey_items, "survey_items")
    master = _log_merge(master, "master", wcst, "wcst")
    master = _log_merge(master, "master", prp, "prp")
    master = _log_merge(master, "master", stroop, "stroop")

    if merge_cognitive_summary:
        summary_path = RESULTS_DIR / "3_cognitive_tests_summary.csv"
        if summary_path.exists():
            cognitive = pd.read_csv(summary_path, encoding="utf-8")
            cognitive = ensure_participant_id(cognitive)
            cognitive = cognitive.sort_values("participant_id").drop_duplicates(subset=["participant_id"])
            master = _log_merge(master, "master", cognitive, "cognitive_summary")

    if add_standardized:
        for col in STANDARDIZE_COLS:
            if col in master.columns:
                std_val = master[col].std()
                master[f"z_{col}"] = (master[col] - master[col].mean()) / std_val if std_val else 0
        # Convenience alias for legacy code
        if 'z_ucla_score' in master.columns and 'z_ucla' not in master.columns:
            master['z_ucla'] = master['z_ucla_score']

    # Cache for fast re-use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        master.to_parquet(cache_path, index=False)
    except Exception:
        # Parquet engine might be missing; fall back to CSV
        fallback = cache_path.with_suffix(".csv")
        master.to_csv(fallback, index=False)
        cache_path = fallback

    print(
        "  Master dataset built: N={}, male={}, female={}, unknown={}".format(
            len(master),
            int(master["gender_male"].fillna(0).sum()),
            int((master["gender_male"] == 0).sum()),
            int(master["gender_male"].isna().sum()),
        )
    )

    return master

def load_exgaussian_params(task='stroop'):
    """
    Load Ex-Gaussian parameters

    .. deprecated::
        This function is deprecated and only used by legacy archive scripts.
        Use analysis.utils.exgaussian.fit_exgaussian_by_participant() instead.

    Parameters
    ----------
    task : str
        'stroop' or 'prp'
    """
    import warnings
    warnings.warn(
        "load_exgaussian_params is deprecated. Use analysis.utils.exgaussian.fit_exgaussian_by_participant() instead.",
        DeprecationWarning,
        stacklevel=2
    )
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
