"""
Task-specific QC criteria and participant filtering.

각 인지검사(Stroop, PRP, WCST)별 품질 기준 및 유효 참가자 필터링 로직.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Optional, Tuple

import pandas as pd

from .constants import (
    RAW_DIR,
    DEFAULT_RT_MIN,
    PRP_RT_MAX,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
    VALID_TASKS,
)

# Windows Unicode 출력 지원
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# QC Criteria Dataclasses
# =============================================================================

@dataclass
class SurveyQCCriteria:
    """설문 품질 기준."""
    ucla_score_required: bool = True
    dass_subscales_required: bool = True
    gender_required: bool = True
    date_filter: Optional[pd.Timestamp] = field(
        default_factory=lambda: pd.Timestamp('2025-09-01', tz='UTC')
    )


@dataclass
class StroopQCCriteria:
    """Stroop 과제 품질 기준."""
    min_accuracy: Optional[float] = None
    max_mean_rt: Optional[float] = None
    require_metrics: bool = True     # mrt_cong/mrt_incong/stroop_effect/accuracy 필수


@dataclass
class PRPQCCriteria:
    """PRP 과제 품질 기준."""
    n_trials_required: int = 120     # 총 시행 수
    min_accuracy: Optional[float] = None
    require_valid_trials: bool = True  # 유효 trial 1건 이상 필수
    require_metrics: bool = True     # mrt_t2/rt2_soa_50/rt2_soa_1200 필수


@dataclass
class WCSTQCCriteria:
    """WCST 과제 품질 기준."""
    max_total_errors: int = 60
    max_perseverative_responses: int = 60
    min_completed_categories: int = 1
    require_metrics: bool = True     # perseverativeErrorCount 필수


# =============================================================================
# Survey Filtering
# =============================================================================

def get_survey_valid_participants(
    data_dir: Path = None,
    criteria: SurveyQCCriteria = None,
    verbose: bool = False,
) -> Set[str]:
    """설문 완료 + 성별 + 날짜 기준 통과자 반환.

    Args:
        data_dir: raw 데이터 디렉토리 (기본: RAW_DIR)
        criteria: 설문 QC 기준 (기본: SurveyQCCriteria())
        verbose: 상세 로그 출력 여부

    Returns:
        유효 참가자 ID 집합
    """
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = SurveyQCCriteria()

    valid_ids: Set[str] = set()

    # 1. UCLA + DASS 설문 완료 확인
    surveys_path = data_dir / '2_surveys_results.csv'
    if not surveys_path.exists():
        if verbose:
            print(f"[WARN] 설문 파일 없음: {surveys_path}")
        return valid_ids

    surveys = pd.read_csv(surveys_path, encoding='utf-8-sig')
    surveys['surveyName'] = surveys['surveyName'].str.lower()

    # UCLA: score가 non-null
    ucla_df = surveys[surveys['surveyName'] == 'ucla']
    if criteria.ucla_score_required:
        ucla_valid = set(ucla_df[ucla_df['score'].notna()]['participantId'].unique())
    else:
        ucla_valid = set(ucla_df['participantId'].unique())

    # DASS: score_D, score_A, score_S 모두 non-null
    dass_df = surveys[surveys['surveyName'].str.contains('dass', na=False)]
    if criteria.dass_subscales_required:
        dass_valid = set(dass_df[
            (dass_df['score_D'].notna()) &
            (dass_df['score_A'].notna()) &
            (dass_df['score_S'].notna())
        ]['participantId'].unique())
    else:
        dass_valid = set(dass_df['participantId'].unique())

    valid_ids = ucla_valid & dass_valid

    if verbose:
        print(f"설문 완료: {len(valid_ids)}명 (UCLA: {len(ucla_valid)}, DASS: {len(dass_valid)})")

    # 2. 성별 누락 제외
    if criteria.gender_required:
        participants_path = data_dir / '1_participants_info.csv'
        if participants_path.exists():
            participants_df = pd.read_csv(participants_path, encoding='utf-8-sig')
            missing_gender_ids = set(
                participants_df[
                    participants_df['gender'].isna() |
                    (participants_df['gender'].astype(str).str.strip() == '')
                ]['participantId'].dropna()
            )
            to_remove = valid_ids & missing_gender_ids
            if to_remove and verbose:
                print(f"  [INFO] 성별 누락으로 제외: {len(to_remove)}명")
            valid_ids -= to_remove

    # 3. 날짜 필터
    if criteria.date_filter is not None:
        participants_path = data_dir / '1_participants_info.csv'
        if participants_path.exists():
            participants_df = pd.read_csv(participants_path, encoding='utf-8-sig')
            if 'createdAt' in participants_df.columns:
                participants_df['createdAt'] = pd.to_datetime(
                    participants_df['createdAt'], errors='coerce', utc=True
                )
                recent_ids = set(
                    participants_df[
                        participants_df['createdAt'] >= criteria.date_filter
                    ]['participantId'].dropna()
                )
                excluded = valid_ids - recent_ids
                if excluded and verbose:
                    print(f"  [INFO] 날짜 기준 미달 제외: {len(excluded)}명")
                valid_ids = valid_ids & recent_ids

    return valid_ids


# =============================================================================
# Stroop Filtering
# =============================================================================

def get_stroop_valid_participants(
    data_dir: Path = None,
    criteria: StroopQCCriteria = None,
    verbose: bool = False,
) -> Set[str]:
    """Stroop 과제 품질 기준 통과자 반환.

    Args:
        data_dir: raw 데이터 디렉토리 (기본: RAW_DIR)
        criteria: Stroop QC 기준 (기본: StroopQCCriteria())
        verbose: 상세 로그 출력 여부

    Returns:
        유효 참가자 ID 집합
    """
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = StroopQCCriteria()

    summary_path = data_dir / '3_cognitive_tests_summary.csv'
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] 인지과제 요약 파일 없음: {summary_path}")
        return set()

    summary_df = pd.read_csv(summary_path, encoding='utf-8-sig')
    summary_df['testName'] = summary_df['testName'].str.lower()

    stroop_df = summary_df[summary_df['testName'] == 'stroop'].copy()
    if stroop_df.empty:
        return set()

    # 필수 지표 존재 여부
    if criteria.require_metrics:
        mask = (
            stroop_df['mrt_cong'].notna() &
            stroop_df['mrt_incong'].notna() &
            stroop_df['stroop_effect'].notna() &
            stroop_df['accuracy'].notna()
        )
        stroop_df = stroop_df[mask]

    # 정확도 기준
    if criteria.min_accuracy is not None:
        stroop_df = stroop_df[stroop_df['accuracy'] >= criteria.min_accuracy]

    # 평균 RT 기준
    if criteria.max_mean_rt is not None:
        # 평균 RT = (mrt_cong + mrt_incong) / 2
        stroop_df = stroop_df[
            ((stroop_df['mrt_cong'] + stroop_df['mrt_incong']) / 2) <= criteria.max_mean_rt
        ]

    valid_ids = set(stroop_df['participantId'].unique())

    if verbose:
        all_stroop = set(summary_df[summary_df['testName'] == 'stroop']['participantId'].unique())
        excluded = all_stroop - valid_ids
        if excluded:
            print(f"  [INFO] Stroop QC 미달: {len(excluded)}명")

    return valid_ids


# =============================================================================
# PRP Filtering
# =============================================================================

def _get_prp_valid_trial_participants(
    data_dir: Path,
    verbose: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """PRP 유효 trial 보유 참가자 확인.

    Returns:
        (valid_ids, no_valid_ids): 유효 trial 있는/없는 참가자 ID 집합
    """
    prp_path = data_dir / '4a_prp_trials.csv'
    if not prp_path.exists():
        if verbose:
            print(f"[WARN] PRP trials 파일 없음: {prp_path}")
        return set(), set()

    prp_trials = pd.read_csv(prp_path, encoding='utf-8-sig')

    # Boolean 컬럼 정규화
    prp_trials['t1_correct'] = prp_trials['t1_correct'].fillna(False).astype(bool)
    prp_trials['t2_correct'] = prp_trials['t2_correct'].fillna(False).astype(bool)
    if 't2_timeout' in prp_trials.columns:
        prp_trials['t2_timeout'] = prp_trials['t2_timeout'].fillna(False).astype(bool)
    else:
        prp_trials['t2_timeout'] = False

    # RT/SOA 컬럼 결정
    rt_col = 't2_rt_ms' if 't2_rt_ms' in prp_trials.columns else (
        't2_rt' if 't2_rt' in prp_trials.columns else None
    )
    soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in prp_trials.columns else (
        'soa' if 'soa' in prp_trials.columns else None
    )

    if rt_col is None or soa_col is None:
        return set(), set(prp_trials['participantId'].unique())

    # 유효 trial 필터
    valid = prp_trials[
        (prp_trials['t1_correct'] == True) &
        (prp_trials['t2_correct'] == True) &
        (prp_trials['t2_timeout'] == False) &
        (prp_trials[rt_col] > DEFAULT_RT_MIN) &
        (prp_trials[rt_col] < PRP_RT_MAX)
    ].copy()

    # SOA binning (short/long만)
    def bin_soa(soa):
        if pd.isna(soa):
            return 'other'
        if soa <= DEFAULT_SOA_SHORT:
            return 'short'
        if soa >= DEFAULT_SOA_LONG:
            return 'long'
        return 'other'

    valid['soa_bin'] = valid[soa_col].apply(bin_soa)
    valid = valid[valid['soa_bin'].isin(['short', 'long'])]

    valid_ids = set(valid['participantId'].unique())
    all_ids = set(prp_trials['participantId'].unique())
    no_valid_ids = all_ids - valid_ids

    return valid_ids, no_valid_ids


def get_prp_valid_participants(
    data_dir: Path = None,
    criteria: PRPQCCriteria = None,
    verbose: bool = False,
) -> Set[str]:
    """PRP 과제 품질 기준 통과자 반환.

    Args:
        data_dir: raw 데이터 디렉토리 (기본: RAW_DIR)
        criteria: PRP QC 기준 (기본: PRPQCCriteria())
        verbose: 상세 로그 출력 여부

    Returns:
        유효 참가자 ID 집합
    """
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = PRPQCCriteria()

    summary_path = data_dir / '3_cognitive_tests_summary.csv'
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] 인지과제 요약 파일 없음: {summary_path}")
        return set()

    summary_df = pd.read_csv(summary_path, encoding='utf-8-sig')
    summary_df['testName'] = summary_df['testName'].str.lower()

    prp_df = summary_df[summary_df['testName'] == 'prp'].copy()
    if prp_df.empty:
        return set()

    # 시행 수 기준 (float 비교 주의)
    if criteria.n_trials_required > 0:
        prp_df = prp_df[prp_df['n_trials'].fillna(0).astype(int) == criteria.n_trials_required]

    # 필수 지표 존재 여부
    if criteria.require_metrics:
        mask = (
            prp_df['mrt_t2'].notna() &
            prp_df['rt2_soa_50'].notna() &
            prp_df['rt2_soa_1200'].notna()
        )
        prp_df = prp_df[mask]

    valid_ids = set(prp_df['participantId'].unique())

    # 정확도 기준 (acc_t2 컬럼 사용, 0-100% 스케일)
    if criteria.min_accuracy is not None:
        # acc_t2는 0-100 스케일, min_accuracy는 0-1 스케일
        acc_threshold = criteria.min_accuracy * 100  # 0.80 -> 80
        if 'acc_t2' in prp_df.columns:
            prp_df_acc = prp_df[prp_df['acc_t2'].fillna(0) >= acc_threshold]
            valid_ids = set(prp_df_acc['participantId'].unique())

    # 유효 trial 존재 여부
    if criteria.require_valid_trials:
        trial_valid, trial_invalid = _get_prp_valid_trial_participants(data_dir, verbose)
        excluded = valid_ids & trial_invalid
        if excluded and verbose:
            print(f"  [INFO] PRP 유효 trial 0건 제외: {len(excluded)}명")
        valid_ids -= trial_invalid

    if verbose:
        all_prp = set(summary_df[summary_df['testName'] == 'prp']['participantId'].unique())
        excluded = all_prp - valid_ids
        if excluded:
            print(f"  [INFO] PRP QC 미달: {len(excluded)}명")

    return valid_ids


# =============================================================================
# WCST Filtering
# =============================================================================

def get_wcst_valid_participants(
    data_dir: Path = None,
    criteria: WCSTQCCriteria = None,
    verbose: bool = False,
) -> Set[str]:
    """WCST 과제 품질 기준 통과자 반환.

    Args:
        data_dir: raw 데이터 디렉토리 (기본: RAW_DIR)
        criteria: WCST QC 기준 (기본: WCSTQCCriteria())
        verbose: 상세 로그 출력 여부

    Returns:
        유효 참가자 ID 집합
    """
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = WCSTQCCriteria()

    summary_path = data_dir / '3_cognitive_tests_summary.csv'
    if not summary_path.exists():
        if verbose:
            print(f"[WARN] 인지과제 요약 파일 없음: {summary_path}")
        return set()

    summary_df = pd.read_csv(summary_path, encoding='utf-8-sig')
    summary_df['testName'] = summary_df['testName'].str.lower()

    wcst_df = summary_df[summary_df['testName'] == 'wcst'].copy()
    if wcst_df.empty:
        return set()

    # 필수 지표 존재 여부
    if criteria.require_metrics:
        mask = (
            (wcst_df['totalTrialCount'] > 0) &
            (wcst_df['completedCategories'] >= criteria.min_completed_categories) &
            (wcst_df['perseverativeErrorCount'].notna())
        )
        wcst_df = wcst_df[mask]

    # 총 오류 수 기준
    if criteria.max_total_errors > 0 and 'totalErrorCount' in wcst_df.columns:
        wcst_df = wcst_df[wcst_df['totalErrorCount'] < criteria.max_total_errors]

    # 보속 반응 수 기준
    if criteria.max_perseverative_responses > 0 and 'perseverativeResponses' in wcst_df.columns:
        wcst_df = wcst_df[wcst_df['perseverativeResponses'] < criteria.max_perseverative_responses]

    valid_ids = set(wcst_df['participantId'].unique())

    if verbose:
        all_wcst = set(summary_df[summary_df['testName'] == 'wcst']['participantId'].unique())
        excluded = all_wcst - valid_ids
        if excluded:
            print(f"  [INFO] WCST QC 미달: {len(excluded)}명")

    return valid_ids


# =============================================================================
# Unified API
# =============================================================================

def get_task_valid_participants(
    task: str,
    data_dir: Path = None,
    survey_criteria: SurveyQCCriteria = None,
    task_criteria = None,
    verbose: bool = False,
) -> Set[str]:
    """통합 API: task별 + 설문 완료자 교집합 반환.

    Args:
        task: 'stroop', 'prp', 'wcst'
        data_dir: raw 데이터 디렉토리 (기본: RAW_DIR)
        survey_criteria: 설문 QC 기준 (기본: SurveyQCCriteria())
        task_criteria: task별 QC 기준 (기본: 각 task의 기본 criteria)
        verbose: 상세 로그 출력 여부

    Returns:
        유효 참가자 ID 집합 (설문 + 해당 task 완료)
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")

    if data_dir is None:
        data_dir = RAW_DIR

    # 설문 유효 참가자
    survey_valid = get_survey_valid_participants(data_dir, survey_criteria, verbose)

    # Task 유효 참가자
    if task == 'stroop':
        task_valid = get_stroop_valid_participants(data_dir, task_criteria, verbose)
    elif task == 'prp':
        task_valid = get_prp_valid_participants(data_dir, task_criteria, verbose)
    elif task == 'wcst':
        task_valid = get_wcst_valid_participants(data_dir, task_criteria, verbose)
    else:
        task_valid = set()

    result = survey_valid & task_valid

    if verbose:
        print(f"{task.upper()} 완료 참가자: {len(result)}명 (설문: {len(survey_valid)}, {task}: {len(task_valid)})")

    return result
